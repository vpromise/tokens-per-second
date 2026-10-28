# Inference, Explained: research notes (factual backbone)

**Fact date: October 2026** (compiled 2026-10-23). All web sources were read on 2026-10-22/23.
Fast-moving facts (hardware specs, rental and API prices, engine versions, metric names) are tagged **[DATED]**. Anything not confirmed against a primary source is tagged **[UNVERIFIED]**. My own derivations are tagged **[DERIVED]**, with the arithmetic shown.

**Conventions**
- GB = 10^9 bytes; GiB = 2^30 bytes; TB/s = 10^12 bytes/s.
- FLOPS figures are **dense** unless stated otherwise. NVIDIA spec pages list tensor-core numbers "with sparsity", so halve them.
- Notation:

| Symbol | Meaning |
|---|---|
| P | parameters |
| b | bytes per parameter |
| B | batch size (sequences decoding together) |
| L | layers |
| d | d_model |
| n_kv | KV heads |
| d_h | head dim |
| T | context length in tokens |
| BW | memory bandwidth |
| C | peak FLOP/s |
| MFU | model FLOPs utilisation |
| MBU | model bandwidth utilisation |

**Reference model for all arithmetic: Llama-3.1-8B (BF16).** Config, from the HF config.json and the Llama 3 paper Table 3:
- 32 layers, d_model 4096, 32 query heads, **8 KV heads**, head_dim 128, FFN 14336, vocab 128,256.
- **8.03B params**, so **16.06 GB** in BF16.

Llama-3.1-70B: 80 layers, d 8192, 64 heads, 8 KV heads, head_dim 128, 70.55B params, 141.1 GB in BF16. Llama 3 paper: "with 8 key-value heads to improve inference speed and to reduce the size of key-value caches during decoding."

Three round numbers carry the course: **8 B params, 16 GB of weights, 128 KiB of KV cache per token.**

---

## §0 Cheat sheet (the 15 numbers worth memorising)

| # | Fact | Value | Source |
|---|---|---|---|
| 1 | FLOPs per generated token (forward) | ≈ 2 × P (8B → 16 GFLOP) | kipply 2022 |
| 2 | Bytes read per decode step, batch 1 | ≈ all weights + this sequence's KV | JAX scaling book |
| 3 | H100 SXM | 80 GB HBM3, 3.35 TB/s, 989 TFLOPS BF16 dense | NVIDIA [DATED] |
| 4 | H100 ridge point | 989e12 / 3.35e12 ≈ 295 FLOPs/byte; JAX book: "about 280 tokens" | [DERIVED] / JAX |
| 5 | 8B BF16 decode ceiling, batch 1 | 208 tok/s on H100; 34 tok/s on M4 Max (546 GB/s) | [DERIVED] |
| 6 | KV per token, Llama-3.1-8B | 2·32·8·128·2 B = 131,072 B = 128 KiB | [DERIVED] |
| 7 | KV of one 128K-token conversation (8B) | 16 GiB, the same size as the weights | [DERIVED] |
| 8 | KV memory actually used by pre-vLLM systems | 20.4–38.2% | Kwon et al. 2023 |
| 9 | Continuous batching headline | 36.9× throughput vs FasterTransformer at the same latency | Orca, OSDI 2022 |
| 10 | Speculative decoding: expected tokens per target pass | (1−α^(γ+1))/(1−α) | Leviathan et al. 2023 |
| 11 | FlashAttention-3 on H100 | up to 740 TFLOPS FP16 (75% utilisation) | Shah et al. 2024 |
| 12 | Batch-1 decode speedup from 4-bit weights | ~3.25× (GPTQ, OPT-175B, A100) | Frantar et al. 2023 |
| 13 | MoE: Mixtral | 46.7B total / 12.9B active; DeepSeek-V3: 671B / 37B | Mistral 2023; DeepSeek 2024 |
| 14 | Cached-input price ratios [DATED Oct 2026] | Anthropic 0.1× (0.05× on Opus/Sonnet 5.5); OpenAI "up to 95%" off; Gemini 0.05–0.1× | official docs |
| 15 | Output/input API price ratio [DATED] | typically 4–6× across Anthropic/OpenAI/Google flagships | official pricing |

---

## §1 The anatomy of one generation

### Mechanism (plain words)
A chat request goes through six stages.
1. **Tokenize** (CPU). The text becomes integer IDs (BPE). This is cheap: microseconds to a millisecond for typical prompts [order of magnitude, UNVERIFIED].
2. **Queue.** The request waits for a slot in the server's running batch.
3. **Prefill.** One forward pass processes *all* prompt tokens in parallel. It writes their keys/values into the **KV cache** and yields the logits for the first output token. Prefill is matrix–matrix work, so it is **compute-bound**.
4. **Decode loop.** Each step feeds *one* new token per sequence. It reads all weights plus that sequence's KV cache, appends one K/V entry per layer, and produces logits. Decode is matrix–vector work at small batch, so it is **memory-bandwidth-bound**.
5. **Sample.** Logits → processors (temperature, top-k/top-p, penalties, grammar mask) → one token ID.
6. **Detokenize incrementally and stream.** Each token is converted to text and sent as a server-sent event. A token can end in the middle of a multi-byte UTF-8 character, so servers buffer until the bytes form valid text.

Generation stops on EOS, a stop string, or max_tokens.

### Key metrics and formulas (Databricks, "LLM Inference Performance Engineering: Best Practices", 12 Oct 2023)

| Metric | Definition |
|---|---|
| TTFT | "How quickly users start seeing the model's output after entering their query." TTFT = network + queue + tokenize + prefill + first sample. |
| TPOT | "Time to generate an output token for each user that is querying our system." Benchmarks compute TPOT = (E2E − TTFT)/(N_out − 1). |
| ITL (inter-token latency) | The distribution of individual gaps between tokens. Its p99 shows hiccups that the mean TPOT hides. |
| Latency | Databricks: "latency = (TTFT) + (TPOT) * (the number of tokens to be generated)" |
| Throughput | "The number of output tokens per second an inference server can generate across all users and requests." Not the same as one user's tokens/s. |
| Goodput (DistServe, OSDI 2024) | The maximum request rate served while meeting the SLO-attainment goal, e.g. 90% of requests within the TTFT and TPOT targets. See §5. |

### Where time goes
At batch 1, decode dominates almost every chat request.
- Databricks: "Output length dominates overall response latency".
- Databricks: "The addition of 512 input tokens increases latency less than the production of 8 additional output tokens" (MPT models).
- Databricks: "Generating the first token is typically compute-bound, while subsequent decoding is memory-bound operation."

### Worked example (animate as a timeline) [DERIVED]
"Owl" (the running example; see Appendix A) sends a 1,500-token prompt and gets a 300-token answer. Model: Llama-3.1-8B BF16 on one H100 SXM, batch 1.
1. **Prefill FLOPs** = 2 × 8.03e9 × 1,500 = **24.1 TFLOP**.
   - At 989 TFLOPS peak that is 24.4 ms; at a realistic 50% MFU, **≈ 49 ms**.
   - Attention adds about 2% at this length (see §11).
2. **One decode step** reads 16.06 GB of weights plus about 0.24 GB of KV (1,800 tokens × 128 KiB) = 16.3 GB.
   - 16.3 GB ÷ 3.35 TB/s = **4.86 ms floor**; at 75% MBU, **≈ 6.5 ms**.
3. **The rest of the decode** is 299 more steps × 6.5 ms ≈ **1.94 s**.
4. **Total**: about 2.0 s plus network and queue.
   - TTFT ≈ 50 ms + queue.
   - TPOT ≈ 6.5 ms, about 154 tok/s for this one user.
   - **Decode is ~97% of the wall-clock time**, even though prefill processed 5× more tokens.

Animation idea: a 1,500-token block lands in one thick "prefill" bar, then 300 thin ticks follow, each one a full sweep over all 16 GB.

### Common misconceptions
- "TTFT is mostly network." On a loaded server it is mostly queueing plus prefill. For long prompts it is dominated by prefill compute (see §11).
- "Tokens/s is one number." Per-user speed (1/TPOT) and server throughput (summed over users) move in opposite directions as batch size grows (§15).
- "A long prompt costs as much time as a long answer." No: prompt tokens are processed in parallel, while output tokens are produced one forward pass at a time.
- "The model writes the whole answer, then streams it." No: streaming sends each token as it is sampled.

### Quotable
- "Output length dominates overall response latency" (Databricks, 2023).
- "Generating the first token is typically compute-bound, while subsequent decoding is memory-bound operation." (Databricks, 2023)

---

## §2 Arithmetic intensity and the roofline

### Mechanism
Every kernel does some FLOPs and moves some bytes between HBM (or unified DRAM) and the compute units. **Arithmetic intensity** I = FLOPs ÷ bytes.
- The hardware has a **ridge point** I* = C ÷ BW.
- If I < I*, the kernel is **memory-bound**: its time is set by bytes ÷ BW.
- If I > I*, it is **compute-bound**: its time is set by FLOPs ÷ C.

In a decoder at batch B, each weight (b bytes) is read once per step and used for 2B FLOPs (one multiply-add per sequence). With BF16 (b = 2), the intensity of the linear layers is **I ≈ B FLOPs/byte**. So:
- **Batch-1 decode** has I ≈ 1, which is hopelessly memory-bound.
- **Prefill** of n tokens has I ≈ n. A 2,048-token prompt is far above the ridge, so it is compute-bound.

### Formulas
- **FLOPs per token (forward):** ≈ 2P.
  - kipply: "We do 2⋅P flops of operations, which can be intuited by the fact that we matmul through all the parameters."
  - Attention adds ≈ 4·L·T·d FLOPs per token at context T. This is negligible at short T.
- **Bytes per decode step:** ≈ P·b_w + B·T·kv_bytes_per_token (+ small activations).
- **Step time.** JAX scaling book: "Theoretical minimum step time = (Batch × KV_cache_size + Param_size) / HBM bandwidth" (memory-bound regime). In general:
  - T_step ≈ B·KV/BW + max(2·B·P/C, P·b/BW)
- **Batch-1 decode ceiling (tokens/s):** ≈ BW ÷ (P·b + T·kv_per_token).
- **Prefill time:** ≈ 2·P·n_prompt ÷ (C·MFU), plus attention (§11).
- **Critical batch** (where decode turns compute-bound) ≈ C ÷ BW × (b/2).
  - kipply, for an A100-40GB (312 TFLOPS / 1.5 TB/s): "we get a distinct ratio here of 208 given this hardware specification." / "Anything below, we're memory bandwidth bound. Above, flops bound."
  - JAX book: "For bf16 activations on TPU v5e, this is 240 tokens. For an H100, it is about 280 tokens."
  - Our spec-sheet arithmetic gives 295 for the H100; the JAX book's "about 280" uses slightly different peak figures. Either way, the order of magnitude is **a few hundred sequences**.
- **MBU** (Databricks): "MBU is defined as (achieved memory bandwidth) / (peak memory bandwidth)" where "achieved memory bandwidth is ((total model parameter size + KV cache size) / TPOT)."
- **MFU** = achieved FLOP/s ÷ peak FLOP/s.

### Hardware table [DATED Oct 2026; vendor pages]

| Device | Memory | BW | BF16 dense | Ridge (FLOPs/B) | 8B BF16 ceiling, B=1 | 8B 4-bit (≈4.5 bpw, 4.52 GB) ceiling |
|---|---|---|---|---|---|---|
| A100 80GB SXM | 80 GB | 2.039 TB/s | 312 TF | 153 | 127 tok/s | 451 |
| H100 SXM | 80 GB | 3.35 TB/s | 989 TF | 295 | **208** | 741 |
| H100 PCIe (older brief) | 80 GB | 2.0 TB/s | ~756 TF | ~378 | 125 | 442 |
| H100 NVL | 94 GB | 3.9 TB/s | ~835 TF | 214 | 243 | 863 |
| H200 SXM | 141 GB | 4.8 TB/s | 989 TF | 206 | 299 | 1,062 |
| B200 (DGX B200 ÷ 8) | 180 GB | 8 TB/s | FP8 4.5 PF dense; BF16 ≈2.25 PF [DERIVED, half of FP8] | ~281 | 498 | 1,770 |
| Apple M4 | ≤32 GB | 120 GB/s | n/p | – | 7.5 | 26.5 |
| Apple M4 Pro | ≤64 GB | 273 GB/s | n/p | – | 17.0 | 60 |
| Apple M4 Max (40-core GPU) | ≤128 GB | 546 GB/s | n/p | – | **34.0** | 121 |
| Apple M5 (Oct 2025) | ≤32 GB | 153 GB/s | n/p | – | 9.5 | 34 |
| Apple M5 Pro (Mar 2026) | ≤64 GB | 307 GB/s | n/p | – | 19.1 | 68 |
| Apple M5 Max (40-core GPU, Mar 2026) | ≤128 GB | 614 GB/s | n/p | – | 38.2 | 136 |
| Apple M3 Ultra (Mar 2025) | ≤512 GB | 819 GB/s | n/p | – | 51 | 181 |
| Apple M5 Ultra (Aug 2026) | ≤512 GB | 1.2 TB/s | n/p | – | 74.7 | 265 |

Notes on the table:
- n/p = Apple does not publish GPU TFLOPS. Apple only states relative claims, e.g. M5 has "over 4x the peak GPU compute performance for AI compared to M4", and M5 Ultra "4.3x the peak AI compute performance of M3 Ultra".
- Ceiling = BW ÷ weight bytes. It ignores KV reads and assumes 100% MBU, so real numbers are 60–90% of it.
- NVIDIA sources:
  - H100: "NVIDIA NVLink™: 900GB/s; PCIe Gen5: 128GB/s" (nvidia.com/en-us/data-center/h100).
  - A100: nvidia.com/en-us/data-center/a100.
  - H200: nvidia.com/en-us/data-center/h200.
  - DGX B200: "1,440 GB total, 64 TB/s HBM3e bandwidth"; FP8 "72 PFLOPS" for 8 GPUs, with footnote "Dense performance is ½ sparse spec shown".
- Apple sources:
  - Newsroom: M4 Pro/Max 2024-10-30; M5 2025-10-15; M5 Pro/Max 2026-03-03, which says "up to 307GB/s" and "up to 614GB/s"; M5 Ultra 2026-08-25.
  - M3 Ultra newsroom 2025-03-05.
  - [UNCERTAIN] Whether the 512 GB M3 Ultra configuration is still sold.

### Worked example 1: "the speed limit is bandwidth ÷ size" [DERIVED]
1. Llama-3.1-8B in BF16 = 8.03e9 × 2 B = **16.06 GB**.
2. **H100:** 3.35e12 ÷ 16.06e9 = **208.6 tokens/s** max, i.e. 4.79 ms per token.
   - Compute needed per token is 16.06 GFLOP ÷ 989e12 = **16 µs**. The GPU's math units are idle about 99.7% of the time at batch 1.
3. **MacBook Pro M4 Max (546 GB/s):** 546 ÷ 16.06 = **34 tok/s** max. In 4-bit (≈4.52 GB) it is about **121 tok/s** max.
4. **Base M4 (120 GB/s):** a 16 GB model does not fit usefully on a 16 GB Mac. In 4-bit: 120 ÷ 4.52 ≈ **26.5 tok/s** max.

### Worked example 2: check the ceiling against measurements
Animation idea: **"multiply speed by size and you get bandwidth".**

llama.cpp Apple Silicon benchmark (github.com/ggml-org/llama.cpp/discussions/4167; LLaMA-2-7B; `llama-bench -p 512 -n 128`). Bytes per token: F16 = 13.48 GB (6.74B × 2); Q4_0 = 3.79 GB (4.5 bpw). M5-generation rows used a newer llama.cpp commit than older rows, so they are not strictly like-for-like.

| Chip (BW) | F16 TG128 | → effective BW (MBU) | Q4_0 TG128 | → eff. BW (MBU) | Q4_0 PP512 |
|---|---|---|---|---|---|
| M1 (68 GB/s) | – | – | 14.15 | 54 GB/s (79%) | 118 |
| M4 (120) | 7.43 | 100 GB/s (83%) | 24.11 | 91 (76%) | 221 |
| M4 Pro (273) | 17.18 | 232 (85%) | 50.74 | 192 (70%) | 440 |
| M4 Max (546) | 31.64 | 426 (78%) | 83.06 | 315 (58%) | 886 |
| M2 Ultra (800) | 41.02 | 553 (69%) | 94.27 | 357 (45%) | 1,238 |
| M5 (153) | – | – | 31.88 | 121 (79%) | 723 |
| M5 Max (614) | 37.11 | 500 (81%) | 119.92 | 455 (74%) | 3,220 |
| M5 Ultra (1,200; table lists 1,228) | 73.60 | 992 (≈83%) | 179.10 | 679 (57%) | 4,945 |

Lessons:
- **(a) Decode tracks bandwidth.** M5 Max vs M4 Max: TG F16 ×1.17, bandwidth ×1.12.
- **(b) Prefill tracks compute.** M5 Max vs M4 Max: PP ×3.4–3.6, because the M5 GPU has Neural Accelerators.
- **(c) Quantized formats use bandwidth less efficiently** (lower MBU), because dequantization costs compute. They still win on absolute speed because they move fewer bytes.

Apple's own confirmation (Apple ML Research, "Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU", 2025-11-19):
- M5 vs M4: TTFT "up to 4x" faster, generation "19-27%" faster, bandwidth "28% higher".
- Quote: "generating the first token is compute-bound, and takes full advantage of the Neural Accelerators." / "Generating subsequent tokens is bounded by memory bandwidth, rather than by compute ability."

Apple ML Research, "Local and Distributed Large LLM Inference with MLX and RDMA on Mac Studio" (2026-10-15):
- "Generation speed (in tokens per second) is memory-bandwidth bound." / "Time to first token (TTFT) is essentially compute bound".
- Decode, M3 Ultra → M5 Ultra:
  - Qwen 3.8 (27B dense, BF16): 12.46 → 19.69 tok/s.
  - Mistral Medium 3.5 (128B, FP8): 4.95 → 7.70.
- [DERIVED] Ceilings: 27B BF16 = 54 GB, so 819/54 = 15.2 and 1,200/54 = 22.2 tok/s → MBU 82% and 89%. The Mistral model gives 77% and 82%.
- Caveats: context lengths per cell are unlabeled on the page, and the text's "up to 4.6×" TTFT claim does not match the tables (3.9×, 4.4×).

GPU cross-check (PyTorch "GPT, Fast" blog, 2023):
- "7B × 2 bytes × 107 tok/s ÷ 2 TB/s" gives "72% MBU! ... even just copying memory struggles to break 85%."
- That run was on an A100-80GB, "power limited to 330W".

### Common misconceptions
- "More TFLOPS = faster chatbot." For one user, bandwidth and bytes decide. TFLOPS decide TTFT and big-batch throughput.
- "nvidia-smi says 100% utilisation, so the GPU is maxed." GPU-Util is the fraction of time *some* kernel was running. GKE's docs say GPU utilization "Does not measure how much work is being done while the GPU is active."
- "Decode is slow because attention is quadratic." At short context, decode is slow because it re-reads 16 GB of *weights* for every token. At long context, KV reads take over (§3, §11).

### Quotable
- "inference from large models is often not bottlenecked on arithmetic operations, but rather on memory bandwidth and communication" (Leviathan et al., ICML 2023).
- "Transformer sampling is typically memory bandwidth bound." (Chen et al., DeepMind, 2023)
- "for BS=1 every single matrix multiplication in a transformer is actually a matrix vector multiplication ... the computations are completely memory-bandwidth bound." (PyTorch, GPT-Fast blog)

---

## §3 The KV cache

### Mechanism
Attention for a new token needs the keys and values of *every previous token*, in every layer.
- **Without a cache**, step t would recompute K and V for all t−1 earlier tokens, so total work grows as O(T²).
- **With a cache**, each step computes K/V only for the new token and appends them. The past is *read*, not recomputed.

The price is memory: the cache grows linearly with tokens × batch, and **every decode step must read the whole cache of each sequence**.

### Formula
**KV bytes per token = 2 (K and V) × L × n_kv × d_h × b.**
- JAX book: "the 2 comes from storing both the keys and values."
- kipply (for multi-head attention, fp16): "Per token, the number of bytes we store is" 2·2·n_layers·n_heads·d_head = 4·n_layers·d_model.
- **Per sequence:** × T tokens. **Per server:** × the number of concurrent sequences.

### Worked numbers [DERIVED from configs]

| Model | L | n_kv × d_h | KV/token (BF16) | 8K ctx | 32K | 128K |
|---|---|---|---|---|---|---|
| Llama-3.1-8B (GQA, 8 KV heads) | 32 | 8×128 | **128 KiB** | 1 GiB | 4 GiB | **16 GiB** |
| same model if it were MHA (32 KV heads) | 32 | 32×128 | 512 KiB | 4 GiB | 16 GiB | 64 GiB |
| Llama-3.1-70B (GQA, 8 KV heads) | 80 | 8×128 | **320 KiB** | 2.5 GiB | 10 GiB | **40 GiB** |
| same if MHA (64 KV heads) | 80 | 64×128 | 2.5 MiB | 20 GiB | 80 GiB | 320 GiB |
| OPT-13B (MHA; vLLM paper's example) | 40 | 5120 | 800 KiB ("800 KB") | – | – | – |
| Llama-3.2-1B (Mac demo model) | 16 | 8×64 | 32 KiB | 256 MiB | 1 GiB | 4 GiB |
| Qwen3-1.7B (Mac demo model) | 28 | 8×128 | 112 KiB | 896 MiB | 3.5 GiB | (max ctx 40,960) |
| DeepSeek-V3 (MLA, 671B) | 61 | latent 512 + RoPE key 64 = 576 elements | 68.6 KiB | 0.54 GiB | 2.1 GiB | 8.6 GiB |

Notes on the table:
- The vLLM paper (§3) confirms the OPT-13B row: "the KV cache of a single token demands 800 KB of space", and one request "can be as much as 1.6 GB".
- The 576-element figure for MLA is **[DERIVED]** from DeepSeek-V2 §2.1.1 and the V3 config (kv_lora_rank 512, qk_rope_head_dim 64). The papers do not state "576" themselves.
- **Surprise for the animation:** DeepSeek-V3, an 84× bigger model, stores *less* KV per token than Llama-3.1-8B. And Qwen3-1.7B stores 3.5× more per token than Llama-3.2-1B.
- JAX book's LLaMA-2-13B example: one 8,192-token bf16 sequence has a 6.7 GB KV cache. "Just 4 of these exceed the memory usage of our parameters!"

### Worked example: "how many Owl conversations fit on one H100?" [DERIVED]
1. Budget 90% of 80 GB = 72 GB.
2. Subtract 16.06 GB of weights → **≈ 56 GB for KV**. This ignores activations and CUDA-graph workspace, so real numbers are a bit lower.
3. 56e9 ÷ 131,072 B/token = **≈ 427,000 token-slots**.
4. At 8K tokens per conversation: **52 concurrent conversations**.
5. Compare:
   - FP8 KV cache: 104.
   - The same model with MHA: 13.
   - 128K-token conversations: only 3.

**Decode cost of the cache.** At T = 32K, each step reads 4 GiB (4.3 GB) of KV per sequence on top of the 16 GB of weights, so batch-1 TPOT rises ≈ 27%. At 128K the cache (17.2 GB) is larger than the weights, and the floor TPOT doubles from 4.9 to 9.9 ms.

### Shrinking the cache (one family of ideas, five tricks)
1. **MQA** (Shazeer 2019, arXiv 1911.02150): one K/V head shared by all query heads.
   - "keys and values are shared across all of the different attention heads, greatly reducing the size of these tensors and hence the memory bandwidth requirements of incremental decoding."
2. **GQA** (Ainslie et al., EMNLP 2023, arXiv 2305.13245): groups of query heads share a K/V head. Llama 3 uses 8 KV heads, a 4× saving for 8B and 8× for 70B.
   - "uptrained GQA achieves quality close to multi-head attention with comparable speed to MQA". Uptraining uses "5% of original pre-training compute".
3. **MLA** (DeepSeek-V2, arXiv 2405.04434): cache a low-rank latent (512) plus a small decoupled RoPE key (64) instead of full K/V. K/V are reconstructed (absorbed into the projections) at attention time.
   - "MLA guarantees efficient inference through significantly compressing the Key-Value (KV) cache into a latent vector".
   - V2 vs DeepSeek 67B: "reduces the KV cache by 93.3%" and "boosts the maximum generation throughput to 5.76 times".
   - V2 is 236B total / 21B active, with 128K context.
4. **Sliding-window / local attention.** Layers only attend to the last W tokens, so the cache is a rolling buffer of W entries (slot i mod W).
   - Mistral 7B (arXiv 2310.06825): W = 4096. "On a sequence length of 32k tokens, this reduces the cache memory usage by 8x, without impacting the model quality."
   - Gemma 3 (arXiv 2503.19786) interleaves "5 local layers for every global layer" and assigns "a smaller span of only 1024 tokens to the local layers". KV memory overhead falls from "60%" (global-only) to "less than 15%". That measurement was on an ablation configuration (1:3, 2B model), not the shipped 5:1, so cite it carefully.
5. **KV quantization.**
   - FP8 KV halves the cache. vLLM docs: `kv_cache_dtype = "fp8" | "fp8_e4m3" | "fp8_e5m2"`; "Quantizing the KV (Key-Value) cache to FP8 format can significantly reduce its memory footprint." Without calibration the scales default to 1.0.
   - **KIVI** (Liu et al., ICML 2024, arXiv 2402.02750), 2-bit, tuning-free: "the key cache should be quantized per-channel", "the value cache should be quantized per-token".
   - KIVI results: 2.6× lower peak memory (including weights), up to 4× larger batch, 2.35–3.47× throughput.
   - Apple's 2025 on-device model: "The KV cache was quantized to 8 bits per weight."
   - mlx-lm: `--kv-bits`, `--kv-group-size`, `--quantized-kv-start`.

### KV offloading and tiering
When GPU memory is full, KV blocks can move to CPU DRAM or SSD and come back later. This is cheaper than recomputing a long prefix.
- **vLLM KV-offloading blog** (2026-01-08; Llama-3.1-8B-Instruct, H100):
  - "loading KV values from the CPU reduces TTFT by X2-X22, depending on the prompt size".
  - CPU↔GPU "DMA achieves 83.4 GB/s" bidirectional.
  - Flags: `--kv_offloading_backend native --kv_offloading_size <GB>` (v0.14.0+).
- **LMCache** (arXiv 2510.09665) moves KV "across GPU, CPU, storage, and network layers"; "up to 15x improvement in throughput" with vLLM.
- **Mooncake** (Kimi) "leverages the underutilized CPU, DRAM, and SSD resources of the GPU cluster" (§5).
- **DeepSeek production** (24 h, Feb 2025): "342B tokens (56.3%) hit the on-disk KV cache".
- **[DERIVED]** Is moving it worth it? Reloading a 32K-token Llama-8B prefix (4.3 GB) over 41.7 GB/s (one direction of 83.4) takes about **0.10 s**. Recomputing it takes ≈ 2·8.03e9·32K + attention ≈ 0.8 PFLOP, i.e. **≈ 1.6 s at 50% MFU** (§11). That is why offloading pays.

### Misconceptions
- "The KV cache is an optional speed-up cache, like a CDN." It is the model's working memory for the conversation. Without it, decode cost explodes quadratically.
- "KV size depends on model size." It depends on L × n_kv × d_h, so attention design dominates (see the DeepSeek-V3 vs Llama-8B row).
- "Once the weights fit, you're done." On a busy server the KV cache, not the weights, decides how many users fit.

### Quotable
- "with 8 key-value heads to improve inference speed and to reduce the size of key-value caches during decoding." (Llama 3 Herd of Models, 2024)
- "Just 4 of these exceed the memory usage of our parameters!" (JAX scaling book, on 8K-token LLaMA-2-13B caches)
- "reduces the KV cache by 93.3%" (DeepSeek-V2, 2024)

---

## §4 Memory management: PagedAttention, prefix caching, copy-on-write

### Mechanism
**Before vLLM, servers reserved one contiguous KV region per request**, sized for the *maximum* possible length. Kwon et al. name three kinds of waste:
- "reserved slots for future tokens";
- "internal fragmentation due to over-provisioning for potential maximum sequence lengths";
- "external fragmentation from the memory allocator like the buddy allocator".

**PagedAttention** (Kwon et al., SOSP 2023, arXiv 2309.06180) treats the KV cache like OS virtual memory.
- KV is stored in fixed-size **blocks** ("vLLM sets its default block size as 16"), which can sit anywhere in GPU memory.
- A per-request **block table** maps logical blocks to physical blocks. The attention kernel gathers through that table.
- Memory is allocated block by block as tokens arrive, so the only waste is the unfilled tail of the last block (< 16 slots per sequence).
- "one can think of blocks as pages, tokens as bytes, and requests as processes."

**Sharing and copy-on-write.**
- Parallel sampling (n > 1) and beam search share the prompt's physical blocks, with a reference count per block.
- When a branch needs to write into a shared *partial* block, it gets its own copy: "with the exception of the final logical block, which is managed by a copy-on-write mechanism."

**Prefix caching** keeps the blocks of finished requests and reuses them when a new request starts with the same tokens.
- **vLLM Automatic Prefix Caching:** "we hash each kv-cache block by the tokens in the block and the tokens in the prefix before the block". "We only cache full blocks." Eviction is LRU. V1 turned it on by default, reporting "near-zero performance degradation, even when the cache hit rate is 0%".
- **SGLang RadixAttention** (Zheng et al., NeurIPS 2024, arXiv 2312.07104) keeps all cached prefixes in a **radix tree**: "our system maintains an LRU cache of the KV cache for all requests within a radix tree".
  - Its scheduler is cache-aware: "we sort the requests by matched prefix length and prioritize requests with longer matched prefixes".
  - Benchmarks: "the cache hit rate ranges from 50% to 99%".

### Numbers (with sources)
- **Kwon et al.:** existing systems store actual token states in "only 20.4% - 38.2% of the KV cache memory". Throughput is 2–4× FasterTransformer and Orca at similar latency.
  - Sharing saves "6.1% - 9.8% memory saving on parallel sampling and 37.6% - 55.2% on beam search."
- **vLLM launch blog** (2023-06-20):
  - Existing systems "waste 60% – 80% of memory due to fragmentation and over-reservation"; vLLM has "a mere waste of under 4%".
  - "up to 24x higher throughput than HuggingFace Transformers" and "up to 3.5x higher throughput than TGI".
  - Parallel-sampling memory is cut "by up to 55%", which "can translate into up to 2.2x improvement in throughput."
- **SGLang:** up to 6.4× throughput (paper). The Jan 2024 LMSYS blog said "up to 5 times higher throughput". Say which one you quote.

### Worked example (animate with coloured blocks) [DERIVED]
1. **Old way.** A request may reach 4,096 tokens, so 4,096 slots are reserved (512 MiB for Llama-8B). It finishes at 600 tokens. **85% of the reservation was never used**, and nobody else could use it meanwhile.
2. **Paged way.** 600 tokens need ⌈600/16⌉ = 38 blocks (608 slots). Only 8 slots are wasted (1.3%).
3. **Sharing.** Owl's 1,024-token system prompt = exactly 64 blocks. Ask for n = 4 answers of 200 tokens each.
   - Without sharing: 4 × (1,024 + 200) = 4,896 slots.
   - With shared prompt blocks: 1,024 + 4 × 200 = 1,824 slots. **63% less.**
4. **Prefix caching across users.** 1,000 Owl users per minute send the same 1,024-token system prompt. Only the first request prefills it; the other 999 skip 64 blocks of prefill.
   - 1,024 tokens × 16 GFLOP = 16.4 TFLOP saved per request, about 33 ms of H100 time at 50% MFU.
5. **Copy-on-write.** Two beams share a half-full block. Beam 2 writes token 9 into it, so the block is copied first and the reference counts are updated.

### Misconceptions
- "PagedAttention means paging to disk." The core idea is *non-contiguous GPU allocation*. Swapping or offloading is a separate feature (§3, §16).
- "Prefix caching matches similar prompts." It needs an **exact token-prefix match**, and in vLLM on full 16-token blocks. Changing one token early in the prompt (a timestamp, say) invalidates everything after it. Same lesson as "Agents, Animated": stable content first, variable content last.

### Quotable
- "only 20.4% - 38.2% of the KV cache memory is used to store the actual token states in the existing systems." (Kwon et al., 2023)
- "an attention algorithm inspired by the classical virtual memory and paging techniques in operating systems" (Kwon et al., 2023)
- "retains the KV cache for both prompts and generation results in a radix tree" (LMSYS SGLang blog, Jan 2024)

---

## §5 Batching and scheduling

### Mechanism
Decode is memory-bound, so adding sequences to a step is almost free until the batch nears the ridge point. A step reads 16 GB of weights either way (§15 has the cost curve). The question is *how* to batch.

**Static (request-level) batching.** Collect N requests, run them together until **all** finish.
- Orca: "requests that have finished earlier than other requests in a batch cannot return to the client, while newly arrived requests have to wait until the current batch completely finishes."

**Dynamic batching.** Static batching that waits up to a time window to fill a batch. Better occupancy, same tail problem.

**Continuous (iteration-level) batching** (Orca, Yu et al., OSDI 2022). The scheduler re-forms the batch at *every decode step*. A finished sequence leaves immediately and a waiting one joins.
- "a newly arrived request can be considered for processing after waiting for only a single iteration of the model."
- **Selective batching** "applies batching only to a selected set of operations". Requests have different lengths, so attention runs per request, while the big matmuls are batched over all tokens.
- vLLM, SGLang, TensorRT-LLM and llama-server all do this today. llama-server calls it "Continuous batching", "a.k.a dynamic batching", and enables it by default.

**Chunked prefill and stall-free scheduling** (Sarathi, arXiv 2308.16369; Sarathi-Serve, OSDI 2024, arXiv 2403.02310). A long prompt arriving mid-stream would otherwise occupy a whole step and freeze every decoding user (a "generation stall").
- "Sarathi-Serve introduces chunked-prefills which splits a prefill request into near equal sized chunks and creates stall-free schedules that adds new requests in a batch without pausing ongoing decodes."
- Each step has a **token budget**. Decodes fill it first, and a prefill chunk takes the remainder ("decode-maximal batching").
- **vLLM V1:** "chunked prefill is enabled by default whenever possible"; the scheduler "batches all pending decode requests before scheduling any prefill operations."
- The budget is a dial: "Smaller values (e.g., 2048) achieve better ITL" versus "Higher values achieve better time to first token (TTFT)".

**Prefill/decode disaggregation.** Run prefill and decode on *different GPU pools* and ship the KV cache between them. Each pool can then be sized, parallelised and even built from different hardware to suit its own bottleneck.
- **DistServe** (OSDI 2024, arXiv 2401.09670): "decoding steps in the batch are delayed by the prefill steps, significantly elongating their TPOT".
- **Splitwise** (ISCA 2024, arXiv 2311.18677) calls the phases "a compute-intensive prompt computation, and a memory-intensive token generation".
- **Mooncake** (Kimi; FAST 2025; arXiv 2407.00079): "a KVCache-centric disaggregated architecture that separates the prefill and decoding clusters".
- **Production:**
  - DeepSeek V3/R1: prefill units of 4 nodes / 32 GPUs, decode units of 18 nodes / 144 GPUs.
  - NVIDIA Dynamo (2025-03-18), the "successor to NVIDIA Triton Inference Server", "uses disaggregated serving to separate the processing and generation phases".
  - llm-d (2025-05-20; Red Hat, Google, IBM): "a Kubernetes-native high-performance distributed LLM inference framework".

**Goodput and SLOs.** DistServe defines goodput as the "maximum request rate that can be served adhering to the SLO attainment goal" (e.g. 90% of requests within the TTFT and TPOT targets).
- Raw throughput can rise while goodput falls, because pushing batch size up blows the TPOT SLO.
- `vllm bench serve --goodput ttft:<ms> tpot:<ms>` measures it directly.

### Numbers
- **Orca:** "36.9× throughput improvement at the same level of latency" (GPT-3 175B vs FasterTransformer).
- **Sarathi** (LLaMA-13B, A6000): decode throughput "up to 10x", end-to-end "up to 1.33x".
- **Sarathi-Serve:**
  - "For Mistral-7B on single A100 GPUs, we achieve 2.6x higher serving capacity".
  - "up to 3.7x higher serving capacity for the Yi-34B model on two A100 GPUs as compared to vLLM".
  - Falcon-180B with pipeline parallelism: "up to 5.6x".
- **DistServe:** "can serve 7.4x more requests or 12.6x tighter SLO, compared to state-of-the-art systems, while staying within latency constraints for > 90% of requests."
- **Splitwise:** "1.4x higher throughput at 20% lower cost", or "2.35x more throughput with the same cost and power budgets".
- **Mooncake:** "up to a 525% increase in throughput in certain simulated scenarios"; "enables Kimi to handle 75% more requests" under real workloads.
- **DeepSeek production** (Feb 2025): per H800 node, prefill "~73.7k tokens/s input (including cache hits)" and decode "~14.8k tokens/s output during decoding" (§9).
- **LMSYS** (2025-05-05): SGLang with PD disaggregation and large-scale EP on 96 H100s reached "52.3k input tokens per second and 22.3k output tokens per second per node" (2,000-token inputs), i.e. "$0.20/1M output tokens".
- **vLLM V1** (2025-01-27): "up to 1.7x higher throughput" than V0, credited mainly to lower CPU overhead.

### Worked example A: static vs continuous batching (animate as a 4-slot grid) [DERIVED]
Setup: four slots, time measured in decode steps.
- A, B, C and D arrive at t = 0 needing 10, 3, 6 and 8 tokens.
- E and F arrive at t = 2 needing 4 and 5 tokens.

**Static batching:**
- {A, B, C, D} runs until t = 10, because the batch lasts as long as its longest member.
- Useful slot-steps: 27 of 40 (67.5%).
- E and F start at t = 10 and finish at t = 14 and 15.
- **E's latency is 12 steps; F's is 13.**

**Continuous batching:**
- B finishes at t = 3, so E takes its slot at t = 3 and finishes at t = 7.
- C finishes at t = 6, so F starts at t = 6 and finishes at t = 11.
- **E's latency is 5; F's is 9.** All work is done by t = 11 instead of 15.

### Worked example B: chunked prefill (8B, H100) [DERIVED]
Model: step time = max(compute at 50% MFU, memory at 80% MBU). 30 users are decoding at 2K context, so a decode-only step takes **≈ 9.0 ms** (memory-bound). Then a 4,000-token prompt arrives.
- **No chunking:** that step must do 4,030 tokens × 16 GFLOP ≈ 64.7 TFLOP, about **131 ms**. All 30 users see a 131 ms hiccup, ~15× their normal ITL.
- **512-token chunks:** 8 steps of ≈ 17.6 ms each. Users see 17.6 ms gaps instead of one 131 ms freeze.
  - The newcomer's TTFT goes from 131 ms to ≈ 141 ms (8 × 17.6).
  - That is the trade the token budget controls.

### Misconceptions
- "Batching makes each user slower in proportion." In the memory-bound region it barely does: going from 1 to 64 sequences only doubles the step time (≈ 6.5 → 13.2 ms in §15's table), while aggregate output grows ~31×.
- "Continuous batching = dynamic batching." Dynamic batching waits to *form* a batch; continuous batching edits the batch every step. Note that llama.cpp uses the two terms interchangeably.
- "Disaggregation always wins." It adds KV transfer and operational complexity. The gains in DistServe and Splitwise come from SLO-bound regimes and heterogeneous hardware.

### Quotable
- "a newly arrived request can be considered for processing after waiting for only a single iteration of the model." (Orca, OSDI 2022)
- "creates stall-free schedules that adds new requests in a batch without pausing ongoing decodes." (Sarathi-Serve, OSDI 2024)
- "a compute-intensive prompt computation, and a memory-intensive token generation" (Splitwise, ISCA 2024)

---

## §6 Faster decoding: speculative decoding and friends

### Mechanism
Decode is memory-bound, so a forward pass that checks k+1 tokens costs about the same as one that checks 1: the weights are read once either way.
- **Speculative decoding** lets a cheap *draft* propose γ tokens.
- The *target* model then scores all of them in one pass and keeps the longest acceptable prefix, plus one token of its own.

**Notation.** Leviathan et al. (ICML 2023, arXiv 2211.17192) call the target Mp with distribution p and the draft Mq with distribution q. *Chen et al. use the opposite letters; pick one convention for the course.*

**Acceptance rule (verbatim).** "To sample x ∼ p(x), we instead sample x ∼ q(x), keeping it if q(x) ≤ p(x), and in case q(x) > p(x) we reject the sample with probability 1− p(x)/q(x) and sample x again from an adjusted distribution p′(x) = norm(max(0, p(x) − q(x))) instead."

**Why it is lossless (sketch).**
- P(output x) = q(x)·min(1, p(x)/q(x)) + P(reject)·p′(x) = min(p,q)(x) + (1 − Σmin(p,q))·[(p − q)⁺(x) / Σ(p − q)⁺].
- Since Σ(p − q)⁺ = 1 − Σmin(p,q), this equals min(p,q)(x) + (p − q)⁺(x) = p(x).
- So the output is **exactly** the target's distribution, up to floating-point numerics.

### Formulas (Leviathan et al.)
- **α**, the expected acceptance rate: α = E[Σ_x min(p(x), q(x))].
- **Expected tokens per target pass:** E = (1 − α^(γ+1)) / (1 − α).
- **c** = draft-run time ÷ target-run time.
- **Walltime speedup:** S = (1 − α^(γ+1)) / ((1 − α)(γc + 1)).
- **Extra arithmetic** (work is wasted on rejected tokens): (1 − α)(γc + γ + 1) / (1 − α^(γ+1)).
- **Corollary 3.9:** if α > c, some γ gives a speedup of at least (1 + α)/(1 + c).
- **Bound** for free drafters (c ≈ 0): 1/(1 − α).
- **Table 1 (c = 0):**

| α | γ | Speedup |
|---|---|---|
| 0.6 | 2 | 1.96× |
| 0.7 | 3 | 2.53× |
| 0.8 | 2 | 2.44× |
| 0.8 | 5 | 3.69× |
| 0.9 | 10 | 6.86× |

### Numbers
- **Leviathan (T5-XXL 11B, TPU-v4):** "a 2X-3X acceleration compared to the standard T5X implementation, with identical outputs".
  - Best case: EnDe with a T5-small draft, T = 0, γ = 7, α = 0.75 → **3.4×**.
  - LaMDA 137B with an 8B draft: α = 0.75.
  - "c was always less than 0.05 and often negligibly close to 0."
  - A draft "around two orders of magnitude smaller than Mp usually performed best."
- **Chen et al.** (DeepMind, arXiv 2302.01318; Chinchilla 70B): "a 2–2.5× decoding speedup in a distributed setup".
  - 4B draft with 8 layers: "1.8ms/token compared to 14.1ms/token for Chinchilla".
  - HumanEval: 14.1 → 5.73 ms/token, **2.46×**.
- **Medusa** (ICML 2024, arXiv 2401.10774): extra decoding heads plus tree attention. "MEDUSA-1 can achieve over 2.2× speedup ... MEDUSA-2 further improves the speedup to 2.3-3.6×."
  - Medusa-1 (frozen backbone) is lossless. Medusa-2's optional "typical acceptance" is *not* distribution-exact.
- **EAGLE** (ICML 2024, arXiv 2401.15077) drafts at the feature level ("second-to-top-layer features"). "For LLaMA2-Chat 70B, EAGLE achieved a latency speedup ratio of 2.7x-3.5x, doubled throughput, while maintaining the distribution of the generated text."
- **EAGLE-2** (arXiv 2406.16858) uses a dynamic draft tree: "speedup ratios 3.05x-4.26x, which is 20%-40% faster than EAGLE-1".
- **EAGLE-3** (arXiv 2503.01840, 2025) uses "multi-layer feature fusion via a technique named training-time test". "a speedup ratio up to 6.5x, with about 1.4x improvement over EAGLE-2. In the SGLang framework, EAGLE-3 achieves a 1.38x throughput improvement at a batch size of 64."
- **Lookahead decoding** (ICML 2024, arXiv 2402.02057): Jacobi iterations fill an n-gram pool, and candidates are verified in parallel with no draft model. "up to 1.8x on MT-bench and 4x with strong scaling on multiple GPUs in code completion tasks."
- **Prompt-lookup decoding** (A. Saxena, GitHub, Nov 2023): "replace the draft model with simple string matching in the prompt". "significant speedups (2x-4x) in input-grounded tasks" such as summarization, document QA and code editing.
  - In HF transformers: `prompt_lookup_num_tokens=10`.
  - vLLM also offers n-gram and suffix speculators.
- **Multi-token prediction** (Gloeckle et al., Meta, ICML 2024, arXiv 2404.19737): "models trained with 4-token prediction are up to 3× faster at inference, even with large batch sizes". Code gets 3.0× with "2.5 accepted tokens out of 3 suggestions".
  - **DeepSeek-V3 MTP:** "the acceptance rate of the second token prediction ranges between 85% and 90%", "delivering 1.8 times TPS".
- **gpt-fast** (PyTorch 2023): CodeLlama-34B with a 7B draft gives "2x"; Llama-7B with a TinyLlama-1B draft gives "about a 1.3x boost".

### Batch size kills the free lunch
EAGLE-3 paper, Table 3: SGLang, H100, Llama-3.1-8B, MT-Bench; speedup vs no speculation at each batch size.

| Batch size | 2 | 4 | 8 | 16 | 24 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|---|
| EAGLE | 1.40× | 1.38× | 1.23× | 1.02× | 0.93× | 0.94× | 0.88× | 0.99× |
| EAGLE-3 | 1.81× | 1.82× | 1.62× | 1.48× | 1.39× | 1.32× | 1.38× | 1.38× |

Reason: speculation spends spare *compute* to save *memory passes*. As batch grows toward the ridge (§2), there is no spare compute left.

### Worked example (animate the accept/reject strip) [DERIVED]
Draft proposes γ = 4 tokens: "sat · on · the · mat". The target scores all 4 (plus one more position) in one pass.
1. "sat": q = 0.6, p = 0.7. q ≤ p, so accept.
2. "on": q = 0.5, p = 0.8. Accept.
3. "the": q = 0.9, p = 0.45. Accept with probability p/q = 0.5. The coin says reject.
4. Resample from norm(max(0, p − q)), which picks "a". Discard "mat".

Result: 3 tokens ("sat on a") for one target pass. Without speculation it would take 3 passes.

**Expected value, with α = 0.8, γ = 4, c = 0.05:**
1. E = (1 − 0.8⁵)/(1 − 0.8) = (1 − 0.32768)/0.2 = **3.36 tokens per target pass**.
2. Each round costs γc + 1 = 4 × 0.05 + 1 = 1.2 target-pass equivalents.
3. S = 3.36/1.2 = **2.80×**.

Sensitivity:
- c = 0.1 gives 2.40×.
- α = 0.6 (γ = 4, c = 0.05) gives 1.92×.
- α = 0.9, γ = 5 gives 3.75×.

### Misconceptions
- "Speculative decoding approximates the big model." With the rejection rule it is exact in distribution. (Some fast variants, like typical acceptance, are not.)
- "It only works with greedy decoding." The rule above is for sampling. Greedy is the special case.
- "Always on is always better." At high batch or low α it can be slower (EAGLE at batch 48: 0.88×).

### Quotable
- "...we can make exact decoding from the large models faster ... without changing the distribution." (Leviathan et al., 2023)
- "...modified rejection sampling scheme which preserves the distribution of the target model within hardware numerics." (Chen et al., 2023)
- "replace the draft model with simple string matching in the prompt to generate candidate token sequences." (Saxena, prompt-lookup decoding)

---

## §7 Kernels: FlashAttention, Flash-Decoding, fusion, CUDA graphs, torch.compile

### Mechanism
A GPU has a small, very fast on-chip SRAM and a large, slower HBM. FlashAttention paper (A100 figures): "40-80GB of high bandwidth memory (HBM) with bandwidth 1.5-2.0TB/s and 192KB of on-chip SRAM per each of 108 streaming multiprocessors with bandwidth estimated around 19TB/s."

**Standard attention** writes the N×N score matrix S and the probability matrix P out to HBM and reads them back.

**FlashAttention** (Dao et al., NeurIPS 2022, arXiv 2205.14135) is "an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM."
- It loads blocks of Q, K and V into SRAM.
- It computes partial softmax-weighted sums, keeping running statistics (max m and normaliser ℓ) to rescale as it goes. This is the "online softmax" trick (Milakov & Gimelshein 2018). FA cites that paper but does not use the phrase.
- It never materialises N×N in HBM, and recomputes in the backward pass instead of storing.
- HBM accesses drop from Ω(Nd + N²) to O(N²d²/M), where M is the SRAM size.

**FlashAttention-2** (arXiv 2307.08691) parallelises over sequence length and splits work better across warps.

**FlashAttention-3** (arXiv 2407.08608) targets the H100: warp specialisation with TMA producer/consumer warps, a "pingpong" that overlaps matmul and softmax, and FP8 with block quantisation and incoherent processing.

**Flash-Decoding** (Dao, Haziza, Massa, Sizov; Oct 2023) handles decode, where the query length is 1. Parallelism then comes only from batch × heads: "With a batch size of 1, FlashAttention will use less than 1% of the GPU!"
- Flash-Decoding splits the *KV sequence* into chunks processed in parallel: "First, we split the keys/values in smaller chunks."
- Each chunk writes its partial output and "the log-sum-exp of the attention values", and a final reduction rescales and sums them. This is split-K.

**FlashInfer** (MLSys 2025, arXiv 2501.01005) is a library of paged/block-sparse attention kernels with JIT-compiled variants and load-balanced scheduling. It is used by vLLM and SGLang.

**Fusion.** Merge chains of small ops so activations do not round-trip through HBM: RMSNorm with quantisation, RoPE with the QKV write into the paged cache, SwiGLU, sampling.

**CUDA Graphs.** Decode steps are tiny: about 5 ms for 8B on an H100, made of hundreds of kernels. The CPU can become the bottleneck just *launching* them.
- CUDA graphs record the whole step once and replay it with one launch: "a graph replay skips all layers of argument setup and kernel dispatch, including Python, C++, and CUDA driver overheads." (PyTorch blog, 2021)
- They need static shapes, so engines capture graphs for a set of batch sizes. In vLLM, `--max-cudagraph-capture-size` is "capped at 512 by default".
- vLLM's default mode is FULL_AND_PIECEWISE: "full CUDA Graph for uniform decode, piecewise CUDA Graphs for others".
- `--enforce-eager` disables CUDA graphs: "This disables both torch.compile and CUDA graphs."

**torch.compile** generates fused Triton kernels. Its `mode="reduce-overhead"` uses CUDA graphs.

### Numbers
- **FA1:** "15% end-to-end wall-clock speedup on BERT-large (seq. length 512) compared to the MLPerf 1.1 training speed record, 3× speedup on GPT-2 (seq. length 1K), and 2.4× speedup on long-range arena".
  - Attention itself is 7.6× faster (Fig. 1). Memory grows linearly in N, "up to 20× more memory efficient".
- **FA2:** "around 2× speedup compared to FlashAttention, reaching 50-73% of the theoretical maximum FLOPs/s on A100". FA1 managed "only 25-40%". Training reaches "up to 225 TFLOPs/s per A100 GPU (72% model FLOPs utilization)".
- **FA3:** "speedup on H100 GPUs by 1.5-2.0× with FP16 reaching up to 740 TFLOPs/s (75% utilization), and with FP8 reaching close to 1.2 PFLOPs/s".
  - FP8 FA3 has "2.6× lower numerical error than a baseline FP8 attention".
  - FA2 achieved "only 35% utilization on the H100".
- **Flash-Decoding:** "up to 8x faster generation for very long sequences" (CodeLlama-34B, 512 to 64K tokens).
- **FlashInfer:** "29-69% inter-token-latency reduction compared to compiler backends", "28-30% latency reduction for long-context inference".
- **PyTorch "GPT, Fast"** (LLaMA-7B, batch 1, A100-80GB at 330 W):

| Step | tok/s |
|---|---|
| Eager baseline | 25.5 |
| torch.compile + static KV cache | 107.0 |
| int8 weight-only | 157.4 |
| int4 GPTQ | 202.1 |
| int4 + speculative decoding | 244.7 |

  - "almost 10x faster than baseline", in "less than 1000 lines of native PyTorch code".
  - Diagnosis of the baseline: "it's heavily CPU overhead bound! What this means is that our CPU is not able to tell the GPU what to do fast enough".

### Worked example: online softmax with two tiles (animate) [DERIVED, checked]
Scores x = [1, 2 | 3, 0], processed in two tiles.
1. **Tile 1:** m = 2, ℓ = e^(1−2) + e^(2−2) = 0.3679 + 1 = 1.3679.
2. **Tile 2:** the new max is 3.
   - Rescale the old sum: 1.3679 × e^(2−3) = 0.5032.
   - Add the new terms: e^0 + e^(−3) = 1.0498.
   - ℓ = **1.5530**.
3. **Direct check:** Σe^(x−3) = e^(−2) + e^(−1) + e^0 + e^(−3) = 0.1353 + 0.3679 + 1 + 0.0498 = **1.5530**. ✓
4. Softmax = [0.087, 0.237, 0.644, 0.032]. The output accumulator is rescaled by the same factor e^(m_old − m_new).

**Why tiling matters.** N = 4,096 with FP16 scores gives a 33.5 MB S matrix per head per layer, or 1.07 GB for 32 heads. Standard attention writes it to HBM and reads it back; FlashAttention keeps it on-chip.

### Why kernels matter especially for decode
Decode moves few FLOPs, so it lives or dies on (a) streaming weights and KV at near-peak bandwidth and (b) not wasting time between kernels.
- gpt-fast went from 25.5 to 107 tok/s on the same GPU and model mainly by removing CPU overhead (compile + CUDA graphs) and using a static KV cache.
- vLLM V1's "up to 1.7x" came mostly from CPU-side work.
- [UNVERIFIED order of magnitude] If launches cost ~5–10 µs each, a few hundred launches per step eat a large share of a 5 ms step.

### Misconceptions
- "FlashAttention approximates attention." It is exact; the speed comes from IO.
- "FlashAttention speeds up decode." FA mostly helps prefill and training. Decode needs split-KV kernels (Flash-Decoding, FlashInfer).
- "torch.compile is about faster math." For decode it is mostly about fewer, fused launches.

### Quotable
- "an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM." (Dao et al., 2022)
- "With a batch size of 1, FlashAttention will use less than 1% of the GPU!" (Flash-Decoding blog, 2023)
- "it's heavily CPU overhead bound!" (PyTorch GPT-Fast blog)

---

## §8 Smaller models: quantization, distillation, pruning

### Mechanism
- **Weight-only quantization (W4A16, W8A16)** stores weights in fewer bits and dequantises on the fly inside the matmul. It cuts *bytes moved*, so it speeds up memory-bound decode. It does not speed up compute-bound prefill.
- **Weight-activation quantization (W8A8 INT8/FP8, W4A4/FP4)** also feeds low-precision activations to low-precision tensor cores. It cuts *FLOP time*, so it helps large-batch and prefill.
- AWQ: "W4A16 quantization can effectively improve the arithmetic intensity by 4×"; "weight-only quantization is more effective for on-device LLMs."
- Kurtic et al. (ACL 2025, arXiv 2411.02355): "W4A16 is the most cost-efficient for synchronous setups, while W8A8 dominates in asynchronous continuous batching."

**Group quantization** (animate). Each group of g weights shares a scale (and maybe a zero point): w ≈ scale × q.
- Overhead is (bits of scale + zero) ÷ g. Examples:
  - g = 32 with an FP16 scale: 4.5 bpw.
  - g = 64 with FP16 scale + bias (MLX's default 4-bit): 4.5 bpw.
  - MXFP4: g = 32 with an 8-bit E8M0 scale, so 4.25 bpw.
  - NVFP4: g = 16 with an FP8 E4M3 scale, so 4.5 bpw (plus a per-tensor FP32 scale).

### Formats and methods (numbers from the papers)

| Format / method | Bits/weight | What it does | Key result (source) |
|---|---|---|---|
| **LLM.int8()** (NeurIPS 2022) | 8 (W8A8 + FP16 outliers) | vector-wise INT8 plus mixed-precision decomposition of outlier dims (threshold α = 6.0); "more than 99.9% of values are multiplied in 8-bit" | "up to 175B parameters without any performance degradation"; outliers appear "At around 6.7B parameters". **Slower** than FP16 below 13B (matmul speed 0.14×–0.86×); BLOOM-176B 253 vs 239 ms/token at batch 1 |
| **SmoothQuant** (ICML 2023) | W8A8 INT8 | "migrating the quantization difficulty from activations to weights with a mathematically equivalent transformation"; s_j = max\|X_j\|^α / max\|W_j\|^(1−α), α = 0.5 | "up to 1.56x speedup and 2x memory reduction ... negligible loss in accuracy"; 530B in one 8×A100 node |
| **GPTQ** (ICLR 2023) | 3–4 (W-only) | one-shot second-order (Hessian) error compensation, column by column | 175B quantized "in approximately four GPU hours"; OPT-175B 3-bit on 1 A100: 71 vs 230 ms/token = **3.24×** (A6000: 4.53×) |
| **AWQ** (MLSys 2024 best paper) | 4 (W-only) | "Protecting only 1% salient weights can greatly reduce quantization error"; chooses per-channel scales from *activation* statistics; "does not rely on any backpropagation or reconstruction" | TinyChat "more than 3x speedup over the Huggingface FP16"; Llama-2-70B on a 64 GB Jetson Orin |
| **FP8 E4M3 / E5M2** (Micikevicius et al. 2022) | 8 | E4M3 max 448, no infinities, one NaN pattern; E5M2 max 57,344, IEEE-like | "E4M3 for weight and activation tensors, and E5M2 for gradient tensors" |
| **NF4** (QLoRA, NeurIPS 2023) | 4 + double-quant | "information theoretically optimal for normally distributed weights"; double quantization saves "about 0.37 bits per parameter (approximately 3 GB for a 65B model)" | 65B *fine-tuning* on one 48 GB GPU (a training claim) |
| **GGUF k-quants** (llama.cpp PR #1684, 2023) | Q2_K 2.5625 · Q3_K 3.4375 · Q4_K 4.5 · Q5_K 5.5 · Q6_K 6.5625 · Q4_0 4.5 · Q8_0 8.5 | 256-weight super-blocks of 16/32-weight blocks with quantized scales; Q4_K_M = Q6_K for half of attn.wv and ffn.w2, else Q4_K | LLaMA-7B wikitext PPL: F16 5.9066, Q4_K_M 5.9601 (+0.9%), Q6_K 5.9110, Q2_K 6.7764 |
| **MXFP4** (OCP MX v1.0, Sep 2023) | 4.25 | E2M1 elements (values ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}), shared E8M0 scale per 32 | gpt-oss: "post-trained with MXFP4 quantization of the MoE weights"; 120b (117B total / 5.1B active) fits "a single 80GB GPU"; 20b "within 16GB" |
| **NVFP4** (NVIDIA, 2025-06-24) | 4.5 | E2M1 + "E4M3 scaling factor to each 16-value micro-block" + per-tensor FP32 | "approximately 3.5x relative to FP16"; "1% or less accuracy degradation" on DeepSeek-R1-0528 (FP8 → NVFP4) |

**Accuracy summary** (Kurtic et al., Llama-3.1 family, "over 500,000 evaluations"): "(1) FP8 (W8A8-FP) is effectively lossless across all model scales, (2) well-tuned INT8 (W8A8-INT) achieves surprisingly low (1-3%) accuracy degradation, and (3) INT4 weight-only (W4A16-INT) is more competitive than expected, rivaling 8-bit quantization."

### Llama-3.1-8B memory by format [DERIVED; GGUF sizes from the llama.cpp quantize README]

| Format | Size |
|---|---|
| BF16 | 16.06 GB |
| FP8/INT8 | ≈ 8.0 GB (+ scales) |
| Q8_0 | 7.95 GiB |
| Q4_K_M (4.89 bpw) | 4.58 GiB ≈ 4.9 GB |
| MXFP4 | ≈ 4.3 GB (if all weights were MXFP4) |
| Q2_K | ≈ 2.6 GB |

The llama.cpp README table (same 8B model; hardware not stated) is a perfect roofline illustration:

| Format | Generation tok/s | Prompt tok/s |
|---|---|---|
| Q4_K_M | 71.93 | 821.81 |
| Q8_0 | 50.93 | 865.09 |
| F16 | 29.17 | 923.49 |

- Decode speeds up as bytes shrink: 2.5× from F16 to Q4_K_M.
- Prefill gets slightly *slower*, because dequantisation adds compute and prefill is already compute-bound.

### Worked example: int4 one group (animate) [DERIVED, checked]
Weights: [0.12, −0.50, 0.33, 0.05, −0.27, 0.48, −0.09, 0.21].
1. Symmetric int4 maps to [−7, 7]. scale = max|w|/7 = 0.50/7 = 0.0714.
2. q = round(w/scale) = **[2, −7, 5, 1, −4, 7, −1, 3]**.
3. Dequantised: [0.143, −0.500, 0.357, 0.071, −0.286, 0.500, −0.071, 0.214].
4. Error: max 0.027, RMSE 0.0185.
5. Storage: 8 × 4 bits + one FP16 scale = 48 bits, i.e. 6 bpw for g = 8. With g = 32 the same scheme costs 4.5 bpw.

Note how the largest-magnitude weight sets the scale for its whole group. Outliers are why LLM.int8(), SmoothQuant and AWQ exist.

### Distillation and pruning
- **Distillation for inference:** train a small student on a big teacher's outputs/logits so a cheaper model serves the traffic.
  - Meta Llama 3.2 1B/3B (2024-09-25): "structured pruning in a single shot manner from the Llama 3.1 8B", plus "logits from the Llama 3.1 8B and 70B models" as "token-level targets".
  - NVIDIA Minitron (arXiv 2407.14679): prune + distil with "<3%" of the original data and "up to 40x fewer training tokens per model compared to training from scratch".
- **2:4 structured sparsity** (Ampere and later): "In each contiguous block of four values, two values must be zero"; tensor cores then "complete the same effective calculation in half the time" (NVIDIA blog, 2021).
  - Real LLM gains are smaller. Sparse-Llama-3.1-8B-2of4 recovers 98.4% of dense accuracy for roughly 30% more vLLM throughput [approximate; sources vary].
  - **SparseGPT** (ICML 2023): "50% sparsity in one-shot, without any retraining"; "60% unstructured sparsity with negligible increase in perplexity" on OPT-175B/BLOOM-176B, "in under 4.5 hours".
  - **Wanda** (ICLR 2024) prunes "weights with the smallest magnitudes multiplied by the corresponding input activations", with no retraining.

### Misconceptions
- "4-bit = 4× faster." At batch 1 the best measured result is about 3.2–3.5× (GPTQ 3.24×; AWQ 3.2–3.3×). Prefill is not faster at all with weight-only formats.
- "INT8 is always faster than FP16." LLM.int8() was slower for models under 13B.
- "Q4 means 4.0 bits." Scales and mixed tensors push real formats to 4.25–4.9 bpw.
- "Quantization only saves memory." In memory-bound decode, saved bytes *are* saved time.

### Quotable
- "information theoretically optimal for normally distributed weights" (QLoRA, NF4)
- "Protecting only 1% salient weights can greatly reduce quantization error." (AWQ)
- "FP8 (W8A8-FP) is effectively lossless across all model scales" (Kurtic et al.)

---

## §9 Mixture-of-Experts at inference time

### Mechanism
Each MoE layer has E expert FFNs, and a router sends each token to k of them.
- **FLOPs per token** scale with *active* parameters.
- **Memory** must hold *all* parameters.
- **Bytes read per step** depend on batch:
  - At batch 1 only the chosen experts are read, which is cheap.
  - As batch grows, different tokens pick different experts, so the step reads the *union* of experts. That quickly approaches all of them.
  - Each expert, however, sees only about B·k/E tokens, so its matmul has low arithmetic intensity.
- **MoE therefore needs much larger global batches to become compute-bound.** That is why big MoE serving uses **expert parallelism (EP)**: experts are spread over many GPUs, and tokens travel to them via **all-to-all** "dispatch" and "combine" steps.

### Numbers
- **Mixtral 8x7B** (Mistral, 2023-12-11): "Mixtral has 46.7B total parameters but only uses 12.9B parameters per token"; "a router network chooses two of these groups"; 32K context.
  - Mistral: it "processes input and generates output at the same speed and for the same cost as a 12.9B model". This is true for compute, but the memory bill is 46.7B (93.4 GB in BF16).
- **DeepSeek-V3:** "671B total parameters with 37B activated for each token".
  - "Each MoE layer consists of 1 shared expert and 256 routed experts ... 8 experts will be activated for each token, and each token will be ensured to be sent to at most 4 nodes."
  - Load balancing is auxiliary-loss-free. FP8 weights take ≈ 671 GB, more than one 8×H100 node (640 GB).
  - Decode quote: "the batch size per expert is relatively small (usually within 256 tokens), and the bottleneck is memory access rather than computation."
  - Deployment in the paper: prefill unit of 4 nodes, EP32, with 32 redundant experts rebalanced periodically. Decode unit of 40 nodes / 320 GPUs, EP320, "each GPU hosts only one expert".
- **DeepSeek Open-Source Week inference overview** (Feb 2025; production differs from the paper):
  - Prefill EP32, decode EP144.
  - Per H800 node: 73.7k input tok/s in prefill and 14.8k output tok/s in decode.
  - Users see 20–22 tok/s.
  - Cost $87,072/day at "$2/GPU-hr", against a theoretical "cost profit margin of 545%". Actual revenue is lower.
- **gpt-oss-120b:** 117B total / 5.1B active, MXFP4 MoE weights, fits on a single 80 GB GPU.
- **Switch Transformer capacity factor:** "Expert Capacity: Batch size of each expert. Calculated as (tokens_per_batch / num_experts) * capacity_factor". Overflowing tokens are dropped, and "A larger capacity factor alleviates this overflow issue, but also increases computation and communication costs".
- **DeepEP** (DeepSeek, README at commit 458cdcb, 2025-03-05; H800, ~160 GB/s NVLink, 400 Gb/s CX7 IB ≈ 50 GB/s):
  - Intranode EP8 dispatch 153 GB/s, combine 158 GB/s.
  - Internode 43–47 GB/s RDMA.
  - Low-latency kernels (128 tokens, top-8, FP8): EP8 dispatch 163 µs, combine 318 µs; EP256 194 µs / 360 µs.

### Worked example: "how many experts does a batch touch?" (animate lighting up experts) [DERIVED]
Assume uniform independent routing: distinct experts per layer = E·(1 − (1 − k/E)^B).

| Batch B | Mixtral (E = 8, k = 2) | DeepSeek-V3 (E = 256, k = 8) | DS-V3 tokens per expert |
|---|---|---|---|
| 1 | 2 | 8 | 0.03 |
| 4 | 5.5 | 30.5 | 0.13 |
| 16 | 7.9 | 102 | 0.5 |
| 64 | 8 | 222 | 2 |
| 256 | 8 | 256 | 8 |

**Lesson:** Mixtral at batch ≥ 16 reads essentially all 46.7B parameters every step, yet each expert sees only ~4 tokens. To give each DeepSeek expert ~256 tokens per step (the scale of DeepSeek's quote) you need about 256 × 32 = **8,192 tokens in flight per step** across the EP group. Hence EP144–EP320 deployments with huge batches.

### Misconceptions
- "MoE 47B runs like a 13B model." Compute does; memory doesn't. Mid-size batches read nearly the whole model.
- "Experts specialise in topics, so routing is predictable." Load imbalance is a real serving problem, handled with redundant experts and EPLB.

### Quotable
- "Mixtral has 46.7B total parameters but only uses 12.9B parameters per token." (Mistral, 2023)
- "the batch size per expert is relatively small (usually within 256 tokens), and the bottleneck is memory access rather than computation." (DeepSeek-V3 report)

---

## §10 Parallelism for serving

### Mechanism
- **Data parallel (replicas).** N independent copies of the model, behind a load balancer. Throughput ×N, latency unchanged. This is the default if the model fits on one GPU.
- **Tensor parallel (TP).** Split every weight matrix across N GPUs, typically column-split then row-split in each MLP and attention block, with an all-reduce after each.
  - Each GPU reads 1/N of the weights per token, so **batch-1 latency drops**.
  - The cost is about 2 all-reduces per layer per step, which needs NVLink-class links.
  - kipply models the communication cost as ≈ 4·n_layers·8 µs per token at small batch.
- **Pipeline parallel (PP).** Put different layers on different GPUs or nodes. A single token still passes through every stage in sequence, so **PP doesn't cut per-token latency**.
  - It adds capacity (memory) and throughput when micro-batches keep all stages busy.
  - It is tolerant of slower inter-node links.
- **Expert parallel (EP).** Spread MoE experts across GPUs, with all-to-all token routing (§9).
- **Sequence / context parallel (SP/CP).** Split one long sequence's tokens across GPUs, e.g. ring attention for prefill, to cut TTFT for very long prompts. DeepSeek also uses "SP" alongside TP4 for attention.

### Interconnect numbers [DATED; vendor pages]

| Link | Bandwidth |
|---|---|
| H100 NVLink 4 | **900 GB/s** total per GPU (≈ 450 GB/s each way) |
| PCIe Gen5 x16 | **128 GB/s** total (≈ 64 each way) |
| A100 NVLink 3 | 600 GB/s |
| PCIe Gen4 | 64 GB/s |
| B200 NVLink 5 | 1.8 TB/s per GPU |
| GB200 NVL72 | a single "72-GPU NVIDIA NVLink™ domain", 130 TB/s total |
| InfiniBand CX7 | 400 Gb/s ≈ 50 GB/s per NIC (DeepEP README) |
| H800 NVLink | ~160 GB/s measured max (DeepEP README) |
| Mac Studio | RDMA over Thunderbolt 5 (macOS 26.2 driver). Apple's Oct 2026 post runs a 816B-total / 42B-active MoE on four M5 Ultras at 36.73 tok/s |

### Numbers
- **Databricks** (MPT-7B): "At batch size 1, going from 2x to 4x only reduces token latency by ~12%". At batch 16 the reduction is 33%. TP pays off more when there is enough work per GPU.
- **gpt-fast:** "80 tok/s" for Llama-70B with TP (int4 + spec).

### Worked example: serving Llama-3.1-70B [DERIVED]
1. **Size.** BF16 weights are 141.1 GB, which doesn't fit in one 80 GB H100.
2. **TP = 2.** 70.6 GB per GPU leaves about 1 GB for KV, which is useless.
3. **TP = 4.**
   - 35.3 GB per GPU, so about 36 GB per GPU is free for KV. Each GPU holds 2 of the 8 KV heads, so the 4 GPUs together hold ≈ 145 GB ÷ 320 KiB ≈ 440K tokens.
   - Aggregate bandwidth 13.4 TB/s, so the batch-1 ceiling is 13.4e12 ÷ 141.1e9 ≈ **95 tok/s**, before communication.
4. **TP = 8.** 17.6 GB per GPU and a ceiling of ≈ **190 tok/s**.
   - But there are 2 × 80 = 160 all-reduces per token. If each costs ~10 µs of latency [order of magnitude, UNVERIFIED], that adds ~1.6 ms to a ~5.3 ms step.
   - Scaling therefore falls short of linear, as Databricks observed.
5. **FP8 on TP = 4** halves the bytes, giving the same ≈ 190 tok/s ceiling on half the GPUs.

### Misconceptions
- "Twice the GPUs, half the latency." Communication grows, and PP gives no latency gain at all.
- "TP over PCIe is fine." At 64 GB/s per direction vs 450 GB/s on NVLink, all-reduces can dominate the step.

### Quotable
- "At batch size 1, going from 2x to 4x only reduces token latency by ~12%" (Databricks, 2023)

---

## §11 Long context at inference

### Mechanism and cost growth
- **Prefill FLOPs:** ≈ 2·P·n (linear part) + ≈ 2·L·n²·d (causal attention: QKᵀ and AV) → **quadratic in prompt length**.
- **Decode** cost per token grows **linearly** with context, because every step reads all n cached K/V. **Memory** also grows linearly (§3).

### Worked table: Llama-3.1-8B on one H100, prefill at 50% MFU [DERIVED]

| Prompt length | Attention share of FLOPs | TTFT (compute only) |
|---|---|---|
| 1K | 1.6% | 33 ms |
| 8K | 12% | 0.29 s |
| 32K | 34% | 1.6 s |
| 128K | **68%** | **13.4 s** |

- At 128K: linear part 2.11 PFLOP + attention 4.50 PFLOP = 6.61 PFLOP.
- Decode floor at 128K is 9.9 ms/token vs 4.9 ms at 2K, and only ~3 such users fit on one H100.

### Techniques
- **Attention sinks / StreamingLLM** (Xiao et al., ICLR 2024, arXiv 2309.17453).
  - Initial tokens act as a "sink" "even if they are not semantically important". The cause is "the Softmax operation, which requires attention scores to sum up to one for all contextual tokens".
  - Keep the KV of 4 initial tokens plus a rolling window of recent tokens: "keeping the KV of initial tokens will largely recover the performance of window attention".
  - It runs stably "with up to 4 million tokens and more", with "up to 22.2x speedup" over sliding-window recomputation.
  - *Caveat:* it streams indefinitely but does **not** remember evicted content. It is not a context-extension method.
- **Architectural options:** sliding/local layers (Gemma 3's 5:1 local:global, Mistral SWA), MLA, KV quantization (§3).
- **Systems options:** chunked prefill (§5), context parallelism for prefill (§10), KV offload and prefix reuse (§3, §4).
- **Context caching APIs** [DATED; official docs read 2026-10-23]:
  - **Anthropic:** "Cache write tokens with a 5-minute TTL are 1.25 times the base input tokens price"; 1-hour TTL writes are 2×; "Cache read tokens are 0.1 times the base input tokens price".
    - Footnotes: Opus 5.5 and Sonnet 5.5 hits are 0.05×; Fable/Mythos 5.1 are 0.025×.
    - Minimum is 512 tokens on the newest models. "By default, the cache has a 5-minute lifetime. The cache is refreshed for no additional cost each time the cached content is used."
  - **OpenAI:** "Prompt caching is enabled by default"; cached input is "discounted up to 95%".
    - Minimum is 1,024 tokens (GPT-5.6+). Routing is by "A hash of the initial tokens"; `prompt_cache_key` influences routing.
    - Per-model discounts on the pricing page: gpt-4o 50%, gpt-4.1 75%, gpt-5.5 90%, gpt-6.1-sol 95%.
  - **Google Gemini:** "Implicit caching is enabled by default for all Gemini 2.5 and newer models".
    - Minimum is 2,048 tokens (2.5) or 4,096 (newer).
    - Pricing page (updated 2026-10-22, promotional): Gemini 4 Argon cached input is 0.05× input; Gemini 3.8 Flash is 0.1× plus storage of $0.50 per MTok per hour.

### Worked example: Owl caches a 20K-token textbook chapter [DERIVED]
A student asks 50 questions within the cache TTL. Model: Sonnet 5.5 at $2/MTok input [DATED].
1. **Without caching:** 50 × 20K × $2/M = **$2.00**.
2. **With caching:** one 5-minute write at 1.25× ($0.05), then 49 reads at 0.05× (49 × 20K × $0.10/M = $0.098). Total **$0.148**, i.e. **−93%**.
3. **Server side:** each hit skips ≈ 0.43 PFLOP of prefill for an 8B-class model (0.32 linear + 0.11 attention; ≈ 0.87 s of H100 time at 50% MFU). That is why the TTFT drops too.

### Misconceptions
- "If it fits in the context window, it's free." TTFT grows quadratically and TPOT linearly, and long-context users crowd out batch slots.
- "StreamingLLM gives infinite memory." It gives infinite *streaming*; the middle is forgotten.
- "Cached tokens are free." There are write premiums, storage fees (Gemini explicit caching) and TTL expiry.

### Quotable
- "keeping the KV of initial tokens will largely recover the performance of window attention" (StreamingLLM, ICLR 2024)
- "The cache is refreshed for no additional cost each time the cached content is used." (Anthropic docs, Oct 2026)

---

## §12 Structured output, sampling cost, and determinism

### Mechanism
**Constrained decoding.** At each step, a grammar engine computes which vocabulary tokens keep the output valid (JSON schema, regex, CFG). It sets every other logit to −∞, then samples.
- The mask is a 128,256-bit vector for Llama 3, i.e. 16 KB per sequence per step.
- **Cost problem:** checking every token against the grammar each step.
  - Outlines (Willard & Louf, arXiv 2307.09702): the naive approach "entails a fixed O(N) cost for each token generated, where N is the size of the LLM's vocabulary".
  - Outlines precompiles the regex or schema into an FSM and builds "an index over a language model's vocabulary", so each step "costs O(1) on average".
- **XGrammar** (Dong et al., MLSys 2025, arXiv 2411.15100) handles nested grammars like JSON with a stack-based (pushdown) matcher.
  - It splits the vocabulary "into context-independent tokens that can be prechecked and context-dependent tokens" that need runtime checks.
  - It overlaps grammar work with GPU compute.
  - "up to 100x speedup over existing solutions"; "near-zero overhead structure generation in end-to-end ... LLM serving".
- **llguidance** (Microsoft/guidance-ai): "on the order of 50μs of CPU time per token (for 128k tokenizer)". It is used by vLLM, SGLang, llama.cpp and OpenAI Structured Outputs.
- **vLLM backends:** xgrammar or guidance; "The default backend is `auto`".
- **Cost to watch:** grammar *compilation* the first time a schema appears (cache it), plus CPU time per step at high batch.

**Sampling cost.** Top-k/top-p normally sorts a 128K-entry vocabulary per sequence per step.
- FlashInfer (2025-03-10): "categorical sampling (token selection) has emerged as a significant performance bottleneck". Its sorting-free rejection-sampling kernels "reduce the overall sampling time by more than 50%" in vLLM on an H100.

**Determinism.** Thinking Machines Lab (Horace He et al., 2025-09-10): temperature-0 outputs vary *because kernels are not batch-invariant*. Server load changes the batch size, which changes reduction order in RMSNorm, matmul and attention.
- "the forward pass of an LLM involves no operations that require atomic adds."
- Qwen3-235B, 1,000 runs at T = 0: "we generate 80 unique completions, with the most common of these occuring 78 times".
- With batch-invariant kernels all 1,000 runs were identical, at a cost of 26 s → 55 s unoptimised, or 42 s with a better attention kernel (Qwen3-8B, vLLM).

### Worked example: a JSON mask (animate) [DERIVED]
Schema: `{"answer": "yes" | "no"}`. The output so far is `{"answer": "`.
1. The model's top candidates are "yes" 0.50, "maybe" 0.30, "no" 0.20.
2. The grammar allows only "yes" and "no", so "maybe" is masked to −∞.
3. Renormalised: yes 0.50/0.70 = **0.714**, no **0.286**.

The format is guaranteed, but the distribution was bent. That is the hidden quality cost when a schema fights what the model wants to say.

### Misconceptions
- "Temperature 0 is deterministic on a server." Not unless the kernels are batch-invariant.
- "Constrained decoding is slow." With modern engines the per-token overhead is near zero; first-use compilation is the visible cost.
- "JSON mode = schema-valid." JSON mode guarantees syntax; schema-constrained decoding guarantees structure.

### Quotable
- "Surprisingly, we generate 80 unique completions, with the most common of these occuring 78 times." (Thinking Machines, 2025)
- "up to 100x speedup over existing solutions" (XGrammar, 2024)

---

## §13 The serving stack

### Layers (animate as a stack, top to bottom)
1. **Client SDK.** Sends OpenAI-compatible requests.
2. **API gateway.** Auth, rate limits, quotas.
3. **Router / load balancer.** Round-robin, least-loaded, or prefix/KV-aware.
   - OpenAI routes by "A hash of the initial tokens"; llm-d and Dynamo do "prefix-aware" / "KV-aware routing".
4. **Engine replicas.** Tokenizer → scheduler → model runner (kernels, CUDA graphs) → sampler → detokenizer.
5. **KV and cache tier.** GPU → CPU → SSD → remote.
6. **Autoscaler and observability.**

### Engines [DATED: latest releases per GitHub, 2026-10-23]

| Engine | Release | Notes |
|---|---|---|
| **vLLM** | v0.32.0 | V1 is the only engine since v0.11.0 ("V1 is the only engine in the codebase now"). PagedAttention, continuous batching, chunked prefill, prefix caching. Speculative methods: n-gram, suffix, EAGLE, DFlash. "OpenAI-compatible API server, plus Anthropic Messages API and gRPC support" |
| **SGLang** | v0.5.22 | RadixAttention, large-scale EP/PD; "optimized for agentic workloads, large-scale serving, and RL rollouts"; lists Apple Silicon via Metal/MLX |
| **TensorRT-LLM** | v1.2.1 stable (2026-04) | NVIDIA. Docs include "Migration Guide: TensorRT Backend Removed"; now a PyTorch-based executor [inferred from docs] |
| **HF TGI** | – | "text-generation-inference is now in maintenance mode" (dated 2025-12-11); repo archived 2026-03-21. Recommends vLLM/SGLang, or llama.cpp/MLX locally |
| **llama.cpp** / llama-server | v0.7.0 | GGUF; "Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks"; OpenAI-compatible chat/responses/embeddings; continuous batching; speculative decoding; `--grammar`; mmap loading. Default port: the flag table says 9931 [UNVERIFIED; historically 8080] |
| **MLX / mlx-lm** (Apple) | mlx-lm v0.32.2 | `mlx_lm.generate`, `.chat`, `.convert`, `.cache_prompt`, `.server`, `.benchmark`; flags `--kv-bits`, `--max-kv-size` (rotating cache), `--draft-model`, `--num-draft-tokens`, `--prompt-cache-file`, `--prefill-step-size` |
| **Ollama** | v0.40.9 | Started on llama.cpp; own engine announced 2025-05-15. On Mac, "now built on top of Apple's machine learning framework, MLX" (2026-03-30, preview) |
| **LM Studio** | – | GUI; OpenAI-compatible at "http://localhost:1234/v1"; MLX engine since 0.3.4 (2024-10-08): "You can even mix and match llama.cpp and MLX models!" |
| **NVIDIA Dynamo** | v1.6.0 | "successor to NVIDIA Triton Inference Server"; disaggregated serving, KV-aware routing |
| **llm-d** | v1.0.0 | Kubernetes-native; vLLM + Inference Gateway; prefix-aware routing, P/D disaggregation |

### OpenAI-compatible streaming (SSE), from the official OpenAPI spec
- With `stream: true` the server "returns server-sent events (`text/event-stream`)".
- Each event is `data: {"object":"chat.completion.chunk", "choices":[{"delta":{"content":"Hello"}, "finish_reason":null}]}`. The final event carries `"finish_reason":"stop"`, then `data: [DONE]`.
- `stream_options.include_usage` adds a usage chunk before `[DONE]` with an empty `choices` array.
- The newer Responses API uses typed events instead (`response.output_text.delta`, `response.completed`).

### Autoscaling and cold starts
- **GKE inference autoscaling guide** (updated 2026-10-22):
  - "Use queue size to maximize throughput and minimize cost within a certain target latency threshold".
  - "We recommend choosing batch size-based autoscaling if you have latency-sensitive workloads".
  - GPU utilization "Does not measure how much work is being done while the GPU is active."
- **Cold start** = provision the node + pull the container + load weights + allocate KV + capture CUDA graphs and compile.
  - **NVIDIA Run:ai Model Streamer** (2025-09-16; Llama 3 8B, 15 GB, A10G):
    - GP3 SSD load: HF safetensors 47.99 s vs streamer 14.34 s.
    - Total time until vLLM is ready: 66.13 s vs 35.08 s, or 23.18 s streaming from S3.
  - **vLLM sleep mode** keeps the process alive, "preserving the allocator, CUDA graphs, and JIT-compiled kernels". Level 1 "will offload the model weights and discard the KV cache".
    - Blog (2025-10-26, A100): wake in 0.85–2.58 s; first request 0.92 s vs 3.72 s from cold.
    - Without sleep mode a model switch takes 30–100+ s.

### Worked example: the cold-start budget for Owl's 16 GB model [DERIVED]
Weight-loading time alone, by source bandwidth:
- 1 GB/s disk: 16 s.
- NVMe at 5 GB/s: 3.2 s.
- 10 GbE network: 12.8 s.
- 100 GbE: 1.3 s.

Add CUDA-graph capture and compilation (seconds to tens of seconds) and node or image provisioning (often minutes; UNVERIFIED typical). **Lesson:** scale on queue depth *before* SLOs break, and keep warm or sleeping replicas.

### Misconceptions
- "OpenAI-compatible means identical." Engines differ in which parameters they honour (logprobs, tool calling, `n`, grammar extensions).
- "Autoscale on GPU utilization." Prefer queue size or batch size, per Google's guide.

---

## §14 On-device and local inference

### Mechanism
**Apple silicon has unified memory.** CPU, GPU and Neural Engine share one LPDDR pool, so the GPU can use most of system RAM for weights.
- This is why a 128 GB MacBook Pro can hold models that need two H100s.
- Its bandwidth (120 GB/s to 1.2 TB/s) is 3–30× below HBM, so **decode tok/s = bandwidth ÷ bytes** dominates everything (§2).
- Prefill is compute-bound. It jumped with M5's per-core GPU "Neural Accelerator" (TTFT "up to 4x" vs M4, per Apple, 2025-11-19).

**Software.**
- **MLX / mlx-lm:** Apple's array framework, lazy evaluation, unified-memory native. Models come from the `mlx-community` HF org; 4-bit uses group 64 with a scale and bias, ≈ 4.5 bpw.
- **llama.cpp:** GGUF k-quants, Metal backend, mmap. Ollama and LM Studio wrap these engines.
- **Wired-memory knob:** `sudo sysctl iogpu.wired_limit_mb=N`, where N is "larger than the size of the model in megabytes but smaller than the memory size of the machine" (mlx-lm README, macOS 15+). The default GPU-usable fraction of RAM is below 100% [exact default UNVERIFIED].

### What fits (≈ 4.5 bpw weights, plus KV and OS headroom) [DERIVED]

| Mac RAM | Comfortable 4-bit model | Weights |
|---|---|---|
| 16 GB | 8B | 4.5 GB |
| 24–32 GB | 14B | 8.3 GB |
| 32 GB | 32B (tight) | 18.4 GB |
| 64 GB | 70B | 39.7 GB |
| 128 GB | gpt-oss-120b in MXFP4 (≈ 63 GB, UNVERIFIED size) or 70B at 8-bit | – |
| 512 GB (M3/M5 Ultra) | DeepSeek-V3 671B at 4.5 bpw | ≈ 377 GB |

- Apple claims M3 Ultra can run "LLMs with over 600 billion parameters" and that M4 Max handles "nearly 200 billion parameters".
- Remember the KV: Llama-8B at 32K context adds 4 GiB (2 GiB with `--kv-bits 8`).

### Estimates vs measurements (LLaMA-2-7B Q4_0 decode, llama.cpp discussion #4167)

| Chip | Ceiling (BW ÷ 3.79 GB) | Measured | Ratio |
|---|---|---|---|
| M1 | 18 tok/s | 14.15 | 79% |
| M4 | 32 | 24.11 | 76% |
| M4 Pro | 72 | 50.74 | 70% |
| M4 Max | 144 | 83.06 | 58% |
| M5 Max | 162 | 119.92 | 74% |
| M5 Ultra | 317 | 179.10 | 57% |

**Rule of thumb for narration:** "expect 55–85% of bandwidth ÷ model size".

### Phones and NPUs
- **Apple Foundation Models, 2024** (2024-06-10): "~3 billion parameter on-device language model"; on iPhone 15 Pro, "time-to-first-token latency of about 0.6 millisecond per prompt token" and a 30 tok/s generation rate; mixed 2/4-bit quantization averaging 3.7 bpw.
- **Apple Foundation Models, 2025** (2025-06-09):
  - "2 bits per weight (bpw) using Quantization-Aware-Training"; 8-bit KV cache.
  - The model is split "into two blocks with a 5:3 depth ratio". Block 2 reuses block 1's final-layer KV caches, "reducing the KV cache memory usage by 37.5% and significantly improving the time-to-first-token."
- **Gemma 3n** (Google, docs updated 2025-11-01): "E" means *effective* parameters. E2B loads "over 5 billion parameters" in standard execution. With Per-Layer-Embedding caching and parameter skipping, the effective memory load is 1.91B.
- **NPUs** (M4 Neural Engine "up to 38 trillion operations per second") help compute-bound prefill. Phone decode is still bandwidth-bound, and Apple does not publish iPhone memory bandwidth [UNVERIFIED].

### Worked example: "will Owl run on my Mac?" [DERIVED]
Machine: a MacBook Pro M4 Pro with 48 GB.
1. Owl-8B in 4-bit = 4.5 GB, so it fits easily.
2. Decode ceiling: 273 ÷ 4.52 = 60 tok/s. At 70–85% MBU, expect **42–51 tok/s**. The measured 7B Q4_0 figure is 50.7.
3. A 2,000-token pasted chapter takes ≈ 2,000 ÷ ~400–450 tok/s ≈ **4–5 s TTFT**, using M4 Pro PP512 ≈ 440 tok/s for 7B Q4_0 as a proxy.
4. On an M5 Pro, the same prefill would be ≈ 1.3 s (PP ≈ 1,620).

### Misconceptions
- "The Neural Engine runs my local LLM." mlx-lm and llama.cpp run on the GPU. NPUs are used by Core ML and Apple's own models.
- "A 70B model needs a datacenter." In 4-bit it is 40 GB, which fits a 64 GB Mac, at ~5–12 tok/s depending on bandwidth [DERIVED: 273–546 GB/s ÷ 39.7 GB × 0.7–0.85].

### Quotable
- "Generating subsequent tokens is bounded by memory bandwidth, rather than by compute ability." (Apple ML Research, 2025)
- "Apple silicon is a first-class citizen" (llama.cpp README)

---

## §15 Cost and capacity

### First principles
- **Cost per million output tokens** = (GPU $/hour) ÷ (output tokens/hour ÷ 10⁶).
- **Tokens/hour** = aggregate tokens/s × 3,600 × utilisation of the fleet (idle time costs money too).
- **Rental prices** [DATED Oct 2026]:
  - Lambda (read 2026-10-23): H100 SXM $3.99 per GPU-hr.
  - RunPod ("Updated October 8, 2026"): H100 SXM $3.99 Secure, $2.69 Community.
  - AWS p5.48xlarge on demand ≈ $55.04/hr for 8 GPUs ≈ $6.88 per GPU-hr. AWS cut P5 on-demand prices 44% from 2025-06-05.
  - Use $2.50–$4 per GPU-hr as the illustrative band.
- **DeepSeek's own accounting:** "$2/GPU-hr" for H800s.

### Worked example: Owl on one H100 [DERIVED]
Model: Llama-8B BF16, 2K average context, 75% MBU. Step time = (16.06 GB + B × 256 MiB KV) ÷ (3.35 TB/s × 0.75).

| Batch | TPOT | Per-user tok/s | Server tok/s | $/M output @ $2.69 | @ $3.99 |
|---|---|---|---|---|---|
| 1 | 6.5 ms | 154 | 154 | $4.86 | $7.20 |
| 8 | 7.3 | 138 | 1,104 | $0.68 | $1.00 |
| 32 | 9.8 | 102 | 3,262 | $0.23 | $0.34 |
| 64 | 13.2 | 76 | 4,838 | $0.15 | $0.23 |
| 128 | 20.1 | 50 | 6,378 | $0.12 | $0.17 |
| 256 | 33.7 | 30 | 7,587 | $0.10 | $0.15 |

- **The curve to animate:** cost per token falls ~50× from batch 1 to 256, while per-user speed falls ~5×. There is a knee around batch 32–64.
- The batch-256 row is hypothetical: 256 × 2K tokens needs 64 GiB of KV, and 68.7 GB + 16.06 GB of weights = 84.8 GB > 80 GB. Memory caps the batch (≈ 200 here) before compute does.
- Real engines add overheads; LMSYS reports DeepSeek on H100s at "$0.20/1M output tokens".

### Why output tokens cost more than input tokens
- **Prefill** at 50% MFU processes ≈ 494.5e12 ÷ 16.06e9 ≈ **30,800 tokens/s** on one H100, i.e. **$0.036/M input tokens** at $3.99/hr.
- **Decode** at batch 64 costs $0.23/M, about **6× more**; at batch 1 it is 200× more.
- Prefill amortises one weight read over thousands of tokens; decode amortises it over only B tokens.
- Published API ratios reflect this [DATED Oct 2026, official pricing pages]:

| Model | Input $/MTok | Output $/MTok | Ratio |
|---|---|---|---|
| Anthropic Opus 5.5 | 4 | 20 | 5× |
| OpenAI gpt-5.5 | 5 | 30 | 6× |
| OpenAI gpt-4.1 | 2 | 8 | 4× |
| Gemini 4 Argon (promo) | 2 | 10 | 5× |

### Utilisation metrics
- **MBU** (Databricks definition, §2). Databricks measured "MBU of 60% on 2xH100-80GBs" at batch 1, and gpt-fast reached 72%.
- **MFU** applies to prefill and large batches. FA2 training reaches 72%, FA3 attention 75%.
- **Batch vs latency** (Databricks): "if we maximize throughput with a batch size of 64, latency increases by 4x while throughput increases by 14x".

### Capacity planning with Little's law
**L = λ × W**: concurrent requests = arrival rate × time in system.

Owl at peak, worked example [DERIVED, fixed-point iteration]:
1. Inputs: λ = 20 requests/s, 300-token answers, TTFT ≈ 0.3 s.
2. Solve B = 20 × (0.3 + 300 × TPOT(B)). It converges at **B ≈ 124 concurrent sequences**.
3. That gives TPOT ≈ 19.6 ms and E2E ≈ 6.2 s, with 33 GB of KV, all on **one H100**.
4. If the SLO is TPOT ≤ 15 ms, cap the batch at ~85 (TPOT 15 ms) and serve the rest from a second replica.

### Benchmarking methodology
- **Tools:**
  - `vllm bench serve` reports TTFT, TPOT, ITL and E2EL percentiles (`--metric-percentiles`, default 99). Use `--request-rate` with `--burstiness 1.0` for Poisson arrivals, `--max-concurrency`, and `--goodput ttft:… tpot:…`.
  - NVIDIA **AIPerf** (successor to GenAI-Perf).
  - **GuideLLM** (vLLM project): sweeps synchronous, constant, poisson and throughput profiles.
  - **LLMPerf** was archived 2025-12-17.
- **Rules:**
  1. Warm up (compile, graphs, caches).
  2. Use realistic input and output length *distributions*, not one fixed length.
  3. Use open-loop Poisson arrivals for latency SLOs and closed-loop max-concurrency for capacity.
  4. Report p50, p90 and p99, not means. Report goodput at the SLO, not peak throughput.
  5. Fix the prefix-cache state, either cold or a known hit rate.
  6. Count reasoning tokens consistently. AIPerf's TTFT counts reasoning tokens, unlike GenAI-Perf.
- **Industry SLOs (MLPerf):**
  - Llama-2-70B server scenario: "TTFT: <= 2 seconds", "TPOT: <= 200 milliseconds" (≈ "~240 words per minute").
  - v5.0 interactive: "99th percentile TTFT limit of 450ms" and "TPOT of 40ms".
  - Llama-3.1-405B: p99 TTFT 6 s, p99 TPOT 175 ms.

### Misconceptions
- "Throughput benchmark = user experience." Peak throughput is usually measured at a latency nobody would accept.
- "Average latency is fine." Users feel p99. The hiccups from un-chunked prefill (§5) show up only in ITL tails.

---

## §16 Observability and reliability

### Metrics (vLLM names, docs "Production Metrics") [DATED; names change, so check your version]

| Signal | Metric |
|---|---|
| Latency | `vllm:time_to_first_token_seconds`, `vllm:inter_token_latency_seconds`, `vllm:request_time_per_output_token_seconds`, `vllm:e2e_request_latency_seconds` (histograms) |
| Queueing | `vllm:request_queue_time_seconds`; gauges `vllm:num_requests_running`, `vllm:num_requests_waiting` |
| Memory | `vllm:kv_cache_usage_perc` (1 = 100%). Older releases used `gpu_cache_usage_perc` [UNVERIFIED in current docs] |
| Preemption | `vllm:num_preemptions` (counter), `vllm:request_num_preemptions` |
| Prefix cache | `vllm:prefix_cache_hits` / `vllm:prefix_cache_queries` (in tokens) |
| Plus | token counters, finished-reason counts, GPU power and temperature (DCGM) |

### Preemption and swapping
When the KV cache is full and a running sequence needs a new block, the scheduler **preempts** a request. It either frees the blocks and **recomputes** later, or **swaps** them to CPU.
- vLLM log: "Sequence group 0 is preempted by PreemptionMode.RECOMPUTE mode because there is not enough KV cache space."
- V1 default: "the default preemption mode is `RECOMPUTE` rather than `SWAP`", because "recomputation has lower overhead in the V1 architecture."
- Suggested fixes: raise `gpu_memory_utilization`, or lower `max_num_seqs` / `max_num_batched_tokens`.
- Frequent preemptions show up as TTFT/E2E tail spikes and wasted prefill compute.

### Reliability patterns
- **Admission control:** a bounded queue, then HTTP 429 with Retry-After once queue time would break the SLO.
- **Timeouts** at every layer, and **client-disconnect detection** so a cancelled stream stops decoding and frees its KV.
- **Cap `max_tokens`**, because a runaway generation holds a batch slot and KV for minutes.
- **Graceful degradation under load:**
  - shorten `max_tokens`;
  - turn off speculative decoding (it hurts at high batch, §6);
  - route to a smaller or quantized model;
  - disable expensive features (`n` > 1, long context).
- **Health and readiness probes** that wait for weights loaded and graphs captured. Warm spares, plus sleep mode for fast swap (§13).
- **Canaries for numerics:** fixed prompts with known outputs, aware of nondeterminism (§12).

### Worked example: diagnosing a p99 TTFT alert (animate as a dashboard)
1. **Symptom:** TTFT p99 jumps 0.4 s → 3 s; TPOT p50 is flat; `num_requests_waiting` climbs; `kv_cache_usage_perc` ≈ 0.98; `num_preemptions` rising.
2. **Diagnosis:** the KV cache is saturated, so new requests queue and running ones get preempted and recomputed. Throughput is not compute-limited.
3. **Fixes:** add a replica (scale on queue size); enable FP8 KV (doubles slots, §3); cap the context per request; route long-context users to a dedicated pool.

### Quotable
- "Use queue size to maximize throughput and minimize cost within a certain target latency threshold" (GKE docs, 2026)

---

## §17 Topics worth adding (not in the brief)

1. **Multi-LoRA serving.** One base model, thousands of fine-tuned adapters, batched together.
   - S-LoRA (Sheng et al., arXiv 2311.03285; MLSys 2024 [venue from memory]): "Unified Paging uses a unified memory pool to manage dynamic adapter weights with different ranks". Up to 4× throughput vs HF PEFT / naive vLLM, and "thousands of LoRA adapters on a single GPU".
   - The vLLM prefix-cache hash includes the LoRA ID.
2. **Reasoning and agent workloads shift cost to decode.** Long hidden reasoning traces mean more output tokens and longer KV.
   - DeepSeek's 24 h production window: 608B input tokens (56.3% cache hits), 168B output tokens, average "KV cache length per output token" 4,989.
   - Agents re-send growing prefixes, so prefix caching and KV tiering matter most (cross-reference "Agents, Animated").
3. **Determinism and numerics** (§12). An episode beat on "why the same prompt gives different answers on a busy server".
4. **Tokenizer effects on cost.** The same text costs different token counts per tokenizer and language. Commonly cited for OpenAI tokenizers: ~4 characters ≈ 1 token ≈ ¾ of an English word [UNVERIFIED: the official help page returned 403].
5. **Energy per token** [not researched; flag for later if wanted].

---

## Appendix A: Proposed course, 25 episodes in 6 modules

### The running example: **Owl**, a study-buddy chatbot that grows up
Students paste notes and ask questions; Owl answers and makes JSON flashcards.

**The model.** Owl runs **Llama-3.1-8B-Instruct** ("Owl-8B"). It gives the course three round numbers to repeat: **8B params, 16 GB of weights, 128 KiB of KV per token**. It is open, uses GQA, and has MLX and GGUF builds. Variants:
- **Owl-mini** (Llama-3.2-1B-Instruct, 4-bit, ≈ 0.7 GB, 32 KiB/token) for live Mac demos.
- **Owl-Pro** (a MoE or 70B) for the scale-up module.

**The story arc mirrors the course:**
1. One student on a MacBook (M1).
2. The cache and memory (M2).
3. Launch day: 100 → 10,000 students on H100s (M3).
4. Making it fast (M4).
5. Smaller and bigger Owls (M5).
6. Production, cost, and back to the Mac (M6).

Every episode ends with an Owl number: tok/s, $/M tokens, users per GPU, TTFT.

### Module 1: One token, two speed limits

| # | Title | The one thing the viewer will understand | Mechanism to animate | Owl beat |
|---|---|---|---|---|
| 1 | Press Enter | A reply is one parallel prefill plus hundreds of one-token decode steps; TTFT ≠ TPOT | Timeline: tokenize → thick prefill bar → 300 thin decode ticks → SSE stream | 1,500-token prompt: TTFT ≈ 50 ms, 300 tokens ≈ 2 s; decode ≈ 97% of time (§1) |
| 2 | The two speed limits | Every step is limited by math *or* memory traffic; for chat it's memory | Roofline chart; a "bytes truck" carrying 16 GB per token; tok/s = bandwidth ÷ bytes | H100 208 tok/s ceiling vs M4 Max 34 (§2) |
| 3 | Prefill vs decode | The same model runs two different workloads: prefill is compute-bound (intensity ≈ prompt length), decode memory-bound (≈ batch) | Two dots sliding on the roofline; M5 vs M4: PP ×3.4, TG ×1.17 | llama.cpp Mac table (§2) |
| 4 | The batching free lunch | Serving 64 users costs about the same weight traffic as serving 1 | One weight sweep feeding 1 → 64 rows; cost-per-token curve | $7.20 → $0.23 per M tokens (§15) |

### Module 2: Memory, the KV cache

| # | Title | The one thing | Mechanism to animate | Owl beat |
|---|---|---|---|---|
| 5 | The KV cache | Models remember past tokens as K/V tensors; the size formula | Cache growing token by token, layer by layer; 2·L·n_kv·d_h·b | 128 KiB/token; 128K chat = 16 GiB = the weights (§3) |
| 6 | Shrinking the cache | MQA → GQA → MLA → sliding windows → KV quantization | Heads merging; latent compression; rolling buffer | 52 → 104 users with FP8 KV; DeepSeek-V3 < Llama-8B per token (§3) |
| 7 | PagedAttention | Fixed-size blocks and a block table end fragmentation; copy-on-write enables sharing | Memory map: contiguous reservations (holes) vs 16-token blocks; CoW on fork | 85% waste → 1.3%; n = 4 answers save 63% (§4) |
| 8 | Never prefill twice | Prefix caching and radix trees; why provider cache pricing exists | Radix tree lighting up shared prefixes; hash chain over blocks | Shared system prompt; 50 questions on a cached chapter: −93% cost (§4, §11) |
| 9 | The cost of a long memory | Prefill grows quadratically, decode linearly; attention sinks; offload | Attention-share bar growing; sinks + rolling window; KV moving to CPU/SSD | 128K prompt ≈ 13 s TTFT on one H100 (§11) |

### Module 3: Many users at once

| # | Title | The one thing | Mechanism to animate | Owl beat |
|---|---|---|---|---|
| 10 | Continuous batching | Re-form the batch every step, not every request | 4-slot grid: static vs iteration-level | E waits 12 steps → 5 (§5) |
| 11 | Chunked prefill | Long prompts must not freeze everyone else; the token budget dial | Step-by-step token budget filled by decodes, then a prefill chunk | 131 ms hiccup → 17.6 ms gaps (§5) |
| 12 | Split the work | Prefill and decode on different machines; goodput vs throughput; SLOs | Two GPU pools, KV transfer arrow; SLO attainment curve | DistServe 7.4×; DeepSeek's 32-GPU prefill / 144-GPU decode units (§5, §9) |

### Module 4: Faster tokens

| # | Title | The one thing | Mechanism to animate | Owl beat |
|---|---|---|---|---|
| 13 | Guess and check | Speculative decoding: a draft guesses, the target verifies; exactly lossless | Accept/reject strip; the E = (1−α^(γ+1))/(1−α) curve | α = 0.8, γ = 4 → 3.36 tokens/pass, 2.8× (§6) |
| 14 | Drafting without a draft model | Medusa heads, EAGLE features, n-gram/prompt lookup, MTP; why it fades at high batch | Heads sprouting from the model; trees of candidates; the EAGLE-3 batch table | Owl quoting the student's notes: prompt lookup 2–4× (§6) |
| 15 | FlashAttention | Exact attention, re-ordered to respect the memory hierarchy | SRAM tiles sliding over Q/K/V; the online-softmax rescale | x = [1,2,3,0] worked example (§7) |
| 16 | The CPU tax | Decode kernels: split-KV, fusion, CUDA graphs, torch.compile | Hundreds of tiny launches collapsing into one graph replay | gpt-fast 25.5 → 107 tok/s (§7) |
| 17 | The last step of a token | Sampling, JSON masks, and why temperature 0 isn't deterministic on a server | Logit bar chart → mask → renormalise; batch-size-dependent sums | Flashcard JSON; 80 unique completions out of 1,000 (§12) |

### Module 5: Fewer bits, more GPUs

| # | Title | The one thing | Mechanism to animate | Owl beat |
|---|---|---|---|---|
| 18 | Fewer bits, same model | Weight-only vs weight+activation quantization; INT8/FP8; outliers | One group quantized step by step; outlier column | 8 weights → int4 (§8); FP8 lossless, INT8 1–3% |
| 19 | Four bits and below | GPTQ, AWQ, GGUF k-quants, MXFP4/NVFP4; distillation and pruning in one minute | Super-blocks and scales; bpw accounting | Owl-8B 16 → 4.9 GB; decode ×2.5, prefill ≈ same (§8) |
| 20 | Mixture of Experts at serve time | Active vs total params; batch lights up all experts; expert parallelism | Router lighting experts; all-to-all dispatch/combine | Mixtral: 2 experts at batch 1, all 8 by batch 16 (§9) |
| 21 | When one GPU isn't enough | TP vs PP vs EP vs CP; interconnects decide | Matrix sliced across GPUs; all-reduce pulses; NVLink vs PCIe pipes | Owl-Pro 70B: TP4 ≈ 95 tok/s ceiling (§10) |

### Module 6: Production and your Mac

| # | Title | The one thing | Mechanism to animate | Owl beat |
|---|---|---|---|---|
| 22 | The serving stack | Engine + OpenAI-compatible API + SSE + router + autoscaler; cold starts | Request travelling down the stack; SSE chunks; cold-start bar | 16 GB load: 3 s (NVMe) to 48 s (HF loader on GP3) (§13) |
| 23 | Your Mac as an inference server | Unified memory; what fits; tok/s ≈ bandwidth ÷ size; MLX vs llama.cpp | RAM jar filling with weights + KV; Mac bandwidth ladder | Owl-8B 4-bit on M4 Pro ≈ 42–51 tok/s; live mlx-lm measurement (App. B) |
| 24 | The price of a token | $/M tokens from $/GPU-hour; why output costs ~5× input; Little's law | Cost-vs-batch curve; prefill vs decode cost bars; L = λW | 20 req/s → 124 concurrent on one H100 (§15) |
| 25 | Keeping it alive | Metrics, percentiles, preemption, admission control, graceful degradation | Dashboard incident replay | p99 TTFT alert from KV saturation (§16) |

**Optional extras:** multi-LoRA serving (§17) as a short to episode 22; "reasoning models make decode king" as a short to episode 24.

---

## Appendix B: What to measure live on an Apple-silicon Mac (mlx-lm), for production time

Nothing has been downloaded. These are the measurements to run later.

### Setup notes
- Record the chip, RAM, macOS version and mlx/mlx-lm versions (mlx-lm 0.32.2 is current as of 2026-10-16).
- Run plugged in, with Low Power Mode off and other GPU apps closed. Do 3–5 trials and report the median.
- **Always pass `--model`** to `mlx_lm.benchmark`, otherwise it downloads a default model.

### Models (download only at production time)

| Model | Size | Use |
|---|---|---|
| `mlx-community/Llama-3.2-1B-Instruct-4bit` (verified to exist; group 64, 4-bit) | ≈ 0.7 GB | Owl-mini |
| `…-8bit` and `…-bf16` siblings | ≈ 1.3 / 2.5 GB | bits comparison [names UNVERIFIED] |
| `mlx-community/Qwen3-1.7B-4bit` (verified) | – | the "small model, big KV" contrast (112 KiB/token) |
| Llama-3.1-8B-Instruct 4-bit | ≈ 4.5 GB | only if RAM ≥ 16 GB and the episode needs it |

### Measurements

| # | Measure | How | What the episode shows | Expected [DERIVED] |
|---|---|---|---|---|
| 1 | Decode tok/s vs bits | `mlx_lm.generate --model <m> --prompt "…" --max-tokens 256` for 4/8/16-bit; read "Generation: N tokens, X tokens-per-sec" | tok/s × weight bytes ≈ bandwidth → MBU (Ep 2, 18) | 55–85% of Apple's bandwidth figure |
| 2 | Prefill tok/s vs prompt length | `mlx_lm.benchmark --model <m> -p 128/512/2048/8192 -g 32 -n 5` (prompt_tps) | Prefill is compute-bound, far faster per token than decode; TTFT ∝ length (Ep 3, 9) | Prompt tok/s ≫ generation tok/s |
| 3 | Decode slowdown with context | `-p 512` vs `-p 8192`, `-g 256` | KV reads add bytes per step (Ep 5) | Llama-1B at 8K: +256 MiB/step on ~0.7 GB → noticeably slower; Qwen3-1.7B (112 KiB/token) slows more |
| 4 | Peak memory vs context | "Peak memory: Z GB" line at several prompt lengths | Weights + KV formula check (Ep 5) | Slope ≈ 32 KiB/token (Llama-1B BF16 KV) |
| 5 | KV quantization | Repeat 4 with `--kv-bits 8` and `--kv-bits 4` (`--kv-group-size 64`) | Memory slope halves/quarters; quality spot-check (Ep 6) | ≈ ½ and ≈ ¼ slope, plus scale overhead |
| 6 | Rotating cache | `--max-kv-size 1024` with long generation | Memory stays flat; old context is forgotten (Ep 9) | Flat peak memory |
| 7 | Prompt caching | `mlx_lm.cache_prompt` on a long system prompt or chapter, then `mlx_lm.generate --prompt-cache-file …` | TTFT with vs without cache (Ep 8) | TTFT drops by roughly prefix/total |
| 8 | Batching free lunch | `mlx_lm.benchmark --model <m> -b 1/2/4/8/16 -p 256 -g 128` | Aggregate tok/s rises much faster than per-sequence falls (Ep 4) | Near-linear aggregate gains at small B |
| 9 | Speculative decoding | `mlx_lm.generate --model <8B-4bit> --draft-model <1B-4bit> --num-draft-tokens 2…6` on (a) open chat and (b) "rewrite these notes" | Speedup depends on α; input-grounded text is easier to guess (Ep 13–14) | 1.3–2× typical; varies [UNVERIFIED] |
| 10 | TTFT/ITL percentiles and queueing | Run `mlx_lm.server --model <m>`, stream with a small Python client timestamping each SSE chunk; then 1, 4, 16 concurrent clients | TTFT and ITL distributions, p50/p99, queueing (Ep 1, 24, 25) | Whether mlx_lm.server batches concurrent requests is version-dependent [UNVERIFIED] |
| 11 | Energy per token | `sudo powermetrics --samplers gpu_power -i 500` during generation; joules/token = watts ÷ tok/s | Cost and energy on-device (Ep 23, 24) | Order of 0.1–1 J/token [UNVERIFIED] |
| 12 | Determinism | Same prompt, temperature 0, 20 runs, at batch 1 and with concurrent load | Batch invariance demo (Ep 17) | Likely identical at batch 1; may differ under batching [UNVERIFIED] |

**Cross-check every result against §2's ceiling (bandwidth ÷ bytes).** If measured > ceiling, the bytes estimate is wrong (e.g. tied embeddings, quantized embeddings).

---

## Appendix C: Sources (read Oct 2026)

### Papers
- **Arithmetic and roofline**
  - kipply, Transformer Inference Arithmetic (2022) — https://kipp.ly/transformer-inference-arithmetic/
  - JAX, How to Scale Your Model, Inference — https://jax-ml.github.io/scaling-book/inference/
- **Memory and KV cache**
  - PagedAttention / vLLM — https://arxiv.org/abs/2309.06180
  - SGLang / RadixAttention — https://arxiv.org/abs/2312.07104
  - MQA — https://arxiv.org/abs/1911.02150
  - GQA — https://arxiv.org/abs/2305.13245
  - DeepSeek-V2 (MLA) — https://arxiv.org/abs/2405.04434
  - Mistral 7B — https://arxiv.org/abs/2310.06825
  - Gemma 3 — https://arxiv.org/abs/2503.19786
  - KIVI — https://arxiv.org/abs/2402.02750
  - LMCache — https://arxiv.org/abs/2510.09665
  - StreamingLLM — https://arxiv.org/abs/2309.17453
- **Batching, scheduling and disaggregation**
  - Orca — https://www.usenix.org/conference/osdi22/presentation/yu
  - Sarathi — https://arxiv.org/abs/2308.16369
  - Sarathi-Serve — https://arxiv.org/abs/2403.02310
  - DistServe — https://arxiv.org/abs/2401.09670
  - Splitwise — https://arxiv.org/abs/2311.18677
  - Mooncake — https://arxiv.org/abs/2407.00079
- **Faster decoding**
  - Leviathan et al. — https://arxiv.org/abs/2211.17192
  - Chen et al. — https://arxiv.org/abs/2302.01318
  - Medusa — https://arxiv.org/abs/2401.10774
  - EAGLE — https://arxiv.org/abs/2401.15077
  - EAGLE-2 — https://arxiv.org/abs/2406.16858
  - EAGLE-3 — https://arxiv.org/abs/2503.01840
  - Lookahead — https://arxiv.org/abs/2402.02057
  - Multi-token prediction — https://arxiv.org/abs/2404.19737
- **Kernels**
  - FlashAttention — https://arxiv.org/abs/2205.14135
  - FlashAttention-2 — https://arxiv.org/abs/2307.08691
  - FlashAttention-3 — https://arxiv.org/abs/2407.08608
  - FlashInfer — https://arxiv.org/abs/2501.01005
- **Quantization, pruning and distillation**
  - LLM.int8() — https://arxiv.org/abs/2208.07339
  - SmoothQuant — https://arxiv.org/abs/2211.10438
  - GPTQ — https://arxiv.org/abs/2210.17323
  - AWQ — https://arxiv.org/abs/2306.00978
  - FP8 formats — https://arxiv.org/abs/2209.05433
  - QLoRA / NF4 — https://arxiv.org/abs/2305.14314
  - Give Me BF16 or Give Me Death — https://arxiv.org/abs/2411.02355
  - SparseGPT — https://arxiv.org/abs/2301.00774
  - Wanda — https://arxiv.org/abs/2306.11695
  - Minitron — https://arxiv.org/abs/2407.14679
- **MoE**
  - DeepSeek-V3 — https://arxiv.org/abs/2412.19437
  - Switch Transformer — https://arxiv.org/abs/2101.03961
- **Serving topics**
  - Outlines — https://arxiv.org/abs/2307.09702
  - XGrammar — https://arxiv.org/abs/2411.15100
  - S-LoRA — https://arxiv.org/abs/2311.03285
- **Models**
  - Llama 3 Herd — https://arxiv.org/abs/2407.21783

### Engineering blogs and docs
- **vLLM and SGLang**
  - vLLM launch — https://vllm.ai/blog/2023-06-20-vllm
  - vLLM V1 — https://vllm.ai/blog/2025-01-27-v1-alpha-release
  - KV offloading — https://vllm.ai/blog/2026-01-08-kv-offloading-connector
  - Sleep mode — https://vllm.ai/blog/2025-10-26-sleep-mode
  - vLLM docs (metrics, optimization, prefix caching, quantized KV, CUDA graphs, structured outputs, bench serve) — https://docs.vllm.ai/en/latest/
  - LMSYS SGLang — https://lmsys.org/blog/2024-01-17-sglang/
  - Large-scale EP — https://lmsys.org/blog/2025-05-05-large-scale-ep/
- **DeepSeek**
  - Inference system overview — https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md
  - DeepEP (old README) — https://github.com/deepseek-ai/DeepEP/blob/458cdcb22a76ae8b58713f1b9f8916ffa4011cfb/README.md
- **Kernels and PyTorch**
  - Flash-Decoding — https://crfm.stanford.edu/2023/10/12/flashdecoding.html
  - FlashInfer sampling — https://flashinfer.ai/2025/03/10/sampling.html
  - GPT-Fast — https://pytorch.org/blog/accelerating-generative-ai-2/
  - CUDA Graphs — https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
- **Performance engineering**
  - Databricks, LLM Inference Performance Engineering — https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices
  - Thinking Machines, nondeterminism — https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- **NVIDIA**
  - NVFP4 — https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
  - 2:4 sparsity — https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/
  - Run:ai Model Streamer — https://developer.nvidia.com/blog/reducing-cold-start-latency-for-llm-inference-with-nvidia-runai-model-streamer
  - Dynamo — https://nvidianews.nvidia.com/news/nvidia-dynamo-open-source-library-accelerates-and-scales-ai-reasoning-models
- **Formats and quantization tools**
  - OCP MX spec — https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
  - llama.cpp k-quants PR — https://github.com/ggml-org/llama.cpp/pull/1684
  - quantize README — https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md
  - Apple-silicon benchmarks — https://github.com/ggml-org/llama.cpp/discussions/4167
- **Models and announcements**
  - Mixtral — https://mistral.ai/news/mixtral-of-experts
  - gpt-oss model card — https://huggingface.co/openai/gpt-oss-120b
  - Llama 3.2 — https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/
  - llm-d — https://llm-d.ai/blog/llm-d-announce
  - Prompt lookup — https://github.com/apoorvumang/prompt-lookup-decoding
- **Engines and tools**
  - TGI — https://github.com/huggingface/text-generation-inference
  - Ollama MLX — https://ollama.com/blog/mlx
  - LM Studio OpenAI compatibility — https://lmstudio.ai/docs/developer/openai-compat
  - mlx-lm — https://github.com/ml-explore/mlx-lm
  - OpenAI OpenAPI spec — https://github.com/openai/openai-openapi
  - llguidance — https://github.com/guidance-ai/llguidance
  - AIPerf — https://github.com/ai-dynamo/aiperf
  - GuideLLM — https://github.com/vllm-project/guidellm
- **Operations and benchmarks**
  - GKE autoscaling — https://docs.cloud.google.com/kubernetes-engine/docs/best-practices/machine-learning/inference/autoscaling
  - MLPerf Llama-2-70B — https://mlcommons.org/2024/03/mlperf-llama2-70b/
  - MLPerf v5.0 — https://mlcommons.org/2025/04/llm-inference-v5/

### Hardware and pricing [DATED]
- **NVIDIA**
  - H100 — https://www.nvidia.com/en-us/data-center/h100/
  - A100 — https://www.nvidia.com/en-us/data-center/a100/
  - H200 — https://www.nvidia.com/en-us/data-center/h200/
  - DGX B200 — https://www.nvidia.com/en-us/data-center/dgx-b200/
  - GB200 NVL72 — https://www.nvidia.com/en-us/data-center/gb200-nvl72/
- **Apple**
  - M4 Pro/Max — https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/
  - M5 — https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/
  - M5 Pro/Max — https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/
  - M5 Ultra — https://www.apple.com/newsroom/2026/08/apple-introduces-new-mac-studio-with-m5-max-and-m5-ultra/
  - M3 Ultra — https://www.apple.com/newsroom/2025/03/apple-reveals-m3-ultra-taking-apple-silicon-to-a-new-extreme/
- **Apple ML Research**
  - MLX on M5 — https://machinelearning.apple.com/research/exploring-llms-mlx-m5
  - MLX + RDMA on Mac Studio — https://machinelearning.apple.com/research/mlx-distributed-inference
  - Foundation Models 2024 — https://machinelearning.apple.com/research/introducing-apple-foundation-models
  - Foundation Models 2025 — https://machinelearning.apple.com/research/apple-foundation-models-2025-updates
- **Google**
  - Gemma 3n — https://ai.google.dev/gemma/docs/gemma-3n
- **GPU rental**
  - Lambda — https://lambda.ai/pricing
  - RunPod — https://www.runpod.io/pricing
  - AWS P5 price cut — https://aws.amazon.com/blogs/aws/announcing-up-to-45-price-reduction-for-amazon-ec2-nvidia-gpu-accelerated-instances/
- **API pricing and caching**
  - Anthropic prompt caching — https://platform.claude.com/docs/en/build-with-claude/prompt-caching
  - OpenAI prompt caching — https://developers.openai.com/api/docs/guides/prompt-caching
  - OpenAI pricing — https://developers.openai.com/api/docs/pricing
  - Gemini caching — https://ai.google.dev/gemini-api/docs/caching
  - Gemini pricing — https://ai.google.dev/gemini-api/docs/pricing

### Open items / could not verify
- Original publication date of the gpt-fast post (Nov 2023; the page shows a migration date).
- Absolute ms cost of top-p sampling.
- Typical kernel-launch overhead per step (order of magnitude only).
- llama-server default port (flag table says 9931; historically 8080).
- The exact current mlx-community repo names for the 8-bit/bf16 siblings.
- Whether the 512 GB M3 Ultra is still sold.
- KIVI's exact sentence (numbers confirmed, wording fragmentary).
- The OpenAI "4 characters per token" help page (HTTP 403).
- gpt-oss release date (the openai.com page returned 403; the HF card was used).
- Venue details for EAGLE-2, FA3 and S-LoRA (from memory).
- Apple's Oct 2026 MLX/RDMA post has unlabeled context lengths and an internally inconsistent TTFT claim.
