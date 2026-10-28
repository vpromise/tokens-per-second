# Inference, Explained: live measurements on this Mac (mlx-lm)

Measured **2026-10-24** (00:40–03:30 local) by the measurement agent. Machine-readable twin: `research/measurements.json` (stable keys; the `headline` block is meant for scene assertions). Raw per-run JSON: `research/measurements/raw/`; logs: `measurements/logs/`; prompts: `measurements/prompts/`; scripts: `measurements/scripts/` (same files as `tools/`).

## Machine, software, conditions

| Item | Value |
|---|---|
| Machine | Mac Studio (2025), Mac16,9 |
| Chip | Apple M4 Max: 16-core CPU (12 performance + 4 efficiency), 40-core GPU |
| Memory | 64 GB unified; MLX recommended working set 55.7 GB |
| Bandwidth (spec) | **546 GB/s**. Apple Support, Mac Studio (2025) Tech Specs (support.apple.com/en-us/122211): "M4 Max with 16‑core CPU, 40‑core GPU, and 16‑core Neural Engine (546GB/s memory bandwidth)" (the 14-core/32-core M4 Max is 410 GB/s) |
| OS | macOS 27.0.1 (build 26A434); AC power; Low Power Mode off |
| Python env | `tools/.venv` (uv 0.12.11, CPython 3.12) |
| Versions | mlx 0.32.4, mlx-metal 0.32.4, **mlx-lm 0.32.2**, transformers 5.20.2, zeus-apple-silicon 1.1.0 (energy only) |

Conditions: no other GPU work (no rendering, narrator TTS never started). macOS background daemons (dasd, ANECompilerService) were using CPU at the start. The 8B 8-bit/bf16 downloads were still running (network and disk only) during the first 1B and 8B-4bit runs; the 8B-4bit decode re-checked after the downloads with the stock CLI gives 102.7 vs 101.9 tok/s, so they had no visible effect.

**Method.** `tools/bench.py` loads each model once, does **1 warm-up then 5 timed trials** (3 for batching, speculative decoding and energy), and reports **median [min–max], n**. Timing uses `time.perf_counter()` at every token yielded by `mlx_lm.stream_generate`:
- **TTFT** = first token time − call time (prompt already tokenized); **prefill tok/s** = prompt tokens ÷ TTFT (this is exactly how mlx-lm computes its `prompt_tps`).
- **Decode tok/s** = (N − 1) ÷ (t_last − t_first). mlx-lm prints N ÷ (that interval), which reads ≈0.4% higher at N = 256.
- Speed runs disable EOS (as `mlx_lm.benchmark` does), so every trial generates exactly the same number of tokens. Sampling is greedy (temperature 0) everywhere.
- Fixed prompts: `prompts/chat.txt` (Owl seasons question, 95 tokens with the Llama chat template), `prompts/notes_rewrite.txt` (copy-these-notes, 370 tokens), `prompts/corpus.txt` (the course notes, used for long prompts), `prompts/chapter.txt` + `question.txt` (prompt caching), `prompts/determinism.txt`.

**Bytes.** *Weights GB* = sum of the `*.safetensors` files on disk. *Ceiling* = 546 GB/s ÷ weights GB (the course's "bandwidth ÷ bytes" rule). *Bytes read per decode step* = every tensor except an **untied** input-embedding table (that is a row lookup: Llama-3.1-8B's 1.05 GB `embed_tokens` is not streamed per token), plus the KV cache at the average context of the run. Llama-3.2-1B and Qwen3-1.7B tie embeddings, so their table doubles as the LM head and is read in full. *MBU* = tok/s × bytes read per step ÷ 546 GB/s (Databricks' definition).

## Models

| Model | Bits | Repo | Weights GB (files) | Bytes read / decode step GB (weights) | Tied emb. | KV per token, bf16 cache |
|---|---|---|---|---|---|---|
| llama-3.2-1b | 4bit | `mlx-community/Llama-3.2-1B-Instruct-4bit` | 0.695 | 0.695 | yes | 2 x 16 x 8 x 64 x 2 B = **32,768 B** (32 KiB) |
| llama-3.2-1b | 8bit | `mlx-community/Llama-3.2-1B-Instruct-8bit` | 1.313 | 1.313 | yes | 2 x 16 x 8 x 64 x 2 B = **32,768 B** (32 KiB) |
| llama-3.2-1b | bf16 | `mlx-community/Llama-3.2-1B-Instruct-bf16` | 2.472 | 2.472 | yes | 2 x 16 x 8 x 64 x 2 B = **32,768 B** (32 KiB) |
| llama-3.1-8b | 4bit | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` | 4.517 | 4.222 | no | 2 x 32 x 8 x 128 x 2 B = **131,072 B** (128 KiB) |
| llama-3.1-8b | 8bit | `mlx-community/Meta-Llama-3.1-8B-Instruct-8bit` | 8.532 | 7.974 | no | 2 x 32 x 8 x 128 x 2 B = **131,072 B** (128 KiB) |
| llama-3.1-8b | bf16 | `mlx-community/Meta-Llama-3.1-8B-Instruct-bf16` | 16.061 | 15.010 | no | 2 x 32 x 8 x 128 x 2 B = **131,072 B** (128 KiB) |
| qwen3-1.7b | 4bit | `mlx-community/Qwen3-1.7B-4bit` | 0.968 | 0.968 | yes | 2 x 28 x 8 x 128 x 2 B = **114,688 B** (112 KiB) |

Quantized repos use MLX affine quantization, group size 64, bf16 scale + bias per group (4-bit ≈ 4.5 bits/weight, 8-bit ≈ 8.5). Parameter counts (from the bf16 headers): Llama-3.2-1B 1.236 B total, 0.973 B in the transformer blocks; Llama-3.1-8B 8.030 B total, 6.980 B in the blocks (embedding and LM head 0.525 B each).

## 0. Machine roofs (micro-benchmarks, `bench.py roofs`)

| Test | Median | Range | vs 546 GB/s spec |
|---|---|---|---|
| Streaming read, `mx.sum` over 4 GB | 482.8 GB/s | [475.3–485.2] | 88% |
| bf16 mat-vec (28672×16384, 0.94 GB) | 474.9 GB/s | [467.6–479.7] | 87% |
| 4-bit mat-vec (same W, quantized_matmul) | 430.5 GB/s | [397.8–438.1] | 79% |
| Copy (read + write), 4 GB | 476.8 GB/s |  |  |
| bf16 matmul 8192³ | 15.2 TFLOPS | [15.2–15.2] |  |
| fp16 / fp32 matmul 8192³ | 15.2 / 13.4 TFLOPS |  |  |

Re-run at the end of the session: read 481.2 GB/s, bf16 mat-vec 465.5 GB/s, matmul 15.2 TFLOPS (stable within 2%).

**For the course.** A pure streaming read on this Mac reaches **≈483 GB/s (88% of Apple's 546)** (488 GB/s = 89% on a 1 GB array), so roughly 88–89% of spec is the practical top of the bandwidth roof. Apple publishes no GPU TFLOPS; the measured bf16 matmul is **15.2 TFLOPS**, which puts the **ridge point at ≈28 FLOPs/byte** (15.2e12 ÷ 546e9), about a tenth of the H100's ~295. The Mac is far more bandwidth-rich relative to its compute.

## 1. Decode speed vs bit width (batch 1)

Command: `tools/.venv/bin/python tools/bench.py decode --model <repo> --tag <tag> --trials 5` (prompt `chat.txt`, 256 generated tokens, EOS off). Stock-CLI equivalent (cross-check below): `mlx_lm.generate --model <repo> --prompt "$(cat prompts/chat.txt)" --max-tokens 256 --temp 0`.

| Model | Bits | Weights GB | Decode tok/s median [range] | n | Ceiling 546/GB | % of ceiling | Bytes read/step GB | Ceiling on bytes read | MBU | Achieved GB/s | % of measured read roof | Speed vs bf16 | Fewer bytes vs bf16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama-3.2-1b | 4bit | 0.695 | **488.85** [487.66–489.92] | 5 | 785.3 | 62% | 0.703 | 777.2 | **62.9%** | 343 | 71% | 2.72× | 3.56× |
| llama-3.2-1b | 8bit | 1.313 | **308.51** [307.47–308.96] | 5 | 415.8 | 74% | 1.320 | 413.5 | **74.6%** | 407 | 84% | 1.72× | 1.88× |
| llama-3.2-1b | bf16 | 2.472 | **179.84** [179.71–179.94] | 5 | 220.9 | 81% | 2.479 | 220.3 | **81.7%** | 446 | 92% | 1.00× | 1.00× |
| llama-3.1-8b | 4bit | 4.517 | **101.93** [101.08–102.21] | 5 | 120.9 | 84% | 4.251 | 128.4 | **79.4%** | 433 | 90% | 3.29× | 3.56× |
| llama-3.1-8b | 8bit | 8.532 | **58.42** [58.19–58.54] | 5 | 64.0 | 91% | 8.003 | 68.2 | **85.6%** | 468 | 97% | 1.88× | 1.88× |
| llama-3.1-8b | bf16 | 16.061 | **31.00** [30.92–31.26] | 5 | 34.0 | 91% | 15.039 | 36.3 | **85.4%** | 466 | 97% | 1.00× | 1.00× |

**Headline.** The course's reference model, **Llama-3.1-8B in bf16 (16.06 GB of weights), decodes at 31.0 tok/s** (32.3 ms per token). The "bandwidth ÷ bytes" ceiling is 546 ÷ 16.06 = **34.0 tok/s**, so the Mac reaches **91% of the ceiling**. Counting only the bytes really streamed per token (15.04 GB: no input-embedding table, plus 23 MB of KV), MBU is **85%**, i.e. 466 GB/s, which is **97% of what a plain streaming read achieves** on this machine. Multiply speed by size and you get bandwidth: 31.0 tok/s × 15.04 GB ≈ 466 GB/s.

- **4-bit is 3.29× faster than bf16** (101.9 vs 31.0 tok/s) while moving 3.56× fewer bytes; 8-bit is 1.88× (bytes ratio 1.88×). This matches GPTQ's "~3.25×" headline for 4-bit batch-1 decode (notes §0 #12).
- **Quantized formats use bandwidth less efficiently** (notes §2 lesson c): MBU 79% (4-bit) vs 86% (8-bit) vs 85% (bf16) on 8B; 63% / 75% / 82% on the 1B. Dequantizing costs compute; the 4-bit mat-vec micro-benchmark also tops out lower (427–431 GB/s vs 466–475 GB/s for bf16).
- **Small models waste a larger share of each step.** Llama-3.2-1B-4bit takes 2.05 ms per token, of which only ≈1.28 ms is the 0.70 GB read at 546 GB/s; the rest is fixed per-step cost (presumably kernel launches for 16 layers, sampling and Python overhead; not profiled). That is why its MBU is only 63%.
- **Cross-check against the ceiling:** every measured number is below both the 546/GB ceiling and the bytes-read ceiling. Without the untied-embedding correction, 8B-4bit would look like 84% of a 120.9 tok/s ceiling; the correct ceiling is 128.4.
- Greedy outputs were identical across the 5 trials for every model.

## 2. Prefill (TTFT) vs prompt length

Command: `bench.py prefill --model <repo> --lengths 128 256 512 1024 2048 4096 8192 16384 --trials 5` (prompt = the first N tokens of `corpus.txt`, max_tokens 1, default `prefill_step_size` 2048). Cross-check: `mlx_lm.benchmark --model <8B-4bit> -p 2048 -g 128 -n 3` gave prompt_tps 943.3 (harness 940.2).

| Prompt tokens | 1B 4-bit tok/s | 1B bf16 tok/s | 8B 4-bit tok/s | 8B 8-bit tok/s | 8B bf16 tok/s | 8B bf16 TTFT s | 8B 4-bit TTFT s |
|---|---|---|---|---|---|---|---|
| 128 | 4,990 [4973–5025] | 4,169 [4163–4203] | 793 [791–794] | 732 [728–734] | 621 [620–622] | 0.206 | 0.161 |
| 256 | 5,645 [5595–5663] | 5,246 [5237–5271] | 880 [879–881] | 835 [834–837] | 749 [747–750] | 0.342 | 0.291 |
| 512 | 6,149 [6129–6193] | 5,879 [5841–5967] | 926 [925–928] | 908 [907–908] | 899 [895–902] | 0.570 | 0.553 |
| 1024 | 6,626 [6599–6634] | 6,726 [6695–6746] | 943 [933–944] | 936 [936–936] | 982 [982–983] | 1.042 | 1.086 |
| 2048 | 6,624 [6308–6626] | 7,019 [7018–7024] | 940 [940–941] | 938 [937–938] | 1,005 [1004–1005] | 2.039 | 2.178 |
| 4096 | 6,052 [6000–6266] | 6,672 [6665–6674] | 911 [908–911] | 911 [911–912] | 980 [980–980] | 4.181 | 4.498 |
| 8192 | 5,546 [5411–5600] | 5,964 [5959–5966] | 849 [844–851] | 853 [853–854] | 915 [914–916] | 8.958 | 9.652 |
| 16384 | 4,622 [4541–4623] | 4,867 [4864–4869] | 740 [739–746] | 752 [752–753] | 801 [800–801] | 20.466 | 22.140 |

| Model | Bits | Prefill tok/s @2048 | Decode tok/s | Prefill ÷ decode | Achieved TFLOPS (2·P_blocks + attention) | of measured 15.2 TFLOPS |
|---|---|---|---|---|---|---|
| llama-3.2-1b | 4bit | 6,624 | 488.8 | **13.6×** | 13.8 | 91% |
| llama-3.2-1b | bf16 | 7,019 | 179.8 | **39.0×** | 14.6 | 96% |
| llama-3.1-8b | 4bit | 940 | 101.9 | **9.2×** | 13.6 | 90% |
| llama-3.1-8b | 8bit | 938 | 58.4 | **16.1×** | 13.6 | 89% |
| llama-3.1-8b | bf16 | 1,005 | 31.0 | **32.4×** | 14.6 | 96% |

**For the course.** Prefill is compute-bound and runs **32× faster per token than decode for 8B bf16** (1005 vs 31 tok/s), and 9.2× for 8B 4-bit. Prefill reaches 13.6–14.6 TFLOPS, 89–96% of the measured matmul roof; batch-1 decode on the same GPU uses only ≈0.5 TFLOPS (31 tok/s × ~15 GFLOP). Quantization helps decode, not prefill: bf16 prefills fastest (1,005 tok/s), because quantized weights add dequantization work to every matmul. TTFT is roughly proportional to prompt length: 8B bf16 needs 0.21 s for 128 tokens, 2.0 s for 2,048 and 20.5 s for 16,384. Per-token prefill speed falls at long prompts because attention work grows with length (1,005 → 801 tok/s from 2K to 16K) and is lower for very short prompts, whose matrices are too small to fill the GPU (621 tok/s at 128).

FLOPs per prompt token = 2 × 6.98 B (block parameters; the LM head is not evaluated during chunked prefill) + 2·L·d·T for causal attention. "Measured roof" = the 8192³ bf16 matmul above; Apple publishes no TFLOPS, so this is a self-measured peak, not a spec.

## 3. Decode slowdown with context length

Command: `bench.py context --model <repo> --lengths 128 512 2048 8192 16384 32768 --gen 128 --trials 5`. The prompt (T−1 corpus tokens) is prefilled once; each trial decodes 128 tokens from that cache and is then trimmed back (`trim_prompt_cache`), so every trial sees the same context. *Predicted* = tok/s at T=128 × bytes(128) ÷ bytes(T), with bytes(T) = weights read + (T + 64) × KV/token.

| Context tokens | 8B bf16 tok/s (pred.) · KV | 8B 4-bit tok/s (pred.) · KV | 1B 4-bit tok/s (pred.) · KV | Qwen3-1.7B 4-bit tok/s (pred.) · KV |
|---|---|---|---|---|
| 128 | 30.8 (30.8) · KV 0.02 GiB | 101.9 (101.9) · KV 0.02 GiB | 497.0 (497.0) · KV 0.00 GiB | 328.0 (328.0) · KV 0.01 GiB |
| 512 | 30.6 (30.7) · KV 0.06 GiB | 99.9 (100.7) · KV 0.06 GiB | 470.6 (488.2) · KV 0.02 GiB | 308.4 (314.1) · KV 0.05 GiB |
| 2048 | 30.1 (30.3) · KV 0.25 GiB | 94.4 (96.2) · KV 0.25 GiB | 410.5 (456.1) · KV 0.06 GiB | 277.4 (268.3) · KV 0.22 GiB |
| 8192 | 28.7 (28.8) · KV 1.00 GiB | 81.2 (81.6) · KV 1.00 GiB | 293.7 (361.0) · KV 0.25 GiB | 200.3 (169.6) · KV 0.88 GiB |
| 16384 | 26.9 (27.0) · KV 2.00 GiB | 67.6 (67.8) · KV 2.00 GiB | 244.7 (282.5) · KV 0.50 GiB | 144.5 (113.8) · KV 1.75 GiB |
| 32768 | 23.8 (24.0) · KV 4.00 GiB | 51.0 (50.8) · KV 4.00 GiB | 216.3 (196.8) · KV 1.00 GiB | 95.0 (68.6) · KV 3.50 GiB |

Ranges are tight (spread ≤ 3.5%; all in the JSON). **For the course.** Every decode step reads the whole KV cache as well as the weights. At 32K tokens the 8B's cache is 4 GiB, about the size of its 4-bit weights, so **8B-4bit decode halves (101.9 → 51.0 tok/s)**, and the bytes model predicts 50.8. For bf16 the same 4 GiB is a smaller share of the bytes (16 GB of weights): 30.8 → 23.8 tok/s against a prediction of 24.0. **Small model, big KV:** Qwen3-1.7B stores 112 KiB per token (3.5× Llama-1B's 32 KiB) and slows 3.5× by 32K (328 → 95 tok/s), when its cache (3.5 GiB) is 3.9× its weights. At that point it streams 449 GB/s. Llama-1B slows 2.3× (497 → 216).

Caveat: for the 8B models the bytes model is within 1–2%. For the 1B-class models it is only a rough guide: at short context their step is partly fixed overhead (MBU 63%), and the attention kernel's efficiency changes with length (it does better at 32K than at 8K).

## 4. Memory vs context (KV-cache slope)

Command: `bench.py memory --model <repo> --lengths 1024 2048 4096 8192 16384 32768 --gen 16`. For each T: prefill T corpus tokens, generate 16, then read `mx.get_peak_memory()` (reset per run), `mx.get_active_memory()` (cache alive) and the cache's own `nbytes`. Slopes are least-squares fits over the six points. Memory is deterministic: two repeats gave identical peaks.

| Model | Formula 2·L·n_kv·d_h·2 B | Active-memory slope B/token | Cache nbytes slope | Slope ÷ formula | Peak-memory slope |
|---|---|---|---|---|---|
| llama-3.1-8b | 131,072 | **131,072** | 131,072 | 1.0000 | 129,486 |
| llama-3.2-1b | 32,768 | **32,768** | 32,768 | 1.0000 | 29,876 |
| qwen3-1.7b | 114,688 | **114,688** | 114,688 | 1.0000 | 110,723 |

| Context | 8B 4-bit: peak GB / active GB / KV MiB | 1B 4-bit: same | Qwen3-1.7B 4-bit: same |
|---|---|---|---|
| 1024 | 5.37 / 4.69 / 160 | 1.48 / 0.74 / 40 | 1.78 / 1.11 / 140 |
| 2048 | 5.54 / 4.82 / 288 | 1.54 / 0.77 / 72 | 1.91 / 1.23 / 252 |
| 4096 | 5.86 / 5.09 / 544 | 1.63 / 0.84 / 136 | 2.23 / 1.47 / 476 |
| 8192 | 6.30 / 5.62 / 1056 | 1.82 / 0.97 / 264 | 2.77 / 1.94 / 924 |
| 16384 | 7.30 / 6.70 / 2080 | 1.99 / 1.24 / 520 | 3.59 / 2.88 / 1820 |
| 32768 | 9.53 / 8.85 / 4128 | 2.46 / 1.78 / 1032 | 5.33 / 4.76 / 3612 |

**For the course.** The KV formula is exact on this machine: the measured slope is **131,072 B/token = 128 KiB** for Llama-3.1-8B, 32,768 B (32 KiB) for Llama-3.2-1B and 114,688 B (112 KiB) for Qwen3-1.7B. A 32K-token conversation adds 4.0 GiB to the 8B (4-bit weights 4.5 GB → 8.85 GB active). mlx-lm allocates the cache in 256-token steps (a 1,040-token cache holds 1,280 slots), so per-T cache sizes are rounded up to the next 256. The peak includes prefill transients (activations for a 2,048-token chunk), so it sits ≈0.6–0.75 GB above active memory.

## 5. KV-cache quantization (`--kv-bits 8 / 4`, group size 64)

Command: as 4 plus `--kv-bits 8` or `--kv-bits 4` (harness passes `kv_bits`, `kv_group_size=64`, **`quantized_kv_start=0`**). Trap: mlx-lm's default `--quantized-kv-start` is **5000**, so `--kv-bits` alone does nothing until the cache passes 5,000 tokens.

| Model (4-bit weights) | KV | Slope B/token | Expected | vs bf16 KV | KV MiB @32K | Active GB @32K | Peak GB @32K | Decode tok/s @32K (1 run, 16 tok) |
|---|---|---|---|---|---|---|---|---|
| llama-3.1-8b | kv16 | 131,072 | 131,072 | 1.000 | 4,128 | 8.85 | 9.53 | 50 |
| llama-3.1-8b | kv8 | 69,627 | 69,632 | 0.531 | 2,193 | 6.82 | 15.96 | 51 |
| llama-3.1-8b | kv4 | 36,859 | 36,864 | 0.281 | 1,161 | 5.74 | 14.83 | 58 |
| llama-3.2-1b | kv16 | 32,768 | 32,768 | 1.000 | 1,032 | 1.78 | 2.46 | 217 |
| llama-3.2-1b | kv8 | 17,408 | 17,408 | 0.531 | 548 | 1.27 | 10.37 | 193 |
| llama-3.2-1b | kv4 | 9,213 | 9,216 | 0.281 | 290 | 1.00 | 10.06 | 212 |
| llama-3.2-1b | kv8 (prefill step 256) | 17,408 | 17,408 | 0.531 | 548 | 1.27 | 3.63 | 190 |

**Size: exactly as predicted.** Each quantized element costs bits/8 bytes plus a bf16 scale and bias per group of 64 (4/64 B), so 8-bit KV = 0.531× and 4-bit KV = 0.281× of bf16: 69,632 and 36,864 B/token for the 8B, which is what was measured.

**But in mlx-lm 0.32.2 it does not lower the peak and does not speed up decode at batch 1.**
- *Peak memory goes up* for prompts longer than one 2,048-token prefill chunk: 8B-4bit at 32K peaks at 16.0 GB (8-bit KV) vs 9.5 GB (bf16 KV). The quantized attention path (`quantized_scaled_dot_product_attention` in `mlx_lm/models/base.py`) is not a fused kernel and materializes the chunk × context score matrix. A smaller prefill chunk tames it (1B: 10.4 → 3.6 GB with `--prefill-step-size 256`), though that is still above bf16 KV's 2.46 GB.
- *Speed:* decode at 32K is about the same (8B: 51 vs 50 tok/s for 8-bit, 58 for 4-bit; 1B: slower). The bytes saved are spent on the unfused dequantize-and-attend path.
- So on this stack KV quantization buys **capacity** (more or longer conversations resident), not speed.

Quality spot-check (`bench.py kvquality`, 8B-4bit, an 8,081-token prompt of course notes + the question "how many bytes of KV cache does Llama-3.1-8B store per token, and why?", 64 greedy tokens): bf16 KV and 8-bit KV give **identical** answers (64/64 tokens); 4-bit KV matches for the first 45 tokens, then rephrases with the same facts. All three answer "128 KiB … 32 layers, 8 KV heads, head dimension 128".

## 6. Rotating (sliding-window) KV cache, `max_kv_size = 1024`

Command: `bench.py rotating --model <repo> --max-kv-size 1024 --gen 8192 --forget-ctx 3000` (EOS off; samples every 512 generated tokens).

**Trap found: `--max-kv-size` is silently ignored for Llama models in mlx-lm 0.32.2.** `llama.Model` defines `make_cache()`, and `make_prompt_cache(model, max_kv_size)` defers to it, which returns plain `KVCache`. The flag therefore has no effect, as confirmed with the stock CLI (`logs/maxkv_cli_check.log`): Llama-1B peak 0.939 vs 0.943 GB with and without `--max-kv-size 256` over 3,000 tokens. Qwen3 has no override, so the flag works there (1.465 → 1.258 GB, 285 → 327 tok/s). For Llama the harness builds mlx-lm's own `RotatingKVCache(max_size=1024, keep=4)` explicitly, which is what the flag intends.

| Model (4-bit) | Tokens generated | Full cache: KV MiB / peak GB / tok/s (last 512) | Rotating 1024: KV MiB / peak GB / tok/s |
|---|---|---|---|
| llama-3.2-1b | 1024 | 40 / 0.914 / 450 | 32 / 0.860 / 446 |
| llama-3.2-1b | 2048 | 72 / 0.914 / 409 | 32 / 0.860 / 436 |
| llama-3.2-1b | 4096 | 136 / 1.057 / 359 | 32 / 0.860 / 436 |
| llama-3.2-1b | 8192 | 264 / 1.308 / 285 | 32 / 0.860 / 434 |
| qwen3-1.7b | 1024 | 140 / 1.214 / 298 | 112 / 1.186 / 299 |
| qwen3-1.7b | 2048 | 252 / 1.351 / 279 | 112 / 1.186 / 300 |
| qwen3-1.7b | 4096 | 476 / 1.621 / 246 | 112 / 1.186 / 301 |
| qwen3-1.7b | 8192 | 924 / 2.089 / 199 | 112 / 1.186 / 298 |

**For the course.** With the window, memory and speed stay **flat**: KV is pinned at 32 MiB (1,024 × 32 KiB) and decode holds ≈434 tok/s, while the full cache grows to 264 MiB and slows to 285 tok/s by 8K tokens. The price is forgetting. A 3,070-token prompt that starts "Remember this: the secret code word is PELICAN" and ends by asking for the code word gets `PELICAN<|eot_id|>` with the full cache and `The answer is: There is no answer, as this question` with the 1,024-token window (the first 4 tokens are kept as attention sinks, but the fact is not among them). Qwen3's demo answer was cut off inside its `<think>` block (12 tokens), so it is not informative.

## 7. Prompt (prefix) caching

Harness: `bench.py promptcache --model <repo> --trials 5 --gen 48`: build the KV cache for the chapter prefix exactly as `mlx_lm.cache_prompt` does (chat template, `continue_final_message=True`), save it with `save_prompt_cache`, then time TTFT for (a) the full prompt from scratch and (b) only the 45-token question suffix on a freshly loaded copy of the cache. Stock CLI run too (`logs/cli_prompt_cache.log`):
```
mlx_lm.cache_prompt --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit --prompt "$(cat prompts/chapter.txt)" --prompt-cache-file chapter.safetensors
mlx_lm.generate --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit --prompt-cache-file chapter.safetensors --prompt "$(cat prompts/question.txt)" --max-tokens 48 --temp 0
mlx_lm.generate --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit --prompt "$(cat prompts/chapter.txt prompts/question.txt)" --max-tokens 48 --temp 0
```
| Model | Bits | Prefix + suffix tokens | TTFT no cache s | TTFT with cache s | …incl. loading the cache file s | Speedup | Cache file MB | Same greedy answer |
|---|---|---|---|---|---|---|---|---|
| llama-3.1-8b | 4bit | 6,910 + 45 | 8.011 [8.010–8.015] | 0.137 [0.136–0.138] | 0.160 | **58×** (50× incl. load) | 906 | yes |
| llama-3.1-8b | bf16 | 6,910 + 45 | 7.468 [7.467–7.473] | 0.188 [0.187–0.188] | 0.211 | **40×** (35× incl. load) | 906 | no (wording differs from token ~16) |
| llama-3.2-1b | 4bit | 6,910 + 45 | 1.205 [1.205–1.210] | 0.029 [0.028–0.030] | 0.036 | **42×** (34× incl. load) | 226 | yes |

Stock CLI, 8B-4bit (n = 3 after a warm-up): 6,954 prompt tokens → TTFT 8.03 s without the cache; 44 prompt tokens → 0.145 s with it (55×). `cache_prompt` itself prefilled the 6,910 tokens at 869 tok/s.

**For the course.** Owl's 6.9K-token chapter costs 8 s of prefill every time without a cache and **0.14 s** with one, which is roughly prefix/total as predicted (the remaining time is the 45-token question plus one decode step). Loading the 906 MB cache file added only ~0.02 s here because it was still in the OS file cache; a cold read from SSD would take longer (not measured). Note the bf16 row: cached and uncached greedy answers state the same facts but diverge in wording after ~16 tokens. The prefix was processed in different chunks, so floating-point sums happened in a different order and a near-tie flipped (see 12).

## 8. Batching: aggregate vs per-sequence throughput

Command: `bench.py batch --model <repo> --sizes 1 2 4 8 16 32 --prompt 256 --gen 128 --trials 3` (plus `--sizes 3 6 10 12 24 48 64` for the 8B). It uses mlx-lm's `BatchGenerator` (the engine behind `batch_generate`, `mlx_lm.benchmark -b` and the server) with `prefill_batch_size = completion_batch_size = B`, B distinct 256-token prompts, EOS off. Aggregate tok/s is measured over the steady-state steps where all B sequences emit a token; per-sequence = aggregate ÷ B = 1 ÷ step time.

| Batch B | 8B bf16: aggregate / per-seq tok/s / step ms | 8B 4-bit: same | 1B 4-bit: same |
|---|---|---|---|
| 1 | 31 / 30.9 / 32.3 | 101 / 101.3 / 9.9 | 479 / 479.3 / 2.1 |
| 2 | 63 / 31.7 / 31.5 | 190 / 95.0 / 10.5 | 838 / 419.0 / 2.4 |
| 3 | 94 / 31.4 / 31.8 | 248 / 82.6 / 12.1 | – |
| 4 | 122 / 30.6 / 32.7 | 267 / 66.8 / 15.0 | 1214 / 303.4 / 3.3 |
| 6 | 183 / 30.4 / 32.9 | 274 / 45.7 / 21.9 | – |
| 8 | 235 / 29.4 / 34.1 | 291 / 36.4 / 27.5 | 1454 / 181.8 / 5.5 |
| 10 | 281 / 28.1 / 35.6 | 300 / 30.0 / 33.3 | – |
| 12 | 301 / 25.1 / 39.8 | 299 / 24.9 / 40.1 | – |
| 16 | 243 / 15.2 / 66.0 | 358 / 22.4 / 44.7 | 1708 / 106.7 / 9.4 |
| 24 | 313 / 13.0 / 76.7 | 514 / 21.4 / 46.7 | – |
| 32 | 415 / 13.0 / 77.1 | 676 / 21.1 / 47.3 | 3378 / 105.6 / 9.5 |
| 48 | 537 / 11.2 / 89.4 | 563 / 11.7 / 85.2 | – |
| 64 | 686 / 10.7 / 93.3 | 717 / 11.2 / 89.3 | – |

Independent check with the stock tool (`mlx_lm.benchmark --model <8B-bf16> -p 256 -g 128 -b B -n 3`): generation_tps 293 (B=12), 240 (B=16), 394 (B=32), which confirms the dip at 16. 8B-4bit with `-p 2048`: 95 / 230 / 288 tok/s at B = 1 / 4 / 16 (lower than the 256-token runs because each step also reads 2K tokens of KV per sequence).

**For the course.** The free lunch is real but short on a Mac. **8B bf16: going from 1 to 8 users multiplies total throughput by 7.6× (31 → 235 tok/s) while each user loses only 5% (30.9 → 29.4 tok/s)**. The same 16 GB of weights are read once per step and shared. The roofline says bf16 decode (intensity ≈ B FLOPs/byte) should stay memory-bound up to the ridge, B ≈ 28. The measured knee comes earlier and the curve is not smooth: the step time is flat at 32–36 ms up to B = 10, rises to 40 ms at B = 12, then jumps to 66–77 ms at B = 16–32 and grows slowly after that (93 ms at B = 64). So aggregate plateaus near 300 tok/s from B = 12 to 24, dips to 243 at B = 16, and rises again to 415 at B = 32 and 686 at B = 64 (≈9.6 TFLOPS, 63% of the matmul peak). Our reading, not verified in the MLX source: MLX switches from a mat-vec kernel at small B to tiled matrix-matrix kernels that are inefficient at these small row counts. The knee here is set by the kernels, not by the roofline. **4-bit weights end the free lunch sooner** (B ≈ 2–3: 101 → 190 → 248, then flat at 267–300 from B = 4 to 12; 358 at 16, 514 at 24, 676 at 32, and again non-monotonic: 563 at 48, 717 at 64). Here the step time grows almost linearly from B = 3 to 12 (12 → 40 ms), then stays at ≈45 ms from B = 16 to 32. Our interpretation: with 3.6× fewer bytes per step, the per-row dequantization work of the quantized mat-vec path dominates early, until a tiled quantized-matmul kernel takes over. Even so, at B = 32 4-bit serves 676 tok/s at 21 tok/s per user. The 1B model scales to 3,379 tok/s at B = 32.

## 9. Speculative decoding (draft = Llama-3.2-1B-Instruct-4bit)

Command: `bench.py spec --model <8B repo> --draft mlx-community/Llama-3.2-1B-Instruct-4bit --k 2 3 4 5 6 --max-tokens 256 --trials 3`; prompts (a) `chat.txt` (open-ended) and (b) `notes_rewrite.txt` ("Copy out the following study notes exactly …"). Greedy, EOS on (all runs hit 256 tokens). Baseline is the same model without a draft, same run. Acceptance = accepted draft tokens ÷ drafted tokens, reconstructed exactly from mlx-lm's per-token `from_draft` flag. Stock-CLI equivalent: `mlx_lm.generate --model <8B> --draft-model mlx-community/Llama-3.2-1B-Instruct-4bit --num-draft-tokens k --prompt … --max-tokens 256 --temp 0`.

| Target | Prompt | Draft tokens | Decode tok/s [range] | Speedup | Speedup incl. prefill | Acceptance α | Tokens per target pass | Same text as greedy baseline |
|---|---|---|---|---|---|---|---|---|
| 8B 4bit | open_chat | baseline | 101.7 [101.6–101.8] | 1.00× |  |  |  |  |
|  |  | k = 2 | 144.6 [144.4–146.8] | **1.42×** | 1.37× | 0.72 | 2.42 | no (from token 221) |
|  |  | k = 3 | 128.3 [128.0–128.4] | 1.26× | 1.23× | 0.58 | 2.72 | no (from token 221) |
|  |  | k = 4 | 118.4 [118.2–118.4] | 1.16× | 1.14× | 0.53 | 3.08 | no (from token 221) |
|  |  | k = 5 | 110.0 [109.8–110.3] | 1.08× | 1.06× | 0.51 | 3.51 | no (from token 221) |
|  |  | k = 6 | 93.5 [93.5–93.6] | 0.92× | 0.91× | 0.44 | 3.61 | no (from token 221) |
| 8B 4bit | copy_notes | baseline | 100.5 [99.4–100.9] | 1.00× |  |  |  |  |
|  |  | k = 2 | 174.7 [173.8–174.8] | 1.74× | 1.51× | 1.00 | 2.98 | yes |
|  |  | k = 3 | 182.0 [181.5–182.0] | 1.81× | 1.55× | 1.00 | 4.00 | yes |
|  |  | k = 4 | 184.3 [184.2–184.3] | **1.83×** | 1.56× | 1.00 | 4.92 | yes |
|  |  | k = 5 | 181.2 [179.7–181.5] | 1.80× | 1.54× | 1.00 | 5.95 | yes |
|  |  | k = 6 | 175.0 [175.0–175.1] | 1.74× | 1.49× | 1.00 | 6.92 | yes |
| 8B bf16 | open_chat | baseline | 31.0 [30.9–31.2] | 1.00× |  |  |  |  |
|  |  | k = 2 | 61.6 [61.3–61.6] | 1.99× | 1.94× | 0.63 | 2.25 | no (from token 154) |
|  |  | k = 3 | 69.7 [69.7–69.7] | 2.25× | 2.18× | 0.58 | 2.72 | no (from token 154) |
|  |  | k = 4 | 68.1 [67.9–68.2] | 2.20× | 2.14× | 0.47 | 2.84 | no (from token 154) |
|  |  | k = 5 | 70.1 [70.1–70.3] | 2.26× | 2.19× | 0.41 | 3.05 | no (from token 154) |
|  |  | k = 6 | 70.7 [70.6–70.8] | **2.28×** | 2.21× | 0.38 | 3.28 | no (from token 154) |
| 8B bf16 | copy_notes | baseline | 30.7 [30.6–30.8] | 1.00× |  |  |  |  |
|  |  | k = 2 | 80.8 [80.3–81.2] | 2.63× | 2.37× | 1.00 | 2.98 | yes |
|  |  | k = 3 | 100.7 [100.7–100.8] | 3.28× | 2.85× | 1.00 | 4.00 | yes |
|  |  | k = 4 | 116.7 [116.5–116.8] | 3.80× | 3.20× | 1.00 | 4.92 | yes |
|  |  | k = 5 | 135.4 [135.2–135.4] | 4.40× | 3.59× | 1.00 | 5.95 | yes |
|  |  | k = 6 | 148.0 [147.8–148.0] | **4.82×** | 3.84× | 1.00 | 6.92 | yes |

**For the course.**
- **Input-grounded text is easy to guess.** When Owl copies notes, the 1B draft is right every time (α = 1.00) and each 8B pass emits k + 1 tokens: up to **4.8× faster for the bf16 target** (30.7 → 148 tok/s at k = 6) and 1.83× for the 4-bit target. Open chat accepts only 38–72%: best 2.28× (bf16) and 1.42× (4-bit at k = 2).
- **The draft must be cheap relative to the target.** A 1B-4bit step is ~2 ms. That is 6% of an 8B-bf16 step (32 ms) but 20% of an 8B-4bit step (9.8 ms). So the same draft helps the bf16 target far more, and for 4-bit chat a longer draft (k = 6) is slower than no draft (0.92×).
- Leviathan's formula E[tokens per pass] = (1 − α^(k+1)) / (1 − α) roughly matches (chat, k = 2, α = 0.72: formula 2.24, measured 2.42; α is not constant along the text).
- **Not bit-identical in practice.** With greedy sampling, speculative decoding should reproduce the target's own output. For the copy prompt it does. For chat the text diverges at token 221 (4-bit) or 154 (bf16): verifying k + 1 tokens in one batched forward pass changes floating-point rounding, and a near-tie flips (compare 12).
- "Speedup incl. prefill" counts the time to prefill both models (the spec path uses prefill_step_size 512), which matters for short answers.

## 10. Serving: TTFT and inter-token latency through `mlx_lm.server`

Commands (`tools/server_runs.sh`): `mlx_lm.server --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit --port 8090 --temp 0 --max-tokens 128`, then `tools/stream_client.py` (stdlib; streams `/v1/chat/completions` with SSE and timestamps every content chunk). Closed loop: C concurrent clients, each sending R requests back to back (C = 1: R = 10; C = 4: R = 5; C = 16: R = 3). Each prompt is a distinct ~64-token "explain <topic> to a 12-year-old in 400 words" request; max_tokens 128 (every request hit the length limit). One warm-up pair of requests first. TTFT is measured from the first connection attempt.

| Concurrent clients | Requests | TTFT p50 / p90 / p99 ms | ITL p50 / p90 / p99 ms | Per-request tok/s (median) | Aggregate tok/s | Connection retries |
|---|---|---|---|---|---|---|
| 1 | 10 | 72 / 74 / 74 | 10.6 / 10.8 / 11.1 | 94.0 | **90** | 0 |
| 4 | 20 | 551 / 626 / 662 | 15.9 / 16.1 / 28.6 | 63.8 | **202** | 0 |
| 16 | 48 | 842 / 1162 / 1210 | 47.1 / 47.7 / 67.4 | 20.8 | **290** | 2 |

**Does this mlx-lm version batch concurrent requests? Yes.** `mlx_lm/server.py` (0.32.2) routes requests through `BatchGenerator` (continuous batching; `--decode-concurrency 32`, `--prompt-concurrency 8`). A request is only served on its own if it uses a draft model, `--kv-bits`, a per-request `seed`, or images. The numbers show it: 16 clients get 290 tok/s in total vs 90 for one client, while each stream slows from 94 to 21 tok/s (ITL 10.6 → 47 ms). That is below the in-process B = 16 result (358 tok/s aggregate, 22.4 tok/s per sequence) because the server prefills newcomers inside the decode loop and adds HTTP/SSE work. The server also keeps a prompt cache (10 entries by default), so ~36 of each ~64 prompt tokens (the shared chat-template header) were served from cache.

**For the course.** At one user, TTFT is ~72 ms (a 64-token prompt) and tokens arrive every 10.6 ms like clockwork (p99 11.1 ms). Under load, TTFT grows to p50 0.55 s at 4 clients and 0.84 s at 16 (p99 1.2 s), although the prompt itself needs only ~70 ms of prefill. The rest is **waiting** while the server interleaves the newcomer with everyone else's decode steps (server internals were not instrumented). The tail ITL spikes (p99 29–67 ms, max 0.4 s) are most likely steps in which a newcomer's prompt was processed inside the decode loop, the stall that chunked prefill is designed to bound (notes §5).

Caveats: (1) a first run lost requests to **connection resets at 15–16 simultaneous connects**. `mlx_lm.server` uses the standard library's `ThreadingHTTPServer` (listen backlog 5) and its Python accept loop competes with the generation thread. The final client staggers starts by 20 ms and retries with backoff: 2 retries in 48 requests at C = 16. The first run's files are kept in `raw/superseded/`. (2) Client and server share the machine; the client's CPU cost is small (stdlib, one thread per client).

## 11. Energy per token (GPU only, no sudo)

`sudo powermetrics` was not used. `zeus-apple-silicon` 1.1.0 (installed in `tools/.venv`) reads the IOReport "Energy Model" counters without root. On this macOS **only the GPU channel updates**: the CPU and DRAM counters stayed frozen, so all figures are **GPU-only, model-based estimates (not a wall meter)**. Command: `bench.py energy --model <repo> --trials 3 --max-tokens 256`. Windows: 256-token batch-1 decode (chat prompt); one 2,048-token prefill; a batch of 16 × 256-token prompts generating 16 × 256 tokens.

| Model | Bits | Decode B=1 J/token | GPU W | Prefill mJ / prompt token | GPU W | B=16 J / generated token (minus prefill est.) | GPU W |
|---|---|---|---|---|---|---|---|
| llama-3.2-1b | 4bit | 0.055 [0.055–0.056] | 25.4 | 9.3 | 61.0 | 0.036 (0.027) | 48.7 |
| llama-3.1-8b | 4bit | 0.316 [0.314–0.319] | 30.6 | 68.0 | 63.8 | 0.210 (0.142) | 54.8 |
| llama-3.1-8b | bf16 | 0.606 [0.605–0.607] | 18.3 | 55.4 | 55.5 | 0.225 (0.170) | 44.2 |

GPU idle reads 0.0 W (the GPU power-gates when idle). **For the course.** One 8B-bf16 token costs ≈0.6 J of GPU energy at batch 1, and 4-bit halves it (0.32 J). An *input* token costs 55–68 mJ to prefill (8B), **4.6–11× less than an output token**, the same direction as API pricing (output ≈ 4–6× input). Batching cuts energy per output token: 8B-bf16 0.61 → 0.17 J (prefill share removed). Caveat: decode is memory-bound and **DRAM energy is not counted**, so true per-token energy is higher, most of all for bf16 decode, which draws only 18 W on the GPU, consistent with a GPU that is mostly waiting on memory.

## 12. Determinism at temperature 0

Commands: `bench.py determinism --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit --runs 20 --gen 128` (prompt `determinism.txt`) and the same with `--prompt-file chat.txt --gen 256`. Each does 20 batch-1 runs, then puts the same prompt as member 0 of a `BatchGenerator` batch of B = 1, 2, 4, 8, 16 (other members: different short questions). Server: `stream_client.py --same-prompt prompts/determinism.txt`, 20 sequential requests alone, then 20 more while 15 other clients keep the batch full.

| Test | Result |
|---|---|
| Batch 1, 20 runs, 128 tokens (octopus prompt) | 1 distinct output(s) of 20 |
| Batch 1, 20 runs, 256 tokens (chat prompt) | 1 distinct output(s) of 20 |
| Same prompt inside batch B = 2 / 4 / 8 / 16, 128 tokens | identical / identical / identical / identical |
| Same prompt inside batch B = 2 / 4 / 8 / 16, 256 tokens | identical / differs from token 221 / differs from token 221 / differs from token 132 |
| Server, 20 sequential requests alone | 1 distinct of 20 |
| Server, 20 requests under 15-client load | 1 distinct of 20; 20/20 identical to the solo answer |
| Speculative decoding vs plain greedy (9) | chat: differs from token 221 (4-bit) / 154 (bf16); copy prompt: identical |
| Prompt cache vs no cache, 8B bf16 (7) | differs from token ~16 (same facts) |

**For the course.** At batch 1 the Mac is perfectly repeatable (20/20 identical twice). Batching *can* change a greedy answer, but rarely and late: the 128-token answer was identical at every batch size and through the busy server, while the 256-token answer diverged at token 221 for B = 4 and 8 and at token 132 for B = 16. Batch-1 continued "your room. If you turn it towards you, the room gets brighter and warmer. If you turn"; inside a batch of 4 it became "your room. If you position it directly above you, the light is strong and warm. But if". The cause is the same each time: a different kernel or chunking sums the same numbers in a different order, and a near-tie between two tokens flips. Once one token differs, the rest of the text follows a different path.

## Cross-check: stock `mlx_lm.generate` vs the harness

`tools/cli_crosscheck.sh`: `mlx_lm.generate --model <repo> --prompt "$(cat prompts/chat.txt)" --max-tokens 256 --temp 0`, 1 warm-up + 3 runs, reading "Generation: N tokens, X tokens-per-sec" (EOS on; every run produced 256 tokens).

| Model | mlx_lm.generate tok/s (median of 3) | Harness tok/s (median of 5) | Difference |
|---|---|---|---|
| Llama-3.2-1B-Instruct-4bit | 486.12 | 488.85 | -0.6% |
| Llama-3.2-1B-Instruct-8bit | 305.84 | 308.51 | -0.9% |
| Llama-3.2-1B-Instruct-bf16 | 178.72 | 179.84 | -0.6% |
| Meta-Llama-3.1-8B-Instruct-4bit | 102.66 | 101.93 | +0.7% |
| Meta-Llama-3.1-8B-Instruct-8bit | 58.39 | 58.42 | -0.1% |
| Meta-Llama-3.1-8B-Instruct-bf16 | 31.01 | 31.00 | +0.0% |

## Caveats, failures and deviations

- 11 energy: powermetrics needs sudo (not used). Used zeus-apple-silicon (IOReport, no sudo): GPU channel only; CPU and DRAM counters did not update on macOS 27.0.1.
- 6 rotating cache: --max-kv-size is silently ignored for Llama models in mlx-lm 0.32.2 (llama.Model.make_cache override); RotatingKVCache built explicitly for Llama; the flag works for Qwen3.
- 5 KV quantization: --kv-bits only applies once the cache reaches --quantized-kv-start (default 5000 tokens); measured with quantized_kv_start=0.
- 9 speculative: acceptance from mlx-lm's per-token from_draft flag; first pass had an off-by-one-pass counting bug (superseded, see raw/superseded); rerun with exact pass reconstruction.
- 10 server: first run lost requests to connection resets at 15-16 concurrent connects (http.server listen backlog 5); rerun with a retrying, 20 ms-staggered client (superseded files kept).
- Thermal: Mac Studio fans; no throttling was visible (decode ranges stay within ±1% across trials and the roofs re-run matched the first run).
- Prefill for 8B used the default 2,048-token chunk; the LM head is evaluated only for the last position, as mlx-lm does.
- All throughput numbers are for mlx-lm 0.32.2 / MLX 0.32.4 kernels; other engines (llama.cpp, a future MLX) will differ, especially for batching, quantized KV and long-context attention.
- "MFU" figures use a self-measured matmul peak, because Apple publishes no GPU FLOPS.

## Disk

Downloaded to `~/.cache/huggingface/hub`: Llama-3.2-1B-Instruct-4bit 0.71 GB, Llama-3.2-1B-Instruct-8bit 1.32 GB, Llama-3.2-1B-Instruct-bf16 2.48 GB, Qwen3-1.7B-4bit 0.98 GB, Meta-Llama-3.1-8B-Instruct-4bit 4.53 GB, Meta-Llama-3.1-8B-Instruct-8bit 8.54 GB, Meta-Llama-3.1-8B-Instruct-bf16 16.08 GB. **Total 34.65 GB** (the HF cache went from 19.05 to 53.68 GB). The 906 MB prompt-cache files were deleted after use.
