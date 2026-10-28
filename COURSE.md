# Tokens per Second — how LLM inference really works

A video course in 26 short episodes (3.5–4.5 minutes each), narrated and animated. It follows one
request — a student pressing Enter in **Owl**, a study-buddy chatbot — from the keyboard to the last
token, and every technique that makes that trip faster, cheaper and possible for thousands of users at
once: the KV cache, batching, paging, speculative decoding, FlashAttention, quantization, MoE,
parallelism, the serving stack, and the Mac on your desk.

- **The two speed limits:** every step of a language model is limited by either **compute** (FLOPs) or
  **memory traffic** (bytes). The whole course is built on telling which one, and on the arithmetic
  that predicts the answer: tokens/s ≈ bandwidth ÷ bytes per token.
- **Owl, the running example:** Owl runs Llama-3.1-8B — **8 B parameters, 16 GB of weights in BF16,
  128 KiB of KV cache per token** — first for one student on a Mac, then for thousands on H100s, then
  back on the Mac. Owl-mini (Llama-3.2-1B, 4-bit) runs the live demos.
- **Measured, not just quoted:** decode speed, prefill speed, memory growth, batching and speculative
  decoding are measured on this Mac Studio (Apple M4 Max, 546 GB/s, 64 GB) with MLX
  (`research/measurements.md`). Every other number is computed in the scene from `owl_reference.py` or
  cited (`research/notes.md`).
- **Who it's for:** developers who know what a transformer and a token are. "AI, Explained" covered the
  model itself (episodes 23–24, 33, 37–39, 42); "Agents, Animated" episode 7 covered prompt caching
  from an agent's side. This course is about running the model.
- **Visual language:** a deep blue-green "silicon" stage; colours fixed across the course —
  compute AMBER · memory traffic CYAN · KV cache VIOLET · weights BLUE · requests PINK · tokens GREEN.
- **Format:** 1080p60 MP4 with captions (.srt) and HTML slides; series kit in `kit.py`.

Status: ✅ produced · 🎬 in production · ○ planned

## Module 1 · One Token, Two Speed Limits — amber

| # | Episode | The viewer will understand | Key concepts | Mechanism to animate | Running example | Status |
|---|---|---|---|---|---|---|
| 1 | **Press Enter** | A reply is one parallel prefill plus hundreds of one-token decode steps | tokenize, queue, prefill, decode loop, sample, detokenize and stream; TTFT, TPOT/ITL, throughput vs latency | a timeline: tokenize → one thick prefill bar → 300 thin decode ticks → SSE chunks; the TinyLM generating token by token | a 1,500-token prompt and a 300-token answer: TTFT ≈ 50 ms, decode ≈ 97% of the time | ✅ `ep01-press-enter` |
| 2 | **The Two Speed Limits** | Every step is limited by math or by memory traffic; for chat it's memory | FLOPs ≈ 2P per token, bytes per step, arithmetic intensity, the roofline, ridge point, MBU | the roofline drawn; a "bytes truck" hauling all 16 GB per token; tok/s = bandwidth ÷ bytes, checked on this Mac | H100 ceiling ≈ 209 tok/s; M4 Max ≈ 34, and Owl-8B BF16 measured here | ✅ `ep02-two-speed-limits` |
| 3 | **Prefill vs Decode** | The same model runs two different workloads | prefill compute-bound (intensity ≈ prompt length), decode memory-bound (≈ batch), TTFT vs TPOT, why the M5 speeds up prefill more than decode | two dots sliding on the roofline; prefill and decode tok/s measured side by side | Owl-mini prefill vs decode tok/s on this Mac | ✅ `ep03-prefill-vs-decode` |
| 4 | **The Batching Free Lunch** | Serving 64 users costs about the same weight traffic as serving one | batch size, aggregate vs per-user tokens/s, critical batch ≈ ridge, cost per token | one weight sweep feeding 1 → 64 rows; the batch curve rising, then bending at the ridge | measured batch 1–16 on this Mac; $/M tokens falling with batch | ✅ `ep04-batching-free-lunch` |

## Module 2 · The KV Cache — violet

| # | Episode | The viewer will understand | Key concepts | Mechanism to animate | Running example | Status |
|---|---|---|---|---|---|---|
| 5 | **The KV Cache** | The model remembers past tokens as keys and values, and that memory has a size | attention needs every past K and V, recompute vs cache, the size formula 2·L·n_kv·d_h·b, memory growth per token | the cache growing token by token, layer by layer; the formula filled in with Owl's numbers; TinyLM: cache == recompute | 128 KiB per token; a 128K-token chat = 16 GiB = the weights; slope measured on this Mac | ✅ `ep05-kv-cache` |
| 6 | **Shrinking the Cache** | Five tricks make the cache smaller | MHA → MQA → GQA → MLA, sliding windows, KV quantization (FP8/INT4), what each costs | query heads sharing K/V heads; latent compression; a rolling buffer; bits shrinking | users per H100 doubling with FP8 KV; KV slope halving with --kv-bits 8 on this Mac | ✅ `ep06-shrinking-the-cache` |
| 7 | **PagedAttention** | Fixed-size blocks and a block table end fragmentation | contiguous reservations, internal/external fragmentation, 16-token blocks, block table, copy-on-write, vLLM | a memory map: reservations with holes vs packed blocks; a fork sharing blocks | 20–38% used before vLLM → near-zero waste; n = 4 answers sharing a prompt | ✅ `ep07-paged-attention` |
| 8 | **Never Prefill Twice** | A shared prefix can be cached and reused | prefix caching, block hashes, radix trees (SGLang), cache hits, why provider cache pricing exists | a radix tree lighting up shared prefixes; a hash chain over blocks; TTFT with vs without | Owl's system prompt + a 50-question study session on one chapter; prompt caching measured | ✅ `ep08-never-prefill-twice` |
| 9 | **The Cost of a Long Memory** | Long contexts cost quadratically in prefill and linearly in decode | attention FLOPs vs context, KV bytes per step, attention sinks, rolling caches, KV offload to CPU/SSD | the attention share growing; a sink + window; KV moving down a memory hierarchy | a 128K-token textbook: TTFT ≈ 13 s on one H100; decode slowdown measured on this Mac | ✅ `ep09-long-memory` |

## Module 3 · Many Users at Once — pink

| # | Episode | The viewer will understand | Key concepts | Mechanism to animate | Running example | Status |
|---|---|---|---|---|---|---|
| 10 | **Continuous Batching** | Re-form the batch every step, not every request | static vs dynamic vs iteration-level batching (Orca), padding waste, slots, preemption | a 4-slot grid stepping: static batches waiting for the longest, continuous refilling slots | `continuous_batching()`: request E waits 12 steps vs 2 | ✅ `ep10-continuous-batching` |
| 11 | **Chunked Prefill** | One long prompt must not freeze everyone else's tokens | the decode stall, token budget per step, chunked prefill (Sarathi-Serve), ITL percentiles | a step-by-step token budget filled with decodes, then a prefill chunk | a student pasting a 10,000-token chapter while 60 others chat | ✅ `ep11-chunked-prefill` |
| 12 | **Split the Work** | Prefill and decode can run on different machines | disaggregation (DistServe, Splitwise, Mooncake), KV transfer, goodput vs throughput, SLOs | two GPU pools with a KV transfer arrow; an SLO-attainment curve | launch day: 10,000 students; DeepSeek's prefill and decode units | ✅ `ep12-split-the-work` |

## Module 4 · Faster Tokens — green

| # | Episode | The viewer will understand | Key concepts | Mechanism to animate | Running example | Status |
|---|---|---|---|---|---|---|
| 13 | **Guess and Check** | A small model drafts, the big one verifies — exactly lossless | speculative decoding, acceptance rate α, draft length γ, E = (1−α^(γ+1))/(1−α), rejection sampling | an accept/reject strip; the expected-tokens curve; the cost ratio | α = 0.8, γ = 4 → 3.36 tokens per pass; Owl-mini drafting for Owl-8B, measured here | ✅ `ep13-guess-and-check` |
| 14 | **Drafting Without a Draft Model** | The model can draft for itself | Medusa heads, EAGLE, n-gram / prompt-lookup decoding, multi-token prediction, why gains shrink at high batch | heads sprouting from the model; a tree of candidates; copying from the prompt | Owl quoting the student's notes back: prompt lookup | ✅ `ep14-drafting-without-a-draft` |
| 15 | **FlashAttention** | Exact attention, reordered to respect the memory hierarchy | HBM vs SRAM, the N×N matrix never stored, tiling, online softmax, IO-awareness, FA2/FA3 | tiles of Q, K, V sliding through fast memory; the running max and sum rescaled | the online softmax on x = [1, 2, 3, 0] in two tiles, checked | ✅ `ep15-flashattention` |
| 16 | **The CPU Tax** | At batch 1, launching work can cost more than doing it | kernel launches, Python overhead, fusion, CUDA graphs, torch.compile, Flash-Decoding (split-KV) | hundreds of tiny launches collapsing into one graph replay; split-KV across the GPU | gpt-fast 25.5 → 107 tok/s | ✅ `ep16-cpu-tax` |
| 17 | **The Last Step of a Token** | Turning logits into one token: sampling, grammars, and determinism | temperature, top-k/top-p, logits processors, JSON grammar masks, why temperature 0 differs on a busy server | a logit bar chart → mask → renormalise → pick; batch-size-dependent sums | Owl's JSON flashcards; determinism tested on this Mac | ✅ `ep17-last-step-of-a-token` |

## Module 5 · Fewer Bits, More GPUs — blue

| # | Episode | The viewer will understand | Key concepts | Mechanism to animate | Running example | Status |
|---|---|---|---|---|---|---|
| 18 | **Fewer Bits, Same Model** | Fewer bits per weight means fewer bytes per token | weight-only vs weight+activation quantization, scales and zero points, INT8, FP8 (E4M3/E5M2), outliers, SmoothQuant | one group of weights quantized step by step; an outlier column breaking INT8 | Owl-8B BF16 → 8-bit: decode measured here | ✅ `ep18-fewer-bits` |
| 19 | **Four Bits and Below** | 4-bit formats keep quality with clever scales | group-wise INT4, GPTQ, AWQ, GGUF k-quants, MXFP4/NVFP4, bits per weight; distillation and pruning in brief | super-blocks and scales; a bits-per-weight ledger | Owl-8B 16 GB → about 4.5 GB; 4-bit decode measured here | ✅ `ep19-four-bits` |
| 20 | **Mixture of Experts at Serve Time** | MoE reads few experts per token but must hold all of them | total vs active parameters, routing, experts lit up across a batch, expert parallelism, all-to-all | a router lighting experts; all-to-all dispatch and combine | Mixtral 46.7B / 12.9B active; DeepSeek-V3 671B / 37B | ✅ `ep20-moe-at-serve-time` |
| 21 | **When One GPU Isn't Enough** | Splitting a model across GPUs, and what the wires cost | tensor, pipeline, expert, context parallelism; all-reduce; NVLink vs PCIe; latency vs throughput | a matrix sliced across GPUs; all-reduce pulses; pipes of different widths | Owl-70B (141 GB) on 4 H100s | ✅ `ep21-one-gpu-isnt-enough` |

## Module 6 · Production & Your Mac — cyan

| # | Episode | The viewer will understand | Key concepts | Mechanism to animate | Running example | Status |
|---|---|---|---|---|---|---|
| 22 | **The Serving Stack** | What sits between the user and the GPU | engines (vLLM, SGLang, TensorRT-LLM, llama.cpp, MLX), OpenAI-compatible APIs, SSE streaming, routers, autoscaling, cold starts, multi-LoRA | a request travelling down the stack; SSE chunks; a cold-start bar | Owl's API: one request traced through the layers | ✅ `ep22-serving-stack` |
| 23 | **Your Mac as an Inference Server** | Unified memory makes big models local; bandwidth sets the speed | unified memory, what fits, tok/s ≈ bandwidth ÷ size, MLX vs llama.cpp, the Mac bandwidth ladder | weights + KV filling the Mac's memory; the bandwidth ladder; a live mlx-lm run | everything measured on this Mac, side by side with the ceilings | ✅ `ep23-your-mac` |
| 24 | **The Price of a Token** | Cost per million tokens from first principles | $/GPU-hour ÷ tokens/hour, utilisation, why output costs more than input, Little's law, capacity planning | the cost-vs-batch curve; prefill vs decode cost bars; L = λW | Owl's monthly bill for 10,000 students | ✅ `ep24-price-of-a-token` |
| 25 | **Keeping It Alive** | What to watch and what to do when load spikes | TTFT/TPOT percentiles, queue time, KV usage, preemption and swapping, admission control, graceful degradation, benchmarking | a dashboard replaying an incident; queues filling; KV saturation → preemptions | exam week: p99 TTFT alert from KV saturation | ✅ `ep25-keeping-it-alive` |
| 26 | **One Request, Revisited** | How every piece fits into one fast, cheap request | the whole path with every optimisation in place; the two speed limits revisited | Episode 1's timeline replayed, layer by layer faster | the same student, the same question, at launch scale | ✅ `ep26-one-request-revisited` |

## Paths through the course

- **Just the core:** 1 → 2 → 3 → 5 → 10 → 13 → 18 → 23.
- **Make it fast:** 2 → 4 → 10 → 11 → 13 → 15 → 16.
- **Make it cheap:** 4 → 6 → 8 → 19 → 20 → 24.
- **Run it yourself:** 1 → 2 → 5 → 19 → 22 → 23.

## Production notes

- Series kit: `kit.py` (palette, inference components, Owl, cards); component tour: `kit_demo.py`;
  canonical numbers and code: `owl_reference.py`; research: `research/notes.md`; measurements on this
  Mac: `research/measurements.md` and `.json` (load with `measured(...)`).
- Episodes are made by subagents from `briefs/epNN.md` (generated by `briefs/make_briefs.py` from
  `briefs/spec.py`), one shared narrator; each episode is reviewed by the coordinator (numbers,
  storyboards, `briefs/check.sh N`).
