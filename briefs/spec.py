"""Per-episode notes for "Tokens per Second" briefs: number → (slug, ClassName, [reference episodes],
research-notes sections, notes). Reference keys are defined in make_briefs.py (REFS)."""

M = "research/measurements.md (load numbers with owl_reference.measured(...))"

SPEC = {
    1: ("press-enter", "PressEnter", ["ag02", "ag06", "ag32"], "§0, §1, §2 (worked example 1), §13 (SSE)",
        "The series OPENER: hook with a student typing a question to Owl and the answer streaming back. Walk the six stages "
        "(tokenize → queue → prefill → decode loop → sample → detokenize/stream) on a Timeline: one thick AMBER prefill bar, "
        "then many thin CYAN decode ticks. Use §1's worked example (1,500-token prompt, 300-token answer on one H100: prefill "
        "≈ 49 ms, a decode step ≈ 6.5 ms at 75% MBU, decode ≈ 97% of the time) — compute it from owl_reference "
        "(prefill_time, step_time) and assert. Define TTFT and TPOT/ITL. Show TinyLM generating token by token (prefill, then "
        "decode with the cache) as the mechanism, with generate() as the ≤10-line code. Introduce Owl and the three numbers "
        "(8 B params, 16 GB of weights, 128 KiB per token) as a teaser for later episodes; end by naming the course's question: "
        "why is decode so slow, and what makes it fast? (Episode 2)."),
    2: ("two-speed-limits", "TheTwoSpeedLimits", ["ag07", "ag29", "ai37"], "§0, §2 (all), §14",
        "The core idea of the course. FLOPs per token ≈ 2P; bytes per decode step ≈ all weights; arithmetic intensity; the "
        "Roofline (kit) with the ridge ≈ 295 for the H100; batch-1 decode at intensity ≈ 1 sits far left on the memory slope. "
        "The ceiling: tok/s = bandwidth ÷ bytes → H100 ≈ 209 tok/s, M4 Max ≈ 34 tok/s for Owl-8B BF16 (decode_ceiling). "
        "Then the measured proof on THIS Mac (" + M + "): Owl-8B BF16 measured tok/s vs its 34 tok/s ceiling, and MBU = "
        "measured × bytes ÷ bandwidth. 'Multiply speed by size and you get bandwidth.' The H100's math units are idle ~99.7% "
        "at batch 1 (16 GFLOP ÷ 989 TFLOP/s = 16 µs vs 4.8 ms). Animate a 'bytes truck' hauling 16 GB per token."),
    3: ("prefill-vs-decode", "PrefillVsDecode", ["ag07", "ag29", "ai37"], "§1, §2 (prefill formulas, the llama.cpp Mac table, Apple M5 notes), §14",
        "Same model, two workloads: prefill intensity ≈ prompt length (compute-bound), decode intensity ≈ batch (memory-bound). "
        "Two dots on the Roofline sliding apart. Measured on this Mac: prefill tok/s vs prompt length and decode tok/s (" + M +
        "): prefill is many times faster per token. TTFT depends on prefill (compute); TPOT on decode (bandwidth). The llama.cpp "
        "table / Apple's M5 post: the M5 Max vs M4 Max speeds prefill ×3.4 but decode only ×1.17, because the M5 GPU adds "
        "compute (Neural Accelerators) while bandwidth grew ~12%."),
    4: ("batching-free-lunch", "TheBatchingFreeLunch", ["ag07", "ag32", "ai37"], "§2 (critical batch), §5 (intro), §15",
        "One weight sweep can serve many sequences: at batch B the same 16 GB is read once and used B times, so aggregate "
        "tok/s ≈ B × per-sequence until the ridge (critical batch ≈ a few hundred on an H100). step_time(batch=…) from "
        "owl_reference gives the curve; per-user speed barely drops at first. Measured on this Mac: aggregate vs per-sequence "
        "tok/s at batch 1–16 (" + M + "). Cost per million tokens falls with batch (§15: $7.20 → $0.23 example — recompute "
        "from a stated $/GPU-hour, dated). The catch: KV cache grows with batch × context (leads to Module 2)."),
    5: ("kv-cache", "TheKVCache", ["ag07", "ag06", "ai24"], "§3 (all), §2",
        "Why attention needs every past token's K and V; recompute (quadratic) vs cache (linear). The size formula "
        "2 · L · n_kv · d_h · bytes filled in with Owl's config → 128 KiB per token (kv_bytes_per_token); a 128K-token chat = "
        "16 GiB = the weights. Animate the cache growing token by token, layer by layer (cell_grid). Show TinyLM's cache == "
        "recompute check (owl_reference asserts it). Measured on this Mac: peak memory vs context, slope vs the formula (" + M +
        "). How many Owl conversations fit on one H100 (§3 worked example): memory, not compute, limits batch size."),
    6: ("shrinking-the-cache", "ShrinkingTheCache", ["in05", "ag07", "ai37"], "§3 (shrinking the cache, offloading), §11",
        "Five tricks, one family: MHA → MQA → GQA (Owl: 32 query heads share 8 KV heads, 4× smaller) → MLA (DeepSeek-V2/V3 "
        "latent compression: per-token KV smaller than Llama-8B's despite 671B params) → sliding windows → KV quantization (FP8, "
        "INT4). Costs of each (quality, compute). Users per H100 doubling with FP8 KV (§3). Measured on this Mac: KV slope with "
        "--kv-bits 8 and 4 (" + M + "). Keep each trick to one clear visual."),
    7: ("paged-attention", "PagedAttention", ["in05", "ag22", "ai37"], "§4 (all)",
        "Pre-vLLM servers reserved contiguous max-length memory per request: internal + external fragmentation; Kwon et al. "
        "2023 measured only 20.4–38.2% of KV memory actually used. PagedAttention: fixed 16-token blocks, a block table per "
        "sequence (like OS virtual memory), near-zero waste; copy-on-write lets parallel samples (n = 4 answers) share the "
        "prompt's blocks. Animate a memory map: reservations with holes vs packed blocks; a fork sharing blocks, then one "
        "block copied on write. Compute the waste numbers in the scene for a small illustrative set of requests (say so)."),
    8: ("never-prefill-twice", "NeverPrefillTwice", ["ag07", "in07", "ai32"], "§4 (prefix caching, radix tree), §11 (context caching APIs), §15",
        "Many requests share a prefix (Owl's system prompt; a whole textbook chapter for a 50-question study session). Prefix "
        "caching keeps their KV blocks keyed by a hash chain; SGLang's RadixAttention keeps a radix tree of prefixes with LRU "
        "eviction. Animate the radix tree lighting up. Measured on this Mac: TTFT with vs without a cached prompt "
        "(mlx_lm.cache_prompt; " + M + "). Why providers price cached input low (dated, sourced ratios only from §0/§11; "
        "Anthropic 0.1× (0.05× on the newest), OpenAI 'up to 95%' off). Link 'Agents, Animated' Episode 7 (the agent side) "
        "without repeating it."),
    9: ("long-memory", "TheCostOfALongMemory", ["in05", "ag06", "ai38"], "§11 (all), §3 (offloading), §7 (Flash-Decoding mention)",
        "Prefill attention grows quadratically with context, decode bytes linearly (KV per step). A 128K-token textbook: TTFT "
        "≈ 13 s on one H100 (§11, recompute and assert). Decode slowdown with context measured on this Mac (" + M + "). "
        "Remedies: attention sinks + rolling window (StreamingLLM), rotating caches (--max-kv-size, measured: memory flat), KV "
        "offload to CPU/SSD (a memory hierarchy). Honest trade-off: forgetting vs memory."),
    10: ("continuous-batching", "ContinuousBatching", ["in04", "ag22", "ag23"], "§5 (all)",
        "Static batching waits for the longest request; iteration-level (continuous) batching (Orca, OSDI 2022) refills a slot "
        "the step after a request finishes. Use owl_reference.static_batching / continuous_batching on the reference's request "
        "set (or your own, asserted): request E waits 12 steps vs 2. Animate a 4-slot grid stepping (cell_grid), PINK requests. "
        "Orca's 36.9× headline (same latency). Mention preemption when KV runs out (link Episode 7)."),
    11: ("chunked-prefill", "ChunkedPrefill", ["in10", "in01", "ag20"], "§5 (chunked prefill, worked example B), §1 (ITL)",
        "A student pastes a 10,000-token chapter while 60 others are mid-answer: one big prefill freezes everyone's decode "
        "(a long gap in their token streams, ITL p99 spike). Chunked prefill (Sarathi-Serve): a per-step token budget; decodes "
        "go first, a prefill chunk fills the rest. §5 worked example B (131 ms hiccup → 17.6 ms gaps, recompute). Animate a "
        "step-by-step token budget bar and the token streams of other users."),
    12: ("split-the-work", "SplitTheWork", ["in11", "ag24", "ag32"], "§5 (disaggregation, goodput), §9 (DeepSeek deployment), §1 (SLOs)",
        "Prefill and decode want different hardware settings and interfere; disaggregation (DistServe, Splitwise, Mooncake) "
        "runs them on separate pools and ships the KV cache between them. Goodput vs throughput: requests served within SLOs "
        "(DistServe: up to 7.4× more requests). The cost: KV transfer (compute its size for a 1,500-token Owl prompt: "
        "1,500 × 128 KiB). DeepSeek's production split (prefill vs decode units, §9). Launch day: 10,000 students."),
    13: ("guess-and-check", "GuessAndCheck", ["in02", "ag29", "ai43"], "§6 (all)",
        "Speculative decoding: a small draft model guesses γ tokens, the target verifies them in ONE pass (memory-bound, so "
        "checking several tokens costs about the same as one). Rejection sampling makes it exactly lossless (Leviathan et al. "
        "2023; Chen et al. 2023). Expected tokens per pass E = (1−α^(γ+1))/(1−α) (expected_tokens_per_pass): α = 0.8, γ = 4 → "
        "3.36. Speedup with draft cost c (speculative_speedup). Animate an accept/reject strip. Measured on this Mac: Owl-8B "
        "with Owl-mini drafting, on chat vs 'rewrite these notes' prompts (" + M + "). Batch size kills the free lunch (§6)."),
    14: ("drafting-without-a-draft", "DraftingWithoutADraftModel", ["in13", "ag30", "ai43"], "§6 (Medusa, EAGLE, n-gram, MTP, batch table)",
        "When there is no separate draft model: Medusa heads, EAGLE (feature-level drafting, trees of candidates), n-gram / "
        "prompt-lookup decoding (copy spans from the prompt — Owl quoting the student's notes back), multi-token prediction "
        "(DeepSeek-V3). Why gains fade at high batch (EAGLE-3's batch table). Animate heads sprouting from the model and a "
        "candidate tree; prompt lookup matching a span of the notes."),
    15: ("flashattention", "FlashAttention", ["ai37", "ai24", "in05"], "§7 (FlashAttention, online softmax worked example)",
        "Standard attention writes the N×N score matrix to HBM; FlashAttention tiles Q, K, V through on-chip SRAM and never "
        "stores it, using the online softmax (running max and sum, rescaled). It is exact. Work §7's example x = [1, 2, 3, 0] "
        "in two tiles step by step and assert it equals the plain softmax (compute in the scene). HBM vs SRAM sizes/bandwidths "
        "(cite). FA2/FA3 numbers (FA3 up to 740 TFLOPS FP16 on H100, ~75%). Link 'AI, Explained' Episode 37 (efficient "
        "attention) — this episode is the IO view, not the approximation view."),
    16: ("cpu-tax", "TheCPUTax", ["in02", "ag32", "ag20"], "§7 (decode kernels, CUDA graphs, torch.compile, Flash-Decoding)",
        "At batch 1 a decode step is hundreds of tiny GPU kernels; launching them from Python can take longer than running "
        "them. Fusion, CUDA graphs (record once, replay), torch.compile; gpt-fast 25.5 → 107 tok/s (§7, cite). Flash-Decoding / "
        "split-KV: spread one long sequence's attention across the GPU. Animate launches as a queue collapsing into one graph "
        "replay; a timeline of GPU busy vs idle."),
    17: ("last-step-of-a-token", "TheLastStepOfAToken", ["ai33", "in01", "ag27"], "§12 (all)",
        "From logits to one token: temperature, top-k, top-p (owl_reference.sample), penalties; JSON grammar masks (XGrammar / "
        "Outlines) for Owl's flashcards — mask invalid tokens, renormalise, pick. Why temperature 0 isn't deterministic on a "
        "busy server: floating-point sums depend on batch size (batch invariance; §12's 80 unique completions out of 1,000). "
        "Measured on this Mac: determinism at batch 1 (" + M + "). Animate a logit bar chart → mask → renormalise → pick. Link "
        "'AI, Explained' Episode 33 (sampling) briefly."),
    18: ("fewer-bits", "FewerBitsSameModel", ["ai42", "in02", "ag29"], "§8 (mechanism, formats, worked example)",
        "Fewer bits per weight = fewer bytes per token = faster decode. Scales and zero points; quantize one group of 8 weights "
        "step by step (§8 worked example, recompute); weight-only vs weight+activation; INT8 outliers (a column breaking the "
        "scale) and SmoothQuant; FP8 E4M3/E5M2 (near-lossless). Measured on this Mac: Owl-8B BF16 vs 8-bit decode tok/s and "
        "MBU (" + M + "). Link 'AI, Explained' Episode 42."),
    19: ("four-bits", "FourBitsAndBelow", ["in18", "ai42", "ag29"], "§8 (formats, Llama-3.1-8B memory by format, distillation and pruning)",
        "Group-wise INT4 with per-group scales; GPTQ (error-compensating), AWQ (protect salient channels), GGUF k-quants "
        "(super-blocks), MXFP4/NVFP4 (micro-scaled floats); bits-per-weight accounting (4-bit + scales ≈ 4.5 bpw → Owl-8B ≈ "
        "4.5 GB). Measured on this Mac: 4-bit decode tok/s vs BF16, and prefill ≈ similar (" + M + "); quantized formats use "
        "bandwidth less efficiently (dequant compute). Distillation and 2:4 pruning in one short section."),
    20: ("moe-at-serve-time", "MixtureOfExpertsAtServeTime", ["ai39", "in04", "ag26"], "§9 (all)",
        "MoE reads few experts per token but must keep all of them in memory: Mixtral 46.7B total / 12.9B active; DeepSeek-V3 "
        "671B / 37B. At batch 1 only 2 experts per layer are read (fast); across a batch, tokens light up all experts, so the "
        "memory-traffic savings shrink (§9: all 8 by batch ~16, recompute the probability). Expert parallelism and the "
        "all-to-all dispatch/combine. Link 'AI, Explained' Episode 39."),
    21: ("one-gpu-isnt-enough", "WhenOneGPUIsntEnough", ["in20", "ag24", "ag25"], "§10 (all)",
        "Owl-70B (141 GB in BF16) does not fit one 80 GB H100. Tensor parallelism splits each matrix (all-reduce every "
        "layer; needs NVLink: 900 GB/s vs PCIe 128 GB/s), pipeline parallelism splits layers (bubbles; throughput not latency), "
        "expert and context parallelism. TP4 ceiling ≈ 4 × bandwidth ÷ 141 GB (§10: ≈ 95 tok/s, recompute). Animate a matrix "
        "sliced across 4 GPUs and all-reduce pulses through pipes of different widths."),
    22: ("serving-stack", "TheServingStack", ["ag32", "ag16", "in10"], "§13 (all), §17 (multi-LoRA)",
        "What sits between Owl's user and the GPU: client → load balancer/router (prefix-aware routing) → OpenAI-compatible API "
        "server → scheduler → engine (vLLM, SGLang, TensorRT-LLM, llama.cpp, MLX) → GPU; streaming as server-sent events; "
        "autoscaling and cold starts (16 GB load: seconds from NVMe vs much longer from a slow disk, §13); multi-LoRA (S-LoRA) "
        "serving many fine-tunes on one base. One request traced down the stack with real JSON request/SSE chunks. Note TGI "
        "is in maintenance mode (dated)."),
    23: ("your-mac", "YourMacAsAnInferenceServer", ["in02", "in19", "ag28"], "§14 (all), §2 (hardware table)",
        "The Mac episode: unified memory (64 GB here) holds weights + KV; what fits; tok/s ≈ bandwidth ÷ size; the Apple "
        "bandwidth ladder (M4 120 → M4 Pro 273 → M4 Max 546 → M5 Max 614 → M5 Ultra 1,200 GB/s, dated); MLX vs llama.cpp. "
        "Put EVERY measurement from " + M + " on one board next to its ceiling: decode by bits, prefill, batching, spec "
        "decoding, TTFT/ITL percentiles through mlx_lm.server. Show the actual commands in a terminal card."),
    24: ("price-of-a-token", "ThePriceOfAToken", ["ag32", "in04", "ag29"], "§15 (all), §0 (#14–15)",
        "Cost per million tokens from first principles: $/GPU-hour ÷ tokens/hour at a given batch and utilisation (state the "
        "$/hour as dated and 'for example' unless the notes give a sourced number). Why output costs ~4–6× input on APIs "
        "(decode is memory-bound, prefill compute-bound). Little's law L = λW: 20 req/s × W → concurrent sequences (§15: 124 "
        "on one H100, recompute). Owl's monthly bill for 10,000 students. Reasoning models shift cost to decode (§17)."),
    25: ("keeping-it-alive", "KeepingItAlive", ["ag32", "ag20", "in11"], "§16 (all), §5 (preemption)",
        "What to watch: TTFT and TPOT percentiles (p50/p99), queue time, KV-cache usage, preemptions/swaps, tokens/s; "
        "benchmarking methodology (load generators, percentiles). Exam-week incident replayed on a dashboard: KV saturation "
        "→ preemptions → p99 TTFT alert; responses: admission control, max-tokens caps, priority, shedding load, graceful "
        "degradation (smaller model, shorter context). Keep numbers illustrative and labelled 'for example' unless sourced."),
    26: ("one-request-revisited", "OneRequestRevisited", ["in01", "in02", "ag33"], "§0, §1, and every section briefly",
        "The FINALE. Replay Episode 1's request (same student, same 1,500-token prompt, 300-token answer) at launch scale, "
        "adding each layer the course taught with its episode number as a badge: batching (4), KV cache (5), paging (7), "
        "prefix cache (8), continuous batching + chunked prefill (10–11), speculative decoding (13), FlashAttention (15), "
        "4-bit weights (19), the serving stack (22) — the Timeline shrinking and tokens/s, users per GPU and $/M tokens "
        "improving, each number computed from owl_reference (assert). End on the two speed limits once more and a recap; "
        "override series_outro() with a course-closing card ('That's Tokens per Second …', the six module names in their "
        "colours, 'Thanks for watching!')."),
}
