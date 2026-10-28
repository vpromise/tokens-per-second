"""Render research/measurements.md from research/measurements.json.
Tables are generated from the JSON (one source of truth); prose is written here.
Usage: python -I write_measurements_md.py"""
import json

R = "/Users/vpromisex/videos/inference-series/research"
d = json.load(open(f"{R}/measurements.json"))
H = d["headline"]
L = []
w = L.append


def f(x, n=1):
    return f"{x:,.{n}f}"


def rg(r, n=1):
    return f"[{r[0]:.{n}f}–{r[1]:.{n}f}]"


def table(head, rows):
    w("| " + " | ".join(head) + " |")
    w("|" + "|".join("---" for _ in head) + "|")
    for r in rows:
        w("| " + " | ".join(str(c) for c in r) + " |")
    w("")


m = d["machine"]
w("# Inference, Explained: live measurements on this Mac (mlx-lm)")
w("")
w("Measured **2026-10-24** (00:40–03:30 local) by the measurement agent. Machine-readable twin: `research/measurements.json` (stable keys; the `headline` block is meant for scene assertions). Raw per-run JSON: `research/measurements/raw/`; logs: `measurements/logs/`; prompts: `measurements/prompts/`; scripts: `measurements/scripts/` (same files as `tools/`).")
w("")
w("## Machine, software, conditions")
w("")
table(["Item", "Value"], [
    ["Machine", "Mac Studio (2025), Mac16,9"],
    ["Chip", "Apple M4 Max: 16-core CPU (12 performance + 4 efficiency), 40-core GPU"],
    ["Memory", "64 GB unified; MLX recommended working set 55.7 GB"],
    ["Bandwidth (spec)", "**546 GB/s**. Apple Support, Mac Studio (2025) Tech Specs (support.apple.com/en-us/122211): \"M4 Max with 16‑core CPU, 40‑core GPU, and 16‑core Neural Engine (546GB/s memory bandwidth)\" (the 14-core/32-core M4 Max is 410 GB/s)"],
    ["OS", "macOS 27.0.1 (build 26A434); AC power; Low Power Mode off"],
    ["Python env", "`tools/.venv` (uv 0.12.11, CPython 3.12)"],
    ["Versions", "mlx 0.32.4, mlx-metal 0.32.4, **mlx-lm 0.32.2**, transformers 5.20.2, zeus-apple-silicon 1.1.0 (energy only)"],
])
w("Conditions: no other GPU work (no rendering, narrator TTS never started). macOS background daemons (dasd, ANECompilerService) were using CPU at the start. The 8B 8-bit/bf16 downloads were still running (network and disk only) during the first 1B and 8B-4bit runs; the 8B-4bit decode re-checked after the downloads with the stock CLI gives 102.7 vs 101.9 tok/s, so they had no visible effect.")
w("")
w("**Method.** `tools/bench.py` loads each model once, does **1 warm-up then 5 timed trials** (3 for batching, speculative decoding and energy), and reports **median [min–max], n**. Timing uses `time.perf_counter()` at every token yielded by `mlx_lm.stream_generate`:")
w("- **TTFT** = first token time − call time (prompt already tokenized); **prefill tok/s** = prompt tokens ÷ TTFT (this is exactly how mlx-lm computes its `prompt_tps`).")
w("- **Decode tok/s** = (N − 1) ÷ (t_last − t_first). mlx-lm prints N ÷ (that interval), which reads ≈0.4% higher at N = 256.")
w("- Speed runs disable EOS (as `mlx_lm.benchmark` does), so every trial generates exactly the same number of tokens. Sampling is greedy (temperature 0) everywhere.")
w("- Fixed prompts: `prompts/chat.txt` (Owl seasons question, 95 tokens with the Llama chat template), `prompts/notes_rewrite.txt` (copy-these-notes, 370 tokens), `prompts/corpus.txt` (the course notes, used for long prompts), `prompts/chapter.txt` + `question.txt` (prompt caching), `prompts/determinism.txt`.")
w("")
w("**Bytes.** *Weights GB* = sum of the `*.safetensors` files on disk. *Ceiling* = 546 GB/s ÷ weights GB (the course's \"bandwidth ÷ bytes\" rule). *Bytes read per decode step* = every tensor except an **untied** input-embedding table (that is a row lookup: Llama-3.1-8B's 1.05 GB `embed_tokens` is not streamed per token), plus the KV cache at the average context of the run. Llama-3.2-1B and Qwen3-1.7B tie embeddings, so their table doubles as the LM head and is read in full. *MBU* = tok/s × bytes read per step ÷ 546 GB/s (Databricks' definition).")
w("")

# ---------------------------------------------------------------- models
w("## Models")
w("")
rows = []
for fam, qs in d["models"].items():
    for q, v in qs.items():
        rows.append([fam, q, f"`{v['repo']}`", f(v["weights_gb"], 3), f(v["decode_bytes_per_token"] / 1e9, 3), "yes" if v["tied_embeddings"] else "no", f"{v['kv_formula']} = **{v['kv_bytes_per_token_bf16']:,} B** ({v['kv_bytes_per_token_bf16'] // 1024} KiB)"])
table(["Model", "Bits", "Repo", "Weights GB (files)", "Bytes read / decode step GB (weights)", "Tied emb.", "KV per token, bf16 cache"], rows)
w("Quantized repos use MLX affine quantization, group size 64, bf16 scale + bias per group (4-bit ≈ 4.5 bits/weight, 8-bit ≈ 8.5). Parameter counts (from the bf16 headers): Llama-3.2-1B 1.236 B total, 0.973 B in the transformer blocks; Llama-3.1-8B 8.030 B total, 6.980 B in the blocks (embedding and LM head 0.525 B each).")
w("")

# ---------------------------------------------------------------- roofs
ro = d["roofs"]
w("## 0. Machine roofs (micro-benchmarks, `bench.py roofs`)")
w("")
table(["Test", "Median", "Range", "vs 546 GB/s spec"], [
    ["Streaming read, `mx.sum` over 4 GB", f"{f(ro['read_GBps']['median'])} GB/s", rg(ro["read_GBps"]["range"]), f"{ro['read_fraction_of_spec']*100:.0f}%"],
    ["bf16 mat-vec (28672×16384, 0.94 GB)", f"{f(ro['matvec_bf16_GBps']['median'])} GB/s", rg(ro["matvec_bf16_GBps"]["range"]), f"{ro['matvec_bf16_GBps']['median']/546*100:.0f}%"],
    ["4-bit mat-vec (same W, quantized_matmul)", f"{f(ro['matvec_q4_GBps']['median'])} GB/s", rg(ro["matvec_q4_GBps"]["range"]), f"{ro['matvec_q4_GBps']['median']/546*100:.0f}%"],
    ["Copy (read + write), 4 GB", f"{f(ro['copy_GBps_read_plus_write'])} GB/s", "", ""],
    ["bf16 matmul 8192³", f"{ro['matmul_bf16_TFLOPS']['median']:.1f} TFLOPS", rg(ro["matmul_bf16_TFLOPS"]["range"]), ""],
    ["fp16 / fp32 matmul 8192³", f"{ro['matmul_fp16_TFLOPS']:.1f} / {ro['matmul_fp32_TFLOPS']:.1f} TFLOPS", "", ""],
])
rr = ro.get("rerun_end_of_session", {})
w(f"Re-run at the end of the session: read {f(rr.get('read_sum_4GB_GBps', 0))} GB/s, bf16 mat-vec {f(rr.get('matvec_bf16_GBps', 0))} GB/s, matmul {rr.get('matmul_bf16_8192_TFLOPS', 0):.1f} TFLOPS (stable within 2%).")
w("")
w(f"**For the course.** A pure streaming read on this Mac reaches **≈{ro['read_GBps']['median']:.0f} GB/s ({ro['read_fraction_of_spec']*100:.0f}% of Apple's 546)** (488 GB/s = 89% on a 1 GB array), so roughly 88–89% of spec is the practical top of the bandwidth roof. Apple publishes no GPU TFLOPS; the measured bf16 matmul is **{ro['matmul_bf16_TFLOPS']['median']:.1f} TFLOPS**, which puts the **ridge point at ≈{ro['ridge_flops_per_byte_spec_bw']:.0f} FLOPs/byte** (15.2e12 ÷ 546e9), about a tenth of the H100's ~295. The Mac is far more bandwidth-rich relative to its compute.")
w("")

# ---------------------------------------------------------------- decode
w("## 1. Decode speed vs bit width (batch 1)")
w("")
w("Command: `tools/.venv/bin/python tools/bench.py decode --model <repo> --tag <tag> --trials 5` (prompt `chat.txt`, 256 generated tokens, EOS off). Stock-CLI equivalent (cross-check below): `mlx_lm.generate --model <repo> --prompt \"$(cat prompts/chat.txt)\" --max-tokens 256 --temp 0`.")
w("")
rows = []
for fam, qs in d["decode"].items():
    for q, v in qs.items():
        rows.append([fam, q, f(v["weights_gb"], 3), f"**{v['tok_s_median']:.2f}** {rg(v['tok_s_range'], 2)}", v["n"], f(v["ceiling_tok_s"]), f"{v['mbu_vs_file_bytes']*100:.0f}%",
                     f(v["bytes_read_per_step"] / 1e9, 3), f(v["ceiling_tok_s_bytes_read"]), f"**{v['mbu']*100:.1f}%**", f(v["achieved_GBps"], 0), f"{v['fraction_of_measured_read_bw']*100:.0f}%", f"{v['speedup_vs_bf16']:.2f}×", f"{v['bytes_ratio_vs_bf16']:.2f}×"])
table(["Model", "Bits", "Weights GB", "Decode tok/s median [range]", "n", "Ceiling 546/GB", "% of ceiling", "Bytes read/step GB", "Ceiling on bytes read", "MBU", "Achieved GB/s", "% of measured read roof", "Speed vs bf16", "Fewer bytes vs bf16"], rows)
d8 = d["decode"]["llama-3.1-8b"]
w(f"**Headline.** The course's reference model, **Llama-3.1-8B in bf16 (16.06 GB of weights), decodes at {d8['bf16']['tok_s_median']:.1f} tok/s** ({1000/d8['bf16']['tok_s_median']:.1f} ms per token). The \"bandwidth ÷ bytes\" ceiling is 546 ÷ 16.06 = **{d8['bf16']['ceiling_tok_s']:.1f} tok/s**, so the Mac reaches **{d8['bf16']['mbu_vs_file_bytes']*100:.0f}% of the ceiling**. Counting only the bytes really streamed per token (15.04 GB: no input-embedding table, plus 23 MB of KV), MBU is **{d8['bf16']['mbu']*100:.0f}%**, i.e. {d8['bf16']['achieved_GBps']:.0f} GB/s, which is **{d8['bf16']['fraction_of_measured_read_bw']*100:.0f}% of what a plain streaming read achieves** on this machine. Multiply speed by size and you get bandwidth: 31.0 tok/s × 15.04 GB ≈ 466 GB/s.")
w("")
w(f"- **4-bit is {d8['4bit']['speedup_vs_bf16']:.2f}× faster than bf16** ({d8['4bit']['tok_s_median']:.1f} vs {d8['bf16']['tok_s_median']:.1f} tok/s) while moving 3.56× fewer bytes; 8-bit is {d8['8bit']['speedup_vs_bf16']:.2f}× (bytes ratio 1.88×). This matches GPTQ's \"~3.25×\" headline for 4-bit batch-1 decode (notes §0 #12).")
w("- **Quantized formats use bandwidth less efficiently** (notes §2 lesson c): MBU 79% (4-bit) vs 86% (8-bit) vs 85% (bf16) on 8B; 63% / 75% / 82% on the 1B. Dequantizing costs compute; the 4-bit mat-vec micro-benchmark also tops out lower (427–431 GB/s vs 466–475 GB/s for bf16).")
w("- **Small models waste a larger share of each step.** Llama-3.2-1B-4bit takes 2.05 ms per token, of which only ≈1.28 ms is the 0.70 GB read at 546 GB/s; the rest is fixed per-step cost (presumably kernel launches for 16 layers, sampling and Python overhead; not profiled). That is why its MBU is only 63%.")
w("- **Cross-check against the ceiling:** every measured number is below both the 546/GB ceiling and the bytes-read ceiling. Without the untied-embedding correction, 8B-4bit would look like 84% of a 120.9 tok/s ceiling; the correct ceiling is 128.4.")
w("- Greedy outputs were identical across the 5 trials for every model.")
w("")

# ---------------------------------------------------------------- prefill
w("## 2. Prefill (TTFT) vs prompt length")
w("")
w("Command: `bench.py prefill --model <repo> --lengths 128 256 512 1024 2048 4096 8192 16384 --trials 5` (prompt = the first N tokens of `corpus.txt`, max_tokens 1, default `prefill_step_size` 2048). Cross-check: `mlx_lm.benchmark --model <8B-4bit> -p 2048 -g 128 -n 3` gave prompt_tps 943.3 (harness 940.2).")
w("")
pf = d["prefill"]
cols = [("llama-3.2-1b", "4bit"), ("llama-3.2-1b", "bf16"), ("llama-3.1-8b", "4bit"), ("llama-3.1-8b", "8bit"), ("llama-3.1-8b", "bf16")]
rows = []
for T in ["128", "256", "512", "1024", "2048", "4096", "8192", "16384"]:
    r = [T]
    for fam, q in cols:
        x = pf[fam][q]["by_prompt_tokens"][T]
        r.append(f"{f(x['tok_s_median'], 0)} {rg(x['tok_s_range'], 0)}")
    x = pf["llama-3.1-8b"]["bf16"]["by_prompt_tokens"][T]
    r.append(f"{x['ttft_s_median']:.3f}")
    r.append(f"{pf['llama-3.1-8b']['4bit']['by_prompt_tokens'][T]['ttft_s_median']:.3f}")
    rows.append(r)
table(["Prompt tokens", "1B 4-bit tok/s", "1B bf16 tok/s", "8B 4-bit tok/s", "8B 8-bit tok/s", "8B bf16 tok/s", "8B bf16 TTFT s", "8B 4-bit TTFT s"], rows)
rows = []
for fam, q in cols:
    x = pf[fam][q]["by_prompt_tokens"]["2048"]
    rows.append([fam, q, f(x["tok_s_median"], 0), f(d["decode"][fam][q]["tok_s_median"], 1), f"**{pf[fam][q]['prefill_over_decode_ratio_at_2048']:.1f}×**", f"{x['achieved_TFLOPS']:.1f}", f"{x['fraction_of_measured_matmul_peak']*100:.0f}%"])
table(["Model", "Bits", "Prefill tok/s @2048", "Decode tok/s", "Prefill ÷ decode", "Achieved TFLOPS (2·P_blocks + attention)", "of measured 15.2 TFLOPS"], rows)
w(f"**For the course.** Prefill is compute-bound and runs **{H['prefill_over_decode_8b_bf16']:.0f}× faster per token than decode for 8B bf16** ({H['prefill_8b_bf16_tok_s_at_2048']:.0f} vs 31 tok/s), and {H['prefill_over_decode_8b_4bit']:.1f}× for 8B 4-bit. Prefill reaches 13.6–14.6 TFLOPS, 89–96% of the measured matmul roof; batch-1 decode on the same GPU uses only ≈0.5 TFLOPS (31 tok/s × ~15 GFLOP). Quantization helps decode, not prefill: bf16 prefills fastest (1,005 tok/s), because quantized weights add dequantization work to every matmul. TTFT is roughly proportional to prompt length: 8B bf16 needs 0.21 s for 128 tokens, 2.0 s for 2,048 and 20.5 s for 16,384. Per-token prefill speed falls at long prompts because attention work grows with length (1,005 → 801 tok/s from 2K to 16K) and is lower for very short prompts, whose matrices are too small to fill the GPU (621 tok/s at 128).")
w("")
w("FLOPs per prompt token = 2 × 6.98 B (block parameters; the LM head is not evaluated during chunked prefill) + 2·L·d·T for causal attention. \"Measured roof\" = the 8192³ bf16 matmul above; Apple publishes no TFLOPS, so this is a self-measured peak, not a spec.")
w("")

# ---------------------------------------------------------------- context
w("## 3. Decode slowdown with context length")
w("")
w("Command: `bench.py context --model <repo> --lengths 128 512 2048 8192 16384 32768 --gen 128 --trials 5`. The prompt (T−1 corpus tokens) is prefilled once; each trial decodes 128 tokens from that cache and is then trimmed back (`trim_prompt_cache`), so every trial sees the same context. *Predicted* = tok/s at T=128 × bytes(128) ÷ bytes(T), with bytes(T) = weights read + (T + 64) × KV/token.")
w("")
cx = d["context"]
rows = []
for T in ["128", "512", "2048", "8192", "16384", "32768"]:
    r = [T]
    for fam, q in (("llama-3.1-8b", "bf16"), ("llama-3.1-8b", "4bit"), ("llama-3.2-1b", "4bit"), ("qwen3-1.7b", "4bit")):
        x = cx[fam][q]["by_context_tokens"][T]
        r.append(f"{x['tok_s_median']:.1f} ({x['predicted_tok_s_bytes_model']:.1f}) · KV {x['kv_GiB']:.2f} GiB")
    rows.append(r)
table(["Context tokens", "8B bf16 tok/s (pred.) · KV", "8B 4-bit tok/s (pred.) · KV", "1B 4-bit tok/s (pred.) · KV", "Qwen3-1.7B 4-bit tok/s (pred.) · KV"], rows)
q32 = cx["qwen3-1.7b"]["4bit"]["by_context_tokens"]["32768"]
w(f"Ranges are tight (spread ≤ 3.5%; all in the JSON). **For the course.** Every decode step reads the whole KV cache as well as the weights. At 32K tokens the 8B's cache is 4 GiB, about the size of its 4-bit weights, so **8B-4bit decode halves (101.9 → 51.0 tok/s)**, and the bytes model predicts 50.8. For bf16 the same 4 GiB is a smaller share of the bytes (16 GB of weights): 30.8 → 23.8 tok/s against a prediction of 24.0. **Small model, big KV:** Qwen3-1.7B stores 112 KiB per token (3.5× Llama-1B's 32 KiB) and slows 3.5× by 32K (328 → 95 tok/s), when its cache (3.5 GiB) is 3.9× its weights. At that point it streams {q32['achieved_GBps']:.0f} GB/s. Llama-1B slows 2.3× (497 → 216).")
w("")
w("Caveat: for the 8B models the bytes model is within 1–2%. For the 1B-class models it is only a rough guide: at short context their step is partly fixed overhead (MBU 63%), and the attention kernel's efficiency changes with length (it does better at 32K than at 8K).")
w("")

# ---------------------------------------------------------------- memory
w("## 4. Memory vs context (KV-cache slope)")
w("")
w("Command: `bench.py memory --model <repo> --lengths 1024 2048 4096 8192 16384 32768 --gen 16`. For each T: prefill T corpus tokens, generate 16, then read `mx.get_peak_memory()` (reset per run), `mx.get_active_memory()` (cache alive) and the cache's own `nbytes`. Slopes are least-squares fits over the six points. Memory is deterministic: two repeats gave identical peaks.")
w("")
mem = d["memory"]
rows = []
for fam, q, k in (("llama-3.1-8b", "4bit", "kv16"), ("llama-3.2-1b", "4bit", "kv16"), ("qwen3-1.7b", "4bit", "kv16")):
    v = mem[fam][q][k]
    rows.append([fam, f"{v['formula_bytes_per_token']:,.0f}", f"**{v['slope_active_bytes_per_token']:,.0f}**", f"{v['slope_kv_cache_bytes_per_token']:,.0f}", f"{v['slope_over_formula']:.4f}", f"{v['slope_peak_bytes_per_token']:,.0f}"])
table(["Model", "Formula 2·L·n_kv·d_h·2 B", "Active-memory slope B/token", "Cache nbytes slope", "Slope ÷ formula", "Peak-memory slope"], rows)
rows = []
for T in ["1024", "2048", "4096", "8192", "16384", "32768"]:
    a = mem["llama-3.1-8b"]["4bit"]["kv16"]["by_context_tokens"][T]
    b = mem["llama-3.2-1b"]["4bit"]["kv16"]["by_context_tokens"][T]
    c = mem["qwen3-1.7b"]["4bit"]["kv16"]["by_context_tokens"][T]
    rows.append([T, f"{a['peak_GB']:.2f} / {a['active_GB']:.2f} / {a['kv_cache_MiB']:.0f}", f"{b['peak_GB']:.2f} / {b['active_GB']:.2f} / {b['kv_cache_MiB']:.0f}", f"{c['peak_GB']:.2f} / {c['active_GB']:.2f} / {c['kv_cache_MiB']:.0f}"])
table(["Context", "8B 4-bit: peak GB / active GB / KV MiB", "1B 4-bit: same", "Qwen3-1.7B 4-bit: same"], rows)
w("**For the course.** The KV formula is exact on this machine: the measured slope is **131,072 B/token = 128 KiB** for Llama-3.1-8B, 32,768 B (32 KiB) for Llama-3.2-1B and 114,688 B (112 KiB) for Qwen3-1.7B. A 32K-token conversation adds 4.0 GiB to the 8B (4-bit weights 4.5 GB → 8.85 GB active). mlx-lm allocates the cache in 256-token steps (a 1,040-token cache holds 1,280 slots), so per-T cache sizes are rounded up to the next 256. The peak includes prefill transients (activations for a 2,048-token chunk), so it sits ≈0.6–0.75 GB above active memory.")
w("")

# ---------------------------------------------------------------- kv quant
w("## 5. KV-cache quantization (`--kv-bits 8 / 4`, group size 64)")
w("")
w("Command: as 4 plus `--kv-bits 8` or `--kv-bits 4` (harness passes `kv_bits`, `kv_group_size=64`, **`quantized_kv_start=0`**). Trap: mlx-lm's default `--quantized-kv-start` is **5000**, so `--kv-bits` alone does nothing until the cache passes 5,000 tokens.")
w("")
rows = []
for fam in ("llama-3.1-8b", "llama-3.2-1b"):
    base = mem[fam]["4bit"]["kv16"]["slope_active_bytes_per_token"]
    for k in ("kv16", "kv8", "kv4", "kv8__step256"):
        if k not in mem[fam]["4bit"]:
            continue
        v = mem[fam]["4bit"][k]
        t32 = v["by_context_tokens"]["32768"]
        rows.append([fam, k.replace("__step256", " (prefill step 256)"), f"{v['slope_active_bytes_per_token']:,.0f}", f"{v['formula_bytes_per_token']:,.0f}", f"{v['slope_active_bytes_per_token']/base:.3f}", f"{t32['kv_cache_MiB']:,.0f}", f"{t32['active_GB']:.2f}", f"{t32['peak_GB']:.2f}", f"{t32['decode_tok_s']:.0f}"])
table(["Model (4-bit weights)", "KV", "Slope B/token", "Expected", "vs bf16 KV", "KV MiB @32K", "Active GB @32K", "Peak GB @32K", "Decode tok/s @32K (1 run, 16 tok)"], rows)
kq = d["kv_quant_quality"]["by_kv_bits"]
w(f"**Size: exactly as predicted.** Each quantized element costs bits/8 bytes plus a bf16 scale and bias per group of 64 (4/64 B), so 8-bit KV = 0.531× and 4-bit KV = 0.281× of bf16: 69,632 and 36,864 B/token for the 8B, which is what was measured.")
w("")
w("**But in mlx-lm 0.32.2 it does not lower the peak and does not speed up decode at batch 1.**")
w("- *Peak memory goes up* for prompts longer than one 2,048-token prefill chunk: 8B-4bit at 32K peaks at 16.0 GB (8-bit KV) vs 9.5 GB (bf16 KV). The quantized attention path (`quantized_scaled_dot_product_attention` in `mlx_lm/models/base.py`) is not a fused kernel and materializes the chunk × context score matrix. A smaller prefill chunk tames it (1B: 10.4 → 3.6 GB with `--prefill-step-size 256`), though that is still above bf16 KV's 2.46 GB.")
w("- *Speed:* decode at 32K is about the same (8B: 51 vs 50 tok/s for 8-bit, 58 for 4-bit; 1B: slower). The bytes saved are spent on the unfused dequantize-and-attend path.")
w("- So on this stack KV quantization buys **capacity** (more or longer conversations resident), not speed.")
w("")
w(f"Quality spot-check (`bench.py kvquality`, 8B-4bit, an {d['kv_quant_quality']['prompt_tokens']:,}-token prompt of course notes + the question \"how many bytes of KV cache does Llama-3.1-8B store per token, and why?\", 64 greedy tokens): bf16 KV and 8-bit KV give **identical** answers (64/64 tokens); 4-bit KV matches for the first {kq['4']['tokens_identical_to_bf16_prefix']} tokens, then rephrases with the same facts. All three answer \"128 KiB … 32 layers, 8 KV heads, head dimension 128\".")
w("")

# ---------------------------------------------------------------- rotating
w("## 6. Rotating (sliding-window) KV cache, `max_kv_size = 1024`")
w("")
w("Command: `bench.py rotating --model <repo> --max-kv-size 1024 --gen 8192 --forget-ctx 3000` (EOS off; samples every 512 generated tokens).")
w("")
w("**Trap found: `--max-kv-size` is silently ignored for Llama models in mlx-lm 0.32.2.** `llama.Model` defines `make_cache()`, and `make_prompt_cache(model, max_kv_size)` defers to it, which returns plain `KVCache`. The flag therefore has no effect, as confirmed with the stock CLI (`logs/maxkv_cli_check.log`): Llama-1B peak 0.939 vs 0.943 GB with and without `--max-kv-size 256` over 3,000 tokens. Qwen3 has no override, so the flag works there (1.465 → 1.258 GB, 285 → 327 tok/s). For Llama the harness builds mlx-lm's own `RotatingKVCache(max_size=1024, keep=4)` explicitly, which is what the flag intends.")
w("")
rc = d["rotating_cache"]
rows = []
for fam in ("llama-3.2-1b", "qwen3-1.7b"):
    v = rc[fam]["4bit"]
    for n in ("1024", "2048", "4096", "8192"):
        a, b = v["full_cache"]["points"][n], v["rotating"]["points"][n]
        rows.append([fam, n, f"{a['kv_MiB']:.0f} / {a['peak_GB']:.3f} / {a['window_tok_s']:.0f}", f"{b['kv_MiB']:.0f} / {b['peak_GB']:.3f} / {b['window_tok_s']:.0f}"])
table(["Model (4-bit)", "Tokens generated", "Full cache: KV MiB / peak GB / tok/s (last 512)", "Rotating 1024: KV MiB / peak GB / tok/s"], rows)
fd = rc["llama-3.2-1b"]["4bit"]["forgetting_demo"]
w(f"**For the course.** With the window, memory and speed stay **flat**: KV is pinned at 32 MiB (1,024 × 32 KiB) and decode holds ≈434 tok/s, while the full cache grows to 264 MiB and slows to 285 tok/s by 8K tokens. The price is forgetting. A {fd['prompt_tokens']:,}-token prompt that starts \"Remember this: the secret code word is PELICAN\" and ends by asking for the code word gets `{fd['full']}` with the full cache and `{fd['rotating_1024']!s}` with the 1,024-token window (the first 4 tokens are kept as attention sinks, but the fact is not among them). Qwen3's demo answer was cut off inside its `<think>` block (12 tokens), so it is not informative.")
w("")

# ---------------------------------------------------------------- prompt cache
w("## 7. Prompt (prefix) caching")
w("")
w("Harness: `bench.py promptcache --model <repo> --trials 5 --gen 48`: build the KV cache for the chapter prefix exactly as `mlx_lm.cache_prompt` does (chat template, `continue_final_message=True`), save it with `save_prompt_cache`, then time TTFT for (a) the full prompt from scratch and (b) only the 45-token question suffix on a freshly loaded copy of the cache. Stock CLI run too (`logs/cli_prompt_cache.log`):")
w("```")
w("mlx_lm.cache_prompt --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit --prompt \"$(cat prompts/chapter.txt)\" --prompt-cache-file chapter.safetensors")
w("mlx_lm.generate --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit --prompt-cache-file chapter.safetensors --prompt \"$(cat prompts/question.txt)\" --max-tokens 48 --temp 0")
w("mlx_lm.generate --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit --prompt \"$(cat prompts/chapter.txt prompts/question.txt)\" --max-tokens 48 --temp 0")
w("```")
pc = d["prompt_cache"]
rows = []
for fam, q in (("llama-3.1-8b", "4bit"), ("llama-3.1-8b", "bf16"), ("llama-3.2-1b", "4bit")):
    v = pc[fam][q]
    rows.append([fam, q, f"{v['prefix_tokens']:,} + {v['suffix_tokens']}", f"{v['ttft_no_cache_s']:.3f} {rg(v['ttft_no_cache_range'], 3)}", f"{v['ttft_with_cache_s']:.3f} {rg(v['ttft_with_cache_range'], 3)}", f"{v['ttft_with_cache_incl_disk_load_s']:.3f}", f"**{v['speedup']:.0f}×** ({v['speedup_incl_load']:.0f}× incl. load)", f"{v['cache_file_MB']:.0f}", "yes" if v["same_greedy_output"] else "no (wording differs from token ~16)"])
table(["Model", "Bits", "Prefix + suffix tokens", "TTFT no cache s", "TTFT with cache s", "…incl. loading the cache file s", "Speedup", "Cache file MB", "Same greedy answer"], rows)
cl = d["prompt_cache_cli_8b_4bit"]
w(f"Stock CLI, 8B-4bit (n = 3 after a warm-up): {cl['no_cache']['prompt_tokens']:,} prompt tokens → TTFT {cl['no_cache']['ttft_s_median']:.2f} s without the cache; {cl['with_cache']['prompt_tokens']} prompt tokens → {cl['with_cache']['ttft_s_median']:.3f} s with it ({cl['no_cache']['ttft_s_median']/cl['with_cache']['ttft_s_median']:.0f}×). `cache_prompt` itself prefilled the 6,910 tokens at 869 tok/s.")
w("")
w("**For the course.** Owl's 6.9K-token chapter costs 8 s of prefill every time without a cache and **0.14 s** with one, which is roughly prefix/total as predicted (the remaining time is the 45-token question plus one decode step). Loading the 906 MB cache file added only ~0.02 s here because it was still in the OS file cache; a cold read from SSD would take longer (not measured). Note the bf16 row: cached and uncached greedy answers state the same facts but diverge in wording after ~16 tokens. The prefix was processed in different chunks, so floating-point sums happened in a different order and a near-tie flipped (see 12).")
w("")

# ---------------------------------------------------------------- batching
w("## 8. Batching: aggregate vs per-sequence throughput")
w("")
w("Command: `bench.py batch --model <repo> --sizes 1 2 4 8 16 32 --prompt 256 --gen 128 --trials 3` (plus `--sizes 3 6 10 12 24 48 64` for the 8B). It uses mlx-lm's `BatchGenerator` (the engine behind `batch_generate`, `mlx_lm.benchmark -b` and the server) with `prefill_batch_size = completion_batch_size = B`, B distinct 256-token prompts, EOS off. Aggregate tok/s is measured over the steady-state steps where all B sequences emit a token; per-sequence = aggregate ÷ B = 1 ÷ step time.")
w("")
bt = d["batching"]
rows = []
Bs = sorted(set(bt["llama-3.1-8b"]["bf16"]["by_batch"]) | set(bt["llama-3.1-8b"]["4bit"]["by_batch"]) | set(bt["llama-3.2-1b"]["4bit"]["by_batch"]), key=int)
for B in Bs:
    r = [B]
    for fam, q in (("llama-3.1-8b", "bf16"), ("llama-3.1-8b", "4bit"), ("llama-3.2-1b", "4bit")):
        x = bt[fam][q]["by_batch"].get(B)
        r.append(f"{x['aggregate_tok_s']:.0f} / {x['per_seq_tok_s']:.1f} / {x['step_ms']:.1f}" if x else "–")
    rows.append(r)
table(["Batch B", "8B bf16: aggregate / per-seq tok/s / step ms", "8B 4-bit: same", "1B 4-bit: same"], rows)
cb = d["cli_crosscheck_mlx_lm_benchmark"]
w(f"Independent check with the stock tool (`mlx_lm.benchmark --model <8B-bf16> -p 256 -g 128 -b B -n 3`): generation_tps {cb['Meta-Llama-3.1-8B-Instruct-bf16__p256_g128_b12']['generation_tps']:.0f} (B=12), {cb['Meta-Llama-3.1-8B-Instruct-bf16__p256_g128_b16']['generation_tps']:.0f} (B=16), {cb['Meta-Llama-3.1-8B-Instruct-bf16__p256_g128_b32']['generation_tps']:.0f} (B=32), which confirms the dip at 16. 8B-4bit with `-p 2048`: {cb['Meta-Llama-3.1-8B-Instruct-4bit__p2048_g128_b1']['generation_tps']:.0f} / {cb['Meta-Llama-3.1-8B-Instruct-4bit__p2048_g128_b4']['generation_tps']:.0f} / {cb['Meta-Llama-3.1-8B-Instruct-4bit__p2048_g128_b16']['generation_tps']:.0f} tok/s at B = 1 / 4 / 16 (lower than the 256-token runs because each step also reads 2K tokens of KV per sequence).")
w("")
b8 = bt["llama-3.1-8b"]["bf16"]["by_batch"]
w(f"**For the course.** The free lunch is real but short on a Mac. **8B bf16: going from 1 to 8 users multiplies total throughput by {b8['8']['aggregate_speedup_vs_b1']:.1f}× ({b8['1']['aggregate_tok_s']:.0f} → {b8['8']['aggregate_tok_s']:.0f} tok/s) while each user loses only {100-b8['8']['per_seq_fraction_of_b1']*100:.0f}% ({b8['1']['per_seq_tok_s']:.1f} → {b8['8']['per_seq_tok_s']:.1f} tok/s)**. The same 16 GB of weights are read once per step and shared. The roofline says bf16 decode (intensity ≈ B FLOPs/byte) should stay memory-bound up to the ridge, B ≈ 28. The measured knee comes earlier and the curve is not smooth: the step time is flat at 32–36 ms up to B = 10, rises to 40 ms at B = 12, then jumps to 66–77 ms at B = 16–32 and grows slowly after that (93 ms at B = 64). So aggregate plateaus near 300 tok/s from B = 12 to 24, dips to 243 at B = 16, and rises again to 415 at B = 32 and {b8['64']['aggregate_tok_s']:.0f} at B = 64 (≈{b8['64']['achieved_TFLOPS']:.1f} TFLOPS, {b8['64']['achieved_TFLOPS']/15.2*100:.0f}% of the matmul peak). Our reading, not verified in the MLX source: MLX switches from a mat-vec kernel at small B to tiled matrix-matrix kernels that are inefficient at these small row counts. The knee here is set by the kernels, not by the roofline. **4-bit weights end the free lunch sooner** (B ≈ 2–3: 101 → 190 → 248, then flat at 267–300 from B = 4 to 12; 358 at 16, 514 at 24, 676 at 32, and again non-monotonic: 563 at 48, 717 at 64). Here the step time grows almost linearly from B = 3 to 12 (12 → 40 ms), then stays at ≈45 ms from B = 16 to 32. Our interpretation: with 3.6× fewer bytes per step, the per-row dequantization work of the quantized mat-vec path dominates early, until a tiled quantized-matmul kernel takes over. Even so, at B = 32 4-bit serves 676 tok/s at 21 tok/s per user. The 1B model scales to 3,379 tok/s at B = 32.")
w("")

# ---------------------------------------------------------------- spec
w("## 9. Speculative decoding (draft = Llama-3.2-1B-Instruct-4bit)")
w("")
w("Command: `bench.py spec --model <8B repo> --draft mlx-community/Llama-3.2-1B-Instruct-4bit --k 2 3 4 5 6 --max-tokens 256 --trials 3`; prompts (a) `chat.txt` (open-ended) and (b) `notes_rewrite.txt` (\"Copy out the following study notes exactly …\"). Greedy, EOS on (all runs hit 256 tokens). Baseline is the same model without a draft, same run. Acceptance = accepted draft tokens ÷ drafted tokens, reconstructed exactly from mlx-lm's per-token `from_draft` flag. Stock-CLI equivalent: `mlx_lm.generate --model <8B> --draft-model mlx-community/Llama-3.2-1B-Instruct-4bit --num-draft-tokens k --prompt … --max-tokens 256 --temp 0`.")
w("")
sp = d["speculative"]
rows = []
for t in ("llama-3.1-8b__4bit", "llama-3.1-8b__bf16"):
    for pn in ("open_chat", "copy_notes"):
        x = sp[t]["prompts"][pn]
        rows.append([t.replace("llama-3.1-8b__", "8B "), pn, "baseline", f"{x['baseline_tok_s']:.1f} {rg(x['baseline_range'])}", "1.00×", "", "", "", ""])
        for k, y in x["by_num_draft_tokens"].items():
            rows.append(["", "", f"k = {k}", f"{y['tok_s_median']:.1f} {rg(y['tok_s_range'])}", f"**{y['speedup']:.2f}×**" if int(k) == x["best_k"] else f"{y['speedup']:.2f}×", f"{y['speedup_end_to_end']:.2f}×", f"{y['acceptance_rate']:.2f}", f"{y['tokens_per_target_pass']:.2f}", "yes" if y["output_identical_to_baseline"] else f"no (from token {y['first_divergence_token']})"])
table(["Target", "Prompt", "Draft tokens", "Decode tok/s [range]", "Speedup", "Speedup incl. prefill", "Acceptance α", "Tokens per target pass", "Same text as greedy baseline"], rows)
w("**For the course.**")
w("- **Input-grounded text is easy to guess.** When Owl copies notes, the 1B draft is right every time (α = 1.00) and each 8B pass emits k + 1 tokens: up to **4.8× faster for the bf16 target** (30.7 → 148 tok/s at k = 6) and 1.83× for the 4-bit target. Open chat accepts only 38–72%: best 2.28× (bf16) and 1.42× (4-bit at k = 2).")
w("- **The draft must be cheap relative to the target.** A 1B-4bit step is ~2 ms. That is 6% of an 8B-bf16 step (32 ms) but 20% of an 8B-4bit step (9.8 ms). So the same draft helps the bf16 target far more, and for 4-bit chat a longer draft (k = 6) is slower than no draft (0.92×).")
w("- Leviathan's formula E[tokens per pass] = (1 − α^(k+1)) / (1 − α) roughly matches (chat, k = 2, α = 0.72: formula 2.24, measured 2.42; α is not constant along the text).")
w("- **Not bit-identical in practice.** With greedy sampling, speculative decoding should reproduce the target's own output. For the copy prompt it does. For chat the text diverges at token 221 (4-bit) or 154 (bf16): verifying k + 1 tokens in one batched forward pass changes floating-point rounding, and a near-tie flips (compare 12).")
w("- \"Speedup incl. prefill\" counts the time to prefill both models (the spec path uses prefill_step_size 512), which matters for short answers.")
w("")

# ---------------------------------------------------------------- server
w("## 10. Serving: TTFT and inter-token latency through `mlx_lm.server`")
w("")
w("Commands (`tools/server_runs.sh`): `mlx_lm.server --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit --port 8090 --temp 0 --max-tokens 128`, then `tools/stream_client.py` (stdlib; streams `/v1/chat/completions` with SSE and timestamps every content chunk). Closed loop: C concurrent clients, each sending R requests back to back (C = 1: R = 10; C = 4: R = 5; C = 16: R = 3). Each prompt is a distinct ~64-token \"explain <topic> to a 12-year-old in 400 words\" request; max_tokens 128 (every request hit the length limit). One warm-up pair of requests first. TTFT is measured from the first connection attempt.")
w("")
sv = d["server"]["llama-3.1-8b__4bit"]
rows = []
for c in ("1", "4", "16"):
    v = sv[c]
    rows.append([c, v["n_requests"], f"{v['ttft_ms']['p50']:.0f} / {v['ttft_ms']['p90']:.0f} / {v['ttft_ms']['p99']:.0f}", f"{v['itl_ms']['p50']:.1f} / {v['itl_ms']['p90']:.1f} / {v['itl_ms']['p99']:.1f}", f"{v['per_request_decode_tok_s']['median']:.1f}", f"**{v['aggregate_tok_s']:.0f}**", v["retries_total"]])
table(["Concurrent clients", "Requests", "TTFT p50 / p90 / p99 ms", "ITL p50 / p90 / p99 ms", "Per-request tok/s (median)", "Aggregate tok/s", "Connection retries"], rows)
w("**Does this mlx-lm version batch concurrent requests? Yes.** `mlx_lm/server.py` (0.32.2) routes requests through `BatchGenerator` (continuous batching; `--decode-concurrency 32`, `--prompt-concurrency 8`). A request is only served on its own if it uses a draft model, `--kv-bits`, a per-request `seed`, or images. The numbers show it: 16 clients get 290 tok/s in total vs 90 for one client, while each stream slows from 94 to 21 tok/s (ITL 10.6 → 47 ms). That is below the in-process B = 16 result (358 tok/s aggregate, 22.4 tok/s per sequence) because the server prefills newcomers inside the decode loop and adds HTTP/SSE work. The server also keeps a prompt cache (10 entries by default), so ~36 of each ~64 prompt tokens (the shared chat-template header) were served from cache.")
w("")
w("**For the course.** At one user, TTFT is ~72 ms (a 64-token prompt) and tokens arrive every 10.6 ms like clockwork (p99 11.1 ms). Under load, TTFT grows to p50 0.55 s at 4 clients and 0.84 s at 16 (p99 1.2 s), although the prompt itself needs only ~70 ms of prefill. The rest is **waiting** while the server interleaves the newcomer with everyone else's decode steps (server internals were not instrumented). The tail ITL spikes (p99 29–67 ms, max 0.4 s) are most likely steps in which a newcomer's prompt was processed inside the decode loop, the stall that chunked prefill is designed to bound (notes §5).")
w("")
w("Caveats: (1) a first run lost requests to **connection resets at 15–16 simultaneous connects**. `mlx_lm.server` uses the standard library's `ThreadingHTTPServer` (listen backlog 5) and its Python accept loop competes with the generation thread. The final client staggers starts by 20 ms and retries with backoff: 2 retries in 48 requests at C = 16. The first run's files are kept in `raw/superseded/`. (2) Client and server share the machine; the client's CPU cost is small (stdlib, one thread per client).")
w("")

# ---------------------------------------------------------------- energy
w("## 11. Energy per token (GPU only, no sudo)")
w("")
w("`sudo powermetrics` was not used. `zeus-apple-silicon` 1.1.0 (installed in `tools/.venv`) reads the IOReport \"Energy Model\" counters without root. On this macOS **only the GPU channel updates**: the CPU and DRAM counters stayed frozen, so all figures are **GPU-only, model-based estimates (not a wall meter)**. Command: `bench.py energy --model <repo> --trials 3 --max-tokens 256`. Windows: 256-token batch-1 decode (chat prompt); one 2,048-token prefill; a batch of 16 × 256-token prompts generating 16 × 256 tokens.")
w("")
en = d["energy"]
rows = []
for fam, q in (("llama-3.2-1b", "4bit"), ("llama-3.1-8b", "4bit"), ("llama-3.1-8b", "bf16")):
    v = en[fam][q]
    rows.append([fam, q, f"{v['decode_b1_J_per_token']:.3f} {rg(v['decode_b1_J_per_token_range'], 3)}", f"{v['decode_b1_gpu_W']:.1f}", f"{v['prefill_mJ_per_prompt_token']:.1f}", f"{v['prefill_gpu_W']:.1f}", f"{v['decode_b16_J_per_token']:.3f} ({v['decode_b16_J_per_token_minus_prefill_est']:.3f})", f"{v['decode_b16_gpu_W']:.1f}"])
table(["Model", "Bits", "Decode B=1 J/token", "GPU W", "Prefill mJ / prompt token", "GPU W", "B=16 J / generated token (minus prefill est.)", "GPU W"], rows)
w("GPU idle reads 0.0 W (the GPU power-gates when idle). **For the course.** One 8B-bf16 token costs ≈0.6 J of GPU energy at batch 1, and 4-bit halves it (0.32 J). An *input* token costs 55–68 mJ to prefill (8B), **4.6–11× less than an output token**, the same direction as API pricing (output ≈ 4–6× input). Batching cuts energy per output token: 8B-bf16 0.61 → 0.17 J (prefill share removed). Caveat: decode is memory-bound and **DRAM energy is not counted**, so true per-token energy is higher, most of all for bf16 decode, which draws only 18 W on the GPU, consistent with a GPU that is mostly waiting on memory.")
w("")

# ---------------------------------------------------------------- determinism
w("## 12. Determinism at temperature 0")
w("")
w("Commands: `bench.py determinism --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit --runs 20 --gen 128` (prompt `determinism.txt`) and the same with `--prompt-file chat.txt --gen 256`. Each does 20 batch-1 runs, then puts the same prompt as member 0 of a `BatchGenerator` batch of B = 1, 2, 4, 8, 16 (other members: different short questions). Server: `stream_client.py --same-prompt prompts/determinism.txt`, 20 sequential requests alone, then 20 more while 15 other clients keep the batch full.")
w("")
dt = d["determinism"]
a = dt["in_process__llama-3.1-8b__4bit"]
b = dt["in_process__llama-3.1-8b__4bit__chat256"]
rows = [
    ["Batch 1, 20 runs, 128 tokens (octopus prompt)", f"{a['batch1_distinct_outputs']} distinct output(s) of {a['batch1_runs']}"],
    ["Batch 1, 20 runs, 256 tokens (chat prompt)", f"{b['batch1_distinct_outputs']} distinct output(s) of {b['batch1_runs']}"],
    ["Same prompt inside batch B = 2 / 4 / 8 / 16, 128 tokens", " / ".join("identical" if a["same_prompt_inside_batch"][B]["identical_to_batch1_stream"] else f"differs from token {a['same_prompt_inside_batch'][B]['first_divergence_token']}" for B in ("2", "4", "8", "16"))],
    ["Same prompt inside batch B = 2 / 4 / 8 / 16, 256 tokens", " / ".join("identical" if b["same_prompt_inside_batch"][B]["identical_to_batch1_stream"] else f"differs from token {b['same_prompt_inside_batch'][B]['first_divergence_token']}" for B in ("2", "4", "8", "16"))],
    ["Server, 20 sequential requests alone", f"{dt['server_det_solo']['distinct_outputs']} distinct of {dt['server_det_solo']['requests']}"],
    ["Server, 20 requests under 15-client load", f"{dt['server_det_underload']['distinct_outputs']} distinct of {dt['server_det_underload']['requests']}; {dt['server_underload_matching_solo_majority']}/20 identical to the solo answer"],
    ["Speculative decoding vs plain greedy (9)", "chat: differs from token 221 (4-bit) / 154 (bf16); copy prompt: identical"],
    ["Prompt cache vs no cache, 8B bf16 (7)", "differs from token ~16 (same facts)"],
]
table(["Test", "Result"], rows)
x4 = b["same_prompt_inside_batch"]["4"]
w(f"**For the course.** At batch 1 the Mac is perfectly repeatable (20/20 identical twice). Batching *can* change a greedy answer, but rarely and late: the 128-token answer was identical at every batch size and through the busy server, while the 256-token answer diverged at token 221 for B = 4 and 8 and at token 132 for B = 16. Batch-1 continued \"{x4['ref_text_after'].strip()}\"; inside a batch of 4 it became \"{x4['batch_text_after'].strip()}\". The cause is the same each time: a different kernel or chunking sums the same numbers in a different order, and a near-tie between two tokens flips. Once one token differs, the rest of the text follows a different path.")
w("")

# ---------------------------------------------------------------- crosscheck
w("## Cross-check: stock `mlx_lm.generate` vs the harness")
w("")
w("`tools/cli_crosscheck.sh`: `mlx_lm.generate --model <repo> --prompt \"$(cat prompts/chat.txt)\" --max-tokens 256 --temp 0`, 1 warm-up + 3 runs, reading \"Generation: N tokens, X tokens-per-sec\" (EOS on; every run produced 256 tokens).")
w("")
cg = d["cli_crosscheck_mlx_lm_generate"]
rows = []
for (fam, q), repo in (((("llama-3.2-1b", "4bit")), "Llama-3.2-1B-Instruct-4bit"), (("llama-3.2-1b", "8bit"), "Llama-3.2-1B-Instruct-8bit"), (("llama-3.2-1b", "bf16"), "Llama-3.2-1B-Instruct-bf16"),
                       (("llama-3.1-8b", "4bit"), "Meta-Llama-3.1-8B-Instruct-4bit"), (("llama-3.1-8b", "8bit"), "Meta-Llama-3.1-8B-Instruct-8bit"), (("llama-3.1-8b", "bf16"), "Meta-Llama-3.1-8B-Instruct-bf16")):
    c = cg[repo]["tok_s_median"]
    h = d["decode"][fam][q]["tok_s_median"]
    rows.append([repo, f"{c:.2f}", f"{h:.2f}", f"{(c/h-1)*100:+.1f}%"])
table(["Model", "mlx_lm.generate tok/s (median of 3)", "Harness tok/s (median of 5)", "Difference"], rows)

# ---------------------------------------------------------------- caveats
w("## Caveats, failures and deviations")
w("")
for s in d["skipped_or_changed"]:
    w(f"- {s}")
w("- Thermal: Mac Studio fans; no throttling was visible (decode ranges stay within ±1% across trials and the roofs re-run matched the first run).")
w("- Prefill for 8B used the default 2,048-token chunk; the LM head is evaluated only for the last position, as mlx-lm does.")
w("- All throughput numbers are for mlx-lm 0.32.2 / MLX 0.32.4 kernels; other engines (llama.cpp, a future MLX) will differ, especially for batching, quantized KV and long-context attention.")
w("- \"MFU\" figures use a self-measured matmul peak, because Apple publishes no GPU FLOPS.")
w("")
w("## Disk")
w("")
dk = d["disk"]
w(f"Downloaded to `~/.cache/huggingface/hub`: " + ", ".join(f"{k} {v:.2f} GB" for k, v in dk["models_GB"].items()) + f". **Total {dk['models_total_GB']:.2f} GB** (the HF cache went from {dk['hf_cache_before_GB']:.2f} to {dk['hf_cache_after_GB']:.2f} GB). The 906 MB prompt-cache files were deleted after use.")
w("")
open(f"{R}/measurements.md", "w").write("\n".join(L))
print("wrote", f"{R}/measurements.md", len(L), "lines")
