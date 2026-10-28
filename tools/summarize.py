"""Build research/measurements.json from research/measurements/raw/*.json.

All derived numbers (ceilings, MBU, slopes, ratios) are computed here so the
markdown and the episode scenes share one source of truth.
Usage: python -I summarize.py   (prints a compact digest)
"""
import glob
import json
import os
import re
import statistics

R = "/Users/vpromisex/videos/inference-series/research"
RAW = os.path.join(R, "measurements", "raw")
BW = 546e9  # Apple spec, support.apple.com/en-us/122211


def J(name):
    p = os.path.join(RAW, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def med(s):
    return s["median"] if isinstance(s, dict) else s


def rng(s):
    return [s["min"], s["max"]]


MODELS = {
    ("llama-3.2-1b", "4bit"): "mlx-community/Llama-3.2-1B-Instruct-4bit",
    ("llama-3.2-1b", "8bit"): "mlx-community/Llama-3.2-1B-Instruct-8bit",
    ("llama-3.2-1b", "bf16"): "mlx-community/Llama-3.2-1B-Instruct-bf16",
    ("llama-3.1-8b", "4bit"): "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    ("llama-3.1-8b", "8bit"): "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit",
    ("llama-3.1-8b", "bf16"): "mlx-community/Meta-Llama-3.1-8B-Instruct-bf16",
    ("qwen3-1.7b", "4bit"): "mlx-community/Qwen3-1.7B-4bit",
}
mb = {m["repo"]: m for m in J("model_bytes")}
pc = {p["repo"]: p for p in J("param_counts")}
PARAMS = {"llama-3.2-1b": pc["mlx-community/Llama-3.2-1B-Instruct-bf16"], "llama-3.1-8b": pc["mlx-community/Meta-Llama-3.1-8B-Instruct-bf16"]}

out = {"schema_version": 1}
out["machine"] = {
    "name": "Mac Studio (2025), Mac16,9",
    "chip": "Apple M4 Max",
    "cpu_cores": 16, "cpu_performance_cores": 12, "cpu_efficiency_cores": 4,
    "gpu_cores": 40,
    "unified_memory_GB": 64,
    "memory_bandwidth_spec_GBps": 546,
    "bandwidth_source": "support.apple.com/en-us/122211: 'M4 Max with 16-core CPU, 40-core GPU, and 16-core Neural Engine (546GB/s memory bandwidth)'",
    "macos": "27.0.1 (26A434)",
    "power": "AC, Low Power Mode off",
    "mlx_max_recommended_working_set_GB": 55.66,
}
out["software"] = {"python": "3.12", "mlx": "0.32.4", "mlx_metal": "0.32.4", "mlx_lm": "0.32.2", "transformers": "5.20.2", "zeus_apple_silicon": "1.1.0", "uv": "0.12.11"}
out["dates"] = {"measured": "2026-10-24"}
out["conventions"] = {"GB": 1e9, "GiB": 2**30, "tok_s": "tokens per second", "ranges": "[min, max] over n timed trials after 1 warm-up"}

# ------------------------------------------------------------------ models
models = {}
for (fam, q), repo in MODELS.items():
    m = mb[repo]
    models.setdefault(fam, {})[q] = {
        "repo": repo,
        "weights_file_bytes": m["file_bytes"],
        "weights_gb": round(m["file_bytes"] / 1e9, 3),
        "decode_bytes_per_token": m["decode_bytes_per_token"],
        "embed_bytes": m["embed_bytes"],
        "tied_embeddings": m["tied_embeddings"],
        "quantization": m["quantization"],
        "layers": m["layers"], "kv_heads": m["kv_heads"], "head_dim": m["head_dim"],
        "kv_bytes_per_token_bf16": m["kv_bytes_per_token_bf16"],
        "kv_formula": f"2 x {m['layers']} x {m['kv_heads']} x {m['head_dim']} x 2 B",
    }
for fam, p in PARAMS.items():
    for q in models[fam]:
        models[fam][q]["params_total"] = p["params_total"]
        models[fam][q]["params_matmul_excl_embed_head"] = p["matmul_params_excl_embed_and_head"]
out["models"] = models

# ------------------------------------------------------------------ roofs
r = J("roofs")
roofs = {
    "read_GBps": {"median": med(r["read_sum_4GB_GBps"]), "range": rng(r["read_sum_4GB_GBps"]), "n": r["read_sum_4GB_GBps"]["n"], "how": "mx.sum over a 4 GB fp32 array"},
    "read_1GB_GBps": med(r["read_sum_1GB_GBps"]),
    "copy_GBps_read_plus_write": med(r["copy_4GB_GBps_read_plus_write"]),
    "matvec_bf16_GBps": {"median": med(r["matvec_bf16_GBps"]), "range": rng(r["matvec_bf16_GBps"]), "how": "28672x16384 bf16 W @ x"},
    "matvec_q4_GBps": {"median": med(r["matvec_q4_GBps"]), "range": rng(r["matvec_q4_GBps"]), "how": "same W, 4-bit group-64 quantized_matmul"},
    "matmul_bf16_TFLOPS": {"median": med(r["matmul_bf16_8192_TFLOPS"]), "range": rng(r["matmul_bf16_8192_TFLOPS"]), "how": "8192^3 bf16 matmul"},
    "matmul_fp16_TFLOPS": med(r["matmul_fp16_8192_TFLOPS"]),
    "matmul_fp32_TFLOPS": med(r["matmul_fp32_8192_TFLOPS"]),
}
roofs["read_fraction_of_spec"] = roofs["read_GBps"]["median"] * 1e9 / BW
roofs["ridge_flops_per_byte_spec_bw"] = roofs["matmul_bf16_TFLOPS"]["median"] * 1e12 / BW
roofs["ridge_flops_per_byte_measured_bw"] = roofs["matmul_bf16_TFLOPS"]["median"] / roofs["read_GBps"]["median"] * 1e3
r2 = J("roofs_rerun")
if r2:
    roofs["rerun_end_of_session"] = {k: med(r2[k]) for k in ("read_sum_4GB_GBps", "matvec_bf16_GBps", "matmul_bf16_8192_TFLOPS")}
out["roofs"] = roofs
READ = roofs["read_GBps"]["median"] * 1e9
PEAK = roofs["matmul_bf16_TFLOPS"]["median"] * 1e12

# ------------------------------------------------------------------ 1 decode
decode = {}
for (fam, q), repo in MODELS.items():
    d = J(f"decode__{fam}__{q}")
    if not d:
        continue
    m = models[fam][q]
    P, N = d["prompt_tokens"], d["max_tokens"]
    kv_read = (P + N / 2) * m["kv_bytes_per_token_bf16"]
    bps = m["decode_bytes_per_token"] + kv_read
    t = med(d["decode_tok_s"])
    decode.setdefault(fam, {})[q] = {
        "tok_s_median": t, "tok_s_range": rng(d["decode_tok_s"]), "n": d["decode_tok_s"]["n"],
        "prompt_tokens": P, "gen_tokens": N,
        "weights_gb": m["weights_gb"],
        "ceiling_tok_s": BW / m["weights_file_bytes"],
        "mbu_vs_file_bytes": t * m["weights_file_bytes"] / BW,
        "bytes_read_per_step": bps,
        "ceiling_tok_s_bytes_read": BW / bps,
        "mbu": t * bps / BW,
        "achieved_GBps": t * bps / 1e9,
        "fraction_of_measured_read_bw": t * bps / READ,
        "ms_per_token": 1e3 / t,
        "ttft_s_median": med(d["ttft_s"]),
        "peak_memory_GB": med(d["peak_memory_bytes"]) / 1e9,
        "greedy_identical_across_trials": d["identical_outputs"],
    }
for fam in decode:
    if "bf16" in decode[fam]:
        for q in decode[fam]:
            decode[fam][q]["speedup_vs_bf16"] = decode[fam][q]["tok_s_median"] / decode[fam]["bf16"]["tok_s_median"]
            decode[fam][q]["bytes_ratio_vs_bf16"] = models[fam]["bf16"]["decode_bytes_per_token"] / models[fam][q]["decode_bytes_per_token"]
out["decode"] = decode

# ------------------------------------------------------------------ 2 prefill
prefill = {}
for (fam, q), repo in MODELS.items():
    d = J(f"prefill__{fam}__{q}")
    if not d:
        continue
    m = models[fam][q]
    pm = m.get("params_matmul_excl_embed_head")
    hidden = mb[repo]["hidden_size"]
    rows = {}
    for T, v in d["results"].items():
        T_ = int(T)
        tps = med(v["prefill_tok_s"])
        flops_tok = 2 * pm + 2 * m["layers"] * hidden * T_  # matmuls + causal attention (avg ctx T/2)
        rows[T] = {
            "tok_s_median": tps, "tok_s_range": rng(v["prefill_tok_s"]), "n": v["prefill_tok_s"]["n"],
            "ttft_s_median": med(v["ttft_s"]), "ttft_s_range": rng(v["ttft_s"]),
            "achieved_TFLOPS": tps * flops_tok / 1e12,
            "fraction_of_measured_matmul_peak": tps * flops_tok / PEAK,
            "peak_memory_GB": med(v["peak_memory_bytes"]) / 1e9,
        }
    best = max(rows.values(), key=lambda x: x["tok_s_median"])
    dec = decode.get(fam, {}).get(q)
    prefill.setdefault(fam, {})[q] = {
        "by_prompt_tokens": rows,
        "best_tok_s": best["tok_s_median"],
        "prefill_step_size": 2048,
        "prefill_over_decode_ratio_at_2048": rows["2048"]["tok_s_median"] / dec["tok_s_median"] if dec else None,
    }
out["prefill"] = prefill

# ------------------------------------------------------------------ 3 context
context = {}
for (fam, q), repo in MODELS.items():
    d = J(f"context__{fam}__{q}")
    if not d:
        continue
    m = models[fam][q]
    G = d["gen_tokens"]
    rows = {}
    base_T = min(int(t) for t in d["results"])
    bytes_at = lambda T: m["decode_bytes_per_token"] + (T + G / 2) * m["kv_bytes_per_token_bf16"]
    t0 = med(d["results"][str(base_T)]["decode_tok_s"])
    for T, v in d["results"].items():
        T_ = int(T)
        t = med(v["decode_tok_s"])
        rows[T] = {
            "tok_s_median": t, "tok_s_range": rng(v["decode_tok_s"]), "n": v["decode_tok_s"]["n"],
            "kv_GiB": T_ * m["kv_bytes_per_token_bf16"] / 2**30,
            "kv_over_weights": T_ * m["kv_bytes_per_token_bf16"] / m["decode_bytes_per_token"],
            "bytes_per_step_GB": bytes_at(T_) / 1e9,
            "predicted_tok_s_bytes_model": t0 * bytes_at(base_T) / bytes_at(T_),
            "achieved_GBps": t * bytes_at(T_) / 1e9,
            "slowdown_vs_short": t / t0,
        }
    context.setdefault(fam, {})[q] = {"gen_tokens": G, "by_context_tokens": rows}
out["context"] = context

# ------------------------------------------------------------------ 4/5 memory + kv quant
memory = {}
for f in sorted(glob.glob(os.path.join(RAW, "memory__*.json"))):
    name = os.path.basename(f)[8:-5]
    d = json.load(open(f))
    fam, q, kv = name.split("__")[:3]
    extra = name.split("__")[3:] if len(name.split("__")) > 3 else []
    m = models[fam][q]
    s = d["slope_bytes_per_token"]
    bits = d["kv_bits"] or 16
    expected = m["kv_bytes_per_token_bf16"] * (1.0 if bits == 16 else (bits / 8 + 4 / 64) / 2)
    key = kv + ("__" + "__".join(extra) if extra else "")
    memory.setdefault(fam, {}).setdefault(q, {})[key] = {
        "kv_bits": bits,
        "prefill_step_size": d.get("prefill_step_size", 2048),
        "slope_active_bytes_per_token": s["active_memory_bytes"],
        "slope_kv_cache_bytes_per_token": s["kv_cache_bytes"],
        "slope_peak_bytes_per_token": s["peak_memory_bytes"],
        "formula_bytes_per_token": expected,
        "formula": "2*L*n_kv*d_h*2 B" + ("" if bits == 16 else f" x (bits/8 + 4/64)/2 [group-64 bf16 scale+bias] = x{(bits/8 + 4/64)/2:.5f}"),
        "slope_over_formula": s["active_memory_bytes"] / expected,
        "weights_active_GB": d["weights_active_bytes"] / 1e9,
        "by_context_tokens": {T: {"peak_GB": v["peak_memory_bytes"] / 1e9, "active_GB": v["active_memory_bytes"] / 1e9, "kv_cache_MiB": v["kv_cache_bytes"] / 2**20, "cache_type": v["cache_type"], "decode_tok_s": v["decode_tok_s"]} for T, v in d["results"].items()},
    }
out["memory"] = memory

# ------------------------------------------------------------------ 6 rotating
rot = {}
for f in sorted(glob.glob(os.path.join(RAW, "rotating__*.json"))):
    d = json.load(open(f))
    fam, q = os.path.basename(f)[10:-5].split("__")
    res = d["results"]
    k = f"rotating_{d['max_kv_size']}"
    pts = lambda lab: {str(p["tokens_generated"]): {"kv_MiB": p["kv_cache_bytes"] / 2**20, "peak_GB": p["peak_memory_bytes"] / 1e9, "window_tok_s": p["window_tok_s"]} for p in res[lab]["points"]}
    rot.setdefault(fam, {})[q] = {
        "max_kv_size": d["max_kv_size"], "gen_tokens": d["gen_tokens"],
        "full_cache": {"cache_type": res["full"]["cache_type"], "points": pts("full")},
        "rotating": {"cache_type": res[k]["cache_type"], "built_explicitly": res[k].get("built_explicitly"), "points": pts(k)},
        "forgetting_demo": res.get("forgetting_demo"),
    }
out["rotating_cache"] = rot
out["rotating_cache_note"] = "mlx-lm 0.32.2: llama.Model defines make_cache(), so make_prompt_cache(model, max_kv_size) and the --max-kv-size CLI flag are silently ignored for Llama models (plain KVCache). Qwen3 has no override, so the flag works there. For Llama the harness builds mlx-lm's RotatingKVCache(max_size, keep=4) explicitly."

# ------------------------------------------------------------------ kv quality
kq = J("kvquality__llama-3.1-8b__4bit")
if kq:
    out["kv_quant_quality"] = {"model": kq["model"], "prompt_tokens": kq["results"]["prompt_tokens"], "by_kv_bits": {b: {k: v for k, v in kq["results"][b].items()} for b in ("16", "8", "4")}}

# ------------------------------------------------------------------ 7 prompt cache
pcache = {}
for f in sorted(glob.glob(os.path.join(RAW, "promptcache__*.json"))):
    d = json.load(open(f))
    fam, q = os.path.basename(f)[13:-5].split("__")
    pcache.setdefault(fam, {})[q] = {
        "prefix_tokens": d["prefix_tokens"], "suffix_tokens": d["suffix_tokens"],
        "ttft_no_cache_s": med(d["ttft_no_cache_s"]), "ttft_no_cache_range": rng(d["ttft_no_cache_s"]),
        "ttft_with_cache_s": med(d["ttft_with_cache_s"]), "ttft_with_cache_range": rng(d["ttft_with_cache_s"]),
        "ttft_with_cache_incl_disk_load_s": med(d["ttft_with_cache_incl_disk_load_s"]),
        "n": d["ttft_no_cache_s"]["n"],
        "speedup": med(d["ttft_no_cache_s"]) / med(d["ttft_with_cache_s"]),
        "speedup_incl_load": med(d["ttft_no_cache_s"]) / med(d["ttft_with_cache_incl_disk_load_s"]),
        "cache_file_MB": d["cache_file_bytes"] / 1e6,
        "same_greedy_output": d["same_output_tokens"],
        "answer_no_cache": d["answer_no_cache"], "answer_with_cache": d["answer_with_cache"],
    }
out["prompt_cache"] = pcache

# ------------------------------------------------------------------ 8 batching
batch = {}
for fam, q in (("llama-3.1-8b", "4bit"), ("llama-3.1-8b", "bf16"), ("llama-3.2-1b", "4bit")):
    rows = {}
    for suffix in ("", "__ext"):
        d = J(f"batch__{fam}__{q}{suffix}")
        if d:
            for B, v in d["results"].items():
                rows[B] = {"aggregate_tok_s": med(v["aggregate_tok_s"]), "aggregate_range": rng(v["aggregate_tok_s"]), "n": v["aggregate_tok_s"]["n"],
                           "per_seq_tok_s": med(v["per_seq_tok_s"]), "step_ms": med(v["step_ms"]), "peak_memory_GB": med(v["peak_memory_bytes"]) / 1e9}
    rows = dict(sorted(rows.items(), key=lambda kv: int(kv[0])))
    b1 = rows["1"]["aggregate_tok_s"]
    for B, v in rows.items():
        v["aggregate_speedup_vs_b1"] = v["aggregate_tok_s"] / b1
        v["per_seq_fraction_of_b1"] = v["per_seq_tok_s"] / b1
        v["achieved_TFLOPS"] = v["aggregate_tok_s"] * 2 * models[fam][q]["params_matmul_excl_embed_head"] / 1e12
    batch.setdefault(fam, {})[q] = {"prompt_tokens": 256, "gen_tokens": 128, "by_batch": rows}
out["batching"] = batch

# ------------------------------------------------------------------ 9 speculative
spec = {}
for f in sorted(glob.glob(os.path.join(RAW, "spec__*.json"))):
    d = json.load(open(f))
    tag = os.path.basename(f)[6:-5]
    fam, q, dr = tag.split("__")
    pr = {}
    for pname, row in d["results"].items():
        key = {"chat.txt": "open_chat", "notes_rewrite.txt": "copy_notes"}[pname]
        ks = {}
        for k, v in row.items():
            if k.startswith("k"):
                ks[k[1:]] = {"tok_s_median": med(v["decode_tok_s"]), "tok_s_range": rng(v["decode_tok_s"]), "n": v["decode_tok_s"]["n"],
                             "speedup": v["speedup_decode"], "speedup_end_to_end": v["speedup_end_to_end"],
                             "acceptance_rate": v["acceptance_rate"], "tokens_per_target_pass": v["tokens_per_target_pass"],
                             "gen_tokens": v["gen_tokens"], "output_identical_to_baseline": v["output_identical_to_baseline"],
                             "first_divergence_token": v["first_divergence"]}
        best = max(ks, key=lambda k: ks[k]["speedup"])
        pr[key] = {"prompt_tokens": row["prompt_tokens"], "baseline_tok_s": med(row["baseline"]["decode_tok_s"]), "baseline_range": rng(row["baseline"]["decode_tok_s"]),
                   "baseline_gen_tokens": row["baseline"]["gen_tokens"], "by_num_draft_tokens": ks, "best_k": int(best), "best_speedup": ks[best]["speedup"]}
    spec[f"{fam}__{q}"] = {"target": d["model"], "draft": d["draft"], "max_tokens": d["max_tokens"], "prompts": pr}
out["speculative"] = spec

# ------------------------------------------------------------------ 10 server
server = {}
for f in sorted(glob.glob(os.path.join(RAW, "server__*__c*.json"))):
    d = json.load(open(f))
    fam, q, c = os.path.basename(f)[8:-5].split("__")
    server.setdefault(f"{fam}__{q}", {})[c[1:]] = {
        "clients": d["clients"], "n_requests": d["n_requests"], "max_tokens": d["max_tokens"],
        "ttft_ms": d["ttft_ms"], "itl_ms": d["itl_ms"], "aggregate_tok_s": d["aggregate_completion_tok_s"],
        "per_request_decode_tok_s": d["per_request_decode_tok_s"], "finish_reasons": d["finish_reasons"],
        "wall_s": d["wall_s"], "retries_total": d.get("retries_total"), "requests_with_retries": d.get("requests_with_retries"),
        "median_prompt_tokens": statistics.median([p for p in d["prompt_tokens"] if p]),
        "cached_prompt_tokens_median": statistics.median([c or 0 for c in d["cached_prompt_tokens"]]),
    }
out["server"] = server
out["server_batches_concurrent_requests"] = {
    "answer": True,
    "evidence": "mlx_lm/server.py 0.32.2 serves batchable requests through BatchGenerator (continuous batching; --decode-concurrency 32, --prompt-concurrency 8). Requests are batchable unless a draft model, --kv-bits, a per-request seed, or images are used. Measured aggregate tok/s rises with concurrency (see server).",
}

# ------------------------------------------------------------------ 11 energy
energy = {}
for f in sorted(glob.glob(os.path.join(RAW, "energy__*.json"))):
    d = json.load(open(f))
    fam, q = os.path.basename(f)[8:-5].split("__")
    energy.setdefault(fam, {})[q] = {
        "gpu_idle_W": d["gpu_idle_W"],
        "decode_b1_J_per_token": med(d["decode_b1"]["J_per_token"]), "decode_b1_J_per_token_range": rng(d["decode_b1"]["J_per_token"]),
        "decode_b1_gpu_W": med(d["decode_b1"]["gpu_W"]),
        "prefill_mJ_per_prompt_token": med(d["prefill_2048"]["J_per_prompt_token"]) * 1e3,
        "prefill_gpu_W": med(d["prefill_2048"]["gpu_W"]),
        "decode_b16_J_per_token": med(d["decode_b16"]["J_per_generated_token"]),
        "decode_b16_gpu_W": med(d["decode_b16"]["gpu_W"]),
        "decode_b16_J_per_token_minus_prefill_est": med(d["decode_b16"]["J_per_generated_token"]) - med(d["prefill_2048"]["J_per_prompt_token"]),
        "n": d["decode_b1"]["J_per_token"]["n"],
    }
out["energy"] = energy
out["energy_note"] = "GPU-only energy from IOReport 'Energy Model' counters via zeus-apple-silicon 1.1.0 (no sudo). Model-based estimates, not a wall-power meter. CPU and DRAM channels did not update on macOS 27.0.1, so they are excluded."

# ------------------------------------------------------------------ 12 determinism
det = {}
for f in sorted(glob.glob(os.path.join(RAW, "determinism__*.json"))):
    d = json.load(open(f))
    lab = os.path.basename(f)[13:-5]
    det[f"in_process__{lab}"] = {"model": d["model"], "prompt_file": d.get("prompt_file", "determinism.txt"), "gen_tokens": d["gen_tokens"],
                                 "batch1_runs": d["batch1"]["runs"], "batch1_distinct_outputs": d["batch1"]["distinct_outputs"],
                                 "same_prompt_inside_batch": {B: {k: v for k, v in x.items() if k != "hash"} for B, x in d["in_batch"].items()}}
for lab in ("det_solo", "det_underload"):
    s = J(f"server__llama-3.1-8b__4bit__{lab}")
    if s:
        hs = s["text_hashes"]
        det[f"server_{lab}"] = {"requests": len(hs), "distinct_outputs": len(set(hs)), "hash_counts": {h: hs.count(h) for h in set(hs)},
                                 "cached_prompt_tokens": s["cached_prompt_tokens"]}
if "server_det_solo" in det and "server_det_underload" in det:
    a = J("server__llama-3.1-8b__4bit__det_solo")["text_hashes"]
    b = J("server__llama-3.1-8b__4bit__det_underload")["text_hashes"]
    det["server_underload_matching_solo_majority"] = sum(1 for h in b if h == max(set(a), key=a.count))
out["determinism"] = det

# ------------------------------------------------------------------ CLI cross-check
cli = {}
p = os.path.join(R, "measurements", "logs", "cli_crosscheck.log")
if os.path.exists(p):
    cur = None
    for line in open(p):
        mm = re.match(r"## run (\d): mlx_lm.generate --model mlx-community/(\S+)", line)
        if mm:
            cur = (int(mm.group(1)), mm.group(2))
            continue
        g = re.match(r"Generation: (\d+) tokens, ([\d.]+) tokens-per-sec", line)
        if g and cur and cur[0] > 0:
            cli.setdefault(cur[1], []).append({"tokens": int(g.group(1)), "tok_s": float(g.group(2))})
    out["cli_crosscheck_mlx_lm_generate"] = {k: {"tok_s_median": statistics.median(x["tok_s"] for x in v), "runs": v} for k, v in cli.items()}
    bench = {}
    cur = None
    extra = os.path.join(R, "measurements", "logs", "cli_crosscheck_batch_bf16.log")
    lines = list(open(p)) + (list(open(extra)) if os.path.exists(extra) else [])
    for line in lines:
        mm = re.match(r"## mlx_lm.benchmark --model mlx-community/(\S+) -p (\d+) -g (\d+) -b (\d+)", line)
        if mm:
            cur = f"{mm.group(1)}__p{mm.group(2)}_g{mm.group(3)}_b{mm.group(4)}"
            continue
        a_ = re.match(r"Averages: prompt_tps=([\d.]+), generation_tps=([\d.]+), peak_memory=([\d.]+)", line)
        if a_ and cur:
            bench[cur] = {"prompt_tps": float(a_.group(1)), "generation_tps": float(a_.group(2)), "peak_memory_GB": float(a_.group(3))}
    out["cli_crosscheck_mlx_lm_benchmark"] = bench

p = os.path.join(R, "measurements", "logs", "cli_prompt_cache.log")
if os.path.exists(p):
    rows = {"with_cache": [], "no_cache": []}
    cur = None
    for line in open(p):
        mm = re.match(r"## run (\d): mlx_lm.generate .*?(--prompt-cache-file)?", line)
        if line.startswith("## run"):
            cur = (int(line.split()[2][0]), "with_cache" if "--prompt-cache-file" in line else "no_cache")
            continue
        g = re.match(r"Prompt: (\d+) tokens, ([\d.]+) tokens-per-sec", line)
        if g and cur and cur[0] > 0:
            rows[cur[1]].append({"prompt_tokens": int(g.group(1)), "ttft_s": int(g.group(1)) / float(g.group(2))})
    out["prompt_cache_cli_8b_4bit"] = {k: {"prompt_tokens": v[0]["prompt_tokens"], "ttft_s_median": statistics.median(x["ttft_s"] for x in v), "n": len(v)} for k, v in rows.items() if v}
    out["prompt_cache_cli_8b_4bit"]["note"] = "TTFT = prompt tokens / mlx_lm.generate's prompt tokens-per-sec; excludes loading the cache file (done before timing)."

# ------------------------------------------------------------------ disk + headline
out["disk"] = {"models_GB": {"Llama-3.2-1B-Instruct-4bit": 0.71, "Llama-3.2-1B-Instruct-8bit": 1.32, "Llama-3.2-1B-Instruct-bf16": 2.48, "Qwen3-1.7B-4bit": 0.98,
                             "Meta-Llama-3.1-8B-Instruct-4bit": 4.53, "Meta-Llama-3.1-8B-Instruct-8bit": 8.54, "Meta-Llama-3.1-8B-Instruct-bf16": 16.08},
               "models_total_GB": 34.65, "hf_cache_before_GB": 19.05, "hf_cache_after_GB": 53.68, "location": "~/.cache/huggingface/hub"}
d8 = decode["llama-3.1-8b"]
bt = batch["llama-3.1-8b"]["bf16"]["by_batch"]
out["headline"] = {
    "decode_8b_bf16_tok_s": d8["bf16"]["tok_s_median"],
    "decode_8b_bf16_ceiling_tok_s": d8["bf16"]["ceiling_tok_s"],
    "decode_8b_bf16_fraction_of_ceiling": d8["bf16"]["mbu_vs_file_bytes"],
    "decode_8b_bf16_mbu_bytes_read": d8["bf16"]["mbu"],
    "decode_8b_bf16_achieved_GBps": d8["bf16"]["achieved_GBps"],
    "decode_8b_4bit_tok_s": d8["4bit"]["tok_s_median"],
    "decode_8b_8bit_tok_s": d8["8bit"]["tok_s_median"],
    "decode_8b_4bit_speedup_vs_bf16": d8["4bit"]["speedup_vs_bf16"],
    "measured_read_bandwidth_GBps": roofs["read_GBps"]["median"],
    "measured_matmul_bf16_TFLOPS": roofs["matmul_bf16_TFLOPS"]["median"],
    "ridge_flops_per_byte": roofs["ridge_flops_per_byte_spec_bw"],
    "prefill_8b_bf16_tok_s_at_2048": prefill["llama-3.1-8b"]["bf16"]["by_prompt_tokens"]["2048"]["tok_s_median"],
    "prefill_over_decode_8b_bf16": prefill["llama-3.1-8b"]["bf16"]["prefill_over_decode_ratio_at_2048"],
    "prefill_over_decode_8b_4bit": prefill["llama-3.1-8b"]["4bit"]["prefill_over_decode_ratio_at_2048"],
    "kv_slope_8b_bytes_per_token": memory["llama-3.1-8b"]["4bit"]["kv16"]["slope_active_bytes_per_token"],
    "kv_formula_8b_bytes_per_token": 131072,
    "kv_slope_1b_bytes_per_token": memory["llama-3.2-1b"]["4bit"]["kv16"]["slope_active_bytes_per_token"],
    "decode_8b_4bit_tok_s_at_32k_context": context["llama-3.1-8b"]["4bit"]["by_context_tokens"]["32768"]["tok_s_median"],
    "decode_8b_bf16_tok_s_at_32k_context": context["llama-3.1-8b"]["bf16"]["by_context_tokens"]["32768"]["tok_s_median"],
    "batch_8b_bf16_aggregate_b8_tok_s": bt["8"]["aggregate_tok_s"],
    "batch_8b_bf16_per_seq_b8_tok_s": bt["8"]["per_seq_tok_s"],
    "batch_8b_bf16_aggregate_b64_tok_s": bt["64"]["aggregate_tok_s"],
    "prompt_cache_8b_4bit_ttft_no_cache_s": pcache["llama-3.1-8b"]["4bit"]["ttft_no_cache_s"],
    "prompt_cache_8b_4bit_ttft_with_cache_s": pcache["llama-3.1-8b"]["4bit"]["ttft_with_cache_s"],
    "spec_8b_4bit_best_speedup_chat": spec["llama-3.1-8b__4bit"]["prompts"]["open_chat"]["best_speedup"],
    "spec_8b_4bit_best_speedup_copy": spec["llama-3.1-8b__4bit"]["prompts"]["copy_notes"]["best_speedup"],
    "spec_8b_bf16_best_speedup_chat": spec["llama-3.1-8b__bf16"]["prompts"]["open_chat"]["best_speedup"],
    "spec_8b_bf16_best_speedup_copy": spec["llama-3.1-8b__bf16"]["prompts"]["copy_notes"]["best_speedup"],
    "server_8b_4bit_ttft_p50_ms": {c: v["ttft_ms"]["p50"] for c, v in server["llama-3.1-8b__4bit"].items()},
    "server_8b_4bit_itl_p50_ms": {c: v["itl_ms"]["p50"] for c, v in server["llama-3.1-8b__4bit"].items()},
    "server_8b_4bit_aggregate_tok_s": {c: v["aggregate_tok_s"] for c, v in server["llama-3.1-8b__4bit"].items()},
}
out["skipped_or_changed"] = [
    "11 energy: powermetrics needs sudo (not used). Used zeus-apple-silicon (IOReport, no sudo): GPU channel only; CPU and DRAM counters did not update on macOS 27.0.1.",
    "6 rotating cache: --max-kv-size is silently ignored for Llama models in mlx-lm 0.32.2 (llama.Model.make_cache override); RotatingKVCache built explicitly for Llama; the flag works for Qwen3.",
    "5 KV quantization: --kv-bits only applies once the cache reaches --quantized-kv-start (default 5000 tokens); measured with quantized_kv_start=0.",
    "9 speculative: acceptance from mlx-lm's per-token from_draft flag; first pass had an off-by-one-pass counting bug (superseded, see raw/superseded); rerun with exact pass reconstruction.",
    "10 server: first run lost requests to connection resets at 15-16 concurrent connects (http.server listen backlog 5); rerun with a retrying, 20 ms-staggered client (superseded files kept).",
]

json.dump(out, open(os.path.join(R, "measurements.json"), "w"), indent=1)

# ------------------------------------------------------------------ digest
print("decode:")
for fam, qs in decode.items():
    for q, v in qs.items():
        print(f"  {fam} {q}: {v['tok_s_median']:.2f} [{v['tok_s_range'][0]:.2f},{v['tok_s_range'][1]:.2f}] ceil(file) {v['ceiling_tok_s']:.1f} ceil(read) {v['ceiling_tok_s_bytes_read']:.1f} MBU {v['mbu']:.3f} (file {v['mbu_vs_file_bytes']:.3f}) {v['achieved_GBps']:.0f} GB/s = {v['fraction_of_measured_read_bw']:.2f} of read roof; x{v.get('speedup_vs_bf16', 0):.2f} vs bf16 (bytes x{v.get('bytes_ratio_vs_bf16', 0):.2f})")
print("prefill:")
for fam, qs in prefill.items():
    for q, v in qs.items():
        r2048 = v["by_prompt_tokens"]["2048"]
        print(f"  {fam} {q}: 2048 -> {r2048['tok_s_median']:.0f} tok/s {r2048['achieved_TFLOPS']:.1f} TFLOPS ({r2048['fraction_of_measured_matmul_peak']:.2f}); ratio vs decode {v['prefill_over_decode_ratio_at_2048']}")
print("context:")
for fam, qs in context.items():
    for q, v in qs.items():
        print(f"  {fam} {q}: " + ", ".join(f"{T}:{x['tok_s_median']:.1f}/{x['predicted_tok_s_bytes_model']:.1f}" for T, x in v["by_context_tokens"].items()))
print("memory slopes:")
for fam, qs in memory.items():
    for q, ks in qs.items():
        for k, v in ks.items():
            print(f"  {fam} {q} {k}: active {v['slope_active_bytes_per_token']:.0f} B/tok formula {v['formula_bytes_per_token']:.0f} ratio {v['slope_over_formula']:.4f}; peak slope {v['slope_peak_bytes_per_token']:.0f}")
print("batch:")
for fam, qs in batch.items():
    for q, v in qs.items():
        print(f"  {fam} {q}: " + ", ".join(f"B{B}:{x['aggregate_tok_s']:.0f}/{x['per_seq_tok_s']:.1f}" for B, x in v["by_batch"].items()))
print("spec:")
for t, v in spec.items():
    for pn, x in v["prompts"].items():
        print(f"  {t} {pn}: base {x['baseline_tok_s']:.1f}; " + ", ".join(f"k{k}: x{y['speedup']:.2f} acc {y['acceptance_rate']:.2f}" for k, y in x["by_num_draft_tokens"].items()))
print("server:", {k: {c: (round(v['aggregate_tok_s']), round(v['ttft_ms']['p50']), round(v['itl_ms']['p50'], 1)) for c, v in x.items()} for k, x in server.items()})
print("energy:", energy)
print("determinism:", {k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk not in ('hashes', 'hash_counts', 'cached_prompt_tokens')}) for k, v in det.items()})
