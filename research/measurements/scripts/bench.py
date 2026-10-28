"""Inference, Explained: measurement harness (mlx-lm, Apple silicon).

One process per sub-command; each loads its model(s) once, does 1 warm-up and
N timed trials, and writes a JSON result to research/measurements/raw/.

Sub-commands
  roofs      machine micro-benchmarks: read bandwidth, bf16 mat-vec, matmul TFLOPS
  decode     batch-1 decode tok/s on a fixed chat prompt (EOS disabled)
  prefill    TTFT / prefill tok/s vs prompt length
  context    decode tok/s vs context length (prefill once, trim cache per trial)
  memory     peak/active memory and KV-cache bytes vs context (+ --kv-bits)
  rotating   memory + speed during a long generation with --max-kv-size
  kvquality  greedy answer to a question over a long context, KV 16/8/4 bit
  promptcache  TTFT with vs without a saved prefix cache
  batch      aggregate / per-sequence decode tok/s vs batch size (BatchGenerator)
  spec       speculative decoding: tok/s and acceptance vs num_draft_tokens
  determinism  20 greedy runs at batch 1 + the same prompt inside batches

Timing: time.perf_counter() at every token yielded by mlx_lm.stream_generate.
  TTFT     = first token time - call time (prompt already tokenized)
  decode   = (N - 1) / (t_last - t_first)    [mlx-lm itself reports N / that]
"""
import argparse
import copy
import hashlib
import json
import os
import statistics
import sys
import time

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.generate import BatchGenerator, generate_step
from mlx_lm.models.cache import (
    RotatingKVCache,
    load_prompt_cache,
    make_prompt_cache,
    save_prompt_cache,
    trim_prompt_cache,
)
from mlx_lm.sample_utils import make_sampler

ROOT = "/Users/vpromisex/videos/inference-series/research/measurements"
PROMPTS = os.path.join(ROOT, "prompts")
RAW = os.path.join(ROOT, "raw")
GREEDY = make_sampler(0.0)


# ---------------------------------------------------------------- helpers
def summ(xs):
    xs = list(xs)
    return {
        "median": statistics.median(xs),
        "min": min(xs),
        "max": max(xs),
        "n": len(xs),
        "values": xs,
    }


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def read(name):
    return open(os.path.join(PROMPTS, name)).read()


def save(name, obj):
    os.makedirs(RAW, exist_ok=True)
    path = os.path.join(RAW, name + ".json")
    obj["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    obj["argv"] = sys.argv
    json.dump(obj, open(path, "w"), indent=1)
    print("saved", path, flush=True)


_CORPUS = {}


def corpus_ids(tok, n, offset=0):
    key = id(tok)
    if key not in _CORPUS:
        _CORPUS[key] = tok.encode(read("corpus.txt"), add_special_tokens=False)
    ids = _CORPUS[key]
    while len(ids) < n + offset:
        ids = ids + ids
    return ids[offset : offset + n]


def chat_ids(tok, text):
    msgs = [{"role": "user", "content": text}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True)


def load_model(repo, no_eos=True):
    model, tok = load(repo)
    if no_eos:
        tok._eos_token_ids = set()  # same trick as mlx_lm.benchmark
    return model, tok


def reset():
    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()


def run_stream(model, tok, ids, max_tokens, **kw):
    """One generation. Returns timing dict + tokens."""
    reset()
    prompt = mx.array(ids)
    t0 = time.perf_counter()
    times, toks, drafts = [], [], []
    for r in stream_generate(model, tok, prompt, max_tokens=max_tokens, **kw):
        times.append(time.perf_counter())
        toks.append(r.token)
        drafts.append(bool(r.from_draft))
    mx.synchronize()
    peak = mx.get_peak_memory()
    n = len(times)
    ttft = times[0] - t0
    dec = (n - 1) / (times[-1] - times[0]) if n > 1 else None
    itl = [b - a for a, b in zip(times, times[1:])]
    return {
        "prompt_tokens": len(ids),
        "gen_tokens": n,
        "ttft_s": ttft,
        "prefill_tok_s": len(ids) / ttft,
        "decode_tok_s": dec,
        "total_s": times[-1] - t0,
        "itl_p50_ms": pct(itl, 50) * 1e3 if itl else None,
        "itl_p99_ms": pct(itl, 99) * 1e3 if itl else None,
        "peak_memory_bytes": peak,
        "tokens": toks,
        "from_draft": drafts,
    }


def trials(fn, n, warmup=1):
    for _ in range(warmup):
        fn()
    return [fn() for _ in range(n)]


def strip(rs):
    return [{k: v for k, v in r.items() if k not in ("tokens", "from_draft")} for r in rs]


# ---------------------------------------------------------------- roofs
def cmd_roofs(a):
    out = {}
    # 1) read bandwidth: sum over a large fp32 array (pure streaming read)
    for gb in (1, 4):
        n = int(gb * 1e9 / 4)
        x = mx.random.uniform(shape=(n,), dtype=mx.float32)
        mx.eval(x)
        ts = []
        for i in range(12):
            mx.synchronize()
            t = time.perf_counter()
            s = mx.sum(x)
            mx.eval(s)
            ts.append(time.perf_counter() - t)
        ts = ts[2:]
        out[f"read_sum_{gb}GB_GBps"] = summ([n * 4 / t / 1e9 for t in ts])
        # copy: read + write
        ts = []
        for i in range(12):
            mx.synchronize()
            t = time.perf_counter()
            y = x * 1.0001
            mx.eval(y)
            ts.append(time.perf_counter() - t)
            del y
        ts = ts[2:]
        out[f"copy_{gb}GB_GBps_read_plus_write"] = summ([2 * n * 4 / t / 1e9 for t in ts])
        del x
        mx.clear_cache()
    # 2) bf16 mat-vec (the batch-1 decode op): W [R x C] @ x
    R, C = 28672, 16384  # 0.94 GB bf16
    W = mx.random.normal((R, C)).astype(mx.bfloat16)
    v = mx.random.normal((C, 1)).astype(mx.bfloat16)
    mx.eval(W, v)
    ts = []
    for i in range(22):
        mx.synchronize()
        t = time.perf_counter()
        y = W @ v
        mx.eval(y)
        ts.append(time.perf_counter() - t)
    ts = ts[2:]
    out["matvec_bf16_GBps"] = summ([R * C * 2 / t / 1e9 for t in ts])
    # 4-bit quantized mat-vec (group 64), as used by the 4-bit models
    wq, sc, bi = mx.quantize(W, group_size=64, bits=4)
    mx.eval(wq, sc, bi)
    qbytes = wq.nbytes + sc.nbytes + bi.nbytes
    vt = v.T
    ts = []
    for i in range(22):
        mx.synchronize()
        t = time.perf_counter()
        y = mx.quantized_matmul(vt, wq, sc, bi, transpose=True, group_size=64, bits=4)
        mx.eval(y)
        ts.append(time.perf_counter() - t)
    ts = ts[2:]
    out["matvec_q4_GBps"] = summ([qbytes / t / 1e9 for t in ts])
    del W, wq, sc, bi
    mx.clear_cache()
    # 3) matmul throughput (compute roof)
    for dt, name in ((mx.bfloat16, "bf16"), (mx.float16, "fp16"), (mx.float32, "fp32")):
        N = 8192
        A = mx.random.normal((N, N)).astype(dt)
        B = mx.random.normal((N, N)).astype(dt)
        mx.eval(A, B)
        ts = []
        for i in range(8):
            mx.synchronize()
            t = time.perf_counter()
            Cm = A @ B
            mx.eval(Cm)
            ts.append(time.perf_counter() - t)
        ts = ts[2:]
        out[f"matmul_{name}_8192_TFLOPS"] = summ([2 * N**3 / t / 1e12 for t in ts])
        del A, B, Cm
        mx.clear_cache()
    out["device_info"] = {k: v for k, v in mx.device_info().items()}
    save("roofs", out)
    for k, v in out.items():
        if isinstance(v, dict) and "median" in v:
            print(f"{k}: median {v['median']:.1f} [{v['min']:.1f}, {v['max']:.1f}]")


# ---------------------------------------------------------------- decode
def cmd_decode(a):
    model, tok = load_model(a.model)
    ids = chat_ids(tok, read("chat.txt"))
    rs = trials(lambda: run_stream(model, tok, ids, a.max_tokens), a.trials)
    text = tok.decode(rs[0]["tokens"][:120])
    out = {
        "model": a.model,
        "max_tokens": a.max_tokens,
        "prompt_tokens": len(ids),
        "decode_tok_s": summ(r["decode_tok_s"] for r in rs),
        "ttft_s": summ(r["ttft_s"] for r in rs),
        "peak_memory_bytes": summ(r["peak_memory_bytes"] for r in rs),
        "identical_outputs": len({tuple(r["tokens"]) for r in rs}) == 1,
        "sample_text_first120tok": text,
        "trials": strip(rs),
    }
    save(f"decode__{a.tag}", out)
    d = out["decode_tok_s"]
    print(f"{a.tag}: decode {d['median']:.2f} tok/s [{d['min']:.2f},{d['max']:.2f}] n={d['n']}")


# ---------------------------------------------------------------- prefill
def cmd_prefill(a):
    model, tok = load_model(a.model)
    res = {}
    for n in a.lengths:
        ids = corpus_ids(tok, n)
        rs = trials(lambda: run_stream(model, tok, ids, 1), a.trials)
        res[str(n)] = {
            "ttft_s": summ(r["ttft_s"] for r in rs),
            "prefill_tok_s": summ(r["prefill_tok_s"] for r in rs),
            "peak_memory_bytes": summ(r["peak_memory_bytes"] for r in rs),
        }
        p = res[str(n)]["prefill_tok_s"]
        print(f"{a.tag} p={n}: {p['median']:.1f} tok/s [{p['min']:.1f},{p['max']:.1f}] ttft {res[str(n)]['ttft_s']['median']:.3f}s", flush=True)
    save(f"prefill__{a.tag}", {"model": a.model, "prefill_step_size": 2048, "results": res})


# ---------------------------------------------------------------- context
def cmd_context(a):
    model, tok = load_model(a.model)
    res = {}
    for T in a.lengths:
        ids = corpus_ids(tok, T)
        cache = make_prompt_cache(model)
        reset()
        for _ in generate_step(mx.array(ids[:-1]), model, max_tokens=0, prompt_cache=cache):
            pass
        mx.eval([c.state for c in cache])
        base = cache[0].offset

        def one():
            r = run_stream(model, tok, ids[-1:], a.gen, prompt_cache=cache)
            trim_prompt_cache(cache, cache[0].offset - base)
            return r

        rs = trials(one, a.trials)
        res[str(T)] = {
            "decode_tok_s": summ(r["decode_tok_s"] for r in rs),
            "itl_p50_ms": summ(r["itl_p50_ms"] for r in rs),
            "kv_cache_bytes_at_start": sum(c.nbytes for c in cache),
        }
        del cache
        mx.clear_cache()
        d = res[str(T)]["decode_tok_s"]
        print(f"{a.tag} ctx={T}: decode {d['median']:.2f} tok/s [{d['min']:.2f},{d['max']:.2f}]", flush=True)
    save(f"context__{a.tag}", {"model": a.model, "gen_tokens": a.gen, "results": res})


# ---------------------------------------------------------------- memory
def cmd_memory(a):
    model, tok = load_model(a.model)
    reset()
    weights_active = mx.get_active_memory()
    res = {}
    kw = {}
    if a.kv_bits:
        kw = dict(kv_bits=a.kv_bits, kv_group_size=64, quantized_kv_start=0)
    kw["prefill_step_size"] = a.prefill_step
    for T in a.lengths:
        ids = corpus_ids(tok, T)
        rows = []
        for rep in range(2):
            cache = make_prompt_cache(model, max_kv_size=a.max_kv_size)
            r = run_stream(model, tok, ids, a.gen, prompt_cache=cache, **kw)
            mx.synchronize()
            rows.append({
                "peak_memory_bytes": r["peak_memory_bytes"],
                "active_memory_bytes": mx.get_active_memory(),
                "kv_cache_bytes": sum(c.nbytes for c in cache),
                "kv_tokens": cache[0].offset,
                "cache_type": type(cache[0]).__name__,
                "decode_tok_s": r["decode_tok_s"],
            })
            del cache
            mx.clear_cache()
        res[str(T)] = rows[-1] | {"repeat_peak_equal": rows[0]["peak_memory_bytes"] == rows[1]["peak_memory_bytes"]}
        x = res[str(T)]
        print(f"{a.tag} T={T}: peak {x['peak_memory_bytes']/1e9:.3f} GB active {x['active_memory_bytes']/1e9:.3f} GB kv {x['kv_cache_bytes']/2**20:.1f} MiB ({x['cache_type']}, {x['kv_tokens']} tok)", flush=True)
    # least-squares slopes (bytes per token) vs T
    Ts = [int(t) for t in res]

    def slope(key):
        ys = [res[str(t)][key] for t in Ts]
        mt, my = statistics.mean(Ts), statistics.mean(ys)
        return sum((t - mt) * (y - my) for t, y in zip(Ts, ys)) / sum((t - mt) ** 2 for t in Ts)

    out = {
        "model": a.model,
        "kv_bits": a.kv_bits,
        "max_kv_size": a.max_kv_size,
        "prefill_step_size": a.prefill_step,
        "gen_tokens": a.gen,
        "weights_active_bytes": weights_active,
        "results": res,
        "slope_bytes_per_token": {k: slope(k) for k in ("peak_memory_bytes", "active_memory_bytes", "kv_cache_bytes")},
    }
    save(f"memory__{a.tag}", out)
    print({k: round(v) for k, v in out["slope_bytes_per_token"].items()})


# ---------------------------------------------------------------- rotating
def rot_cache(model, mks):
    """make_prompt_cache(model, max_kv_size) -- but mlx-lm 0.32.2's llama.Model
    defines make_cache(), so max_kv_size is silently ignored for Llama. In that
    case build mlx-lm's own RotatingKVCache(max_size, keep=4) explicitly."""
    c = make_prompt_cache(model, max_kv_size=mks)
    if mks is not None and not isinstance(c[0], RotatingKVCache):
        return [RotatingKVCache(max_size=mks, keep=4) for _ in model.layers], True
    return c, False


def cmd_rotating(a):
    model, tok = load_model(a.model)
    ids = chat_ids(tok, read("chat.txt"))
    res = {}
    for label, mks in (("full", None), (f"rotating_{a.max_kv_size}", a.max_kv_size)):
        cache, manual = rot_cache(model, mks)
        reset()
        pts = []
        t_prev = None
        t0 = time.perf_counter()
        for i, r in enumerate(stream_generate(model, tok, mx.array(ids), max_tokens=a.gen, prompt_cache=cache)):
            if (i + 1) % 512 == 0:
                now = time.perf_counter()
                pts.append({
                    "tokens_generated": i + 1,
                    "kv_cache_bytes": sum(c.nbytes for c in cache),
                    "peak_memory_bytes": mx.get_peak_memory(),
                    "active_memory_bytes": mx.get_active_memory(),
                    "window_tok_s": 512 / (now - t_prev) if t_prev else None,
                })
                t_prev = now
        res[label] = {"cache_type": type(cache[0]).__name__, "built_explicitly": manual, "points": pts, "total_s": time.perf_counter() - t0}
        del cache
        mx.clear_cache()
        last = pts[-1]
        print(f"{label}: kv {last['kv_cache_bytes']/2**20:.1f} MiB peak {last['peak_memory_bytes']/1e9:.3f} GB last-window {last['window_tok_s']:.1f} tok/s", flush=True)
    # forgetting demo: a fact at the start of a long prompt
    tok2 = load(a.model)[1]
    fact = "Remember this: the secret code word is PELICAN.\n\n"
    filler = tok2.decode(corpus_ids(tok2, a.forget_ctx))
    q = "\n\nWhat is the secret code word mentioned at the very start of this message? Reply with just the word."
    pids = chat_ids(tok2, fact + filler + q)
    demo = {"prompt_tokens": len(pids)}
    for label, mks in (("full", None), (f"rotating_{a.max_kv_size}", a.max_kv_size)):
        cache, _ = rot_cache(model, mks)
        out = []
        for r in stream_generate(model, tok2, mx.array(pids), max_tokens=12, prompt_cache=cache, sampler=GREEDY):
            out.append(r.token)
        demo[label] = tok2.decode(out)
        print(f"forget demo {label}: {demo[label]!r}", flush=True)
    res["forgetting_demo"] = demo
    save(f"rotating__{a.tag}", {"model": a.model, "max_kv_size": a.max_kv_size, "gen_tokens": a.gen, "results": res})


# ---------------------------------------------------------------- kv quality
def cmd_kvquality(a):
    model, tok = load(a.model)
    filler = tok.decode(corpus_ids(tok, a.ctx))
    q = read("question.txt")
    ids = chat_ids(tok, "Here are my course notes.\n\n" + filler + q)
    res = {"prompt_tokens": len(ids)}
    base = None
    for bits in (None, 8, 4):
        kw = dict(kv_bits=bits, kv_group_size=64, quantized_kv_start=0) if bits else {}
        r = run_stream(model, tok, ids, a.gen, **kw)
        toks = r["tokens"]
        if base is None:
            base = toks
        match = next((i for i, (x, y) in enumerate(zip(base, toks)) if x != y), min(len(base), len(toks)))
        res[str(bits or 16)] = {
            "text": tok.decode(toks),
            "tokens_identical_to_bf16_prefix": match,
            "gen_tokens": len(toks),
            "decode_tok_s": r["decode_tok_s"],
            "prefill_tok_s": r["prefill_tok_s"],
            "peak_memory_bytes": r["peak_memory_bytes"],
        }
        print(f"kv{bits or 16}: match {match}/{len(toks)} | {tok.decode(toks)[:160]!r}", flush=True)
    save(f"kvquality__{a.tag}", {"model": a.model, "results": res})


# ---------------------------------------------------------------- prompt cache
def cmd_promptcache(a):
    model, tok = load(a.model)
    chapter, question = read("chapter.txt"), read("question.txt")
    full = chat_ids(tok, chapter + question)
    # prefix exactly as mlx_lm.cache_prompt builds it
    prefix = tok.apply_chat_template([{"role": "user", "content": chapter}], add_generation_prompt=False, continue_final_message=True)
    assert full[: len(prefix)] == prefix, "prefix mismatch"
    suffix = full[len(prefix):]
    path = os.path.join(ROOT, "logs", f"prefix_cache__{a.tag}.safetensors")
    # build + save the prefix cache (timed once, after a warm-up build)
    def build():
        c = make_prompt_cache(model)
        for _ in generate_step(mx.array(prefix), model, max_tokens=0, prompt_cache=c):
            pass
        mx.eval([x.state for x in c])
        return c
    build()
    t = time.perf_counter(); c = build(); t_build = time.perf_counter() - t
    save_prompt_cache(path, c)
    del c

    def no_cache():
        return run_stream(model, tok, full, a.gen, sampler=GREEDY)

    def with_cache():
        t = time.perf_counter()
        c = load_prompt_cache(path)
        mx.eval([x.state for x in c])
        t_load = time.perf_counter() - t
        r = run_stream(model, tok, suffix, a.gen, prompt_cache=c, sampler=GREEDY)
        r["cache_load_s"] = t_load
        r["ttft_incl_load_s"] = r["ttft_s"] + t_load
        return r

    r0 = trials(no_cache, a.trials)
    r1 = trials(with_cache, a.trials)
    same = r0[0]["tokens"] == r1[0]["tokens"]
    out = {
        "model": a.model,
        "prefix_tokens": len(prefix),
        "suffix_tokens": len(suffix),
        "full_tokens": len(full),
        "prefix_build_s": t_build,
        "cache_file_bytes": os.path.getsize(path),
        "ttft_no_cache_s": summ(r["ttft_s"] for r in r0),
        "ttft_with_cache_s": summ(r["ttft_s"] for r in r1),
        "ttft_with_cache_incl_disk_load_s": summ(r["ttft_incl_load_s"] for r in r1),
        "same_output_tokens": same,
        "answer_no_cache": tok.decode(r0[0]["tokens"]),
        "answer_with_cache": tok.decode(r1[0]["tokens"]),
    }
    os.remove(path)
    save(f"promptcache__{a.tag}", out)
    print(f"{a.tag}: prefix {len(prefix)} suffix {len(suffix)} | TTFT no-cache {out['ttft_no_cache_s']['median']:.3f}s, cache {out['ttft_with_cache_s']['median']:.3f}s, cache+load {out['ttft_with_cache_incl_disk_load_s']['median']:.3f}s same={same}")


# ---------------------------------------------------------------- batch
def run_batch(model, prompts, gen):
    B = len(prompts)
    reset()
    g = BatchGenerator(model, max_tokens=gen, completion_batch_size=B, prefill_batch_size=B)
    t0 = time.perf_counter()
    g.insert(prompts, [gen] * B)
    steps = []  # (time, n_tokens)
    first = None
    while True:
        pr, gr = g.next()
        if gr:
            now = time.perf_counter()
            if first is None:
                first = now
            steps.append((now, len(gr)))
        if not gr and not pr and not len(g._generation_batch) and not g._unprocessed_sequences:
            break
        if steps and sum(n for _, n in steps) >= B * gen:
            break
    mx.synchronize()
    g.close()
    peak = mx.get_peak_memory()
    full = [(t, n) for t, n in steps if n == B]
    # steady state: consecutive steps where all B sequences produced a token
    dt = full[-1][0] - full[0][0]
    toks = sum(n for _, n in full[1:])
    agg = toks / dt
    return {
        "batch": B,
        "ttft_all_s": first - t0,
        "aggregate_tok_s": agg,
        "per_seq_tok_s": agg / B,
        "step_ms": dt / (len(full) - 1) * 1e3,
        "steady_steps": len(full),
        "total_tokens": sum(n for _, n in steps),
        "peak_memory_bytes": peak,
    }


def cmd_batch(a):
    model, tok = load_model(a.model)
    res = {}
    for B in a.sizes:
        prompts = [corpus_ids(tok, a.prompt, offset=997 * i) for i in range(B)]
        rs = trials(lambda: run_batch(model, prompts, a.gen), a.trials)
        res[str(B)] = {k: summ(r[k] for r in rs) for k in ("aggregate_tok_s", "per_seq_tok_s", "step_ms", "ttft_all_s", "peak_memory_bytes")}
        x = res[str(B)]
        print(f"{a.tag} B={B}: agg {x['aggregate_tok_s']['median']:.1f} tok/s [{x['aggregate_tok_s']['min']:.1f},{x['aggregate_tok_s']['max']:.1f}] per-seq {x['per_seq_tok_s']['median']:.1f}", flush=True)
    save(f"batch__{a.tag}", {"model": a.model, "prompt_tokens": a.prompt, "gen_tokens": a.gen, "results": res})


# ---------------------------------------------------------------- spec
def cmd_spec(a):
    model, tok = load(a.model)  # keep EOS; prompts ask for long answers
    draft, dtok = load(a.draft)
    assert dtok.vocab_size == tok.vocab_size
    res = {}
    for pname in ("chat.txt", "notes_rewrite.txt"):
        ids = chat_ids(tok, read(pname))
        base = trials(lambda: run_stream(model, tok, ids, a.max_tokens, sampler=GREEDY), a.trials)
        bt = summ(r["decode_tok_s"] for r in base)
        btot = summ(r["gen_tokens"] / r["total_s"] for r in base)
        row = {"prompt_tokens": len(ids), "baseline": {"decode_tok_s": bt, "end_to_end_tok_s": btot, "gen_tokens": base[0]["gen_tokens"]}}
        print(f"{pname} baseline {bt['median']:.2f} tok/s ({base[0]['gen_tokens']} tok)", flush=True)
        for k in a.k:
            rs = trials(lambda: run_stream(model, tok, ids, a.max_tokens, sampler=GREEDY, draft_model=draft, num_draft_tokens=k), a.trials)
            r = rs[0]
            # Reconstruct verify passes exactly as speculative_generate_step runs
            # them: each pass drafts min(k, remaining) tokens, yields the accepted
            # prefix (from_draft=True) then one target token (False) unless
            # max_tokens is reached first.
            fd, i, ntok, rounds, drafted, acc = r["from_draft"], 0, 0, 0, 0, 0
            while i < len(fd):
                nd = min(k, a.max_tokens - ntok)
                rounds += 1
                drafted += nd
                while i < len(fd) and fd[i]:
                    acc += 1; i += 1; ntok += 1
                if i < len(fd):  # the target's own token ends the pass
                    i += 1; ntok += 1
            d = summ(x["decode_tok_s"] for x in rs)
            e2e = summ(x["gen_tokens"] / x["total_s"] for x in rs)
            row[f"k{k}"] = {
                "decode_tok_s": d,
                "end_to_end_tok_s": e2e,
                "speedup_decode": d["median"] / bt["median"],
                "speedup_end_to_end": e2e["median"] / btot["median"],
                "gen_tokens": r["gen_tokens"],
                "accepted_draft_tokens": acc,
                "drafted_tokens": drafted,
                "target_passes": rounds,
                "acceptance_rate": acc / drafted if drafted else None,
                "from_draft_sequence": "".join("D" if x else "T" for x in fd),
                "tokens_per_target_pass": r["gen_tokens"] / rounds if rounds else None,
                "output_identical_to_baseline": r["tokens"] == base[0]["tokens"],
                "first_divergence": next((i for i, (x, y) in enumerate(zip(r["tokens"], base[0]["tokens"])) if x != y), None),
            }
            x = row[f"k{k}"]
            print(f"  k={k}: {d['median']:.2f} tok/s x{x['speedup_decode']:.2f} accept {x['acceptance_rate']:.2f} tok/pass {x['tokens_per_target_pass']:.2f} same={x['output_identical_to_baseline']}", flush=True)
        row["baseline_text"] = tok.decode(base[0]["tokens"])
        res[pname] = row
    save(f"spec__{a.tag}", {"model": a.model, "draft": a.draft, "max_tokens": a.max_tokens, "results": res})


# ---------------------------------------------------------------- determinism
def cmd_determinism(a):
    model, tok = load(a.model)
    ids = chat_ids(tok, read(a.prompt_file))
    h = lambda t: hashlib.sha256(json.dumps(t).encode()).hexdigest()[:16]
    runs = [run_stream(model, tok, ids, a.gen, sampler=GREEDY)["tokens"] for _ in range(a.runs)]
    hashes = [h(r) for r in runs]
    ref = runs[0]
    out = {"model": a.model, "prompt_file": a.prompt_file, "gen_tokens": a.gen, "batch1": {"runs": a.runs, "distinct_outputs": len(set(hashes)), "hashes": hashes}}
    print(f"batch-1: {len(set(hashes))} distinct of {a.runs}", flush=True)
    # the same prompt as member 0 of a batch with other (different) prompts
    others = [chat_ids(tok, t) for t in (
        "Explain photosynthesis to a ten-year-old.",
        "Write a haiku about rain on a tin roof.",
        "What is the capital of Australia, and why is it not Sydney?",
        "Give three tips for revising for a maths exam.",
        "Describe how a bicycle gear works.",
        "Summarise the plot of Romeo and Juliet in four sentences.",
        "Why is the sea salty?",
        "Name three uses of magnets at home.",
    )]
    bres = {}
    for B in (1, 2, 4, 8, 16):
        prompts = [ids] + [others[i % len(others)] for i in range(B - 1)]
        g = BatchGenerator(model, max_tokens=a.gen, completion_batch_size=B, prefill_batch_size=B)
        uids = g.insert(prompts, [a.gen] * B)
        seq = {u: [] for u in uids}
        while True:
            _, gr = g.next()
            for r in gr:
                seq[r.uid].append(r.token)
            if all(len(v) >= a.gen for v in seq.values()):
                break
        g.close()
        mine = seq[uids[0]]
        div = next((i for i, (x, y) in enumerate(zip(mine, ref)) if x != y), None)
        bres[str(B)] = {"identical_to_batch1_stream": mine == ref, "first_divergence_token": div, "hash": h(mine)}
        if div is not None:
            bres[str(B)]["ref_text_after"] = tok.decode(ref[max(0, div - 5): div + 15])
            bres[str(B)]["batch_text_after"] = tok.decode(mine[max(0, div - 5): div + 15])
        print(f"B={B}: identical={mine == ref} first_div={div}", flush=True)
    out["in_batch"] = bres
    out["reference_text"] = tok.decode(ref)
    save(f"determinism__{a.tag}", out)


# ---------------------------------------------------------------- energy
def cmd_energy(a):
    """GPU energy from IOReport 'Energy Model' counters via zeus-apple-silicon
    (no sudo). On this macOS only the GPU channel updates; CPU and DRAM stay
    frozen, so these are GPU-only, model-based estimates."""
    from zeus_apple_silicon import AppleEnergyMonitor

    mon = AppleEnergyMonitor()
    model, tok = load_model(a.model)
    ids = chat_ids(tok, read("chat.txt"))

    def window(fn):
        mx.synchronize()
        mon.begin_window("w")
        t = time.perf_counter()
        r = fn()
        mx.synchronize()
        dt = time.perf_counter() - t
        e = mon.end_window("w")
        return r, e.gpu_mj / 1e3, dt

    time.sleep(2)
    _, j, dt = window(lambda: time.sleep(5))
    idle_w = j / dt
    out = {"model": a.model, "gpu_idle_W": idle_w}
    run_stream(model, tok, ids, 64)  # warm-up
    rows = []
    for _ in range(a.trials):
        r, j, dt = window(lambda: run_stream(model, tok, ids, a.max_tokens))
        rows.append({"gpu_J": j, "s": dt, "tokens": r["gen_tokens"], "gpu_W": j / dt,
                     "J_per_token": j / r["gen_tokens"], "decode_tok_s": r["decode_tok_s"]})
    out["decode_b1"] = {k: summ(x[k] for x in rows) for k in ("J_per_token", "gpu_W", "decode_tok_s")}
    pids = corpus_ids(tok, 2048)
    run_stream(model, tok, pids, 1)
    rows = []
    for _ in range(a.trials):
        r, j, dt = window(lambda: run_stream(model, tok, pids, 1))
        rows.append({"J_per_prompt_token": j / 2048, "gpu_W": j / dt})
    out["prefill_2048"] = {k: summ(x[k] for x in rows) for k in ("J_per_prompt_token", "gpu_W")}
    B = 16
    prompts = [corpus_ids(tok, 256, offset=997 * i) for i in range(B)]
    run_batch(model, prompts, 32)
    rows = []
    for _ in range(a.trials):
        r, j, dt = window(lambda: run_batch(model, prompts, a.max_tokens))
        rows.append({"J_per_generated_token": j / r["total_tokens"], "gpu_W": j / dt, "aggregate_tok_s": r["aggregate_tok_s"]})
    out[f"decode_b{B}"] = {k: summ(x[k] for x in rows) for k in ("J_per_generated_token", "gpu_W", "aggregate_tok_s")}
    save(f"energy__{a.tag}", out)
    print(f"{a.tag}: idle {idle_w:.2f} W | b1 decode {out['decode_b1']['J_per_token']['median']:.3f} J/tok at {out['decode_b1']['gpu_W']['median']:.1f} W | "
          f"prefill {out['prefill_2048']['J_per_prompt_token']['median']*1e3:.1f} mJ/prompt-tok at {out['prefill_2048']['gpu_W']['median']:.1f} W | "
          f"b16 {out['decode_b16']['J_per_generated_token']['median']:.3f} J/tok at {out['decode_b16']['gpu_W']['median']:.1f} W")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd")
    p.add_argument("--model")
    p.add_argument("--draft")
    p.add_argument("--tag", default="x")
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--gen", type=int, default=128)
    p.add_argument("--prompt", type=int, default=256)
    p.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096, 8192])
    p.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--k", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    p.add_argument("--kv-bits", type=int, default=None)
    p.add_argument("--max-kv-size", type=int, default=None)
    p.add_argument("--forget-ctx", type=int, default=3000)
    p.add_argument("--ctx", type=int, default=8000)
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--prefill-step", type=int, default=2048)
    p.add_argument("--prompt-file", default="determinism.txt")
    a = p.parse_args()
    globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
