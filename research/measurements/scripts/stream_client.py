"""Streaming load client for mlx_lm.server (OpenAI-compatible SSE).

N concurrent client threads, each sending R requests back-to-back
(closed loop). Every SSE chunk carrying text is timestamped with
time.perf_counter():
  TTFT = first text chunk - request sent
  ITL  = gaps between consecutive text chunks (one chunk per token in
         mlx-lm 0.32.2's server, except tokens that decode to no text)
Usage counts (prompt/completion/cached tokens) come from stream_options.

stdlib only.  Usage:
  python -I stream_client.py --port 8090 --clients 4 --requests 5 \
      --max-tokens 128 --tag x --out raw/server__x.json [--same-prompt FILE]
"""
import argparse
import hashlib
import http.client
import json
import statistics
import threading
import time

TOPICS = [
    "how volcanoes form", "why leaves change colour in autumn", "how a rainbow forms",
    "how vaccines train the immune system", "why the Moon has phases", "how bees make honey",
    "how a steam engine works", "why ice floats", "how the heart pumps blood",
    "how tides work", "why the sky is blue", "how plants drink water",
    "how sound travels", "why we yawn", "how a compass works", "how bread rises",
    "how earthquakes are measured", "why cats purr", "how glaciers move",
    "how batteries store energy", "how clouds form", "why bridges sway",
    "how a lighthouse lens works", "how coral reefs grow",
]


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def one_request(port, content, max_tokens):
    """Retries on connection reset (mlx_lm.server uses http.server's
    ThreadingHTTPServer, listen backlog 5). TTFT is measured from the FIRST
    attempt, i.e. what the user waits."""
    t_first = time.perf_counter()
    for attempt in range(30):
        try:
            r = _one_request(port, content, max_tokens)
            r["retries"] = attempt
            r["ttft_s"] = (r["ttft_s"] + (r["t_send"] - t_first)) if r["ttft_s"] is not None else None
            r["t_send"] = t_first
            return r
        except (ConnectionResetError, BrokenPipeError, ConnectionRefusedError, OSError) as e:
            last = e
            time.sleep(0.05 * (attempt + 1))
    raise last


def _one_request(port, content, max_tokens):
    body = json.dumps({
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    })
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    t0 = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    resp = conn.getresponse()
    stamps, text, usage, finish = [], [], None, None
    while True:
        line = resp.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            break
        now = time.perf_counter()
        d = json.loads(payload)
        if d.get("usage"):
            usage = d["usage"]
        for ch in d.get("choices", []):
            piece = (ch.get("delta") or {}).get("content")
            if piece:
                stamps.append(now)
                text.append(piece)
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    t_end = time.perf_counter()
    conn.close()
    return {
        "t_send": t0,
        "t_end": t_end,
        "ttft_s": stamps[0] - t0 if stamps else None,
        "itl_s": [b - a for a, b in zip(stamps, stamps[1:])],
        "chunks": len(stamps),
        "usage": usage,
        "finish_reason": finish,
        "text": "".join(text),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--clients", type=int, default=1)
    p.add_argument("--requests", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--same-prompt", default=None, help="file: every request sends exactly this text")
    p.add_argument("--tag", default="x")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    same = open(a.same_prompt).read() if a.same_prompt else None
    results = [[] for _ in range(a.clients)]
    barrier = threading.Barrier(a.clients)

    def client(ci):
        barrier.wait()
        time.sleep(0.02 * ci)  # 20 ms stagger so 16 connects do not arrive in one burst
        for r in range(a.requests):
            if same is not None:
                content = same
            else:
                topic = TOPICS[(ci * a.requests + r) % len(TOPICS)]
                content = (f"[client {ci} request {r}] Write a detailed, 400-word explanation of "
                           f"{topic} for a curious 12-year-old.")
            results[ci].append(one_request(a.port, content, a.max_tokens))

    t0 = time.perf_counter()
    th = [threading.Thread(target=client, args=(i,)) for i in range(a.clients)]
    [t.start() for t in th]
    [t.join() for t in th]
    wall = time.perf_counter() - t0

    reqs = [r for rs in results for r in rs]
    ttft = [r["ttft_s"] for r in reqs if r["ttft_s"] is not None]
    itl = [x for r in reqs for x in r["itl_s"]]
    comp = sum((r["usage"] or {}).get("completion_tokens", r["chunks"]) for r in reqs)
    per_req_tps = [(r["chunks"] - 1) / sum(r["itl_s"]) for r in reqs if r["itl_s"]]
    out = {
        "tag": a.tag,
        "clients": a.clients,
        "requests_per_client": a.requests,
        "max_tokens": a.max_tokens,
        "n_requests": len(reqs),
        "wall_s": wall,
        "aggregate_completion_tok_s": comp / wall,
        "completion_tokens_total": comp,
        "ttft_ms": {k: pct(ttft, q) * 1e3 for k, q in (("p50", 50), ("p90", 90), ("p99", 99))} | {"max": max(ttft) * 1e3, "min": min(ttft) * 1e3},
        "itl_ms": {k: pct(itl, q) * 1e3 for k, q in (("p50", 50), ("p90", 90), ("p99", 99))} | {"max": max(itl) * 1e3, "n": len(itl)},
        "per_request_decode_tok_s": {"median": statistics.median(per_req_tps), "min": min(per_req_tps), "max": max(per_req_tps)},
        "retries_total": sum(r["retries"] for r in reqs),
        "requests_with_retries": sum(1 for r in reqs if r["retries"]),
        "finish_reasons": {f: sum(1 for r in reqs if r["finish_reason"] == f) for f in {r["finish_reason"] for r in reqs}},
        "cached_prompt_tokens": [((r["usage"] or {}).get("prompt_tokens_details") or {}).get("cached_tokens") for r in reqs],
        "prompt_tokens": [(r["usage"] or {}).get("prompt_tokens") for r in reqs],
        "text_hashes": [hashlib.sha256(r["text"].encode()).hexdigest()[:16] for r in reqs],
        "requests": [{k: v for k, v in r.items() if k != "itl_s"} for r in reqs],
    }
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{a.tag}: clients={a.clients} n={len(reqs)} agg {out['aggregate_completion_tok_s']:.1f} tok/s | "
          f"TTFT p50 {out['ttft_ms']['p50']:.0f} p90 {out['ttft_ms']['p90']:.0f} p99 {out['ttft_ms']['p99']:.0f} ms | "
          f"ITL p50 {out['itl_ms']['p50']:.1f} p90 {out['itl_ms']['p90']:.1f} p99 {out['itl_ms']['p99']:.1f} ms | "
          f"per-req {out['per_request_decode_tok_s']['median']:.1f} tok/s | finish {out['finish_reasons']} | retries {out['retries_total']} on {out['requests_with_retries']} req")


if __name__ == "__main__":
    main()
