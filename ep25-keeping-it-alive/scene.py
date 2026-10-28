"""Tokens per Second · Episode 25 · Keeping It Alive (Module 6 · Production & Your Mac)

What to watch, and what to do, when load spikes. Running example: exam week. Owl-8B (BF16) serves students
from one H100; one evening the alert fires: the ninety-ninth percentile time to first token (TTFT) is over its
two-second target. The episode replays that evening on a dashboard of six panels (TTFT and the gap between
tokens as p50 / p99, tokens per second, requests waiting, KV-cache usage, preemptions), reads the diagnosis
(the KV cache is full, so requests queue and running ones are preempted and recomputed: out of cache memory,
not out of math; Little's law from Episode 24 gives the limit), pulls the levers (admission control with
HTTP 429 + Retry-After, a max-tokens cap, priority, shedding load, graceful degradation) and replays the
evening again. Then the rehearsal: a load generator against mlx_lm.server on this Mac (1 / 4 / 16 clients),
percentiles not means, and the breaking point it found. Episodes 6, 7, 10, 11, 12, 22–24 are linked by number.

THE EVENING IS A SIMULATION, computed below (illustrative, "for example"): a vLLM-V1-like scheduler on one
H100 serving Owl-8B BF16 — continuous batching (owl_reference.continuous_batching, Episode 10) with a memory
limit: 16-token KV blocks (Episode 7) in a pool of 90% of 80 GB minus the weights; a token budget per step
with chunked prefill, decodes first (Episode 11); a step lasts max(weights + every running request's cache
at 75% of 3.35 TB/s, its tokens × 2P FLOPs at 50% of 989 TFLOP/s) (owl_reference.step_time / prefill_time,
the MBU of Episode 24, the MFU of Episode 11); when a running request needs a block and none is free, the most
recently admitted request is preempted: its blocks are freed and it is recomputed later (vLLM V1's default,
RECOMPUTE rather than SWAP). The workload is invented: chats (2,000-token prompts, 90% a cached prefix as in
Episode 24, 300-token answers) and flashcard decks (4,000 fresh prompt tokens, 1,500-token answers), Poisson
arrivals, 14 chats + 0.1 decks a second at 20:00, ramping to 17 + 0.4 between 20:10 and 20:16; a student gives
up after 30 s without a first token. Levers at 20:20 (four of the five are simulated): admission control
(queue capped at about one second of work, the rest get HTTP 429), max_tokens 1,024, priority (chats first,
decks preempted first) and shedding (a queued deck makes room for a chat). Every number on screen is computed
from that run (or from owl_reference / measured()) and asserted.

Sources (facts dated October 2026; research/notes.md §16 Observability and reliability, §15 Benchmarking
methodology, §5 preemption; docs re-read 2026-10-24):
- vLLM docs, Production Metrics: vllm:time_to_first_token_seconds, vllm:inter_token_latency_seconds,
  vllm:request_time_per_output_token_seconds, vllm:request_queue_time_seconds, vllm:num_requests_waiting,
  vllm:num_requests_running, vllm:kv_cache_usage_perc ("KV-cache usage. 1 means 100 percent usage."),
  vllm:num_preemptions ("Cumulative number of preemption from the engine.").
- vLLM docs, Optimization and Tuning, Preemption: "Sequence group 0 is preempted by PreemptionMode.RECOMPUTE
  mode because there is not enough KV cache space."; "In vLLM V1, the default preemption mode is RECOMPUTE
  rather than SWAP, as recomputation has lower overhead in the V1 architecture."; fixes: increase
  gpu_memory_utilization, decrease max_num_seqs or max_num_batched_tokens, increase tensor_parallel_size.
- Notes §16 reliability patterns: admission control ("a bounded queue, then HTTP 429 with Retry-After once queue
  time would break the SLO"), timeouts and client-disconnect detection, cap max_tokens ("a runaway generation
  holds a batch slot and KV for minutes"), graceful degradation (shorter max_tokens, no speculative decoding,
  a smaller or quantized model, no long context); the notes' worked p99 TTFT alert (KV ≈ 0.98, preemptions
  rising, TPOT p50 flat; fixes: a replica scaled on queue size, FP8 KV, a context cap).
- Notes §15: MLPerf Llama-2-70B server scenario "TTFT: <= 2 seconds" (the example SLO here); benchmarking
  rules: warm up, realistic length distributions, open-loop Poisson arrivals for latency SLOs, report p50 / p90 /
  p99 not means, goodput at the SLO, a fixed prefix-cache state; tools: vllm bench serve, GuideLLM, AIPerf.
- GKE inference autoscaling guide (notes §13): "Use queue size to maximize throughput and minimize cost within a
  certain target latency threshold".
- Measured on this Mac (research/measurements.md §10, 2026-10-24, mlx-lm 0.32.2): mlx_lm.server, Owl-8B 4-bit,
  tools/stream_client.py (closed loop, ~64-token prompts, 128-token answers); 1 / 4 / 16 clients: TTFT p99 74 /
  662 / 1,210 ms, ITL p50 10.6 / 15.9 / 47.1 ms, aggregate 90 / 202 / 290 tok/s; ceiling 546 GB/s ÷ 4.52 GB =
  8.3 ms per token; a first run "lost requests to connection resets at 15–16 simultaneous connects"
  (ThreadingHTTPServer, listen backlog 5); the final client staggers starts and retries.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import heapq
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from owl_reference import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- one H100 serving Owl-8B

GB = 1e9
KVB = kv_bytes_per_token()                       # 128 KiB per token (Episode 5)
BLOCK = 16                                       # tokens per KV block, vLLM's default (Episode 7)
MEM_UTIL = 0.9                                   # vLLM's default gpu_memory_utilization (notes §3: 90% of 80 GB)
POOL = MEM_UTIL * GPU["memory"] - weight_bytes() # bytes left for the KV cache
BLOCKS = int(POOL // (BLOCK * KVB))
SLOTS = BLOCKS * BLOCK
MBU, MFU = 0.75, 0.5                             # Episode 24's 75% of bandwidth, Episode 11's 50% of peak math
BUDGET = 2048                                    # tokens per step, decodes first, then prefill chunks (Episode 11)
MAX_SEQS = 256
FPT = flops_per_token()
FULL_GAP = (weight_bytes() + SLOTS * KVB) / (GPU["bandwidth"] * MBU)   # a decode step with the cache full
CHUNK_GAP = BUDGET * FPT / (GPU["peak"] * MFU)                          # a step with a full prefill chunk
assert round(POOL / GB, 1) == 55.9 and BLOCKS == 26_674 and SLOTS == 426_784
assert round(FULL_GAP * 1e3, 1) == 28.7 and round(CHUNK_GAP * 1e3, 1) == 66.5

# ---------------------------------------------------------------------------- the evening (illustrative)

SPAN = 32                     # minutes shown, from 20:00
WARM = 4                      # minutes simulated before 20:00, so the server isn't empty at the start
RAMP = (10, 16)               # exam-week traffic ramps up between 20:10 and 20:16
CALM = (14.0, 0.1)            # chats, decks per second before
EXAM = (17.0, 0.4)            # … and after
GIVE_UP = 30.0                # a student gives up after 30 s without a first token
LEVERS_AT = 20                # minute the on-call engineer pulls the levers
CAP = 1024                    # max_tokens after the levers
SLO = 2.0                     # p99 TTFT target, s (MLPerf's Llama-2-70B server scenario: TTFT <= 2 s)


def rates(minute):
    f = float(np.clip((minute - RAMP[0]) / (RAMP[1] - RAMP[0]), 0, 1))
    return tuple(a + (b - a) * f for a, b in zip(CALM, EXAM))


def workload(seed=25):
    """Poisson arrivals, one second at a time; each an owl_reference.Request (arrival in seconds)."""
    rng = np.random.default_rng(seed)
    out = []
    for s in range(-WARM * 60, SPAN * 60):
        for kind, lam in zip(("chat", "deck"), rates(s / 60)):
            for a in np.sort(rng.uniform(s, s + 1, rng.poisson(lam))):
                if kind == "chat":   # 2,000-token prompt (90% a cached prefix), 300-token answer
                    p = int(np.clip(rng.lognormal(np.log(2000), 0.35), 300, 8000))
                    o = int(np.clip(rng.lognormal(np.log(300), 0.5), 20, 2000))
                else:                # a pasted chapter → a deck of flashcards
                    p = int(np.clip(rng.lognormal(np.log(4000), 0.4), 800, 16000))
                    o = int(np.clip(rng.lognormal(np.log(1500), 0.4), 200, 4096))
                out.append((float(a), kind, p, o))
    out.sort()
    return [Request(f"{k[0]}{i}", a, p, o) for i, (a, k, p, o) in enumerate(out)]


class Job:
    """A Request inside the server: its cache, its answer so far, and what happened to it."""
    __slots__ = ("req", "kind", "out", "new", "gen", "cached", "target", "state", "c0", "d0", "seq", "first",
                 "last", "admit", "fin", "preempted")

    def __init__(self, req):
        self.req, self.kind, self.out = req, ("chat" if req.rid[0] == "c" else "deck"), req.output
        self.new = req.prompt // 10 if self.kind == "chat" else req.prompt      # tokens that need prefill math
        self.gen = self.cached = self.seq = self.preempted = 0
        self.target, self.state = req.prompt, "new"
        self.c0 = self.d0 = 0
        self.first = self.last = self.admit = self.fin = None


def blocks_of(tokens):
    return -(-tokens // BLOCK)


def simulate(levers_at=None, q_max=None, cap=None):
    """Steps of one engine. Returns per-step logs and per-request events (times in seconds from 20:00)."""
    jobs = [Job(r) for r in workload()]
    t, D, used, seq = -WARM * 60.0, 0, 0, 0
    chats, decks = deque(), deque()        # waiting; without priority everybody queues in `chats`
    prefill, stack, heap = [], [], []
    phase = [0] * BLOCK                    # decoders by (c0 - d0) mod 16: who needs a fresh block on step D
    s_c0 = s_d0 = n_dec = 0
    steps, ttft, done, rejected, shed, gave_up, stalls = [], [], [], [], [], [], []
    i = 0

    def on():
        return levers_at is not None and t >= levers_at * 60

    def cached_now(j):
        return j.c0 + (D - j.d0) if j.state == "dec" else j.cached

    def leave(j):                          # out of the running set: free its blocks
        nonlocal used, s_c0, s_d0, n_dec
        c = cached_now(j)
        if j.state == "dec":
            j.gen += D - j.d0
            phase[(j.c0 - j.d0) % BLOCK] -= 1
            s_c0, s_d0, n_dec = s_c0 - j.c0, s_d0 - j.d0, n_dec - 1
        else:
            prefill.remove(j)
        used -= blocks_of(c)

    def preempt():                         # the newest (with priority: the newest deck) loses its cache
        for k in (("deck", "chat") if on() else (None,)):
            for x in range(len(stack) - 1, -1, -1):
                j = stack[x]
                if j.state in ("dec", "pre") and (k is None or j.kind == k):
                    stack.pop(x)
                    leave(j)
                    j.state, j.last, j.cached = "wait", t, 0
                    j.target = j.req.prompt + j.gen          # recompute the prompt and the answer so far
                    j.preempted += 1
                    (decks if on() and j.kind == "deck" else chats).appendleft(j)
                    return True
        return False

    while t < SPAN * 60:
        while i < len(jobs) and jobs[i].req.arrival <= t:
            j = jobs[i]
            i += 1
            if on():
                j.out = min(j.out, cap)
                if len(chats) + len(decks) >= q_max:         # admission control, shedding decks first
                    if j.kind == "chat" and decks:
                        s = decks.pop()
                        s.state = "shed"
                        shed.append(s.req.arrival)
                    else:
                        j.state = "rejected"
                        rejected.append(j.req.arrival)
                        continue
            j.state = "wait"
            (decks if on() and j.kind == "deck" else chats).append(j)
        for q in (chats, decks):                             # students who waited too long give up
            while q and q[0].gen == 0 and t - q[0].req.arrival > GIVE_UP:
                q.popleft().state = "gave up"
                gave_up.append(t)
        need, pre = phase[(-D) % BLOCK], 0                   # decoders crossing into a new block this step
        while need > BLOCKS - used and preempt():
            pre += 1
            need = phase[(-D) % BLOCK]
        used += need
        budget, chunks = BUDGET - n_dec, []
        for j in list(prefill):                              # prompts already admitted go on
            if budget <= 0:
                break
            c = min(budget, j.target - j.cached)
            nb = blocks_of(j.cached + c) - blocks_of(j.cached)
            if nb <= BLOCKS - used:
                used += nb
                chunks.append((j, c))
                budget -= c
        while budget > 0 and n_dec + len(prefill) < MAX_SEQS and (chats or decks):
            j = chats[0] if chats else decks[0]
            hit = j.req.prompt - j.new if j.preempted == 0 else 0     # a cached prefix: blocks, but no math
            c = min(budget, j.target - hit)
            if used + blocks_of(hit + c) > BLOCKS:
                break
            (chats if chats and j is chats[0] else decks).popleft()
            used += blocks_of(hit + c)
            seq += 1
            j.state, j.cached, j.seq = "pre", hit, seq
            j.admit = t if j.admit is None else j.admit
            stack.append(j)
            prefill.append(j)
            chunks.append((j, c))
            budget -= c
        kv = s_c0 - s_d0 + n_dec * D + sum(j.cached for j in prefill)
        work = n_dec + sum(c for _, c in chunks)
        dt = max((weight_bytes() + kv * KVB) / (GPU["bandwidth"] * MBU), work * FPT / (GPU["peak"] * MFU))
        tokens, running = n_dec, n_dec + len(prefill)
        deck_blocks = -1
        if len(steps) % 40 == 0:                             # sample who holds the cache
            deck_blocks = sum(blocks_of(cached_now(j)) for j in stack if j.state in ("dec", "pre") and j.kind == "deck")
        t += dt
        D += 1
        for j, c in chunks:
            j.cached += c
            if j.cached < j.target:
                continue
            prefill.remove(j)
            j.gen += 1
            tokens += 1
            if j.first is None:
                j.first = t
                ttft.append((t, t - j.req.arrival))
            else:
                stalls.append((t, t - j.last))
            j.last = t
            if j.gen >= j.out:
                used -= blocks_of(j.cached)
                j.state, j.fin = "done", t
                done.append((t, j.kind, j.fin - j.admit))
                continue
            j.state, j.c0, j.d0 = "dec", j.cached, D
            phase[(j.c0 - j.d0) % BLOCK] += 1
            s_c0, s_d0, n_dec = s_c0 + j.c0, s_d0 + j.d0, n_dec + 1
            heapq.heappush(heap, (D + j.out - j.gen, j.seq, j))
        while heap and heap[0][0] <= D:
            d, s, j = heapq.heappop(heap)
            if j.state != "dec" or j.seq != s or d != j.d0 + j.out - j.gen:
                continue                                     # a stale entry (preempted since)
            leave(j)
            j.state, j.fin = "done", t
            done.append((t, j.kind, j.fin - j.admit))
        steps.append((t, dt, n_dec, used, len(chats) + len(decks), pre, tokens, running, deck_blocks))
    return dict(steps=np.array(steps), ttft=np.array(ttft), stalls=np.array(stalls), done=done,
                rejected=np.array(rejected), shed=np.array(shed), gave_up=np.array(gave_up), jobs=jobs)


# the evening without levers (for the diagnosis), and the same evening with the levers pulled at 20:20
CAPACITY_PROBE = simulate()
SAT = [s for s in CAPACITY_PROBE["steps"] if 15 * 60 <= s[0] < LEVERS_AT * 60]
SAT_DT = np.array([s[1] for s in SAT])
N_FLIGHT = float(np.average([s[7] for s in SAT], weights=SAT_DT))           # requests holding cache
DONE_SAT = [d for d in CAPACITY_PROBE["done"] if 15 * 60 <= d[0] < LEVERS_AT * 60]
MU = len(DONE_SAT) / ((LEVERS_AT - 15) * 60)                                 # finished per second
W_STAY = float(np.mean([d[2] for d in DONE_SAT]))                            # admitted → done, seconds
LAM = sum(EXAM)
DECK_SHARE = float(np.mean([s[8] / s[3] for s in SAT if s[8] >= 0]))        # share of used blocks held by decks
DECK_REQ = EXAM[1] / LAM

Q_MAX = round(MU * 1.0)                    # admission control: about one second of work may wait
RUN = simulate(LEVERS_AT, Q_MAX, CAP)

# ---------------------------------------------------------------------------- the dashboard's series (30 s bins)

ST = RUN["steps"]
TT = RUN["ttft"]
GRID = np.arange(0, SPAN * 60 + 1, 30.0)   # seconds from 20:00


def window(t, width=60.0):
    return (ST[:, 0] > t - width) & (ST[:, 0] <= t)


def wpct(values, weights, q):
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w) / w.sum()
    return float(v[np.searchsorted(cum, q / 100)])


def series():
    out = {k: [] for k in ("ttft50", "ttft99", "gap50", "gap99", "tok", "wait", "kv", "pre", "arr")}
    stalls = RUN["stalls"]
    for t in GRID:
        sel = window(t)
        tt = TT[(TT[:, 0] > t - 60) & (TT[:, 0] <= t), 1]
        out["ttft50"].append(float(np.percentile(tt, 50)))
        out["ttft99"].append(float(np.percentile(tt, 99)))
        gaps = np.concatenate([ST[sel, 1], stalls[(stalls[:, 0] > t - 60) & (stalls[:, 0] <= t), 1]])
        w = np.concatenate([ST[sel, 2], np.ones(int(((stalls[:, 0] > t - 60) & (stalls[:, 0] <= t)).sum()))])
        out["gap50"].append(wpct(gaps, w, 50) * 1e3)
        out["gap99"].append(wpct(gaps, w, 99) * 1e3)
        half = window(t, 30.0)
        out["tok"].append(float(ST[half, 6].sum() / ST[half, 1].sum()))
        out["wait"].append(float(np.average(ST[half, 4], weights=ST[half, 1])))
        out["kv"].append(float(np.average(ST[half, 3], weights=ST[half, 1]) / BLOCKS * 100))
        out["pre"].append(float(ST[sel, 5].sum()))
        out["arr"].append(sum(rates(t / 60)))
    return {k: np.array(v) for k, v in out.items()}


S = series()


def at(key, minute):
    return float(S[key][int(round(minute * 2))])


ALERT = next(m / 2 for m in range(len(GRID)) if S["ttft99"][m] > SLO)         # first 30 s with p99 over the SLO
FULL = next(m / 2 for m in range(len(GRID)) if S["kv"][m] > 98)               # the cache is full
FIRST_PRE = next(m / 2 for m in range(len(GRID)) if S["pre"][m] > 0 and m / 2 > RAMP[0])
BACK = next(m / 2 for m in range(len(GRID)) if m / 2 > LEVERS_AT and S["ttft99"][m] < SLO)
AFTER = [m for m in range(len(GRID)) if m / 2 >= BACK]
REJ = RUN["rejected"]
ARRIVED_AFTER = sum(1 for j in RUN["jobs"] if j.req.arrival >= LEVERS_AT * 60)
REJ_SHARE = len(REJ) / ARRIVED_AFTER
SHED_N = len(RUN["shed"])

ALERT_FIRES = ALERT + 1.0                  # the rule: p99 over the target for a whole minute
CALM_KV = float(np.mean([at("kv", m / 2) for m in range(0, 21)]))
PRE_SAT = float(np.mean([at("pre", m / 2) for m in range(30, 41)]))

# the narration quotes these: if the run changes, the scene refuses to render
assert ALERT == 14.0 and ALERT_FIRES == 15.0 and FULL == 14.0 and FIRST_PRE == 13.0      # "a quarter past eight"
assert 40 < CALM_KV < 55 and max(at("wait", m / 2) for m in range(0, 21)) < 1          # "about half full, nobody waits"
assert round(EXAM[0] / CALM[0] - 1, 1) == 0.2 and EXAM[1] / CALM[1] == 4                # "a fifth more … four times"
assert at("kv", FULL) > 98 and 90 < PRE_SAT < 115                                        # "full", "about a hundred a minute"
assert at("wait", 16) > 400 and at("ttft99", 16) > 20                                    # "past four hundred"
assert abs(at("gap50", 16) - FULL_GAP * 1e3) < 0.5 and round(FULL_GAP * 1e3) == 29      # "about twenty-nine"
assert round(at("gap50", 5)) == 17 and round(at("gap99", 16), 1) == round(CHUNK_GAP * 1e3, 1)
TOK_SAT = float(np.mean([at("tok", k / 2) for k in range(30, 41)]))
assert all(4300 < at("tok", k / 2) < 5700 for k in range(0, 41)) and round(TOK_SAT, -3) == 5000   # "about five thousand, flat"
assert round(N_FLIGHT, -1) == 160 and round(W_STAY) == 12 and round(MU) == 14             # Little's law, in numbers
assert abs(N_FLIGHT / W_STAY - MU) / MU < 0.03                                            # L = λW holds in the run
assert round(LAM, 1) == 17.4 and round(LAM - MU, 1) == 3.5                                # "three and a half more"
assert round(DECK_REQ * 100) == 2 and 0.18 < DECK_SHARE < 0.22                            # "two in a hundred … a fifth"
assert Q_MAX == 14 and BACK == 21.5 and all(S["ttft99"][m] < SLO for m in AFTER)          # "by twenty-two past"
assert all(at("kv", k / 2) > 84 for k in AFTER)                                           # the cache stays nearly full
assert round(REJ_SHARE * 100) == 6 and round(SHED_N, -2) == 200                           # "six in a hundred", "two hundred"
assert round(1 - 0.99 ** 30, 2) == 0.26                                                   # "one student in four"

# ---------------------------------------------------------------------------- measured on this Mac (§10)

SRV = measured("server", "llama-3.1-8b__4bit")
CLIENTS = ("1", "4", "16")
AGG = [SRV[c]["aggregate_tok_s"] for c in CLIENTS]
TTFT99 = [SRV[c]["ttft_ms"]["p99"] for c in CLIENTS]
ITL50 = [SRV[c]["itl_ms"]["p50"] for c in CLIENTS]
W4_GB = measured("decode", "llama-3.1-8b", "4bit", "weights_gb")
FLOOR_MAC = W4_GB * 1e9 / MAC["bandwidth"] * 1e3
assert [round(a) for a in AGG] == [90, 202, 290]                                   # "ninety … two hundred and ninety"
assert round(TTFT99[0]) == 74 and round(TTFT99[2] / 1e3, 1) == 1.2                 # "seventy-four ms … one point two s"
assert round(ITL50[0], 1) == 10.6 and round(ITL50[2]) == 47 and round(FLOOR_MAC, 1) == 8.3
assert SRV["16"]["retries_total"] == 2 and SRV["1"]["retries_total"] == 0

# ---------------------------------------------------------------------------- look

FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPRSTUVWXYZ0123456789")
GPU_ICON = recolored("gpu", MUTED, CARD_HI, prefix="ep25")
MAC_ICON = recolored("mac", MUTED, CARD_HI, prefix="ep25")
CHAT_KV, CHAT_KV_FILL = "#7E68C9", "#2E2450"          # chats' blocks: a quieter violet; decks' blocks: full VIOLET


def cite(s, size=20):
    return text(s, size, MUTED)


def baseline(words, s):
    chars = [c for c in s if not c.isspace()]
    if len(words) == len(chars):
        bottoms = sorted(g.get_bottom()[1] for g, c in zip(words, chars) if c in FLAT)
        if bottoms:
            return bottoms[len(bottoms) // 2]
    return float(np.median([g.get_bottom()[1] for g in words]))


def sit(words, s, y):
    """Put the baseline of Text `words` (made from s) at height y."""
    return words.shift((y - baseline(words, s)) * m.UP)


def mono(s, size=20, color=INK, rules=None, **kw):
    return code(s, size, color, t2c={re.escape(k): v for k, v in (rules or {}).items()}, **kw)


def clock_label(minute):
    return f"20:{int(minute):02d}"


# ---------------------------------------------------------------------------- the dashboard

PW, PH = 4.25, 2.85
COLS, ROWS = (-4.45, 0.0, 4.45), (1.1, -1.95)
LOG_LO, LOG_HI = 0.01, 100.0


class Panel:
    """One dashboard panel: a title, a big current value, a small legend, and a chart of the evening."""

    def __init__(self, title, legend, col, row, lines, ymax=None, ticks=(), log=False, value=None, sub=""):
        cx, cy = COLS[col], ROWS[row]
        self.lines, self.ymax, self.log, self.value, self.sub_s = lines, ymax, log, value, sub
        L, R, T, B = cx - PW / 2, cx + PW / 2, cy + PH / 2, cy - PH / 2
        self.L, self.R, self.T, self.B = L, R, T, B
        self.box = card(PW, PH, EDGE, CARD, radius=0.16, stroke=2).move_to([cx, cy, 0])
        self.title = text(title, 22, INK, weight=m.BOLD).align_to([L + 0.25, 0, 0], m.LEFT)
        sit(self.title, title, T - 0.45)
        self.legend = legend.align_to([L + 0.25, 0, 0], m.LEFT)
        sit(self.legend, legend.text if hasattr(legend, "text") else "x", T - 0.8)
        self.x0, self.x1, self.y0, self.y1 = L + 0.8, R - 0.45, B + 0.45, T - 1.05
        axis = m.VGroup(m.Line([self.x0, self.y0, 0], [self.x1 + 0.1, self.y0, 0]),
                        m.Line([self.x0, self.y0, 0], [self.x0, self.y1 + 0.08, 0])).set_stroke(FAINT, 2)
        grid = m.VGroup(*[m.Line([self.x0, self.Y(v), 0], [self.x1, self.Y(v), 0]).set_stroke(FAINT, 1.2, 0.5) for v, _ in ticks])
        ylabs = m.VGroup(*[code(s, 20, MUTED).move_to([0, self.Y(v), 0]).align_to([self.x0 - 0.1, 0, 0], m.RIGHT) for v, s in ticks])
        xlabs = m.VGroup()
        for mm in (0, 15, 30):
            xlabs.add(code(clock_label(mm), 20, MUTED).move_to([self.X(mm), self.y0 - 0.24, 0]))
        self.frame = m.VGroup(axis, grid, ylabs, xlabs)
        self.extra = m.VGroup()
        self.curves = [self.curve(key, 0.5, color, width, opacity) for key, color, width, opacity in lines]
        self.readout = self.make_readout(0.5)
        self.subtag = code(sub, 20, MUTED).align_to([R - 0.25, 0, 0], m.RIGHT) if sub else m.VGroup()
        if sub:
            sit(self.subtag, sub, T - 0.8)

    def X(self, minute):
        return self.x0 + (self.x1 - self.x0) * minute / SPAN

    def Y(self, v):
        if self.log:
            f = (np.log10(np.clip(v, LOG_LO, LOG_HI)) - np.log10(LOG_LO)) / (np.log10(LOG_HI) - np.log10(LOG_LO))
        else:
            f = min(max(v, 0.0), self.ymax) / self.ymax
        return self.y0 + (self.y1 - self.y0) * f

    def points(self, key, upto):
        data = S[key]
        n = int(np.floor(upto * 2))
        pts = [[self.X(k / 2), self.Y(data[k]), 0] for k in range(n + 1)]
        if upto * 2 > n + 1e-9 and n + 1 < len(data):
            f = upto * 2 - n
            v = data[n] + (data[n + 1] - data[n]) * f if not self.log else 10 ** (np.log10(max(data[n], LOG_LO)) * (1 - f) + np.log10(max(data[n + 1], LOG_LO)) * f)
            pts.append([self.X(upto), self.Y(v), 0])
        return pts

    def curve(self, key, upto, color, width, opacity):
        return m.VMobject().set_points_as_corners(self.points(key, upto)).set_stroke(color, width, opacity).set_z_index(2)

    def make_readout(self, minute, color=None):
        key, fmt, unit, c = self.value
        v = title_text(fmt(at(key, minute)), 30, color or c)
        u = code(unit, 24, MUTED).next_to(v, m.RIGHT, buff=0.1).align_to(v, m.DOWN)
        group = m.VGroup(v, u)
        group.align_to([self.R - 0.25, 0, 0], m.RIGHT)
        return group.shift((self.T - 0.45 - v.get_bottom()[1]) * m.UP)

    def group(self):
        return m.VGroup(self.box, self.title, self.legend, self.subtag, self.frame, *self.curves, self.readout, self.extra)

    def grow(self, t0, t1, run_time):
        anims = []
        for crv, (key, color, width, opacity) in zip(self.curves, self.lines):
            def update(mob, a, key=key, color=color, width=width, opacity=opacity):
                mob.become(self.curve(key, t0 + (t1 - t0) * a, color, width, opacity))
            anims.append(m.UpdateFromAlphaFunc(crv, update, run_time=run_time, rate_func=m.linear))
        return anims

    def to_value(self, minute, color=None):
        return m.Transform(self.readout, self.make_readout(minute, color))


def legend_pair(acr, c50, c99):
    """'TTFT  p50  p99' with each percentile in its line's colour."""
    s = f"{acr} · p50 · p99"
    words = mono(s, 20, MUTED, rules={"p50": c50, "p99": c99})
    words.text = s
    return words


def dim(color, f=0.55):
    """A colour mixed toward the card, for the p50 lines."""
    a, b = m.ManimColor(color).to_rgb(), m.ManimColor(CARD).to_rgb()
    return m.ManimColor.from_rgb(a * f + b * (1 - f)).to_hex()


class Dashboard:
    def __init__(self, upto):
        A50, C50 = dim(AMBER), dim(CYAN)
        s1 = lambda v: f"{v:.2f}" if v < 1 else f"{v:.1f}"
        self.ttft = Panel("Time to first token", legend_pair("TTFT", A50, AMBER), 0, 0,
                          [("ttft50", A50, 2.5, 1), ("ttft99", AMBER, 3.5, 1)], log=True,
                          ticks=((0.1, "0.1"), (10, "10")), value=("ttft99", s1, "s", AMBER), sub="p99")
        self.gap = Panel("Between tokens", legend_pair("ITL", C50, CYAN), 1, 0,
                         [("gap50", C50, 2.5, 1), ("gap99", CYAN, 3.5, 1)], ymax=80,
                         ticks=((25, "25"), (50, "50"), (75, "75")), value=("gap50", lambda v: f"{v:.1f}", "ms", CYAN), sub="p50")
        tok_leg = text("tokens per second, all", 20, MUTED)
        self.tok = Panel("Throughput", tok_leg, 2, 0, [("tok", TOKEN, 3.5, 1)], ymax=8000,
                         ticks=((4000, "4k"), (8000, "8k")), value=("tok", lambda v: f"{v:,.0f}", "tok/s", TOKEN))
        self.wait = Panel("Requests waiting", text("in the queue", 20, MUTED), 0, 1, [("wait", REQUEST, 3.5, 1)], ymax=600,
                          ticks=((300, "300"), (600, "600")), value=("wait", lambda v: f"{v:.0f}", "", REQUEST))
        self.kv = Panel("KV cache usage", text("share of blocks in use", 20, MUTED), 1, 1, [("kv", KV, 3.5, 1)], ymax=100,
                        ticks=((50, "50"), (100, "100")), value=("kv", lambda v: f"{v:.0f}", "%", KV))
        self.pre = Panel("Preemptions", text("per minute", 20, MUTED), 2, 1, [("pre", RED, 3.5, 1)], ymax=150,
                         ticks=((50, "50"), (100, "100")), value=("pre", lambda v: f"{v:.0f}", "/min", RED))
        self.panels = [self.ttft, self.gap, self.tok, self.wait, self.kv, self.pre]
        for p in self.panels:
            for crv, (key, color, width, opacity) in zip(p.curves, p.lines):
                crv.become(p.curve(key, upto, color, width, opacity))
            p.readout.become(p.make_readout(upto))
        # the target line on the TTFT chart
        y = self.ttft.Y(SLO)
        self.slo = m.DashedLine([self.ttft.x0, y, 0], [self.ttft.x1, y, 0], dash_length=0.09).set_stroke(RED, 2.2)
        self.slo_lab = code("2", 20, RED, weight=m.BOLD).move_to([0, y, 0]).align_to([self.ttft.x0 - 0.1, 0, 0], m.RIGHT)

    def grow(self, t0, t1, run_time, panels=None):
        anims = []
        for p in panels or self.panels:
            anims += p.grow(t0, t1, run_time)
        return m.AnimationGroup(*anims)

    def values(self, minute, delay=0.0, panels=None):
        anims = [p.to_value(minute) for p in (panels or self.panels)]
        return m.Succession(m.Wait(delay), m.AnimationGroup(*anims, run_time=0.5)) if delay else m.AnimationGroup(*anims, run_time=0.5)

    def all(self):
        return m.VGroup(*[p.group() for p in self.panels], self.slo, self.slo_lab)


def clock_pill(minute, color=MUTED, fill=CARD, extra=None, stamp=True):
    s = f"{clock_label(minute)} · {sum(rates(minute)):.1f} requests/s" if extra is None else f"{clock_label(minute)} · {extra}"
    if not stamp:
        s = extra
    p = pill(s, color, fill, 22, mono=True)
    return p.move_to([0, 3.19, 0]).align_to([6.6, 0, 0], m.RIGHT)


def advance(dash, clock, t0, t1, run_time, clock_extra=None, color=MUTED, fill=CARD):
    """The replay: every curve grows from t0 to t1 on one clock; the readouts and the clock land at t1."""
    return m.AnimationGroup(
        dash.grow(t0, t1, run_time),
        m.Succession(m.Wait(max(run_time - 0.5, 0)), m.AnimationGroup(
            *[p.to_value(t1) for p in dash.panels], m.Transform(clock, clock_pill(t1, color, fill, clock_extra)), run_time=0.5)),
    )


def student_grid(rows, cols, size, gap, color=REQUEST, fill=PINK_FILL, stroke=2.0):
    return m.VGroup(*[m.RoundedRectangle(corner_radius=size * 0.18, width=size, height=size)
                      .set_stroke(color, stroke).set_fill(fill, 1) for _ in range(rows * cols)]
                    ).arrange_in_grid(rows=rows, cols=cols, buff=gap)


def contents(p):
    return m.VGroup(p.title, p.legend, p.subtag, p.frame, *p.curves, p.readout)


# ---------------------------------------------------------------------------- the episode


class Ep25KeepingItAlive(Episode):
    number = 25

    def lesson(self):
        self.hook()
        self.watch()
        self.replay()
        self.diagnosis()
        self.levers()
        self.again()
        self.rehearse()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes=f"Exam week (illustrative). Owl-8B on one H100 serves the students. At {clock_label(ALERT_FIRES)} an alert fires: the p99 time to first token (TTFT) has been over its {SLO:.0f} s target for a whole minute (it first crossed at {clock_label(ALERT)}). The server is up and the GPU busy every step — nothing crashed. The whole evening is the simulation described in the module docstring.")
        grid = student_grid(6, 10, 0.3, 0.1).move_to([-4.1, -0.15, 0])
        who = text("exam week · cramming with Owl", 22, MUTED).next_to(grid, m.UP, buff=0.3)
        mascot = owl(0.55).next_to(grid, m.DOWN, buff=0.35)
        box = card(6.3, 2.75, RED, RED_FILL, radius=0.2, stroke=2.5).move_to([2.75, 0.35, 0])
        bx = box.get_left()[0] + 0.4
        warn = icon("alert", 0.5)
        tag = code("ALERT · owl-prod", 24, RED, weight=m.BOLD)
        row1 = m.VGroup(warn, tag).arrange(m.RIGHT, buff=0.25).align_to([bx, 0, 0], m.LEFT).set_y(1.2)
        what = title_text(f"TTFT p99 > {SLO:.0f} s", 52, INK).align_to([bx, 0, 0], m.LEFT).set_y(0.3)
        since = code(f"for 1 min · since {clock_label(ALERT_FIRES)}", 20, MUTED).align_to([bx, 0, 0], m.LEFT).set_y(-0.55)
        for w_ in (row1, what, since):
            w_.set_z_index(3)
        ok = m.VGroup(pill("✓ server up", GREEN, GREEN_FILL, 22), pill("✓ GPU busy", GREEN, GREEN_FILL, 22))
        ok.arrange(m.RIGHT, buff=0.35).next_to(box, m.DOWN, buff=0.35)
        ask = title_text("What to watch, and what to do?", 40, INK).move_to([1.6, -2.9, 0])

        self.say(
            "[It's exam week], [and students are cramming with Owl all evening.]",
            m.AnimationGroup(self.new_header("00 · hook", "The pager goes off"), m.FadeIn(who, shift=0.1 * m.UP)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(s, scale=0.6) for s in grid], lag_ratio=0.02), m.FadeIn(mascot, shift=0.15 * m.UP)),
        )
        self.say(
            "[At a quarter past eight], [an alert fires]: [the ninety-ninth percentile time to first token is over two seconds.]",
            m.AnimationGroup(m.FadeIn(box, shift=0.2 * m.LEFT), m.FadeIn(since, shift=0.2 * m.LEFT)),
            m.AnimationGroup(m.FadeIn(row1, shift=0.1 * m.UP), m.Flash(warn.get_center(), color=RED, line_length=0.2, flash_radius=0.45)),
            m.FadeIn(what, shift=0.1 * m.UP),
        )
        self.say(
            "[Nothing has crashed], [and the GPU is busy all the time.]",
            m.FadeIn(ok[0], shift=0.1 * m.UP),
            m.FadeIn(ok[1], shift=0.1 * m.UP),
        )
        self.say(
            "[So what should you watch], [and what do you do], [when the load spikes?]",
            m.FadeIn(ask, shift=0.15 * m.UP),
            m.Circumscribe(box, color=RED, buff=0.08),
            m.Succession(*[m.Indicate(grid, color=PINK, scale_factor=1.04, run_time=0.6) for _ in range(2)]),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ what to watch

    def watch(self):
        self.next_section("1 · What to watch", notes=f"Six panels, each a Prometheus metric in vLLM (Production Metrics docs): TTFT vllm:time_to_first_token_seconds; the gap between tokens vllm:inter_token_latency_seconds (ITL; TPOT is its per-request mean, vllm:request_time_per_output_token_seconds); generated tokens per second; vllm:num_requests_waiting (queue; vllm:request_queue_time_seconds for queue time); vllm:kv_cache_usage_perc (1 = 100%: the share of KV blocks in use, Episode 7); vllm:num_preemptions (Episode 10). Latency as percentiles over a trailing minute: p50 (median) and p99 (one request in a hundred is slower); Episode 11 showed how the mean hides hiccups. A student who asks 30 questions a day (Episode 24's example) meets at least one p99-slow answer with probability 1 - 0.99^30 = {1 - 0.99 ** 30:.2f}. The target is an SLO (Episode 12): here, for example, p99 TTFT <= {SLO:.0f} s — MLPerf's Llama-2-70B server scenario uses 'TTFT: <= 2 seconds' at the 99th percentile (notes §15; MLCommons, 'Llama 2 70B: An MLPerf Inference Benchmark', 2024). TTFT is on a log scale (0.01–100 s). Dashboard data: the simulated evening (illustrative), 30-s points, percentiles over the trailing minute.")
        dash = Dashboard(2.0)
        self.dash = dash
        boxes = m.VGroup(dash.ttft.box, dash.gap.box, dash.tok.box)
        ep11 = pill("Episode 11 · the average hides the hiccups", REQUEST, PINK_FILL, 20).move_to([0, -1.0, 0])
        call = card(8.8, 1.75, EDGE, CARD_HI, radius=0.18, stroke=2).move_to([0, -2.45, 0])
        c1 = text("a student asks 30 questions a day", 24, INK)
        c2 = mono("1 - 0.99**30 = 0.26", 22, MUTED, rules={"0.26": AMBER})
        cl = m.VGroup(c1, c2).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.2).align_to(call, m.LEFT).shift(0.45 * m.RIGHT)
        cl.set_y(call.get_y())
        big = title_text("26%", 64, AMBER)
        big_lab = text("meet the p99 at least once", 22, MUTED)
        cr = m.VGroup(big, big_lab).arrange(m.DOWN, buff=0.1).align_to(call, m.RIGHT).shift(0.45 * m.LEFT).set_y(call.get_y())
        callout = m.VGroup(call, cl, cr)
        slo_cite = cite("for example · MLPerf, Llama 2 70B server: p99 TTFT ≤ 2 s").move_to([0, 3.19, 0]).align_to([6.6, 0, 0], m.RIGHT)

        self.say(
            "[Here's Owl's dashboard for that evening.] [First, how the wait feels]: [the time to first token], [and the gap between tokens.]",
            m.AnimationGroup(self.new_header("01 · what to watch", "Six panels"), m.LaggedStart(*[m.FadeIn(b) for b in boxes], lag_ratio=0.12)),
            m.Circumscribe(m.VGroup(dash.ttft.box, dash.gap.box), color=MUTED, buff=0.06),
            m.FadeIn(contents(dash.ttft), shift=0.1 * m.UP),
            m.FadeIn(contents(dash.gap), shift=0.1 * m.UP),
        )
        self.say(
            "[As Episode eleven showed, watch percentiles, not the average]: [the median], [and the ninety-ninth percentile, the slowest request in a hundred.]",
            m.FadeIn(ep11, shift=0.1 * m.UP),
            m.Indicate(m.VGroup(glyphs(dash.ttft.legend, "p50"), glyphs(dash.gap.legend, "p50")), color=INK),
            m.Indicate(m.VGroup(glyphs(dash.ttft.legend, "p99"), glyphs(dash.gap.legend, "p99"), dash.ttft.curves[1], dash.gap.curves[1]), color=INK, scale_factor=1.05),
        )
        self.say(
            "[With thirty questions a day], [one student in four meets that slowest one in a hundred.]",
            m.AnimationGroup(m.FadeIn(call), m.FadeIn(cl, shift=0.1 * m.UP)),
            m.FadeIn(cr, shift=0.1 * m.UP),
        )
        self.say(
            "[Next, how full the server is]: [requests waiting in the queue], [and KV cache usage, Episode seven's share of blocks in use.]",
            m.AnimationGroup(m.FadeOut(callout), m.FadeOut(ep11)),
            m.AnimationGroup(m.FadeIn(dash.wait.box), m.FadeIn(contents(dash.wait), shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(dash.kv.box), m.FadeIn(contents(dash.kv), shift=0.1 * m.UP)),
        )
        self.say(
            "[And what gets done]: [tokens per second], [and preemptions, Episode ten's last resort when the cache runs out.]",
            m.Circumscribe(dash.tok.box, color=MUTED, buff=0.06),
            m.FadeIn(contents(dash.tok), shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(dash.pre.box), m.FadeIn(contents(dash.pre), shift=0.1 * m.UP)),
        )
        self.say(
            "[Each latency gets a target, as in Episode twelve]: [for example, a first token within two seconds, for ninety-nine requests in a hundred.]",
            m.FadeIn(slo_cite, shift=0.1 * m.DOWN),
            m.AnimationGroup(m.Create(dash.slo), m.FadeIn(dash.slo_lab)),
        )
        self.slo_cite = slo_cite
        self.wait(0.3)

    # ------------------------------------------------------------------ the replay

    def replay(self):
        d = self.dash
        self.next_section("2 · The replay", notes=f"The simulated evening (illustrative): 20:00 calm, {CALM[0]:.0f} chats + {CALM[1]} decks a second, the cache about {CALM_KV:.0f}% full, nobody waiting, gap p50 ≈ {at('gap50', 5):.0f} ms. 20:10–20:16 exam-week traffic ramps to {EXAM[0]:.0f} chats + {EXAM[1]} decks a second (+{EXAM[0] / CALM[0] - 1:.0%} questions, 4x decks). {clock_label(FIRST_PRE)}: first preemptions; {clock_label(FULL)}: KV {at('kv', FULL):.1f}% full; then ≈ {PRE_SAT:.0f} preemptions a minute. p99 TTFT crosses {SLO:.0f} s at {clock_label(ALERT)} ({at('ttft99', ALERT):.1f} s); the alert (rule: over target for 1 min) fires at {clock_label(ALERT_FIRES)}. By 20:16: {at('wait', 16):.0f} waiting, p99 TTFT {at('ttft99', 16):.1f} s (it plateaus near {GIVE_UP:.0f} s, when students give up). Tokens per second stay ≈ {TOK_SAT:,.0f} (calm {at('tok', 5):,.0f}); the p50 gap rises to its full-cache ceiling: (16.06 GB weights + {POOL / GB:.1f} GB of KV) ÷ (75% × 3.35 TB/s) = {FULL_GAP * 1e3:.1f} ms; the p99 gap ({at('gap99', 16):.1f} ms) is a step carrying a full 2,048-token prefill chunk (Episode 11's dial), mostly re-prefill of preempted requests.")
        clock = clock_pill(2.0)
        self.wall = clock
        self.say(
            "[Now press play.] [At eight o'clock, all is calm]: [the cache is about half full, and nobody waits.]",
            m.AnimationGroup(self.new_header("02 · the replay", "Exam week, live"), m.FadeOut(self.slo_cite), m.FadeIn(clock)),
            advance(d, clock, 2.0, 10.0, 3.6),
            m.Circumscribe(m.VGroup(d.wait.box, d.kv.box), color=MUTED, buff=0.06),
        )
        self.say(
            "[At ten past, exam-week traffic arrives]: [a fifth more questions, and four times as many flashcard decks.]",
            advance(d, clock, 10.0, 12.5, 2.6),
            m.Circumscribe(clock, color=REQUEST, buff=0.06),
        )
        self.say(
            "[KV usage climbs], [and by fourteen past, the cache is full.]",
            advance(d, clock, 12.5, 14.0, 2.4),
            m.Circumscribe(d.kv.box, color=KV, buff=0.06),
        )
        self.say(
            "[Now every new block means preempting someone]: [about a hundred preemptions a minute.]",
            advance(d, clock, 14.0, 15.0, 2.2),
            m.Circumscribe(d.pre.box, color=RED, buff=0.06),
        )
        red_box = card(PW, PH, RED, RED_FILL, radius=0.16, stroke=2.5).move_to(d.ttft.box)
        alert = clock_pill(16.0, RED, RED_FILL, extra="ALERT · TTFT p99 > 2 s", stamp=False)
        self.say(
            "[The queue grows], [the slowest first tokens shoot past two seconds], [and the alert fires.]",
            m.AnimationGroup(d.grow(15.0, 16.0, 3.0), m.Succession(m.Wait(2.5), d.values(16.0))),
            m.Circumscribe(d.ttft.box, color=AMBER, buff=0.06),
            m.AnimationGroup(m.Transform(d.ttft.box, red_box), m.Transform(clock, alert)),
        )
        self.say(
            "[Yet tokens per second barely move], [and the gap between tokens stops at about twenty-nine milliseconds], [the pace of a full cache.]",
            m.Circumscribe(d.tok.box, color=TOKEN, buff=0.06),
            m.Circumscribe(d.gap.box, color=CYAN, buff=0.06),
            m.Indicate(d.gap.readout, color=CYAN),
        )
        self.wait(0.3)

    # ------------------------------------------------------------------ diagnosis

    def diagnosis(self):
        d = self.dash
        self.next_section("3 · Diagnosis", notes=f"Read together: TTFT up while the gap between tokens is steady ⇒ the time is spent waiting (queue), not decoding; KV ≈ {at('kv', 16):.1f}% and ≈ {PRE_SAT:.0f} preemptions a minute while tokens/s stay ≈ {TOK_SAT:,.0f} ⇒ out of KV-cache memory, not out of compute (notes §16's worked alert: KV ≈ 0.98, preemptions rising, TPOT p50 flat). Inside (simulation, 20:15–20:20): ≈ {N_FLIGHT:.0f} requests hold cache at once; decks are {DECK_REQ:.1%} of requests but hold {DECK_SHARE:.0%} of the used blocks (sampled every 40 steps). The grid is a schematic: 160 squares, each ≈ 1/160 of the {BLOCKS:,} blocks; deck squares in proportion. Preemption (vLLM V1): when a running request needs a block and none is free, the most recently admitted is preempted, its blocks freed, and it is recomputed later — 'the default preemption mode is RECOMPUTE rather than SWAP' (vLLM docs); recompute = prompt + answer so far, wasted prefill. Little's law (Episode 24), measured in the run, 20:15–20:20: L ≈ {N_FLIGHT:.1f} in flight, W ≈ {W_STAY:.2f} s from admission to the last token, L ÷ W = {N_FLIGHT / W_STAY:.2f}/s vs {MU:.2f}/s actually finished (≈). Arrivals λ = {LAM:.1f}/s ⇒ {LAM - MU:.1f} more per second than finish; the queue grows until students give up after {GIVE_UP:.0f} s (simulated give-up; vLLM aborts a request whose client disconnects). Queue shown: {at('wait', 16):.0f} waiting at 20:16.")
        self.say(
            "[Read the panels together.] [Slow first tokens, but steady gaps]: [the time goes to waiting, not to decoding.]",
            self.new_header("03 · diagnosis", "Out of cache, not math"),
            m.AnimationGroup(m.Circumscribe(d.ttft.box, color=AMBER, buff=0.06), m.Circumscribe(d.gap.box, color=CYAN, buff=0.06)),
            m.Circumscribe(d.wait.box, color=REQUEST, buff=0.06),
        )
        self.say(
            "[A full cache, and constant preemptions]: [the server is out of KV memory], [not out of math.]",
            m.AnimationGroup(m.Circumscribe(d.kv.box, color=KV, buff=0.06), m.Circumscribe(d.pre.box, color=RED, buff=0.06)),
            m.Indicate(d.kv.readout, color=KV),
            m.Circumscribe(d.tok.box, color=TOKEN, buff=0.06),
        )

        # inside: the queue and the cache, as a schematic
        rng = np.random.default_rng(7)
        cells = cell_grid(8, 20, size=0.22, gap=0.05, color=CHAT_KV, fill=CHAT_KV_FILL)
        flat = [c for row in cells for c in row]
        gpu_box = card(6.3, 3.95, EDGE, CARD, radius=0.2, stroke=2).move_to([-3.35, -0.25, 0])
        gpu_cap = m.VGroup(icon(GPU_ICON, 0.42), code(f"one H100 · {POOL / GB:.0f} GB of KV blocks", 20, MUTED)).arrange(m.RIGHT, buff=0.2)
        gpu_cap.align_to(gpu_box, m.UL).shift(0.3 * m.RIGHT + 0.25 * m.DOWN)
        cells.move_to([gpu_box.get_x(), -0.12, 0])
        held = text(f"about {round(N_FLIGHT, -1):.0f} requests share it", 22, KV).move_to([gpu_box.get_x(), 0, 0])
        sit(held, held.text, gpu_box.get_bottom()[1] + 0.3)
        n_deck = round(DECK_SHARE * len(flat))
        order = rng.permutation(len(flat))
        free_i = int(order[0])                                    # 99.5% full: one square left
        deck_i = [int(k) for k in order[1:1 + n_deck]]
        needy = int(order[1 + n_deck])
        flat[free_i].set_fill(CARD, 1).set_stroke(EDGE, 1.5)
        deck_on = [flat[k].copy().set_fill(VIOLET, 1).set_stroke(VIOLET, 1.5) for k in deck_i]
        queue_lab = text("waiting", 22, REQUEST, weight=m.BOLD).move_to([-6.0, 2.2, 0]).align_to([-6.45, 0, 0], m.LEFT)
        chips = m.VGroup(*[m.RoundedRectangle(corner_radius=0.06, width=0.3, height=0.3).set_stroke(REQUEST, 2).set_fill(PINK_FILL, 1)
                           for _ in range(7)]).arrange(m.RIGHT, buff=0.1)
        chips.next_to(queue_lab, m.RIGHT, buff=0.3)
        n_more = round(at('wait', 16)) - len(chips)
        more = code(f"+ {n_more} more", 20, REQUEST).next_to(chips, m.RIGHT, buff=0.2)
        back = m.RoundedRectangle(corner_radius=0.06, width=0.3, height=0.3).set_stroke(RED, 2.5).set_fill(RED_FILL, 1)
        back.move_to(chips[0])
        shifted = [c.get_center() + np.array([0.4, 0, 0]) for c in chips]
        more_to = more.copy().shift(0.4 * m.RIGHT)
        rec_lab = code("recompute", 20, RED).next_to(back, m.UP, buff=0.1)
        leg_chat = m.VGroup(m.Square(0.22).set_fill(CHAT_KV_FILL, 1).set_stroke(CHAT_KV, 1.5), text("chats", 20, MUTED)).arrange(m.RIGHT, buff=0.15)
        leg_deck = m.VGroup(m.Square(0.22).set_fill(VIOLET, 1).set_stroke(VIOLET, 1.5),
                            text(f"flashcard decks: {DECK_REQ * 100:.0f}% of requests, 1/5 of the cache", 20, MUTED)).arrange(m.RIGHT, buff=0.15)
        legend = m.VGroup(leg_chat, leg_deck).arrange(m.RIGHT, buff=0.4).next_to(gpu_box, m.DOWN, buff=0.25).align_to(gpu_box, m.LEFT)
        modes = m.VGroup(pill("RECOMPUTE · vLLM's default", RED, RED_FILL, 20, mono=True),
                         pill("or SWAP to CPU memory", FAINT, CARD, 20, mono=True)).arrange(m.RIGHT, buff=0.3)
        modes.next_to(legend, m.DOWN, buff=0.28).align_to(gpu_box, m.LEFT)

        # Little's law, measured in the run
        law = card(6.0, 4.55, EDGE, CARD, radius=0.2, stroke=2).move_to([3.5, -0.05, 0])
        lx = law.get_left()[0] + 0.4
        head = m.VGroup(title_text("L = λ × W", 44, INK), cite("Little's law · Episode 24")).arrange(m.RIGHT, buff=0.35, aligned_edge=m.DOWN)
        head.align_to([lx, 0, 0], m.LEFT).set_y(1.65)
        rows_spec = [
            ("in the cache", f"L ≈ {round(N_FLIGHT, -1):.0f} requests", KV),
            ("each stays", f"W ≈ {W_STAY:.0f} s", INK),
            ("they finish", f"L ÷ W ≈ {MU:.0f} per s", TOKEN),
            ("arriving", f"λ = {LAM:.1f} per s", REQUEST),
            ("left waiting", f"+ {LAM - MU:.1f} per s", RED),
        ]
        rows = []
        for k, (lab, val, col) in enumerate(rows_spec):
            y = 0.92 - 0.56 * k - (0.12 if k >= 3 else 0)
            a = text(lab, 22, MUTED).align_to([lx, 0, 0], m.LEFT)
            sit(a, lab, y)
            b = code(val, 22, col, weight=m.BOLD).align_to([lx + 2.15, 0, 0], m.LEFT)
            sit(b, val, y)
            rows.append(m.VGroup(a, b))
        rule = m.Line([lx, 0.92 - 0.56 * 2 - 0.3, 0], [law.get_right()[0] - 0.4, 0.92 - 0.56 * 2 - 0.3, 0]).set_stroke(EDGE, 2)
        give = text(f"… until students give up, after {GIVE_UP:.0f} s", 22, RED).align_to([lx, 0, 0], m.LEFT)
        sit(give, give.text, law.get_bottom()[1] + 0.35)

        inside = m.VGroup(gpu_box, gpu_cap, cells, queue_lab, chips, more, held)
        self.say(
            "[Inside, about a hundred and sixty requests share the full cache.] [Flashcard decks are only two requests in a hundred], [yet they hold a fifth of the cache.]",
            m.AnimationGroup(m.FadeOut(d.all()), m.FadeOut(self.wall), m.FadeIn(inside)),
            m.FadeIn(legend, shift=0.1 * m.UP),
            m.LaggedStart(*[m.FadeIn(c) for c in deck_on], lag_ratio=0.03),
        )
        victims = m.VGroup(deck_on[0], deck_on[1], deck_on[2])            # the newest request: a deck, three squares
        red_v = [deck_on[k].copy().set_fill(RED, 1).set_stroke(RED, 2) for k in (0, 1, 2)]
        free_v = [deck_on[k].copy().set_fill(CARD, 1).set_stroke(EDGE, 1.5) for k in (0, 1, 2)]
        grow_into = deck_on[0].copy().set_fill(CHAT_KV_FILL, 1).set_stroke(CHAT_KV, 1.5)
        self.say(
            "[When a growing answer needs one more block], [the newest request is preempted]: [its blocks are freed], [and it goes back to the queue, to be recomputed from scratch.]",
            m.Circumscribe(flat[needy], color=INK, buff=0.05),
            m.AnimationGroup(*[m.Transform(deck_on[k], red_v[k]) for k in (0, 1, 2)]),
            m.Succession(m.AnimationGroup(*[m.Transform(deck_on[k], free_v[k]) for k in (0, 1, 2)], run_time=0.6),
                         m.Transform(deck_on[0], grow_into, run_time=0.6)),
            m.AnimationGroup(*[c.animate.move_to(p) for c, p in zip(chips, shifted)], m.Transform(more, more_to),
                             m.FadeIn(back, target_position=victims), m.FadeIn(rec_lab)),
        )
        self.say(
            "[By default, vLLM recomputes it], [rather than swapping its cache out to CPU memory.]",
            m.FadeIn(modes[0], shift=0.1 * m.UP),
            m.FadeIn(modes[1], shift=0.1 * m.UP),
        )
        self.say(
            "[Episode twenty-four's Little's law gives the limit]: [a hundred and sixty requests, each staying about twelve seconds], [means about fourteen finish per second.]",
            m.AnimationGroup(m.FadeIn(law), m.FadeIn(head, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(rows[0], shift=0.1 * m.UP), m.FadeIn(rows[1], shift=0.1 * m.UP)),
            m.FadeIn(rows[2], shift=0.1 * m.UP),
        )
        self.say(
            "[Exam week sends seventeen point four a second]: [the other three and a half wait in the queue], [until students give up, after thirty seconds.]",
            m.AnimationGroup(m.Create(rule), m.FadeIn(rows[3], shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(rows[4], shift=0.1 * m.UP), m.Circumscribe(m.VGroup(back, chips, more), color=REQUEST, buff=0.08)),
            m.FadeIn(give, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the levers

    def levers(self):
        self.next_section("4 · What to do", notes=f"Five levers (notes §16 reliability patterns). 1 Admission control: a bounded queue, then HTTP 429 with Retry-After once queue time would break the SLO — here the queue is capped at {Q_MAX} requests ≈ one second of work at {MU:.1f}/s. 2 Cap max_tokens ({CAP:,} here): 'a runaway generation holds a batch slot and KV for minutes'. 3 Priority: interactive chats before background flashcard decks; decks are preempted first (vLLM has a priority scheduling policy). 4 Shed load: when the queue is full, drop the lowest-priority work first (here a queued deck makes room for a chat). 5 Graceful degradation: a smaller or quantized model (Owl-mini), a shorter context, shorter max_tokens, no speculative decoding at high batch (Episode 14). The replay simulates levers 1–4; degradation is not simulated.")
        specs = [
            ("1", "Admission control", "cap the queue; the rest hear “retry soon”", "429 · Retry-After", REQUEST, PINK_FILL),
            ("2", "Cap max tokens", "no runaway answer holds blocks for minutes", f"max_tokens = {CAP}", TOKEN, GREEN_FILL),
            ("3", "Priority", "a waiting chat goes before a background deck", "chat  >  deck", REQUEST, PINK_FILL),
            ("4", "Shed load", "when full, drop the lowest priority first", "drop: newest deck", RED, RED_FILL),
            ("5", "Degrade gracefully", "a smaller model, a shorter context", "Owl-mini · 4K context", WEIGHTS, BLUE_FILL),
        ]
        rows = []
        for k, (n, title, desc, setting, col, fill) in enumerate(specs):
            y = 2.05 - 1.1 * k
            box = card(12.9, 0.95, EDGE, CARD, radius=0.16, stroke=2).move_to([0, y, 0])
            num = title_text(n, 36, col).move_to([-5.95, y, 0])
            t = text(title, 26, INK, weight=m.BOLD).align_to([-5.4, 0, 0], m.LEFT)
            sit(t, title, y + 0.05)
            ds = text(desc, 20, MUTED).align_to([-5.4, 0, 0], m.LEFT)
            sit(ds, desc, y - 0.33)
            chip_ = pill(setting, col, fill, 22, mono=True).move_to([1.85, y, 0])
            rows.append((box, num, t, ds, chip_, y))
        hd = self.new_header("04 · what to do", "Five levers")

        # mini mechanisms, one per lever, in the right-hand zone
        zx = 5.35
        y1 = rows[0][5]
        gate = m.Line([zx + 0.35, y1 - 0.3, 0], [zx + 0.35, y1 + 0.3, 0]).set_stroke(RED, 5)
        dot = m.Dot([zx - 0.8, y1, 0], radius=0.11, color=REQUEST)
        no = code("429", 20, RED, weight=m.BOLD).next_to(gate, m.RIGHT, buff=0.15)
        y2 = rows[1][5]
        cap_x = zx + 0.6
        cap_line = m.Line([cap_x, y2 - 0.3, 0], [cap_x, y2 + 0.3, 0]).set_stroke(RED, 4)
        bar = m.Rectangle(width=cap_x - (zx - 0.9), height=0.24).set_stroke(width=0).set_fill(TOKEN, 1)
        bar.move_to([(zx - 0.9 + cap_x) / 2, y2, 0])
        y3 = rows[2][5]
        mk = lambda s, c, f: m.VGroup(card(0.95, 0.42, c, f, radius=0.1, stroke=2), code(s, 20, INK)).arrange(m.ORIGIN)
        deck_c = mk("deck", VIOLET, VIOLET_FILL).move_to([zx - 0.55, y3, 0])
        chat_c = mk("chat", REQUEST, PINK_FILL).move_to([zx + 0.55, y3, 0])
        y4 = rows[3][5]
        drop = mk("deck", VIOLET, VIOLET_FILL).move_to([zx, y4, 0])
        y5 = rows[4][5]
        big_owl = owl(0.52).move_to([zx - 0.6, y5 - 0.04, 0])
        small_owl = owl(0.36).move_to([zx + 0.75, y5 - 0.08, 0])
        to = code("→", 24, MUTED).move_to([zx + 0.12, y5, 0])

        def row_in(r):
            box, num, t, ds, chip_, _ = r
            return m.AnimationGroup(m.FadeIn(box), m.FadeIn(num, shift=0.1 * m.RIGHT), m.FadeIn(t, shift=0.1 * m.RIGHT), m.FadeIn(ds))

        self.say(
            "[So what do you do? First, admission control]: [cap the queue, and tell the rest to retry in a few seconds], [instead of letting them wait thirty.]",
            m.AnimationGroup(hd, row_in(rows[0])),
            m.AnimationGroup(m.FadeIn(rows[0][4], shift=0.1 * m.UP), m.FadeIn(gate), m.FadeIn(dot)),
            m.Succession(dot.animate(run_time=0.6).move_to([zx + 0.15, y1, 0]),
                         m.AnimationGroup(dot.animate(run_time=0.6).move_to([zx - 0.8, y1, 0]), m.FadeIn(no, scale=0.8))),
        )
        self.say(
            "[Second, cap max tokens], [so no runaway answer holds its blocks for minutes.]",
            m.AnimationGroup(row_in(rows[1]), m.FadeIn(rows[1][4], shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(cap_line), m.GrowFromEdge(bar, m.LEFT, run_time=1.4)),
        )
        self.say(
            "[Third, priority]: [a student waiting in a chat goes before a background flashcard deck.]",
            m.AnimationGroup(row_in(rows[2]), m.FadeIn(rows[2][4], shift=0.1 * m.UP), m.FadeIn(deck_c), m.FadeIn(chat_c)),
            m.AnimationGroup(chat_c.animate(path_arc=-1.2).move_to([zx - 0.55, y3, 0]),
                             deck_c.animate(path_arc=-1.2).move_to([zx + 0.55, y3, 0]), run_time=1.0),
        )
        self.say(
            "[Fourth, shed load]: [when the queue is full, drop the lowest priority work first.]",
            m.AnimationGroup(row_in(rows[3]), m.FadeIn(rows[3][4], shift=0.1 * m.UP), m.FadeIn(drop)),
            m.FadeOut(drop, shift=0.35 * m.DOWN),
        )
        self.say(
            "[And fifth, degrade gracefully]: [a smaller model, or a shorter context], [beats no answer at all.]",
            m.AnimationGroup(row_in(rows[4]), m.FadeIn(big_owl)),
            m.AnimationGroup(m.FadeIn(rows[4][4], shift=0.1 * m.UP), m.FadeIn(to), m.FadeIn(small_owl, target_position=big_owl)),
            m.Circumscribe(rows[4][0], color=WEIGHTS, buff=0.05),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the replay, with the levers

    def again(self):
        self.next_section("5 · The replay, again", notes=f"The same simulated evening, levers 1–4 pulled at {clock_label(LEVERS_AT)} (illustrative on-call response): queue capped at {Q_MAX}, max_tokens {CAP:,} for new requests, chats first and decks preempted first, a queued deck shed to make room for a chat. p99 TTFT (trailing minute) is back under {SLO:.0f} s from {clock_label(BACK)}:30 and stays {min(S['ttft99'][m_] for m_ in AFTER):.2f}–{max(S['ttft99'][m_] for m_ in AFTER):.2f} s to 20:32. The price, 20:20–20:32: {len(REJ):,} of {ARRIVED_AFTER:,} arrivals told to retry (HTTP 429, {REJ_SHARE:.1%}); {SHED_N} queued decks dropped; long decks cut at {CAP:,} tokens. KV usage stays {min(at('kv', m_ / 2) for m_ in AFTER):.0f}–{max(at('kv', m_ / 2) for m_ in AFTER):.0f}%, so the real fix is capacity: a replica scaled on queue length (GKE guide: 'Use queue size to maximize throughput and minimize cost within a certain target latency threshold'; Episode 22's autoscaler) or FP8 KV for twice the blocks (Episode 6; notes §3: 52 → 104 conversations per H100).")
        d = Dashboard(16.0)
        red_box = card(PW, PH, RED, RED_FILL, radius=0.16, stroke=2.5).move_to(d.ttft.box)
        d.ttft.box.become(red_box)
        clock = clock_pill(16.0, RED, RED_FILL, extra="ALERT · TTFT p99 > 2 s", stamp=False)
        marks = m.VGroup(*[m.DashedLine([p.X(LEVERS_AT), p.y0, 0], [p.X(LEVERS_AT), p.y1 + 0.05, 0], dash_length=0.07).set_stroke(GREEN, 2.2)
                           for p in d.panels]).set_z_index(1)
        calm_box = card(PW, PH, EDGE, CARD, radius=0.16, stroke=2).move_to(d.ttft.box)
        self.say(
            "[Back to the dashboard.] [At twenty past, the on-call engineer pulls the first four levers.]",
            m.AnimationGroup(self.new_header("05 · the replay, again", "Levers at 20:20"), m.FadeIn(d.all()), m.FadeIn(clock)),
            m.Succession(m.AnimationGroup(d.grow(16.0, 20.0, 2.2), m.Succession(m.Wait(1.7), d.values(20.0))),
                         m.AnimationGroup(m.Create(marks), m.Transform(clock, clock_pill(20.0, GREEN, GREEN_FILL, extra="levers pulled")))),
        )
        self.say(
            "[By twenty-two past, the queue has drained], [and the ninety-ninth percentile is back under two seconds.]",
            advance(d, clock, 20.0, 22.0, 2.4, clock_extra="queue drained", color=GREEN, fill=GREEN_FILL),
            m.AnimationGroup(m.Transform(d.ttft.box, calm_box), m.Circumscribe(d.ttft.box, color=GREEN, buff=0.06)),
        )
        self.say(
            "[It stays there for the rest of the evening], [with the cache still nearly full.]",
            advance(d, clock, 22.0, 32.0, 3.2),
            m.Circumscribe(d.kv.box, color=KV, buff=0.06),
        )
        # the price, and the real fix
        ledger = card(8.6, 2.75, EDGE, CARD, radius=0.2, stroke=2).move_to([-2.0, 0.95, 0])
        lx = ledger.get_left()[0] + 0.4
        lt = text("the price, 20:20 to 20:32", 22, MUTED).align_to([lx, 0, 0], m.LEFT)
        sit(lt, lt.text, 1.9)
        spec = [
            ("told to retry · HTTP 429", f"{len(REJ):,} of {ARRIVED_AFTER:,}", f"{REJ_SHARE * 100:.0f}%", REQUEST),
            ("decks dropped · shed", f"{SHED_N}", "", RED),
            ("long decks cut at", f"{CAP:,} tokens", "", TOKEN),
        ]
        lrows = []
        for k, (a_s, b_s, c_s, col) in enumerate(spec):
            y = 1.25 - 0.62 * k
            a = text(a_s, 24, INK).align_to([lx, 0, 0], m.LEFT)
            sit(a, a_s, y)
            b = code(b_s, 24, col, weight=m.BOLD).align_to([lx + 4.35, 0, 0], m.LEFT)
            sit(b, b_s, y)
            g = m.VGroup(a, b)
            if c_s:
                c = title_text(c_s, 34, col).next_to(b, m.RIGHT, buff=0.3)
                c.shift((b.get_bottom()[1] - c.get_bottom()[1]) * m.UP)
                g.add(c)
            lrows.append(g)
        fixes = m.VGroup(
            pill("scale out on queue length · Episode 22", MEMORY, CYAN_FILL, 22),
            pill("FP8 KV cache: twice the blocks · Episode 6", KV, VIOLET_FILL, 22),
        ).arrange(m.DOWN, buff=0.3, aligned_edge=m.LEFT).next_to(ledger, m.DOWN, buff=1.0).align_to(ledger, m.LEFT)
        fix_lab = text("the real fix: more cache", 24, INK, weight=m.BOLD).next_to(fixes, m.UP, buff=0.2).align_to(fixes, m.LEFT)
        fix_lab.shift(0.0 * m.UP)
        kv_card = m.VGroup(card(3.6, 2.75, EDGE, CARD, radius=0.2, stroke=2).move_to([4.75, 0.95, 0]))
        kv_big = title_text(f"{min(at('kv', m_ / 2) for m_ in AFTER):.0f}–{max(at('kv', m_ / 2) for m_ in AFTER):.0f}%", 54, KV)
        kv_lab = m.VGroup(text("KV cache usage", 22, MUTED), text("after the levers", 22, MUTED)).arrange(m.DOWN, buff=0.1)
        m.VGroup(kv_big, kv_lab).arrange(m.DOWN, buff=0.25).move_to(kv_card)
        kv_card.add(kv_big, kv_lab)
        self.say(
            "[The price]: [six requests in a hundred were told to retry], [about two hundred decks were dropped], [and long decks were cut at about a thousand tokens.]",
            m.AnimationGroup(m.FadeOut(d.all()), m.FadeOut(marks), m.FadeOut(clock), m.FadeIn(ledger), m.FadeIn(lt)),
            m.FadeIn(lrows[0], shift=0.1 * m.UP),
            m.FadeIn(lrows[1], shift=0.1 * m.UP),
            m.FadeIn(lrows[2], shift=0.1 * m.UP),
        )
        self.say(
            "[So the real fix is more cache]: [scale out on queue length, as in Episode twenty-two], [or fit twice the blocks with Episode six's FP8.]",
            m.AnimationGroup(m.FadeIn(kv_card, shift=0.1 * m.LEFT), m.FadeIn(fix_lab, shift=0.1 * m.UP)),
            m.FadeIn(fixes[0], shift=0.1 * m.UP),
            m.FadeIn(fixes[1], shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ rehearse: benchmark before exam week

    def rehearse(self):
        self.next_section("6 · Rehearse", notes=f"Measured on this Mac (research/measurements.md §10, 2026-10-24, mlx-lm 0.32.2): mlx_lm.server --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit --temp 0 --max-tokens 128; tools/stream_client.py streams /v1/chat/completions (SSE) and timestamps every chunk; closed loop, C clients each sending R requests back to back (1×10, 4×5, 16×3), ~64-token prompts, 128-token answers, one warm-up pair; TTFT from the first connection attempt. Aggregate tok/s {AGG[0]:.0f} / {AGG[1]:.0f} / {AGG[2]:.0f}; ITL p50 {ITL50[0]:.1f} / {ITL50[1]:.1f} / {ITL50[2]:.1f} ms; TTFT p99 {TTFT99[0]:.0f} / {TTFT99[1]:.0f} / {TTFT99[2]:,.0f} ms. One client's gap vs the ceiling: {W4_GB:.2f} GB ÷ 546 GB/s = {FLOOR_MAC:.1f} ms per token. A first run 'lost requests to connection resets at 15–16 simultaneous connects' (ThreadingHTTPServer, listen backlog 5); the final client staggers starts by 20 ms and retries ({SRV['16']['retries_total']} retries in 48 requests at 16). Rules (notes §15): warm up; realistic input/output length distributions; open-loop Poisson arrivals for latency SLOs, closed-loop max concurrency for capacity; report p50/p90/p99, not means; goodput at the SLO (Episode 12); fix the prefix-cache state. Tools: vllm bench serve (TTFT/TPOT/ITL/E2EL percentiles, --request-rate, --goodput), GuideLLM, NVIDIA AIPerf.")
        # left: the load generator and the server
        gen = card(2.25, 2.7, REQUEST, PINK_FILL, radius=0.2, stroke=2).move_to([-5.3, 0.75, 0])
        dots = m.VGroup(*[m.Dot(radius=0.09, color=REQUEST) for _ in range(16)]).arrange_in_grid(rows=4, cols=4, buff=0.2)
        dots.move_to(gen).shift(0.15 * m.DOWN)
        gen_lab = text("load generator", 22, INK, weight=m.BOLD).next_to(gen, m.UP, buff=0.18)
        mac_box = card(2.75, 2.7, EDGE, CARD, radius=0.2, stroke=2).move_to([-2.05, 0.75, 0])
        mac_body = m.VGroup(icon(MAC_ICON, 0.85), code("mlx_lm.server", 20, INK), text("Owl-8B · 4-bit", 20, MUTED)).arrange(m.DOWN, buff=0.16)
        mac_body.move_to(mac_box)
        mac_lab = text("this Mac", 22, INK, weight=m.BOLD).next_to(mac_box, m.UP, buff=0.18)
        wire = arrow([gen.get_right()[0] + 0.05, 0.75, 0], [mac_box.get_left()[0] - 0.05, 0.75, 0], REQUEST)
        how = m.VGroup(text("streamed requests, closed loop", 20, MUTED), text("64-token prompts · 128-token answers", 20, MUTED))
        how.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.1).next_to(m.VGroup(gen, mac_box), m.DOWN, buff=0.3).align_to(gen, m.LEFT)
        n_on = {1: dots[:1], 4: dots[:4], 16: dots}
        for d_ in dots:
            d_.set_opacity(0.0)

        # right: the results, measured
        x_n, x_bar, x_gap, x_tt = -0.05, 0.65, 4.0, 5.65
        hy = 2.15
        heads = m.VGroup(
            text("students", 20, MUTED).move_to([x_n, hy, 0]),
            text("tokens/s, all", 20, MUTED).move_to([x_bar + 0.95, hy, 0]),
            text("gap p50", 20, MUTED).move_to([x_gap, hy, 0]),
            text("first token p99", 20, MUTED).move_to([x_tt, hy, 0]),
        )
        for h in heads:
            sit(h, h.text, hy)
        ys = [1.45, 0.75, 0.05]
        nums = m.VGroup(*[title_text(c, 34, REQUEST).move_to([x_n, y, 0]) for c, y in zip(CLIENTS, ys)])
        bars, bar_vals = m.VGroup(), m.VGroup()
        for a, y in zip(AGG, ys):
            w = 1.6 * a / 300
            b = m.Rectangle(width=w, height=0.34).set_stroke(width=0).set_fill(TOKEN, 1).move_to([x_bar + w / 2, y, 0])
            bars.add(b)
            bar_vals.add(code(f"{a:.0f}", 22, TOKEN, weight=m.BOLD).next_to(b, m.RIGHT, buff=0.12))
        gaps = m.VGroup(*[mono(f"{g:.1f} ms", 22, MEMORY, weight=m.BOLD).move_to([x_gap, y, 0]) for g, y in zip(ITL50, ys)])
        tts = m.VGroup(*[mono(f"{t:.0f} ms" if t < 1000 else f"{t / 1000:.2f} s", 22, AMBER, weight=m.BOLD).move_to([x_tt, y, 0])
                         for t, y in zip(TTFT99, ys)])
        src = cite("measured on this Mac · research/measurements.md §10").move_to([0, -1.0, 0]).align_to([-0.35, 0, 0], m.LEFT)
        ceil_note = text(f"alone vs the ceiling: {W4_GB:.2f} GB ÷ 546 GB/s = {FLOOR_MAC:.1f} ms", 20, MEMORY).move_to([0, -0.58, 0]).align_to([-0.35, 0, 0], m.LEFT)
        reset = card(6.9, 0.95, RED, RED_FILL, radius=0.16, stroke=2).move_to([3.1, -1.85, 0])
        reset_t = m.VGroup(text("first run: connection resets", 24, RED, weight=m.BOLD),
                           text("at 15–16 simultaneous connects · listen backlog 5", 20, INK)).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.08)
        reset_t.move_to(reset).align_to(reset, m.LEFT).shift(0.35 * m.RIGHT)
        rules = card(13.0, 0.8, EDGE, CARD, radius=0.16, stroke=2).move_to([0, -2.95, 0])
        r_words = ["warm up", "realistic lengths", "Poisson arrivals", "p50 · p90 · p99", "goodput at the SLO"]
        r_items = m.VGroup(*[m.VGroup(code("✓", 22, GREEN, weight=m.BOLD), text(w, 22, INK)).arrange(m.RIGHT, buff=0.12) for w in r_words])
        r_items.arrange(m.RIGHT, buff=0.4).move_to(rules)
        if r_items.width > rules.width - 0.7:
            r_items.scale((rules.width - 0.7) / r_items.width)
        tools = cite("tools: vllm bench serve · GuideLLM · AIPerf").move_to([0, -2.05, 0]).align_to([-6.45, 0, 0], m.LEFT)

        self.say(
            "[Better still, find the limits before exam week], [with a load generator.]",
            m.AnimationGroup(self.new_header("06 · rehearse", "Benchmark before exam week"), m.FadeIn(mac_box), m.FadeIn(mac_body), m.FadeIn(mac_lab)),
            m.AnimationGroup(m.FadeIn(gen), m.FadeIn(gen_lab), m.GrowArrow(wire)),
        )
        self.say(
            "[A small client streamed requests to Episode twenty-three's MLX server on this Mac], [with one, four, then sixteen students at once.]",
            m.AnimationGroup(m.FadeIn(how, shift=0.1 * m.UP), pulse(wire, REQUEST)),
            m.Succession(m.AnimationGroup(n_on[1].animate.set_opacity(1), m.FadeIn(nums[0]), run_time=0.5),
                         m.AnimationGroup(n_on[4].animate.set_opacity(1), m.FadeIn(nums[1]), run_time=0.5),
                         m.AnimationGroup(n_on[16].animate.set_opacity(1), m.FadeIn(nums[2]), m.FadeIn(heads[0]), run_time=0.5)),
        )
        self.say(
            "[Together they got more done]: [from ninety to two hundred and ninety tokens per second.]",
            m.AnimationGroup(m.FadeIn(heads[1]), m.FadeIn(src)),
            m.LaggedStart(*[m.AnimationGroup(m.GrowFromEdge(b, m.LEFT), m.FadeIn(v)) for b, v in zip(bars, bar_vals)], lag_ratio=0.4),
        )
        self.say(
            "[Alone, a student got a token every ten point six milliseconds], [near the eight point three millisecond ceiling]; [with sixteen, every forty-seven.]",
            m.AnimationGroup(m.FadeIn(heads[2]), m.FadeIn(gaps[0], shift=0.1 * m.UP)),
            m.FadeIn(ceil_note, shift=0.1 * m.DOWN),
            m.AnimationGroup(m.FadeIn(gaps[1], shift=0.1 * m.UP), m.FadeIn(gaps[2], shift=0.1 * m.UP)),
        )
        self.say(
            "[And the slowest first token, the ninety-ninth percentile], [went from seventy-four milliseconds to one point two seconds.]",
            m.AnimationGroup(m.FadeIn(heads[3]), m.FadeIn(tts[0], shift=0.1 * m.UP)),
            m.LaggedStart(m.FadeIn(tts[1], shift=0.1 * m.UP), m.FadeIn(tts[2], shift=0.1 * m.UP), lag_ratio=0.5),
        )
        self.say(
            "[It also found a real limit]: [the first run lost requests to connection resets], [at fifteen or sixteen simultaneous connections.]",
            m.Circumscribe(nums[2], color=RED, buff=0.08),
            m.AnimationGroup(m.FadeIn(reset), m.FadeIn(reset_t[0], shift=0.1 * m.UP)),
            m.FadeIn(reset_t[1], shift=0.1 * m.UP),
        )
        self.say(
            "[So warm up first, and use realistic lengths and arrivals]; [report percentiles, and goodput at your target], [never just the mean.]",
            m.AnimationGroup(m.FadeIn(rules), m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.UP) for r in r_items[:3]], lag_ratio=0.3)),
            m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.UP) for r in r_items[3:]], lag_ratio=0.4),
            m.FadeIn(tools),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Watch the tail (p99 time to first token, the gap between tokens, the queue, KV usage, preemptions); protect the cache (admission control, max-tokens caps, priority, shedding, graceful degradation; then more capacity); rehearse the load (a load generator, realistic lengths and arrivals, percentiles and goodput, before the spike).")
        halo = glow(CYAN, radius=4.0)
        words = m.VGroup(*[title_text(w, 80, c) for w, c in (("Watch", AMBER), ("Protect", KV), ("Rehearse", TOKEN))])
        subs = m.VGroup(*[code(s, 22, MUTED) for s in ("the tail, the queue", "the cache", "the load, first")])
        slots = [max(w.width, s.width) for w, s in zip(words, subs)]
        x, gap = 0.0, 0.9
        for w, s, slot_w in zip(words, subs, slots):
            w.move_to([x + slot_w / 2, 0, 0])
            s.move_to([x + slot_w / 2, 0, 0])
            x += slot_w + gap
        on_baseline(words)
        line_y = min(w.get_bottom()[1] for w in words) - 0.45
        for s in subs:
            sit(s, s.text, line_y)
        mid = words[0][0].get_center()[1]
        dots = m.VGroup(*[m.Dot([(a.get_right()[0] + b.get_left()[0]) / 2, mid, 0], radius=0.07, color=MUTED) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, dots, subs).move_to(0.45 * m.UP)
        tagline = text("Keep the queue short, and even the slowest student gets an answer.", 30, MUTED).next_to(row, m.DOWN, buff=0.7)
        mascot = owl(0.5).next_to(tagline, m.DOWN, buff=0.45)

        self.say(
            "[To sum up]: [watch the tail], [protect the cache], [rehearse the load.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say("[Keep the queue short], [and even the slowest student gets an answer.]",
                 m.Write(tagline), m.FadeIn(mascot, shift=0.15 * m.UP))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline, mascot)], run_time=0.9)
