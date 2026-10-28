"""Tokens per Second · Episode 12 · Split the Work (Module 3 · Many Users at Once)

Launch day: ten thousand students open Owl at once, and every GPU in the fleet serves both kinds of work, so
every long prompt stalls the streams sharing its GPU. Episode 3's two jobs sit at opposite ends of the H100
roofline (prefill ≈ 1,500 FLOPs/byte, decode ≈ 60), and even Episode 11's chunks stretch a shared step from
12.0 to 16.6 ms. Disaggregation (DistServe, Splitwise, Mooncake) gives each job its own pool and ships the
KV cache in between: animated as two GPU pools, a prompt, its first token, the cache crossing a VIOLET arrow
and the answer streaming back. The price: Owl's 1,500-token prompt is 1,500 × 128 KiB = 187.5 MiB of cache
(0.44 ms over NVLink, 3.9 ms over 400 Gb/s InfiniBand, 63 ms over a 25 Gb/s network: longer than the 49 ms
prefill), hidden layer by layer (Splitwise); launch day ships 33 GB of cache a second. Goodput vs throughput
and SLOs, with DistServe's own example as an SLO-attainment sketch (1.6 → 3.3 requests/s per GPU, 2.1×; up
to 7.4× vs a chunked-prefill system). DeepSeek's production units (32-GPU prefill, 144-GPU decode), the
other systems, and the honest catch. Episodes 1, 3, 5, 10 and 11 are linked by number.

Every number on screen is computed below from owl_reference.py, or cited with its source, and asserted.

Sources (facts dated October 2026; research/notes.md, original papers re-read 2026-10-24):
- §5 disaggregation. DistServe (Zhong et al., OSDI 2024, arXiv 2401.09670): goodput = "the maximum request
  rate that can be served adhering to the SLO attainment goal (say, 90%)"; Fig. 1 (13B model, one A100,
  512-token inputs, 64-token outputs): a colocated system "is about 1.6 requests per second (rps)"; "per-GPU
  goodput of 5.6 rps for the prefill phase and 10 rps for decoding"; "by allocating 2 GPUs for prefill and 1
  GPU for decoding, we can effectively serve the model with an overall goodput of 10 rps, or equally 3.3 rps
  per GPU", "2.1x higher than existing systems". Headline: "7.4× more requests or 12.6× tighter SLO"; the 7.4×
  is against DeepSpeed-MII (which "supports chunked-prefill") on chatbot workloads, keeping > 90% of requests
  within the latency targets. Prefill "benefits from more intra-op parallelism" (splitting one prompt's
  operators across GPUs). Chatbot SLOs, OPT-13B (Table 1): TTFT 0.25 s, TPOT 0.1 s. Testbed: "The
  cross-node bandwidth is 25Gbps". Chunked prefill "does not eliminate" interference (§2.3).
- Splitwise (Patel et al., ISCA 2024, arXiv 2311.18677): "a compute-intensive prompt computation, and a
  memory-intensive token generation"; token generation "can be run on less compute-capable hardware" (A100s in
  Splitwise-HA) and "incurs almost no latency impact when power capping by over 50%"; the KV-cache transfer
  overlaps the prompt computation layer by layer and "only incurs 0.8% of E2E".
- Mooncake (Kimi, FAST 2025, arXiv 2407.00079): "a KVCache-centric disaggregated architecture that separates
  the prefill and decoding clusters". NVIDIA Dynamo (2025): "disaggregated serving"; SGLang: large-scale PD
  disaggregation (LMSYS 2025-05-05); llm-d: P/D disaggregation (notes §13).
- §9 / DeepSeek, "Day 6: DeepSeek-V3/R1 inference system overview" (Feb 2025, re-read 2026-10-24): prefill
  "Each deployment unit spans 4 nodes" (EP32), decode "spans 18 nodes" (EP144), 8 H800 GPUs per node; per node
  "~73.7k tokens/s input" in prefill and ~14.8k output tokens/s in decode.
- §10 links [DATED Oct 2026, vendor pages]: H100 NVLink 4 900 GB/s total per GPU (≈ 450 GB/s each way);
  InfiniBand CX7 400 Gb/s ≈ 50 GB/s per NIC (DeepEP README).
- §1 / owl_reference: Owl-8B (Llama-3.1-8B, BF16) 128 KiB of KV per token; prefill_time(1,500) at 50% MFU on an
  H100 SXM (989 TFLOP/s, 3.35 TB/s) ≈ 49 ms. Episode 11's model: 60 students at 2,048 tokens, a decode-only
  step 12.0 ms (80% MBU), a 512-token budgeted step 16.6 ms (50% MFU).

Illustrative (said "say" / "for example", or labelled): the 400-square crowd (25 students a square) and its
prefill/decode split, the fleet's lane patterns, "one question a minute" on launch day, the ten finished
answers of which three are late, and the SHAPES of the SLO-attainment curves (only their 90% crossings are
DistServe's numbers; the sketch is a logistic through them).

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from owl_reference import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- the numbers

KVT = kv_bytes_per_token()                       # 128 KiB per token (Episode 5)
PROMPT = 1500                                    # Owl's worked-example prompt (Episode 1)
KV_PROMPT = PROMPT * KVT
KV_MIB, KV_MB = KV_PROMPT / 2 ** 20, KV_PROMPT / 1e6
T_PRE = prefill_time(PROMPT)                     # 50% MFU on one H100
ms = lambda s: s * 1e3

NVLINK, IB, SLOW = 450e9, 400e9 / 8, 25e9 / 8    # bytes/s one way: NVLink 4, a 400 Gb/s NIC, DistServe's 25 Gb/s
T_NV, T_IB, T_SLOW = KV_PROMPT / NVLINK, KV_PROMPT / IB, KV_PROMPT / SLOW
assert KVT == 131_072 and KV_PROMPT == 196_608_000                       # "1,500 × 128 KiB"
assert KV_MIB == 187.5 and round(KV_MB) == 197                            # "187.5 MiB ≈ 197 MB", "about two hundred megabytes"
assert round(ms(T_PRE)) == 49 and round(ms(T_PRE), 1) == 48.7             # "forty-nine milliseconds"
assert round(ms(T_NV), 2) == 0.44 and ms(T_NV) < 0.5                       # "under half a millisecond"
assert round(ms(T_IB), 1) == 3.9 and T_IB < T_PRE / 10                     # "about four", "less than a tenth"
assert round(ms(T_SLOW)) == 63 and T_SLOW > T_PRE                          # "sixty-three", "longer than the prefill"

LAYERS = OWL["layers"]                           # 32: Splitwise ships each layer's cache as it is done
assert LAYERS == 32

STUDENTS, ASK_EVERY = 10_000, 60.0               # launch day; "say, one question a minute" (illustrative)
RATE = STUDENTS / ASK_EVERY
KV_FLOW = RATE * KV_PROMPT
LINK_SHARE = KV_FLOW / IB
assert round(RATE) == 167 and round(KV_FLOW / 1e9, 1) == 32.8 and round(KV_FLOW / 1e9) == 33   # "thirty-three gigabytes"
assert round(LINK_SHARE, 2) == 0.66                                        # "two thirds of one such link"

# Episode 11's shared GPU (notes §5 model): 60 students at 2,048 tokens, a 512-token budget
USERS, CTX, BUDGET, MFU, MBU = 60, 2048, 512, 0.5, 0.8
T_DEC = step_time(batch=USERS, context=CTX) / MBU
T_MIX = max(prefill_time(BUDGET, mfu=MFU), T_DEC)
assert round(ms(T_DEC), 1) == 12.0 and round(ms(T_MIX), 1) == 16.6        # "twelve", "sixteen point six"

# the roofline (Episode 3): intensity ≈ prompt length for prefill, ≈ batch for decode
I_PRE, I_DEC = PROMPT, USERS
assert I_DEC < ridge(GPU) < I_PRE and round(ridge(GPU)) == 295

# DistServe, Fig. 1 (cited): goodput at 90% SLO attainment, requests/s
SHARED, PRE_ONLY, DEC_ONLY, N_PRE, N_DEC = 1.6, 5.6, 10.0, 2, 1
SPLIT_TOTAL = min(N_PRE * PRE_ONLY, N_DEC * DEC_ONLY)
SPLIT_PER = SPLIT_TOTAL / (N_PRE + N_DEC)
GAIN = SPLIT_PER / SHARED
HEADLINE = 7.4
assert SPLIT_TOTAL == 10 and round(SPLIT_PER, 1) == 3.3 and round(GAIN, 1) == 2.1   # "ten on three", "3.3", "2.1×"
TTFT_SLO, TPOT_SLO = 250, 100                    # ms, DistServe Table 1, chatbot OPT-13B (cited)

# DeepSeek V3/R1 in production (cited)
NODE = 8
PRE_NODES, DEC_NODES = 4, 18
TOK_IN, TOK_OUT = 73.7e3, 14.8e3
assert PRE_NODES * NODE == 32 and DEC_NODES * NODE == 144                 # "thirty-two", "a hundred and forty-four"
assert round(TOK_IN, -3) == 74e3 and round(TOK_OUT, -3) == 15e3           # "about seventy-four thousand", "fifteen"

# the illustrative crowd: 400 squares, 25 students each
CROWD_ROWS, CROWD_COLS = 16, 25
PER_SQUARE = STUDENTS // (CROWD_ROWS * CROWD_COLS)
assert PER_SQUARE * CROWD_ROWS * CROWD_COLS == STUDENTS


def attainment(r, goodput, spread):
    """The sketched SLO-attainment curve: a logistic that crosses 90% exactly at `goodput`."""
    centre = goodput + spread * np.log(9)
    return 1 / (1 + np.exp((r - centre) / spread))


assert abs(attainment(SHARED, SHARED, 0.22) - 0.9) < 1e-9 and abs(attainment(SPLIT_PER, SPLIT_PER, 0.3) - 0.9) < 1e-9

# ---------------------------------------------------------------------------- look

FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPRSTUVWXYZ0123456789")
GPU_ICON = recolored("gpu", MUTED, CARD_HI, prefix="ep12")
GPU_PRE = recolored("gpu", AMBER, AMBER_FILL, prefix="ep12")
GPU_DEC = recolored("gpu", CYAN, CYAN_FILL, prefix="ep12")
USER_ICON = recolored("user", PINK, PINK_FILL, prefix="ep12")


def cite(s, size=20):
    return text(s, size, MUTED)


def baseline(words, s):
    """The baseline of a Text made from s: the median bottom of its flat-bottomed letters."""
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
    """code() with literal-substring colouring (rules: {substring: colour}, in priority order)."""
    return code(s, size, color, t2c={re.escape(k): v for k, v in (rules or {}).items()}, **kw)


def ep_pill(n, size=20):
    """'EP NN' in the accent of that episode's module."""
    _, accent, tint = MODULES[EPISODES[n][1]]
    return pill(f"EP {n:02d}", accent, tint, size, mono=True)


def student_grid(rows, cols, size, gap, color=REQUEST, fill=PINK_FILL, stroke=1.6):
    return m.VGroup(*[m.RoundedRectangle(corner_radius=size * 0.2, width=size, height=size)
                      .set_stroke(color, stroke).set_fill(fill, 1) for _ in range(rows * cols)]
                    ).arrange_in_grid(rows=rows, cols=cols, buff=gap)


def tinted(squares, color, fill, stroke=1.6):
    out = m.VGroup(*[s.copy() for s in squares])
    for s in out:
        s.set_fill(fill, 1).set_stroke(color, stroke)
    return out.set_z_index(2)


def roof_path(roof, a, b):
    """The path along the roof from intensity a to b (it turns the corner at the ridge)."""
    pts = [roof.point(a)]
    if a < roof.ridge_x < b:
        pts.append(roof.point(roof.ridge_x))
    pts.append(roof.point(b))
    return m.VMobject().set_points_as_corners(pts)


def lane(pattern, x0, x1, y, h=0.36):
    """A GPU's lane of steps: 'd' a short decode step (CYAN), 'P' a long prefill (AMBER)."""
    blocks, x = [], x0
    for c in pattern:
        w = 0.16 if c == "d" else 0.86
        if x + w > x1:
            break
        color = MEMORY if c == "d" else COMPUTE
        blocks.append(m.Rectangle(width=w, height=h).set_stroke(color, 1.5).set_fill(FILLS[color], 1)
                      .move_to([x + w / 2, y, 0]))
        x += w + 0.06
    return m.VGroup(*blocks)


def mixed_block(x0, w, y, h, prompt_tokens, users=USERS):
    """One budgeted step: a CYAN sliver for the decodes, AMBER for the prompt slice (by token count)."""
    share = users / (users + prompt_tokens)
    dec = m.Rectangle(width=w * share, height=h).set_stroke(MEMORY, 1.5).set_fill(CYAN_FILL, 1)
    dec.move_to([x0 + w * share / 2, y, 0])
    pre = m.Rectangle(width=w * (1 - share), height=h).set_stroke(COMPUTE, 1.5).set_fill(AMBER_FILL, 1)
    pre.move_to([x0 + w * share + w * (1 - share) / 2, y, 0])
    return m.VGroup(dec, pre)


def kv_block(width=0.62, rows=4):
    """A small KV cache: a few VIOLET layer slices."""
    return m.VGroup(*[m.RoundedRectangle(corner_radius=0.03, width=width, height=0.09).set_stroke(width=0)
                      .set_fill(KV, 1) for _ in range(rows)]).arrange(m.DOWN, buff=0.05)


def big_stat(value, unit, label, color, size=56):
    number = title_text(value, size, color)
    u = code(unit, 24, MUTED).next_to(number, m.RIGHT, buff=0.14).align_to(number, m.DOWN)
    lab = text(label, 20, MUTED).next_to(number, m.DOWN, buff=0.16).align_to(number, m.LEFT)
    return m.VGroup(number, u, lab)


def blink(overlay, on=0.12, off=0.22):
    return m.Succession(m.FadeIn(overlay, run_time=on), m.FadeOut(overlay, run_time=off))


class Pool:
    """A pool of GPUs: a card, a title, a mono subtitle, up to four GPU slots and a work meter."""

    def __init__(self, title, sub, color, fill, gpu, center, width=2.8):
        self.box = card(width, 3.1, color, CARD, radius=0.2, stroke=2.5).move_to(center)
        name = text(title, 24, color, weight=m.BOLD)
        tag = code(sub, 20, MUTED)
        self.head = m.VGroup(name, tag).arrange(m.DOWN, buff=0.1).move_to(self.box).align_to(self.box, m.UP).shift(0.22 * m.DOWN)
        self.slots = m.VGroup(*[icon(gpu, 0.5) for _ in range(4)]).arrange_in_grid(rows=2, cols=2, buff=(0.45, 0.22))
        self.slots.move_to(self.box).shift(0.12 * m.DOWN)
        self.track_w = width - 0.8
        self.track = m.RoundedRectangle(corner_radius=0.07, width=self.track_w, height=0.14).set_stroke(EDGE, 1.5).set_fill(CARD_HI, 1)
        self.track.move_to(self.box).align_to(self.box, m.DOWN).shift(0.28 * m.UP)
        self.color = color

    def bar(self, fraction=1.0):
        b = m.RoundedRectangle(corner_radius=0.07, width=self.track_w * fraction, height=0.14).set_stroke(width=0).set_fill(self.color, 1)
        return b.align_to(self.track, m.LEFT).set_y(self.track.get_y()).set_z_index(3)

    def shell(self):
        return m.VGroup(self.box, self.head, self.track)


# ---------------------------------------------------------------------------- the episode


class Ep12SplitTheWork(Episode):
    number = 12

    def lesson(self):
        self.hook()
        self.two_jobs()
        self.split()
        self.price()
        self.goodput()
        self.production()
        self.recap()

    # ------------------------------------------------------------------ hook: launch day

    def hook(self):
        self.next_section("Hook", notes=f"Launch day (the course's Module 3 story): {STUDENTS:,} students open Owl at once (one square = {PER_SQUARE} students; the prefill/decode split of the crowd and the fleet's lane patterns are illustrative). Every GPU serves both jobs, so a long prompt (AMBER prefill) stalls the decode streams (CYAN) sharing its GPU: Episode 11's decode stall, fleet-wide.")
        grid = student_grid(CROWD_ROWS, CROWD_COLS, 0.13, 0.055).move_to([-3.85, -0.5, 0])
        count = title_text(f"{STUDENTS:,}", 56, REQUEST)
        count_lab = text("students on launch day", 24, MUTED)
        count_row = m.VGroup(count, count_lab).arrange(m.RIGHT, buff=0.25, aligned_edge=m.DOWN)
        count_row.next_to(grid, m.UP, buff=0.32).align_to(grid, m.LEFT)
        per = text(f"each square = {PER_SQUARE} students", 20, MUTED).next_to(grid, m.DOWN, buff=0.22).align_to(grid, m.LEFT)

        rng = np.random.default_rng(12)
        pasting = set(rng.choice(len(grid), 56, replace=False).tolist())
        amber = tinted([s for i, s in enumerate(grid) if i in pasting], COMPUTE, AMBER_FILL)
        cyan = tinted([s for i, s in enumerate(grid) if i not in pasting], MEMORY, CYAN_FILL)
        key_p = m.VGroup(m.Square(0.2).set_stroke(COMPUTE, 1.6).set_fill(AMBER_FILL, 1), text("pasting a chapter: prefill", 20, COMPUTE))
        key_d = m.VGroup(m.Square(0.2).set_stroke(MEMORY, 1.6).set_fill(CYAN_FILL, 1), text("mid-answer: decode", 20, MEMORY))
        for k in (key_p, key_d):
            k.arrange(m.RIGHT, buff=0.15)
        keys = m.VGroup(key_p, key_d).arrange(m.RIGHT, buff=0.45).move_to(per).align_to(grid, m.LEFT)

        lanes_y = [0.95, -0.5, -1.95]
        fleet_lab = text("the fleet: every GPU does both", 22, MUTED).move_to([0, 1.85, 0]).align_to([0.35, 0, 0], m.LEFT)
        mascot = owl(0.42).move_to([6.05, 1.95, 0])
        gpus = m.VGroup(*[icon(GPU_ICON, 0.55).move_to([0.75, y, 0]) for y in lanes_y])
        tracks = m.VGroup(*[m.Rectangle(width=5.0, height=0.5).set_stroke(EDGE, 1.5).set_fill(CARD, 1).move_to([3.85, y, 0])
                            for y in lanes_y])
        patterns = ["ddddddPdddddddPdddd", "ddPdddddddddPddddd", "dddddddddPddddddPd"]
        lanes = [lane(p, 1.42, 6.28, y) for p, y in zip(patterns, lanes_y)]
        prefills = m.VGroup(*[b for ln in lanes for b in ln if b.get_width() > 0.5])
        stalls = m.VGroup(*[m.Rectangle(width=b.get_width() + 0.08, height=0.5).set_stroke(RED, 2.5).set_fill(RED, 0.18)
                            .move_to(b) for b in prefills]).set_z_index(3)
        stall_lab = text("decode streams wait", 22, RED, weight=m.BOLD).move_to([3.85, -2.75, 0])
        inbound = arrow([-1.45, -0.5, 0], [0.25, -0.5, 0], REQUEST)
        ask = title_text("What if prefill and decode ran on different machines?", 36, INK).move_to([0, -3.1, 0])

        self.say(
            "[It's launch day]: [ten thousand students open Owl at once.]",
            m.AnimationGroup(self.new_header("00 · launch day", "Ten thousand students, one fleet"), m.FadeIn(count_row, shift=0.15 * m.UP)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(s, scale=0.5) for s in grid], lag_ratio=0.004, run_time=2.0), m.FadeIn(per)),
        )
        self.say(
            "[Some paste a whole chapter]; [others are mid-answer], [waiting for their next word.]",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(s) for s in amber], lag_ratio=0.03, run_time=1.2), m.FadeOut(per), m.FadeIn(key_p)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(s) for s in cyan], lag_ratio=0.002, run_time=1.4), m.FadeIn(key_d)),
            m.Circumscribe(key_d, color=MEMORY, buff=0.08),
        )
        self.say(
            "[Every GPU in the fleet serves both kinds of work], [taking turns.]",
            m.AnimationGroup(m.FadeIn(fleet_lab), m.FadeIn(mascot, shift=0.15 * m.UP), m.FadeIn(gpus), m.FadeIn(tracks), m.GrowArrow(inbound)),
            m.LaggedStart(*[m.LaggedStart(*[m.GrowFromEdge(b, m.LEFT) for b in ln], lag_ratio=0.25) for ln in lanes], lag_ratio=0.15, run_time=2.4),
        )
        self.say(
            "[So every long prompt] [stalls the streams that share its GPU.]",
            m.LaggedStart(*[m.Indicate(b, color=RED, scale_factor=1.04) for b in prefills], lag_ratio=0.2),
            m.AnimationGroup(m.FadeIn(stalls), m.FadeIn(stall_lab, shift=0.1 * m.UP)),
        )
        self.say(
            "[What if prefill and decode] [ran on different machines?]",
            m.AnimationGroup(m.FadeOut(stall_lab), m.FadeOut(keys)),
            m.FadeIn(ask, shift=0.15 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 1 · two jobs that clash

    def two_jobs(self):
        self.next_section("1 · Two jobs", notes=f"Episode 3 on the H100 roofline (989 TFLOP/s dense BF16, 3.35 TB/s; ridge ≈ {ridge(GPU):.0f} FLOPs/byte): prefill's intensity ≈ prompt length ({I_PRE:,}, on the compute roof), decode's ≈ batch ({I_DEC} students, on the memory slope). Episode 11's shared GPU (notes §5 model): a decode-only step for {USERS} students at {CTX:,} tokens = {ms(T_DEC):.1f} ms (80% MBU); a step that also carries a prompt chunk (budget {BUDGET}) = {ms(T_MIX):.1f} ms (50% MFU), {T_MIX / T_DEC - 1:.0%} longer. DistServe: chunked prefill alleviates but 'does not eliminate' the interference; prefill 'benefits from more intra-op parallelism' (one prompt's operators split across GPUs) to cut TTFT; decode's best setup depends on its batch. Splitwise: token generation 'can be run on less compute-capable hardware' and 'incurs almost no latency impact when power capping by over 50%'.")
        roof = Roofline(GPU["bandwidth"], GPU["peak"], width=4.6, height=3.2, x_decades=(0, 4), y_decades=(12, 15.5),
                        center=(-3.5, -0.45, 0))
        cap = code("one H100 · Owl-8B", 20, MUTED).move_to([0, 1.95, 0]).align_to(roof.axes[1], m.LEFT)
        ep03 = ep_pill(3).next_to(cap, m.RIGHT, buff=0.3)
        ridge_lab = code(f"ridge ≈ {roof.ridge_x:.0f}", 20, MUTED)
        ridge_lab.next_to(roof.at(roof.ridge_x, 10 ** 12), m.UR, buff=0.12).shift(0.18 * m.UP)
        p_dot = roof.dot(1, COMPUTE)
        d_dot = roof.dot(1, MEMORY)
        p_lab = code(f"prefill · {I_PRE:,}", 20, COMPUTE, weight=m.BOLD)
        p_lab.next_to(roof.point(I_PRE), m.UP, buff=0.25)
        d_lab = code(f"decode · {I_DEC}", 20, MEMORY, weight=m.BOLD)
        d_lab.next_to(roof.point(I_DEC), m.UL, buff=0.08).shift(0.12 * m.UP)

        rx = 0.45
        share_title = text("sharing one GPU", 24, INK, weight=m.BOLD)
        share_row = m.VGroup(share_title, ep_pill(11)).arrange(m.RIGHT, buff=0.3).move_to([0, 2.0, 0]).align_to([rx, 0, 0], m.LEFT)
        k = 2.0 / ms(T_MIX)
        bx0 = 3.3
        r1y, r2y = 1.38, 0.78
        r1_lab = code("decodes only", 20, MEMORY).move_to([0, r1y, 0]).align_to([rx, 0, 0], m.LEFT)
        r2_lab = code("+ a prompt chunk", 20, COMPUTE).move_to([0, r2y, 0]).align_to([rx, 0, 0], m.LEFT)
        r1_blk = m.Rectangle(width=k * ms(T_DEC), height=0.38).set_stroke(MEMORY, 1.5).set_fill(CYAN_FILL, 1)
        r1_blk.move_to([bx0 + k * ms(T_DEC) / 2, r1y, 0])
        r2_blk = mixed_block(bx0, k * ms(T_MIX), r2y, 0.38, BUDGET - USERS)
        r1_val = code(f"{ms(T_DEC):.1f} ms", 20, INK).next_to(r1_blk, m.RIGHT, buff=0.18)
        r2_val = code(f"{ms(T_MIX):.1f} ms", 20, INK, weight=m.BOLD).next_to(r2_blk, m.RIGHT, buff=0.18)

        def want(title, color, sub, lines, y, h=1.3):
            box = card(6.15, h, color, CARD, radius=0.16, stroke=2).move_to([rx + 3.075, y, 0])
            t = text(title, 24, color, weight=m.BOLD)
            s = code(sub, 20, MUTED).next_to(t, m.RIGHT, buff=0.25)
            sit(s, sub, baseline(t, title))
            head = m.VGroup(t, s).align_to(box, m.UL).shift(0.3 * m.RIGHT + 0.2 * m.DOWN)
            body = m.VGroup(*[text(line, 20, INK) for line in lines]).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.1)
            body.next_to(head, m.DOWN, buff=0.16).align_to(head, m.LEFT)
            return m.VGroup(box, head, body)

        w_pre = want("prefill wants", COMPUTE, "compute-bound", ["raw math, even one prompt split across GPUs"], -0.5)
        w_dec = want("decode wants", MEMORY, "memory-bound", ["bandwidth, and memory for big batches",
                                                              "barely slowed by older or power-capped GPUs"], -2.12, 1.62)
        src = cite("DistServe, OSDI 2024 · Splitwise, ISCA 2024").next_to(w_dec[0], m.DOWN, buff=0.12).align_to(w_dec[0], m.LEFT)

        self.say(
            "[Episode three showed that one request is two different jobs.]",
            m.AnimationGroup(self.new_header("01 · two jobs", "Two jobs, one GPU, a clash"), m.FadeIn(cap), m.FadeIn(ep03, shift=0.1 * m.UP),
                             m.FadeIn(roof), m.FadeIn(ridge_lab), m.FadeIn(p_dot, scale=0.4), m.FadeIn(d_dot, scale=0.4)),
        )
        self.say(
            "[Prefill runs the whole prompt through each weight it reads]: [fifteen hundred FLOPs per byte], [up on the compute roof.]",
            m.Circumscribe(roof.compute, color=COMPUTE, buff=0.08),
            m.AnimationGroup(m.MoveAlongPath(p_dot, roof_path(roof, 1, I_PRE)), m.FadeIn(p_lab), run_time=1.6),
            m.Flash(roof.point(I_PRE), color=COMPUTE, flash_radius=0.3),
        )
        self.say(
            "[Decode reads every weight for one token per student]: [with sixty students, sixty FLOPs per byte], [down on the memory slope.]",
            m.Circumscribe(roof.memory, color=MEMORY, buff=0.08),
            m.AnimationGroup(m.MoveAlongPath(d_dot, roof_path(roof, 1, I_DEC)), m.FadeIn(d_lab), run_time=1.4),
            m.Flash(roof.point(I_DEC), color=MEMORY, flash_radius=0.3),
        )
        self.say(
            "[On one GPU they share every step.] [Even with Episode eleven's chunks], [a step with prompt work takes sixteen point six milliseconds, not twelve.]",
            m.AnimationGroup(m.FadeIn(share_row, shift=0.1 * m.UP), m.FadeIn(r1_lab), m.GrowFromEdge(r1_blk, m.LEFT), m.FadeIn(r1_val)),
            m.Circumscribe(share_row[1], color=REQUEST, buff=0.06),
            m.AnimationGroup(m.FadeIn(r2_lab), m.GrowFromEdge(r2_blk, m.LEFT, run_time=1.4), m.FadeIn(r2_val)),
        )
        self.say(
            "[And the two jobs want different setups.] [Prefill wants raw math], [even one prompt split across several GPUs], [for a faster first token.]",
            m.FadeIn(w_pre[0]),
            m.FadeIn(w_pre[1], shift=0.1 * m.UP),
            m.FadeIn(w_pre[2], shift=0.1 * m.UP),
            m.Circumscribe(p_dot, color=COMPUTE, buff=0.12),
        )
        self.say(
            "[Decode wants bandwidth, and memory for big batches.] [Splitwise found it barely slows on older or power-capped GPUs.]",
            m.AnimationGroup(m.FadeIn(w_dec[0]), m.FadeIn(w_dec[1], shift=0.1 * m.UP), m.FadeIn(w_dec[2][0], shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(w_dec[2][1], shift=0.1 * m.UP), m.FadeIn(src)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 2 · split the work

    def split(self):
        self.next_section("2 · Split the work", notes=f"Disaggregation: a prefill pool (compute-bound, AMBER) and a decode pool (memory-bound, CYAN). The prefill GPU runs the {PROMPT:,}-token prompt in one pass and produces the first token (so the time to first token is the prefill pool's business); the request's KV cache ({PROMPT:,} × 128 KiB, Episode 5) is sent to a decode GPU, which streams the rest of the answer (continuous batching, Episode 10). Prompts never enter a decode step, so no decode stall; each pool is sized and set up for its own job (DistServe picks parallelism and instance counts per phase; Splitwise even uses different hardware). Names: DistServe (OSDI 2024), Splitwise (ISCA 2024), Mooncake (Kimi, FAST 2025).")
        pre = Pool("prefill pool", "compute-bound", COMPUTE, AMBER_FILL, GPU_PRE, (-2.7, 0.55, 0))
        dec = Pool("decode pool", "memory-bound", MEMORY, CYAN_FILL, GPU_DEC, (3.05, 0.55, 0))
        sx, sy = -5.85, 0.62
        face = icon(USER_ICON, 0.75).move_to([sx, sy, 0])
        face_lab = text("a student", 20, MUTED).next_to(face, m.DOWN, buff=0.15)
        prompt = m.VGroup(card(1.55, 0.42, REQUEST, PINK_FILL, radius=0.1, stroke=1.8), code(f"{PROMPT:,} tokens", 20, INK))
        prompt[1].move_to(prompt[0])
        prompt.move_to([sx, 1.62, 0])
        to_pre = arrow([sx + 0.45, sy, 0], [pre.box.get_left()[0] - 0.05, sy, 0], REQUEST)
        first_path = m.Line([pre.box.get_center()[0], sy - 0.32, 0], [sx + 0.45, sy - 0.32, 0])
        first = m.Dot(first_path.get_start(), radius=0.09, color=TOKEN).set_z_index(-1)   # emerges from under the pool's card
        first_lab = code("first token", 20, TOKEN, weight=m.BOLD).move_to(pre.box).set_y(pre.track.get_y() + 0.42)

        kv_arrow = arrow([pre.box.get_right()[0] + 0.05, sy, 0], [dec.box.get_left()[0] - 0.05, sy, 0], KV)
        kv_lab = code("KV cache", 20, KV, weight=m.BOLD).next_to(kv_arrow, m.UP, buff=0.14)
        kv_size = code(f"{PROMPT:,} × 128 KiB", 20, MUTED).next_to(kv_arrow, m.DOWN, buff=0.14)
        cache = kv_block().move_to(pre.slots[0]).set_z_index(5)

        down_x, low_y = dec.box.get_center()[0], -1.72
        ret_pts = [[down_x, dec.box.get_bottom()[1] - 0.05, 0], [down_x, low_y, 0], [sx, low_y, 0], [sx, face_lab.get_bottom()[1] - 0.12, 0]]
        ret_path = m.VMobject().set_points_as_corners([[down_x, dec.box.get_y(), 0]] + ret_pts[1:])   # tokens start under the card
        ret_line = m.VGroup(m.Line(ret_pts[0], ret_pts[1]), m.Line(ret_pts[1], ret_pts[2])).set_stroke(TOKEN, 4)
        ret_tip = arrow(ret_pts[2], ret_pts[3], TOKEN, buff=0)
        ret = m.VGroup(ret_line, ret_tip)
        ret_lab = code("the rest of the answer, token by token", 20, TOKEN).move_to([(sx + down_x) / 2, low_y + 0.27, 0])

        def tokens(n, run_time=2.4):
            dots = [m.Dot(ret_path.get_start(), radius=0.075, color=TOKEN).set_z_index(-1) for _ in range(n)]
            return m.LaggedStart(*[m.Succession(m.MoveAlongPath(d, ret_path, run_time=1.4), m.FadeOut(d, run_time=0.1))
                                   for d in dots], lag_ratio=0.18, run_time=run_time)

        chapter = m.VGroup(card(1.75, 0.42, REQUEST, PINK_FILL, radius=0.1, stroke=1.8), code("10,000 tokens", 20, INK))
        chapter[1].move_to(chapter[0])
        chapter.move_to([sx, 1.62, 0])
        no_stall = pill("no stalls", TOKEN, GREEN_FILL, 22).next_to(dec.box, m.RIGHT, buff=0.25).shift(0.2 * m.DOWN)
        dec_busy = tinted(dec.slots[:2], MEMORY, MEMORY)

        name = title_text("disaggregation", 44, REQUEST).move_to([0, -2.5, 0]).align_to([sx - 0.3, 0, 0], m.LEFT)
        c1 = cite("DistServe · OSDI 2024")
        c2 = cite("Splitwise · ISCA 2024")
        c3 = cite("Mooncake (Kimi) · FAST 2025")
        cites = m.VGroup(c1, c2, c3).arrange(m.RIGHT, buff=0.5).next_to(name, m.DOWN, buff=0.22).align_to(name, m.LEFT)
        for c in cites[1:]:
            c.align_to(cites[0], m.DOWN)

        self.say(
            "[So split them]: [a pool of GPUs for prefill], [and another pool for decode.]",
            self.new_header("02 · split the work", "Two pools, one cache in between"),
            m.AnimationGroup(m.FadeIn(pre.shell()), m.LaggedStart(*[m.FadeIn(s, scale=0.6) for s in pre.slots[:2]], lag_ratio=0.3)),
            m.AnimationGroup(m.FadeIn(dec.shell()), m.LaggedStart(*[m.FadeIn(s, scale=0.6) for s in dec.slots[:2]], lag_ratio=0.3)),
        )
        bar1 = pre.bar()
        self.say(
            "[A student's prompt goes to a GPU in the prefill pool], [which runs it all in one pass], [and picks the first token.]",
            m.AnimationGroup(m.FadeIn(face, shift=0.15 * m.UP), m.FadeIn(face_lab), m.FadeIn(prompt, shift=0.1 * m.DOWN), m.GrowArrow(to_pre)),
            m.AnimationGroup(m.FadeOut(prompt, target_position=pre.slots[0], scale=0.3), m.Succession(m.Wait(0.3), m.GrowFromEdge(bar1, m.LEFT, run_time=1.2))),
            m.AnimationGroup(m.MoveAlongPath(first, first_path, run_time=1.1), m.FadeIn(first_lab)),
        )
        self.say(
            "[Then its KV cache] [travels to a decode GPU]: [a hundred and twenty-eight kibibytes for every prompt token.]",
            m.AnimationGroup(m.FadeOut(first), m.FadeOut(first_lab), m.FadeOut(bar1), m.FadeIn(cache, scale=0.5)),
            m.AnimationGroup(m.GrowArrow(kv_arrow), m.FadeIn(kv_lab), cache.animate(run_time=1.3).move_to(dec.slots[0])),
            m.FadeIn(kv_size, shift=0.1 * m.UP),
        )
        self.say(
            "[That GPU streams the rest of the answer], [one token at a time.]",
            m.AnimationGroup(m.FadeOut(cache, scale=0.6), m.FadeIn(ret), m.FadeIn(ret_lab)),
            m.AnimationGroup(tokens(7), blink(dec_busy)),
        )
        bar2 = pre.bar()
        self.say(
            "[Now a long chapter never lands in a decode step], [so every stream keeps flowing.]",
            m.AnimationGroup(m.FadeIn(chapter, shift=0.1 * m.DOWN), m.Succession(m.Wait(0.5), m.FadeOut(chapter, target_position=pre.slots[1], scale=0.3)),
                             m.Succession(m.Wait(0.8), m.GrowFromEdge(bar2, m.LEFT, run_time=2.6)), tokens(9, 3.4)),
            m.FadeIn(no_stall, shift=0.1 * m.LEFT),
        )
        self.say(
            "[And each pool grows on its own]: [more prefill GPUs for long prompts], [more decode GPUs for long answers.]",
            m.FadeOut(bar2),
            m.LaggedStart(*[m.FadeIn(s, scale=0.6) for s in pre.slots[2:]], lag_ratio=0.4),
            m.LaggedStart(*[m.FadeIn(s, scale=0.6) for s in dec.slots[2:]], lag_ratio=0.4),
        )
        self.say(
            "[This is disaggregation]: [DistServe and Splitwise in twenty twenty-four], [then Kimi's Mooncake.]",
            m.FadeIn(name, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(c1, shift=0.1 * m.UP), m.FadeIn(c2, shift=0.1 * m.UP)),
            m.FadeIn(c3, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 3 · the price: shipping the cache

    def price(self):
        self.next_section("3 · The price", notes=f"Owl's {PROMPT:,}-token prompt: {PROMPT:,} × {KVT:,} B (128 KiB, Episode 5) = {KV_PROMPT:,} B = {KV_MIB} MiB ≈ {KV_MB:.1f} MB. One way: NVLink 4 inside an H100 server ≈ 450 GB/s (900 GB/s total per GPU) → {ms(T_NV):.2f} ms; a 400 Gb/s InfiniBand NIC ≈ 50 GB/s → {ms(T_IB):.2f} ms ({T_IB / T_PRE:.0%} of the prefill); DistServe's testbed cross-node link, 25 Gb/s = 3.125 GB/s → {ms(T_SLOW):.1f} ms, longer than the prefill itself ({ms(T_PRE):.1f} ms: prefill_time at 50% MFU on one H100). DistServe therefore places matching prefill and decode stages in one node so the transfer goes 'only via NVLINK'. Splitwise overlaps the transfer with the prompt computation layer by layer ({LAYERS} layers in Owl-8B; drawn schematically): it 'only incurs 0.8% of E2E'. Launch day (illustrative: one question a minute per student): {STUDENTS:,} / 60 = {RATE:.1f} prompts/s × {KV_MB:.1f} MB = {KV_FLOW / 1e9:.1f} GB/s, {LINK_SHARE:.0%} of one 400 Gb/s link. Link speeds dated October 2026 (vendor pages, DeepEP README).")
        f_s = f"{PROMPT:,} tokens × 128 KiB = {KV_MIB} MiB ≈ {KV_MB:.0f} MB"
        formula = mono(f_s, 24, rules={f"{PROMPT:,} tokens": REQUEST, "128 KiB": KV, f"{KV_MIB} MiB ≈ {KV_MB:.0f} MB": KV}, weight=m.BOLD)
        formula.move_to([0, 2.1, 0]).align_to([-6.3, 0, 0], m.LEFT)
        cut = f_s.index("=")
        n_left = sum(not c.isspace() for c in f_s[:cut])
        n_tokens = sum(not c.isspace() for c in f_s[:f_s.index("×")])
        stack = kv_block(0.9, 6).next_to(formula, m.RIGHT, buff=0.45)

        x0, scale = -2.45, 7.0 / 70
        rows_y = [1.3, 0.66, 0.02, -0.62]
        specs = [
            ("the prefill itself", f"{PROMPT:,} tokens, one H100", T_PRE, COMPUTE, AMBER_FILL),
            ("NVLink", "inside one server", T_NV, KV, VIOLET_FILL),
            ("InfiniBand", "between servers · 400 Gb/s", T_IB, KV, VIOLET_FILL),
            ("a slow network", "25 Gb/s · DistServe's testbed", T_SLOW, RED, RED_FILL),
        ]
        rows = []
        for (name, sub, t, color, fill), y in zip(specs, rows_y):
            lab = m.VGroup(text(name, 22, COMPUTE if color == COMPUTE else INK, weight=m.BOLD), text(sub, 20, MUTED))
            lab.arrange(m.DOWN, buff=0.06, aligned_edge=m.LEFT).move_to([0, y, 0]).align_to([-6.3, 0, 0], m.LEFT)
            w = max(scale * ms(t), 0.03)
            bar = m.Rectangle(width=w, height=0.36).set_stroke(color, 1.5).set_fill(fill, 1).move_to([x0 + w / 2, y, 0])
            val = code(f"{ms(t):.2f} ms" if t < 1e-3 else f"{ms(t):.1f} ms", 20, INK if color == KV else color, weight=m.BOLD)
            val.next_to(bar, m.RIGHT, buff=0.18)
            rows.append(m.VGroup(lab, bar, val))
        ay = -1.0
        axis = m.Line([x0, ay, 0], [x0 + 7.15, ay, 0]).set_stroke(FAINT, 2)
        base = m.Line([x0, ay, 0], [x0, 1.6, 0]).set_stroke(FAINT, 2)
        ticks = m.VGroup()
        for t in (0, 20, 40, 60):
            tk = m.Line([x0 + scale * t, ay - 0.07, 0], [x0 + scale * t, ay + 0.07, 0]).set_stroke(FAINT, 2)
            ticks.add(tk, code(f"{t} ms", 20, MUTED).next_to(tk, m.DOWN, buff=0.08))
        chart = m.VGroup(axis, base, ticks)
        tenth = text("less than a tenth of the prefill", 20, MUTED).next_to(rows[2][2], m.RIGHT, buff=0.3)

        # Splitwise: each layer's cache leaves while the next layer computes (schematic)
        cw, gap, lx0, ly = 0.13, 0.03, -6.3, -2.2
        comp = m.VGroup(*[m.Rectangle(width=cw, height=0.24).set_stroke(width=0).set_fill(COMPUTE, 1).move_to([lx0 + cw / 2 + i * (cw + gap), ly, 0])
                          for i in range(LAYERS)])
        send = m.VGroup(*[m.Rectangle(width=cw, height=0.24).set_stroke(width=0).set_fill(KV, 1)
                          .move_to([lx0 + cw / 2 + (i + 1) * (cw + gap), ly - 0.36, 0]) for i in range(LAYERS)])
        sw_title = text("Splitwise: each layer's cache leaves as the next computes", 20, INK)
        sw_title.move_to([0, -1.8, 0]).align_to([lx0, 0, 0], m.LEFT)
        sw_keys = m.VGroup(code("compute", 20, COMPUTE), code("send", 20, KV)).arrange(m.DOWN, buff=0.14, aligned_edge=m.LEFT)
        sw_keys.next_to(m.VGroup(comp, send), m.RIGHT, buff=0.2)
        sw_src = cite("Splitwise, ISCA 2024: transfer ≈ 0.8% of end-to-end").move_to([0, -3.05, 0]).align_to([lx0, 0, 0], m.LEFT)

        # launch day
        box = card(5.25, 2.05, EDGE, CARD, radius=0.18, stroke=2)
        box.move_to([0, -2.47, 0]).align_to([6.6, 0, 0], m.RIGHT)
        bx = box.get_left()[0] + 0.3
        l1s = "launch day · a question a minute each"
        l1 = text(l1s, 20, MUTED)
        l2s = f"{RATE:.0f} prompts/s × {KV_MB:.0f} MB"
        l2 = mono(l2s, 20, rules={f"{RATE:.0f} prompts/s": REQUEST})
        l3s = f"= {KV_FLOW / 1e9:.1f} GB/s of KV cache"
        l3 = mono(l3s, 20, KV, weight=m.BOLD)
        meter = Meter(width=1.9, fraction=0.0, color=KV, height=0.22)
        m_lab = code(f"{LINK_SHARE:.0%} of one link", 20, INK)
        for wds, s, y in ((l1, l1s, -1.78), (l2, l2s, -2.2), (l3, l3s, -2.6)):
            wds.align_to([bx, 0, 0], m.LEFT)
            sit(wds, s, y)
        meter.move_to([0, -3.06, 0]).align_to([bx, 0, 0], m.LEFT)
        m_lab.next_to(meter, m.RIGHT, buff=0.2)

        self.say(
            "[Splitting has a price]: [the cache must travel.] [Owl's prompt is fifteen hundred tokens], [at a hundred and twenty-eight kibibytes each.]",
            self.new_header("03 · the price", "Shipping the cache"),
            m.FadeIn(stack, shift=0.2 * m.RIGHT),
            m.FadeIn(formula[:n_tokens], shift=0.1 * m.UP),
            m.FadeIn(formula[n_tokens:n_left], shift=0.1 * m.UP),
        )
        self.say(
            "[That's about two hundred megabytes of keys and values to ship.]",
            m.AnimationGroup(m.FadeIn(formula[n_left:], shift=0.1 * m.UP), m.Circumscribe(stack, color=KV, buff=0.08)),
        )
        self.say(
            "[For scale], [the prefill itself takes forty-nine milliseconds.]",
            m.AnimationGroup(m.FadeIn(chart), m.FadeIn(rows[0][0])),
            m.AnimationGroup(m.GrowFromEdge(rows[0][1], m.LEFT, run_time=1.2), m.FadeIn(rows[0][2])),
        )
        self.say(
            "[Inside one server, over NVLink], [the cache moves in under half a millisecond.]",
            m.FadeIn(rows[1][0], shift=0.1 * m.UP),
            m.AnimationGroup(m.GrowFromEdge(rows[1][1], m.LEFT), m.FadeIn(rows[1][2])),
        )
        self.say(
            "[Between servers, over four hundred gigabit InfiniBand], [about four milliseconds], [less than a tenth of the prefill.]",
            m.FadeIn(rows[2][0], shift=0.1 * m.UP),
            m.AnimationGroup(m.GrowFromEdge(rows[2][1], m.LEFT), m.FadeIn(rows[2][2])),
            m.FadeIn(tenth, shift=0.1 * m.LEFT),
        )
        self.say(
            "[But on a slow twenty-five gigabit network], [it takes sixty-three milliseconds]: [longer than the prefill itself.]",
            m.FadeIn(rows[3][0], shift=0.1 * m.UP),
            m.AnimationGroup(m.GrowFromEdge(rows[3][1], m.LEFT, run_time=1.4), m.FadeIn(rows[3][2])),
            m.Circumscribe(m.VGroup(rows[3][1], rows[3][2]), color=RED, buff=0.08),
        )
        self.say(
            "[In practice, Splitwise hides most of the trip]: [each layer's cache leaves while the next layer computes.]",
            m.AnimationGroup(m.FadeIn(sw_title), m.FadeIn(sw_keys), m.FadeIn(sw_src)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c) for c in comp], lag_ratio=0.6, run_time=2.4),
                             m.Succession(m.Wait(2.4 / LAYERS), m.LaggedStart(*[m.FadeIn(s, shift=0.1 * m.DOWN) for s in send], lag_ratio=0.6, run_time=2.4))),
        )
        self.say(
            "[On launch day], [ten thousand students asking once a minute] [ship about thirty-three gigabytes of cache every second]: [two thirds of one such link.]",
            m.AnimationGroup(m.FadeIn(box), m.FadeIn(meter)),
            m.FadeIn(l1, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(l2, shift=0.1 * m.UP), m.FadeIn(l3, shift=0.1 * m.UP)),
            m.AnimationGroup(meter.fill_to(LINK_SHARE, run_time=1.2), m.FadeIn(m_lab)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 4 · goodput and SLOs

    def goodput(self):
        self.next_section("4 · Goodput", notes=f"Throughput counts every finished answer; students feel two clocks (Episode 1): time to first token (TTFT) and time per output token (TPOT). A target for each is a service level objective (SLO); the targets shown are DistServe's chatbot SLOs for OPT-13B (Table 1: TTFT {TTFT_SLO} ms, TPOT {TPOT_SLO} ms); the ten answers and the three late ones are illustrative. Goodput (DistServe): 'the maximum request rate that can be served adhering to the SLO attainment goal (say, 90%)'. DistServe Fig. 1 (13B model on A100s, 512-token prompts, 64-token answers, its own two targets): colocated {SHARED} req/s per GPU; prefill-only {PRE_ONLY}, decode-only {DEC_ONLY:.0f} per GPU; {N_PRE} prefill + {N_DEC} decode GPU = min({N_PRE} × {PRE_ONLY}, {DEC_ONLY:.0f}) = {SPLIT_TOTAL:.0f} req/s on {N_PRE + N_DEC} GPUs = {SPLIT_PER:.2f} per GPU = {GAIN:.2f}× ('2.1x'). The curve SHAPES are a sketch (logistics through the 90% crossings); only the crossings are DistServe's. Headline: 'up to 7.4× more requests' within the targets for > 90% of requests, against DeepSpeed-MII (chunked prefill) on chatbot workloads.")
        # ten finished answers, three of them late (illustrative)
        chips = m.VGroup(*[m.RoundedRectangle(corner_radius=0.08, width=0.42, height=0.42).set_stroke(REQUEST, 2).set_fill(PINK_FILL, 1)
                           for _ in range(10)]).arrange(m.RIGHT, buff=0.12).move_to([0, 2.05, 0]).align_to([-6.2, 0, 0], m.LEFT)
        late = [2, 6, 7]
        reds = tinted([chips[i] for i in late], RED, RED_FILL, 2)
        greens = tinted([c for i, c in enumerate(chips) if i not in late], TOKEN, GREEN_FILL, 2)
        thr_s, on_s = "throughput · 10 answers", "on time · 7"
        thr = mono(thr_s, 20, REQUEST, weight=m.BOLD).move_to([0, 1.5, 0]).align_to(chips, m.LEFT)
        ontime = mono(on_s, 20, TOKEN, weight=m.BOLD).next_to(thr, m.RIGHT, buff=0.45)
        sit(ontime, on_s, baseline(thr, thr_s))

        tx = 0.75
        tag1 = code("TTFT", 20, COMPUTE, weight=m.BOLD)
        nm1 = text("time to first token", 22, INK)
        v1 = code(f"≤ {TTFT_SLO} ms", 20, COMPUTE, weight=m.BOLD)
        tag2 = code("TPOT", 20, MEMORY, weight=m.BOLD)
        nm2 = text("time per output token", 22, INK)
        v2 = code(f"≤ {TPOT_SLO} ms", 20, MEMORY, weight=m.BOLD)
        xv = tx + tag2.width + 0.22 + max(nm1.width, nm2.width) + 0.35
        for (tg, nm, v), y in (((tag1, nm1, v1), 2.2), ((tag2, nm2, v2), 1.68)):
            tg.move_to([0, y, 0]).align_to([tx, 0, 0], m.LEFT)
            nm.next_to(tg, m.RIGHT, buff=0.22)
            sit(nm, nm.text, baseline(tg, tg.text))
            v.move_to([0, y, 0]).align_to([xv, 0, 0], m.LEFT)
            sit(v, v.text, baseline(tg, tg.text))
        slo_lab = text("SLOs · DistServe's chatbot targets", 20, MUTED)
        slo_lab.move_to([0, 1.22, 0]).align_to([tx, 0, 0], m.LEFT)
        ep01 = ep_pill(1).move_to([0, 2.68, 0]).align_to([tx, 0, 0], m.LEFT)
        assert v2.get_right()[0] < 6.6

        # the SLO-attainment sketch
        ox, oy, W, H, xmax = -5.5, -2.7, 6.1, 3.35, 5.0
        px = lambda r: ox + W * r / xmax
        py = lambda a: oy + H * a
        x_axis = m.Line([ox, oy, 0], [ox + W + 0.15, oy, 0]).set_stroke(FAINT, 2)
        y_axis = m.Line([ox, oy, 0], [ox, oy + H + 0.1, 0]).set_stroke(FAINT, 2)
        xt = m.VGroup(*[code(f"{r}", 20, MUTED).move_to([px(r), oy - 0.27, 0]) for r in range(6)])
        yt = m.VGroup(*[code(f"{a}%", 20, MUTED).move_to([0, py(a / 100), 0]).align_to([ox - 0.15, 0, 0], m.RIGHT) for a in (0, 50, 100)])
        x_name = text("requests per second, per GPU", 20, MUTED).move_to([ox + W / 2, oy - 0.62, 0])
        y_name = text("answers on time", 20, MUTED).move_to([0, oy + H + 0.32, 0]).align_to([ox - 0.15, 0, 0], m.LEFT)
        axes_ = m.VGroup(x_axis, y_axis, xt, yt, x_name, y_name)
        line90 = m.DashedLine([ox, py(0.9), 0], [ox + W, py(0.9), 0], dash_length=0.1).set_stroke(TOKEN, 2)
        lab90 = code("90%", 20, TOKEN, weight=m.BOLD).next_to(line90, m.RIGHT, buff=0.12)

        rs = np.linspace(0, xmax, 120)
        shared_curve = m.VMobject().set_points_smoothly([[px(r), py(attainment(r, SHARED, 0.22)), 0] for r in rs]).set_stroke(INK, 4.5)
        split_curve = m.VMobject().set_points_smoothly([[px(r), py(attainment(r, SPLIT_PER, 0.3)), 0] for r in rs]).set_stroke(TOKEN, 5)

        def marker(g, color, name, value, level):
            d = m.Dot([px(g), py(0.9), 0], radius=0.1, color=color).set_z_index(4)
            drop = m.DashedLine([px(g), py(0.9), 0], [px(g), oy, 0], dash_length=0.08).set_stroke(color, 2)
            lab = m.VGroup(text(name, 22, color, weight=m.BOLD), code(value, 20, color, weight=m.BOLD)).arrange(m.DOWN, buff=0.08, aligned_edge=m.RIGHT)
            lab.move_to([0, py(level), 0]).align_to([px(g) - 0.15, 0, 0], m.RIGHT)
            return m.VGroup(drop, d, lab)

        mk_shared = marker(SHARED, INK, "shared", f"{SHARED} req/s", 0.7)
        mk_split = marker(SPLIT_PER, TOKEN, "split", f"{SPLIT_PER:.1f} req/s", 0.76)

        box = card(5.25, 3.85, EDGE, CARD, radius=0.18, stroke=2).move_to([0, -1.03, 0]).align_to([6.6, 0, 0], m.RIGHT)
        bx, brx = box.get_left()[0] + 0.3, box.get_right()[0] - 0.3
        head = text("DistServe's example: a 13B model, A100s", 20, MUTED)
        sketch = cite("curve shapes: a sketch").align_to([bx, 0, 0], m.LEFT)
        sit(sketch, sketch.text, -2.75)

        def row(left, right, color, y, bold=False):
            a = text(left, 22, color, weight=m.BOLD if bold else m.NORMAL).align_to([bx, 0, 0], m.LEFT)
            b = code(right, 20, color if bold else INK, weight=m.BOLD).align_to([brx, 0, 0], m.RIGHT)
            sit(a, left, y)
            sit(b, right, y)
            return m.VGroup(a, b)

        head.align_to([bx, 0, 0], m.LEFT)
        sit(head, head.text, 0.52)
        r_pre = row("prefill GPU alone", f"{PRE_ONLY} req/s", COMPUTE, -0.02)
        r_dec = row("decode GPU alone", f"{DEC_ONLY:.0f} req/s", MEMORY, -0.5)
        r_mix = row(f"{N_PRE} prefill + {N_DEC} decode", f"{SPLIT_TOTAL:.0f} req/s", INK, -1.08)
        r_per = row(f"per GPU, on {N_PRE + N_DEC} GPUs", f"{SPLIT_PER:.1f} req/s", TOKEN, -1.56, bold=True)
        gain = m.VGroup(title_text(f"{GAIN:.1f}×", 54, TOKEN), text(f"the shared {SHARED}", 22, MUTED))
        gain.arrange(m.RIGHT, buff=0.25, aligned_edge=m.DOWN).move_to([0, -2.2, 0]).align_to([bx, 0, 0], m.LEFT)
        rule = m.Line([bx, -0.8, 0], [brx, -0.8, 0]).set_stroke(EDGE, 1.5)
        body = m.VGroup(head, r_pre, r_dec, rule, r_mix, r_per, gain, sketch)

        hl = m.VGroup(
            text("against a chunked-prefill server", 22, INK),
            m.VGroup(title_text(f"{HEADLINE}×", 80, TOKEN), text("more requests", 24, INK)).arrange(m.RIGHT, buff=0.25, aligned_edge=m.DOWN),
            text("on time, for over 90% of requests", 20, MUTED),
            m.VGroup(cite("DistServe, OSDI 2024 · chatbot workloads"), cite("vs DeepSpeed-MII (chunked prefill)")).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.08),
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.22)
        hl[1].shift(0.06 * m.DOWN)
        hl.move_to(box).align_to([bx, 0, 0], m.LEFT)

        self.say(
            "[How should we score a server?] [Throughput counts every answer it finishes], [late or not.]",
            self.new_header("04 · goodput", "Count only answers on time"),
            m.LaggedStart(*[m.FadeIn(c, shift=0.2 * m.RIGHT) for c in chips], lag_ratio=0.15),
            m.FadeIn(thr, shift=0.1 * m.UP),
        )
        self.say(
            "[But students feel the two clocks from Episode one]: [time to first token], [and time per output token.]",
            m.FadeIn(ep01, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(tag1), m.FadeIn(nm1)),
            m.AnimationGroup(m.FadeIn(tag2), m.FadeIn(nm2)),
        )
        self.say(
            "[Give each clock a target], [a service level objective], [and the late answers stop counting.]",
            m.AnimationGroup(m.FadeIn(v1, shift=0.1 * m.LEFT), m.FadeIn(v2, shift=0.1 * m.LEFT)),
            m.FadeIn(slo_lab),
            m.AnimationGroup(m.FadeIn(reds), m.FadeIn(greens), m.FadeIn(ontime, shift=0.1 * m.UP)),
        )
        self.say(
            "[Goodput is the highest request rate] [at which, say, ninety percent of answers meet both targets.]",
            m.FadeIn(axes_),
            m.AnimationGroup(m.Create(line90), m.FadeIn(lab90)),
        )
        self.say(
            "[In an example from the DistServe paper], [a thirteen billion parameter model sharing each GPU between both jobs] [reaches one point six requests a second.]",
            m.AnimationGroup(m.FadeIn(box), m.FadeIn(head)),
            m.Create(shared_curve, run_time=1.8),
            m.FadeIn(mk_shared, scale=0.8),
        )
        self.say(
            "[On their own], [prefill sustains five point six per GPU], [and decode ten]: [so give prefill two GPUs for every decode GPU.]",
            m.Wait(0.1),
            m.FadeIn(r_pre, shift=0.1 * m.UP),
            m.FadeIn(r_dec, shift=0.1 * m.UP),
            m.AnimationGroup(m.Create(rule), m.FadeIn(r_mix, shift=0.1 * m.UP)),
        )
        self.say(
            "[That's ten requests a second on three GPUs]: [three point three each], [two point one times the shared goodput.]",
            m.Circumscribe(r_mix, color=INK, buff=0.08),
            m.AnimationGroup(m.FadeIn(r_per, shift=0.1 * m.UP), m.Create(split_curve, run_time=1.8), m.FadeIn(mk_split, scale=0.8)),
            m.AnimationGroup(m.FadeIn(gain, shift=0.1 * m.UP), m.FadeIn(sketch)),
        )
        self.say(
            "[Against a server already using chunked prefill], [DistServe served up to seven point four times more requests on time.]",
            m.AnimationGroup(m.FadeOut(body), m.FadeIn(hl[0])),
            m.AnimationGroup(m.FadeIn(hl[1], scale=0.8), m.FadeIn(hl[2]), m.FadeIn(hl[3])),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 5 · in production

    def production(self):
        self.next_section("5 · In production", notes=f"DeepSeek, 'Day 6: DeepSeek-V3/R1 inference system overview' (Feb 2025): 'prefill-decode disaggregation'; prefill units of {PRE_NODES} nodes (EP32, {PRE_NODES * NODE} H800 GPUs), decode units of {DEC_NODES} nodes (EP144, {DEC_NODES * NODE} GPUs); per H800 node ≈ {TOK_IN / 1e3:.1f}k input tokens/s in prefill and ≈ {TOK_OUT / 1e3:.1f}k output tokens/s in decode (24 h average, Feb 2025; cache hits included in the input figure). Also disaggregated: Mooncake (Kimi), NVIDIA Dynamo, SGLang (large-scale PD, LMSYS 2025), llm-d (vLLM). The catch (notes §5 misconceptions): disaggregation adds the KV transfer and operational complexity; its gains come in SLO-bound regimes and with heterogeneous hardware; for a small fleet, chunked prefill (Episode 11) may be enough.")
        sq, gp = 0.2, 0.07

        def unit(nodes, color, fill):
            return m.VGroup(*[m.VGroup(*[m.Square(sq).set_stroke(color, 1.4).set_fill(fill, 1) for _ in range(NODE)]).arrange(m.DOWN, buff=gp)
                              for _ in range(nodes)]).arrange(m.RIGHT, buff=gp + 0.05)

        pre_u = unit(PRE_NODES, COMPUTE, AMBER_FILL).move_to([-4.95, 0.15, 0])
        dec_u = unit(DEC_NODES, MEMORY, CYAN_FILL).move_to([2.35, 0.15, 0])

        def unit_lab(title, sub, color, u):
            t = text(title, 24, color, weight=m.BOLD)
            s = code(sub, 20, MUTED)
            return m.VGroup(t, s).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT).next_to(u, m.UP, buff=0.25).align_to(u, m.LEFT)

        pre_lab = unit_lab("prefill unit", f"{PRE_NODES} nodes · {PRE_NODES * NODE} GPUs", COMPUTE, pre_u)
        dec_lab = unit_lab("decode unit", f"{DEC_NODES} nodes · {DEC_NODES * NODE} GPUs", MEMORY, dec_u)
        link = arrow([pre_u.get_right()[0] + 0.15, 0.15, 0], [dec_u.get_left()[0] - 0.15, 0.15, 0], KV)
        link_lab = code("KV cache", 20, KV, weight=m.BOLD).next_to(link, m.UP, buff=0.12)
        src = cite("DeepSeek-V3/R1 inference overview, Feb 2025 · 8 H800 GPUs a node")
        src.move_to([0, -1.25, 0]).align_to(pre_u, m.LEFT)

        s_in = big_stat(f"{TOK_IN / 1e3:.1f}k", "tokens/s in", "per prefill node", COMPUTE, 50)
        s_out = big_stat(f"{TOK_OUT / 1e3:.1f}k", "tokens/s out", "per decode node", MEMORY, 50)
        s_in.move_to([0, -2.3, 0]).align_to(pre_u, m.LEFT)
        s_out.move_to([0, -2.3, 0]).align_to(dec_u, m.LEFT)

        others = m.VGroup(*[pill(s, REQUEST, PINK_FILL, 20) for s in ("Mooncake · Kimi", "Dynamo · NVIDIA", "SGLang", "llm-d")])
        others.arrange(m.RIGHT, buff=0.3)
        others_lab = text("also split:", 20, MUTED).next_to(others, m.LEFT, buff=0.3)
        m.VGroup(others_lab, others).move_to([0, -2.4, 0])

        catch = card(12.4, 1.55, RED, CARD, radius=0.18, stroke=2).move_to([0, -2.45, 0])
        c_head = text("the catch", 24, RED, weight=m.BOLD)
        items = m.VGroup(*[text(s, 22, INK) for s in ("a fast link", "two pools to balance", "more to run")]).arrange(m.RIGHT, buff=0.7)
        c_row = m.VGroup(c_head, items).arrange(m.RIGHT, buff=0.7).move_to(catch).shift(0.3 * m.UP)
        c_small = m.VGroup(text("for a small fleet,", 22, MUTED), ep_pill(11), text("chunked prefill may be enough", 22, MUTED)).arrange(m.RIGHT, buff=0.2)
        c_small.next_to(c_row, m.DOWN, buff=0.22).set_x(0)

        self.say(
            "[Even DeepSeek splits the work in production]: [a prefill unit is four nodes], [thirty-two GPUs.]",
            m.AnimationGroup(self.new_header("05 · in production", "How DeepSeek splits it"), m.FadeIn(src)),
            m.AnimationGroup(m.FadeIn(pre_lab), m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.UP) for c in pre_u], lag_ratio=0.25)),
            m.Circumscribe(pre_lab[1], color=COMPUTE, buff=0.06),
        )
        self.say(
            "[A decode unit is eighteen nodes], [a hundred and forty-four GPUs.]",
            m.AnimationGroup(m.GrowArrow(link), m.FadeIn(link_lab), m.FadeIn(dec_lab)),
            m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.UP) for c in dec_u], lag_ratio=0.12, run_time=1.8),
        )
        self.say(
            "[Each prefill node takes in about seventy-four thousand tokens a second]; [each decode node writes about fifteen thousand.]",
            m.FadeIn(s_in, shift=0.1 * m.UP),
            m.FadeIn(s_out, shift=0.1 * m.UP),
        )
        self.say(
            "[Others split the work the same way]: [Kimi's Mooncake], [Dynamo], [and SGLang.]",
            m.AnimationGroup(m.FadeOut(s_in, shift=0.1 * m.UP), m.FadeOut(s_out, shift=0.1 * m.UP), m.FadeIn(others_lab)),
            m.FadeIn(others[0], shift=0.1 * m.UP),
            m.FadeIn(others[1], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(others[2], shift=0.1 * m.UP), m.FadeIn(others[3], shift=0.1 * m.UP)),
        )
        self.say(
            "[But splitting isn't free]: [a fast link, two pools to balance, more to run.] [For a small fleet, Episode eleven's chunked prefill may be enough.]",
            m.AnimationGroup(m.FadeOut(others_lab), m.FadeOut(others), m.FadeIn(catch), m.FadeIn(c_head)),
            m.LaggedStart(*[m.FadeIn(i, shift=0.1 * m.UP) for i in items], lag_ratio=0.5),
            m.FadeIn(c_small, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes=f"Split: prefill and decode on their own pools (no stalls, each sized and set up for its job). Ship: the KV cache travels between them ({PROMPT:,} × 128 KiB ≈ {KV_MB:.0f} MB for Owl's prompt; {ms(T_IB):.1f} ms over 400 Gb/s InfiniBand). Goodput: count requests within both targets (DistServe: {SHARED} → {SPLIT_PER:.1f} req/s per GPU in its example; up to {HEADLINE}×).")
        halo = glow(PINK, radius=4.0)
        words = m.VGroup(*[title_text(w, 80, c) for w, c in (("Split", REQUEST), ("Ship", KV), ("Goodput", TOKEN))])
        subs = m.VGroup(*[code(s, 22, MUTED) for s in ("two pools", "the KV cache", "answers on time")])
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
        tagline = text("Give each job its own machines, and count only the answers on time.", 30, MUTED).next_to(row, m.DOWN, buff=0.7)
        mascot = owl(0.5).next_to(tagline, m.DOWN, buff=0.45)

        self.say(
            "[To sum up]: [split the jobs], [ship the cache], [count goodput.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say("[Give each job its own machines], [and count only the answers that arrive on time.]",
                 m.Write(tagline), m.FadeIn(mascot, shift=0.15 * m.UP))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline, mascot)], run_time=0.9)
