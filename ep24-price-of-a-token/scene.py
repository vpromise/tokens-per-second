"""Tokens per Second · Episode 24 · The Price of a Token (Module 6 · Production & Your Mac).

Owl's monthly GPU bill for 10,000 students, from first principles: $ per million tokens = $ per GPU-hour ÷
millions of tokens per hour. Episode 4's cost-vs-batch curve on one rented H100 ($3.99/GPU-hour, Lambda, dated
Oct 2026); why output costs ~6× input (prefill compute-bound, decode memory-bound; energy measured on this Mac;
API price ratios and cached-input pricing, dated); Little's law L = λW sizing the evening peak (20 questions/s →
124 in flight on one H100, solved by fixed-point iteration); utilisation (a GPU paid for 24 hours, busy 17% of
the day); the bill ($2,872.80, 29 cents per student); earlier episodes as price cuts (4-bit, MoE, speculative
decoding) and why they shrink at the peak; reasoning models moving the bill to decode (1 → 13 H100s).

Every number is computed below from owl_reference (step_time, prefill_time, kv_bytes_per_token,
speculative_speedup, measured()) or cited from research/notes.md §0 #14–15, §9, §15, §17. The usage pattern
(30 questions a day, the hourly curve, 2,000 tokens in / 300 out, 90% cached) is illustrative ("for example").

    VOICE=draft manimgx inspect scene.py      # silent layout pass
    manimgx render scene.py                   # with the shared narrator running
"""

import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from owl_reference import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- the numbers (all computed)

GB = 1e9
W_GB = weight_bytes() / GB                      # 16.06 GB of BF16 weights

# Episode 4's price model (notes §15): one H100 at a dated rental price, 2K-token chats, 75% of peak bandwidth
PRICE = 3.99            # $ per H100 SXM GPU-hour, Lambda (read 2026-10-23) — dated, October 2026
CTX, MBU = 2048, 0.75   # for example (Episode 4)


def cost_step(b, ctx=CTX, bits=16):
    """One decode step at batch b, at 75% of peak bandwidth (owl_reference.step_time ÷ MBU)."""
    t = step_time(batch=b, context=ctx, bits=bits)
    assert t * GPU["peak"] > b * flops_per_token()      # still memory-bound, so dividing by MBU is right
    return t / MBU


def tps(b, ctx=CTX, bits=16):
    """Output tokens per second of the whole batch."""
    return b / cost_step(b, ctx, bits)


def usd_per_m(b, ctx=CTX, bits=16):
    """$ per million output tokens = $ per GPU-hour ÷ millions of tokens per hour."""
    return PRICE / (tps(b, ctx, bits) * 3600 / 1e6)


def dollars(x, places=2):
    return f"${x:,.{places}f}"


COST_B = [1, 8, 32, 64]
assert [f"{usd_per_m(b):.2f}" for b in COST_B] == ["7.20", "1.00", "0.34", "0.23"]       # Episode 4, unchanged
assert [round(tps(b)) for b in COST_B] == [154, 1104, 3262, 4838]
KV_TOK = kv_bytes_per_token()
KV_SEQ = CTX * KV_TOK                                                                     # 256 MiB per 2K chat
MAX_B = int((GPU["memory"] - weight_bytes()) // KV_SEQ)                                   # memory full
assert KV_TOK == 128 * 2 ** 10 and MAX_B == 238 and f"{usd_per_m(MAX_B):.2f}" == "0.15"

# input vs output (notes §15): prefill at 50% MFU is compute-bound; decode at batch 64 is memory-bound
PREFILL_TPS = 1 / prefill_time(1)                                                        # tokens/s, one H100
USD_IN = PRICE / (PREFILL_TPS * 3600 / 1e6)
RATIO_64, RATIO_1 = usd_per_m(64) / USD_IN, usd_per_m(1) / USD_IN
assert round(PREFILL_TPS, -2) == 30_800 and round(PREFILL_TPS / 1000) == 31 and f"{USD_IN:.3f}" == "0.036" and USD_IN < 0.04   # "under four cents"
assert round(RATIO_64) == 6 and int(RATIO_64) == 6 and round(RATIO_1, -1) == 200

# measured on this Mac (research/measurements.md §11, GPU-only energy counters, Llama-3.1-8B BF16)
E_IN = measured("energy", "llama-3.1-8b", "bf16", "prefill_mJ_per_prompt_token")          # mJ per prompt token
E_OUT = measured("energy", "llama-3.1-8b", "bf16", "decode_b1_J_per_token") * 1e3          # mJ per generated token
assert round(E_IN) == 55 and round(E_OUT) == 606 and 10 < E_OUT / E_IN < 12                # "about a tenth"

# published API prices, $ per million tokens (notes §15 table, official pricing pages, DATED Oct 2026)
API = [("Opus 5.5", 4, 20), ("gpt-5.5", 5, 30), ("gpt-4.1", 2, 8)]      # Anthropic, OpenAI, OpenAI
assert [o // i for _, i, o in API] == [5, 6, 4] and all(o % i == 0 for _, i, o in API)
CACHE_READ, CACHE_READ_55, OPENAI_OFF = 0.1, 0.05, 0.95   # notes §0 #14 (checked on the pricing pages 2026-10-24)

# Little's law at the evening peak (notes §15, recomputed): B = λ × (TTFT + 300 × TPOT(B))
LAM, TTFT, OUT = 20, 0.3, 300            # questions per second at the peak, first-token wait, answer tokens


def in_flight(L):
    return LAM * (TTFT + OUT * cost_step(L))


ITER = [0.0]
for _ in range(200):
    ITER.append(in_flight(ITER[-1]))
L_PEAK = ITER[-1]
TPOT_PEAK = cost_step(L_PEAK)
W_PEAK = TTFT + OUT * TPOT_PEAK
KV_PEAK = L_PEAK * KV_SEQ
assert abs(in_flight(L_PEAK) - L_PEAK) < 1e-6 and round(L_PEAK) == 124
assert [round(x) for x in ITER[1:5]] == [44, 73, 91, 103]
assert round(TPOT_PEAK * 1e3, 1) == 19.6 and round(TPOT_PEAK * 1e3) == 20 and round(W_PEAK, 1) == 6.2
assert round(KV_PEAK / GB) == 33 and round((weight_bytes() + KV_PEAK) / GB) == 49
assert weight_bytes() + KV_PEAK < GPU["memory"] and L_PEAK < MAX_B                           # fits on one H100
assert f"{usd_per_m(L_PEAK):.2f}" == "0.18"

# the day (for example): 30 questions per student per day, quiet at night, peaking at 20/s in the evening
STUDENTS, PER_DAY, DAYS = 10_000, 30, 30
SHAPE = [2.0, 0.9, 0.4, 0.2, 0.2, 0.3, 0.6, 1.2, 1.8, 2.2, 2.4, 2.4,
         2.6, 2.5, 2.6, 3.0, 3.6, 4.4, 6.0, 9.5, 15.0, 20.0, 13.0, 5.0]           # illustrative shape
PEAK_H = SHAPE.index(max(SHAPE))
_k = (STUDENTS * PER_DAY / 3600 - LAM) / (sum(SHAPE) - SHAPE[PEAK_H])
DAY = [float(LAM) if h == PEAK_H else s * _k for h, s in enumerate(SHAPE)]      # questions per second, hourly
LAM_AVG = sum(DAY) / 24
UTIL = LAM_AVG / LAM                                                             # busy share of the paid day
assert max(DAY) == LAM and abs(sum(DAY) * 3600 - STUDENTS * PER_DAY) < 1e-6
assert round(LAM_AVG, 1) == 3.5 and round(UTIL * 100) == 17 and round(24 * UTIL) == 4

# the bill (for example: 2,000 tokens in, 90% a cached prefix, 300 out)
QUESTIONS = STUDENTS * PER_DAY * DAYS
IN_TOK, CACHED = 2000, 0.9
TOK_IN, TOK_OUT = QUESTIONS * IN_TOK, QUESTIONS * OUT
HOURS = 24 * DAYS
GPUS_NOW = math.ceil(LAM * OUT / tps(MAX_B))                    # capacity at the peak: one H100 is enough
BILL = GPUS_NOW * HOURS * PRICE
PEAK_USD = PRICE / (LAM * OUT * 3600 / 1e6)                     # $ per M output tokens in the peak hour
IDEAL = TOK_OUT / 1e6 * PEAK_USD                                # if every hour were as busy as the peak
REAL_USD = BILL / (TOK_OUT / 1e6)                               # what each million output tokens really costs
PER_STUDENT = BILL / STUDENTS
assert QUESTIONS == 9_000_000 and TOK_IN == 18e9 and TOK_OUT == 2.7e9 and HOURS == 720 and GPUS_NOW == 1
assert f"{BILL:,.2f}" == "2,872.80" and round(BILL) == 2873
assert round(IDEAL) == 499 and round(IDEAL, -2) == 500 and abs(IDEAL - BILL * UTIL) < 1e-6
assert f"{PEAK_USD:.2f}" == "0.18" and f"{REAL_USD:.2f}" == "1.06" and round(REAL_USD / PEAK_USD) == 6
assert f"{PER_STUDENT:.2f}" == "0.29"
# the new 10% of each prompt prefills inside the peak's memory-bound decode steps (chunked prefill, Episode 11)
NEW_IN_PEAK = LAM * IN_TOK * (1 - CACHED)                                        # 4,000 tokens/s at the peak
STEP_MATH = (L_PEAK + NEW_IN_PEAK * TPOT_PEAK) * flops_per_token() / (GPU["peak"] * 0.5)
assert round(NEW_IN_PEAK) == 4000 and STEP_MATH < TPOT_PEAK * MBU                # math fits inside the memory time

# earlier episodes as price cuts
Q4_BITS = 4.5                                                                    # Episode 19: ≈ 4.5 bits per weight
assert round(weight_bytes(bits=Q4_BITS) / GB, 1) == 4.5
CUT_Q4 = usd_per_m(1) / usd_per_m(1, bits=Q4_BITS)                               # batch 1
SAVE_Q4_PEAK = 1 - usd_per_m(L_PEAK, bits=Q4_BITS) / usd_per_m(L_PEAK)            # at the evening peak
Q4_MAC = measured("decode", "llama-3.1-8b", "4bit", "speedup_vs_bf16")            # measured on this Mac
assert round(CUT_Q4, 1) == 3.4 and round(SAVE_Q4_PEAK * 100) == 23 and SAVE_Q4_PEAK < 0.25 and round(Q4_MAC, 1) == 3.3
MIX_TOTAL, MIX_ACTIVE = 46.7, 12.9                                               # Mixtral (notes §9, Mistral 2023)
CUT_MOE = MIX_TOTAL / MIX_ACTIVE
assert round(CUT_MOE, 1) == 3.6 and 0.25 <= round(MIX_ACTIVE / MIX_TOTAL, 2) <= 0.3     # "about a quarter"
ALPHA, GAMMA = 0.8, 4                                                            # Episode 13's example
C_DRAFT = measured("decode", "llama-3.2-1b", "4bit", "ms_per_token") / measured("decode", "llama-3.1-8b", "bf16", "ms_per_token")
CUT_SPEC = speculative_speedup(ALPHA, GAMMA, C_DRAFT)
E_PASS = expected_tokens_per_pass(ALPHA, GAMMA)
assert round(CUT_SPEC, 1) == 2.7 and round(E_PASS, 2) == 3.36

# reasoning (notes §17): for example 2,000 hidden thinking tokens before each 300-token answer
THINK = 2000
OUT_R = THINK + OUT
CTX_R = 4096                                                                     # for example: ≈ 4K-token caches
MAX_R = int((GPU["memory"] - weight_bytes()) // (CTX_R * KV_TOK))
TPS_R = tps(MAX_R, ctx=CTX_R)
GPUS_R = math.ceil(LAM * OUT_R / TPS_R)
assert round(OUT_R / OUT, 1) == 7.7 and MAX_R == 119 and round(TPS_R) == 3740 and GPUS_R == 13

USER = recolored("user", REQUEST, PINK_FILL, prefix="ep24")
GPU_ICON = recolored("gpu", MUTED, CARD_HI, prefix="ep24")

# ---------------------------------------------------------------------------- helpers

_FLAT = set("0123456789abcdefhiklmnorstuvwxzABCDEFGHIKLMNOPRSTUVWXYZ")


def baseline(t):
    """The baseline of a Text: median bottom of its letters that sit on the line."""
    chars = t.text.replace(" ", "").replace("\n", "")
    if len(chars) == len(t):
        ys = [g.get_bottom()[1] for g, c in zip(t, chars) if c in _FLAT]
        if ys:
            return float(np.median(ys))
    return float(np.median([g.get_bottom()[1] for g in t]))


def sit(t, y):
    t.shift((y - baseline(t)) * m.UP)
    return t


def cite(s, size=20):
    return text(s, size, MUTED)


def left_at(mob, x, y):
    return mob.move_to([x, y, 0], aligned_edge=m.LEFT)


def right_at(mob, x, y):
    return mob.move_to([x, y, 0], aligned_edge=m.RIGHT)


def row(label, value, color, y, xl, xr, size=22, label_color=MUTED):
    """A readout row: a label on the left, a coloured value right-aligned, on one baseline."""
    lab = left_at(code(label, size, label_color), xl, y)
    val = right_at(code(value, size, color), xr, y)
    sit(val, baseline(lab))
    return lab, val


def swap_value(old, value, color, size=22):
    """A new value Text in the same place (right edge and baseline) as `old`."""
    new = right_at(code(value, size, color), old.get_right()[0], old.get_y())
    return sit(new, baseline(old))


def stat_at(value, label, color, x, y, size=48, label_size=20):
    """A big DIN number with a MUTED label (one line, or a tuple of lines) under it, both centred on x."""
    num = title_text(value, size, color).move_to([x, y, 0])
    lines = (label,) if isinstance(label, str) else label
    lab = m.VGroup(*[text(t, label_size, MUTED) for t in lines]).arrange(m.DOWN, buff=0.07)
    lab.next_to(num, m.DOWN, buff=0.16).set_x(x)
    return num, lab


class CostChart(m.VGroup):
    """$ per million output tokens against batch size, both on log axes."""

    def __init__(self, origin=(-5.45, -2.3), width=6.6, height=4.25, xd=(0, 2.6), yd=(-1, 1)):
        super().__init__()
        self.o, self.w, self.h, self.xd, self.yd = np.array([*origin, 0.0]), width, height, xd, yd
        x_axis = m.Arrow(self.o, self.o + [width + 0.4, 0, 0], buff=0, color=FAINT, stroke_width=3,
                         max_tip_length_to_length_ratio=0.04)
        y_axis = m.Arrow(self.o, self.o + [0, height + 0.45, 0], buff=0, color=FAINT, stroke_width=3,
                         max_tip_length_to_length_ratio=0.05)
        self.axes = m.VGroup(x_axis, y_axis)
        ticks = m.VGroup()
        for b in (1, 10, 100):
            p = self.p(b, 10 ** yd[0])
            ticks.add(m.Line(p, p + [0, -0.08, 0]).set_stroke(FAINT, 2))
            ticks.add(code(str(b), 20, MUTED).next_to(p, m.DOWN, buff=0.18))
        for v, s in ((0.1, "$0.10"), (1, "$1"), (10, "$10")):
            p = self.p(1, v)
            ticks.add(m.Line(p, p + [-0.08, 0, 0]).set_stroke(FAINT, 2))
            ticks.add(right_at(code(s, 20, MUTED), p[0] - 0.15, p[1]))
        on_baseline(m.VGroup(*[t for t in ticks if isinstance(t, m.Text)][:3]))
        self.ticks = ticks
        self.x_label = text("batch size (students in one step)", 22, MUTED).move_to(
            [self.o[0] + width / 2, self.o[1] - 0.78, 0])
        self.y_label = left_at(text("$ per million output tokens", 22, MUTED), self.o[0] + 0.22, self.o[1] + height + 0.32)
        self.add(self.axes, ticks, self.x_label, self.y_label)

    def p(self, b, usd):
        fx = (np.log10(b) - self.xd[0]) / (self.xd[1] - self.xd[0])
        fy = (np.log10(usd) - self.yd[0]) / (self.yd[1] - self.yd[0])
        return self.o + np.array([fx * self.w, fy * self.h, 0])

    def curve(self, f, a, b, color, n=40, width=5):
        pts = [self.p(x, f(x)) for x in np.logspace(np.log10(a), np.log10(b), n)]
        return m.VMobject().set_points_as_corners(pts).set_stroke(color, width)


# ---------------------------------------------------------------------------- the episode


class Ep24ThePriceOfAToken(Episode):
    number = 24

    def lesson(self):
        self.hook()
        self.curve()
        self.in_out()
        self.little()
        self.utilisation()
        self.bill()
        self.levers()
        self.reasoning()
        self.recap()

    # ------------------------------------------------------------------ 1 · hook

    def hook(self):
        self.next_section(
            "1 · Owl's first GPU bill",
            notes=f"Running example: Owl (Owl-8B = Llama-3.1-8B, BF16, {W_GB:.2f} GB) now serves {STUDENTS:,} students. "
                  "The crowd is schematic (56 icons stand for 10,000 students). The bill is worked out from one "
                  "division (notes §15): $ per million tokens = $ per GPU-hour ÷ millions of tokens per hour.")
        mascot = owl(0.85).move_to([-3.25, 1.45, 0])
        crowd = m.VGroup(*[icon(USER, 0.3) for _ in range(56)]).arrange_in_grid(rows=4, cols=14, buff=(0.12, 0.12))
        crowd.move_to([-3.25, -0.6, 0])
        ask = arrow([-3.25, crowd.get_top()[1], 0], [-3.25, mascot.get_bottom()[1], 0], REQUEST)
        n_students = title_text("10,000 students", 44, REQUEST).move_to([-3.25, -2.05, 0])
        every = cite("studying with Owl, every day").next_to(n_students, m.DOWN, buff=0.18)

        box = card(5.0, 4.9, EDGE, CARD, radius=0.18, stroke=2).move_to([3.55, 0.1, 0])
        head = left_at(code("GPU BILL · OWL", 22, INK, weight=m.BOLD), 1.45, 2.1)
        sub = left_at(cite("one month · 10,000 students"), 1.45, 1.62)
        rule = m.Line([1.45, 1.22, 0], [5.65, 1.22, 0]).set_stroke(EDGE, 2)
        q = title_text("$ ?", 120, INK).move_to([3.55, -0.35, 0])
        bill_card = m.VGroup(box, head, sub, rule)

        f0 = code("$ per million tokens =", 22, INK)
        f1 = code("$ per GPU-hour", 22, INK)
        f2 = code("÷ million tokens per hour", 22, TOKEN)
        formula = m.VGroup(f0, f1, f2).arrange(m.RIGHT, buff=0.3).move_to([0, -3.2, 0])
        for f in (f1, f2):
            sit(f, baseline(f0))

        self.say(
            "[Owl has grown up]: [ten thousand students now study with it every day.]",
            m.AnimationGroup(self.new_header("01 · the bill", "Owl's first GPU bill"), m.FadeIn(mascot, shift=0.15 * m.UP)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(u, shift=0.1 * m.UP) for u in crowd], lag_ratio=0.03, run_time=1.6),
                             m.GrowArrow(ask), m.FadeIn(n_students, shift=0.1 * m.UP), m.FadeIn(every)),
        )
        self.say(
            "[And at the end of the month], [the GPU bill arrives.]",
            m.FadeIn(bill_card, shift=0.15 * m.LEFT),
            m.FadeIn(q, scale=0.8),
        )
        self.say(
            "[What will it say?] [We'll work it out from one division]: [dollars per GPU-hour], [over tokens per hour.]",
            m.Circumscribe(q, color=INK, buff=0.15),
            m.FadeIn(f0, shift=0.1 * m.UP),
            m.FadeIn(f1, shift=0.1 * m.UP),
            m.FadeIn(f2, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 2 · the cost curve

    def curve(self):
        self.next_section(
            "2 · The rent is fixed, the tokens aren't",
            notes=f"Episode 4's model, unchanged (notes §15): one H100 SXM at ${PRICE}/GPU-hour (Lambda, read 2026-10-23 — "
                  f"DATED, October 2026; RunPod $2.69–3.99); Owl-8B BF16, {CTX:,}-token chats, 75% of peak bandwidth (for "
                  f"example). step = owl_reference.step_time(batch=B, context={CTX}) ÷ 0.75. $ per M = ${PRICE} ÷ (tok/s × "
                  f"3,600 ÷ 10⁶). B = 1, 8, 64 → {', '.join(dollars(usd_per_m(b)) for b in (1, 8, 64))} "
                  f"({', '.join(f'{tps(b):,.0f}' for b in (1, 8, 64))} tok/s). Memory is full at B = {MAX_B} "
                  f"((80 − 16.06) GB ÷ 256 MiB per chat): {dollars(usd_per_m(MAX_B), 3)} per M, {tps(MAX_B):,.0f} tok/s.")
        ch = CostChart()
        X = 1.75
        kick = left_at(text("one H100, rented", 24, MUTED), X, 2.15)
        price = left_at(title_text(dollars(PRICE), 64, INK), X, 1.45)
        per_h = sit(code("per GPU-hour", 22, MUTED).next_to(price, m.RIGHT, buff=0.2), baseline(price))
        lam = left_at(cite("Lambda · H100 SXM · October 2026"), X, 0.82)
        a1 = left_at(cite("Owl-8B · BF16 · 2K-token chats"), X, 0.4)
        a2 = left_at(cite("75% of peak bandwidth · for example"), X, 0.02)

        box = card(4.8, 1.95, EDGE, CARD, radius=0.16, stroke=2).move_to([4.15, -1.45, 0])
        xl, xr = 2.1, 6.2
        spec = [("batch", "1", REQUEST), ("tokens/s", f"{tps(1):,.0f}", TOKEN), ("$ per M", dollars(usd_per_m(1)), INK)]
        labs, vals = [], []
        for (lab, v, c), yy in zip(spec, (-0.88, -1.43, -1.98)):
            a, b = row(lab, v, c, yy, xl, xr)
            labs.append(a)
            vals.append(b)
        stops = [1, 8, 64, MAX_B]
        targets = {}
        for b in stops[1:]:
            new = [f"{b:,}", f"{tps(b):,.0f}", dollars(usd_per_m(b))]
            targets[b] = [swap_value(v, t, c) for v, t, c in zip(vals, new, (REQUEST, TOKEN, INK))]

        dot = m.Dot(ch.p(1, usd_per_m(1)), radius=0.1, color=INK).set_z_index(4)
        segs = [ch.curve(usd_per_m, a, b, INK) for a, b in zip(stops, stops[1:])]
        tags = {}
        for b in stops:
            p = ch.p(b, usd_per_m(b))
            t = code(dollars(usd_per_m(b)), 20, INK)
            if b == 1:
                left_at(t, p[0] + 0.25, p[1] + 0.02)
            elif b == MAX_B:
                left_at(t, p[0] + 0.24, p[1] + 0.05)
            else:
                left_at(t, p[0] + 0.12, p[1] + 0.34)
            tags[b] = t

        def stage(k, run_time=1.2):
            b = stops[k + 1]
            return m.AnimationGroup(m.Create(segs[k]), dot.animate.move_to(ch.p(b, usd_per_m(b))),
                                    *[m.Transform(v, t) for v, t in zip(vals, targets[b])],
                                    m.FadeIn(tags[b]), run_time=run_time)

        wx = ch.p(MAX_B, 1)[0]
        wall = m.DashedLine([wx, ch.o[1], 0], [wx, ch.o[1] + ch.h, 0], dash_length=0.12).set_stroke(KV, 2.5)
        wall_lab = m.VGroup(code("memory full", 20, KV), code(f"{MAX_B} students", 20, KV)).arrange(
            m.DOWN, buff=0.1, aligned_edge=m.RIGHT)
        right_at(wall_lab, wx - 0.15, 1.45)

        self.say(
            "[As in episode four], [rent one H100 at three dollars ninety-nine an hour], [Lambda's price this October.]",
            m.AnimationGroup(self.new_header("02 · cost per token", "The rent is fixed, the tokens aren't"), m.FadeIn(ch)),
            m.AnimationGroup(m.FadeIn(kick), m.FadeIn(price, shift=0.1 * m.UP), m.FadeIn(per_h)),
            m.AnimationGroup(m.FadeIn(lam), m.FadeIn(a1), m.FadeIn(a2)),
        )
        self.say(
            "[Serving one student at a time, it makes a hundred and fifty-four tokens a second]: [seven dollars twenty per million.]",
            m.AnimationGroup(m.FadeIn(box), *[m.FadeIn(x) for x in labs], m.FadeIn(vals[0]), m.FadeIn(vals[1])),
            m.AnimationGroup(m.FadeIn(dot, scale=0.5), m.FadeIn(vals[2]), m.FadeIn(tags[1])),
        )
        self.say(
            "[Fill the batch, and the same hour makes many more tokens]: [a dollar per million at eight], [twenty-three cents at sixty-four.]",
            m.Circumscribe(m.VGroup(price, per_h), color=MUTED, buff=0.12),
            stage(0),
            stage(1),
        )
        self.say(
            "[Past that, the curve flattens], [until memory runs out at two hundred and thirty-eight students]: [about fifteen cents.]",
            stage(2, run_time=1.4),
            m.AnimationGroup(m.Create(wall), m.FadeIn(wall_lab, shift=0.1 * m.LEFT)),
            m.Circumscribe(vals[2], color=INK, buff=0.08),
        )
        self.say(
            "[The rent never changes]: [only the tokens per hour do.]",
            m.Circumscribe(m.VGroup(price, per_h), color=MUTED, buff=0.12),
            m.Circumscribe(vals[1], color=TOKEN, buff=0.08),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 3 · input vs output

    def in_out(self):
        self.next_section(
            "3 · Why output costs more",
            notes=f"Notes §15, recomputed. Prefill is compute-bound: at 50% MFU one H100 does 494.5 TFLOP/s ÷ 16.06 GFLOP "
                  f"per token = {PREFILL_TPS:,.0f} prompt tokens/s (owl_reference.prefill_time) → {dollars(USD_IN, 4)} per M "
                  f"input tokens at ${PRICE}/h. Decode is memory-bound: {dollars(usd_per_m(64))} per M at batch 64 "
                  f"(×{RATIO_64:.1f}), {dollars(usd_per_m(1))} at batch 1 (×{RATIO_1:.0f}). Bars to scale (the batch-1 bar is "
                  f"cut off). Measured on this Mac (measurements.md §11, GPU-only IOReport energy counters, 8B BF16): "
                  f"{E_IN:.1f} mJ per prompt token vs {E_OUT:.0f} mJ per generated token at batch 1 (×{E_OUT / E_IN:.1f}). "
                  "API prices (official pricing pages, DATED October 2026, notes §15): Anthropic Opus 5.5 $4/$20, OpenAI "
                  "gpt-5.5 $5/$30, gpt-4.1 $2/$8 per M tokens; cached input (notes §0 #14): Anthropic cache reads 0.1× "
                  "(0.05× on Opus/Sonnet 5.5), OpenAI up to 95% off, Gemini 0.05–0.1×. Episode 8 covers the server side.")
        base = -2.15
        s = 3.75 / usd_per_m(64)
        h_in, h_out = USD_IN * s, usd_per_m(64) * s
        xs = (-5.15, -3.15, -0.95)
        bw = 1.15
        axis = m.Line([-6.0, base, 0], [-0.25, base, 0]).set_stroke(FAINT, 2.5)
        title = left_at(cite("one H100 · $ per million tokens"), -6.0, 2.3)
        bar_in = m.Rectangle(width=bw, height=h_in).set_stroke(COMPUTE, 2.5).set_fill(AMBER_FILL, 1).move_to([xs[0], base + h_in / 2, 0])
        bar_out = m.Rectangle(width=bw, height=h_out).set_stroke(MEMORY, 2.5).set_fill(CYAN_FILL, 1).move_to([xs[1], base + h_out / 2, 0])
        v_in = code(dollars(USD_IN, 3), 22, COMPUTE).next_to(bar_in, m.UP, buff=0.15)
        v_out = code(dollars(usd_per_m(64)), 22, MEMORY).next_to(bar_out, m.UP, buff=0.15)
        n_slices = int(RATIO_64)
        slices = m.VGroup(*[m.Line([xs[1] - bw / 2, base + k * h_in, 0], [xs[1] + bw / 2, base + k * h_in, 0]).set_stroke(MEMORY, 2)
                            for k in range(1, n_slices + 1)])
        guide = m.DashedLine([xs[0] + bw / 2, base + h_in, 0], [xs[1] - bw / 2, base + h_in, 0], dash_length=0.08).set_stroke(COMPUTE, 2)
        times = title_text(f"{n_slices}×", 40, MEMORY)
        times.move_to([(xs[1] + xs[2]) / 2, base + h_out * 0.55, 0])
        top = 2.05
        ghost = m.DashedVMobject(m.Rectangle(width=bw, height=top - base), num_dashes=40).set_stroke(MEMORY, 2)
        ghost.move_to([xs[2], (top + base) / 2, 0])
        g_arrow = m.Arrow([xs[2], top - 0.2, 0], [xs[2], top + 0.45, 0], buff=0, color=MEMORY, stroke_width=4,
                          max_tip_length_to_length_ratio=0.4)
        g_val = code(dollars(usd_per_m(1)), 22, MEMORY).move_to([xs[2], 0.6, 0])
        g_x = title_text(f"{RATIO_1:.0f}×", 34, MEMORY).move_to([xs[2], 0.0, 0])

        def caps(x, name, phase, rate, color):
            a = text(name, 22, color, weight=m.BOLD).move_to([x, base - 0.35, 0])
            b = code(phase, 20, MUTED).move_to([x, base - 0.75, 0])
            c = code(rate, 20, color).move_to([x, base - 1.12, 0])
            return a, b, c

        c_in = caps(xs[0], "input", "prefill", f"{PREFILL_TPS:,.0f}/s", COMPUTE)
        c_out = caps(xs[1], "output", "batch 64", f"{tps(64):,.0f}/s", MEMORY)
        c_g = caps(xs[2], "output", "batch 1", f"{tps(1):,.0f}/s", MEMORY)
        for k in range(3):
            on_baseline(m.VGroup(c_in[k], c_out[k], c_g[k]))
        rate_cap = left_at(cite("tokens per second, one H100"), -6.0, base - 1.5)

        # right column: measured energy, API prices, cached input
        X0, X1 = 0.75, 6.45
        e_title = left_at(cite("measured on this Mac · GPU energy per token"), X0, 2.25)
        e1 = row("prompt token", f"{E_IN:.0f} mJ", COMPUTE, 1.78, X0, X1, label_color=INK)
        e2 = row("generated token", f"{E_OUT:.0f} mJ", MEMORY, 1.33, X0, X1, label_color=INK)
        api_box = card(X1 - X0 + 0.4, 2.85, EDGE, CARD, radius=0.16, stroke=2)
        api_box.move_to([(X0 + X1) / 2, -0.5, 0])
        api_title = left_at(cite("API prices · $ per million tokens · Oct 2026"), X0 + 0.2, 0.62)
        api_src = left_at(cite("Anthropic · OpenAI pricing pages"), X0 + 0.2, -1.62)
        cols = (3.95, 5.05, 6.15)
        hdr = [right_at(code(h, 20, MUTED), x, 0.18) for h, x in zip(("in", "out", ""), cols)][:2]
        api_rows = []
        for (name, i, o), y in zip(API, (-0.3, -0.75, -1.2)):
            n = left_at(code(name, 20, INK), X0 + 0.2, y)
            vi = right_at(code(f"${i}", 20, COMPUTE), cols[0], y)
            vo = right_at(code(f"${o}", 20, MEMORY), cols[1], y)
            r = right_at(code(f"{o // i}×", 20, INK, weight=m.BOLD), cols[2], y)
            for v in (vi, vo, r):
                sit(v, baseline(n))
            api_rows.append(m.VGroup(n, vi, vo, r))
        on_baseline(m.VGroup(*hdr))
        api = m.VGroup(api_box, api_title, api_src, *hdr)
        c_title = left_at(text("cached input", 22, TOKEN, weight=m.BOLD), X0, -2.3)
        c1 = left_at(code(f"Anthropic: {CACHE_READ:g}× ({CACHE_READ_55:g}× on 5.5)", 20, INK), X0, -2.75)
        c2 = left_at(code(f"OpenAI: up to {OPENAI_OFF:.0%} off", 20, INK), X0, -3.15)
        c_ep = right_at(cite("Episode 8"), X1, -2.3)
        sit(c_ep, baseline(c_title))

        self.say(
            "[Input and output tokens don't cost the same.] [Prefill is compute-bound, as in episode three]: "
            "[one H100 reads about thirty-one thousand prompt tokens a second.]",
            m.AnimationGroup(self.new_header("03 · input vs output", "Why output costs more"), m.FadeIn(axis), m.FadeIn(title)),
            m.AnimationGroup(m.FadeIn(c_in[0]), m.FadeIn(c_in[1])),
            m.AnimationGroup(m.FadeIn(c_in[2]), m.FadeIn(rate_cap)),
        )
        self.say(
            "[That's under four cents per million input tokens.]",
            m.AnimationGroup(m.GrowFromEdge(bar_in, m.DOWN), m.FadeIn(v_in, shift=0.1 * m.UP)),
        )
        self.say(
            "[Decode is memory-bound]: [even at batch sixty-four, it's twenty-three cents per million], [six times more], "
            "[and two hundred times more at batch one.]",
            m.AnimationGroup(m.FadeIn(c_out[0]), m.FadeIn(c_out[1]), m.FadeIn(c_out[2])),
            m.AnimationGroup(m.GrowFromEdge(bar_out, m.DOWN, run_time=1.2), m.FadeIn(v_out, shift=0.1 * m.UP)),
            m.AnimationGroup(m.Create(guide), m.LaggedStart(*[m.Create(x) for x in slices], lag_ratio=0.25), m.FadeIn(times)),
            m.AnimationGroup(m.Create(ghost, run_time=1.2), m.GrowArrow(g_arrow), m.FadeIn(g_val), m.FadeIn(g_x),
                             *[m.FadeIn(x) for x in c_g]),
        )
        self.say(
            "[Measured on this Mac], [a prompt token takes about a tenth of the GPU energy of a generated one.]",
            m.FadeIn(e_title),
            m.AnimationGroup(m.FadeIn(m.VGroup(*e1)), m.FadeIn(m.VGroup(*e2))),
        )
        self.say(
            "[So API price lists charge four to six times more for output than for input.]",
            m.AnimationGroup(m.FadeIn(api), m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.UP) for r in api_rows], lag_ratio=0.3)),
        )
        self.say(
            "[And a cached prompt skips prefill altogether]: [providers bill those tokens at a tenth of the input price or less, as in episode eight.]",
            m.AnimationGroup(m.FadeIn(c_title), m.FadeIn(c_ep)),
            m.AnimationGroup(m.FadeIn(c1, shift=0.1 * m.UP), m.FadeIn(c2, shift=0.1 * m.UP)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 4 · Little's law

    def little(self):
        self.next_section(
            "4 · How many GPUs at the peak?",
            notes=f"Little's law L = λW (notes §15, recomputed): λ = {LAM} questions/s at the evening peak (for example), "
                  f"W = TTFT {TTFT} s + {OUT} × TPOT(L), TPOT(L) = step_time(batch=L, context={CTX}) ÷ 0.75 on an H100. "
                  f"Fixed-point iteration from L = 0: {', '.join(f'{x:.1f}' for x in ITER[1:8])} … → {L_PEAK:.2f} ≈ "
                  f"{round(L_PEAK)} in flight; TPOT {TPOT_PEAK * 1e3:.1f} ms, W {W_PEAK:.2f} s; KV {round(L_PEAK)} × 256 MiB = "
                  f"{KV_PEAK / GB:.1f} GB + {W_GB:.2f} GB of weights = {(weight_bytes() + KV_PEAK) / GB:.1f} of 80 GB. "
                  "The dots in the box are schematic. The prefill of each prompt's new tokens fits in the spare math of these "
                  f"memory-bound steps ({NEW_IN_PEAK:,.0f} new prompt tokens/s at the peak: {STEP_MATH * 1e3:.1f} ms of math at "
                  f"50% MFU per step vs {TPOT_PEAK * MBU * 1e3:.1f} ms of memory time; chunked prefill, Episode 11).")
        # layout A: the room, the law
        room = card(5.3, 3.1, EDGE, CARD).move_to([-3.45, 0.55, 0])
        room_cap = left_at(code("one H100 · in flight", 20, MUTED), -5.85, 1.82)
        slots = [np.array([-5.55 + c * 0.6, 1.2 - r * 0.58, 0]) for r in range(4) for c in range(8)]
        rng = np.random.default_rng(24)
        order = rng.permutation(len(slots))[:26]
        dots = [m.Dot([-6.45, float(rng.uniform(-0.6, 1.4)), 0], radius=0.09, color=REQUEST).set_z_index(3) for _ in order]
        arrivals = left_at(text("20 questions arrive every second", 22, REQUEST), -6.1, -1.42)
        flow = m.LaggedStart(*[m.Succession(m.FadeIn(d, run_time=0.12), d.animate(run_time=0.55).move_to(slots[i]))
                               for d, i in zip(dots, order)], lag_ratio=0.18, run_time=3.2)
        room_all = m.VGroup(room, room_cap, arrivals, *dots)

        law = code("L = λ × W", 54, INK, t2c={"L": REQUEST, "λ": REQUEST})
        law.move_to([3.65, 1.85, 0])
        d1 = left_at(code("L · requests in flight", 20, REQUEST), 1.3, 1.08)
        d2 = left_at(code("λ · arrivals per second", 20, REQUEST), 1.3, 0.7)
        d3 = left_at(code("W · time each one stays", 20, INK), 1.3, 0.32)
        w_eq = code("W = 0.3 s + 300 × TPOT", 24, INK, t2c={r"0\.3 s": COMPUTE, r"300 × TPOT": MEMORY})
        left_at(w_eq, 1.3, -0.3)
        w_sub = left_at(cite("first token · then 300 decode steps"), 1.3, -0.72)

        # layout B: the iteration table, the answer, the memory
        cx = (-5.6, -3.85, -2.1, -0.4)
        hdr = m.VGroup(*[text(h, 20, MUTED).move_to([x, 1.85, 0]) for h, x in zip(("in flight", "per token", "per answer"), cx)],
                       code("λ × W", 20, MUTED).move_to([cx[3], 1.85, 0]))
        for h in hdr[1:]:
            sit(h, baseline(hdr[0]))
        rule = m.Line([-6.35, 1.58, 0], [0.35, 1.58, 0]).set_stroke(EDGE, 2)

        def table_row(L, y, last=False):
            t = cost_step(L)
            cells = [code(f"{L:.0f}", 22, REQUEST), code(f"{t * 1e3:.1f} ms", 22, MEMORY),
                     code(f"{TTFT + OUT * t:.1f} s", 22, INK), code(f"{in_flight(L):.0f}", 22, REQUEST)]
            for c, x in zip(cells, cx):
                c.move_to([x, y, 0])
            on_baseline(m.VGroup(*cells))
            return cells

        rows_ = [table_row(ITER[k], y) for k, y in zip(range(4), (1.2, 0.72, 0.24, -0.24))]
        more = m.VGroup(*[m.Dot([cx[0], -0.52 - 0.11 * k, 0], radius=0.03, color=MUTED) for k in range(3)])
        final = table_row(L_PEAK, -1.05)
        tick = code("✓", 24, TOKEN).next_to(final[3], m.RIGHT, buff=0.2)
        ring = m.SurroundingRectangle(m.VGroup(*final), buff=0.12, corner_radius=0.08).set_stroke(TOKEN, 2)

        xs3 = (1.75, 3.7, 5.75)
        s_l = stat_at(f"{round(L_PEAK)}", "in flight", REQUEST, xs3[0], -1.45, size=44)
        s_t = stat_at(f"{TPOT_PEAK * 1e3:.1f} ms", "per token", MEMORY, xs3[1], -1.45, size=44)
        s_w = stat_at(f"{W_PEAK:.1f} s", "per answer", INK, xs3[2], -1.45, size=44)
        for s_ in (s_t, s_w):
            s_[0].align_to(s_l[0], m.DOWN)
            s_[1].align_to(s_l[1], m.UP)

        mb = MemoryBar(GPU["memory"], width=12.4, height=0.5, title="H100 · 80 GB of memory").move_to([0, -2.9, 0])
        w_anim = mb.set("weights", weight_bytes(), WEIGHTS)
        kv_anim = mb.set("kv", KV_PEAK, KV).animations[-1]
        w_lab = code("weights 16 GB", 20, INK).move_to(mb.segments["weights"]).set_z_index(5)
        kv_lab = code(f"{round(L_PEAK)} KV caches · {KV_PEAK / GB:.0f} GB", 20, INK).move_to(mb.segments["kv"]).set_z_index(5)
        free_x = (mb.segments["kv"].get_right()[0] + mb.track.get_right()[0]) / 2
        used = code(f"{(weight_bytes() + KV_PEAK) / GB:.0f} of 80 GB", 20, MUTED).move_to([free_x, mb.track.get_y(), 0]).set_z_index(5)
        fits = pill("fits on one H100", TOKEN, GREEN_FILL, 20)
        right_at(fits, mb.track.get_right()[0], mb.caption.get_y())

        self.say(
            "[Now, how many GPUs does Owl need?] [Say that at the evening peak], [twenty questions arrive every second.]",
            m.AnimationGroup(self.new_header("04 · capacity", "How many GPUs at the peak?"), m.FadeIn(room), m.FadeIn(room_cap)),
            m.FadeIn(arrivals, shift=0.1 * m.UP),
            flow,
        )
        self.say(
            "[Little's law]: [the number of requests in flight] [equals the arrival rate], [times the time each one stays.]",
            m.FadeIn(law, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(d1), m.Circumscribe(m.VGroup(*dots), color=REQUEST, buff=0.12)),
            m.AnimationGroup(m.FadeIn(d2), m.Indicate(arrivals, color=REQUEST, scale_factor=1.05)),
            m.FadeIn(d3),
        )
        self.say(
            "[Each answer waits a third of a second for its first token], [then decodes three hundred tokens.]",
            m.FadeIn(glyphs(w_eq, "W = 0.3 s"), shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(glyphs(w_eq, "+ 300 × TPOT"), shift=0.1 * m.UP), m.FadeIn(w_sub)),
        )
        self.say(
            "[But a bigger batch makes every token slower], [so we go round again]: [forty-four in flight], [seventy-three], "
            "[ninety-one], [settling at a hundred and twenty-four.]",
            m.AnimationGroup(m.FadeOut(room_all), m.FadeIn(hdr), m.Create(rule), *[m.FadeIn(c) for c in rows_[0]]),
            m.FadeIn(rows_[1][0], target_position=rows_[0][3], path_arc=0.6),
            m.AnimationGroup(*[m.FadeIn(c) for c in rows_[1][1:]]),
            m.AnimationGroup(m.FadeIn(rows_[2][0], target_position=rows_[1][3], path_arc=0.6), *[m.FadeIn(c) for c in rows_[2][1:]]),
            m.AnimationGroup(m.FadeIn(rows_[3][0], target_position=rows_[2][3], path_arc=0.6), *[m.FadeIn(c) for c in rows_[3][1:]]),
            m.AnimationGroup(m.FadeIn(more), *[m.FadeIn(c) for c in final], m.FadeIn(tick), m.Create(ring),
                             m.FadeIn(s_l[0], shift=0.1 * m.UP), m.FadeIn(s_l[1])),
        )
        self.say(
            "[That's twenty milliseconds per token], [and about six seconds per answer.]",
            m.AnimationGroup(m.FadeIn(s_t[0], shift=0.1 * m.UP), m.FadeIn(s_t[1])),
            m.AnimationGroup(m.FadeIn(s_w[0], shift=0.1 * m.UP), m.FadeIn(s_w[1])),
        )
        self.say(
            "[With sixteen gigabytes of weights], [their caches add thirty-three]: [forty-nine of eighty], [so it all fits on one H100.]",
            m.AnimationGroup(m.FadeIn(mb.track), m.FadeIn(mb.caption), w_anim, m.FadeIn(w_lab)),
            m.AnimationGroup(kv_anim, m.FadeIn(kv_lab)),
            m.FadeIn(used),
            m.FadeIn(fits, scale=0.9),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 5 · utilisation

    def utilisation(self):
        self.next_section(
            "5 · Paid for 24 hours",
            notes=f"Illustrative day (for example): {STUDENTS:,} students × {PER_DAY} questions = {STUDENTS * PER_DAY:,} a day, "
                  f"quiet at night and peaking at {LAM}/s from 21:00 (hourly bars; the shape is invented, scaled so the day "
                  f"sums to {STUDENTS * PER_DAY:,}). Average {LAM_AVG:.2f}/s. One H100 is sized for the peak (Little's law, "
                  f"section 4) and rented 24 h: utilisation = average ÷ peak = {UTIL:.1%} (≈ {24 * UTIL:.1f} busy hours). "
                  f"notes §15: tokens/hour = aggregate tok/s × 3,600 × utilisation. At the peak the GPU makes {LAM} × {OUT} = "
                  f"{LAM * OUT:,} output tok/s → {dollars(PEAK_USD, 4)} per M; ÷ {UTIL:.3f} = {dollars(REAL_USD, 3)} per M "
                  f"(×{REAL_USD / PEAK_USD:.2f}).")
        x0, slot, y0, sy = -6.0, 0.5, -2.5, 0.19
        y_cap = y0 + LAM * sy
        x_axis = m.Line([x0, y0, 0], [x0 + 24 * slot + 0.15, y0, 0]).set_stroke(FAINT, 2.5)
        y_axis = m.Line([x0, y0, 0], [x0, y_cap + 0.3, 0]).set_stroke(FAINT, 2.5)
        ticks = m.VGroup()
        for v in (0, 10, 20):
            y = y0 + v * sy
            ticks.add(m.Line([x0 - 0.08, y, 0], [x0, y, 0]).set_stroke(FAINT, 2))
            ticks.add(right_at(code(str(v), 20, MUTED), x0 - 0.15, y))
        hours = m.VGroup(*[code(f"{h}:00", 20, MUTED).move_to([x0 + h * slot, y0 - 0.32, 0]) for h in (0, 6, 12, 18, 24)])
        on_baseline(hours)
        y_lab = left_at(text("questions per second", 22, MUTED), x0 + 0.2, 2.2)
        day_cite = right_at(cite("a day of Owl · for example"), 6.45, y0 - 0.75)
        chart = m.VGroup(x_axis, y_axis, ticks, hours, y_lab)
        bars = [m.Rectangle(width=slot * 0.76, height=max(r * sy, 0.02)).set_stroke(REQUEST, 1.5).set_fill(PINK_FILL, 1)
                .move_to([x0 + (h + 0.5) * slot, y0 + r * sy / 2, 0]).set_z_index(2) for h, r in enumerate(DAY)]

        cap = m.DashedLine([x0, y_cap, 0], [x0 + 24 * slot, y_cap, 0], dash_length=0.12).set_stroke(INK, 2.5).set_z_index(3)
        cap_lab = left_at(code("one H100 · sized for the peak · 20/s", 20, INK), x0 + 0.2, y_cap + 0.3)
        paid = m.Rectangle(width=24 * slot, height=LAM * sy).set_stroke(width=0).set_fill(MUTED, 0.1)
        paid.move_to([x0 + 12 * slot, y0 + LAM * sy / 2, 0]).set_z_index(-1)
        paid_lab = right_at(code(f"paid: 24 h × {dollars(PRICE)}", 20, MUTED), x0 + 24 * slot, y_cap + 0.3)
        sit(paid_lab, baseline(cap_lab))

        y_avg = y0 + LAM_AVG * sy
        avg = m.DashedLine([x0, y_avg, 0], [x0 + 24 * slot, y_avg, 0], dash_length=0.1).set_stroke(TOKEN, 2.5).set_z_index(3)
        avg_lab = left_at(code(f"average {LAM_AVG:.1f}/s", 20, TOKEN), x0 + 0.2, y_avg + 0.28)
        busy = pill(f"busy {UTIL:.0%} of the day", TOKEN, GREEN_FILL, 24).move_to([0, 0, 0])
        left_at(busy, x0 + 0.2, 0.55)
        f1 = left_at(code("tokens per hour = tokens/s × 3,600 × utilisation", 20, INK,
                          t2c={"utilisation": TOKEN}), x0 + 0.2, -0.2)
        f2a = left_at(code(f"{dollars(PEAK_USD)} per M ÷ {UTIL:.0%}", 22, INK, t2c={f"{UTIL:.0%}": TOKEN}), x0 + 0.2, -0.75)
        f2b = code(f"≈ {dollars(REAL_USD)} per M output tokens", 22, INK, t2c={re.escape(dollars(REAL_USD)): RED})
        sit(f2b.next_to(f2a, m.RIGHT, buff=0.25), baseline(f2a))

        self.say(
            "[But twenty a second is only the peak.] [For example, here's a day of Owl]: [quiet at night, busy in the evening.]",
            m.AnimationGroup(self.new_header("05 · utilisation", f"Paid for 24 hours, busy for {round(24 * UTIL)}"), m.FadeIn(chart)),
            m.AnimationGroup(m.FadeIn(day_cite), m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in bars[:17]], lag_ratio=0.15)),
            m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in bars[17:]], lag_ratio=0.2, run_time=1.4),
        )
        self.say(
            "[The GPU is sized for that peak], [and paid for all twenty-four hours.]",
            m.AnimationGroup(m.Create(cap), m.FadeIn(cap_lab)),
            m.AnimationGroup(m.FadeIn(paid), m.FadeIn(paid_lab)),
        )
        self.say(
            "[On average, three and a half questions arrive each second]: [the H100 is busy just seventeen percent of the day.]",
            m.AnimationGroup(m.Create(avg), m.FadeIn(avg_lab)),
            m.FadeIn(busy, scale=0.9),
        )
        self.say(
            "[Idle hours cost money too]: [divide by seventeen percent], [and every token costs about six times more.]",
            m.FadeIn(f1, shift=0.1 * m.UP),
            m.FadeIn(f2a, shift=0.1 * m.UP),
            m.FadeIn(f2b, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 6 · the bill

    def bill(self):
        self.next_section(
            "6 · Owl's monthly bill",
            notes=f"For example: {STUDENTS:,} students × {PER_DAY} questions a day × {DAYS} days = {QUESTIONS:,} questions; "
                  f"{IN_TOK:,} tokens in each ({CACHED:.0%} a cached prefix: system prompt + the student's notes, Episode 8) = "
                  f"{TOK_IN / 1e9:.0f} B in, {OUT} out each = {TOK_OUT / 1e9:.1f} B out. Little's law put the whole peak on "
                  f"{GPUS_NOW} H100 (the memory-full batch makes {tps(MAX_B):,.0f} tok/s ≥ {LAM * OUT:,} needed). Rent: "
                  f"{HOURS} h × ${PRICE} = {dollars(BILL)} (no spare replica; Episode 25 is about keeping it alive). "
                  f"If every hour were as busy as the peak: {TOK_OUT / 1e6:,.0f} M × {dollars(PEAK_USD, 4)} = "
                  f"{dollars(IDEAL)} (= bill × {UTIL:.3f}). Per student {dollars(PER_STUDENT, 3)}; per M output tokens "
                  f"{dollars(REAL_USD, 3)}. Price DATED (Lambda, Oct 2026).")
        box = card(6.2, 5.55, EDGE, CARD, radius=0.18, stroke=2).move_to([-3.3, -0.3, 0])
        xl, xr = -6.0, -0.6
        head = left_at(code("GPU BILL · OWL", 22, INK, weight=m.BOLD), xl, 2.0)
        mascot = owl(0.42).move_to([-0.95, 1.9, 0])
        ex = left_at(cite("one month · for example"), xl, 1.58)
        r_st = row("students", f"{STUDENTS:,}", REQUEST, 1.05, xl, xr)
        r_pd = row("questions a day, each", f"{PER_DAY}", REQUEST, 0.62, xl, xr)
        r_q = row("questions a month", f"{QUESTIONS:,}", REQUEST, 0.19, xl, xr)
        r_in = row(f"tokens in · {CACHED:.0%} cached", f"{TOK_IN / 1e9:.0f} B", COMPUTE, -0.24, xl, xr)
        r_out = row(f"tokens out · {OUT} each", f"{TOK_OUT / 1e9:.1f} B", TOKEN, -0.67, xl, xr)
        rule = m.Line([xl, -0.98, 0], [xr, -0.98, 0]).set_stroke(EDGE, 2)
        r_g = row("H100s at the peak", f"{GPUS_NOW}", INK, -1.3, xl, xr)
        r_h = row("hours rented", f"{HOURS}", INK, -1.73, xl, xr)
        r_p = row("price per hour", dollars(PRICE), INK, -2.16, xl, xr)
        tot_lab = left_at(text("total", 26, INK, weight=m.BOLD), xl, -2.66)
        tot = right_at(title_text(dollars(BILL), 40, INK), xr, -2.66)
        sit(tot, baseline(tot_lab))

        base, sc = -2.3, 4.0 / BILL
        bx = (1.35, 3.2)
        h_ideal, h_rent = IDEAL * sc, BILL * sc
        b_ideal = m.Rectangle(width=1.1, height=h_ideal).set_stroke(TOKEN, 2.5).set_fill(GREEN_FILL, 1).move_to([bx[0], base + h_ideal / 2, 0])
        b_rent = m.Rectangle(width=1.1, height=h_rent).set_stroke(INK, 2.5).set_fill(CARD_HI, 1).move_to([bx[1], base + h_rent / 2, 0])
        axis = m.Line([0.55, base, 0], [4.0, base, 0]).set_stroke(FAINT, 2.5)
        v_ideal = code(dollars(IDEAL, 0), 22, TOKEN).next_to(b_ideal, m.UP, buff=0.15)
        v_rent = code(dollars(BILL, 0), 22, INK).next_to(b_rent, m.UP, buff=0.15)
        l_ideal = m.VGroup(text("always as busy", 20, MUTED), text("as the peak", 20, MUTED)).arrange(m.DOWN, buff=0.08).move_to([bx[0], base - 0.5, 0])
        l_rent = m.VGroup(text("rented", 20, MUTED), text("all month", 20, MUTED)).arrange(m.DOWN, buff=0.08).move_to([bx[1], base - 0.5, 0])
        on_baseline(m.VGroup(l_ideal[0], l_rent[0]))
        on_baseline(m.VGroup(l_ideal[1], l_rent[1]))
        bend = m.CurvedArrow([bx[0] + 0.1, base + h_ideal + 0.55, 0], [bx[1] - 0.62, base + h_rent - 0.3, 0], angle=-0.7,
                             color=MUTED, stroke_width=3)
        bend_lab = code(f"÷ {UTIL:.0%}", 22, TOKEN).move_to([bx[0] - 0.05, 0.75, 0])

        s_ps = stat_at(dollars(PER_STUDENT), ("per student", "per month"), TOKEN, 5.35, 1.35, size=60)
        s_pm = stat_at(dollars(REAL_USD), ("per million", "output tokens"), INK, 5.35, -0.85, size=48)

        self.say(
            "[So here's Owl's bill.] [For example: ten thousand students], [thirty questions a day each], [nine million questions a month.]",
            m.AnimationGroup(self.new_header("06 · the bill", "Owl's monthly bill"), m.FadeIn(box), m.FadeIn(head),
                             m.FadeIn(mascot, shift=0.1 * m.UP), m.FadeIn(ex)),
            m.FadeIn(m.VGroup(*r_st)),
            m.FadeIn(m.VGroup(*r_pd)),
            m.FadeIn(m.VGroup(*r_q)),
        )
        self.say(
            "[Each sends two thousand tokens, mostly a cached prefix], [and gets three hundred back]: [two point seven billion output tokens.]",
            m.FadeIn(m.VGroup(*r_in)),
            m.FadeIn(r_out[0]),
            m.FadeIn(r_out[1], shift=0.1 * m.LEFT),
        )
        self.say(
            "[If every hour were as busy as the peak], [that would cost about five hundred dollars.]",
            m.AnimationGroup(m.FadeIn(axis), m.FadeIn(l_ideal)),
            m.AnimationGroup(m.GrowFromEdge(b_ideal, m.DOWN), m.FadeIn(v_ideal, shift=0.1 * m.UP)),
        )
        self.say(
            "[But the H100 is rented all month]: [seven hundred and twenty hours at three dollars ninety-nine], "
            "[two thousand eight hundred and seventy-three dollars.]",
            m.AnimationGroup(m.Create(rule), m.FadeIn(m.VGroup(*r_g)), m.FadeIn(l_rent)),
            m.AnimationGroup(m.FadeIn(m.VGroup(*r_h)), m.FadeIn(m.VGroup(*r_p))),
            m.AnimationGroup(m.FadeIn(tot_lab), m.FadeIn(tot, scale=0.9), m.GrowFromEdge(b_rent, m.DOWN, run_time=1.2),
                             m.FadeIn(v_rent, shift=0.1 * m.UP), m.Create(bend), m.FadeIn(bend_lab)),
        )
        self.say(
            "[About twenty-nine cents per student per month], [or a dollar six per million output tokens.]",
            m.AnimationGroup(m.FadeIn(s_ps[0], shift=0.1 * m.UP), m.FadeIn(s_ps[1])),
            m.AnimationGroup(m.FadeIn(s_pm[0], shift=0.1 * m.UP), m.FadeIn(s_pm[1])),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 7 · price cuts

    def levers(self):
        self.next_section(
            "7 · Every episode is a price cut",
            notes=f"Same price model. 4-bit (Episode 19: ≈ 4.5 bits per weight → {weight_bytes(bits=Q4_BITS) / GB:.2f} GB): "
                  f"batch 1 {dollars(usd_per_m(1))} → {dollars(usd_per_m(1, bits=Q4_BITS))} per M (×{CUT_Q4:.2f}; measured "
                  f"on this Mac: {Q4_MAC:.2f}× faster decode, measurements.md §1); at the peak (batch {L_PEAK:.0f}, KV now "
                  f"twice the weights) {dollars(usd_per_m(L_PEAK), 3)} → {dollars(usd_per_m(L_PEAK, bits=Q4_BITS), 3)} "
                  f"(−{SAVE_Q4_PEAK:.0%}). MoE (Episode 20; notes §9, Mistral 2023): Mixtral reads {MIX_ACTIVE} of {MIX_TOTAL} B "
                  f"parameters per token → ×{CUT_MOE:.2f} fewer bytes than a dense model of its size at small batches; "
                  "by batch ~16 every expert is lit (Episode 20). Speculative decoding (Episode 13): "
                  f"speculative_speedup(α = {ALPHA}, γ = {GAMMA}, c = {C_DRAFT:.3f}) = {CUT_SPEC:.2f}× at batch 1, c = Owl-mini "
                  "4-bit step ÷ Owl-8B BF16 step measured on this Mac; gains shrink at high batch (Episode 14). Utilisation "
                  "(section 5) is Owl's biggest lever: batch work in the quiet hours, or switching off (Episode 22, autoscaling).")
        cxs, cy, cw, chh = (-4.45, 0.0, 4.45), 0.62, 4.05, 3.3
        specs = [
            ("EP 18–19", "4-bit weights", f"{CUT_Q4:.1f}×", ("cheaper", "at batch 1"), "16 → 4.5 GB a step",
             f"this Mac: {Q4_MAC:.1f}× faster", WEIGHTS, BLUE_FILL),
            ("EP 20", "Mixture of experts", f"{CUT_MOE:.1f}×", ("vs a dense model", "of the same size"),
             f"reads {MIX_ACTIVE} of {MIX_TOTAL} B", "Mixtral · Mistral, 2023", MEMORY, CYAN_FILL),
            ("EP 13", "Speculative decoding", f"{CUT_SPEC:.1f}×", ("faster", "at batch 1"),
             f"{E_PASS:.2f} tokens per pass", f"α = {ALPHA} · γ = {GAMMA}", TOKEN, GREEN_FILL),
        ]
        cards = []
        for cx, (ep, name, big, cap, det, src, col, fill) in zip(cxs, specs, ):
            box = card(cw, chh, col, CARD, radius=0.18, stroke=2.5).move_to([cx, cy, 0])
            xl = cx - cw / 2 + 0.3
            chip_ = left_at(pill(ep, col, fill, 22, mono=True), xl, cy + 1.25)
            title = left_at(text(name, 24, INK, weight=m.BOLD), xl, cy + 0.68)
            num = left_at(title_text(big, 56, col), xl, cy - 0.12)
            caps = m.VGroup(*[text(c, 20, MUTED) for c in cap]).arrange(m.DOWN, buff=0.07, aligned_edge=m.LEFT)
            caps.next_to(num, m.RIGHT, buff=0.2).set_y(num.get_y())
            detail = left_at(code(det, 20, col), xl, cy - 0.88)
            source = left_at(cite(src), xl, cy - 1.3)
            cards.append(dict(box=box, chip=chip_, title=title, num=num, caps=caps, detail=detail, source=source))

        peak_lab = left_at(cite(f"at Owl's evening peak · batch {round(L_PEAK)}:"), -6.4, -1.42)
        p1 = left_at(code(f"−{SAVE_Q4_PEAK:.0%} per token", 22, WEIGHTS), cxs[0] - cw / 2 + 0.3, -1.88)
        p2 = left_at(code("every expert lit", 22, MUTED), cxs[1] - cw / 2 + 0.3, -1.88)
        p3 = left_at(code("little gain · Ep 14", 22, MUTED), cxs[2] - cw / 2 + 0.3, -1.88)
        on_baseline(m.VGroup(p1, p2, p3))
        lever = pill("Owl's biggest lever: utilisation", TOKEN, GREEN_FILL, 24)
        left_at(lever, -6.4, -2.85)
        u1 = left_at(text("fill the quiet hours with batch work", 20, INK), lever.get_right()[0] + 0.45, -2.66)
        u2 = left_at(text("or switch the GPU off when the room is empty", 20, INK), lever.get_right()[0] + 0.45, -3.06)

        def intro(c, *keys):
            return m.AnimationGroup(*[m.FadeIn(c[k]) for k in keys])

        self.say(
            "[Every lever in this course is a price cut.] [Four-bit weights, from episodes eighteen and nineteen], "
            "[make a token three point four times cheaper at batch one.]",
            self.new_header("07 · price cuts", "Every episode is a price cut"),
            m.AnimationGroup(intro(cards[0], "box", "chip", "title"), m.FadeIn(cards[0]["detail"]), m.FadeIn(cards[0]["source"])),
            m.AnimationGroup(m.FadeIn(cards[0]["num"], shift=0.1 * m.UP), m.FadeIn(cards[0]["caps"])),
        )
        self.say(
            "[A mixture of experts, from episode twenty], [reads about a quarter of its weights per token]: "
            "[three and a half times cheaper than a dense model its size.]",
            intro(cards[1], "box", "chip", "title"),
            m.AnimationGroup(m.FadeIn(cards[1]["detail"]), m.FadeIn(cards[1]["source"])),
            m.AnimationGroup(m.FadeIn(cards[1]["num"], shift=0.1 * m.UP), m.FadeIn(cards[1]["caps"])),
        )
        self.say(
            "[Speculative decoding, from episode thirteen], [makes tokens about two point seven times faster at batch one.]",
            m.AnimationGroup(intro(cards[2], "box", "chip", "title"), m.FadeIn(cards[2]["detail"]), m.FadeIn(cards[2]["source"])),
            m.AnimationGroup(m.FadeIn(cards[2]["num"], shift=0.1 * m.UP), m.FadeIn(cards[2]["caps"])),
        )
        self.say(
            "[But these cuts shrink as the batch fills], [and Owl pays for its evening peak]: [there, four-bit saves under a quarter.]",
            m.AnimationGroup(m.FadeIn(p2), m.FadeIn(p3)),
            m.FadeIn(peak_lab),
            m.AnimationGroup(m.FadeIn(p1, shift=0.1 * m.UP), m.Circumscribe(p1, color=WEIGHTS, buff=0.1)),
        )
        self.say(
            "[So Owl's biggest lever is utilisation]: [fill the quiet hours with batch work], [or switch the GPU off when the room is empty.]",
            m.FadeIn(lever, scale=0.9),
            m.FadeIn(u1, shift=0.1 * m.LEFT),
            m.FadeIn(u2, shift=0.1 * m.LEFT),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 8 · reasoning

    def reasoning(self):
        self.next_section(
            "8 · Thinking moves the bill to decode",
            notes=f"notes §17: reasoning and agent workloads shift cost to decode (DeepSeek's production day: 608 B input, "
                  f"168 B output tokens, average KV length per output token 4,989). For example: Owl thinks {THINK:,} hidden "
                  f"tokens before each {OUT}-token answer → {OUT_R:,} decode tokens per question (×{OUT_R / OUT:.1f}), with "
                  f"≈ {CTX_R:,}-token caches: memory-full batch {MAX_R} per H100, {TPS_R:,.0f} tok/s each. Peak need "
                  f"{LAM} × {OUT_R:,} = {LAM * OUT_R:,} tok/s → {GPUS_R} H100s (vs {LAM * OUT:,} tok/s → {GPUS_NOW} today; "
                  f"same capacity method). Per-user speed at that batch ≈ {1 / cost_step(MAX_R, ctx=CTX_R):.0f} tok/s.")
        sc = 9.0 / (IN_TOK + OUT_R)
        x_start = -4.3
        y1, y2, hh = 1.55, 0.5, 0.6

        def seg(x, n, color, fill, y):
            w = n * sc
            return m.Rectangle(width=w, height=hh).set_stroke(color, 2).set_fill(fill, 1).move_to([x + w / 2, y, 0])

        a_in = seg(x_start, IN_TOK, COMPUTE, AMBER_FILL, y1)
        a_out = seg(a_in.get_right()[0], OUT, MEMORY, MEMORY, y1)
        b_in = seg(x_start, IN_TOK, COMPUTE, AMBER_FILL, y2)
        b_th = seg(b_in.get_right()[0], THINK, MEMORY, CYAN_FILL, y2)
        b_out = seg(b_th.get_right()[0], OUT, MEMORY, MEMORY, y2)
        lab_today = right_at(text("today", 22, INK), x_start - 0.25, y1)
        lab_think = right_at(text("thinking", 22, INK), x_start - 0.25, y2)
        t_in = code(f"{IN_TOK:,} in", 20, INK).move_to(a_in).set_z_index(3)
        t_in2 = code(f"{IN_TOK:,} in", 20, INK).move_to(b_in).set_z_index(3)
        t_th = code(f"{THINK:,} thinking", 20, INK).move_to(b_th).set_z_index(3)
        t_out = left_at(code(f"{OUT} out", 20, MEMORY), a_out.get_right()[0] + 0.15, y1)
        t_out2 = code(f"{OUT}", 20, MEMORY).next_to(b_out, m.UP, buff=0.12)
        times = left_at(code(f"decode per question: {OUT} → {OUT_R:,} tokens · ×{OUT_R / OUT:.1f}", 22, MEMORY), x_start, -0.3)
        longer = left_at(code(f"each reading a ≈ {CTX_R // 1024}K-token KV cache", 20, KV), x_start, -0.72)

        y_a, y_b = -1.5, -2.35
        g_lab_a = right_at(text("today", 22, INK), x_start - 0.25, y_a)
        g_lab_b = right_at(text("thinking", 22, INK), x_start - 0.25, y_b)
        gpu_a = icon(GPU_ICON, 0.4).move_to([x_start + 0.28, y_a, 0])
        gpus_b = m.VGroup(*[icon(GPU_ICON, 0.4) for _ in range(GPUS_R)]).arrange(m.RIGHT, buff=0.1)
        left_at(gpus_b, x_start, y_b)
        n_a = left_at(code(f"{GPUS_NOW} H100 at the peak", 22, INK), gpu_a.get_right()[0] + 0.25, y_a)
        n_b = left_at(code(f"{GPUS_R} H100s", 22, RED), gpus_b.get_right()[0] + 0.25, y_b)
        moral = title_text("Reasoning moves the bill to decode", 36, INK, t2c={"decode": MEMORY}).move_to([0, -3.2, 0])

        self.say(
            "[One more shift is under way.] [Reasoning models think before they answer]: "
            "[for example, two thousand hidden tokens before Owl's three hundred.]",
            m.AnimationGroup(self.new_header("08 · reasoning", "Thinking moves the bill to decode"), m.FadeIn(lab_today),
                             m.GrowFromEdge(a_in, m.LEFT), m.FadeIn(t_in), m.GrowFromEdge(a_out, m.LEFT), m.FadeIn(t_out)),
            m.AnimationGroup(m.FadeIn(lab_think), m.GrowFromEdge(b_in, m.LEFT), m.FadeIn(t_in2)),
            m.AnimationGroup(m.GrowFromEdge(b_th, m.LEFT, run_time=1.4), m.FadeIn(t_th),
                             m.Succession(m.Wait(1.0), m.AnimationGroup(m.GrowFromEdge(b_out, m.LEFT), m.FadeIn(t_out2)))),
        )
        self.say(
            "[That's almost eight times the decode per question], [and every one of those tokens reads a longer cache.]",
            m.FadeIn(times, shift=0.1 * m.UP),
            m.FadeIn(longer, shift=0.1 * m.UP),
        )
        self.say(
            "[At the evening peak, Owl would need thirteen H100s instead of one]: [reasoning moves the bill to decode.]",
            m.AnimationGroup(m.FadeIn(g_lab_a), m.FadeIn(gpu_a), m.FadeIn(n_a), m.FadeIn(g_lab_b),
                             m.LaggedStart(*[m.FadeIn(g, shift=0.1 * m.UP) for g in gpus_b], lag_ratio=0.12), m.FadeIn(n_b)),
            m.FadeIn(moral, shift=0.15 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Divide the rent by the tokens per hour (and know that output costs ~6× input); fill "
                                         "the batch and the day (utilisation multiplies the price); plan for the peak with "
                                         "Little's law, L = λW.")
        halo = glow(MEMORY, radius=4.0)
        words = m.VGroup(*[title_text(w, 80, c) for w, c in (("Divide", INK), ("Fill", TOKEN), ("Plan", REQUEST))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("rent ÷ tokens per hour", "the batch and the day", "L = λ × W")])
        slots = [max(w.width, s.width) for w, s in zip(words, subs)]
        x, gap = 0.0, 1.0
        for w, s, slot in zip(words, subs, slots):
            w.move_to([x + slot / 2, 0, 0])
            s.move_to([x + slot / 2, 0, 0])
            x += slot + gap
        on_baseline(words)
        line_y = min(w.get_bottom()[1] for w in words) - 0.45
        for s in subs:
            s.set_y(line_y)
        mid = words[0][0].get_center()[1]
        dots = m.VGroup(*[m.Dot([(a.get_right()[0] + b.get_left()[0]) / 2, mid, 0], radius=0.07, color=MUTED)
                          for a, b in zip(words, words[1:])])
        row_ = m.VGroup(words, dots, subs).move_to(0.45 * m.UP)
        tagline = text("Divide the rent by the tokens. Keep the GPU busy. Plan for the peak.", 28, MUTED).next_to(
            row_, m.DOWN, buff=0.75)

        self.say(
            "[To sum up]: [divide], [fill], [plan.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[Divide the rent by the tokens, keep the GPU busy, and plan for the peak with Little's law.]",
            m.Write(tagline),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row_, tagline)], run_time=0.9)

