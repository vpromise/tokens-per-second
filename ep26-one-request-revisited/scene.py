"""Tokens per Second · Episode 26 · One Request, Revisited (Module 6 · Production & Your Mac) — the finale.

How every piece fits into one fast, cheap request. Episode 1's request is replayed — the same student, the same
"Why does Earth have seasons?", the same 1,500-token prompt and 300-token answer, the same Timeline on one H100 —
and every layer the course built is added with its episode number lit on a 26-cell course rail, while three
readouts move: tokens per second, students at once, and dollars per million tokens.

Arc: hook (Episode 1's window and timeline: 49 ms of prefill, 299 decode ticks of 6.5 ms, 1.99 s, decode 97.5%) ·
1 already inside (KV cache, Ep 5: without it 16.1 s; GQA, Ep 6: 128 KiB not 512; FlashAttention, Ep 15: 4.6 GB of
scores never written) · 2 quiet hours, batch 1 (prefix cache, Ep 8: prefill 49 → 15 ms; speculative decoding,
Ep 13: ×2.68; 4-bit weights, Ep 18–19: steps ×3.43; together ×6.28, not ×9.19, because the draft costs more against
a lighter target; her reply 0.32 s; still $1.15 per M tokens with a whole H100) · 3 evening peak, sharing (batching,
Ep 4: 64 students, $0.23; paging, Ep 7: 238 fit vs 59 with 8K reservations, $0.15; continuous batching and chunked
prefill, Ep 10–11; the evening peak, Ep 24: 20 questions/s → 124 in flight, $0.18, her reply 6.2 s) · 4 both at
once (at 124 in flight a step reads 16 GB of weights + 33 GB of caches: 4-bit trims 23% vs 71% alone; math fills
0.5% of a step alone, 20% at the peak, 102% if four guesses are checked per student: speculation waits for the quiet
hours) · 5 the stack (Ep 22 engine and stack, Ep 16 graphs, Ep 17 sampler, Ep 25 monitoring, Ep 9/12/20/21 dark
today, Ep 23 this Mac: 102 tok/s measured vs a 121 ceiling; 10,000 students on one H100 at 29 cents each a month) ·
6 the two speed limits (Episode 2's roofline: prefill on the AMBER roof; her decode at ≈1 FLOP/byte on the CYAN
slope; 4-bit → 3.4; the peak batch → 40, with 2K-token caches capping plain decode near 60; checking four guesses
each → ≈200, toward the ridge 295) · recap (cache, batch, shrink) · a course-closing card instead of "next episode".

Every number on screen is computed below from owl_reference.py (prefill_time, step_time, decode_ceiling,
weight_bytes, kv_bytes_per_token, flops_per_token, ridge, expected_tokens_per_pass, speculative_speedup) or loaded
with measured(), and asserted against the words the narrator says and the numbers earlier episodes showed.

Sources and assumptions (facts dated October 2026; research/notes.md):
- Episode 1 / notes §1: Llama-3.1-8B BF16 on one H100 SXM (989 TFLOP/s dense BF16, 3.35 TB/s, 80 GB; NVIDIA spec),
  batch 1, 50% MFU for prefill and 75% MBU for decode (assumptions), KV read at the end-of-reply size (1,800 tokens).
- Episode 4 / 24 / notes §15 price model: $3.99 per H100-hour (Lambda, read 2026-10-23, dated), 2K-token chats,
  step = step_time(batch, 2048) ÷ 0.75; Little's law at the evening peak (20 questions/s, 0.3 s to the first token,
  300-token answers) solved by fixed-point iteration; one H100 for 10,000 students: $2,872.80 a month = 29 cents each.
- Episode 7 / Kwon et al. 2023: earlier systems held tokens in only 20.4–38.2% of the KV memory; here a reservation
  of 8,192 tokens each (Episode 7's reservation) vs paged 2K caches, both on the whole 80 GB minus the weights.
- Episode 8: Owl's 1,024-token system prompt (notes §4); the rest of her prompt (476 tokens) is new.
- Episode 13: α = 0.8, γ = 4 (3.36 tokens per pass); draft cost c = Owl-mini 4-bit step ÷ Owl-8B step, both
  measured on this Mac (2.05 ms ÷ 32.3 ms BF16 → ×2.68; ÷ 9.81 ms 4-bit → ×1.83), applied to the H100 timeline as
  Episodes 13 and 24 did. Episode 14 (EAGLE-3 paper, Table 3) measured the shrinking gain at higher batch.
- Episode 15 / Dao et al. 2022: standard attention writes the N×N scores to HBM; FlashAttention never does. Her
  prompt's score matrix: 32 layers × 32 heads × 1,500² × 2 bytes = 4.6 GB.
- Episode 6: Llama-3.1-8B has 32 query heads and 8 KV heads (config.json): 128 KiB per token instead of 512 KiB.
- Episode 19: about 4.5 bits per weight (4-bit plus a BF16 scale and offset per 64) → 4.52 GB.
- Episode 23 / measurements §1 (this Mac, M4 Max 546 GB/s, mlx-lm 0.32.2): Owl-8B 4-bit decodes 101.9 tok/s;
  ceiling 546 GB/s ÷ 4.52 GB = 120.9 tok/s.
- Math share of a step: tokens × 2·P ÷ (989 TFLOP/s × 50%) over the memory-bound step time (75% MBU), the same
  utilisations as Episode 1. Intensities on the roofline: FLOPs ÷ bytes per step (weights + every sequence's KV).

Illustrative (said "for example" or schematic): the student, her notes and the 1,500/300 token counts (the course's
worked example); the guessed tokens in the speculation strip; the slot grid of continuous batching; the usage
pattern behind the evening peak (Episode 24's "for example"); the stack's layer cards.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from owl_reference import *  # noqa: E402,F403

GB = 1e9
KIB = 2 ** 10

# ---------------------------------------------------------------------------- Episode 1's request (notes §1)

N_PROMPT, N_OUT = 1500, 300
MFU, MBU = 0.5, 0.75                              # Episode 1's assumed utilisations
CONTEXT = N_PROMPT + N_OUT                        # KV read per step, the end-of-reply size (as in Episode 1)
STEPS = N_OUT - 1                                 # the first token comes from prefill's scores
T_PRE = prefill_time(N_PROMPT, mfu=MFU)
T_STEP = step_time(context=CONTEXT) / MBU
T_DEC = STEPS * T_STEP
T_TOTAL = T_PRE + T_DEC
SHARE = T_DEC / T_TOTAL
assert round(T_PRE * 1e3, 1) == 48.7 and round(T_PRE * 1e3) == 49                 # "forty-nine milliseconds"
assert round(T_STEP * 1e3, 2) == 6.49 and round(T_STEP * 1e3, 1) == 6.5           # Episode 1: 6.5 ms per step
assert round(T_DEC, 2) == 1.94 and round(T_TOTAL, 2) == 1.99 and round(T_TOTAL) == 2   # "about two seconds"
assert f"{SHARE:.1%}" == "97.5%"                                                  # "ninety-seven and a half percent"

# ---------------------------------------------------------------------------- Episode 4 / 24's price model (notes §15)

PRICE, CTX = 3.99, 2048          # $ per H100-hour (Lambda, dated Oct 2026); 2K-token chats
KV_TOK = kv_bytes_per_token()
KV_SEQ = CTX * KV_TOK            # 256 MiB per 2K-token chat


def cost_step(b, ctx=CTX, bits=16):
    """One decode step at batch b, at 75% of peak bandwidth (Episode 4)."""
    t = step_time(batch=b, context=ctx, bits=bits)
    assert t * GPU["peak"] > b * flops_per_token(model=OWL)    # memory-bound, so dividing by MBU is right
    return t / MBU


def tps(b, ctx=CTX, bits=16):
    return b / cost_step(b, ctx, bits)


def usd(tok_s):
    """$ per million output tokens = $ per GPU-hour ÷ millions of tokens per hour."""
    return PRICE / (tok_s * 3600 / 1e6)


TPS1, USD1 = tps(1), usd(tps(1))
assert round(TPS1) == 154 and f"{USD1:.2f}" == "7.20"                            # Episode 4, batch 1

# ---------------------------------------------------------------------------- 1 · already inside

NOCACHE = [prefill_time(N_PROMPT + k, mfu=MFU) for k in range(1, STEPS + 1)]     # recompute the whole sequence
T_NOCACHE = T_PRE + sum(NOCACHE)
assert round(T_NOCACHE, 1) == 16.1 and round(T_NOCACHE) == 16                    # "sixteen seconds, not two"
assert round(T_NOCACHE / T_TOTAL) == 8
KV_MHA = kv_bytes_per_token(dict(OWL, kv_heads=OWL["heads"]))                    # one K/V head per query head
assert KV_TOK == 128 * KIB and KV_MHA == 512 * KIB and OWL["heads"] == 32 and OWL["kv_heads"] == 8
SCORES = OWL["layers"] * OWL["heads"] * N_PROMPT ** 2 * 2                         # the N×N scores, BF16
assert round(SCORES / GB, 1) == 4.6                                              # "four point six gigabytes"

# ---------------------------------------------------------------------------- 2 · quiet hours (batch 1)

SYSTEM = 1024                                                                     # Episode 8's system prompt
T_PRE8 = prefill_time(N_PROMPT - SYSTEM, mfu=MFU)
assert round(T_PRE8 * 1e3) == 15 and round((T_PRE - T_PRE8) * 1e3) == 33         # "forty-nine … to fifteen"

ALPHA, GAMMA = 0.8, 4
E_PASS = expected_tokens_per_pass(ALPHA, GAMMA)
T_MINI = measured("decode", "llama-3.2-1b", "4bit", "ms_per_token")
T_BF16 = measured("decode", "llama-3.1-8b", "bf16", "ms_per_token")
T_4BIT = measured("decode", "llama-3.1-8b", "4bit", "ms_per_token")
C16, C4 = T_MINI / T_BF16, T_MINI / T_4BIT
S16, S4 = speculative_speedup(ALPHA, GAMMA, C16), speculative_speedup(ALPHA, GAMMA, C4)
assert round(E_PASS, 2) == 3.36 and round(S16, 2) == 2.68 and round(S4, 2) == 1.83   # Episode 13
PASSES = round(STEPS / E_PASS)                                                    # target passes for 299 tokens
PASS16 = (GAMMA * C16 + 1) * T_STEP                                               # one round: 4 drafts + 1 check
T_DEC13 = T_DEC / S16
assert PASSES == 89 and abs(PASSES * PASS16 - T_DEC13) < 0.002 and T_DEC13 < 0.75  # "under three quarters of a second"

Q4 = 4.5                                                                          # Episode 19: bits per weight
W4 = weight_bytes(bits=Q4)
T_STEP4 = step_time(context=CONTEXT, bits=Q4) / MBU
PASS4 = (GAMMA * C4 + 1) * T_STEP4
T_DEC19 = STEPS * T_STEP4 / S4
STEP_X = T_STEP / T_STEP4
NAIVE, REAL = S16 * STEP_X, S4 * STEP_X
assert round(W4 / GB, 2) == 4.52 and round(W4 / GB, 1) == 4.5                    # "four and a half gigabytes"
assert round(STEP_X, 2) == 3.43 and round(STEP_X, 1) == 3.4                       # "three point four times faster"
assert round(NAIVE, 1) == 9.2 and round(REAL, 1) == 6.3                           # 2.68 × 3.43 vs 1.83 × 3.43
assert abs(PASSES * PASS4 - T_DEC19) < 0.002
T_ALONE = T_PRE8 + T_DEC19
assert round(T_ALONE, 2) == 0.32 and 0.3 < T_ALONE < 1 / 3 + 0.01                # "a third of a second"
assert round(T_TOTAL / T_ALONE) == 6 and round(T_TOTAL / T_ALONE, 1) == 6.1      # "six times faster"
TPS13, TPS19 = TPS1 * S16, tps(1, bits=Q4) * S4
assert round(TPS13) == 413 and f"{usd(TPS13):.2f}" == "2.69"
assert round(TPS19) == 962 and f"{usd(TPS19):.2f}" == "1.15"                     # "a dollar fifteen"

# ---------------------------------------------------------------------------- 3 · the evening peak (sharing)

TPS64 = tps(64)
assert round(TPS64) == 4838 and f"{usd(TPS64):.2f}" == "0.23"                     # Episode 4: "twenty-three cents"
FREE = GPU["memory"] - weight_bytes()
MAX_B = int(FREE // KV_SEQ)                                                       # memory full, paged 2K caches
RESERVE = 8192                                                                    # Episode 7's reservation
R8 = int(FREE // (RESERVE * KV_TOK))
assert MAX_B == 238 and R8 == 59 and round(MAX_B / R8) == 4                       # "238 … a quarter as many" (59 on all 80 GB; Episode 7 kept 10% spare and said 52)
TPS_MAX = tps(MAX_B)
assert round(TPS_MAX, -1) == 7480 and f"{usd(TPS_MAX):.2f}" == "0.15"

LAM, TTFT, OUT = 20, 0.3, 300                                                     # Episode 24's evening peak


def in_flight(L):
    return LAM * (TTFT + OUT * cost_step(L))


L_PEAK = 0.0
for _ in range(200):
    L_PEAK = in_flight(L_PEAK)
TPOT = cost_step(L_PEAK)
W_PEAK = TTFT + OUT * TPOT
T_HER = TTFT + STEPS * TPOT                                                       # her timeline: 299 gaps after the first token
PEAK_TPS = LAM * OUT                                                              # tokens the GPU makes at the peak
PEAK_USD = usd(PEAK_TPS)
KV_PEAK = L_PEAK * KV_SEQ
assert abs(in_flight(L_PEAK) - L_PEAK) < 1e-6 and round(L_PEAK) == 124            # "a hundred and twenty-four"
assert round(TPOT * 1e3, 1) == 19.6 and round(TPOT * 1e3) == 20                   # "twenty milliseconds a token"
assert round(W_PEAK, 1) == 6.2 and round(T_HER, 1) == 6.2 and round(W_PEAK) == 6  # "about six seconds"
assert PEAK_TPS == 6000 and f"{PEAK_USD:.2f}" == "0.18"                           # "eighteen cents"
assert round(USD1 / PEAK_USD) == 39                                               # "about forty times less"
assert round(1 / TPOT) == 51                                                      # "still faster than anyone reads"
assert round(KV_PEAK / GB) == 33 and round((weight_bytes() + KV_PEAK) / GB) == 49

# ---------------------------------------------------------------------------- 4 · both at once

BYTES16 = weight_bytes() + KV_PEAK                                                # one step at the peak
BYTES4 = W4 + KV_PEAK
SAVE_PEAK = 1 - BYTES4 / BYTES16
SAVE_ALONE = 1 - T_STEP4 / T_STEP
assert round(SAVE_PEAK * 100) == 23 and SAVE_PEAK < 0.25                          # "under a quarter"
assert round(SAVE_ALONE * 100) == 71 and SAVE_ALONE > 2 / 3                        # "over two thirds"
MATH = flops_per_token() / (GPU["peak"] * MFU)                                    # one token's math, s
SHARE1 = MATH / cost_step(1)
SHARE_PK = L_PEAK * MATH / cost_step(L_PEAK)
SHARE_SPEC = (GAMMA + 1) * SHARE_PK                                               # every student checks γ + 1 tokens
assert round(SHARE1 * 100, 1) == 0.5 and round(SHARE_PK * 100) == 20 and round(SHARE_SPEC * 100) == 102
PROMPT_CUT = SYSTEM / N_PROMPT
assert round(PROMPT_CUT * 100) == 68

# ---------------------------------------------------------------------------- 5 · the stack and this Mac

MAC_TOK = measured("decode", "llama-3.1-8b", "4bit", "tok_s_median")
MAC_W4 = measured("decode", "llama-3.1-8b", "4bit", "weights_gb")
MAC_CEIL = MAC["bandwidth"] / (MAC_W4 * GB)
assert round(MAC_TOK) == 102 and round(MAC_CEIL) == 121                           # "a hundred and two … a hundred and twenty-one"
assert round(decode_ceiling(hw=MAC, bits=Q4)) == 121
STUDENTS = 10_000
PER_STUDENT = 1 * 24 * 30 * PRICE / STUDENTS                                      # one H100 for a 30-day month
assert f"{PER_STUDENT:.2f}" == "0.29" and math.ceil(PEAK_TPS / TPS_MAX) == 1       # "twenty-nine cents"
KV_USE = KV_PEAK / FREE
assert round(KV_USE * 100) == 52

# ---------------------------------------------------------------------------- 6 · the two speed limits

RIDGE = ridge(GPU)
I_ALONE = flops_per_token() / (weight_bytes() + CONTEXT * KV_TOK)
I_4BIT = flops_per_token() / (W4 + CONTEXT * KV_TOK)
I_PEAK = L_PEAK * flops_per_token() / BYTES16
I_SPEC = (GAMMA + 1) * I_PEAK                                                     # same bytes, five tokens each
I_CAP = flops_per_token() / KV_SEQ                                                # batch → ∞: the caches set the bytes
assert round(RIDGE) == 295 and round(I_ALONE) == 1 and round(I_4BIT, 1) == 3.4
assert round(I_PEAK) == 40 and round(I_SPEC, -1) == 200 and round(I_CAP) == 60 and I_SPEC < RIDGE

# ---------------------------------------------------------------------------- look

PREFIX = "ep26"
DOC_ICON = recolored("doc", PINK, PINK_FILL, prefix=PREFIX)
GPU_ICON = recolored("gpu", MUTED, CARD_HI, prefix=PREFIX)
MAC_ICON = recolored("mac", MUTED, CARD_HI, prefix=PREFIX)
QUESTION = "Why does Earth have seasons?"
FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPRSTUVWXYZ0123456789")

PILL_Y, VIS_Y = 0.5, -0.3          # the layer badge row and the mechanism row
TL_KW = dict(width=11.0, origin=(-5.5, -2.45, 0), lane_height=0.7)


def cite(s, size=20):
    return text(s, size, MUTED)


def baseline(words):
    chars = [c for c in words.text if not c.isspace()]
    if len(chars) == len(words):
        bottoms = sorted(g.get_bottom()[1] for g, c in zip(words, chars) if c in FLAT)
        if bottoms:
            return bottoms[len(bottoms) // 2]
    return float(np.median([g.get_bottom()[1] for g in words]))


def sit(words, y):
    return words.shift((y - baseline(words)) * m.UP)


def mono(s, size=20, color=INK, rules=None, **kw):
    """code() with literal-substring colouring ({substring: colour}, in priority order)."""
    return code(s, size, color, t2c={re.escape(k): v for k, v in (rules or {}).items()}, **kw)


def left_at(mob, x, y):
    return mob.move_to([x, y, 0], aligned_edge=m.LEFT)


def module_of(n):
    return MODULES[EPISODES[n][1]]


def badge(n, label):
    _, color, fill = module_of(n)
    return pill(f"EP {n} · {label}", color, fill, 22, mono=True)


def cell(n, lit, size=0.4):
    _, color, fill = module_of(n)
    box = card(size, size, color if lit else EDGE, fill if lit else CARD, radius=0.07, stroke=2.2 if lit else 1.5)
    num = code(str(n), 20, INK if lit else FAINT, weight=m.BOLD if lit else m.NORMAL).move_to(box)
    return m.VGroup(box, num)


class Rail:
    """The whole course as 26 numbered cells in module colours, lit as each episode's layer is added."""

    def __init__(self, y=2.32, size=0.4, gap=0.05, module_gap=0.2):
        xs, x = [], 0.0
        for n in range(1, 27):
            if n > 1:
                x += size + gap + (module_gap if EPISODES[n][1] != EPISODES[n - 1][1] else 0)
            xs.append(x)
        mid = (xs[0] + xs[-1]) / 2
        self.x = {n: xs[n - 1] - mid for n in range(1, 27)}
        self.y = y
        self.dim = {n: cell(n, False, size).move_to([self.x[n], y, 0]) for n in range(1, 27)}
        self.lit = {n: cell(n, True, size).move_to([self.x[n], y, 0]).set_z_index(1) for n in range(1, 27)}
        self.on = set()

    def intro(self):
        return m.LaggedStart(*[m.FadeIn(self.dim[n], shift=0.1 * m.DOWN) for n in range(1, 27)], lag_ratio=0.04)

    def light(self, *ns):
        anims = []
        for n in ns:
            anims += [m.FadeOut(self.dim[n]), m.FadeIn(self.lit[n], scale=1.5)]
            self.on.add(n)
        return m.AnimationGroup(*anims)

    def shown(self):
        return [self.lit[n] if n in self.on else self.dim[n] for n in range(1, 27)]


class Readout:
    """A caption and a big value that can be swapped (tokens/s, students, $ per M)."""

    def __init__(self, x, caption, color, value):
        self.x, self.color = x, color
        self.cap = code(caption, 20, MUTED).move_to([x, 1.78, 0])
        self.value = self._make(value)

    def _make(self, s):
        return sit(title_text(s, 50, self.color).move_to([self.x, 1.2, 0]), 0.98)

    def to(self, s):
        old, new = self.value, self._make(s)
        self.value = new
        return m.AnimationGroup(m.FadeOut(old, shift=0.25 * m.UP), m.FadeIn(new, shift=0.25 * m.UP))


def bracket(x0, x1, y, color=INK, tick=0.12):
    return m.VGroup(m.Line([x0, y, 0], [x1, y, 0]), m.Line([x0, y, 0], [x0, y - tick, 0]),
                    m.Line([x1, y, 0], [x1, y - tick, 0])).set_stroke(color, 2.5)


def ticks_at(tl, times, color=MEMORY, width=1.8):
    return m.VGroup(*[tl.tick(t, color).set_stroke(color, width) for t in times])


def hbar(x0, y, width, height, color, opacity=1.0):
    r = m.Rectangle(width=max(width, 0.02), height=height).set_stroke(width=0).set_fill(color, opacity)
    return r.move_to([x0 + max(width, 0.02) / 2, y, 0])


def layer_card(title, sub, width=5.3, height=0.62, color=EDGE):
    box = card(width, height, color, CARD, radius=0.14, stroke=2)
    t = text(title, 24, INK, weight=m.BOLD)
    s = code(sub, 20, MUTED)
    row = m.VGroup(t, s).arrange(m.RIGHT, buff=0.3)
    row.move_to(box).align_to(box, m.LEFT).shift(0.3 * m.RIGHT)
    y = box.get_y() - 0.09
    sit(t, y)
    sit(s, y)
    return m.VGroup(box, row)


# ---------------------------------------------------------------------------- the episode


class Ep26OneRequestRevisited(Episode):
    number = 26

    def lesson(self):
        self.rail = Rail()
        self.hook()
        self.inside()
        self.quiet()
        self.peak()
        self.both()
        self.stack()
        self.limits()
        self.recap()

    def keep(self, *extra):
        """Everything that stays between sections: the rail (and whatever else is passed)."""
        return [*self.rail.shown(), *extra]

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes=f"Episode 1's request, replayed: the student pastes her earth-science notes into Owl and asks '{QUESTION}' (Episode 1's window). Episode 1's timeline on one H100 (notes §1): prefill = 2·P·1,500 = {flops_per_token() * N_PROMPT / 1e12:.1f} TFLOP at 50% MFU = {T_PRE * 1e3:.1f} ms; each decode step reads {weight_bytes() / GB:.2f} GB + {CONTEXT * KV_TOK / GB:.2f} GB of KV at 75% MBU = {T_STEP * 1e3:.2f} ms; {STEPS} steps = {T_DEC:.2f} s; total {T_TOTAL:.2f} s; decode {SHARE:.1%}. The readouts use Episode 4's price model ($3.99 per H100-hour, Lambda, dated October 2026; 2K-token chats; 75% MBU): {TPS1:.0f} tok/s, ${USD1:.2f} per million output tokens. The rail: the course's 26 episodes in their module colours; 1–3 (the request, the two speed limits, prefill vs decode) are the base.")
        win = card(9.2, 4.3, EDGE, CARD, radius=0.22, stroke=2).move_to([0, -0.4, 0])
        top = win.get_top()[1]
        rule = m.Line([win.get_left()[0], top - 0.66, 0], [win.get_right()[0], top - 0.66, 0]).set_stroke(EDGE, 2)
        face = owl(0.36).move_to([win.get_left()[0] + 0.55, top - 0.33, 0])
        name = title_text("Owl", 28).next_to(face, m.RIGHT, buff=0.22)
        role = text("study buddy", 22, MUTED).next_to(name, m.RIGHT, buff=0.22)
        sit(name, top - 0.43)
        sit(role, top - 0.43)
        field_ = card(8.3, 0.72, EDGE, "#0A1013", radius=0.14, stroke=1.5).move_to([0, win.get_bottom()[1] + 0.56, 0])
        key = m.VGroup(card(1.15, 0.48, MUTED, CARD_HI, radius=0.1, stroke=1.5), code("Enter", 20, INK))
        key[1].move_to(key[0])
        key.move_to(field_).align_to(field_, m.RIGHT).shift(0.14 * m.LEFT)
        typed = text(QUESTION, 26, INK).move_to(field_).align_to(field_, m.LEFT).shift(0.3 * m.RIGHT)
        sit(typed, field_.get_y() - 0.1)
        attach = m.VGroup(icon(DOC_ICON, 0.36), code("earth-science-notes.txt", 20, MUTED)).arrange(m.RIGHT, buff=0.14)
        attach_box = card(attach.width + 0.4, 0.52, EDGE, CARD_HI, radius=0.12, stroke=1.5)
        attach = m.VGroup(attach_box, attach.move_to(attach_box))
        attach.next_to(field_, m.UP, buff=0.14).align_to(field_, m.LEFT)
        window = m.VGroup(win, rule, face, name, role, field_, key)

        q_words = text(QUESTION, 26, INK)
        u_label = code("student", 22, PINK, weight=m.BOLD)
        u_file = code("earth-science-notes.txt", 20, MUTED)
        u_note = m.VGroup(u_label, icon(DOC_ICON, 0.3), u_file).arrange(m.RIGHT, buff=0.16)
        sit(u_file, baseline(u_label))
        u_inner = m.VGroup(u_note, q_words).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.16)
        u_box = card(u_inner.width + 0.75, u_inner.height + 0.4, EDGE, CARD_HI, radius=0.14, stroke=1.5)
        u_stripe = m.RoundedRectangle(corner_radius=0.04, width=0.08, height=u_box.height - 0.18, stroke_width=0,
                                      fill_color=PINK, fill_opacity=1).move_to(u_box).align_to(u_box, m.LEFT).shift(0.1 * m.RIGHT)
        u_inner.move_to(u_box).align_to(u_box, m.LEFT).shift(0.35 * m.RIGHT)
        user_msg = m.VGroup(u_box, u_stripe, u_inner)
        user_msg.align_to([win.get_right()[0] - 0.35, 0, 0], m.RIGHT).align_to([0, top - 0.86, 0], m.UP)

        # Episode 1's tags and timeline, exactly as it drew them
        tags = m.VGroup(
            pill("Owl-8B · BF16", WEIGHTS, BLUE_FILL, 22, mono=True),
            pill("one H100", MUTED, CARD, 22, mono=True),
            pill(f"prompt {N_PROMPT:,} tokens", REQUEST, PINK_FILL, 22, mono=True),
            pill(f"answer {N_OUT} tokens", TOKEN, GREEN_FILL, 22, mono=True),
        ).arrange(m.RIGHT, buff=0.3).move_to([0, 2.2, 0])
        tl = Timeline(2.0, label_every=0.5, **TL_KW)
        self.tl = tl
        self.pre = tl.bar(0, T_PRE, COMPUTE)
        self.ticks = ticks_at(tl, [T_PRE + k * T_STEP for k in range(1, STEPS + 1)])
        self.pre_lab = code(f"prefill {T_PRE * 1e3:.0f} ms", 20, COMPUTE).next_to(self.pre, m.UP, buff=0.22).align_to(self.pre, m.LEFT)
        self.dec_lab = code(f"{STEPS} decode steps × {T_STEP * 1e3:.1f} ms = {T_DEC:.2f} s", 20, MEMORY)
        self.dec_lab.next_to(self.ticks, m.UP, buff=0.22).align_to(self.ticks, m.RIGHT)
        sit(self.dec_lab, baseline(self.pre_lab))

        x0, x1 = tl.x(0)[0], tl.x(T_TOTAL)[0]
        total_br = bracket(x0, x1, -0.86, INK)
        q_spot = [-6.4 + user_msg.width / 2, 0.55, 0]
        total_lab = mono(f"≈ {T_TOTAL:.2f} s for her reply · decode {SHARE:.1%} of it", 22,
                         rules={f"≈ {T_TOTAL:.2f} s": INK, f"decode {SHARE:.1%}": MEMORY}).next_to(total_br, m.UP, buff=0.18)

        rail = self.rail
        self.meters = [Readout(-4.4, "tokens per second", TOKEN, f"{TPS1:.0f}"),
                       Readout(0.0, "students at once", REQUEST, "1"),
                       Readout(4.4, "$ per million tokens", INK, f"${USD1:.2f}")]
        price_src = cite("Ep 4's price model: $3.99 per H100-hour · 75% of bandwidth").to_corner(m.DR, buff=0.5)
        self.price_src = price_src

        self.say(
            "[Twenty-five episodes ago], [a student pasted her notes into Owl], [asked why Earth has seasons], [and pressed Enter.]",
            m.AnimationGroup(self.new_header("00 · hook", "The same request, revisited"), m.FadeIn(window, shift=0.2 * m.UP)),
            m.FadeIn(attach, shift=0.15 * m.UP),
            m.AddTextLetterByLetter(typed, time_per_char=0.04),
            m.Succession(
                m.Indicate(key, color=INK, scale_factor=0.86, run_time=0.45),
                m.AnimationGroup(m.FadeOut(typed, shift=0.3 * m.UP), m.FadeOut(attach, shift=0.3 * m.UP),
                                 m.FadeIn(user_msg, shift=0.4 * m.UP), run_time=0.7),
            ),
        )
        self.say(
            "[On a single H100], [fifteen hundred tokens went in]: [one thick prefill bar], [then two hundred and ninety-nine thin decode ticks.]",
            m.AnimationGroup(m.FadeOut(window, shift=0.3 * m.UP, scale=0.85), user_msg.animate.move_to(q_spot), m.FadeIn(tl),
                             m.FadeIn(tags[0], shift=0.1 * m.UP),
                             m.FadeIn(tags[1], shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(tags[2], shift=0.1 * m.UP), m.FadeIn(tags[3], shift=0.1 * m.UP)),
            m.AnimationGroup(m.GrowFromEdge(self.pre, m.LEFT), m.FadeIn(self.pre_lab, shift=0.1 * m.UP)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(t) for t in self.ticks], lag_ratio=0.03, run_time=2.4),
                             m.FadeIn(self.dec_lab, shift=0.1 * m.UP)),
        )
        self.say(
            "[About two seconds in all], [and decode took ninety-seven and a half percent of it.]",
            m.AnimationGroup(m.Create(total_br), m.FadeIn(total_lab[: len(total_lab) - len(f"·decode{SHARE:.1%}ofit")], shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(total_lab[len(total_lab) - len(f"·decode{SHARE:.1%}ofit"):], shift=0.1 * m.UP),
                             m.Circumscribe(self.ticks, color=MEMORY, buff=0.08)),
        )
        self.say(
            "[Today we replay that request], [adding each layer the course built], [badged with its episode.]",
            m.AnimationGroup(m.FadeOut(tags, shift=0.2 * m.UP), m.FadeOut(total_br), m.FadeOut(total_lab), m.FadeOut(user_msg, shift=0.2 * m.UP)),
            rail.intro(),
            m.LaggedStart(rail.light(1), rail.light(2), rail.light(3), lag_ratio=0.3),
        )
        mt = self.meters
        self.say(
            "[And we watch three numbers]: [tokens per second], [students served at once], [and the price of a million tokens.]",
            m.FadeIn(price_src),
            m.AnimationGroup(m.FadeIn(mt[0].cap), m.FadeIn(mt[0].value, shift=0.15 * m.UP)),
            m.AnimationGroup(m.FadeIn(mt[1].cap), m.FadeIn(mt[1].value, shift=0.15 * m.UP)),
            m.AnimationGroup(m.FadeIn(mt[2].cap), m.FadeIn(mt[2].value, shift=0.15 * m.UP)),
        )
        self.wait(0.4)

    def dashboard(self):
        mt = self.meters
        return [*self.rail.shown(), *[r.cap for r in mt], *[r.value for r in mt], self.price_src]

    def timeline_parts(self):
        return [self.tl, self.pre, self.ticks, self.pre_lab, self.dec_lab]

    # ------------------------------------------------------------------ 1 · already inside

    def inside(self):
        self.next_section("1 · Already inside", notes=f"Three layers Episode 1's timeline already assumed. KV cache (Episode 5): without it, step k recomputes the whole sequence (1,500 + k tokens, prefill_time at 50% MFU, {NOCACHE[0] * 1e3:.1f} → {NOCACHE[-1] * 1e3:.1f} ms a step): decode {sum(NOCACHE):.1f} s, reply {T_NOCACHE:.1f} s instead of {T_TOTAL:.2f} s (×{T_NOCACHE / T_TOTAL:.1f}); the amber bars are the first steps of that, drawn to scale. Grouped-query attention (Episode 6): Llama-3.1-8B's 32 query heads share 8 KV heads (config.json), so its cache is 2 × 32 × 8 × 128 × 2 B = 128 KiB per token; one K/V head per query head would be 512 KiB. FlashAttention (Episode 15; Dao et al. 2022): standard attention writes the N×N score matrix to HBM and reads it back; for her prompt that is 32 layers × 32 heads × 1,500² × 2 B = {SCORES / GB:.2f} GB; FlashAttention computes it tile by tile in on-chip SRAM and never writes it (the tile walk is schematic). Episode 1's 49 ms ignored attention, which FlashAttention keeps small at this length.")
        rail, tl = self.rail, self.tl
        p5 = badge(5, "KV cache")
        p6 = badge(6, "shrinking the cache")
        p15 = badge(15, "FlashAttention")
        for p in (p5, p6, p15):
            left_at(p, -6.4, PILL_Y)

        # no cache: every step recomputes the whole sequence (amber bars, to scale, running off the axis)
        ghost, t = [], T_PRE
        for d in NOCACHE:
            ghost.append(tl.bar(t + 0.0015, min(t + d, 2.0) - 0.0015, COMPUTE).set_z_index(2))   # the last one runs off the axis
            t += d
            if t >= 2.0:
                break
        ghost = m.VGroup(*ghost)
        end = tl.x(2.0)
        more = arrow(end + np.array([0.05, 0, 0]), end + np.array([0.95, 0, 0]), RED, buff=0.0)
        g1 = left_at(mono("without it, each step recomputes all 1,500 + k tokens", 20, MUTED,
                          rules={"recomputes all 1,500 + k tokens": COMPUTE}), -6.4, VIS_Y + 0.22)
        g2 = left_at(mono(f"her reply: {T_NOCACHE:.1f} s, not {T_TOTAL:.2f} s", 20, MUTED,
                          rules={f"{T_NOCACHE:.1f} s": RED}), -6.4, VIS_Y - 0.22)

        # GQA: 32 query heads share 8 KV heads
        qx0, qdx = -6.25, 0.125
        qd = m.VGroup(*[m.Dot([qx0 + i * qdx, VIS_Y + 0.25, 0], radius=0.04, color=MUTED) for i in range(32)])
        kd = m.VGroup(*[m.Dot([qx0 + (4 * j + 1.5) * qdx, VIS_Y - 0.3, 0], radius=0.07, color=KV) for j in range(8)])
        links = m.VGroup(*[m.Line(qd[i].get_center(), kd[i // 4].get_center(), buff=0.07).set_stroke(FAINT, 1.5) for i in range(32)])
        gq1 = left_at(mono("32 query heads share 8 K/V heads", 20, MUTED, rules={"8 K/V heads": KV}), -1.85, VIS_Y + 0.22)
        gq2 = left_at(mono(f"{int(KV_TOK // KIB)} KiB per token, not {int(KV_MHA // KIB)} KiB", 20, MUTED,
                           rules={f"{int(KV_TOK // KIB)} KiB per token": KV}), -1.85, VIS_Y - 0.22)
        gq_src = cite("Llama 3.1 config").move_to([0, VIS_Y - 0.22, 0]).align_to([6.5, 0, 0], m.RIGHT)

        # FlashAttention: the N×N scores never leave the chip
        grid = cell_grid(6, 6, size=0.14, gap=0.03, color=MEMORY, fill=CYAN_FILL).move_to([-5.75, VIS_Y, 0])
        step_ = 0.17
        corner = grid[0][0].get_center()
        spots = [corner + np.array([(c + 0.5) * step_, -(r + 0.5) * step_, 0]) for r, c in ((0, 0), (0, 2), (0, 4), (2, 0), (2, 2), (2, 4), (4, 4))]
        tile = m.Square(0.33).set_stroke(TOKEN, 3).set_fill(TOKEN, 0.15).move_to(spots[0]).set_z_index(2)
        fa1 = left_at(mono(f"{SCORES / GB:.1f} GB of scores for her prompt", 20, MUTED,
                           rules={f"{SCORES / GB:.1f} GB of scores": MEMORY}), -5.0, VIS_Y + 0.22)
        fa2 = left_at(text("computed tile by tile on the chip, never written to memory", 20, MUTED), -5.0, VIS_Y - 0.22)
        fa_src = cite("Dao et al., 2022").move_to([0, VIS_Y - 0.22, 0]).align_to([6.5, 0, 0], m.RIGHT)

        self.say(
            "[Three layers were hiding in that timeline already.] [First, the KV cache, from episode five.]",
            self.new_header("01 · already inside", "Three layers were already there"),
            m.AnimationGroup(rail.light(5), m.FadeIn(p5, shift=0.1 * m.UP)),
        )
        self.say(
            "[Without it, every step would recompute all fifteen hundred tokens], [and her reply would take sixteen seconds, not two.]",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(b) for b in ghost], lag_ratio=0.12, run_time=2.4), m.FadeIn(g1, shift=0.1 * m.UP)),
            m.AnimationGroup(m.GrowArrow(more), m.FadeIn(g2, shift=0.1 * m.UP)),
        )
        self.say(
            "[Second, from episode six]: [Owl's attention heads share their keys and values], "
            "[so each token's cache is a hundred and twenty-eight K, not five hundred and twelve.]",
            m.AnimationGroup(m.FadeOut(ghost), m.FadeOut(more), m.FadeOut(g1), m.FadeOut(g2), m.FadeOut(p5),
                             rail.light(6), m.FadeIn(p6, shift=0.1 * m.UP)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in qd], lag_ratio=0.03), m.Create(links),
                             m.FadeIn(kd, scale=0.5), m.FadeIn(gq1)),
            m.AnimationGroup(m.FadeIn(gq2, shift=0.1 * m.UP), m.FadeIn(gq_src)),
        )
        self.say(
            "[Third, from episode fifteen, FlashAttention computes attention in tiles], "
            "[so the four point six gigabytes of scores for her prompt are never written out.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(qd, kd, links, gq1, gq2, gq_src, p6)), rail.light(15), m.FadeIn(p15, shift=0.1 * m.UP),
                             m.FadeIn(grid),
                             m.Succession(m.FadeIn(tile, run_time=0.4), *[tile.animate(run_time=0.38).move_to(p) for p in spots[1:]])),
            m.AnimationGroup(m.FadeIn(fa1, shift=0.1 * m.UP), m.FadeIn(fa2, shift=0.1 * m.UP), m.FadeIn(fa_src)),
        )
        self.wait(0.3)
        self.play(*[m.FadeOut(x) for x in (p15, grid, tile, fa1, fa2, fa_src)], run_time=0.7)

    # ------------------------------------------------------------------ 2 · quiet hours

    def quiet(self):
        self.next_section("2 · Quiet hours", notes=f"Batch 1: she is alone on the H100. Prefix cache (Episode 8): Owl's 1,024-token system prompt is cached, so prefill covers the other {N_PROMPT - SYSTEM} tokens: {T_PRE * 1e3:.1f} → {T_PRE8 * 1e3:.1f} ms. Speculative decoding (Episode 13): α = {ALPHA}, γ = {GAMMA}: {E_PASS:.2f} tokens per check; c = {T_MINI:.2f} ms ÷ {T_BF16:.1f} ms (measured on this Mac) = {C16:.3f}, speedup {S16:.2f}×: {PASSES} rounds of 4 drafts + 1 check = {PASS16 * 1e3:.2f} ms each, decode {T_DEC13:.3f} s (the strip's guesses are illustrative). 4-bit weights (Episodes 18–19, 4.5 bits per weight): {W4 / GB:.2f} GB, a step {T_STEP4 * 1e3:.2f} ms ({STEP_X:.2f}× faster). Together not {S16:.2f} × {STEP_X:.2f} = {NAIVE:.2f}×: against the 4-bit target the draft costs c = {C4:.3f} and speculation gives {S4:.2f}× (Episode 13), so {REAL:.2f}×: {PASSES} rounds of {PASS4 * 1e3:.2f} ms, decode {T_DEC19:.3f} s, reply {T_ALONE:.3f} s ({T_TOTAL / T_ALONE:.1f}× sooner). Readouts (Episode 4's model, 2K context): {TPS1:.0f} → {TPS13:.0f} → {TPS19:.0f} tok/s; ${USD1:.2f} → ${usd(TPS13):.2f} → ${usd(TPS19):.2f} per M (one student pays for the whole GPU).")
        rail, tl, mt = self.rail, self.tl, self.meters
        alone = left_at(pill("quiet hours · she's alone · batch 1", REQUEST, PINK_FILL, 22, mono=True), -6.4, PILL_Y)
        p8 = left_at(badge(8, "never prefill twice"), -6.4, PILL_Y)
        p13 = left_at(badge(13, "speculative decoding"), -6.4, PILL_Y)
        p19 = left_at(pill("EP 18–19 · four bits", WEIGHTS, BLUE_FILL, 22, mono=True), -6.4, PILL_Y)

        # prefix cache: the system prompt's share of the prefill bar is already there
        frac = SYSTEM / N_PROMPT
        cached = m.Rectangle(width=self.pre.width * frac, height=self.pre.height).set_stroke(KV, 1.5).set_fill(VIOLET_FILL, 1)
        cached.align_to(self.pre, m.LEFT).set_y(self.pre.get_y()).set_z_index(2)
        pre8 = tl.bar(0, T_PRE8, COMPUTE)
        dx = tl.x(T_PRE8)[0] - tl.x(T_PRE)[0]
        lab8 = code(f"prefill {T_PRE8 * 1e3:.0f} ms", 20, COMPUTE).move_to(self.pre_lab, aligned_edge=m.LEFT)
        def chip_of(s, color, fill):
            words = code(s, 20, INK)
            box = card(words.width + 0.45, 0.5, color, fill, radius=0.12, stroke=2)
            return m.VGroup(box, words.move_to(box))

        sys_chip = chip_of(f"system prompt · {SYSTEM:,} cached", KV, VIOLET_FILL)
        new_chip = chip_of(f"her notes + question · {N_PROMPT - SYSTEM}", COMPUTE, AMBER_FILL)
        chips = m.VGroup(sys_chip, new_chip).arrange(m.RIGHT, buff=0.15)
        left_at(chips, -6.4, VIS_Y)

        # speculative decoding: four guesses, one check (illustrative tokens from Owl's answer)
        drafts = token_row([" come", " from", " the", " Sun"], TOKEN, size=22, gap=0.1)
        left_at(drafts, -6.4, VIS_Y + 0.05)
        verdict = [TOKEN, TOKEN, RED, FAINT]
        judged = m.VGroup(*[token_row([t], c, size=22)[0].move_to(d) for t, c, d in zip([" come", " from", " the", " Sun"], verdict, drafts)])
        bonus = token_row([" Earth"], TOKEN, size=22)[0].next_to(drafts, m.RIGHT, buff=0.35)
        plus = code("+", 26, TOKEN, weight=m.BOLD).next_to(bonus, m.LEFT, buff=0.1)
        bonus = m.VGroup(plus, bonus)
        bonus.next_to(drafts, m.RIGHT, buff=0.2)
        sd1 = left_at(mono(f"{E_PASS:.2f} tokens per check, on average", 20, MUTED, rules={f"{E_PASS:.2f} tokens": TOKEN}),
                      bonus.get_right()[0] + 0.45, VIS_Y + 0.22)
        sd2 = left_at(code(f"α = {ALPHA} · γ = {GAMMA} · Ep 13", 20, MUTED), bonus.get_right()[0] + 0.45, VIS_Y - 0.22)

        # rounds of speculation on the timeline
        t13 = [T_PRE8 + (k + 1) * PASS16 for k in range(PASSES)]
        ticks13 = ticks_at(tl, t13, width=2.2)
        dec13 = code(f"{PASSES} checks × {PASS16 * 1e3:.1f} ms = {T_DEC13:.2f} s", 20, MEMORY).next_to(ticks13, m.RIGHT, buff=0.25)
        dec13.set_y(tl.x(0)[1])

        # 4-bit: each step reads 4.5 GB, not 16
        wscale = 0.3
        wbar = hbar(-6.4, VIS_Y, weight_bytes() / GB * wscale, 0.4, WEIGHTS)
        wbar4 = hbar(-6.4, VIS_Y, W4 / GB * wscale, 0.4, WEIGHTS)
        wlab = code(f"{weight_bytes() / GB:.2f} GB", 20, WEIGHTS).next_to(wbar, m.RIGHT, buff=0.15)
        wlab4 = code(f"{W4 / GB:.2f} GB · {Q4} bits a weight", 20, WEIGHTS).next_to(wbar4, m.RIGHT, buff=0.15)
        t19 = [T_PRE8 + (k + 1) * PASS4 for k in range(PASSES)]
        ticks19 = ticks_at(tl, t19, width=1.1)
        dec19 = code(f"{PASSES} checks × {PASS4 * 1e3:.1f} ms = {T_DEC19:.2f} s", 20, MEMORY).next_to(ticks19, m.RIGHT, buff=0.25)
        dec19.set_y(tl.x(0)[1])

        fx = 0.75
        naive_s = f"{S16:.2f} × {STEP_X:.2f} = {NAIVE:.1f}×"
        real_s = f"{S4:.2f} × {STEP_X:.2f} = {REAL:.1f}×"
        naive = left_at(mono(naive_s, 22, MUTED), fx, VIS_Y + 0.22)
        real = left_at(mono(real_s, 22, INK, rules={f"= {REAL:.1f}×": TOKEN}), fx, VIS_Y - 0.28)
        naive_tag = left_at(text("if they multiplied", 20, RED), naive.get_right()[0] + 0.4, VIS_Y + 0.22)
        real_tag = left_at(text("together · Ep 13", 20, MUTED), naive.get_right()[0] + 0.4, VIS_Y - 0.28)
        sit(naive_tag, baseline(naive))
        sit(real_tag, baseline(real))

        br = bracket(tl.x(0)[0], tl.x(T_ALONE)[0], -0.86, TOKEN)
        br_lab = mono(f"{T_ALONE:.2f} s · {T_TOTAL / T_ALONE:.1f}× sooner than {T_TOTAL:.2f} s", 22, MUTED,
                      rules={f"{T_ALONE:.2f} s": TOKEN, f"{T_TOTAL / T_ALONE:.1f}× sooner": TOKEN})
        left_at(br_lab, br.get_right()[0] + 0.3, -0.84)

        self.say(
            "[Late at night, she's alone on the GPU.] [Three more layers make her own reply faster.]",
            m.AnimationGroup(self.new_header("02 · quiet hours", "Faster for her alone"), m.FadeIn(alone, shift=0.1 * m.UP)),
            m.Circumscribe(m.VGroup(self.pre, self.ticks), color=TOKEN, buff=0.1),
        )
        self.say(
            "[The prefix cache, from episode eight]: [Owl's thousand-token system prompt is already cached], "
            "[so prefill drops from forty-nine milliseconds to fifteen.]",
            m.AnimationGroup(m.FadeOut(alone), rail.light(8), m.FadeIn(p8, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(cached), m.FadeIn(chips, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeOut(cached), m.Transform(self.pre, pre8), self.ticks.animate.shift(dx * m.RIGHT),
                             self.dec_lab.animate.shift(dx * m.RIGHT), m.FadeOut(self.pre_lab), m.FadeIn(lab8)),
        )
        self.pre_lab = lab8
        self.say(
            "[Speculative decoding, from episode thirteen]: [a small model guesses four tokens], [and the big one checks them all in one pass.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(chips, p8)), rail.light(13), m.FadeIn(p13, shift=0.1 * m.UP)),
            m.LaggedStart(*[m.FadeIn(d, shift=0.15 * m.RIGHT) for d in drafts], lag_ratio=0.25),
            m.AnimationGroup(m.LaggedStart(*[m.ReplacementTransform(d, j) for d, j in zip(drafts, judged)], lag_ratio=0.2),
                             m.FadeIn(bonus, shift=0.1 * m.LEFT),
                             m.FadeIn(sd1), m.FadeIn(sd2)),
        )
        self.say(
            "[Decode runs two point seven times faster]: [under three quarters of a second.]",
            m.AnimationGroup(m.ReplacementTransform(self.ticks, ticks13), m.FadeOut(self.dec_lab), m.FadeIn(dec13, shift=0.1 * m.LEFT)),
            m.AnimationGroup(mt[0].to(f"{TPS13:.0f}"), mt[2].to(f"${usd(TPS13):.2f}")),
        )
        self.ticks, self.dec_lab = ticks13, dec13
        spec = m.VGroup(judged, bonus, sd1, sd2, p13)
        self.say(
            "[Four-bit weights, from episodes eighteen and nineteen]: [each step reads four and a half gigabytes, not sixteen], "
            "[so steps run three point four times faster.]",
            m.AnimationGroup(m.FadeOut(spec), rail.light(18), rail.light(19), m.FadeIn(p19, shift=0.1 * m.UP), m.FadeIn(wbar), m.FadeIn(wlab)),
            m.AnimationGroup(m.Transform(wbar, wbar4), m.FadeOut(wlab), m.FadeIn(wlab4)),
            m.AnimationGroup(m.ReplacementTransform(self.ticks, ticks19), m.FadeOut(self.dec_lab), m.FadeIn(dec19, shift=0.1 * m.LEFT),
                             mt[0].to(f"{TPS19:.0f}"), mt[2].to(f"${usd(TPS19):.2f}")),
        )
        self.ticks, self.dec_lab = ticks19, dec19
        self.say(
            "[But don't multiply the two.] [Against a lighter model, the draft costs more], [so speculation now gives one point eight, not two point seven.]",
            m.AnimationGroup(m.FadeIn(naive, shift=0.1 * m.UP), m.FadeIn(naive_tag)),
            m.AnimationGroup(naive.animate.set_opacity(0.4), naive_tag.animate.set_opacity(0.6)),
            m.AnimationGroup(m.FadeIn(real, shift=0.1 * m.UP), m.FadeIn(real_tag)),
        )
        self.say(
            "[Her reply now takes a third of a second], [six times faster than in episode one.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(p19, wbar, wlab4, naive, naive_tag, real, real_tag)), m.Create(br),
                             m.FadeIn(br_lab[:len(f"{T_ALONE:.2f}s")])),
            m.FadeIn(br_lab[len(f"{T_ALONE:.2f}s"):], shift=0.1 * m.RIGHT),
        )
        self.say(
            "[But she has a whole H100 to herself]: [a dollar fifteen per million tokens.]",
            m.Circumscribe(m.VGroup(mt[1].cap, mt[1].value), color=REQUEST, buff=0.12),
            m.Circumscribe(m.VGroup(mt[2].cap, mt[2].value), color=INK, buff=0.12),
        )
        self.wait(0.3)
        self.clear(*self.dashboard(), *self.timeline_parts())

    # ------------------------------------------------------------------ 3 · the evening peak

    def peak(self):
        self.next_section("3 · Evening peak", notes=f"Sharing the H100, from Episode 1's plain request (BF16, no speculation), with Episode 4's price model (2K-token caches, {KV_SEQ / 2**20:.0f} MiB each). Batching (Episode 4): batch 64 → {TPS64:,.0f} tok/s, ${usd(TPS64):.2f} per M. Paging (Episode 7): reservations of {RESERVE:,} tokens ({RESERVE * KV_TOK / GB:.2f} GB each) fit {R8} students in the {FREE / GB:.2f} GB left after the weights, three quarters of each empty (Kwon et al. 2023: earlier systems held tokens in only 20.4–38.2% of KV memory); paged 2K caches fit {MAX_B}: {TPS_MAX:,.0f} tok/s, ${usd(TPS_MAX):.2f}. Continuous batching (Episode 10) refills a finished slot at the next step; chunked prefill (Episode 11) slices a new prompt across steps (the slot grid is schematic). The evening peak (Episode 24, for example 20 questions/s, 0.3 s to the first token, 300-token answers): Little's law L = λW solved with step_time: {L_PEAK:.1f} in flight, {TPOT * 1e3:.1f} ms per token, {W_PEAK:.2f} s per answer; KV {KV_PEAK / GB:.1f} GB + weights {weight_bytes() / GB:.2f} GB = {BYTES16 / GB:.1f} of 80 GB. The GPU makes {PEAK_TPS:,} tok/s at the peak → ${PEAK_USD:.3f} per M (rent ÷ tokens), {USD1 / PEAK_USD:.0f}× less than alone at batch 1. Her timeline: 0.3 s to the first token, then {STEPS} gaps of {TPOT * 1e3:.1f} ms = {T_HER:.2f} s; {1 / TPOT:.0f} tok/s for her.")
        rail, mt = self.rail, self.meters
        evening = left_at(pill(f"evening · {STUDENTS:,} students", REQUEST, PINK_FILL, 22, mono=True), -6.4, PILL_Y)
        plain = left_at(pill("from episode 1: BF16, no guessing", MUTED, CARD, 22, mono=True), evening.get_right()[0] + 0.25, PILL_Y)
        p4 = left_at(badge(4, "batching"), -6.4, PILL_Y)
        p7 = left_at(badge(7, "PagedAttention"), -6.4, PILL_Y)
        p1011 = left_at(pill("EP 10–11 · continuous batching, chunked prefill", REQUEST, PINK_FILL, 22, mono=True), -6.4, PILL_Y)
        p24 = left_at(badge(24, "the price of a token"), -6.4, PILL_Y)

        mem = MemoryBar(GPU["memory"], width=11.0, height=0.55, title="one H100 · 80 GB of memory").move_to([0, -1.85, 0])
        x_w = mem.track.get_left()[0] + mem.bar_width * weight_bytes() / GPU["memory"]
        unit = mem.bar_width / GPU["memory"]
        seg_h = mem.bar_height - 0.08
        y_m = mem.track.get_y()
        wseg = hbar(mem.track.get_left()[0], y_m, weight_bytes() * unit, seg_h, WEIGHTS)
        wl = code(f"weights {weight_bytes() / GB:.0f} GB", 20, WEIGHTS).next_to(wseg, m.DOWN, buff=0.18).align_to(wseg, m.LEFT)

        def kv(n):
            return hbar(x_w, y_m, n * KV_SEQ * unit, seg_h, KV)

        def kv_lab(s):
            return left_at(code(s, 20, KV), x_w + 0.5, wl.get_y())

        kv1, kv64, kvmax, kvpk = kv(1), kv(64), kv(MAX_B), kv(L_PEAK)
        kl1 = kv_lab(f"her cache · {KV_SEQ / GB:.2f} GB")
        kl64 = kv_lab(f"64 caches · {64 * KV_SEQ / GB:.0f} GB")
        klmax = kv_lab(f"{MAX_B} caches, packed in 16-token blocks · {MAX_B * KV_SEQ / GB:.0f} GB")
        klpk = kv_lab(f"{L_PEAK:.0f} caches · {KV_PEAK / GB:.0f} GB · {BYTES16 / GB:.0f} of 80 GB in use")
        rw = RESERVE * KV_TOK * unit
        reserved = m.VGroup()
        for i in range(R8):
            x0 = x_w + i * rw
            reserved.add(m.Rectangle(width=rw - 0.012, height=seg_h).set_stroke(KV, 1).set_fill(CARD, 1).move_to([x0 + rw / 2, y_m, 0]))
            reserved.add(hbar(x0 + 0.006, y_m, KV_SEQ * unit, seg_h - 0.04, KV))
        kl_res = kv_lab("8K reserved each: a quarter as many")

        # students sharing one weight read
        dots = m.VGroup(*[m.Dot(radius=0.045, color=REQUEST) for _ in range(64)]).arrange_in_grid(rows=2, buff=(0.06, 0.1))
        left_at(dots, -6.4, VIS_Y)
        dots_lab = left_at(code("64 students share each read", 20, MUTED), dots.get_right()[0] + 0.35, VIS_Y)

        # continuous batching and chunked prefill (schematic slot grid: 5 slots × 12 steps)
        rows = [
            ["d"] * 5 + ["n"] + ["d"] * 6,
            ["p"] * 3 + ["d"] * 9,
            ["d"] * 8 + ["n"] + ["d"] * 3,
            ["d"] * 12,
            ["d"] * 3 + ["n"] + ["d"] * 8,
        ]
        colors = {"d": MEMORY, "n": REQUEST, "p": COMPUTE}
        cs, cg = 0.15, 0.045
        gx, gy = -6.35, VIS_Y + 2 * (cs + cg)
        slots = m.VGroup()
        cols = [m.VGroup() for _ in range(12)]
        for r, row in enumerate(rows):
            for c, kind in enumerate(row):
                sq = m.Square(cs).set_stroke(width=0).set_fill(colors[kind], 0.9 if kind != "d" else 0.55)
                sq.move_to([gx + c * (cs + cg) + cs / 2, gy - r * (cs + cg), 0])
                slots.add(sq)
                cols[c].add(sq)
        lx = gx + 12 * (cs + cg) + 0.35
        cb1 = left_at(mono("a free slot refills at the very next step", 20, MUTED, rules={"refills": REQUEST}), lx, VIS_Y + 0.22)
        cb2 = left_at(mono("her prompt rides along in slices", 20, MUTED, rules={"in slices": COMPUTE}), lx, VIS_Y - 0.22)

        little = left_at(mono(f"in flight = {LAM} questions/s × {W_PEAK:.1f} s ≈ {L_PEAK:.0f}", 22, INK,
                              rules={f"{LAM} questions/s": REQUEST, f"≈ {L_PEAK:.0f}": REQUEST}), -6.4, VIS_Y)
        little_src = cite("Little's law · Ep 24's evening peak").move_to([0, VIS_Y, 0]).align_to([6.5, 0, 0], m.RIGHT)
        ratio = code(f"÷ {USD1 / PEAK_USD:.0f} vs. alone", 22, TOKEN).move_to([4.4, PILL_Y, 0])

        # her request at the peak, on an 8-second axis
        tl8 = Timeline(8.0, label_every=2, **TL_KW)
        first = tl8.bar(0, TTFT, COMPUTE)
        first_lab = code(f"first token {TTFT:.1f} s", 20, COMPUTE).next_to(first, m.UP, buff=0.22).align_to(first, m.LEFT)
        pk_ticks = ticks_at(tl8, [TTFT + k * TPOT for k in range(1, STEPS + 1)], width=1.4)
        pk_lab = code(f"{STEPS} × {TPOT * 1e3:.1f} ms", 20, MEMORY).next_to(pk_ticks, m.UP, buff=0.22).align_to(pk_ticks, m.RIGHT)
        sit(pk_lab, baseline(first_lab))
        her_br = bracket(tl8.x(0)[0], tl8.x(T_HER)[0], -0.86, INK)
        her_lab = mono(f"her reply ≈ {T_HER:.1f} s at the peak", 22, INK, rules={f"≈ {T_HER:.1f} s": INK}).next_to(her_br, m.UP, buff=0.15)
        read_lab = left_at(mono(f"{1 / TPOT:.0f} tokens/s for her", 20, MUTED, rules={f"{1 / TPOT:.0f} tokens/s": TOKEN}),
                           pk_ticks.get_right()[0] + 0.3, tl8.x(0)[1])

        self.say(
            "[In the evening, the whole class logs on.] [To cut the bill, share the GPU], [starting again from episode one's plain request.]",
            m.AnimationGroup(self.new_header("03 · evening peak", "Cheaper for everyone"), *[m.FadeOut(p) for p in self.timeline_parts()],
                             m.FadeIn(evening, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(mem.track), m.FadeIn(mem.caption), m.GrowFromEdge(wseg, m.LEFT), m.FadeIn(wl),
                             m.GrowFromEdge(kv1, m.LEFT), m.FadeIn(kl1)),
            m.AnimationGroup(m.FadeIn(plain, shift=0.1 * m.UP), mt[0].to(f"{TPS1:.0f}"), mt[2].to(f"${USD1:.2f}")),
        )
        self.say(
            "[Batching, from episode four]: [sixty-four students share every read of the weights], [and a million tokens costs twenty-three cents.]",
            m.AnimationGroup(m.FadeOut(evening), m.FadeOut(plain), rail.light(4), m.FadeIn(p4, shift=0.1 * m.UP)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.02), m.FadeIn(dots_lab),
                             m.ReplacementTransform(kv1, kv64), m.FadeOut(kl1), m.FadeIn(kl64)),
            m.AnimationGroup(mt[0].to(f"{TPS64:,.0f}"), mt[1].to("64"), mt[2].to(f"${usd(TPS64):.2f}")),
        )
        self.say(
            "[Paging, from episode seven, packs their caches in small blocks]: [two hundred and thirty-eight students fit], "
            "[where eight K reservations held only a quarter as many.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(dots, dots_lab, p4)), rail.light(7), m.FadeIn(p7, shift=0.1 * m.UP)),
            m.AnimationGroup(m.ReplacementTransform(kv64, kvmax), m.FadeOut(kl64), m.FadeIn(klmax),
                             mt[0].to(f"{TPS_MAX:,.0f}"), mt[1].to(f"{MAX_B}"), mt[2].to(f"${usd(TPS_MAX):.2f}")),
            m.AnimationGroup(m.FadeIn(reserved), m.FadeOut(klmax), m.FadeIn(kl_res)),
        )
        self.say(
            "[Continuous batching and chunked prefill, from episodes ten and eleven], [refill each free slot at the next step], "
            "[and slice her prompt so nobody stalls.]",
            m.AnimationGroup(m.FadeOut(reserved), m.FadeOut(kl_res), m.FadeIn(klmax), m.FadeOut(p7), rail.light(10), rail.light(11),
                             m.FadeIn(p1011, shift=0.1 * m.UP)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c) for c in cols], lag_ratio=0.25, run_time=2.2), m.FadeIn(cb1, shift=0.1 * m.UP)),
            m.FadeIn(cb2, shift=0.1 * m.UP),
        )
        self.say(
            "[At Owl's evening peak, from episode twenty-four], [twenty questions a second keep a hundred and twenty-four students in flight.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(slots, cb1, cb2, p1011)), rail.light(24), m.FadeIn(p24, shift=0.1 * m.UP), m.FadeIn(little_src)),
            m.AnimationGroup(m.FadeIn(little, shift=0.1 * m.UP), m.ReplacementTransform(kvmax, kvpk), m.FadeOut(klmax), m.FadeIn(klpk),
                             mt[0].to(f"{PEAK_TPS:,}"), mt[1].to(f"{L_PEAK:.0f}"), mt[2].to(f"${PEAK_USD:.2f}")),
        )
        self.say(
            "[A million tokens now costs eighteen cents], [about forty times less than alone.]",
            m.Circumscribe(m.VGroup(mt[2].cap, mt[2].value), color=TOKEN, buff=0.12),
            m.FadeIn(ratio, shift=0.1 * m.UP),
        )
        memory = m.VGroup(mem.track, mem.caption, wseg, wl, kvpk, klpk)
        self.say(
            "[Her own reply is slower]: [about six seconds], [at twenty milliseconds a token], [still faster than anyone reads.]",
            m.AnimationGroup(m.FadeOut(memory), m.FadeOut(m.VGroup(little, little_src)), m.FadeIn(tl8)),
            m.AnimationGroup(m.GrowFromEdge(first, m.LEFT), m.FadeIn(first_lab),
                             m.LaggedStart(*[m.FadeIn(t) for t in pk_ticks], lag_ratio=0.02, run_time=1.6),
                             m.Create(her_br), m.FadeIn(her_lab)),
            m.FadeIn(pk_lab, shift=0.1 * m.UP),
            m.FadeIn(read_lab, shift=0.1 * m.LEFT),
        )
        self.wait(0.4)
        self.clear(*self.rail.shown())

    # ------------------------------------------------------------------ 4 · both at once

    def both(self):
        self.next_section("4 · Both at once", notes=f"Her speed layers at the evening peak ({L_PEAK:.0f} in flight, BF16, 2K-token caches). One step reads {weight_bytes() / GB:.2f} GB of weights + {KV_PEAK / GB:.1f} GB of caches = {BYTES16 / GB:.1f} GB; with 4-bit weights {BYTES4 / GB:.1f} GB: −{SAVE_PEAK:.1%} (Episode 24: 'under a quarter'), where alone it cut a step by {SAVE_ALONE:.1%} (6.49 → {T_STEP4 * 1e3:.2f} ms). Speculation spends spare math: one token's math at 50% MFU = {MATH * 1e6:.1f} µs; alone that is {SHARE1:.2%} of a {cost_step(1) * 1e3:.2f} ms step; at the peak {L_PEAK:.0f} tokens = {SHARE_PK:.1%} of a {TPOT * 1e3:.1f} ms step; checking γ + 1 = 5 tokens per student needs {SHARE_SPEC:.0%}: no spare math, so speculation waits for the quiet hours (Episode 14: EAGLE-3 measured 1.8× at batch 2 falling to 1.38× at batch 64, H100, Li et al. 2025). Table: KV cache ×{T_NOCACHE / T_TOTAL:.0f} alone; prefix cache −{(T_PRE - T_PRE8) * 1e3:.0f} ms alone, {PROMPT_CUT:.0%} less prompt math at the peak; sharing ÷{USD1 / PEAK_USD:.0f} per token.")
        rail = self.rail
        cap1 = left_at(text(f"one decode step at the peak · {L_PEAK:.0f} students", 22, MUTED), -6.4, 1.62)
        sc = 0.095
        xb, yb = -6.4, 1.02
        w16 = hbar(xb, yb, weight_bytes() / GB * sc, 0.42, WEIGHTS)
        kvb = hbar(xb + weight_bytes() / GB * sc, yb, KV_PEAK / GB * sc, 0.42, KV)
        w4b = hbar(xb, yb, W4 / GB * sc, 0.42, WEIGHTS)
        wlab = code(f"weights {weight_bytes() / GB:.0f} GB", 20, WEIGHTS).next_to(w16, m.DOWN, buff=0.15).align_to(w16, m.LEFT)
        klab = code(f"caches {KV_PEAK / GB:.0f} GB", 20, BG, weight=m.BOLD).move_to(kvb).set_z_index(2)
        wlab4 = code(f"weights {W4 / GB:.1f} GB", 20, WEIGHTS).move_to(wlab, aligned_edge=m.LEFT)
        tot = code(f"{BYTES16 / GB:.0f} GB", 20, INK).next_to(kvb, m.RIGHT, buff=0.15)
        tot4 = code(f"{BYTES4 / GB:.0f} GB", 20, INK).next_to(w4b, m.RIGHT, buff=0.15 + KV_PEAK / GB * sc)
        cut = left_at(mono(f"4-bit: −{SAVE_PEAK:.0%} per step · alone: −{SAVE_ALONE:.0%}", 20, MUTED,
                           rules={f"−{SAVE_PEAK:.0%} per step": TOKEN}), -6.4, 0.05)

        cap2 = left_at(text("math used per step", 22, MUTED), -6.4, -0.55)
        full = 3.0
        x_bar = -4.25
        names = ["alone", "at the peak", "+ 4 guesses each"]
        fr = [SHARE1, SHARE_PK, SHARE_SPEC]
        cols = [COMPUTE, COMPUTE, RED]
        ys = [-1.05, -1.6, -2.15]
        track = m.VGroup(*[m.Rectangle(width=full, height=0.3).set_stroke(EDGE, 1.5).set_fill(CARD, 1).move_to([x_bar + full / 2, y, 0]) for y in ys])
        limit = m.DashedLine([x_bar + full, ys[0] + 0.3, 0], [x_bar + full, ys[-1] - 0.3, 0], dash_length=0.08).set_stroke(MUTED, 2)
        bars = [hbar(x_bar, y, max(f * full, 0.03), 0.22, c) for f, y, c in zip(fr, ys, cols)]
        labs = [m.VGroup(text(n, 20, MUTED).move_to([0, y, 0]).align_to([x_bar - 0.2, 0, 0], m.RIGHT)) for n, y in zip(names, ys)]
        vals = [code(s, 20, c, weight=m.BOLD).next_to(b, m.RIGHT, buff=0.15)
                for s, b, c in zip([f"{SHARE1:.1%}", f"{SHARE_PK:.0%}", f"{SHARE_SPEC:.0%}"], bars, [COMPUTE, COMPUTE, RED])]
        vals[0].next_to(track[0], m.RIGHT, buff=0.15)
        vals[1].next_to(track[1], m.RIGHT, buff=0.15)
        mfu_src = cite("half the peak FLOPs, as in Ep 1").move_to([0, -2.75, 0]).align_to([-6.4, 0, 0], m.LEFT)

        # the table: alone (batch 1) vs at the peak
        tx = 0.05
        c_name, c_alone, c_peak = tx + 0.55, tx + 2.55, tx + 3.85
        hy = 1.62
        h1 = left_at(code("alone", 20, MUTED, weight=m.BOLD), c_alone, hy)
        h2 = left_at(code("at the peak", 20, MUTED, weight=m.BOLD), c_peak, hy)
        rule = m.Line([tx, hy - 0.28, 0], [6.4, hy - 0.28, 0]).set_stroke(EDGE, 2)

        def row(n, name, a, b, y, ca=TOKEN, cb=TOKEN):
            c = cell(n, True, 0.4).move_to([tx + 0.2, y, 0])
            nm = left_at(text(name, 22, INK), c_name, y)
            va = left_at(code(a, 20, ca, weight=m.BOLD), c_alone, y)
            vb = left_at(code(b, 20, cb, weight=m.BOLD), c_peak, y)
            base = baseline(va)
            sit(nm, base)
            sit(vb, base)
            return m.VGroup(c, nm, va, vb)

        r19 = row(19, "4-bit weights", f"{STEP_X:.1f}×", f"−{SAVE_PEAK:.0%}", 1.0, TOKEN, MUTED)
        r13 = row(13, "speculation", f"{S16:.1f}×", "no spare math", 0.42, TOKEN, RED)
        r5 = row(5, "KV cache", f"{T_NOCACHE / T_TOTAL:.0f}×", "essential", -0.16)
        r8 = row(8, "prefix cache", f"−{(T_PRE - T_PRE8) * 1e3:.0f} ms", f"−{PROMPT_CUT:.0%} math", -0.74)
        r4 = row(4, "sharing", "1×", f"÷ {USD1 / PEAK_USD:.0f} per token", -1.32, MUTED, TOKEN)
        verdict = left_at(text("speedups don't multiply", 26, INK, weight=m.BOLD), tx, -2.1)

        self.say(
            "[Now bring her speed layers to the peak.] [Not all of them survive.]",
            m.AnimationGroup(self.new_header("04 · both at once", "Which speed layers survive?"), m.FadeIn(cap1)),
            m.AnimationGroup(m.FadeIn(h1), m.FadeIn(h2), m.Create(rule)),
        )
        self.say(
            "[Each step reads sixteen gigabytes of weights], [plus every student's cache]: [thirty-three more.]",
            m.AnimationGroup(m.GrowFromEdge(w16, m.LEFT), m.FadeIn(wlab)),
            m.GrowFromEdge(kvb, m.LEFT),
            m.AnimationGroup(m.FadeIn(klab), m.FadeIn(tot)),
        )
        self.say(
            "[So four-bit weights trim under a quarter of each step], [where alone they cut over two thirds.]",
            m.AnimationGroup(m.Transform(w16, w4b), kvb.animate.shift((w4b.width - w16.width) * m.RIGHT),
                             klab.animate.shift((w4b.width - w16.width) * m.RIGHT),
                             m.FadeOut(tot), m.FadeIn(tot4), m.FadeOut(wlab), m.FadeIn(wlab4), m.FadeIn(cut[:len("4-bit:−23%perstep")], shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(cut[len("4-bit:−23%perstep"):]), m.FadeIn(r19, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[Speculation spends spare math.] [Alone, her steps used half a percent of it]. [At the peak, a fifth.]",
            m.AnimationGroup(m.FadeIn(cap2), m.FadeIn(track), m.Create(limit), m.FadeIn(mfu_src), *[m.FadeIn(lb) for lb in labs]),
            m.AnimationGroup(m.GrowFromEdge(bars[0], m.LEFT), m.FadeIn(vals[0])),
            m.AnimationGroup(m.GrowFromEdge(bars[1], m.LEFT), m.FadeIn(vals[1])),
        )
        self.say(
            "[Checking four guesses for every student needs five times that]: [all of it]. [So speculation waits for the quiet hours.]",
            m.AnimationGroup(m.GrowFromEdge(bars[2], m.LEFT, run_time=1.4), m.FadeIn(vals[2])),
            m.Circumscribe(m.VGroup(bars[2], vals[2]), color=RED, buff=0.08),
            m.FadeIn(r13, shift=0.1 * m.LEFT),
        )
        self.say(
            "[Both caches help at any load], [but speedups don't multiply]: [sharing uses up the spare math], [and the students' caches take over the bytes.]",
            m.LaggedStart(m.FadeIn(r5, shift=0.1 * m.LEFT), m.FadeIn(r8, shift=0.1 * m.LEFT), lag_ratio=0.3),
            m.FadeIn(verdict, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(r4, shift=0.1 * m.LEFT), m.Circumscribe(m.VGroup(track[1], bars[1]), color=COMPUTE, buff=0.08)),
            m.Circumscribe(kvb, color=KV, buff=0.08),
        )
        self.wait(0.4)
        self.clear(*self.rail.shown())

    # ------------------------------------------------------------------ 5 · the stack

    def stack(self):
        self.next_section("5 · The stack", notes=f"Everything ships in one engine (Episode 22, notes §13: vLLM and SGLang implement PagedAttention, continuous batching, chunked prefill, prefix caching, speculative decoding; engines dated October 2026) behind an OpenAI-compatible API with SSE streaming, a router and an autoscaler. The engine also brings CUDA graphs (Episode 16) and the sampler with JSON grammars (Episode 17). Episode 25 watches token gaps and KV use: at the peak {TPOT * 1e3:.1f} ms per token and {KV_USE:.0%} of the KV memory ({KV_PEAK / GB:.1f} of {FREE / GB:.1f} GB). Dark today: Episode 9 (long contexts: her chat is 1,800 tokens), 12 (split prefill and decode pools), 20 (mixture of experts), 21 (many GPUs): Owl-8B fits on one H100. Episode 23, measured on this Mac (measurements §1, mlx-lm 0.32.2): Owl-8B 4-bit {MAC_TOK:.1f} tok/s; ceiling {MAC['bandwidth'] / GB:.0f} GB/s ÷ {MAC_W4:.2f} GB = {MAC_CEIL:.1f} tok/s ({MAC_TOK / MAC_CEIL:.0%}). Episode 24's bill: one H100 × 720 h × $3.99 = ${720 * PRICE:,.2f} for {STUDENTS:,} students = ${PER_STUDENT:.2f} each.")
        rail = self.rail
        sx, cw = -3.6, 5.7
        api = layer_card("API", "OpenAI-style · streamed", width=cw).move_to([sx, 1.05, 0])
        router = layer_card("router", "routing · autoscaling", width=cw).move_to([sx, 0.25, 0])
        eng_box = card(cw, 1.25, EDGE, CARD, radius=0.16, stroke=2).move_to([sx, -0.85, 0])
        eng_t = text("engine", 24, INK, weight=m.BOLD)
        eng_s = code("vLLM · SGLang", 20, MUTED)
        eng_row = m.VGroup(eng_t, eng_s).arrange(m.RIGHT, buff=0.3).align_to(eng_box, m.UL).shift(0.3 * m.RIGHT + 0.16 * m.DOWN)
        sit(eng_s, baseline(eng_t))
        layers = [4, 5, 6, 7, 8, 10, 11, 13, 15, 19]
        extra = [16, 17]
        allb = layers + extra
        chips = {n: cell(n, True, 0.4) for n in allb}
        row = m.VGroup(*[chips[n] for n in sorted(allb)]).arrange(m.RIGHT, buff=0.04)
        row.move_to(eng_box).align_to(eng_box, m.DOWN).shift(0.17 * m.UP)
        engine = m.VGroup(eng_box, eng_row)
        gpu = layer_card("one H100", "Owl-8B · 80 GB", width=cw).move_to([sx, -2.05, 0])
        g_icon = icon(GPU_ICON, 0.4).move_to(gpu[0]).align_to(gpu[0], m.RIGHT).shift(0.3 * m.LEFT)
        gpu.add(g_icon)
        arrows = m.VGroup(*[m.Arrow(a.get_bottom(), b.get_top(), buff=0.02, color=FAINT, stroke_width=3,
                                    max_tip_length_to_length_ratio=0.5) for a, b in ((api, router), (router, eng_box), (eng_box, gpu))])
        path = m.Line(api.get_top() + 0.05 * m.DOWN, gpu.get_bottom() + 0.05 * m.UP)
        pulse_ = m.ShowPassingFlash(path.copy().set_stroke(REQUEST, 8), time_width=0.4, run_time=1.4)

        rx, rw = -0.35, 6.8
        mon_box = card(rw, 1.2, EDGE, CARD, radius=0.16, stroke=2)
        left_at(mon_box, rx, 0.9)
        mon_t = left_at(text("watching it · Ep 25", 22, INK, weight=m.BOLD), rx + 0.3, 1.24)
        g1_lab = left_at(code(f"token gap {TPOT * 1e3:.0f} ms", 20, MUTED), rx + 0.3, 0.82)
        g2_lab = left_at(code(f"KV in use {KV_USE:.0%}", 20, MUTED), rx + 3.55, 0.82)
        m1 = Meter(width=2.9, fraction=TPOT / 0.05, color=MEMORY, height=0.16)
        left_at(m1, rx + 0.3, 0.55)
        m2 = Meter(width=2.9, fraction=KV_USE, color=KV, height=0.16)
        left_at(m2, rx + 3.55, 0.55)

        dark = [9, 12, 20, 21]
        rings = {n: m.DashedVMobject(m.RoundedRectangle(corner_radius=0.09, width=0.52, height=0.52), num_dashes=16)
                 .set_stroke(RED, 2).move_to(rail.dim[n]) for n in dark}
        d1 = code("dark today: 9 long memory", 20, MUTED)
        d2 = code("· 12 split pools · 20 experts · 21 many GPUs", 20, MUTED)
        drow = m.VGroup(d1, d2).arrange(m.RIGHT, buff=0.15).move_to([0, 1.8, 0])
        sit(d2, baseline(d1))

        mac_box = card(rw, 1.4, EDGE, CARD, radius=0.16, stroke=2)
        left_at(mac_box, rx, -0.6)
        mac_i = icon(MAC_ICON, 0.62).move_to([rx + 0.65, -0.6, 0])
        mac_t = left_at(text("this Mac · Ep 23", 22, INK, weight=m.BOLD), rx + 1.25, -0.25)
        mac_v = left_at(mono(f"measured: {MAC_TOK:.0f} tok/s, 4-bit", 20, MUTED, rules={f"{MAC_TOK:.0f} tok/s": TOKEN}), rx + 1.25, -0.63)
        mac_c = left_at(mono(f"ceiling {MAC['bandwidth'] / GB:.0f} GB/s ÷ {MAC_W4:.2f} GB = {MAC_CEIL:.0f}", 20, MUTED,
                             rules={f"= {MAC_CEIL:.0f}": MEMORY}), rx + 1.25, -0.98)

        fin = m.VGroup(
            stat(f"{STUDENTS:,}", "", "students", REQUEST, 50),
            stat("1", "", "H100 at peak", INK, 50),
            stat(f"${PER_STUDENT:.2f}", "", "each a month", TOKEN, 50),
        ).arrange(m.RIGHT, buff=0.45, aligned_edge=m.DOWN)
        left_at(fin, rx + 0.15, -2.2)
        fin_src = cite("Ep 24's bill · $3.99 per H100-hour, dated").move_to([0, -3.4, 0]).align_to([6.5, 0, 0], m.RIGHT)

        self.say(
            "[All of it ships as one engine, like vLLM or SGLang], [behind the serving stack from episode twenty-two.]",
            m.AnimationGroup(self.new_header("05 · the stack", "Shipped as one server"), m.FadeIn(engine),
                             m.LaggedStart(*[m.FadeIn(chips[n], scale=0.6) for n in layers], lag_ratio=0.12)),
            m.AnimationGroup(rail.light(22), m.FadeIn(api, shift=0.1 * m.DOWN), m.FadeIn(router, shift=0.1 * m.DOWN),
                             m.FadeIn(gpu, shift=0.1 * m.UP), m.FadeIn(arrows), m.Succession(m.Wait(0.5), pulse_)),
        )
        self.say(
            "[The engine brings the rest]: [graphs that cut the CPU tax], [and the sampler that writes her flashcards.]",
            m.Circumscribe(engine, color=MUTED, buff=0.06),
            m.AnimationGroup(rail.light(16), m.FadeIn(chips[16], scale=0.6)),
            m.AnimationGroup(rail.light(17), m.FadeIn(chips[17], scale=0.6)),
        )
        self.say(
            "[Episode twenty-five keeps it alive through exam week], [watching the gaps between tokens], [and the cache filling up.]",
            m.AnimationGroup(rail.light(25), m.FadeIn(mon_box), m.FadeIn(mon_t)),
            m.AnimationGroup(m.FadeIn(m1.track), m.GrowFromEdge(m1.bar, m.LEFT), m.FadeIn(g1_lab)),
            m.AnimationGroup(m.FadeIn(m2.track), m.GrowFromEdge(m2.bar, m.LEFT), m.FadeIn(g2_lab)),
        )
        self.say(
            "[Four stay dark today]: [her chat is short], [and Owl's dense model fits on one GPU.]",
            m.LaggedStart(*[m.Create(rings[n]) for n in dark], lag_ratio=0.2),
            m.FadeIn(d1, shift=0.1 * m.DOWN),
            m.FadeIn(d2, shift=0.1 * m.DOWN),
        )
        self.say(
            "[And on your desk, from episode twenty-three], [the same four-bit Owl makes a hundred and two tokens a second], "
            "[against a ceiling of a hundred and twenty-one.]",
            m.AnimationGroup(rail.light(23), m.FadeIn(mac_box), m.FadeIn(mac_i, shift=0.1 * m.UP), m.FadeIn(mac_t)),
            m.FadeIn(mac_v, shift=0.1 * m.UP),
            m.FadeIn(mac_c, shift=0.1 * m.UP),
        )
        self.say(
            "[Ten thousand students], [one H100 at the peak], [twenty-nine cents each a month.]",
            m.AnimationGroup(m.FadeIn(fin[0], shift=0.15 * m.UP), m.FadeIn(fin_src)),
            m.FadeIn(fin[1], shift=0.15 * m.UP),
            m.FadeIn(fin[2], shift=0.15 * m.UP),
        )
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ 6 · the two speed limits

    def limits(self):
        self.next_section("6 · Two speed limits", notes=f"Episode 2's H100 roofline (attainable FLOP/s = min(989 TFLOP/s, 3.35 TB/s × intensity), ridge {RIDGE:.0f} FLOPs/byte). Intensity = FLOPs ÷ bytes of one step. Prefill of 1,500 tokens ≈ 1,500 (Episode 2's dot): on the roof. Her decode alone: 2·P ÷ (16.06 GB + 1,800 × 128 KiB) = {I_ALONE:.2f}; with 4-bit weights {I_4BIT:.2f}. The evening peak: {L_PEAK:.0f} × 2·P ÷ {BYTES16 / GB:.1f} GB = {I_PEAK:.1f}: still memory-bound, and as the batch grows the caches set the bytes: 2·P ÷ (2,048 × 128 KiB) = {I_CAP:.1f} FLOPs/byte is the most plain decode with 2K-token caches can reach. Checking γ + 1 = 5 tokens per student reads the same bytes for 5× the math: {I_SPEC:.0f}, near the ridge (at Episode 1's 50% MFU and 75% MBU the effective ridge is {RIDGE * MFU / MBU:.0f}, where the math share passed 100%).")
        roof = Roofline(GPU["bandwidth"], GPU["peak"], width=6.8, height=4.3, x_decades=(-1, 4), y_decades=(10.5, 15.5),
                        center=(-1.6, -0.45, 0))
        y_title = text("FLOPs per second you can reach", 22, MUTED)
        y_title.move_to([0, 2.15, 0]).align_to([roof.origin[0] + 0.3, 0, 0], m.LEFT)
        rx = roof.get_right()[0] + 0.45

        def legend(color, title, desc):
            sw = m.Line([0, 0, 0], [0.45, 0, 0]).set_stroke(color, 6)
            t = code(title, 22, color, weight=m.BOLD)
            d = text(desc, 20, MUTED)
            words = m.VGroup(t, d).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.08)
            sw.next_to(t, m.LEFT, buff=0.2)
            return m.VGroup(sw, words)

        leg_mem = left_at(legend(MEMORY, "memory", "bandwidth × intensity"), rx, 1.55)
        leg_cmp = left_at(legend(COMPUTE, "math", "peak · 989 TFLOP/s"), rx, 0.75)
        ridge_lab = code(f"ridge {RIDGE:.0f}", 20, MUTED).next_to(roof.at(RIDGE, 10 ** roof.yd[0]), m.UR, buff=0.12)

        p_pre = roof.point(N_PROMPT)
        pre_dot = roof.dot(N_PROMPT, COMPUTE)
        pre_lab = code("prefill", 20, COMPUTE, weight=m.BOLD).next_to(p_pre, m.UP, buff=0.18)
        dec_dot = roof.dot(I_ALONE, MEMORY)
        dec_lab = code("her decode, alone", 20, MEMORY, weight=m.BOLD).next_to(roof.point(I_ALONE), m.DR, buff=0.2)
        p4b = roof.point(I_4BIT)
        dec4_lab = code("4-bit", 20, MEMORY, weight=m.BOLD).next_to(p4b, m.DR, buff=0.2)
        trail = m.Line(roof.point(I_ALONE), p4b).set_stroke(MEMORY, 10, opacity=0.35)
        p_pk = roof.point(I_PEAK)
        pk_dot = m.Dot(p_pk, radius=0.11, color=REQUEST).set_z_index(3)
        pk_lab = code(f"{L_PEAK:.0f} students", 20, REQUEST, weight=m.BOLD).next_to(p_pk, m.UL, buff=0.2)
        cap_x = roof.at(I_CAP, 10 ** roof.yd[0])
        cap_line = m.DashedLine(cap_x, roof.point(I_CAP), dash_length=0.1).set_stroke(KV, 2.5)
        cap_lab = code(f"2K caches\ncap it at {I_CAP:.0f}", 20, KV).next_to(cap_line, m.LEFT, buff=0.15)
        cap_lab.set_y(roof.origin[1] + 0.75)
        p_sp = roof.point(I_SPEC)
        sp_dot = m.Circle(radius=0.12).set_stroke(TOKEN, 3).move_to(p_sp).set_z_index(3)
        sp_lab = code("+ 4 guesses each", 20, TOKEN, weight=m.BOLD).next_to(p_sp, m.UL, buff=0.2)

        rows = [("skip work", "KV cache · prefix cache", KV), ("share each byte", "batching · speculation", REQUEST),
                ("move fewer bytes", "4-bit · GQA · FlashAttention", WEIGHTS)]
        ways = m.VGroup()
        for i, (t, d, c) in enumerate(rows):
            g = m.VGroup(code(t, 22, c, weight=m.BOLD), text(d, 20, MUTED)).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.08)
            ways.add(left_at(g, rx, -0.35 - 0.8 * i))

        self.say(
            "[Underneath it all are the two speed limits from episode two]: [memory traffic], [and math.]",
            m.AnimationGroup(self.new_header("06 · two speed limits", "Two limits, one chart"), m.FadeIn(roof.axes), m.FadeIn(roof.ticks),
                             m.FadeIn(roof.x_label), m.FadeIn(y_title)),
            m.AnimationGroup(m.Create(roof.memory), m.FadeIn(leg_mem, shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.Create(roof.compute), m.Create(roof.ridge), m.FadeIn(ridge_lab), m.FadeIn(leg_cmp, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[Prefill sits on the amber roof.] [Her decode, alone, sits low on the cyan slope.]",
            m.AnimationGroup(m.FadeIn(pre_dot, scale=0.4), m.FadeIn(pre_lab)),
            m.AnimationGroup(m.FadeIn(dec_dot, scale=0.4), m.FadeIn(dec_lab)),
        )
        self.say(
            "[Four-bit weights slide her right.] [Sharing the step slides much further], [until the caches set the bytes.]",
            m.AnimationGroup(dec_dot.animate.move_to(p4b), m.Create(trail), m.FadeOut(dec_lab), m.FadeIn(dec4_lab)),
            m.AnimationGroup(m.FadeIn(pk_dot, target_position=p4b), m.FadeIn(pk_lab)),
            m.AnimationGroup(m.Create(cap_line), m.FadeIn(cap_lab)),
        )
        self.say(
            "[Checking four guesses per student pushes the peak toward the ridge], [where the spare math runs out.]",
            m.AnimationGroup(m.FadeIn(sp_dot, target_position=p_pk), m.FadeIn(sp_lab)),
            m.Circumscribe(m.VGroup(roof.ridge, ridge_lab), color=COMPUTE, buff=0.1),
        )
        self.say(
            "[Every layer either skipped work], [shared each byte across more tokens], [or moved fewer bytes.]",
            m.FadeIn(ways[0], shift=0.1 * m.LEFT),
            m.FadeIn(ways[1], shift=0.1 * m.LEFT),
            m.FadeIn(ways[2], shift=0.1 * m.LEFT),
        )
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Cache (never compute twice: the KV cache, the prefix cache), batch (share every read: batching, continuous batching, speculation), shrink (move fewer bytes: GQA, 4-bit weights, FlashAttention).")
        halo = glow(MEMORY, radius=4.0)
        spec = (("Cache", KV, "never compute twice"), ("Batch", REQUEST, "share every read"), ("Shrink", WEIGHTS, "move fewer bytes"))
        words = m.VGroup(*[title_text(w, 80, c) for w, c, _ in spec])
        subs = m.VGroup(*[code(s, 22, MUTED) for _, _, s in spec])
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
        dots = m.VGroup(*[m.Dot([(a.get_right()[0] + b.get_left()[0]) / 2, mid, 0], radius=0.07, color=MUTED) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, dots, subs).move_to(0.45 * m.UP)

        self.say(
            "[Three words to keep]: [cache], [batch], [shrink.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say(
            "[Never compute twice], [share every read], [and move fewer bytes.]",
            m.FadeIn(subs[0], shift=0.1 * m.UP),
            m.FadeIn(subs[1], shift=0.1 * m.UP),
            m.FadeIn(subs[2], shift=0.1 * m.UP),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row)], run_time=0.9)

    # ------------------------------------------------------------------ the course-closing card

    def series_outro(self):
        self.next_section("The end", notes="The course's closing card: Tokens per Second, from one token to ten thousand students; the six modules in their colours.")
        if self.head:
            self.play(self.drop_header(), run_time=0.6)
        halo = glow(CYAN, radius=4.3).move_to(0.2 * m.UP)
        brand = title_text("That's Tokens per Second", 64, INK, gradient=(CYAN, AMBER))
        sub = text("from one token to ten thousand students", 30, MUTED)
        top = m.VGroup(brand, sub).arrange(m.DOWN, buff=0.3)
        mods = m.VGroup(*[pill(f"{k} · {name}", color, fill, 22, mono=True) for k, (name, color, fill) in MODULES.items()])
        rows = m.VGroup(*[m.VGroup(mods[i], mods[i + 1]).arrange(m.RIGHT, buff=0.25) for i in (0, 2, 4)])
        grid = rows.arrange(m.DOWN, buff=0.18)
        thanks = title_text("Thanks for watching!", 42, INK)
        mascot = owl(0.6)
        foot = m.VGroup(mascot, thanks).arrange(m.RIGHT, buff=0.4)
        group = m.VGroup(top, grid, foot).arrange(m.DOWN, buff=0.45).move_to(0.05 * m.UP)

        self.say(
            "[That's Tokens per Second]: [from one token to ten thousand students.] [Thanks for watching!]",
            m.AnimationGroup(m.FadeIn(halo, scale=0.6), m.FadeIn(brand, shift=0.2 * m.DOWN)),
            m.AnimationGroup(m.FadeIn(sub, shift=0.1 * m.UP), m.LaggedStart(*[m.FadeIn(p, shift=0.1 * m.UP) for p in mods], lag_ratio=0.15)),
            m.AnimationGroup(m.FadeIn(mascot, shift=0.2 * m.UP), m.Write(thanks)),
        )
        self.wait(1.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, group)], run_time=1.0)
        self.wait(0.5)
