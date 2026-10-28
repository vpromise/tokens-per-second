"""Tokens per Second · Episode 13 · Guess and Check (Module 4 · Faster Tokens)

Speculative decoding: a small draft model (Owl-mini) guesses γ tokens, the big target (Owl-8B) checks them all in
ONE pass — decode is memory-bound (episode 2), and a pass over several rows costs about the same as over one
(episode 4) — then keeps the longest acceptable prefix plus one token of its own. Rejection sampling makes the
output exactly the target's distribution. Expected tokens per pass E = (1 − α^(γ+1)) / (1 − α): α = 0.8, γ = 4
→ 3.36. Speedup E / (1 + γc) with c = draft step ÷ target pass. Then measured on this Mac (1B 4-bit draft for the
8B target in BF16 and 4-bit, chat vs copy-the-notes prompts), and why big batches end the free lunch.

Sections: the idea (one pass per token; ep 4's 1 row vs 8 rows; guess and check) · draft and verify (four
guesses, one pass; kept/rejected/thrown away; Owl-8B's own token; the bonus token) · exactly lossless (p and q
for the third guess; keep if q ≤ p, else with chance p/q; resample from (p − q)+; min(p, q) + (p − q)+ = p;
greedy = keep iff top pick) · tokens per pass (α = average overlap, γ; 1 + α + … + α^γ = 3.36; 100,000
simulated passes; the E-vs-γ curve and its 1/(1 − α) limit; α = 0.9 → 4.10) · the cost ratio (c measured here:
2.0 ms ÷ 32.3 ms ≈ 0.06 in BF16, ÷ 9.8 ms ≈ 0.21 in 4-bit → 2.7× vs 1.8× at α = 0.8, γ = 4) · measured (copy
the notes: every guess kept, 30.7 → 148 tok/s = 4.82×, past the one-token ceiling of 34, inside the new ceiling
of 187; open chat: ragged strip, 3.28 tokens per pass, 38–72% of drafted tokens kept; 2.28× BF16, 1.42× 4-bit;
why 4-bit gains less; the drift after token 154) · busy servers (EAGLE-3 paper, Table 3) · recap.

Sources (facts dated October 2026):
- owl_reference.py: expected_tokens_per_pass, speculative_speedup (Leviathan et al., ICML 2023, arXiv 2211.17192),
  weight_bytes, decode_ceiling, MAC (M4 Max 546 GB/s, Apple Mac Studio (2025) tech specs).
- research/notes.md §6: acceptance rule (Leviathan, verbatim), the losslessness sketch, α = E[Σ min(p, q)],
  E and S formulas, the worked example (q/p = 0.6/0.7, 0.5/0.8, 0.9 vs 0.45 → here 0.9 vs 0.36), "Batch size
  kills the free lunch" (EAGLE-3, arXiv 2503.01840, Table 3: SGLang, H100, Llama-3.1-8B-Instruct, MT-Bench —
  re-checked against the paper on 2026-10-24; the paper also has a batch-56 column, omitted here).
- research/measurements.md §9 / .json "speculative" (2026-10-24, this Mac Studio, mlx-lm 0.32.2, greedy, 256
  tokens, 3 trials, median): acceptance = accepted draft tokens ÷ drafted tokens, reconstructed exactly from
  mlx-lm's per-token from_draft flag (a first pass had an off-by-one counting bug and was superseded). The strips
  are the real from_draft sequences of the BF16 k = 6 runs (raw/spec__llama-3.1-8b__bf16__draft-1b-4bit.json,
  first 42 tokens). Step times: §1 decode ms per token; §8 batching step_ms (episode 4).

Illustrative (said "for example" or marked on screen): the token boundaries; the guesses "tilt of the planet" and
every probability in the lossless section (chosen so p and q each sum to 1); the draw 0.71; α = 0.8 / 0.6 / 0.9
as example acceptance rates; the 8 simulated passes (seed 1; their 100,000-pass mean is computed and asserted).

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from owl_reference import *  # noqa: E402,F403

SERIES_DIR = Path(__file__).resolve().parent.parent
GB = 1e9

# ---------------------------------------------------------------------------- the numbers

ALPHA, GAMMA = 0.8, 4
E = expected_tokens_per_pass(ALPHA, GAMMA)               # 3.36 tokens per target pass
SURVIVE = [ALPHA ** i for i in range(1, GAMMA + 1)]      # guess i survives: 0.8, 0.64, 0.512, 0.41
LIMIT = 1 / (1 - ALPHA)                                   # 5: the most any number of guesses can give at α = 0.8
E09 = expected_tokens_per_pass(0.9, GAMMA)               # 4.10


def simulate(alpha, gamma, n, seed):
    """Kept guesses in n simulated passes: each guess is kept with chance alpha, until the first miss."""
    rng = np.random.default_rng(seed)
    return np.cumprod(rng.random((n, gamma)) < alpha, axis=1).sum(1)


SIM = simulate(ALPHA, GAMMA, 100_000, seed=1)
SIM_MEAN = float((SIM + 1).mean())
ROWS = [int(k) for k in SIM[:8]]                          # the first 8 simulated passes, drawn on screen

W_8B = weight_bytes()                                     # 16.06 GB of BF16 weights
W_MINI = measured("decode", "llama-3.2-1b", "4bit", "weights_gb") * GB   # 0.695 GB on disk
T_MINI = measured("decode", "llama-3.2-1b", "4bit", "ms_per_token")      # 2.05 ms per draft step
T_BF16 = measured("decode", "llama-3.1-8b", "bf16", "ms_per_token")      # 32.3 ms per target pass
T_4BIT = measured("decode", "llama-3.1-8b", "4bit", "ms_per_token")      # 9.8 ms
C_BF16, C_4BIT = T_MINI / T_BF16, T_MINI / T_4BIT
S_BF16 = speculative_speedup(ALPHA, GAMMA, C_BF16)       # 2.68× (example α)
S_4BIT = speculative_speedup(ALPHA, GAMMA, C_4BIT)       # 1.83× (example α)
PLAIN_BF16, SPEC_BF16 = E * T_BF16, GAMMA * T_MINI + T_BF16   # ms for E tokens without / with the draft
PLAIN_4BIT, SPEC_4BIT = E * T_4BIT, GAMMA * T_MINI + T_4BIT

BATCH = measured("batching", "llama-3.1-8b")


def step_ms(bits, rows):
    return BATCH[bits]["by_batch"][str(rows)]["step_ms"]


ROW1_BF16, ROW8_BF16 = step_ms("bf16", 1), step_ms("bf16", 8)   # 32.3 vs 34.1 ms: episode 4
ROW1_4BIT, ROW4_4BIT = step_ms("4bit", 1), step_ms("4bit", 4)   # 9.9 vs 15.0 ms

SPEC = measured("speculative")


def spec(target, prompt, k=None):
    p = SPEC[f"llama-3.1-8b__{target}"]["prompts"][prompt]
    return p if k is None else p["by_num_draft_tokens"][str(k)]


COPY = spec("bf16", "copy_notes", 6)
CHAT = spec("bf16", "open_chat", 6)
CHAT4 = spec("4bit", "open_chat", 2)
COPY4 = spec("4bit", "copy_notes", 4)
BASE_COPY = spec("bf16", "copy_notes")["baseline_tok_s"]          # 30.7 tok/s
BASE_CHAT4 = spec("4bit", "open_chat")["baseline_tok_s"]
CHAT_ACC = [spec(t, "open_chat", k)["acceptance_rate"] for t in ("bf16", "4bit") for k in range(2, 7)]
COPY_ACC = [spec(t, "copy_notes", k)["acceptance_rate"] for t in ("bf16", "4bit") for k in range(2, 7)]
CEIL = decode_ceiling(hw=MAC)                                       # 34 tok/s: one token per pass
ROUND_BYTES = W_8B + 6 * W_MINI                                     # one big pass + six draft steps
SPEC_CEIL = MAC["bandwidth"] * COPY["tokens_per_target_pass"] / ROUND_BYTES   # 187 tok/s
OF_SPEC_CEIL = COPY["tok_s_median"] / SPEC_CEIL                     # 79%

RAW = json.loads((SERIES_DIR / "research" / "measurements" / "raw" / "spec__llama-3.1-8b__bf16__draft-1b-4bit.json").read_text())["results"]
RAW_COPY, RAW_CHAT = RAW["notes_rewrite.txt"]["k6"], RAW["chat.txt"]["k6"]
SEQ_COPY, SEQ_CHAT = RAW_COPY["from_draft_sequence"], RAW_CHAT["from_draft_sequence"]
N_STRIP = 42
assert abs(RAW_COPY["tokens_per_target_pass"] - COPY["tokens_per_target_pass"]) < 1e-9   # raw file == summary
assert abs(RAW_CHAT["tokens_per_target_pass"] - CHAT["tokens_per_target_pass"]) < 1e-9
assert set(SEQ_COPY[:N_STRIP]) == {"D", "T"} and SEQ_COPY[:N_STRIP] == "DDDDDDT" * 6   # six kept, one own, repeated

# the lossless example (illustrative): Owl-8B's p and Owl-mini's q for the third guess's position
CANDS = ["the", "Earth", "our", "its"]
P = [0.36, 0.50, 0.10, 0.04]
Q = [0.90, 0.05, 0.03, 0.02]
OVERLAP = [min(p, q) for p, q in zip(P, Q)]
LEFT = [max(0.0, p - q) for p, q in zip(P, Q)]
KEEP = P[0] / Q[0]                                        # 0.40
DRAW = 0.71                                               # illustrative random draw

EAGLE_B = [2, 4, 8, 16, 24, 32, 48, 64]                   # EAGLE-3 paper, Table 3 (notes §6), cited not computed
EAGLE1 = [1.40, 1.38, 1.23, 1.02, 0.93, 0.94, 0.88, 0.99]
EAGLE3 = [1.81, 1.82, 1.62, 1.48, 1.39, 1.32, 1.38, 1.38]

# every number the narration says
assert round(E, 2) == 3.36 and abs(sum(SURVIVE) + 1 - E) < 1e-12                 # "three point three six"
assert [round(s, 2) for s in SURVIVE] == [0.8, 0.64, 0.51, 0.41]
assert round(LIMIT, 6) == 5                                                       # "five tokens"
assert round(E09, 1) == 4.1                                                       # "four point one"
assert round(SIM_MEAN, 2) == 3.36 and abs(SIM_MEAN - E) < 0.01                    # simulation agrees
assert round(W_8B / W_MINI) == 23                                                 # "twenty-three times fewer bytes"
assert round(T_MINI) == 2 and round(T_BF16) == 32 and T_4BIT < 10                 # "two", "thirty-two", "under ten"
assert round(C_BF16 * 100) == 6 and round(C_4BIT, 1) == 0.2                      # "six percent", "a fifth"
assert round(S_BF16, 1) == 2.7 and round(S_4BIT, 1) == 1.8                       # "two point seven", "one point eight"
assert round(ROW1_BF16) == 32 and round(ROW8_BF16) == 34 and ROW8_BF16 / ROW1_BF16 < 1.06   # "barely longer"
assert round(ROW4_4BIT / ROW1_4BIT, 1) == 1.5                                     # "half again as long"
assert all(a == 1.0 for a in COPY_ACC)                                            # "every guess was kept"
assert round(BASE_COPY) == 31 and round(COPY["tok_s_median"]) == 148              # "thirty-one … a hundred and forty-eight"
assert round(COPY["speedup"], 2) == 4.82 and round(COPY["tokens_per_target_pass"], 2) == 6.92   # "almost seven"
assert round(CEIL) == 34 and round(SPEC_CEIL) == 187 and round(ROUND_BYTES / GB) == 20   # "about twenty gigabytes"
assert round(CHAT["tokens_per_target_pass"], 2) == 3.28                           # "about three tokens"
assert round(min(CHAT_ACC) * 100) == 38 and round(max(CHAT_ACC) * 100) == 72      # "thirty-eight to seventy-two"
assert round(CHAT["speedup"], 2) == 2.28 and round(CHAT4["speedup"], 2) == 1.42   # "two point two eight", "one point four two"
assert round(COPY4["speedup"], 2) == 1.83
assert spec("bf16", "open_chat", 6)["first_divergence_token"] == 154 and COPY["output_identical_to_baseline"]
assert abs(sum(P) - 1) < 1e-9 and abs(sum(Q) - 1) < 1e-9 and round(KEEP, 2) == 0.40     # "four times in ten"
assert abs(sum(OVERLAP) + sum(LEFT) - 1) < 1e-9 and all(abs(o + l - p) < 1e-9 for o, l, p in zip(OVERLAP, LEFT, P))
assert P.index(max(P)) == 1                                                       # Earth is Owl-8B's top pick (greedy)
assert min(EAGLE1[EAGLE_B.index(24):]) < 1 and all(e >= 1 for e in EAGLE1[:EAGLE_B.index(24)])   # "below one from twenty-four"
assert round(EAGLE3[-1], 1) == 1.4                                                # "about one point four"

# ---------------------------------------------------------------------------- look

PREFIX = "ep13"
STATES = {   # a token's state: (outline, fill)
    "ctx": (EDGE, CARD),
    "guess": (MUTED, CARD_HI),
    "kept": (TOKEN, GREEN_FILL),
    "rejected": (RED, RED_FILL),
    "own": (WEIGHTS, BLUE_FILL),       # written by Owl-8B itself: the pass's own token
    "gone": (FAINT, CARD),
}


def cite(s, size=20):
    return text(s, size, MUTED)


def tchip(word, state="guess", width=1.45, height=0.62):
    color, fill = STATES[state]
    return chip(word, color, fill, width=width, height=height, size=26, mono=True)


def restyle(ch, state):
    color, fill = STATES[state]
    return ch[0].animate.set_stroke(color).set_fill(fill, 1)


def cell(state, size=0.2):
    color, fill = STATES[state]
    sq = m.RoundedRectangle(corner_radius=size * 0.18, width=size, height=size).set_stroke(color, 1.6)
    return sq.set_fill(fill if state in ("gone", "guess", "ctx") else color, 1 if state in ("gone", "guess", "ctx") else 0.8)


def tick_mark(color=TOKEN, size=0.34):
    pts = [np.array(p) * size for p in ([-0.5, 0.02, 0], [-0.14, -0.36, 0], [0.5, 0.42, 0])]
    return m.VMobject().set_points_as_corners(pts).set_stroke(color, 6)


def cross_mark(color=RED, size=0.3):
    h = size / 2
    return m.VGroup(m.Line([-h, -h, 0], [h, h, 0]), m.Line([-h, h, 0], [h, -h, 0])).set_stroke(color, 6)


def model_card(name, lines, color, width=3.9, owl_size=0.42):
    """A model as a card: Owl, a bold name, one or two mono spec lines. Parts: [0] box, [1] body."""
    words = m.VGroup(text(name, 28, INK, weight=m.BOLD), *[code(t, 20, MUTED) for t in lines])
    words.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.08)
    body = m.VGroup(owl(owl_size), words).arrange(m.RIGHT, buff=0.3)
    box = card(max(width, body.width + 0.7), body.height + 0.45, color, CARD)
    body.move_to(box).align_to(box, m.LEFT).shift(0.35 * m.RIGHT)
    return m.VGroup(box, body)


def hbar(x0, y, width, height, color, fill, stroke=2):
    bar = m.Rectangle(width=max(width, 0.04), height=height).set_stroke(color, stroke).set_fill(fill, 1)
    return bar.move_to([x0 + max(width, 0.04) / 2, y, 0])


def strip(seq, n, size=0.2, gap=0.035, group_gap=0.13):
    """The real accept strip: D = drafted by Owl-mini and kept (GREEN), T = Owl-8B's own token (BLUE); a gap
    after each T closes one pass of Owl-8B."""
    cells, x = m.VGroup(), 0.0
    for ch in seq[:n]:
        cells.add(cell("kept" if ch == "D" else "own", size).move_to([x + size / 2, 0, 0]))
        x += size + gap + (group_gap if ch == "T" else 0)
    return cells


def swatch(state, label, size=0.26):
    words = text(label, 20, MUTED)
    return m.VGroup(cell(state, size), words).arrange(m.RIGHT, buff=0.15)


def polyline(points, color, width=4):
    return m.VMobject().set_points_as_corners(points).set_stroke(color, width=width)


class Chart:
    """Quiet axes with the origin at the lower-left corner (as in "Agents, Animated" Ep29)."""

    def __init__(self, x_range, y_range, width, height, center):
        self.x0, self.y0 = x_range[0], y_range[0]
        self.ax = m.Axes(
            x_range=x_range, y_range=y_range, x_length=width, y_length=height,
            axis_config={"color": FAINT, "stroke_width": 2.5, "include_tip": True, "tip_length": 0.18, "include_ticks": False},
        ).move_to(center)

    def p(self, x, y):
        return self.ax.c2p(x, y)

    def xlabel(self, x, s):
        return code(s, 20, MUTED).move_to(self.p(x, self.y0) + 0.3 * m.DOWN)

    def ylabel(self, y, s):
        return code(s, 20, MUTED).next_to(self.p(self.x0, y), m.LEFT, buff=0.15)


# ---------------------------------------------------------------------------- the episode


class Ep13GuessAndCheck(Episode):
    number = 13

    def lesson(self):
        self.idea()
        self.draft_verify()
        self.lossless()
        self.per_pass()
        self.cost()
        self.measured_copy()
        self.measured_chat()
        self.busy()
        self.recap()

    # ------------------------------------------------------------------ 1 · the idea

    def idea(self):
        self.next_section("1 · The idea", notes=(
            f"Episode 2: a decode step reads all {W_8B / GB:.0f} GB of Owl-8B's BF16 weights for one token; on this Mac a "
            f"step takes {T_BF16:.1f} ms (measured, research/measurements.md §1). Episode 4: a pass over 8 rows takes "
            f"{ROW8_BF16:.1f} ms vs {ROW1_BF16:.1f} ms for one row (measured batching step times, §8). The answer tokens "
            "are the start of the measured run's real greedy answer (token boundaries illustrative)."))
        big = model_card("Owl-8B", ["8B · BF16", f"{W_8B / GB:.0f} GB of weights"], WEIGHTS).move_to([-4.3, 1.5, 0])
        toks = m.VGroup(*[tchip(w, "kept", width=1.55) for w in ("Hello", "there,", "friend.")])
        toks.arrange(m.RIGHT, buff=0.18).move_to([0, 1.5, 0]).align_to([-1.1, 0, 0], m.LEFT)
        lane = m.Line(big.get_right() + [0.12, 0, 0], toks.get_left() + [-0.12, 0, 0])
        passes = m.VGroup(*[code(f"pass {i + 1}", 20, MUTED).next_to(t, m.DOWN, buff=0.18) for i, t in enumerate(toks)])
        each = code(f"1 pass = all {W_8B / GB:.0f} GB read = {T_BF16:.0f} ms here", 22, MEMORY)
        each.next_to(passes, m.DOWN, buff=0.3).align_to(toks, m.LEFT)

        def flash():
            return m.ShowPassingFlash(lane.copy().set_stroke(MEMORY, 8), time_width=0.7)

        sweep = []
        for t, lab in zip(toks, passes):
            sweep += [m.AnimationGroup(flash(), run_time=0.55),
                      m.AnimationGroup(m.FadeIn(t, shift=0.15 * m.UP), m.FadeIn(lab), run_time=0.4)]

        x0, scale = -1.6, 0.125
        rows_title = text("one pass of Owl-8B, over 1 row or 8 rows · episode 4", 22, MUTED).move_to([0, -0.6, 0]).align_to([x0 - 1.3, 0, 0], m.LEFT)
        bars = []
        for y, label, ms in ((-1.2, "1 row", ROW1_BF16), (-1.85, "8 rows", ROW8_BF16)):
            lab = code(label, 22, INK).move_to([0, y, 0]).align_to([x0 - 0.25, 0, 0], m.RIGHT)
            bar = hbar(x0, y, scale * ms, 0.42, MEMORY, CYAN_FILL)
            val = code(f"{ms:.1f} ms", 22, MEMORY).next_to(bar, m.RIGHT, buff=0.2)
            bars.append(m.VGroup(lab, bar, val))
        rows_src = cite("measured on this Mac · mlx-lm").next_to(bars[1], m.DOWN, buff=0.25).align_to(rows_title, m.LEFT)
        ep4 = m.VGroup(rows_title, *bars, rows_src)

        mini = model_card("Owl-mini", ["1B · 4-bit", f"{W_MINI / GB:.1f} GB of weights"], MUTED, owl_size=0.32).move_to([-4.3, -1.5, 0])
        guesses = m.VGroup(*[tchip(w, "guess", width=1.75 if len(w) > 5 else 1.4) for w in ("I'm", "Owl,", "your", "friendly")])
        guesses.arrange(m.RIGHT, buff=0.14).move_to([0, -1.5, 0]).align_to([-1.1, 0, 0], m.LEFT)
        box = m.SurroundingRectangle(guesses, buff=0.16, corner_radius=0.14).set_stroke(WEIGHTS, 3)
        box_lab = code("checked together, in one pass of Owl-8B", 20, WEIGHTS).next_to(box, m.DOWN, buff=0.18).align_to(box, m.LEFT)

        hero = title_text("speculative decoding", 72).move_to([0, 0.9, 0])
        p1 = pill("exactly lossless", TOKEN, GREEN_FILL, 24)
        p2 = pill("measured on this Mac", MEMORY, CYAN_FILL, 24)
        pills = m.VGroup(p1, p2).arrange(m.RIGHT, buff=0.6).move_to([0, -0.55, 0])
        mascot = owl(0.7).move_to([0, -2.3, 0])
        frame1 = m.VGroup(big, toks, passes, each, mini, guesses, box, box_lab)

        self.say(
            "[Every token Owl writes] [costs one pass of the big model]: [all sixteen gigabytes read, for a single token.]",
            m.AnimationGroup(self.new_header("01 · the idea", "One pass, one token"), m.FadeIn(big, shift=0.2 * m.UP)),
            m.Succession(*sweep),
            m.FadeIn(each, shift=0.1 * m.UP),
        )
        self.say(
            "[Episode four found the strange part]: [a pass over eight rows] [takes barely longer than a pass over one.]",
            m.AnimationGroup(m.FadeIn(rows_title), m.FadeIn(bars[0][0]), m.GrowFromEdge(bars[0][1], m.LEFT), m.FadeIn(bars[0][2])),
            m.AnimationGroup(m.FadeIn(bars[1][0]), m.GrowFromEdge(bars[1][1], m.LEFT)),
            m.AnimationGroup(m.FadeIn(bars[1][2]), m.FadeIn(rows_src)),
        )
        self.say(
            "[So what if a small model guessed Owl's next few tokens], [and the big one checked them all], [in one pass?]",
            m.AnimationGroup(m.FadeOut(ep4), m.FadeIn(mini, shift=0.2 * m.UP),
                             m.LaggedStart(*[m.FadeIn(g, shift=0.15 * m.RIGHT) for g in guesses], lag_ratio=0.25)),
            m.Create(box),
            m.FadeIn(box_lab, shift=0.1 * m.UP),
        )
        self.say(
            "[That's speculative decoding.] [Today: why it's exact], [and what it buys on this Mac.]",
            m.AnimationGroup(m.FadeOut(frame1, shift=0.2 * m.UP), m.FadeIn(hero, shift=0.2 * m.UP)),
            m.FadeIn(p1, shift=0.15 * m.UP),
            m.AnimationGroup(m.FadeIn(p2, shift=0.15 * m.UP), m.FadeIn(mascot, shift=0.2 * m.UP)),
        )
        self.wait(0.15)
        self.clear()

    # ------------------------------------------------------------------ 2 · draft and verify

    def draft_verify(self):
        self.next_section("2 · Draft and verify", notes=(
            f"Owl-mini = Llama-3.2-1B-Instruct 4-bit ({W_MINI / GB:.3f} GB on disk, {W_8B / W_MINI:.0f}× fewer bytes than "
            f"Owl-8B's {W_8B / GB:.2f} GB); one draft step takes {T_MINI:.2f} ms on this Mac (measured, §1). The target scores "
            "all γ guesses (plus one more position) in one pass and keeps the longest acceptable prefix, plus one token of "
            "its own (notes §6). The sentence comes from the measured chat answer ('The real reason for the seasons is the "
            "tilt of Earth's axis'); the guesses and the verdicts are illustrative (adapted from notes §6's worked example). "
            "GREEN = a drafted token that was kept; BLUE = Owl-8B's own token; RED = rejected; dim = thrown away."))
        mini = model_card("Owl-mini", ["draft · 1B · 4-bit", f"{W_MINI / GB:.1f} GB of weights"], MUTED, owl_size=0.32)
        mini.move_to([0, 1.85, 0]).align_to([-6.3, 0, 0], m.LEFT)
        big = model_card("Owl-8B", ["target · 8B · BF16", f"{W_8B / GB:.0f} GB of weights"], WEIGHTS)
        big.move_to([0, 1.85, 0]).align_to([6.3, 0, 0], m.RIGHT)
        ctx = m.VGroup(*[tchip(w, "ctx", width=1.1) for w in ("is", "the")]).arrange(m.RIGHT, buff=0.1)
        gs = m.VGroup(*[tchip(w, "guess") for w in ("tilt", "of", "the", "planet")]).arrange(m.RIGHT, buff=0.12)
        row = m.VGroup(ctx, gs).arrange(m.RIGHT, buff=0.4).move_to([-0.3, 0.2, 0])
        dots = code("…seasons", 24, MUTED).next_to(ctx, m.LEFT, buff=0.2)
        draft_lab = code(f"4 guesses · {T_MINI:.0f} ms each", 20, MUTED).next_to(mini, m.DOWN, buff=0.16).align_to(mini, m.LEFT)
        box = m.SurroundingRectangle(gs, buff=0.14, corner_radius=0.12).set_stroke(WEIGHTS, 3)
        pass_lab = code(f"1 pass · {W_8B / GB:.0f} GB read once", 20, WEIGHTS).next_to(big, m.DOWN, buff=0.16).align_to(big, m.RIGHT)
        sweep_line = m.Line(box.get_left(), box.get_right())
        marks = [tick_mark(), tick_mark(), cross_mark(), None]
        y_mark = -0.62
        for mk, g in zip(marks, gs):
            if mk is not None:
                mk.move_to([g.get_x(), y_mark, 0])
        gone_lab = code("thrown away", 20, FAINT).move_to([gs[3].get_x(), y_mark, 0])

        result = m.VGroup(tchip("tilt", "kept"), tchip("of", "kept"), tchip("Earth", "own"))
        for r, g in zip(result, gs):
            r.move_to([g.get_x(), -1.55, 0])
        res_lab = code("= 3 tokens", 24, TOKEN, weight=m.BOLD).move_to([0, -1.55, 0]).align_to(gs[3], m.LEFT)

        all_lab = text("if all four pass:", 22, MUTED).move_to([0, -2.75, 0]).align_to([gs[0].get_left()[0] - 0.3, 0, 0], m.RIGHT)
        bonus = m.VGroup(*[cell("kept", 0.46).move_to([g.get_x(), -2.75, 0]) for g in gs])
        extra = cell("own", 0.46).move_to([gs[3].get_x() + 0.62, -2.75, 0])
        bonus_lab = code("= 5 tokens", 24, TOKEN, weight=m.BOLD).next_to(extra, m.RIGHT, buff=0.3)
        legend = m.VGroup(swatch("kept", "drafted, kept"), swatch("own", "Owl-8B's own"))
        legend.arrange(m.DOWN, buff=0.18, aligned_edge=m.LEFT).move_to([0, -1.55, 0]).align_to([-6.2, 0, 0], m.LEFT)

        self.say(
            "[Meet the drafter]: [Owl-mini, a one-billion-parameter model in four bits], [with twenty-three times fewer bytes than Owl-8B.]",
            m.AnimationGroup(self.new_header("02 · draft and verify", "Four guesses, one pass"), m.FadeIn(mini, shift=0.2 * m.UP)),
            m.Circumscribe(mini[1][1][2], color=MUTED, buff=0.06),
            m.FadeIn(big, shift=0.2 * m.UP),
        )
        self.say(
            "[It guesses the next four tokens], [one at a time, two milliseconds each]: [tilt, of, the, planet.]",
            m.AnimationGroup(m.FadeIn(dots), m.FadeIn(ctx, shift=0.1 * m.UP)),
            m.FadeIn(draft_lab, shift=0.1 * m.UP),
            m.LaggedStart(*[m.FadeIn(g, target_position=mini) for g in gs], lag_ratio=0.45),
        )
        self.say(
            "[Then Owl-8B checks all four in a single pass], [reading all sixteen gigabytes just once.]",
            m.AnimationGroup(m.Create(box), m.ShowPassingFlash(sweep_line.copy().set_stroke(WEIGHTS, 10), time_width=0.6)),
            m.AnimationGroup(m.FadeIn(pass_lab, shift=0.1 * m.UP), m.Circumscribe(big, color=WEIGHTS, buff=0.06)),
        )
        self.say(
            "[Tilt, kept.] [Of, kept.] [But 'the' is rejected], [and the guess after it is thrown away.]",
            m.AnimationGroup(restyle(gs[0], "kept"), m.Create(marks[0])),
            m.AnimationGroup(restyle(gs[1], "kept"), m.Create(marks[1])),
            m.AnimationGroup(restyle(gs[2], "rejected"), m.Create(marks[2])),
            m.AnimationGroup(gs[3].animate.set_opacity(0.3), m.FadeIn(gone_lab)),
        )
        self.say(
            "[In its place, Owl-8B writes its own token: Earth.] [Three tokens, from one pass of the big model.]",
            m.AnimationGroup(m.FadeIn(result[0], target_position=gs[0]), m.FadeIn(result[1], target_position=gs[1]),
                             m.FadeIn(result[2], target_position=big)),
            m.AnimationGroup(m.FadeIn(res_lab, shift=0.1 * m.LEFT), m.FadeIn(legend)),
        )
        self.say(
            "[And when all four guesses pass], [the same pass adds a fifth token of its own.]",
            m.AnimationGroup(m.FadeIn(all_lab), m.LaggedStart(*[m.FadeIn(b, scale=0.6) for b in bonus], lag_ratio=0.2)),
            m.AnimationGroup(m.FadeIn(extra, target_position=big), m.FadeIn(bonus_lab, shift=0.1 * m.LEFT)),
        )
        self.wait(0.15)
        self.clear()

    # ------------------------------------------------------------------ 3 · exactly lossless

    def lossless(self):
        self.next_section("3 · Exactly lossless", notes=(
            "Leviathan et al. 2023 (verbatim, notes §6): 'To sample x ∼ p(x), we instead sample x ∼ q(x), keeping it if "
            "q(x) ≤ p(x), and in case q(x) > p(x) we reject the sample with probability 1 − p(x)/q(x) and sample x again from "
            "an adjusted distribution p′(x) = norm(max(0, p(x) − q(x))).' Then P(x) = min(p, q)(x) + (p − q)⁺(x) = p(x): exactly "
            "the target's distribution (Chen et al. 2023: 'within hardware numerics'). p = Owl-8B, q = Owl-mini (Leviathan's "
            f"letters). The probabilities are illustrative: p = {P}, q = {Q} over {CANDS}; keep chance {KEEP:.2f}; overlap "
            f"Σ min = {sum(OVERLAP):.2f}; the draw 0.71 is illustrative. Greedy (temperature 0, as measured here) is the special "
            "case: a guess is kept only if it is the target's argmax — here 'Earth', so 'the' is rejected either way."))
        top = m.VGroup(tchip("tilt", "kept", width=1.2), tchip("of", "kept", width=1.2), tchip("the", "guess", width=1.2))
        top.arrange(m.RIGHT, buff=0.1).move_to([0, 2.05, 0]).align_to([-6.2, 0, 0], m.LEFT)
        third = text("the third guess", 22, MUTED).next_to(top, m.RIGHT, buff=0.35)
        earth = tchip("Earth", "own", width=1.2).move_to(top[2])
        the_red = tchip("the", "rejected", width=1.2).move_to(top[2])
        leg_p = m.VGroup(m.Square(0.26).set_stroke(WEIGHTS, 2.5).set_fill(BLUE_FILL, 1), code("p · Owl-8B", 20, WEIGHTS)).arrange(m.RIGHT, buff=0.15)
        leg_q = m.VGroup(m.Square(0.26).set_stroke(MUTED, 0).set_fill(MUTED, 0.85), code("q · Owl-mini", 20, MUTED)).arrange(m.RIGHT, buff=0.15)
        legend = m.VGroup(leg_p, leg_q).arrange(m.RIGHT, buff=0.45).move_to([0, 2.05, 0]).align_to([1.4, 0, 0], m.LEFT)

        xs, width, base, s = [-5.2, -3.5, -1.8, -0.1], 1.15, -1.75, 2.75
        pbars, qbars, names, vals, over, left = (m.VGroup() for _ in range(6))
        for x, name, p, q, o, lft in zip(xs, CANDS, P, Q, OVERLAP, LEFT):
            pbars.add(m.Rectangle(width=width, height=p * s).set_stroke(WEIGHTS, 3).set_fill(BLUE_FILL, 1).move_to([x, base + p * s / 2, 0]))
            qbars.add(m.Rectangle(width=0.4, height=q * s).set_stroke(width=0).set_fill(MUTED, 0.85).move_to([x, base + q * s / 2, 0]).set_z_index(1))
            names.add(code(name, 24, INK).move_to([x, base - 0.33, 0]))
            vals.add(m.VGroup(code(f"p {p:.2f}", 20, WEIGHTS), code(f"q {q:.2f}", 20, MUTED)).arrange(m.DOWN, buff=0.08).move_to([x, base - 1.0, 0]))
            over.add(m.Rectangle(width=width, height=max(o * s, 0.001)).set_stroke(width=0).set_fill(TOKEN, 0.85).move_to([x, base + o * s / 2, 0]).set_z_index(2))
            lr = m.Rectangle(width=width, height=max(lft * s, 0.001)).set_stroke(width=0).set_fill(WEIGHTS, 0.85)
            left.add(lr.move_to([x, base + o * s + lft * s / 2, 0]).set_z_index(2))
        axis = m.Line([xs[0] - 0.8, base, 0], [xs[-1] + 0.8, base, 0]).set_stroke(FAINT, 2.5)
        chart = m.VGroup(axis, pbars, qbars, names, vals)

        px = 1.4
        rule1 = m.VGroup(tick_mark(TOKEN, 0.3), text("q ≤ p: drafter less sure · keep", 22, INK)).arrange(m.RIGHT, buff=0.22)
        rule1.move_to([0, 1.0, 0]).align_to([px, 0, 0], m.LEFT)
        rule2 = text("q > p: keep with chance p ÷ q", 22, INK).move_to([0, 0.2, 0]).align_to([px, 0, 0], m.LEFT)
        ratio = code(f"{P[0]:.2f} ÷ {Q[0]:.2f} = {KEEP:.2f}", 26, INK, t2c={rf"^{P[0]:.2f}": WEIGHTS, rf"÷ {Q[0]:.2f}": MUTED, rf"{KEEP:.2f}$": TOKEN})
        ratio.next_to(rule2, m.DOWN, buff=0.25).align_to([px, 0, 0], m.LEFT)
        draw = pill(f"draw {DRAW:.2f} > {KEEP:.2f} → reject", RED, RED_FILL, 22, mono=True).move_to([0, -1.25, 0]).align_to([px, 0, 0], m.LEFT)
        resample = text("resample from the leftover, p − q", 22, WEIGHTS).move_to([0, -2.05, 0]).align_to([px, 0, 0], m.LEFT)
        f_parts = m.MathTex(r"\min(p,q)", "+", r"(p-q)^{+}", "=", "p", font_size=46, color=INK)
        f_parts[0].set_color(TOKEN)
        f_parts[2].set_color(WEIGHTS)
        f_parts.move_to([0, -2.95, 0]).align_to([px, 0, 0], m.LEFT)
        f_words = text("kept  +  resampled  =  Owl-8B's own", 20, MUTED).next_to(f_parts, m.UP, buff=0.18).align_to([px, 0, 0], m.LEFT)
        lossless = pill("exactly lossless", TOKEN, GREEN_FILL, 24).move_to([0, 1.05, 0]).align_to([px, 0, 0], m.LEFT)
        src = cite("Leviathan et al., 2023 · Chen et al., 2023").next_to(lossless, m.DOWN, buff=0.18).align_to([px, 0, 0], m.LEFT)
        greedy = text("greedy: keep only Owl-8B's top pick", 22, INK).move_to([0, -0.45, 0]).align_to([px, 0, 0], m.LEFT)
        top_pick = code("top pick", 20, WEIGHTS, weight=m.BOLD).next_to(pbars[1], m.UP, buff=0.12)

        self.say(
            "[Why is the result exactly Owl-8B's?] [Look at the third guess], [where the models disagree.]",
            m.AnimationGroup(self.new_header("03 · exactly lossless", "Keep, or resample"), m.FadeIn(top, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(third), m.Circumscribe(top[2], color=MUTED, buff=0.06)),
            m.AnimationGroup(m.Create(axis), m.FadeIn(names, shift=0.1 * m.UP)),
        )
        self.say(
            "[Each model gives every candidate a probability]: [Owl-8B in blue], [Owl-mini in grey.]",
            m.FadeIn(vals),
            m.AnimationGroup(m.FadeIn(leg_p), m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in pbars], lag_ratio=0.15)),
            m.AnimationGroup(m.FadeIn(leg_q), m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in qbars], lag_ratio=0.15)),
        )
        self.say(
            "[When the drafter is less sure than Owl-8B, a guess is always kept]: [that's how tilt and of got through.]",
            m.FadeIn(rule1, shift=0.1 * m.LEFT),
            m.AnimationGroup(m.Circumscribe(top[0], color=TOKEN, buff=0.05), m.Circumscribe(top[1], color=TOKEN, buff=0.05)),
        )
        self.say(
            "[But Owl-mini was ninety percent sure of 'the', and Owl-8B says thirty-six]: [so it's kept only four times in ten.]",
            m.AnimationGroup(m.FadeIn(rule2, shift=0.1 * m.LEFT), m.Circumscribe(m.VGroup(pbars[0], qbars[0]), color=INK, buff=0.08)),
            m.FadeIn(ratio, shift=0.1 * m.UP),
        )
        self.say(
            "[This time, the draw says no.] [So Owl-8B samples from the leftover]: [what it wanted more than the drafter offered.]",
            m.AnimationGroup(m.FadeIn(draw, shift=0.1 * m.UP), m.FadeOut(top[2]), m.FadeIn(the_red)),
            m.AnimationGroup(m.FadeIn(resample), m.LaggedStart(*[m.GrowFromEdge(lr, m.DOWN) for lr in left[1:]], lag_ratio=0.2)),
            m.AnimationGroup(m.FadeOut(the_red, shift=0.25 * m.UP), m.FadeIn(earth, shift=0.25 * m.UP), m.Circumscribe(left[1], color=WEIGHTS, buff=0.06)),
        )
        self.say(
            "[Kept guesses fill the overlap], [the leftover fills the rest], [and together they rebuild Owl-8B's own probabilities, exactly.]",
            m.AnimationGroup(qbars.animate.set_opacity(0.25), m.LaggedStart(*[m.GrowFromEdge(o, m.DOWN) for o in over], lag_ratio=0.15)),
            m.AnimationGroup(*[m.Indicate(lr, color=WEIGHTS, scale_factor=1.05) for lr in left[1:]]),
            m.AnimationGroup(m.FadeOut(m.VGroup(rule1, rule2, ratio, draw), shift=0.1 * m.UP), m.FadeIn(f_words), m.FadeIn(f_parts, shift=0.1 * m.UP),
                             *[m.Circumscribe(b, color=WEIGHTS, buff=0.04) for b in pbars]),
        )
        self.say(
            "[So the output is exactly the big model's.] [With greedy decoding, the rule is simpler]: "
            "[keep a guess only if it's Owl-8B's top pick.]",
            m.AnimationGroup(m.FadeIn(lossless, shift=0.1 * m.UP), m.FadeIn(src)),
            m.FadeIn(greedy, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(top_pick, shift=0.1 * m.DOWN), m.Circumscribe(pbars[1], color=WEIGHTS, buff=0.06)),
        )
        self.wait(0.15)
        self.clear()

    # ------------------------------------------------------------------ 4 · tokens per pass

    def per_pass(self):
        self.next_section("4 · Tokens per pass", notes=(
            "Leviathan et al. 2023: α = E[Σ_x min(p(x), q(x))], the expected acceptance rate (the average overlap); with γ "
            f"guesses, E[tokens per target pass] = (1 − α^(γ+1)) / (1 − α) (owl_reference.expected_tokens_per_pass). α = {ALPHA}, "
            f"γ = {GAMMA}: 1 + 0.8 + 0.64 + 0.512 + 0.4096 = {E:.4f}. The simulation draws each guess kept with chance α "
            f"until the first miss (NumPy, seed 1): the 8 rows drawn are its first passes; the mean of 100,000 passes is "
            f"{SIM_MEAN:.4f}. Limit 1/(1 − α) = {LIMIT:.0f} at α = 0.8; α = 0.9, γ = 4 → {E09:.2f}. α values are examples."))
        a_def = m.VGroup(tex(r"\alpha", 44, TOKEN), text(f"= {ALPHA} · the chance a guess is kept: the average overlap", 24, INK)).arrange(m.RIGHT, buff=0.2)
        g_def = m.VGroup(tex(r"\gamma", 44, INK), text(f"= {GAMMA} · guesses per pass", 24, INK)).arrange(m.RIGHT, buff=0.2)
        for d, y in ((a_def, 2.1), (g_def, 1.45)):
            d.move_to([0, y, 0]).align_to([-6.2, 0, 0], m.LEFT)
        g_def[0].align_to(a_def[0], m.LEFT)
        g_def[1].align_to(a_def[1], m.LEFT)

        xs, width, base, s = [-5.6, -4.55, -3.5, -2.45, -1.05], 0.78, -2.15, 2.5
        heights = SURVIVE + [1.0]
        bars, tops, subs = m.VGroup(), m.VGroup(), m.VGroup()
        for i, (x, h) in enumerate(zip(xs, heights)):
            own = i == GAMMA
            color, fill = STATES["own" if own else "kept"]
            bars.add(m.Rectangle(width=width, height=h * s).set_stroke(color, 2.5).set_fill(fill, 1).move_to([x, base + h * s / 2, 0]))
            tops.add(code(f"{h:.2f}" if not own else "1", 20, color).next_to(bars[-1], m.UP, buff=0.12))
            sub = code("own", 24, WEIGHTS) if own else tex(r"\alpha" if i == 0 else rf"\alpha^{{{i + 1}}}", 32, TOKEN)
            subs.add(sub.move_to([x, base - 0.4, 0]))
        axis = m.Line([xs[0] - 0.6, base, 0], [xs[-1] + 0.6, base, 0]).set_stroke(FAINT, 2.5)

        fx = 0.7
        f_sum = m.MathTex("E", "=", "1", "+", r"\alpha", "+", r"\alpha^{2}", "+", r"\alpha^{3}", "+", r"\alpha^{4}", font_size=46, color=INK)
        f_sum[0].set_color(TOKEN)
        f_sum.move_to([0, 0.55, 0]).align_to([fx, 0, 0], m.LEFT)
        f_closed = m.MathTex("=", r"\frac{1-\alpha^{\gamma+1}}{1-\alpha}", font_size=46, color=INK)
        f_closed.next_to(f_sum, m.DOWN, buff=0.35).align_to(f_sum[1], m.LEFT)
        nums = code("1 + 0.8 + 0.64 + 0.51 + 0.41", 24, INK, t2c={r"^1": WEIGHTS, r"0\.\d+": TOKEN})
        nums.move_to([0, -1.55, 0]).align_to([fx, 0, 0], m.LEFT)
        result = stat(f"{E:.2f}", "tokens per pass", "", TOKEN, 64).move_to([0, -2.6, 0]).align_to([fx, 0, 0], m.LEFT)
        frame1 = m.VGroup(a_def, g_def, axis, bars, tops, subs, f_sum, f_closed, nums, result)

        self.say(
            "[That overlap, averaged over the text, is the acceptance rate: alpha.] [And gamma is the number of guesses.]",
            m.AnimationGroup(self.new_header("04 · tokens per pass", "Alpha and gamma"), m.FadeIn(a_def, shift=0.1 * m.UP)),
            m.FadeIn(g_def, shift=0.1 * m.UP),
        )
        self.say(
            "[The first guess survives with chance alpha.] [The second needs both]: [alpha squared], [and so on.]",
            m.AnimationGroup(m.Create(axis), m.GrowFromEdge(bars[0], m.DOWN), m.FadeIn(tops[0]), m.FadeIn(subs[0])),
            m.AnimationGroup(m.GrowFromEdge(bars[1], m.DOWN), m.FadeIn(tops[1])),
            m.FadeIn(subs[1]),
            m.AnimationGroup(*[m.AnimationGroup(m.GrowFromEdge(bars[i], m.DOWN), m.FadeIn(tops[i]), m.FadeIn(subs[i])) for i in (2, 3)]),
        )
        self.say(
            "[Owl-8B's own token always comes], [so one pass makes one, plus alpha, plus alpha squared, and so on.]",
            m.AnimationGroup(m.GrowFromEdge(bars[4], m.DOWN), m.FadeIn(tops[4]), m.FadeIn(subs[4])),
            m.AnimationGroup(m.FadeIn(f_sum, shift=0.1 * m.UP), m.FadeIn(f_closed, shift=0.1 * m.UP)),
        )
        self.say(
            "[With alpha point eight and four guesses], [that adds up to three point three six tokens per pass.]",
            m.AnimationGroup(m.FadeIn(nums, shift=0.1 * m.UP), m.Circumscribe(a_def[0], color=TOKEN, buff=0.06)),
            m.FadeIn(result, scale=0.9),
        )

        rows_title = text("8 simulated passes", 22, MUTED).move_to([0, 2.15, 0]).align_to([-6.2, 0, 0], m.LEFT)
        sim_rows = m.VGroup()
        for r, kept in enumerate(ROWS):
            y = 1.55 - r * 0.45
            cells = m.VGroup()
            for j in range(GAMMA):
                state = "kept" if j < kept else ("rejected" if j == kept else "gone")
                cells.add(cell(state, 0.32).move_to([-6.0 + j * 0.4, y, 0]))
            cells.add(cell("own", 0.32).move_to([-6.0 + GAMMA * 0.4 + 0.18, y, 0]))
            count = code(str(kept + 1), 24, TOKEN).move_to([-3.6, y, 0])
            sim_rows.add(m.VGroup(cells, count))
        mean_lab = text("mean of 100,000 passes", 20, MUTED).move_to([0, -2.25, 0]).align_to([-6.2, 0, 0], m.LEFT)
        mean = stat(f"{SIM_MEAN:.2f}", "tokens", "", TOKEN, 52).move_to([0, -2.95, 0]).align_to([-6.2, 0, 0], m.LEFT)

        ch = Chart([0, 10, 1], [0, 7, 1], 5.6, 4.3, (2.4, -0.45, 0))
        x_ticks = m.VGroup(*[ch.xlabel(g, str(g)) for g in (0, 2, 4, 6, 8, 10)])
        y_ticks = m.VGroup(*[ch.ylabel(v, str(v)) for v in (1, 3, 5, 7)])
        x_name = text("γ · guesses per pass", 22, MUTED).next_to(x_ticks, m.DOWN, buff=0.12).set_x(ch.p(5, 0)[0])
        y_name = text("tokens per pass", 22, MUTED).next_to(ch.p(0, 7), m.RIGHT, buff=0.25)
        curves = {}
        for a, color in ((0.6, MUTED), (0.8, TOKEN), (0.9, INK)):
            curves[a] = polyline([ch.p(g, expected_tokens_per_pass(a, g)) for g in np.linspace(0, 10, 41)], color, 4)
        lab06 = code("α = 0.6", 20, MUTED).next_to(ch.p(10, expected_tokens_per_pass(0.6, 10)), m.RIGHT, buff=0.15)
        lab08 = code("α = 0.8", 20, TOKEN).next_to(ch.p(10, expected_tokens_per_pass(0.8, 10)), m.RIGHT, buff=0.15)
        lab09 = code("α = 0.9", 20, INK).next_to(ch.p(10, expected_tokens_per_pass(0.9, 10)), m.RIGHT, buff=0.15)
        limit = m.DashedLine(ch.p(0, LIMIT), ch.p(10, LIMIT), dash_length=0.12).set_stroke(TOKEN, 2.5, opacity=0.6)
        limit_lab = code("1 ÷ (1 − α) = 5", 20, TOKEN).next_to(ch.p(0.3, LIMIT), m.UR, buff=0.1)
        d08 = m.Dot(ch.p(4, E), radius=0.09, color=TOKEN).set_z_index(3)
        d08_lab = code(f"{E:.2f}", 20, TOKEN, weight=m.BOLD).next_to(d08, m.DR, buff=0.08)
        d09 = m.Dot(ch.p(4, E09), radius=0.09, color=INK).set_z_index(3)
        d09_lab = code(f"{E09:.2f}", 20, INK, weight=m.BOLD).next_to(d09, m.UL, buff=0.08)

        self.say(
            "[Simulate a hundred thousand passes], [and the average is three point three six too.]",
            m.AnimationGroup(m.FadeOut(frame1), m.FadeIn(rows_title),
                             m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.RIGHT) for r in sim_rows], lag_ratio=0.2)),
            m.AnimationGroup(m.FadeIn(mean_lab), m.FadeIn(mean, scale=0.9)),
        )
        self.say(
            "[More guesses help less and less]: [the curve flattens toward one over one minus alpha], [five tokens at alpha point eight.]",
            m.AnimationGroup(m.FadeIn(ch.ax), m.FadeIn(x_ticks), m.FadeIn(y_ticks), m.FadeIn(x_name), m.FadeIn(y_name),
                             m.Create(curves[0.8], run_time=1.6), m.FadeIn(d08, scale=0.5), m.FadeIn(d08_lab)),
            m.AnimationGroup(m.Create(limit), m.FadeIn(limit_lab), m.FadeIn(lab08)),
            m.Circumscribe(limit_lab, color=TOKEN, buff=0.06),
        )
        self.say(
            "[A better drafter beats a longer draft]: [at alpha point nine, four guesses already give four point one.]",
            m.AnimationGroup(m.Create(curves[0.6]), m.FadeIn(lab06), m.Create(curves[0.9]), m.FadeIn(lab09)),
            m.AnimationGroup(m.FadeIn(d09, scale=0.5), m.FadeIn(d09_lab)),
        )
        self.wait(0.15)
        self.clear()

    # ------------------------------------------------------------------ 5 · the cost ratio

    def cost(self):
        self.next_section("5 · The cost ratio", notes=(
            f"Leviathan et al.: c = draft step time ÷ target step time; speedup S = E / (γc + 1) (owl_reference."
            f"speculative_speedup). Measured on this Mac (§1, batch-1 decode): Owl-mini (1B 4-bit) {T_MINI:.2f} ms/token, Owl-8B "
            f"BF16 {T_BF16:.2f} ms, 4-bit {T_4BIT:.2f} ms → c = {C_BF16:.3f} and {C_4BIT:.3f}. With the example α = 0.8, γ = 4: "
            f"E = {E:.2f} tokens cost {PLAIN_BF16:.0f} ms without a draft vs {SPEC_BF16:.1f} ms with one in BF16 ({S_BF16:.2f}×), "
            f"{PLAIN_4BIT:.0f} vs {SPEC_4BIT:.1f} ms in 4-bit ({S_4BIT:.2f}×). Assumes checking γ + 1 tokens costs one pass."))
        f_s = m.MathTex("S", "=", r"\frac{E}{1+\gamma c}", font_size=50, color=INK)
        f_s[0].set_color(TOKEN)
        f_s.move_to([-4.9, 1.85, 0])
        c_def = code("c = one draft step ÷ one big pass", 22, MEMORY).move_to([0, 2.1, 0]).align_to([-2.9, 0, 0], m.LEFT)
        same = text(f"the same {E:.2f} tokens either way · α = 0.8, 4 guesses, for example", 20, MUTED)
        same.next_to(c_def, m.DOWN, buff=0.22).align_to(c_def, m.LEFT)

        x0, k = -3.2, 0.078

        def block(y_title, title, t_pass, plain_ms, spec_ms, c, speed):
            head = m.VGroup(text(title, 24, INK, weight=m.BOLD),
                            code(f"c = {T_MINI:.1f} ÷ {t_pass:.1f} ms ≈ {c:.2f}", 20, MEMORY)).arrange(m.RIGHT, buff=0.4)
            head.move_to([0, y_title, 0]).align_to([-6.2, 0, 0], m.LEFT)
            y1, y2 = y_title - 0.58, y_title - 1.16
            l1 = code("no draft", 20, MUTED).move_to([0, y1, 0]).align_to([x0 - 0.25, 0, 0], m.RIGHT)
            l2 = code("with draft", 20, MUTED).move_to([0, y2, 0]).align_to([x0 - 0.25, 0, 0], m.RIGHT)
            plain = hbar(x0, y1, k * plain_ms, 0.4, MEMORY, CYAN_FILL)
            seps = m.VGroup(*[m.Line([x0 + k * t_pass * i, y1 - 0.2, 0], [x0 + k * t_pass * i, y1 + 0.2, 0]).set_stroke(MEMORY, 2)
                              for i in (1, 2, 3)])
            v1 = code(f"{plain_ms:.0f} ms", 20, MEMORY).next_to(plain, m.RIGHT, buff=0.15)
            drafts = m.VGroup(*[hbar(x0 + i * k * T_MINI, y2, k * T_MINI - 0.025, 0.4, INK, CARD_HI, stroke=1.5) for i in range(GAMMA)])
            big = hbar(x0 + GAMMA * k * T_MINI, y2, k * t_pass, 0.4, MEMORY, CYAN_FILL)
            v2 = code(f"{spec_ms:.0f} ms", 20, MEMORY).next_to(big, m.RIGHT, buff=0.15)
            sp = title_text(f"{speed:.1f}×", 44, TOKEN).move_to([0, y2, 0]).align_to([1.6, 0, 0], m.LEFT)
            return head, m.VGroup(l1, plain, seps, v1), m.VGroup(l2, drafts, big, v2), sp

        b_head, b_plain, b_spec, b_sp = block(0.75, "Owl-8B · BF16", T_BF16, PLAIN_BF16, SPEC_BF16, C_BF16, S_BF16)
        q_head, q_plain, q_spec, q_sp = block(-1.35, "Owl-8B · 4-bit", T_4BIT, PLAIN_4BIT, SPEC_4BIT, C_4BIT, S_4BIT)
        q_sp.align_to(b_sp, m.LEFT)
        mini_lab = code(f"4 guesses × {T_MINI:.1f} ms", 20, INK).next_to(b_spec[1], m.DOWN, buff=0.12).align_to(b_spec[1], m.LEFT)
        src = cite("step times measured on this Mac · mlx-lm").move_to([0, -3.35, 0]).align_to([-6.2, 0, 0], m.LEFT)

        self.say(
            "[But guessing isn't free.] [One round is four quick draft steps plus one big pass], [and that sets the speedup.]",
            m.AnimationGroup(self.new_header("05 · the cost ratio", "Guessing isn't free"), m.FadeIn(c_def)),
            m.FadeIn(same),
            m.FadeIn(f_s, shift=0.1 * m.UP),
        )
        self.say(
            "[Here, Owl-mini takes two milliseconds a token], [Owl-8B in sixteen-bit, thirty-two]: [the draft costs six percent of a pass.]",
            m.AnimationGroup(m.FadeIn(b_head[0]), m.FadeIn(b_spec[0]), m.LaggedStart(*[m.GrowFromEdge(d, m.LEFT) for d in b_spec[1]], lag_ratio=0.3),
                             m.FadeIn(mini_lab), m.FadeIn(src)),
            m.AnimationGroup(m.GrowFromEdge(b_spec[2], m.LEFT), m.FadeIn(b_spec[3])),
            m.FadeIn(b_head[1], shift=0.1 * m.LEFT),
        )
        self.say(
            "[Without a draft, that's three point three six passes]: [so drafting is about two point seven times faster.]",
            m.AnimationGroup(m.FadeIn(b_plain[0]), m.GrowFromEdge(b_plain[1], m.LEFT, run_time=1.4), m.FadeIn(b_plain[2]), m.FadeIn(b_plain[3])),
            m.FadeIn(b_sp, scale=0.85),
        )
        self.say(
            "[In four bits, Owl-8B's pass is under ten milliseconds]: [the same draft costs a fifth of it], [and the gain drops to one point eight.]",
            m.AnimationGroup(m.FadeIn(q_head[0]), m.FadeIn(q_plain), m.FadeIn(q_spec)),
            m.FadeIn(q_head[1], shift=0.1 * m.LEFT),
            m.FadeIn(q_sp, scale=0.85),
        )
        self.wait(0.15)
        self.clear()

    # ------------------------------------------------------------------ 6 · measured: copy the notes

    def measured_copy(self):
        self.next_section("6 · Measured: copy the notes", notes=(
            "research/measurements.md §9 (this Mac, 2026-10-24, mlx-lm 0.32.2): target mlx-community/Meta-Llama-3.1-8B-Instruct-"
            "bf16, draft mlx-community/Llama-3.2-1B-Instruct-4bit, greedy, 256 tokens, 3 trials, median; prompt notes_rewrite.txt "
            f"('Copy out the following study notes exactly…'). k = 6 draft tokens: {COPY['tok_s_median']:.1f} tok/s vs "
            f"{BASE_COPY:.1f} without a draft ({COPY['speedup']:.2f}×); acceptance 100% at every k; "
            f"{COPY['tokens_per_target_pass']:.2f} tokens per target pass (256 tokens, {RAW_COPY['target_passes']} passes); "
            "text identical to the plain greedy run. The strip is the real from_draft sequence (first 42 tokens). Ceilings: "
            f"546 GB/s ÷ {W_8B / GB:.2f} GB = {CEIL:.1f} tok/s at one token per pass; one round reads ≈ {W_8B / GB:.2f} + 6 × "
            f"{W_MINI / GB:.3f} = {ROUND_BYTES / GB:.1f} GB for {COPY['tokens_per_target_pass']:.2f} tokens → {SPEC_CEIL:.0f} tok/s "
            f"(KV reads ignored); measured = {OF_SPEC_CEIL * 100:.0f}% of it. The terminal shows the stock-CLI equivalent, shortened."))
        term = terminal([
            "$ mlx_lm.generate --temp 0 --max-tokens 256 \\",
            "  --model …Llama-3.1-8B-Instruct-bf16 \\",
            "  --draft-model …Llama-3.2-1B-Instruct-4bit \\",
            "  --num-draft-tokens 6 --prompt \"Copy out …\"",
        ], width=7.6, size=18, title="this Mac · M4 Max · 546 GB/s").move_to([0, 1.62, 0]).align_to([-6.2, 0, 0], m.LEFT)
        src = cite("measured on this Mac · mlx-lm").next_to(term, m.RIGHT, buff=0.4).align_to(term, m.DOWN)
        mascot = owl(0.75).next_to(src, m.UP, buff=0.35).align_to(src, m.LEFT)

        s = strip(SEQ_COPY, N_STRIP).move_to([0, -0.55, 0])
        s_title = text("copy the notes", 24, INK, weight=m.BOLD).next_to(s, m.UP, buff=0.22).align_to(s, m.LEFT)
        s_passes = code(f"256 tokens in {RAW_COPY['target_passes']} passes, not 256", 20, TOKEN).next_to(s, m.UP, buff=0.22).align_to(s, m.RIGHT)
        legend = m.VGroup(swatch("kept", "drafted by Owl-mini, kept"), swatch("own", "Owl-8B's own token"))
        legend.arrange(m.RIGHT, buff=0.6).next_to(s, m.DOWN, buff=0.22).align_to(s, m.LEFT)

        x0, k = -2.7, 0.045
        y1, y2 = -1.85, -2.65
        l1 = code("no draft", 20, MUTED).move_to([0, y1, 0]).align_to([x0 - 0.25, 0, 0], m.RIGHT)
        l2 = code("Owl-mini drafting", 20, MUTED).move_to([0, y2, 0]).align_to([x0 - 0.25, 0, 0], m.RIGHT)
        b1 = hbar(x0, y1, k * BASE_COPY, 0.44, TOKEN, GREEN_FILL)
        b2 = hbar(x0, y2, k * COPY["tok_s_median"], 0.58, TOKEN, GREEN_FILL)
        c1 = m.DashedLine([x0 + k * CEIL, y1 - 0.32, 0], [x0 + k * CEIL, y1 + 0.32, 0], dash_length=0.07).set_stroke(MEMORY, 3)
        c2 = m.DashedLine([x0 + k * SPEC_CEIL, y2 - 0.32, 0], [x0 + k * SPEC_CEIL, y2 + 0.32, 0], dash_length=0.07).set_stroke(MEMORY, 3)
        v1 = code(f"{BASE_COPY:.1f} tok/s", 22, TOKEN).next_to(c1, m.RIGHT, buff=0.2).set_y(y1)
        c1_lab = code(f"ceiling {CEIL:.0f}: 1 token per pass", 20, MEMORY).next_to(v1, m.RIGHT, buff=0.35)
        v2 = code(f"{COPY['tok_s_median']:.0f} tok/s", 24, INK, weight=m.BOLD).move_to(b2).align_to(b2, m.LEFT).shift(0.25 * m.RIGHT)
        sp = title_text(f"{COPY['speedup']:.2f}×", 40, TOKEN).move_to(b2).align_to(b2, m.RIGHT).shift(0.25 * m.LEFT)
        c2_lab = code(f"ceiling {SPEC_CEIL:.0f}", 20, MEMORY, weight=m.BOLD).next_to(c2, m.DOWN, buff=0.1).align_to(c2, m.RIGHT)
        c2_note = code(f"≈ {ROUND_BYTES / GB:.0f} GB per round · {COPY['tokens_per_target_pass']:.2f} tokens", 20, MEMORY)
        c2_note.next_to(c2_lab, m.LEFT, buff=0.35)

        self.say(
            "[Now on this Mac]: [Owl-8B in sixteen-bit, with Owl-mini drafting six tokens at a time], [in MLX.]",
            m.AnimationGroup(self.new_header("06 · measured", "Copy the notes"), m.FadeIn(term[:4])),
            m.LaggedStart(*[m.FadeIn(r, shift=0.05 * m.RIGHT) for r in term[4]], lag_ratio=0.3),
            m.AnimationGroup(m.FadeIn(src), m.FadeIn(mascot, shift=0.2 * m.UP)),
        )
        self.say(
            "[First prompt: copy a student's notes.] [Every guess was kept]: [six from Owl-mini, one from Owl-8B, over and over.]",
            m.FadeIn(s_title, shift=0.1 * m.UP),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c, scale=0.5) for c in s], lag_ratio=0.03, run_time=2.0), m.FadeIn(s_passes)),
            m.AnimationGroup(m.FadeIn(legend), m.Circumscribe(m.VGroup(*s[:7]), color=TOKEN, buff=0.06)),
        )
        self.say(
            "[Thirty-one tokens a second became a hundred and forty-eight]: [four point eight times faster.]",
            m.AnimationGroup(m.FadeIn(l1), m.GrowFromEdge(b1, m.LEFT), m.Create(c1), m.FadeIn(v1), m.FadeIn(c1_lab),
                             m.FadeIn(l2), m.GrowFromEdge(b2, m.LEFT, run_time=1.6), m.FadeIn(v2)),
            m.FadeIn(sp, scale=0.85),
        )
        self.say(
            "[Bandwidth over bytes still rules]: [about twenty gigabytes per round, for almost seven tokens], [a new ceiling of a hundred and eighty-seven.]",
            m.Circumscribe(c1_lab, color=MEMORY, buff=0.06),
            m.FadeIn(c2_note, shift=0.1 * m.UP),
            m.AnimationGroup(m.Create(c2), m.FadeIn(c2_lab, shift=0.1 * m.DOWN)),
        )
        self.wait(0.15)
        self.clear()

    # ------------------------------------------------------------------ 7 · measured: open chat

    def measured_chat(self):
        self.next_section("7 · Measured: open chat", notes=(
            "research/measurements.md §9, prompt chat.txt (the seasons question). BF16 target, k = 6: "
            f"{CHAT['tok_s_median']:.1f} tok/s ({CHAT['speedup']:.2f}×), {CHAT['tokens_per_target_pass']:.2f} tokens per pass "
            f"({RAW_CHAT['target_passes']} passes), acceptance {CHAT['acceptance_rate'] * 100:.1f}%. 4-bit target, best k = 2: "
            f"{CHAT4['tok_s_median']:.1f} vs {BASE_CHAT4:.1f} tok/s ({CHAT4['speedup']:.2f}×, acceptance {CHAT4['acceptance_rate'] * 100:.0f}%); "
            f"copy prompt 4-bit best k = 4: {COPY4['speedup']:.2f}×. Acceptance as the file counts it: accepted draft tokens ÷ "
            "drafted tokens, from mlx-lm's per-token from_draft flag; drafted guesses discarded after a miss count as drafted, "
            "so it falls as k grows (chat: 72% → 44% in 4-bit, 63% → 38% in BF16 from k = 2 to 6) — it is not Leviathan's α. "
            f"Why 4-bit gains less: c = {C_4BIT:.2f} vs {C_BF16:.2f}, and a 4-bit pass over 4 rows takes {ROW4_4BIT:.1f} ms vs "
            f"{ROW1_4BIT:.1f} ms for one (episode 4's batching step times; checking k + 1 positions is the same matrix shape — "
            "our reading). 'Not bit-identical in practice': greedy speculative output differs from the plain greedy run from token "
            "154 (BF16) / 221 (4-bit) on chat; verifying k + 1 tokens in one pass changes floating-point rounding and a near-tie "
            "flips (the file's explanation); copy prompt identical."))
        s = strip(SEQ_CHAT, N_STRIP).move_to([0, 1.55, 0])
        s_title = text("open chat · the seasons question", 24, INK, weight=m.BOLD).next_to(s, m.UP, buff=0.22).align_to(s, m.LEFT)
        s_passes = code(f"{RAW_CHAT['target_passes']} passes · {CHAT['tokens_per_target_pass']:.2f} tokens each", 20, TOKEN)
        s_passes.next_to(s, m.UP, buff=0.22).align_to(s, m.RIGHT)
        kept = code(f"kept: {min(CHAT_ACC) * 100:.0f}–{max(CHAT_ACC) * 100:.0f}% of drafted tokens", 26, TOKEN, weight=m.BOLD)
        kept.move_to([0, 0.55, 0]).align_to(s, m.LEFT)
        how = text("kept ÷ drafted, from mlx-lm's from_draft flag · guesses thrown away after a miss count as drafted", 20, MUTED)
        how.next_to(kept, m.DOWN, buff=0.16).align_to(s, m.LEFT)

        cx = {"label": -6.0, "chat": 0.3, "copy": 3.9}
        y_head, y_bf, y_q = -1.05, -1.8, -2.75
        heads = m.VGroup(code("open chat", 22, MUTED).move_to([cx["chat"], y_head, 0]), code("copy the notes", 22, MUTED).move_to([cx["copy"], y_head, 0]))
        rlabs = m.VGroup(*[m.VGroup(text(t, 24, INK, weight=m.BOLD), code(sub, 20, MUTED)).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.06)
                           .move_to([0, y, 0]).align_to([cx["label"], 0, 0], m.LEFT)
                           for t, sub, y in (("Owl-8B · BF16", f"{BASE_COPY:.0f} tok/s alone", y_bf), ("Owl-8B · 4-bit", f"{BASE_CHAT4:.0f} tok/s alone", y_q))])
        rule = m.Line([cx["label"], (y_head + y_bf) / 2 + 0.05, 0], [5.6, (y_head + y_bf) / 2 + 0.05, 0]).set_stroke(EDGE, 2)

        def value(v, kk, x, y, color):
            num = title_text(f"{v:.2f}×", 44, color)
            sub = code(f"k = {kk}", 20, MUTED).next_to(num, m.RIGHT, buff=0.2).align_to(num, m.DOWN)
            return m.VGroup(num, sub).move_to([x, y, 0])

        chat_bf = value(CHAT["speedup"], 6, cx["chat"], y_bf, TOKEN)
        chat_q = value(CHAT4["speedup"], 2, cx["chat"], y_q, TOKEN)
        copy_bf = value(COPY["speedup"], 6, cx["copy"], y_bf, TOKEN)
        copy_q = value(COPY4["speedup"], 4, cx["copy"], y_q, TOKEN)
        table = m.VGroup(heads, rlabs, rule)

        rows_title = text("one pass of Owl-8B · episode 4", 22, MUTED).move_to([0, 0.7, 0]).align_to([-6.2, 0, 0], m.LEFT)
        x0, k = -3.0, 0.12
        step_rows = []
        for y, label, ms, color, fill in ((0.15, "BF16 · 1 row", ROW1_BF16, MEMORY, CYAN_FILL), (-0.35, "BF16 · 8 rows", ROW8_BF16, MEMORY, CYAN_FILL),
                                          (-0.95, "4-bit · 1 row", ROW1_4BIT, MEMORY, CYAN_FILL), (-1.45, "4-bit · 4 rows", ROW4_4BIT, RED, RED_FILL)):
            lab = code(label, 20, INK).move_to([0, y, 0]).align_to([x0 - 0.25, 0, 0], m.RIGHT)
            bar = hbar(x0, y, k * ms, 0.38, color, fill)
            val = code(f"{ms:.1f} ms", 20, color).next_to(bar, m.RIGHT, buff=0.15)
            step_rows.append(m.VGroup(lab, bar, val))
        draft_note = text(f"draft step: {T_MINI:.1f} ms · {C_BF16 * 100:.0f}% of a BF16 pass, {C_4BIT * 100:.0f}% of a 4-bit one", 22, INK)
        draft_note.move_to([0, -2.0, 0]).align_to([-6.2, 0, 0], m.LEFT)
        steps = m.VGroup(rows_title, *step_rows, draft_note)

        foot = card(12.4, 1.0, EDGE, CARD, radius=0.14, stroke=1.5).move_to([0, -2.95, 0])
        foot_words = m.VGroup(
            text("chat: the text left plain greedy's path at token 154, when a near-tie flipped", 22, INK),
            cite("checking 7 tokens in one pass rounds differently · copy prompt: identical · measured on this Mac"),
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.1).move_to(foot).align_to(foot, m.LEFT).shift(0.35 * m.RIGHT)

        self.say(
            "[Open chat is harder to guess.] [The strip is ragged], [and each pass made about three tokens.]",
            m.AnimationGroup(self.new_header("07 · measured", "Open chat"), m.FadeIn(s_title, shift=0.1 * m.UP)),
            m.LaggedStart(*[m.FadeIn(c, scale=0.5) for c in s], lag_ratio=0.03, run_time=1.8),
            m.FadeIn(s_passes, shift=0.1 * m.UP),
        )
        self.say(
            "[Only thirty-eight to seventy-two percent of drafted tokens were kept], [counting the guesses thrown away after each miss.]",
            m.FadeIn(kept, shift=0.1 * m.UP),
            m.FadeIn(how, shift=0.1 * m.UP),
        )
        self.say(
            "[Chat still ran two point two eight times faster in sixteen-bit], [but only one point four two times in four-bit.]",
            m.AnimationGroup(m.FadeIn(table), m.FadeIn(copy_bf), m.FadeIn(copy_q), m.FadeIn(chat_bf, scale=0.85)),
            m.FadeIn(chat_q, scale=0.85),
        )
        self.say(
            "[Four-bit gains less]: [its pass is short], [and in episode four, four rows took half again as long as one.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(kept, how, table, chat_bf, chat_q, copy_bf, copy_q)), m.FadeIn(rows_title),
                             m.FadeIn(step_rows[0]), m.FadeIn(step_rows[1])),
            m.AnimationGroup(m.FadeIn(step_rows[2][0]), m.GrowFromEdge(step_rows[2][1], m.LEFT), m.FadeIn(step_rows[2][2]), m.FadeIn(draft_note)),
            m.AnimationGroup(m.FadeIn(step_rows[3][0]), m.GrowFromEdge(step_rows[3][1], m.LEFT), m.FadeIn(step_rows[3][2])),
        )
        self.say(
            "[One honest footnote]: [in chat, the text left plain decoding's path after a hundred and fifty-four tokens], [when rounding flipped a near-tie.]",
            m.FadeIn(foot),
            m.FadeIn(foot_words[0], shift=0.1 * m.UP),
            m.FadeIn(foot_words[1]),
        )
        self.wait(0.15)
        self.clear()

    # ------------------------------------------------------------------ 8 · busy servers

    def busy(self):
        self.next_section("8 · Busy servers", notes=(
            "Notes §6 'Batch size kills the free lunch': EAGLE-3 paper (Li et al., arXiv 2503.01840, 2025), Table 3: SGLang, "
            "H100, LLaMA-Instruct 3.1 8B, MT-Bench, throughput vs SGLang without speculation at each batch size (re-checked "
            "against the paper 2026-10-24; its batch-56 column, 0.99× / 1.34×, is omitted). Reason (notes §6): speculation "
            "spends spare compute to save memory passes; as the batch grows toward the ridge (episodes 2 and 4) none is left. "
            "EAGLE and EAGLE-3 draft from the model's own features: episode 14."))
        ch = Chart([0, 64, 8], [0.5, 2.0, 0.5], 7.0, 4.2, (-1.7, -0.5, 0))
        x_ticks = m.VGroup(*[code(str(b), 20, REQUEST).move_to(ch.p(b, 0.5) + 0.3 * m.DOWN) for b in (2, 8, 16, 24, 32, 48, 64)])
        y_ticks = m.VGroup(*[ch.ylabel(v, f"{v:.1f}×") for v in (1.0, 1.5, 2.0)])
        x_name = text("batch size · users at once", 22, REQUEST).next_to(x_ticks, m.DOWN, buff=0.12).set_x(ch.p(32, 0.5)[0])
        y_name = text("speedup vs no speculation", 22, MUTED).move_to(ch.p(0, 2.0) + [0, 0.45, 0]).align_to(ch.p(0, 2.0) + [-0.5, 0, 0], m.LEFT)
        one = m.DashedLine(ch.p(0, 1.0), ch.p(64, 1.0), dash_length=0.12).set_stroke(MUTED, 2.5)
        e1 = polyline([ch.p(b, v) for b, v in zip(EAGLE_B, EAGLE1)], MUTED, 4)
        e3 = polyline([ch.p(b, v) for b, v in zip(EAGLE_B, EAGLE3)], TOKEN, 4)
        d1 = m.VGroup(*[m.Dot(ch.p(b, v), radius=0.07, color=RED if v < 1 else MUTED).set_z_index(3) for b, v in zip(EAGLE_B, EAGLE1)])
        d3 = m.VGroup(*[m.Dot(ch.p(b, v), radius=0.07, color=TOKEN).set_z_index(3) for b, v in zip(EAGLE_B, EAGLE3)])
        lab1 = code("EAGLE", 22, MUTED, weight=m.BOLD).next_to(ch.p(64, EAGLE1[-1]), m.RIGHT, buff=0.2)
        lab3 = code("EAGLE-3", 22, TOKEN, weight=m.BOLD).next_to(ch.p(64, EAGLE3[-1]), m.RIGHT, buff=0.2)
        v_first = code(f"{EAGLE1[0]:.2f}×", 20, MUTED).next_to(d1[0], m.UP, buff=0.16).shift(0.45 * m.RIGHT)
        v_low = code(f"{min(EAGLE1):.2f}×", 20, RED).next_to(d1[EAGLE1.index(min(EAGLE1))], m.DOWN, buff=0.14)
        v3 = code(f"{EAGLE3[-1]:.2f}×", 20, TOKEN).next_to(d3[-1], m.UP, buff=0.14)
        src = m.VGroup(cite("Li et al., 2025"), cite("EAGLE-3 paper, Table 3"), cite("SGLang on an H100"), cite("Llama-3.1-8B · MT-Bench"))
        src.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.08).move_to([0, -2.2, 0]).align_to([3.5, 0, 0], m.LEFT)
        why = m.VGroup(text("speculation spends", 22, MUTED), text("spare compute", 26, COMPUTE, weight=m.BOLD),
                       text("to save trips", 22, MUTED), text("through memory", 26, MEMORY, weight=m.BOLD))
        why.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.1).move_to([0, 1.1, 0]).align_to([3.5, 0, 0], m.LEFT)

        self.say(
            "[On a busy server, the lunch runs out]: [speculation spends spare compute to save trips through memory], [and a big batch has none.]",
            m.AnimationGroup(self.new_header("08 · busy servers", "Big batches, small gains"), m.FadeIn(ch.ax), m.FadeIn(x_ticks),
                             m.FadeIn(y_ticks), m.FadeIn(x_name), m.FadeIn(y_name), m.Create(one)),
            m.FadeIn(why, shift=0.1 * m.LEFT),
            m.AnimationGroup(m.Create(e1, run_time=1.6), m.FadeIn(d1), m.FadeIn(lab1), m.FadeIn(v_first), m.FadeIn(src)),
        )
        self.say(
            "[With this very model on an H100], [the EAGLE drafter fell below one from batch twenty-four], [while EAGLE-3 held about one point four.]",
            m.Circumscribe(src, color=MUTED, buff=0.06),
            m.AnimationGroup(m.FadeIn(v_low, shift=0.1 * m.DOWN), m.Indicate(d1[4:], color=RED, scale_factor=1.4)),
            m.AnimationGroup(m.Create(e3, run_time=1.4), m.FadeIn(d3), m.FadeIn(lab3), m.FadeIn(v3)),
        )
        self.wait(0.15)
        self.clear()

    # ------------------------------------------------------------------ 9 · recap

    def recap(self):
        self.next_section("9 · Recap", notes=(
            f"Guess (a small draft, Owl-mini), check (one pass of Owl-8B over every guess), keep (exactly the target's tokens: "
            f"rejection sampling). E = (1 − α^(γ+1))/(1 − α) = {E:.2f} at α = 0.8, γ = 4. Measured here: {BASE_COPY:.1f} → "
            f"{COPY['tok_s_median']:.0f} tok/s copying notes ({COPY['speedup']:.2f}×), {CHAT['speedup']:.2f}× in open chat (BF16). "
            "Next: episode 14, drafting without a draft model."))
        halo = glow(TOKEN, radius=3.4).move_to([0, 0.7, 0])
        words = m.VGroup(title_text("Guess", 76, MUTED), title_text("Check", 76, WEIGHTS), title_text("Keep", 76, TOKEN))
        words.arrange(m.RIGHT, buff=1.3).move_to([0, 0.8, 0])
        on_baseline(words)
        a1 = arrow(words[0].get_right(), words[1].get_left(), MUTED, buff=0.25)
        a2 = arrow(words[1].get_right(), words[2].get_left(), MUTED, buff=0.25)
        f_e = m.MathTex("E", "=", r"\frac{1-\alpha^{\gamma+1}}{1-\alpha}", font_size=44, color=INK)
        f_e[0].set_color(TOKEN)
        f_e.move_to([0, -1.0, 0])
        proof = code(f"this Mac: {BASE_COPY:.0f} → {COPY['tok_s_median']:.0f} tok/s copying notes · {CHAT['speedup']:.2f}× in chat", 22, TOKEN)
        proof.next_to(f_e, m.DOWN, buff=0.4)
        mascot = owl(0.75).move_to([0, -3.0, 0])

        self.say(
            "[To sum up]: [guess], [check], [keep.]",
            m.AnimationGroup(self.new_header("09 · recap", "Guess, check, keep"), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.2 * m.UP),
            m.AnimationGroup(m.GrowArrow(a1), m.FadeIn(words[1], shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowArrow(a2), m.FadeIn(words[2], shift=0.2 * m.UP)),
        )
        self.say(
            "[A small model guesses], [the big one checks every guess in one pass], [and keeps exactly what it would have written itself.]",
            m.Indicate(words[0], color=INK, scale_factor=1.08),
            m.Indicate(words[1], color=WEIGHTS, scale_factor=1.08),
            m.AnimationGroup(m.Indicate(words[2], color=TOKEN, scale_factor=1.08), m.FadeIn(f_e, shift=0.1 * m.UP)),
        )
        self.say(
            "[Easy guesses made Owl almost five times faster here]; [hard guesses and big batches, much less.]",
            m.FadeIn(proof, shift=0.1 * m.UP),
            m.FadeIn(mascot, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()
