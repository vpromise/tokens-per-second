"""Tokens per Second · Episode 2 · The Two Speed Limits (Module 1 · One Token, Two Speed Limits)

Every step of a language model is limited by math (FLOPs) or by memory traffic (bytes); for chat, one token at
a time, it is memory. Running example: Owl-8B (Llama-3.1-8B: 8.03 B parameters, 16.06 GB of BF16 weights)
answering a student's "Why does Earth have seasons?", on an H100 and on this Mac Studio (M4 Max, 546 GB/s).

Sections: the question (31.0 tok/s measured here: why that number?) · math (2 FLOPs per weight, 16 GFLOP per
token, 16 µs on an H100, "over 60,000 tok/s" if math were the limit) · memory traffic (the bytes truck: all
16 GB per token over 3.35 TB/s = 4.8 ms, 295× the math; math units idle 99.7%) · intensity (FLOPs ÷ bytes:
Owl ≈ 1, the H100's balance ≈ 295) · the roofline (slope, roof, ridge; Owl's dot rises to 3.35 TFLOP/s = 0.34%
of the roof; this Mac's lower roofline, ridge ≈ 28 from a measured matmul peak; long prompts and batches move
right → episodes 3 and 4) · the ceiling (tok/s ≈ bandwidth ÷ bytes: 209 on an H100, 34 here; only bandwidth or
fewer bytes move it) · measured (31.0 tok/s = 91% of 34; 15.04 GB really read per token; 31.0 × 15.04 GB =
466 GB/s = 85% MBU = 97% of a plain streaming read) · recap (math, memory, measure).

Sources (facts dated October 2026):
- owl_reference.py: MODELS (Llama-3.1-8B config.json), HARDWARE (H100 SXM 3.35 TB/s and 989 TFLOP/s BF16 dense,
  NVIDIA H100 page; M4 Max 546 GB/s, Apple Mac Studio (2025) tech specs), flops_per_token, weight_bytes, ridge,
  decode_ceiling.
- research/notes.md §0 (#1 2·P FLOPs, kipply 2022; #2 bytes per step ≈ all weights, JAX scaling book; #3–#5),
  §2 (intensity, ridge, roofline, MBU = achieved ÷ peak bandwidth (Databricks); worked example 1: 16 µs of math
  vs 4.79 ms of reading, idle ≈ 99.7%; worked example 2: "multiply speed by size and you get bandwidth"), §14.
- research/measurements.md / .json, measured on this Mac on 2026-10-24 with mlx-lm 0.32.2 (loaded with
  measured()): decode Llama-3.1-8B bf16 31.0 tok/s (median of 5, 256 tokens, greedy); bytes read per decode
  step 15.04 GB (the untied 1.05 GB input-embedding table is a row lookup, so it is not streamed; plus the KV
  cache at the run's average context); MBU 85.4% = 466 GB/s; a plain streaming read reaches 482.8 GB/s; a bf16
  8192³ matmul reaches 15.2 TFLOPS (Apple publishes no GPU TFLOPS), so this Mac's ridge ≈ 28 FLOPs/byte;
  stock mlx_lm.generate cross-check 31.0 tok/s. The answer text in the hook is that run's own greedy output.

Illustrative: the 6 × 8 weight grid (standing for 8 B weights), the truck and its road, the token boundaries
in the answer, "300?" / "3,000?" (rhetorical), the bars' lengths (to scale, except the 16 µs sliver, widened
to stay visible). The terminal shows the stock command in shortened form. Asserts below tie every quoted
number to the data.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from owl_reference import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- the numbers

P = OWL["params"]                       # 8.03e9
FLOPS = flops_per_token()               # 2·P = 16.06 GFLOP per token
WBYTES = weight_bytes()                 # 16.06 GB of BF16 weights
INTENSITY = FLOPS / WBYTES              # ≈ 1 FLOP per byte at batch 1
H_BW, H_PEAK = GPU["bandwidth"], GPU["peak"]
RIDGE_H = ridge(GPU)                    # ≈ 295 FLOPs/byte
T_MATH = FLOPS / H_PEAK                 # ≈ 16 µs of math per token
T_MEM = WBYTES / H_BW                   # ≈ 4.8 ms to read the weights
IDLE = 1 - T_MATH / T_MEM               # ≈ 99.7%
MATH_ONLY = H_PEAK / FLOPS              # tok/s if math were the only limit
ON_SLOPE = H_BW * INTENSITY             # FLOP/s reached at intensity 1 (= 3.35 TFLOP/s)
ROOF_FRAC = ON_SLOPE / H_PEAK           # ≈ 0.34% of the roof
CEIL_H = decode_ceiling(hw=GPU)         # ≈ 209 tok/s
CEIL_M = decode_ceiling(hw=MAC)         # ≈ 34 tok/s
M_BW = MAC["bandwidth"]

MAC_PEAK = measured("roofs", "matmul_bf16_TFLOPS", "median") * 1e12   # measured: Apple publishes none
RIDGE_M = MAC_PEAK / M_BW               # ≈ 28 FLOPs/byte
DEC = measured("decode", "llama-3.1-8b", "bf16")
TOK = DEC["tok_s_median"]               # 31.0 tok/s measured
READ = DEC["bytes_read_per_step"]       # 15.04 GB really streamed per token
EMBED = measured("models", "llama-3.1-8b", "bf16", "embed_bytes")    # 1.05 GB: a lookup, not streamed
CLI = measured("cli_crosscheck_mlx_lm_generate", "Meta-Llama-3.1-8B-Instruct-bf16", "tok_s_median")
ACHIEVED = TOK * READ                   # "multiply speed by size": 466 GB/s
MBU = ACHIEVED / M_BW                   # 85%
READ_ROOF = measured("roofs", "read_GBps", "median") * 1e9           # 483 GB/s, a plain streaming read
OF_READ = ACHIEVED / READ_ROOF          # 97%
OF_CEIL = TOK / CEIL_M                  # 91%

RAW = Path(__file__).resolve().parent.parent / "research" / "measurements" / "raw" / "decode__llama-3.1-8b__bf16.json"
SAMPLE = json.loads(RAW.read_text())["sample_text_first120tok"]   # the measured run's own answer
ANSWER = ["Hello", " there,", " friend.", " I'm", " Owl,", " your", " friendly", " study", " buddy."]
assert SAMPLE.startswith("".join(ANSWER))  # the tokens shown are the start of the real answer

assert round(P / 1e9) == 8 and round(FLOPS / 1e9) == 16 and round(WBYTES / 1e9) == 16   # "eight billion … sixteen"
assert f"{P / 1e9:.2f}" == "8.03" and f"{FLOPS / 1e9:.2f}" == "16.06" == f"{WBYTES / 1e9:.2f}"
assert round(INTENSITY) == 1                                       # "about one FLOP per byte"
assert round(H_PEAK / 1e12) == 989 and round(H_BW / 1e12, 2) == 3.35   # "nine hundred and eighty-nine trillion", "three point three five"
assert round(T_MATH * 1e6) == 16                                   # "sixteen microseconds"
assert MATH_ONLY > 60_000 and round(MATH_ONLY, -2) == 61_600       # "over sixty thousand tokens a second"
assert round(T_MEM * 1e3, 1) == 4.8                                # "four point eight milliseconds"
assert round(T_MEM / T_MATH) == 295 and 250 < T_MEM / T_MATH < 350  # "about three hundred times longer"
assert round(IDLE * 100, 1) == 99.7                                # "ninety-nine point seven percent"
assert round(RIDGE_H) == 295                                       # "two hundred and ninety-five"
assert round(ON_SLOPE / 1e12, 2) == 3.35                           # "three point three five trillion FLOPs per second"
assert round(ROOF_FRAC * 100, 2) == 0.34 and abs(ROOF_FRAC * 100 - 1 / 3) < 0.01   # "a third of a percent"
assert round(RIDGE_M) == 28                                        # "a ridge near twenty-eight"
assert round(CEIL_H) == 209 and round(CEIL_M) == 34                # "two hundred and nine", "thirty-four"
assert round(M_BW / 1e9) == 546                                    # "five hundred and forty-six"
assert f"{TOK:.1f}" == "31.0" and f"{CLI:.1f}" == "31.0"           # "thirty-one tokens a second"
assert round(READ / 1e9, 2) == 15.04 and round(READ / 1e9) == 15   # "fifteen gigabytes, not sixteen"
assert round(EMBED / 1e9, 2) == 1.05
assert round(ACHIEVED / 1e9) == 466                                # "four hundred and sixty-six"
assert round(MBU * 100) == 85                                      # "eighty-five percent"
assert round(READ_ROOF / 1e9) == 483                               # "four hundred and eighty-three"
assert round(OF_READ * 100) == 97                                  # "ninety-seven percent"
assert round(OF_CEIL * 100) == 91 and round((1 - OF_CEIL) * 100) == 9   # "ninety-one percent", "within nine percent"

# ---------------------------------------------------------------------------- look

PREFIX = "ep02"
GPU_AMBER = recolored("gpu", COMPUTE, AMBER_FILL, PREFIX)     # math units
RAM_CYAN = recolored("ram", MEMORY, CYAN_FILL, PREFIX)        # memory
GPU_INK = recolored("gpu", INK, CARD_HI, PREFIX)              # a GPU as hardware (neutral)

FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIKLMNOPRSTUVWXYZ0123456789")
SYMBOLS = {"≈", "÷", "×", "=", "+"}


def cite(s, size=20):
    return text(s, size, MUTED)


def baseline(words):
    """The baseline of a Text: the median bottom of its flat-bottomed glyphs."""
    chars = [c for c in words.text if not c.isspace()]
    if len(chars) != len(words):  # a ligature merged two glyphs
        return float(np.median([g.get_bottom()[1] for g in words]))
    bottoms = sorted(g.get_bottom()[1] for g, c in zip(words, chars) if c in FLAT)
    return bottoms[len(bottoms) // 2] if bottoms else words.get_bottom()[1]


def sit(words, y):
    """Put a Text's baseline at y."""
    return words.shift((y - baseline(words)) * m.UP)


def formula(parts, size=48, buff=0.24):
    """A display formula in DIN: words on one baseline, symbols (≈ ÷ × =) centred on the x-height."""
    mobs = [title_text(s, size, c) for s, c in parts]
    row = m.VGroup(*mobs).arrange(m.RIGHT, buff=buff)
    wordy = [mb for mb, (s, _) in zip(mobs, parts) if s not in SYMBOLS]
    y0 = baseline(wordy[0])
    for mb in wordy:
        sit(mb, y0)
    x_height = title_text("x", size).height
    for mb, (s, _) in zip(mobs, parts):
        if mb not in wordy:
            mb.set_y(y0 + x_height / 2)
    return row


def hbar(x0, y, width, height, color, fill, stroke=2):
    """A horizontal bar whose left edge is at x0."""
    bar = m.Rectangle(width=max(width, 0.04), height=height).set_stroke(color, stroke).set_fill(fill, 1)
    return bar.move_to([x0 + max(width, 0.04) / 2, y, 0])


def truck(load):
    """The bytes truck: a cab (memory traffic, CYAN) pulling a load of weights (BLUE)."""
    cargo = card(2.0, 1.0, WEIGHTS, BLUE_FILL, radius=0.1, stroke=3)
    cubes = m.VGroup(*[m.Square(0.24).set_stroke(WEIGHTS, 2).set_fill(WEIGHTS, 0.55) for _ in range(10)])
    cubes.arrange_in_grid(rows=2, buff=0.11).move_to(cargo)
    cab = m.RoundedRectangle(corner_radius=0.12, width=0.78, height=0.74, stroke_color=MEMORY, stroke_width=3,
                             fill_color=CYAN_FILL, fill_opacity=1)
    cab.next_to(cargo, m.RIGHT, buff=0.06).align_to(cargo, m.DOWN)
    window = m.RoundedRectangle(corner_radius=0.05, width=0.32, height=0.25).set_stroke(MEMORY, 2).set_fill(BG, 1)
    window.move_to(cab.get_corner(m.UR) + np.array([-0.24, -0.21, 0]))
    y_axle = cargo.get_bottom()[1] - 0.07
    chassis = m.Line([cargo.get_left()[0] + 0.05, y_axle, 0], [cab.get_right()[0] - 0.03, y_axle, 0]).set_stroke(MUTED, 5)
    wheels = m.VGroup(*[
        m.Circle(radius=0.16).set_stroke(INK, 3).set_fill(BG, 1).move_to([x, y_axle - 0.04, 0])
        for x in (cargo.get_left()[0] + 0.42, cargo.get_right()[0] - 0.42, cab.get_x())
    ])
    label = code(load, 20, WEIGHTS).next_to(cargo, m.UP, buff=0.14)
    return m.VGroup(label, cargo, cubes, cab, window, chassis, wheels)


def mac_studio(width=1.4, color=MUTED):
    """A Mac Studio, front view: a squat rounded box with two ports, a card slot and a power light."""
    h = width * 0.42
    body = card(width, h, color, CARD_HI, radius=0.14 * width, stroke=3)
    foot = m.RoundedRectangle(corner_radius=0.03, width=width * 0.8, height=0.06, stroke_width=0,
                              fill_color=EDGE, fill_opacity=1).next_to(body, m.DOWN, buff=0.0)
    ports = m.VGroup(*[
        m.RoundedRectangle(corner_radius=0.015, width=0.045 * width, height=0.2 * h, stroke_width=0,
                           fill_color=BG, fill_opacity=1) for _ in range(2)
    ]).arrange(m.RIGHT, buff=0.05 * width)
    slot = m.RoundedRectangle(corner_radius=0.01, width=0.16 * width, height=0.05 * h, stroke_width=0,
                              fill_color=BG, fill_opacity=1)
    front = m.VGroup(ports, slot).arrange(m.RIGHT, buff=0.08 * width)
    front.move_to(body.get_center() + np.array([-0.18 * width, -0.14 * h, 0]))
    light = m.Dot(radius=0.025 * width, color=GREEN).move_to(body.get_center() + np.array([0.36 * width, -0.14 * h, 0]))
    return m.VGroup(foot, body, front, light)


def answer_card(tokens, width):
    """Owl's reply: an assistant card (GREEN stripe) holding a row of token chips."""
    label = code("assistant", 22, TOKEN, weight=m.BOLD)
    inner = m.VGroup(label, tokens).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.16)
    box = card(width, inner.height + 0.42, EDGE, CARD, radius=0.12, stroke=1.5)
    stripe = m.RoundedRectangle(corner_radius=0.05, width=0.09, height=box.height - 0.16, stroke_width=0,
                                fill_color=TOKEN, fill_opacity=1)
    stripe.move_to(box).align_to(box, m.LEFT).shift(0.1 * m.RIGHT)
    inner.move_to(box).align_to(box, m.LEFT).shift(0.32 * m.RIGHT)
    return m.VGroup(box, stripe, label)


def legend(color, title, desc, width=6):
    """A legend row for the roofline: a short line in the colour, a bold mono title, a muted description."""
    sample = m.Line([0, 0, 0], [0.45, 0, 0]).set_stroke(color, width)
    words = m.VGroup(code(title, 20, color, weight=m.BOLD), text(desc, 20, MUTED)).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.08)
    row = m.VGroup(sample, words).arrange(m.RIGHT, buff=0.22)
    sample.set_y(words[0].get_y())
    return row


def limit_tile(name, title, desc, color, width=6.0, height=1.9):
    box = card(width, height, color, CARD, stroke=2.5)
    words = m.VGroup(text(title, 32, INK, weight=m.BOLD), text(desc, 22, MUTED)).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.16)
    body = m.VGroup(icon(name, 0.8), words).arrange(m.RIGHT, buff=0.45).move_to(box)
    return m.VGroup(box, body)


def limit_tiles():
    """The two speed limits as tiles: math (AMBER) and memory traffic (CYAN)."""
    a = limit_tile(GPU_AMBER, "Math", "compute · FLOPs per second", COMPUTE)
    b = limit_tile(RAM_CYAN, "Memory traffic", "bytes moved per second", MEMORY)
    return a, b


# ---------------------------------------------------------------------------- the episode


class Ep02TheTwoSpeedLimits(Episode):
    number = 2

    def lesson(self):
        self.question()
        self.math()
        self.memory()
        self.intensity()
        self.roofline()
        self.ceiling()
        self.check()
        self.recap()

    # ------------------------------------------------------------------ 1 · the question

    def question(self):
        self.next_section("1 · The question", notes=(
            f"Owl-8B (Llama-3.1-8B, BF16) answering the course's seasons prompt on this Mac: {TOK:.1f} tok/s decode, "
            "measured on this Mac with mlx-lm 0.32.2 (research/measurements.md §1, median of 5). The answer text is that "
            "run's own greedy output (token boundaries illustrative). '300?' and '3,000?' are rhetorical. The two speed "
            "limits: compute (FLOPs, AMBER) and memory traffic (bytes, CYAN) — the course's colours."))
        mascot = owl(1.15).move_to([-5.3, 0.6, 0])
        ask = message("user", "Why does Earth have seasons?", width=7.6, size=24).move_to([1.0, 1.75, 0])
        toks = token_row(ANSWER, size=20, max_width=6.8)
        frame = answer_card(toks, 7.6)
        m.VGroup(frame, toks).next_to(ask, m.DOWN, buff=0.28)
        rate = stat(f"{TOK:.1f}", "tok/s", "", TOKEN, 84)
        rate.move_to([0, -1.95, 0]).align_to(ask, m.LEFT)
        src = cite("Owl-8B · BF16 · measured on this Mac · mlx-lm").next_to(rate, m.DOWN, buff=0.2).align_to(rate, m.LEFT)
        ghosts = m.VGroup(title_text("300?", 64, FAINT), title_text("3,000?", 64, FAINT)).arrange(m.RIGHT, buff=0.6)
        ghosts.next_to(rate, m.RIGHT, buff=0.8)
        for g in ghosts:
            sit(g, baseline(rate[0]))
        hook = m.VGroup(mascot, ask, frame, toks, rate, src, ghosts)

        t_math, t_mem = limit_tiles()
        m.VGroup(t_math, t_mem).arrange(m.RIGHT, buff=0.4).move_to([0, 0.75, 0])
        teaser = formula([("tok/s", INK), ("≈", MUTED), ("?", COMPUTE), ("÷", MUTED), ("?", MEMORY)], 56).move_to([0.4, -1.75, 0])
        small = owl(0.7).next_to(teaser, m.LEFT, buff=0.7)

        self.say(
            "[Ask Owl why Earth has seasons], [and on this Mac, its answer streams in at about thirty-one tokens a second.]",
            m.AnimationGroup(self.new_header("01 · the question", "Why 31 tok/s?"), m.FadeIn(mascot, shift=0.2 * m.UP),
                             m.FadeIn(ask, shift=0.15 * m.UP)),
            m.AnimationGroup(m.FadeIn(frame), m.LaggedStart(*[m.FadeIn(t, shift=0.1 * m.UP) for t in toks], lag_ratio=0.3),
                             m.FadeIn(rate, shift=0.15 * m.UP), m.FadeIn(src)),
        )
        self.say(
            "[Why thirty-one?] [Why not three hundred], [or three thousand?]",
            m.Circumscribe(rate[0], color=TOKEN, buff=0.1),
            m.FadeIn(ghosts[0], shift=0.15 * m.UP),
            m.FadeIn(ghosts[1], shift=0.15 * m.UP),
        )
        self.say(
            "[Every step of a language model obeys two speed limits]: [math], [and memory traffic.]",
            m.FadeOut(hook, shift=0.2 * m.UP),
            m.FadeIn(t_math, shift=0.2 * m.UP),
            m.FadeIn(t_mem, shift=0.2 * m.UP),
        )
        self.say(
            "[Today: which one sets Owl's speed], [and how to predict it with a single division.]",
            m.FadeIn(small, shift=0.2 * m.UP),
            m.FadeIn(teaser, shift=0.15 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 2 · math

    def math(self):
        self.next_section("2 · Math", notes=(
            f"FLOPs per token ≈ 2·P (kipply 2022, notes §0 #1, §2): each weight does one multiply and one add per token. "
            f"Owl-8B: P = {P / 1e9:.2f} B → {FLOPS / 1e9:.2f} GFLOP per token (flops_per_token()). H100 SXM: "
            f"{H_PEAK / 1e12:.0f} TFLOP/s BF16 dense (NVIDIA spec, Oct 2026) → {T_MATH * 1e6:.1f} µs of math per token, "
            f"so math alone would allow {MATH_ONLY:,.0f} tok/s. The 6 × 8 grid is illustrative."))
        W = cell_grid(6, 8, size=0.36, gap=0.08, color=WEIGHTS, fill=BLUE_FILL).move_to([-3.7, -0.55, 0])
        x = m.VGroup(*[m.Square(0.36).set_stroke(TOKEN, 1.5).set_fill(GREEN_FILL, 1) for _ in range(8)])
        for j, sq in enumerate(x):
            sq.move_to([W[0][j].get_x(), W.get_top()[1] + 0.48, 0])
        y = m.VGroup(*[m.Square(0.36).set_stroke(EDGE, 1.5).set_fill(CARD, 1) for _ in range(6)])
        for i, sq in enumerate(y):
            sq.move_to([W.get_right()[0] + 0.62, W[i][0].get_y(), 0])
        y_on = m.VGroup(*[m.Square(0.36).set_stroke(COMPUTE, 1.5).set_fill(AMBER_FILL, 1).move_to(sq) for sq in y])
        x_lab = code("x · this token", 20, TOKEN).next_to(x, m.UP, buff=0.16).align_to(x, m.LEFT)
        w_lab = code("W · the weights", 20, WEIGHTS).next_to(W, m.DOWN, buff=0.22).align_to(W, m.LEFT)
        y_lab = code("y", 20, COMPUTE).next_to(y, m.UP, buff=0.16)
        scan = m.SurroundingRectangle(W[0], buff=0.06, corner_radius=0.06).set_stroke(COMPUTE, 3)
        left = m.VGroup(W, x, y, y_on, x_lab, w_lab, y_lab, scan)

        cx = 3.15
        eq = code("y[i] += W[i,j] × x[j]", 30, INK,
                  t2c={r"W\[i,j\]": WEIGHTS, r"x\[j\]": TOKEN, r"y\[i\]": COMPUTE, "×": COMPUTE, r"\+=": COMPUTE}).move_to([cx, 1.6, 0])
        two = pill("1 multiply + 1 add = 2 FLOPs", COMPUTE, AMBER_FILL, 24).move_to([cx, 0.75, 0])
        block = code_block('def flops_per_token(model=OWL):\n    return 2 * model["params"]  # 8.03e9',
                           20, title="owl_reference.py", accent=COMPUTE).move_to([cx, -0.45, 0])
        per_tok = stat(f"{FLOPS / 1e9:.2f}", "GFLOP", "per token: 2 × 8.03 B weights", COMPUTE, 64)
        per_tok.move_to([cx, -2.15, 0])

        hw = card(5.2, 1.75, COMPUTE, CARD)
        hw_words = m.VGroup(title_text("H100", 40), code(f"{H_PEAK / 1e12:.0f} TFLOP/s · BF16", 20, COMPUTE))
        hw_words.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.14)
        hw_body = m.VGroup(icon(GPU_AMBER, 0.8), hw_words).arrange(m.RIGHT, buff=0.35).move_to(hw)
        hw_card = m.VGroup(hw, hw_body).move_to([-3.8, 1.25, 0])
        hw_src = cite("NVIDIA, H100 SXM spec").next_to(hw_card, m.DOWN, buff=0.15).align_to(hw_card, m.LEFT)
        div = code(f"{FLOPS / 1e9:.2f} GFLOP ÷ {H_PEAK / 1e12:.0f} TFLOP/s", 26, INK,
                   t2c={r"[\d.]+ GFLOP": COMPUTE, r"\d+ TFLOP/s": COMPUTE}).move_to([-3.8, -0.6, 0])
        t_math = stat(f"{T_MATH * 1e6:.0f}", "µs", "of math per token", COMPUTE, 72).move_to([-3.8, -1.85, 0])

        ifm = text("if math were the only limit", 24, MUTED).move_to([cx, 1.55, 0])
        fast = stat(f"{round(MATH_ONLY, -2):,.0f}", "tok/s", "", COMPUTE, 84).move_to([cx, 0.55, 0])
        nope = text("it doesn't come close", 26, RED, weight=m.BOLD).next_to(fast, m.DOWN, buff=0.45)

        sweep = [m.AnimationGroup(m.FadeIn(scan), m.FadeIn(y_on[0]), run_time=0.35)]
        sweep += [m.AnimationGroup(scan.animate.move_to(W[r].get_center()), m.FadeIn(y_on[r]), run_time=0.35) for r in range(1, 6)]
        self.say(
            "[First, the math.] [To make one token], [every weight in the model is used once.]",
            m.AnimationGroup(self.new_header("02 · math", "Two FLOPs per weight"), m.FadeIn(W), m.FadeIn(w_lab),
                             m.FadeIn(y), m.FadeIn(y_lab)),
            m.AnimationGroup(m.FadeIn(x, shift=0.2 * m.DOWN), m.FadeIn(x_lab, shift=0.2 * m.DOWN)),
            m.Succession(*sweep),
        )
        self.say(
            "[Each weight does one multiply] [and one add]: [two FLOPs.]",
            m.AnimationGroup(m.FadeIn(eq, shift=0.1 * m.UP), m.Circumscribe(glyphs(eq, "×"), color=COMPUTE, buff=0.06)),
            m.Circumscribe(glyphs(eq, "+="), color=COMPUTE, buff=0.06),
            m.FadeIn(two, shift=0.1 * m.UP),
        )
        self.say(
            "[Owl has eight billion weights], [so one token costs about sixteen billion FLOPs.]",
            m.FadeIn(block, shift=0.1 * m.UP),
            m.FadeIn(per_tok, scale=0.9),
        )
        self.say(
            "[An H100 can do nine hundred and eighty-nine trillion FLOPs per second], [so that math takes about sixteen microseconds.]",
            m.AnimationGroup(m.FadeOut(left, shift=0.2 * m.LEFT), m.FadeIn(hw_card, shift=0.2 * m.UP), m.FadeIn(hw_src)),
            m.AnimationGroup(m.FadeIn(div, shift=0.1 * m.UP), m.FadeIn(t_math, scale=0.9)),
        )
        self.say(
            "[If math were the only limit], [Owl would write over sixty thousand tokens a second.] [It doesn't come close.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(eq, two, block), shift=0.15 * m.UP), m.FadeIn(ifm, shift=0.1 * m.UP)),
            m.FadeIn(fast, scale=0.9),
            m.AnimationGroup(fast.animate.set_color(FAINT), m.FadeIn(nope, shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 3 · memory traffic: the bytes truck

    def memory(self):
        self.next_section("3 · The bytes truck", notes=(
            f"Bytes per decode step ≈ all the weights (JAX scaling book, notes §0 #2): {WBYTES / 1e9:.2f} GB for Owl-8B in BF16, "
            f"read again for every token. H100 HBM3: {H_BW / 1e12:.2f} TB/s (NVIDIA, Oct 2026) → {T_MEM * 1e3:.2f} ms per token, "
            f"{T_MEM / T_MATH:.0f}× the {T_MATH * 1e6:.1f} µs of math; the math units idle {IDLE * 100:.1f}% of the time "
            "(notes §2 worked example 1). The truck, road and token boundaries are illustrative; bars drawn to scale (the "
            "math sliver is widened to stay visible)."))
        road_top, road_bot = -0.7, -0.95
        road = m.VGroup(
            m.Line([-4.05, road_top, 0], [4.05, road_top, 0]).set_stroke(EDGE, 2.5),
            m.Line([-4.05, road_bot, 0], [4.05, road_bot, 0]).set_stroke(EDGE, 2.5),
            m.DashedLine([-4.05, (road_top + road_bot) / 2, 0], [4.05, (road_top + road_bot) / 2, 0], dash_length=0.18).set_stroke(FAINT, 2),
        )
        lane = m.Line([-4.05, (road_top + road_bot) / 2, 0], [4.05, (road_top + road_bot) / 2, 0])

        def flash():
            return m.ShowPassingFlash(lane.copy().set_stroke(MEMORY, 8), time_width=0.6)

        mem_box = card(2.5, 2.7, MEMORY, CARD)
        mem_body = m.VGroup(text("memory", 26, MEMORY, weight=m.BOLD), icon(RAM_CYAN, 0.7),
                            code("H100 · 80 GB", 20, MUTED), code(f"weights {WBYTES / 1e9:.0f} GB", 20, WEIGHTS))
        mem_body.arrange(m.DOWN, buff=0.17).move_to(mem_box)
        mem_card = m.VGroup(mem_box, mem_body).move_to([-5.3, 0.35, 0])
        chip_box = card(2.5, 2.7, COMPUTE, CARD)
        chip_icon = icon(GPU_AMBER, 0.9)
        chip_body = m.VGroup(text("math units", 26, COMPUTE, weight=m.BOLD), chip_icon, code(f"{H_PEAK / 1e12:.0f} TFLOP/s", 20, COMPUTE))
        chip_body.arrange(m.DOWN, buff=0.2).move_to(chip_box)
        chip_card = m.VGroup(chip_box, chip_body).move_to([5.3, 0.35, 0])

        rig = truck(f"{WBYTES / 1e9:.0f} GB of weights")
        lift = road_top - rig.get_bottom()[1]
        START = np.array([-2.5, rig.get_y() + lift, 0])
        END = np.array([2.5, rig.get_y() + lift, 0])
        rig.move_to(START)

        out = token_row(ANSWER[:3], size=20)
        out.move_to([0, 2.25, 0]).align_to(chip_card, m.RIGHT)
        bw_lab = code(f"bandwidth · {H_BW / 1e12:.2f} TB/s", 24, MEMORY).move_to([0, -1.32, 0])

        x0, full = -4.35, 9.6
        y_mem, y_math = -2.2, -2.9
        mem_bar = hbar(x0, y_mem, full, 0.38, MEMORY, CYAN_FILL)
        math_bar = hbar(x0, y_math, full * T_MATH / T_MEM, 0.38, COMPUTE, AMBER_FILL)
        mem_name = code("one trip", 20, MEMORY).move_to([0, y_mem, 0]).align_to([x0 - 0.2, 0, 0], m.RIGHT)
        math_name = code("the math", 20, COMPUTE).move_to([0, y_math, 0]).align_to([x0 - 0.2, 0, 0], m.RIGHT)
        mem_t = code(f"{T_MEM * 1e3:.1f} ms", 22, MEMORY).next_to(mem_bar, m.RIGHT, buff=0.15)
        math_t = code(f"{T_MATH * 1e6:.0f} µs", 22, COMPUTE).next_to(math_bar, m.RIGHT, buff=0.15)
        ratio = text(f"{T_MEM / T_MATH:.0f}× shorter", 22, MUTED).next_to(math_t, m.RIGHT, buff=0.35)
        idle = pill(f"idle {IDLE * 100:.1f}%", COMPUTE, AMBER_FILL, 22).next_to(chip_card, m.DOWN, buff=0.32)

        self.say(
            "[The catch: the weights don't live next to the math.] [They sit in memory], [a drive away from the math units.]",
            m.AnimationGroup(self.new_header("03 · memory traffic", "The bytes truck"), m.FadeIn(chip_card, shift=0.2 * m.UP)),
            m.FadeIn(mem_card, shift=0.2 * m.UP),
            m.Create(road),
        )
        self.say(
            "[Picture a truck.] [For every token], [it hauls all sixteen gigabytes of weights to the math units.]",
            m.FadeIn(rig, shift=0.3 * m.RIGHT),
            m.Circumscribe(mem_card, color=MEMORY, buff=0.08),
            m.Succession(
                m.AnimationGroup(rig.animate(run_time=2.0).move_to(END), flash()),
                m.AnimationGroup(m.Indicate(chip_icon, color=COMPUTE, scale_factor=1.12), m.FadeIn(out[0], shift=0.2 * m.UP), run_time=0.7),
            ),
        )
        self.say(
            "[One trip, one token.] [For the next token, it goes back], [and hauls the same sixteen gigabytes again.]",
            m.Circumscribe(out[0], color=TOKEN, buff=0.06),
            rig.animate(run_time=1.1).move_to(START),
            m.Succession(
                m.AnimationGroup(rig.animate(run_time=1.6).move_to(END), flash()),
                m.AnimationGroup(m.Indicate(chip_icon, color=COMPUTE, scale_factor=1.12), m.FadeIn(out[1], shift=0.2 * m.UP), run_time=0.7),
            ),
        )
        self.say(
            "[The road is the memory bandwidth]: [on an H100, three point three five terabytes per second.]",
            m.AnimationGroup(flash(), road[0].animate.set_stroke(MEMORY), road[1].animate.set_stroke(MEMORY)),
            m.AnimationGroup(
                m.FadeIn(bw_lab, shift=0.1 * m.UP),
                m.Succession(rig.animate(run_time=0.6).move_to(START),
                             m.AnimationGroup(rig.animate(run_time=1.0).move_to(END), flash()),
                             m.FadeIn(out[2], shift=0.2 * m.UP, run_time=0.4)),
            ),
        )
        self.say(
            "[So one trip takes four point eight milliseconds], [about three hundred times longer than the math.]",
            m.AnimationGroup(m.FadeIn(mem_name), m.GrowFromEdge(mem_bar, m.LEFT), m.FadeIn(mem_t)),
            m.AnimationGroup(m.FadeIn(math_name), m.GrowFromEdge(math_bar, m.LEFT), m.FadeIn(math_t), m.FadeIn(ratio, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[The math units sit idle] [ninety-nine point seven percent of the time.]",
            m.AnimationGroup(chip_body.animate.set_opacity(0.35), chip_box.animate.set_stroke(FAINT)),
            m.FadeIn(idle, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 4 · arithmetic intensity

    def build_roof(self):
        if getattr(self, "roof", None) is None:
            kw = dict(width=6.8, height=4.3, x_decades=(-1, 4), y_decades=(10.5, 15.5), center=(-1.6, -0.4, 0))
            self.roof = Roofline(H_BW, H_PEAK, **kw)
            self.mac_roof = Roofline(M_BW, MAC_PEAK, **kw)
            assert np.allclose(self.roof.origin, self.mac_roof.origin)  # one set of axes for both machines
        return self.roof

    def intensity(self):
        self.next_section("4 · Intensity", notes=(
            f"Arithmetic intensity I = FLOPs ÷ bytes (notes §2). Owl at batch 1: {FLOPS / 1e9:.2f} GFLOP ÷ {WBYTES / 1e9:.2f} GB "
            f"= {INTENSITY:.2f} FLOP/byte. The H100's balance (ridge) = {H_PEAK / 1e12:.0f} TFLOP/s ÷ {H_BW / 1e12:.2f} TB/s "
            f"= {RIDGE_H:.0f} FLOPs/byte (owl_reference.ridge(); the JAX book's 'about 280' uses slightly different peaks). "
            "Below the ridge a kernel is memory-bound, above it compute-bound."))
        roof = self.build_roof()
        y_ax = roof.origin[1]
        f_int = formula([("intensity", INK), ("=", MUTED), ("FLOPs", COMPUTE), ("÷", MUTED), ("bytes", MEMORY)], 54).move_to([0, 2.0, 0])
        sub = text("FLOPs done for every byte moved", 22, MUTED).next_to(f_int, m.DOWN, buff=0.22)
        val1 = code(f"{FLOPS / 1e9:.2f} GFLOP ÷ {WBYTES / 1e9:.2f} GB ≈ {INTENSITY:.0f} FLOP/byte", 22, INK,
                    t2c={r"[\d.]+ GFLOP": COMPUTE, r"[\d.]+ GB": MEMORY})
        val2 = code(f"{H_PEAK / 1e12:.0f} TFLOP/s ÷ {H_BW / 1e12:.2f} TB/s ≈ {RIDGE_H:.0f} FLOPs/byte", 22, INK,
                    t2c={r"\d+ TFLOP/s": COMPUTE, r"[\d.]+ TB/s": MEMORY})
        col = 6.3 - max(val1.width, val2.width)
        val1.move_to([0, 0.5, 0]).align_to([col, 0, 0], m.LEFT)
        val2.move_to([0, -0.25, 0]).align_to([col, 0, 0], m.LEFT)
        lab1 = text("Owl, one token per step", 24, INK).move_to([0, 0.5, 0]).align_to([col - 0.35, 0, 0], m.RIGHT)
        lab2 = text("H100, its balance point", 24, INK).move_to([0, -0.25, 0]).align_to([col - 0.35, 0, 0], m.RIGHT)
        sit(lab1, baseline(val1))
        sit(lab2, baseline(val2))

        x_axis = m.VGroup(roof.axes[0], roof.ticks[:6], roof.x_label)
        band = 0.62
        xr = roof.at(RIDGE_H, 10 ** roof.yd[0])[0]
        x_lo, x_hi = roof.origin[0], roof.origin[0] + roof.w
        cy_rect = m.Rectangle(width=xr - x_lo, height=band).set_stroke(width=0).set_fill(MEMORY, 0.14)
        cy_rect.move_to([(x_lo + xr) / 2, y_ax + band / 2 + 0.02, 0])
        am_rect = m.Rectangle(width=x_hi - xr, height=band).set_stroke(width=0).set_fill(COMPUTE, 0.14)
        am_rect.move_to([(xr + x_hi) / 2, y_ax + band / 2 + 0.02, 0])
        self.owl_dot = dot = m.Dot(roof.at(INTENSITY, 10 ** roof.yd[0]), radius=0.085, color=MEMORY).set_z_index(3)
        cy_lab = text("memory-bound", 22, MEMORY, weight=m.BOLD).move_to([(dot.get_x() + xr) / 2 + 0.35, y_ax + band / 2 + 0.02, 0])
        am_lab = text("compute-bound", 22, COMPUTE, weight=m.BOLD).move_to(am_rect).align_to([xr + 0.1, 0, 0], m.LEFT)
        tick = m.DashedLine([xr, y_ax + band + 0.06, 0], [xr, y_ax + band + 0.34, 0], dash_length=0.07).set_stroke(MUTED, 2)
        tick_lab = code(f"ridge {RIDGE_H:.0f}", 20, MUTED).next_to(tick, m.UP, buff=0.1)
        dot_lab = code("Owl", 20, MEMORY, weight=m.BOLD).move_to([dot.get_x(), y_ax + band / 2 + 0.02, 0])
        self.s4_extras = m.VGroup(f_int, sub, lab1, val1, lab2, val2, cy_rect, am_rect, cy_lab, am_lab, tick, tick_lab, dot_lab)

        self.say(
            "[The ratio that decides is arithmetic intensity]: [FLOPs done for every byte moved.]",
            m.AnimationGroup(self.new_header("04 · intensity", "FLOPs per byte"), m.FadeIn(f_int, shift=0.15 * m.UP)),
            m.FadeIn(sub, shift=0.1 * m.UP),
        )
        self.say(
            "[For Owl, one token per step]: [sixteen billion FLOPs over sixteen billion bytes], [about one FLOP per byte.]",
            m.AnimationGroup(m.FadeIn(lab1), m.FadeIn(x_axis)),
            m.FadeIn(val1, shift=0.1 * m.LEFT),
            m.AnimationGroup(m.FadeIn(dot, scale=0.4), m.FadeIn(dot_lab), m.Circumscribe(glyphs(val1, f"≈ {INTENSITY:.0f} FLOP/byte"), color=MEMORY, buff=0.06)),
        )
        self.say(
            "[The chip has a ratio too]: [its peak FLOPs over its bandwidth.] [For the H100, about two hundred and ninety-five.]",
            m.FadeIn(lab2),
            m.FadeIn(val2, shift=0.1 * m.LEFT),
            m.AnimationGroup(m.Create(tick), m.FadeIn(tick_lab), m.Circumscribe(glyphs(val2, f"≈ {RIDGE_H:.0f} FLOPs/byte"), color=INK, buff=0.06)),
        )
        self.say(
            "[Below that balance point, memory sets the pace]. [Above it, math does.]",
            m.AnimationGroup(m.FadeIn(cy_rect), m.FadeIn(cy_lab)),
            m.AnimationGroup(m.FadeIn(am_rect), m.FadeIn(am_lab)),
        )
        self.wait(0.3)

    # ------------------------------------------------------------------ 5 · the roofline

    def roofline(self):
        self.next_section("5 · The roofline", notes=(
            f"Roofline: attainable FLOP/s = min(peak, bandwidth × intensity) (notes §2), log–log. H100: slope {H_BW / 1e12:.2f} TB/s, "
            f"roof {H_PEAK / 1e12:.0f} TFLOP/s, ridge {RIDGE_H:.0f}. Owl at batch 1 (I ≈ 1) reaches {ON_SLOPE / 1e12:.2f} TFLOP/s = "
            f"{ROOF_FRAC * 100:.2f}% of the roof: memory-bound. This Mac (M4 Max): slope {M_BW / 1e9:.0f} GB/s (Apple spec), roof "
            f"{MAC_PEAK / 1e12:.1f} TFLOP/s = a bf16 matmul MEASURED on this Mac (Apple publishes no GPU TFLOPS; "
            f"research/measurements.md §0), ridge ≈ {RIDGE_M:.0f}. Prefill of a long prompt has I ≈ prompt length (episode 3); "
            "batching B users gives I ≈ B (episode 4). Dot for prefill at 1,500 tokens, the users arrow schematic."))
        roof, mac = self.roof, self.mac_roof
        dot = self.owl_dot
        y_title = text("FLOPs per second you can reach", 22, MUTED)
        y_title.move_to([0, 2.2, 0]).align_to([roof.origin[0] + 0.3, 0, 0], m.LEFT)
        rx = roof.get_right()[0] + 0.45   # the right column, beside the chart
        heading = code("H100 roofline", 22, INK, weight=m.BOLD).move_to([0, 2.15, 0]).align_to([rx, 0, 0], m.LEFT)
        leg_mem = legend(MEMORY, "memory", "bandwidth × intensity").move_to([0, 1.5, 0]).align_to([rx, 0, 0], m.LEFT)
        leg_cmp = legend(COMPUTE, "math", f"peak · {H_PEAK / 1e12:.0f} TFLOP/s").move_to([0, 0.75, 0]).align_to([rx, 0, 0], m.LEFT)
        bend = roof.point(RIDGE_H)
        ridge_lab = code(f"ridge ≈ {RIDGE_H:.0f}", 20, MUTED).next_to(bend, m.UP, buff=0.25)

        def slope_y(x, rl=roof):
            """The y of a roofline's memory slope at scene x."""
            a, b = rl.memory.get_start(), rl.memory.get_end()
            return a[1] + (x - a[0]) * (b[1] - a[1]) / (b[0] - a[0])

        p1 = roof.point(INTENSITY)
        guide = m.DashedLine(dot.get_center(), p1, dash_length=0.08).set_stroke(FAINT, 2)
        h_lab = code("H100", 20, MEMORY, weight=m.BOLD).next_to(p1, m.UL, buff=0.12)
        h_stat = stat(f"{ON_SLOPE / 1e12:.2f}", "TFLOP/s", "Owl decode on an H100", MEMORY, 50)
        h_stat.move_to([0, -0.25, 0]).align_to([rx, 0, 0], m.LEFT)
        gap = m.DashedLine(p1, roof.at(INTENSITY, H_PEAK), dash_length=0.1).set_stroke(COMPUTE, 2.5)
        gap_lab = code(f"×{H_PEAK / ON_SLOPE:.0f}", 22, COMPUTE).next_to(gap, m.RIGHT, buff=0.15)
        frac = text(f"= {ROOF_FRAC * 100:.2f}% of the roof", 22, INK).next_to(h_stat, m.DOWN, buff=0.18).align_to(h_stat, m.LEFT)
        bound = pill("memory-bound", MEMORY, CYAN_FILL, 22).next_to(frac, m.DOWN, buff=0.28).align_to(h_stat, m.LEFT)

        for line in (mac.memory, mac.compute):
            line.set_stroke(width=3.5, opacity=0.9)
        mac.ridge.set_stroke(FAINT, 2)
        roof_end = mac.point(10 ** roof.xd[1])
        mac_roof_lab = code("M4 Max roof", 20, COMPUTE).next_to(roof_end, m.DOWN, buff=0.16).align_to(roof_end, m.RIGHT)
        m1 = mac.point(INTENSITY)
        mac_dot = m.Dot(m1, radius=0.1, color=MEMORY).set_z_index(3)
        mac_lab = code("M4 Max", 20, MEMORY, weight=m.BOLD).next_to(m1, m.DR, buff=0.12)
        mac_stat = stat(f"≈ {RIDGE_M:.0f}", "FLOPs/byte", "Mac ridge · measured", INK, 44)
        mac_stat.move_to([0, -2.55, 0]).align_to([rx, 0, 0], m.LEFT)

        pre_dot = roof.dot(1500, COMPUTE)
        pre_lab = code("prefill · ep 3", 20, COMPUTE).next_to(pre_dot, m.DOWN, buff=0.2).align_to([bend[0] + 0.22, 0, 0], m.LEFT)
        d = roof.point(100) - roof.point(2)
        normal = np.array([-d[1], d[0], 0]) / np.linalg.norm(d)
        users = m.Arrow(roof.point(2) + 0.3 * normal, roof.point(60) + 0.3 * normal, buff=0, color=REQUEST, stroke_width=5,
                        max_tip_length_to_length_ratio=0.12)
        users_lab = code("more users\nep 4", 20, REQUEST)
        x_l = users.get_start()[0] - 0.85
        users_lab.move_to([x_l + users_lab.width / 2, slope_y(x_l + users_lab.width) + 0.6 + users_lab.height / 2, 0])

        self.say(
            "[Now add a second axis]: [the FLOPs per second you can actually reach.]",
            m.AnimationGroup(self.new_header("05 · roofline", "Two limits, one chart"), m.FadeOut(self.s4_extras), m.FadeIn(roof.axes[1])),
            m.AnimationGroup(m.FadeIn(roof.ticks[6:]), m.FadeIn(y_title)),
        )
        self.say(
            "[On the left, a slope]: [bandwidth times intensity, where memory is the limit.]",
            m.Create(roof.memory),
            m.AnimationGroup(m.FadeIn(heading), m.FadeIn(leg_mem, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[On the right, a flat roof]: [the chip's peak, where math is the limit.]",
            m.Create(roof.compute),
            m.FadeIn(leg_cmp, shift=0.1 * m.LEFT),
        )
        self.say(
            "[Together they draw the roofline], [and they meet at the ridge, two hundred and ninety-five FLOPs per byte.]",
            m.Succession(pulse(roof.memory, MEMORY, 12), pulse(roof.compute, COMPUTE, 12)),
            m.AnimationGroup(m.Create(roof.ridge), m.FadeIn(ridge_lab, shift=0.1 * m.UP)),
        )
        self.say(
            "[Now place Owl], [at one FLOP per byte.] [It only rises as far as the slope]: [three point three five trillion FLOPs per second.]",
            m.Circumscribe(dot, color=MEMORY, buff=0.08),
            m.Indicate(roof.ticks[1], color=MEMORY, scale_factor=1.3),
            m.AnimationGroup(dot.animate(run_time=1.4).move_to(p1), m.Create(guide, run_time=1.4), m.FadeIn(h_lab)),
            m.FadeIn(h_stat, shift=0.1 * m.LEFT),
        )
        self.say(
            "[That's a third of a percent of the roof.] [For one user chatting, Owl is memory-bound.]",
            m.AnimationGroup(m.Create(gap), m.FadeIn(gap_lab), m.FadeIn(frac, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(bound, shift=0.1 * m.UP), m.Circumscribe(roof.memory, color=MEMORY, buff=0.1)),
        )
        self.say(
            "[This Mac's roofline sits lower], [with a ridge near twenty-eight], [and chat is memory-bound here too.]",
            m.AnimationGroup(m.Create(mac.memory), m.Create(mac.compute), m.FadeIn(mac_roof_lab)),
            m.AnimationGroup(m.Create(mac.ridge), m.FadeIn(mac_stat, shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.FadeIn(mac_dot, scale=0.4), m.FadeIn(mac_lab)),
        )
        self.say(
            "[Long prompts and many users move right along the chart]: [that's episodes three and four.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(gap, gap_lab)), m.FadeIn(pre_dot, scale=0.4), m.FadeIn(pre_lab),
                             m.GrowArrow(users), m.FadeIn(users_lab)),
            m.AnimationGroup(m.Circumscribe(pre_lab, color=COMPUTE, buff=0.08), m.Circumscribe(users_lab, color=REQUEST, buff=0.08)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 6 · the ceiling

    def ceiling(self):
        self.next_section("6 · The ceiling", notes=(
            f"Batch-1 decode ceiling = bandwidth ÷ bytes per token (owl_reference.decode_ceiling(); notes §0 #5, §2). "
            f"H100: {H_BW / 1e12:.2f} TB/s ÷ {WBYTES / 1e9:.2f} GB = {CEIL_H:.1f} tok/s. M4 Max: {M_BW / 1e9:.0f} GB/s ÷ "
            f"{WBYTES / 1e9:.2f} GB = {CEIL_M:.1f} tok/s. Specs dated October 2026 (NVIDIA; Apple Mac Studio 2025 tech specs). "
            "Peak FLOPs do not appear in the formula: only more bandwidth or fewer bytes per token (quantization, episodes "
            "18–19) raise it. The code is an excerpt of owl_reference (decode_ceiling = 1 / step_time at batch 1, which is "
            "this division while memory-bound). Bars to scale."))
        top = formula([("tok/s", INK), ("≈", MUTED), ("bandwidth", MEMORY), ("÷", MUTED), ("bytes per token", WEIGHTS)], 50)
        top.move_to([0, 1.95, 0])
        src = ('def decode_ceiling(model=OWL, hw=GPU, bits=16):\n'
               '    # batch 1, memory-bound: one full read per token\n'
               '    return hw["bandwidth"] / weight_bytes(model, bits)')
        block = code_block(src, 20, title="owl_reference.py (excerpt)", accent=MEMORY).move_to([0, 0.42, 0])

        x0, full = -3.65, 4.8
        rows = []
        for y, name, sub, ic, bw, ceil in (
            (-1.3, "H100", "NVIDIA spec", icon(GPU_INK, 0.62), H_BW, CEIL_H),
            (-2.35, "M4 Max", "this Mac", mac_studio(0.95), M_BW, CEIL_M),
        ):
            ic.move_to([-6.05, y, 0])
            names = m.VGroup(text(name, 24, INK, weight=m.BOLD), cite(sub)).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.06)
            names.next_to(ic, m.RIGHT, buff=0.22)
            bar = hbar(x0, y, full * bw / H_BW, 0.46, MEMORY, CYAN_FILL)
            bw_txt = f"{bw / 1e12:.2f} TB/s" if bw >= 1e12 else f"{bw / 1e9:.0f} GB/s"
            bw_lab = code(bw_txt, 22, MEMORY, weight=m.BOLD).set_z_index(2)
            if bw_lab.width + 0.3 < bar.width:
                bw_lab.move_to(bar)
            else:
                bw_lab.next_to(bar, m.RIGHT, buff=0.15)
            div = code(f"÷ {WBYTES / 1e9:.2f} GB", 22, WEIGHTS).move_to([2.45, y, 0])
            res = stat(f"{ceil:.0f}", "tok/s", "", MEMORY, 56)
            res.move_to([0, y, 0]).align_to([3.7, 0, 0], m.LEFT)
            sit(div, baseline(res[0]))
            rows.append((m.VGroup(ic, names), bar, bw_lab, div, res))
        h, mc = rows

        nomath = pill("faster math: no change", RED, RED_FILL, 22)
        more_bw = pill("more bandwidth", MEMORY, CYAN_FILL, 22)
        fewer = pill("fewer bytes · ep 18–19", WEIGHTS, BLUE_FILL, 22)
        levers = m.VGroup(nomath, more_bw, fewer).arrange(m.RIGHT, buff=0.55).move_to([0, -3.3, 0])

        self.say(
            "[When memory is the limit, speed is one division]: [bandwidth, over bytes per token.]",
            m.AnimationGroup(self.new_header("06 · the ceiling", "Bandwidth ÷ bytes"), m.FadeIn(top, shift=0.15 * m.UP)),
            m.FadeIn(block, shift=0.1 * m.UP),
        )
        self.say(
            "[An H100 moves three point three five terabytes per second]. [Over sixteen gigabytes], [that's about two hundred and nine tokens a second.]",
            m.AnimationGroup(m.FadeIn(h[0], shift=0.1 * m.RIGHT), m.GrowFromEdge(h[1], m.LEFT), m.FadeIn(h[2])),
            m.FadeIn(h[3], shift=0.1 * m.LEFT),
            m.FadeIn(h[4], scale=0.85),
        )
        self.say(
            "[This Mac, an M4 Max, moves five hundred and forty-six gigabytes per second]: [about thirty-four tokens a second.]",
            m.AnimationGroup(m.FadeIn(mc[0], shift=0.1 * m.RIGHT), m.GrowFromEdge(mc[1], m.LEFT), m.FadeIn(mc[2]), m.FadeIn(mc[3], shift=0.1 * m.LEFT)),
            m.FadeIn(mc[4], scale=0.85),
        )
        self.say(
            "[Faster math changes neither ceiling.] [Only more bandwidth], [or fewer bytes.]",
            m.FadeIn(nomath, shift=0.1 * m.UP),
            m.FadeIn(more_bw, shift=0.1 * m.UP),
            m.FadeIn(fewer, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 7 · checked on this Mac

    def check(self):
        self.next_section("7 · Checked on this Mac", notes=(
            f"Measured on this Mac (Mac Studio, M4 Max 546 GB/s, mlx-lm 0.32.2, 2026-10-24; research/measurements.md §1): "
            f"Llama-3.1-8B-Instruct bf16 decodes at {TOK:.2f} tok/s (median of 5 trials, 256 tokens, greedy; the stock "
            f"mlx_lm.generate cross-check gives {CLI:.2f}). Ceiling {M_BW / 1e9:.0f} GB/s ÷ {WBYTES / 1e9:.2f} GB = {CEIL_M:.1f} tok/s, "
            f"so the Mac reaches {OF_CEIL * 100:.0f}% of it. The terminal shows the command shortened."))
        box = mac_studio(1.35).move_to([-5.75, 1.95, 0])
        who = m.VGroup(text("Mac Studio · M4 Max", 24, INK, weight=m.BOLD), code(f"{M_BW / 1e9:.0f} GB/s · 64 GB", 20, MUTED))
        who.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.08).next_to(box, m.RIGHT, buff=0.3)
        term = terminal([
            "$ mlx_lm.generate --temp 0 \\",
            "  --model …Llama-3.1-8B-Instruct-bf16 \\",
            "  --max-tokens 256 --prompt …",
            f"Generation: 256 tokens, {CLI:.1f} tokens-per-sec",
        ], width=6.3, size=18, title="this Mac").move_to([0, 0.15, 0])
        term.align_to([-6.6, 0, 0], m.LEFT)
        term[4][3].set_color(TOKEN)
        src = cite("measured on this Mac · mlx-lm").next_to(term, m.DOWN, buff=0.15).align_to(term, m.LEFT)
        mascot = owl(0.8).move_to([-5.85, -2.65, 0])

        rx, bw_full = 1.2, 3.0
        c_cap = code(f"ceiling · {M_BW / 1e9:.0f} GB/s ÷ {WBYTES / 1e9:.2f} GB", 20, MEMORY)
        c_cap.move_to([0, 2.0, 0]).align_to([rx, 0, 0], m.LEFT)
        c_bar = hbar(rx, 1.5, bw_full, 0.44, MEMORY, CYAN_FILL)
        c_val = stat(f"{CEIL_M:.1f}", "tok/s", "", MEMORY, 40).next_to(c_bar, m.RIGHT, buff=0.25)
        m_cap = code("measured · mlx-lm, 256 tokens", 20, TOKEN).move_to([0, 0.8, 0]).align_to([rx, 0, 0], m.LEFT)
        m_bar = hbar(rx, 0.3, bw_full * OF_CEIL, 0.44, TOKEN, GREEN_FILL)
        m_val = stat(f"{TOK:.1f}", "tok/s", "", TOKEN, 40).move_to([0, 0.3, 0]).align_to(c_val, m.LEFT)
        pct = stat(f"{OF_CEIL * 100:.0f}%", "", "of the ceiling, measured on this Mac", TOKEN, 84)
        pct.move_to([0, -1.55, 0]).align_to([rx, 0, 0], m.LEFT)

        self.say(
            "[Now the check.] [We ran Owl's model in sixteen-bit on this Mac], [with MLX.]",
            m.AnimationGroup(self.new_header("07 · measured", "Checked on this Mac"), m.FadeIn(box, shift=0.1 * m.UP), m.FadeIn(who)),
            m.AnimationGroup(m.FadeIn(term[:4]), m.LaggedStart(*[m.FadeIn(r, shift=0.05 * m.RIGHT) for r in term[4][:3]], lag_ratio=0.4)),
            m.AnimationGroup(m.FadeIn(term[4][3], shift=0.05 * m.UP), m.FadeIn(src)),
        )
        self.say(
            "[It decoded thirty-one tokens a second], [against a ceiling of thirty-four]: [ninety-one percent.]",
            m.AnimationGroup(m.FadeIn(m_cap), m.GrowFromEdge(m_bar, m.LEFT), m.FadeIn(m_val)),
            m.AnimationGroup(m.FadeIn(c_cap), m.GrowFromEdge(c_bar, m.LEFT), m.FadeIn(c_val)),
            m.AnimationGroup(m.FadeIn(pct, shift=0.1 * m.UP), m.FadeIn(mascot, shift=0.2 * m.UP)),
        )
        self.wait(0.3)
        self.clear()
        self.bandwidth()

    # ------------------------------------------------------------------ 8 · speed × size = bandwidth

    def bandwidth(self):
        self.next_section("8 · Speed × size = bandwidth", notes=(
            f"Bytes really read per token: {READ / 1e9:.2f} GB = every tensor except the untied {EMBED / 1e9:.2f} GB input-embedding "
            "table (a row lookup: one 4,096-wide row per token), plus the KV cache at the run's average context (research/"
            f"measurements.md, 'Bytes'). Multiply speed by size: {TOK:.1f} tok/s × {READ / 1e9:.2f} GB = {ACHIEVED / 1e9:.0f} GB/s. "
            f"MBU = achieved ÷ peak bandwidth (Databricks, notes §2) = {MBU * 100:.1f}% of Apple's {M_BW / 1e9:.0f} GB/s. A plain "
            f"streaming read (mx.sum over 4 GB) reaches {READ_ROOF / 1e9:.1f} GB/s here, so decode streams at {OF_READ * 100:.0f}% "
            "of the practical roof. All measured on this Mac, 2026-10-24."))
        summary = stat(f"{TOK:.1f}", "tok/s", "measured on this Mac", TOKEN, 60)
        summary.move_to([0, 1.85, 0]).align_to([-6.3, 0, 0], m.LEFT)

        rx, rw = 0.55, 5.85
        bytes_cap = code(f"Owl-8B BF16 · {WBYTES / 1e9:.2f} GB of weights", 20, MUTED).move_to([0, 2.3, 0]).align_to([rx, 0, 0], m.LEFT)
        w_bar = hbar(rx, 1.82, rw, 0.4, WEIGHTS, BLUE_FILL)
        e_w = rw * EMBED / WBYTES
        e_seg = hbar(rx, 1.82, e_w, 0.4, INK, "#5C7FB8")
        e_off = hbar(rx, 1.82, e_w, 0.4, FAINT, CARD).set_z_index(1)
        e_lab = code(f"embedding {EMBED / 1e9:.2f} GB · a lookup", 20, MUTED).next_to(w_bar, m.DOWN, buff=0.16).align_to(w_bar, m.LEFT)
        r_lab = code(f"{READ / 1e9:.2f} GB read per token", 20, WEIGHTS, weight=m.BOLD).next_to(e_lab, m.DOWN, buff=0.12).align_to(w_bar, m.RIGHT)

        eq_title = text("multiply speed by size", 24, MUTED).move_to([0, 0.55, 0])
        full_eq = code(f"{TOK:.1f} tok/s × {READ / 1e9:.2f} GB = {ACHIEVED / 1e9:.0f} GB/s", 32, INK,
                       t2c={r"[\d.]+ tok/s": TOKEN, r"[\d.]+ GB(?!/)": WEIGHTS, r"\d+ GB/s": MEMORY})
        full_eq.next_to(eq_title, m.DOWN, buff=0.22)
        lhs = glyphs(full_eq, f"{TOK:.1f} tok/s × {READ / 1e9:.2f} GB")
        rhs = glyphs(full_eq, f"= {ACHIEVED / 1e9:.0f} GB/s")

        gx, gw = -2.6, 6.6
        g_cap = text("bandwidth on this Mac", 22, MUTED).move_to([0, -0.95, 0]).align_to([gx, 0, 0], m.LEFT)
        g_rows = {}
        for key, y, name, value, color, fill in (
            ("spec", -1.45, "Apple spec", M_BW, MUTED, CARD),
            ("read", -2.0, "plain read", READ_ROOF, INK, CARD_HI),
            ("dec", -2.55, "Owl decode", ACHIEVED, MEMORY, CYAN_FILL),
        ):
            lab = code(name, 20, color).move_to([0, y, 0]).align_to([gx - 0.25, 0, 0], m.RIGHT)
            bar = hbar(gx, y, gw * value / M_BW, 0.4, MEMORY if key == "spec" else color, fill)
            val = code(f"{value / 1e9:.0f} GB/s", 20, color)
            val.move_to([0, y, 0]).align_to([gx + gw + 0.2, 0, 0], m.LEFT)
            g_rows[key] = (lab, bar, val)
        mbu = pill(f"MBU = {ACHIEVED / 1e9:.0f} ÷ {M_BW / 1e9:.0f} = {MBU * 100:.0f}%", MEMORY, CYAN_FILL, 22)
        of_read = pill(f"{OF_READ * 100:.0f}% of a plain read", INK, CARD_HI, 22)
        pills = m.VGroup(mbu, of_read).arrange(m.RIGHT, buff=0.45).move_to([0, -3.2, 0]).align_to([gx, 0, 0], m.LEFT)
        sp, rd, dc = g_rows["spec"], g_rows["read"], g_rows["dec"]

        self.say(
            "[Now count the bytes.] [The embedding table is a lookup, one row per token], [so each token really reads fifteen gigabytes, not sixteen.]",
            m.AnimationGroup(self.new_header("08 · bandwidth", "Speed × size = bandwidth"), m.FadeIn(summary, shift=0.1 * m.UP),
                             m.FadeIn(bytes_cap), m.GrowFromEdge(w_bar, m.LEFT), m.FadeIn(e_seg)),
            m.AnimationGroup(m.FadeIn(e_off), m.FadeIn(e_lab, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(r_lab, shift=0.1 * m.UP), m.Circumscribe(w_bar, color=WEIGHTS, buff=0.06)),
        )
        self.say(
            "[Multiply speed by size], [and you get bandwidth]: [four hundred and sixty-six gigabytes per second.]",
            m.AnimationGroup(m.FadeIn(eq_title), m.Circumscribe(summary[0], color=TOKEN, buff=0.08)),
            m.FadeIn(lhs, shift=0.1 * m.UP),
            m.FadeIn(rhs, shift=0.1 * m.LEFT),
        )
        self.say(
            "[Against the spec's five hundred and forty-six], [that's eighty-five percent]: [the model bandwidth utilization.]",
            m.AnimationGroup(m.FadeIn(g_cap), m.FadeIn(sp[0]), m.GrowFromEdge(sp[1], m.LEFT), m.FadeIn(sp[2])),
            m.AnimationGroup(m.FadeIn(dc[0]), m.GrowFromEdge(dc[1], m.LEFT), m.FadeIn(dc[2])),
            m.FadeIn(mbu, shift=0.1 * m.UP),
        )
        self.say(
            "[A plain memory read here tops out at four hundred and eighty-three], [so decode gets ninety-seven percent of what the memory can really give.]",
            m.AnimationGroup(m.FadeIn(rd[0]), m.GrowFromEdge(rd[1], m.LEFT), m.FadeIn(rd[2])),
            m.FadeIn(of_read, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 8 · recap

    def recap(self):
        self.next_section("9 · Recap", notes=(
            f"Every step is limited by math or by memory traffic; for chat (batch 1) it is memory: tok/s ≈ bandwidth ÷ bytes. "
            f"On this Mac: {TOK:.1f} of {CEIL_M:.1f} tok/s ({OF_CEIL * 100:.0f}%, i.e. within {(1 - OF_CEIL) * 100:.0f}%). "
            "Three words: math, memory, measure. Next: episode 3, prefill vs decode."))
        t_math, t_mem = limit_tiles()
        tiles = m.VGroup(t_math, t_mem).arrange(m.RIGHT, buff=0.4).move_to([0, 0.9, 0])
        chat = pill("for chat: this one", MEMORY, CYAN_FILL, 22).next_to(t_mem, m.DOWN, buff=0.35)

        halo = glow(MEMORY, radius=3.4).move_to([0, 0.55, 0])
        words = m.VGroup(title_text("Math", 76, COMPUTE), title_text("Memory", 76, MEMORY), title_text("Measure", 76, TOKEN))
        words.arrange(m.RIGHT, buff=1.3).move_to([0, 0.7, 0])
        on_baseline(words)
        a1 = arrow(words[0].get_right(), words[1].get_left(), MUTED, buff=0.25)
        a2 = arrow(words[1].get_right(), words[2].get_left(), MUTED, buff=0.25)
        tag = formula([("tok/s", INK), ("≈", MUTED), ("bandwidth", MEMORY), ("÷", MUTED), ("bytes", WEIGHTS)], 40).move_to([0, -1.05, 0])
        proof = code(f"this Mac: {TOK:.1f} of {CEIL_M:.1f} tok/s · {OF_CEIL * 100:.0f}%", 22, TOKEN).next_to(tag, m.DOWN, buff=0.35)
        mascot = owl(0.8).move_to([0, -2.85, 0])

        self.say(
            "[Every step is limited by math or by memory traffic]. [For chat, it's memory.]",
            m.AnimationGroup(self.new_header("09 · recap", "Math, memory, measure"), m.FadeIn(tiles, shift=0.15 * m.UP)),
            m.AnimationGroup(t_math.animate.set_opacity(0.35), m.Circumscribe(t_mem, color=MEMORY, buff=0.1), m.FadeIn(chat, shift=0.1 * m.UP)),
        )
        self.say(
            "[Three words to keep]: [math], [memory], [measure.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(tiles, chat), shift=0.2 * m.UP), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.2 * m.UP),
            m.AnimationGroup(m.GrowArrow(a1), m.FadeIn(words[1], shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowArrow(a2), m.FadeIn(words[2], shift=0.2 * m.UP)),
        )
        self.say(
            "[Tokens per second is bandwidth over bytes], [and this Mac lands within nine percent of it.]",
            m.FadeIn(tag, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(proof, shift=0.1 * m.UP), m.FadeIn(mascot, shift=0.2 * m.UP)),
        )
        self.wait(0.6)
        self.clear()
