"""Tokens per Second · Episode 4 · The Batching Free Lunch (Module 1 · One Token, Two Speed Limits).

One decode step reads all 16 GB of Owl's weights whether it serves one student or sixty-four: the batch
shares the weight sweep, so total tokens/s grows with the batch until the math catches up at the ridge
(the critical batch). Then the real curve, measured on this Mac (8B BF16 and 4-bit, batch 1-16, with its
dip at 16), the cost per million tokens falling with batch, and the catch: every student's KV cache.

    VOICE=draft manimgx inspect scene.py      # silent layout pass
    manimgx render scene.py                   # with the shared narrator running
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from owl_reference import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- the numbers (all computed)

GB = 1e9
W_GB = weight_bytes() / GB                       # 16.06 GB of BF16 weights
STEP1 = step_time()                              # one H100 decode step at batch 1 (weights only)
MATH1 = flops_per_token() / GPU["peak"]          # the math of one token on an H100
RIDGE = ridge(GPU)                               # 295 FLOPs/byte = the critical batch for BF16


def total(b):
    """Tokens/s of the whole batch on an H100 (owl_reference.step_time, weights only)."""
    return b / step_time(batch=b)


def each(b):
    """Tokens/s that each student sees."""
    return 1 / step_time(batch=b)


assert round(W_GB) == 16 and round(STEP1 * 1e3, 1) == 4.8 and round(MATH1 * 1e6) == 16
assert round(each(1)) == 209 and step_time(batch=64) == STEP1               # the step doesn't move
assert round(total(64)) == 13_350 and round(RIDGE) == 295
assert round(total(1000)) == 61_582 and round(each(1000)) == 62               # past the ridge: flat total

# measured on this Mac (research/measurements.md §8: mlx-lm 0.32.2, 256-token prompts, 128 tokens, 3 trials)
RUN = measured("batching", "llama-3.1-8b")
BS = [1, 2, 3, 4, 6, 8, 10, 12, 16]
BF = {b: RUN["bf16"]["by_batch"][str(b)] for b in BS + [32, 64]}
Q4 = {b: RUN["4bit"]["by_batch"][str(b)] for b in BS + [32]}


def agg(run, b):
    return run[b]["aggregate_tok_s"]


def per(run, b):
    return run[b]["per_seq_tok_s"]


CEIL_BF = MAC["bandwidth"] / weight_bytes()                                  # 546 GB/s ÷ 16.06 GB
CEIL_Q4 = measured("decode", "llama-3.1-8b", "4bit", "ceiling_tok_s")         # 546 GB/s ÷ 4.52 GB
Q4_GB = measured("models", "llama-3.1-8b", "4bit", "weights_gb")
BYTES_RATIO = measured("decode", "llama-3.1-8b", "4bit", "bytes_ratio_vs_bf16")
MAC_RIDGE = measured("headline", "ridge_flops_per_byte")                     # 15.2 TFLOPS ÷ 546 GB/s
STOCK = measured("cli_crosscheck_mlx_lm_benchmark")
STOCK12 = STOCK["Meta-Llama-3.1-8B-Instruct-bf16__p256_g128_b12"]["generation_tps"]
STOCK16 = STOCK["Meta-Llama-3.1-8B-Instruct-bf16__p256_g128_b16"]["generation_tps"]

assert round(agg(BF, 1)) == 31 and round(CEIL_BF) == 34 and round(measured("decode", "llama-3.1-8b", "bf16", "ceiling_tok_s")) == 34
assert [round(agg(BF, b)) for b in (2, 4, 8)] == [63, 122, 235]
assert round(per(BF, 8)) == 29 and round(agg(BF, 8) / agg(BF, 1), 1) == 7.6 and round(100 * (1 - per(BF, 8) / per(BF, 1))) == 5
assert round(agg(BF, 12)) == 301 and round(agg(BF, 16)) == 243 and agg(BF, 16) < agg(BF, 12)    # the dip
assert round(agg(BF, 32)) == 415 and round(agg(BF, 64)) == 686
assert round(STOCK12) == 293 and round(STOCK16) == 240                                           # same dip, stock tool
assert round(MAC_RIDGE) == 28
assert round(agg(Q4, 1)) == 101 and round(CEIL_Q4) == 121 and round(Q4_GB, 1) == 4.5 and 3.4 < BYTES_RATIO < 3.7
assert round(agg(Q4, 2)) == 190 and round(agg(Q4, 3)) == 248
assert all(265 < agg(Q4, b) <= 300.5 for b in (4, 6, 8, 10, 12))                                  # flat near 300
assert round(per(Q4, 1)) == 101 and round(per(Q4, 12)) == 25 and round(agg(Q4, 16)) == 358
assert round(agg(Q4, 32)) == 676 and round(per(Q4, 32)) == 21

# cost per million tokens (notes §15): one H100 at a dated rental price, 2K-token chats, 75% of peak bandwidth
PRICE = 3.99            # $ per H100 SXM GPU-hour, Lambda (read 2026-10-23) — dated, October 2026
CTX, MBU = 2048, 0.75   # for example


def cost_step(b):
    t = step_time(batch=b, context=CTX)
    assert t * GPU["peak"] > b * flops_per_token()      # still memory-bound, so dividing by MBU is right
    return t / MBU


def tps(b):
    return b / cost_step(b)


def usd_per_m(b):
    return PRICE / (tps(b) * 3600 / 1e6)


COST_B = [1, 8, 32, 64]
assert [f"{usd_per_m(b):.2f}" for b in COST_B] == ["7.20", "1.00", "0.34", "0.23"]
assert [round(tps(b)) for b in COST_B] == [154, 1104, 3262, 4838]
assert [round(1 / cost_step(b)) for b in COST_B] == [154, 138, 102, 76]
assert round(usd_per_m(1) / usd_per_m(64)) == 31

# the catch: every student's KV cache
KV_TOK = kv_bytes_per_token()
KV_SEQ = CTX * KV_TOK
KV64 = 64 * KV_SEQ
MAX_B = int((GPU["memory"] - weight_bytes()) // KV_SEQ)
assert KV_TOK == 128 * 2 ** 10 and KV_SEQ == 256 * 2 ** 20 and KV64 == 16 * 2 ** 30
assert round(KV64 / GB, 1) == 17.2 and KV64 > weight_bytes() and MAX_B == 238 and MAX_B < RIDGE
READ64 = (weight_bytes() + KV64) / GB
assert round(READ64) == 33

USER = recolored("user", REQUEST, PINK_FILL, prefix="ep04")

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


def row(label, value, color, y, xl, xr, size=22):
    """A readout row: a MUTED label on the left, a coloured value right-aligned, on one baseline."""
    lab = left_at(code(label, size, MUTED), xl, y)
    val = right_at(code(value, size, color), xr, y)
    sit(val, baseline(lab))
    return lab, val


def swap_value(old, value, color, size=22):
    """A new value Text in the same place (right edge and baseline) as `old`."""
    new = right_at(code(value, size, color), old.get_right()[0], old.get_y())
    return sit(new, baseline(old))


def counter(value, label, color, x, y, size=48):
    num = title_text(value, size, color).move_to([x, y, 0])
    lab = text(label, 22, MUTED).next_to(num, m.DOWN, buff=0.18)
    lab.set_x(x)
    return num, lab


def bar_rect(width, height, color, x_left, y):
    return m.Rectangle(width=width, height=height).set_stroke(width=0).set_fill(color, 1).move_to(
        [x_left + width / 2, y, 0])


class LogChart(m.VGroup):
    """Tokens/s against batch size, both on log axes (the shape of a roofline)."""

    def __init__(self, origin=(-5.6, -2.45), width=7.0, height=4.2, xd=(0, 3.3), yd=(1, 5)):
        super().__init__()
        self.o, self.w, self.h, self.xd, self.yd = np.array([*origin, 0.0]), width, height, xd, yd
        x_axis = m.Arrow(self.o, self.o + [width + 0.45, 0, 0], buff=0, color=FAINT, stroke_width=3,
                         max_tip_length_to_length_ratio=0.04)
        y_axis = m.Arrow(self.o, self.o + [0, height + 0.55, 0], buff=0, color=FAINT, stroke_width=3,
                         max_tip_length_to_length_ratio=0.05)
        self.axes = m.VGroup(x_axis, y_axis)
        ticks = m.VGroup()
        for b, s in ((1, "1"), (10, "10"), (100, "100"), (1000, "1k")):
            p = self.p(b, 10 ** yd[0])
            ticks.add(m.Line(p, p + [0, -0.08, 0]).set_stroke(FAINT, 2))
            ticks.add(code(s, 20, MUTED).next_to(p, m.DOWN, buff=0.18))
        for v, s in ((100, "100"), (1e3, "1k"), (1e4, "10k"), (1e5, "100k")):
            p = self.p(1, v)
            ticks.add(m.Line(p, p + [-0.08, 0, 0]).set_stroke(FAINT, 2))
            ticks.add(right_at(code(s, 20, MUTED), p[0] - 0.15, p[1]))
        self.ticks = ticks
        self.x_label = text("batch size (students in one step)", 22, MUTED).move_to(
            [self.o[0] + width / 2, self.o[1] - 0.78, 0])
        self.y_label = left_at(text("tokens per second", 22, MUTED), self.o[0] + 0.22, self.o[1] + height + 0.35)
        self.add(self.axes, ticks, self.x_label, self.y_label)

    def p(self, b, v):
        fx = (np.log10(b) - self.xd[0]) / (self.xd[1] - self.xd[0])
        fy = (np.log10(v) - self.yd[0]) / (self.yd[1] - self.yd[0])
        return self.o + np.array([fx * self.w, fy * self.h, 0])

    def curve(self, f, a, b, color, n=48):
        pts = [self.p(x, f(x)) for x in np.logspace(np.log10(a), np.log10(b), n)]
        return m.VMobject().set_points_as_corners(pts).set_stroke(color, 5)


class Panels(m.VGroup):
    """Two linear charts sharing a batch axis 0-17: total tokens/s on top, per student below."""

    X0, W, XMAX = -5.85, 7.4, 17

    def __init__(self, top=(-0.4, 2.6, 500), bottom=(-2.85, 1.6, 120)):
        super().__init__()
        self.top, self.bottom = top, bottom
        parts = m.VGroup()
        for (y0, h, vmax), steps in ((top, (100, 200, 300, 400)), (bottom, (50, 100))):
            o = np.array([self.X0, y0, 0])
            parts.add(m.Arrow(o, o + [self.W + 0.35, 0, 0], buff=0, color=FAINT, stroke_width=3,
                              max_tip_length_to_length_ratio=0.04))
            parts.add(m.Arrow(o, o + [0, h + 0.3, 0], buff=0, color=FAINT, stroke_width=3,
                              max_tip_length_to_length_ratio=0.1))
            for v in steps:
                y = y0 + v / vmax * h
                parts.add(m.Line([self.X0 - 0.08, y, 0], [self.X0, y, 0]).set_stroke(FAINT, 2))
                parts.add(right_at(code(str(v), 20, MUTED), self.X0 - 0.15, y))
            for b in (1, 4, 8, 12, 16):
                x = self.x(b)
                parts.add(m.Line([x, y0, 0], [x, y0 - 0.08, 0]).set_stroke(FAINT, 2))
        y0b = bottom[0]
        labels = m.VGroup(*[code(str(b), 20, MUTED).move_to([self.x(b), y0b - 0.3, 0]) for b in (1, 4, 8, 12, 16)])
        on_baseline(labels)
        unit = left_at(code("batch", 20, MUTED), self.x(16) + 0.45, labels[-1].get_y())
        sit(unit, baseline(labels[-1]))
        self.t_label = left_at(text("total tokens per second", 22, TOKEN), self.X0 + 0.22, top[0] + top[1] + 0.25)
        self.b_label = left_at(text("tokens per second, per student", 22, REQUEST), self.X0 + 0.22,
                               bottom[0] + bottom[1] + 0.22)
        self.frame = m.VGroup(parts, labels, unit)
        self.add(self.frame, self.t_label, self.b_label)

    def x(self, b):
        return self.X0 + b / self.XMAX * self.W

    def pt(self, b, v, top=True):
        y0, h, vmax = self.top if top else self.bottom
        return np.array([self.x(b), y0 + v / vmax * h, 0])

    def series(self, run, top, color, bs=BS):
        """Dots and joining segments for one measured series (each dot animates with the segment to it)."""
        f = agg if top else per
        dots = [m.Dot(self.pt(b, f(run, b), top), radius=0.075, color=color).set_z_index(3) for b in bs]
        segs = [m.Line(self.pt(a, f(run, a), top), self.pt(b, f(run, b), top)).set_stroke(color, 3)
                for a, b in zip(bs, bs[1:])]
        return dots, segs

    def ceiling(self, rate, color=MEMORY):
        """The memory ceiling B × (bandwidth ÷ bytes), cut where it leaves the top chart."""
        y0, h, vmax = self.top
        b_end = min(vmax / rate, 16.5)
        return m.DashedLine(self.pt(1, rate), self.pt(b_end, rate * b_end), dash_length=0.12).set_stroke(color, 2.5)


# ---------------------------------------------------------------------------- the episode


class Ep04TheBatchingFreeLunch(Episode):
    number = 4

    def lesson(self):
        self.hook()
        self.sweep()
        self.curve()
        self.measured_bf16()
        self.measured_4bit()
        self.cost()
        self.catch()
        self.recap()

    # ------------------------------------------------------------------ 1 · hook

    def hook(self):
        self.next_section(
            "1 · One student, 16 GB",
            notes=f"Episode 2's lesson in one line: at batch 1 every token reads all {W_GB:.2f} GB of Owl-8B's BF16 "
                  f"weights. On an H100 SXM (NVIDIA spec, dated Oct 2026: 3.35 TB/s, 989 TFLOPS dense BF16) that is "
                  f"{STEP1 * 1e3:.2f} ms of memory traffic (weight_bytes ÷ bandwidth) against {MATH1 * 1e6:.1f} µs of "
                  f"math (2 × 8.03 B FLOPs ÷ peak). Bars to scale. The 64 students are the episode's question.")
        y = 1.25
        mem = card(3.7, 1.9, MEMORY, CARD).move_to([-4.65, y, 0])
        mem_cap = code("memory", 20, MEMORY).move_to(mem.get_corner(m.UL) + np.array([0.28, -0.25, 0]), aligned_edge=m.UL)
        wblock = m.RoundedRectangle(corner_radius=0.1, width=3.0, height=0.72).set_stroke(WEIGHTS, 2.5).set_fill(BLUE_FILL, 1)
        wblock.move_to(mem).shift(0.22 * m.DOWN)
        wlab = code("weights · 16 GB", 22, INK).move_to(wblock)
        alu = card(2.3, 1.9, COMPUTE, CARD).move_to([0.15, y, 0])
        alu_cap = code("math units", 20, COMPUTE).move_to(alu.get_corner(m.UL) + np.array([0.28, -0.25, 0]), aligned_edge=m.UL)
        cells = cell_grid(2, 5, size=0.26, gap=0.09, color=COMPUTE, fill=AMBER_FILL).move_to(alu).shift(0.22 * m.DOWN)
        pipe = arrow(mem.get_right(), alu.get_left(), MEMORY)
        pipe_lab = code("16 GB", 20, MEMORY).next_to(pipe, m.UP, buff=0.14)
        out = arrow(alu.get_right(), [2.05, y, 0], TOKEN)
        tok = token_row(["cells"], size=22)[0].move_to([2.85, y, 0])
        mascot = owl(0.55).move_to([4.55, y, 0])
        machine = m.VGroup(mem, mem_cap, wblock, wlab, alu, alu_cap, cells)

        span = 9.0
        lab_m = right_at(text("math units busy", 22, MUTED), -3.3, -0.85)
        bar_m = bar_rect(max(span * MATH1 / STEP1, 0.035), 0.36, COMPUTE, -3.0, -0.85)
        val_m = code("16 µs", 22, COMPUTE).next_to(bar_m, m.RIGHT, buff=0.2)
        lab_w = right_at(text("waiting for memory", 22, MUTED), -3.3, -1.65)
        bar_w = bar_rect(span, 0.36, MEMORY, -3.0, -1.65)
        val_w = code("4.8 ms", 22, MEMORY).next_to(bar_w, m.DOWN, buff=0.16).align_to(bar_w, m.RIGHT)
        note = left_at(cite("one token · Owl-8B · H100 · batch 1"), -3.0, -2.2)
        bars = m.VGroup(lab_m, bar_m, val_m, lab_w, bar_w, val_w, note)

        crowd = m.VGroup(*[icon(USER, 0.3) for _ in range(64)]).arrange_in_grid(rows=4, cols=16, buff=(0.1, 0.1))
        crowd.move_to([0.15, -1.75, 0])
        ask = arrow([0.15, crowd.get_top()[1], 0], [0.15, alu.get_bottom()[1], 0], REQUEST)
        toks = m.VGroup(*[m.RoundedRectangle(corner_radius=0.03, width=0.17, height=0.17).set_stroke(TOKEN, 1.5).set_fill(GREEN_FILL, 1)
                          for _ in range(64)]).arrange_in_grid(rows=8, cols=8, buff=0.05).move_to([2.9, y, 0])
        toks_lab = code("64 tokens", 20, TOKEN).next_to(toks, m.DOWN, buff=0.18)
        claim = title_text("64 students, one trip through the weights", 40, INK,
                           t2c={"64 students": REQUEST, "one trip through the weights": WEIGHTS}).move_to([0, -3.2, 0])

        self.say(
            "[In episode two, every token of Owl's answer] [hauled all sixteen gigabytes of weights through the chip.]",
            m.AnimationGroup(self.new_header("01 · the waste", "One student, sixteen gigabytes"),
                             m.FadeIn(machine, shift=0.15 * m.UP), m.GrowArrow(pipe),
                             m.FadeIn(mascot, shift=0.15 * m.UP)),
            m.AnimationGroup(pulse(pipe, MEMORY), m.FadeIn(pipe_lab),
                             m.Succession(m.Wait(0.4), m.GrowArrow(out), m.FadeIn(tok, shift=0.15 * m.RIGHT))),
        )
        self.say(
            "[The math took sixteen microseconds.] [Waiting for memory took almost five milliseconds.]",
            m.AnimationGroup(m.FadeIn(lab_m), m.GrowFromEdge(bar_m, m.LEFT), m.FadeIn(val_m)),
            m.AnimationGroup(m.FadeIn(lab_w), m.GrowFromEdge(bar_w, m.LEFT, run_time=1.6), m.FadeIn(val_w), m.FadeIn(note)),
        )
        self.say(
            "[So what happens when sixty-four students press Enter at the same moment?]",
            m.AnimationGroup(m.FadeOut(bars, shift=0.15 * m.DOWN),
                             m.LaggedStart(*[m.FadeIn(u, shift=0.1 * m.UP) for u in crowd], lag_ratio=0.02, run_time=1.6),
                             m.GrowArrow(ask)),
        )
        self.say(
            "[Here's the surprise]: [all sixty-four can share one trip through the weights], [for about the cost of one.]",
            m.Circumscribe(wblock, color=WEIGHTS, buff=0.08),
            m.AnimationGroup(pulse(pipe, MEMORY), pulse(ask, REQUEST), m.FadeOut(tok, shift=0.15 * m.RIGHT),
                             m.LaggedStart(*[m.FadeIn(t, scale=0.5) for t in toks], lag_ratio=0.02, run_time=1.4),
                             m.FadeIn(toks_lab)),
            m.FadeIn(claim, shift=0.15 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 2 · one sweep, many rows

    def sweep(self):
        self.next_section(
            "2 · One sweep, many rows",
            notes="Schematic: one decode step drawn as a single weight matrix (8 × 12 tiles stand for all 16 GB). "
                  "Each column is read from memory once and multiplied with every student's row (a matrix-vector "
                  "product at batch 1, matrix-matrix at batch B). BF16: 2 bytes per weight, 2 FLOPs (multiply + add) "
                  "per weight per row, so arithmetic intensity ≈ B FLOPs per byte (notes §2). The bytes stay at "
                  "16 GB per step; the math grows with B.")
        Y = 0.45
        W = cell_grid(8, 12, size=0.28, gap=0.06, color=WEIGHTS, fill=BLUE_FILL).move_to([-4.55, Y, 0])
        w_cap = left_at(code("weights · 16 GB · in memory", 20, WEIGHTS), W.get_left()[0], 2.25)
        panel = card(2.2, 3.1, COMPUTE, CARD).move_to([-0.4, Y, 0])
        p_cap = code("math units", 20, COMPUTE).move_to([-0.4, 2.25, 0])
        sit(p_cap, baseline(w_cap))
        out_x0 = 1.55
        o_cap = left_at(code("outputs", 20, TOKEN), out_x0, 2.25)
        sit(o_cap, baseline(w_cap))
        arr_in = arrow([W.get_right()[0], Y, 0], [panel.get_left()[0], Y, 0], MEMORY)
        arr_out = arrow([panel.get_right()[0], Y, 0], [out_x0, Y, 0], TOKEN)

        def out_x(j):
            return out_x0 + 0.14 + j * 0.34

        def rows(b):
            if b <= 4:
                h, pitch = 0.28, 0.42
            else:
                pitch = 2.66 / b
                h = pitch * 0.72
            ys = [Y + (b - 1) / 2 * pitch - i * pitch for i in range(b)]
            strips = m.VGroup(*[m.Rectangle(width=1.6, height=h).set_stroke(width=0).set_fill(REQUEST, 1)
                                .move_to([-0.4, yy, 0]) for yy in ys])
            return strips, ys, h

        def outs(b):
            _, ys, h = rows(b)
            return m.VGroup(*[
                m.VGroup(*[m.Rectangle(width=0.28, height=h).set_stroke(width=0).set_fill(TOKEN, 1).move_to([out_x(j), yy, 0])
                           for yy in ys])
                for j in range(12)])

        def col_at(j):
            return np.array([W[0][j].get_x(), W.get_y(), 0])

        r1, r2, r4, r64 = rows(1)[0], rows(2)[0], rows(4)[0], rows(64)[0]
        o1, o64 = outs(1), outs(64)
        blabs = {b: code(f"batch = {b}", 22, REQUEST).move_to([-0.4, -1.42, 0]) for b in (1, 2, 4, 64)}
        hl = m.RoundedRectangle(corner_radius=0.06, width=0.42, height=W.height + 0.18).set_stroke(MEMORY, 3.5)
        hl.set_fill(BG, 0).move_to(col_at(0)).set_z_index(2)
        colc = m.VGroup(*[W[i][0].copy() for i in range(8)]).move_to([-0.4, Y, 0]).set_z_index(3)
        one_tok = code("one output row: one token", 20, TOKEN).move_to([out_x0 + 2.0, Y - 0.55, 0])

        n_bytes, l_bytes = counter("16 GB", "weights read per step", MEMORY, -4.55, -2.35)
        n_int, l_int = counter("1", "FLOPs per byte", COMPUTE, -0.4, -2.35)
        n_tok, l_tok = counter("1", "tokens per step", TOKEN, 3.56, -2.35)
        for n in (n_int, n_tok):
            n.align_to(n_bytes, m.DOWN)
        for lab in (l_int, l_tok):
            lab.align_to(l_bytes, m.UP)

        def steps(cells, js, run_time=0.22):
            return [m.AnimationGroup(hl.animate.move_to(col_at(j)), pulse(arr_in, MEMORY), m.FadeIn(cells[j]),
                                     run_time=run_time) for j in js]

        self.say(
            "[Here is one decode step, shrunk to a single matrix]: [the weights, waiting in memory], "
            "[and one row of numbers for one student.]",
            m.AnimationGroup(self.new_header("02 · one sweep", "Read once, use many times"),
                             m.FadeIn(panel), m.FadeIn(p_cap), m.GrowArrow(arr_in)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(r) for r in W], lag_ratio=0.08), m.FadeIn(w_cap)),
            m.AnimationGroup(m.FadeIn(r1, shift=0.1 * m.UP), m.FadeIn(blabs[1]), m.GrowArrow(arr_out), m.FadeIn(o_cap)),
        )
        self.say(
            "[Each column of weights is read from memory], [multiplied with the row], [and dropped again.]",
            m.AnimationGroup(m.FadeIn(hl), pulse(arr_in, MEMORY), m.FadeIn(colc, target_position=col_at(0))),
            m.AnimationGroup(m.Indicate(r1, color=COMPUTE, scale_factor=1.06), m.FadeIn(o1[0], shift=0.1 * m.RIGHT)),
            m.FadeOut(colc, scale=0.6),
        )
        self.say(
            "[Column after column, the whole matrix streams through]: [one pass gives one output row], [and one token.]",
            m.Succession(*steps(o1, range(1, 12))),
            m.Circumscribe(m.VGroup(*o1), color=TOKEN, buff=0.08),
            m.FadeIn(one_tok, shift=0.1 * m.UP),
        )
        self.say(
            "[Each weight is two bytes, used for one multiply and one add]: [one FLOP per byte of traffic.]",
            m.AnimationGroup(m.FadeIn(n_bytes, shift=0.1 * m.UP), m.FadeIn(l_bytes)),
            m.AnimationGroup(m.FadeIn(n_int, shift=0.1 * m.UP), m.FadeIn(l_int), m.FadeIn(n_tok, shift=0.1 * m.UP), m.FadeIn(l_tok)),
        )
        self.say(
            "[Now a second student joins], [then four], [then sixty-four.]",
            m.AnimationGroup(*[m.FadeOut(c) for c in o1], m.FadeOut(one_tok), hl.animate.move_to(col_at(0)),
                             m.ReplacementTransform(r1, r2), m.Transform(blabs[1], blabs[2])),
            m.AnimationGroup(m.ReplacementTransform(r2, r4), m.Transform(blabs[1], blabs[4])),
            m.AnimationGroup(m.ReplacementTransform(r4, r64), m.Transform(blabs[1], blabs[64])),
        )
        last_col = m.VGroup(*[W[i][11] for i in range(8)])
        self.say(
            "[The sweep is exactly the same, column after column]: [each one read once], [and used sixty-four times.]",
            m.Succession(*steps(o64, range(12), run_time=0.21)),
            m.Circumscribe(last_col, color=MEMORY, buff=0.1),
            m.AnimationGroup(m.Circumscribe(o64[11], color=TOKEN, buff=0.08), m.Indicate(r64, color=COMPUTE, scale_factor=1.03)),
        )
        self.say(
            "[Sixty-four tokens from one trip through the weights]: [still sixteen gigabytes read], [but sixty-four FLOPs per byte.]",
            m.Transform(n_tok, title_text("64", 48, TOKEN).move_to(n_tok)),
            m.Circumscribe(m.VGroup(n_bytes, l_bytes), color=MEMORY, buff=0.12),
            m.Transform(n_int, title_text("64", 48, COMPUTE).move_to(n_int)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 3 · the batch curve (H100, theory)

    def curve(self):
        self.next_section(
            "3 · The batch curve",
            notes=f"owl_reference.step_time(batch=B) on an H100 SXM (3.35 TB/s, 989 TFLOPS dense BF16; dated Oct 2026), "
                  f"weights only (context 0, the KV cache comes at the end): step = max(16.06 GB ÷ bandwidth, B × 16 GFLOP ÷ peak). "
                  f"Batch 1: {STEP1 * 1e3:.2f} ms, {each(1):.0f} tok/s. Batch 64: same step, {total(64):,.0f} tok/s total. "
                  f"Ridge = peak ÷ bandwidth = {RIDGE:.0f} FLOPs/byte; for BF16 decode intensity ≈ B, so the critical batch "
                  f"≈ {RIDGE:.0f} (JAX scaling book: 'about 280 tokens' for an H100; kipply 2022: 208 for an A100-40GB). "
                  f"Batch 1,000: {step_time(batch=1000) * 1e3:.1f} ms, {total(1000):,.0f} tok/s total, {each(1000):.0f} each. "
                  "On log-log axes, total tok/s × 2P = attained FLOP/s and B = intensity, so the batch curve IS the roofline.")
        src = ("def step_time(batch=1, context=0):\n"
               "    weights = weight_bytes()          # read once\n"
               "    kv = batch * context * kv_bytes_per_token()\n"
               "    memory = (weights + kv) / GPU[\"bandwidth\"]\n"
               "    compute = batch * flops_per_token() / GPU[\"peak\"]\n"
               "    return max(memory, compute)")
        block = code_block(src, 20, title="owl_reference.py · simplified").move_to([0, 0.2, 0])
        block[1].set_z_index(2)
        body = block[1]

        def line_hl(key, name, color):
            """A gutter mark beside one line, and its variable name in the line's colour."""
            g = glyphs(body, key)
            mark = m.RoundedRectangle(corner_radius=0.03, width=0.08, height=g.height).set_stroke(width=0).set_fill(color, 1)
            mark.move_to([block[0].get_left()[0] + 0.17, g.get_y(), 0])
            return m.AnimationGroup(m.FadeIn(mark, shift=0.1 * m.RIGHT), glyphs(body, name).animate.set_color(color)), mark

        hl_mem, mark_mem = line_hl('memory = (weights + kv) / GPU["bandwidth"]', "memory", MEMORY)
        hl_cmp, mark_cmp = line_hl('compute = batch * flops_per_token() / GPU["peak"]', "compute", COMPUTE)

        ch = LogChart()
        xl, xr = 2.75, 6.2
        box = card(4.05, 2.45, EDGE, CARD, radius=0.16, stroke=2).move_to([4.47, 0.55, 0])
        title = text("H100 · Owl-8B · weights only", 20, MUTED).next_to(box, m.UP, buff=0.18)
        spec = [("batch", "1", REQUEST), ("step", f"{STEP1 * 1e3:.1f} ms", INK),
                ("total", f"{total(1):,.0f} tok/s", TOKEN), ("each", f"{each(1):,.0f} tok/s", REQUEST)]
        labs, vals = [], []
        for (lab, v, c), yy in zip(spec, (1.32, 0.82, 0.32, -0.18)):
            a, b = row(lab, v, c, yy, xl, xr)
            labs.append(a)
            vals.append(b)
        readout = m.VGroup(box, title, *labs)

        stops = [1, 8, 64, RIDGE, 1000]
        targets = {}
        for b in stops[1:]:  # built now, while every value still has its first shape
            new = [f"{b:,.0f}", f"{step_time(batch=b) * 1e3:.1f} ms", f"{total(b):,.0f} tok/s", f"{each(b):,.0f} tok/s"]
            targets[b] = [swap_value(v, t, c) for v, t, c in zip(vals, new, (REQUEST, INK, TOKEN, REQUEST))]

        def readout_to(b):
            return [m.Transform(v, t) for v, t in zip(vals, targets[b])]

        dot_t = m.Dot(ch.p(1, total(1)), radius=0.1, color=TOKEN).set_z_index(4)
        dot_e = m.Dot(ch.p(1, each(1)), radius=0.1, color=REQUEST).set_z_index(3)
        segs_t = [ch.curve(total, a, b, TOKEN) for a, b in zip(stops, stops[1:])]
        segs_e = [ch.curve(each, a, b, REQUEST) for a, b in zip(stops, stops[1:])]

        def stage(k, run_time=1.2):
            b = stops[k + 1]
            return m.AnimationGroup(m.Create(segs_t[k]), m.Create(segs_e[k]),
                                    dot_t.animate.move_to(ch.p(b, total(b))), dot_e.animate.move_to(ch.p(b, each(b))),
                                    *readout_to(b), run_time=run_time)

        lab_total = text("total", 22, TOKEN).move_to([-3.95, 0.4, 0])
        lab_each = text("per student", 22, REQUEST).move_to([-4.2, -1.45, 0])
        bend = ch.p(RIDGE, total(RIDGE))
        ridge_line = m.DashedLine([bend[0], ch.o[1], 0], [bend[0], bend[1] + 0.3, 0], dash_length=0.12).set_stroke(MUTED, 2)
        ridge_lab = m.VGroup(code("critical batch", 20, COMPUTE), code(f"≈ ridge ≈ {RIDGE:.0f}", 20, COMPUTE)).arrange(
            m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        left_at(ridge_lab, bend[0] + 0.2, 0.15)
        jax = m.VGroup(cite("JAX scaling book: “about 280”"), cite("for an H100")).arrange(
            m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        left_at(jax, 2.45, -1.2)
        lab_mem = text("memory-bound", 22, MEMORY).move_to([-1.6, -0.3, 0])
        lab_cmp = left_at(text("compute-bound", 22, COMPUTE), -0.15, 1.15)
        same = pill("the roofline in disguise", COMPUTE, AMBER_FILL, 22).move_to([4.47, -2.45, 0])

        self.say(
            "[In our reference code, a step takes the slower of two times]: [reading the weights], [or doing every student's math.]",
            m.AnimationGroup(self.new_header("03 · the batch curve", "Rising, then bending at the ridge"),
                             m.FadeIn(block, shift=0.15 * m.UP)),
            hl_mem,
            hl_cmp,
        )
        self.say(
            "[On an H100 at batch one], [that's four point eight milliseconds a step]: [two hundred and nine tokens per second.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(block, mark_mem, mark_cmp), shift=0.2 * m.UP),
                             m.FadeIn(ch), m.FadeIn(readout, shift=0.15 * m.LEFT)),
            m.AnimationGroup(m.FadeIn(vals[0]), m.FadeIn(vals[1])),
            m.AnimationGroup(m.FadeIn(vals[2]), m.FadeIn(vals[3]), m.FadeIn(dot_e, scale=0.5), m.FadeIn(dot_t, scale=0.5)),
        )
        self.say(
            "[At batch eight, then sixty-four, the step time doesn't move]: [so the total grows sixty-four times], "
            "[and each student keeps two hundred and nine.]",
            m.Succession(stage(0), stage(1), m.Circumscribe(vals[1], color=INK, buff=0.08)),
            m.AnimationGroup(m.Circumscribe(vals[2], color=TOKEN, buff=0.08), m.FadeIn(lab_total)),
            m.AnimationGroup(m.Circumscribe(vals[3], color=REQUEST, buff=0.08), m.FadeIn(lab_each)),
        )
        self.say(
            "[The free lunch lasts until the ridge]: [about two hundred and ninety-five students], "
            "[where the math takes as long as the memory.]",
            m.AnimationGroup(stage(2, run_time=1.4), m.Create(ridge_line)),
            m.AnimationGroup(m.FadeIn(ridge_lab, shift=0.1 * m.DOWN), m.FadeIn(jax)),
            m.Flash(bend, color=COMPUTE, flash_radius=0.3, line_length=0.25),
        )
        self.say(
            "[Past the ridge, the math units are full]: [the total stops growing], [and every extra student slows the others down.]",
            stage(3, run_time=1.4),
            m.Circumscribe(vals[2], color=TOKEN, buff=0.08),
            m.Circumscribe(vals[3], color=REQUEST, buff=0.08),
        )
        self.say(
            "[Rising, then bending]: [that's the roofline from episode two], [drawn in tokens per second.]",
            m.Flash(bend, color=COMPUTE, flash_radius=0.3, line_length=0.25),
            m.AnimationGroup(*[s.animate.set_stroke(MEMORY) for s in segs_t[:3]], segs_t[3].animate.set_stroke(COMPUTE),
                             m.FadeIn(lab_mem), m.FadeIn(lab_cmp)),
            m.FadeIn(same, scale=0.9),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 4 · measured on this Mac (BF16)

    def measured_bf16(self):
        self.next_section(
            "4 · Measured: batch 1 to 16",
            notes=f"research/measurements.md §8 (measured 2026-10-24 on this Mac Studio, M4 Max 546 GB/s, mlx-lm 0.32.2 / "
                  f"MLX 0.32.4, BatchGenerator, 256-token prompts, 128 generated tokens, EOS off, median of 3). Llama-3.1-8B "
                  f"BF16 total tok/s at B = 1…16: {', '.join(f'{agg(BF, b):.0f}' for b in BS)}; per sequence "
                  f"{', '.join(f'{per(BF, b):.1f}' for b in BS)}. Batch-1 ceiling 546 ÷ 16.06 = {CEIL_BF:.1f} tok/s. "
                  f"1 → 8: ×{agg(BF, 8) / agg(BF, 1):.1f} total, −{100 * (1 - per(BF, 8) / per(BF, 1)):.0f}% each. "
                  f"Mac ridge ≈ {MAC_RIDGE:.1f} FLOPs/byte (measured 15.2 TFLOPS bf16 matmul ÷ 546 GB/s). The dip at 16 is "
                  f"confirmed by the stock tool (mlx_lm.benchmark: {STOCK12:.0f} tok/s at 12, {STOCK16:.0f} at 16). "
                  "Explanation = the measurement agent's reading, NOT verified in the MLX source: MLX switches from a "
                  "mat-vec kernel at small B to tiled matrix-matrix kernels that are inefficient at these small row "
                  f"counts. Bigger batches: {agg(BF, 32):.0f} tok/s at 32, {agg(BF, 64):.0f} at 64.")
        P = self.panels = Panels()
        bf_t, bf_ts = P.series(BF, True, TOKEN)
        bf_e, bf_es = P.series(BF, False, REQUEST)
        ceil = P.ceiling(CEIL_BF)
        ceil_lab = right_at(code(f"ceiling · B × {CEIL_BF:.0f}", 20, MEMORY), -1.6, 1.75)

        X = 2.6
        name = left_at(text("Owl-8B · BF16", 30, INK, weight=m.BOLD), X, 2.2)
        size = left_at(code("16 GB of weights", 20, WEIGHTS), X, 1.72)
        src = left_at(cite("measured on this Mac · mlx-lm"), X, 1.3)
        xr = 6.5
        r1 = row("batch 1", f"{agg(BF, 1):.0f} tok/s", TOKEN, 0.6, X, xr)
        r2 = row("ceiling", f"{CEIL_BF:.0f} tok/s", MEMORY, 0.15, X, xr)
        r2b = left_at(code("546 GB/s ÷ 16 GB", 20, MUTED), X, -0.25)
        r3 = row("batch 8", f"{agg(BF, 8):.0f} tok/s", TOKEN, -0.85, X, xr)
        r4 = row("each", f"{per(BF, 8):.1f} tok/s", REQUEST, -1.3, X, xr)
        r5 = left_at(text(f"×{agg(BF, 8) / agg(BF, 1):.1f} total · −{100 * (1 - per(BF, 8) / per(BF, 1)):.0f}% each", 22, INK), X, -1.8)
        rows_ = m.VGroup(*r1, *r2, r2b, *r3, *r4, r5)

        i12, i16 = BS.index(12), BS.index(16)
        lab12 = code(f"{agg(BF, 12):.0f}", 20, TOKEN).next_to(bf_t[i12], m.DOWN, buff=0.17)
        lab16 = code(f"{agg(BF, 16):.0f}", 20, RED).next_to(bf_t[i16], m.DOWN, buff=0.32)
        ring = m.Circle(radius=0.22).set_stroke(RED, 3).move_to(bf_t[i16])
        ridge_note = right_at(code(f"Mac ridge ≈ {MAC_RIDGE:.0f} →", 20, COMPUTE), 1.55, -0.1)

        why = m.VGroup(text("Around 16 rows, MLX switches", 22, INK), text("from a matrix-vector kernel", 22, INK),
                       text("to tiled matrix kernels, which", 22, INK), text("are inefficient at so few rows.", 22, INK)
                       ).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
        reading = pill("our reading · not verified", MUTED, CARD, 20)
        expl = m.VGroup(reading, why).arrange(m.DOWN, buff=0.25, aligned_edge=m.LEFT)
        expl.move_to([X, 0.9, 0], aligned_edge=m.UL)
        stock = m.VGroup(text("the stock mlx_lm.benchmark", 20, MUTED),
                         text(f"shows it too: {STOCK12:.0f} → {STOCK16:.0f} tok/s", 20, MUTED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        stock.move_to([X, -1.6, 0], aligned_edge=m.UL)
        knee = left_at(text("the kernels set this knee", 24, COMPUTE, weight=m.BOLD), X, -2.45)
        big = m.VGroup(*row("batch 32", f"{agg(BF, 32):.0f} tok/s", TOKEN, 0.6, X, xr),
                       *row("batch 64", f"{agg(BF, 64):.0f} tok/s", TOKEN, 0.15, X, xr))

        def grow(k, dots, segs):
            """Dot k and the segment leading to it."""
            parts = [m.FadeIn(dots[k], scale=0.5)]
            if k:
                parts.insert(0, m.Create(segs[k - 1]))
            return m.Succession(*parts) if k else parts[0]

        def upto(ks, dots, segs):
            return m.Succession(*[grow(k, dots, segs) for k in ks], run_time=0.45 * len(ks))

        self.say(
            "[Now the real thing]: [Owl's eight-billion model in sixteen-bit, on this Mac], [measured with MLX from batch one to sixteen.]",
            m.AnimationGroup(self.new_header("04 · measured on this Mac", "Batch 1 to 16, for real"), m.FadeIn(P.frame)),
            m.AnimationGroup(m.FadeIn(name, shift=0.1 * m.UP), m.FadeIn(size), m.FadeIn(src)),
            m.FadeIn(P.t_label),
        )
        self.say(
            "[Batch one: thirty-one tokens per second], [against a ceiling of thirty-four.]",
            m.AnimationGroup(grow(0, bf_t, bf_ts), m.FadeIn(m.VGroup(*r1))),
            m.AnimationGroup(m.Create(ceil), m.FadeIn(ceil_lab), m.FadeIn(m.VGroup(*r2)), m.FadeIn(r2b)),
        )
        self.say(
            "[Two students: sixty-three.] [Four: a hundred and twenty-two.] [Eight: two hundred and thirty-five], [right under the ceiling.]",
            grow(1, bf_t, bf_ts),
            upto([2, 3], bf_t, bf_ts),
            m.AnimationGroup(upto([4, 5], bf_t, bf_ts), m.FadeIn(m.VGroup(*r3))),
            m.Circumscribe(m.VGroup(*bf_t[:6]), color=MEMORY, buff=0.12),
        )
        self.say(
            "[And each student still gets twenty-nine tokens a second]: [seven point six times the total], [for five percent less each.]",
            m.AnimationGroup(m.FadeIn(P.b_label), upto(range(6), bf_e, bf_es), m.FadeIn(m.VGroup(*r4))),
            m.FadeIn(r5, shift=0.1 * m.UP),
            m.Circumscribe(m.VGroup(*bf_e[:6]), color=REQUEST, buff=0.12),
        )
        self.say(
            "[Then the curve stops being smooth]: [at twelve it flattens near three hundred], "
            "[and at sixteen it drops, to two hundred and forty-three.]",
            m.AnimationGroup(grow(6, bf_t, bf_ts), grow(6, bf_e, bf_es)),
            m.AnimationGroup(grow(7, bf_t, bf_ts), grow(7, bf_e, bf_es), m.FadeIn(lab12)),
            m.AnimationGroup(grow(8, bf_t, bf_ts), grow(8, bf_e, bf_es), m.FadeIn(lab16), m.Create(ring)),
        )
        self.say(
            "[The roofline puts this Mac's ridge near twenty-eight], [so this knee comes early.]",
            m.FadeIn(ridge_note, shift=0.15 * m.RIGHT),
            m.Circumscribe(m.VGroup(bf_t[i12], bf_t[i16], lab12, lab16), color=RED, buff=0.15),
        )
        self.say(
            "[Our reading, not checked in the MLX source]: [around sixteen rows, MLX switches to tiled matrix kernels], "
            "[which are slow with so few rows.]",
            m.AnimationGroup(m.FadeOut(rows_, shift=0.15 * m.DOWN), m.FadeIn(reading, shift=0.1 * m.UP)),
            m.FadeIn(why[:2]),
            m.FadeIn(why[2:]),
        )
        self.say(
            "[The stock benchmark tool shows the same dip]: [here, the kernels set the knee, not the roofline.]",
            m.FadeIn(stock),
            m.FadeIn(knee, shift=0.1 * m.UP),
        )
        self.say(
            "[And bigger batches recover]: [four hundred and fifteen at thirty-two], [six hundred and eighty-six at sixty-four.]",
            m.FadeOut(expl, shift=0.15 * m.UP),
            m.FadeIn(m.VGroup(*big[:2]), shift=0.1 * m.UP),
            m.FadeIn(m.VGroup(*big[2:]), shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.bf = dict(t=bf_t, ts=bf_ts, e=bf_e, es=bf_es, ceil=ceil, ceil_lab=ceil_lab, lab12=lab12, lab16=lab16,
                       ring=ring, name=name, size=size, src=src, expl=expl, stock=stock, knee=knee, big=big,
                       ridge_note=ridge_note)

    # ------------------------------------------------------------------ 5 · measured on this Mac (4-bit)

    def measured_4bit(self):
        self.next_section(
            "5 · Four bits: a shorter lunch",
            notes=f"Same run, Llama-3.1-8B-Instruct-4bit (MLX affine, group 64, {Q4_GB:.2f} GB; {BYTES_RATIO:.2f}× fewer bytes "
                  f"read per step than BF16). Total tok/s at B = 1…16: {', '.join(f'{agg(Q4, b):.0f}' for b in BS)}; per "
                  f"sequence {', '.join(f'{per(Q4, b):.1f}' for b in BS)}. Batch-1 ceiling 546 ÷ {Q4_GB:.2f} = {CEIL_Q4:.1f} tok/s. "
                  "The free lunch ends at B ≈ 2–3. Explanation = the measurement agent's interpretation, not verified: with "
                  "3.6× fewer bytes per step, the per-row dequantization work of the quantized mat-vec path dominates early, "
                  f"until a tiled quantized-matmul kernel takes over (step ≈ 45 ms from 16 to 32). At B = 32: {agg(Q4, 32):.0f} tok/s, "
                  f"{per(Q4, 32):.1f} each (and non-monotonic again: 563 at 48, 717 at 64).")
        P, bf = self.panels, self.bf
        q_t, q_ts = P.series(Q4, True, TOKEN)
        q_e, q_es = P.series(Q4, False, REQUEST)
        ceil = P.ceiling(CEIL_Q4)
        ceil_lab = left_at(code(f"ceiling · B × {CEIL_Q4:.0f}", 20, MEMORY), -4.0, 1.9)
        old = m.VGroup(*bf["t"], *bf["ts"], *bf["e"], *bf["es"], bf["ceil"])
        tag_bf = left_at(code("BF16", 20, FAINT), P.x(16) + 0.22, P.pt(16, agg(BF, 16))[1])
        tag_q = left_at(code("4-bit", 20, TOKEN), P.x(16) + 0.22, P.pt(16, agg(Q4, 16))[1])

        X, xr = 2.6, 6.5
        name = left_at(text("Owl-8B · 4-bit", 30, INK, weight=m.BOLD), X, 2.2)
        sit(name, baseline(bf["name"]))
        size = left_at(code(f"{Q4_GB:.1f} GB of weights", 20, WEIGHTS), X, 1.72)
        r1 = row("batch 1", f"{agg(Q4, 1):.0f} tok/s", TOKEN, 0.6, X, xr)
        r2 = row("ceiling", f"{CEIL_Q4:.0f} tok/s", MEMORY, 0.15, X, xr)
        why = m.VGroup(text("With so few bytes per step,", 22, INK), text("unpacking 4-bit weights row", 22, INK),
                       text("by row dominates early, until", 22, INK), text("a matrix kernel takes over.", 22, INK)
                       ).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
        reading = pill("our interpretation", MUTED, CARD, 20)
        expl = m.VGroup(reading, why).arrange(m.DOWN, buff=0.25, aligned_edge=m.LEFT)
        expl.move_to([X, -0.25, 0], aligned_edge=m.UL)
        big = m.VGroup(*row("batch 32", f"{agg(Q4, 32):.0f} tok/s", TOKEN, -0.55, X, xr),
                       *row("each", f"{per(Q4, 32):.0f} tok/s", REQUEST, -1.0, X, xr))

        gone = m.VGroup(bf["ceil_lab"], bf["lab12"], bf["lab16"], bf["ring"], bf["name"], bf["size"],
                        bf["stock"], bf["knee"], bf["big"])

        def grow(k, dots, segs):
            parts = [m.FadeIn(dots[k], scale=0.5)]
            if k:
                parts.insert(0, m.Create(segs[k - 1]))
            return m.Succession(*parts) if k else parts[0]

        def upto(ks, dots, segs):
            return m.Succession(*[grow(k, dots, segs) for k in ks], run_time=0.4 * len(ks))

        self.say(
            "[Four-bit weights are about three and a half times smaller]: [a hundred and one tokens a second at batch one], "
            "[out of a possible hundred and twenty-one.]",
            m.AnimationGroup(self.new_header("05 · four bits", "A shorter free lunch"), m.FadeOut(gone),
                             old.animate.set_color(FAINT), m.FadeIn(tag_bf), m.FadeIn(name, shift=0.1 * m.UP), m.FadeIn(size)),
            m.AnimationGroup(grow(0, q_t, q_ts), grow(0, q_e, q_es), m.FadeIn(m.VGroup(*r1))),
            m.AnimationGroup(m.Create(ceil), m.FadeIn(ceil_lab), m.FadeIn(m.VGroup(*r2))),
        )
        self.say(
            "[But its free lunch ends after about three students]: [a hundred and ninety at two], [two hundred and forty-eight at three], "
            "[then flat near three hundred up to twelve.]",
            m.Circumscribe(ceil, color=MEMORY, buff=0.1),
            grow(1, q_t, q_ts),
            grow(2, q_t, q_ts),
            upto([3, 4, 5, 6, 7], q_t, q_ts),
        )
        self.say(
            "[Per student, that's a slide from a hundred and one], [down to twenty-five.]",
            upto(range(4), q_e, q_es),
            upto([4, 5, 6, 7], q_e, q_es),
        )
        self.say(
            "[Our interpretation]: [with so few bytes per step, unpacking the four-bit weights row by row dominates early], "
            "[until a matrix kernel takes over at sixteen.]",
            m.FadeIn(reading, shift=0.1 * m.UP),
            m.FadeIn(why),
            m.AnimationGroup(grow(8, q_t, q_ts), grow(8, q_e, q_es), m.FadeIn(tag_q)),
        )
        self.say(
            "[At batch thirty-two it still serves six hundred and seventy-six tokens a second], [twenty-one for each student.]",
            m.AnimationGroup(m.FadeOut(expl, shift=0.15 * m.DOWN), m.FadeIn(big[:2], shift=0.1 * m.UP)),
            m.FadeIn(big[2:], shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 6 · cost per token

    def cost(self):
        self.next_section(
            "6 · Cost per million tokens",
            notes=f"Notes §15, recomputed: $ per M output tokens = ($/GPU-hour) ÷ (tokens/s × 3,600 ÷ 10⁶). Price: H100 SXM "
                  f"${PRICE}/GPU-hour, Lambda (read 2026-10-23) — DATED, October 2026 (RunPod $2.69–3.99). For example: "
                  f"Owl-8B BF16, 2,048-token chats (each step also reads every sequence's KV), 75% of peak bandwidth "
                  f"(step = step_time(batch=B, context=2048) ÷ 0.75). B = 1, 8, 32, 64 → "
                  f"{', '.join(f'{tps(b):,.0f}' for b in COST_B)} tok/s total, "
                  f"{', '.join(f'{1 / cost_step(b):.0f}' for b in COST_B)} each, "
                  f"{', '.join(f'${usd_per_m(b):.2f}' for b in COST_B)} per M tokens (×{usd_per_m(1) / usd_per_m(64):.0f} cheaper). "
                  "Real engines add overhead; LMSYS reports DeepSeek on H100s at $0.20/M output tokens.")
        X0 = -6.4
        price_card = card(4.75, 5.6, EDGE, CARD, radius=0.18, stroke=2).move_to([X0 + 2.375, -0.35, 0])
        kick = left_at(text("one H100, rented", 24, MUTED), X0 + 0.35, 2.05)
        price = left_at(title_text(f"${PRICE:.2f}", 64, INK), X0 + 0.35, 1.3)
        per_h = sit(code("per GPU-hour", 22, MUTED).next_to(price, m.RIGHT, buff=0.2), baseline(price))
        lam = left_at(cite("Lambda · H100 SXM · Oct 2026"), X0 + 0.35, 0.55)
        rule = m.Line([X0 + 0.35, 0.15, 0], [X0 + 4.4, 0.15, 0]).set_stroke(EDGE, 2)
        a1 = left_at(code("Owl-8B · BF16 · 16 GB", 20, WEIGHTS), X0 + 0.35, -0.25)
        a2 = left_at(code("2K-token chats", 20, KV), X0 + 0.35, -0.65)
        a3 = left_at(code("75% of peak bandwidth", 20, MEMORY), X0 + 0.35, -1.05)
        a4 = left_at(cite("for example"), X0 + 0.35, -1.45)
        formula = m.VGroup(code("$ per M tokens =", 20, INK), code("$/hour ÷ M tokens/hour", 20, INK)).arrange(
            m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        left_at(formula, X0 + 0.35, -2.45)
        assume = m.VGroup(rule, a1, a2, a3, a4)

        base, scale, pitch, x_first = -1.25, 2.85 / usd_per_m(1), 1.95, -0.35
        colors = {1: RED, 8: MUTED, 32: MUTED, 64: GREEN}
        fills = {1: RED_FILL, 8: CARD_HI, 32: CARD_HI, 64: GREEN_FILL}
        bars, vals, labs, totals, eaches = [], [], [], [], []
        for i, b in enumerate(COST_B):
            x = x_first + i * pitch
            h = max(usd_per_m(b) * scale, 0.04)
            r = m.Rectangle(width=1.0, height=h).set_stroke(colors[b], 2.5).set_fill(fills[b], 1).move_to([x, base + h / 2, 0])
            bars.append(r)
            vals.append(code(f"${usd_per_m(b):.2f}", 24, INK if b in (8, 32) else colors[b]).next_to(r, m.UP, buff=0.15))
            labs.append(code(f"batch {b}", 22, REQUEST).move_to([x, base - 0.32, 0]))
            totals.append(code(f"{tps(b):,.0f}", 22, TOKEN).move_to([x, base - 0.78, 0]))
            eaches.append(code(f"{1 / cost_step(b):.0f} each", 20, REQUEST).move_to([x, base - 1.18, 0]))
        on_baseline(labs)
        on_baseline(totals)
        on_baseline(eaches)
        axis = m.Line([x_first - 0.75, base, 0], [x_first + 3 * pitch + 0.75, base, 0]).set_stroke(FAINT, 2.5)
        y_cap = left_at(text("$ per million tokens", 22, MUTED), x_first - 0.75, 2.42)
        cheaper = title_text(f"{usd_per_m(1) / usd_per_m(64):.0f}× cheaper", 48, GREEN).move_to([3.4, 1.15, 0])
        reads = right_at(cite("far faster than anyone reads"), 6.4, eaches[-1].get_y() - 0.42)
        legend = left_at(text("tokens per second: total · each student", 20, MUTED,
                              t2c={"total": TOKEN, "each student": REQUEST}), x_first - 0.75, -3.25)

        def show(i):
            return m.AnimationGroup(m.GrowFromEdge(bars[i], m.DOWN), m.FadeIn(vals[i], shift=0.1 * m.UP),
                                    m.FadeIn(labs[i]), m.FadeIn(totals[i]))

        self.say(
            "[What does sharing do to the bill?] [Say an H100 rents for three dollars ninety-nine an hour], [Lambda's price this October.]",
            m.AnimationGroup(self.new_header("06 · cost per token", "The bill falls with the batch"), m.FadeIn(axis), m.FadeIn(y_cap)),
            m.AnimationGroup(m.FadeIn(price_card), m.FadeIn(kick), m.FadeIn(price, shift=0.1 * m.UP), m.FadeIn(per_h)),
            m.FadeIn(lam),
        )
        self.say(
            "[With two-thousand-token chats, batch one makes a hundred and fifty-four tokens a second]: [seven dollars twenty per million.]",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(p) for p in assume], lag_ratio=0.2), m.FadeIn(formula), m.FadeIn(legend)),
            show(0),
        )
        self.say(
            "[Batch eight costs a dollar.] [Batch thirty-two costs thirty-four cents.] [Batch sixty-four costs twenty-three cents]: [thirty-one times cheaper.]",
            show(1),
            show(2),
            show(3),
            m.FadeIn(cheaper, scale=0.85),
        )
        self.say(
            "[Same rental, same sixteen gigabytes a step], [and each student still gets seventy-six tokens a second], "
            "[far faster than anyone reads.]",
            m.Circumscribe(m.VGroup(price, per_h), color=MUTED, buff=0.12),
            m.LaggedStart(*[m.FadeIn(e) for e in eaches], lag_ratio=0.25),
            m.AnimationGroup(m.FadeIn(reads), m.Circumscribe(eaches[-1], color=REQUEST, buff=0.08)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 7 · the catch

    def catch(self):
        self.next_section(
            "7 · The catch",
            notes=f"KV per token = 2 × 32 layers × 8 KV heads × 128 × 2 B = {KV_TOK:,} B = 128 KiB (kv_bytes_per_token). "
                  f"At {CTX:,} tokens per chat: {KV_SEQ / 2 ** 20:.0f} MiB per student; 64 students = {KV64 / 2 ** 30:.0f} GiB "
                  f"= {KV64 / GB:.1f} GB, more than the {W_GB:.2f} GB of weights, and private to each sequence (not shared). "
                  f"Every step reads it: {READ64:.1f} GB per step at B = 64. An 80 GB H100 holds the weights plus "
                  f"{MAX_B} such students ((80 − 16.06) GB ÷ 256 MiB, ignoring activations and engine overhead; notes §15 say "
                  f"≈ 200 in practice) — below the ridge ({RIDGE:.0f}). Memory caps the batch before compute does. Module 2.")
        mb = MemoryBar(GPU["memory"], width=11.6, height=0.7, title="H100 · 80 GB of memory").move_to([0, 1.85, 0])
        sq = lambda c: m.Square(0.24).set_stroke(width=0).set_fill(c, 1)
        leg_w = m.VGroup(sq(WEIGHTS), code("weights · 16 GB · one copy, shared by everyone", 20, WEIGHTS)).arrange(m.RIGHT, buff=0.2)
        leg_k = m.VGroup(sq(KV), code("KV cache · one per student, never shared", 20, KV)).arrange(m.RIGHT, buff=0.2)
        left_at(leg_w, -5.8, 1.1)
        left_at(leg_k, -5.8, 0.68)
        f1 = code("128 KiB per token", 24, KV)
        f2 = code(f"× {CTX:,} tokens = 256 MiB per student", 24, INK, t2c={"256 MiB": KV})
        formula = m.VGroup(f1, f2).arrange(m.RIGHT, buff=0.25).move_to([0, -0.05, 0])
        sit(f2, baseline(f1))

        xs = (-4.2, 0, 4.2)
        n_b, l_b = counter("64", "students", REQUEST, xs[0], -1.1)
        n_k, l_k = counter(f"{KV64 / GB:.1f} GB", "of KV cache", KV, xs[1], -1.1)
        n_r, l_r = counter(f"{READ64:.0f} GB", "read every step", MEMORY, xs[2], -1.1)
        for n in (n_k, n_r):
            n.align_to(n_b, m.DOWN)
        for lab in (l_k, l_r):
            lab.align_to(l_b, m.UP)
        full_b = title_text(f"{MAX_B}", 48, REQUEST).move_to(n_b)
        full_k = title_text(f"{(GPU['memory'] - weight_bytes()) / GB:.0f} GB", 48, KV).move_to(n_k)
        full_r = title_text("80 GB", 48, MEMORY).move_to(n_r)
        verdict = pill(f"memory full at {MAX_B} students · the ridge is at {RIDGE:.0f}", RED, RED_FILL, 22).move_to([0, -2.45, 0])
        moral = title_text("Memory, not math, ends the free lunch", 40, INK, t2c={"Memory": KV}).move_to([0, -3.2, 0])

        self.say(
            "[Here's the catch.] [Every student also brings a KV cache]: [a hundred and twenty-eight kibibytes for each token of their chat.]",
            m.AnimationGroup(self.new_header("07 · the catch", "Every student brings a cache"), m.FadeIn(mb.track),
                             m.FadeIn(mb.caption), mb.set("weights", weight_bytes(), WEIGHTS), m.FadeIn(leg_w)),
            m.FadeIn(leg_k),
            m.FadeIn(f1, shift=0.1 * m.UP),
        )
        self.say(
            "[At two thousand tokens each], [sixty-four students hold seventeen gigabytes]: [more than the weights], "
            "[and none of it is shared.]",
            m.FadeIn(f2, shift=0.1 * m.LEFT),
            m.AnimationGroup(mb.set("kv", KV64, KV, run_time=1.2), m.FadeIn(n_b), m.FadeIn(l_b), m.FadeIn(n_k), m.FadeIn(l_k)),
            m.Circumscribe(mb.segments["kv"], color=KV, buff=0.08),
            m.Indicate(leg_k, color=KV, scale_factor=1.04),
        )
        self.say(
            "[Every step reads all of it], [and at two hundred and thirty-eight students the memory is full]: "
            "[before the ridge at two hundred and ninety-five.]",
            m.AnimationGroup(m.FadeIn(n_r), m.FadeIn(l_r), pulse(m.Line(mb.track.get_left(), mb.track.get_right()), MEMORY, 12)),
            m.AnimationGroup(mb.set("kv", MAX_B * KV_SEQ, KV, run_time=1.6), m.Transform(n_b, full_b), m.Transform(n_k, full_k),
                             m.Transform(n_r, full_r)),
            m.FadeIn(verdict, scale=0.9),
        )
        self.say(
            "[So memory, not math, ends the free lunch]: [and that's what module two is about.]",
            m.FadeIn(moral, shift=0.15 * m.UP),
            m.Circumscribe(mb.segments["kv"], color=KV, buff=0.08),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Share one weight sweep across the batch (bytes per step stay ~16 GB); fill the batch "
                                         "up to the ridge (≈ 295 on an H100, earlier on this Mac's kernels); make every "
                                         "student's KV cache fit (Module 2).")
        halo = glow(COMPUTE, radius=4.0)
        words = m.VGroup(*[title_text(w, 80, c) for w, c in (("Share", WEIGHTS), ("Fill", COMPUTE), ("Fit", KV))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("one weight sweep", "up to the ridge", "every KV cache")])
        slots = [max(w.width, s.width) for w, s in zip(words, subs)]
        x, gap = 0.0, 1.1
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
        tagline = text("One trip through the weights, a whole classroom served.", 30, MUTED).next_to(row_, m.DOWN, buff=0.75)

        self.say(
            "[To sum up]: [share], [fill], [fit.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[Share one trip through the weights, fill the batch up to the ridge, and make sure every cache fits.]",
            m.Write(tagline),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row_, tagline)], run_time=0.9)
