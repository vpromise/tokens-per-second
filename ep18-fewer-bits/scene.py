"""Tokens per Second · Episode 18 · Fewer Bits, Same Model (Module 5 · Fewer Bits, More GPUs)

Fewer bits per weight means fewer bytes per token, and in memory-bound decode, fewer bytes are less time.
Running example: Owl-8B (Llama-3.1-8B: 8.03 B parameters, 16.06 GB of BF16 weights) on this Mac Studio
(M4 Max, 546 GB/s), stored in 16 bits and in MLX's 8-bit format.

Sections: fewer bytes (Episode 2's tok/s ≈ bandwidth ÷ bytes: 546 GB/s ÷ 16.06 GB = 34 tok/s; 8 bits halve the
bytes, the ceiling doubles to 68) · one scale (a group of 8 weights quantized to INT8 step by step: max |w|
0.50, scale = 0.50 ÷ 127 = 0.00394, q = round(w / scale), 10 bytes instead of 16, error under half a step)
· zero points and groups (a lopsided group wastes 118 of 255 symmetric levels; a zero point maps 256 levels
onto [min, max]; MLX: one bf16 scale + one bf16 bias per 64 weights = 8.5 bits per weight = 8.53 GB = the
file on disk) · measured (BF16 31.0 of 34.0 tok/s, 8-bit 58.4 of 64.0: both 91%; the bits=8 model says 68;
1.88× faster for 1.88× fewer bytes; MBU 85.4% / 85.6%) · two kinds (weight-only W8A16 halves bytes: chat ×2;
a batch of 256 near the H100 ridge gains only 15%; W8A8 on FP8/INT8 tensor cores doubles the roof: ×2) ·
outliers (an activation channel ~100× the rest, in every token; one INT8 scale per token leaves ordinary
values on −1, 0, 1, off by 50% on average; LLM.int8() keeps those columns in 16 bits) · SmoothQuant (divide
the column by s = √(50 ÷ 0.5) = 10, multiply the weight row by 10: X·W unchanged; error 50% → 5%) · FP8
(E4M3: 4 exponent + 3 mantissa bits, max 448; E5M2 max 57,344 for gradients; with the same giant, 149 FP8
levels vs 3 INT8 levels cover the ordinary values, error 3%; Kurtic et al.: FP8 effectively lossless, INT8
1–3%) · recap (shrink, scale, smooth).

Sources (facts dated October 2026):
- owl_reference.py: OWL (Llama-3.1-8B config), HARDWARE (M4 Max 546 GB/s, Apple Mac Studio (2025) tech specs;
  H100 SXM 3.35 TB/s, 989 TFLOP/s BF16 dense, NVIDIA), weight_bytes, step_time, decode_ceiling, ridge,
  measured().
- research/measurements.md §1 / .json (this Mac, mlx-lm 0.32.2, 2026-10-24, median of 5 trials, 256 tokens,
  greedy): Meta-Llama-3.1-8B-Instruct bf16 31.00 tok/s, 8bit 58.42 tok/s; weights on disk 16.061 / 8.532 GB;
  MBU (bytes really read per step × tok/s ÷ 546 GB/s) 85.4% / 85.6%; MLX affine quantization, group size 64,
  bf16 scale + bias per group ("8-bit ≈ 8.5" bits per weight). Stock mlx_lm.generate cross-check: 31.01 / 58.39.
- MLX docs, mlx.core.quantize (read 2026-10-24): affine mode, per group: s = (max − min) / (2^b − 1),
  ŵ = round((w − min) / s), scales and biases (the group minimum) stored; default group size 64.
- research/notes.md §8: weight-only vs weight-activation quantization; group overhead; the worked example's
  eight weights [0.12, −0.50, 0.33, 0.05, −0.27, 0.48, −0.09, 0.21] (recomputed here at INT8: symmetric
  absmax, scale = max|w| / 127); LLM.int8() (Dettmers et al., NeurIPS 2022, arXiv 2208.07339: "isolates the
  outlier feature dimensions into a 16-bit matrix multiplication while still more than 99.9% of values are
  multiplied in 8-bit"); SmoothQuant (Xiao et al., ICML 2023, arXiv 2211.10438: "The scale of outliers in
  activations is ~100× larger than most of the activation values"; "If one channel has an outlier, it
  persistently appears in all tokens"; s_j = max|X_j|^α / max|W_j|^(1−α), "α=0.5 is a well-balanced point";
  "up to 1.56x speedup and 2x memory reduction for LLMs with negligible loss in accuracy" — all read on
  arxiv.org 2026-10-24); FP8 formats (Micikevicius et al. 2022, arXiv 2209.05433, Table 1: E4M3 max normal
  1.75 × 2^8 = 448, E5M2 1.75 × 2^15 = 57,344; "E4M3 for weight and activation tensors, and E5M2 for gradient
  tensors"); Kurtic et al. (ACL 2025, arXiv 2411.02355): "FP8 (W8A8-FP) is effectively lossless across all
  model scales", "well-tuned INT8 (W8A8-INT) achieves surprisingly low (1-3%) accuracy degradation", "over
  500,000 evaluations" on the Llama-3.1 family.
- NVIDIA H100 page (read 2026-10-24): H100 SXM FP8 Tensor Core 3,958 TFLOPS and BF16 1,979 "with sparsity";
  dense = half: FP8 1,979, BF16 989 (owl_reference's peak). So 8-bit tensor cores do twice the BF16 math.

Illustrative (said "for example" or obvious): the lopsided group of section 3; the 4 × 8 activation matrix
(ordinary values ±0.5, one channel ≈ 100× larger, after SmoothQuant's reported ratio) and the 8 × 3 weight
matrix (rows chosen so every ordinary channel's smoothing factor is exactly 1); the bit boxes; the batch of
256 (Episode 4's regime near the H100 ridge, weights only, no KV). Asserts below tie every quoted number to
the data.

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

GB = 1e9
M_BW = MAC["bandwidth"]                                  # 546 GB/s
W16 = weight_bytes(bits=16)                               # 16.06 GB
W8 = weight_bytes(bits=8)                                 # 8.03 GB
CEIL16 = decode_ceiling(hw=MAC, bits=16)                  # 34.0 tok/s
CEIL8 = decode_ceiling(hw=MAC, bits=8)                    # 68.0 tok/s: the bytes model at 8 bits

# §2 · one group of eight weights, symmetric INT8 (notes §8's worked example, recomputed at 8 bits)
GROUP = np.array([0.12, -0.50, 0.33, 0.05, -0.27, 0.48, -0.09, 0.21])
QMAX = 2 ** (8 - 1) - 1                                   # 127, the largest 8-bit integer
G_MAX = float(np.abs(GROUP).max())                        # 0.50
SCALE = G_MAX / QMAX                                      # 0.00394
QUOT = GROUP / SCALE
Q = np.round(QUOT).astype(int)                            # [30, −127, 84, 13, −69, 122, −23, 53]
BACK = Q * SCALE
ERR = np.abs(BACK - GROUP)
BYTES_BF16 = GROUP.size * 2                               # 16 bytes
BYTES_INT8 = GROUP.size * 1 + 2                           # 8 one-byte integers + one 2-byte scale = 10

# §3 · a lopsided group (for example): symmetric vs a zero point; then MLX's groups of 64
LOP = np.array([0.05, 0.31, 0.12, 0.58, 0.22, -0.04, 0.40, 0.17])
SYM_STEP = float(np.abs(LOP).max()) / QMAX
SYM_Q = np.round(LOP / SYM_STEP)
SYM_UNUSED = int(SYM_Q.min() - (-QMAX))                   # levels −127 … −10 never used: 118
SYM_LEVELS = 2 * QMAX + 1                                 # 255
ZP_STEP = float(LOP.max() - LOP.min()) / 255              # 256 levels across [min, max]
ZERO = int(round(-LOP.min() / ZP_STEP))                   # 16: the integer that stands for zero
ZP_Q = np.round(LOP / ZP_STEP) + ZERO
GROUP_SIZE = measured("models", "llama-3.1-8b", "8bit", "quantization", "group_size")    # 64
Q_BITS = measured("models", "llama-3.1-8b", "8bit", "quantization", "bits")              # 8
BPW = Q_BITS + (16 + 16) / GROUP_SIZE                     # 8.5 bits per weight (bf16 scale + bf16 bias)
W85 = weight_bytes(bits=BPW)                              # 8.53 GB
DISK8 = measured("models", "llama-3.1-8b", "8bit", "weights_file_bytes")                 # 8.532 GB on disk
CEIL85 = decode_ceiling(hw=MAC, bits=BPW)                 # 64.0 tok/s

# §4 · measured on this Mac
D16 = measured("decode", "llama-3.1-8b", "bf16")
D8 = measured("decode", "llama-3.1-8b", "8bit")
TOK16, TOK8 = D16["tok_s_median"], D8["tok_s_median"]    # 31.0 and 58.4 tok/s
OF16, OF85 = TOK16 / CEIL16, TOK8 / CEIL85               # 91% and 91%
OF8 = TOK8 / CEIL8                                        # 86% of the bits=8 model
SPEEDUP = TOK8 / TOK16                                    # 1.88×
FEWER = measured("models", "llama-3.1-8b", "bf16", "weights_file_bytes") / DISK8   # 1.88× fewer bytes
MBU16, MBU8 = D16["mbu"], D8["mbu"]                       # 85.4%, 85.6%

# §5 · weight-only vs weight + activation, on the H100 roofline (weights only, no KV)
H_BW, H_PEAK = GPU["bandwidth"], GPU["peak"]
FP8_HW = dict(GPU, name="H100 SXM, FP8", peak=3958e12 / 2)  # NVIDIA: FP8 3,958 TFLOPS "with sparsity" → 1,979 dense
BATCH = 256
RIDGE_H, RIDGE_F8 = ridge(GPU), ridge(FP8_HW)             # 295 and 591 FLOPs/byte
CHAT_GAIN = step_time(hw=GPU, bits=16) / step_time(hw=GPU, bits=8)                          # 2.0
BATCH_W8 = step_time(hw=GPU, batch=BATCH, bits=16) / step_time(hw=GPU, batch=BATCH, bits=8)  # 1.15
BATCH_F8 = step_time(hw=GPU, batch=BATCH, bits=16) / step_time(hw=FP8_HW, batch=BATCH, bits=8)  # 2.0


def intensity(batch, bits):
    """FLOPs per byte of one decode step (weights only): 2·P·B ÷ (P·bits/8)."""
    return batch * flops_per_token() / weight_bytes(bits=bits)


# §6–7 · activations with an outlier channel (for example), and SmoothQuant
X = np.array([[0.3, -0.4, 0.1, 47.0, -0.2, 0.5, -0.3, 0.2],
              [-0.2, 0.5, -0.3, 50.0, 0.4, -0.1, 0.2, -0.5],
              [0.4, -0.1, 0.3, 46.0, -0.5, 0.2, 0.3, 0.1],
              [-0.1, 0.3, -0.2, 44.0, 0.3, -0.4, -0.1, 0.4]])
W = np.array([[0.4, -0.2, 0.1], [-0.3, 0.5, -0.1], [0.2, 0.1, -0.3], [0.5, -0.4, 0.3],
              [-0.1, 0.3, 0.5], [0.3, -0.5, 0.2], [-0.2, 0.1, 0.3], [0.1, 0.4, -0.5]])
OUT = 3                                                    # the outlier channel (shown as channel 4)
ORD = [j for j in range(X.shape[1]) if j != OUT]
GIANT = float(np.abs(X[:, OUT]).max() / np.abs(X[:, ORD]).max())   # 100×


def int8_row(row):
    """One token, one symmetric INT8 scale (per token): integers, read-back values, scale."""
    s = float(np.abs(row).max()) / QMAX
    q = np.round(row / s)
    return q.astype(int), q * s, s


def rel_err(row, back):
    return float(np.mean(np.abs(back - row)[ORD] / np.abs(row)[ORD]))


T1_Q, T1_BACK, T1_S = int8_row(X[0])                       # scale 0.370; ordinary values → −1, 0, 1
T1_ERR = rel_err(X[0], T1_BACK)                            # 50%
SMOOTH = np.sqrt(np.abs(X).max(0) / np.abs(W).max(1))      # SmoothQuant, α = 0.5: [1, 1, 1, 10, 1, …]
XS, WS = X / SMOOTH, W * SMOOTH[:, None]
S_OUT = float(SMOOTH[OUT])                                 # 10
XS_RATIO = float(np.abs(XS[:, OUT]).max() / np.abs(XS[:, ORD]).max())   # tenfold
WS_RATIO = float(np.abs(WS[OUT]).max() / np.abs(np.delete(WS, OUT, 0)).max())   # tenfold
S1_Q, S1_BACK, S1_S = int8_row(XS[0])
S1_ERR = rel_err(XS[0], S1_BACK)                           # 5%


def e4m3_values():
    """Every finite FP8 E4M3 value (bias 7, no infinities, S.1111.111 is NaN)."""
    vals = {0.0}
    for e in range(16):
        for f in range(8):
            if e == 15 and f == 7:
                continue
            v = (f / 8) * 2.0 ** -6 if e == 0 else (1 + f / 8) * 2.0 ** (e - 7)
            vals |= {v, -v}
    return np.array(sorted(vals))


E4M3 = e4m3_values()
E4M3_MAX = float(E4M3.max())                               # 448
E5M2_MAX = 1.75 * 2 ** 15                                  # 57,344 (Micikevicius et al., Table 1)
F8_S = float(np.abs(X[0]).max()) / E4M3_MAX                # the giant maps to 448
T1_F8 = E4M3[np.abs(X[0][:, None] / F8_S - E4M3[None, :]).argmin(1)] * F8_S
T1_F8_ERR = rel_err(X[0], T1_F8)                           # 3%
SPAN = 0.55                                                # the ordinary values' range, ±0.55
INT8_LEVELS = [k * T1_S for k in range(-QMAX, QMAX + 1) if abs(k * T1_S) <= SPAN]
F8_LEVELS = [v * F8_S for v in E4M3 if abs(v * F8_S) <= SPAN]

# ---------------------------------------------------------------------------- asserts: the words match the data
assert round(M_BW / GB) == 546 and round(W16 / GB) == 16 and round(CEIL16) == 34          # "five hundred and forty-six … sixteen … thirty-four"
assert round(W8 / GB, 2) == 8.03 and round(CEIL8) == 68 and round(CEIL8 / CEIL16) == 2    # "half the bytes", "sixty-eight"
assert G_MAX == 0.5 and QMAX == 127                                                        # "zero point five", "a hundred and twenty-seven"
assert round(SCALE * 1000) == 4 and f"{SCALE:.5f}" == "0.00394"                            # "about four thousandths"
assert Q.tolist() == [30, -127, 84, 13, -69, 122, -23, 53]
assert BYTES_BF16 == 16 and BYTES_INT8 == 10                                               # "ten bytes instead of sixteen"
assert ERR.max() <= SCALE / 2 and ERR.max() < 0.002 and int(ERR.argmax()) == 0             # "under two thousandths"
assert SYM_UNUSED == 118 and 0.4 < SYM_UNUSED / SYM_LEVELS < 0.5                           # "almost half its levels"
assert ZERO == 16 and ZP_Q.min() == 0 and ZP_Q.max() == 255                                 # "two hundred and fifty-six levels"
assert GROUP_SIZE == 64 and Q_BITS == 8 and BPW == 8.5                                     # "groups of sixty-four", "eight and a half bits"
assert round(W85 / GB, 2) == 8.53 and abs(W85 - DISK8) / DISK8 < 1e-3                      # "eight point five three gigabytes", "exactly … on disk"
assert f"{TOK16:.1f}" == "31.0" and round(OF16 * 100) == 91                                # "thirty-one", "ninety-one percent"
assert f"{TOK8:.1f}" == "58.4" and round(CEIL85) == 64 and round(OF85 * 100) == 91         # "fifty-eight point four", "sixty-four", "again"
assert f"{SPEEDUP:.2f}" == "1.88" and f"{FEWER:.2f}" == "1.88"                             # "one point eight eight … one point eight eight"
assert 0.85 <= MBU16 < 0.86 and 0.85 <= MBU8 < 0.86                                        # "about eighty-five percent"
assert round(RIDGE_H) == 295 and round(RIDGE_F8) == 591 and round(FP8_HW["peak"] / 1e12) == 1979
assert round(FP8_HW["peak"] / H_PEAK, 2) == 2.0                                             # "twice the math"
assert round(intensity(1, 16)) == 1 and round(intensity(1, 8)) == 2 and round(intensity(BATCH, 8)) == 512
assert CHAT_GAIN == 2.0 and round((BATCH_W8 - 1) * 100) == 15 and round(BATCH_F8, 6) == 2.0  # "twice", "fifteen percent", "twice"
assert GIANT == 100.0                                                                     # "about a hundred times larger"
assert f"{T1_S:.3f}" == "0.370" and set(T1_Q[ORD]) == {-1, 0, 1}                           # "minus one, zero, or one"
assert round(T1_ERR * 100) == 50                                                          # "off by half their size"
assert np.allclose(XS @ WS, X @ W) and S_OUT == 10.0 and all(SMOOTH[ORD] == 1.0)           # "changes nothing", "by ten"
assert XS_RATIO == 10.0 and round(WS_RATIO) == 10                                         # "tenfold, on each side"
assert round(S1_ERR * 100) == 5                                                           # "about five percent"
assert E4M3_MAX == 448 and E5M2_MAX == 57_344                                             # "four hundred and forty-eight", "fifty-seven thousand"
assert len(INT8_LEVELS) == 3 and len(F8_LEVELS) > 100 and round(T1_F8_ERR * 100) == 3     # "over a hundred levels", "about three percent"

# ---------------------------------------------------------------------------- look

PREFIX = "ep18"
FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIKLMNOPRSTUVWXYZ0123456789")
SYMBOLS = {"≈", "÷", "×", "=", "+"}


def cite(s, size=20):
    return text(s, size, MUTED)


def baseline(words):
    """The baseline of a Text: the median bottom of its flat-bottomed glyphs."""
    chars = [c for c in words.text if not c.isspace()]
    if len(chars) != len(words):
        return float(np.median([g.get_bottom()[1] for g in words]))
    bottoms = sorted(g.get_bottom()[1] for g, c in zip(words, chars) if c in FLAT)
    return bottoms[len(bottoms) // 2] if bottoms else words.get_bottom()[1]


def sit(words, y):
    return words.shift((y - baseline(words)) * m.UP)


def formula(parts, size=48, buff=0.24):
    """A display formula in DIN: words on one baseline, symbols centred on the x-height."""
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
    bar = m.Rectangle(width=max(width, 0.04), height=height).set_stroke(color, stroke).set_fill(fill, 1)
    return bar.move_to([x0 + max(width, 0.04) / 2, y, 0])


def minus(s):
    return s.replace("-", "−")


def num_cell(s, w, h, color, fill, size=22, ink=INK):
    box = card(w, h, color, fill, radius=0.08, stroke=2)
    return m.VGroup(box, code(minus(s), size, ink).move_to(box))


def cell_row(strings, x0, y, w, h, pitch, color, fill, size=22, ink=INK):
    cells = m.VGroup()
    for i, s in enumerate(strings):
        cells.add(num_cell(s, w, h, color, fill, size, ink).move_to([x0 + i * pitch + w / 2, y, 0]))
    return cells


def relabel(cell, s, size=22, ink=INK):
    """The animation that swaps a cell's number."""
    new = code(minus(s), size, ink).move_to(cell[0])
    return m.ReplacementTransform(cell[1], new), new


def row_label(title, sub, x, y, color=INK):
    words = m.VGroup(code(title, 22, color, weight=m.BOLD), code(sub, 20, MUTED))
    words.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.08)
    return words.move_to([0, y, 0]).align_to([x, 0, 0], m.LEFT)


def squares(n, x0, y, color, fill, size=0.22, gap=0.05):
    """n small squares in a row: one per byte."""
    row = m.VGroup(*[m.Square(size).set_stroke(color, 1.5).set_fill(fill, 1) for _ in range(n)])
    row.arrange(m.RIGHT, buff=gap)
    return row.move_to([x0 + row.width / 2, y, 0])


BIT_STYLE = {"s": (MUTED, CARD), "e": (WEIGHTS, BLUE_FILL), "m": (INK, CARD_HI), "b": (WEIGHTS, BLUE_FILL)}


def bit_row(layout, x0, y, size=0.34, gap=0.05):
    """Bit boxes: layout like 'seeeemmm' (sign, exponent, mantissa) or 'bbbb' (plain bits)."""
    boxes = m.VGroup()
    for i, kind in enumerate(layout):
        color, fill = BIT_STYLE[kind]
        sq = m.RoundedRectangle(corner_radius=0.05, width=size, height=size * 1.25).set_stroke(color, 2).set_fill(fill, 1)
        boxes.add(sq.move_to([x0 + i * (size + gap) + size / 2, y, 0]))
    return boxes


def part(boxes, layout, kind):
    return m.VGroup(*[b for b, k in zip(boxes, layout) if k == kind])


def mac_studio(width=1.4, color=MUTED):
    """A Mac Studio, front view (as in Episode 2)."""
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


def info_card(title, tag, lines, color, width=4.1):
    head = m.VGroup(text(title, 24, color, weight=m.BOLD), code(tag, 20, MUTED)).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.1)
    body = m.VGroup(*[text(s, 20, INK if i == 0 else MUTED) for i, s in enumerate(lines)])
    body.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.12)
    inner = m.VGroup(head, body).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.2)
    box = card(width, inner.height + 0.5, color, CARD, radius=0.16, stroke=2.5)
    inner.move_to(box).align_to(box, m.LEFT).shift(0.3 * m.RIGHT)
    return m.VGroup(box, inner)


def fmt1(v):
    return f"{v:.1f}"


# ---------------------------------------------------------------------------- the episode


class Ep18FewerBitsSameModel(Episode):
    number = 18

    def lesson(self):
        self.fewer_bytes()
        self.one_scale()
        self.zero_points()
        self.check()
        self.two_kinds()
        self.outliers()
        self.smooth()
        self.fp8()
        self.recap()

    # ------------------------------------------------------------------ 1 · fewer bytes

    def fewer_bytes(self):
        self.next_section("1 · Fewer bytes", notes=(
            f"Episode 2: batch-1 decode is memory-bound, tok/s ≈ bandwidth ÷ bytes per token (owl_reference.decode_ceiling). "
            f"This Mac (M4 Max, {M_BW / GB:.0f} GB/s, Apple spec): {M_BW / GB:.0f} ÷ {W16 / GB:.2f} GB = {CEIL16:.1f} tok/s for "
            f"Owl-8B in BF16. Eight bits per weight: {W8 / GB:.2f} GB → {CEIL8:.1f} tok/s (decode_ceiling(hw=MAC, bits=8)). "
            "The bit boxes are schematic."))
        top = formula([("tok/s", INK), ("≈", MUTED), ("bandwidth", MEMORY), ("÷", MUTED), ("bytes per token", WEIGHTS)], 48)
        top.move_to([0, 1.95, 0])
        x0, full = -5.6, 8.6
        cap = code(f"Owl-8B · {OWL['params'] / 1e9:.2f} B weights, all read for every token", 20, MUTED)
        cap.move_to([0, 0.95, 0]).align_to([x0, 0, 0], m.LEFT)
        bar = hbar(x0, 0.45, full, 0.46, WEIGHTS, BLUE_FILL)
        val = code(f"{W16 / GB:.2f} GB", 22, WEIGHTS, weight=m.BOLD).next_to(bar, m.RIGHT, buff=0.2)
        half = hbar(x0, 0.45, full * W8 / W16, 0.46, WEIGHTS, BLUE_FILL)
        val8 = code(f"{W8 / GB:.2f} GB", 22, WEIGHTS, weight=m.BOLD).next_to(half, m.RIGHT, buff=0.2)

        one = text("one weight", 22, MUTED).move_to([0, -0.7, 0]).align_to([x0, 0, 0], m.LEFT)
        bits = bit_row("b" * 16, -3.55, -0.7, size=0.34, gap=0.08)
        n16 = code("16 bits", 22, WEIGHTS).next_to(bits, m.RIGHT, buff=0.3)
        n8 = code("8 bits", 22, WEIGHTS).next_to(bits[:8], m.RIGHT, buff=0.3)

        bw = stat(f"{M_BW / GB:.0f}", "GB/s", "bandwidth · M4 Max", MEMORY, 60).move_to([0, -2.25, 0]).align_to([x0, 0, 0], m.LEFT)
        lock = pill("fixed by the chip", MUTED, CARD, 20).next_to(bw[0], m.RIGHT, buff=1.4)
        lock.align_to(bw[0], m.DOWN)
        ceil = stat(f"{CEIL16:.0f}", "tok/s", "ceiling · batch 1", MEMORY, 60).move_to([0, -2.25, 0]).align_to([1.3, 0, 0], m.LEFT)
        ceil68 = title_text(f"{CEIL8:.0f}", 60, MEMORY).move_to(ceil[0]).align_to(ceil[0], m.LEFT)
        mascot = owl(0.8).move_to([5.75, -2.35, 0])
        slogan = pill("same model, fewer bits", TOKEN, GREEN_FILL, 22).move_to([0, 0.45, 0]).align_to([1.4, 0, 0], m.LEFT)

        self.say(
            "[In episode two, Owl's speed on this Mac came down to one division]: [bandwidth, over bytes per token.]",
            m.AnimationGroup(self.new_header("01 · fewer bytes", "Same model, fewer bytes"), m.FadeIn(top, shift=0.15 * m.UP),
                             m.FadeIn(mascot, shift=0.2 * m.UP)),
            m.AnimationGroup(m.Circumscribe(top[2], color=MEMORY, buff=0.08), m.Circumscribe(top[4], color=WEIGHTS, buff=0.08)),
        )
        self.say(
            "[Five hundred and forty-six gigabytes per second, over sixteen gigabytes of weights], [gives at most thirty-four tokens a second.]",
            m.AnimationGroup(m.FadeIn(bw, shift=0.1 * m.UP), m.FadeIn(cap), m.GrowFromEdge(bar, m.LEFT), m.FadeIn(val)),
            m.FadeIn(ceil, shift=0.1 * m.UP),
        )
        self.say(
            "[The chip fixes the bandwidth.] [The bytes are a choice]: [right now, every weight takes sixteen bits.]",
            m.FadeIn(lock, shift=0.1 * m.LEFT),
            m.Circumscribe(m.VGroup(bar, val), color=WEIGHTS, buff=0.08),
            m.AnimationGroup(m.FadeIn(one), m.LaggedStart(*[m.FadeIn(b, shift=0.1 * m.UP) for b in bits], lag_ratio=0.12),
                             m.FadeIn(n16)),
        )
        self.say(
            "[Store each one in eight bits instead], [and every token reads half the bytes.] [Same model, fewer bits.]",
            m.AnimationGroup(m.FadeOut(bits[8:], shift=0.15 * m.DOWN), m.ReplacementTransform(n16, n8)),
            m.AnimationGroup(m.Transform(bar, half), m.ReplacementTransform(val, val8),
                             m.ReplacementTransform(ceil[0], ceil68)),
            m.FadeIn(slogan, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 2 · one scale

    def one_scale(self):
        self.next_section("2 · One scale", notes=(
            f"Symmetric (absmax) INT8 on one group of eight weights (notes §8's worked example, recomputed at 8 bits; the "
            f"values are illustrative): max|w| = {G_MAX:.2f}; scale = {G_MAX:.2f} ÷ {QMAX} = {SCALE:.6f}; "
            f"q = round(w ÷ scale) = {Q.tolist()}; read back q × scale; max error {ERR.max():.4f} ≤ half a step "
            f"({SCALE / 2:.4f}). Storage: {BYTES_BF16} bytes in BF16 → {GROUP.size} one-byte integers + one 2-byte scale = "
            f"{BYTES_INT8} bytes (one scale per 8 weights is for the example; real formats use bigger groups, next section). "
            "The rounding itself is covered in 'AI, Explained' Episode 42."))
        x0, w, h, pitch = -4.2, 1.12, 0.6, 1.3
        lab_x = -6.55
        y_w, y_q, y_b = 1.75, 0.6, -0.55
        wts = cell_row([f"{v:.2f}" for v in GROUP], x0, y_w, w, h, pitch, WEIGHTS, BLUE_FILL)
        l_w = row_label("BF16", "2 bytes each", lab_x, y_w, WEIGHTS)
        quot = cell_row([f"{v:.1f}" for v in QUOT], x0, y_q, w, h, pitch, WEIGHTS, CARD, size=20)
        l_q = row_label("INT8", "1 byte each", lab_x, y_q, WEIGHTS)
        back = cell_row([f"{v:.3f}" for v in BACK], x0, y_b, w, h, pitch, EDGE, CARD, size=20)
        l_b = row_label("read back", "q × scale", lab_x, y_b, INK)

        scan = m.SurroundingRectangle(wts[0], buff=0.07, corner_radius=0.1).set_stroke(INK, 3)
        biggest = int(np.abs(GROUP).argmax())
        lx = -6.55
        l1 = code(f"largest |w| = {G_MAX:.2f}", 24, INK).move_to([0, -1.65, 0]).align_to([lx, 0, 0], m.LEFT)
        l2a = code(f"scale = {G_MAX:.2f} ÷ {QMAX}", 24, INK)
        l2a.move_to([0, -2.3, 0]).align_to([lx, 0, 0], m.LEFT)
        l2b = code(f"= {SCALE:.5f}", 24, TOKEN, weight=m.BOLD).next_to(l2a, m.RIGHT, buff=0.2)
        sit(l2b, baseline(l2a))

        sx = 0.2
        strip16 = squares(BYTES_BF16, sx, -1.65, WEIGHTS, BLUE_FILL)
        s_lab16 = code(f"{BYTES_BF16} bytes", 22, WEIGHTS).next_to(strip16, m.RIGHT, buff=0.25)
        sc_sq = squares(2, 0, -1.65, INK, CARD_HI)
        sc_sq.next_to(strip16[GROUP.size - 1], m.RIGHT, buff=0.05)
        s_lab10 = code(f"{BYTES_INT8} bytes", 22, TOKEN, weight=m.BOLD).next_to(sc_sq, m.RIGHT, buff=0.25)
        s_cap = text("storage, one square per byte", 20, MUTED).next_to(strip16, m.UP, buff=0.18).align_to(strip16, m.LEFT)
        sc_tag = code("scale", 20, INK).next_to(sc_sq, m.DOWN, buff=0.18)

        err = code(f"max error {ERR.max():.4f} < half a step", 22, TOKEN).move_to([0, -3.25, 0]).align_to([lx, 0, 0], m.LEFT)
        ep42 = cite("rounding in depth: AI, Explained · Episode 42").move_to([0, -3.25, 0]).align_to([6.5, 0, 0], m.RIGHT)
        sit(ep42, baseline(err))

        sweep = [m.FadeIn(scan, run_time=0.25)]
        sweep += [scan.animate(run_time=0.22).move_to(wts[i]) for i in range(1, GROUP.size)]
        sweep += [scan.animate(run_time=0.4).move_to(wts[biggest])]

        self.say(
            "[Here's one group of eight weights, for example], [each stored in sixteen bits: sixteen bytes in all.]",
            m.AnimationGroup(self.new_header("02 · one scale", "Eight bits and a scale"), m.FadeIn(l_w),
                             m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.UP) for c in wts], lag_ratio=0.15)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(s, scale=0.6) for s in strip16], lag_ratio=0.1), m.FadeIn(s_lab16), m.FadeIn(s_cap)),
        )
        self.say(
            "[Step one: find the largest size, ignoring the sign]: [zero point five.]",
            m.Succession(*sweep),
            m.AnimationGroup(m.FadeOut(scan), m.Circumscribe(wts[biggest], color=INK, buff=0.08), m.FadeIn(l1, shift=0.1 * m.UP)),
        )
        self.say(
            "[Step two: divide it by a hundred and twenty-seven, the biggest eight-bit integer.] [That's the scale: one integer step is worth about four thousandths.]",
            m.FadeIn(l2a, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(l2b, shift=0.1 * m.LEFT), m.Circumscribe(l2b, color=TOKEN, buff=0.08)),
        )
        rounds = [relabel(c, str(v), 22)[0] for c, v in zip(quot, Q)]
        self.say(
            "[Step three: divide every weight by the scale], [and round to the nearest whole number.]",
            m.AnimationGroup(m.FadeIn(l_q), m.LaggedStart(*[m.FadeIn(q, target_position=c) for q, c in zip(quot, wts)], lag_ratio=0.1)),
            m.LaggedStart(*rounds, lag_ratio=0.1),
        )
        self.say(
            "[Now each weight fits in one byte], [plus one two-byte scale for the group]: [ten bytes instead of sixteen.]",
            m.AnimationGroup(m.FadeOut(strip16[GROUP.size:], shift=0.1 * m.DOWN), m.Circumscribe(m.VGroup(*quot), color=WEIGHTS, buff=0.08)),
            m.AnimationGroup(m.FadeIn(sc_sq, shift=0.1 * m.LEFT), m.FadeIn(sc_tag)),
            m.ReplacementTransform(s_lab16, s_lab10),
        )
        self.say(
            "[To read a weight back, multiply by the scale.] [Nothing moves by more than half a step]: [under two thousandths.]",
            m.AnimationGroup(m.FadeIn(l_b), m.LaggedStart(*[m.FadeIn(b, target_position=q) for b, q in zip(back, quot)], lag_ratio=0.1)),
            m.Circumscribe(back[int(ERR.argmax())], color=TOKEN, buff=0.08),
            m.AnimationGroup(m.FadeIn(err, shift=0.1 * m.UP), m.FadeIn(ep42)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 3 · zero points and groups

    def zero_points(self):
        self.next_section("3 · Zero points and groups", notes=(
            f"A lopsided group (illustrative): {LOP.tolist()}. Symmetric INT8: step {SYM_STEP:.5f}, integers used "
            f"{int(SYM_Q.min())}…{int(SYM_Q.max())}, so {SYM_UNUSED} of {SYM_LEVELS} levels never used. With a zero point "
            f"(asymmetric, 256 levels over [min, max]): step {ZP_STEP:.5f}, zero point {ZERO}, q = round(w ÷ step) + {ZERO} "
            f"spans {int(ZP_Q.min())}…{int(ZP_Q.max())}. MLX's affine mode (mlx.core.quantize docs) stores the group minimum "
            f"('bias') instead of an integer zero point — the same job. Owl's 8-bit model (mlx-community): group size "
            f"{GROUP_SIZE}, {Q_BITS} bits, one bf16 scale + one bf16 bias per group → {Q_BITS} + 32/{GROUP_SIZE} = {BPW} bits "
            f"per weight → {OWL['params'] / 1e9:.2f} B × {BPW} ÷ 8 = {W85 / GB:.3f} GB; files on disk {DISK8 / GB:.3f} GB "
            "(research/measurements.md, Models). Group-wise INT4, GPTQ, AWQ and GGUF: Episode 19."))
        lo, hi, ax0, ax1, y_ax = -0.65, 0.65, -4.2, 5.4, 1.75

        def X_(v):
            return ax0 + (v - lo) / (hi - lo) * (ax1 - ax0)

        axis = m.Line([ax0, y_ax, 0], [ax1, y_ax, 0]).set_stroke(FAINT, 2.5)
        ticks = m.VGroup()
        for v in (-0.6, -0.3, 0.0, 0.3, 0.6):
            ticks.add(m.Line([X_(v), y_ax - 0.08, 0], [X_(v), y_ax + 0.08, 0]).set_stroke(FAINT, 2))
            ticks.add(code(minus(f"{v:.1f}"), 20, MUTED).next_to([X_(v), y_ax, 0], m.UP, buff=0.16))
        dots = m.VGroup(*[m.Dot([X_(v), y_ax, 0], radius=0.075, color=WEIGHTS).set_z_index(3) for v in LOP])
        l_axis = text("a lopsided group", 22, WEIGHTS).move_to([0, y_ax, 0]).align_to([-6.55, 0, 0], m.LEFT)

        bh = 0.38
        y_sym, y_zp = 0.95, 0.2
        top_sym = float(np.abs(LOP).max())
        sym = m.Rectangle(width=X_(top_sym) - X_(-top_sym), height=bh).set_stroke(WEIGHTS, 2).set_fill(BLUE_FILL, 1)
        sym.move_to([(X_(top_sym) + X_(-top_sym)) / 2, y_sym, 0])
        used_lo = float(SYM_Q.min()) * SYM_STEP
        waste = m.Rectangle(width=X_(used_lo) - X_(-top_sym), height=bh).set_stroke(RED, 2).set_fill(RED_FILL, 1)
        waste.move_to([(X_(used_lo) + X_(-top_sym)) / 2, y_sym, 0]).set_z_index(1)
        w_lab = code(f"{SYM_UNUSED} of {SYM_LEVELS} unused", 20, RED).move_to(waste).set_z_index(2)
        l_sym = text("symmetric", 22, INK).move_to([0, y_sym, 0]).align_to([-6.55, 0, 0], m.LEFT)

        zp = m.Rectangle(width=X_(LOP.max()) - X_(LOP.min()), height=bh).set_stroke(TOKEN, 2).set_fill(BLUE_FILL, 1)
        zp.move_to([(X_(LOP.max()) + X_(LOP.min())) / 2, y_zp, 0])
        zp_lab = code("256 levels", 20, TOKEN, weight=m.BOLD).move_to(zp).shift(0.6 * m.RIGHT).set_z_index(2)
        l_zp = text("with a zero point", 22, INK).move_to([0, y_zp, 0]).align_to([-6.55, 0, 0], m.LEFT)
        z_mark = m.Line([X_(0), y_zp - bh / 2 - 0.12, 0], [X_(0), y_zp + bh / 2 + 0.12, 0]).set_stroke(INK, 3).set_z_index(3)
        z_lab = code(f"zero point: q = {ZERO} means 0.0", 20, INK).next_to(z_mark, m.DOWN, buff=0.12)
        z_lab.align_to([X_(0) - 0.25, 0, 0], m.LEFT)
        guides = m.VGroup(*[m.DashedLine([X_(v), y_ax - 0.1, 0], [X_(v), y_zp - bh / 2, 0], dash_length=0.07).set_stroke(FAINT, 1.5)
                            for v in (LOP.min(), LOP.max())])

        # a group of 64 weights + one scale + one zero point (MLX: a bias)
        n, cw, cg = GROUP_SIZE, 0.1, 0.035
        gx0, gy = -5.95, -1.35
        cells = m.VGroup(*[m.Rectangle(width=cw, height=0.38).set_stroke(WEIGHTS, 1.2).set_fill(BLUE_FILL, 1) for _ in range(n)])
        cells.arrange(m.RIGHT, buff=cg).move_to([gx0, gy, 0], aligned_edge=m.LEFT)
        extra = m.VGroup(num_cell("scale", 1.3, 0.5, INK, CARD_HI, 22), num_cell("zero", 1.2, 0.5, INK, CARD_HI, 22))
        extra.arrange(m.RIGHT, buff=0.1).next_to(cells, m.RIGHT, buff=0.18)
        g_cap = code(f"one group: {n} weights × {Q_BITS} bits", 20, WEIGHTS).next_to(cells, m.UP, buff=0.14).align_to(cells, m.LEFT)
        e_cap = code("16 bits each", 20, MUTED).next_to(extra, m.UP, buff=0.14)
        sit(e_cap, baseline(g_cap))
        rule = code(f"{Q_BITS} + (16 + 16) ÷ {n} = {BPW} bits per weight", 26, INK,
                    t2c={rf"= {BPW}": TOKEN, r"\(16 \+ 16\)": INK}).move_to([0, -2.1, 0])
        owl_gb = code(f"Owl: {OWL['params'] / 1e9:.2f} B × {BPW} bits ÷ 8 = {W85 / GB:.2f} GB", 24, WEIGHTS,
                      t2c={rf"{W85 / GB:.2f} GB": TOKEN}).move_to([0, -2.8, 0]).align_to([gx0, 0, 0], m.LEFT)
        disk = cite(f"8-bit model on disk: {DISK8 / GB:.3f} GB · measured on this Mac").next_to(owl_gb, m.DOWN, buff=0.18)
        disk.align_to(owl_gb, m.LEFT)
        ep19 = cite("4-bit groups, GPTQ, AWQ, GGUF: Episode 19").move_to(disk).align_to([6.5, 0, 0], m.RIGHT)

        self.say(
            "[Now a lopsided group, for example, with every weight but one above zero.] [A symmetric grid spans both signs], [so almost half its levels go unused.]",
            m.AnimationGroup(self.new_header("03 · zero points", "Zero points and groups"), m.FadeIn(axis), m.FadeIn(ticks),
                             m.FadeIn(l_axis), m.LaggedStart(*[m.FadeIn(d, scale=0.4) for d in dots], lag_ratio=0.15)),
            m.AnimationGroup(m.FadeIn(l_sym), m.GrowFromCenter(sym)),
            m.AnimationGroup(m.FadeIn(waste), m.FadeIn(w_lab)),
        )
        self.say(
            "[A zero point fixes that]: [it says which integer stands for zero], [so all two hundred and fifty-six levels cover just the group's range.]",
            m.AnimationGroup(m.FadeIn(l_zp), m.Create(guides)),
            m.AnimationGroup(m.FadeIn(zp, target_position=sym), m.Create(z_mark), m.FadeIn(z_lab, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(zp_lab), m.Circumscribe(zp, color=TOKEN, buff=0.06)),
        )
        self.say(
            "[Real formats share a scale, and often a zero point, per group.] [MLX uses groups of sixty-four], [so two sixteen-bit numbers add half a bit per weight]: [eight and a half bits.]",
            m.AnimationGroup(m.FadeIn(g_cap), m.LaggedStart(*[m.FadeIn(c) for c in cells], lag_ratio=0.03)),
            m.Circumscribe(cells, color=WEIGHTS, buff=0.06),
            m.AnimationGroup(m.FadeIn(extra, shift=0.1 * m.LEFT), m.FadeIn(e_cap)),
            m.FadeIn(rule, shift=0.1 * m.UP),
        )
        self.say(
            "[For Owl, that's eight point five three gigabytes], [exactly the size of the eight-bit model on disk.]",
            m.FadeIn(owl_gb, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(disk, shift=0.1 * m.UP), m.FadeIn(ep19)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 4 · checked on this Mac

    def check(self):
        self.next_section("4 · Checked on this Mac", notes=(
            f"Measured on this Mac (Mac Studio, M4 Max 546 GB/s, mlx-lm 0.32.2, 2026-10-24; research/measurements.md §1; "
            f"median of 5, 256 tokens, greedy): Llama-3.1-8B-Instruct bf16 {TOK16:.2f} tok/s, 8bit {TOK8:.2f} tok/s. Ceilings "
            f"(decode_ceiling(hw=MAC, bits)): 16 bits {CEIL16:.1f}; 8 bits {CEIL8:.1f} (the simple bytes model; measured = "
            f"{OF8 * 100:.0f}% of it); {BPW} bits (with MLX's per-64 scale and bias) {CEIL85:.1f}. Measured ÷ ceiling: "
            f"{OF16 * 100:.1f}% and {OF85 * 100:.1f}%. Speed ratio {SPEEDUP:.3f}×; file-bytes ratio {FEWER:.3f}×. MBU (bytes "
            f"really read per step × tok/s ÷ 546 GB/s; the untied embedding table is a lookup): {MBU16 * 100:.1f}% and "
            f"{MBU8 * 100:.1f}%."))
        box = mac_studio(1.25).move_to([-5.85, 2.0, 0])
        who = m.VGroup(text("Mac Studio · M4 Max", 24, INK, weight=m.BOLD), code(f"{M_BW / GB:.0f} GB/s · mlx-lm", 20, MUTED))
        who.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.08).next_to(box, m.RIGHT, buff=0.3)
        src = cite("measured on this Mac · median of 5 runs · 256 tokens").move_to([0, 2.0, 0]).align_to([6.5, 0, 0], m.RIGHT)

        bx, full, top = -2.9, 5.4, 70.0
        lab_x = -6.55

        def bar_row(y, value, color, fill):
            return hbar(bx, y, full * value / top, 0.36, color, fill)

        def bar_val(bar, s, color):
            return code(s, 20, color).next_to(bar, m.RIGHT, buff=0.18)

        y1c, y1m, y2c, y2m = 0.95, 0.5, -0.45, -0.9
        l16 = row_label("16-bit", f"{W16 / GB:.2f} GB", lab_x, (y1c + y1m) / 2, WEIGHTS)
        l8 = row_label("8-bit", f"{DISK8 / GB:.2f} GB · {BPW} bits", lab_x, (y2c + y2m) / 2, WEIGHTS)
        c16 = bar_row(y1c, CEIL16, MEMORY, CYAN_FILL)
        c16v = bar_val(c16, f"ceiling {CEIL16:.1f}", MEMORY)
        m16 = bar_row(y1m, TOK16, TOKEN, GREEN_FILL)
        m16v = bar_val(m16, f"measured {TOK16:.1f} tok/s", TOKEN)
        p16 = title_text(f"{OF16 * 100:.0f}%", 44, TOKEN).move_to([5.9, (y1c + y1m) / 2, 0])
        c8 = bar_row(y2c, CEIL8, MEMORY, CYAN_FILL)
        c8v = bar_val(c8, f"model {CEIL8:.1f}", MEMORY)
        c85 = bar_row(y2c, CEIL85, MEMORY, CYAN_FILL)
        c85v = bar_val(c85, f"ceiling {CEIL85:.1f}", MEMORY)
        m8 = bar_row(y2m, TOK8, TOKEN, GREEN_FILL)
        m8v = bar_val(m8, f"measured {TOK8:.1f} tok/s", TOKEN)
        p8 = title_text(f"{OF85 * 100:.0f}%", 44, TOKEN).move_to([5.9, (y2c + y2m) / 2, 0])

        lines = [f"decode_ceiling(hw=MAC, bits=16)   # {CEIL16:.1f}",
                 f"decode_ceiling(hw=MAC, bits=8)    # {CEIL8:.1f}",
                 f"decode_ceiling(hw=MAC, bits={BPW})  # {CEIL85:.1f}"]
        block = code_block("\n".join(lines), 19, title="owl_reference.py · the bytes model", accent=MEMORY)
        block.move_to([0, -2.45, 0]).align_to([lab_x, 0, 0], m.LEFT)
        body = block[1]
        row_len = [len(s.replace(" ", "")) for s in lines]
        cut = [sum(row_len[:i]) for i in range(4)]
        code_rows = [body[cut[i]:cut[i + 1]] for i in range(3)]
        hide = m.VGroup(*code_rows[1], *code_rows[2])
        hide.set_opacity(0)

        px = 2.1
        ratio = pill(f"{SPEEDUP:.2f}× faster  ·  {FEWER:.2f}× fewer bytes", TOKEN, GREEN_FILL, 20)
        ratio.move_to([0, -2.0, 0]).align_to([px, 0, 0], m.LEFT)
        mbu = pill(f"MBU {MBU16 * 100:.1f}% · {MBU8 * 100:.1f}%", MEMORY, CYAN_FILL, 20)
        mbu.next_to(ratio, m.DOWN, buff=0.3).align_to(ratio, m.LEFT)
        mbu_c = cite("bandwidth used, of 546 GB/s").next_to(mbu, m.DOWN, buff=0.14).align_to(mbu, m.LEFT)

        self.say(
            "[Now the check, on this Mac]: [Owl's model in sixteen-bit, and in MLX's eight-bit format.]",
            m.AnimationGroup(self.new_header("04 · measured", "Checked on this Mac"), m.FadeIn(box, shift=0.1 * m.UP), m.FadeIn(who),
                             m.FadeIn(src)),
            m.AnimationGroup(m.FadeIn(l16, shift=0.1 * m.RIGHT), m.FadeIn(l8, shift=0.1 * m.RIGHT)),
        )
        self.say(
            "[Sixteen-bit decodes thirty-one tokens a second], [against a ceiling of thirty-four]: [ninety-one percent.]",
            m.AnimationGroup(m.GrowFromEdge(m16, m.LEFT), m.FadeIn(m16v)),
            m.AnimationGroup(m.GrowFromEdge(c16, m.LEFT), m.FadeIn(c16v), m.FadeIn(block[0]), m.FadeIn(block[2]),
                             m.FadeIn(code_rows[0])),
            m.FadeIn(p16, scale=0.8),
        )
        self.say(
            "[Eight-bit decodes fifty-eight point four.] [Our bytes model, at eight bits, promised sixty-eight.]",
            m.AnimationGroup(m.GrowFromEdge(m8, m.LEFT), m.FadeIn(m8v)),
            m.AnimationGroup(m.GrowFromEdge(c8, m.LEFT), m.FadeIn(c8v), code_rows[1].animate.set_opacity(1)),
        )
        self.say(
            "[But with its scales and zero points, it's eight and a half bits]: [a ceiling of sixty-four], [and again, ninety-one percent.]",
            code_rows[2].animate.set_opacity(1),
            m.AnimationGroup(m.Transform(c8, c85), m.ReplacementTransform(c8v, c85v)),
            m.FadeIn(p8, scale=0.8),
        )
        self.say(
            "[One point eight eight times the speed, for one point eight eight times fewer bytes], [with both streaming at about eighty-five percent of the bandwidth.]",
            m.FadeIn(ratio, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(mbu, shift=0.1 * m.UP), m.FadeIn(mbu_c)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 5 · weight-only vs weights + activations

    def two_kinds(self):
        self.next_section("5 · Two kinds", notes=(
            f"Weight-only (W8A16): 8-bit weights, dequantized to 16 bits inside the matmul — fewer bytes, same math (notes §8). "
            f"Weight + activation (W8A8, INT8 or FP8): the multiply runs on 8-bit tensor cores — H100 SXM {FP8_HW['peak'] / 1e12:,.0f} "
            f"vs {H_PEAK / 1e12:.0f} TFLOP/s dense (NVIDIA lists 3,958 and 1,979 'with sparsity'; dated Oct 2026). Roofline "
            f"(owl_reference.step_time, weights only, no KV): batch 1 intensity {intensity(1, 16):.0f} → {intensity(1, 8):.0f} "
            f"FLOPs/byte, ×{CHAT_GAIN:.1f}; a batch of {BATCH} (Episode 4's regime, near the ridge {RIDGE_H:.0f}): "
            f"{intensity(BATCH, 16):.0f} → {intensity(BATCH, 8):.0f} FLOPs/byte, capped by the BF16 roof: ×{BATCH_W8:.3f} "
            f"(+{(BATCH_W8 - 1) * 100:.0f}%); with FP8 math the roof doubles (ridge {RIDGE_F8:.0f}): ×{BATCH_F8:.1f}."))
        kw = dict(width=6.8, height=4.3, x_decades=(-1, 4), y_decades=(11, 15.5), center=(0, -0.3, 0))
        roof = Roofline(H_BW, H_PEAK, **kw)
        f8 = Roofline(H_BW, FP8_HW["peak"], **kw)
        assert np.allclose(roof.origin, f8.origin)
        dx = -2.55 - roof.get_left()[0]          # the chart's left edge (tick labels) at x = -2.55
        for rl in (roof, f8):
            rl.shift(dx * m.RIGHT)
            rl.origin = rl.origin + dx * m.RIGHT
        y_title = code("FLOP/s · H100", 20, MUTED).next_to(roof.axes[1].get_end(), m.RIGHT, buff=0.15)
        bf_lab = m.VGroup(code("16-bit math", 20, COMPUTE), code(f"{H_PEAK / 1e12:.0f} TF/s", 20, COMPUTE))
        bf_lab.arrange(m.DOWN, aligned_edge=m.RIGHT, buff=0.08).next_to(roof.compute, m.DOWN, buff=0.28)
        bf_lab.align_to(roof.compute, m.RIGHT)
        ext = m.Line(roof.point(RIDGE_H), f8.point(RIDGE_F8)).set_stroke(MEMORY, 6)
        f8_roof = f8.compute.copy()
        f8_lab = m.VGroup(code("8-bit math", 20, COMPUTE, weight=m.BOLD), code(f"{FP8_HW['peak'] / 1e12:,.0f} TF/s", 20, COMPUTE, weight=m.BOLD))
        f8_lab.arrange(m.DOWN, aligned_edge=m.RIGHT, buff=0.08).next_to(f8.compute, m.UP, buff=0.14).align_to(f8.compute, m.RIGHT)
        x_r = roof.point(RIDGE_H)[0]
        r_lab = code(f"ridge {RIDGE_H:.0f}", 20, MUTED).next_to(roof.at(RIDGE_H, 10 ** 11), m.UP, buff=0.12)
        r_lab.align_to([x_r + 0.12, 0, 0], m.LEFT)

        chat = roof.dot(intensity(1, 16), MEMORY)
        chat_lab = code("chat · batch 1", 20, MEMORY, weight=m.BOLD).next_to(chat, m.DR, buff=0.08)
        chat_to = roof.point(intensity(1, 8))
        x2a = code(f"×{CHAT_GAIN:.0f}", 24, TOKEN, weight=m.BOLD).move_to(chat_to + np.array([-0.55, 0.32, 0]))
        batch = roof.dot(intensity(BATCH, 16), REQUEST)
        b_lab = code(f"batch {BATCH} · ep 4", 20, REQUEST, weight=m.BOLD).next_to(batch, m.LEFT, buff=0.3).shift(0.35 * m.UP)
        b_to = roof.point(intensity(BATCH, 8))
        b_up = f8.point(intensity(BATCH, 8))
        plus = code(f"batch: +{(BATCH_W8 - 1) * 100:.0f}%", 20, RED, weight=m.BOLD)
        plus.next_to(bf_lab, m.DOWN, buff=0.3).align_to(bf_lab, m.RIGHT)
        x2b = code(f"batch: ×{BATCH_F8:.0f}", 20, TOKEN, weight=m.BOLD).move_to(plus).align_to(plus, m.RIGHT)
        guide = m.DashedLine(b_to, b_up, dash_length=0.05).set_stroke(TOKEN, 2.5)

        c1 = info_card("Weight-only", "W8A16", ["8-bit weights, 16-bit math", "cuts the bytes"], WEIGHTS, width=3.6)
        c2 = info_card("Weights + activations", "W8A8", ["8-bit math on tensor cores", "cuts the math time"], COMPUTE, width=3.6)
        cards = m.VGroup(c1, c2).arrange(m.DOWN, buff=0.35).move_to([-4.75, -0.2, 0])
        spec = cite("NVIDIA H100 SXM, dense").next_to(cards, m.DOWN, buff=0.22).align_to(cards, m.LEFT)

        chart = m.VGroup(roof.axes, roof.ticks, roof.x_label, roof.memory, roof.compute, roof.ridge)
        self.say(
            "[There are two ways to use eight bits.] [Weight-only stores the weights small, and widens them back to sixteen bits inside the multiply.]",
            m.AnimationGroup(self.new_header("05 · two kinds", "Fewer bytes, or faster math"), m.FadeIn(chart), m.FadeIn(y_title),
                             m.FadeIn(bf_lab), m.FadeIn(r_lab)),
            m.FadeIn(c1, shift=0.15 * m.RIGHT),
        )
        self.say(
            "[Half the bytes means twice the FLOPs per byte], [so Owl's chat dot climbs the memory slope]: [twice as fast.]",
            m.AnimationGroup(m.FadeIn(chat, scale=0.4), m.FadeIn(chat_lab)),
            chat.animate(run_time=1.3).move_to(chat_to),
            m.FadeIn(x2a, shift=0.1 * m.UP),
        )
        self.say(
            "[But batching, from episode four, pushes a server toward the ridge.] [At a batch of two hundred and fifty-six], [fewer bytes buy only fifteen percent.]",
            m.AnimationGroup(m.FadeIn(batch, scale=0.4), m.FadeIn(b_lab), m.Circumscribe(r_lab, color=MUTED, buff=0.06)),
            batch.animate(run_time=1.2).move_to(b_to),
            m.FadeIn(plus, shift=0.1 * m.UP),
        )
        self.say(
            "[Weight-and-activation quantization does the multiply itself in eight bits.] [An H100's eight-bit tensor cores do twice the math], [so the roof rises, and the batch runs twice as fast.]",
            m.FadeIn(c2, shift=0.15 * m.RIGHT),
            m.AnimationGroup(m.Create(ext), m.Create(f8_roof), m.FadeIn(f8_lab), m.FadeIn(spec)),
            m.AnimationGroup(batch.animate(run_time=1.2).move_to(b_up), m.Create(guide), m.ReplacementTransform(plus, x2b)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 6 · outliers

    def act_grid(self):
        """The activation matrix X (4 tokens × 8 channels), built once and kept into section 7."""
        if getattr(self, "xg", None) is None:
            w, h, px, py = 0.8, 0.46, 0.84, 0.55
            self.xgeo = (w, h, px, py, -6.55)
            rows = m.VGroup()
            for r, vals in enumerate(X):
                strings = [f"{v:.0f}" if j == OUT else fmt1(v) for j, v in enumerate(vals)]
                rows.add(cell_row(strings, -6.55, 1.75 - r * py, w, h, px, TOKEN, GREEN_FILL, size=19))
            self.xg = rows
        return self.xg

    def outliers(self):
        self.next_section("6 · Outliers", notes=(
            f"Activations (illustrative 4 tokens × 8 channels): ordinary values ±0.5, channel 4 ≈ {GIANT:.0f}× larger in every token "
            "(SmoothQuant, Xiao et al. 2023: outliers '~100× larger than most of the activation values'; 'If one channel has an "
            f"outlier, it persistently appears in all tokens'). Token 1 with one symmetric INT8 scale: {np.abs(X[0]).max():.0f} ÷ "
            f"{QMAX} = {T1_S:.3f}; integers {T1_Q.tolist()}; ordinary values read back off by {T1_ERR * 100:.0f}% on average. "
            "LLM.int8() (Dettmers et al. 2022) 'isolates the outlier feature dimensions into a 16-bit matrix multiplication while "
            "still more than 99.9% of values are multiplied in 8-bit'."))
        xg = self.act_grid()
        w, h, px, py, x0 = self.xgeo
        col = m.VGroup(*[row[OUT] for row in xg])
        x_cap = code("activations X", 20, TOKEN, weight=m.BOLD).move_to([0, 2.35, 0]).align_to([x0, 0, 0], m.LEFT)
        frame = m.SurroundingRectangle(col, buff=0.08, corner_radius=0.08).set_stroke(RED, 3)

        rx = 0.6
        g1 = text(f"channel 4: about {GIANT:.0f}× the rest", 24, RED, weight=m.BOLD)
        g2 = text("the same channel, in every token", 22, MUTED)
        g3 = cite("Xiao et al., 2023 (SmoothQuant)")
        gtxt = m.VGroup(g1, g2, g3).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.14).move_to([0, 1.25, 0]).align_to([rx, 0, 0], m.LEFT)

        y_q, y_b = -0.85, -1.75
        qrow = cell_row([str(v) for v in T1_Q], x0, y_q, w, h, px, TOKEN, CARD, size=19)
        brow = cell_row([f"{v:.1f}" if j == OUT else f"{v:.2f}" for j, v in enumerate(T1_BACK)], x0, y_b, w, h, px, EDGE, CARD, size=18)
        for row in (qrow, brow):
            row[OUT][0].set_stroke(RED, 2)
        q_lab = code(f"one scale: {np.abs(X[0]).max():.0f} ÷ {QMAX} = {T1_S:.3f}", 22, INK).move_to([0, y_q, 0]).align_to([rx, 0, 0], m.LEFT)
        q_sub = text("token 1: ordinary values become −1, 0 or 1", 20, RED).next_to(q_lab, m.DOWN, buff=0.1).align_to(q_lab, m.LEFT)
        b_lab = code(f"read back: off by {T1_ERR * 100:.0f}%", 22, RED, weight=m.BOLD).move_to([0, y_b, 0]).align_to([rx, 0, 0], m.LEFT)
        b_lab.shift(0.15 * m.DOWN)

        fix_box = card(13.1, 0.9, EDGE, CARD, radius=0.16, stroke=2)
        fix_t = code("LLM.int8()", 22, INK, weight=m.BOLD)
        p16 = pill("outlier columns: 16-bit", RED, RED_FILL, 20)
        p8 = pill("the rest: 8-bit, over 99.9%", TOKEN, GREEN_FILL, 20)
        fix_c = cite("Dettmers et al., 2022")
        fix_row = m.VGroup(fix_t, p16, p8, fix_c).arrange(m.RIGHT, buff=0.3)
        fix = m.VGroup(fix_box, fix_row.move_to(fix_box)).move_to([0, -3.0, 0])

        self.say(
            "[Activations are harder than weights.] [In big models, a few channels run about a hundred times larger than the rest], [in the same place, in every token.]",
            m.AnimationGroup(self.new_header("06 · outliers", "One column breaks INT8"), m.FadeIn(x_cap),
                             m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.UP) for r in xg], lag_ratio=0.2)),
            m.AnimationGroup(*[c[0].animate.set_stroke(RED, 2.5).set_fill(RED_FILL, 1) for c in col], m.FadeIn(g1), m.FadeIn(g3)),
            m.AnimationGroup(m.Create(frame), m.FadeIn(g2, shift=0.1 * m.UP)),
        )
        self.say(
            "[Quantize one token with one eight-bit scale], [and the giant claims the whole range]: [every ordinary value rounds to minus one, zero, or one.]",
            m.AnimationGroup(m.Circumscribe(xg[0], color=INK, buff=0.06), m.FadeIn(q_lab, shift=0.1 * m.UP)),
            m.Circumscribe(xg[0][OUT], color=RED, buff=0.06),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(q, target_position=c) for q, c in zip(qrow, xg[0])], lag_ratio=0.1),
                             m.FadeIn(q_sub)),
        )
        self.say(
            "[Read back, they're off by half their size, on average.] [One fix, from twenty twenty-two, pulls those columns out into sixteen bits], [and keeps the rest in eight.]",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(b, target_position=q) for b, q in zip(brow, qrow)], lag_ratio=0.1),
                             m.FadeIn(b_lab, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(fix_box), m.FadeIn(fix_t), m.FadeIn(fix_c), m.FadeIn(p16, shift=0.1 * m.UP)),
            m.FadeIn(p8, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.lower = m.VGroup(gtxt, qrow, brow, q_lab, q_sub, b_lab, fix)
        self.x_extras = m.VGroup(x_cap, frame)

    # ------------------------------------------------------------------ 7 · SmoothQuant

    def smooth(self):
        self.next_section("7 · SmoothQuant", notes=(
            f"SmoothQuant (Xiao et al., ICML 2023): s_j = max|X_j|^α ÷ max|W_j|^(1−α), α = 0.5; X ← X ÷ s, W ← s · W, so "
            f"X·W is unchanged (asserted). Here s = {SMOOTH.tolist()} (the example's ordinary channels were chosen so their "
            f"factor is exactly 1): channel 4 gets √({np.abs(X[:, OUT]).max():.0f} ÷ {np.abs(W[OUT]).max():.1f}) = {S_OUT:.0f}. "
            f"After: the giant is {XS_RATIO:.0f}× its neighbours in X and {WS_RATIO:.0f}× in W. Token 1 with one INT8 scale "
            f"({S1_S:.4f}): integers {S1_Q.tolist()}, ordinary values off by {S1_ERR * 100:.1f}% on average (was "
            f"{T1_ERR * 100:.0f}%). Paper: 'up to 1.56x speedup and 2x memory reduction for LLMs with negligible loss in accuracy'."))
        xg = self.xg
        w, h, px, py, x0 = self.xgeo
        col = m.VGroup(*[row[OUT] for row in xg])
        wx0, wpy, wh, ww, wpx = 0.85, 0.5, 0.42, 0.74, 0.8
        wg = m.VGroup(*[cell_row([fmt1(v) for v in W[r]], wx0, 1.95 - r * wpy, ww, wh, wpx, WEIGHTS, BLUE_FILL, size=19)
                        for r in range(W.shape[0])])
        w_cap = code("weights W", 20, WEIGHTS, weight=m.BOLD).next_to(wg, m.UP, buff=0.14).align_to(wg, m.LEFT)
        times = title_text("×", 44, MUTED).move_to([(xg.get_right()[0] + wg.get_left()[0]) / 2, xg.get_y(), 0])
        wrow = wg[OUT]
        div = code(f"÷ {S_OUT:.0f}", 22, TOKEN, weight=m.BOLD).next_to(col, m.UP, buff=0.12)
        mul = code(f"row 4 × {S_OUT:.0f}", 22, WEIGHTS, weight=m.BOLD).next_to(w_cap, m.RIGHT, buff=0.35)
        sit(mul, baseline(w_cap))

        x_swaps, x_new = [], []
        for r, row in enumerate(xg):
            anim, new = relabel(row[OUT], fmt1(XS[r, OUT]), 19)
            x_swaps.append(anim)
            x_new.append(new)
        w_swaps = [relabel(c, fmt1(WS[OUT, j]), 19)[0] for j, c in enumerate(wrow)]
        recolor = [c[0].animate.set_stroke(TOKEN, 3).set_fill(GREEN_FILL, 1) for c in col]

        rx = 3.5
        eq1 = code("X · W", 22, INK, t2c={"X": TOKEN, "W": WEIGHTS})
        eq2 = code("= (X÷s) · (s×W)", 22, INK, t2c={"X": TOKEN, "W": WEIGHTS, "s": INK})
        eq = m.VGroup(eq1, eq2).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.16).move_to([0, 1.45, 0]).align_to([rx, 0, 0], m.LEFT)
        same = pill("same product", TOKEN, GREEN_FILL, 20).next_to(eq, m.DOWN, buff=0.25).align_to(eq, m.LEFT)
        s_eq = code(f"s = √({np.abs(X[:, OUT]).max():.0f}÷{np.abs(W[OUT]).max():.1f}) = {S_OUT:.0f}", 20, MUTED)
        s_eq.next_to(same, m.DOWN, buff=0.22).align_to(eq, m.LEFT)
        ten = m.VGroup(code(f"X: {GIANT:.0f}× → {XS_RATIO:.0f}×", 22, TOKEN, weight=m.BOLD),
                       code(f"W:   1× → {WS_RATIO:.0f}×", 22, WEIGHTS, weight=m.BOLD))
        ten.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.1).next_to(s_eq, m.DOWN, buff=0.28).align_to(eq, m.LEFT)

        y_q = -0.85
        srow = cell_row([str(v) for v in S1_Q], x0, y_q, w, h, px, TOKEN, CARD, size=19)
        s_lab = code(f"token 1 now: scale {S1_S:.3f}", 20, INK).next_to(srow, m.DOWN, buff=0.2).align_to(srow, m.LEFT)
        s_err = pill(f"off by {S1_ERR * 100:.0f}%, not {T1_ERR * 100:.0f}%", TOKEN, GREEN_FILL, 22)
        s_err.next_to(s_lab, m.DOWN, buff=0.25).align_to(srow, m.LEFT)
        paper = m.VGroup(text("up to 1.56× faster, 2× less memory, negligible loss", 22, INK),
                         cite("Xiao et al., 2023 · SmoothQuant, INT8 weights and activations"))
        paper.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.1).move_to([0, -3.0, 0]).align_to([x0, 0, 0], m.LEFT)

        self.say(
            "[The SmoothQuant fix is a rescale that changes nothing]: [divide the giant column by ten], [and multiply the matching row of weights by ten.]",
            m.AnimationGroup(self.new_header("07 · smoothquant", "Move the difficulty"), m.FadeOut(self.lower), m.FadeOut(self.x_extras[1]),
                             m.FadeIn(times), m.FadeIn(w_cap), m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.LEFT) for r in wg], lag_ratio=0.08)),
            m.AnimationGroup(m.FadeIn(div, shift=0.1 * m.DOWN), m.LaggedStart(*x_swaps, lag_ratio=0.15), *recolor),
            m.AnimationGroup(m.FadeIn(mul, shift=0.1 * m.LEFT), m.LaggedStart(*w_swaps, lag_ratio=0.15),
                             m.Circumscribe(wrow, color=WEIGHTS, buff=0.06)),
        )
        self.say(
            "[The product stays exactly the same], [but now the giant is only ten times its neighbours, on both sides, instead of a hundred times on one.]",
            m.AnimationGroup(m.FadeIn(eq, shift=0.1 * m.UP), m.FadeIn(same, shift=0.1 * m.UP), m.FadeIn(s_eq)),
            m.FadeIn(ten, shift=0.1 * m.UP),
        )
        self.say(
            "[Quantize the same token again], [and it's off by about five percent, not half.] [The paper reports up to one point five six times faster, with negligible loss.]",
            m.AnimationGroup(m.Circumscribe(xg[0], color=INK, buff=0.06),
                             m.LaggedStart(*[m.FadeIn(q, target_position=c) for q, c in zip(srow, xg[0])], lag_ratio=0.1), m.FadeIn(s_lab)),
            m.FadeIn(s_err, shift=0.1 * m.UP),
            m.FadeIn(paper, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()
        self.xg = None

    # ------------------------------------------------------------------ 8 · FP8

    def fp8(self):
        self.next_section("8 · FP8", notes=(
            f"FP8 (Micikevicius et al. 2022, arXiv 2209.05433): E4M3 = 1 sign + 4 exponent + 3 mantissa bits, max normal "
            f"1.75 × 2^8 = {E4M3_MAX:.0f}, no infinities; E5M2 = 1 + 5 + 2, max {E5M2_MAX:,.0f}, IEEE-like; 'E4M3 for weight and "
            f"activation tensors, and E5M2 for gradient tensors'. BF16 = 1 + 8 + 7. Token 1 (giant 47) with one scale per "
            f"token: INT8 levels within ±{SPAN}: {len(INT8_LEVELS)}; E4M3 levels (47 ↦ 448): {len(F8_LEVELS)}; ordinary values "
            f"off by {T1_ERR * 100:.0f}% (INT8) vs {T1_F8_ERR * 100:.1f}% (FP8) on average. Kurtic et al. (ACL 2025, 'Give Me BF16 "
            "or Give Me Death', Llama-3.1, over 500,000 evaluations): 'FP8 (W8A8-FP) is effectively lossless across all model "
            "scales'; 'well-tuned INT8 (W8A8-INT) achieves surprisingly low (1-3%) accuracy degradation'."))
        bx, lab_x = -3.6, -6.55
        y16, y43, y52 = 1.95, 1.2, 0.45
        L16, L43, L52 = "s" + "e" * 8 + "m" * 7, "s" + "e" * 4 + "m" * 3, "s" + "e" * 5 + "m" * 2
        r16, r43, r52 = bit_row(L16, bx, y16), bit_row(L43, bx, y43), bit_row(L52, bx, y52)
        n16 = code("BF16", 22, INK, weight=m.BOLD).move_to([0, y16, 0]).align_to([lab_x, 0, 0], m.LEFT)
        n43 = m.VGroup(code("FP8", 22, WEIGHTS, weight=m.BOLD), code("E4M3", 20, MUTED)).arrange(m.RIGHT, buff=0.2)
        n43.move_to([0, y43, 0]).align_to([lab_x, 0, 0], m.LEFT)
        n52 = m.VGroup(code("FP8", 22, WEIGHTS, weight=m.BOLD), code("E5M2", 20, MUTED)).arrange(m.RIGHT, buff=0.2)
        n52.move_to([0, y52, 0]).align_to([lab_x, 0, 0], m.LEFT)
        for n in (n43, n52):
            sit(n[1], baseline(n[0]))
        ix = 2.95
        i16 = code("16 bits", 20, MUTED).move_to([0, y16, 0]).align_to([ix, 0, 0], m.LEFT)
        i43 = code(f"max {E4M3_MAX:.0f}", 22, INK, weight=m.BOLD).move_to([0, y43, 0]).align_to([ix, 0, 0], m.LEFT)
        i52 = m.VGroup(code(f"max {E5M2_MAX:,.0f}", 22, INK, weight=m.BOLD), code("for gradients", 20, MUTED))
        i52.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.08).move_to([0, y52, 0]).align_to([ix, 0, 0], m.LEFT)

        def swatch(kind, label):
            color, fill = BIT_STYLE[kind]
            sq = m.RoundedRectangle(corner_radius=0.04, width=0.26, height=0.32).set_stroke(color, 2).set_fill(fill, 1)
            return m.VGroup(sq, text(label, 20, MUTED)).arrange(m.RIGHT, buff=0.15)

        lg_s, lg_e, lg_m = swatch("s", "sign"), swatch("e", "exponent: range"), swatch("m", "mantissa: precision")
        legend = m.VGroup(lg_s, lg_e, lg_m).arrange(m.RIGHT, buff=0.6).move_to([0, -0.25, 0]).align_to([bx, 0, 0], m.LEFT)

        lo, hi, a0, a1 = -SPAN, SPAN, -3.6, 2.4

        def X_(v):
            return a0 + (v - lo) / (hi - lo) * (a1 - a0)

        def level_line(y, levels, color):
            axis = m.Line([a0, y, 0], [a1, y, 0]).set_stroke(FAINT, 2.5)
            ticks = m.VGroup(*[m.Line([X_(v), y - 0.17, 0], [X_(v), y + 0.17, 0]).set_stroke(color, 1.6) for v in levels])
            return m.VGroup(axis, ticks)

        y_i, y_f = -1.6, -2.6
        li = level_line(y_i, INT8_LEVELS, RED)
        lf = level_line(y_f, F8_LEVELS, TOKEN)
        cap = code(f"token 1, the giant ({np.abs(X[0]).max():.0f}) sets the scale · values from −{SPAN} to {SPAN}", 20, MUTED)
        cap.move_to([0, -0.85, 0]).align_to([lab_x, 0, 0], m.LEFT)
        ni = code("INT8", 22, INK, weight=m.BOLD).move_to([0, y_i, 0]).align_to([lab_x, 0, 0], m.LEFT)
        nf = m.VGroup(code("FP8", 22, WEIGHTS, weight=m.BOLD), code("E4M3", 20, MUTED)).arrange(m.RIGHT, buff=0.2)
        nf.move_to([0, y_f, 0]).align_to([lab_x, 0, 0], m.LEFT)
        sit(nf[1], baseline(nf[0]))
        ords = [X[0][j] for j in ORD]
        di = m.VGroup(*[m.Dot([X_(v), y_i + 0.32, 0], radius=0.06, color=INK) for v in ords])
        df = m.VGroup(*[m.Dot([X_(v), y_f + 0.32, 0], radius=0.06, color=INK) for v in ords])
        ri = code(f"{len(INT8_LEVELS)} levels · off {T1_ERR * 100:.0f}%", 20, RED, weight=m.BOLD).next_to(li, m.RIGHT, buff=0.3)
        rf = code(f"{len(F8_LEVELS)} levels · off {T1_F8_ERR * 100:.0f}%", 20, TOKEN, weight=m.BOLD).next_to(lf, m.RIGHT, buff=0.3)
        lines = m.VGroup(cap, li, lf, ni, nf, di, df, ri, rf)

        k_t = text("Llama-3.1 · over 500,000 evaluations", 24, INK, weight=m.BOLD)
        k_1 = pill("FP8 weights + activations: effectively lossless", TOKEN, GREEN_FILL, 22)
        k_2 = pill("well-tuned INT8: 1–3% lower accuracy", INK, CARD_HI, 22)
        k_c = cite("Kurtic et al., 2025 · Give Me BF16 or Give Me Death")
        verdict = m.VGroup(k_t, k_1, k_2, k_c).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.24).move_to([0, -2.0, 0])
        verdict.align_to([lab_x, 0, 0], m.LEFT)

        self.say(
            "[The other eight-bit option is a tiny float]: [FP8.] [Like sixteen-bit, it spends some bits on an exponent, for range], [and the rest on a mantissa, for precision.]",
            m.AnimationGroup(self.new_header("08 · fp8", "A tiny float"), m.FadeIn(n16), m.FadeIn(r16), m.FadeIn(i16)),
            m.AnimationGroup(m.FadeIn(n43), m.LaggedStart(*[m.FadeIn(b, shift=0.1 * m.UP) for b in r43], lag_ratio=0.1)),
            m.AnimationGroup(m.FadeIn(lg_s), m.FadeIn(lg_e), m.Circumscribe(part(r16, L16, "e"), color=WEIGHTS, buff=0.05),
                             m.Circumscribe(part(r43, L43, "e"), color=WEIGHTS, buff=0.05)),
            m.AnimationGroup(m.FadeIn(lg_m), m.Circumscribe(part(r16, L16, "m"), color=INK, buff=0.05),
                             m.Circumscribe(part(r43, L43, "m"), color=INK, buff=0.05)),
        )
        self.say(
            "[Its usual flavour has four exponent bits and three mantissa bits], [reaching four hundred and forty-eight.] [A five-and-two flavour reaches fifty-seven thousand, for gradients in training.]",
            m.Circumscribe(m.VGroup(n43, r43), color=WEIGHTS, buff=0.08),
            m.FadeIn(i43, shift=0.1 * m.LEFT),
            m.AnimationGroup(m.FadeIn(n52), m.FadeIn(r52, shift=0.1 * m.UP), m.FadeIn(i52, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[A float's levels crowd near zero], [so even with the giant at forty-seven], [the ordinary values get over a hundred levels, and are off by about three percent.]",
            m.AnimationGroup(m.FadeIn(cap), m.FadeIn(ni), m.FadeIn(nf), m.Create(li[0]), m.Create(lf[0]),
                             m.LaggedStart(*[m.FadeIn(t) for t in li[1]], lag_ratio=0.2),
                             m.LaggedStart(*[m.FadeIn(t) for t in lf[1]], lag_ratio=0.01)),
            m.AnimationGroup(m.FadeIn(di, shift=0.15 * m.DOWN), m.FadeIn(df, shift=0.15 * m.DOWN)),
            m.AnimationGroup(m.FadeIn(ri, shift=0.1 * m.LEFT), m.FadeIn(rf, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[Across over half a million evaluations of Llama three point one], [FP8 was effectively lossless], [and well-tuned INT8 lost one to three percent.]",
            m.AnimationGroup(m.FadeOut(lines, shift=0.2 * m.DOWN), m.FadeIn(k_t, shift=0.1 * m.UP), m.FadeIn(k_c)),
            m.FadeIn(k_1, shift=0.1 * m.UP),
            m.FadeIn(k_2, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 9 · recap

    def recap(self):
        self.next_section("9 · Recap", notes=(
            f"Fewer bits per weight = fewer bytes per token = faster memory-bound decode. On this Mac: {TOK16:.1f} → {TOK8:.1f} "
            f"tok/s ({SPEEDUP:.2f}×) for {W16 / GB:.2f} → {DISK8 / GB:.2f} GB ({FEWER:.2f}× fewer bytes). Three words: shrink "
            "(the bits), scale (per group, with a zero point), smooth (the outliers: SmoothQuant, or a float). Next: Episode 19, "
            "four bits and below."))
        x0, full = -3.2, 5.6
        rows = m.VGroup()
        for y, name, gb, tok in ((1.55, "16-bit", W16, TOK16), (0.6, "8-bit", DISK8, TOK8)):
            lab = code(name, 24, WEIGHTS, weight=m.BOLD).move_to([0, y, 0]).align_to([-6.3, 0, 0], m.LEFT)
            bar = hbar(x0, y, full * gb / W16, 0.5, WEIGHTS, BLUE_FILL)
            gbt = code(f"{gb / GB:.2f} GB", 20, WEIGHTS).move_to(bar).align_to(bar, m.LEFT).shift(0.15 * m.RIGHT)
            gbt.set_z_index(2)
            rate = stat(f"{tok:.1f}", "tok/s", "", TOKEN, 48).move_to([0, y, 0]).align_to([3.4, 0, 0], m.LEFT)
            rows.add(m.VGroup(lab, bar, gbt, rate))
        both = pill(f"{FEWER:.2f}× fewer bytes  →  {SPEEDUP:.2f}× faster · measured on this Mac", TOKEN, GREEN_FILL, 22).move_to([0, -0.5, 0])

        halo = glow(WEIGHTS, radius=3.4).move_to([0, 0.55, 0])
        words = m.VGroup(title_text("Shrink", 76, WEIGHTS), title_text("Scale", 76, INK), title_text("Smooth", 76, TOKEN))
        words.arrange(m.RIGHT, buff=1.3).move_to([0, 0.7, 0])
        on_baseline(words)
        a1 = arrow(words[0].get_right(), words[1].get_left(), MUTED, buff=0.25)
        a2 = arrow(words[1].get_right(), words[2].get_left(), MUTED, buff=0.25)
        subs = m.VGroup(text("fewer bits per weight", 22, MUTED), text("a scale and zero point per group", 22, MUTED),
                        text("tame the outliers", 22, MUTED))
        for s, wd in zip(subs, words):
            s.next_to(wd, m.DOWN, buff=0.3)
        tag = formula([("tok/s", INK), ("≈", MUTED), ("bandwidth", MEMORY), ("÷", MUTED), ("fewer bytes", WEIGHTS)], 40).move_to([0, -1.55, 0])
        nxt = pill("below 8 bits: episode 19", WEIGHTS, BLUE_FILL, 22).next_to(tag, m.DOWN, buff=0.4)
        mascot = owl(0.75).move_to([5.6, -2.75, 0])

        self.say(
            "[Fewer bits per weight means fewer bytes per token]: [on this Mac, eight bits made Owl one point eight eight times faster.]",
            m.AnimationGroup(self.new_header("09 · recap", "Shrink, scale, smooth"),
                             m.LaggedStart(*[m.AnimationGroup(m.FadeIn(r[0]), m.GrowFromEdge(r[1], m.LEFT), m.FadeIn(r[2])) for r in rows], lag_ratio=0.4)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(r[3], shift=0.1 * m.LEFT) for r in rows], lag_ratio=0.4), m.FadeIn(both, shift=0.1 * m.UP)),
        )
        self.say(
            "[Three words to keep]: [shrink], [scale], [smooth.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(rows, both), shift=0.2 * m.UP), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.2 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.GrowArrow(a1), m.FadeIn(words[1], shift=0.2 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.GrowArrow(a2), m.FadeIn(words[2], shift=0.2 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[Below eight bits, the scales have to get clever], [and that's where we go next.]",
            m.FadeIn(tag, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(nxt, shift=0.1 * m.UP), m.FadeIn(mascot, shift=0.2 * m.UP)),
        )
        self.wait(0.6)
        self.clear()
