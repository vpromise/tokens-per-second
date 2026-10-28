"""Tokens per Second · Episode 19 · Four Bits and Below (Module 5 · Fewer Bits, More GPUs)

Four-bit formats keep quality with clever scales, and the scales are bytes too. Running example: Owl-8B
(Llama-3.1-8B: 8.03 B parameters, 16.06 GB of BF16 weights) shrinking to about 4.5 GB, measured on this Mac.

Sections: hook (Episode 18's 8-bit Owl, 16.06 → 8.53 GB; today 4-bit, 4.52 GB; tok/s ≈ bandwidth ÷ bytes, Episode
2) · groups (16 INT4 levels; one scale and offset per row vs per group; a sample row: error 31% → 9%) · the ledger
(64 × 4 + 16 + 16 bits = 4.5 bits per weight; 4.25 with a scale alone; 4.5 for 32 weights + one scale; Owl:
4.02 GB + 0.50 GB = 4.52 GB, the file on this Mac 4.517 GB) · smarter rounding (GPTQ column by column with error
compensation; AWQ scales salient channels; Kurtic et al.: 4-bit weight-only rivals 8-bit) · GGUF k-quants
(256-weight super-blocks of 8 × 32, 6-bit scales and mins, a 16-bit pair per super-block: 1,152 bits ÷ 256 =
4.5; Q4_1 costs 5.0; the ladder Q2_K … Q6_K with Owl sizes; perplexity +0.9% for Q4_K_M, +15% for Q2_K) ·
micro-scaled floats (E2M1 levels; MXFP4 4.25, NVFP4 4.5; Blackwell FP4 math; gpt-oss in MXFP4) · measured on this
Mac (decode 31.0 / 58.4 / 101.9 tok/s vs ceilings 34.0 / 64.0 / 120.9; 136 without scales; 3.3× faster from
3.6× fewer bytes; 84% vs 91% of the ceiling; 4-bit mat-vec 431 vs 475 GB/s; prefill 940 vs 1,005 tok/s) ·
fewer weights (distillation: Owl-mini = Llama 3.2 1B, pruned from 8B, taught by 8B and 70B logits; 0.70 GB, 489
tok/s measured vs a 785 ceiling; 2:4 sparsity) · recap (group, protect, count).

Sources (facts dated October 2026):
- research/notes.md §8: group quantization overheads (g = 32 + FP16 scale 4.5 bpw; MLX g = 64 + scale + bias
  4.5; MXFP4 4.25; NVFP4 4.5), GPTQ (Frantar et al., ICLR 2023, arXiv 2210.17323: "approximately four GPU hours"
  for 175 B, 3–4 bits, 3.25× on A100), AWQ (Lin et al., MLSys 2024, arXiv 2306.00978: "Protecting only 1% salient
  weights can greatly reduce quantization error", salience from activations, per-channel scaling, no
  backpropagation), Kurtic et al. (ACL 2025, arXiv 2411.02355, "over 500,000 evaluations": "INT4 weight-only
  (W4A16-INT) is more competitive than expected, rivaling 8-bit quantization"), distillation (Meta, Llama 3.2,
  2024-09-25: structured pruning from Llama 3.1 8B, logits from 8B and 70B as token-level targets), 2:4 sparsity
  (NVIDIA blog, 2021: two of every four values zero, "the same effective calculation in half the time").
  Abstracts of GPTQ and AWQ re-read on arxiv.org on 2026-10-24.
- GGUF k-quants: llama.cpp PR #1684 (2023; descriptions and the LLaMA-7B perplexity table F16 5.9066, Q4_K_M
  5.9601, Q2_K 6.7764, Q6_K 5.9110; Q4_K_M = Q6_K for half of attention.wv and feed_forward.w2, else Q4_K) and
  ggml/src/ggml-common.h (block layouts; read 2026-10-24). Bits per weight below are computed from the layouts:
  Q2_K 2.625 (the PR text says 2.5625, the struct gives 2.625), Q3_K 3.4375, Q4_K 4.5, Q5_K 5.5, Q6_K 6.5625.
  Q4_K_M for Llama-3.1-8B: 4.89 bpw (llama.cpp quantize README, notes §8).
- OCP Microscaling Formats (MX) v1.0, 2023: MXFP4 = 32 E2M1 elements + an E8M0 (power-of-two) scale. NVIDIA,
  "Introducing NVFP4 …" (2025-06-24, read 2026-10-24): E2M1, 16-value micro-blocks with an E4M3 scale, a per-tensor
  FP32 scale; Blackwell Tensor Cores implement it; DeepSeek-R1-0528 FP8 → NVFP4 "1% or less accuracy degradation".
  OpenAI gpt-oss (2025): MoE weights post-trained in MXFP4; 120b fits a single 80 GB GPU (notes §8).
- research/measurements.md / .json (this Mac Studio, M4 Max 546 GB/s, mlx-lm 0.32.2, 2026-10-24), loaded with
  measured(): §1 decode (median of 5, 256 tokens), §0 roofs (4-bit vs bf16 mat-vec), §2 prefill at 2,048 tokens,
  models (file sizes; MLX affine quantization, group size 64, bf16 scale + bias).
- owl_reference.py: weight_bytes(bits=…), decode_ceiling(hw=MAC, bits=…), MODELS.

Illustrative (said "for example" or schematic): the 24 weights and their three groups of 8 in section 1 (the real
groups hold 64), the 4,096-weight sample row (seeded Gaussian, σ = 0.02, one weight set to 0.25), the GPTQ and
AWQ grids and activation bars, the 16- and 32-cell blocks, the bell curve, the 2:4 cells.

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

PREFIX = "ep19"
GB = 1e9
SCALE, SCALE_FILL = "#A9C9FF", "#1A2F52"  # scales and offsets: part of the weights, so a lighter blue
ROUNDED = "#2B4C80"  # a weight cell once it has been rounded to four bits

# Owl-8B's bytes at 16, 8.5 and 4.5 bits per weight (the bytes model) against the files measured on this Mac
P8 = OWL["params"]
W16, W4 = weight_bytes(), weight_bytes(bits=4)
FILE = {k: measured("models", "llama-3.1-8b", k, "weights_file_bytes") for k in ("bf16", "8bit", "4bit")}
assert round(W16 / GB, 2) == 16.06 and round(W4 / GB, 1) == 4.0  # "sixteen gigabytes", "four gigabytes"


def bpw(bits, group, extra_bits):
    """Bits per weight: the weight's own bits plus its share of the group's scales."""
    return bits + extra_bits / group


GROUP = 64
MLX_BPW = bpw(4, GROUP, 16 + 16)   # a BF16 scale and a BF16 offset per 64 weights (MLX, this Mac)
S64_BPW = bpw(4, 64, 16)           # one 16-bit scale per 64
S32_BPW = bpw(4, 32, 16)           # one 16-bit scale per 32 (GGUF Q4_0)
Q8_BPW = bpw(8, GROUP, 16 + 16)
assert (MLX_BPW, S64_BPW, S32_BPW, Q8_BPW) == (4.5, 4.25, 4.5, 8.5)  # "four and a half", "four and a quarter"
assert measured("models", "llama-3.1-8b", "4bit", "quantization") == {"group_size": GROUP, "bits": 4}
GROUP_BITS = GROUP * 4
assert GROUP_BITS == 256 and GROUP_BITS + 32 == 288  # "two hundred and fifty-six bits", "thirty-two more"
W45, W85 = weight_bytes(bits=MLX_BPW), weight_bytes(bits=Q8_BPW)
SCALES = W45 - W4
assert round(W45 / GB, 2) == 4.52 and round(SCALES / GB, 1) == 0.5  # "plus half a gigabyte"
for k, w in (("bf16", W16), ("8bit", W85), ("4bit", W45)):
    assert abs(FILE[k] - w) / w < 0.001  # the ledger balances: the bytes model matches every file on disk
assert round(FILE["4bit"] / GB, 2) == 4.52 and round(W85 / GB, 2) == 8.53
assert 0.5 < W85 / W16 < 0.55  # "nearly halved"

# section 1: one large weight in a row (a seeded sample row, for example)
_rng = np.random.default_rng(0)
ROW = _rng.normal(0, 0.02, 4096)
ROW[_rng.integers(4096)] = 0.25


def affine(w, group):
    """INT4 with a scale and an offset per group: 16 levels from the group's smallest weight to its largest."""
    g = w.reshape(-1, group)
    lo, hi = g.min(1, keepdims=True), g.max(1, keepdims=True)
    step = (hi - lo) / 15
    return (np.clip(np.round((g - lo) / step), 0, 15) * step + lo).ravel()


def rel_error(w, group):
    return float(np.linalg.norm(w - affine(w, group)) / np.linalg.norm(w))


ERR_ROW, ERR_GROUP = rel_error(ROW, ROW.size), rel_error(ROW, GROUP)
assert round(ERR_ROW * 100) == 31 and round(ERR_GROUP * 100) == 9  # "from thirty-one percent to nine"

# the 24 weights on screen (schematic, in three groups of 8)
VALS = [0.35, -0.2, 0.5, 2.2, -0.45, 0.1, 0.25, -0.3,
        -0.15, 0.4, -0.5, 0.2, 0.05, -0.35, 0.55, -0.05,
        0.3, -0.4, 0.15, -0.25, 0.45, -0.1, 0.6, -0.55]
NG = 8


def level_index(v, lo, hi):
    return int(np.clip(np.round((v - lo) / ((hi - lo) / 15)), 0, 15))


def snap(v, lo, hi):
    return lo + level_index(v, lo, hi) * (hi - lo) / 15


ROW_LO, ROW_HI = min(VALS), max(VALS)
GROUPS = [VALS[i:i + NG] for i in range(0, len(VALS), NG)]
SNAP_ROW = [snap(v, ROW_LO, ROW_HI) for v in VALS]
SNAP_GROUP = [snap(v, min(g), max(g)) for g in GROUPS for v in g]
USED = len({level_index(v, ROW_LO, ROW_HI) for v in VALS if v != ROW_HI})
assert USED == 7  # "crowd onto just seven of them"
assert max(abs(a - b) for a, b in zip(VALS[NG:], SNAP_GROUP[NG:])) < 0.5 * max(abs(a - b) for a, b in zip(VALS[NG:], SNAP_ROW[NG:]))

# GGUF k-quants: bits per weight from the block layouts (ggml-common.h)
QK = 256


def kquant(bits, blocks, scale_bits, with_min, halves):
    """A 256-weight super-block: the weights, a scale (and a min) per block, 16-bit numbers for the whole block."""
    return (QK * bits + blocks * scale_bits * (2 if with_min else 1) + 16 * halves) / QK


Q4K_PARTS = (QK * 4, 8 * (6 + 6), 2 * 16)
assert Q4K_PARTS == (1024, 96, 32) and sum(Q4K_PARTS) == 1152 and sum(Q4K_PARTS) / QK == 4.5
LADDER = [("Q2_K", kquant(2, 16, 4, True, 2)), ("Q3_K", kquant(3, 16, 6, False, 1)), ("Q4_K", kquant(4, 8, 6, True, 2)),
          ("Q4_K_M", 4.89), ("Q5_K", kquant(5, 8, 6, True, 2)), ("Q6_K", kquant(6, 16, 8, False, 1))]
assert [b for _, b in LADDER] == [2.625, 3.4375, 4.5, 4.89, 5.5, 6.5625]
Q41_BPW = bpw(4, 32, 16 + 16)
assert Q41_BPW == 5.0  # "the simpler format … costs five"
PPL = {"F16": 5.9066, "Q6_K": 5.9110, "Q4_K_M": 5.9601, "Q2_K": 6.7764}  # LLaMA-7B, wikitext (PR #1684)
PPL_UP = {k: (PPL[k] / PPL["F16"] - 1) * 100 for k in ("Q6_K", "Q4_K_M", "Q2_K")}
assert PPL_UP["Q4_K_M"] < 1 and round(PPL_UP["Q2_K"]) == 15  # "under one percent", "about fifteen"

# micro-scaled floats
E2M1 = sorted({(m / 2) if e == 0 else (1 + m / 2) * 2 ** (e - 1) for e in range(4) for m in range(2)})
assert E2M1 == [0, 0.5, 1, 1.5, 2, 3, 4, 6]  # "reaches out to six"
MX_BPW, NV_BPW = bpw(4, 32, 8), bpw(4, 16, 8)
assert (MX_BPW, NV_BPW) == (4.25, 4.5)

# measured on this Mac: decode against the bandwidth ÷ bytes ceiling
KEYS = ("bf16", "8bit", "4bit")
BITS = {"bf16": 16, "8bit": Q8_BPW, "4bit": MLX_BPW}
DEC = {k: measured("decode", "llama-3.1-8b", k) for k in KEYS}
TOK = {k: DEC[k]["tok_s_median"] for k in KEYS}
CEIL = {k: decode_ceiling(hw=MAC, bits=BITS[k]) for k in KEYS}
FLAT4 = decode_ceiling(hw=MAC, bits=4)  # the bytes model at a flat four bits, no scales
OF = {k: TOK[k] / CEIL[k] for k in KEYS}
for k in KEYS:
    assert abs(CEIL[k] - DEC[k]["ceiling_tok_s"]) / CEIL[k] < 0.002  # the same ceiling as 546 GB/s ÷ the file
assert [round(TOK[k]) for k in KEYS] == [31, 58, 102] and [round(CEIL[k]) for k in KEYS] == [34, 64, 121]
assert round(FLAT4) == 136 and [round(OF[k] * 100) for k in KEYS] == [91, 91, 84]  # "eighty-four … not ninety-one"
SPEEDUP = TOK["4bit"] / TOK["bf16"]
FEWER = FILE["bf16"] / FILE["4bit"]
assert round(SPEEDUP, 1) == 3.3 and round(FEWER, 1) == 3.6  # "three point three times … three point six"
MV4, MV16 = measured("roofs", "matvec_q4_GBps", "median"), measured("roofs", "matvec_bf16_GBps", "median")
assert (round(MV4), round(MV16)) == (431, 475)
PRE = {k: measured("prefill", "llama-3.1-8b", k, "by_prompt_tokens", "2048", "tok_s_median") for k in ("bf16", "4bit")}
assert (round(PRE["bf16"]), round(PRE["4bit"])) == (1005, 940)  # "nine hundred and forty … against a thousand"

# fewer weights: Owl-mini (Llama-3.2-1B, pruned and distilled from Llama-3.1-8B), 4-bit, on this Mac
MINI = MODELS["owl-mini"]
MINI_DEC = measured("decode", "llama-3.2-1b", "4bit")
MINI_GB = measured("models", "llama-3.2-1b", "4bit", "weights_file_bytes") / GB
assert round(MINI_GB, 2) == 0.70 and round(MINI_DEC["tok_s_median"]) == 489  # "seven hundred megabytes", "almost 490"
assert abs(weight_bytes(MINI, bits=MLX_BPW) / GB - MINI_GB) < 0.01
MINI_CEIL = MAC["bandwidth"] / (MINI_GB * GB)
assert round(MINI_CEIL) == round(MINI_DEC["ceiling_tok_s"]) == 785

# ---------------------------------------------------------------------------- look


def cite(s, size=20):
    return text(s, size, MUTED)


def cell(w, h, color=WEIGHTS, fill=BLUE_FILL, stroke=1.6, radius=0.04):
    return m.RoundedRectangle(corner_radius=radius, width=w, height=h, stroke_color=color, stroke_width=stroke,
                              fill_color=fill, fill_opacity=1)


def hbar(x0, y, w, h, color, fill, stroke=2):
    return m.Rectangle(width=max(w, 0.02), height=h).set_stroke(color, stroke).set_fill(fill, 1).move_to([x0 + w / 2, y, 0])


def bracket_under(x0, x1, y, color=MUTED, lip=0.12):
    return m.VGroup(m.Line([x0, y, 0], [x1, y, 0]), m.Line([x0, y, 0], [x0, y + lip, 0]),
                    m.Line([x1, y, 0], [x1, y + lip, 0])).set_stroke(color, 2)


def note_pill(words, color=INK, fill=CARD_HI, size=20):
    return pill(words, color, fill, size)


# ---------------------------------------------------------------------------- the episode


class Ep19FourBitsAndBelow(Episode):
    number = 19

    def lesson(self):
        self.hook()
        self.groups()
        self.ledger()
        self.rounding()
        self.kquants()
        self.floats()
        self.on_this_mac()
        self.fewer_weights()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes=f"Owl-8B ({P8 / 1e9:.2f} B parameters) in BF16 is {W16 / GB:.2f} GB; Episode 18's 8-bit "
                          f"(a scale and offset per 64 weights, {Q8_BPW} bits per weight) is {W85 / GB:.2f} GB; 4-bit ({MLX_BPW} bits per "
                          f"weight) is {W45 / GB:.2f} GB. The files on this Mac (mlx-community, MLX affine quantization): "
                          f"{FILE['bf16'] / GB:.3f} / {FILE['8bit'] / GB:.3f} / {FILE['4bit'] / GB:.3f} GB. Episode 2: at batch 1, decode "
                          "tok/s ≈ bandwidth ÷ bytes per token.")
        x0, per = -4.4, 8.6 / (W16 / GB)
        rows = []
        for i, (name, nbytes, y) in enumerate((("BF16", W16, 1.55), ("8-bit", W85, 0.65), ("4-bit", W45, -0.25))):
            lab = code(name, 24, WEIGHTS, weight=m.BOLD).move_to([-6.3, y, 0], aligned_edge=m.LEFT)
            bar = hbar(x0, y, per * nbytes / GB, 0.5, WEIGHTS, BLUE_FILL if i < 2 else WEIGHTS)
            val = code(f"{nbytes / GB:.2f} GB", 24, INK).next_to(bar, m.RIGHT, buff=0.25)
            rows.append(m.VGroup(lab, bar, val))
        ep18 = note_pill("Episode 18", MUTED, CARD, 20).next_to(rows[1][2], m.RIGHT, buff=0.4)
        ep2 = m.VGroup(
            code("tok/s ≈ bandwidth ÷ bytes per token", 30, INK, t2c={r"tok/s": TOKEN, r"bandwidth": MEMORY, r"bytes per token": MEMORY}),
            cite("Episode 2"),
        ).arrange(m.RIGHT, buff=0.45)
        ep2_box = card(ep2.width + 0.7, ep2.height + 0.5, EDGE, CARD, radius=0.16, stroke=2).move_to([0, -1.55, 0])
        ep2.move_to(ep2_box)
        formula = m.VGroup(ep2_box, ep2)
        q1 = note_pill("why 4.5 GB, and not 4.0?", INK, CARD_HI, 22)
        q2 = note_pill("are 16 levels per weight enough?", INK, CARD_HI, 22)
        qs = m.VGroup(q1, q2).arrange(m.RIGHT, buff=0.5).move_to([-0.6, -2.95, 0])
        mascot = owl(0.6).move_to([5.75, -2.9, 0])

        self.say(
            "[In episode eighteen], [eight bits, plus a scale per group], [nearly halved Owl's sixteen gigabytes of weights.]",
            m.AnimationGroup(self.new_header("hook", "Owl in four and a half gigabytes"),
                             m.FadeIn(rows[0][0]), m.GrowFromEdge(rows[0][1], m.LEFT), m.FadeIn(rows[0][2])),
            m.AnimationGroup(m.FadeIn(rows[1][0]), m.GrowFromEdge(rows[1][1], m.LEFT), m.FadeIn(rows[1][2], shift=0.1 * m.LEFT),
                             m.FadeIn(ep18, shift=0.1 * m.LEFT)),
            m.Circumscribe(m.VGroup(rows[1][1], rows[1][2]), color=WEIGHTS, buff=0.08),
        )
        self.say(
            "[Today, four bits]: [the same eight billion weights, in about four and a half gigabytes.]",
            m.FadeIn(rows[2][0], shift=0.1 * m.RIGHT),
            m.AnimationGroup(m.GrowFromEdge(rows[2][1], m.LEFT), m.FadeIn(rows[2][2], shift=0.1 * m.LEFT), run_time=1.4),
        )
        self.say(
            "[Since decode speed is bandwidth over bytes], as in episode two, [every byte we drop is speed.]",
            m.FadeIn(formula, shift=0.1 * m.UP),
            m.Circumscribe(ep2[0], color=MEMORY, buff=0.1),
        )
        self.say(
            "[Two questions: why four and a half, and not four?] [And can sixteen levels per weight keep Owl's answers right?]",
            m.AnimationGroup(m.FadeIn(mascot, shift=0.2 * m.UP), m.FadeIn(q1, shift=0.1 * m.UP)),
            m.FadeIn(q2, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 1 · group-wise INT4

    def groups(self):
        self.next_section("1 · A scale for every group", notes=f"INT4 has 16 codes. Affine (asymmetric) quantization: each group "
                          "stores a scale and an offset (zero point, Episode 18), so its 16 levels run from the group's smallest weight to "
                          "its largest. The 24 weights here are schematic, in three groups of 8 (MLX's real groups hold 64). With one scale "
                          f"for the row, the large weight stretches the levels and the other 23 share {USED} of the 16; with a scale per "
                          "group, groups 2 and 3 use all 16 over their own range (the large weight's group stays coarse). Sample row, "
                          f"for example: 4,096 Gaussian weights (σ = 0.02, seed 0) with one set to 0.25; relative RMS rounding error "
                          f"{ERR_ROW * 100:.1f}% with one scale and offset for the row, {ERR_GROUP * 100:.1f}% with groups of {GROUP}.")
        base_y, k = -0.55, 1.15
        y = lambda v: base_y + k * v  # noqa: E731
        xs = [-6.0 + i * 0.3 + (i // NG) * 0.35 for i in range(len(VALS))]
        bw = 0.2

        def bar(v, x):
            h = max(abs(k * v), 0.03)
            r = m.Rectangle(width=bw, height=h).set_stroke(WEIGHTS, 1.8).set_fill(BLUE_FILL, 1)
            return r.move_to([x, base_y + np.sign(v) * h / 2, 0]).set_z_index(1)

        bars = m.VGroup(*[bar(v, x) for v, x in zip(VALS, xs)])
        base = m.Line([xs[0] - 0.25, base_y, 0], [xs[-1] + 0.25, base_y, 0]).set_stroke(EDGE, 2)
        spans = [(xs[g * NG] - bw / 2 - 0.08, xs[g * NG + NG - 1] + bw / 2 + 0.08) for g in range(3)]

        def levels(per_group):
            lines = m.VGroup()
            for g, (a, b) in enumerate(spans):
                lo, hi = (min(GROUPS[g]), max(GROUPS[g])) if per_group else (ROW_LO, ROW_HI)
                for j in range(16):
                    yy = y(lo + j * (hi - lo) / 15)
                    lines.add(m.Line([a, yy, 0], [b, yy, 0]).set_stroke(FAINT, 1.6))
            return lines

        lines_row, lines_group = levels(False), levels(True)

        def marks(snapped):
            ticks, errs = m.VGroup(), m.VGroup()
            for v, s, x in zip(VALS, snapped, xs):
                ticks.add(m.Line([x - 0.17, y(s), 0], [x + 0.17, y(s), 0]).set_stroke(INK, 4).set_z_index(3))
                end = y(s) if abs(y(s) - y(v)) > 0.004 else y(v) + 0.004
                errs.add(m.Line([x, y(v), 0], [x, end, 0]).set_stroke(RED, 7).set_z_index(4))
            return ticks, errs

        ticks_row, errs_row = marks(SNAP_ROW)
        ticks_group, errs_group = marks(SNAP_GROUP)

        under = -1.5
        row_tag = note_pill("one scale + offset for the whole row", SCALE, SCALE_FILL, 20).move_to([(xs[0] + xs[-1]) / 2, under - 0.45, 0])
        row_br = bracket_under(spans[0][0], spans[2][1], under, SCALE)
        g_tags = m.VGroup(*[note_pill("scale + offset", SCALE, SCALE_FILL, 20).move_to([(a + b) / 2, under - 0.45, 0]) for a, b in spans])
        g_brs = m.VGroup(*[bracket_under(a, b, under, SCALE) for a, b in spans])

        px = 2.45
        int4 = code("INT4 · 16 levels", 26, WEIGHTS, weight=m.BOLD).move_to([px, 1.95, 0], aligned_edge=m.LEFT)
        lv_note = text("smallest weight to largest", 20, MUTED).move_to([px, 1.5, 0], aligned_edge=m.LEFT)
        crowd = text(f"23 weights share {USED} levels", 22, RED).move_to([px, 0.85, 0], aligned_edge=m.LEFT)
        spread = text("each group: its own 16 levels", 22, GREEN).move_to([px, 0.85, 0], aligned_edge=m.LEFT)
        sample = m.VGroup(text("a sample row, for example:", 20, MUTED), text("4,096 weights, one of them large", 20, MUTED))
        sample.arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT).move_to([px, -0.05, 0], aligned_edge=m.LEFT)
        err_lab = text("rounding error", 22, INK).move_to([px, -0.75, 0], aligned_edge=m.LEFT)
        before = m.VGroup(title_text(f"{ERR_ROW * 100:.0f}%", 64, RED), code("one scale", 20, MUTED)).arrange(m.DOWN, buff=0.12)
        after = m.VGroup(title_text(f"{ERR_GROUP * 100:.0f}%", 64, GREEN), code(f"groups of {GROUP}", 20, MUTED)).arrange(m.DOWN, buff=0.12)
        before.move_to([px, -1.65, 0], aligned_edge=m.LEFT)
        after.move_to([6.4, -1.65, 0], aligned_edge=m.RIGHT)
        e_arrow = arrow(before[0].get_right() + [0.2, 0, 0], after[0].get_left() - [0.2, 0, 0], MUTED)
        assert after.get_right()[0] < 6.6

        self.say(
            "[Four bits give each weight just sixteen levels], [spread from the smallest weight to the largest.]",
            m.AnimationGroup(self.new_header("01 · groups", "A scale for every group"), m.FadeIn(base), m.FadeIn(int4),
                             m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN if v > 0 else m.UP) for b, v in zip(bars, VALS)], lag_ratio=0.05)),
            m.AnimationGroup(m.LaggedStart(*[m.Create(l) for l in lines_row], lag_ratio=0.01), m.FadeIn(lv_note)),
        )
        self.say(
            "[With one scale for the whole row], [one large weight stretches the levels], [and the rest crowd onto just seven of them.]",
            m.AnimationGroup(m.Create(row_br), m.FadeIn(row_tag, shift=0.1 * m.UP)),
            m.Circumscribe(bars[3], color=RED, buff=0.08),
            m.AnimationGroup(m.FadeIn(ticks_row), m.LaggedStart(*[m.Create(e) for e in errs_row], lag_ratio=0.03), m.FadeIn(crowd, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[So give each small group its own scale and offset.] [Now every group spreads its sixteen levels over its own weights.]",
            m.AnimationGroup(m.ReplacementTransform(row_br, g_brs), m.ReplacementTransform(row_tag, g_tags), run_time=1.2),
            m.AnimationGroup(m.Transform(lines_row, lines_group), m.Transform(ticks_row, ticks_group), m.Transform(errs_row, errs_group),
                             m.FadeOut(crowd, shift=0.1 * m.UP), m.FadeIn(spread, shift=0.1 * m.UP), run_time=1.6),
        )
        self.say(
            "[On a sample row of four thousand weights, for example], [the rounding error falls from thirty-one percent], [to nine.]",
            m.FadeIn(sample, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(err_lab), m.FadeIn(before, shift=0.1 * m.UP)),
            m.AnimationGroup(m.GrowArrow(e_arrow), m.FadeIn(after, shift=0.15 * m.LEFT)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 2 · the bits-per-weight ledger

    def ledger(self):
        self.next_section("2 · The ledger", notes=f"One MLX group: {GROUP} weights × 4 bits = {GROUP_BITS} bits, plus a BF16 scale and a BF16 "
                          f"offset (32 bits) = {GROUP_BITS + 32} bits ÷ {GROUP} = {MLX_BPW} bits per weight (the 4-bit model measured on this "
                          f"Mac: mlx-community/Meta-Llama-3.1-8B-Instruct-4bit, group size 64). A scale alone per 64: {S64_BPW}; a 16-bit "
                          f"scale per 32 (GGUF Q4_0): {S32_BPW}. Owl-8B: {P8 / 1e9:.2f} B × 4 bits = {W4 / GB:.3f} GB (owl_reference "
                          f"weight_bytes(bits=4)), + {SCALES / GB:.3f} GB of scales and offsets = {W45 / GB:.3f} GB; the file on this Mac is "
                          f"{FILE['4bit'] / GB:.4f} GB (the rest: BF16 norm weights and the file header).")
        sy = 1.55
        pitch, cw = 0.15, 0.135
        x_start = -5.45
        wcells = m.VGroup(*[cell(cw, 0.46, WEIGHTS, BLUE_FILL, 1.2, 0.02).move_to([x_start + i * pitch + cw / 2, sy, 0]) for i in range(GROUP)])
        sw = 4 * pitch - 0.015
        sx0 = wcells[-1].get_right()[0] + 0.12
        s_cell = cell(sw, 0.46, SCALE, SCALE_FILL, 2, 0.04).move_to([sx0 + sw / 2, sy, 0])
        o_cell = cell(sw, 0.46, SCALE, SCALE_FILL, 2, 0.04).move_to([sx0 + sw + 0.06 + sw / 2, sy, 0])
        s_t = code("s", 26, SCALE).move_to(s_cell)
        o_t = code("z", 26, SCALE).move_to(o_cell)
        extras = m.VGroup(s_cell, o_cell, s_t, o_t)
        w_lab = code(f"{GROUP} weights × 4 bits = {GROUP_BITS} bits", 20, WEIGHTS).move_to([x_start, sy + 0.62, 0], aligned_edge=m.LEFT)
        e_lab = code("scale + offset = 2 × 16 bits", 20, SCALE).move_to([o_cell.get_right()[0], sy + 0.62, 0], aligned_edge=m.RIGHT)
        eq = m.VGroup(code(f"({GROUP_BITS} + 32) bits ÷ {GROUP} weights =", 26, INK, t2c={str(GROUP_BITS): WEIGHTS, r"\+ 32": SCALE}),
                      title_text(f"{MLX_BPW} bits per weight", 40, INK)).arrange(m.RIGHT, buff=0.3)
        eq[1].align_to(eq[0], m.DOWN).shift(0.02 * m.DOWN)
        eq.move_to([0, 0.55, 0])

        cols = [-6.2, -3.5, 0.3, 3.4]
        top = -0.25
        heads = m.VGroup(*[code(h, 20, MUTED).move_to([x, top, 0], aligned_edge=m.LEFT)
                           for h, x in zip(("group", "extra bits", "bits per weight", "used by"), cols)])
        rule = m.Line([-6.2, top - 0.24, 0], [6.4, top - 0.24, 0]).set_stroke(EDGE, 2)

        def lrow(i, group, extra, formula, value, who, hi=False):
            yy = top - 0.6 - 0.45 * i
            return m.VGroup(
                code(f"{group} weights", 20, INK).move_to([cols[0], yy, 0], aligned_edge=m.LEFT),
                code(extra, 20, SCALE).move_to([cols[1], yy, 0], aligned_edge=m.LEFT),
                code(f"{formula} = {value:g}", 20, INK, t2c={rf"= {value:g}": GREEN if hi else INK}).move_to([cols[2], yy, 0], aligned_edge=m.LEFT),
                (code(who, 20, INK if hi else MUTED).move_to([cols[3], yy, 0], aligned_edge=m.LEFT) if who else m.VGroup()),
            )

        r1 = lrow(0, GROUP, "scale + offset", f"4 + 32/{GROUP}", MLX_BPW, "MLX · this Mac", True)
        r2 = lrow(1, 64, "scale", "4 + 16/64", S64_BPW, "")
        r3 = lrow(2, 32, "scale", "4 + 16/32", S32_BPW, "GGUF Q4_0")

        oy = -2.55
        tx0, twidth = -4.6, 10.6
        per = twidth / (W16 / GB)
        track = m.Rectangle(width=twidth, height=0.5).set_stroke(EDGE, 2).set_fill(CARD, 1).move_to([tx0 + twidth / 2, oy, 0])
        t_lab = code(f"BF16 · {W16 / GB:.2f} GB", 20, MUTED).move_to([tx0 + twidth - 0.2, oy, 0], aligned_edge=m.RIGHT)
        o_lab = m.VGroup(text("Owl-8B", 22, INK, weight=m.BOLD), code("4-bit", 20, MUTED)).arrange(m.DOWN, buff=0.06, aligned_edge=m.LEFT)
        o_lab.move_to([-6.3, oy, 0], aligned_edge=m.LEFT)
        wseg = m.Rectangle(width=per * W4 / GB, height=0.42).set_stroke(width=0).set_fill(WEIGHTS, 1)
        wseg.move_to([tx0 + wseg.width / 2, oy, 0])
        sseg = m.Rectangle(width=per * SCALES / GB, height=0.42).set_stroke(width=0).set_fill(SCALE, 1)
        sseg.move_to([wseg.get_right()[0] + sseg.width / 2, oy, 0])
        w_gb = code(f"{W4 / GB:.3f} GB weights", 20, WEIGHTS).next_to(wseg, m.DOWN, buff=0.14).align_to(wseg, m.LEFT)
        s_gb = code(f"+ {SCALES / GB:.3f} GB scales", 20, SCALE).next_to(w_gb, m.RIGHT, buff=0.3)
        total = code(f"= {W45 / GB:.3f} GB", 24, INK, weight=m.BOLD).next_to(sseg, m.RIGHT, buff=0.25)
        file = m.VGroup(code(f"on this Mac: {FILE['4bit'] / GB:.3f} GB", 20, INK), code("✓", 26, GREEN, weight=m.BOLD)).arrange(m.RIGHT, buff=0.2)
        file.move_to([tx0 + twidth + 0.3, w_gb.get_y(), 0], aligned_edge=m.RIGHT)
        assert file.get_left()[0] > s_gb.get_right()[0] + 0.3

        self.say(
            "[Now the price.] [One group of sixty-four four-bit weights] [is two hundred and fifty-six bits.]",
            self.new_header("02 · the ledger", "Count every bit"),
            m.LaggedStart(*[m.FadeIn(c, scale=0.5) for c in wcells], lag_ratio=0.03),
            m.FadeIn(w_lab, shift=0.1 * m.DOWN),
        )
        self.say(
            "[As in episode eighteen, a scale and an offset add thirty-two more], [so each weight really costs four and a half bits.]",
            m.AnimationGroup(m.LaggedStart(m.GrowFromEdge(m.VGroup(s_cell, s_t), m.LEFT), m.GrowFromEdge(m.VGroup(o_cell, o_t), m.LEFT), lag_ratio=0.4),
                             m.FadeIn(e_lab, shift=0.1 * m.DOWN)),
            m.FadeIn(eq, shift=0.15 * m.UP),
        )
        self.say(
            "[That's the format measured on this Mac.] [A scale alone: four and a quarter.] [A scale per thirty-two weights: four and a half again.]",
            m.AnimationGroup(m.FadeIn(heads), m.Create(rule), m.FadeIn(r1, shift=0.1 * m.UP)),
            m.FadeIn(r2, shift=0.1 * m.UP),
            m.FadeIn(r3, shift=0.1 * m.UP),
        )
        self.say(
            "[For Owl, that's four gigabytes of four-bit weights], [plus half a gigabyte of scales and offsets.]",
            m.AnimationGroup(m.FadeIn(track), m.FadeIn(t_lab), m.FadeIn(o_lab), m.GrowFromEdge(wseg, m.LEFT), m.FadeIn(w_gb)),
            m.AnimationGroup(m.GrowFromEdge(sseg, m.LEFT), m.FadeIn(s_gb, shift=0.1 * m.LEFT), m.FadeIn(total, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[The file on this Mac: four point five two gigabytes.] [The ledger balances.]",
            m.FadeIn(file[0], shift=0.1 * m.LEFT),
            m.AnimationGroup(m.FadeIn(file[1], scale=1.6), m.Circumscribe(file, color=GREEN, buff=0.1)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 3 · GPTQ and AWQ

    def rounding(self):
        self.next_section("3 · Round with a plan", notes="Both are post-training, weight-only (W4A16) methods with a small calibration set and "
                          "no retraining. GPTQ (Frantar et al., ICLR 2023): 'a new one-shot weight quantization method based on approximate "
                          "second-order information'; it quantizes a layer's weights column by column and updates the not-yet-quantized "
                          "columns to compensate the error; 175 B parameters 'in approximately four GPU hours', 3–4 bits, ≈3.25× faster "
                          "end to end on an A100. AWQ (Lin et al., MLSys 2024 best paper): 'Protecting only 1% salient weights can greatly "
                          "reduce quantization error'; salient channels are found from activation statistics; scaling a weight channel up by "
                          "s and its input down by s leaves the product unchanged but shrinks that channel's relative rounding error; 'does "
                          "not rely on any backpropagation or reconstruction'. Kurtic et al. (ACL 2025, Llama-3.1 family, over 500,000 "
                          "evaluations): 'INT4 weight-only (W4A16-INT) is more competitive than expected, rivaling 8-bit quantization.' "
                          "The grids and activation bars are schematic.")
        rows_, cols_, sz, gap = 4, 8, 0.4, 0.08
        gy = -0.35

        def grid(cx):
            g = m.VGroup(*[m.VGroup(*[cell(sz, sz, WEIGHTS, BLUE_FILL, 1.6, 0.05) for _ in range(cols_)]).arrange(m.RIGHT, buff=gap)
                           for _ in range(rows_)]).arrange(m.DOWN, buff=gap)
            return g.move_to([cx, gy, 0])

        def panel(cx, name, who):
            box = card(6.1, 4.6, EDGE, CARD, radius=0.2, stroke=2).move_to([cx, -0.35, 0])
            t = m.VGroup(title_text(name, 32, WEIGHTS), cite(who)).arrange(m.RIGHT, buff=0.3)
            t.align_to(box, m.UL).shift(0.35 * m.RIGHT + 0.25 * m.DOWN)
            return m.VGroup(box, t).set_z_index(-1)

        lp, rp = panel(-3.25, "GPTQ", "Frantar et al., 2023"), panel(3.25, "AWQ", "Lin et al., 2024")
        lg, rg = grid(-3.25), grid(3.25)
        col = lambda g, j: m.VGroup(*[g[r][j] for r in range(rows_)])  # noqa: E731

        # GPTQ: a cursor sweeps the columns; each rounded column pushes its error onto the rest
        cursor = m.SurroundingRectangle(col(lg, 0), buff=0.07, corner_radius=0.06).set_stroke(INK, 3).set_z_index(3)
        top_y = lg.get_top()[1]

        def err_arrow(j):
            a = [col(lg, j).get_x() + 0.12, top_y + 0.3, 0]
            b = [lg.get_right()[0] + 0.05, top_y + 0.3, 0]
            return arrow(a, b, RED, buff=0).set_stroke(width=4)

        e_arr = err_arrow(0)
        e_lab = code("error →", 20, RED).next_to(e_arr, m.UP, buff=0.08).align_to(lg, m.LEFT)
        steps = []
        for j in range(cols_):
            parts = [cursor.animate.move_to(col(lg, j).get_center()), col(lg, j).animate.set_fill(ROUNDED, 1)]
            if j < cols_ - 1:
                rest = m.VGroup(*[col(lg, i) for i in range(j + 1, cols_)])
                parts += [rest.animate(rate_func=m.there_and_back).set_fill(RED_FILL, 1), m.Transform(e_arr, err_arrow(j))]
            else:
                parts += [m.FadeOut(e_arr), m.FadeOut(e_lab)]
            steps.append(m.AnimationGroup(*parts, run_time=0.45))
        l_cap = text("round a column, push its error onto the rest", 20, INK).next_to(lg, m.DOWN, buff=0.52)
        l_fact = cite("175 B weights in ≈ 4 GPU hours").next_to(l_cap, m.DOWN, buff=0.16)

        # AWQ: activation bars above the columns; one salient channel is scaled before rounding
        heights = [0.13, 0.2, 0.1, 0.17, 0.12, 0.5, 0.18, 0.12]
        hot = int(np.argmax(heights))
        a_base = rg.get_top()[1] + 0.12
        acts = m.VGroup(*[m.Rectangle(width=0.22, height=h).set_stroke(width=0).set_fill(INK if j == hot else MUTED, 1)
                          .move_to([col(rg, j).get_x(), a_base + h / 2, 0]) for j, h in enumerate(heights)])
        a_lab = text("inputs", 20, MUTED).next_to(acts, m.LEFT, buff=0.25).align_to(acts, m.DOWN)
        a_lab.shift(0.04 * m.UP)
        guard = m.SurroundingRectangle(col(rg, hot), buff=0.07, corner_radius=0.06).set_stroke(GREEN, 3).set_z_index(3)
        small_act = m.Rectangle(width=0.22, height=heights[hot] / 2).set_stroke(width=0).set_fill(INK, 1)
        small_act.move_to([col(rg, hot).get_x(), a_base + heights[hot] / 4, 0])
        div = code("÷ s", 22, INK, weight=m.BOLD).next_to(small_act, m.UP, buff=0.1)
        mul = code("× s", 22, GREEN, weight=m.BOLD).next_to(guard, m.DOWN, buff=0.08)
        r_cap = text("salient channel up, its input down", 20, INK).next_to(rg, m.DOWN, buff=0.52)
        r_fact = cite("“protecting only 1% salient weights …”").next_to(r_cap, m.DOWN, buff=0.16)

        verdict = m.VGroup(text("Llama 3.1, over 500,000 evaluations: 4-bit weights rival 8-bit quantization", 22, INK,
                                t2c={"rival 8-bit": GREEN}), cite("Kurtic et al., 2025")).arrange(m.RIGHT, buff=0.35).move_to([0, -3.2, 0])

        self.say(
            "[Rounding to the nearest level is the simple way.] [Two methods do better, without retraining.]",
            m.AnimationGroup(self.new_header("03 · rounding", "Round with a plan"), m.FadeIn(lg, shift=0.1 * m.UP), m.FadeIn(rg, shift=0.1 * m.UP)),
            m.LaggedStart(m.FadeIn(lp), m.FadeIn(rp), lag_ratio=0.4),
        )
        self.say(
            "[GPTQ rounds one column at a time], [and nudges the columns not yet rounded], [to cancel the error it just made.]",
            m.AnimationGroup(m.Create(cursor), m.FadeIn(e_arr), m.FadeIn(e_lab)),
            m.Succession(*steps),
            m.FadeIn(l_cap, shift=0.1 * m.UP),
        )
        self.say(
            "[It needs only a small sample of inputs], [and about four GPU hours, even for a hundred and seventy-five billion weights.]",
            m.FadeOut(cursor),
            m.FadeIn(l_fact, shift=0.1 * m.UP),
        )
        self.say(
            "[AWQ looks at the activations instead]: [the channels that meet large inputs matter most.]",
            m.AnimationGroup(m.FadeIn(a_lab), m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in acts], lag_ratio=0.1)),
            m.AnimationGroup(m.Create(guard), m.Circumscribe(acts[hot], color=GREEN, buff=0.06)),
        )
        self.say(
            "[It scales those few channels up before rounding, and their inputs down], [so the math is unchanged], [but their error shrinks.]",
            m.AnimationGroup(m.Transform(acts[hot], small_act), m.FadeIn(div, shift=0.1 * m.LEFT), m.FadeIn(mul, shift=0.1 * m.LEFT),
                             col(rg, hot).animate.set_fill(ROUNDED, 1)),
            m.FadeIn(r_cap, shift=0.1 * m.UP),
            m.FadeIn(r_fact, shift=0.1 * m.UP),
        )
        self.say(
            "[And in episode eighteen's study of half a million evaluations], [four-bit weights like these rivaled eight-bit quantization.]",
            m.FadeIn(verdict[1]),
            m.FadeIn(verdict[0], shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 4 · GGUF k-quants

    def kquants(self):
        self.next_section("4 · Scales for the scales", notes=f"GGUF k-quants (llama.cpp PR #1684, 2023; ggml-common.h): Q4_K = 'type-1' "
                          "(w ≈ scale · q + min) 4-bit quantization in super-blocks of 8 blocks × 32 weights; 'Scales and mins are quantized "
                          f"with 6 bits'; one FP16 scale and one FP16 min per super-block scale them. Bits: {Q4K_PARTS[0]} + {Q4K_PARTS[1]} + "
                          f"{Q4K_PARTS[2]} = {sum(Q4K_PARTS)} per {QK} weights = 4.5 bpw. Q4_1 (a 16-bit scale and min per 32) = {Q41_BPW} bpw. "
                          "Ladder (bits per weight from the block layouts in ggml-common.h; Owl-8B size = 8.03 B × bpw ÷ 8): "
                          + ", ".join(f"{n} {b:g} → {P8 * b / 8 / GB:.1f} GB" for n, b in LADDER)
                          + ". Q4_K_M uses Q6_K for half of attention.wv and feed_forward.w2, else Q4_K (4.89 bpw for Llama-3.1-8B, llama.cpp "
                          f"quantize README). Perplexity, LLaMA-7B on wikitext (PR #1684): F16 {PPL['F16']}, Q6_K {PPL['Q6_K']} "
                          f"(+{PPL_UP['Q6_K']:.2f}%), Q4_K_M {PPL['Q4_K_M']} (+{PPL_UP['Q4_K_M']:.1f}%), Q2_K {PPL['Q2_K']} (+{PPL_UP['Q2_K']:.1f}%); "
                          "the Q2_K file is itself a mix (Q4_K for attention.wv and feed_forward.w2).")
        bw_, bh, bgap, by = 1.25, 0.55, 0.1, 0.75
        bx = [-(8 * bw_ + 7 * bgap) / 2 + bw_ / 2 + i * (bw_ + bgap) for i in range(8)]
        blocks = m.VGroup(*[m.VGroup(cell(bw_, bh, WEIGHTS, BLUE_FILL, 2, 0.06), code("32", 22, INK)).move_to([x, by, 0]) for x in bx])
        for b in blocks:
            b[1].move_to(b[0])
        br = bracket_under(bx[0] - bw_ / 2, bx[-1] + bw_ / 2, by - bh / 2 - 0.2, MUTED)
        br_lab = text("256 weights: one super-block of eight blocks", 22, MUTED).next_to(br, m.DOWN, buff=0.12)
        cy = 1.55
        chips = m.VGroup(*[m.VGroup(cell(1.08, 0.38, SCALE, SCALE_FILL, 2, 0.08), code("6 + 6", 20, SCALE)).move_to([x, cy, 0]) for x in bx])
        for c in chips:
            c[1].move_to(c[0])
        drops = m.VGroup(*[m.Line([x, cy - 0.19, 0], [x, by + bh / 2, 0]).set_stroke(SCALE, 2) for x in bx])
        sup_t = code("2 × 16 bits · scale + min", 22, SCALE, weight=m.BOLD)
        sup = m.VGroup(cell(sup_t.width + 0.5, 0.44, SCALE, SCALE_FILL, 2.5, 0.1), sup_t)
        sup[1].move_to(sup[0])
        sup.move_to([0, 2.33, 0])
        bus_y = 1.98
        bus = m.VGroup(m.Line([bx[0], bus_y, 0], [bx[-1], bus_y, 0]), m.Line([0, bus_y, 0], [0, sup.get_bottom()[1], 0]),
                       *[m.Line([x, bus_y, 0], [x, cy + 0.19, 0]) for x in bx]).set_stroke(SCALE, 2)

        lx = [-6.2, -2.5, 0.55]
        rows = [("weights", f"{QK} × 4", Q4K_PARTS[0], WEIGHTS), ("block scales + mins", "8 × (6 + 6)", Q4K_PARTS[1], SCALE),
                ("super-block pair", "2 × 16", Q4K_PARTS[2], SCALE)]
        led = m.VGroup()
        for i, (name, f, v, c) in enumerate(rows):
            yy = -0.85 - 0.42 * i
            led.add(m.VGroup(code(name, 20, c).move_to([lx[0], yy, 0], aligned_edge=m.LEFT),
                             code(f, 20, MUTED).move_to([lx[1], yy, 0], aligned_edge=m.LEFT),
                             code(f"{v:,}", 20, INK).move_to([lx[2], yy, 0], aligned_edge=m.RIGHT)))
        l_rule = m.Line([lx[0], -1.9, 0], [lx[2], -1.9, 0]).set_stroke(EDGE, 2)
        l_tot = m.VGroup(code(f"{sum(Q4K_PARTS):,} bits ÷ {QK} =", 22, INK), title_text("4.5 bits per weight", 32, INK)).arrange(m.RIGHT, buff=0.25)
        l_tot.move_to([lx[0], -2.3, 0], aligned_edge=m.LEFT)

        cx0 = 1.45
        cmp_t = text("a scale and a min for every 32 weights:", 22, MUTED).move_to([cx0, -0.85, 0], aligned_edge=m.LEFT)
        c1 = m.VGroup(code("Q4_1", 24, INK, weight=m.BOLD), text("both in 16 bits", 22, INK),
                      code(f"{Q41_BPW:g} bits", 24, RED, weight=m.BOLD)).arrange(m.RIGHT, buff=0.3)
        c2 = m.VGroup(code("Q4_K", 24, INK, weight=m.BOLD), text("both in 6 bits", 22, INK),
                      code("4.5 bits", 24, GREEN, weight=m.BOLD)).arrange(m.RIGHT, buff=0.3)
        c1.move_to([cx0, -1.4, 0], aligned_edge=m.LEFT)
        c2.move_to([cx0, -1.9, 0], aligned_edge=m.LEFT)
        c2[2].align_to(c1[2], m.LEFT)
        c2[1].align_to(c1[1], m.LEFT)
        src = cite("llama.cpp · PR #1684, 2023 · ggml-common.h").move_to([cx0, -2.45, 0], aligned_edge=m.LEFT)
        diagram = m.VGroup(blocks, br, br_lab, chips, drops, sup, bus, led, l_rule, l_tot, cmp_t, c1, c2, src)

        # the ladder (replaces the diagram on the last line)
        top = 2.2
        lad_x0, unit = -4.3, 0.75
        heads = m.VGroup(code("type", 20, MUTED).move_to([-6.2, top, 0], aligned_edge=m.LEFT),
                         code("bits per weight", 20, MUTED).move_to([lad_x0, top, 0], aligned_edge=m.LEFT),
                         code("Owl-8B", 20, MUTED).move_to([3.6, top, 0], aligned_edge=m.RIGHT),
                         code("perplexity", 20, MUTED).move_to([6.4, top, 0], aligned_edge=m.RIGHT))
        lad = m.VGroup()
        for i, (name, b) in enumerate(LADDER):
            yy = top - 0.6 - 0.65 * i
            is_m = name == "Q4_K_M"
            colr = GREEN if is_m else WEIGHTS
            bar = hbar(lad_x0, yy, unit * b, 0.36, colr, GREEN_FILL if is_m else BLUE_FILL)
            up = PPL_UP.get(name)
            lad.add(m.VGroup(
                code(name, 22, INK, weight=m.BOLD).move_to([-6.2, yy, 0], aligned_edge=m.LEFT),
                bar,
                code(f"{b:g}", 22, colr).next_to(bar, m.RIGHT, buff=0.18),
                code(f"{P8 * b / 8 / GB:.1f} GB", 22, INK).move_to([3.6, yy, 0], aligned_edge=m.RIGHT),
                m.VGroup() if up is None else code(f"+{up:.2f}%" if up < 0.1 else f"+{up:.1f}%" if up < 1 else f"+{up:.0f}%", 22,
                                                   RED if up > 5 else GREEN).move_to([6.4, yy, 0], aligned_edge=m.RIGHT),
            ))
        lad_src = cite("perplexity vs F16 on LLaMA-7B (lower is better) · llama.cpp PR #1684, 2023").move_to([0, -2.55, 0])
        mix = cite("Q4_K_M: 6-bit for a few sensitive tensors, Q4_K elsewhere").move_to([0, -2.95, 0])
        ladder = m.VGroup(heads, lad, lad_src, mix)

        self.say(
            "[The GGUF files of llama.cpp go one step further], [with k-quants.]",
            self.new_header("04 · super-blocks", "Scales for the scales"),
            m.LaggedStart(*[m.FadeIn(b, shift=0.1 * m.UP) for b in blocks], lag_ratio=0.1),
        )
        self.say(
            "[Two hundred and fifty-six weights form a super-block], [made of eight blocks of thirty-two.]",
            m.AnimationGroup(m.Create(br), m.FadeIn(br_lab, shift=0.1 * m.UP)),
            m.LaggedStart(*[m.Indicate(b[1], color=WEIGHTS, scale_factor=1.25) for b in blocks], lag_ratio=0.1),
        )
        self.say(
            "[Each block gets its own scale and minimum], [stored in just six bits each.]",
            m.LaggedStart(*[m.AnimationGroup(m.Create(d), m.FadeIn(c, shift=0.1 * m.DOWN)) for d, c in zip(drops, chips)], lag_ratio=0.1),
            m.LaggedStart(*[m.Indicate(c[1], color=SCALE, scale_factor=1.2) for c in chips], lag_ratio=0.06),
        )
        self.say(
            "[Then one sixteen-bit pair for the whole super-block] [scales those scales.]",
            m.AnimationGroup(m.FadeIn(sup, shift=0.1 * m.DOWN), m.Create(bus[1])),
            m.AnimationGroup(m.Create(bus[0]), m.LaggedStart(*[m.Create(l) for l in bus[2:]], lag_ratio=0.08),
                             m.LaggedStart(*[c.animate(rate_func=m.there_and_back).scale(1.12) for c in chips], lag_ratio=0.08)),
        )
        self.say(
            "[Add it up]: [four and a half bits per weight.] [The simpler format, with both in sixteen bits], [costs five.]",
            m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.UP) for r in led], lag_ratio=0.3),
            m.AnimationGroup(m.Create(l_rule), m.FadeIn(l_tot, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(cmp_t), m.FadeIn(c1, shift=0.1 * m.LEFT), m.FadeIn(src)),
            m.FadeIn(c2, shift=0.1 * m.LEFT),
        )
        self.say(
            "[The same idea spans a ladder from two bits to six.] [The popular four-bit mix costs under one percent in perplexity]; [the two-bit one, about fifteen.]",
            m.AnimationGroup(m.FadeOut(diagram), m.FadeIn(heads),
                             m.LaggedStart(*[m.AnimationGroup(m.FadeIn(r[0]), m.GrowFromEdge(r[1], m.LEFT), m.FadeIn(r[2]), m.FadeIn(r[3])) for r in lad],
                                           lag_ratio=0.15), run_time=2.2),
            m.AnimationGroup(m.FadeIn(lad[3][4], shift=0.1 * m.LEFT), m.FadeIn(lad[5][4], shift=0.1 * m.LEFT), m.FadeIn(lad_src), m.FadeIn(mix),
                             m.Circumscribe(lad[3], color=GREEN, buff=0.08)),
            m.AnimationGroup(m.FadeIn(lad[0][4], shift=0.1 * m.LEFT), m.Circumscribe(lad[0], color=RED, buff=0.08)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 5 · micro-scaled floats

    def floats(self):
        self.next_section("5 · Four-bit floats", notes=f"FP4 E2M1 (1 sign, 2 exponent, 1 mantissa bit) represents ±{{{', '.join(f'{v:g}' for v in E2M1)}}}: "
                          "levels close together near zero, where most weights are, and far apart near the ends; INT4's 16 levels are evenly "
                          f"spaced. MXFP4 (OCP Microscaling Formats v1.0, 2023): 32 E2M1 values share one 8-bit E8M0 (power-of-two) scale = 4 + 8/32 = "
                          f"{MX_BPW} bits per weight. NVFP4 (NVIDIA, 2025-06-24): 16 values share an FP8 E4M3 scale (+ one FP32 scale per tensor) = "
                          f"4 + 8/16 = {NV_BPW}; 'approximately 3.5x relative to FP16'; DeepSeek-R1-0528 FP8 → NVFP4 '1% or less accuracy degradation'. "
                          "Blackwell Tensor Cores implement NVFP4 (and MXFP4), so activations can be 4-bit too (W4A4) and compute-bound prefill gets "
                          "faster, not just decode (notes §8). OpenAI gpt-oss (2025): MoE weights post-trained in MXFP4; gpt-oss-120b (117 B total, "
                          "5.1 B active) fits a single 80 GB GPU. Experts at serve time: Episode 20. Bell curve and cells are schematic.")
        sx = 0.78  # scene units per value
        x_of = lambda v: v * sx  # noqa: E731
        cy0, ch = 1.45, 1.0
        xs = np.linspace(-6.2, 6.2, 90)
        pts = [[x_of(v), cy0 + ch * np.exp(-v * v / (2 * 1.1 ** 2)), 0] for v in xs]
        hill = m.Polygon(*pts, [x_of(6.2), cy0, 0], [x_of(-6.2), cy0, 0], stroke_width=0, fill_color=BLUE_FILL, fill_opacity=1)
        hill_l = m.VMobject().set_points_as_corners(pts).set_stroke(WEIGHTS, 3)
        hill_t = text("most weights sit near zero", 20, MUTED).move_to([3.3, cy0 + 0.75, 0], aligned_edge=m.LEFT)
        curve = m.VGroup(hill, hill_l)

        def number_line(y, values, name, sub, color):
            axis = m.Line([x_of(-6.3), y, 0], [x_of(6.3), y, 0]).set_stroke(FAINT, 2)
            ticks = m.VGroup(*[m.Line([x_of(v), y - 0.14, 0], [x_of(v), y + 0.14, 0]).set_stroke(color, 3.5) for v in values])
            lab = m.VGroup(code(name, 22, color, weight=m.BOLD), code(sub, 20, MUTED)).arrange(m.DOWN, buff=0.06, aligned_edge=m.LEFT)
            lab.move_to([-6.35, y, 0], aligned_edge=m.LEFT)
            return axis, ticks, lab

        int_vals = [-6 + 12 * j / 15 for j in range(16)]
        fp_vals = sorted({s * v for s in (1, -1) for v in E2M1})
        assert len(int_vals) == 16 and len(fp_vals) == 15
        i_axis, i_ticks, i_lab = number_line(0.85, int_vals, "INT4", "uniform", WEIGHTS)
        f_axis, f_ticks, f_lab = number_line(0.0, fp_vals, "FP4", "E2M1", GREEN)
        f_nums = m.VGroup(*[code(f"{v:g}", 20, MUTED).next_to([x_of(v), -0.14, 0], m.DOWN, buff=0.1) for v in (-6, -4, -2, 0, 2, 4, 6)])
        i_lab.shift(0.0 * m.UP)

        def block(n, y, scale_label):
            cells = m.VGroup(*[cell(0.16, 0.42, GREEN, GREEN_FILL, 1.4, 0.03).move_to([-6.2 + 0.08 + i * 0.19, y, 0]) for i in range(n)])
            sc = cell(0.37, 0.42, SCALE, SCALE_FILL, 2, 0.05).move_to([cells[-1].get_right()[0] + 0.08 + 0.185, y, 0])
            return m.VGroup(cells, sc)

        mx = block(32, -1.3, "E8M0")
        nv = block(16, -2.3, "E4M3")
        tx = 0.75
        mx_t = m.VGroup(code(f"MXFP4 · 4 + 8/32 = {MX_BPW:g} bits", 22, INK, t2c={rf"{MX_BPW:g}": GREEN}),
                        cite("one power-of-two scale per 32 · OCP MX, 2023"))
        mx_t.arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT).move_to([tx, -1.3, 0], aligned_edge=m.LEFT)
        nv_t = m.VGroup(code(f"NVFP4 · 4 + 8/16 = {NV_BPW:g} bits", 22, INK, t2c={rf"{NV_BPW:g}": GREEN}),
                        cite("16 values + an FP8 scale · NVIDIA, 2025"), cite("DeepSeek-R1: within ≈ 1% of FP8 accuracy"))
        nv_t.arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT).move_to([tx, -2.3, 0], aligned_edge=m.LEFT)
        hw = note_pill("Blackwell: FP4 tensor-core math · faster prefill", COMPUTE, AMBER_FILL, 20)
        oss = note_pill("gpt-oss in MXFP4: 120b on one 80 GB GPU", INK, CARD_HI, 20)
        bottom = m.VGroup(hw, oss).arrange(m.RIGHT, buff=0.35).move_to([0, -3.2, 0])
        assert bottom.width < 13.2

        self.say(
            "[The newest GPUs go further]: [they compute directly in four bits], [with tiny floating-point numbers.]",
            m.AnimationGroup(self.new_header("05 · FP4", "Four-bit floats, tiny scales"), m.FadeIn(curve), m.FadeIn(hill_t)),
            m.AnimationGroup(m.Create(i_axis), m.FadeIn(i_lab), m.LaggedStart(*[m.FadeIn(t, scale=0.4) for t in i_ticks], lag_ratio=0.05)),
            m.AnimationGroup(m.Create(f_axis), m.FadeIn(f_lab)),
        )
        self.say(
            "[An INT4 grid is evenly spaced.] [A four-bit float, like FP8, packs its levels near zero, where most weights live], [and reaches out to six.]",
            m.LaggedStart(*[t.animate(rate_func=m.there_and_back).scale(1.6) for t in i_ticks], lag_ratio=0.04),
            m.LaggedStart(*[m.FadeIn(t, shift=0.15 * m.UP) for t in sorted(f_ticks, key=lambda t: abs(t.get_x()))], lag_ratio=0.06),
            m.FadeIn(f_nums, shift=0.1 * m.UP),
        )
        self.say(
            "[The MXFP4 format shares one eight-bit scale, a power of two, across thirty-two values]: [four and a quarter bits per weight.]",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c, scale=0.5) for c in mx[0]], lag_ratio=0.03), m.GrowFromEdge(mx[1], m.LEFT)),
            m.FadeIn(mx_t, shift=0.1 * m.LEFT),
        )
        self.say(
            "[The NVFP4 format uses blocks of sixteen and a finer scale]: [four and a half bits.] [On DeepSeek-R1, it stays within about one percent of FP8.]",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c, scale=0.5) for c in nv[0]], lag_ratio=0.05), m.GrowFromEdge(nv[1], m.LEFT)),
            m.AnimationGroup(m.FadeIn(nv_t[0], shift=0.1 * m.LEFT), m.FadeIn(nv_t[1], shift=0.1 * m.LEFT)),
            m.FadeIn(nv_t[2], shift=0.1 * m.UP),
        )
        self.say(
            "[On Blackwell GPUs, the tensor cores multiply these directly], [so activations can shrink too, and prefill gets faster, not just decode.]",
            m.FadeIn(hw, shift=0.1 * m.UP),
            m.Circumscribe(hw, color=COMPUTE, buff=0.08),
        )
        self.say(
            "[Even OpenAI's largest open model stores its experts this way], [so it fits on one eighty-gigabyte GPU.]",
            m.FadeIn(oss, shift=0.1 * m.UP),
            m.Circumscribe(oss, color=INK, buff=0.08),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 6 · measured on this Mac

    def on_this_mac(self):
        self.next_section("6 · Four bits on this Mac", notes=f"research/measurements.md §1 (2026-10-24, Mac Studio M4 Max 546 GB/s, mlx-lm 0.32.2, "
                          "median of 5 runs, 256 tokens, greedy): Llama-3.1-8B decode " + ", ".join(f"{k} {TOK[k]:.1f} tok/s vs ceiling "
                          f"{CEIL[k]:.1f} ({OF[k] * 100:.0f}%)" for k in KEYS) + f". Ceilings from owl_reference decode_ceiling(hw=MAC, bits=16 / "
                          f"{Q8_BPW} / {MLX_BPW}) = 546 GB/s ÷ the file sizes; a flat bits=4 (no scales) would give {FLAT4:.0f}. 4-bit is "
                          f"{SPEEDUP:.2f}x faster than BF16 from {FEWER:.2f}x fewer bytes; MBU on the bytes really read: {DEC['4bit']['mbu'] * 100:.0f}% "
                          f"(4-bit) vs {DEC['bf16']['mbu'] * 100:.0f}% (BF16). §0: a 4-bit group-64 mat-vec reaches {MV4:.0f} GB/s vs {MV16:.0f} for "
                          f"BF16 (dequantization costs compute). §2: prefill at 2,048 tokens {PRE['bf16']:,.0f} (BF16) vs {PRE['4bit']:,.0f} (4-bit) "
                          "tok/s: prefill is compute-bound (Episode 3), and weight-only quantization adds unpacking work. Episode 23: what fits on which Mac.")
        mac = icon(recolored("mac", MUTED, CARD, PREFIX), 0.45)
        stamp = m.VGroup(mac, m.VGroup(cite("measured on this Mac"), cite("M4 Max · 546 GB/s · mlx-lm")).arrange(m.DOWN, buff=0.06, aligned_edge=m.LEFT))
        stamp.arrange(m.RIGHT, buff=0.22).move_to([6.5, 2.3, 0], aligned_edge=m.RIGHT)
        legend = m.VGroup(
            m.VGroup(m.Square(0.24).set_stroke(MEMORY, 2).set_fill(CYAN_FILL, 1), code("ceiling: bandwidth ÷ bytes", 20, MEMORY)).arrange(m.RIGHT, buff=0.15),
            m.VGroup(m.Square(0.24).set_stroke(width=0).set_fill(TOKEN, 1), code("measured decode", 20, TOKEN)).arrange(m.RIGHT, buff=0.15),
        ).arrange(m.RIGHT, buff=0.5).move_to([-6.3, 2.3, 0], aligned_edge=m.LEFT)

        x0, per = -4.2, 6.0 / FLAT4
        names = {"bf16": ("BF16", f"{FILE['bf16'] / GB:.2f} GB"), "8bit": ("8-bit", f"{FILE['8bit'] / GB:.2f} GB"),
                 "4bit": ("4-bit", f"{FILE['4bit'] / GB:.2f} GB")}
        rows = {}
        for k, yy in zip(KEYS, (1.4, 0.45, -0.5)):
            lab = m.VGroup(code(names[k][0], 24, WEIGHTS, weight=m.BOLD), code(names[k][1], 20, MUTED)).arrange(m.DOWN, buff=0.06, aligned_edge=m.LEFT)
            lab.move_to([-6.3, yy, 0], aligned_edge=m.LEFT)
            ceil = hbar(x0, yy, per * CEIL[k], 0.46, MEMORY, CYAN_FILL)
            meas = m.Rectangle(width=per * TOK[k], height=0.26).set_stroke(width=0).set_fill(TOKEN, 1).move_to([x0 + per * TOK[k] / 2, yy, 0])
            meas.set_z_index(2)
            val = code(f"{TOK[k]:.1f} / {CEIL[k]:.1f} tok/s", 20, INK, t2c={rf"^{TOK[k]:.1f}": TOKEN, rf"{CEIL[k]:.1f}": MEMORY})
            val.move_to([2.2, yy, 0], aligned_edge=m.LEFT)
            pct = title_text(f"{OF[k] * 100:.0f}%", 40, GREEN if k != "4bit" else AMBER).move_to([6.45, yy, 0], aligned_edge=m.RIGHT)
            rows[k] = (lab, ceil, meas, val, pct)
        ghost = m.DashedVMobject(m.Rectangle(width=per * (FLAT4 - CEIL["4bit"]), height=0.46), num_dashes=14).set_stroke(MEMORY, 2, opacity=0.6)
        ghost.move_to([x0 + per * (CEIL["4bit"] + FLAT4) / 2, -0.5, 0])
        ghost_t = code(f"flat 4 bits: {FLAT4:.0f}", 20, MUTED).next_to(ghost, m.UP, buff=0.1).align_to(ghost, m.RIGHT)

        sp = m.VGroup(title_text(f"{SPEEDUP:.1f}×", 60, TOKEN), m.VGroup(text("faster than BF16", 22, INK),
                      code(f"from {FEWER:.1f}× fewer bytes", 20, MEMORY)).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)).arrange(m.RIGHT, buff=0.3)
        sp.move_to([-6.3, -1.65, 0], aligned_edge=m.LEFT)
        mv = m.VGroup(text("unpacking costs math:", 20, COMPUTE),
                      code(f"mat-vec: 4-bit {MV4:.0f} · BF16 {MV16:.0f} GB/s", 20, MUTED)).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
        mv.move_to([-6.3, -2.65, 0], aligned_edge=m.LEFT)

        px0, pper = 1.75, 2.4 / PRE["bf16"]
        p_t = code("prefill · 2,048-token prompt", 20, MUTED).move_to([px0 - 0.95, -1.3, 0], aligned_edge=m.LEFT)
        pre = m.VGroup()
        for k, yy, nm in (("bf16", -1.8, "BF16"), ("4bit", -2.3, "4-bit")):
            b = hbar(px0, yy, pper * PRE[k], 0.34, COMPUTE, AMBER_FILL)
            pre.add(m.VGroup(code(nm, 20, WEIGHTS).move_to([px0 - 0.15, yy, 0], aligned_edge=m.RIGHT), b,
                             code(f"{PRE[k]:,.0f} tok/s", 20, INK).next_to(b, m.RIGHT, buff=0.15)))
        p_note = text("already compute-bound (Episode 3)", 20, COMPUTE).move_to([px0 - 0.95, -2.8, 0], aligned_edge=m.LEFT)
        ep23 = note_pill("what fits on which Mac · Episode 23", MEMORY, CYAN_FILL, 20).move_to([3.6, -3.27, 0])
        assert pre[0].get_right()[0] < 6.6 and p_note.get_right()[0] < 6.6

        r16, r8, r4 = rows["bf16"], rows["8bit"], rows["4bit"]
        grow = lambda r: m.AnimationGroup(m.FadeIn(r[0]), m.GrowFromEdge(r[1], m.LEFT), m.GrowFromEdge(r[2], m.LEFT), m.FadeIn(r[3], shift=0.1 * m.LEFT))  # noqa: E731
        self.say(
            "[Now the check, on this Mac.] [Sixteen-bit Owl decodes thirty-one tokens a second], [against a ceiling of thirty-four.]",
            m.AnimationGroup(self.new_header("06 · measured", "Four bits on this Mac"), m.FadeIn(stamp), m.FadeIn(legend)),
            m.AnimationGroup(m.FadeIn(r16[0]), m.GrowFromEdge(r16[2], m.LEFT)),
            m.AnimationGroup(m.GrowFromEdge(r16[1], m.LEFT), m.FadeIn(r16[3], shift=0.1 * m.LEFT), m.FadeIn(r16[4])),
        )
        self.say(
            "[Eight-bit: fifty-eight, against sixty-four.] [Four-bit: a hundred and two], [against a hundred and twenty-one], [because the scales are bytes too.]",
            m.AnimationGroup(grow(r8), m.FadeIn(r8[4])),
            m.AnimationGroup(m.FadeIn(r4[0]), m.GrowFromEdge(r4[2], m.LEFT)),
            m.AnimationGroup(m.GrowFromEdge(r4[1], m.LEFT), m.FadeIn(r4[3], shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.Create(ghost), m.FadeIn(ghost_t, shift=0.1 * m.DOWN)),
        )
        self.say(
            "[That's three point three times faster], [from three point six times fewer bytes.]",
            m.FadeIn(sp[0], shift=0.1 * m.UP),
            m.FadeIn(sp[1], shift=0.1 * m.LEFT),
        )
        self.say(
            "[Four-bit reaches eighty-four percent of its ceiling, not ninety-one]: [unpacking each weight costs math], [so the bandwidth is used less well.]",
            m.AnimationGroup(m.FadeIn(r4[4], scale=1.2), m.Circumscribe(r4[4], color=AMBER, buff=0.08)),
            m.FadeIn(mv[0], shift=0.1 * m.UP),
            m.FadeIn(mv[1], shift=0.1 * m.UP),
        )
        self.say(
            "[Prefill gets no faster], [since it's limited by math.] [What fits on which Mac is episode twenty-three.]",
            m.AnimationGroup(m.FadeIn(p_t), m.LaggedStart(*[m.AnimationGroup(m.FadeIn(r[0]), m.GrowFromEdge(r[1], m.LEFT), m.FadeIn(r[2]))
                                                           for r in pre], lag_ratio=0.5)),
            m.FadeIn(p_note, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(ep23, shift=0.1 * m.UP), m.Circumscribe(ep23, color=MEMORY, buff=0.08)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 7 · distillation and pruning

    def fewer_weights(self):
        self.next_section("7 · Or keep fewer weights", notes="Distillation: a small student trained on a big teacher's outputs (logits). Llama 3.2 1B "
                          "and 3B (Meta, 2024-09-25): 'structured pruning in a single shot manner from the Llama 3.1 8B', then 'logits from the "
                          "Llama 3.1 8B and 70B models' as 'token-level targets'. Owl-mini is Llama-3.2-1B "
                          f"({MINI['params'] / 1e9:.2f} B parameters); its 4-bit MLX file is {MINI_GB:.3f} GB and decodes at "
                          f"{MINI_DEC['tok_s_median']:.1f} tok/s on this Mac vs a ceiling of {MINI_CEIL:.0f} (546 GB/s ÷ the file): small models "
                          "lose more of each step to fixed per-step costs (Episode 16). 2:4 structured sparsity (NVIDIA, 2021): 'In each contiguous "
                          "block of four values, two values must be zero'; sparse Tensor Cores 'complete the same effective calculation in half the "
                          "time' — a theoretical 2x on the matmul; real end-to-end LLM gains are smaller and quality must be checked. Cells schematic.")
        big = owl(1.15).move_to([-5.2, 1.0, 0])
        small = owl(0.62).move_to([-1.5, 1.0, 0])
        big_t = m.VGroup(text("teacher", 22, INK, weight=m.BOLD), code(f"Owl-8B · {P8 / 1e9:.0f} B", 20, MUTED)).arrange(m.DOWN, buff=0.06)
        big_t.next_to(big, m.DOWN, buff=0.25)
        small_t = m.VGroup(text("student", 22, INK, weight=m.BOLD), code(f"Owl-mini · {MINI['params'] / 1e9:.2f} B", 20, MUTED)).arrange(m.DOWN, buff=0.06)
        small_t.next_to(small, m.DOWN, buff=0.25).set_y(big_t.get_y())
        flow = arrow(big.get_right() + [0.2, 0, 0], small.get_left() - [0.2, 0, 0], TOKEN)
        probs = [0.55, 0.25, 0.12, 0.08]
        logit_bars = m.VGroup(*[m.Rectangle(width=0.14, height=0.7 * p + 0.04).set_stroke(width=0).set_fill(TOKEN, 1) for p in probs])
        logit_bars.arrange(m.RIGHT, buff=0.06, aligned_edge=m.DOWN).next_to(flow, m.UP, buff=0.12)
        flow_t = code("its outputs", 20, TOKEN).next_to(flow, m.DOWN, buff=0.12)
        made = m.VGroup(text("Owl-mini = Llama 3.2 1B:", 22, INK, weight=m.BOLD),
                        text("pruned from the 8B model,", 22, INK),
                        text("then taught by 8B and 70B", 22, INK),
                        cite("Meta, 2024")).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        made.move_to([0.6, 1.25, 0], aligned_edge=m.LEFT)
        mini_stat = m.VGroup(
            stat(f"{MINI_GB * 1000:.0f}", "MB", "", WEIGHTS, 52),
            stat(f"{MINI_DEC['tok_s_median']:.0f}", "tok/s", "", TOKEN, 52),
        ).arrange(m.RIGHT, buff=0.6, aligned_edge=m.DOWN).move_to([0.6, -0.55, 0], aligned_edge=m.LEFT)
        mini_ceil = m.VGroup(cite("4-bit · measured on this Mac"), cite(f"ceiling {MINI_CEIL:.0f} tok/s = 546 GB/s ÷ {MINI_GB:.2f} GB"))
        mini_ceil.arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT).next_to(mini_stat, m.DOWN, buff=0.15).align_to(mini_stat, m.LEFT)
        assert mini_ceil.get_right()[0] < 6.6 and mini_stat.get_right()[0] < 6.6

        sy = -2.45
        sq = 0.42
        cells = m.VGroup(*[cell(sq, sq, WEIGHTS, BLUE_FILL, 1.6, 0.05) for _ in range(8)])
        for i, c in enumerate(cells):
            c.move_to([-6.1 + sq / 2 + i * 0.5 + (i // 4) * 0.25, sy, 0])
        zero_idx = [1, 2, 4, 7]
        ghosts = m.VGroup(*[m.DashedVMobject(m.Square(sq), num_dashes=12).set_stroke(FAINT, 1.6).move_to(cells[i]) for i in zero_idx])
        s_lab = code("2:4 sparsity", 22, WEIGHTS, weight=m.BOLD).next_to(cells, m.UP, buff=0.22).align_to(cells, m.LEFT)
        s_t = m.VGroup(text("two of every four weights are zero:", 22, INK),
                       text("sparse tensor cores skip them, half the math in theory", 22, INK),
                       cite("NVIDIA, 2021 · real gains are smaller")).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        s_t.move_to([-1.3, sy - 0.05, 0], aligned_edge=m.LEFT)
        assert s_t.get_right()[0] < 6.6

        self.say(
            "[Fewer bits is one way to shrink Owl. Fewer weights is the other]: [distillation trains a small student on a big teacher's outputs.]",
            m.AnimationGroup(self.new_header("07 · fewer weights", "Or keep fewer weights"), m.FadeIn(big, shift=0.2 * m.UP), m.FadeIn(big_t)),
            m.AnimationGroup(m.GrowArrow(flow), m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in logit_bars], lag_ratio=0.2),
                             m.FadeIn(flow_t), m.FadeIn(small, shift=0.2 * m.UP), m.FadeIn(small_t)),
        )
        self.say(
            "[Owl-mini was made that way]: [pruned from the eight-billion model], [then taught by it, and by a seventy-billion one.]",
            m.FadeIn(made[0], shift=0.1 * m.LEFT),
            m.FadeIn(made[1], shift=0.1 * m.LEFT),
            m.AnimationGroup(m.FadeIn(made[2], shift=0.1 * m.LEFT), m.FadeIn(made[3])),
        )
        self.say(
            "[At four bits, it's about seven hundred megabytes], [and on this Mac it writes almost four hundred and ninety tokens a second.]",
            m.FadeIn(mini_stat[0], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(mini_stat[1], shift=0.1 * m.UP), m.FadeIn(mini_ceil)),
        )
        self.say(
            "[Pruning can also follow the hardware]: [with two-to-four sparsity, two of every four weights are zero], [and the tensor cores skip them]: [in theory, half the math.]",
            m.AnimationGroup(m.FadeIn(s_lab), m.LaggedStart(*[m.FadeIn(c, scale=0.6) for c in cells], lag_ratio=0.04)),
            m.AnimationGroup(*[m.FadeOut(cells[i]) for i in zero_idx], m.FadeIn(ghosts), m.FadeIn(s_t[0], shift=0.1 * m.LEFT)),
            m.FadeIn(s_t[1], shift=0.1 * m.LEFT),
            m.FadeIn(s_t[2]),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes=f"Group (a scale and offset per small group: 31% → 9% error on the sample row), protect (GPTQ compensates "
                          f"rounding error, AWQ protects salient channels; k-quants and FP4 scale the scales), count (bits per weight include the "
                          f"scales: {MLX_BPW} → {W45 / GB:.2f} GB for Owl-8B, matching the {FILE['4bit'] / GB:.3f} GB file). On this Mac: "
                          f"{W45 / W16 * 100:.0f}% of the BF16 bytes, {SPEEDUP:.2f}x the decode speed. Next: Episode 20, Mixture of Experts at Serve Time.")
        words = [("Group", "a scale for every small group"), ("Protect", "the weights that matter most"), ("Count", "every bit, scales included")]
        cols = m.VGroup()
        for w, sub in words:
            cols.add(m.VGroup(title_text(w, 72, WEIGHTS), text(sub, 22, MUTED)).arrange(m.DOWN, buff=0.25))
        cols.arrange(m.RIGHT, buff=1.2).move_to([0, 0.75, 0])
        base = 0.85
        for c in cols:
            c[0].shift((base - c[0][0].get_bottom()[1]) * m.UP)
            c[1].move_to([c[0].get_x(), base - 0.55, 0])
        mid = base + 0.3
        arrows = m.VGroup(*[arrow([a[0].get_right()[0] + 0.15, mid, 0], [b[0].get_left()[0] - 0.15, mid, 0], MUTED) for a, b in zip(cols, cols[1:])])
        assert W45 / W16 < 1 / 3 and SPEEDUP > 3
        tagline = text("4.5 bits per weight: under a third of the bytes, over three times the tokens.", 28, INK,
                       t2c={"4.5 bits per weight": WEIGHTS, "under a third of the bytes": MEMORY, "over three times the tokens": TOKEN}).move_to([0, -1.35, 0])
        mascot = owl(0.75).move_to([0, -2.7, 0])

        self.say(
            "[So: group], [protect], [and count.]",
            m.AnimationGroup(self.new_header("recap", "Four bits in three words"), m.FadeIn(cols[0], shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowArrow(arrows[0]), m.FadeIn(cols[1], shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowArrow(arrows[1]), m.FadeIn(cols[2], shift=0.2 * m.UP)),
        )
        self.say(
            "[Four and a half bits per weight]: [under a third of the bytes], [and over three times the tokens.]",
            m.FadeIn(tagline, shift=0.1 * m.UP),
            m.Circumscribe(tagline, color=MEMORY, buff=0.1),
            m.FadeIn(mascot, shift=0.2 * m.UP),
        )
        self.wait(0.4)
        self.clear()
