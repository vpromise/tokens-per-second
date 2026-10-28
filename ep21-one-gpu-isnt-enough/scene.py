"""Tokens per Second · Episode 21 · When One GPU Isn't Enough (Module 5 · Fewer Bits, More GPUs)

Splitting a model across GPUs, and what the wires between them cost. Running example: Owl moves up to
Owl-70B (Llama-3.1-70B: 70.55 B parameters, 141.1 GB of BF16 weights, 80 layers, d_model 8,192), which
does not fit one 80 GB H100, so it runs tensor-parallel on four.

Sections: too big (Owl-8B fits one GPU, more students = more copies; Owl-70B is 141.1 GB, 61 GB over one
H100; Episode 18's 8 bits squeeze it in with 9.4 GB left for the KV cache) · tensor parallel (every matrix
sliced four ways, the first of each pair by columns, the second by rows; 35.3 GB per GPU; all four read at
once: Episode 2's ceiling 3.35 TB/s ÷ 141.1 GB = 23.7 → 4 × 3.35 TB/s ÷ 141.1 GB = 95.0 tok/s) · all-reduce
(each GPU ends a block with a partial sum; a ring all-reduce adds them, exactly; 2 per layer × 80 layers =
160 per token, each batch × d_model × 2 bytes = 16 KiB per token, 2.6 MB per token) · the wires (a batch of
64: 168 MB per step; NVLink 450 GB/s each way → 0.37 ms, PCIe Gen5 64 GB/s → 2.62 ms, InfiniBand 400 Gb/s
= 50 GB/s → 3.36 ms; against the 10.53 ms step: +3.5% vs +25%; so tensor parallel stays inside one
8-GPU NVLink server; Owl-70B uses half of it) · pipeline parallel (20 layers per GPU; one 16 KiB vector per
token per hop, 3 hops = 48 KiB; stages can sit in different servers; one token visits all four: 4 × 10.53
ms = 42 ms → 23.7 tok/s, no latency gain; one batch in flight → bubbles, each GPU busy 25%; four in flight
→ 4× throughput, same latency) · experts and long prompts (Episode 20's expert parallelism and all-to-all;
context parallelism: a long prompt's tokens split across GPUs, keys and values passed around a ring) · the
price (batch 1: 2.6 MB over NVLink takes 6 µs, but 160 syncs at, for example, 10 µs = 1.6 ms → 95 → 82
tok/s; Databricks: 2 → 4 GPUs cut token latency only ~12% at batch 1) · which split when (a table) · recap
(slice, sum, stage).

Sources (facts dated October 2026):
- owl_reference.py: MODELS["owl-70b"] (Llama-3.1-70B config.json: 70.55 B params, 80 layers, d_model 8192,
  8 KV heads × 128), MODELS["owl-8b"], HARDWARE["H100"] (H100 SXM 3.35 TB/s, 989 TFLOP/s BF16 dense, 80 GB;
  NVIDIA H100 page), weight_bytes, decode_ceiling, step_time.
- research/notes.md §10: data / tensor / pipeline / expert / sequence-context parallelism; the Llama-3.1-70B
  worked example (141.1 GB; TP4 35.3 GB per GPU, 13.4 TB/s → ≈ 95 tok/s "before communication"); interconnect
  table [vendor pages, dated]: H100 NVLink 4 900 GB/s total per GPU (≈ 450 GB/s each way), PCIe Gen5 x16
  128 GB/s total (≈ 64 each way), InfiniBand CX7 400 Gb/s ≈ 50 GB/s per NIC (DeepEP README) — the same link
  speeds as Episode 12; Databricks quote.
- Shoeybi et al., "Megatron-LM" (2019, arXiv 1909.08053, §3, read 2026-10-24): "we partition the first GEMM in
  this column parallel fashion and split the second GEMM along its rows so it takes the output of the GeLU
  layer directly without requiring any communication"; "using only two all-reduces in the forward path and
  two in the backward path" per transformer layer.
- kipply, "Transformer Inference Arithmetic" (2022, read 2026-10-24): "approximately 8 microseconds per
  message sent" (from a Citadel paper, "for V100 NVLink") — used only as the order of magnitude behind the
  illustrative 10 µs per sync.
- Databricks, "LLM Inference Performance Engineering: Best Practices" (2023, read 2026-10-24): "At batch size
  1, going from 2x to 4x only reduces token latency by ~12%"; "At batch size 16, latency with 4x is 33% lower
  than with 2x" (MPT-7B on A100-40GB).
- vLLM docs, "Parallelism and scaling" (read 2026-10-24): single node → tensor parallelism; multi-node →
  "Set tensor_parallel_size to the number of GPUs per node and pipeline_parallel_size to the number of
  nodes"; without NVLink, prefer pipeline parallelism.
- NVIDIA HGX H100: eight H100s joined by NVLink through NVSwitch in one server.
- Liu et al., "Ring Attention with Blockwise Transformers" (2023): a sequence split across devices, key-value
  blocks passed around a ring (context parallelism for long prompts, notes §10/§11).

Illustrative (said "for example" or obvious): the 8 × 8 matrices and the toy partial sums (a seeded NumPy
example, checked exact below); the batch of 64 students; the 10 µs per sync; two servers with two pipeline
stages each; the pipeline schedule (micro-batches 1–4 over 8 time slots); the expert routes and the 16-token
prompt in section 6. Times over a link are bytes ÷ one-way link speed (a lower bound: a ring all-reduce sends
2(N−1)/N = 1.5× the message per GPU for N = 4, and latency comes on top). Asserts below tie every quoted
number to the data.

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

GB, MB, KIB = 1e9, 1e6, 1024
O70 = MODELS["owl-70b"]                                   # Llama-3.1-70B
N = 4                                                     # GPUs
H_BW, H_MEM = GPU["bandwidth"], GPU["memory"]             # 3.35 TB/s, 80 GB
W8B = weight_bytes()                                      # Owl-8B: 16.06 GB
W70 = weight_bytes(O70)                                   # 141.1 GB
OVER = W70 - H_MEM                                        # 61.1 GB too many
W70_8 = weight_bytes(O70, bits=8)                         # 70.55 GB (Episode 18's 8 bits)
ROOM8 = H_MEM - W70_8                                     # 9.45 GB left for the KV cache
PER_GPU = W70 / N                                         # 35.3 GB per GPU
TP4 = dict(GPU, name="4 × H100 SXM, tensor parallel", bandwidth=N * H_BW, peak=N * GPU["peak"])
CEIL1 = decode_ceiling(O70, GPU)                          # 23.7 tok/s: one H100's bandwidth
CEIL4 = decode_ceiling(O70, TP4)                          # 95.0 tok/s: four, reading at once
LAYERS, D = O70["layers"], O70["d_model"]                 # 80, 8192
AR_PER_LAYER = 2                                          # Megatron-LM: after attention, after the MLP
N_AR = AR_PER_LAYER * LAYERS                              # 160 all-reduces per token


def allreduce_bytes(batch=1):
    """One all-reduce carries one d_model vector per token in the step, in 16-bit."""
    return batch * D * 2


AR1 = allreduce_bytes(1)                                  # 16,384 B = 16 KiB
PER_TOKEN = N_AR * AR1                                    # 2.62 MB per token
BATCH = 64                                                # for example: 64 students at once
STEP_BYTES = N_AR * allreduce_bytes(BATCH)                # 167.8 MB per step
NVLINK, PCIE, IB = 450e9, 64e9, 400e9 / 8                 # bytes/s one way (Episode 12's link speeds)
T_NV, T_PCIE, T_IB = STEP_BYTES / NVLINK, STEP_BYTES / PCIE, STEP_BYTES / IB
STEP = step_time(O70, TP4, batch=BATCH)                   # 10.53 ms: each GPU reads its 35.3 GB
SHARE_NV, SHARE_PCIE = T_NV / STEP, T_PCIE / STEP         # 3.5% and 25%
STEP1 = step_time(O70, TP4)                               # 10.53 ms at batch 1
T1_NV = PER_TOKEN / NVLINK                                # 5.8 µs: the bytes of one token over NVLink
SYNC = 10e-6                                              # for example: one sync's fixed cost
SYNC_COST = N_AR * SYNC                                   # 1.6 ms per token
CEIL4_SYNC = 1 / (STEP1 + SYNC_COST)                      # 82 tok/s
STAGE = W70 / N / H_BW                                    # 10.53 ms: one pipeline stage reads 35.3 GB
PP_TOKEN = N * STAGE                                      # 42.1 ms per token through four stages
PP_RATE = 1 / PP_TOKEN                                    # 23.7 tok/s
PP_HOPS = N - 1
PP_BYTES = PP_HOPS * AR1                                  # 48 KiB per token between stages
PER_STAGE = LAYERS // N                                   # 20 layers per GPU
SERVER_GPUS = 8

# tensor parallelism is exact: a toy MLP block, split Megatron-style across four GPUs (for example)
_rng = np.random.default_rng(25)
XV = _rng.normal(0, 1, 8)
A_M = _rng.normal(0, 0.5, (8, 8))
B_M = _rng.normal(0, 0.5, (8, 8))
FULL = np.maximum(XV @ A_M, 0) @ B_M                      # the unsplit block
SPLIT = np.split(np.arange(8), N)                         # columns of A = rows of B, two per GPU
PARTS = [np.maximum(XV @ A_M[:, c], 0) @ B_M[c, :] for c in SPLIT]   # each GPU's partial sum

# ---------------------------------------------------------------------------- asserts: the words match the data
assert round(W70 / GB, 1) == 141.1 and round(W70 / GB) == 141 and H_MEM == 80e9   # "a hundred and forty-one", "eighty"
assert round(W8B / GB) == 16 and round(OVER / GB) == 61
assert round(W70_8 / GB, 2) == 70.55 and ROOM8 < 10e9 and round(ROOM8 / GB, 2) == 9.45  # "under ten gigabytes"
assert round(PER_GPU / GB, 1) == 35.3                                                # "thirty-five point three"
assert round(CEIL1) == 24 and round(CEIL4) == 95 and abs(CEIL4 / CEIL1 - N) < 1e-9   # "twenty-four", "ninety-five"
assert round(N * H_BW / 1e12, 1) == 13.4
assert N_AR == 160 and D == 8192 and AR1 == 16 * KIB                                 # "a hundred and sixty", "sixteen kibibytes"
assert round(PER_TOKEN / MB, 1) == 2.6                                               # "two point six megabytes"
assert round(STEP_BYTES / MB) == 168                                                 # "a hundred and sixty-eight megabytes"
assert f"{T_NV * 1e3:.2f}" == "0.37" and f"{T_PCIE * 1e3:.1f}" == "2.6" and f"{T_IB * 1e3:.1f}" == "3.4"
assert round(STEP * 1e3, 1) == 10.5 and round(STEP1 * 1e3, 1) == 10.5               # "ten and a half milliseconds"
assert SHARE_NV < 0.04 and round(SHARE_PCIE, 2) == 0.25                              # "under four percent", "a quarter"
assert IB == 50e9 and round(T1_NV * 1e6) == 6
assert round(SYNC_COST * 1e3, 1) == 1.6 and round(CEIL4_SYNC) == 82                  # "one point six", "eighty-two"
assert round(PP_TOKEN * 1e3) == 42 and round(PP_RATE) == 24 and PER_STAGE == 20      # "forty-two", "twenty layers"
assert PP_BYTES == 48 * KIB
assert np.allclose(sum(PARTS), FULL) and min(np.abs(p).max() for p in PARTS) > 0.5   # "exactly"

# ---------------------------------------------------------------------------- look

PREFIX = "ep21"
GPU_ICON = recolored("gpu", MUTED, CARD_HI, prefix=PREFIX)
XS = [-4.8, -1.6, 1.6, 4.8]                    # four columns: one per GPU
FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIKLMNOPRSTUVWXYZ0123456789")


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


def row_of(parts, buff=0.22):
    """Texts in a row on one baseline."""
    row = m.VGroup(*parts).arrange(m.RIGHT, buff=buff)
    y0 = baseline(parts[0])
    for p in parts[1:]:
        sit(p, y0)
    return row


def left(mob, x, y):
    return mob.move_to([0, y, 0]).align_to([x, 0, 0], m.LEFT)


def hbar(x0, y, width, height, color, fill, stroke=2):
    bar = m.Rectangle(width=max(width, 0.02), height=height).set_stroke(color, stroke).set_fill(fill, 1)
    return bar.move_to([x0 + max(width, 0.02) / 2, y, 0])


def gpu_box(x, y, w, h, title, sub=None, title_size=22):
    """A GPU drawn as a card with its name in the top-left corner (and an optional MUTED tag on the right)."""
    box = card(w, h, EDGE, CARD, radius=0.16).move_to([x, y, 0])
    name = code(title, title_size, INK, weight=m.BOLD)
    name.move_to(box.get_corner(m.UL) + np.array([0.22 + name.width / 2, -0.3, 0]))
    parts = [box, name]
    if sub:
        tag = code(sub, 20, MUTED)
        tag.move_to(box.get_corner(m.UR) + np.array([-0.22 - tag.width / 2, 0, 0]))
        sit(tag, baseline(name))
        parts.append(tag)
    return m.VGroup(*parts)


def bars(values, x, y0, scale, color, fill, width=0.2, pitch=0.27):
    """A vector as small bars around a baseline at y0 (up for positive, down for negative)."""
    group = m.VGroup()
    x_first = x - pitch * (len(values) - 1) / 2
    for i, v in enumerate(values):
        h = max(abs(v) * scale, 0.03)
        b = m.Rectangle(width=width, height=h).set_stroke(color, 1.5).set_fill(fill, 1)
        b.move_to([x_first + i * pitch, y0 + (h / 2 if v >= 0 else -h / 2), 0])
        group.add(b)
    return group


def introducing(animation):
    """Mark an animation of parts of a just-introduced group so play() waits for its cue (gotcha 21)."""
    animation.introducer = True
    return animation


def flash(path, color=MEMORY, width=7, run_time=0.45, time_width=0.6):
    return m.ShowPassingFlash(path.copy().set_stroke(color, width), time_width=time_width, run_time=run_time)


def table_row(cells, xs, y, sizes, colors, mono):
    out = []
    for s, x, size, c, mo in zip(cells, xs, sizes, colors, mono):
        t = code(s, size, c, weight=m.BOLD if mo == "b" else m.NORMAL) if mo else text(s, size, c)
        out.append(left(t, x, y))
    y0 = baseline(out[0])
    for t in out[1:]:
        sit(t, y0)
    return m.VGroup(*out)


# ---------------------------------------------------------------------------- the episode


class Ep21WhenOneGPUIsntEnough(Episode):
    number = 21

    def lesson(self):
        self.too_big()
        self.tensor()
        self.all_reduce()
        self.wires()
        self.pipeline()
        self.more_splits()
        self.price()
        self.which()
        self.recap()

    # ------------------------------------------------------------------ 1 · too big

    def too_big(self):
        self.next_section("1 · Too big for one GPU", notes=(
            f"Owl-8B ({W8B / GB:.2f} GB in BF16) fits one H100 (80 GB): more students → more copies (data parallel). "
            f"Owl-70B = Llama-3.1-70B: {O70['params'] / 1e9:.2f} B parameters × 2 bytes = {W70 / GB:.1f} GB "
            f"(weight_bytes(MODELS['owl-70b'])), {OVER / GB:.1f} GB more than one H100's {H_MEM / GB:.0f} GB (NVIDIA spec). "
            f"Episode 18's 8 bits: {W70_8 / GB:.2f} GB, leaving {ROOM8 / GB:.2f} GB for the KV cache (activations and "
            "runtime overhead come out of that too). Bar lengths to scale."))
        mascot = owl(0.8).move_to([5.75, -2.65, 0])
        copies = m.VGroup()
        yc = 0.6
        for x in XS:
            box = card(2.85, 1.9, EDGE, CARD, radius=0.16).move_to([x, yc, 0])
            name = code("H100 · 80 GB", 20, MUTED).move_to([x, yc + 0.68, 0])
            chip_icon = icon(GPU_ICON, 0.5).move_to([x, yc + 0.17, 0])
            lab = code(f"Owl-8B · {W8B / GB:.0f} GB", 20, WEIGHTS).move_to([x, yc - 0.38, 0])
            meter = Meter(2.2, W8B / H_MEM, WEIGHTS, 0.18).move_to([x, yc - 0.68, 0])
            copies.add(m.VGroup(box, name, chip_icon, lab, meter))
        dp = code("more students → more copies · data parallel", 20, MUTED).move_to([0, -0.85, 0])

        x0, u = -5.6, 0.065                                  # scene units per GB
        cap = left(code(f"Owl-70B · {O70['name']} · {O70['params'] / 1e9:.2f} B parameters × 2 bytes", 20, MUTED), x0, 2.2)
        bar16 = hbar(x0, 1.0, W70 / GB * u, 0.5, WEIGHTS, BLUE_FILL)
        val16 = code(f"{W70 / GB:.1f} GB", 22, WEIGHTS, weight=m.BOLD).next_to(bar16, m.RIGHT, buff=0.2)
        cap_w = H_MEM / GB * u
        box16 = m.Rectangle(width=cap_w + 0.2, height=0.9).set_stroke(INK, 2.5).move_to([x0 + cap_w / 2, 1.0, 0])
        box_lab = code("one H100 · 80 GB", 20, INK).next_to(box16, m.UP, buff=0.12).align_to(box16, m.LEFT)
        over = m.Rectangle(width=OVER / GB * u, height=0.5).set_stroke(RED, 2).set_fill(RED_FILL, 1)
        over.move_to([x0 + cap_w + OVER / GB * u / 2, 1.0, 0]).set_z_index(1)
        over_lab = code(f"{OVER / GB:.0f} GB too many", 20, RED).next_to(over, m.DOWN, buff=0.3)

        tag8 = left(code("8-bit · Episode 18", 20, WEIGHTS), x0, -0.3)
        bar8 = hbar(x0, -0.95, W70_8 / GB * u, 0.5, WEIGHTS, BLUE_FILL)
        val8 = code(f"{W70_8 / GB:.2f} GB", 20, INK).move_to(bar8).align_to(bar8, m.LEFT).shift(0.15 * m.RIGHT).set_z_index(2)
        box8 = box16.copy().set_y(-0.95)
        kv8 = m.Rectangle(width=ROOM8 / GB * u, height=0.5).set_stroke(KV, 2).set_fill(VIOLET_FILL, 1)
        kv8.move_to([x0 + (W70_8 + ROOM8 / 2) / GB * u, -0.95, 0])
        kv_lab = code(f"{ROOM8 / GB:.2f} GB left for the KV cache", 20, KV).next_to(box8, m.RIGHT, buff=0.3)

        gpus = m.VGroup(*[icon(GPU_ICON, 0.55).move_to([-5.3 + i * 1.3, -2.45, 0]) for i in range(N)])
        links = m.VGroup(*[m.Line(gpus[i].get_right() + [0.06, 0, 0], gpus[i + 1].get_left() - [0.06, 0, 0]).set_stroke(MEMORY, 4)
                           for i in range(N - 1)])
        wire_lab = code("the wires", 20, MEMORY).next_to(gpus, m.DOWN, buff=0.25)
        ways = m.VGroup(*[pill(w, WEIGHTS, BLUE_FILL, 20, mono=True) for w in ("tensor", "pipeline", "expert", "context")])
        ways.arrange_in_grid(rows=2, cols=2, buff=(0.3, 0.25)).move_to([1.5, -2.55, 0])

        self.say(
            "[Until now, Owl fit on one GPU], [and more students just meant more copies.]",
            m.AnimationGroup(self.new_header("01 · too big", "One GPU isn't enough"), m.FadeIn(copies[0], shift=0.1 * m.UP),
                             m.FadeIn(mascot, shift=0.2 * m.UP)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.LEFT) for c in copies[1:]], lag_ratio=0.3), m.FadeIn(dp)),
        )
        self.say(
            "[Now Owl moves up to a seventy-billion-parameter model.] [In sixteen-bit, that's a hundred and forty-one gigabytes of weights.]",
            m.AnimationGroup(m.FadeOut(copies, shift=0.2 * m.UP), m.FadeOut(dp), m.FadeIn(cap, shift=0.1 * m.UP)),
            m.AnimationGroup(m.GrowFromEdge(bar16, m.LEFT), m.FadeIn(val16)),
        )
        self.say(
            "[One H100 holds eighty.] [It doesn't fit.]",
            m.AnimationGroup(m.Create(box16), m.FadeIn(box_lab)),
            m.AnimationGroup(m.FadeIn(over), m.FadeIn(over_lab, shift=0.1 * m.UP)),
        )
        self.say(
            "[Episode eighteen's fix is fewer bits]: [at eight bits, it squeezes in], [leaving under ten gigabytes for the KV cache.]",
            m.FadeIn(tag8, shift=0.1 * m.UP),
            m.AnimationGroup(m.Create(box8), m.GrowFromEdge(bar8, m.LEFT), m.FadeIn(val8)),
            m.AnimationGroup(m.FadeIn(kv8), m.FadeIn(kv_lab, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[The other fix is more GPUs.] [Today, four ways to split a model across them], [and what the wires between them cost.]",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(g, shift=0.1 * m.UP) for g in gpus], lag_ratio=0.2), m.Create(links)),
            m.LaggedStart(*[m.FadeIn(w, shift=0.1 * m.UP) for w in ways], lag_ratio=0.2),
            m.AnimationGroup(m.FadeIn(wire_lab), *[flash(l, MEMORY, 9, 0.8) for l in links]),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 2 · tensor parallel

    def tensor(self):
        self.next_section("2 · Tensor parallel", notes=(
            "Tensor parallelism (Megatron-LM, Shoeybi et al. 2019 §3): the first matrix of each pair (Q/K/V, the MLP's "
            "up projection) is split by columns, the second (the attention output, the MLP's down projection) by rows, so "
            "each GPU's slice of the first feeds its slice of the second with no communication in between. The 8 × 8 "
            f"grids are schematic. Per GPU: {W70 / GB:.1f} ÷ {N} = {PER_GPU / GB:.2f} GB of weights, "
            f"{(H_MEM - PER_GPU) / GB:.1f} GB of the 80 GB left (KV cache, activations, runtime). Episode 2's ceiling "
            f"(decode_ceiling): one H100's {H_BW / 1e12:.2f} TB/s ÷ {W70 / GB:.1f} GB = {CEIL1:.1f} tok/s (if one GPU could "
            f"hold it); four reading their quarters at once: {N} × {H_BW / 1e12:.2f} = {N * H_BW / 1e12:.1f} TB/s ÷ "
            f"{W70 / GB:.1f} GB = {CEIL4:.1f} tok/s — before communication (notes §10)."))
        cell, gap = 0.26, 0.03
        A = cell_grid(8, 8, cell, gap, WEIGHTS, BLUE_FILL).move_to([-2.3, 1.2, 0]).set_z_index(2)
        B = cell_grid(8, 8, cell, gap, WEIGHTS, BLUE_FILL).move_to([2.3, 1.2, 0]).set_z_index(2)
        la = code("cut by columns", 20, MUTED).next_to(A, m.DOWN, buff=0.25)
        lb = code("cut by rows", 20, MUTED).next_to(B, m.DOWN, buff=0.25)
        feed = arrow(A.get_right(), B.get_left(), MUTED, buff=0.25)

        def xmid(c):
            return (A[0][c - 1].get_right()[0] + A[0][c].get_left()[0]) / 2

        def ymid(r):
            return (B[r - 1][0].get_bottom()[1] + B[r][0].get_top()[1]) / 2

        cuts_a = m.VGroup(*[m.DashedLine([xmid(c), A.get_top()[1] + 0.1, 0], [xmid(c), A.get_bottom()[1] - 0.1, 0],
                                         dash_length=0.08).set_stroke(INK, 3) for c in (2, 4, 6)]).set_z_index(3)
        cuts_b = m.VGroup(*[m.DashedLine([B.get_left()[0] - 0.1, ymid(r), 0], [B.get_right()[0] + 0.1, ymid(r), 0],
                                         dash_length=0.08).set_stroke(INK, 3) for r in (2, 4, 6)]).set_z_index(3)
        strips_a = [m.VGroup(*[A[r][c] for r in range(8) for c in cols]) for cols in SPLIT]
        strips_b = [m.VGroup(*[B[r][c] for r in rows for c in range(8)]) for rows in SPLIT]

        y_card, k_s, y_strip = -1.6, 0.68, -1.4
        cards, outlines, tracks, w_segs, w_labs, kv_segs, kv_labs = (m.VGroup() for _ in range(7))
        for k, x in enumerate(XS):
            cards.add(gpu_box(x, y_card, 3.0, 3.1, f"GPU {k + 1}", "H100"))
            ta = strips_a[k].copy().scale(k_s).move_to([x - 0.9, y_strip, 0])
            tb = strips_b[k].copy().scale(k_s).move_to([x + 0.35, y_strip, 0])
            outlines.add(m.VGroup(m.SurroundingRectangle(ta, buff=0.05), m.SurroundingRectangle(tb, buff=0.05)))
            tw, ty = 2.6, -2.85
            track = m.Rectangle(width=tw, height=0.22).set_stroke(EDGE, 2).set_fill(CARD_HI, 1).move_to([x, ty, 0]).set_z_index(1)
            ww = tw * PER_GPU / H_MEM
            wseg = m.Rectangle(width=ww, height=0.22).set_stroke(WEIGHTS, 1.5).set_fill(WEIGHTS, 1)
            wseg.move_to([x - tw / 2 + ww / 2, ty, 0]).set_z_index(2)
            kvseg = m.Rectangle(width=tw - ww, height=0.22).set_stroke(KV, 1.5).set_fill(VIOLET_FILL, 1)
            kvseg.move_to([x - tw / 2 + ww + (tw - ww) / 2, ty, 0]).set_z_index(2)
            wl = left(code(f"{PER_GPU / GB:.1f} GB", 20, WEIGHTS), x - tw / 2, -2.5).set_z_index(1)
            kl = code("KV room", 20, KV).move_to([0, -2.5, 0]).align_to([x + tw / 2, 0, 0], m.RIGHT).set_z_index(1)
            sit(kl, baseline(wl))
            tracks.add(track), w_segs.add(wseg), kv_segs.add(kvseg), w_labs.add(wl), kv_labs.add(kl)
        targets_a = [strips_a[k].copy().scale(k_s).move_to([x - 0.9, y_strip, 0]) for k, x in enumerate(XS)]
        targets_b = [strips_b[k].copy().scale(k_s).move_to([x + 0.35, y_strip, 0]) for k, x in enumerate(XS)]

        bw = code(f"{N} × {H_BW / 1e12:.2f} TB/s = {N * H_BW / 1e12:.1f} TB/s, for the same {W70 / GB:.1f} GB", 22, INK,
                  t2c={rf"{N * H_BW / 1e12:.1f} TB/s": MEMORY, rf"{W70 / GB:.1f} GB": WEIGHTS}).move_to([0, 2.25, 0])
        ep2 = cite("tok/s ≈ bandwidth ÷ bytes per token · Episode 2").move_to([0, 1.8, 0])
        s1 = stat(f"{CEIL1:.0f}", "tok/s", "one H100's bandwidth", MEMORY, 60)
        left(s1, -5.0, 0.78)
        s4 = stat(f"{CEIL4:.0f}", "tok/s", "four H100s, reading at once", MEMORY, 60)
        left(s4, 1.2, 0.78)
        s4.align_to(s1, m.UP)
        go = arrow(s1[0].get_right() + [1.9, 0, 0], s4[0].get_left(), MUTED, buff=0.3)

        self.say(
            "[Tensor parallelism slices every weight matrix]: [the first of each pair by columns, the second by rows.]",
            m.AnimationGroup(self.new_header("02 · tensor parallel", "Slice every matrix"), m.FadeIn(A), m.FadeIn(B),
                             m.GrowArrow(feed)),
            m.AnimationGroup(m.Create(cuts_a), m.Create(cuts_b), m.FadeIn(la), m.FadeIn(lb)),
        )
        self.say(
            "[Each of four GPUs gets one strip of each.]",
            m.AnimationGroup(m.FadeOut(cuts_a), m.FadeOut(cuts_b), m.FadeOut(la), m.FadeOut(lb), m.FadeOut(feed),
                             m.LaggedStart(*[m.FadeIn(c) for c in cards], lag_ratio=0.1),
                             m.LaggedStart(*[m.AnimationGroup(m.Transform(strips_a[k], targets_a[k]), m.Transform(strips_b[k], targets_b[k]))
                                             for k in range(N)], lag_ratio=0.15)),
        )
        self.say(
            "[So each GPU holds a quarter of the weights]: [thirty-five point three gigabytes], [leaving room for the KV cache.]",
            m.LaggedStart(*[m.Circumscribe(m.VGroup(strips_a[k], strips_b[k]), color=WEIGHTS, buff=0.1) for k in range(N)], lag_ratio=0.15),
            m.AnimationGroup(*[m.AnimationGroup(m.FadeIn(tracks[k]), m.GrowFromEdge(w_segs[k], m.LEFT), m.FadeIn(w_labs[k])) for k in range(N)]),
            m.AnimationGroup(*[m.AnimationGroup(m.GrowFromEdge(kv_segs[k], m.LEFT), m.FadeIn(kv_labs[k])) for k in range(N)]),
        )
        self.say(
            "[For every token, all four GPUs read their own quarter at the same time.]",
            m.Succession(*[m.AnimationGroup(*[flash(r, MEMORY, 6, 1.0, 0.5) for o in outlines for r in o]) for _ in range(2)]),
        )
        self.say(
            "[That's four times the bandwidth, for the same bytes.] [Episode two's speed limit goes from twenty-four tokens a second], [to ninety-five.]",
            m.AnimationGroup(m.FadeIn(bw, shift=0.1 * m.UP), m.FadeIn(ep2)),
            m.FadeIn(s1, shift=0.1 * m.UP),
            m.AnimationGroup(m.GrowArrow(go), m.FadeIn(s4, shift=0.1 * m.LEFT)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 3 · all-reduce

    def all_reduce(self):
        self.next_section("3 · All-reduce", notes=(
            "Each GPU multiplies by its column strip, then by its row strip, and ends the block holding a full-size partial "
            "sum; the four partial sums add up to the unsplit block's output exactly (checked here on a seeded toy block: "
            "x · 8 × 8 · ReLU · 8 × 8, two columns and two rows per GPU — illustrative values). The sum is an all-reduce, "
            "drawn as a ring (reduce-scatter then all-gather: 3 + 3 steps for 4 GPUs). Megatron-LM: 'only two all-reduces "
            f"in the forward path' per layer (after attention, after the MLP) → {AR_PER_LAYER} × {LAYERS} = {N_AR} per token. "
            f"Each carries batch × d_model × 2 bytes = 1 × {D:,} × 2 = {AR1:,} B = 16 KiB per token; {N_AR} × {AR1:,} B = "
            f"{PER_TOKEN:,} B ≈ {PER_TOKEN / MB:.1f} MB ({PER_TOKEN / 2 ** 20:.1f} MiB) per token."))
        ys, ch = 0.95, 2.1
        cards = m.VGroup(*[gpu_box(x, ys, 2.6, ch, f"GPU {k + 1}") for k, x in enumerate(XS)])
        scale = 0.5 / max(np.abs(np.array(PARTS)).max(), np.abs(FULL).max())
        y_base = 0.92
        part_bars = [bars(PARTS[k], x, y_base, scale, MUTED, CARD_HI).set_z_index(1) for k, x in enumerate(XS)]
        sum_bars = [bars(FULL, x, y_base, scale, INK, INK).set_z_index(1) for x in XS]
        axes_ = m.VGroup(*[m.Line([x - 1.15, y_base, 0], [x + 1.15, y_base, 0]).set_stroke(FAINT, 1.5) for x in XS])
        p_labs = [code("partial sum", 20, MUTED).move_to([x, 0.15, 0]) for x in XS]
        s_labs = [code("full sum", 20, INK, weight=m.BOLD).move_to([x, 0.15, 0]) for x in XS]
        for a, b in zip(p_labs, s_labs):
            sit(b, baseline(a))

        top, bot = ys + ch / 2, ys - ch / 2
        paths = [m.ArcBetweenPoints([XS[k] + 0.6, top, 0], [XS[k + 1] - 0.2, top, 0], angle=-np.pi / 2) for k in range(N - 1)]
        paths.append(m.ArcBetweenPoints([XS[-1] + 0.85, bot, 0], [XS[0] - 0.85, bot, 0], angle=-np.pi / 8))
        ring = m.VGroup(*[p.copy().set_stroke(MUTED, 3).add_tip(tip_length=0.16) for p in paths])
        steps = [m.AnimationGroup(*[flash(p, MEMORY, 8, 0.42) for p in paths]) for _ in range(2 * (N - 1))]
        ar_pill = pill("all-reduce", MEMORY, CYAN_FILL, 22, mono=True).move_to([0, -1.05, 0])
        check = pill("the four partial sums add up to the unsplit layer, exactly", TOKEN, GREEN_FILL, 20).move_to([0, -1.72, 0])

        x_first, pitch, y_t = -4.72, 0.12, -1.75
        ticks = m.VGroup()
        for layer in range(LAYERS):
            for j in range(AR_PER_LAYER):
                t = m.Rectangle(width=0.032, height=0.3).set_stroke(width=0).set_fill(MEMORY, 1)
                ticks.add(t.move_to([x_first + layer * pitch + j * 0.052, y_t, 0]))
        l1 = code("layer 1", 20, MUTED).move_to([0, y_t, 0]).align_to([ticks.get_left()[0] - 0.25, 0, 0], m.RIGHT)
        l80 = left(code("layer 80", 20, MUTED), ticks.get_right()[0] + 0.25, y_t)
        both = left(code("attention · MLP", 20, MUTED), ticks.get_left()[0], y_t + 0.38)
        line1 = code(f"{AR_PER_LAYER} per layer × {LAYERS} layers = {N_AR} all-reduces per token", 22, INK,
                     t2c={rf"{N_AR} all-reduces": MEMORY}).move_to([0, -2.35, 0])
        p1 = code("each: batch × d_model × 2 bytes", 22, INK)
        p2 = code(f"= 1 × {D:,} × 2", 22, INK)
        p3 = code("= 16 KiB", 22, MEMORY, weight=m.BOLD)
        line2 = row_of([p1, p2, p3], 0.25).move_to([0, -2.85, 0])
        line3 = code(f"{N_AR} × 16 KiB = {PER_TOKEN / MB:.1f} MB for every token", 22, INK,
                     t2c={rf"{PER_TOKEN / MB:.1f} MB": MEMORY}).move_to([0, -3.35, 0])

        self.say(
            "[But each slice computes only part of the answer]: [a partial sum.]",
            m.AnimationGroup(self.new_header("03 · all-reduce", "Sum the pieces"), m.LaggedStart(*[m.FadeIn(c) for c in cards], lag_ratio=0.1)),
            m.AnimationGroup(m.FadeIn(axes_), *[m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in pb], lag_ratio=0.08) for pb in part_bars],
                             *[m.FadeIn(lab) for lab in p_labs]),
        )
        self.say(
            "[To finish the layer, every GPU needs the total.] [So they pass their pieces around a ring, adding as they go]: [the step is called all-reduce.]",
            m.Create(ring),
            m.Succession(*steps, m.AnimationGroup(*[m.ReplacementTransform(a, b) for a, b in zip(part_bars, sum_bars)],
                                                  *[m.ReplacementTransform(a, b) for a, b in zip(p_labs, s_labs)], run_time=0.8)),
            m.FadeIn(ar_pill, shift=0.1 * m.UP),
        )
        self.say(
            "[The result is exactly what one giant GPU would compute.]",
            m.AnimationGroup(m.FadeIn(check, shift=0.1 * m.UP), *[m.Circumscribe(m.VGroup(*sb), color=TOKEN, buff=0.06) for sb in sum_bars]),
        )
        self.say(
            "[It happens twice in every layer], [after attention and after the feed-forward block]: [eighty layers, a hundred and sixty all-reduces per token.]",
            m.AnimationGroup(m.FadeOut(check, shift=0.1 * m.DOWN), m.FadeIn(ticks[0]), m.FadeIn(ticks[1]), m.FadeIn(l1)),
            m.FadeIn(both, shift=0.1 * m.UP),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(t) for t in ticks[2:]], lag_ratio=0.03), m.FadeIn(l80), m.FadeIn(line1)),
        )
        self.say(
            "[Each one carries one vector per token]: [eight thousand one hundred and ninety-two numbers, at two bytes each]: [sixteen kibibytes.]",
            m.FadeIn(p1, shift=0.1 * m.UP),
            m.FadeIn(p2, shift=0.1 * m.UP),
            m.FadeIn(p3, shift=0.1 * m.LEFT),
        )
        self.say(
            "[A hundred and sixty of them]: [two point six megabytes for every token.]",
            m.Circumscribe(ticks, color=MEMORY, buff=0.08),
            m.FadeIn(line3, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 4 · the wires

    def wires(self):
        self.next_section("4 · The wires", notes=(
            f"For example, {BATCH} students decoding together: {BATCH} × {PER_TOKEN / MB:.2f} MB = {STEP_BYTES / MB:.1f} MB of "
            "all-reduce messages per step. Time = bytes ÷ one-way link speed (a lower bound; a ring all-reduce sends 1.5× "
            f"the message per GPU at N = 4, plus latency): NVLink 4 ≈ 450 GB/s each way (900 GB/s total per H100) → "
            f"{T_NV * 1e3:.3f} ms; PCIe Gen5 x16 ≈ 64 GB/s each way (128 total) → {T_PCIE * 1e3:.2f} ms; a 400 Gb/s "
            f"InfiniBand NIC ≈ 50 GB/s → {T_IB * 1e3:.2f} ms (notes §10, vendor pages and DeepEP README, dated October "
            f"2026; Episode 12 used the same speeds). The step itself (step_time on 4 × H100, batch {BATCH}, KV ignored): "
            f"each GPU reads {PER_GPU / GB:.1f} GB at 3.35 TB/s = {STEP * 1e3:.2f} ms → NVLink adds {SHARE_NV:.1%}, PCIe "
            f"{SHARE_PCIE:.1%}. An HGX H100 server joins 8 GPUs by NVLink through NVSwitch (NVIDIA): tensor parallelism stays "
            "inside it (vLLM docs: without NVLink, prefer pipeline parallelism). Owl-70B on TP4 uses half the server; the "
            "other four GPUs can hold a second copy."))
        students = m.VGroup(*[m.Dot(radius=0.04, color=REQUEST) for _ in range(BATCH)])
        students.arrange_in_grid(rows=2, cols=BATCH // 2, buff=(0.04, 0.08))
        left(students, -6.3, 2.2)
        payload = code(f"{BATCH} students × {PER_TOKEN / MB:.1f} MB = {STEP_BYTES / MB:.0f} MB per step", 24, INK,
                       t2c={rf"{STEP_BYTES / MB:.0f} MB": MEMORY})
        left(payload, -2.1, 2.2)

        px0, px1, h_nv = -1.3, 3.5, 0.8
        rows = []
        for y, name, speed, bw, t in ((1.05, "NVLink", "450 GB/s each way", NVLINK, T_NV),
                                      (-0.15, "PCIe Gen5", "64 GB/s each way", PCIE, T_PCIE),
                                      (-1.15, "InfiniBand", "400 Gb/s ≈ 50 GB/s", IB, T_IB)):
            h = h_nv * bw / NVLINK
            label = m.VGroup(code(name, 22, INK, weight=m.BOLD), code(speed, 20, MUTED)).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.08)
            left(label, -6.3, y)
            tube = m.Rectangle(width=px1 - px0 + 0.1, height=h + 0.1).set_stroke(MUTED, 2.5).set_fill(CARD_HI, 1)
            tube.move_to([(px0 + px1) / 2, y, 0])
            fill = m.Rectangle(width=px1 - px0, height=h).set_stroke(width=0).set_fill(MEMORY, 1).move_to(tube).set_z_index(1)
            ms = code(f"{t * 1e3:.2f} ms", 24, MEMORY, weight=m.BOLD)
            left(ms, px1 + 0.35, y)
            rows.append((label, tube, fill, ms, t))
        both_ways = cite("both ways: NVLink 900 GB/s, PCIe Gen5 128 GB/s · NVIDIA, H100 spec")
        both_ways.move_to([0, 1.72, 0]).align_to([6.5, 0, 0], m.RIGHT)

        u, sx = 0.45, -4.6                                   # scene units per millisecond on the step bars
        tl = []
        for y, name, t, good in ((-2.45, "NVLink", T_NV, True), (-3.1, "PCIe", T_PCIE, False)):
            lab = left(code(name, 20, INK), -6.3, y)
            step = hbar(sx, y, STEP * 1e3 * u, 0.42, WEIGHTS, BLUE_FILL)
            sl = code(f"read the weights · {STEP * 1e3:.1f} ms", 20, INK).move_to(step).align_to(step, m.LEFT).shift(0.15 * m.RIGHT)
            sl.set_z_index(2)
            com = hbar(sx + STEP * 1e3 * u, y, t * 1e3 * u, 0.42, MEMORY, MEMORY)
            share = t / STEP
            pct = pill(f"+{share * 100:.1f}%" if good else f"+{share * 100:.0f}%", TOKEN if good else RED,
                       GREEN_FILL if good else RED_FILL, 20, mono=True)
            pct.move_to([0, y, 0]).align_to([sx + (STEP + T_PCIE) * 1e3 * u + 0.35, 0, 0], m.LEFT)
            tl.append((lab, step, sl, com, pct))
        tl_cap = left(code(f"one decode step, batch {BATCH}", 20, MUTED), sx, -2.05)

        wires_all = m.VGroup(students, payload, both_ways, tl_cap, *[m.VGroup(*r[:4]) for r in rows],
                             *[m.VGroup(*r) for r in tl])

        sw, shh = 11.0, 4.7
        server = card(sw, shh, EDGE, CARD, radius=0.2).move_to([0, -0.5, 0])
        srv_lab = left(code("one H100 server · 8 GPUs", 22, INK, weight=m.BOLD), -5.2, 1.45)
        g8 = m.VGroup(*[icon(GPU_ICON, 0.7).move_to([x, y, 0]) for y in (0.55, -1.55) for x in (-3.9, -1.3, 1.3, 3.9)])
        switch = m.RoundedRectangle(corner_radius=0.12, width=9.4, height=0.42).set_stroke(MEMORY, 2.5).set_fill(CYAN_FILL, 1)
        switch.move_to([0, -0.5, 0])
        sw_lab = code("NVLink switch · 900 GB/s per GPU", 20, MEMORY).move_to(switch).set_z_index(2)
        spokes = m.VGroup()
        for i, g in enumerate(g8):
            a = g.get_bottom() + [0, -0.06, 0] if i < 4 else g.get_top() + [0, 0.06, 0]
            b = [g.get_x(), switch.get_top()[1] if i < 4 else switch.get_bottom()[1], 0]
            spokes.add(m.Line(a, b).set_stroke(MEMORY, 4))
        tp_box = m.SurroundingRectangle(g8[:4], buff=0.22, corner_radius=0.12).set_stroke(WEIGHTS, 3)
        tp_lab = code("Owl-70B · tensor parallel × 4", 20, WEIGHTS).move_to([0, 1.45, 0]).align_to([5.2, 0, 0], m.RIGHT)
        sit(tp_lab, baseline(srv_lab))
        dp_box = m.DashedVMobject(m.SurroundingRectangle(g8[4:], buff=0.22, corner_radius=0.12).set_stroke(MUTED, 2.5), num_dashes=60)
        dp_lab = code("a second copy · more students", 20, MUTED).move_to([0, -3.2, 0])

        self.say(
            "[Now serve sixty-four students at once]: [every step moves a hundred and sixty-eight megabytes between the GPUs.]",
            m.AnimationGroup(self.new_header("04 · the wires", "Pipes of different widths"),
                             m.LaggedStart(*[m.FadeIn(s, scale=0.5) for s in students], lag_ratio=0.02)),
            m.FadeIn(payload, shift=0.1 * m.UP),
        )
        nv, pc, ib = rows
        self.say(
            "[Inside an H100 server, NVLink moves four hundred and fifty gigabytes per second each way]: [the bytes cross in zero point three seven milliseconds.]",
            m.AnimationGroup(m.FadeIn(nv[0], shift=0.1 * m.RIGHT), m.FadeIn(nv[1]), m.FadeIn(both_ways)),
            m.AnimationGroup(m.GrowFromEdge(nv[2], m.LEFT, run_time=0.7), m.FadeIn(nv[3])),
        )
        self.say(
            "[Over PCIe, it's sixty-four], [so the same bytes take two point six milliseconds.]",
            m.AnimationGroup(m.FadeIn(pc[0], shift=0.1 * m.RIGHT), m.FadeIn(pc[1])),
            m.AnimationGroup(m.GrowFromEdge(pc[2], m.LEFT, run_time=0.7 * T_PCIE / T_NV, rate_func=m.linear),
                             m.FadeIn(pc[3], run_time=0.7 * T_PCIE / T_NV, rate_func=lambda t: max(0.0, (t - 0.85) / 0.15))),
        )
        (n_lab, n_step, n_sl, n_com, n_pct), (p_lab, p_step, p_sl, p_com, p_pct) = tl
        self.say(
            "[Next to a ten-and-a-half-millisecond step], [that's under four percent on NVLink], [and a quarter on PCIe.]",
            m.AnimationGroup(m.FadeIn(tl_cap), m.FadeIn(n_lab), m.FadeIn(p_lab), m.GrowFromEdge(n_step, m.LEFT),
                             m.GrowFromEdge(p_step, m.LEFT), m.FadeIn(n_sl), m.FadeIn(p_sl)),
            m.AnimationGroup(m.GrowFromEdge(n_com, m.LEFT), m.FadeIn(n_pct, shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.GrowFromEdge(p_com, m.LEFT), m.FadeIn(p_pct, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[Between servers, a four-hundred-gigabit InfiniBand link is narrower still]: [fifty gigabytes per second.]",
            m.AnimationGroup(m.FadeIn(ib[0], shift=0.1 * m.RIGHT), m.FadeIn(ib[1])),
            m.AnimationGroup(m.GrowFromEdge(ib[2], m.LEFT, run_time=0.7 * T_IB / T_NV, rate_func=m.linear),
                             m.FadeIn(ib[3], run_time=0.7 * T_IB / T_NV, rate_func=lambda t: max(0.0, (t - 0.85) / 0.15))),
        )
        self.say(
            "[So tensor parallelism stays inside one server], [where NVLink joins up to eight GPUs.]",
            m.AnimationGroup(m.FadeOut(wires_all, shift=0.2 * m.DOWN), m.FadeIn(server), m.FadeIn(srv_lab)),
            m.Succession(m.AnimationGroup(m.LaggedStart(*[m.FadeIn(g, shift=0.1 * m.UP) for g in g8], lag_ratio=0.08),
                                          m.FadeIn(switch), m.FadeIn(sw_lab), m.Create(spokes)),
                         m.AnimationGroup(*[flash(s, MEMORY, 9, 0.6) for s in spokes])),
        )
        self.say(
            "[Owl's model takes half of it], [four GPUs], [and the other four can hold a second copy.]",
            m.AnimationGroup(m.Create(tp_box), m.FadeIn(tp_lab)),
            m.LaggedStart(*[m.Indicate(g, color=WEIGHTS, scale_factor=1.12) for g in g8[:4]], lag_ratio=0.15),
            m.AnimationGroup(m.Create(dp_box), m.FadeIn(dp_lab, shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 5 · pipeline parallel

    def pipeline(self):
        self.next_section("5 · Pipeline parallel", notes=(
            f"Pipeline parallelism: {LAYERS} layers ÷ {N} = {PER_STAGE} per GPU. Between stages only the hidden state "
            f"crosses: 1 × {D:,} × 2 bytes = 16 KiB per token, {PP_HOPS} hops = 48 KiB per token (vs {PER_TOKEN / MB:.1f} MB "
            "of all-reduces for tensor parallel), so stages can sit in different servers (two servers of two stages are "
            f"illustrative). A token passes the stages in sequence: each reads {PER_GPU / GB:.1f} GB in {STAGE * 1e3:.2f} ms, "
            f"so {N} × {STAGE * 1e3:.2f} = {PP_TOKEN * 1e3:.1f} ms per token = {PP_RATE:.1f} tok/s, the same as one GPU with "
            "all the bandwidth of one H100 — no latency gain (notes §10). Schedule (illustrative): one batch in flight → "
            "each GPU busy 1/4 of the time (bubbles: decode needs the previous token before the next starts); four batches "
            "in flight → every stage busy after warm-up, 4× the tokens per second, the same 42 ms per token."))
        y_c, h_c = 1.7, 1.1
        stages = m.VGroup()
        for k, x in enumerate(XS):
            box = card(2.6, h_c, EDGE, CARD, radius=0.16).move_to([x, y_c, 0])
            name = code(f"GPU {k + 1}", 22, INK, weight=m.BOLD).move_to([x, y_c + 0.3, 0])
            lay = code(f"layers {PER_STAGE * k + 1}–{PER_STAGE * (k + 1)}", 20, WEIGHTS).move_to([x, y_c - 0.05, 0])
            stages.add(m.VGroup(box, name, lay))
        hops = m.VGroup(*[arrow(stages[k][0].get_right(), stages[k + 1][0].get_left(), MEMORY, buff=0.05) for k in range(N - 1)])
        cap = code(f"{PP_HOPS} hops × 16 KiB = 48 KiB per token  ·  tensor parallel: {PER_TOKEN / MB:.1f} MB", 20, MUTED,
                   t2c={"48 KiB": MEMORY}).move_to([0, 0.5, 0])
        srv1 = code("server 1", 20, MUTED).move_to([(XS[0] + XS[1]) / 2, 2.55, 0])
        srv2 = code("server 2", 20, MUTED).move_to([(XS[2] + XS[3]) / 2, 2.55, 0])
        divider = m.DashedLine([0, 2.7, 0], [0, 1.12, 0], dash_length=0.1).set_stroke(MUTED, 2)
        ib_lab = code("InfiniBand", 20, MEMORY).move_to([0, 0.9, 0])

        lane = y_c - 0.33
        dot = m.Dot(radius=0.12, color=TOKEN).move_to([-6.5, lane, 0]).set_z_index(4)
        trip = []
        for k, x in enumerate(XS):
            trip.append(dot.animate(run_time=0.35).move_to([x, lane, 0]))
            trip.append(flash(stages[k][0], MEMORY, 6, 0.55, 0.7))
        trip.append(dot.animate(run_time=0.35).move_to([6.5, lane, 0]))
        math1 = code(f"{N} × {STAGE * 1e3:.1f} ms = {PP_TOKEN * 1e3:.0f} ms per token", 22, INK)
        math2 = code(f"→ {PP_RATE:.0f} tok/s", 22, MEMORY, weight=m.BOLD)
        math3 = code("like one big GPU", 22, MUTED)
        maths = row_of([math1, math2, math3], 0.3).move_to([0, 0.0, 0])

        gx0, pitch, cw, chh = -4.0, 0.74, 0.64, 0.4
        g_ys = [-0.65, -1.15, -1.65, -2.15]
        slots = 8

        def cell_at(s, t):
            return [gx0 + t * pitch + cw / 2, g_ys[s], 0]

        empties = m.VGroup(*[m.Rectangle(width=cw, height=chh).set_stroke(FAINT, 1.5).set_fill(CARD, 1).move_to(cell_at(s, t))
                             for s in range(N) for t in range(slots)])
        g_labs = m.VGroup(*[left(code(f"GPU {s + 1}", 20, MUTED), -5.15, g_ys[s]) for s in range(N)])
        t_axis = m.Arrow([gx0, -2.6, 0], [gx0 + slots * pitch + 0.2, -2.6, 0], buff=0, color=FAINT, stroke_width=3,
                         max_tip_length_to_length_ratio=0.04)
        t_lab = code("time", 20, MUTED).next_to(t_axis, m.RIGHT, buff=0.15)

        def block(s, t, b):
            box = m.Rectangle(width=cw, height=chh).set_stroke(REQUEST, 2).set_fill(PINK_FILL, 1).move_to(cell_at(s, t)).set_z_index(1)
            return m.VGroup(box, code(str(b), 20, INK).move_to(box).set_z_index(2))

        one = m.VGroup(*[block(s, t, 1) for s in range(N) for t in range(slots) if t % N == s])
        bubbles = m.VGroup(*[m.Rectangle(width=cw, height=chh).set_stroke(RED, 1.5).set_fill(RED_FILL, 1).move_to(cell_at(s, t))
                             for s in range(N) for t in range(slots) if t % N != s])
        fill_in = m.VGroup(*[block(s, t, (t - s) % N + 1) for t in range(slots) for s in range(N) if t >= s and t % N != s])
        sx = 3.2
        st1 = m.VGroup(text("1 batch in flight", 22, INK, weight=m.BOLD), code("each GPU busy 25%", 20, RED))
        st1.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.1)
        left(st1, sx, -0.9)
        bub_lab = left(code("bubbles", 22, RED, weight=m.BOLD), sx, -1.65)
        st4 = m.VGroup(text("4 batches in flight", 22, INK, weight=m.BOLD), code("every GPU busy", 20, TOKEN))
        st4.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.1)
        left(st4, sx, -0.9)
        thr = left(pill(f"{N}× throughput", TOKEN, GREEN_FILL, 20), sx, -1.7)
        lat = left(pill(f"same {PP_TOKEN * 1e3:.0f} ms per token", INK, CARD_HI, 20), sx, -2.35)

        self.say(
            "[Pipeline parallelism splits the model the other way]: [by layers, twenty on each GPU.]",
            m.AnimationGroup(self.new_header("05 · pipeline parallel", "Split the layers"),
                             m.LaggedStart(*[m.FadeIn(VGroup_(s[0], s[1])) for s in stages], lag_ratio=0.12)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(s[2], shift=0.1 * m.UP) for s in stages], lag_ratio=0.15),
                             m.LaggedStart(*[m.GrowArrow(h) for h in hops], lag_ratio=0.2)),
        )
        self.say(
            "[Between stages, only one small vector per token crosses]: [sixteen kibibytes, three times, instead of a hundred and sixty all-reduces.]",
            m.Succession(*[flash(m.Line(h.get_start(), h.get_end()), MEMORY, 10, 0.5) for h in hops]),
            m.FadeIn(cap, shift=0.1 * m.UP),
        )
        self.say(
            "[So the stages can sit in different servers], [joined by slower links, like InfiniBand.]",
            m.AnimationGroup(m.FadeIn(srv1), m.FadeIn(srv2), m.Create(divider)),
            m.AnimationGroup(m.FadeIn(ib_lab, shift=0.1 * m.UP), flash(m.Line(hops[1].get_start(), hops[1].get_end()), MEMORY, 10, 0.8)),
        )
        self.say(
            "[But each token still visits every stage in turn]: [ten and a half milliseconds each, forty-two in all], [no faster than one big GPU.]",
            m.Succession(m.FadeIn(dot, run_time=0.3), *trip),
            m.FadeIn(m.VGroup(math1, math2), shift=0.1 * m.UP),
            m.FadeIn(math3, shift=0.1 * m.LEFT),
        )
        self.say(
            "[With one batch in flight, three of the four GPUs sit idle at any moment]: [these gaps are called bubbles.]",
            m.AnimationGroup(m.FadeIn(empties), m.FadeIn(g_labs), m.GrowArrow(t_axis), m.FadeIn(t_lab),
                             m.LaggedStart(*[m.FadeIn(b, scale=0.8) for b in sorted(one, key=lambda b: b.get_x())], lag_ratio=0.12),
                             m.FadeIn(st1)),
            m.AnimationGroup(m.FadeIn(bubbles), m.FadeIn(bub_lab, shift=0.1 * m.UP)),
        )
        self.say(
            "[Keep four batches in flight], [and every stage stays busy]: [four times the throughput], [at the same latency.]",
            m.AnimationGroup(m.FadeOut(bub_lab), m.LaggedStart(*[m.FadeIn(b, scale=0.8) for b in fill_in], lag_ratio=0.05)),
            m.AnimationGroup(m.FadeOut(st1, shift=0.1 * m.UP), m.FadeIn(st4, shift=0.1 * m.UP)),
            m.FadeIn(thr, shift=0.1 * m.UP),
            m.FadeIn(lat, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 6 · experts and long prompts

    def more_splits(self):
        self.next_section("6 · Experts and long prompts", notes=(
            "Expert parallelism (Episode 20, notes §9–§10): a mixture of experts' experts are spread across GPUs; each token "
            "is dispatched to its top-k experts and the results combined, an all-to-all both ways (routes drawn here are "
            "illustrative: 4 tokens, 2 experts each, 8 experts on 4 GPUs). Context (sequence) parallelism (notes §10–§11): one "
            "very long prompt's tokens are split across GPUs, each prefills its slice while key-value blocks travel around "
            "a ring (Ring Attention, Liu et al. 2023), cutting time to first token for very long prompts. 16 prompt tokens "
            "drawn for a long prompt."))
        lc = card(6.3, 5.3, EDGE, CARD, radius=0.2).move_to([-3.35, -0.45, 0])
        rc = card(6.3, 5.3, EDGE, CARD, radius=0.2).move_to([3.35, -0.45, 0])
        lt = m.VGroup(text("Expert parallel", 26, INK, weight=m.BOLD), cite("Episode 20 · a mixture of experts"))
        lt.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.12)
        left(lt, -6.2, 1.62)
        rt = m.VGroup(text("Context parallel", 26, INK, weight=m.BOLD), cite("very long prompts · Liu et al., 2023"))
        rt.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.12)
        left(rt, 0.5, 1.62)

        ex = [-5.6, -4.25, -2.9, -1.55]
        g_boxes = m.VGroup()
        experts = []
        for k, x in enumerate(ex):
            box = card(1.2, 1.25, EDGE, CARD_HI, radius=0.12, stroke=2).move_to([x, -1.65, 0])
            lab = code(f"GPU {k + 1}", 20, MUTED).move_to([x, -2.05, 0])
            pair = [m.Square(0.36).set_stroke(WEIGHTS, 2).set_fill(BLUE_FILL, 1).move_to([x + dx, -1.5, 0]) for dx in (-0.24, 0.24)]
            experts += pair
            g_boxes.add(m.VGroup(box, lab, *pair))
        toks = m.VGroup(*[m.Dot(radius=0.12, color=TOKEN).move_to([x, 0.45, 0]) for x in ex])
        tok_lab = left(code("4 tokens · 2 experts each", 20, MUTED), -6.2, 0.88)
        routes = [(1, 6), (3, 4), (0, 7), (2, 5)]
        lines = m.VGroup(*[m.Line(toks[j].get_bottom(), experts[e].get_top()).set_stroke(MUTED, 2)
                           for j, pair in enumerate(routes) for e in pair])
        a2a = code("all-to-all: dispatch, then combine", 20, MEMORY).move_to([-3.35, -2.65, 0])

        prompt = m.VGroup(*[m.Square(0.26).set_stroke(REQUEST, 1.5).set_fill(PINK_FILL, 1) for _ in range(16)])
        prompt.arrange(m.RIGHT, buff=0.06).move_to([3.35, 0.45, 0]).set_z_index(3)
        p_lab = left(code("one very long prompt", 20, REQUEST), prompt.get_left()[0], 0.88)
        cx = [1.33, 2.68, 4.03, 5.38]
        c_boxes = m.VGroup()
        for k, x in enumerate(cx):
            box = card(1.2, 1.0, EDGE, CARD_HI, radius=0.12, stroke=2).move_to([x, -1.3, 0])
            c_boxes.add(m.VGroup(box, code(f"GPU {k + 1}", 20, MUTED).move_to([x, -1.6, 0])))
        slices = [m.VGroup(*prompt[4 * k: 4 * k + 4]) for k in range(N)]
        slice_to = [s.copy().scale(0.8).move_to([cx[k], -1.08, 0]) for k, s in enumerate(slices)]
        ctop, cbot = -0.8, -1.8
        c_paths = [m.ArcBetweenPoints([cx[k] + 0.25, ctop, 0], [cx[k + 1] - 0.25, ctop, 0], angle=-np.pi / 2) for k in range(N - 1)]
        c_paths.append(m.ArcBetweenPoints([cx[-1] + 0.45, cbot, 0], [cx[0] - 0.45, cbot, 0], angle=-np.pi / 5))
        c_ring = m.VGroup(*[p.copy().set_stroke(MUTED, 2.5).add_tip(tip_length=0.14) for p in c_paths[:-1]],
                          c_paths[-1].copy().set_stroke(MUTED, 2.5))   # the way back: no tip under GPU 1's label
        kv_lab = code("keys and values travel the ring", 20, KV).move_to([3.35, -2.75, 0])
        ttft = pill("time to first token ↓ · TTFT", TOKEN, GREEN_FILL, 20)
        left(ttft, 0.5, 0.88)

        self.say(
            "[Episode twenty's mixture of experts adds a third split]: [its experts spread across GPUs], [and each token travels to its experts and back], [all-to-all.]",
            m.AnimationGroup(self.new_header("06 · more splits", "Experts and long prompts"), m.FadeIn(lc), m.FadeIn(lt)),
            m.LaggedStart(*[m.FadeIn(g, shift=0.1 * m.UP) for g in g_boxes], lag_ratio=0.15),
            m.Succession(m.AnimationGroup(m.FadeIn(toks), m.FadeIn(tok_lab), m.Create(lines)),
                         m.AnimationGroup(*[flash(l, MEMORY, 6, 0.7) for l in lines]),
                         m.AnimationGroup(*[flash(m.Line(l.get_end(), l.get_start()), MEMORY, 6, 0.7) for l in lines])),
            m.FadeIn(a2a, shift=0.1 * m.UP),
        )
        self.say(
            "[And for a very long prompt, context parallelism splits the tokens themselves], [a slice per GPU.]",
            m.AnimationGroup(m.FadeIn(rc), m.FadeIn(rt), m.FadeIn(p_lab), m.LaggedStart(*[m.FadeIn(p, scale=0.6) for p in prompt], lag_ratio=0.04)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(b) for b in c_boxes], lag_ratio=0.1),
                             m.LaggedStart(*[introducing(m.Transform(s, t)) for s, t in zip(slices, slice_to)], lag_ratio=0.15)),
        )
        self.say(
            "[Each GPU works on its slice, passing keys and values around a ring], [so the first token comes sooner.]",
            m.Succession(m.Create(c_ring), *[m.AnimationGroup(*[flash(p, KV, 7, 0.45) for p in c_paths]) for _ in range(3)],
                         m.FadeIn(kv_lab)),
            m.AnimationGroup(m.FadeOut(p_lab), m.FadeIn(ttft, shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 7 · the price of splitting

    def price(self):
        self.next_section("7 · The price of splitting", notes=(
            f"Batch 1 on TP4: each GPU reads {PER_GPU / GB:.1f} GB per token in {STEP1 * 1e3:.2f} ms; the token's "
            f"{PER_TOKEN / MB:.1f} MB of all-reduce messages cross NVLink in {T1_NV * 1e6:.1f} µs — bandwidth is no problem, "
            f"the count is: {N_AR} synchronisations per token. For example at 10 µs each (order of magnitude: kipply 2022 "
            f"quotes ≈ 8 µs per message on V100 NVLink; notes §10 use ~10 µs, unverified): {SYNC_COST * 1e3:.1f} ms → "
            f"1 ÷ ({STEP1 * 1e3:.2f} + {SYNC_COST * 1e3:.1f}) ms = {CEIL4_SYNC:.1f} tok/s instead of {CEIL4:.1f}. Databricks "
            "(2023, MPT-7B on A100-40GB): 'At batch size 1, going from 2x to 4x only reduces token latency by ~12%'; at "
            "batch 16, 33%."))
        u, x0 = 0.55, -6.3
        cap = left(code("one token, batch 1, four H100s", 20, MUTED), x0, 2.2)
        step = hbar(x0, 1.55, STEP1 * 1e3 * u, 0.5, WEIGHTS, BLUE_FILL)
        s_lab = code(f"read {PER_GPU / GB:.1f} GB per GPU · {STEP1 * 1e3:.1f} ms", 20, INK).move_to(step).align_to(step, m.LEFT)
        s_lab.shift(0.15 * m.RIGHT).set_z_index(2)
        sx = x0 + STEP1 * 1e3 * u
        sw = SYNC_COST * 1e3 * u
        sync_box = m.Rectangle(width=sw, height=0.5).set_stroke(MEMORY, 2).set_fill(CYAN_FILL, 1).move_to([sx + sw / 2, 1.55, 0])
        sync_ticks = m.VGroup(*[m.Line([sx + (i + 0.5) * sw / 16, 1.37, 0], [sx + (i + 0.5) * sw / 16, 1.73, 0]).set_stroke(MEMORY, 2)
                                for i in range(16)]).set_z_index(1)
        sync_lab = code(f"{N_AR} syncs", 20, MEMORY).next_to(sync_box, m.RIGHT, buff=0.25)
        tiny = left(code(f"{PER_TOKEN / MB:.1f} MB over NVLink: {T1_NV * 1e6:.0f} µs", 20, MUTED), x0, 0.9)
        sync_math = code(f"{N_AR} × {SYNC * 1e6:.0f} µs = {SYNC_COST * 1e3:.1f} ms · for example", 20, MEMORY)
        left(sync_math, -1.2, 0.9)
        sit(sync_math, baseline(tiny))

        s95 = title_text(f"{CEIL4:.0f}", 64, MUTED)
        left(s95, x0, -0.2)
        s82 = stat(f"{CEIL4_SYNC:.0f}", "tok/s", "", MEMORY, 64)
        left(s82, x0 + 2.4, -0.2)
        s82.align_to(s95, m.DOWN)
        go = arrow(s95.get_right(), s82.get_left(), MUTED, buff=0.3)
        kipply = left(cite("order of magnitude: about 8 µs per message on V100 NVLink · kipply, 2022"), x0, -1.0)

        quote_a = text("From 2 to 4 GPUs at batch 1, token latency fell only", 24, INK)
        quote_b = title_text("~12%", 34, RED)
        quote = row_of([quote_a, quote_b], 0.25)
        src = cite("Databricks, 2023 · MPT-7B on A100-40GB · at batch 16: 33%")
        inner = m.VGroup(quote, src).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.18)
        qbox = card(inner.width + 0.8, inner.height + 0.6, EDGE, CARD, radius=0.16)
        qbox.move_to([0, -2.3, 0])
        inner.move_to(qbox)

        self.say(
            "[More GPUs don't cut latency for free.] [For one student, the bytes are tiny], [but four GPUs must sync a hundred and sixty times per token.]",
            m.AnimationGroup(self.new_header("07 · the price", "Latency isn't free"), m.FadeIn(cap), m.GrowFromEdge(step, m.LEFT), m.FadeIn(s_lab)),
            m.FadeIn(tiny, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(sync_box), m.LaggedStart(*[m.Create(t) for t in sync_ticks], lag_ratio=0.1), m.FadeIn(sync_lab)),
        )
        self.say(
            "[If each sync costs ten microseconds, for example, that adds one point six milliseconds]: [ninety-five tokens a second becomes about eighty-two.]",
            m.FadeIn(sync_math, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(s95), m.GrowArrow(go), m.FadeIn(s82, shift=0.1 * m.LEFT), m.FadeIn(kipply)),
        )
        self.say(
            "[In fact, Databricks measured it]: [going from two GPUs to four cut token latency by only about twelve percent], [at batch one.]",
            m.AnimationGroup(m.FadeIn(qbox), m.FadeIn(quote_a)),
            m.AnimationGroup(m.FadeIn(quote_b, scale=0.8), m.Circumscribe(quote_b, color=RED, buff=0.08)),
            m.FadeIn(src, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 8 · which split when

    def which(self):
        self.next_section("8 · Which split when", notes=(
            "Summary table. Tensor parallel: cuts every matrix, lowers per-token latency, needs NVLink-class links (inside one "
            "server). Pipeline: cuts the layers, adds room and (with micro-batches) throughput, not latency; tolerant of "
            "slower links (across servers). Expert: MoE experts, all-to-all (Episode 20). Context: one long prompt's tokens, "
            "a ring of KV blocks, faster long prefills. Copies (data parallel): when the model fits, more replicas serve more "
            "users. vLLM docs (read 2026-10-24): 'Set tensor_parallel_size to the number of GPUs per node and "
            "pipeline_parallel_size to the number of nodes.'"))
        xs = [-6.2, -3.75, -0.95, 2.4]
        heads = table_row(["split", "cuts", "buys", "wires"], xs, 2.05, [20] * 4, [MUTED] * 4, [True] * 4)
        data = [("tensor", "every matrix", "lower latency", "NVLink, in a server"),
                ("pipeline", "the layers", "room, throughput", "any link, across servers"),
                ("expert", "the experts", "room for big MoEs", "all-to-all"),
                ("context", "the prompt", "faster long prefill", "a ring of KV"),
                ("copies", "nothing", "more students", "none needed")]
        ys = [1.35, 0.65, -0.05, -0.75, -1.45]
        rows = [table_row(list(r), xs, y, [22, 22, 22, 20], [INK, INK, TOKEN, MEMORY], ["b", False, False, True])
                for r, y in zip(data, ys)]
        rules = m.VGroup(*[m.Line([-6.3, y + 0.35, 0], [6.4, y + 0.35, 0]).set_stroke(FAINT, 1.5) for y in ys])
        vllm = left(cite("vLLM docs: tensor parallel across a server's GPUs, pipeline parallel across servers"), -6.3, -2.2)
        mascot = owl(0.75).move_to([5.75, -2.75, 0])

        self.say(
            "[Putting it together]: [tensor parallel buys latency, inside one server], [and pipelines buy room and throughput, across servers.]",
            m.AnimationGroup(self.new_header("08 · which split", "Which split when"), m.FadeIn(heads)),
            m.AnimationGroup(m.Create(rules[0]), m.FadeIn(rows[0], shift=0.1 * m.UP)),
            m.AnimationGroup(m.Create(rules[1]), m.FadeIn(rows[1], shift=0.1 * m.UP), m.FadeIn(vllm)),
        )
        self.say(
            "[Expert and context parallelism handle huge mixtures and huge prompts], [and when a model fits on one GPU, plain copies serve more students.]",
            m.LaggedStart(m.AnimationGroup(m.Create(rules[2]), m.FadeIn(rows[2], shift=0.1 * m.UP)),
                          m.AnimationGroup(m.Create(rules[3]), m.FadeIn(rows[3], shift=0.1 * m.UP)), lag_ratio=0.4),
            m.AnimationGroup(m.Create(rules[4]), m.FadeIn(rows[4], shift=0.1 * m.UP), m.FadeIn(mascot, shift=0.2 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 9 · recap

    def recap(self):
        self.next_section("9 · Recap", notes=(
            f"Owl-70B ({W70 / GB:.1f} GB) on four H100s: {PER_GPU / GB:.1f} GB per GPU, ceiling {CEIL4:.0f} tok/s (×4), "
            f"{N_AR} all-reduces per token. Three words: slice (every matrix, tensor parallel), sum (the all-reduce, over "
            "NVLink), stage (layers across servers, pipeline parallel). Next: Episode 22, the serving stack."))
        top = code(f"Owl-70B · {W70 / GB:.1f} GB · four H100s", 22, MUTED).move_to([0, 2.15, 0])
        tiles = m.VGroup(
            stat(f"{PER_GPU / GB:.1f}", "GB", "weights per GPU", WEIGHTS, 64),
            stat(f"{CEIL4:.0f}", "tok/s", "decode ceiling, four times one", MEMORY, 64),
            stat(f"{N_AR}", "", "all-reduces per token", MEMORY, 64),
        )
        tiles.arrange(m.RIGHT, buff=1.1, aligned_edge=m.UP).move_to([0, 0.9, 0])

        halo = glow(WEIGHTS, radius=3.4).move_to([0, 0.55, 0])
        words = m.VGroup(title_text("Slice", 76, WEIGHTS), title_text("Sum", 76, MEMORY), title_text("Stage", 76, INK))
        words.arrange(m.RIGHT, buff=1.7).move_to([0, 0.7, 0])
        on_baseline(words)
        ya = words[0].get_y()
        a1 = arrow([words[0].get_right()[0], ya, 0], [words[1].get_left()[0], ya, 0], MUTED, buff=0.25)
        a2 = arrow([words[1].get_right()[0], ya, 0], [words[2].get_left()[0], ya, 0], MUTED, buff=0.25)
        subs = m.VGroup(text("every matrix", 22, MUTED), text("all-reduce over NVLink", 22, MUTED),
                        text("layers across servers", 22, MUTED))
        for sb, wd in zip(subs, words):
            sb.move_to([wd.get_x(), 0, 0])
            sit(sb, baseline(words[0]) - 0.55)
        nxt = pill("next: the serving stack · episode 22", MEMORY, CYAN_FILL, 22).move_to([0, -1.7, 0])
        mascot = owl(0.75).move_to([5.6, -2.75, 0])

        self.say(
            "[Owl's seventy-billion model on four H100s]: [a quarter of the weights each], [four times the bandwidth], [and a hundred and sixty all-reduces per token.]",
            m.AnimationGroup(self.new_header("09 · recap", "Slice, sum, stage"), m.FadeIn(top)),
            m.FadeIn(tiles[0], shift=0.1 * m.UP),
            m.FadeIn(tiles[1], shift=0.1 * m.UP),
            m.FadeIn(tiles[2], shift=0.1 * m.UP),
        )
        self.say(
            "[Three words to keep]: [slice], [sum], [stage.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(top, tiles), shift=0.2 * m.UP), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.2 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.GrowArrow(a1), m.FadeIn(words[1], shift=0.2 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.GrowArrow(a2), m.FadeIn(words[2], shift=0.2 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[Now Owl's big model runs across four GPUs.] [Next, the software that serves it to students.]",
            m.Circumscribe(m.VGroup(words, subs), color=WEIGHTS, buff=0.2),
            m.AnimationGroup(m.FadeIn(nxt, shift=0.1 * m.UP), m.FadeIn(mascot, shift=0.2 * m.UP)),
        )
        self.wait(0.6)
        self.clear()


def VGroup_(*parts):
    return m.VGroup(*parts)
