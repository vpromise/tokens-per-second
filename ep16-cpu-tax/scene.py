"""Tokens per Second · Episode 16 · The CPU Tax (Module 4 · Faster Tokens)

At batch 1, launching work can cost more than doing it. Episode 2 put a floor under every decode step (read all
16 GB of Owl-8B's weights once: 4.8 ms on an H100). A plain PyTorch decode loop gets nowhere near its floor:
PyTorch's "GPT, Fast" baseline ran a 7B Llama at 25.5 tok/s on an A100, about a sixth of its ~149 tok/s ceiling.
The missing time is the CPU tax: a decode step is hundreds of tiny kernels (for example 32 × 18 + 4 = 580 for
Owl, about 8 µs of GPU work each), and the CPU must launch every one of them from Python. Fixes, animated:
fusion (TinyLM's RMS-norm line = 5 ops → 1 kernel; a layer's 18 kernels → 9), CUDA graphs (the launch queue
recorded once, then replayed with one call: the gaps close), torch.compile (both, in one line: gpt-fast 25.5 →
107 tok/s), and Flash-Decoding (split-KV: one long chat's attention spread over the whole GPU).

Sections: the missing time · hundreds of tiny kernels · the CPU tells, the GPU does · measured (gpt-fast's
trace; Owl-mini on this Mac) · fusion · CUDA graphs · torch.compile · Flash-Decoding · recap (fuse, replay,
split). Episodes 2 (the floor), 5/9 (the 128K-token cache), 15 (online softmax) are linked by number.

Sources (facts dated October 2026):
- owl_reference.py: OWL (32 layers, 32 heads), weight_bytes, kv_bytes_per_token, step_time (the batch-1 floor),
  HARDWARE (H100 SXM 3.35 TB/s; M4 Max 546 GB/s); TinyLM.forward's RMS-norm line (shown verbatim, asserted).
- research/notes.md §7 + the PyTorch blog "Accelerating Generative AI with PyTorch II: GPT, Fast" (2023, read
  2026-10-24): LLaMA-7B, batch 1, A100-80GB power-limited to 330 W; eager baseline 25.5 tok/s; "torch.compile +
  static KV cache" 107.0 tok/s ("more than 4x"); "72% MBU" at 107 tok/s; "it's heavily CPU overhead bound! ...
  our CPU is not able to tell the GPU what to do fast enough"; torch.compile(decode_one_token,
  mode="reduce-overhead", fullgraph=True); reduce-overhead uses CUDA graphs; compile generates fused kernels.
- PyTorch blog "Accelerating PyTorch with CUDA Graphs" (2021, notes §7): "CUDA work issued to a capturing stream
  doesn't actually run on the GPU. Instead, the work is recorded in a graph"; "Each replay runs the same kernels
  with the same arguments ... the same memory addresses"; "a graph replay skips all layers of argument setup and
  kernel dispatch, including Python, C++, and CUDA driver overheads"; graph-safe usually means static shapes.
- vLLM docs (notes §7): CUDA graphs captured for a set of batch sizes, --max-cudagraph-capture-size "capped at
  512 by default"; --enforce-eager disables them.
- Flash-Decoding (Dao, Haziza, Massa, Sizov; Oct 2023, notes §7, blog read 2026-10-24): FlashAttention
  parallelizes over query blocks and batch, so at decode (one query) a small batch uses a small part of the GPU;
  "First, we split the keys/values in smaller chunks"; each split writes its partial output and the log-sum-exp;
  a final reduction rescales and sums; "up to 8x" faster generation for very long sequences (CodeLlama-34B).
- NVIDIA H100 Tensor Core GPU Architecture whitepaper (2022): H100 SXM5 has 132 SMs.
- Measured on this Mac (research/measurements.md §1, mlx-lm 0.32.2, 2026-10-24; loaded with measured()):
  Owl-mini (Llama-3.2-1B 4-bit) decodes at 488.8 tok/s = 2.05 ms per token; its 0.70 GB read per step takes
  1.29 ms at 546 GB/s (ceiling 777 tok/s, MBU 63%); "the rest is fixed per-step cost (presumably kernel
  launches..., sampling and Python overhead; not profiled)".

Derived (computed below and asserted): the gpt-fast ceiling 107 ÷ 0.72 ≈ 149 tok/s (6.7 ms per token); the
baseline at 17% of it; 39.2 ms → 9.3 ms per token; ×4.2. Owl: 580 launches per step, 4.8 ms ÷ 580 ≈ 8 µs per
kernel. Flash-Decoding on an H100: 32 heads → 32 of 132 SMs busy (≈ 76% idle); 4 splits per head → 128 busy.

Illustrative (said "for example" / "say", or labelled): the 18-op layer and the 4 extra ops (an unfused,
one-kernel-per-op count; eager PyTorch launches more), the 9-kernel fused layer, the schematic timelines (launch
and kernel widths, no time scale), the 292-cell launch queue, the 4-way split, the batch sizes on the graph cards.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import inspect as _inspect
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from owl_reference import *  # noqa: E402,F403
import owl_reference  # noqa: E402

# ---------------------------------------------------------------------------- the numbers

LAYERS, HEADS = OWL["layers"], OWL["heads"]        # 32, 32
W_BYTES = weight_bytes()                           # 16.06 GB of BF16 weights
H_BW = GPU["bandwidth"]                            # 3.35 TB/s
T_FLOOR = step_time()                              # Episode 2's floor: 4.79 ms per decode step

# one layer of Owl, op by op (illustrative: one kernel per op, nothing fused)
LAYER = [("norm", MUTED), ("q", WEIGHTS), ("k", WEIGHTS), ("v", WEIGHTS), ("rope q", MUTED), ("rope k", MUTED),
         ("cache k", KV), ("cache v", KV), ("attention", KV),
         ("out", WEIGHTS), ("add", MUTED), ("norm", MUTED), ("gate", WEIGHTS), ("up", WEIGHTS), ("silu", MUTED),
         ("mul", MUTED), ("down", WEIGHTS), ("add", MUTED)]
EXTRA = [("embed", MUTED), ("norm", MUTED), ("lm head", WEIGHTS), ("sample", MUTED)]   # once per step
PER_LAYER = len(LAYER)                             # 18
N_LAUNCH = LAYERS * PER_LAYER + len(EXTRA)         # 580 launches per token
AVG_US = T_FLOOR / N_LAUNCH * 1e6                  # ≈ 8 µs of GPU work per kernel
# the same layer fused (illustrative): which unfused ops each fused kernel absorbs
FUSED = [("norm", MUTED, [0]), ("qkv", WEIGHTS, [1, 2, 3]), ("rope+kv", KV, [4, 5, 6, 7]), ("attention", KV, [8]),
         ("out+add", WEIGHTS, [9, 10]), ("norm", MUTED, [11]), ("gate+up", WEIGHTS, [12, 13]),
         ("silu*mul", MUTED, [14, 15]), ("down+add", WEIGHTS, [16, 17])]
N_FUSED = LAYERS * len(FUSED) + len(EXTRA)         # 292

# PyTorch "GPT, Fast" (notes §7): LLaMA-7B, batch 1, A100-80GB at 330 W
GF_EAGER, GF_COMPILED, GF_MBU = 25.5, 107.0, 0.72  # tok/s, tok/s, the blog's MBU at 107 tok/s
GF_CEIL = GF_COMPILED / GF_MBU                     # ≈ 149 tok/s at 100% of the bandwidth
GF_EAGER_FRAC = GF_EAGER / GF_CEIL                 # ≈ 17% of the ceiling
GF_SPEEDUP = GF_COMPILED / GF_EAGER                # ≈ 4.2×
MS_EAGER, MS_COMP, MS_FLOOR = 1e3 / GF_EAGER, 1e3 / GF_COMPILED, 1e3 / GF_CEIL   # 39.2, 9.3, 6.7 ms

# measured on this Mac: Owl-mini (Llama-3.2-1B, 4-bit), batch 1
MINI = measured("decode", "llama-3.2-1b", "4bit")
MINI_TOK, MINI_MS = MINI["tok_s_median"], MINI["ms_per_token"]              # 488.8 tok/s, 2.05 ms
MINI_READ = MINI["bytes_read_per_step"]                                      # 0.70 GB per step
MINI_FLOOR_MS = MINI_READ / MAC["bandwidth"] * 1e3                           # 1.29 ms at 546 GB/s
MINI_CEIL = MINI["ceiling_tok_s_bytes_read"]                                 # 777 tok/s
MINI_FRAC = MINI_TOK / MINI_CEIL                                             # 63%
MINI_GAP = MINI_MS - MINI_FLOOR_MS                                           # 0.76 ms not reading bytes

# Flash-Decoding on an H100
SMS = 132                                          # H100 SXM5 (NVIDIA H100 whitepaper, 2022)
CTX = 131_072                                      # a 128K-token chat (Episodes 5 and 9)
KV_LONG = CTX * kv_bytes_per_token()               # 16 GiB
SPLITS = SMS // HEADS                              # 4 chunks per head (illustrative choice that fills the chip)
BLOCKS = HEADS * SPLITS                            # 128 blocks of work
BUSY = HEADS / SMS                                 # ≈ 24% of the SMs busy without splitting

NORM_LINE = 'h = x / np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-6)'
COMPILE_SRC = ('torch.compile(\n'
               '    decode_one_token,\n'
               '    mode="reduce-overhead",\n'
               '    fullgraph=True)')

assert LAYERS == 32 and HEADS == 32                                         # "thirty-two layers", "thirty-two heads"
assert round(T_FLOOR * 1e3, 1) == 4.8                                       # "four point eight milliseconds"
assert round(W_BYTES / 1e9) == 16                                           # "sixteen gigabytes"
assert PER_LAYER == 18 and N_LAUNCH == 580 and 550 < N_LAUNCH < 600         # "nearly six hundred launches"
assert round(AVG_US) == 8                                                   # "just eight microseconds"
assert len(FUSED) == 9 and sorted(i for *_, ix in FUSED for i in ix) == list(range(PER_LAYER))   # "eighteen ... become nine"
assert N_FUSED == 292
assert round(GF_CEIL) == 149 and 140 < GF_CEIL < 160                        # "about a hundred and fifty"
assert 1 - GF_EAGER_FRAC > 0.8 and round(GF_EAGER_FRAC * 100) == 17        # "more than eighty percent of the time"
assert GF_SPEEDUP > 4 and round(GF_SPEEDUP, 1) == 4.2                       # "over four times faster"
assert round(MS_EAGER) == 39 and MS_FLOOR < 7 and 9 < MS_COMP < 9.5         # "thirty-nine", "under seven", "just over nine"
assert round(GF_MBU * 100) == 72                                            # "seventy-two percent"
assert round(MINI_MS, 2) == 2.05 and round(MINI_MS) == 2                    # "two milliseconds"
assert round(MINI_FLOOR_MS, 1) == 1.3                                       # "one point three"
assert round(MINI_FRAC * 100) == 63                                         # "sixty-three percent"
assert round(MINI_READ / 1e9, 2) == 0.70 and round(MINI_CEIL) == 777
assert KV_LONG == 16 * 2 ** 30 and abs(KV_LONG / W_BYTES - 1) < 0.1         # "as big as its weights"
assert SPLITS == 4 and BLOCKS == 128                                        # "four per head", "a hundred and twenty-eight"
assert round((1 - BUSY) * 4) == 3                                           # "three quarters of the chip waits"
assert NORM_LINE in _inspect.getsource(owl_reference.TinyLM.forward)        # the line shown is the real one

# ---------------------------------------------------------------------------- look

PREFIX = "ep16"
GPU_CYAN = recolored("gpu", MEMORY, CYAN_FILL, PREFIX)    # decode kernels: memory traffic
GPU_INK = recolored("gpu", INK, CARD_HI, PREFIX)          # a GPU as hardware (neutral)
MAC_ICON = recolored("mac", MUTED, CARD, PREFIX)
CPU_ICON = "ep16-cpu"                                     # drawn for this episode (MUTED, the messenger)

LAUNCH, LAUNCH_FILL = MUTED, "#22343B"                    # a CPU launch: structure, not compute


def cite(s, size=20):
    return text(s, size, MUTED)


def hbar(x0, y, width, height, color, fill, stroke=2):
    """A horizontal bar whose left edge is at x0."""
    w = max(width, 0.04)
    return m.Rectangle(width=w, height=height).set_stroke(color, stroke).set_fill(fill, 1).move_to([x0 + w / 2, y, 0])


def span(x0, x1, y, height, color, fill, stroke=1.5):
    """A block on a timeline lane from x0 to x1."""
    return m.Rectangle(width=x1 - x0, height=height).set_stroke(color, stroke).set_fill(fill, 1).move_to([(x0 + x1) / 2, y, 0])


def mono_in(box, s, size=20, color=INK):
    """Mono text centred in a box on a cap-height baseline (a leading M sets the baseline, then is dropped), so
    chips with and without descenders ("q", "up") line up."""
    full = code("M" + s, size, color)
    rest = m.VGroup(*full[1:])
    full.shift([box.get_x() - rest.get_x(), box.get_y() - full[0].height / 2 - full[0].get_bottom()[1], 0])
    return rest


def kchip(label, color, size=20, height=0.5, width=None):
    """One kernel as a chip: outline in its colour (BLUE reads weights, VIOLET the KV cache, MUTED small ops)."""
    w = width or code(label, size).width + 0.34
    box = card(w, height, color, FILLS.get(color, CARD) if color != MUTED else CARD_HI, radius=0.1, stroke=2)
    return m.VGroup(box, mono_in(box, label, size))


def thin_arrow(start, end, color=MUTED, buff=0.06, stroke=3.5):
    return m.Arrow(start, end, buff=buff, color=color, stroke_width=stroke, max_tip_length_to_length_ratio=0.16)


def cell(color, size):
    fill = {MUTED: FAINT}.get(color, color)
    return m.Square(size).set_stroke(width=0).set_fill(fill, 0.95)


def lane_label(s, x, y):
    return code(s, 20, MUTED).move_to([0, y, 0]).align_to([x, 0, 0], m.RIGHT)


def quote_card(lines, width, size=24):
    words = m.VGroup(*[text(s, size, INK) for s in lines]).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.14)
    box = card(width, words.height + 0.55, EDGE, CARD, radius=0.14, stroke=2)
    stripe = m.RoundedRectangle(corner_radius=0.05, width=0.09, height=box.height - 0.2, stroke_width=0,
                                fill_color=MUTED, fill_opacity=1)
    stripe.move_to(box).align_to(box, m.LEFT).shift(0.12 * m.RIGHT)
    words.move_to(box).align_to(box, m.LEFT).shift(0.4 * m.RIGHT)
    return m.VGroup(box, stripe, words)


def legend_item(color, label):
    sq = cell(color, 0.22)
    words = text(label, 20, MUTED)
    return m.VGroup(sq, words).arrange(m.RIGHT, buff=0.14)


# ---------------------------------------------------------------------------- the episode


class Ep16TheCPUTax(Episode):
    number = 16

    def lesson(self):
        self.hook()
        self.kernels()
        self.launches()
        self.measured_tax()
        self.fusion()
        self.graphs()
        self.compiler()
        self.split_kv()
        self.recap()

    # ------------------------------------------------------------------ 1 · the missing time

    def hook(self):
        self.next_section("1 · The missing time", notes=(
            f"Episode 2's floor: Owl-8B reads {W_BYTES / 1e9:.2f} GB per decode step; at the H100's {H_BW / 1e12:.2f} TB/s "
            f"that is {T_FLOOR * 1e3:.2f} ms (owl_reference.step_time). PyTorch 'GPT, Fast' (2023, notes §7): a plain "
            f"PyTorch decode loop, LLaMA-7B, batch 1, A100-80GB at 330 W: {GF_EAGER} tok/s. The blog reports {GF_MBU:.0%} MBU "
            f"at {GF_COMPILED:.0f} tok/s, so 100% of the bandwidth is {GF_CEIL:.1f} tok/s [DERIVED]; the baseline reaches "
            f"{GF_EAGER_FRAC:.1%} of it. The blog's diagnosis: 'heavily CPU overhead bound'."))
        mascot = owl(0.6).move_to([-6.0, 1.7, 0])
        who = m.VGroup(text("Owl-8B on an H100", 24, INK, weight=m.BOLD),
                       code(f"{W_BYTES / 1e9:.0f} GB of weights", 20, WEIGHTS))
        who.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.1).next_to(mascot, m.RIGHT, buff=0.3)
        x0 = -0.9
        floor_bar = hbar(x0, 1.55, 3.0, 0.52, MEMORY, CYAN_FILL)
        floor_val = code(f"{T_FLOOR * 1e3:.1f} ms", 22, MEMORY, weight=m.BOLD).move_to(floor_bar)
        floor_cap = text("one decode step: read every weight once", 22, MUTED)
        floor_cap.next_to(floor_bar, m.UP, buff=0.16).align_to(floor_bar, m.LEFT)
        floor_eq = code(f"{W_BYTES / 1e9:.2f} GB ÷ {H_BW / 1e12:.2f} TB/s", 20, INK,
                        t2c={r"[\d.]+ GB(?!/)": WEIGHTS, r"[\d.]+ TB/s": MEMORY}).next_to(floor_bar, m.RIGHT, buff=0.3)
        floor_src = cite("Episode 2").next_to(floor_eq, m.DOWN, buff=0.12).align_to(floor_eq, m.LEFT)
        band1 = m.VGroup(mascot, who, floor_bar, floor_val, floor_cap, floor_eq, floor_src)

        gf_icon = icon(GPU_INK, 0.75)
        gf_words = m.VGroup(text("a plain PyTorch decode loop", 26, INK, weight=m.BOLD),
                            code("Llama-7B · A100-80GB · batch 1", 20, MUTED),
                            cite("PyTorch, “GPT, Fast”, 2023"))
        gf_words.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.1)
        gf_body = m.VGroup(gf_icon, gf_words).arrange(m.RIGHT, buff=0.35)
        gf_box = card(gf_body.width + 0.7, gf_body.height + 0.5, EDGE, CARD)
        gf_body.move_to(gf_box)
        gf_card = m.VGroup(gf_box, gf_body).move_to([-3.05, -0.25, 0])
        eager = stat(f"{GF_EAGER:.1f}", "tok/s", "", RED, 84).move_to([3.0, -0.2, 0])

        mx0, mw, my = -5.4, 10.8, -2.3
        track = m.Rectangle(width=mw, height=0.42).set_stroke(MEMORY, 2).set_fill(CARD, 1).move_to([mx0 + mw / 2, my, 0])
        got = hbar(mx0, my, mw * GF_EAGER_FRAC, 0.42, RED, RED_FILL)
        got_lab = code(f"{GF_EAGER:.1f} · {GF_EAGER_FRAC * 100:.0f}%", 22, RED, weight=m.BOLD)
        got_lab.next_to(track, m.UP, buff=0.14).align_to(track, m.LEFT)
        ceil_lab = code(f"ceiling ≈ {GF_CEIL:.0f} tok/s", 22, MEMORY, weight=m.BOLD).next_to(track, m.UP, buff=0.14)
        ceil_lab.align_to(track, m.RIGHT)
        ceil_src = cite(f"the blog: {GF_COMPILED:.0f} tok/s used {GF_MBU * 100:.0f}% of the bandwidth, so 100% ≈ {GF_CEIL:.0f}")
        ceil_src.next_to(track, m.DOWN, buff=0.16).align_to(track, m.RIGHT)
        missing = text(f"{(1 - GF_EAGER_FRAC) * 100:.0f}% of the time: where did it go?", 22, INK, weight=m.BOLD)
        missing.move_to([(got.get_right()[0] + track.get_right()[0]) / 2, my, 0]).set_z_index(2)

        p_math = pill("not math", COMPUTE, AMBER_FILL, 24)
        p_mem = pill("not memory", MEMORY, CYAN_FILL, 24)
        cpu = icon(CPU_ICON, 0.75)
        p_cpu = pill("the CPU, telling the GPU what to do", RED, RED_FILL, 24)
        pills = m.VGroup(p_math, p_mem, m.VGroup(cpu, p_cpu).arrange(m.RIGHT, buff=0.2)).arrange(m.RIGHT, buff=0.4)
        pills.move_to([0, 1.6, 0])
        for p in (p_math, p_mem):
            p.set_opacity(0.45)

        self.say(
            "[In episode two, every decode step had a floor]: [read all the weights once.] "
            "[For Owl on an H100, that's four point eight milliseconds.]",
            m.AnimationGroup(self.new_header("01 · the missing time", "Where did the time go?"),
                             m.FadeIn(mascot, shift=0.2 * m.UP), m.FadeIn(who)),
            m.AnimationGroup(m.FadeIn(floor_cap), m.GrowFromEdge(floor_bar, m.LEFT)),
            m.AnimationGroup(m.FadeIn(floor_val), m.FadeIn(floor_eq, shift=0.1 * m.LEFT), m.FadeIn(floor_src)),
        )
        self.say(
            "[PyTorch's own team timed a plain decode loop]: [a seven-billion-parameter Llama on an A100], "
            "[twenty-five and a half tokens a second.]",
            m.FadeIn(gf_box),
            m.FadeIn(gf_body, shift=0.1 * m.UP),
            m.FadeIn(eager, scale=0.9),
        )
        self.say(
            "[Its ceiling is about a hundred and fifty.] [So more than eighty percent of the time went somewhere else.]",
            m.AnimationGroup(m.FadeIn(track), m.FadeIn(ceil_lab), m.FadeIn(ceil_src)),
            m.AnimationGroup(m.GrowFromEdge(got, m.LEFT), m.FadeIn(got_lab), m.FadeIn(missing, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[Not to math], [not to memory], [but to the CPU, telling the GPU what to do.]",
            m.AnimationGroup(m.FadeOut(band1, shift=0.2 * m.UP), m.FadeIn(p_math, shift=0.15 * m.UP)),
            m.FadeIn(p_mem, shift=0.15 * m.UP),
            m.AnimationGroup(m.FadeIn(pills[2], shift=0.15 * m.UP), m.Circumscribe(eager[0], color=RED, buff=0.1)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 2 · hundreds of tiny kernels

    def kernels(self):
        self.next_section("2 · Hundreds of tiny kernels", notes=(
            f"One Llama layer written op by op: {PER_LAYER} ops (norm, q, k, v, RoPE on q and k, two cache writes, attention, "
            "output projection, residual add, norm, gate, up, SiLU, multiply, down, residual add) — illustrative: one kernel "
            "per op, nothing fused; eager PyTorch launches more (an RMS norm alone is several kernels). "
            f"{LAYERS} layers × {PER_LAYER} + {len(EXTRA)} (embedding, final norm, LM head, sampling) = {N_LAUNCH} launches per "
            f"token. Episode 2's floor {T_FLOOR * 1e3:.2f} ms ÷ {N_LAUNCH} = {AVG_US:.1f} µs of GPU work per kernel on average "
            "(the weight matmuls take most; norms and adds a few µs). Notes §7: 'about 5 ms for 8B on an H100, made of "
            "hundreds of kernels'. BLUE = reads weights, VIOLET = KV cache, grey = small element-wise ops."))
        chips = [kchip(name, color) for name, color in LAYER]
        row1 = m.VGroup(*chips[:9]).arrange(m.RIGHT, buff=0.12).move_to([0, 1.5, 0])
        row2 = m.VGroup(*chips[9:]).arrange(m.RIGHT, buff=0.12).move_to([0, 0.78, 0])
        caption = text("one layer of Owl, op by op · for example", 22, MUTED).move_to([0, 2.25, 0])
        legend = m.VGroup(legend_item(WEIGHTS, "reads weights"), legend_item(KV, "KV cache"),
                          legend_item(MUTED, "small element-wise op")).arrange(m.RIGHT, buff=0.6).move_to([0, -0.05, 0])
        per_layer = text(f"{PER_LAYER} kernels per layer, each launched on its own", 26, INK, weight=m.BOLD).move_to([0, -0.85, 0])

        size, gap = 0.105, 0.045
        grid = m.VGroup(*[
            m.VGroup(*[cell(color, size) for _, color in LAYER]).arrange(m.RIGHT, buff=gap) for _ in range(LAYERS)
        ]).arrange(m.DOWN, buff=gap).move_to([4.2, -0.45, 0])
        extras = m.VGroup(*[cell(color, size) for _, color in EXTRA]).arrange(m.RIGHT, buff=gap)
        extras.next_to(grid, m.DOWN, buff=0.16).align_to(grid, m.LEFT)
        plus = code(f"+{len(EXTRA)}", 20, MUTED).next_to(extras, m.RIGHT, buff=0.15)
        l1 = code("layer 1", 20, MUTED).next_to(grid[0], m.LEFT, buff=0.2)
        l32 = code(f"layer {LAYERS}", 20, MUTED).next_to(grid[-1], m.LEFT, buff=0.2)

        cx = -3.0
        big = stat(f"{N_LAUNCH}", "launches", "for one token · for example", INK, 104)
        big.move_to([cx, 1.15, 0])
        recipe = code(f"{LAYERS} layers × {PER_LAYER} + {len(EXTRA)} = {N_LAUNCH}", 26, INK).move_to([cx, -0.3, 0]).align_to(big, m.LEFT)
        extra_note = cite("+4: embedding, final norm, LM head, sampling").next_to(recipe, m.DOWN, buff=0.16).align_to(big, m.LEFT)
        work = code(f"{T_FLOOR * 1e3:.1f} ms ÷ {N_LAUNCH} ≈ {AVG_US:.0f} µs", 30, MEMORY, weight=m.BOLD)
        work.move_to([cx, -1.75, 0]).align_to(big, m.LEFT)
        work_lab = text("of GPU work per kernel, on average", 22, MUTED).next_to(work, m.DOWN, buff=0.16).align_to(big, m.LEFT)
        work_src = cite(f"Episode 2: the whole step is {T_FLOOR * 1e3:.1f} ms on an H100").next_to(work_lab, m.DOWN, buff=0.12)
        work_src.align_to(big, m.LEFT)

        flashes = [m.ShowPassingFlash(c[0].copy().set_stroke(MEMORY, 5), time_width=0.6, run_time=0.5) for c in chips]
        into_grid = [m.ReplacementTransform(c[0], grid[0][i]) for i, c in enumerate(chips)]
        self.say(
            "[Look inside one decode step.] [One layer of Owl is a string of small operations]: "
            "[norms, projections, attention, the feed-forward.]",
            m.AnimationGroup(self.new_header("02 · one step", "Hundreds of tiny kernels"), m.FadeIn(caption)),
            m.LaggedStart(*[m.FadeIn(c, shift=0.12 * m.DOWN) for c in chips], lag_ratio=0.12),
            m.LaggedStart(*[m.FadeIn(it, shift=0.1 * m.UP) for it in legend], lag_ratio=0.3),
        )
        self.say(
            "[On the GPU, each one is its own kernel], [and each kernel must be launched.]",
            m.LaggedStart(*flashes, lag_ratio=0.12),
            m.FadeIn(per_layer, shift=0.1 * m.UP),
        )
        self.say(
            "[Owl has thirty-two layers], [so one token takes, for example, nearly six hundred launches.]",
            m.AnimationGroup(
                m.FadeOut(m.VGroup(caption, legend, per_layer)), *into_grid, *[m.FadeOut(c[1]) for c in chips],
                m.LaggedStart(*[m.FadeIn(r) for r in grid[1:]], lag_ratio=0.12), m.FadeIn(l1), m.FadeIn(l32), run_time=2.6),
            m.AnimationGroup(m.FadeIn(extras), m.FadeIn(plus), m.FadeIn(big, scale=0.9), m.FadeIn(recipe, shift=0.1 * m.UP),
                             m.FadeIn(extra_note)),
        )
        self.remove(*chips)
        self.say(
            "[The step's real work is four point eight milliseconds], [so the average kernel runs for just eight microseconds.]",
            m.AnimationGroup(m.FadeIn(work_src), m.Circumscribe(grid, color=MEMORY, buff=0.08)),
            m.AnimationGroup(m.FadeIn(work, shift=0.1 * m.UP), m.FadeIn(work_lab)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 3 · the CPU tells, the GPU does

    def launches(self):
        self.next_section("3 · The CPU tells, the GPU does", notes=(
            "A GPU kernel runs only when the CPU launches it; in eager PyTorch every op goes through Python, the framework "
            "(argument setup, dispatch) and the CUDA driver (PyTorch CUDA Graphs blog, 2021: 'argument setup and kernel "
            "dispatch, including Python, C++, and CUDA driver overheads'). PyTorch executes asynchronously: while kernels "
            "run longer than launches, the CPU runs ahead and its overhead is hidden (He, 'Making Deep Learning Go Brrrr', "
            "2022); when kernels are shorter, the GPU idles between them (PyTorch 2021: 'GPUs go idle between kernel "
            f"calls'). Owl's kernels average {AVG_US:.0f} µs (section 2). The timelines are schematic: launch and kernel "
            "widths are illustrative, no time scale. The code is an excerpt of owl_reference's TinyLM.forward (reshape "
            "omitted)."))
        cpu_box = card(5.2, 1.55, MUTED, CARD).move_to([-3.9, 1.6, 0])
        cpu_icon = icon(CPU_ICON, 0.8)
        cpu_name = code("CPU", 20, MUTED, weight=m.BOLD)
        lines = m.VGroup(*[code(f'{n} = h @ b["{n}"]', 20, INK, t2c={r"\"\w\"": GREEN}) for n in "qkv"])
        lines.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.14)
        cpu_side = m.VGroup(cpu_icon, cpu_name).arrange(m.DOWN, buff=0.12)
        cpu_body = m.VGroup(cpu_side, lines).arrange(m.RIGHT, buff=0.5).move_to(cpu_box)
        cpu_card = m.VGroup(cpu_box, cpu_body)
        hi = m.RoundedRectangle(corner_radius=0.06, width=lines.width + 0.3, height=lines[0].height + 0.16,
                                stroke_width=0, fill_color=MUTED, fill_opacity=0.18).move_to(lines[0]).set_z_index(1)

        gpu_box = card(4.4, 1.55, MEMORY, CARD).move_to([4.2, 1.6, 0])
        gpu_icon = icon(GPU_CYAN, 0.8)
        gpu_name = code("GPU", 20, MEMORY, weight=m.BOLD)
        gpu_side = m.VGroup(gpu_icon, gpu_name).arrange(m.DOWN, buff=0.12)
        slots = m.VGroup(*[kchip(n, WEIGHTS, width=0.62) for n in "qkv"]).arrange(m.RIGHT, buff=0.12)
        gpu_body = m.VGroup(gpu_side, slots).arrange(m.RIGHT, buff=0.45).move_to(gpu_box)
        gpu_card = m.VGroup(gpu_box, gpu_side)
        go = arrow(cpu_box.get_right(), gpu_box.get_left(), MUTED, buff=0.12)
        go_lab = code("launch", 20, MUTED).next_to(go, m.UP, buff=0.1)

        T0, lab_x, lane_h = -5.3, -5.5, 0.36
        ya_cpu, ya_gpu, yb_cpu, yb_gpu = 0.05, -0.6, -1.75, -2.4
        wl, pitch = 0.9, 0.95

        def launch(i, y):
            return span(T0 + i * pitch, T0 + i * pitch + wl, y, lane_h, LAUNCH, LAUNCH_FILL)

        a_launch = [launch(i, ya_cpu) for i in range(8)]
        a_kernel, end = [], T0
        for i in range(8):
            start = max(T0 + i * pitch + wl + 0.04, end + 0.03)
            end = start + 1.25
            a_kernel.append(span(start, end, ya_gpu, lane_h, MEMORY, CYAN_FILL))
        b_launch = [launch(i, yb_cpu) for i in range(11)]
        b_kernel = [span(T0 + i * pitch + wl + 0.04, T0 + i * pitch + wl + 0.26, yb_gpu, lane_h, MEMORY, CYAN_FILL)
                    for i in range(11)]
        b_gaps = [span(b_kernel[i].get_right()[0] + 0.02, b_kernel[i + 1].get_left()[0] - 0.02, yb_gpu, lane_h * 0.8,
                       RED, RED_FILL, 1.2) for i in range(10)]
        a_labels = m.VGroup(lane_label("CPU", lab_x, ya_cpu), lane_label("GPU", lab_x, ya_gpu))
        b_labels = m.VGroup(lane_label("CPU", lab_x, yb_cpu), lane_label("GPU", lab_x, yb_gpu))
        a_title = text("long kernels: the CPU runs ahead", 22, MUTED).move_to([0, ya_cpu + 0.45, 0]).align_to([6.4, 0, 0], m.RIGHT)
        b_title = text(f"Owl's tiny kernels: ≈ {AVG_US:.0f} µs each", 22, INK, weight=m.BOLD)
        b_title.move_to([0, yb_cpu + 0.45, 0]).align_to([6.4, 0, 0], m.RIGHT)
        brace_lab = code("Python → framework → driver", 20, LAUNCH).next_to(a_launch[0], m.UP, buff=0.14).align_to(a_launch[0], m.LEFT)
        idle = code("idle", 20, RED).next_to(b_gaps[1], m.DOWN, buff=0.12)
        tax = pill("the CPU tax", RED, RED_FILL, 24).move_to([0, -3.25, 0]).align_to([6.4, 0, 0], m.RIGHT)

        launch_pulses = []
        for i in range(3):
            launch_pulses.append(m.AnimationGroup(pulse(go, MUTED, 8), m.FadeIn(slots[i], scale=0.8), run_time=0.55))
        self.say(
            "[A GPU doesn't read Python.] [The CPU walks through the code], [launching one kernel after another.]",
            m.AnimationGroup(self.new_header("03 · launches", "The CPU tells, the GPU does"), m.FadeIn(cpu_card, shift=0.15 * m.UP),
                             m.FadeIn(gpu_card, shift=0.15 * m.UP), m.GrowArrow(go), m.FadeIn(go_lab)),
            m.Succession(m.FadeIn(hi, run_time=0.3), hi.animate(run_time=0.5).move_to(lines[1]),
                         hi.animate(run_time=0.5).move_to(lines[2])),
            m.Succession(*launch_pulses),
        )
        self.say(
            "[Each launch costs the CPU a few microseconds]: [Python, the framework, the driver.]",
            m.AnimationGroup(m.FadeIn(a_labels), m.GrowFromEdge(a_launch[0], m.LEFT)),
            m.FadeIn(brace_lab, shift=0.1 * m.UP),
        )
        self.say(
            "[When kernels run longer than that], [the CPU gets ahead], [and its cost hides behind the GPU's work.]",
            m.AnimationGroup(m.GrowFromEdge(a_kernel[0], m.LEFT), m.FadeIn(a_title)),
            m.LaggedStart(*[m.GrowFromEdge(b, m.LEFT) for b in a_launch[1:]], lag_ratio=0.35, run_time=1.2),
            m.LaggedStart(*[m.GrowFromEdge(k, m.LEFT) for k in a_kernel[1:]], lag_ratio=0.5, run_time=1.8),
        )
        pairs = [m.Succession(m.GrowFromEdge(b_launch[i], m.LEFT, run_time=0.3), m.FadeIn(b_kernel[i], run_time=0.15))
                 for i in range(11)]
        self.say(
            "[But Owl's kernels are tiny.] [The GPU finishes each one], [then waits for the next launch.]",
            m.AnimationGroup(m.FadeIn(b_labels), m.FadeIn(b_title, shift=0.1 * m.UP)),
            m.Succession(*pairs[:3]),
            m.Succession(*pairs[3:]),
        )
        self.say(
            "[That idle time between kernels is the CPU tax.]",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(g) for g in b_gaps], lag_ratio=0.15), m.FadeIn(idle),
                             m.FadeIn(tax, shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 4 · measured

    def measured_tax(self):
        self.next_section("4 · Measured", notes=(
            "PyTorch 'GPT, Fast' (2023, notes §7; blog read 2026-10-24): the baseline's trace 'reveals the answer – it's "
            "heavily CPU overhead bound! What this means is that our CPU is not able to tell the GPU what to do fast enough "
            f"for the GPU to be fully utilized.' {GF_EAGER} tok/s = {MS_EAGER:.1f} ms per token; the bandwidth floor at 100% "
            f"is {MS_FLOOR:.2f} ms ({GF_CEIL:.0f} tok/s, derived from the blog's {GF_MBU:.0%} MBU at {GF_COMPILED:.0f} tok/s). "
            f"Measured on this Mac (research/measurements.md §1, mlx-lm 0.32.2, 2026-10-24): Owl-mini (Llama-3.2-1B 4-bit) "
            f"{MINI_TOK:.1f} tok/s = {MINI_MS:.2f} ms per token; its {MINI_READ / 1e9:.2f} GB per step take {MINI_FLOOR_MS:.2f} ms "
            f"at 546 GB/s (ceiling {MINI_CEIL:.0f} tok/s, {MINI_FRAC:.0%}); the other {MINI_GAP:.2f} ms is fixed per-step cost "
            "(launches, sampling, Python — not profiled)."))
        q = quote_card(["“it's heavily CPU overhead bound! … our CPU is not able",
                        "to tell the GPU what to do fast enough”"], 9.6, 26).move_to([0, 1.75, 0])
        q_src = cite("PyTorch, “GPT, Fast”, 2023 · Llama-7B · A100-80GB · batch 1").next_to(q, m.DOWN, buff=0.14)
        q_src.align_to(q, m.LEFT)

        x0, per_ms = -2.6, 8.4 / MS_EAGER
        y1, y2 = 0.05, -0.6
        n1 = code("plain PyTorch", 20, INK).move_to([0, y1, 0]).align_to([x0 - 0.2, 0, 0], m.RIGHT)
        n2 = code("the floor", 20, MEMORY).move_to([0, y2, 0]).align_to([x0 - 0.2, 0, 0], m.RIGHT)
        bar1 = hbar(x0, y1, MS_EAGER * per_ms, 0.48, LAUNCH, LAUNCH_FILL)
        v1 = code(f"{MS_EAGER:.1f} ms per token", 20, INK).next_to(bar1, m.UP, buff=0.1).align_to(bar1, m.RIGHT)
        bar2 = hbar(x0, y2, MS_FLOOR * per_ms, 0.48, MEMORY, CYAN_FILL)
        v2 = code(f"{MS_FLOOR:.1f} ms · weights at full bandwidth", 20, MEMORY).next_to(bar2, m.RIGHT, buff=0.15)
        waste = hbar(x0 + MS_FLOOR * per_ms, y1, (MS_EAGER - MS_FLOOR) * per_ms, 0.48, RED, RED_FILL).set_z_index(1)
        waste_lab = code(f"≈ {MS_EAGER - MS_FLOOR:.0f} ms: mostly the GPU waiting", 20, RED).move_to(waste).set_z_index(2)

        mac_box = card(13.0, 2.4, EDGE, CARD).move_to([0, -2.3, 0])
        mac_icon = icon(MAC_ICON, 0.85).move_to([-5.85, -2.35, 0])
        mac_head = m.VGroup(text("measured on this Mac", 22, INK, weight=m.BOLD),
                            cite("Owl-mini 4-bit · mlx-lm"))
        mac_head.arrange(m.RIGHT, buff=0.3).move_to([0, -1.45, 0]).align_to([-5.2, 0, 0], m.LEFT)
        mx0, m_per = -2.0, 3.6 / MINI_MS
        ym1, ym2 = -2.0, -2.6
        m1n = code("one step", 20, TOKEN).move_to([0, ym1, 0]).align_to([mx0 - 0.2, 0, 0], m.RIGHT)
        m2n = code("its bytes alone", 20, MEMORY).move_to([0, ym2, 0]).align_to([mx0 - 0.2, 0, 0], m.RIGHT)
        mb1 = hbar(mx0, ym1, MINI_MS * m_per, 0.4, TOKEN, GREEN_FILL)
        mv1 = code(f"{MINI_MS:.2f} ms", 20, TOKEN).next_to(mb1, m.RIGHT, buff=0.15)
        mb2 = hbar(mx0, ym2, MINI_FLOOR_MS * m_per, 0.4, MEMORY, CYAN_FILL)
        mv2 = code(f"{MINI_FLOOR_MS:.2f} ms · {MINI_READ / 1e9:.2f} GB ÷ 546 GB/s", 20, MEMORY).next_to(mb2, m.RIGHT, buff=0.15)
        m_gap = hbar(mx0 + MINI_FLOOR_MS * m_per, ym1, MINI_GAP * m_per, 0.4, RED, RED_FILL).set_z_index(1)
        m_frac = pill(f"{MINI_TOK:.0f} of {MINI_CEIL:.0f} tok/s · {MINI_FRAC * 100:.0f}%", TOKEN, GREEN_FILL, 22)
        m_frac.move_to([0, -1.45, 0]).align_to([6.25, 0, 0], m.RIGHT)
        m_note = cite(f"{MINI_GAP:.2f} ms per step of other, fixed costs · not profiled").move_to([0, -3.12, 0]).align_to([mx0, 0, 0], m.LEFT)
        mac_card = m.VGroup(mac_box, mac_icon, mac_head)

        self.say(
            "[PyTorch's team saw exactly this in their trace.] [In their words]: [it's heavily CPU overhead bound.]",
            m.AnimationGroup(self.new_header("04 · measured", "CPU-bound, in the wild"), m.FadeIn(q[0]), m.FadeIn(q[1])),
            m.FadeIn(q_src),
            m.Succession(*[m.AddTextLetterByLetter(line, time_per_char=0.025) for line in q[2]]),
        )
        self.say(
            "[Each token took thirty-nine milliseconds.] [Reading the weights needs under seven.]",
            m.AnimationGroup(m.FadeIn(n1), m.GrowFromEdge(bar1, m.LEFT), m.FadeIn(v1)),
            m.AnimationGroup(m.FadeIn(n2), m.GrowFromEdge(bar2, m.LEFT), m.FadeIn(v2),
                             m.FadeIn(waste), m.FadeIn(waste_lab)),
        )
        self.say(
            "[Even on this Mac, measured]: [Owl-mini's step takes two milliseconds], [but its bytes need only one point three.]",
            m.FadeIn(mac_card, shift=0.15 * m.UP),
            m.AnimationGroup(m.FadeIn(m1n), m.GrowFromEdge(mb1, m.LEFT), m.FadeIn(mv1)),
            m.AnimationGroup(m.FadeIn(m2n), m.GrowFromEdge(mb2, m.LEFT), m.FadeIn(mv2), m.FadeIn(m_gap)),
        )
        self.say(
            "[That's sixty-three percent of its ceiling]: [the shorter the step, the bigger the share any fixed cost takes.]",
            m.FadeIn(m_frac, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(m_note), m.Circumscribe(m_gap, color=RED, buff=0.06)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 5 · fusion

    def fusion(self):
        self.next_section("5 · Fusion", notes=(
            "The RMS-norm line is TinyLM.forward's own (owl_reference.py, asserted). Run eagerly, it is five ops — square, "
            "mean, add epsilon, square root, divide — so five kernel launches and four intermediate results written to GPU "
            "memory and read back. A fused kernel reads x once, keeps the intermediates in registers/on-chip memory and "
            "writes h once (notes §7: 'Merge chains of small ops so activations do not round-trip through HBM'). The fused "
            f"layer ({len(FUSED)} kernels: norm, qkv, RoPE + cache write, attention, output + residual, norm, gate + up, "
            f"SiLU × up, down + residual) is illustrative: {LAYERS} × {len(FUSED)} + {len(EXTRA)} = {N_FUSED} launches."))
        block = code_block(NORM_LINE, 22, title="owl_reference.py · TinyLM.forward · RMS norm", accent=EDGE).move_to([0, 2.05, 0])
        body = block[1]
        ops = [("square", "x ** 2"), ("mean", ".mean"), ("add", "+ 1e-6"), ("sqrt", "np.sqrt"), ("divide", "x /")]
        chips = m.VGroup(*[kchip(name, MUTED) for name, _ in ops]).arrange(m.RIGHT, buff=1.05).move_to([0, 0.35, 0])
        launches = m.VGroup(*[arrow([c.get_x(), 1.2, 0], c.get_top(), LAUNCH, buff=0.06) for c in chips])
        n_launch = code("5 launches", 20, LAUNCH).move_to([0, 0.95, 0]).align_to([-6.4, 0, 0], m.LEFT)
        mem = hbar(-5.2, -1.45, 10.4, 0.5, MEMORY, CYAN_FILL)
        mem_lab = code("GPU memory (HBM)", 20, MEMORY).move_to(mem)
        trips = m.VGroup()
        for a, b in zip(chips[:-1], chips[1:]):
            down = thin_arrow([a.get_right()[0] - 0.12, a.get_bottom()[1], 0], [a.get_right()[0] - 0.12, mem.get_top()[1], 0], MEMORY)
            up = thin_arrow([b.get_left()[0] + 0.12, mem.get_top()[1], 0], [b.get_left()[0] + 0.12, b.get_bottom()[1], 0], MEMORY)
            trips.add(m.VGroup(down, up))
        trip_lab = text("4 in-between results: written to memory, read back", 22, MUTED).next_to(mem, m.DOWN, buff=0.2)

        fused = kchip("rms_norm · one fused kernel", TOKEN, 22, 0.6).move_to([0, 0.35, 0])
        one_launch = arrow([0, 1.2, 0], fused.get_top(), LAUNCH, buff=0.06)
        x_in = thin_arrow([-3.2, mem.get_top()[1], 0], [fused.get_left()[0] + 0.3, fused.get_bottom()[1], 0], MEMORY)
        h_out = thin_arrow([fused.get_right()[0] - 0.3, fused.get_bottom()[1], 0], [3.2, mem.get_top()[1], 0], MEMORY)
        io_lab = text("x read once, h written once", 22, TOKEN).next_to(mem, m.DOWN, buff=0.2)

        strip = [kchip(name, color) for name, color in LAYER]
        s1 = m.VGroup(*strip[:9]).arrange(m.RIGHT, buff=0.12).move_to([0, 1.35, 0])
        s2 = m.VGroup(*strip[9:]).arrange(m.RIGHT, buff=0.12).move_to([0, 0.65, 0])
        s_cap = text(f"one layer: {PER_LAYER} kernels", 22, MUTED).move_to([0, 2.15, 0])
        f_chips = [kchip(name, color) for name, color, _ in FUSED]
        f1 = m.VGroup(*f_chips[:4]).arrange(m.RIGHT, buff=0.16).move_to([0, -0.7, 0])
        f2 = m.VGroup(*f_chips[4:]).arrange(m.RIGHT, buff=0.16).move_to([0, -1.4, 0])
        f_cap = text(f"fused: {len(FUSED)} kernels · for example", 22, TOKEN).move_to([0, -2.15, 0])
        f_total = code(f"{LAYERS} × {len(FUSED)} + {len(EXTRA)} = {N_FUSED} launches per token", 22, INK).move_to([0, -2.85, 0])
        merges = [m.ReplacementTransform(m.VGroup(*[strip[i].copy() for i in ix]), f_chips[j])
                  for j, (_, _, ix) in enumerate(FUSED)]
        norm_stuff = m.VGroup(block, chips, launches, n_launch, mem, mem_lab, fused, one_launch, x_in, h_out, io_lab)

        circ = [m.AnimationGroup(m.Circumscribe(glyphs(body, key), color=INK, buff=0.05, run_time=0.7),
                                 m.FadeIn(chips[i], shift=0.15 * m.DOWN, run_time=0.5))
                for i, (_, key) in enumerate(ops)]
        self.say(
            "[First fix: fusion.] [Owl's norm is one line of code], [but five operations]: "
            "[square], [mean], [add], [square root], [divide.]",
            self.new_header("05 · fusion", "Fewer, bigger kernels"),
            m.FadeIn(block, shift=0.1 * m.DOWN),
            m.Circumscribe(body, color=MUTED, buff=0.08),
            *circ,
        )
        self.say(
            "[That's five launches], [and four in-between results, each written to memory and read back.]",
            m.AnimationGroup(m.LaggedStart(*[m.GrowArrow(a) for a in launches], lag_ratio=0.2), m.FadeIn(n_launch)),
            m.AnimationGroup(m.FadeIn(mem), m.FadeIn(mem_lab), m.FadeIn(trip_lab),
                             m.LaggedStart(*[m.Succession(m.GrowArrow(t[0], run_time=0.4), m.GrowArrow(t[1], run_time=0.4))
                                             for t in trips], lag_ratio=0.5)),
        )
        self.say(
            "[Fused, it's one kernel]: [one launch], [and the numbers never leave the chip.]",
            m.AnimationGroup(*[m.FadeOut(c, target_position=fused) for c in chips], m.FadeIn(fused, scale=0.8),
                             m.FadeOut(trips), m.FadeOut(trip_lab)),
            m.AnimationGroup(m.FadeOut(launches), m.FadeOut(n_launch), m.GrowArrow(one_launch)),
            m.AnimationGroup(m.GrowArrow(x_in), m.GrowArrow(h_out), m.FadeIn(io_lab, shift=0.1 * m.UP)),
        )
        self.say(
            "[Fuse across the layer], [and, for example, eighteen kernels become nine.]",
            m.AnimationGroup(m.FadeOut(norm_stuff, shift=0.2 * m.UP),
                             m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.DOWN) for c in strip], lag_ratio=0.06), m.FadeIn(s_cap)),
            m.AnimationGroup(*merges, m.FadeIn(f_cap), m.FadeIn(f_total, shift=0.1 * m.UP), run_time=1.6),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 6 · CUDA graphs

    def graphs(self):
        self.next_section("6 · CUDA graphs", notes=(
            "PyTorch, 'Accelerating PyTorch with CUDA Graphs' (2021; notes §7): in capture mode 'CUDA work issued to a "
            "capturing stream doesn't actually run on the GPU. Instead, the work is recorded in a graph'; 'Each replay runs "
            "the same kernels with the same arguments. For pointer arguments this means the same memory addresses are used'; "
            "'a graph replay skips all layers of argument setup and kernel dispatch, including Python, C++, and CUDA driver "
            "overheads'; graph-safe usually means static shapes. Hence a static (pre-allocated) KV cache (gpt-fast) and "
            "graphs captured for a set of batch sizes (vLLM: --max-cudagraph-capture-size, 'capped at 512 by default'; "
            f"--enforce-eager disables them). The {N_FUSED}-cell queue (section 5's fused step) and the timelines are "
            "schematic; the batch sizes on the cards are examples."))
        cols = 37
        colors = [c for _ in range(LAYERS) for _, c, _ in FUSED] + [c for _, c in EXTRA]
        assert len(colors) == N_FUSED
        queue = m.VGroup(*[cell(c, 0.12) for c in colors]).arrange_in_grid(rows=8, cols=cols, buff=0.045)
        queue.move_to([-2.75, 1.5, 0])
        q_cap = text(f"one step's launch queue: {N_FUSED} launches", 22, MUTED).next_to(queue, m.UP, buff=0.18).align_to(queue, m.LEFT)

        g_box = card(4.5, 1.95, TOKEN, GREEN_FILL).move_to([4.15, 1.45, 0])
        g_title = code("CUDA graph · one step", 20, TOKEN, weight=m.BOLD).move_to(g_box.get_top() + [0, -0.3, 0])
        g_slot = g_box.get_center() + np.array([0, -0.2, 0])
        capture = pill("capture: record, don't run", TOKEN, GREEN_FILL, 22).next_to(queue, m.DOWN, buff=0.18).align_to(queue, m.LEFT)

        T0, lab_x, lane_h = -5.3, -5.5, 0.36
        y_cpu, y_gpu = -0.85, -1.75
        lanes = m.VGroup(lane_label("CPU", lab_x, y_cpu), lane_label("GPU", lab_x, y_gpu))
        n, pitch, wl = 10, 1.0, 0.92
        e_launch = [span(T0 + i * pitch, T0 + i * pitch + wl, y_cpu, lane_h, LAUNCH, LAUNCH_FILL) for i in range(n)]
        e_kernel = [span(T0 + i * pitch + wl + 0.04, T0 + i * pitch + wl + 0.29, y_gpu, lane_h, MEMORY, CYAN_FILL)
                    for i in range(n)]
        e_gaps = [span(e_kernel[i].get_right()[0] + 0.02, e_kernel[i + 1].get_left()[0] - 0.02, y_gpu, lane_h * 0.8,
                       RED, RED_FILL, 1.2) for i in range(n - 1)]
        launches_g, gaps_g = m.VGroup(*e_launch), m.VGroup(*e_gaps)
        e_title = text("eager: one launch per kernel", 22, MUTED).move_to([0, y_cpu + 0.5, 0]).align_to([6.4, 0, 0], m.RIGHT)
        replay = span(T0, T0 + 0.62, y_cpu, lane_h, TOKEN, GREEN_FILL)
        r_lab = code("replay", 20, TOKEN).next_to(replay, m.RIGHT, buff=0.15)
        packed = [np.array([T0 + 0.66 + i * 0.28 + 0.125, y_gpu, 0]) for i in range(n)]
        r_title = text("graph replay: one launch per step", 22, TOKEN, weight=m.BOLD)
        r_title.move_to([0, y_cpu + 0.5, 0]).align_to([6.4, 0, 0], m.RIGHT)
        done = packed[-1][0] + 0.125
        saved = m.Line([done, y_gpu - 0.5, 0], [e_kernel[-1].get_right()[0], y_gpu - 0.5, 0]).set_stroke(TOKEN, 3)
        ticks = m.VGroup(*[m.Line([x, y_gpu - 0.6, 0], [x, y_gpu - 0.4, 0]).set_stroke(TOKEN, 3) for x in (saved.get_start()[0], saved.get_end()[0])])
        saved_lab = text("time saved, every token", 22, TOKEN).next_to(saved, m.DOWN, buff=0.14)

        q = quote_card(["“a graph replay skips all layers of argument",
                        "setup and kernel dispatch, including Python,",
                        "C++, and CUDA driver overheads.”"], 7.4, 22).move_to([-2.85, 1.6, 0])
        q_src = cite("PyTorch, “Accelerating PyTorch with CUDA Graphs”, 2021").next_to(q, m.DOWN, buff=0.12).align_to(q, m.LEFT)

        frozen = text("a graph is frozen:", 24, INK, weight=m.BOLD)
        same = m.VGroup(*[pill(s, TOKEN, GREEN_FILL, 22) for s in ("same kernels", "same shapes", "same addresses")])
        same.arrange(m.RIGHT, buff=0.25)
        frozen.move_to([0, 2.25, 0]).align_to([-6.4, 0, 0], m.LEFT)
        same.move_to([0, 1.7, 0]).align_to([-6.4, 0, 0], m.LEFT)

        kv_x0, kv_w, kv_y = -6.4, 3.6, 0.6
        kv_track = m.Rectangle(width=kv_w, height=0.36).set_stroke(KV, 2).set_fill(VIOLET_FILL, 1).move_to([kv_x0 + kv_w / 2, kv_y, 0])
        kv_used = hbar(kv_x0, kv_y, kv_w * 0.3, 0.36, KV, KV).set_z_index(1)
        kv_lab = text("KV cache, allocated full size", 20, KV).next_to(kv_track, m.UP, buff=0.1).align_to(kv_track, m.LEFT)
        sizes = m.VGroup(*[kchip(s, TOKEN, width=0.5 if len(s) < 3 else 0.72) for s in ("1", "2", "4", "8", "…", "512")])
        sizes.arrange(m.RIGHT, buff=0.1).move_to([0, kv_y, 0]).align_to([-2.2, 0, 0], m.LEFT)
        sizes_lab = text("one graph per batch size", 20, TOKEN).next_to(sizes, m.UP, buff=0.1).align_to(sizes, m.LEFT)
        sizes_src = cite("vLLM docs: up to 512 by default").next_to(sizes, m.DOWN, buff=0.1).align_to(sizes, m.LEFT)

        self.say(
            "[Fusion shortens the queue.] [The next fix, CUDA graphs, takes the CPU out of it.]",
            m.AnimationGroup(self.new_header("06 · CUDA graphs", "Record once, replay"), m.FadeIn(q_cap),
                             m.LaggedStart(*[m.FadeIn(c) for c in queue], lag_ratio=0.004, run_time=1.2),
                             m.FadeIn(lanes), m.FadeIn(launches_g), m.FadeIn(m.VGroup(*e_kernel)), m.FadeIn(gaps_g), m.FadeIn(e_title)),
            m.AnimationGroup(m.FadeIn(g_box), m.FadeIn(g_title)),
        )
        self.say(
            "[Run one step in capture mode]: [no kernel runs], [every launch is recorded into a graph.]",
            m.FadeIn(capture, shift=0.1 * m.UP),
            m.AnimationGroup(*[k.animate.set_opacity(0.15) for k in e_kernel], gaps_g.animate.set_opacity(0.15)),
            queue.animate(run_time=1.6).scale(0.55).move_to(g_slot),
        )
        self.say(
            "[Then, for every token], [the CPU makes one call], [and the GPU replays every kernel, back to back.]",
            m.AnimationGroup(m.FadeOut(capture), m.FadeOut(q_cap), *[k.animate.set_opacity(1) for k in e_kernel],
                             m.Indicate(g_box, color=TOKEN, scale_factor=1.04)),
            m.AnimationGroup(m.ReplacementTransform(launches_g, replay), m.FadeIn(r_lab), m.FadeOut(e_title), m.FadeIn(r_title)),
            m.AnimationGroup(*[k.animate.move_to(p) for k, p in zip(e_kernel, packed)], m.FadeOut(gaps_g),
                             m.Create(saved), m.FadeIn(ticks), m.FadeIn(saved_lab), run_time=1.5),
        )
        self.say(
            "[A replay skips Python, the framework and the driver.]",
            m.AnimationGroup(m.FadeIn(q, shift=0.1 * m.UP), m.FadeIn(q_src)),
        )
        self.say(
            "[The catch: a graph is frozen]: [same kernels, same shapes, same memory addresses.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(q, q_src), shift=0.15 * m.UP), m.FadeIn(frozen, shift=0.1 * m.UP)),
            m.LaggedStart(*[m.FadeIn(s, shift=0.1 * m.UP) for s in same], lag_ratio=0.45),
        )
        self.say(
            "[So the KV cache is allocated up front], [and servers like vLLM capture a graph for each of many batch sizes.]",
            m.AnimationGroup(m.FadeIn(kv_lab), m.FadeIn(kv_track), m.GrowFromEdge(kv_used, m.LEFT)),
            m.AnimationGroup(m.FadeIn(sizes_lab), m.LaggedStart(*[m.FadeIn(s, shift=0.1 * m.UP) for s in sizes], lag_ratio=0.2),
                             m.FadeIn(sizes_src)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 7 · torch.compile

    def compiler(self):
        self.next_section("7 · torch.compile", notes=(
            "PyTorch 'GPT, Fast' (2023; notes §7): 'torch.compile(decode_one_token, mode=\"reduce-overhead\", fullgraph=True)'; "
            "reduce-overhead applies CUDA graphs; compile generates the kernels (fused Triton); a static KV cache keeps the "
            f"shapes fixed. LLaMA-7B, batch 1, A100-80GB at 330 W: eager {GF_EAGER} → compile + static KV cache {GF_COMPILED} "
            f"tok/s (×{GF_SPEEDUP:.2f}, the blog: 'more than 4x'); '72% MBU' at {GF_COMPILED:.0f} tok/s, so the ceiling at "
            f"100% is {GF_CEIL:.1f} tok/s [DERIVED]. Per token: {MS_EAGER:.1f} → {MS_COMP:.2f} ms. The same post continues "
            "with int8 (157.4), int4 (202.1) and speculative decoding (244.7) — Episodes 13 and 18–19."))
        block = code_block(COMPILE_SRC, 22, width=6.4, title="gpt-fast, as in the PyTorch blog", accent=TOKEN).move_to([-3.3, 1.35, 0])
        block.align_to([-6.4, 0, 0], m.LEFT)
        p_fuse = pill("fuses ops into new kernels", MEMORY, CYAN_FILL, 22)
        p_graph = pill("reduce-overhead → CUDA graphs", TOKEN, GREEN_FILL, 22)
        m.VGroup(p_fuse, p_graph).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.22).next_to(block, m.DOWN, buff=0.4).align_to(block, m.LEFT)

        x0, full = 1.0, 5.2
        per = full / GF_CEIL
        y1, y2 = 1.5, 0.35
        lab1 = code("plain PyTorch", 20, MUTED).move_to([0, y1 + 0.45, 0]).align_to([x0, 0, 0], m.LEFT)
        lab2 = code("compile + static KV cache", 20, TOKEN).move_to([0, y2 + 0.45, 0]).align_to([x0, 0, 0], m.LEFT)
        tracks = m.VGroup(*[hbar(x0, y, full, 0.5, EDGE, CARD, 1.5) for y in (y1, y2)])
        bar1 = hbar(x0, y1, GF_EAGER * per, 0.5, RED, RED_FILL)
        bar2 = hbar(x0, y2, GF_COMPILED * per, 0.5, TOKEN, GREEN_FILL)
        val1 = code(f"{GF_EAGER:.1f} tok/s", 22, RED, weight=m.BOLD).next_to(bar1, m.RIGHT, buff=0.15)
        val2 = code(f"{GF_COMPILED:.0f} tok/s", 22, TOKEN, weight=m.BOLD).move_to(bar2).set_z_index(2)
        cx = x0 + full
        ceil_line = m.DashedLine([cx, y1 + 0.75, 0], [cx, y2 - 0.45, 0], dash_length=0.1).set_stroke(MEMORY, 2.5)
        ceil_lab = code(f"ceiling ≈ {GF_CEIL:.0f}", 20, MEMORY).next_to(ceil_line, m.DOWN, buff=0.1).align_to([cx + 0.2, 0, 0], m.RIGHT)

        times = stat(f"×{GF_SPEEDUP:.1f}", "", "same GPU, same model", TOKEN, 110).move_to([-4.4, -2.55, 0])
        mbu = pill(f"{GF_MBU * 100:.0f}% of the bandwidth ceiling (MBU)", MEMORY, CYAN_FILL, 22)
        mbu.move_to([0, -1.0, 0]).align_to([x0, 0, 0], m.LEFT)
        per_tok = code(f"{MS_EAGER:.1f} ms → {MS_COMP:.1f} ms per token", 26, INK,
                       t2c={rf"{MS_EAGER:.1f} ms": RED, rf"{MS_COMP:.1f} ms": TOKEN}).move_to([0, -2.15, 0]).align_to([-0.2, 0, 0], m.LEFT)
        src = cite("PyTorch, “GPT, Fast”, 2023 · Llama-7B · A100-80GB at 330 W · batch 1")
        src.next_to(per_tok, m.DOWN, buff=0.25).align_to([6.4, 0, 0], m.RIGHT)

        self.say(
            "[In PyTorch, the compiler does both]: [it fuses operations into new kernels], "
            "[and in reduce-overhead mode, it records CUDA graphs.]",
            m.AnimationGroup(self.new_header("07 · torch.compile", "25.5 → 107 tok/s"), m.FadeIn(block, shift=0.1 * m.UP)),
            m.FadeIn(p_fuse, shift=0.1 * m.UP),
            m.FadeIn(p_graph, shift=0.1 * m.UP),
        )
        self.say(
            "[Same GPU, same model, plus a static KV cache]: [twenty-five and a half tokens a second] [became a hundred and seven.]",
            m.AnimationGroup(m.FadeIn(lab1), m.FadeIn(lab2), m.FadeIn(tracks), m.Create(ceil_line), m.FadeIn(ceil_lab)),
            m.AnimationGroup(m.GrowFromEdge(bar1, m.LEFT), m.FadeIn(val1)),
            m.AnimationGroup(m.GrowFromEdge(bar2, m.LEFT, run_time=1.4), m.FadeIn(val2)),
        )
        self.say(
            "[Over four times faster], [at seventy-two percent of the bandwidth ceiling.]",
            m.FadeIn(times, scale=0.9),
            m.FadeIn(mbu, shift=0.1 * m.UP),
        )
        self.say(
            "[Each token went from thirty-nine milliseconds to just over nine.]",
            m.AnimationGroup(m.FadeIn(per_tok, shift=0.1 * m.UP), m.FadeIn(src)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 8 · Flash-Decoding

    def split_kv(self):
        self.next_section("8 · Flash-Decoding", notes=(
            f"Owl at a 128K-token context: {CTX:,} × 128 KiB = {KV_LONG / 2 ** 30:.0f} GiB of KV cache (Episodes 5, 9), as big "
            f"as the {W_BYTES / 1e9:.2f} GB of weights. Flash-Decoding (Dao, Haziza, Massa, Sizov, Oct 2023; notes §7): "
            "FlashAttention parallelizes over query blocks and batch, so at decode (one query) a small batch 'will only use "
            "a small part of the GPU'; Flash-Decoding splits the keys/values into chunks processed in parallel, each writes "
            "its partial output and log-sum-exp, a final reduction rescales and sums ('up to 8x' faster generation for very "
            "long sequences, CodeLlama-34B, up to 64K). H100 SXM5: 132 SMs (NVIDIA whitepaper, 2022). Schematic count: one "
            f"block of work per head → {HEADS} of {SMS} SMs busy ({BUSY:.0%}); split {SPLITS} ways (illustrative; real kernels "
            f"pick the split count to fill the chip) → {BLOCKS} blocks. The reduction is Episode 15's online softmax."))
        grid = m.VGroup(*[m.Square(0.3).set_stroke(EDGE, 1.5).set_fill(CARD, 1) for _ in range(SMS)])
        grid.arrange_in_grid(rows=11, cols=12, buff=0.07).move_to([4.25, -0.3, 0])
        g_title = code(f"H100 · {SMS} SMs", 22, INK, weight=m.BOLD).next_to(grid, m.UP, buff=0.2)
        g_src = cite("NVIDIA H100 whitepaper").next_to(grid, m.DOWN, buff=0.14)
        busy1 = code(f"{HEADS} of {SMS} busy · {BUSY * 100:.0f}%", 24, KV, weight=m.BOLD).next_to(g_src, m.DOWN, buff=0.14)
        busy2 = code(f"{BLOCKS} of {SMS} busy · {BLOCKS / SMS * 100:.0f}%", 24, KV, weight=m.BOLD).move_to(busy1)
        lit1 = [sq.copy().set_stroke(KV, 1.5).set_fill(KV, 0.85) for sq in grid[:HEADS]]
        lit2 = [sq.copy().set_stroke(KV, 1.5).set_fill(KV, 0.85) for sq in grid[HEADS:BLOCKS]]

        x0, w = -5.0, 6.2
        kv_bar = hbar(x0, 1.8, w, 0.42, KV, VIOLET_FILL)
        kv_cap = code(f"KV cache · 128K tokens · {KV_LONG / 2 ** 30:.0f} GiB", 20, KV).next_to(kv_bar, m.UP, buff=0.12)
        kv_cap.align_to(kv_bar, m.LEFT)
        w_bar = hbar(x0, 1.15, w * W_BYTES / KV_LONG, 0.42, WEIGHTS, BLUE_FILL)
        w_lab = code(f"weights · {W_BYTES / 1e9:.0f} GB", 20, INK).move_to(w_bar)
        w_src = cite("Episodes 5 and 9").next_to(w_bar, m.DOWN, buff=0.1).align_to(w_bar, m.LEFT)
        top = m.VGroup(kv_bar, kv_cap, w_bar, w_lab, w_src)

        ys = [0.35, -0.15, -0.65, -1.45]
        names = ["head 1", "head 2", "head 3", f"head {HEADS}"]
        strips = [hbar(x0, y, w, 0.26, KV, VIOLET_FILL, 1.5) for y in ys]
        h_labs = m.VGroup(*[code(s, 20, MUTED).move_to([0, y, 0]).align_to([x0 - 0.15, 0, 0], m.RIGHT) for s, y in zip(names, ys)])
        dots = code("⋮", 24, MUTED).move_to([x0 + w / 2, -1.02, 0])
        crawl = [hbar(x0, y, w, 0.26, KV, KV, 0).set_z_index(1) for y in ys]
        cw = w / SPLITS
        chunks = [m.VGroup(*[hbar(x0 + j * cw + 0.04, y, cw - 0.08, 0.26, KV, VIOLET_FILL, 1.5) for j in range(SPLITS)]) for y in ys]
        fills = [[hbar(x0 + j * cw + 0.04, y, cw - 0.08, 0.26, KV, KV, 0).set_z_index(1) for j in range(SPLITS)] for y in ys]
        idle_lab = text("three quarters of the chip waits", 22, RED, weight=m.BOLD).next_to(g_title, m.UP, buff=0.12)

        yp = -2.3
        parts = m.VGroup(*[kchip(f"part {j + 1}", KV, width=1.2) for j in range(SPLITS)])
        for j, p in enumerate(parts):
            p.move_to([x0 + (j + 0.5) * cw, yp, 0])
        out = kchip("output", TOKEN, width=1.5).move_to([x0 + w / 2, -3.15, 0])
        joins = m.VGroup(*[m.Line(p.get_bottom() + np.array([0, -0.04, 0]), out.get_top() + np.array([(j - 1.5) * 0.3, 0.04, 0]))
                           .set_stroke(MUTED, 3) for j, p in enumerate(parts)])
        part_lab = text("each: a partial output, its max and its sum", 20, MUTED).next_to(parts, m.UP, buff=0.12)
        part_lab.align_to(parts, m.LEFT).set_z_index(3)
        ep15 = pill("online softmax · Ep 15", TOKEN, GREEN_FILL, 20).next_to(out, m.LEFT, buff=0.35)
        rescale = code("rescale + add", 20, MUTED).next_to(out, m.RIGHT, buff=0.35)

        eight = stat("up to 8×", "", "faster generation on very long sequences", TOKEN, 64)
        eight.move_to([-2.0, 1.85, 0]).align_to([-6.4, 0, 0], m.LEFT)
        eight_src = cite("Dao, Haziza, Massa, Sizov · Flash-Decoding, 2023 · CodeLlama-34B").next_to(eight, m.DOWN, buff=0.12)
        eight_src.align_to(eight, m.LEFT)

        self.say(
            "[One more place where the GPU starves]: [attention over a long chat.] "
            "[At a hundred and twenty-eight thousand tokens, Owl's cache is as big as its weights.]",
            m.AnimationGroup(self.new_header("08 · split-KV", "Flash-Decoding"), m.FadeIn(grid), m.FadeIn(g_title), m.FadeIn(g_src)),
            m.AnimationGroup(m.FadeIn(kv_cap), m.GrowFromEdge(kv_bar, m.LEFT)),
            m.AnimationGroup(m.GrowFromEdge(w_bar, m.LEFT), m.FadeIn(w_lab), m.FadeIn(w_src)),
        )
        self.say(
            "[At batch one, attention hands out one block of work per head], [and Owl has thirty-two heads.]",
            m.AnimationGroup(*[m.GrowFromEdge(s, m.LEFT) for s in strips], m.FadeIn(h_labs)),
            m.FadeIn(dots),
        )
        self.say(
            "[So only thirty-two of the H100's hundred and thirty-two streaming multiprocessors get any work.]",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(sq, scale=0.6) for sq in lit1], lag_ratio=0.05), m.FadeIn(busy1)),
        )
        self.say(
            "[Each one crawls through the whole cache alone], [while three quarters of the chip waits.]",
            m.AnimationGroup(*[m.GrowFromEdge(c, m.LEFT, run_time=3.0, rate_func=m.linear) for c in crawl]),
            m.AnimationGroup(m.FadeIn(idle_lab, shift=0.1 * m.UP),
                             m.LaggedStart(*[m.Indicate(sq, color=RED, scale_factor=1.0) for sq in grid[HEADS:]], lag_ratio=0.01)),
        )
        self.say(
            "[Flash-Decoding splits the cache into chunks], [say four per head], [so a hundred and twenty-eight blocks run at once.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(*crawl)), m.FadeOut(idle_lab),
                             *[m.ReplacementTransform(s, c) for s, c in zip(strips, chunks)]),
            m.AnimationGroup(*[m.GrowFromEdge(f, m.LEFT, run_time=0.9, rate_func=m.linear) for row in fills for f in row]),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(sq, scale=0.6) for sq in lit2], lag_ratio=0.02),
                             m.ReplacementTransform(busy1, busy2)),
        )
        self.say(
            "[Each chunk keeps its own running max and sum], [and one last small step rescales and adds them]: "
            "[the online softmax from episode fifteen.]",
            m.AnimationGroup(m.FadeOut(top, shift=0.15 * m.UP),
                             *[m.FadeIn(p, target_position=c) for p, c in zip(parts, chunks[-1])], m.FadeIn(part_lab)),
            m.AnimationGroup(m.LaggedStart(*[m.Create(j) for j in joins], lag_ratio=0.15), m.FadeIn(out, shift=0.1 * m.UP),
                             m.FadeIn(rescale)),
            m.FadeIn(ep15, shift=0.1 * m.LEFT),
        )
        self.say(
            "[On very long sequences], [its authors measured up to eight times faster generation.]",
            m.FadeIn(eight_src),
            m.FadeIn(eight, scale=0.9),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 9 · recap

    def recap(self):
        self.next_section("9 · Recap", notes=(
            "At batch 1 a decode step is hundreds of tiny kernels; launching them from Python can take longer than running "
            "them (the CPU tax). Fuse (fewer, bigger kernels), replay (CUDA graphs: one launch per step; torch.compile does "
            f"both: gpt-fast {GF_EAGER} → {GF_COMPILED:.0f} tok/s), split (Flash-Decoding spreads a long chat's attention "
            "over every SM). Next: Episode 17, the last step of a token."))
        x0, y1, y2 = -2.6, 1.55, 0.85
        t1 = code("launch it · CPU", 20, LAUNCH).move_to([0, y1, 0]).align_to([x0 - 0.2, 0, 0], m.RIGHT)
        t2 = code("do it · GPU", 20, MEMORY).move_to([0, y2, 0]).align_to([x0 - 0.2, 0, 0], m.RIGHT)
        b1 = hbar(x0, y1, 4.2, 0.46, LAUNCH, LAUNCH_FILL)
        b2 = hbar(x0, y2, 1.3, 0.46, MEMORY, CYAN_FILL)
        cap = text("one tiny kernel at batch one · schematic", 20, MUTED).next_to(b1, m.RIGHT, buff=0.3)
        lead = m.VGroup(t1, t2, b1, b2, cap)

        halo = glow(TOKEN, radius=3.4).move_to([0, -0.9, 0])
        words = m.VGroup(title_text("Fuse", 80, MEMORY), title_text("Replay", 80, TOKEN), title_text("Split", 80, KV))
        words.arrange(m.RIGHT, buff=1.4).move_to([0, -0.75, 0])
        on_baseline(words)
        a1 = arrow(words[0].get_right(), words[1].get_left(), MUTED, buff=0.25)
        a2 = arrow(words[1].get_right(), words[2].get_left(), MUTED, buff=0.25)
        subs = m.VGroup(*[text(s, 22, MUTED) for s in ("fewer launches", "no gaps", "the whole chip at work")])
        for s, w in zip(subs, words):
            s.move_to([w.get_x(), words.get_bottom()[1] - 0.35, 0])
        mascot = owl(0.7).move_to([0, -3.1, 0])

        self.say(
            "[At batch one, the GPU is so fast that launching work can cost more than doing it.]",
            m.AnimationGroup(self.new_header("09 · recap", "Fuse, replay, split"),
                             m.AnimationGroup(m.FadeIn(t1), m.GrowFromEdge(b1, m.LEFT), m.FadeIn(t2), m.GrowFromEdge(b2, m.LEFT),
                                              m.FadeIn(cap))),
        )
        self.say(
            "[Three words to keep]: [fuse], [replay], [split.]",
            m.FadeIn(halo, scale=0.6),
            m.FadeIn(words[0], shift=0.2 * m.UP),
            m.AnimationGroup(m.GrowArrow(a1), m.FadeIn(words[1], shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowArrow(a2), m.FadeIn(words[2], shift=0.2 * m.UP)),
        )
        self.say(
            "[Fewer launches, no gaps], [and the whole chip at work.]",
            m.AnimationGroup(m.FadeIn(subs[0], shift=0.1 * m.UP), m.FadeIn(subs[1], shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(subs[2], shift=0.1 * m.UP), m.FadeIn(mascot, shift=0.2 * m.UP)),
        )
        self.wait(0.6)
        self.clear()
