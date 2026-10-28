"""Tokens per Second · Episode 15 · FlashAttention (Module 4 · Faster Tokens)

Exact attention, reordered to respect the memory hierarchy. Standard attention writes the N×N score matrix to
HBM and reads it back; FlashAttention tiles Q, K and V through on-chip SRAM, keeps a running max and sum (the
online softmax, rescaled as it goes) and never stores the matrix. Running example: Owl-8B (Llama-3.1-8B: 8 B
parameters, 16 GB of BF16 weights, 128 KiB of KV cache per token) reading a student's 4,096-token chapter
(prefill), and notes §7's online softmax on x = [1, 2, 3, 0] in two tiles, computed and checked below.

Sections: hook (a 4,096 × 4,096 score grid per head; the textbook way round-trips it through memory) · two
memories (H100: HBM 80 GB at 3.35 TB/s vs SRAM 228 KiB × 132 SMs ≈ 30.8 MB at 31 TB/s: 9× faster, 2,600×
smaller) · the textbook way (one head: S written, read, P written, read; a byte counter: 138.4 MB, of which
134.2 MB is the grid) · tiling (a Q tile stays in SRAM, K/V tiles stream through, score blocks live and die on
the chip, only O goes back; the catch: softmax needs the row's max and sum) · online softmax (x = [1, 2 | 3, 0]:
sum 1.368 → rescaled 0.503 → 1.553 = the plain sum; the output accumulator rescaled the same way: 26.21 both
ways) · the bytes, counted (Owl's whole prefill: 140 GB = 41.8 ms of traffic vs 8.9 ms of math, against
2.7 GB = 0.8 ms; the paper's GPT-2 measurement; this Mac's unfused path: 16.0 vs 9.5 GB peak) · FA2 / FA3
(share of peak FLOPs) · recap (tile, rescale, exact).

Sources (facts dated October 2026):
- research/notes.md §7: FlashAttention (Dao et al., NeurIPS 2022, arXiv 2205.14135): "an IO-aware exact
  attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth
  memory (HBM) and GPU on-chip SRAM"; standard attention writes S and P to HBM (Algorithm 0, re-read in the
  paper 2026-10-24); online softmax (Milakov & Gimelshein 2018); worked example x = [1, 2 | 3, 0] → 1.3679,
  0.5032, 1.0498, 1.5530, softmax [0.087, 0.237, 0.644, 0.032]; N = 4,096 FP16 scores = 33.5 MB per head.
- FlashAttention paper, Fig. 2 (re-read 2026-10-24): GPT-2 medium (seq. 1024, head dim 64, 16 heads, batch 64),
  A100, forward + backward: standard 66.6 GFLOPs, 40.3 GB HBM R/W, 41.7 ms; FlashAttention 75.2 GFLOPs, 4.4 GB,
  7.3 ms. Fig. 1: 7.6× on the attention computation.
- FlashAttention-2 (Dao 2023, arXiv 2307.08691; notes §7): "around 2× speedup compared to FlashAttention,
  reaching 50-73% of the theoretical maximum FLOPs/s on A100"; FlashAttention "only 25-40%".
- FlashAttention-3 (Shah et al. 2024, arXiv 2407.08608, re-read 2026-10-24): Table 1, H100 SXM5: GMEM (HBM)
  "80 GiB @ 3.35 TB/s", L2 "50 MiB @ 12 TB/s", SMEM "228 KiB per SM, 31TB/s per GPU"; abstract: "FP16 reaching
  up to 740 TFLOPs/s (75% utilization), and with FP8 reaching close to 1.2 PFLOPs/s"; "FlashAttention-2
  achieving only 35% utilization on the H100 GPU". Techniques: warp specialisation / TMA (asynchronous loads),
  pingpong overlap of softmax and matmul, FP8.
- H100 SXM: 132 SMs (NVIDIA H100 Tensor Core GPU architecture whitepaper); 80 GB HBM3, 3.35 TB/s, 989 TFLOPS
  BF16 dense (NVIDIA spec, owl_reference HARDWARE; the course writes 80 GB).
- research/measurements.md §5 (this Mac Studio, M4 Max, mlx-lm 0.32.2, 2026-10-24), via measured(): Llama-3.1-8B
  4-bit at 32K tokens peaks at 9.53 GB with the BF16 cache (fused attention kernel) and 15.96 GB with the 8-bit
  cache, whose attention path "is not a fused kernel and materializes the chunk × context score matrix".
- Episode 2 (the two speed limits, bandwidth ÷ bytes), Episode 9 (attention's cost grows with context),
  Episode 16 (Flash-Decoding), 'AI, Explained' Episode 37 (the idea of FlashAttention; this episode is the IO
  view, counted in bytes).

Computed here (owl_reference: OWL config, GPU): one head of a 4,096-token prefill, BF16: Q, K, V, O 1 MiB each;
S (and P) 33.55 MB; S and P each written once and read once = 134.2 MB, plus 4.2 MB of inputs and output. Owl's
whole prefill (32 query heads, 8 KV heads, 32 layers): standard 140.1 GB (41.8 ms at 3.35 TB/s), FlashAttention
at least 2.68 GB (0.80 ms), counting each input read once and the output written once; real kernels re-read K/V
tiles once per query tile (much of it served by the 50 MB L2 cache), which is why the paper's measured
reduction (9×) is smaller than this idealised 52×. Attention math over the full grid (QKᵀ and PV): 8.8 TFLOP =
8.9 ms at the H100's 989 TFLOPS peak.

Illustrative (said as such or schematic): the 16 × 16 and 12 × 12 grids (standing for 4,096 × 4,096), the bar
proportions of Q/K/V (N × d drawn wide), four tiles per matrix, the values v = [10, 20, 30, 40] ("for example":
one coordinate of each value vector), the package drawing (6 HBM stacks, 12 × 11 SMs = 132).

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from owl_reference import *  # noqa: E402,F403

PREFIX = "ep15"
KIB, MIB = 2**10, 2**20

# ---------------------------------------------------------------------------- the numbers

# the memory hierarchy of one H100 (FA3 Table 1; NVIDIA)
HBM_BYTES, HBM_BW = GPU["memory"], GPU["bandwidth"]
SMS, SMEM = 132, 228 * KIB
SRAM_BYTES = SMS * SMEM
SRAM_BW = 31e12
FASTER, SMALLER = SRAM_BW / HBM_BW, HBM_BYTES / SRAM_BYTES
assert HBM_BYTES == 80e9 and HBM_BW == 3.35e12  # "eighty gigabytes … three point three five terabytes per second"
assert round(SRAM_BYTES / 1e6) == 31  # "about thirty-one megabytes in all"
assert round(FASTER) == 9 and round(SMALLER, -2) == 2600  # "nine times faster … twenty-six hundred times smaller"
assert SMS == 12 * 11  # the drawing's 12 × 11 grid of SMs

# Owl reading a 4,096-token chapter (prefill), BF16
N, D, B = 4096, OWL["head_dim"], 2
H, HKV, L = OWL["heads"], OWL["kv_heads"], OWL["layers"]
SCORES = N * N
GRID = SCORES * B  # S (or P) for one head in one layer
MAT = N * D * B  # Q, K, V or O for one head
assert (D, H, HKV, L) == (128, 32, 8, 32) and MAT == MIB  # "four thousand rows of a hundred and twenty-eight numbers"
assert round(SCORES / 1e6, 1) == 16.8  # "about seventeen million scores"
assert abs(GRID / 1e6 - 33.5) < 0.1  # 33.55 MB: "thirty-three and a half megabytes" (notes §7: 33.5 MB)
GRID_TRAFFIC = 4 * GRID  # S written + read back, P written + read back
IO_HEAD = 4 * MAT  # Q and K read, V read, O written
assert round(GRID_TRAFFIC / 1e6) == 134 and round(IO_HEAD / 1e6) == 4  # "a hundred and thirty-four … four"
# the byte counter, step by step (one head)
COUNT = [2 * MAT + GRID, 2 * MAT + 3 * GRID, 4 * MAT + 4 * GRID]
assert COUNT[-1] == GRID_TRAFFIC + IO_HEAD and [round(c / 1e6, 1) for c in COUNT] == [35.7, 102.8, 138.4]

# the whole prefill: 32 query heads (Q, O), 8 KV heads (K, V), 32 layers
IO_LAYER = (2 * H + 2 * HKV) * MAT
STD_BYTES = L * (H * GRID_TRAFFIC + IO_LAYER)
FLASH_BYTES = L * IO_LAYER  # each input read once, the output written once
STD_MS, FLASH_MS = STD_BYTES / HBM_BW * 1e3, FLASH_BYTES / HBM_BW * 1e3
ATT_FLOPS = 2 * 2 * SCORES * D * H * L  # QKᵀ and PV over the full grid, 2 FLOPs per multiply-add
MATH_MS = ATT_FLOPS / GPU["peak"] * 1e3
assert round(STD_BYTES / 1e9) == 140 and round(FLASH_BYTES / 1e9, 1) == 2.7 < 3  # "a hundred and forty … under three"
assert round(STD_MS) == 42 and FLASH_MS < 1 and round(MATH_MS) == 9  # "forty-two … less than a millisecond … nine"
assert round(ATT_FLOPS / 1e12, 1) == 8.8

# the paper's measurement (Dao et al. 2022, Fig. 2: GPT-2 medium, A100, forward + backward)
PAPER = dict(gflops=(66.6, 75.2), hbm_gb=(40.3, 4.4), ms=(41.7, 7.3))
assert 1.1 < PAPER["gflops"][1] / PAPER["gflops"][0] < 1.15  # "a little more math"
assert round(PAPER["hbm_gb"][0] / PAPER["hbm_gb"][1]) == 9  # "nine times fewer bytes"
assert 5.5 < PAPER["ms"][0] / PAPER["ms"][1] < 6  # "almost six times faster"

# measured on this Mac: peak memory at 32K tokens, Owl-8B 4-bit (fused kernel vs the unfused path)
CTX = 32768
PEAK_FUSED = measured("memory", "llama-3.1-8b", "4bit", "kv16", "by_context_tokens", str(CTX), "peak_GB")
PEAK_UNFUSED = measured("memory", "llama-3.1-8b", "4bit", "kv8", "by_context_tokens", str(CTX), "peak_GB")
assert round(PEAK_FUSED, 1) == 9.5 and round(PEAK_UNFUSED) == 16  # "sixteen gigabytes, not nine and a half"

# versions: share of the GPU's peak 16-bit FLOPs (FA2 and FA3 papers)
VERSIONS = [  # (name, gpu, year, low %, high %)
    ("FlashAttention-1", "A100", 2022, 25, 40),
    ("FlashAttention-2", "A100", 2023, 50, 73),
    ("FlashAttention-2", "H100", 2023, 35, 35),
    ("FlashAttention-3", "H100", 2024, 75, 75),
]
FA3_TFLOPS, FA3_FP8_PFLOPS = 740, 1.2
assert round(FA3_TFLOPS / 989 * 100) == 75  # 740 of 989 dense BF16/FP16 TFLOPS = "three quarters of the chip"

# the online softmax (notes §7): x = [1, 2 | 3, 0] in two tiles; v = one coordinate of each value, for example
X = np.array([1.0, 2.0, 3.0, 0.0])
V = np.array([10.0, 20.0, 30.0, 40.0])


def online_softmax(x_tiles, v_tiles):
    """FlashAttention's inner loop for one row: a running max, a running sum and a running output, rescaled
    by exp(old max − new max) whenever a tile raises the max."""
    run_max, run_sum, run_out, steps = -np.inf, 0.0, 0.0, []
    for xt, vt in zip(x_tiles, v_tiles):
        new_max = max(run_max, float(xt.max()))
        scale = np.exp(run_max - new_max)  # 0 on the first tile
        p = np.exp(xt - new_max)
        steps.append(dict(max=new_max, scale=scale, old=run_sum * scale, new=p.sum(),
                          out_old=run_out * scale, out_new=float(p @ vt)))
        run_sum = run_sum * scale + p.sum()
        run_out = run_out * scale + float(p @ vt)
        run_max = new_max
    return run_max, run_sum, run_out, steps


MAX, SUM, ACC, STEPS = online_softmax([X[:2], X[2:]], [V[:2], V[2:]])
PLAIN = np.exp(X - X.max())
SOFT = PLAIN / PLAIN.sum()
OUT, PLAIN_OUT = ACC / SUM, float(SOFT @ V)
f3 = lambda v: f"{v:.3f}"  # noqa: E731
assert MAX == 3 and STEPS[0]["max"] == 2 and STEPS[1]["max"] == 3
assert f3(STEPS[0]["new"]) == "1.368"  # "one point three six eight"
assert f3(STEPS[1]["scale"]) == "0.368" and f3(STEPS[1]["old"]) == "0.503"  # "zero point five zero three"
assert f3(STEPS[1]["new"]) == "1.050" and f3(SUM) == f3(PLAIN.sum()) == "1.553"  # "one point five five three"
assert abs(SUM - PLAIN.sum()) < 1e-12 and [f3(p) for p in SOFT] == ["0.087", "0.237", "0.644", "0.032"]
assert abs(OUT - PLAIN_OUT) < 1e-12 and f"{OUT:.2f}" == f"{PLAIN_OUT:.2f}" == "26.21"  # "agree … down to rounding"
assert f"{STEPS[0]['out_new']:.2f}" == "23.68" and f"{ACC:.2f}" == "40.70"

# ---------------------------------------------------------------------------- look


def cite(s, size=20):
    return text(s, size, MUTED)


def mb(n, digits=1):
    return f"{n / 1e6:.{digits}f} MB"


def rect(w, h, color, fill, stroke=2.5, opacity=1.0):
    return m.Rectangle(width=w, height=h).set_stroke(color, stroke).set_fill(fill, opacity)


def mini_grid(n, size, color=COMPUTE, fill=AMBER_FILL, gap=None):
    """An n × n grid of score cells, `size` units across."""
    gap = size * 0.012 if gap is None else gap
    cell = (size - (n - 1) * gap) / n
    return m.VGroup(*[
        m.VGroup(*[m.Square(cell).set_stroke(color, 1.2).set_fill(fill, 1) for _ in range(n)]).arrange(m.RIGHT, buff=gap)
        for _ in range(n)
    ]).arrange(m.DOWN, buff=gap)


def dashed_box(w, h, color=MUTED, dashes=40):
    return m.DashedVMobject(m.Rectangle(width=w, height=h), num_dashes=dashes).set_stroke(color, 2)


def mac_stamp():
    mac = icon(recolored("mac", MUTED, CARD, PREFIX), 0.42)
    return m.VGroup(mac, cite("measured on this Mac · mlx-lm")).arrange(m.RIGHT, buff=0.2)


# the stage of sections 2 and 3: HBM on the left, the chip (SRAM + math units) on the right
BAR_W, BAR_H, BAR_Y = 0.45, 3.0, -0.95
BAR_X = {"Q": -6.0, "K": -5.4, "V": -4.8, "O": -4.2}
BAR_COL = {"Q": (TOKEN, GREEN_FILL), "K": (KV, VIOLET_FILL), "V": (KV, VIOLET_FILL), "O": (TOKEN, GREEN_FILL)}
GRID_C, GRID_S = np.array([-2.05, BAR_Y, 0]), 3.0
SLOT_Y = 0.1
SLOT_X = {"Q": 2.95, "K": 3.55, "V": 4.15, "S": 5.05, "O": 5.85}
TILES, TILE_H = 4, BAR_H / 4


def stage():
    hbm = card(6.75, 4.9, MEMORY, CARD, radius=0.2, stroke=2.5).move_to([-3.25, -0.6, 0])
    hbm_t = code("HBM · 80 GB · 3.35 TB/s", 20, MEMORY, weight=m.BOLD).align_to(hbm, m.UL).shift(0.28 * m.RIGHT + 0.22 * m.DOWN)
    chip = card(4.45, 4.9, COMPUTE, CARD, radius=0.2, stroke=2.5).move_to([4.35, -0.6, 0])
    chip_t = code("on the chip", 20, COMPUTE, weight=m.BOLD).align_to(chip, m.UL).shift(0.28 * m.RIGHT + 0.22 * m.DOWN)
    sram = card(3.85, 2.25, MEMORY, CYAN_FILL, radius=0.14, stroke=2).move_to([4.35, 0.2, 0])
    sram_t = code("SRAM · fast, small", 20, MEMORY).align_to(sram, m.UL).shift(0.2 * m.RIGHT + 0.14 * m.DOWN)
    maths = card(3.85, 0.9, COMPUTE, AMBER_FILL, radius=0.14, stroke=2).move_to([4.35, -2.3, 0])
    maths_t = text("math units", 24, INK, weight=m.BOLD).move_to(maths)
    read = arrow([0.22, 0.45, 0], [2.0, 0.45, 0], MEMORY, buff=0)
    write = arrow([2.0, -1.35, 0], [0.22, -1.35, 0], MEMORY, buff=0)
    read_t = code("read", 20, MEMORY).next_to(read, m.UP, buff=0.12)
    write_t = code("write", 20, MEMORY).next_to(write, m.DOWN, buff=0.12)
    labels = m.VGroup(*[code(k, 24, BAR_COL[k][0], weight=m.BOLD).move_to([BAR_X[k], BAR_Y + BAR_H / 2 + 0.47, 0])
                        for k in ("Q", "K", "V", "O")])
    slot_labs = m.VGroup(*[code(k, 20, MUTED).move_to([SLOT_X[k], SLOT_Y - 0.62, 0]) for k in SLOT_X])
    return dict(hbm=m.VGroup(hbm, hbm_t), chip=m.VGroup(chip, chip_t, sram, sram_t, slot_labs), maths=m.VGroup(maths, maths_t),
                road=m.VGroup(read, write, read_t, write_t), read=read, write=write, labels=labels, maths_box=maths)


def whole_bar(k):
    color, fill = BAR_COL[k]
    return rect(BAR_W, BAR_H, color, fill).move_to([BAR_X[k], BAR_Y, 0])


def bar_tiles(k, spread=0.0):
    color, fill = BAR_COL[k]
    top = BAR_Y + BAR_H / 2 + 1.5 * spread
    return m.VGroup(*[rect(BAR_W, TILE_H, color, fill).move_to([BAR_X[k], top - (i + 0.5) * TILE_H - i * spread, 0])
                      for i in range(TILES)])


def slot(k, w=BAR_W, h=TILE_H):
    color, fill = BAR_COL.get(k, (COMPUTE, AMBER_FILL))
    return rect(w, h, color, fill).move_to([SLOT_X[k], SLOT_Y, 0])


# ---------------------------------------------------------------------------- the episode


class Ep15FlashAttention(Episode):
    number = 15

    def lesson(self):
        self.hook()
        self.two_memories()
        self.textbook()
        self.tiling()
        self.online()
        self.counted()
        self.versions()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes=f"Owl reads a student's chapter: {N:,} tokens of prefill. In every layer and every head, attention scores each token against every other: {N:,} × {N:,} = {SCORES / 1e6:.1f} M scores (the 16 × 16 grid is schematic). Standard attention writes this grid to the GPU's memory (HBM) and reads it back; FlashAttention (Dao et al. 2022) never stores it, and is exact. 'AI, Explained' Episode 37 sketched the idea; this episode is the IO view, counted in bytes.")
        doc = icon(recolored("doc", REQUEST, PINK_FILL, PREFIX), 1.45).move_to([-5.3, 1.15, 0])
        toks = code(f"{N:,} tokens", 26, REQUEST, weight=m.BOLD).next_to(doc, m.DOWN, buff=0.3)
        toks_sub = cite("a chapter of notes").next_to(toks, m.DOWN, buff=0.14)
        mascot = owl(0.6).move_to([-5.3, -2.55, 0])
        grid = mini_grid(16, 3.1).move_to([-1.15, 0.75, 0])
        cells = [c for row in grid for c in row]
        order = sorted(range(len(cells)), key=lambda i: (i // 16) + (i % 16))
        g_big = title_text(f"{N:,} × {N:,}", 46, COMPUTE).move_to([1.15, 1.75, 0], aligned_edge=m.LEFT)
        g_sub = code(f"≈ {SCORES / 1e6:.1f} M scores", 24, INK).next_to(g_big, m.DOWN, buff=0.25).align_to(g_big, m.LEFT)
        g_sub2 = cite("for every head, in every layer").next_to(g_sub, m.DOWN, buff=0.16).align_to(g_big, m.LEFT)
        hbm = card(5.6, 0.75, MEMORY, CYAN_FILL, radius=0.14, stroke=2.5).move_to([-1.15, -2.45, 0])
        hbm_t = code("HBM · the GPU's memory", 22, INK).move_to(hbm)
        down = arrow([-1.75, -0.92, 0], [-1.75, -1.98, 0], MEMORY, buff=0)
        up = arrow([-0.55, -1.98, 0], [-0.55, -0.92, 0], MEMORY, buff=0)
        down_t = code("write", 20, MEMORY).next_to(down, m.LEFT, buff=0.15)
        up_t = code("read", 20, MEMORY).next_to(up, m.RIGHT, buff=0.15)
        ghost = dashed_box(3.1, 3.1, MUTED, 48).move_to(grid)
        never = pill("FlashAttention: never stored", GOOD, GOOD_FILL, 24).move_to([3.55, -0.35, 0])
        exact = code("same answer, exactly", 22, GOOD).next_to(never, m.DOWN, buff=0.22)
        ep37 = m.VGroup(code("AI, Explained · Episode 37", 20, MUTED), cite("the idea")).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        ep15 = m.VGroup(code("this episode", 20, MEMORY, weight=m.BOLD), text("count the bytes", 22, INK)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        ep37.move_to([1.95, -1.75, 0], aligned_edge=m.LEFT)
        ep15.move_to([1.95, -2.65, 0], aligned_edge=m.LEFT)

        self.say(
            "[A student pastes a chapter of notes into Owl]: [about four thousand tokens to read before the first word.]",
            m.AnimationGroup(self.new_header("hook", "The grid that never exists"), m.FadeIn(doc, shift=0.2 * m.UP),
                             m.FadeIn(mascot, shift=0.2 * m.UP)),
            m.AnimationGroup(m.FadeIn(toks, shift=0.1 * m.UP), m.FadeIn(toks_sub)),
        )
        self.say(
            "[In every layer, attention scores each token against every other one]: [about seventeen million scores, for every head.]",
            m.LaggedStart(*[m.FadeIn(cells[i], scale=0.6) for i in order], lag_ratio=0.006, run_time=2.4),
            m.AnimationGroup(m.FadeIn(g_big, shift=0.1 * m.UP), m.FadeIn(g_sub), m.FadeIn(g_sub2)),
        )
        self.say(
            "[The textbook way writes that grid out to memory], [and reads it back.]",
            m.AnimationGroup(m.FadeIn(hbm), m.FadeIn(hbm_t), m.GrowArrow(down), m.FadeIn(down_t),
                             m.Succession(m.Wait(0.5), pulse(down, MEMORY))),
            m.AnimationGroup(m.GrowArrow(up), m.FadeIn(up_t), m.Succession(m.Wait(0.4), pulse(up, MEMORY))),
        )
        self.say(
            "[But FlashAttention never stores it at all], [and still gets exactly the same answer.]",
            m.AnimationGroup(grid.animate.set_opacity(0.12), m.Create(ghost),
                             m.VGroup(down, up, down_t, up_t).animate.set_opacity(0.2), m.FadeIn(never, scale=0.9)),
            m.FadeIn(exact, shift=0.1 * m.UP),
        )
        self.say(
            "[In AI, Explained, episode thirty-seven sketched the idea.] [Today, we count the bytes.]",
            m.FadeIn(ep37, shift=0.1 * m.UP),
            m.FadeIn(ep15, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ two memories

    def two_memories(self):
        self.next_section("1 · Two memories", notes=f"Episode 2's two speed limits: math units and the memory that feeds them. H100 SXM (NVIDIA spec; FA3 paper Table 1, Shah et al. 2024): HBM (off-chip) 80 GB at 3.35 TB/s; shared memory (SRAM) 228 KiB per SM × {SMS} SMs = {SRAM_BYTES / 1e6:.1f} MB at 31 TB/s aggregate; L2 (not shown) 50 MiB at 12 TB/s. SRAM is {FASTER:.1f}× faster and {SMALLER:,.0f}× smaller (80 GB ÷ 30.8 MB). FlashAttention paper's A100 figures for comparison: HBM 40-80 GB at 1.5-2.0 TB/s, 192 KB × 108 SMs at ~19 TB/s. The package drawing is schematic (6 HBM stacks, 12 × 11 SMs).")
        die = card(3.85, 3.6, COMPUTE, CARD, radius=0.12, stroke=2.5).move_to([-2.85, -0.55, 0])
        sms = m.VGroup()
        srams = m.VGroup()
        for _ in range(SMS):
            sq = m.Square(0.24).set_stroke(COMPUTE, 1.2).set_fill(AMBER_FILL, 1)
            dot = m.Square(0.08).set_stroke(width=0).set_fill(MEMORY, 0.35).move_to(sq).shift(0.05 * m.UP + 0.05 * m.RIGHT)
            sms.add(m.VGroup(sq, dot))
            srams.add(dot)
        sms.arrange_in_grid(rows=11, cols=12, buff=0.052).move_to(die)
        stacks = m.VGroup(*[
            card(0.8, 1.02, MEMORY, CYAN_FILL, radius=0.08, stroke=2).move_to([x, -0.55 + dy, 0])
            for x in (-5.85, 0.15) for dy in (1.2, 0.0, -1.2)
        ])
        package = card(7.35, 4.35, EDGE, "#0F191D", radius=0.2, stroke=2).move_to([-2.85, -0.55, 0])
        hbm_l = code("HBM", 22, MEMORY, weight=m.BOLD).move_to([-5.85, 2.0, 0])
        hbm_r = code("HBM", 22, MEMORY, weight=m.BOLD).move_to([0.15, 2.0, 0])
        die_l = code(f"the chip · {SMS} SMs", 22, COMPUTE, weight=m.BOLD).move_to([-2.85, 2.0, 0])
        links = m.VGroup()
        for x0, x1 in ((-5.42, -4.8), (-0.27, -0.9)):
            for dy in (1.2, 0.0, -1.2):
                links.add(m.DoubleArrow([x0, -0.55 + dy, 0], [x1, -0.55 + dy, 0], buff=0, stroke_width=4, color=MEMORY,
                                        tip_length=0.12, max_tip_length_to_length_ratio=0.5))
        flows = [m.ShowPassingFlash(m.Line(a.get_start(), a.get_end()).set_stroke(MEMORY, 8), time_width=0.8, run_time=1.0) for a in links]

        rx = 1.75
        h_head = code("HBM · off the chip", 22, MEMORY, weight=m.BOLD).move_to([rx, 1.95, 0], aligned_edge=m.LEFT)
        h_big = title_text("80 GB", 60, INK)
        h_bw = code(f"{HBM_BW / 1e12:.2f} TB/s", 28, MEMORY)
        h_row = m.VGroup(h_big, h_bw).arrange(m.RIGHT, buff=0.4, aligned_edge=m.DOWN).move_to([rx, 1.2, 0], aligned_edge=m.LEFT)
        s_head = code("SRAM · on the chip", 22, MEMORY, weight=m.BOLD).move_to([rx, 0.25, 0], aligned_edge=m.LEFT)
        s_big = title_text(f"{SRAM_BYTES / 1e6:.0f} MB", 60, INK)
        s_bw = code(f"{SRAM_BW / 1e12:.0f} TB/s", 28, MEMORY)
        s_row = m.VGroup(s_big, s_bw).arrange(m.RIGHT, buff=0.4, aligned_edge=m.DOWN).move_to([rx, -0.5, 0], aligned_edge=m.LEFT)
        s_sub = code(f"228 KiB × {SMS} SMs", 20, MUTED).next_to(s_row, m.DOWN, buff=0.18).align_to(s_row, m.LEFT)
        faster = pill(f"≈ {FASTER:.0f}× faster", GOOD, GOOD_FILL, 22, mono=True)
        smaller = pill(f"≈ {round(SMALLER, -2):,.0f}× smaller", BAD, BAD_FILL, 22, mono=True)
        faster.move_to([rx, -1.75, 0], aligned_edge=m.LEFT)
        smaller.move_to([rx, -2.4, 0], aligned_edge=m.LEFT)
        src = cite("NVIDIA H100 SXM · Shah et al., 2024").move_to([rx, -3.05, 0], aligned_edge=m.LEFT)
        rule = pill("keep the work on the chip, touch HBM rarely", COMPUTE, AMBER_FILL, 24).move_to([-2.85, -3.2, 0])

        self.say(
            "[Episode two set up two speed limits]: [the math units, and the memory that feeds them.]",
            m.AnimationGroup(self.new_header("01 · two memories", "HBM and SRAM"), m.FadeIn(package), m.FadeIn(die), m.FadeIn(die_l),
                             m.LaggedStart(*[m.FadeIn(s, scale=0.5) for s in sms], lag_ratio=0.01, run_time=1.6)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(s, shift=0.1 * m.UP) for s in stacks], lag_ratio=0.12),
                             m.FadeIn(hbm_l), m.FadeIn(hbm_r), m.LaggedStart(*[m.GrowFromCenter(a) for a in links], lag_ratio=0.08)),
        )
        self.say(
            "[On an H100, the main memory is HBM]: [eighty gigabytes, feeding the chip at three point three five terabytes per second.]",
            m.AnimationGroup(m.Circumscribe(stacks, color=MEMORY, buff=0.08), m.FadeIn(h_head, shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.FadeIn(h_row, shift=0.1 * m.UP), m.LaggedStart(*flows, lag_ratio=0.1, run_time=1.8)),
        )
        self.say(
            "[Each of the chip's hundred and thirty-two processors keeps a small, fast scratchpad of its own]: [SRAM.]",
            m.LaggedStart(*[d.animate.set_fill(MEMORY, 1) for d in srams], lag_ratio=0.012, run_time=2.4),
            m.FadeIn(s_head, shift=0.1 * m.LEFT),
        )
        self.say(
            "[About thirty-one megabytes in all], [nine times faster than HBM], [but some twenty-six hundred times smaller.]",
            m.AnimationGroup(m.FadeIn(s_row, shift=0.1 * m.UP), m.FadeIn(s_sub)),
            m.AnimationGroup(m.FadeIn(faster, scale=0.9), m.FadeIn(src)),
            m.FadeIn(smaller, scale=0.9),
        )
        self.say(
            "[So the rule is simple]: [keep the work on the chip], [and touch HBM as little as you can.]",
            m.Circumscribe(die, color=COMPUTE, buff=0.08),
            m.FadeIn(rule, shift=0.1 * m.UP),
            m.LaggedStart(*[m.ShowPassingFlash(m.Line(a.get_start(), a.get_end()).set_stroke(MEMORY, 8), time_width=0.8, run_time=0.8)
                            for a in links[:2]], lag_ratio=0.3),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the textbook way

    def textbook(self):
        self.next_section("2 · The textbook way", notes=f"Standard attention (FlashAttention paper, Algorithm 0), one head of Owl-8B's prefill of the {N:,}-token chapter, BF16: Q, K, V are {N:,} × {D} = 1 MiB each. 1) read Q, K, compute S = QKᵀ, write S ({mb(GRID, 2)}) to HBM; 2) read S, softmax, write P; 3) read P and V, compute O = PV, write O. HBM traffic counter: {', '.join(mb(c) for c in COUNT)}; S and P each written once and read once = 4 × {mb(GRID, 2)} = {mb(GRID_TRAFFIC)}, against {mb(IO_HEAD)} for Q, K, V and O. Real implementations add more (masking, an FP32 softmax), so this is a lower bound. The 12 × 12 grid and the bars' proportions are schematic.")
        st = stage()
        bars = {k: whole_bar(k) for k in ("Q", "K", "V")}
        o_bar = whole_bar("O")
        dims = m.VGroup(code(f"{N:,} × {D} numbers", 20, MUTED), code("BF16 · 1 MiB each", 20, MUTED)).arrange(m.DOWN, buff=0.14).move_to(GRID_C)
        grid = mini_grid(12, GRID_S).move_to(GRID_C)
        cells = [c for row in grid for c in row]
        s_lab = code(f"S = QKᵀ · {N:,} × {N:,}", 20, COMPUTE).move_to([GRID_C[0], 1.0, 0])
        p_lab = code("P = softmax(S)", 20, COMPUTE).move_to([GRID_C[0], 1.0, 0])
        sp_lab = code(f"S, P: {mb(GRID, 2)} each", 20, INK, weight=m.BOLD).move_to([GRID_C[0], 1.0, 0])
        q_in, k_in, v_in = slot("Q"), slot("K"), slot("V")
        s_in = mini_grid(4, 0.75).move_to([SLOT_X["S"], SLOT_Y, 0])
        p_in = mini_grid(4, 0.75, COMPUTE, COMPUTE).move_to([SLOT_X["S"], SLOT_Y, 0])
        p_in2 = mini_grid(4, 0.75, COMPUTE, COMPUTE).move_to([SLOT_X["S"], SLOT_Y, 0])
        o_in = slot("O")

        def counter(nbytes):
            label = code("HBM traffic, one head:", 20, MUTED)
            value = code(mb(nbytes), 26, MEMORY, weight=m.BOLD)
            return m.VGroup(label, value).arrange(m.RIGHT, buff=0.22, aligned_edge=m.DOWN).move_to([6.5, 2.38, 0], aligned_edge=m.RIGHT)

        count = counter(0)

        self.say(
            "[Here's attention the textbook way, for one head.] [Queries, keys and values sit in HBM]: [four thousand rows of a hundred and twenty-eight numbers.]",
            m.AnimationGroup(self.new_header("02 · the textbook way", "The grid's round trip"), m.FadeIn(st["hbm"]),
                             m.FadeIn(st["chip"]), m.FadeIn(st["maths"]), m.FadeIn(st["road"]), m.FadeIn(count)),
            m.AnimationGroup(*[m.FadeIn(bars[k], shift=0.15 * m.UP) for k in bars], m.FadeIn(st["labels"][:3])),
            m.FadeIn(dims, shift=0.1 * m.UP),
        )
        self.say(
            "[Step one: read the queries and keys], [multiply them on the chip], [and write the score grid out to HBM.]",
            m.AnimationGroup(pulse(st["read"], MEMORY), m.FadeIn(q_in, target_position=bars["Q"]), m.FadeIn(k_in, target_position=bars["K"])),
            m.AnimationGroup(m.Indicate(st["maths_box"], color=COMPUTE, scale_factor=1.04), m.FadeIn(s_in, scale=0.6)),
            m.AnimationGroup(pulse(st["write"], MEMORY), m.FadeOut(dims), m.FadeIn(s_lab),
                             m.LaggedStart(*[m.FadeIn(c, shift=0.12 * m.LEFT) for c in cells], lag_ratio=0.008, run_time=1.6),
                             m.FadeOut(q_in), m.FadeOut(k_in), m.FadeOut(s_in), m.Transform(count, counter(COUNT[0]))),
        )
        self.say(
            "[Step two: read the grid back], [take each row's softmax], [and write the probabilities out again.]",
            m.AnimationGroup(pulse(st["read"], MEMORY), m.FadeIn(s_in, target_position=grid)),
            m.AnimationGroup(m.Indicate(st["maths_box"], color=COMPUTE, scale_factor=1.04), m.FadeIn(p_in)),
            m.AnimationGroup(pulse(st["write"], MEMORY), m.FadeOut(s_in), m.FadeOut(p_in),
                             m.LaggedStart(*[c.animate.set_fill(COMPUTE, 0.75) for c in cells], lag_ratio=0.006, run_time=1.4),
                             m.FadeOut(s_lab), m.FadeIn(p_lab), m.Transform(count, counter(COUNT[1]))),
        )
        self.say(
            "[Step three: read those back once more], [and multiply by the values to make the output.]",
            m.AnimationGroup(pulse(st["read"], MEMORY), m.FadeIn(p_in2, target_position=grid), m.FadeIn(v_in, target_position=bars["V"])),
            m.AnimationGroup(m.Indicate(st["maths_box"], color=COMPUTE, scale_factor=1.04), pulse(st["write"], MEMORY),
                             m.FadeIn(o_in, scale=0.6),
                             m.Succession(m.Wait(0.4), m.AnimationGroup(m.FadeIn(o_bar, target_position=o_in), m.FadeIn(st["labels"][3]))),
                             m.Transform(count, counter(COUNT[2]))),
        )
        tally = card(5.9, 3.4, MEMORY, CARD, radius=0.18, stroke=2).move_to([3.6, -0.75, 0])
        t_title = code("one head, one layer", 20, MUTED).align_to(tally, m.UL).shift(0.3 * m.RIGHT + 0.22 * m.DOWN)
        unit = 4.6 / GRID_TRAFFIC
        g_bar = rect(unit * GRID_TRAFFIC, 0.4, MEMORY, MEMORY, stroke=0).move_to([tally.get_left()[0] + 0.3, 0.05, 0], aligned_edge=m.LEFT)
        g_txt = m.VGroup(code("S and P, each written and read", 20, INK),
                         code(f"4 × {mb(GRID, 2)} = {mb(GRID_TRAFFIC)}", 22, MEMORY, weight=m.BOLD))
        g_txt.arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT).next_to(g_bar, m.DOWN, buff=0.15).align_to(g_bar, m.LEFT)
        i_bar = rect(max(unit * IO_HEAD, 0.05), 0.4, MEMORY, MEMORY, stroke=0).move_to([tally.get_left()[0] + 0.3, -1.55, 0], aligned_edge=m.LEFT)
        i_txt = m.VGroup(code("Q, K, V in · O out", 20, INK), code(f"4 × 1 MiB = {mb(IO_HEAD)}", 22, MEMORY, weight=m.BOLD))
        i_txt.arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT).next_to(i_bar, m.RIGHT, buff=0.25)
        g_ring = m.SurroundingRectangle(grid, buff=0.08).set_stroke(COMPUTE, 3)
        chip_side = m.VGroup(st["chip"], st["maths"], st["road"], p_in2, v_in, o_in)

        self.say(
            "[One grid is thirty-three and a half megabytes.] [Written and read twice, that's a hundred and thirty-four], [against four for the inputs and output.]",
            m.AnimationGroup(m.Create(g_ring), m.FadeOut(p_lab), m.FadeIn(sp_lab, scale=0.9)),
            m.AnimationGroup(m.FadeOut(chip_side), m.FadeIn(tally), m.FadeIn(t_title), m.GrowFromEdge(g_bar, m.LEFT), m.FadeIn(g_txt, shift=0.1 * m.UP)),
            m.AnimationGroup(m.GrowFromEdge(i_bar, m.LEFT), m.FadeIn(i_txt, shift=0.1 * m.LEFT)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ tiling

    def tiling(self):
        self.next_section("3 · Tiles through fast memory", notes="FlashAttention (Dao et al. 2022; loop order as in FlashAttention-2): Q, K, V are cut into tiles; one Q tile is loaded into SRAM and stays; K and V tiles stream through; each block of scores is computed on the chip, used to update a running output, and discarded; only the finished O tile is written back to HBM. The N × N matrix S never exists in HBM. The catch: softmax normalises over a whole row (its max and its sum), which the online softmax solves (next section). Four tiles per matrix are schematic (real tiles: e.g. 128 rows).")
        st = stage()
        packed = {k: bar_tiles(k) for k in ("Q", "K", "V")}
        spread = {k: bar_tiles(k, spread=0.12) for k in ("Q", "K", "V")}
        o_slots = m.VGroup(*[dashed_box(BAR_W, TILE_H, TOKEN, 10).move_to(t) for t in bar_tiles("O", spread=0.12)])
        o_done = bar_tiles("O", spread=0.12)
        q_in = slot("Q")
        kv_in = [(slot("K"), slot("V")) for _ in range(TILES)]
        score = mini_grid(4, 0.75).move_to([SLOT_X["S"], SLOT_Y, 0])
        acc = rect(BAR_W, TILE_H, TOKEN, GREEN_FILL).move_to([SLOT_X["O"], SLOT_Y, 0])
        acc_full = rect(BAR_W, TILE_H, TOKEN, TOKEN, opacity=0.85).move_to([SLOT_X["O"], SLOT_Y, 0])
        ghost = dashed_box(GRID_S, GRID_S, MUTED, 56).move_to(GRID_C)
        never = m.VGroup(code("S", 26, COMPUTE, weight=m.BOLD), code("never stored", 22, GOOD, weight=m.BOLD)).arrange(m.DOWN, buff=0.12).move_to(GRID_C + [0, 0.45, 0])
        row = mini_grid(12, GRID_S)[9].move_to(GRID_C + [0, -0.95, 0])
        formula = code("softmax(x) = exp(x - max) / sum", 24, INK, t2c={r"\bmax\b": COMPUTE, r"\bsum\b": COMPUTE})
        formula.move_to([-1.2, 2.35, 0])
        f_max, f_sum = glyphs(formula, "max"), glyphs(formula, "sum")

        self.say(
            "[So FlashAttention does the same math in a different order.] [First, cut everything into tiles.]",
            m.AnimationGroup(self.new_header("03 · tiling", "Tiles through fast memory"), m.FadeIn(st["hbm"]), m.FadeIn(st["chip"]),
                             m.FadeIn(st["maths"]), m.FadeIn(st["road"]), m.FadeIn(st["labels"]),
                             *[m.FadeIn(packed[k]) for k in packed]),
            m.AnimationGroup(*[t.animate.move_to(s) for k in packed for t, s in zip(packed[k], spread[k])],
                             m.Succession(m.Wait(0.3), m.FadeIn(o_slots))),
        )
        self.say(
            "[Load one tile of queries into SRAM], [and keep it there.]",
            m.AnimationGroup(pulse(st["read"], MEMORY), m.FadeIn(q_in, target_position=spread["Q"][0])),
            m.Circumscribe(m.VGroup(q_in, packed["Q"][0]), color=TOKEN, buff=0.08, run_time=1.2),
        )
        stream = []
        for j, (k_t, v_t) in enumerate(kv_in):
            glide = m.AnimationGroup(pulse(st["read"], MEMORY), m.FadeIn(k_t, target_position=spread["K"][j]),
                                     m.FadeIn(v_t, target_position=spread["V"][j]), run_time=0.55)
            stream.append(glide)
            if j < TILES - 1:
                stream.append(m.AnimationGroup(m.FadeOut(k_t, scale=0.7), m.FadeOut(v_t, scale=0.7), run_time=0.3))
        self.say(
            "[Then stream the keys and values through it, one tile at a time.]",
            m.Succession(*stream),
        )
        last_k, last_v = kv_in[-1]
        self.say(
            "[Each small block of scores is made on the chip], [used right away], [and thrown away.]",
            m.AnimationGroup(m.Indicate(st["maths_box"], color=COMPUTE, scale_factor=1.04), m.FadeIn(score, scale=0.5)),
            m.AnimationGroup(m.FadeIn(acc), m.FadeIn(acc_full, target_position=score)),
            m.AnimationGroup(m.FadeOut(score, scale=0.4), m.FadeOut(last_k, scale=0.7), m.FadeOut(last_v, scale=0.7)),
        )
        self.say(
            "[Only the finished output goes back to HBM.] [The full grid never exists anywhere.]",
            m.AnimationGroup(pulse(st["write"], MEMORY), m.FadeIn(o_done[0], target_position=acc_full),
                             m.FadeOut(acc_full), m.FadeOut(o_slots[0])),
            m.AnimationGroup(m.Create(ghost), m.FadeIn(never, scale=0.9)),
        )
        self.say(
            "[But there's a catch]: [softmax needs the whole row], [its maximum and its sum], [before a single weight is known.]",
            m.FadeIn(formula, shift=0.1 * m.DOWN),
            m.LaggedStart(*[m.FadeIn(c, scale=0.6) for c in row], lag_ratio=0.08),
            m.AnimationGroup(m.Circumscribe(f_max, color=COMPUTE, buff=0.06), m.Circumscribe(f_sum, color=COMPUTE, buff=0.06)),
            m.ShowPassingFlash(m.Line(row.get_left(), row.get_right()).set_stroke(COMPUTE, 8), time_width=0.6, run_time=1.2),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ online softmax

    def online(self):
        s0, s1 = STEPS
        self.next_section("4 · Online softmax", notes=f"notes §7 worked example (Milakov & Gimelshein 2018; FlashAttention §3.1), computed in this file by online_softmax(): scores x = [1, 2 | 3, 0] in two tiles. Tile 1: max 2, sum = e^-1 + e^0 = {f3(s0['new'])}. Tile 2: new max 3; the old sum is rescaled by exp(2 - 3) = {f3(s1['scale'])} → {f3(s1['old'])}; new terms e^0 + e^-3 = {f3(s1['new'])}; sum = {f3(SUM)}. Plain: e^-2 + e^-1 + e^0 + e^-3 = {f3(PLAIN.sum())} (asserted equal to 1e-12). Softmax = [{', '.join(f3(p) for p in SOFT)}]. The running output is rescaled by the same factor: with values v = [10, 20, 30, 40] (illustrative: one coordinate of each value vector), out = {STEPS[0]['out_new']:.2f} after tile 1, then × {f3(s1['scale'])} + {s1['out_new']:.2f} = {ACC:.2f}; ÷ {f3(SUM)} = {OUT:.2f} = softmax · v = {PLAIN_OUT:.2f}.")
        xs = [-5.15, -4.0, -2.45, -1.3]
        chips = m.VGroup(*[chip(str(int(v)), COMPUTE, AMBER_FILL, width=0.95, height=0.8, size=32, mono=True).move_to([x, 1.45, 0])
                           for v, x in zip(X, xs)])
        x_lab = code("x =", 24, MUTED).move_to([-6.25, 1.45, 0])
        vals = m.VGroup(*[code(str(int(v)), 24, MUTED).move_to([x, 0.62, 0]) for v, x in zip(V, xs)])
        v_lab = code("v =", 24, MUTED).move_to([-6.25, 0.62, 0])
        v_note = cite("values, for example").move_to([-3.2, 0.12, 0])

        def bracket(a, b, label):
            y = 2.05
            line = m.VGroup(m.Line([a, y, 0], [b, y, 0]), m.Line([a, y, 0], [a, y - 0.12, 0]), m.Line([b, y, 0], [b, y - 0.12, 0]))
            line.set_stroke(MEMORY, 2.5)
            return m.VGroup(line, code(label, 20, MEMORY, weight=m.BOLD).move_to([(a + b) / 2, y + 0.25, 0]))

        t1 = bracket(xs[0] - 0.47, xs[1] + 0.47, "tile 1")
        t2 = bracket(xs[2] - 0.47, xs[3] + 0.47, "tile 2")

        # the running state, on the chip
        box = card(5.0, 2.55, COMPUTE, CARD, radius=0.18, stroke=2).move_to([4.0, 1.0, 0])
        b_title = code("running state, on the chip", 20, MUTED).align_to(box, m.UL).shift(0.3 * m.RIGHT + 0.2 * m.DOWN)
        rows_y = [1.45, 0.82, 0.19]
        r_labs = m.VGroup(*[code(s, 22, MUTED).move_to([1.9, y, 0], aligned_edge=m.LEFT) for s, y in zip(("max", "sum", "out"), rows_y)])
        vx = 2.75

        def val(s, y, color=INK):
            return code(s, 26, color, weight=m.BOLD).move_to([vx, y, 0], aligned_edge=m.LEFT)

        bx, bw = 4.0, 2.2 / SUM

        def sum_bar(v):
            return rect(max(bw * v, 0.02), 0.3, COMPUTE, COMPUTE, stroke=0).move_to([bx, rows_y[1], 0], aligned_edge=m.LEFT)

        m_val = val("-inf", rows_y[0], MUTED)
        m2, m3 = val("2", rows_y[0]), val("3", rows_y[0])
        s_val = val("0", rows_y[1], MUTED)
        s1v, s2v, s3v = val(f3(s0["new"]), rows_y[1], COMPUTE), val(f3(s1["old"]), rows_y[1], COMPUTE), val(f3(SUM), rows_y[1], COMPUTE)
        o_val = val("0", rows_y[2], MUTED)
        o1v, o2v = val(f"{s0['out_new']:.2f}", rows_y[2], TOKEN), val(f"{ACC:.2f}", rows_y[2], TOKEN)
        bar = sum_bar(0)
        bar1, bar2, bar3 = sum_bar(s0["new"]), sum_bar(s1["old"]), sum_bar(SUM)

        # the log, one line per step
        lx, ex, rx = -6.45, -5.0, 2.95
        ys = [-0.75, -1.3, -1.85, -2.5, -3.1]

        def log(i, label, expr, result, color=INK, label_color=MEMORY, indent=0.0):
            parts = [code(label, 22, label_color, weight=m.BOLD).move_to([lx, ys[i], 0], aligned_edge=m.LEFT)] if label else []
            parts.append(code(expr, 22, color).move_to([ex + indent, ys[i], 0], aligned_edge=m.LEFT))
            res = code(result, 22, COMPUTE, weight=m.BOLD, t2c={"✓": GOOD}).move_to([rx, ys[i], 0], aligned_edge=m.LEFT)
            return m.VGroup(*parts), res

        l1, r1 = log(0, "tile 1", "max 2 · sum = exp(1-2) + exp(2-2)", f"= {f3(s0['new'])}")
        l2, r2 = log(1, "tile 2", f"max 3 · old sum {f3(s0['new'])} × exp(2-3)", f"= {f3(s1['old'])}")
        l3, r3 = log(2, "", "+ exp(3-3) + exp(0-3)", f"+ {f3(s1['new'])} = {f3(SUM)}", indent=1.6)
        l4, r4 = log(3, "plain", "exp(1-3) + exp(2-3) + exp(3-3) + exp(0-3)", f"= {f3(PLAIN.sum())} ✓", label_color=GOOD)
        l5, r5 = log(4, "output", f"online {ACC:.2f} / {f3(SUM)} = plain softmax · v", f"= {OUT:.2f} ✓", label_color=TOKEN)
        verdict = pill("✓ exact", GOOD, GOOD_FILL, 24).move_to([5.45, 0.19, 0])

        self.say(
            "[The fix is an online softmax.] [Take one row of four scores]: [one, two, three and zero], [in two tiles.]",
            m.AnimationGroup(self.new_header("04 · online softmax", "A running max and sum"), m.FadeIn(box), m.FadeIn(b_title),
                             m.FadeIn(r_labs), m.FadeIn(m_val), m.FadeIn(s_val), m.FadeIn(o_val)),
            m.FadeIn(x_lab),
            m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.DOWN) for c in chips], lag_ratio=0.3),
            m.AnimationGroup(m.FadeIn(t1, shift=0.1 * m.DOWN), m.FadeIn(t2, shift=0.1 * m.DOWN), m.FadeIn(v_lab), m.FadeIn(vals), m.FadeIn(v_note)),
        )
        self.say(
            "[Tile one: the running max is two], [and measured from it, the running sum is one point three six eight.]",
            m.AnimationGroup(m.Circumscribe(m.VGroup(chips[0], chips[1]), color=MEMORY, buff=0.08), m.Transform(m_val, m2), m.FadeIn(l1[0])),
            m.AnimationGroup(m.FadeIn(l1[1]), m.FadeIn(r1), m.Transform(s_val, s1v), m.FadeIn(bar), m.Transform(bar, bar1)),
        )
        self.say(
            "[Tile two brings a three]: [a new maximum.]",
            m.AnimationGroup(m.Circumscribe(m.VGroup(chips[2], chips[3]), color=MEMORY, buff=0.08), m.FadeIn(l2[0])),
            m.AnimationGroup(m.Circumscribe(chips[2], color=COMPUTE, buff=0.06), m.Transform(m_val, m3)),
        )
        self.say(
            "[The old sum was measured from the old max], [so it's rescaled down], [to zero point five zero three.]",
            m.Circumscribe(m.VGroup(r_labs[1], s_val, bar), color=COMPUTE, buff=0.1),
            m.AnimationGroup(m.FadeIn(l2[1]), m.Transform(bar, bar2)),
            m.AnimationGroup(m.FadeIn(r2), m.Transform(s_val, s2v)),
        )
        self.say(
            "[Add the new tile's terms], [and the running sum is one point five five three.]",
            m.FadeIn(l3[0]),
            m.AnimationGroup(m.FadeIn(r3), m.Transform(s_val, s3v), m.Transform(bar, bar3)),
        )
        self.say(
            "[Now the plain way, all four scores at once]: [one point five five three.] [Exactly the same.]",
            m.FadeIn(l4),
            m.FadeIn(r4),
            m.AnimationGroup(m.Circumscribe(r3, color=GOOD, buff=0.08), m.Circumscribe(r4, color=GOOD, buff=0.08)),
        )
        self.say(
            "[The running output is rescaled by the same factor], [so the weighted values agree too], [down to rounding.]",
            m.Succession(m.Transform(o_val, o1v, run_time=0.6), m.Wait(0.3), m.Transform(o_val, o2v, run_time=0.6)),
            m.FadeIn(l5),
            m.FadeIn(r5),
        )
        self.say(
            "[So FlashAttention is no approximation]: [it's exact attention, in a different order.]",
            m.FadeIn(verdict, scale=0.85),
            m.Circumscribe(m.VGroup(t1, t2, chips), color=GOOD, buff=0.12),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the bytes, counted

    def counted(self):
        self.next_section("5 · The bytes, counted", notes=f"Owl-8B reading the {N:,}-token chapter on one H100, attention only, BF16. Textbook: per layer, 32 query heads × {mb(GRID_TRAFFIC)} of S/P traffic + Q, K, V, O ({(2 * H + 2 * HKV)} × 1 MiB with 8 KV heads) = {(H * GRID_TRAFFIC + IO_LAYER) / 1e9:.2f} GB; × 32 layers = {STD_BYTES / 1e9:.1f} GB = {STD_MS:.1f} ms at 3.35 TB/s. FlashAttention, counting each input read once and the output written once: {FLASH_BYTES / 1e9:.2f} GB = {FLASH_MS:.2f} ms. Attention math over the full grid (QKᵀ and PV): {ATT_FLOPS / 1e12:.1f} TFLOP = {MATH_MS:.1f} ms at 989 TFLOPS. A step takes the longer of the two (Episode 2). Real kernels re-read K/V tiles once per query tile (largely from the 50 MB L2), so the measured reduction is smaller: FlashAttention paper Fig. 2 (GPT-2 medium, seq. 1024, A100, fwd + bwd): 66.6 → 75.2 GFLOPs, 40.3 → 4.4 GB HBM R/W, 41.7 → 7.3 ms. Measured on this Mac (research/measurements.md §5, mlx-lm 0.32.2, 2026-10-24): Owl-8B 4-bit at 32K tokens peaks at {PEAK_FUSED:.2f} GB with the fused attention kernel (BF16 cache) and {PEAK_UNFUSED:.2f} GB with the 8-bit cache, whose attention path is not fused and materializes the chunk × context score matrix (the 8-bit cache itself is smaller).")
        ctx = code(f"Owl-8B · {N:,}-token prefill · {H} heads × {L} layers · BF16 · one H100", 20, MUTED).move_to([0, 2.3, 0])
        mascot = owl(0.45).move_to([6.05, 3.15, 0])
        x0, per_ms = -2.85, 9.2 / STD_MS
        rows = {"std": (1.35, 0.8), "flash": (-0.35, -0.9)}

        def tbar(ms, y, color, fill):
            return rect(max(per_ms * ms, 0.05), 0.4, color, fill, stroke=2).move_to([x0, y, 0], aligned_edge=m.LEFT)

        std_lab = m.VGroup(title_text("Textbook", 34, INK), code("S and P in HBM", 20, MUTED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        std_lab.move_to([-6.45, sum(rows["std"]) / 2, 0], aligned_edge=m.LEFT)
        fl_lab = m.VGroup(title_text("FlashAttention", 34, GOOD), code("tiles in SRAM", 20, MUTED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        fl_lab.move_to([-6.45, sum(rows["flash"]) / 2, 0], aligned_edge=m.LEFT)
        std_mem = tbar(STD_MS, rows["std"][0], MEMORY, CYAN_FILL)
        std_mem_t = code(f"HBM traffic {STD_BYTES / 1e9:.0f} GB · {STD_MS:.1f} ms", 22, INK).move_to(std_mem).align_to(std_mem, m.LEFT).shift(0.2 * m.RIGHT)
        std_math = tbar(MATH_MS, rows["std"][1], COMPUTE, AMBER_FILL)
        std_math_t = code(f"math {MATH_MS:.1f} ms", 22, COMPUTE).next_to(std_math, m.RIGHT, buff=0.2)
        fl_mem = tbar(FLASH_MS, rows["flash"][0], MEMORY, CYAN_FILL)
        fl_mem_t = code(f"HBM traffic {FLASH_BYTES / 1e9:.1f} GB · {FLASH_MS:.1f} ms", 22, MEMORY).next_to(fl_mem, m.RIGHT, buff=0.2)
        fl_math = tbar(MATH_MS, rows["flash"][1], COMPUTE, AMBER_FILL)
        fl_math_t = code(f"math {MATH_MS:.1f} ms", 22, COMPUTE).next_to(fl_math, m.RIGHT, buff=0.2)
        pace = code("longer bar sets the time", 20, MUTED).move_to([1.9, rows["flash"][1], 0], aligned_edge=m.LEFT)
        bound = pill("memory-bound", MEMORY, CYAN_FILL, 20).next_to(std_math_t, m.RIGHT, buff=0.5)

        # the paper's measurement and this Mac
        paper = card(5.7, 2.0, EDGE, CARD, radius=0.16, stroke=2).move_to([-3.6, -2.45, 0])
        p_title = cite("Dao et al., 2022 · GPT-2 medium · A100").align_to(paper, m.UL).shift(0.3 * m.RIGHT + 0.18 * m.DOWN)
        p_rows = m.VGroup(
            code(f"math     {PAPER['gflops'][0]} → {PAPER['gflops'][1]} GFLOPs", 22, COMPUTE),
            code(f"HBM      {PAPER['hbm_gb'][0]} → {PAPER['hbm_gb'][1]} GB", 22, MEMORY),
            code(f"time     {PAPER['ms'][0]} → {PAPER['ms'][1]} ms", 22, GOOD),
        ).arrange(m.DOWN, buff=0.16, aligned_edge=m.LEFT).next_to(p_title, m.DOWN, buff=0.2).align_to(p_title, m.LEFT)
        mac = card(6.85, 2.0, EDGE, CARD, radius=0.16, stroke=2).move_to([3.05, -2.45, 0])
        m_title = mac_stamp().align_to(mac, m.UL).shift(0.25 * m.RIGHT + 0.12 * m.DOWN)
        unit = 2.2 / 16

        def peak_row(label, gb, y, color):
            lab = code(label, 20, INK).move_to([mac.get_left()[0] + 0.3, y, 0], aligned_edge=m.LEFT)
            b = rect(unit * gb, 0.3, color, color, stroke=0).move_to([mac.get_left()[0] + 2.65, y, 0], aligned_edge=m.LEFT)
            val_ = code(f"{gb:.1f} GB", 20, color, weight=m.BOLD).next_to(b, m.RIGHT, buff=0.15)
            return m.VGroup(lab, b, val_)

        fused = peak_row("fused kernel", PEAK_FUSED, -2.25, MEMORY)
        unfused = peak_row("unfused path", PEAK_UNFUSED, -2.75, COMPUTE)
        m_sub = cite(f"Owl-8B 4-bit · {CTX // 1024}K tokens · peak memory").move_to([mac.get_x(), -3.2, 0])

        self.say(
            "[Now scale up to Owl's whole chapter]: [thirty-two heads, in each of thirty-two layers.]",
            m.AnimationGroup(self.new_header("05 · the bytes", "Owl's prefill, counted"), m.FadeIn(mascot, shift=0.1 * m.UP)),
            m.FadeIn(ctx, shift=0.1 * m.UP),
        )
        self.say(
            "[The textbook way moves a hundred and forty gigabytes]: [forty-two milliseconds of traffic], [for nine milliseconds of math.]",
            m.FadeIn(std_lab, shift=0.1 * m.RIGHT),
            m.AnimationGroup(m.GrowFromEdge(std_mem, m.LEFT, run_time=1.4), m.FadeIn(std_mem_t)),
            m.AnimationGroup(m.GrowFromEdge(std_math, m.LEFT), m.FadeIn(std_math_t), m.Succession(m.Wait(0.5), m.FadeIn(bound, scale=0.9))),
        )
        self.say(
            "[With FlashAttention, it's under three gigabytes], [less than a millisecond], [so now the math sets the pace.]",
            m.AnimationGroup(m.FadeIn(fl_lab, shift=0.1 * m.RIGHT), m.GrowFromEdge(fl_mem, m.LEFT), m.FadeIn(fl_mem_t)),
            m.Circumscribe(m.VGroup(fl_mem, fl_mem_t), color=MEMORY, buff=0.08),
            m.AnimationGroup(m.GrowFromEdge(fl_math, m.LEFT), m.FadeIn(fl_math_t), m.FadeIn(pace)),
        )
        self.say(
            "[Real kernels re-read some tiles. The paper measured GPT-2]: [a little more math], [nine times fewer bytes], [and almost six times faster.]",
            m.AnimationGroup(m.FadeIn(paper), m.FadeIn(p_title)),
            m.FadeIn(p_rows[0], shift=0.1 * m.UP),
            m.FadeIn(p_rows[1], shift=0.1 * m.UP),
            m.FadeIn(p_rows[2], shift=0.1 * m.UP),
        )
        self.say(
            "[On this Mac, it shows up in memory]: [an unfused attention path in MLX builds the grid], [and peaks at sixteen gigabytes, not nine and a half.]",
            m.AnimationGroup(m.FadeIn(mac), m.FadeIn(m_title), m.FadeIn(m_sub)),
            m.AnimationGroup(m.FadeIn(unfused[0]), m.GrowFromEdge(unfused[1], m.LEFT), m.FadeIn(unfused[2])),
            m.AnimationGroup(m.FadeIn(fused[0]), m.GrowFromEdge(fused[1], m.LEFT), m.FadeIn(fused[2])),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ FA2 and FA3

    def versions(self):
        self.next_section("6 · FlashAttention-2 and -3", notes="Share of the GPU's peak 16-bit FLOPs (dense). FlashAttention (2022): 25-40% on A100 (as quoted by the FA2 paper). FlashAttention-2 (Dao 2023, arXiv 2307.08691): better parallelism over the sequence and work split across warps, 'around 2× speedup', 50-73% on A100; only 35% on H100 (FA3 paper). FlashAttention-3 (Shah et al. 2024, arXiv 2407.08608): warp specialisation with asynchronous TMA loads, pingpong overlap of softmax and matmul, FP8 with block quantisation; FP16 up to 740 TFLOPS (75% of 989), FP8 close to 1.2 PFLOPS. It helps prefill and training (many queries); decode (one query per sequence) needs split-KV kernels: Flash-Decoding, Episode 16 (notes §7).")
        x0, w100 = -2.05, 6.2
        ys = [1.75, 0.85, -0.05, -0.95]
        rows = m.VGroup()
        for (name, gpu, year, lo, hi), y in zip(VERSIONS, ys):
            lab = m.VGroup(text(name, 24, INK, weight=m.BOLD), code(f"{gpu} · {year}", 20, MUTED)).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
            lab.move_to([-6.45, y, 0], aligned_edge=m.LEFT)
            track = rect(w100, 0.42, EDGE, CARD, stroke=1.5).move_to([x0, y, 0], aligned_edge=m.LEFT)
            solid = rect(w100 * lo / 100, 0.42, COMPUTE, COMPUTE, stroke=0).move_to([x0, y, 0], aligned_edge=m.LEFT)
            parts = [lab, track, solid]
            if hi > lo:
                parts.append(rect(w100 * (hi - lo) / 100, 0.42, COMPUTE, COMPUTE, stroke=0, opacity=0.4)
                             .move_to([x0 + w100 * lo / 100, y, 0], aligned_edge=m.LEFT))
            share = f"{lo}–{hi}%" if hi > lo else f"{lo}%"
            parts.append(code(share, 22, COMPUTE, weight=m.BOLD).move_to([x0 + w100 * hi / 100 + 0.2, y, 0], aligned_edge=m.LEFT))
            rows.add(m.VGroup(*parts))
        ticks = m.VGroup()
        for p in (0, 25, 50, 75, 100):
            x = x0 + w100 * p / 100
            ticks.add(m.Line([x, -1.35, 0], [x, -1.45, 0]).set_stroke(FAINT, 2), code(f"{p}%", 20, MUTED).move_to([x, -1.7, 0]))
        axis = m.Line([x0, -1.4, 0], [x0 + w100, -1.4, 0]).set_stroke(FAINT, 2)
        a_title = cite("share of the GPU's peak 16-bit math").move_to([x0 + w100 / 2, -2.1, 0])
        tflops = code(f"{FA3_TFLOPS} TFLOPS · FP16", 22, BG, weight=m.BOLD).move_to(rows[3][2])
        fp8 = m.VGroup(code("FP8:", 20, COMPUTE), code(f"≈ {FA3_FP8_PFLOPS} PFLOPS", 20, COMPUTE)).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
        fp8.move_to([x0 + w100 + 0.3, ys[3], 0], aligned_edge=m.LEFT)
        twice = code("≈ 2× faster", 20, GOOD).move_to([x0 + w100 + 0.3, ys[1], 0], aligned_edge=m.LEFT)
        src = cite("Dao 2023 · Shah et al., 2024").move_to([4.9, 2.4, 0])
        tricks = m.VGroup(code("FA3: softmax ∥ matmul", 20, INK, t2c={"FA3:": MUTED}), code("· async tile loads", 20, INK)).arrange(m.RIGHT, buff=0.25)
        tricks.move_to([x0 + w100 / 2 - 0.3, -2.6, 0])
        where = m.VGroup(pill("prefill · training ✓", GOOD, GOOD_FILL, 22),
                         pill("decode: one query → Flash-Decoding, Episode 16", MUTED, CARD, 22)).arrange(m.RIGHT, buff=0.4).move_to([0, -3.2, 0])

        self.say(
            "[The kernel kept improving.] [Version one used a quarter to two fifths of an A100's math.]",
            m.AnimationGroup(self.new_header("06 · versions", "FlashAttention-2 and -3"), m.FadeIn(axis), m.FadeIn(ticks), m.FadeIn(a_title), m.FadeIn(src)),
            m.AnimationGroup(m.FadeIn(rows[0][0]), m.FadeIn(rows[0][1]), m.GrowFromEdge(rows[0][2], m.LEFT), m.FadeIn(rows[0][3]), m.FadeIn(rows[0][4])),
        )
        self.say(
            "[Version two split the work better across the chip]: [about twice as fast], [up to seventy-three percent.]",
            m.AnimationGroup(m.FadeIn(rows[1][0]), m.FadeIn(rows[1][1])),
            m.AnimationGroup(m.GrowFromEdge(rows[1][2], m.LEFT), m.FadeIn(twice)),
            m.AnimationGroup(m.FadeIn(rows[1][3]), m.FadeIn(rows[1][4])),
        )
        self.say(
            "[On the H100, version two reached only thirty-five percent.] [Version three overlaps the softmax with the matrix math], [and loads tiles while it computes.]",
            m.AnimationGroup(m.FadeIn(rows[2][0]), m.FadeIn(rows[2][1]), m.GrowFromEdge(rows[2][2], m.LEFT), m.FadeIn(rows[2][3])),
            m.AnimationGroup(m.FadeIn(rows[3][0]), m.FadeIn(rows[3][1]), m.FadeIn(tricks[0], shift=0.1 * m.UP)),
            m.FadeIn(tricks[1], shift=0.1 * m.UP),
        )
        self.say(
            "[It reaches seven hundred and forty teraflops in sixteen-bit], [three quarters of the chip], [and close to one point two petaflops in FP8.]",
            m.AnimationGroup(m.GrowFromEdge(rows[3][2], m.LEFT, run_time=1.2), m.FadeIn(tflops)),
            m.FadeIn(rows[3][3], scale=0.8),
            m.FadeIn(fp8, shift=0.1 * m.UP),
        )
        self.say(
            "[It shines in prefill and training, where many queries meet many keys.] [Decode, one query at a time, needs another split]: [episode sixteen.]",
            m.FadeIn(where[0], shift=0.1 * m.UP),
            m.FadeIn(where[1], shift=0.1 * m.UP),
            m.Circumscribe(where[1], color=MUTED, buff=0.06),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes=f"Tile (Q, K, V through SRAM; the N × N grid never stored), rescale (a running max and sum, notes §7: 1.368 → 0.503 → 1.553), exact (the same softmax and output as plain attention). Owl's prefill: {STD_BYTES / 1e9:.0f} GB of HBM traffic → {FLASH_BYTES / 1e9:.1f} GB. Next: Episode 16, the CPU tax (kernel launches, CUDA graphs, Flash-Decoding).")
        words = [("Tile", MEMORY, "Q, K, V through SRAM"),
                 ("Rescale", COMPUTE, "a running max and sum"),
                 ("Exact", GOOD, "same answer, fewer bytes")]
        cols = m.VGroup()
        for w, color, sub in words:
            cols.add(m.VGroup(title_text(w, 72, color), text(sub, 22, MUTED)).arrange(m.DOWN, buff=0.25))
        cols.arrange(m.RIGHT, buff=1.3).move_to([0, 0.75, 0])
        base = 0.85
        for c in cols:
            c[0].shift((base - c[0][0].get_bottom()[1]) * m.UP)  # first letters (T, R, E) sit on the baseline
            c[1].move_to([c[0].get_x(), base - 0.55, 0])
        mid = base + 0.3
        arrows = m.VGroup(*[
            arrow([a[0].get_right()[0] + 0.15, mid, 0], [b[0].get_left()[0] - 0.15, mid, 0], MUTED) for a, b in zip(cols, cols[1:])
        ])
        tagline = text("The score grid never leaves the chip.", 32, INK).move_to([0, -1.35, 0])
        mascot = owl(0.75).move_to([0, -2.7, 0])

        self.say(
            "[So: tile], [rescale], [and stay exact.]",
            m.AnimationGroup(self.new_header("recap", "FlashAttention in three words"), m.FadeIn(cols[0], shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowArrow(arrows[0]), m.FadeIn(cols[1], shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowArrow(arrows[1]), m.FadeIn(cols[2], shift=0.2 * m.UP)),
        )
        self.say(
            "[Same math, better order]: [the score grid never leaves the chip], [and attention runs at the speed of math, not memory.]",
            m.Circumscribe(cols, color=MUTED, buff=0.2),
            m.FadeIn(tagline, shift=0.1 * m.UP),
            m.FadeIn(mascot, shift=0.2 * m.UP),
        )
        self.wait(0.4)
        self.clear()
