"""Tokens per Second · Episode 9 · The Cost of a Long Memory (Module 2 · The KV Cache)

Long contexts cost quadratically in prefill and linearly in decode. Running example: a student pastes a whole
textbook into Owl, 128K tokens (131,072: Llama 3.1's entire context), with Owl-8B (Llama-3.1-8B: 8 B parameters,
16 GB of BF16 weights, 128 KiB of KV cache per token). On one H100 the first token takes 13.4 s; every later step
reads 16 GB of weights plus 16 GiB of cache. The decode slowdown is measured on this Mac (up to 32K tokens), and
the rotating-cache remedy is measured with Owl-mini (Llama-3.2-1B 4-bit), as the measurement agent did.

Sections: hook (the textbook fills the context: 13.4 s to the first token, 9.9 vs 4.9 ms per token) · prefill (FLOPs
vs prompt length: a straight line for the weights plus a parabola for attention; attention share 1.6% → 35% → 68%;
the Mac's prefill speed falls 1,005 → 801 tok/s from 2K to 16K, the attention model predicts 798) · decode (bytes
per step = weights + the whole cache; H100 step 4.9 → 9.9 ms) · measured on this Mac (ms per token vs context: a
straight line, +0.29 ms per 1,000 tokens; 30.8 → 23.8 tok/s at 89–91% of the ceiling; 4-bit 102 → 51 tok/s;
Flash-Decoding pointer) · sinks + window (why window attention breaks, attention sinks, StreamingLLM; measured:
flat memory and speed, and the forgotten code word) · offload (GPU → CPU → disk; reload 17.2 GB at ~50 GB/s =
0.34 s vs 13.4 s of prefill; vLLM 2–22× TTFT; DeepSeek 56.3% disk-cache hits) · the trade-off · recap.

Sources (facts dated October 2026):
- research/notes.md §11: prefill FLOPs ≈ 2·P·n + 2·L·n²·d (causal attention, QKᵀ and AV); worked table for
  Llama-3.1-8B on one H100 at 50% MFU (attention share 1.6% / 12% / 34% / 68% at 1K / 8K / 32K / 128K; 13.4 s at
  128K; decode floor 9.9 ms at 128K vs 4.9 ms at 2K). Recomputed here with owl_reference (flops_per_token, GPU,
  step_time) and asserted; the 1K TTFT computes to 33.8 ms (shown "34 ms"; the notes' table says 33).
- StreamingLLM (Xiao et al., ICLR 2024, arXiv 2309.17453; abstract re-read 2026-10-24): window attention "fails
  when the text length surpasses the cache size"; "strong attention scores towards initial tokens as a 'sink' even
  if they are not semantically important"; cause: softmax "requires attention scores to sum up to one"; keeping the
  KV of 4 initial tokens "will largely recover the performance of window attention"; stable "with up to 4 million
  tokens and more" (notes §11).
- vLLM blog "KV offloading connector" (2026-01-08, Llama-3.1-8B-Instruct on an H100; re-read 2026-10-24): "loading
  KV values from the CPU reduces TTFT by X2-X22, depending on the prompt size"; CPU↔GPU "for single-direction both
  can get up to about 50GB/s" (notes §3 used 41.7 GB/s = half of the 83.4 GB/s bidirectional figure; the primary
  source's one-way figure is used here: 17.18 GB ÷ 50 GB/s = 0.34 s; with 41.7 GB/s it would be 0.41 s).
- DeepSeek, inference system overview (Feb 2025, 24 h): "342B tokens (56.3%) hit the on-disk KV cache" (notes §3).
- Flash-Decoding (Dao, Haziza, Massa, Sizov, Oct 2023): splits the keys/values into chunks processed in parallel;
  "up to 8x faster generation for very long sequences" (notes §7). Episode 16 covers it.
- H100 SXM: 80 GB HBM3, 3.35 TB/s, 989 TFLOPS BF16 dense (NVIDIA, dated Oct 2026; owl_reference HARDWARE).
- research/measurements.md (this Mac Studio, M4 Max 546 GB/s, mlx-lm 0.32.2, 2026-10-24), via measured():
  §2 prefill speed per prompt length (8B BF16: 1,005 tok/s at 2,048, 801 at 16,384; FLOPs per prompt token =
  2 × 6.98 B block parameters + 2·L·d·T); §3 decode vs context (8B BF16 30.8 → 23.8 tok/s and 4-bit 101.9 → 51.0
  from 128 to 32,768 tokens); §6 rotating cache (Llama-3.2-1B 4-bit, RotatingKVCache(max_size=1024, keep=4) built
  explicitly because --max-kv-size is silently ignored for Llama models in mlx-lm 0.32.2: KV pinned at 32 MiB and
  ≈434 tok/s, vs 264 MiB and 285 tok/s for the full cache after 8,192 tokens; the PELICAN code word, forgotten).
- Episodes 3 (TTFT growing with prompt length on this Mac), 5 (the KV memory slope), 6 (the rolling buffer and the
  forgotten code word), 8 (prefix caching), 16 (Flash-Decoding) are referenced, not repeated.

Illustrative (said "for example" or schematic): the 4- and 8-token pair triangles, the 24-square token strip and
its attention weights, the block counts in the memory hierarchy, the other students' blocks, the trade-off cards'
mini drawings. The textbook is a stand-in for any 128K-token prompt.

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

PREFIX = "ep09"
KIB, MIB, GIB = 2**10, 2**20, 2**30
KV_TOKEN = int(kv_bytes_per_token())
W_BYTES = weight_bytes()
BOOK = 128 * 1024  # the textbook fills Llama 3.1's whole context: 131,072 tokens
BOOK_KV = BOOK * KV_TOKEN
MFU = 0.5  # prefill at half the H100's peak (notes §11)
assert KV_TOKEN == 128 * KIB and BOOK == 131_072
assert BOOK_KV == 16 * GIB and 1.0 < BOOK_KV / W_BYTES < 1.1  # "as big as the weights themselves"
D_ATT = OWL["heads"] * OWL["head_dim"]
assert D_ATT == OWL["d_model"] == 4096


def attention_flops(n, model=OWL):
    """Causal attention in prefill (notes §11): QKᵀ and AV over every (query, earlier key) pair, every layer."""
    return 2 * model["layers"] * n**2 * model["heads"] * model["head_dim"]


def weight_flops(n, model=OWL):
    """Every prompt token through every weight: 2 FLOPs per parameter per token."""
    return flops_per_token(model) * n


def prefill_flops(n):
    return weight_flops(n) + attention_flops(n)


def ttft(n, hw=GPU):
    return prefill_flops(n) / (hw["peak"] * MFU)


def share(n):
    return attention_flops(n) / prefill_flops(n)


assert abs(weight_flops(BOOK) / (GPU["peak"] * MFU) - prefill_time(BOOK)) < 1e-9  # owl_reference ignores attention
STOPS = (1024, 32768, BOOK)
assert share(1024) < 0.02 and round(100 * share(1024), 1) == 1.6  # "under two percent"
assert 0.33 < share(32768) < 0.36  # "about a third"
assert round(100 * share(BOOK)) == 68 and 0.66 < share(BOOK) < 0.7  # "two thirds"
assert round(ttft(BOOK), 1) == 13.4 and round(ttft(BOOK)) == 13  # notes §11: 13.4 s; "about thirteen seconds"
assert round(prefill_flops(BOOK) / 1e15, 1) == 6.6
assert round(ttft(1024) * 1000) == 34 and round(ttft(32768), 1) == 1.6


def pairs(n):
    return n * (n + 1) // 2  # each token with itself and every earlier token


assert pairs(4) == 10 and pairs(8) == 36  # "twice the tokens, about four times the pairs"

# prefill on this Mac: speed per prompt token falls as attention grows (measurements.md §2)
BLOCK_PARAMS = measured("models", "llama-3.1-8b", "bf16", "params_matmul_excl_embed_head")
PRE = measured("prefill", "llama-3.1-8b", "bf16", "by_prompt_tokens")


def mac_flops_per_prompt_token(n):
    """measurements.md §2: 2 × block parameters (no embedding lookup, LM head for the last token only) + attention."""
    return 2 * BLOCK_PARAMS + 2 * OWL["layers"] * D_ATT * n


PRE_SHORT, PRE_LONG = PRE["2048"]["tok_s_median"], PRE["16384"]["tok_s_median"]
PRE_PRED = PRE_SHORT * mac_flops_per_prompt_token(2048) / mac_flops_per_prompt_token(16384)
assert round(PRE_SHORT) == 1005 and round(PRE_LONG) == 801 and round(PRE_PRED) == 798  # "a thousand … eight hundred"
assert abs(PRE_PRED / PRE_LONG - 1) < 0.01  # "as the math predicts"

# decode on one H100: weights + the whole cache, every step
SHORT = 2048
STEP_SHORT, STEP_BOOK = step_time(context=SHORT), step_time(context=BOOK)
STEP_BYTES = W_BYTES + BOOK_KV
assert round(STEP_SHORT * 1e3, 1) == 4.9 and round(STEP_BOOK * 1e3, 1) == 9.9  # notes §11
assert round(STEP_BOOK * 1e3) == 10 and round(STEP_SHORT * 1e3) == 5  # "ten milliseconds instead of five"
assert 0.48 < STEP_SHORT / STEP_BOOK < 0.5  # "half the speed of a short chat"
assert round(STEP_BYTES / 1e9, 1) == 33.2 and 2.0 < STEP_BYTES / W_BYTES < 2.2  # "twice the bytes"

# capacity on one H100 (notes §3 worked example, Episode 5): 90% of 80 GB, minus the weights, in textbook-sized caches
KV_ROOM = 0.9 * GPU["memory"] - W_BYTES
BOOK_USERS = int(KV_ROOM // BOOK_KV)
assert BOOK_USERS == 3  # "only three textbook-sized chats fit on one H100"
STREAM_SPEEDUP = 22.2  # StreamingLLM vs sliding-window recomputation: "up to 22.2x speedup"
assert round(STREAM_SPEEDUP) == 22  # "up to twenty-two times faster"

# decode on this Mac (measurements.md §3): time per token vs context
CTX = {b: sorted((int(t), r["tok_s_median"]) for t, r in measured("context", "llama-3.1-8b", b, "by_context_tokens").items())
       for b in ("bf16", "4bit")}
MS = {b: [(t, 1000 / s) for t, s in pts] for b, pts in CTX.items()}
FIT = {b: np.polyfit([t for t, _ in MS[b]], [ms for _, ms in MS[b]], 1) for b in MS}
SLOPE = FIT["bf16"][0] * 1000  # ms per 1,000 tokens of context
CEIL_SLOPE = 1000 * KV_TOKEN / MAC["bandwidth"] * 1000
assert round(SLOPE, 2) == 0.29 and round(SLOPE, 1) == 0.3 and round(CEIL_SLOPE, 2) == 0.24  # "three tenths"
assert round(FIT["4bit"][0] * 1000, 1) == 0.3  # the same cache, the same slope
BF, Q4 = CTX["bf16"], CTX["4bit"]


def ceiling(t, bits=16):
    return decode_ceiling(hw=MAC, context=t, bits=bits)


FRACS = [s / ceiling(t) for t, s in BF]
assert BF[-1][0] == 32768 and len(BF) == 6  # "up to thirty-two thousand tokens"
assert round(BF[0][1]) == 31 and round(BF[-1][1]) == 24  # "thirty-one … fall to twenty-four"
assert all(0.88 < f < 0.92 for f in FRACS)  # "about ninety percent of the bandwidth ceiling, all the way"
assert round(Q4[0][1]) == 102 and round(Q4[-1][1]) == 51 and 0.49 < Q4[-1][1] / Q4[0][1] < 0.51  # "halves"
Q4_FILE = measured("models", "llama-3.1-8b", "4bit", "weights_file_bytes")
assert abs(weight_bytes(bits=4.5) / Q4_FILE - 1) < 0.001  # MLX 4-bit ≈ 4.5 bits per weight with its scales
Q4_CEIL = (ceiling(Q4[0][0], 4.5), ceiling(Q4[-1][0], 4.5))
assert round(Q4_CEIL[0]) == 120 and round(Q4_CEIL[1]) == 62

# the rotating cache on this Mac (measurements.md §6): Owl-mini, 1,024 slots, the first 4 kept as sinks
ROT = measured("rotating_cache", "llama-3.2-1b", "4bit")
WINDOW, SINKS = ROT["max_kv_size"], 4  # RotatingKVCache(max_size=1024, keep=4)


def rot_points(kind):
    return sorted((int(t), r["kv_MiB"], r["window_tok_s"]) for t, r in ROT[kind]["points"].items())


FULL, ROLL = rot_points("full_cache"), rot_points("rotating")
FULL_SPEED = [(t, s) for t, _, s in FULL if s]
ROLL_SPEED = [(t, s) for t, _, s in ROLL if s]
assert WINDOW == 1024 and FULL[-1][0] == ROLL[-1][0] == 8192  # "a thousand-token window"
assert ROLL[-1][1] == 32 == WINDOW * kv_bytes_per_token(MODELS["owl-mini"]) / MIB
assert all(mib == 32 for t, mib, _ in ROLL if t >= WINDOW)  # flat
assert round(FULL[-1][1] / ROLL[-1][1]) == 8  # "grew eightfold"
assert 1 - FULL_SPEED[-1][1] / FULL_SPEED[0][1] > 1 / 3  # "slowed by more than a third"
assert (max(s for _, s in ROLL_SPEED) - min(s for _, s in ROLL_SPEED)) / max(s for _, s in ROLL_SPEED) < 0.03  # flat
DEMO = ROT["forgetting_demo"]
assert DEMO["full"].startswith("PELICAN") and "There is no answer" in DEMO["rotating_1024"]  # "the code word was lost"

# offload: bring the textbook's cache back over PCIe instead of prefilling it again
LINK = 50e9  # vLLM blog 2026-01-08 (H100): "for single-direction both can get up to about 50GB/s"
RELOAD = BOOK_KV / LINK
assert round(RELOAD, 2) == 0.34 and 0.3 < RELOAD < 0.37  # "about a third of a second"
assert ttft(BOOK) / RELOAD > 30  # "instead of thirteen seconds of prefill"
VLLM_GAIN = (2, 22)  # "reduces TTFT by X2-X22, depending on the prompt size"
DEEPSEEK_HIT, DEEPSEEK_TOKENS = 0.563, 342e9  # "342B tokens (56.3%) hit the on-disk KV cache"
assert DEEPSEEK_HIT > 0.5  # "more than half"

# ---------------------------------------------------------------------------- look


def cite(s):
    return text(s, 20, MUTED)


def kv_cell(size=0.36, color=KV, fill=VIOLET_FILL):
    return m.RoundedRectangle(corner_radius=0.06, width=size, height=size, stroke_color=color, stroke_width=2.5,
                              fill_color=fill, fill_opacity=1)


def nice_bytes(n):
    if n >= GIB:
        return f"{n / GIB:g} GiB"
    if n >= MIB:
        return f"{n / MIB:g} MiB"
    return f"{n / KIB:g} KiB"


def mac_stamp():
    mac = icon(recolored("mac", MUTED, CARD, PREFIX), 0.5)
    where = m.VGroup(cite("measured on this Mac"), cite("M4 Max · mlx-lm")).arrange(m.DOWN, buff=0.06, aligned_edge=m.LEFT)
    return m.VGroup(mac, where).arrange(m.RIGHT, buff=0.22)


class Plot:
    """A small chart in scene units: p(x, y) maps data to the stage; axes() draws quiet axes with tick labels."""

    def __init__(self, origin, width, height, x_max, y_max):
        self.o = np.array([origin[0], origin[1], 0.0])
        self.w, self.h, self.xm, self.ym = width, height, x_max, y_max

    def p(self, x, y):
        return self.o + np.array([self.w * x / self.xm, self.h * y / self.ym, 0])

    def axes(self, xticks, yticks, x_title=None, y_title=None):
        xa = m.Arrow(self.p(0, 0), self.p(self.xm, 0) + [0.75, 0, 0], buff=0, color=FAINT, stroke_width=3,
                     max_tip_length_to_length_ratio=0.25 / (self.w + 0.75))
        ya = m.Arrow(self.p(0, 0), self.p(0, self.ym) + [0, 0.28, 0], buff=0, color=FAINT, stroke_width=3,
                     max_tip_length_to_length_ratio=0.07)
        xt = m.VGroup(*[code(lab, 20, MUTED).next_to(self.p(x, 0), m.DOWN, buff=0.14) for x, lab in xticks])
        yt = m.VGroup(*[code(lab, 20, MUTED).next_to(self.p(0, y), m.LEFT, buff=0.14) for y, lab in yticks])
        group = m.VGroup(xa, ya, xt, yt)
        if x_title:
            group.add(text(x_title, 20, MUTED).next_to(xt, m.DOWN, buff=0.1).set_x(self.p(self.xm / 2, 0)[0]))
        if y_title:
            group.add(text(y_title, 20, MUTED).move_to(self.p(0, self.ym) + [0.3, 0.22, 0], aligned_edge=m.LEFT))
        return group

    def polyline(self, pts, color, width=4):
        return m.VMobject().set_points_as_corners([self.p(x, y) for x, y in pts]).set_stroke(color, width)


def bracket(x0, x1, y, color, lip=0.12):
    """A horizontal bracket under a span, lips pointing up."""
    return m.VGroup(m.Line([x0, y, 0], [x1, y, 0]), m.Line([x0, y, 0], [x0, y + lip, 0]),
                    m.Line([x1, y, 0], [x1, y + lip, 0])).set_stroke(color, 2.5)


# the token strip in the sinks section: illustrative attention of the newest token (sums to one)
STRIP = 24
ATT = [0.30, 0.12, 0.07, 0.05] + [0.012] * 12 + [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.071]
assert len(ATT) == STRIP and abs(sum(ATT) - 1) < 1e-9 and sum(ATT[:SINKS]) > 0.5


# ---------------------------------------------------------------------------- the episode


class Ep09TheCostOfALongMemory(Episode):
    number = 9

    def lesson(self):
        self.hook()
        self.prefill()
        self.decode()
        self.on_this_mac()
        self.sinks()
        self.offload()
        self.trade_off()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes=f"A student pastes a whole textbook into Owl: {BOOK:,} tokens, Llama 3.1's entire 128K context (the textbook is a stand-in for any prompt that long). Owl-8B on one H100 (989 TFLOPS BF16 at 50% MFU, 3.35 TB/s; NVIDIA, dated Oct 2026): prefill {prefill_flops(BOOK) / 1e15:.2f} PFLOP → TTFT {ttft(BOOK):.1f} s (compute only, notes §11). Every later decode step reads {W_BYTES / 1e9:.2f} GB of weights plus {BOOK_KV / 1e9:.1f} GB of cache: {STEP_BOOK * 1e3:.2f} ms per token vs {STEP_SHORT * 1e3:.2f} ms with a 2K-token chat (roofline floor, step_time).")
        book = icon(recolored("book", REQUEST, PINK_FILL, PREFIX), 1.2).move_to([-5.55, 1.3, 0])
        track = m.RoundedRectangle(corner_radius=0.12, width=9.6, height=0.5).set_stroke(EDGE, 2).set_fill(CARD, 1)
        track.move_to([-4.45 + 4.8, 1.3, 0])
        bar = m.RoundedRectangle(corner_radius=0.1, width=9.6 - 0.1, height=0.4, stroke_width=0).set_fill(REQUEST, 1).move_to(track)
        cap_l = code("Owl's context window", 20, MUTED).next_to(track, m.UP, buff=0.15).align_to(track, m.LEFT)
        cap_r = code("128K tokens", 20, MUTED).next_to(track, m.UP, buff=0.15).align_to(track, m.RIGHT)
        count = m.VGroup(title_text(f"{BOOK:,}", 64, REQUEST), code("tokens · a whole textbook", 24, MUTED))
        count[1].next_to(count[0], m.RIGHT, buff=0.3).align_to(count[0], m.DOWN)
        count.next_to(track, m.DOWN, buff=0.35).align_to(track, m.LEFT)

        def big(value, unit, label, sub, color, x):
            num = title_text(value, 80, color)
            u = code(unit, 30, MUTED).next_to(num, m.RIGHT, buff=0.15).align_to(num, m.DOWN)
            lab = text(label, 24, INK).next_to(num, m.DOWN, buff=0.22).align_to(num, m.LEFT)
            s = code(sub, 20, MUTED).next_to(lab, m.DOWN, buff=0.12).align_to(num, m.LEFT)
            return m.VGroup(num, u, lab, s).move_to([x, -0.75, 0], aligned_edge=m.UL)

        first = big(f"{ttft(BOOK):.1f}", "s", "until the first token", "TTFT · one H100", COMPUTE, -5.6)
        after = big(f"{STEP_BOOK * 1e3:.1f}", "ms", "per token after that", f"short chat: {STEP_SHORT * 1e3:.1f} ms", MEMORY, 0.6)
        tease = m.VGroup(
            pill("prefill: quadratic", COMPUTE, AMBER_FILL, 22),
            pill("decode: linear", MEMORY, CYAN_FILL, 22),
            pill("sinks · windows · offload", KV, VIOLET_FILL, 22),
        ).arrange(m.RIGHT, buff=0.35).move_to([-0.75, -3.25, 0])
        mascot = owl(0.6).move_to([5.95, -3.05, 0])

        self.say(
            "[A student pastes a whole textbook into Owl]: [a hundred and twenty-eight thousand tokens, its entire context.]",
            m.AnimationGroup(self.new_header("hook", "A whole textbook"), m.FadeIn(book, shift=0.2 * m.RIGHT),
                             m.FadeIn(track), m.FadeIn(cap_l), m.FadeIn(cap_r), m.FadeIn(mascot, shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowFromEdge(bar, m.LEFT, run_time=2.4), m.FadeIn(count, shift=0.15 * m.UP)),
        )
        self.say(
            "[On one H100, the first word of the answer arrives after about thirteen seconds.]",
            m.FadeIn(first, shift=0.2 * m.UP),
        )
        self.say(
            "[And every word after it comes at half the speed of a short chat.]",
            m.FadeIn(after, shift=0.2 * m.UP),
        )
        self.say(
            "[Reading it grows with the square of its length]; [keeping it grows in a straight line.]",
            m.FadeIn(tease[0], shift=0.15 * m.UP),
            m.FadeIn(tease[1], shift=0.15 * m.UP),
        )
        self.say(
            "[Today: why a long memory costs so much], [and what servers do about it.]",
            m.Circumscribe(m.VGroup(first, after), color=KV, buff=0.15),
            m.FadeIn(tease[2], shift=0.15 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ prefill: quadratic

    def prefill(self):
        self.next_section("1 · Prefill grows with the square", notes=f"Prefill FLOPs (notes §11) = 2·P·n (every prompt token through every weight; P = 8.03 B) + 2·L·n²·d (causal attention, QKᵀ and AV; L = 32, d = 32 heads × 128 = 4,096). At 1K / 32K / 128K tokens: attention share {100 * share(1024):.1f}% / {100 * share(32768):.0f}% / {100 * share(BOOK):.0f}%; TTFT on one H100 at 50% of 989 TFLOPS: {ttft(1024) * 1000:.0f} ms / {ttft(32768):.2f} s / {ttft(BOOK):.1f} s (total at 128K: {weight_flops(BOOK) / 1e15:.2f} + {attention_flops(BOOK) / 1e15:.2f} = {prefill_flops(BOOK) / 1e15:.2f} PFLOP). The 4- and 8-token triangles count (query, key) pairs: 10 and 36. Measured on this Mac (measurements.md §2, Owl-8B BF16): prefill {PRE_SHORT:,.0f} tok/s at 2,048 tokens, {PRE_LONG:.0f} at 16,384; the model 2 × {BLOCK_PARAMS / 1e9:.2f} B + 2·L·d·T FLOPs per prompt token predicts {PRE_PRED:.0f}. Episode 3 showed TTFT itself growing with prompt length on this Mac.")
        plot = Plot((-6.0, -2.4), 6.3, 4.2, BOOK, 7e15)
        ax = plot.axes([(0, "0"), (32768, "32K"), (65536, "64K"), (98304, "96K"), (BOOK, "128K")],
                       [(0, "0"), (2e15, "2"), (4e15, "4"), (6e15, "6")], "prompt tokens", "prefill FLOPs (peta)")
        ns = np.linspace(0, BOOK, 90)
        lin_line = m.Line(plot.p(0, 0), plot.p(BOOK, weight_flops(BOOK))).set_stroke(COMPUTE, 3)
        lin_area = m.Polygon(plot.p(0, 0), plot.p(BOOK, 0), plot.p(BOOK, weight_flops(BOOK))).set_stroke(width=0).set_fill(COMPUTE, 0.22)
        curve = plot.polyline([(n, prefill_flops(n)) for n in ns], COMPUTE, 4)
        att_area = m.Polygon(*[plot.p(n, prefill_flops(n)) for n in ns], *[plot.p(n, weight_flops(n)) for n in reversed(ns)])
        att_area.set_stroke(width=0).set_fill(COMPUTE, 0.6)
        lin_lab = code("weights", 20, COMPUTE).move_to(plot.p(106000, 0.42e15))
        att_lab = code("attention", 22, COMPUTE, weight=m.BOLD).move_to(plot.p(84000, 4.6e15))

        px = 4.05
        sw1 = m.Rectangle(width=0.42, height=0.28).set_stroke(COMPUTE, 2).set_fill(COMPUTE, 0.22)
        sw2 = m.Rectangle(width=0.42, height=0.28).set_stroke(COMPUTE, 2).set_fill(COMPUTE, 0.6)
        leg1 = m.VGroup(sw1, text("every token through every weight", 22, INK)).arrange(m.RIGHT, buff=0.2)
        leg2 = m.VGroup(sw2, text("attention: every pair of tokens", 22, INK)).arrange(m.RIGHT, buff=0.2)
        legend = m.VGroup(leg1, leg2).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.2).move_to([px, 2.1, 0])

        def triangle(n, x, size=0.17, gap=0.04):
            pitch = size + gap
            cells = m.VGroup()
            for r in range(n):
                for c in range(r + 1):
                    cells.add(m.Square(size).set_stroke(COMPUTE, 1.2).set_fill(COMPUTE, 0.6).move_to([x + c * pitch, -r * pitch, 0]))
            cells.move_to([x, 0, 0]).align_to([0, -0.55, 0], m.DOWN)
            return cells

        t4, t8 = triangle(4, 2.75), triangle(8, 5.0)
        lab4 = m.VGroup(code("4 tokens", 20, MUTED), code(f"{pairs(4)} pairs", 22, COMPUTE)).arrange(m.DOWN, buff=0.08).next_to(t4, m.DOWN, buff=0.2)
        lab8 = m.VGroup(code("8 tokens", 20, MUTED), code(f"{pairs(8)} pairs", 22, COMPUTE)).arrange(m.DOWN, buff=0.08).next_to(t8, m.DOWN, buff=0.2)
        lab8.align_to(lab4, m.UP)
        tris = m.VGroup(t4, lab4, t8, lab8)

        # the readout for one prompt length
        bar_w, bar_y = 4.6, 0.2
        track = m.Rectangle(width=bar_w, height=0.42).set_stroke(EDGE, 2).set_fill(COMPUTE, 0.22).move_to([px, bar_y, 0])

        def n_label(n):
            g = m.VGroup(code("prompt", 22, MUTED), title_text(f"{n // 1024}K tokens", 44, INK))
            g[0].next_to(g[1], m.LEFT, buff=0.25).align_to(g[1], m.DOWN)
            return g.move_to([px, 1.0, 0])

        def seg(n):
            w = max(bar_w * share(n), 0.04)
            return m.Rectangle(width=w, height=0.42).set_stroke(width=0).set_fill(COMPUTE, 0.9).move_to([px - bar_w / 2 + w / 2, bar_y, 0])

        def share_lab(n):
            pct = f"{100 * share(n):.1f}" if share(n) < 0.1 else f"{100 * share(n):.0f}"
            return code(f"attention: {pct}% of the math", 22, COMPUTE).move_to([px, bar_y - 0.48, 0])

        def ttft_lab(n):
            t = ttft(n)
            value = f"{t * 1000:.0f} ms" if t < 1 else f"{t:.1f} s"
            return title_text(value, 64, COMPUTE).move_to([px, -1.05, 0])

        def cursor(n):
            guide = m.DashedLine(plot.p(n, 0), plot.p(n, 7e15), dash_length=0.08).set_stroke(INK, 2, opacity=0.7)
            dot = m.Dot(plot.p(n, prefill_flops(n)), radius=0.09, color=INK).set_z_index(3)
            return m.VGroup(guide, dot)

        n0 = STOPS[0]
        cur, nlab, sg, slab, tlab = cursor(n0), n_label(n0), seg(n0), share_lab(n0), ttft_lab(n0)
        t_sub = m.VGroup(text("until the first token", 22, INK), code("one H100 · 50% of peak", 20, MUTED)).arrange(m.DOWN, buff=0.1)
        t_sub.move_to([px, -1.75, 0])
        readout = m.VGroup(nlab, track, sg, slab, tlab, t_sub)

        mac_rows = m.VGroup(
            code("measured on this Mac · Owl-8B", 20, MUTED),
            code(f"2K: {PRE_SHORT:,.0f} → 16K: {PRE_LONG:.0f} tok/s", 22, INK),
            code(f"attention model: {PRE_PRED:.0f} ✓", 22, INK, t2c={"✓": GOOD}),
        ).arrange(m.DOWN, buff=0.12).move_to([px, -2.95, 0])

        self.say(
            "[Reading the textbook is prefill], [and its math comes in two parts.]",
            m.AnimationGroup(self.new_header("01 · prefill", "Reading grows with the square"), m.FadeIn(ax)),
            m.FadeIn(legend, shift=0.1 * m.UP),
        )
        self.say(
            "[First, every token passes through every weight]: [twice the tokens, twice the work.]",
            m.Circumscribe(leg1, color=COMPUTE, buff=0.08),
            m.AnimationGroup(m.Create(lin_line), m.FadeIn(lin_area), m.FadeIn(lin_lab)),
        )
        self.say(
            "[Second, attention compares each token with every token before it]: [twice the tokens, about four times the pairs.]",
            m.AnimationGroup(m.Circumscribe(leg2, color=COMPUTE, buff=0.08),
                             m.LaggedStart(*[m.FadeIn(c, scale=0.6) for c in t4], lag_ratio=0.12), m.FadeIn(lab4)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c, scale=0.6) for c in t8], lag_ratio=0.04), m.FadeIn(lab8)),
        )
        self.say(
            "[Add the two, and the total bends upward.]",
            m.AnimationGroup(m.Create(curve), m.FadeIn(att_area), m.FadeIn(att_lab), run_time=2.0),
        )
        self.say(
            "[At a thousand tokens, attention is under two percent of the math.] [At thirty-two thousand, about a third.]",
            m.AnimationGroup(m.FadeOut(tris), m.FadeIn(readout), m.FadeIn(cur)),
            m.AnimationGroup(m.Transform(cur, cursor(STOPS[1])), m.Transform(nlab, n_label(STOPS[1])), m.Transform(sg, seg(STOPS[1])),
                             m.Transform(slab, share_lab(STOPS[1])), m.Transform(tlab, ttft_lab(STOPS[1])), run_time=1.4),
        )
        self.say(
            "[At the textbook's full length, it's two thirds]: [thirteen point four seconds on one H100, at half its peak.]",
            m.AnimationGroup(m.Transform(cur, cursor(STOPS[2])), m.Transform(nlab, n_label(STOPS[2])), m.Transform(sg, seg(STOPS[2])),
                             m.Transform(slab, share_lab(STOPS[2])), run_time=1.6),
            m.AnimationGroup(m.Transform(tlab, ttft_lab(STOPS[2])), m.Circumscribe(t_sub, color=COMPUTE, buff=0.08)),
        )
        self.say(
            "[This Mac measures the same bend]: [about a thousand prompt tokens per second at two thousand tokens], [and eight hundred at sixteen thousand, as the math predicts.]",
            m.FadeIn(mac_rows[0], shift=0.1 * m.UP),
            m.FadeIn(mac_rows[1], shift=0.1 * m.UP),
            m.FadeIn(mac_rows[2], shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ decode: linear

    def decode(self):
        self.next_section("2 · Every step reads it all", notes=f"Each decode step reads every weight ({W_BYTES / 1e9:.2f} GB, BF16) and the whole KV cache (notes §0 #2, §11). A 2K-token chat: {SHORT * KV_TOKEN / 1e9:.2f} GB of cache; the textbook: {BOOK:,} × 128 KiB = 16 GiB = {BOOK_KV / 1e9:.2f} GB, about the size of the weights (Episode 5). Bytes per step {STEP_BYTES / 1e9:.1f} GB ÷ 3.35 TB/s (H100 SXM, NVIDIA, dated Oct 2026) = {STEP_BOOK * 1e3:.2f} ms vs {STEP_SHORT * 1e3:.2f} ms (owl_reference.step_time; a roofline floor, batch 1).")
        x0, total_w, by = -6.0, 11.6, 1.45
        per = total_w / STEP_BYTES
        w_w = per * W_BYTES
        w_rect = m.Rectangle(width=w_w, height=0.6).set_stroke(width=0).set_fill(WEIGHTS, 1).move_to([x0 + w_w / 2, by, 0])

        def kv_rect(nbytes):
            w = max(per * nbytes, 0.05)
            return m.Rectangle(width=w, height=0.6).set_stroke(width=0).set_fill(KV, 1).move_to([x0 + w_w + w / 2, by, 0])

        kv_r = kv_rect(SHORT * KV_TOKEN)
        kv_book = kv_rect(BOOK_KV)
        w_lab = code(f"weights · {W_BYTES / 1e9:.0f} GB", 22, WEIGHTS).move_to([x0 + w_w / 2, by - 0.6, 0])
        kv_lab = code("cache · 2K tokens", 20, KV).move_to([x0 + w_w + 0.15, by - 0.6, 0], aligned_edge=m.LEFT)
        kv_lab_book = code(f"cache · 128K tokens · {BOOK_KV / 1e9:.1f} GB", 22, KV).move_to([x0 + w_w + per * BOOK_KV / 2, by - 0.6, 0])
        caption = code("bytes read for every new token", 20, MUTED).move_to([x0, by + 0.7, 0], aligned_edge=m.LEFT)
        total = code(f"= {STEP_BYTES / 1e9:.1f} GB per token", 22, INK).move_to([x0 + total_w, by + 0.7, 0], aligned_edge=m.RIGHT)
        sweep = lambda: m.ShowPassingFlash(m.Line([x0, by, 0], [x0 + total_w, by, 0]).set_stroke(MEMORY, 14), time_width=0.5, run_time=1.6)
        sweep_w = m.ShowPassingFlash(m.Line([x0, by, 0], [x0 + w_w, by, 0]).set_stroke(MEMORY, 14), time_width=0.6, run_time=1.2)

        bx, scale = -1.2, 5.4 / (STEP_BOOK * 1e3)
        rows = m.VGroup()
        for i, (label, step) in enumerate((("short chat · 2K tokens", STEP_SHORT), ("the textbook · 128K tokens", STEP_BOOK))):
            y = -0.55 - 0.95 * i
            lab = text(label, 24, INK).move_to([x0, y, 0], aligned_edge=m.LEFT)
            w = scale * step * 1e3
            r = m.Rectangle(width=w, height=0.5).set_stroke(MEMORY, 2).set_fill(CYAN_FILL, 1).move_to([bx + w / 2, y, 0])
            t = code(f"{step * 1e3:.1f} ms", 24, INK, weight=m.BOLD).next_to(r, m.RIGHT, buff=0.2)
            rows.add(m.VGroup(lab, r, t))
        note = code("time per step = bytes ÷ 3.35 TB/s · one H100", 20, MUTED).move_to([0, -2.45, 0])
        verdict = pill("half the speed, for one student", MEMORY, CYAN_FILL, 24).move_to([0, -3.2, 0])
        fits = pill(f"{BOOK_USERS} textbook chats per H100 · Episode 5", KV, VIOLET_FILL, 24).move_to([3.3, -3.2, 0])

        self.say(
            "[After prefill, every new token reads all the weights], [and the whole cache.]",
            m.AnimationGroup(self.new_header("02 · decode", "Every step reads it all"), m.FadeIn(caption),
                             m.GrowFromEdge(w_rect, m.LEFT), m.FadeIn(w_lab, shift=0.1 * m.UP), sweep_w),
            m.AnimationGroup(m.GrowFromEdge(kv_r, m.LEFT), m.FadeIn(kv_lab, shift=0.1 * m.UP)),
        )
        self.say(
            "[For the textbook, that cache is as big as the weights themselves]: [so every step moves twice the bytes.]",
            m.AnimationGroup(m.Transform(kv_r, kv_book), m.FadeOut(kv_lab), m.FadeIn(kv_lab_book, shift=0.1 * m.UP), run_time=1.6),
            m.AnimationGroup(m.FadeIn(total, shift=0.1 * m.LEFT), sweep()),
        )
        self.say(
            "[On the H100, a step takes ten milliseconds instead of five.]",
            m.AnimationGroup(
                m.LaggedStart(*[m.AnimationGroup(m.FadeIn(r[0]), m.GrowFromEdge(r[1], m.LEFT), m.FadeIn(r[2])) for r in rows], lag_ratio=0.5),
                m.FadeIn(note), m.FadeIn(verdict, scale=0.9, run_time=1.2)),
        )
        self.say(
            "[And memory runs out first]: [only three textbook-sized chats fit on a single H100], [as episode five showed.]",
            verdict.animate(run_time=0.8).move_to([-3.3, -3.2, 0]),
            m.FadeIn(fits, shift=0.1 * m.LEFT),
            m.Circumscribe(fits, color=KV, buff=0.08),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ measured on this Mac

    def on_this_mac(self):
        bf_pts = ", ".join(f"{t}: {s:.1f}" for t, s in BF)
        q4_pts = ", ".join(f"{t}: {s:.1f}" for t, s in Q4)
        self.next_section("3 · The slowdown, measured", notes=f"measurements.md §3 (this Mac, 2026-10-24, mlx-lm 0.32.2): the prompt is prefilled once, then 128 tokens are decoded from that cache, 5 trials. Owl-8B BF16 tok/s by context: {bf_pts}. 4-bit weights: {q4_pts}. Plotted as ms per token (1000 ÷ tok/s): a straight line, least-squares slope {SLOPE:.3f} ms per 1,000 tokens (4-bit: {FIT['4bit'][0] * 1000:.3f}: the same BF16 cache); the ceiling slope is 131,072 B ÷ 546 GB/s = {CEIL_SLOPE:.3f} ms per 1,000 tokens. Ceiling = 546 GB/s ÷ (weights + T × 128 KiB) (owl_reference.decode_ceiling): measured BF16 is {100 * min(FRACS):.0f}–{100 * max(FRACS):.0f}% of it at every length. 4-bit ceiling with MLX's 4.5 bits per weight ({weight_bytes(bits=4.5) / 1e9:.2f} GB = the file size): {Q4_CEIL[0]:.0f} → {Q4_CEIL[1]:.0f} tok/s. Flash-Decoding (Dao et al., Oct 2023): split-KV kernels for long-context decode, 'up to 8x faster generation for very long sequences' (notes §7) → Episode 16.")
        plot = Plot((-6.0, -2.35), 6.8, 4.0, 34000, 45)
        ax = plot.axes([(0, "0"), (8192, "8K"), (16384, "16K"), (24576, "24K"), (32768, "32K")],
                       [(0, "0"), (10, "10"), (20, "20"), (30, "30"), (40, "40")], "tokens of context", "milliseconds per token")
        end = 33500
        ceil_ms = lambda t: (W_BYTES + t * KV_TOKEN) / MAC["bandwidth"] * 1000
        fit = lambda b, t: FIT[b][1] + FIT[b][0] * t
        bf_dots = m.VGroup(*[m.Dot(plot.p(t, ms), radius=0.085, color=INK).set_z_index(3) for t, ms in MS["bf16"]])
        q4_dots = m.VGroup(*[m.Dot(plot.p(t, ms), radius=0.085, color=INK).set_z_index(3) for t, ms in MS["4bit"]])
        bf_line = m.Line(plot.p(0, fit("bf16", 0)), plot.p(end, fit("bf16", end))).set_stroke(MEMORY, 4)
        q4_line = m.Line(plot.p(0, fit("4bit", 0)), plot.p(end, fit("4bit", end))).set_stroke(MEMORY, 4)
        ceil_line = m.DashedLine(plot.p(0, ceil_ms(0)), plot.p(end, ceil_ms(end)), dash_length=0.12).set_stroke(MEMORY, 2.5, opacity=0.75)
        bf_lab = code("BF16", 20, INK).next_to(plot.p(32768, MS["bf16"][-1][1]), m.RIGHT, buff=0.26)
        ceil_lab = code("ceiling: bytes ÷ 546 GB/s", 20, MEMORY).move_to(plot.p(9000, 27.2), aligned_edge=m.LEFT)
        q4_lab = code("4-bit", 20, INK).next_to(plot.p(32768, MS["4bit"][-1][1]), m.RIGHT, buff=0.26)
        stamp = mac_stamp().move_to([-1.15, 2.15, 0])

        px = 4.55
        s_head = text("each 1,000 tokens of context adds", 22, MUTED).move_to([px, 2.05, 0])
        s_num = title_text(f"+{SLOPE:.2f} ms", 60, MEMORY).move_to([px, 1.4, 0])
        s_sub = code(f"per token · ceiling +{CEIL_SLOPE:.2f}", 20, MUTED).move_to([px, 0.85, 0])
        bf_tok = code(f"BF16: {BF[0][1]:.1f} → {BF[-1][1]:.1f} tok/s", 22, INK).move_to([px, 0.05, 0])
        bf_frac = code(f"{100 * min(FRACS):.0f}–{100 * max(FRACS):.0f}% of the ceiling ✓", 20, GOOD).move_to([px, -0.37, 0])
        q4_tok = code(f"4-bit: {Q4[0][1]:.0f} → {Q4[-1][1]:.0f} tok/s", 22, INK).move_to([px, -1.15, 0])
        q4_ceil = code(f"ceiling {Q4_CEIL[0]:.0f} → {Q4_CEIL[1]:.0f} · halved", 20, MUTED).move_to([px, -1.57, 0])
        fd = m.VGroup(code("Flash-Decoding: split-KV", 20, INK), cite("Dao et al., 2023 · Episode 16")).arrange(m.DOWN, buff=0.1)
        fd_box = m.SurroundingRectangle(fd, buff=0.16, corner_radius=0.12).set_stroke(EDGE, 2)
        flash = m.VGroup(fd_box, fd).move_to([px, -2.75, 0])
        split = m.VGroup(*[m.Line(plot.p(a, 0) + [0, 0.12, 0], plot.p(a + 7000, 0) + [0, 0.12, 0]).set_stroke(MEMORY, 8) for a in (500, 8700, 16900, 25100)])

        self.say(
            "[We measured this on the Mac], [with Owl in sixteen-bit, up to thirty-two thousand tokens.]",
            m.AnimationGroup(self.new_header("03 · measured", "The slowdown, on this Mac"), m.FadeIn(ax), m.FadeIn(stamp)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in bf_dots], lag_ratio=0.2), m.FadeIn(bf_lab)),
        )
        self.say(
            "[The time per token climbs in a straight line]: [about three tenths of a millisecond per thousand tokens of context.]",
            m.Create(bf_line),
            m.AnimationGroup(m.FadeIn(s_head), m.FadeIn(s_num, shift=0.1 * m.UP), m.FadeIn(s_sub)),
        )
        self.say(
            "[Thirty-one tokens per second fall to twenty-four], [and stay at about ninety percent of the bandwidth ceiling, all the way.]",
            m.FadeIn(bf_tok, shift=0.1 * m.UP),
            m.AnimationGroup(m.Create(ceil_line), m.FadeIn(ceil_lab), m.FadeIn(bf_frac, shift=0.1 * m.UP)),
        )
        self.say(
            "[With four-bit weights, the same cache weighs more]: [the speed halves, from a hundred and two to fifty-one.]",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in q4_dots], lag_ratio=0.15), m.Create(q4_line), m.FadeIn(q4_lab)),
            m.AnimationGroup(m.FadeIn(q4_tok, shift=0.1 * m.UP), m.FadeIn(q4_ceil, shift=0.1 * m.UP)),
        )
        self.say(
            "[Kernels matter here too]: [Flash-Decoding splits a long cache across the whole GPU], [and that's episode sixteen.]",
            m.FadeIn(flash, shift=0.1 * m.UP),
            m.LaggedStart(*[m.ShowPassingFlash(s.copy(), time_width=0.8, run_time=1.2) for s in split], lag_ratio=0.0),
            m.Circumscribe(flash, color=MEMORY, buff=0.06),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ sinks + window

    def sinks(self):
        self.next_section("4 · A sink and a window", notes=f"StreamingLLM (Xiao et al., ICLR 2024, arXiv 2309.17453): window attention 'fails when the text length surpasses the cache size'; models put 'strong attention scores towards initial tokens as a sink even if they are not semantically important', because softmax 'requires attention scores to sum up to one'; keeping the KV of 4 initial tokens plus a rolling window 'will largely recover the performance of window attention'; stable 'with up to 4 million tokens and more' (notes §11). The 24-square strip and its attention weights are illustrative. Measured on this Mac (measurements.md §6, Owl-mini = Llama-3.2-1B 4-bit, mlx-lm RotatingKVCache(max_size={WINDOW}, keep={SINKS}) built explicitly: --max-kv-size is silently ignored for Llama in mlx-lm 0.32.2): after 8,192 generated tokens the full cache holds {FULL[-1][1]:.0f} MiB and decodes {FULL_SPEED[-1][1]:.0f} tok/s (from {FULL_SPEED[0][1]:.0f} at 1,024); the rotating cache stays at {ROLL[-1][1]:.0f} MiB and {min(s for _, s in ROLL_SPEED):.0f}–{max(s for _, s in ROLL_SPEED):.0f} tok/s. The forgetting demo (Episode 6): a {DEMO['prompt_tokens']:,}-token prompt that opens with a code word; the full cache answers PELICAN, the window 'There is no answer…': the code word sat after the 4 sink tokens, so it was evicted.")
        pitch, size, cy = 0.46, 0.36, 0.15
        xs = [-(STRIP - 1) * pitch / 2 + i * pitch for i in range(STRIP)]
        cells = m.VGroup(*[kv_cell(size).move_to([x, cy, 0]) for x in xs])
        win_lo = STRIP - 8
        edge = lambda i, side: xs[i] + side * size / 2
        win_br = bracket(edge(win_lo, -1), edge(STRIP - 1, 1), cy - 0.38, KV)
        win_lab = code("window · the latest tokens", 20, KV).next_to(win_br, m.DOWN, buff=0.12)
        ep6 = code("→ Episode 6", 20, MUTED).next_to(win_lab, m.DOWN, buff=0.1)
        sink_br = bracket(edge(0, -1), edge(SINKS - 1, 1), cy - 0.38, KV)
        sink_lab = code(f"{SINKS} sinks", 20, KV, weight=m.BOLD).next_to(sink_br, m.DOWN, buff=0.12)
        caption = code("one square = one token's keys and values (schematic)", 20, MUTED).move_to([0, -1.35, 0])
        base, k = cy + 0.38, 3.0
        bars = m.VGroup(*[
            m.Rectangle(width=0.3, height=max(k * w, 0.03)).set_stroke(width=0).set_fill(TOKEN, 0.9).move_to([x, base + k * w / 2, 0])
            for x, w in zip(xs, ATT)
        ])
        sum_lab = code("weights add up to 1", 20, TOKEN).move_to([3.4, base + 1.05, 0])
        att_cap = code("the newest token's attention, for example", 20, MUTED).move_to([2.6, base + 1.5, 0])
        sink_word = code("attention sinks", 22, TOKEN, weight=m.BOLD).move_to([(xs[0] + xs[SINKS - 1]) / 2 + 0.55, base + 1.25, 0])
        evict = lambda c: c.animate.set_fill(CARD, 1).set_stroke(FAINT, 1.5)
        relight = lambda c: c.animate.set_fill(VIOLET_FILL, 1).set_stroke(INK, 3)
        fail = pill("✗ window alone: breaks once the first tokens are gone", BAD, BAD_FILL, 22).move_to([0, -2.2, 0])
        ok = pill(f"✓ {SINKS} sinks + a window: steady for 4 million+ tokens", GOOD, GOOD_FILL, 22).move_to([0, -2.2, 0])
        src = cite("StreamingLLM · Xiao et al., ICLR 2024").move_to([0, -2.85, 0])
        fast = code(f"up to {STREAM_SPEEDUP}× faster than recomputing a sliding window", 20, GOOD).move_to([0, -3.3, 0])

        self.say(
            "[Remedy one: keep only the recent past.] [Episode six showed the rolling window.]",
            m.AnimationGroup(self.new_header("04 · sinks", "A sink and a window"),
                             m.LaggedStart(*[m.FadeIn(c, scale=0.8) for c in cells], lag_ratio=0.04), m.FadeIn(caption)),
            m.AnimationGroup(m.LaggedStart(*[evict(c) for c in cells[:win_lo]], lag_ratio=0.05),
                             m.Create(win_br), m.FadeIn(win_lab), m.FadeIn(ep6)),
        )
        self.say(
            "[But once the first few tokens slide out of the window], [the model's output falls apart.]",
            m.Circumscribe(m.VGroup(*cells[:SINKS]), color=BAD, buff=0.08),
            m.AnimationGroup(m.FadeIn(fail, scale=0.9), m.FadeIn(src)),
        )
        self.say(
            "[Attention weights must add up to one], [so heads park their spare attention on the first tokens]: [attention sinks.]",
            m.AnimationGroup(m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in bars], lag_ratio=0.04),
                             m.FadeIn(att_cap), m.FadeIn(sum_lab)),
            m.Circumscribe(m.VGroup(*bars[:SINKS]), color=TOKEN, buff=0.08),
            m.FadeIn(sink_word, shift=0.1 * m.DOWN),
        )
        self.say(
            "[Keep four sink tokens plus the window], [and the model streams millions of tokens at a steady cost.]",
            m.AnimationGroup(*[relight(c) for c in cells[:SINKS]], m.Create(sink_br), m.FadeIn(sink_lab)),
            m.AnimationGroup(m.FadeOut(fail), m.FadeIn(ok, scale=0.9)),
        )
        self.say(
            "[And it runs up to twenty-two times faster than recomputing a sliding window at every step.]",
            m.FadeIn(fast, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

        # measured on this Mac: Owl-mini with a rotating cache
        lp = Plot((-5.9, -0.75), 4.7, 2.45, 8192, 300)
        rp = Plot((1.15, -0.75), 4.7, 2.45, 8192, 500)
        ticks = [(0, "0"), (4096, "4K"), (8192, "8K")]
        l_ax = lp.axes(ticks, [(0, "0"), (100, "100"), (200, "200"), (300, "300")])
        r_ax = rp.axes(ticks, [(0, "0"), (250, "250"), (500, "500")])
        l_title = text("cache size (MiB)", 22, INK, weight=m.BOLD).move_to([-5.9, 2.15, 0], aligned_edge=m.LEFT)
        r_title = text("decode speed (tokens/s)", 22, INK, weight=m.BOLD).move_to([1.15, 2.15, 0], aligned_edge=m.LEFT)
        x_note = cite("tokens generated · Owl-mini on this Mac · mlx-lm").move_to([0, -1.55, 0])
        full_mem = lp.polyline([(t, mib) for t, mib, _ in FULL], MUTED, 3)
        roll_mem = lp.polyline([(t, mib) for t, mib, _ in ROLL], KV, 5)
        full_spd = rp.polyline(FULL_SPEED, MUTED, 3)
        roll_spd = rp.polyline(ROLL_SPEED, MEMORY, 5)
        l_full = code("full cache", 20, MUTED).move_to(lp.p(2000, 200))
        l_roll = code("sinks + window", 20, KV).move_to(lp.p(5800, 32) + [0, 0.28, 0])
        r_full = code("full cache", 20, MUTED).move_to(rp.p(5600, 250))
        r_roll = code("sinks + window", 20, MEMORY).move_to(rp.p(4300, 434) + [0, 0.27, 0])
        l_vals = m.VGroup(code(f"{FULL[-1][1]:.0f}", 20, MUTED).next_to(lp.p(8192, FULL[-1][1]), m.RIGHT, buff=0.12),
                          code(f"{ROLL[-1][1]:.0f}", 20, KV).next_to(lp.p(8192, ROLL[-1][1]), m.RIGHT, buff=0.12))
        r_vals = m.VGroup(code(f"{FULL_SPEED[-1][1]:.0f}", 20, MUTED).next_to(rp.p(8192, FULL_SPEED[-1][1]), m.RIGHT, buff=0.12),
                          code(f"{ROLL_SPEED[-1][1]:.0f}", 20, MEMORY).next_to(rp.p(8192, ROLL_SPEED[-1][1]), m.RIGHT, buff=0.12))
        setup = code(f"{SINKS} sinks + {WINDOW - SINKS:,} recent tokens = {WINDOW:,} slots", 20, INK).move_to([0, -1.95, 0])

        # where the code word was
        sy = -2.7
        s_cells = m.VGroup(*[kv_cell(0.34) for _ in range(SINKS)]).arrange(m.RIGHT, buff=0.07).move_to([-5.55, sy, 0])
        gone = m.DashedVMobject(m.Rectangle(width=5.4, height=0.5), num_dashes=46).set_stroke(FAINT, 2).move_to([-1.8, sy, 0])
        word = code("“the code word is PELICAN”", 20, REQUEST).move_to(gone)
        win_box = m.Rectangle(width=5.1, height=0.5).set_stroke(KV, 2).set_fill(KV, 0.75).move_to([3.7, sy, 0])
        win_txt = code(f"last {WINDOW - SINKS:,} tokens", 20, INK).move_to(win_box)
        s_tag = code("sinks", 20, KV).next_to(s_cells, m.DOWN, buff=0.12)
        g_tag = code("evicted ✗", 20, BAD).next_to(gone, m.DOWN, buff=0.12)
        w_tag = code("window", 20, KV).next_to(win_box, m.DOWN, buff=0.12)
        strip = m.VGroup(s_cells, gone, win_box, win_txt, s_tag, g_tag, w_tag)

        self.say(
            "[On this Mac, Owl-mini ran with four sinks and a thousand-token window.]",
            m.AnimationGroup(m.FadeIn(l_ax), m.FadeIn(r_ax), m.FadeIn(l_title), m.FadeIn(r_title), m.FadeIn(x_note),
                             m.FadeIn(setup, shift=0.1 * m.UP)),
        )
        self.say(
            "[Its cache and its speed stayed flat], [while the full cache grew eightfold and slowed by more than a third.]",
            m.AnimationGroup(m.Create(roll_mem), m.Create(roll_spd), m.FadeIn(l_roll), m.FadeIn(r_roll),
                             m.FadeIn(l_vals[1]), m.FadeIn(r_vals[1]), run_time=1.8),
            m.AnimationGroup(m.Create(full_mem), m.Create(full_spd), m.FadeIn(l_full), m.FadeIn(r_full),
                             m.FadeIn(l_vals[0]), m.FadeIn(r_vals[0]), run_time=2.2),
        )
        self.say(
            "[The price: the window forgets.] [Episode six's code word was lost], [because sinks keep the model steady, not its memory.]",
            m.AnimationGroup(m.FadeOut(setup), m.FadeIn(strip, shift=0.1 * m.UP)),
            m.Succession(m.FadeIn(word, run_time=0.6), word.animate(run_time=0.8).set_opacity(0.35)),
            m.Circumscribe(s_cells, color=KV, buff=0.08),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ offload

    def offload(self):
        self.next_section("5 · Move it down, not out", notes=f"KV offloading (notes §3): when GPU memory is full, KV blocks move to CPU DRAM or disk and come back later, cheaper than recomputing. vLLM KV-offloading blog (2026-01-08, Llama-3.1-8B-Instruct on an H100): CPU↔GPU 'for single-direction both can get up to about 50GB/s'; 'loading KV values from the CPU reduces TTFT by X2-X22, depending on the prompt size'. Derived: the textbook's {BOOK_KV / 1e9:.1f} GB at 50 GB/s = {RELOAD:.2f} s vs {ttft(BOOK):.1f} s of prefill on one H100 (§11). DeepSeek inference system overview (Feb 2025, 24 h): '342B tokens (56.3%) hit the on-disk KV cache'. Mooncake and LMCache also tier KV across GPU, CPU and SSD. H100: 80 GB HBM3, 3.35 TB/s (NVIDIA, dated Oct 2026). Block counts, the other students' blocks and the tier drawing are schematic; Episode 8 is prefix caching.")
        tx, tw, th = -3.15, 6.1, 1.2
        ys = (1.5, 0.0, -1.5)
        specs = (("gpu", "GPU memory · HBM", "80 GB · 3.35 TB/s"), ("ram", "CPU memory", "link ≈ 50 GB/s"),
                 ("ssd", "disk · SSD", "biggest · slowest"))
        tiers = m.VGroup()
        for (name, title, spec), y in zip(specs, ys):
            box = card(tw, th, EDGE, CARD, radius=0.16, stroke=2).move_to([tx, y, 0])
            ic = icon(f"{PREFIX}-ssd" if name == "ssd" else recolored(name, MUTED, CARD, PREFIX), 0.6)
            ic.scale(0.78 / ic.width).move_to([tx - tw / 2 + 0.62, y, 0])
            words = m.VGroup(text(title, 24, INK, weight=m.BOLD), code(spec, 20, MUTED)).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.1)
            words.move_to([tx - tw / 2 + 1.25, y, 0], aligned_edge=m.LEFT)
            tiers.add(m.VGroup(box, ic, words))
        cs, cp = 0.28, 0.36

        def slots(y):
            right = tx + tw / 2 - 0.22 - cs / 2
            return [np.array([right - (3 - c) * cp, y + (0.19 if r == 0 else -0.19), 0]) for r in range(2) for c in range(4)]

        gpu_s, cpu_s, ssd_s = slots(ys[0]), slots(ys[1]), slots(ys[2])
        blocks = m.VGroup(*[kv_cell(cs).move_to(p).set_z_index(2) for p in gpu_s])
        others = m.VGroup(*[kv_cell(cs, KV, VIOLET_FILL).set_stroke(REQUEST, 2.5).move_to(gpu_s[i]).set_z_index(2) for i in range(2, 8)])
        down1 = [m.AnimationGroup(*[blocks[i].animate.move_to(cpu_s[i - 2]) for i in range(2, 8)]), ]
        down2 = [blocks[i].animate.move_to(ssd_s[i - 5]) for i in range(5, 8)]
        up = [blocks[i].animate.move_to(gpu_s[i]) for i in range(2, 8)]
        lx = tx + tw / 2 + 0.4
        link = m.Arrow([lx, ys[2] - 0.3, 0], [lx, ys[0] + 0.3, 0], buff=0, color=EDGE, stroke_width=3, max_tip_length_to_length_ratio=0.06)
        flows = m.LaggedStart(*[m.ShowPassingFlash(m.Line([lx, ys[2] - 0.3, 0], [lx, ys[0] + 0.1, 0]).set_stroke(MEMORY, 9),
                                                   time_width=0.5, run_time=1.1) for _ in range(3)], lag_ratio=0.35)
        side = text("each floor down: bigger, cheaper, slower", 20, MUTED).move_to([tx, -2.5, 0])
        legend = m.VGroup(
            m.VGroup(kv_cell(0.26), code("the textbook", 20, KV)).arrange(m.RIGHT, buff=0.15),
            m.VGroup(kv_cell(0.26).set_stroke(REQUEST, 2.5), code("other students", 20, REQUEST)).arrange(m.RIGHT, buff=0.15),
        ).arrange(m.RIGHT, buff=0.5).move_to([tx, 2.55, 0])
        back = code("the student asks again ↑", 20, REQUEST).move_to([tx + 1.2, 2.55, 0])

        px = 3.8
        r_title = text("Bring the textbook back", 26, INK, weight=m.BOLD).move_to([px, 2.1, 0])
        bmax = 3.7
        bx0 = px - 2.45
        r_lab = code(f"reload {BOOK_KV / 1e9:.1f} GB at 50 GB/s", 20, MEMORY).move_to([bx0, 1.45, 0], aligned_edge=m.LEFT)
        r_bar = m.Rectangle(width=max(bmax * RELOAD / ttft(BOOK), 0.06), height=0.42).set_stroke(MEMORY, 2).set_fill(MEMORY, 0.85)
        r_bar.move_to([bx0, 1.0, 0], aligned_edge=m.LEFT)
        r_val = code(f"{RELOAD:.2f} s", 24, INK, weight=m.BOLD).next_to(r_bar, m.RIGHT, buff=0.18)
        p_lab = code("prefill it again", 20, COMPUTE).move_to([bx0, 0.4, 0], aligned_edge=m.LEFT)
        p_bar = m.Rectangle(width=bmax, height=0.42).set_stroke(COMPUTE, 2).set_fill(COMPUTE, 0.6).move_to([bx0, -0.05, 0], aligned_edge=m.LEFT)
        p_val = code(f"{ttft(BOOK):.1f} s", 24, INK, weight=m.BOLD).next_to(p_bar, m.RIGHT, buff=0.18)
        vllm = m.VGroup(code(f"first token {VLLM_GAIN[0]}–{VLLM_GAIN[1]}× sooner", 22, GOOD), cite("vLLM, measured on an H100, 2026"))
        vllm.arrange(m.DOWN, buff=0.1).move_to([px, -0.95, 0])
        ds = m.VGroup(title_text(f"{100 * DEEPSEEK_HIT:.1f}%", 48, KV), text("of input tokens hit\nits on-disk KV cache", 20, INK))
        ds.arrange(m.RIGHT, buff=0.3)
        ds_src = cite(f"DeepSeek, 24 hours, Feb 2025 · {DEEPSEEK_TOKENS / 1e9:.0f}B tokens")
        deep = m.VGroup(ds, ds_src).arrange(m.DOWN, buff=0.12).move_to([px, -2.25, 0])
        ep8 = pill("prefix caching, one floor down → Episode 8", KV, VIOLET_FILL, 22).move_to([0, -3.25, 0])

        self.say(
            "[Remedy two: keep everything, somewhere cheaper.]",
            m.AnimationGroup(self.new_header("05 · offload", "Move it down, not out"),
                             m.LaggedStart(*[m.FadeIn(t, shift=0.15 * m.UP) for t in tiers], lag_ratio=0.25),
                             m.LaggedStart(*[m.FadeIn(b, scale=0.6) for b in blocks], lag_ratio=0.06), m.FadeIn(side), m.FadeIn(link)),
        )
        self.say(
            "[When the GPU fills up, cache blocks slide down into CPU memory], [and then onto disk.]",
            m.AnimationGroup(m.FadeIn(legend), *down1, m.LaggedStart(*[m.FadeIn(o, scale=0.6) for o in others], lag_ratio=0.08), run_time=1.8),
            m.AnimationGroup(*down2, run_time=1.2),
        )
        self.say(
            "[When the student comes back], [the textbook's cache flows back up, at about fifty gigabytes per second.]",
            m.AnimationGroup(m.FadeOut(legend), m.FadeIn(back, shift=0.1 * m.UP), m.FadeOut(others)),
            m.AnimationGroup(*up, flows, run_time=1.8),
        )
        self.say(
            "[That's about a third of a second], [instead of thirteen seconds of prefill.]",
            m.AnimationGroup(m.FadeIn(r_title), m.FadeIn(r_lab), m.GrowFromEdge(r_bar, m.LEFT), m.FadeIn(r_val)),
            m.AnimationGroup(m.FadeIn(p_lab), m.GrowFromEdge(p_bar, m.LEFT, run_time=1.4), m.FadeIn(p_val)),
        )
        self.say(
            "[Moving bytes is cheap]; [redoing all that quadratic math is not.]",
            m.Circumscribe(m.VGroup(r_lab, r_bar, r_val), color=MEMORY, buff=0.1),
            m.Circumscribe(m.VGroup(p_lab, p_bar, p_val), color=COMPUTE, buff=0.1),
        )
        self.say(
            "[In vLLM's tests, loading the cache from CPU memory made the first token two to twenty-two times faster.]",
            m.FadeIn(vllm, shift=0.1 * m.UP),
        )
        self.say(
            "[And even DeepSeek reported more than half of its input tokens served from a cache on disk.]",
            m.AnimationGroup(m.FadeIn(deep, shift=0.1 * m.UP), m.Circumscribe(tiers[2], color=KV, buff=0.06)),
        )
        self.say(
            "[It's prefix caching from episode eight], [one floor down.]",
            m.FadeIn(ep8, scale=0.9),
            m.Circumscribe(tiers[1], color=KV, buff=0.06),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the trade-off

    def trade_off(self):
        self.next_section("6 · Pick your price", notes="The honest trade-off between forgetting and memory. Keep the whole cache: exact, but memory and step time grow with every token (Episodes 5 and 9; only 3 textbook-length chats fit on one H100). Sinks + a window (StreamingLLM): flat memory and speed, but evicted tokens are gone (the code word). Offload (vLLM, LMCache, Mooncake): nothing is lost, but bytes must travel back over a ~50 GB/s link, and the step still reads the whole cache once it is back. The mini drawings are schematic.")
        xs = (-4.45, 0.0, 4.45)
        cw, ch, cy = 4.05, 3.6, -0.4

        def mini_all():
            return m.VGroup(*[kv_cell(0.26) for _ in range(10)]).arrange(m.RIGHT, buff=0.05)

        def mini_window():
            cells = m.VGroup(*[kv_cell(0.26) for _ in range(10)]).arrange(m.RIGHT, buff=0.05)
            for c in cells[2:6]:
                c.set_fill(CARD, 1).set_stroke(FAINT, 1.5)
            for c in cells[:2]:
                c.set_stroke(INK, 2.5)
            return cells

        def mini_down():
            rows = m.VGroup(*[m.RoundedRectangle(corner_radius=0.06, width=2.4 - 0.3 * i, height=0.24).set_stroke(EDGE, 2).set_fill(CARD, 1)
                              for i in range(3)]).arrange(m.DOWN, buff=0.06)
            dots = m.VGroup(kv_cell(0.2).move_to(rows[1]).shift(0.25 * m.LEFT), kv_cell(0.2).move_to(rows[2]).shift(0.1 * m.RIGHT))
            arr = m.Arrow(rows[2].get_right() + [0.2, 0, 0], rows[0].get_right() + [0.2, 0, 0], buff=0, color=MEMORY, stroke_width=4,
                          max_tip_length_to_length_ratio=0.25)
            return m.VGroup(rows, dots, arr)

        data = (
            ("Keep it all", mini_all(), "remembers every token", "memory and time grow\nwith every token"),
            ("Keep a window", mini_window(), "flat memory, flat speed", "the middle is forgotten"),
            ("Move it down", mini_down(), "nothing is lost", "bytes must travel back"),
        )
        cards = []
        for (title, mini, good, bad), x in zip(data, xs):
            box = card(cw, ch, EDGE, CARD, radius=0.2, stroke=2).move_to([x, cy, 0])
            head = title_text(title, 36, INK).move_to([x, 0, 0]).align_to([0, cy + ch / 2 - 0.25, 0], m.UP)
            mini.move_to([x, cy + ch / 2 - 1.32, 0])
            g = m.VGroup(code("✓", 26, GOOD, weight=m.BOLD), text(good, 22, INK)).arrange(m.RIGHT, buff=0.18, aligned_edge=m.UP)
            b = m.VGroup(code("✗", 26, BAD, weight=m.BOLD), text(bad, 22, INK)).arrange(m.RIGHT, buff=0.18, aligned_edge=m.UP)
            lines = m.VGroup(g, b).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.3).move_to([x, 0, 0])
            lines.align_to([x - cw / 2 + 0.35, cy + ch / 2 - 2.05, 0], m.UL)
            cards.append(m.VGroup(box, head, mini, lines))

        self.say(
            "[So every long memory has a price.] [Keep it all, and it costs memory and speed.]",
            self.new_header("06 · trade-off", "Pick your price"),
            m.FadeIn(cards[0], shift=0.2 * m.UP),
        )
        self.say(
            "[Keep a window, and the cost stays flat], [but the middle is forgotten.]",
            m.FadeIn(cards[1][:3], shift=0.2 * m.UP),
            m.FadeIn(cards[1][3], shift=0.1 * m.UP),
        )
        self.say(
            "[Move it down, and nothing is lost], [but the bytes must travel back.]",
            m.FadeIn(cards[2][:3], shift=0.2 * m.UP),
            m.FadeIn(cards[2][3], shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes=f"Quadratic (prefill: every pair of tokens; attention is {100 * share(BOOK):.0f}% of the textbook's prefill, {ttft(BOOK):.1f} s on one H100), linear (decode: every step reads the whole cache; +{SLOPE:.2f} ms per 1,000 tokens measured on this Mac), trade-off (sinks + window forget; offload moves bytes). Next: Episode 10, continuous batching.")
        words = [("Quadratic", COMPUTE, "prefill: every pair of tokens"),
                 ("Linear", MEMORY, "decode: every step reads it all"),
                 ("Trade-off", KV, "forget it, or move it down")]
        cols = m.VGroup()
        for w, color, sub in words:
            cols.add(m.VGroup(title_text(w, 66, color), text(sub, 22, MUTED)).arrange(m.DOWN, buff=0.25))
        cols.arrange(m.RIGHT, buff=1.15).move_to([0, 0.75, 0])
        base = 0.85
        for c in cols:
            c[0].shift((base - c[0].get_bottom()[1]) * m.UP)
            c[1].move_to([c[0].get_x(), base - 0.55, 0])
        mid = base + 0.3
        arrows = m.VGroup(*[
            arrow([a[0].get_right()[0] + 0.12, mid, 0], [b[0].get_left()[0] - 0.12, mid, 0], MUTED) for a, b in zip(cols, cols[1:])
        ])
        tagline = text("A long memory costs twice: once to read it, then on every step.", 30, INK).move_to([0, -1.35, 0])
        mascot = owl(0.75).move_to([0, -2.75, 0])

        self.say(
            "[So: quadratic], [linear], [and a trade-off.]",
            m.AnimationGroup(self.new_header("recap", "Long memory in three words"), m.FadeIn(cols[0], shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowArrow(arrows[0]), m.FadeIn(cols[1], shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowArrow(arrows[1]), m.FadeIn(cols[2], shift=0.2 * m.UP)),
        )
        self.say(
            "[A long memory costs twice]: [once to read it, and again on every step.]",
            m.FadeIn(tagline, shift=0.1 * m.UP),
            m.FadeIn(mascot, shift=0.2 * m.UP),
        )
        self.wait(0.4)
        self.clear()
