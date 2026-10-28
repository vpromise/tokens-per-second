"""Tokens per Second · Episode 5 · The KV Cache (Module 2 · The KV Cache)

The model remembers past tokens as keys and values, and that memory has a size. Running example: Owl-8B
(Llama-3.1-8B: 8 B parameters, 16 GB of BF16 weights, 128 KiB of KV cache per token).

Sections: hook (Owl answering a student; every token leaves a key and a value) · attention looks back (the
newest token's query meets every earlier key and mixes their values, in each of 32 layers) · recompute or
remember (a toy 8-step triangle vs a line; Owl's 1,500 + 300-token request: 496,650 vs 1,800 token passes) ·
cache == recompute (owl_reference's TinyLM: five decode steps with the cache vs recomputing from scratch,
run in this file) · the size formula (2 · L · n_kv · d_h · b = 131,072 B = 128 KiB per token; GQA: 4x
smaller than 32 KV heads) · growth (a 32-layer column per token; 1 GiB at 8K, 4 GiB at 32K, 16 GiB at 128K
≈ the weights) · measured on this Mac (active memory vs context, slope 131,072 B/token = the formula) ·
capacity (one H100: 52 students at 8K tokens, 3 at 128K; compute would allow ≈ 295) · recap.

Sources (facts dated October 2026):
- research/notes.md §3 (KV cache): mechanism (O(T²) recompute vs append), the formula 2·L·n_kv·d_h·b (JAX
  scaling book: "the 2 comes from storing both the keys and values"; kipply 2022), the worked table (8B:
  128 KiB, 1 / 4 / 16 GiB at 8K / 32K / 128K; MHA 512 KiB) and the H100 worked example (90% of 80 GB, minus
  16.06 GB of weights ≈ 56 GB ≈ 427,000 token slots → 52 conversations at 8K, 3 at 128K; activations and
  CUDA-graph workspace ignored, so real numbers are a bit lower).
- Llama-3.1-8B config (HF config.json, Llama 3 paper Table 3): 32 layers, 32 query heads, 8 KV heads, head
  dim 128; "with 8 key-value heads to improve inference speed and to reduce the size of key-value caches
  during decoding" (Llama 3 Herd of Models, 2024). Context window 128K tokens.
- H100 SXM: 80 GB HBM3, 3.35 TB/s, 989 TFLOPS BF16 dense (NVIDIA, dated Oct 2026); ridge ≈ 295 FLOPs/byte
  (notes §2: critical batch ≈ ridge for BF16; Episode 4).
- research/measurements.md §4 (measured on this Mac Studio, M4 Max, mlx-lm 0.32.2, 2026-10-24):
  Llama-3.1-8B 4-bit weights with a BF16 cache, active memory at 1K…32K tokens; least-squares slope
  131,072 B/token = the formula, exactly; 4.52 GB of weights → 8.85 GB active at 32K.
- Episode 1's request: a 1,500-token prompt and a 300-token answer (COURSE.md).
- owl_reference.py: every number via kv_bytes_per_token, weight_bytes, ridge, measured; TinyLM's
  cache == recompute check is the file's own self-check, re-run below.

Illustrative (said as such or schematic): the hook's tokenisation ("What is osmosis?" → "Water moves
through a membrane"), the attention weights' line widths, the 8-step toy grids, the 44-token grid.

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

PREFIX = "ep05"
KIB, MIB, GIB = 2**10, 2**20, 2**30

# the size formula, filled with Owl's config
KV_TOKEN = int(kv_bytes_per_token())
FACTORS = [2, OWL["layers"], OWL["kv_heads"], OWL["head_dim"], 16 // 8]
KV_CELL = KV_TOKEN // OWL["layers"]  # one layer's keys and values, one token
MHA_TOKEN = int(kv_bytes_per_token(dict(OWL, kv_heads=OWL["heads"])))
CONTEXT = 128 * 1024  # Llama 3.1's context window: 128K tokens
assert KV_TOKEN == 131_072 == 128 * KIB and int(np.prod(FACTORS)) == KV_TOKEN  # "131,072 bytes, 128 kibibytes"
assert FACTORS == [2, 32, 8, 128, 2]  # "thirty-two layers, eight key-value heads, 128 numbers, two bytes"
assert KV_CELL == 4 * KIB  # "each cell is four kibibytes"
assert OWL["heads"] == 32 and MHA_TOKEN == 4 * KV_TOKEN  # "shared by thirty-two query heads … four times more"
assert KV_TOKEN * 8192 == GIB and KV_TOKEN * 32768 == 4 * GIB and KV_TOKEN * CONTEXT == 16 * GIB
assert 1.0 < KV_TOKEN * CONTEXT / weight_bytes() < 1.1  # 17.2 GB vs 16.06 GB: "about as much as the weights"
GRID_TOKENS = 44
assert GRID_TOKENS * KV_TOKEN == 5.5 * MIB

# recompute vs remember: Episode 1's request, counted in token passes through the model
PROMPT_N, ANSWER_N = 1500, 300
RECOMPUTE = PROMPT_N + sum(PROMPT_N + i for i in range(1, ANSWER_N + 1))  # every step re-runs the whole sequence
CACHED = PROMPT_N + ANSWER_N  # prefill once, then one new token per step
RATIO = RECOMPUTE / CACHED
assert RECOMPUTE == 496_650 and CACHED == 1_800 and round(RECOMPUTE, -5) == 500_000  # "half a million … eighteen hundred"
assert 270 < RATIO < 280  # "more than two hundred and seventy times the work"
TOY = 8
assert sum(range(1, TOY + 1)) == 36

# cache == recompute: owl_reference's TinyLM, run here
TINY = TinyLM()
TINY_PROMPT = [3, 14, 15, 9, 2, 6]


def tiny_run(steps=5):
    logits, cache = TINY.prefill(TINY_PROMPT)
    seq, rows = list(TINY_PROMPT), []
    for step in range(1, steps + 1):
        tok = sample(logits, 0)
        seq.append(tok)
        logits = TINY.decode(tok, cache)
        fresh = TINY.forward(seq, KVCache())[-1]
        assert np.allclose(logits, fresh) and np.argmax(logits) == np.argmax(fresh)
        rows.append((step, tok, len(cache), float(np.abs(logits - fresh).max())))
    return rows


RUN = tiny_run()
WORST = max(r[3] for r in RUN)
assert [r[2] for r in RUN] == [7, 8, 9, 10, 11]
assert 0 < WORST < 1e-14 and -int(np.floor(np.log10(WORST))) == 15  # "rounding noise in the fifteenth decimal place"

# measured on this Mac: memory vs context (Llama-3.1-8B, 4-bit weights, BF16 cache)
MEM = measured("memory", "llama-3.1-8b", "4bit", "kv16")
POINTS = sorted((int(t), r["active_GB"]) for t, r in MEM["by_context_tokens"].items())
FIT = np.polyfit([t for t, _ in POINTS], [g * 1e9 for _, g in POINTS], 1)
SLOPE = MEM["slope_active_bytes_per_token"]
W4 = MEM["weights_active_GB"]
assert len(POINTS) == 6  # "at six context lengths"
assert SLOPE == MEM["formula_bytes_per_token"] == KV_TOKEN and abs(FIT[0] - KV_TOKEN) < 1  # "exactly the formula"
assert measured("headline", "kv_slope_8b_bytes_per_token") == KV_TOKEN
MINI_SLOPE = int(measured("headline", "kv_slope_1b_bytes_per_token"))
assert MINI_SLOPE == kv_bytes_per_token(MODELS["owl-mini"]) == 32 * KIB  # "Owl-mini … thirty-two kibibytes per token"
assert POINTS[-1][0] * KV_TOKEN == 4 * GIB and 1.9 < POINTS[-1][1] / W4 < 2.0  # "four gibibytes, nearly doubling"

# one H100 (notes §3 worked example)
BUDGET = 0.9 * GPU["memory"]
KV_ROOM = BUDGET - weight_bytes()
SLOTS = KV_ROOM / KV_TOKEN
CHAT = 8192
USERS, USERS_LONG = int(SLOTS // CHAT), int(SLOTS // CONTEXT)
CRIT = ridge(GPU)
assert GPU["memory"] == 80e9 and round(weight_bytes() / 1e9) == 16  # "eighty gigabytes", "sixteen"
assert round(KV_ROOM / 1e9) == 56 and round(SLOTS, -3) == 427_000  # "fifty-six gigabytes … 427 thousand tokens"
assert USERS == 52 and USERS_LONG == 3 and round(CRIT) == 295  # "fifty-two", "three", "about three hundred"

# ---------------------------------------------------------------------------- look


def cite(s):
    return text(s, 20, MUTED)


def kv_cell(size=0.42, color=KV, fill=VIOLET_FILL):
    return m.RoundedRectangle(corner_radius=0.07, width=size, height=size, stroke_color=color, stroke_width=2.5,
                              fill_color=fill, fill_opacity=1)


def nice_bytes(n):
    if n >= GIB:
        return f"{n / GIB:g} GiB"
    if n >= MIB:
        return f"{n / MIB:g} MiB"
    return f"{n / KIB:g} KiB"


def tick_bracket(x, y0, y1, color, lip=0.12):
    """A vertical bracket with lips pointing left."""
    return m.VGroup(
        m.Line([x, y0, 0], [x, y1, 0]), m.Line([x, y0, 0], [x - lip, y0, 0]), m.Line([x, y1, 0], [x - lip, y1, 0])
    ).set_stroke(color, 2.5)


# hook and attention: an illustrative exchange (tokenisation simplified)
PROMPT_TOKS = ["What", " is", " osmosis", "?"]
ANSWER_TOKS = ["Water", " moves", " through", " a", " membrane"]
ATTN = [0.04, 0.03, 0.31, 0.03, 0.17, 0.12, 0.08, 0.04, 0.18]  # illustrative attention of "membrane"
assert abs(sum(ATTN) - 1) < 1e-9 and len(ATTN) == len(PROMPT_TOKS) + len(ANSWER_TOKS)


def exchange(size):
    """The student's tokens (PINK) followed by Owl's (GREEN), in one row."""
    ask = token_row(PROMPT_TOKS, REQUEST, size=size)
    ans = token_row(ANSWER_TOKS, TOKEN, size=size)
    return ask, ans


# ---------------------------------------------------------------------------- the episode


class Ep05TheKVCache(Episode):
    number = 5

    def lesson(self):
        self.hook()
        self.looks_back()
        self.recompute()
        self.exact()
        self.formula()
        self.growth()
        self.on_this_mac()
        self.capacity()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="Owl answers one token at a time; to choose each next token it attends to every earlier token. Instead of rereading them, it keeps a key and a value per token (and per layer): the KV cache. Teasers: 128 KiB per token (kv_bytes_per_token), 52 students per H100 at 8K tokens each (notes §3 worked example). The exchange and its tokenisation are illustrative.")
        ask, ans = exchange(24)
        ref = token_row(["Mx"], MUTED, size=24)[0]
        slot = card(0.72, ref.height, MUTED, CARD, radius=0.08, stroke=1.8)
        nxt = m.VGroup(m.DashedVMobject(slot, num_dashes=14),
                       m.VGroup(*[m.Dot(radius=0.045, color=MUTED) for _ in range(3)]).arrange(m.RIGHT, buff=0.08))
        row = m.VGroup(*ask, *ans, nxt).arrange(m.RIGHT, buff=0.14).move_to([0.25, 0.7, 0])
        chips = m.VGroup(*ask, *ans)
        arcs = m.VGroup(*[
            m.ArcBetweenPoints(nxt.get_top() + 0.06 * m.UP, c.get_top() + 0.06 * m.UP, angle=0.85).set_stroke(TOKEN, 2.5, opacity=0.85)
            for c in chips
        ])
        ks = m.VGroup(*[kv_cell(0.36).move_to([c.get_x(), -0.3, 0]) for c in chips])
        vs = m.VGroup(*[kv_cell(0.36).move_to([c.get_x(), -0.78, 0]) for c in chips])
        k_lab = code("K", 22, KV, weight=m.BOLD).move_to([row.get_left()[0] - 0.35, -0.3, 0])
        v_lab = code("V", 22, KV, weight=m.BOLD).move_to([row.get_left()[0] - 0.35, -0.78, 0])
        notes = m.VGroup(*[m.VGroup(k, v) for k, v in zip(ks, vs)])
        box = m.SurroundingRectangle(m.VGroup(ks, vs, k_lab, v_lab), buff=0.14, corner_radius=0.14).set_stroke(KV, 2.5)
        tag = pill("KV cache", KV, VIOLET_FILL, 24).next_to(box, m.DOWN, buff=0.3)
        tease = m.VGroup(
            pill(f"{nice_bytes(KV_TOKEN)} per token", KV, VIOLET_FILL, 24, mono=True),
            pill(f"{USERS} students per H100", REQUEST, PINK_FILL, 24, mono=True),
        ).arrange(m.RIGHT, buff=0.6).move_to([-0.4, -2.85, 0])
        mascot = owl(0.62).move_to([5.75, -2.8, 0])

        self.say(
            "[A student asks Owl a question], [and Owl answers, one token at a time.]",
            m.AnimationGroup(self.new_header("hook", "What does Owl remember?"),
                             m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.UP) for c in ask], lag_ratio=0.15),
                             m.FadeIn(mascot, shift=0.2 * m.UP)),
            m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.RIGHT) for c in ans], lag_ratio=0.35),
        )
        self.say(
            "[To choose the next one], [it looks back at every token so far.]",
            m.FadeIn(nxt, scale=0.8),
            m.LaggedStart(*[m.Create(a) for a in reversed(arcs)], lag_ratio=0.12),
        )
        self.say(
            "[It doesn't reread them]: [for each token, it kept a small note, a key and a value.]",
            arcs.animate.set_stroke(opacity=0.25),
            m.AnimationGroup(m.FadeIn(k_lab), m.FadeIn(v_lab),
                             m.LaggedStart(*[m.FadeIn(n, shift=0.15 * m.DOWN) for n in notes], lag_ratio=0.1)),
        )
        self.say(
            "[Those notes are the KV cache.] [Today: how big it gets], [and what it limits.]",
            m.AnimationGroup(m.Create(box), m.FadeIn(tag, shift=0.1 * m.UP)),
            m.FadeIn(tease[0], shift=0.15 * m.UP),
            m.FadeIn(tease[1], shift=0.15 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ attention looks back

    def looks_back(self):
        self.next_section("1 · Attention looks back", notes="Inside one layer, the newest token's query is compared with the key of every earlier token (and its own); the values are mixed, weighted by those matches (attention; covered in AI, Explained, episode 24). Owl-8B has 32 layers, each with its own keys and values for every token, so every past K and V is needed again at every decode step (notes §3). Line widths (attention weights) are illustrative; the strongest match is 'osmosis'.")
        ask, ans = exchange(22)
        chips = m.VGroup(*ask, *ans).arrange(m.RIGHT, buff=0.16).move_to([0.45, 0.2, 0])
        new = chips[-1]
        ks = m.VGroup(*[kv_cell().move_to([c.get_x(), 1.05, 0]) for c in chips])
        vs = m.VGroup(*[kv_cell().move_to([c.get_x(), -0.65, 0]) for c in chips])
        q = kv_cell(color=TOKEN, fill=GREEN_FILL).move_to([new.get_x(), 2.05, 0])
        q_words = code("q", 22, TOKEN, weight=m.BOLD).move_to(q)
        out = kv_cell(color=TOKEN, fill=GREEN_FILL).move_to([new.get_x(), -1.65, 0])
        out_words = code("out", 20, TOKEN, weight=m.BOLD).next_to(out, m.LEFT, buff=0.18)
        left = chips.get_left()[0] - 0.3
        k_lab = text("keys", 22, KV, weight=m.BOLD).move_to([left, 1.05, 0], aligned_edge=m.RIGHT)
        v_lab = text("values", 22, KV, weight=m.BOLD).move_to([left, -0.65, 0], aligned_edge=m.RIGHT)
        q_lab = text("query", 22, TOKEN, weight=m.BOLD).next_to(q, m.LEFT, buff=0.22)
        q_lines = m.VGroup(*[
            m.Line(q.get_bottom(), k.get_top()).set_stroke(TOKEN, 1.5 + 14 * w, opacity=0.9) for k, w in zip(ks, ATTN)
        ])
        v_lines = m.VGroup(*[
            m.Line(v.get_bottom(), out.get_top()).set_stroke(KV, 1.5 + 14 * w, opacity=0.9) for v, w in zip(vs, ATTN)
        ])
        best = int(np.argmax(ATTN))
        layers = m.VGroup(*[
            m.Square(0.2).set_stroke(KV, 1.5).set_fill(VIOLET_FILL, 1) for _ in range(OWL["layers"])
        ]).arrange(m.RIGHT, buff=0.05).move_to([-0.2, -2.95, 0])
        l_first = code("layer 1", 20, MUTED).next_to(layers, m.LEFT, buff=0.25)
        l_last = code(f"layer {OWL['layers']}", 20, MUTED).next_to(layers, m.RIGHT, buff=0.25)
        l_cap = text(f"Owl-8B: {OWL['layers']} layers, each with its own keys and values", 22, INK).move_to([-0.2, -2.4, 0])
        panel = m.VGroup(chips[:-1], ks[:-1], vs[:-1], k_lab, v_lab)

        self.say(
            "[Inside one layer], [the newest token makes a query],",
            m.AnimationGroup(self.new_header("01 · attention", "Every token, every layer"),
                             m.FadeIn(panel, shift=0.15 * m.UP), m.FadeIn(new, shift=0.15 * m.LEFT)),
            m.AnimationGroup(m.FadeIn(q, shift=0.2 * m.DOWN), m.FadeIn(q_words, shift=0.2 * m.DOWN), m.FadeIn(q_lab),
                             m.FadeIn(ks[-1], shift=0.1 * m.DOWN), m.FadeIn(vs[-1], shift=0.1 * m.UP)),
        )
        self.say(
            "[and compares it with the key of every earlier token],",
            m.LaggedStart(*[m.Create(line) for line in reversed(q_lines)], lag_ratio=0.12),
        )
        self.say(
            "[then mixes their values], [weighted by how well each one matched.]",
            m.AnimationGroup(m.LaggedStart(*[m.Create(line) for line in v_lines], lag_ratio=0.08),
                             m.FadeIn(out, shift=0.15 * m.DOWN), m.FadeIn(out_words)),
            m.Circumscribe(m.VGroup(ks[best], chips[best], vs[best]), color=TOKEN, buff=0.08),
        )
        self.say(
            "[Owl does this in all thirty-two layers], [and each layer keeps its own keys and values.]",
            m.AnimationGroup(m.FadeIn(l_cap, shift=0.1 * m.UP), m.FadeIn(l_first), m.FadeIn(l_last),
                             m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.UP) for c in layers], lag_ratio=0.04)),
            m.LaggedStart(*[c.animate.set_fill(KV, 1) for c in layers], lag_ratio=0.05),
        )
        self.say(
            "[So at every step, all those keys and values are needed again.]",
            m.AnimationGroup(m.Circumscribe(m.VGroup(k_lab, ks), color=KV, buff=0.1),
                             m.Circumscribe(m.VGroup(v_lab, vs), color=KV, buff=0.1)),
        )
        kv_rows = m.VGroup(ks, vs)
        self.say(
            "[Throw them away], [and every one of them must be computed again.]",
            kv_rows.animate(run_time=0.8).set_opacity(0.15),
            m.LaggedStart(*[m.AnimationGroup(k.animate.set_opacity(1).set_stroke(COMPUTE), v.animate.set_opacity(1).set_stroke(COMPUTE))
                            for k, v in zip(ks, vs)], lag_ratio=0.12),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recompute or remember

    def recompute(self):
        self.next_section("2 · Recompute or remember", notes=f"Without a cache, every decode step would run the whole sequence through the model again: the work is a triangle, O(T²) (notes §3). With a cache, each token's keys and values are computed once and read back afterwards: O(T) computed, plus reads. Owl's request from Episode 1 (1,500-token prompt, 300-token answer), counted in token passes: recompute = 1,500 + Σ(1,501…1,800) = {RECOMPUTE:,}; cache = {CACHED:,}; ratio {RATIO:.1f}x. The cache trades compute for memory; every engine keeps one (vLLM, SGLang, llama.cpp, MLX). The 8-step grids are a toy.")
        size, gap = 0.34, 0.06
        pitch = size + gap

        def triangle(cx, cy):
            cells = {}
            x0 = cx - (TOY - 1) * pitch / 2
            y0 = cy + (TOY - 1) * pitch / 2
            for r in range(TOY):
                for c in range(r + 1):
                    sq = m.Square(size).set_stroke(EDGE, 1.5).set_fill(CARD, 1).move_to([x0 + c * pitch, y0 - r * pitch, 0])
                    cells[r, c] = sq
            return cells

        lc, rc = (-3.4, 0.3), (3.4, 0.3)
        left, right = triangle(*lc), triangle(*rc)
        l_all = m.VGroup(*left.values())
        r_all = m.VGroup(*right.values())
        l_title = title_text("Recompute", 34, COMPUTE).move_to([lc[0], 2.3, 0])
        r_title = title_text("Remember", 34, KV).move_to([rc[0], 2.3, 0])
        l_step = code("step", 20, MUTED).next_to(left[0, 0], m.LEFT, buff=0.3)
        r_step = code("step", 20, MUTED).next_to(right[0, 0], m.LEFT, buff=0.3)
        l_rows = [m.VGroup(*[left[r, c] for c in range(r + 1)]) for r in range(TOY)]
        r_new = [right[r, r] for r in range(TOY)]
        r_old = [right[r, c] for r in range(TOY) for c in range(r)]
        l_count = code(f"1 + 2 + … + {TOY} = {sum(range(1, TOY + 1))} computed", 20, COMPUTE).move_to([lc[0], -1.6, 0])
        l_grow = text("grows with the length squared", 22, INK).move_to([lc[0], -2.08, 0])
        r_count = code(f"{TOY} computed · {sum(range(TOY))} read", 20, INK, t2c={r"\d+ computed": COMPUTE, r"\d+ read": KV}).move_to([rc[0], -1.6, 0])
        r_grow = text("grows in a straight line", 22, INK).move_to([rc[0], -2.08, 0])
        owl_line = m.VGroup(
            code(f"Owl · {PROMPT_N:,}-token prompt + {ANSWER_N}-token answer", 20, MUTED),
            code(f"{RECOMPUTE:,}", 26, COMPUTE, weight=m.BOLD),
            code("vs", 24, MUTED),
            code(f"{CACHED:,}", 26, KV, weight=m.BOLD),
            code("token passes", 22, MUTED),
        )
        owl_line[1:].arrange(m.RIGHT, buff=0.25)
        ratio = title_text(f"{RATIO:.0f}× the work", 40, INK)
        bottom = m.VGroup(owl_line[1:], ratio).arrange(m.RIGHT, buff=0.6).move_to([0, -3.2, 0])
        owl_line[0].move_to([0, -2.68, 0])
        trade = m.VGroup(code("compute", 24, COMPUTE, weight=m.BOLD), code("↓", 30, MUTED), code("memory", 24, KV, weight=m.BOLD))
        trade.arrange(m.DOWN, buff=0.15).move_to([0, 0.75, 0])
        engines = m.VGroup(code("vLLM · SGLang", 20, MUTED), code("llama.cpp · MLX", 20, MUTED)).arrange(m.DOWN, buff=0.12)
        engines.next_to(trade, m.DOWN, buff=0.35)

        self.say(
            "[Option one: recompute.] [Every step runs the whole conversation through the model again.]",
            m.AnimationGroup(self.new_header("02 · recompute", "Recompute, or remember?"),
                             m.FadeIn(l_title, shift=0.1 * m.UP), m.FadeIn(l_all), m.FadeIn(l_step)),
            m.LaggedStart(*[row.animate.set_fill(COMPUTE, 0.9).set_stroke(COMPUTE, 1.5) for row in l_rows], lag_ratio=0.3),
        )
        self.say(
            "[The work piles up like a triangle]: [it grows with the square of the length.]",
            m.FadeIn(l_count, shift=0.1 * m.UP),
            m.FadeIn(l_grow, shift=0.1 * m.UP),
        )
        self.say(
            "[Option two: remember.] [Compute each token's key and value once], [and read the rest from the cache.]",
            m.AnimationGroup(m.FadeIn(r_title, shift=0.1 * m.UP), m.FadeIn(r_all), m.FadeIn(r_step)),
            m.LaggedStart(*[c.animate.set_fill(COMPUTE, 0.9).set_stroke(COMPUTE, 1.5) for c in r_new], lag_ratio=0.25),
            m.LaggedStart(*[c.animate.set_fill(KV, 0.9).set_stroke(KV, 1.5) for c in r_old], lag_ratio=0.03),
        )
        self.say(
            "[Now the computing grows in a straight line.]",
            m.AnimationGroup(m.FadeIn(r_count, shift=0.1 * m.UP), m.FadeIn(r_grow, shift=0.1 * m.UP)),
        )
        self.say(
            "[For Owl's fifteen-hundred-token prompt and three-hundred-token answer], [that's half a million token passes against eighteen hundred.]",
            m.FadeIn(owl_line[0], shift=0.1 * m.UP),
            m.FadeIn(owl_line[1:], shift=0.1 * m.UP),
        )
        self.say("[More than two hundred and seventy times the work.]", m.FadeIn(ratio, scale=0.85))
        self.say(
            "[The cache trades compute for memory], [and every inference engine makes that trade.]",
            m.LaggedStart(*[m.FadeIn(t, shift=0.1 * m.DOWN) for t in trade], lag_ratio=0.3),
            m.FadeIn(engines, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ cache == recompute

    def exact(self):
        worst = f"{WORST:.1e}"
        self.next_section("3 · Cache equals recompute", notes=f"owl_reference.py's TinyLM (2 layers, 4 query heads, 2 KV heads, head_dim 4, float64), prompt {TINY_PROMPT}: five greedy decode steps with the cache, and at each step a full recompute from an empty cache. Run in this scene (and asserted in owl_reference's self-check): the logits agree at every step, max |difference| {worst} (rounding in the 15th decimal place), same argmax tokens {[r[1] for r in RUN]}; the cache grows 7 → 11 tokens. Real engines in BF16 may differ by rounding between cached and recomputed paths (different summation order); mathematically the cache is exact.")
        src = ("prompt = [3, 14, 15, 9, 2, 6]\n"
               "logits, cache = tiny.prefill(prompt)\n"
               "seq = list(prompt)\n"
               "for _ in range(5):\n"
               "    tok = sample(logits, 0)\n"
               "    seq.append(tok)\n"
               "    logits = tiny.decode(tok, cache)\n"
               "    fresh = tiny.forward(seq, KVCache())\n"
               "    assert np.allclose(logits, fresh[-1])")
        assert len(src.splitlines()) <= 10
        block = code_block(src, 20, title="owl_reference.py · TinyLM").move_to([-3.05, 0.35, 0])
        body = block[1]
        hl_cache = glyphs(body, "tiny.decode(tok, cache)")
        hl_fresh = glyphs(body, "tiny.forward(seq, KVCache())")
        tag_cache = code("with the cache", 20, KV).next_to(block, m.DOWN, buff=0.3).align_to(block, m.LEFT)
        tag_fresh = code("recomputed from scratch", 20, COMPUTE).next_to(tag_cache, m.RIGHT, buff=0.5)

        # the run, as a table
        box = card(5.2, 4.1, EDGE, CARD, radius=0.16, stroke=2).move_to([3.95, 0.55, 0])
        title = code("cache vs recompute", 22, INK, t2c={"cache": KV, "recompute": COMPUTE}, weight=m.BOLD)
        title.align_to(box, m.UL).shift(0.35 * m.RIGHT + 0.25 * m.DOWN)
        cols = [box.get_left()[0] + d for d in (0.7, 1.75, 2.85, 4.2)]
        heads = m.VGroup(*[code(h, 20, MUTED).move_to([x, box.get_top()[1] - 1.05, 0]) for h, x in zip(("step", "token", "cache", "max diff"), cols)])
        rule = m.Line([box.get_left()[0] + 0.3, box.get_top()[1] - 1.35, 0], [box.get_right()[0] - 0.3, box.get_top()[1] - 1.35, 0]).set_stroke(EDGE, 2)
        rows = m.VGroup()
        for i, (step, tok, n, diff) in enumerate(RUN):
            y = box.get_top()[1] - 1.7 - 0.48 * i
            rows.add(m.VGroup(
                code(str(step), 22, MUTED).move_to([cols[0], y, 0]),
                code(str(tok), 22, TOKEN).move_to([cols[1], y, 0]),
                code(str(n), 22, KV).move_to([cols[2], y, 0]),
                code(f"{diff:.1e}", 22, INK).move_to([cols[3], y, 0]),
            ))
        table = m.VGroup(box, title, heads, rule)
        diffs = m.VGroup(*[r[3] for r in rows])
        verdict = pill("✓ cache == recompute, 5 of 5 steps", GOOD, GOOD_FILL, 24).move_to([3.95, -2.2, 0])
        note = cite("asserted in owl_reference.py's self-check").next_to(verdict, m.DOWN, buff=0.25)

        self.say(
            "[Is the cache an approximation?] [Let's test it on the tiny model in our reference code.]",
            self.new_header("03 · exact", "Cache equals recompute"),
            m.FadeIn(block, shift=0.15 * m.UP),
        )
        self.say(
            "[It decodes five tokens with the cache], [and at each step, it also recomputes everything from scratch.]",
            m.AnimationGroup(hl_cache.animate.set_color(KV), m.FadeIn(tag_cache)),
            m.AnimationGroup(hl_fresh.animate.set_color(COMPUTE), m.FadeIn(tag_fresh)),
        )
        self.say(
            "[The logits agree at every step]: [the only differences are rounding noise, in the fifteenth decimal place.]",
            m.AnimationGroup(m.FadeIn(table), m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.UP) for r in rows], lag_ratio=0.25)),
            m.Circumscribe(diffs, color=INK, buff=0.1),
        )
        self.say(
            "[So the cache is no approximation]: [same logits, same tokens, and a fraction of the work.]",
            m.FadeIn(verdict, scale=0.9),
            m.FadeIn(note, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the size formula

    def formula(self):
        self.next_section("4 · The size formula", notes=f"KV bytes per token = 2 (K and V) × L layers × n_kv KV heads × d_h head dim × b bytes (notes §3; JAX scaling book, kipply 2022). Owl-8B (Llama-3.1-8B config, owl_reference MODELS): 2 × 32 × 8 × 128 × 2 = {KV_TOKEN:,} B = 128 KiB per token (kv_bytes_per_token). GQA: 8 KV heads shared by 32 query heads; with 32 KV heads (MHA) it would be {MHA_TOKEN // KIB} KiB, 4× more (notes §3 table). Llama 3 paper (2024): 'with 8 key-value heads … to reduce the size of key-value caches during decoding'. Shrinking tricks are Episode 6.")
        xs = [-5.0, -2.5, 0.0, 2.5, 5.0]
        symbols = ["2", "L", "n_kv", "d_h", "b"]
        labels = ["K and V", "layers", "KV heads", "numbers per head", "bytes · BF16"]
        sym = m.VGroup(*[code(s, 44, KV, weight=m.BOLD).move_to([x, 1.75, 0]) for s, x in zip(symbols, xs)])
        times_top = m.VGroup(*[code("×", 36, MUTED).move_to([(a + b) / 2, 1.75, 0]) for a, b in zip(xs, xs[1:])])
        vals = [title_text(str(v), 64, INK).move_to([x, 0.75, 0]) for v, x in zip(FACTORS, xs)]
        times_mid = m.VGroup(*[code("×", 36, MUTED).move_to([(a + b) / 2, 0.75, 0]) for a, b in zip(xs, xs[1:])])
        labs = [text(s, 22, MUTED).move_to([x, 0.0, 0]) for s, x in zip(labels, xs)]
        res_bytes = title_text(f"= {KV_TOKEN:,} bytes", 52, INK)
        res_kib = m.VGroup(title_text(f"= {nice_bytes(KV_TOKEN)}", 64, KV), text("per token", 26, MUTED))
        res_kib[1].next_to(res_kib[0], m.RIGHT, buff=0.25).align_to(res_kib[0], m.DOWN)
        result = m.VGroup(res_bytes, res_kib).arrange(m.RIGHT, buff=0.6).move_to([0, -1.25, 0])
        gqa = text(f"{OWL['kv_heads']} KV heads, shared by {OWL['heads']} query heads", 24, INK).move_to([0, -2.45, 0])
        mha = code(f"with {OWL['heads']} KV heads: {nice_bytes(MHA_TOKEN)} per token, 4× more", 20, MUTED).next_to(gqa, m.DOWN, buff=0.22)
        src = cite("Llama-3.1-8B config · Llama 3 paper, 2024").move_to([0, -3.4, 0])
        cols = [m.VGroup(v, l) for v, l in zip(vals, labs)]

        self.say(
            "[How big is it?] [Each token stores a key and a value],",
            m.AnimationGroup(self.new_header("04 · the formula", "How big is one token?"), m.FadeIn(sym), m.FadeIn(times_top)),
            m.FadeIn(cols[0], shift=0.2 * m.DOWN),
        )
        self.say(
            "[in each of thirty-two layers], [for each of eight key-value heads],",
            m.AnimationGroup(m.FadeIn(cols[1], shift=0.2 * m.DOWN), m.FadeIn(times_mid[0])),
            m.AnimationGroup(m.FadeIn(cols[2], shift=0.2 * m.DOWN), m.FadeIn(times_mid[1])),
        )
        self.say(
            "[each a vector of a hundred and twenty-eight numbers], [at two bytes per number.]",
            m.AnimationGroup(m.FadeIn(cols[3], shift=0.2 * m.DOWN), m.FadeIn(times_mid[2])),
            m.AnimationGroup(m.FadeIn(cols[4], shift=0.2 * m.DOWN), m.FadeIn(times_mid[3])),
        )
        self.say(
            "[Multiply it out]: [a hundred and thirty-one thousand and seventy-two bytes], [exactly a hundred and twenty-eight kibibytes per token.]",
            m.Circumscribe(m.VGroup(*vals, times_mid), color=KV, buff=0.12),
            m.FadeIn(res_bytes, shift=0.15 * m.UP),
            m.FadeIn(res_kib, scale=0.85),
        )
        self.say(
            "[Only eight key-value heads, shared by thirty-two query heads]: [without that sharing, it would be four times bigger.]",
            m.AnimationGroup(m.Circumscribe(cols[2], color=KV, buff=0.1), m.FadeIn(gqa, shift=0.1 * m.UP), m.FadeIn(src)),
            m.FadeIn(mha, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ growth

    def growth(self):
        self.next_section("5 · Token by token", notes=f"The cache grows by one column per token: {OWL['layers']} layers, each cell = one layer's K and V for one token = 2 × 8 × 128 × 2 B = 4 KiB; a column = 128 KiB. {GRID_TOKENS} tokens = {nice_bytes(GRID_TOKENS * KV_TOKEN)}. 8K tokens = 1 GiB, 32K = 4 GiB, 128K (Llama 3.1's full context) = 16 GiB = 17.2 GB, about the size of the 16.06 GB of weights (notes §0 #7, §3 table). The 44-column grid is schematic.")
        rows_n, cols_n = OWL["layers"], GRID_TOKENS
        size, gap = 0.085, 0.03
        pitch = size + gap
        grid = m.VGroup(*[
            m.VGroup(*[m.Square(size).set_stroke(width=0).set_fill(EDGE, 0.7) for _ in range(cols_n)]).arrange(m.RIGHT, buff=gap)
            for _ in range(rows_n)
        ]).arrange(m.DOWN, buff=gap).move_to([-2.85, 0.05, 0])
        col = lambda j: [grid[r][j] for r in reversed(range(rows_n))]  # layer 1 (bottom) first
        top_lab = code(f"layer {rows_n}", 20, MUTED).next_to(grid[0], m.LEFT, buff=0.22)
        bot_lab = code("layer 1", 20, MUTED).next_to(grid[-1], m.LEFT, buff=0.22)
        tok_lab = code("tokens →", 20, MUTED).next_to(grid, m.DOWN, buff=0.22).align_to(grid, m.LEFT)

        def counter(n):
            return code(f"{n} token{'s' if n > 1 else ''} · {nice_bytes(n * KV_TOKEN)}", 26, INK, t2c={r"[\d.]+ [KMG]iB": KV}).move_to([3.7, 2.1, 0])

        count = counter(1)
        cell = grid[-1][1]
        ring = m.SurroundingRectangle(cell, buff=0.05).set_stroke(INK, 2)
        call_box = card(5.3, 1.35, KV, CARD, radius=0.16, stroke=2.5).move_to([3.85, 0.95, 0])
        call_big = title_text(nice_bytes(KV_CELL), 48, KV).move_to(call_box).align_to(call_box, m.LEFT).shift(0.35 * m.RIGHT)
        call_text = m.VGroup(text("one layer, one token", 22, INK), code("2 × 8 × 128 × 2 B", 20, MUTED))
        call_text.arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT).next_to(call_big, m.RIGHT, buff=0.4)

        scale = 4.95 / (16 * GIB)  # units per byte
        x0 = 1.35

        def bar(nbytes, color, y):
            w = max(scale * nbytes, 0.02)
            return m.Rectangle(width=w, height=0.38).set_stroke(width=0).set_fill(color, 1).move_to([x0 + w / 2, y, 0])

        def kv_label(tokens, nbytes):
            return code(f"{tokens} tokens · {nice_bytes(nbytes)}", 22, KV).move_to([x0, -0.55, 0], aligned_edge=m.LEFT)

        kv_bar = bar(8192 * KV_TOKEN, KV, -0.98)
        kv_lab = kv_label("8K", 8192 * KV_TOKEN)
        kv_bar4, kv_lab4 = bar(32768 * KV_TOKEN, KV, -0.98), kv_label("32K", 32768 * KV_TOKEN)
        kv_bar16, kv_lab16 = bar(CONTEXT * KV_TOKEN, KV, -0.98), kv_label("128K", CONTEXT * KV_TOKEN)
        w_bar = bar(weight_bytes(), WEIGHTS, -2.05)
        w_lab = code(f"weights · {weight_bytes() / 1e9:.0f} GB", 22, WEIGHTS).move_to([x0, -1.62, 0], aligned_edge=m.LEFT)
        w_note = cite("16 GiB = 17.2 GB").move_to([x0, -2.65, 0], aligned_edge=m.LEFT)

        fill = lambda c: c.animate.set_fill(KV, 1)
        self.say(
            "[Watch it grow.] [Each new token adds a column of thirty-two cells], [filled layer by layer.]",
            m.AnimationGroup(self.new_header("05 · growth", "Token by token, layer by layer"),
                             m.FadeIn(grid), m.FadeIn(top_lab), m.FadeIn(bot_lab), m.FadeIn(tok_lab)),
            m.AnimationGroup(m.LaggedStart(*[fill(c) for c in col(0)], lag_ratio=0.12, run_time=2.0), m.FadeIn(count)),
            m.AnimationGroup(m.LaggedStart(*[fill(c) for c in col(1)], lag_ratio=0.1, run_time=1.2),
                             m.Transform(count, counter(2))),
        )
        self.say(
            "[Each cell is four kibibytes]: [one layer's key and value, for one token.]",
            m.AnimationGroup(m.Create(ring), m.FadeIn(call_box, target_position=cell), m.FadeIn(call_big, target_position=cell)),
            m.FadeIn(call_text, shift=0.1 * m.LEFT),
        )
        self.say(
            "[And the columns keep coming], [one per token, for as long as the chat goes on.]",
            m.AnimationGroup(m.FadeOut(ring),
                             m.LaggedStart(*[m.LaggedStart(*[fill(c) for c in col(j)], lag_ratio=0.02, run_time=0.5)
                                             for j in range(2, cols_n)], lag_ratio=0.12, run_time=3.2)),
            m.Transform(count, counter(cols_n)),
        )
        self.say(
            "[Eight thousand tokens make one gibibyte.] [Thirty-two thousand make four.]",
            m.AnimationGroup(m.GrowFromEdge(kv_bar, m.LEFT), m.FadeIn(kv_lab, shift=0.1 * m.UP)),
            m.AnimationGroup(m.Transform(kv_bar, kv_bar4), m.Transform(kv_lab, kv_lab4)),
        )
        self.say(
            "[Owl's full context, a hundred and twenty-eight thousand tokens, takes sixteen]: [about as much memory as the weights themselves.]",
            m.AnimationGroup(m.Transform(kv_bar, kv_bar16), m.Transform(kv_lab, kv_lab16)),
            m.AnimationGroup(m.GrowFromEdge(w_bar, m.LEFT), m.FadeIn(w_lab, shift=0.1 * m.UP), m.FadeIn(w_note)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ measured on this Mac

    def on_this_mac(self):
        self.next_section("6 · Measured on this Mac", notes=f"research/measurements.md §4 (2026-10-24, Mac Studio M4 Max, mlx-lm 0.32.2): Llama-3.1-8B with 4-bit weights ({W4:.2f} GB) and a BF16 cache; prefill T tokens, generate 16, read mx.get_active_memory(). Active GB at {', '.join(f'{t}' for t, _ in POINTS)} tokens: {', '.join(f'{g:.2f}' for _, g in POINTS)}. Least-squares slope {SLOPE:,.0f} B/token (re-fitted here: {FIT[0]:,.1f}) = the formula 2·32·8·128·2 = {KV_TOKEN:,}. 32K tokens add 4 GiB: {W4:.2f} → {POINTS[-1][1]:.2f} GB. mlx-lm allocates the cache in 256-token steps, so each point carries up to 255 spare slots (same for all points; the slope is unaffected).")
        ox, oy, cw, ch = -5.2, -2.45, 6.8, 4.5
        t_max, g_max = 34_000, 10.0

        def p(t, gb):
            return np.array([ox + cw * t / t_max, oy + ch * gb / g_max, 0])

        x_axis = m.Arrow(p(0, 0), p(t_max, 0) + [0.35, 0, 0], buff=0, color=FAINT, stroke_width=3, max_tip_length_to_length_ratio=0.03)
        y_axis = m.Arrow(p(0, 0), p(0, g_max) + [0, 0.3, 0], buff=0, color=FAINT, stroke_width=3, max_tip_length_to_length_ratio=0.05)
        xt = m.VGroup(*[code(lab, 20, MUTED).next_to(p(t, 0), m.DOWN, buff=0.15) for t, lab in ((0, "0"), (8192, "8K"), (16384, "16K"), (24576, "24K"), (32768, "32K"))])
        yt = m.VGroup(*[code(f"{g}", 20, MUTED).next_to(p(0, g), m.LEFT, buff=0.15) for g in (0, 2, 4, 6, 8)])
        x_title = text("tokens in the conversation", 20, MUTED).next_to(xt, m.DOWN, buff=0.15).set_x(p(t_max / 2, 0)[0])
        y_title = text("memory in use (GB)", 20, MUTED).move_to(p(0, g_max) + [0.3, 0.2, 0], aligned_edge=m.LEFT)
        axes_g = m.VGroup(x_axis, y_axis, xt, yt, x_title, y_title)

        band = m.Polygon(p(0, 0), p(t_max, 0), p(t_max, W4), p(0, W4)).set_stroke(width=0).set_fill(WEIGHTS, 0.2)
        base = m.DashedLine(p(0, W4), p(t_max, W4), dash_length=0.1).set_stroke(WEIGHTS, 2.5)
        base_lab = code(f"weights · {W4:.1f} GB (4-bit)", 20, WEIGHTS).move_to(p(1500, W4 / 2), aligned_edge=m.LEFT)
        fit = lambda t: (FIT[1] + FIT[0] * t) / 1e9
        wedge = m.Polygon(p(0, W4), p(t_max, W4), p(t_max, fit(t_max)), p(0, fit(0))).set_stroke(width=0).set_fill(KV, 0.22)
        line = m.Line(p(0, fit(0)), p(t_max, fit(t_max))).set_stroke(KV, 4)
        dots = m.VGroup(*[m.Dot(p(t, g), radius=0.085, color=INK).set_z_index(3) for t, g in POINTS])
        last = POINTS[-1]
        brk = tick_bracket(p(last[0], 0)[0] - 0.25, p(0, W4)[1], p(0, last[1])[1], KV)
        brk_lab = code(f"+{nice_bytes(last[0] * KV_TOKEN)}", 22, KV, weight=m.BOLD).next_to(brk, m.LEFT, buff=0.25)
        top_lab = code(f"{W4:.1f} → {last[1]:.1f} GB", 20, INK).next_to(dots[-1], m.UP, buff=0.25).shift(0.3 * m.LEFT)

        px = 4.7
        s_head = text("slope, measured", 22, MUTED).move_to([px, 1.75, 0])
        s_num = title_text(f"{SLOPE:,.0f}", 64, KV).move_to([px, 1.05, 0])
        s_unit = code("bytes per token", 20, MUTED).next_to(s_num, m.DOWN, buff=0.18)
        f_line = m.VGroup(code("formula  2·32·8·128·2", 20, MUTED), code(f"= {KV_TOKEN:,}", 20, INK)).arrange(m.DOWN, buff=0.12)
        f_line.move_to([px, -0.45, 0])
        check = pill("✓ to the byte", GOOD, GOOD_FILL, 24).move_to([px, -1.3, 0])
        mini = code(f"Owl-mini: {MINI_SLOPE // KIB} KiB/token ✓", 20, INK, t2c={"✓": GOOD}).move_to([px, -1.95, 0])
        reads = code("read at every decode step", 20, MEMORY).move_to(p(1200, 8.4), aligned_edge=m.LEFT)
        sweep = m.ShowPassingFlash(m.Line(p(0, fit(0)), p(t_max, fit(t_max))).set_stroke(MEMORY, 10), time_width=0.6, run_time=1.4)
        ep9 = code("→ Episode 9", 20, MUTED).next_to(reads, m.DOWN, buff=0.12).align_to(reads, m.LEFT)
        mac = icon(recolored("mac", MUTED, CARD, PREFIX), 0.55)
        where = m.VGroup(cite("measured on this Mac"), cite("M4 Max · mlx-lm")).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
        stamp = m.VGroup(mac, where).arrange(m.RIGHT, buff=0.25).move_to([px, -2.75, 0])

        self.say(
            "[Does the formula hold on a real machine?] [We measured Owl's memory on this Mac], [at six context lengths.]",
            m.AnimationGroup(self.new_header("06 · measured", "The slope, on this Mac"), m.FadeIn(axes_g)),
            m.AnimationGroup(m.FadeIn(band), m.Create(base), m.FadeIn(base_lab), m.FadeIn(stamp)),
            m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.2),
        )
        self.say(
            "[The points fall on a straight line], [and its slope is the cache's size per token.]",
            m.AnimationGroup(m.Create(line), m.FadeIn(wedge)),
            m.AnimationGroup(m.FadeIn(s_head), m.FadeIn(s_num, shift=0.1 * m.UP), m.FadeIn(s_unit)),
        )
        self.say(
            "[A hundred and thirty-one thousand and seventy-two bytes per token]: [exactly the formula], [to the byte.]",
            m.Circumscribe(s_num, color=KV, buff=0.1),
            m.FadeIn(f_line, shift=0.1 * m.UP),
            m.FadeIn(check, scale=0.9),
        )
        self.say(
            "[Thirty-two thousand tokens add four gibibytes], [nearly doubling what the four-bit model needs.]",
            m.AnimationGroup(m.Create(brk), m.FadeIn(brk_lab, shift=0.1 * m.LEFT)),
            m.FadeIn(top_lab, shift=0.1 * m.DOWN),
        )
        self.say(
            "[The small Owl-mini matched its own formula too], [at thirty-two kibibytes per token.]",
            m.Circumscribe(check, color=GOOD, buff=0.08),
            m.FadeIn(mini, shift=0.1 * m.UP),
        )
        self.say(
            "[And every decode step reads this whole cache], [so long chats also decode slower]: [that's episode nine.]",
            m.AnimationGroup(m.FadeIn(reads, shift=0.1 * m.DOWN), sweep),
            m.Succession(m.Wait(0.2), m.ShowPassingFlash(m.Line(p(0, fit(0)), p(t_max, fit(t_max))).set_stroke(MEMORY, 10), time_width=0.6, run_time=1.4)),
            m.FadeIn(ep9, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ capacity

    def capacity(self):
        self.next_section("7 · How many students fit", notes=f"notes §3 worked example (H100 SXM 80 GB, NVIDIA spec, dated Oct 2026): budget 90% = {BUDGET / 1e9:.0f} GB; minus {weight_bytes() / 1e9:.2f} GB of weights = {KV_ROOM / 1e9:.2f} GB for KV = {SLOTS:,.0f} token slots (activations and CUDA-graph workspace ignored, so real numbers are a bit lower). 8K tokens per chat → {USERS} students; 128K → {USERS_LONG}. Compute would only bind at batch ≈ ridge = {CRIT:.0f} (989 TFLOPS ÷ 3.35 TB/s; critical batch ≈ ridge for BF16, notes §2, Episode 4). Memory, not compute, limits the batch.")
        x0, width, y = -5.8, 11.6, 0.65
        per = width / GPU["memory"]
        xb = lambda nbytes: x0 + per * nbytes
        track = m.Rectangle(width=width, height=0.72).set_stroke(EDGE, 2).set_fill(CARD, 1).move_to([x0 + width / 2, y, 0])

        def seg(a, b, color, h=0.56, opacity=1.0, inset=0.0):
            w = per * (b - a) - inset
            return m.Rectangle(width=w, height=h).set_stroke(width=0).set_fill(color, opacity).move_to([(xb(a) + xb(b)) / 2, y, 0])

        gpu = icon(recolored("gpu", MUTED, CARD, PREFIX), 0.6)
        cap = code("H100 · 80 GB of HBM", 22, INK)
        title = m.VGroup(gpu, cap).arrange(m.RIGHT, buff=0.25).move_to([x0, 1.75, 0], aligned_edge=m.LEFT)
        w_seg = seg(0, weight_bytes(), WEIGHTS)
        r_seg = seg(BUDGET, GPU["memory"], FAINT, opacity=0.6)
        room = m.DashedVMobject(
            m.Rectangle(width=per * KV_ROOM - 0.06, height=0.62).move_to([(xb(weight_bytes()) + xb(BUDGET)) / 2, y, 0]),
            num_dashes=60,
        ).set_stroke(KV, 2)
        w_lab = code(f"weights {weight_bytes() / 1e9:.0f} GB", 20, WEIGHTS).move_to([(x0 + xb(weight_bytes())) / 2, y - 0.68, 0])
        r_lab = code("10% free", 20, MUTED).move_to([(xb(BUDGET) + xb(GPU["memory"])) / 2, y - 0.68, 0])
        k_lab = code(f"cache ≈ {KV_ROOM / 1e9:.0f} GB ≈ {round(SLOTS, -3):,.0f} tokens", 20, KV).move_to([(xb(weight_bytes()) + xb(BUDGET)) / 2, y - 0.68, 0])

        def chats(n, tokens):
            nbytes = tokens * KV_TOKEN
            segs = m.VGroup(*[seg(weight_bytes() + i * nbytes, weight_bytes() + (i + 1) * nbytes, KV, h=0.5, inset=0.03) for i in range(n)])
            heads = m.VGroup(*[m.Dot(radius=0.055 if n > 10 else 0.11, color=REQUEST).move_to([s.get_x(), y + 0.62, 0]) for s in segs])
            return segs, heads

        segs52, dots52 = chats(USERS, CHAT)
        segs3, dots3 = chats(USERS_LONG, CONTEXT)

        def stat_block(number, label, sub, color, x):
            num = title_text(number, 84, color)
            g = m.VGroup(num, text(label, 22, INK), code(sub, 20, MUTED)).arrange(m.DOWN, buff=0.14)
            return g.move_to([x, -1.85, 0])

        st52 = stat_block(str(USERS), "students at once", "8K-token chats", REQUEST, -4.3)
        st3 = stat_block(str(USERS_LONG), "students at once", "128K-token chats", REQUEST, -0.6)
        st295 = stat_block(f"≈{CRIT:.0f}", "before compute runs out", "the ridge · Episode 4", COMPUTE, 3.9)
        verdict = pill("memory, not compute, limits the batch", KV, VIOLET_FILL, 24).move_to([0, -3.35, 0])

        self.say(
            "[Now scale up to a server]: [one H100, with eighty gigabytes of memory.]",
            self.new_header("07 · capacity", "How many students fit?"),
            m.AnimationGroup(m.FadeIn(title, shift=0.1 * m.UP), m.FadeIn(track)),
        )
        self.say(
            "[Owl's weights take sixteen], [and we keep a tenth free for everything else.]",
            m.AnimationGroup(m.GrowFromEdge(w_seg, m.LEFT), m.FadeIn(w_lab, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(r_seg), m.FadeIn(r_lab, shift=0.1 * m.UP)),
        )
        self.say(
            "[The other fifty-six gigabytes hold the cache]: [room for about four hundred and twenty-seven thousand tokens.]",
            m.Create(room),
            m.FadeIn(k_lab, shift=0.1 * m.UP),
        )
        self.say(
            "[At eight thousand tokens per chat], [that's fifty-two students at once.]",
            m.LaggedStart(*[m.FadeIn(s, shift=0.1 * m.RIGHT) for s in segs52], lag_ratio=0.06),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in dots52], lag_ratio=0.03), m.FadeIn(st52, shift=0.15 * m.UP)),
        )
        self.say(
            "[If every student fills Owl's whole context], [only three fit.]",
            m.AnimationGroup(m.FadeOut(segs52), m.FadeOut(dots52),
                             m.LaggedStart(*[m.GrowFromEdge(s, m.LEFT) for s in segs3], lag_ratio=0.3)),
            m.AnimationGroup(m.FadeIn(dots3, scale=0.5), m.FadeIn(st3, shift=0.15 * m.UP)),
        )
        self.say(
            "[Yet the math units would only run out at about three hundred.] [Memory, not compute, limits the batch.]",
            m.FadeIn(st295, shift=0.15 * m.UP),
            m.FadeIn(verdict, scale=0.9),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Remember (every token's keys and values, every layer), reuse (exactly equal to recomputing, far less work), budget (128 KiB per token: 52 students per H100 at 8K tokens). Next: Episode 6, five tricks that shrink the cache.")
        words = [("Remember", KV, "keys and values, every layer"),
                 ("Reuse", TOKEN, "exact, and far less work"),
                 ("Budget", REQUEST, f"{nice_bytes(KV_TOKEN)} per token")]
        cols = m.VGroup()
        for w, color, sub in words:
            cols.add(m.VGroup(title_text(w, 72, color), text(sub, 22, MUTED)).arrange(m.DOWN, buff=0.25))
        cols.arrange(m.RIGHT, buff=1.3).move_to([0, 0.75, 0])
        base = 0.85
        for c in cols:
            c[0].shift((base - c[0][0].get_bottom()[1]) * m.UP)  # first letters (R, R, B) sit on the baseline
            c[1].move_to([c[0].get_x(), base - 0.55, 0])
        mid = base + 0.3
        arrows = m.VGroup(*[
            arrow([a[0].get_right()[0] + 0.15, mid, 0], [b[0].get_left()[0] - 0.15, mid, 0], MUTED) for a, b in zip(cols, cols[1:])
        ])
        tagline = text("Every token leaves 128 KiB behind.", 32, INK).move_to([0, -1.35, 0])
        mascot = owl(0.75).move_to([0, -2.7, 0])

        self.say(
            "[So: remember], [reuse], [and budget.]",
            m.AnimationGroup(self.new_header("recap", "The KV cache in three words"), m.FadeIn(cols[0], shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowArrow(arrows[0]), m.FadeIn(cols[1], shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowArrow(arrows[1]), m.FadeIn(cols[2], shift=0.2 * m.UP)),
        )
        self.say(
            "[Every token leaves a hundred and twenty-eight kibibytes behind], [and that memory, not math, decides how many students Owl can serve.]",
            m.FadeIn(tagline, shift=0.1 * m.UP),
            m.FadeIn(mascot, shift=0.2 * m.UP),
        )
        self.wait(0.4)
        self.clear()
