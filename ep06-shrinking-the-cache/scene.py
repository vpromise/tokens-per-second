"""Tokens per Second · Episode 6 · Shrinking the Cache (Module 2 · The KV Cache)

Five tricks make the cache smaller, and each one pays for it somewhere. Running example: Owl-8B (Llama-3.1-8B:
8 B parameters, 16 GB of BF16 weights, 128 KiB of KV cache per token; 52 students per H100 at 8K tokens,
Episode 5).

Sections: hook (Episode 5's formula 2 · L · n_kv · d_h · b, times the tokens kept; four targets for five
tricks) · share the heads (MHA 512 KiB → MQA 16 KiB → GQA 128 KiB per token for Owl; 13 / 416 / 52 students
per H100) · compress (MLA: a 512-number latent + a 64-number position key per layer; DeepSeek-V3, 84x Owl's
parameters, 68.6 KiB per token) · forget (a rolling buffer, slot = token mod W; Mistral 7B: 8x smaller at
32K; on this Mac Owl-mini with a 1,024-token window forgot a code word; Gemma 3: 5 local per global layer)
· fewer bits (BF16 → FP8 → INT4; 52 → 104 students per H100 with FP8) · on this Mac (mlx-lm --kv-bits 8 / 4:
slope 131,072 → 69,627 → 36,859 B/token; quality spot-check; decode at 32K vs its ceiling) · the bill (what
each trick shrinks and costs; 13 → 52 → 104 students) · recap.

Sources (facts dated October 2026):
- research/notes.md §3 "Shrinking the cache": MQA (Shazeer 2019, arXiv 1911.02150), GQA (Ainslie et al.,
  EMNLP 2023, arXiv 2305.13245: "MQA can lead to quality degradation and training instability"; uptraining
  with "5% of original pre-training compute"; "uptrained GQA achieves quality close to multi-head attention
  with comparable speed to MQA"; both checked on arxiv.org 2026-10-24), MLA (DeepSeek-V2, arXiv 2405.04434:
  cache a 512-dim latent plus a decoupled 64-dim RoPE key; V2 vs DeepSeek 67B "reduces the KV cache by
  93.3%"), Mistral 7B (arXiv 2310.06825, Table 1: 32 layers, 8 KV heads, head_dim 128, window 4096; "On a
  sequence length of 32k tokens, this reduces the cache memory usage by 8x, without impacting the model
  quality"; rolling buffer at i mod W; checked 2026-10-24), Gemma 3 (arXiv 2503.19786: "5 local layers for
  every global layer", local span 1024), vLLM kv_cache_dtype fp8, mlx-lm --kv-bits.
- DeepSeek-V3 config.json (huggingface.co/deepseek-ai/DeepSeek-V3, read 2026-10-24): num_hidden_layers 61,
  kv_lora_rank 512, qk_rope_head_dim 64; 671 B parameters (notes §0). Per token in BF16: 61 × 576 × 2 B =
  70,272 B = 68.6 KiB (notes §3 table, [DERIVED]).
- research/notes.md §3 worked example (H100 SXM 80 GB, NVIDIA spec, dated Oct 2026): 90% budget minus 16.06 GB
  of weights ≈ 56 GB for KV; 8K-token chats → 52 students (BF16), 104 (FP8), 13 (if MHA).
- research/measurements.md (this Mac Studio, M4 Max 546 GB/s, mlx-lm 0.32.2, 2026-10-24): §5 KV quantization
  (Llama-3.1-8B 4-bit weights, group size 64, quantized_kv_start = 0; active-memory slopes, quality
  spot-check on an 8,081-token prompt, decode at 32K: 1 run of 16 tokens); §6 rotating cache (Llama-3.2-1B
  4-bit, RotatingKVCache max 1,024, keep 4; the 3,070-token code-word test).
- owl_reference.py: kv_bytes_per_token, weight_bytes, measured(); every number below is computed and asserted.

Illustrative (schematic, said as such or obvious): the 8-slot rolling buffer and its token numbers, the bit
boxes of one cached number, the widths of the MLA bars (proportional to element counts), the chat segments.

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

PREFIX = "ep06"
KIB, MIB, GIB = 2**10, 2**20, 2**30

# Episode 5's formula, Owl's config
KV_TOKEN = int(kv_bytes_per_token())
HEADS, KV_HEADS = OWL["heads"], OWL["kv_heads"]
GROUP = HEADS // KV_HEADS
assert KV_TOKEN == 128 * KIB and HEADS == 32 and KV_HEADS == 8 and GROUP == 4


def owl_kv(kv_heads=KV_HEADS, bits=16):
    """Owl's KV bytes per token with another number of KV heads or another precision."""
    return int(kv_bytes_per_token(dict(OWL, kv_heads=kv_heads), bits))


MHA_TOKEN, MQA_TOKEN = owl_kv(HEADS), owl_kv(1)
FP8_TOKEN, INT4_TOKEN = owl_kv(bits=8), owl_kv(bits=4)
assert MHA_TOKEN == 512 * KIB and MQA_TOKEN == 16 * KIB  # "five hundred and twelve kibibytes"
assert MHA_TOKEN // MQA_TOKEN == 32 and MHA_TOKEN // KV_TOKEN == 4  # "thirty-two times", "four times smaller"
assert FP8_TOKEN == 64 * KIB and INT4_TOKEN == 32 * KIB  # "halves, to sixty-four kibibytes", "halves again"

# one H100 (Episode 5, notes §3 worked example)
BUDGET = 0.9 * GPU["memory"]
KV_ROOM = BUDGET - weight_bytes()
CHAT = 8192


def students(per_token, tokens=CHAT):
    return int(KV_ROOM / per_token // tokens)


USERS, USERS_MHA, USERS_MQA, USERS_FP8 = students(KV_TOKEN), students(MHA_TOKEN), students(MQA_TOKEN), students(FP8_TOKEN)
assert round(KV_ROOM / 1e9) == 56 and round(weight_bytes() / 1e9) == 16  # "fifty-six gigabytes"
assert (USERS, USERS_MHA, USERS_MQA, USERS_FP8) == (52, 13, 416, 104)  # "fifty-two", "thirteen", "a hundred and four"
assert USERS_FP8 == 2 * USERS and USERS_FP8 // USERS_MHA == 8

# multi-head latent attention: DeepSeek-V3 (config.json) against Owl, both in BF16
DSV3 = dict(name="DeepSeek-V3", params=671e9, layers=61, kv_lora_rank=512, rope_dim=64)
MLA_NUMBERS = DSV3["kv_lora_rank"] + DSV3["rope_dim"]
MLA_TOKEN = DSV3["layers"] * MLA_NUMBERS * 2
OWL_LAYER_NUMBERS = 2 * KV_HEADS * OWL["head_dim"]
assert MLA_NUMBERS == 576 and OWL_LAYER_NUMBERS == 2048 and OWL_LAYER_NUMBERS * 2 == 4 * KIB  # "five hundred and twelve … plus sixty-four"
assert MLA_TOKEN == 70_272 and round(MLA_TOKEN / KIB, 1) == 68.6
assert 0.5 < MLA_TOKEN / KV_TOKEN < 0.55  # "about half as much per token"
PARAM_RATIO = DSV3["params"] / OWL["params"]
assert round(PARAM_RATIO) == 84  # "eighty-four times bigger"

# sliding window: Mistral 7B (Table 1) has Owl's per-token cache; on this Mac, Owl-mini's rolling cache
MISTRAL = dict(name="Mistral 7B", layers=32, kv_heads=8, head_dim=128, window=4096)
MISTRAL_TOKEN = int(kv_bytes_per_token(MISTRAL))
LONG = 32768
FULL_LONG, WIN_LONG = LONG * MISTRAL_TOKEN, MISTRAL["window"] * MISTRAL_TOKEN
assert MISTRAL_TOKEN == KV_TOKEN and FULL_LONG == 4 * GIB and WIN_LONG == 512 * MIB
assert FULL_LONG // WIN_LONG == 8  # "eight times smaller"
ROT = measured("rotating_cache", "llama-3.2-1b", "4bit")
DEMO = ROT["forgetting_demo"]
ROT_WINDOW = ROT["max_kv_size"]
assert ROT_WINDOW == 1024 and DEMO["prompt_tokens"] == 3070  # "a thousand-token window"
assert DEMO["full"].startswith("PELICAN") and "There is no answer" in DEMO["rotating_1024"]  # "forgot a code word"
assert ROT["rotating"]["points"]["8192"]["kv_MiB"] == 32 == ROT_WINDOW * kv_bytes_per_token(MODELS["owl-mini"]) / MIB
FORGOT = DEMO["rotating_1024"].split(":")[1].strip().split(",")[0]  # "There is no answer"
GEMMA_LOCAL, GEMMA_SPAN = 5, 1024  # Gemma 3 report: "5 local layers for every global layer", span 1024

# fewer bits, measured on this Mac (Llama-3.1-8B, 4-bit weights)
MEMQ = {b: measured("memory", "llama-3.1-8b", "4bit", f"kv{b}") for b in (16, 8, 4)}
SLOPE = {b: MEMQ[b]["slope_active_bytes_per_token"] for b in MEMQ}
EXPECT = {b: MEMQ[b]["formula_bytes_per_token"] for b in MEMQ}
GROUP_SIZE = 64
for b in (8, 4):  # b bits plus a BF16 scale and a BF16 offset per 64 numbers = b + 0.5 bits per number
    assert EXPECT[b] == KV_TOKEN * (b + 2 * 16 / GROUP_SIZE) / 16
    assert abs(SLOPE[b] - EXPECT[b]) < 5  # "the prediction matches within five bytes"
assert SLOPE[16] == KV_TOKEN == EXPECT[16]
assert round(SLOPE[8] / KIB) == 68 and round(SLOPE[4] / KIB) == 36  # "to sixty-eight, and then thirty-six"
MAC_POINTS = {b: sorted((int(t), r["active_GB"]) for t, r in MEMQ[b]["by_context_tokens"].items()) for b in MEMQ}
assert all(len(p) == 6 for p in MAC_POINTS.values())
QUAL = measured("kv_quant_quality")
SAME8 = QUAL["by_kv_bits"]["8"]["tokens_identical_to_bf16_prefix"]
SAME4 = QUAL["by_kv_bits"]["4"]["tokens_identical_to_bf16_prefix"]
GEN = QUAL["by_kv_bits"]["8"]["gen_tokens"]
assert SAME8 == GEN == 64 and SAME4 == 45 and QUAL["prompt_tokens"] == 8081  # "word for word", "after forty-five tokens"
W_READ = measured("models", "llama-3.1-8b", "4bit", "decode_bytes_per_token")  # weights streamed per decode step


def mac_ceiling(per_token, tokens=LONG):
    """This Mac's decode speed limit: bandwidth ÷ (weights read + the whole cache)."""
    return MAC["bandwidth"] / (W_READ + tokens * per_token)


SPEED = {b: MEMQ[b]["by_context_tokens"][str(LONG)]["decode_tok_s"] for b in (16, 8)}
CEIL = {b: mac_ceiling(EXPECT[b]) for b in (16, 8)}
assert (round(SPEED[16]), round(SPEED[8])) == (50, 51) and (round(CEIL[16]), round(CEIL[8])) == (64, 84)  # "the measured speed stays put"

# ---------------------------------------------------------------------------- look


def cite(s):
    return text(s, 20, MUTED)


def nice_bytes(n):
    if n >= GIB:
        return f"{n / GIB:g} GiB"
    if n >= MIB:
        return f"{n / MIB:g} MiB"
    return f"{n / KIB:g} KiB"


def kv_cell(size=0.42, color=KV, fill=VIOLET_FILL):
    return m.RoundedRectangle(corner_radius=min(0.07, size / 5), width=size, height=size, stroke_color=color,
                              stroke_width=2.2, fill_color=fill, fill_opacity=1)


def head_sq(size, color, fill):
    return m.Square(size).set_stroke(color, 2).set_fill(fill, 1)


def price(words, color=RED, fill=RED_FILL, size=22):
    """'price' pill + what the trick costs."""
    tag = pill("price", color, fill, 20)
    return m.VGroup(tag, text(words, size, INK)).arrange(m.RIGHT, buff=0.25)


def funnel(x0, x1, y, h0, h1, color=COMPUTE, fill=AMBER_FILL):
    """A trapezoid from height h0 at x0 to h1 at x1 (a projection: squeeze or unpack)."""
    return m.Polygon([x0, y + h0 / 2, 0], [x1, y + h1 / 2, 0], [x1, y - h1 / 2, 0], [x0, y - h0 / 2, 0],
                     stroke_color=color, stroke_width=2.5, fill_color=fill, fill_opacity=1)


def tick_bracket(x, y0, y1, color, lip=0.12):
    """A vertical bracket with lips pointing left."""
    return m.VGroup(
        m.Line([x, y0, 0], [x, y1, 0]), m.Line([x, y0, 0], [x - lip, y0, 0]), m.Line([x, y1, 0], [x - lip, y1, 0])
    ).set_stroke(color, 2.5)


# ---------------------------------------------------------------------------- the episode


class Ep06ShrinkingTheCache(Episode):
    number = 6

    def lesson(self):
        self.hook()
        self.share()
        self.compress()
        self.forget()
        self.fewer_bits()
        self.on_this_mac()
        self.bill()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes=f"Episode 5's result: {nice_bytes(KV_TOKEN)} per token (2 × 32 × 8 × 128 × 2 B), {USERS} students per H100 at 8K tokens (notes §3). The cache of one conversation is that formula times the tokens kept, T. Five tricks attack four parts of the product: sharing heads (MQA, GQA) shrinks n_kv; a latent (MLA) replaces 2 · n_kv · d_h; a sliding window caps T; quantization (FP8, INT4) shrinks b. Each has a price.")
        big = m.VGroup(
            stat("128", "KiB per token", "Owl-8B, from Episode 5", KV, 96),
            stat(str(USERS), "students per H100", "8K-token chats", REQUEST, 96),
        ).arrange(m.RIGHT, buff=1.2).move_to([0, 0.5, 0])
        small = m.VGroup(
            pill(f"Owl-8B · {nice_bytes(KV_TOKEN)} per token", KV, VIOLET_FILL, 22, mono=True),
            pill(f"{USERS} students per H100 · 8K chats", REQUEST, PINK_FILL, 22, mono=True),
        ).arrange(m.RIGHT, buff=0.5).move_to([-0.5, -3.05, 0])
        mascot = owl(0.55).move_to([5.95, -3.0, 0])

        symbols = ["2", "L", "n_kv", "d_h", "b", "T"]
        labels = ["K and V", "layers", "KV heads", "head size", "bytes", "tokens"]
        fy, ly = 0.72, 0.22  # formula baseline, label row
        parts = []
        for i, s in enumerate(symbols):
            parts.append(code(s, 44, INK, weight=m.BOLD))
            if i < len(symbols) - 1:
                parts.append(code("×", 34, MUTED))
        gap = 0.55
        x = -(sum(p.width for p in parts) + gap * (len(parts) - 1)) / 2
        for p in parts:
            p.set_x(x + p.width / 2)
            x += p.width + gap
        syms, times = m.VGroup(*parts[0::2]), m.VGroup(*parts[1::2])
        for s, word in zip(syms, symbols):  # every symbol on one baseline (descenders: _ in n_kv, d_h)
            sitting = [g for g, c in zip(s, word) if c not in "_"]
            s.shift((fy - min(g.get_bottom()[1] for g in sitting)) * m.UP)
        for t in times:
            t.set_y(fy + 0.2)
        syms[0].set_color(KV)
        xs = [s.get_x() for s in syms]
        labs = m.VGroup(*[text(s, 20, MUTED).move_to([x, ly, 0]) for s, x in zip(labels, xs)])
        formula = m.VGroup(syms[0], labs[0], times)

        def target(name, sub, x, y):
            body = m.VGroup(title_text(name, 28, KV), code(sub, 20, MUTED)).arrange(m.DOWN, buff=0.1)
            box = card(max(2.2, body.width + 0.45), 1.0, KV, CARD, radius=0.16, stroke=2)
            body.move_to(box)
            return m.VGroup(box, body).move_to([x, y, 0])

        lab_bottom = labs[0].get_bottom()[1] - 0.08
        comb_y = lab_bottom - 0.25
        share = target("share", "MQA · GQA", xs[2], 2.05)
        forget = target("forget", "sliding window", xs[5], 2.05)
        squeeze = target("compress", "MLA", (xs[0] + xs[3]) / 2, -1.25)
        bits = target("fewer bits", "FP8 · INT4", xs[4], -1.25)
        link = lambda a, b: m.Line(a, b).set_stroke(KV, 2.5)
        share_l = link(share.get_bottom(), [xs[2], syms[2].get_top()[1] + 0.08, 0])
        forget_l = link(forget.get_bottom(), [xs[5], syms[5].get_top()[1] + 0.08, 0])
        bits_l = link(bits.get_top(), [xs[4], lab_bottom, 0])
        comb = m.VGroup(
            m.Line([xs[0], comb_y, 0], [xs[3], comb_y, 0]),
            *[m.Line([xs[i], comb_y, 0], [xs[i], lab_bottom, 0]) for i in (0, 2, 3)],
            m.Line(squeeze.get_top(), [squeeze.get_x(), comb_y, 0]),
        ).set_stroke(KV, 2.5)
        cards = [m.VGroup(share, share_l), m.VGroup(squeeze, comb), m.VGroup(forget, forget_l), m.VGroup(bits, bits_l)]
        pays = text("and each one has a price", 24, INK).move_to([0, -2.25, 0])
        light = lambda i: m.AnimationGroup(syms[i].animate.set_color(KV), m.FadeIn(labs[i], shift=0.1 * m.UP))

        self.say(
            "[Last episode], Owl's cache came to a hundred and twenty-eight kibibytes per token: [room for fifty-two students] on one H100.",
            m.AnimationGroup(self.new_header("hook", "Make the cache smaller"), m.FadeIn(big[0], shift=0.15 * m.UP),
                             m.FadeIn(mascot, shift=0.2 * m.UP)),
            m.FadeIn(big[1], shift=0.15 * m.UP),
        )
        self.say(
            "[That size is a simple product]: [layers, key-value heads, head size, bytes per number], [and the tokens kept.]",
            m.AnimationGroup(m.FadeTransform(big[0], small[0]), m.FadeTransform(big[1], small[1]),
                             m.FadeIn(formula), m.FadeIn(syms[1:])),
            m.LaggedStart(*[light(i) for i in (1, 2, 3, 4)], lag_ratio=0.45),
            light(5),
        )
        self.say(
            "[Shrink any one of them], [and more students fit.]",
            m.LaggedStart(*[m.Indicate(s, color=KV, scale_factor=1.15) for s in syms], lag_ratio=0.15),
            m.Circumscribe(small[1], color=REQUEST, buff=0.08),
        )
        self.say(
            "[Today: five tricks, aimed at four parts of that product], [and the price each one pays.]",
            m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.UP) for c in cards], lag_ratio=0.3),
            m.FadeIn(pays, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 1 · share the heads

    def share(self):
        self.next_section("1 · Share the heads", notes=f"Owl-8B has {HEADS} query heads per layer. Multi-head attention (MHA, Vaswani et al. 2017) gives each its own K/V head: Owl would store {nice_bytes(MHA_TOKEN)} per token → {USERS_MHA} students per H100 at 8K. Multi-query (MQA, Shazeer 2019): one K/V head, {nice_bytes(MQA_TOKEN)} per token, {MHA_TOKEN // MQA_TOKEN}x smaller → {USERS_MQA} students; 'MQA can lead to quality degradation and training instability' (Ainslie et al. 2023). Grouped-query (GQA, Ainslie et al. 2023): {KV_HEADS} K/V heads, groups of {GROUP} → {nice_bytes(KV_TOKEN)}, Owl's actual design (Llama 3, 2024), {USERS} students; 'uptrained GQA achieves quality close to multi-head attention with comparable speed to MQA'; uptraining an MHA checkpoint uses '5% of original pre-training compute'. Student counts: 56 GB of KV room on an H100 (Episode 5), activations ignored.")
        sq, pitch = 0.2, 0.28
        qy, ky = 1.5, 0.35
        qx = [(i - (HEADS - 1) / 2) * pitch + 0.7 for i in range(HEADS)]
        queries = m.VGroup(*[head_sq(sq, TOKEN, GREEN_FILL).move_to([x, qy, 0]) for x in qx])
        q_lab = text("query heads", 22, TOKEN, weight=m.BOLD).next_to(queries, m.LEFT, buff=0.3)
        mha_x = qx
        mqa_x = [float(np.mean(qx))] * HEADS
        gqa_x = [float(np.mean(qx[GROUP * (k // GROUP): GROUP * (k // GROUP) + GROUP])) for k in range(HEADS)]

        def kv_at(xs_):
            return m.VGroup(*[kv_cell(sq).move_to([x, ky, 0]) for x in xs_])

        def links(xs_):
            return m.VGroup(*[m.Line([qx[k], qy - sq / 2 - 0.04, 0], [xs_[k], ky + sq / 2 + 0.04, 0]).set_stroke(MUTED, 1.6)
                              for k in range(HEADS)])

        kvs, lines = kv_at(mha_x), links(mha_x)
        kv_lab = text("key-value heads", 22, KV, weight=m.BOLD).next_to(kvs, m.LEFT, buff=0.3)
        kv_lab.align_to(q_lab, m.RIGHT)

        def mode(label, source):
            p = pill(label, INK, CARD_HI, 22)
            g = m.VGroup(p, cite(source)).arrange(m.RIGHT, buff=0.3)
            return g.move_to([0.7, 2.3, 0])

        mha_mode = mode("multi-head attention · MHA", "Vaswani et al., 2017")
        mqa_mode = mode("multi-query attention · MQA", "Shazeer, 2019")
        gqa_mode = mode("grouped-query attention · GQA", "Ainslie et al., 2023")

        x0, per = -3.2, 5.0 / MHA_TOKEN
        cap = cite(f"KV cache per token · students per H100 at 8K tokens (Episode 5)").move_to([-6.3, -0.35, 0], aligned_edge=m.LEFT)

        def row(y, name, short, nbytes, n_students):
            label = m.VGroup(text(name, 22, INK, weight=m.BOLD), code(short, 20, MUTED)).arrange(m.RIGHT, buff=0.15)
            label.move_to([-6.3, y, 0], aligned_edge=m.LEFT)
            w = max(per * nbytes, 0.04)
            b = m.Rectangle(width=w, height=0.36).set_stroke(width=0).set_fill(KV, 1).move_to([x0 + w / 2, y, 0])
            size = code(nice_bytes(nbytes), 22, KV, weight=m.BOLD).next_to(b, m.RIGHT, buff=0.2)
            who = code(f"· {n_students} students", 22, REQUEST).next_to(size, m.RIGHT, buff=0.22)
            return m.VGroup(label, b, size), who

        mha_row, mha_who = row(-0.95, "multi-head", "MHA", MHA_TOKEN, USERS_MHA)
        mqa_row, mqa_who = row(-1.65, "multi-query", "MQA", MQA_TOKEN, USERS_MQA)
        gqa_row, gqa_who = row(-2.35, "grouped-query", "GQA", KV_TOKEN, USERS)
        bad = pill("quality can drop", RED, RED_FILL, 20).next_to(mqa_who, m.RIGHT, buff=0.35)
        good = pill("quality close to MHA", GREEN, GREEN_FILL, 20).next_to(gqa_who, m.RIGHT, buff=0.35)
        mascot = owl(0.36).next_to(good, m.RIGHT, buff=0.25)
        assert mascot.get_right()[0] <= 6.6  # ≥ 0.5 inside the right edge
        cost = price("chosen in training · uptraining costs ≈ 5% of the original compute", AMBER, AMBER_FILL)
        cost.move_to([0, -3.15, 0])
        grow = lambda r: m.AnimationGroup(m.FadeIn(r[0]), m.GrowFromEdge(r[1], m.LEFT), m.FadeIn(r[2], shift=0.1 * m.LEFT))

        self.say(
            "[First: share the heads.] [Owl's attention has thirty-two query heads in every layer.]",
            self.new_header("01 · share", "Share the heads"),
            m.AnimationGroup(m.FadeIn(q_lab), m.LaggedStart(*[m.FadeIn(q, scale=0.5) for q in queries], lag_ratio=0.03)),
        )
        self.say(
            "[In classic multi-head attention], [each query head has its own key and value head.]",
            m.FadeIn(mha_mode, shift=0.1 * m.DOWN),
            m.AnimationGroup(m.FadeIn(kv_lab),
                             m.LaggedStart(*[m.AnimationGroup(m.Create(l), m.FadeIn(h, shift=0.1 * m.DOWN)) for l, h in zip(lines, kvs)],
                                           lag_ratio=0.04)),
        )
        self.say(
            "[For Owl, that would be five hundred and twelve kibibytes per token]: [room for just thirteen students.]",
            m.AnimationGroup(m.FadeIn(cap), grow(mha_row)),
            m.FadeIn(mha_who, shift=0.1 * m.LEFT),
        )
        self.say(
            "[Multi-query attention] goes to the other extreme: [one key-value head, shared by all thirty-two.]",
            m.AnimationGroup(m.FadeOut(mha_mode, shift=0.1 * m.UP), m.FadeIn(mqa_mode, shift=0.1 * m.UP)),
            m.AnimationGroup(m.Transform(kvs, kv_at(mqa_x)), m.Transform(lines, links(mqa_x)), run_time=1.6),
        )
        self.say(
            "[The cache shrinks thirty-two times], [but quality can drop], and training can turn unstable.",
            m.AnimationGroup(grow(mqa_row), m.FadeIn(mqa_who, shift=0.1 * m.LEFT)),
            m.FadeIn(bad, shift=0.1 * m.LEFT),
        )
        self.say(
            "[Grouped-query attention sits in between]: [eight key-value heads, each shared by a group of four.]",
            m.AnimationGroup(m.FadeOut(mqa_mode, shift=0.1 * m.UP), m.FadeIn(gqa_mode, shift=0.1 * m.UP)),
            m.AnimationGroup(m.Transform(kvs, kv_at(gqa_x)), m.Transform(lines, links(gqa_x)), run_time=1.6),
        )
        self.say(
            "[That's what Owl uses]: four times smaller than multi-head, [with quality close to it.]",
            m.AnimationGroup(grow(gqa_row), m.FadeIn(gqa_who, shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.FadeIn(good, shift=0.1 * m.LEFT), m.FadeIn(mascot, shift=0.1 * m.UP)),
        )
        self.say(
            "[The price is paid in training]: a model is built this way, or [uptrained] with about five percent of its original compute.",
            m.FadeIn(cost[0], scale=0.9),
            m.FadeIn(cost[1], shift=0.1 * m.LEFT),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 2 · compress (MLA)

    def compress(self):
        self.next_section("2 · Compress into a latent", notes=f"Owl (GQA) caches, per layer and token, K and V: 2 × {KV_HEADS} × {OWL['head_dim']} = {OWL_LAYER_NUMBERS:,} numbers = 4 KiB in BF16. Multi-head latent attention (MLA, DeepSeek-V2, arXiv 2405.04434) caches instead a {DSV3['kv_lora_rank']}-number latent per layer plus a decoupled {DSV3['rope_dim']}-number RoPE key = {MLA_NUMBERS} numbers; at attention time the latent is up-projected to keys and values for all 128 heads, or the up-projections are absorbed into the query and output matrices (V2 §2.1.2) — either way, more math per byte read. DeepSeek-V3 (config.json: {DSV3['layers']} layers, kv_lora_rank 512, qk_rope_head_dim 64; 671 B parameters, {PARAM_RATIO:.0f}x Owl's 8.03 B) stores {DSV3['layers']} × {MLA_NUMBERS} × 2 B = {MLA_TOKEN:,} B = {MLA_TOKEN / KIB:.1f} KiB per token in BF16 vs Owl's 128 KiB ({MLA_TOKEN / KV_TOKEN:.2f}x). DeepSeek-V2 vs DeepSeek 67B: 'reduces the KV cache by 93.3%'. Bar widths are proportional to element counts.")
        s = 6.0 / OWL_LAYER_NUMBERS  # units per cached number
        ay, by = 1.85, 0.45
        a_lab = text("Owl caches, per layer and per token:", 22, INK).move_to([-6.3, ay + 0.55, 0], aligned_edge=m.LEFT)
        half = OWL_LAYER_NUMBERS // 2
        k_bar = m.Rectangle(width=s * half - 0.04, height=0.42).set_stroke(KV, 2).set_fill(VIOLET_FILL, 1)
        v_bar = k_bar.copy()
        kv_bar = m.VGroup(k_bar, v_bar).arrange(m.RIGHT, buff=0.04).move_to([-6.3, ay, 0], aligned_edge=m.LEFT)
        k_t = code(f"K · {half:,}", 20, INK).move_to(k_bar)
        v_t = code(f"V · {half:,}", 20, INK).move_to(v_bar)
        a_size = title_text("4 KiB", 40, KV).next_to(kv_bar, m.RIGHT, buff=0.4)
        row_a = m.VGroup(a_lab, kv_bar, k_t, v_t, a_size)
        no = pill("MLA: not cached", RED, RED_FILL, 20).next_to(a_size, m.RIGHT, buff=0.45)

        tok = chip("token", TOKEN, GREEN_FILL, width=1.1, height=0.56, size=22).move_to([-5.85, by, 0])
        f_in = funnel(-4.8, -3.95, by, 1.0, 0.36)
        a1 = arrow(tok.get_right(), [f_in.get_left()[0], by, 0], MUTED, buff=0.06)
        lat = m.Rectangle(width=s * DSV3["kv_lora_rank"], height=0.34).set_stroke(width=0).set_fill(KV, 1)
        lat.move_to([-3.8 + lat.width / 2, by, 0])
        rope = m.Rectangle(width=s * DSV3["rope_dim"], height=0.34).set_stroke(width=0).set_fill("#C9B8FF", 1)
        rope.move_to([lat.get_right()[0] + 0.03 + rope.width / 2, by, 0])
        box = m.DashedVMobject(m.SurroundingRectangle(m.VGroup(lat, rope), buff=0.14, corner_radius=0.08), num_dashes=34).set_stroke(KV, 2.2)
        cached = code("cached", 20, KV, weight=m.BOLD).next_to(box, m.UP, buff=0.12)
        lat_t = m.VGroup(code(f"{MLA_NUMBERS} numbers", 22, KV, weight=m.BOLD),
                         text(f"{DSV3['kv_lora_rank']} latent + {DSV3['rope_dim']} for position", 20, MUTED)).arrange(m.DOWN, buff=0.1)
        lat_t.move_to([box.get_x(), by - 0.95, 0])
        f_out = funnel(box.get_right()[0] + 0.15, box.get_right()[0] + 1.0, by, 0.36, 1.0)
        kx0, kw = f_out.get_right()[0] + 0.15, 5.6

        def ghost(y, label):
            r = m.DashedVMobject(m.Rectangle(width=kw, height=0.4), num_dashes=48).set_stroke(MUTED, 1.8).move_to([kx0 + kw / 2, y, 0])
            return m.VGroup(r, code(label, 20, MUTED).move_to(r))

        k_ghost = ghost(by + 0.24, "K · rebuilt for all 128 heads")
        v_ghost = ghost(by - 0.24, "V · rebuilt, never stored")
        squeeze_t = code("squeeze", 20, COMPUTE).next_to(f_in, m.UP, buff=0.12)
        unpack_t = code("unpack", 20, COMPUTE).next_to(f_out, m.UP, buff=0.12)
        trade = code("+ math   − memory", 22, INK, t2c={r"\+ math": COMPUTE, "− memory": KV}).move_to([kx0 + kw / 2, by - 0.95, 0])

        cy0, cy1, cx = -1.45, -2.1, -1.0
        cper = 3.6 / KV_TOKEN

        def crow(y, name, nbytes):
            lab = code(name, 20, INK).move_to([cx - 0.25, y, 0], aligned_edge=m.RIGHT)
            w = cper * nbytes
            b = m.Rectangle(width=w, height=0.38).set_stroke(width=0).set_fill(KV, 1).move_to([cx + w / 2, y, 0])
            v = code(f"{nbytes / KIB:.3g} KiB per token", 20, KV, weight=m.BOLD).next_to(b, m.RIGHT, buff=0.2)
            return m.VGroup(lab, b, v)

        owl_row = crow(cy0, f"Owl-8B · {OWL['params'] / 1e9:.0f} B params", KV_TOKEN)
        ds_row = crow(cy1, f"DeepSeek-V3 · {DSV3['params'] / 1e9:.0f} B params", MLA_TOKEN)
        verdict = m.VGroup(code(f"{PARAM_RATIO:.0f}× the parameters, ½ the cache per token", 20, INK),
                           cite("both BF16 · DeepSeek-AI, 2024 · V3 config")).arrange(m.RIGHT, buff=0.4).move_to([0, -2.72, 0])
        cost = m.VGroup(pill("price", RED, RED_FILL, 20), text("more math per byte read", 22, INK),
                        text("· decode can afford it", 22, MEMORY), text("· built into the model", 22, INK))
        cost.arrange(m.RIGHT, buff=0.22).move_to([0, -3.32, 0])
        grow = lambda r: m.AnimationGroup(m.FadeIn(r[0]), m.GrowFromEdge(r[1], m.LEFT), m.FadeIn(r[2], shift=0.1 * m.LEFT))

        self.say(
            "[Second: compress.] Multi-head latent attention, or MLA, [doesn't cache keys and values] at all.",
            m.AnimationGroup(self.new_header("02 · compress", "Squeeze it into a latent"), m.FadeIn(row_a, shift=0.1 * m.UP)),
            m.AnimationGroup(m.VGroup(kv_bar, k_t, v_t, a_size).animate.set_opacity(0.4), m.FadeIn(no, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[It squeezes each token into one small latent vector]: [five hundred and twelve numbers], [plus sixty-four that carry position.]",
            m.AnimationGroup(m.FadeIn(tok, shift=0.1 * m.RIGHT), m.GrowArrow(a1), m.FadeIn(f_in), m.FadeIn(squeeze_t)),
            m.AnimationGroup(m.GrowFromEdge(lat, m.LEFT), m.Create(box), m.FadeIn(cached)),
            m.AnimationGroup(m.GrowFromEdge(rope, m.LEFT), m.FadeIn(lat_t, shift=0.1 * m.UP)),
        )
        self.say(
            "[When attending, the model unpacks keys and values from it on the fly], [spending math to save memory.]",
            m.AnimationGroup(m.FadeIn(f_out), m.FadeIn(unpack_t),
                             m.LaggedStart(m.FadeIn(k_ghost, shift=0.15 * m.RIGHT), m.FadeIn(v_ghost, shift=0.15 * m.RIGHT), lag_ratio=0.3)),
            m.AnimationGroup(m.Indicate(f_in, color=COMPUTE), m.Indicate(f_out, color=COMPUTE), m.FadeIn(trade, shift=0.1 * m.UP)),
        )
        self.say(
            "[The effect is striking]: even DeepSeek-V3, eighty-four times bigger than Owl, [stores about half as much per token.]",
            grow(owl_row),
            m.AnimationGroup(grow(ds_row), m.FadeIn(verdict, shift=0.1 * m.UP)),
        )
        self.say(
            "[The price: more math for every byte read], [a trade that memory-bound decode can afford]; [and like grouped-query, it's built into the model.]",
            m.AnimationGroup(m.FadeIn(cost[0], scale=0.9), m.FadeIn(cost[1], shift=0.1 * m.LEFT)),
            m.FadeIn(cost[2], shift=0.1 * m.LEFT),
            m.FadeIn(cost[3], shift=0.1 * m.LEFT),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 3 · forget (sliding window)

    def forget(self):
        self.next_section("3 · Keep only a window", notes=f"Sliding-window attention: each layer attends only to the last W tokens, so its cache is a rolling buffer of W slots; token i goes to slot i mod W (Mistral 7B paper, arXiv 2310.06825). The 8-slot buffer and its token numbers are schematic. Mistral 7B (Table 1: 32 layers, 8 KV heads, head_dim 128 — the same {nice_bytes(MISTRAL_TOKEN)} per token as Owl; W = 4,096): at 32K tokens a full cache is {nice_bytes(FULL_LONG)}, the window {nice_bytes(WIN_LONG)}: 8x smaller, matching the paper's 'reduces the cache memory usage by 8x, without impacting the model quality'. The price, measured on this Mac (research/measurements.md §6, mlx-lm 0.32.2): Owl-mini (Llama-3.2-1B 4-bit) with RotatingKVCache(max_size={ROT_WINDOW}, keep=4 — mlx-lm's --max-kv-size is silently ignored for Llama, so the harness built it explicitly); a {DEMO['prompt_tokens']:,}-token prompt starting 'Remember this: the secret code word is PELICAN' and ending with the question: full cache → '{DEMO['full'].split('<')[0]}', window → '{DEMO['rotating_1024']}'. KV pinned at 32 MiB. Gemma 3 (arXiv 2503.19786): '5 local layers for every global layer', local span {GEMMA_SPAN}. Attention sinks and long contexts: Episode 9.")
        n_slots, size, pitch = 8, 0.6, 0.7
        sy = 1.0
        sx = [-6.0 + size / 2 + i * pitch for i in range(n_slots)]
        slots = m.VGroup(*[kv_cell(size).move_to([x, sy, 0]) for x in sx])
        s_top = code("rolling buffer · slot = token mod 8", 20, MUTED).next_to(slots, m.UP, buff=0.25).align_to(slots, m.LEFT)
        s_bot = code("window W = 8 slots (schematic)", 20, MUTED).next_to(slots, m.DOWN, buff=0.25).align_to(slots, m.LEFT)
        first = [code(str(t + 1), 24, INK, weight=m.BOLD).move_to(slots[t]) for t in range(n_slots)]
        later = [code(str(n_slots + t + 1), 24, INK, weight=m.BOLD).move_to(slots[t]) for t in range(4)]

        def write(t):
            return m.AnimationGroup(m.FadeIn(first[t], shift=0.25 * m.DOWN), slots[t].animate(rate_func=m.there_and_back).set_stroke(TOKEN))

        def overwrite(t):
            return m.Succession(first[t].animate(run_time=0.25).set_color(RED),
                                m.AnimationGroup(m.FadeOut(first[t], shift=0.25 * m.DOWN), m.FadeIn(later[t], shift=0.25 * m.DOWN), run_time=0.5))

        # the chart: cache size vs tokens, full vs a 4,096-token window (Mistral 7B)
        ox, oy, cw, ch = 0.6, -0.5, 4.6, 1.85

        def p(t, nbytes):
            return np.array([ox + cw * t / LONG, oy + ch * nbytes / FULL_LONG, 0])

        x_axis = m.Arrow(p(0, 0), p(LONG, 0) + [0.3, 0, 0], buff=0, color=FAINT, stroke_width=3, max_tip_length_to_length_ratio=0.05)
        y_axis = m.Arrow(p(0, 0), p(0, FULL_LONG) + [0, 0.25, 0], buff=0, color=FAINT, stroke_width=3, max_tip_length_to_length_ratio=0.1)
        xt = m.VGroup(*[code(lab, 20, MUTED).next_to(p(t, 0), m.DOWN, buff=0.2) for t, lab in ((0, "0"), (16384, "16K"), (LONG, "32K"))])
        x_name = code("tokens", 20, MUTED).next_to(xt[2], m.RIGHT, buff=0.35)
        title = m.VGroup(
            m.VGroup(code(f"Mistral 7B · W = {MISTRAL['window']:,}", 20, INK), cite("Jiang et al., 2023")).arrange(m.RIGHT, buff=0.3),
            code(f"{nice_bytes(MISTRAL_TOKEN)} per token, like Owl", 20, MUTED),
        ).arrange(m.DOWN, buff=0.14, aligned_edge=m.LEFT).move_to([ox, 2.12, 0], aligned_edge=m.LEFT)
        full = m.DashedLine(p(0, 0), p(LONG, FULL_LONG), dash_length=0.12).set_stroke(KV, 3, opacity=0.55)
        win = m.VGroup(m.Line(p(0, 0), p(MISTRAL["window"], WIN_LONG)), m.Line(p(MISTRAL["window"], WIN_LONG), p(LONG, WIN_LONG))).set_stroke(KV, 4)
        full_lab = code(f"full · {nice_bytes(FULL_LONG)}", 20, KV).move_to(p(13500, FULL_LONG * 0.84))
        win_lab = code(f"window · {nice_bytes(WIN_LONG)}", 20, KV, weight=m.BOLD).move_to(p(23500, WIN_LONG) + [0, 0.25, 0])
        brk = tick_bracket(p(LONG, 0)[0] + 0.45, p(0, WIN_LONG)[1], p(0, FULL_LONG)[1], INK)
        eight = title_text(f"{FULL_LONG // WIN_LONG}×", 40, INK).next_to(brk, m.LEFT, buff=0.2).set_y(p(0, (WIN_LONG + FULL_LONG) / 2)[1] - 0.1)
        chart = m.VGroup(x_axis, y_axis, xt, x_name, title)

        # the price, measured on this Mac
        d_title = text(f"Measured on this Mac: Owl-mini, a {DEMO['prompt_tokens']:,}-token prompt with a code word in its first line, asked at the end",
                       20, MUTED).move_to([0, -1.2, 0])

        def answer(where, said, color, mark, x):
            box = card(6.1, 1.0, color, CARD, radius=0.14, stroke=2).move_to([x, -2.0, 0])
            head = code(where, 20, MUTED).align_to(box, m.UL).shift(0.3 * m.RIGHT + 0.14 * m.DOWN)
            body = code(said, 24, color, weight=m.BOLD).align_to(box, m.DL).shift(0.3 * m.RIGHT + 0.16 * m.UP)
            tick = code(mark, 30, color, weight=m.BOLD).move_to(box).align_to(box, m.RIGHT).shift(0.3 * m.LEFT)
            return m.VGroup(box, head, body, tick)

        ok = answer("full cache answers", DEMO["full"].split("<")[0], GREEN, "✓", -3.25)
        lost = answer(f"{ROT_WINDOW:,}-token window answers", f"\"{FORGOT}…\"", RED, "✗", 3.25)
        gemma = text(f"Gemma 3: {GEMMA_LOCAL} windowed layers ({GEMMA_SPAN:,} tokens) for every full-attention layer", 22, INK).move_to([0, -2.88, 0])
        g_src = m.VGroup(cite("Gemma Team, 2025"), code("attention sinks, long chats → Episode 9", 20, MUTED)).arrange(m.RIGHT, buff=0.6)
        g_src.move_to([0, -3.3, 0])

        self.say(
            "[Third: forget.] [With a sliding window, a layer attends only to the most recent tokens.]",
            self.new_header("03 · forget", "Keep only a window"),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c, scale=0.8) for c in slots], lag_ratio=0.08), m.FadeIn(s_top), m.FadeIn(s_bot)),
        )
        self.say(
            "[So its cache becomes a rolling buffer]: [each new token overwrites the oldest slot.]",
            m.LaggedStart(*[write(t) for t in range(n_slots)], lag_ratio=0.3, run_time=2.4),
            m.LaggedStart(*[overwrite(t) for t in range(4)], lag_ratio=0.6, run_time=2.6),
        )
        self.say(
            "[The first Mistral model used a window of about four thousand tokens]: at thirty-two thousand, [its cache is eight times smaller.]",
            m.AnimationGroup(m.FadeIn(chart), m.Create(full), m.FadeIn(full_lab)),
            m.AnimationGroup(m.Create(win), m.FadeIn(win_lab), m.Create(brk), m.FadeIn(eight, scale=0.8)),
        )
        self.say(
            "[The price is the past]: on this Mac, Owl-mini with a thousand-token window [forgot a code word] from the start of its prompt.",
            m.AnimationGroup(m.FadeIn(d_title), m.FadeIn(ok, shift=0.1 * m.UP)),
            m.FadeIn(lost, shift=0.1 * m.UP),
        )
        self.say(
            "[So Gemma 3 mixes the two]: [five windowed layers for every full-attention layer.]",
            m.FadeIn(gemma, shift=0.1 * m.UP),
            m.FadeIn(g_src, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 4 · fewer bits

    def fewer_bits(self):
        self.next_section("4 · Fewer bits per number", notes=f"KV quantization: Owl caches BF16 (16 bits per number) → {nice_bytes(KV_TOKEN)} per token; FP8 (8 bits) → {nice_bytes(FP8_TOKEN)}; INT4 → {nice_bytes(INT4_TOKEN)} plus a scale per group (e.g. MLX: a BF16 scale and offset per 64 numbers). vLLM: kv_cache_dtype = 'fp8' | 'fp8_e4m3' | 'fp8_e5m2' (docs; scales default to 1.0 without calibration). H100 (notes §3 worked example): {KV_ROOM / 1e9:.1f} GB of KV room → {USERS} students at 8K tokens with a BF16 cache, {USERS_FP8} with FP8. The bit boxes and chat segments are schematic. Costs: rounding error (KIVI, ICML 2024, quantizes keys per channel and values per token to reach 2 bits), and dequantization work in the attention kernel.")
        bsz, bp = 0.28, 0.34
        by = 1.65
        bx = [-6.1 + bsz / 2 + i * bp for i in range(16)]
        bits = m.VGroup(*[m.Square(bsz).set_stroke(KV, 1.8).set_fill(VIOLET_FILL, 1).move_to([x, by, 0]) for x in bx])
        b_top = code("one cached number", 20, MUTED).next_to(bits, m.UP, buff=0.22).align_to(bits, m.LEFT)
        scale = m.VGroup(card(1.05, 0.46, MUTED, CARD, radius=0.1, stroke=1.6), code("scale", 20, MUTED))
        scale[1].move_to(scale[0])
        scale.move_to([bx[3] + bsz / 2 + 0.2 + 0.525, by, 0])

        def fmt(s, after):
            return code(s, 24, KV, weight=m.BOLD).move_to([after.get_right()[0] + 0.4, by, 0], aligned_edge=m.LEFT)

        f16, f8, f4 = fmt("BF16 · 16 bits", bits[15]), fmt("FP8 · 8 bits", bits[7]), fmt("INT4 · 4 bits + scale", scale)
        sx = 4.75

        def per_token(nbytes, extra=""):
            g = m.VGroup(title_text(nice_bytes(nbytes), 60, KV), text("per token" + extra, 22, MUTED)).arrange(m.DOWN, buff=0.12)
            return g.move_to([sx, by - 0.1, 0])

        t16, t8, t4 = per_token(KV_TOKEN), per_token(FP8_TOKEN), per_token(INT4_TOKEN, ", plus scales")

        # one H100, as in Episode 5
        x0, width, ty = -5.8, 11.6, -0.4
        per = width / GPU["memory"]
        xb = lambda nbytes: x0 + per * nbytes
        track = m.Rectangle(width=width, height=0.64).set_stroke(EDGE, 2).set_fill(CARD, 1).move_to([x0 + width / 2, ty, 0])

        def seg(a, b, color, h=0.5, opacity=1.0, inset=0.0):
            return m.Rectangle(width=per * (b - a) - inset, height=h).set_stroke(width=0).set_fill(color, opacity).move_to([(xb(a) + xb(b)) / 2, ty, 0])

        gpu = icon(recolored("gpu", MUTED, CARD, PREFIX), 0.5)
        h_title = m.VGroup(gpu, code("H100 · 80 GB of HBM", 22, INK)).arrange(m.RIGHT, buff=0.25).move_to([x0, 0.55, 0], aligned_edge=m.LEFT)
        w_seg = seg(0, weight_bytes(), WEIGHTS)
        r_seg = seg(BUDGET, GPU["memory"], FAINT, opacity=0.6)
        w_lab = code(f"weights {weight_bytes() / 1e9:.0f} GB", 20, WEIGHTS).move_to([(x0 + xb(weight_bytes())) / 2, ty - 0.6, 0])
        k_lab = code(f"cache ≈ {KV_ROOM / 1e9:.0f} GB", 20, KV).move_to([(xb(weight_bytes()) + xb(BUDGET)) / 2, ty - 0.6, 0])
        r_lab = code("10% free", 20, MUTED).move_to([(xb(BUDGET) + xb(GPU["memory"])) / 2, ty - 0.6, 0])
        h100 = m.VGroup(h_title, track, w_seg, r_seg, w_lab, k_lab, r_lab)

        def chats(n, per_tok):
            nbytes = CHAT * per_tok
            segs = m.VGroup(*[seg(weight_bytes() + i * nbytes, weight_bytes() + (i + 1) * nbytes, KV, h=0.46, inset=0.02) for i in range(n)])
            dots = m.VGroup(*[m.Dot(radius=0.05 if n < 60 else 0.032, color=REQUEST).move_to([c.get_x(), ty + 0.5, 0]) for c in segs])
            return m.VGroup(segs, dots)

        c52, c104 = chats(USERS, KV_TOKEN), chats(USERS_FP8, FP8_TOKEN)

        def who(n, sub, x):
            g = m.VGroup(title_text(str(n), 72, REQUEST), m.VGroup(text("students at once", 22, INK), code(sub, 20, MUTED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT))
            g.arrange(m.RIGHT, buff=0.3)
            return g.move_to([x, -1.95, 0])

        w52, w104 = who(USERS, "16-bit cache", -2.6), who(USERS_FP8, "FP8 cache", 2.9)
        w_arrow = arrow(w52.get_right() + [0.3, 0, 0], w104.get_left() - [0.3, 0, 0], MUTED)
        cost = price("rounding error · a little work to unpack each number")
        cost.move_to([0, -3.15, 0])

        self.say(
            "[Fourth: fewer bits.] [Owl stores every cached number in sixteen bits.]",
            self.new_header("04 · fewer bits", "Fewer bits per number"),
            m.AnimationGroup(m.FadeIn(b_top), m.LaggedStart(*[m.FadeIn(b, scale=0.6) for b in bits], lag_ratio=0.05),
                             m.FadeIn(f16), m.FadeIn(t16, shift=0.1 * m.UP)),
        )
        self.say(
            "[Store them as eight-bit FP8], [and each token's cache halves], to sixty-four kibibytes.",
            m.AnimationGroup(m.LaggedStart(*[m.FadeOut(b, scale=0.4) for b in reversed(bits[8:])], lag_ratio=0.1),
                             m.FadeTransform(f16, f8)),
            m.FadeTransform(t16, t8),
        )
        self.say(
            "[With four-bit INT4, it halves again], plus [a small scale] stored for each group of numbers.",
            m.AnimationGroup(m.LaggedStart(*[m.FadeOut(b, scale=0.4) for b in reversed(bits[4:8])], lag_ratio=0.15),
                             m.FadeTransform(f8, f4), m.FadeTransform(t8, t4)),
            m.FadeIn(scale, shift=0.15 * m.LEFT),
        )
        self.say(
            "[Back on one H100], [the same fifty-six gigabytes now hold twice the tokens]: [a hundred and four students] instead of fifty-two.",
            m.AnimationGroup(m.FadeIn(h100), m.FadeIn(c52), m.FadeIn(w52, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeOut(c52), m.LaggedStart(*[m.FadeIn(s, shift=0.05 * m.RIGHT) for s in c104[0]], lag_ratio=0.02),
                             m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in c104[1]], lag_ratio=0.02), run_time=2.0),
            m.AnimationGroup(m.GrowArrow(w_arrow), m.FadeIn(w104, shift=0.15 * m.LEFT)),
        )
        self.say(
            "[The price is rounding error], [and a little extra work] to unpack each number before attention uses it.",
            m.FadeIn(cost[0], scale=0.9),
            m.FadeIn(cost[1], shift=0.1 * m.LEFT),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 5 · measured on this Mac

    def on_this_mac(self):
        self.next_section("5 · Fewer bits, on this Mac", notes=f"research/measurements.md §5 (2026-10-24, Mac Studio M4 Max 546 GB/s, mlx-lm 0.32.2): Llama-3.1-8B, 4-bit weights, KV quantized with kv_bits 8 or 4, group size 64, quantized_kv_start = 0 (mlx-lm's default --quantized-kv-start is 5000: --kv-bits alone does nothing until the cache passes 5,000 tokens). Active memory at 1K…32K tokens; least-squares slopes {SLOPE[16]:,.0f} / {SLOPE[8]:,.0f} / {SLOPE[4]:,.0f} B per token (16 / 8 / 4 bits) = {SLOPE[8] / SLOPE[16]:.3f}x and {SLOPE[4] / SLOPE[16]:.3f}x; predicted {EXPECT[8]:,.0f} and {EXPECT[4]:,.0f} (b bits + a BF16 scale and offset per 64 numbers = 8.5 and 4.5 bits per number). Quality spot-check (8,081-token prompt, 64 greedy tokens): 8-bit identical to the BF16-cache answer ({SAME8}/{GEN} tokens), 4-bit identical for {SAME4} tokens, then rephrased with the same facts. Decode at 32K (1 run of 16 tokens): {SPEED[16]:.1f} tok/s (16-bit cache) vs {SPEED[8]:.1f} (8-bit); ceilings 546 GB/s ÷ ({W_READ / 1e9:.2f} GB of weights read + 32,768 × bytes per token) = {CEIL[16]:.1f} → {CEIL[8]:.1f} tok/s. The quantized attention path (quantized_scaled_dot_product_attention) is not a fused kernel: here KV quantization buys capacity, not speed (peak memory during long prefills even rises: 9.5 → 16.0 GB at 32K).")
        ox, oy, cw, ch = -5.6, -2.3, 5.0, 4.0
        t_max, g_lo, g_hi = 34_000, 4.0, 9.5

        def p(t, gb):
            return np.array([ox + cw * t / t_max, oy + ch * (gb - g_lo) / (g_hi - g_lo), 0])

        x_axis = m.Arrow(p(0, g_lo), p(t_max, g_lo) + [0.3, 0, 0], buff=0, color=FAINT, stroke_width=3, max_tip_length_to_length_ratio=0.03)
        y_axis = m.Arrow(p(0, g_lo), p(0, g_hi) + [0, 0.25, 0], buff=0, color=FAINT, stroke_width=3, max_tip_length_to_length_ratio=0.05)
        xt = m.VGroup(*[code(lab, 20, MUTED).next_to(p(t, g_lo), m.DOWN, buff=0.14) for t, lab in ((0, "0"), (8192, "8K"), (16384, "16K"), (24576, "24K"), (32768, "32K"))])
        yt = m.VGroup(*[code(f"{g}", 20, MUTED).next_to(p(0, g), m.LEFT, buff=0.14) for g in (4, 5, 6, 7, 8, 9)])
        x_title = text("tokens in the conversation", 20, MUTED).next_to(xt, m.DOWN, buff=0.12).set_x(p(t_max / 2, 0)[0])
        y_title = text("memory in use (GB)", 20, MUTED).move_to(p(0, g_hi) + [0.3, 0.22, 0], aligned_edge=m.LEFT)
        mac = icon(recolored("mac", MUTED, CARD, PREFIX), 0.45)
        where = m.VGroup(cite("measured on this Mac"), cite("M4 Max · mlx-lm")).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
        stamp = m.VGroup(mac, where).arrange(m.RIGHT, buff=0.22).move_to([ox + 0.35, 1.25, 0], aligned_edge=m.LEFT)
        axes_g = m.VGroup(x_axis, y_axis, xt, yt, x_title, y_title, stamp)

        def series(b, color, dashed=False, opacity=1.0):
            pts = MAC_POINTS[b]
            fit = np.polyfit([t for t, _ in pts], [g for _, g in pts], 1)
            f = lambda t: fit[1] + fit[0] * t
            a, z = p(0, f(0)), p(32768, f(32768))
            line = (m.DashedLine(a, z, dash_length=0.12) if dashed else m.Line(a, z)).set_stroke(color, 4, opacity=opacity)
            dots = m.VGroup(*[m.Dot(p(t, g), radius=0.07, color=INK).set_z_index(3) for t, g in pts])
            label = code("16-bit" if b == 16 else f"--kv-bits {b}", 20, color, weight=m.BOLD).next_to(z, m.RIGHT, buff=0.18)
            return line, dots, label

        l16, d16, n16 = series(16, MUTED)
        l8, d8, n8 = series(8, KV)
        l4, d4, n4 = series(4, KV, dashed=True)

        px = 4.3
        cx3 = [px - 1.7, px + 0.05, px + 1.7]
        hdr = m.VGroup(*[code(h, 20, MUTED).move_to([x, 2.2, 0]) for h, x in zip(("bits", "bytes/token", "ratio"), cx3)])

        def trow(b, y):
            s = SLOPE[b]
            return m.VGroup(
                code(f"{b}", 24, KV if b < 16 else MUTED, weight=m.BOLD).move_to([cx3[0], y, 0]),
                code(f"{s:,.0f}", 24, INK).move_to([cx3[1], y, 0]),
                code(f"{s / SLOPE[16]:.2f}×", 24, KV if b < 16 else MUTED).move_to([cx3[2], y, 0]),
            )

        r16, r8, r4 = trow(16, 1.72), trow(8, 1.27), trow(4, 0.82)
        ratios = m.VGroup(r8[2], r4[2])
        why = text("+ a scale and an offset per 64 numbers", 20, INK).move_to([px, 0.22, 0])
        bits_per = code("= 8.5 and 4.5 bits each", 20, MUTED).move_to([px, -0.16, 0])
        pred = code(f"expected {EXPECT[8]:,.0f} · {EXPECT[4]:,.0f} ✓", 20, INK, t2c={"✓": GOOD}).move_to([px, -0.56, 0])
        q_head = text("same tokens as the 16-bit answer", 20, MUTED).move_to([px, -1.18, 0])
        q_body = code(f"8-bit {SAME8}/{GEN} · 4-bit {SAME4}/{GEN}", 20, INK, t2c={f"{SAME8}/{GEN}": GOOD, f"{SAME4}/{GEN}": AMBER}).move_to([px, -1.54, 0])
        s_head = text("decode at 32K: 16-bit vs 8-bit cache", 20, MUTED).move_to([px, -2.16, 0])
        s_meas = code(f"measured {SPEED[16]:.0f} → {SPEED[8]:.0f} tok/s", 22, INK).move_to([px, -2.54, 0])
        s_ceil = code(f"ceiling  {CEIL[16]:.0f} → {CEIL[8]:.0f} tok/s", 22, MEMORY).move_to([px, -2.92, 0])
        s_ceil.align_to(s_meas, m.LEFT)

        self.say(
            "[On this Mac, we switched on MLX's cache quantization], [at eight bits, and at four.]",
            m.AnimationGroup(self.new_header("05 · measured", "Fewer bits, on this Mac"), m.FadeIn(axes_g),
                             m.FadeIn(d16), m.Create(l16), m.FadeIn(n16)),
            m.AnimationGroup(m.FadeIn(n8, shift=0.1 * m.LEFT), m.FadeIn(n4, shift=0.1 * m.LEFT), m.FadeIn(hdr), m.FadeIn(r16)),
        )
        self.say(
            "[Each line still climbs straight], [but the slope falls from a hundred and twenty-eight kibibytes per token], [to sixty-eight, and then thirty-six.]",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in d8], lag_ratio=0.15),
                             m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in d4], lag_ratio=0.15)),
            m.AnimationGroup(m.Create(l8), m.Create(l4), m.Circumscribe(r16, color=MUTED, buff=0.08)),
            m.LaggedStart(m.FadeIn(r8, shift=0.1 * m.UP), m.FadeIn(r4, shift=0.1 * m.UP), lag_ratio=0.5),
        )
        self.say(
            "[Not quite a half and a quarter]: [every sixty-four numbers] share a scale and an offset, and [the prediction matches] within five bytes.",
            m.Circumscribe(ratios, color=KV, buff=0.1),
            m.AnimationGroup(m.FadeIn(why, shift=0.1 * m.UP), m.FadeIn(bits_per, shift=0.1 * m.UP)),
            m.FadeIn(pred, shift=0.1 * m.UP),
        )
        self.say(
            "[And the answers?] [At eight bits], a long test answer came out word for word the same; [at four], it drifted after forty-five tokens, but kept the facts.",
            m.FadeIn(q_head),
            m.FadeIn(q_body[: len(f"8-bit{SAME8}/{GEN}·")], shift=0.1 * m.UP),
            m.FadeIn(q_body[len(f"8-bit{SAME8}/{GEN}·"):], shift=0.1 * m.UP),
        )
        self.say(
            "[One catch on this Mac]: [quantized attention isn't one fused kernel], so [the ceiling rises], but the measured speed stays put.",
            m.FadeIn(s_head),
            m.FadeIn(s_meas, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(s_ceil, shift=0.1 * m.UP), m.Circumscribe(s_meas, color=INK, buff=0.08)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 6 · the bill

    def bill(self):
        self.next_section("6 · What each trick costs", notes=f"Summary of sections 1–5. Shrinks: MQA n_kv 32 → 1 (÷32 vs MHA, {nice_bytes(MQA_TOKEN)}); GQA n_kv 32 → 8 (÷4, {nice_bytes(KV_TOKEN)}, Owl); MLA 2·n_kv·d_h → {MLA_NUMBERS} numbers per layer (DeepSeek-V3 {MLA_TOKEN / KIB:.1f} KiB ≈ ½ Owl's at {PARAM_RATIO:.0f}x the parameters); window T → W (Mistral 7B 8x at 32K); FP8 / INT4 b 2 → 1 / ½ byte. Costs: MQA quality and training stability (Ainslie et al. 2023); GQA, MLA set in training (uptraining ≈ 5% of pre-training compute for GQA); MLA more math per byte; windows forget (measured on this Mac); quantization rounding error and dequantization work (on this Mac's mlx-lm, no speed-up). Stacking on one H100 at 8K tokens: MHA {USERS_MHA} → GQA {USERS} → GQA + FP8 {USERS_FP8} students (notes §3).")
        cols = [-6.3, -3.85, -0.7, 2.3]
        heads = ["TRICK", "SHRINKS", "SAVES", "COSTS"]
        rows = [
            ("multi-query", f"n_kv {HEADS} → 1", f"÷{MHA_TOKEN // MQA_TOKEN} vs MHA", "quality, stability"),
            ("grouped-query", f"n_kv {HEADS} → {KV_HEADS}", f"÷{MHA_TOKEN // KV_TOKEN} · {nice_bytes(KV_TOKEN)}", "fixed in training"),
            ("latent (MLA)", f"2·n_kv·d_h → {MLA_NUMBERS}", f"≈ ½, at {PARAM_RATIO:.0f}× size", "more math, in training"),
            ("sliding window", f"T → W", f"÷{FULL_LONG // WIN_LONG} at 32K", "forgets the past"),
            ("FP8 · INT4", "b 2 → 1 or ½ B", "÷2 · ÷4 + scales", "rounding, unpacking"),
        ]
        top = 2.1
        head_row = m.VGroup(*[code(h, 20, MUTED).move_to([x, top, 0], aligned_edge=m.LEFT) for h, x in zip(heads, cols)])
        rule = m.Line([-6.3, top - 0.3, 0], [6.4, top - 0.3, 0]).set_stroke(EDGE, 2)
        table = m.VGroup()
        for i, (name, shrinks, saves, costs) in enumerate(rows):
            y = top - 0.78 - 0.6 * i
            table.add(m.VGroup(
                text(name, 22, INK, weight=m.BOLD).move_to([cols[0], y, 0], aligned_edge=m.LEFT),
                code(shrinks, 20, KV).move_to([cols[1], y, 0], aligned_edge=m.LEFT),
                code(saves, 20, GREEN).move_to([cols[2], y, 0], aligned_edge=m.LEFT),
                text(costs, 22, RED).move_to([cols[3], y, 0], aligned_edge=m.LEFT),
            ))
        costs_col = m.VGroup(*[r[3] for r in table])
        when = m.VGroup(pill("MQA · GQA · MLA: fixed when the model is trained", WEIGHTS, BLUE_FILL, 20),
                        pill("FP8 · INT4: a switch when serving", INK, CARD_HI, 20)).arrange(m.RIGHT, buff=0.4).move_to([0, -1.88, 0])

        def step(n, label):
            g = m.VGroup(title_text(str(n), 60, REQUEST), text(label, 22, INK)).arrange(m.RIGHT, buff=0.22)
            return g

        stack = m.VGroup(step(USERS_MHA, "multi-head"), step(USERS, "grouped-query"), step(USERS_FP8, "+ FP8 cache")).arrange(m.RIGHT, buff=1.3)
        stack.move_to([0, -2.82, 0])
        for g in stack[1:]:
            g.align_to(stack[0], m.DOWN)
        s_arrows = m.VGroup(*[arrow(a.get_right() + [0.15, 0, 0], b.get_left() - [0.15, 0, 0], MUTED) for a, b in zip(stack, stack[1:])])
        s_note = cite("students per H100 at 8K tokens").next_to(stack, m.DOWN, buff=0.15)

        self.say(
            "[Here's the bill.] [Each trick shrinks one part of the formula], [and pays for it somewhere else.]",
            m.AnimationGroup(self.new_header("06 · the bill", "What each trick costs"), m.FadeIn(head_row), m.Create(rule)),
            m.LaggedStart(*[m.FadeIn(r[:3], shift=0.1 * m.UP) for r in table], lag_ratio=0.25),
            m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.LEFT) for c in costs_col], lag_ratio=0.2),
        )
        self.say(
            "[Heads and latents] are fixed when the model is trained; [cache precision] is a switch you flip when serving.",
            m.FadeIn(when[0], shift=0.1 * m.UP),
            m.FadeIn(when[1], shift=0.1 * m.UP),
        )
        self.say(
            "[And they stack]: [multi-head Owl would fit thirteen students], [grouped-query fits fifty-two], [and FP8 on top, a hundred and four.]",
            m.FadeIn(s_note),
            m.FadeIn(stack[0], shift=0.1 * m.UP),
            m.AnimationGroup(m.GrowArrow(s_arrows[0]), m.FadeIn(stack[1], shift=0.1 * m.UP)),
            m.AnimationGroup(m.GrowArrow(s_arrows[1]), m.FadeIn(stack[2], shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Share (fewer key-value heads: MQA, GQA), squeeze (a latent, MLA; fewer bits, FP8 / INT4), forget (a sliding window). Every byte saved is paid for in quality, in math, or in what the model can still see. Next: Episode 7, PagedAttention — fixed-size blocks that end fragmentation.")
        words = [("Share", "fewer key-value heads"), ("Squeeze", "a latent, fewer bits"), ("Forget", "a window of recent tokens")]
        cols = m.VGroup()
        for w, sub in words:
            cols.add(m.VGroup(title_text(w, 72, KV), text(sub, 22, MUTED)).arrange(m.DOWN, buff=0.25))
        cols.arrange(m.RIGHT, buff=1.3).move_to([0, 0.75, 0])
        base = 0.85
        for c in cols:
            c[0].shift((base - c[0][0].get_bottom()[1]) * m.UP)  # first letters (S, S, F) sit on the baseline
            c[1].move_to([c[0].get_x(), base - 0.55, 0])
        mid = base + 0.3
        arrows = m.VGroup(*[
            arrow([a[0].get_right()[0] + 0.15, mid, 0], [b[0].get_left()[0] - 0.15, mid, 0], MUTED) for a, b in zip(cols, cols[1:])
        ])
        tagline = text("Every byte saved is paid for: in quality, in math, or in memory of the past.", 30, INK,
                       t2c={"quality": RED, "math": COMPUTE, "memory of the past": KV}).move_to([0, -1.35, 0])
        mascot = owl(0.75).move_to([0, -2.7, 0])

        self.say(
            "[So: share], [squeeze], [and forget.]",
            m.AnimationGroup(self.new_header("recap", "Shrinking the cache in three words"), m.FadeIn(cols[0], shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowArrow(arrows[0]), m.FadeIn(cols[1], shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowArrow(arrows[1]), m.FadeIn(cols[2], shift=0.2 * m.UP)),
        )
        self.say(
            "[Every byte you save is paid for somewhere]: [in quality, in math, or in memory of the past.]",
            m.FadeIn(tagline, shift=0.1 * m.UP),
            m.FadeIn(mascot, shift=0.2 * m.UP),
        )
        self.wait(0.4)
        self.clear()
