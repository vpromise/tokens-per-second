"""Tokens per Second · Episode 20 · Mixture of Experts at Serve Time (Module 5 · Fewer Bits, More GPUs).

A mixture of experts reads few experts per token but must hold all of them. Episode 2's speed limit
(tokens/s ≈ bandwidth ÷ bytes per token) at batch 1: the Mixtral model holds 46.7 B parameters and reads
12.9 B per token, so it decodes like a 13 B dense model (ceiling on one H200: 186 vs 51 tok/s for a dense
model of its full size). Then Episode 4's batch: every student's token lights its own two experts, so the
bytes per step climb from 25.8 GB to the whole 93.4 GB by batch 16 (expected experts lit = E·(1 − (1 − k/E)^B))
and the free lunch shrinks (830 tok/s at batch 16 = ×4.5, not ×16). DeepSeek-V3 (256 experts, top 8) needs
thousands of tokens in flight, more memory than a server, hence expert parallelism: all-to-all dispatch and
combine, DeepEP's measured latencies, overlap, and DeepSeek's production decode unit (EP144, 32 spare experts).

No MoE was run on this Mac: every number is computed below from the published configs (Mixtral config.json,
DeepSeek-V3 config.json), owl_reference.HARDWARE, or cited from research/notes.md §9–§10 and the sources it
names. The routings drawn (which experts each token picks) are seeded random examples, labelled as such.

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
H200 = HARDWARE["H200"]

# Mixtral 8x7B, from its config.json (dim 4096, 32 layers, SwiGLU experts of 14336, 32 heads, 8 KV heads,
# vocab 32000, 8 experts, top 2). Mistral (2023): "46.7B total parameters but only uses 12.9B parameters per token".
MX = dict(dim=4096, layers=32, head_dim=128, hidden=14336, heads=32, kv_heads=8, vocab=32000, experts=8, top_k=2)
E, K = MX["experts"], MX["top_k"]
_ffn = 3 * MX["dim"] * MX["hidden"]                                                        # one expert, one layer
_attn = 2 * MX["dim"] * MX["heads"] * MX["head_dim"] + 2 * MX["dim"] * MX["kv_heads"] * MX["head_dim"]
EXPERT = MX["layers"] * _ffn                                                               # one expert, all layers
SHARED = MX["layers"] * (_attn + 2 * MX["dim"] + MX["dim"] * E) + 2 * MX["vocab"] * MX["dim"] + MX["dim"]
TOTAL = E * EXPERT + SHARED
ACTIVE = K * EXPERT + SHARED
assert round(TOTAL / 1e9, 1) == 46.7 and round(ACTIVE / 1e9, 1) == 12.9                    # "46.7", "12.9"
HELD_GB, READ_GB = 2 * TOTAL / GB, 2 * ACTIVE / GB                                          # BF16: 2 bytes each
assert round(HELD_GB, 1) == 93.4 and round(READ_GB, 1) == 25.8 and round(READ_GB) == 26  # "about twenty-six"
OWL_B = OWL["params"] / 1e9
assert round(weight_bytes() / GB) == 16                                                    # Owl-8B: 16 GB


def lit(b, e=E, k=K):
    """Expected distinct experts a batch of b tokens reads in one layer (uniform, independent routing)."""
    return e * (1 - (1 - k / e) ** b)


def step_gb(b):
    """GB of Mixtral weights one decode step reads at batch b (BF16, weights only)."""
    return 2 * (SHARED + lit(b) * EXPERT) / GB


def total_tps(b, gb):
    """Tokens/s of the whole batch on one H200 when a step reads `gb` (memory-bound: checked below)."""
    return b * H200["bandwidth"] / (gb * GB)


CEIL_MOE = total_tps(1, READ_GB)
CEIL_DENSE = total_tps(1, HELD_GB)
assert round(CEIL_MOE) == 186 and round(CEIL_DENSE) == 51 and H200["memory"] / GB == 141  # "186", "51", "141 GB"
assert round(lit(2), 1) == 3.5 and round(lit(4), 1) == 5.5 and round(lit(16), 1) == 7.9   # "five and a half", "seven point nine"
assert round(0.75 ** 16, 2) == 0.01                                                        # "one chance in a hundred"
assert round(step_gb(1), 1) == 25.8 and 92 < step_gb(16) < 93 and round(step_gb(16), 1) == 92.5  # "over ninety-two"
T16, D47, D13 = total_tps(16, step_gb(16)), total_tps(16, HELD_GB), total_tps(16, READ_GB)
assert round(T16) == 830 and round(D47) == 822 and round(D13, -3) == 3000                  # "830", "822", "three thousand"
assert round(T16 / CEIL_MOE, 1) == 4.5                                                     # "four and a half times"
assert 16 * flops_per_token(dict(params=ACTIVE)) / H200["peak"] < step_gb(16) * GB / H200["bandwidth"] / 20  # memory-bound
BATCHES = [1, 2, 4, 8, 16, 32]

# DeepSeek-V3 (DeepSeek-AI 2024; config.json: 61 layers, the first 3 dense, 256 routed experts of 3 × 7168 × 2048,
# 1 shared, 8 per token): "671B total parameters with 37B activated for each token".
DS_TOTAL, DS_ACTIVE, DS_E, DS_K = 671, 37, 256, 8
DS_MOE_LAYERS = 61 - 3
DS_EXPERT = DS_MOE_LAYERS * 3 * 7168 * 2048 / 1e9                     # one routed expert, all MoE layers (B)
DS_SHARED = DS_ACTIVE - DS_K * DS_EXPERT                               # what every token reads besides its 8 experts
assert DS_MOE_LAYERS == 58 and abs(DS_E * DS_EXPERT + DS_SHARED - DS_TOTAL) / DS_TOTAL < 0.01  # the configs add up
assert round(lit(16, DS_E, DS_K)) == 102 and round(lit(64, DS_E, DS_K)) == 222 and lit(256, DS_E, DS_K) > 255.8
PER_EXPERT_64 = 64 * DS_K / DS_E
IN_FLIGHT = 256 * DS_E // DS_K                                        # tokens per step for 256 per expert
assert PER_EXPERT_64 == 2 and IN_FLIGHT == 8192                      # "two tokens", "about eight thousand"
FP8_GB = DS_TOTAL * 1                                                 # one byte per weight
NODE_GB = 8 * HARDWARE["H100"]["memory"] / GB
assert FP8_GB == 671 and NODE_GB == 640 and FP8_GB > NODE_GB          # "more than a server of eight H100s holds"

# DeepEP README (commit 458cdcb, 2025-03-05): low-latency kernels, EP8, 128 tokens, top-8, FP8 dispatch, H800
DISPATCH_US, COMBINE_US = 163, 318
assert round(DISPATCH_US, -1) == 160 and round(COMBINE_US, -1) == 320  # "about 160", "about 320"
# DeepSeek, "Day 6: DeepSeek-V3/R1 inference system overview" (Feb 2025): decode = EP144, 2 routed experts + 1 shared
# per GPU, "32 redundant routed experts"
DEC_GPUS, PER_GPU, SPARES = 144, 2, 32
assert DEC_GPUS * PER_GPU == DS_E + SPARES == 288                     # "288 slots for 256 experts"

# the drawn routings: seeded random examples (illustrative)
_rng = np.random.default_rng(20)
ROUTE = [sorted(_rng.choice(E, K, replace=False).tolist()) for _ in range(16)]
UNION = {b: len(set(sum(ROUTE[:b], []))) for b in BATCHES[:5]}
assert ROUTE[:3] == [[2, 6], [3, 7], [1, 4]] and [UNION[b] for b in (1, 2, 4, 8, 16)] == [2, 4, 6, 7, 8]
_rng = np.random.default_rng(146)
DS_ROUTE = [_rng.choice(DS_E, DS_K, replace=False).tolist() for _ in range(256)]
DS_UNION = {b: sorted(set(sum(DS_ROUTE[:b], []))) for b in (1, 16, 64, 256)}
assert [len(DS_UNION[b]) for b in (1, 16, 64, 256)] == [8, 102, 222, 256]

# router scores for the three tokens of section 2 (illustrative): the chosen two score highest
_rng = np.random.default_rng(7)
SCORES = []
for picks in ROUTE[:3]:
    s = _rng.uniform(0.14, 0.4, E)
    for j, p in enumerate(picks):
        s[p] = 0.56 - 0.06 * j
    SCORES.append(s)
assert all(set(np.argsort(-s)[:K]) == set(p) for s, p in zip(SCORES, ROUTE[:3]))

# expert parallelism: 4 GPUs × 2 experts × 2 home tokens, each token to 2 experts (illustrative, balanced)
EP_PICKS = [[2, 5], [4, 7], [0, 6], [1, 5], [0, 3], [2, 7], [1, 4], [3, 6]]   # token t → experts (0-based)
assert all(sum(e in p for p in EP_PICKS) == 2 for e in range(8))           # every expert gets two tokens

USER = recolored("user", REQUEST, PINK_FILL, prefix="ep20")
GPU_ICON = recolored("gpu", MUTED, CARD_HI, prefix="ep20")

# ---------------------------------------------------------------------------- helpers

_FLAT = set("0123456789abcdefhiklmnorstuvwxzABCDEFGHIKLMNOPRSTUVWXYZ")


def baseline(t):
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


def counter(value, label, color, x, y, size=48):
    num = title_text(value, size, color).move_to([x, y, 0])
    lab = text(label, 22, MUTED).next_to(num, m.DOWN, buff=0.16)
    lab.set_x(x)
    return num, lab


def quote_card(lines, width, size=22):
    words = m.VGroup(*[text(s, size, INK) for s in lines]).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.14)
    box = card(width, words.height + 0.5, EDGE, CARD, radius=0.14, stroke=2)
    stripe = m.RoundedRectangle(corner_radius=0.05, width=0.09, height=box.height - 0.2, stroke_width=0,
                                fill_color=MUTED, fill_opacity=1)
    stripe.move_to(box).align_to(box, m.LEFT).shift(0.12 * m.RIGHT)
    words.move_to(box).align_to(box, m.LEFT).shift(0.4 * m.RIGHT)
    return m.VGroup(box, stripe, words)


STYLE = {  # an expert block's look: held in memory · read this step · idle (held, not read)
    "held": (WEIGHTS, 2.0, BLUE_FILL, WEIGHTS),
    "lit": (MEMORY, 4.0, CYAN_FILL, MEMORY),
    "lit_s": (MEMORY, 2.5, CYAN_FILL, MEMORY),
    "idle": (FAINT, 2.0, CARD, FAINT),
}


def expert(label, w, h, size=20, state="held"):
    box = m.RoundedRectangle(corner_radius=0.1, width=w, height=h)
    words = code(label, size, INK)
    block = m.VGroup(box, words)
    styled(block, state)
    words.move_to(box)
    return block


def styled(block, state):
    stroke, width, fill, ink = STYLE[state]
    block[0].set_stroke(stroke, width).set_fill(fill, 1)
    block[1].set_color(ink)
    return block


def restyle(block, state):
    return m.Transform(block, styled(block.copy(), state))


def model_bar(total, spans, scale, x0, y, h=0.46, dividers=()):
    """A model's weights as one bar (BLUE, width ∝ parameters) with the parts one token reads in CYAN."""
    track = m.Rectangle(width=total * scale, height=h).set_stroke(WEIGHTS, 2).set_fill(BLUE_FILL, 1)
    track.move_to([x0 + total * scale / 2, y, 0])
    lits = m.VGroup(*[
        m.Rectangle(width=max((b - a) * scale, 0.014), height=h - 0.1).set_stroke(width=0).set_fill(MEMORY, 1)
        .move_to([x0 + (a + b) / 2 * scale, y, 0]) for a, b in spans])
    divs = m.VGroup(*[m.Line([x0 + d * scale, y - h / 2, 0], [x0 + d * scale, y + h / 2, 0]).set_stroke(WEIGHTS, 1.5)
                      for d in dividers])
    return m.VGroup(track, divs, lits)


# ---------------------------------------------------------------------------- the episode


class Ep20MixtureOfExpertsAtServeTime(Episode):
    number = 20

    def lesson(self):
        self.hook()
        self.batch_one()
        self.batch()
        self.at_scale()
        self.expert_parallel()
        self.in_production()
        self.recap()

    # ------------------------------------------------------------------ 1 · hold everything, read a little

    def hook(self):
        self.next_section(
            "1 · Hold everything, read a little",
            notes=f"Episode 2: tokens/s ≈ bandwidth ÷ bytes per token. Owl-8B (dense) reads all {OWL_B:.2f} B parameters "
                  f"per token. Mixtral 8x7B, computed from its config.json: one expert = 32 layers × 3 × 4096 × 14336 = "
                  f"{EXPERT / 1e9:.2f} B, shared parts (attention, router, norms, embeddings) {SHARED / 1e9:.2f} B; total "
                  f"{TOTAL / 1e9:.2f} B, active (2 experts + shared) {ACTIVE / 1e9:.2f} B — Mistral (2023): '46.7B total "
                  f"parameters but only uses 12.9B parameters per token'. DeepSeek-V3 (DeepSeek-AI 2024): '671B total "
                  f"parameters with 37B activated for each token'; 256 routed experts + 1 shared, 8 per token. Bars: width ∝ "
                  f"parameters (two scales: the zoom-out divides by {DS_TOTAL / (TOTAL / 1e9):.1f}). Which experts are lit "
                  "is illustrative. 'AI, Explained' Episode 39 covers routing and load balancing.")
        formula = code("tokens/s ≈ bandwidth ÷ bytes per token", 28, INK,
                       t2c={r"tokens/s": TOKEN, r"bandwidth": MEMORY, r"bytes per token": MEMORY}).move_to([0, 2.1, 0])
        under = m.Line(glyphs(formula, "bytes per token").get_corner(m.DL) + [0, -0.1, 0],
                       glyphs(formula, "bytes per token").get_corner(m.DR) + [0, -0.1, 0]).set_stroke(MEMORY, 3)

        X0, W = -2.4, 8.9
        s1, s2 = W / (TOTAL / 1e9), W / DS_TOTAL
        ys = (0.95, -0.55, -2.05)
        names = [("Owl-8B", "dense"), ("Mixtral 8x7B", "8 experts · top 2"), ("DeepSeek-V3", "256 experts · top 8")]
        heads = m.VGroup()
        for (n, sub), y in zip(names, ys):
            a = left_at(text(n, 26, INK, weight=m.BOLD), -6.4, y + 0.14)
            b = left_at(code(sub, 20, MUTED), -6.4, y - 0.27)
            heads.add(m.VGroup(a, b))

        sh, ex = SHARED / 1e9, EXPERT / 1e9
        mx_div = [sh + i * ex for i in range(E)]
        mx_lit = [(0, sh)] + [(sh + i * ex, sh + (i + 1) * ex) for i in ROUTE[0]]
        ds_lit = [(0, DS_SHARED)] + [(DS_SHARED + i * DS_EXPERT, DS_SHARED + (i + 1) * DS_EXPERT) for i in DS_ROUTE[0]]
        owl1, owl2 = (model_bar(OWL_B, [(0, OWL_B)], s, X0, ys[0]) for s in (s1, s2))
        mx1, mx2 = (model_bar(TOTAL / 1e9, mx_lit, s, X0, ys[1], dividers=mx_div) for s in (s1, s2))
        ds = model_bar(DS_TOTAL, ds_lit, s2, X0, ys[2])
        owl_read = owl1[2]
        owl1.remove(owl_read)

        def label(held, read, y):
            a = left_at(code(f"holds {held} B ·", 20, MUTED, t2c={rf"holds {held} B": WEIGHTS}), X0, y - 0.5)
            b = code(f"reads {read} B per token", 20, MUTED, t2c={rf"reads {read} B": MEMORY}).next_to(a, m.RIGHT, buff=0.2)
            sit(b, baseline(a))
            return m.VGroup(a, b)

        lab_owl = label("8", "8", ys[0])
        lab_mx = label("46.7", "12.9", ys[1])
        lab_ds = label("671", "37", ys[2])
        gb_owl = code("= 16 GB in BF16", 20, WEIGHTS).next_to(lab_owl, m.RIGHT, buff=0.25)
        sit(gb_owl, baseline(lab_owl[0]))
        mascot = owl(0.42).move_to([X0 + OWL_B * s1 + 0.55, ys[0], 0])
        zoom = code("zoomed out ×14", 20, MUTED)
        right_at(zoom, X0 + W, ys[0] + 0.55)
        deal = title_text("Hold everything, read a little", 40, INK,
                          t2c={"Hold everything": WEIGHTS, "read a little": MEMORY}).move_to([0, -3.15, 0])
        link = m.VGroup(pill("AI, Explained · EP 39", MUTED, CARD, 20, mono=True), text("how the router learns", 24, MUTED),
                        text("today: serving it", 24, INK, weight=m.BOLD)).arrange(m.RIGHT, buff=0.3)
        link[2].shift(0.3 * m.RIGHT)
        link.move_to([0, -3.15, 0])

        self.say(
            "[In episode two, Owl's speed limit was one division]: [memory bandwidth, over the bytes each token reads.]",
            m.AnimationGroup(self.new_header("01 · the deal", "Hold everything, read a little"),
                             m.FadeIn(formula, shift=0.15 * m.UP)),
            m.Create(under),
        )
        self.say(
            "[Owl's eight-billion model is dense]: [every token reads every weight], [sixteen gigabytes.]",
            m.AnimationGroup(m.FadeIn(heads[0]), m.GrowFromEdge(owl1, m.LEFT), m.FadeIn(mascot, shift=0.1 * m.UP)),
            m.AnimationGroup(m.GrowFromEdge(owl_read, m.LEFT, run_time=1.0), m.FadeIn(lab_owl)),
            m.FadeIn(gb_owl, shift=0.1 * m.LEFT),
        )
        self.say(
            "[A mixture of experts breaks that link]: [it holds many expert networks], [but each token reads only a few.]",
            m.FadeIn(heads[1]),
            m.AnimationGroup(m.GrowFromEdge(mx1[0], m.LEFT, run_time=1.2), m.Create(mx1[1], run_time=1.2)),
            m.LaggedStart(*[m.FadeIn(r, scale=0.6) for r in mx1[2]], lag_ratio=0.3),
        )
        self.say(
            "[The Mixtral model holds forty-six point seven billion parameters], [yet each token reads only twelve point nine billion.]",
            m.AnimationGroup(m.FadeIn(lab_mx[0]), m.Circumscribe(mx1[0], color=WEIGHTS, buff=0.06)),
            m.AnimationGroup(m.FadeIn(lab_mx[1]), m.Indicate(mx1[2], color=MEMORY, scale_factor=1.15)),
        )
        self.say(
            "[Zoom out], [and the DeepSeek-V3 model holds six hundred and seventy-one billion], [reading just thirty-seven.]",
            m.AnimationGroup(m.Transform(owl1, owl2[:2]), m.Transform(owl_read, owl2[2]), m.Transform(mx1, mx2),
                             mascot.animate.move_to([X0 + OWL_B * s2 + 0.55, ys[0], 0]), m.FadeOut(gb_owl), m.FadeIn(zoom)),
            m.AnimationGroup(m.FadeIn(heads[2]), m.GrowFromEdge(ds[0], m.LEFT, run_time=1.4), m.FadeIn(lab_ds[0])),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(r, scale=0.6) for r in ds[2]], lag_ratio=0.12), m.FadeIn(lab_ds[1])),
        )
        self.say(
            "[Hold everything], [read a little]: [that's the deal.]",
            m.Circumscribe(ds[0], color=WEIGHTS, buff=0.06),
            m.Indicate(ds[2], color=MEMORY, scale_factor=1.2),
            m.FadeIn(deal, shift=0.15 * m.UP),
        )
        self.say(
            "[Episode thirty-nine of AI, Explained showed how the router learns.] [Today, we serve one.]",
            m.AnimationGroup(m.FadeOut(deal, shift=0.15 * m.UP), m.FadeIn(link[:2], shift=0.15 * m.UP)),
            m.FadeIn(link[2], shift=0.1 * m.LEFT),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 2 · batch one: a router lights two experts

    def batch_one(self):
        self.next_section(
            "2 · Batch one: two of eight",
            notes=f"One of Mixtral's 32 layers, schematic (blocks not to scale): the shared part (attention, router, norms: "
                  f"{SHARED / 1e9:.2f} B over the whole model) and 8 SwiGLU experts ({EXPERT / 1e9:.2f} B each over the whole "
                  f"model). Router scores and the experts chosen are illustrative (seeded). Per token: {ACTIVE / 1e9:.2f} B "
                  f"parameters = {READ_GB:.2f} GB in BF16 read, {HELD_GB:.2f} GB held. Ceiling on one H200 (NVIDIA spec, dated "
                  f"Oct 2026: 141 GB, 4.8 TB/s), weights only, batch 1: {H200['bandwidth'] / 1e12:.1f} TB/s ÷ {READ_GB:.1f} GB = "
                  f"{CEIL_MOE:.0f} tok/s; a dense 46.7 B model: ÷ {HELD_GB:.1f} GB = {CEIL_DENSE:.0f} tok/s. (Mixtral in BF16 "
                  "does not fit one 80 GB H100.) Mistral: it 'processes input and generates output at the same speed and for "
                  "the same cost as a 12.9B model' — true for compute; the memory bill is 46.7 B.")
        XS = [-2.75 + 1.17 * i for i in range(E)]
        EW, EH, EY = 0.97, 1.1, 0.2
        experts = m.VGroup(*[expert(f"E{i + 1}", EW, EH).move_to([x, EY, 0]) for i, x in enumerate(XS)])
        shared = expert("shared", 1.3, EH).move_to([-4.3, EY, 0])
        mem_cap = left_at(code("in memory · one of 32 layers", 20, WEIGHTS), -4.83, 1.0)
        RY = 1.75
        router = card(9.4, 0.8, COMPUTE, CARD, radius=0.14, stroke=2).move_to([1.3, RY, 0])
        r_cap = left_at(code("router · one score per expert", 20, COMPUTE), -3.4, 2.42)

        def bars(t):
            out = m.VGroup()
            for i, (x, s) in enumerate(zip(XS, SCORES[t])):
                chosen = i in ROUTE[t]
                r = m.Rectangle(width=0.46, height=s).set_stroke(COMPUTE, 1.5).set_fill(COMPUTE if chosen else AMBER_FILL, 1)
                out.add(r.move_to([x, RY - 0.31 + s / 2, 0]))
            return out

        bar_sets = [bars(t) for t in range(3)]
        words = ["energy", "for", "the"]
        toks = [token_row([w], size=22)[0].move_to([-5.5, RY, 0]) for w in words]
        feed = arrow([-4.75, RY, 0], [router.get_left()[0], RY, 0], TOKEN)
        math_box = card(4.6, 0.64, COMPUTE, CARD, radius=0.12, stroke=2).move_to([1.15, -1.2, 0])
        math_cap = code("math units", 20, COMPUTE).move_to(math_box)
        top_y = math_box.get_top()[1]

        def reads(t):
            arrows = m.VGroup(arrow([shared.get_x(), EY - EH / 2, 0], [math_box.get_left()[0] + 0.25, top_y, 0], MEMORY, buff=0.06))
            for i in ROUTE[t]:
                tx = float(np.clip(XS[i], math_box.get_left()[0] + 1.2, math_box.get_right()[0] - 0.3))
                arrows.add(arrow([XS[i], EY - EH / 2, 0], [tx, top_y, 0], MEMORY, buff=0.06))
            return arrows

        read_sets = [reads(t) for t in range(3)]

        def states(t):
            return [restyle(b, "lit" if i in ROUTE[t] else "idle") for i, b in enumerate(experts)]

        n_read, l_read = counter(f"{READ_GB:.1f} GB", "read per token", MEMORY, -4.9, -2.3)
        n_held, l_held = counter(f"{HELD_GB:.1f} GB", "held in memory", WEIGHTS, -1.6, -2.3)
        n_tps, l_tps = counter(f"{CEIL_MOE:.0f} tok/s", "ceiling on one H200", TOKEN, 1.9, -2.3)
        for n in (n_held, n_tps):
            n.align_to(n_read, m.DOWN)
        for lab in (l_held, l_tps):
            lab.align_to(l_read, m.UP)
        dense = m.VGroup(code("a dense 46.7 B:", 22, MUTED), code(f"{CEIL_DENSE:.0f} tok/s", 22, RED)).arrange(
            m.DOWN, aligned_edge=m.LEFT, buff=0.12)
        left_at(dense, n_tps.get_right()[0] + 0.4, n_tps.get_y() - 0.2)
        h200 = cite("H200: 141 GB · 4.8 TB/s (NVIDIA, Oct 2026) · weights only, BF16").move_to([1.2, -3.35, 0])

        self.say(
            "[Here is one layer of the Mixtral model, as it sits in memory]: [eight experts], [and a small shared part.]",
            m.AnimationGroup(self.new_header("02 · batch one", "Two experts out of eight"), m.FadeIn(mem_cap)),
            m.LaggedStart(*[m.FadeIn(b, shift=0.1 * m.UP) for b in experts], lag_ratio=0.12),
            m.FadeIn(shared, shift=0.1 * m.UP),
        )
        self.say(
            "[A token arrives]; [the router gives every expert a score], [and the top two light up.]",
            m.AnimationGroup(m.FadeIn(toks[0], shift=0.15 * m.RIGHT), m.GrowArrow(feed)),
            m.AnimationGroup(m.FadeIn(router), m.FadeIn(r_cap),
                             m.LaggedStart(*[m.GrowFromEdge(r, m.DOWN) for r in bar_sets[0]], lag_ratio=0.1)),
            m.AnimationGroup(*[restyle(experts[i], "lit") for i in ROUTE[0]]),
        )
        self.say(
            "[Only those two are read from memory], [plus the shared part]: [the other six sit idle, taking up space.]",
            m.AnimationGroup(m.FadeIn(math_box), m.FadeIn(math_cap),
                             *[m.GrowArrow(a) for a in read_sets[0][1:]], *[pulse(a, MEMORY) for a in read_sets[0][1:]]),
            m.AnimationGroup(m.GrowArrow(read_sets[0][0]), pulse(read_sets[0][0], MEMORY)),
            m.AnimationGroup(*[restyle(b, "idle") for i, b in enumerate(experts) if i not in ROUTE[0]]),
        )

        def next_token(t):
            return m.AnimationGroup(
                m.FadeOut(toks[t - 1], shift=0.15 * m.RIGHT), m.FadeIn(toks[t], shift=0.15 * m.RIGHT),
                m.Transform(bar_sets[0], bar_sets[t]), *states(t),
                m.FadeOut(read_sets[t - 1]), m.FadeIn(read_sets[t]), *[pulse(a, MEMORY) for a in read_sets[t]],
                run_time=1.1)

        self.say(
            "[The next token picks a different pair], [and so does the one after.]",
            next_token(1),
            next_token(2),
        )
        self.say(
            "[So each token reads about twenty-six gigabytes in sixteen-bit], [while all ninety-three stay in memory.]",
            m.AnimationGroup(m.FadeIn(n_read, shift=0.1 * m.UP), m.FadeIn(l_read)),
            m.AnimationGroup(m.FadeIn(n_held, shift=0.1 * m.UP), m.FadeIn(l_held)),
        )
        self.say(
            "[On one H200, that's a ceiling of a hundred and eighty-six tokens a second]: "
            "[a dense model of the same size would manage fifty-one.]",
            m.AnimationGroup(m.FadeIn(n_tps, shift=0.1 * m.UP), m.FadeIn(l_tps), m.FadeIn(h200)),
            m.FadeIn(dense, shift=0.1 * m.LEFT),
        )
        self.say(
            "[At batch one, it decodes like a thirteen-billion model], [but it needs the memory of a forty-seven-billion one.]",
            m.Circumscribe(m.VGroup(n_read, l_read), color=MEMORY, buff=0.12),
            m.Circumscribe(m.VGroup(n_held, l_held), color=WEIGHTS, buff=0.12),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 3 · a batch lights them all

    def batch(self):
        self.next_section(
            "3 · A batch lights them all",
            notes=f"Episode 4: a dense model's bytes per step stay flat with the batch. In an MoE every token picks its own "
                  f"k experts per layer. With uniform, independent routing an expert stays unread with probability "
                  f"(1 − k/E)^B, so the expected number read per layer = E·(1 − (1 − k/E)^B) (notes §9, recomputed): Mixtral "
                  f"(E = 8, k = 2) B = 1, 2, 4, 8, 16 → {', '.join(f'{lit(b):.2f}' for b in BATCHES[:5])}. GB read per step "
                  f"(BF16, weights only) = 2 × (shared + lit × one expert): {', '.join(f'{step_gb(b):.1f}' for b in BATCHES)} "
                  f"for B = {', '.join(map(str, BATCHES))}. The grid is one seeded random routing (illustrative): "
                  f"{UNION[2]}, {UNION[4]}, {UNION[16]} experts lit at 2, 4, 16 students. Total tok/s on one H200 at B = 16 = "
                  f"16 × 4.8 TB/s ÷ {step_gb(16):.1f} GB = {T16:.0f} (×{T16 / CEIL_MOE:.2f} of batch 1); a dense 46.7 B: "
                  f"{D47:.0f}; a dense 12.9 B: {D13:,.0f}. Still memory-bound (math ≈ 0.4 ms vs 19 ms of traffic). Real "
                  "routing is not uniform (hot experts), which lights fewer experts but loads some GPUs more.")
        # left: the routing grid (rows = students, columns = experts)
        CX = [-5.6 + 0.54 * j for j in range(E)]
        ROW0, PITCH = 1.62, 0.232
        heads = m.VGroup(*[expert(f"E{j + 1}", 0.48, 0.38, size=20).move_to([x, 2.1, 0]) for j, x in enumerate(CX)])
        grid = m.VGroup(*[
            m.VGroup(*[m.Rectangle(width=0.48, height=0.16).set_stroke(EDGE, 1.2).set_fill(CARD, 1).move_to([x, ROW0 - r * PITCH, 0])
                       for x in CX]) for r in range(16)])
        users = m.VGroup(*[icon(USER, 0.19).move_to([-6.2, ROW0 - r * PITCH, 0]) for r in range(16)])

        def light_rows(rows):
            return [grid[r][j].animate.set_fill(REQUEST, 1).set_stroke(REQUEST, 1.2) for r in rows for j in ROUTE[r]]

        def light_heads(b, before):
            new = set(sum(ROUTE[:b], [])) - set(sum(ROUTE[:before], []))
            return [restyle(heads[j], "lit_s") for j in sorted(new)]

        def count_text(b):
            t = code(f"batch {b:>2} · {UNION[b]} of 8 experts read", 22, INK,
                     t2c={rf"batch {b:>2}": REQUEST, rf"{UNION[b]} of 8 experts": MEMORY})
            return t.move_to([-3.8, -2.3, 0])

        counts = {b: count_text(b) for b in (1, 2, 4, 16)}
        formula = code("lit = 8 × (1 − 0.75^B)", 22, INK, t2c={r"lit": MEMORY, r"B": REQUEST}).move_to([-3.8, -2.85, 0])
        at16 = code(f"B = 16: 1 − 0.01 → {lit(16):.1f} of 8", 22, MUTED, t2c={rf"{lit(16):.1f} of 8": MEMORY}).move_to([-3.8, -3.3, 0])
        sample = cite("one random example").next_to(heads, m.UP, buff=0.14).align_to(heads, m.LEFT)

        # right: GB read per step vs batch
        OX, OY, W, H = 1.15, -1.25, 5.0, 3.2

        def px(b):
            return OX + np.log2(b) / 5 * W

        def py(gb):
            return OY + gb / 100 * H

        axes_ = m.VGroup(
            m.Arrow([OX, OY, 0], [OX + W + 0.35, OY, 0], buff=0, color=FAINT, stroke_width=3, max_tip_length_to_length_ratio=0.04),
            m.Arrow([OX, OY, 0], [OX, OY + H + 0.3, 0], buff=0, color=FAINT, stroke_width=3, max_tip_length_to_length_ratio=0.06))
        ticks = m.VGroup()
        for v in (25, 50, 75, 100):
            ticks.add(m.Line([OX - 0.08, py(v), 0], [OX, py(v), 0]).set_stroke(FAINT, 2))
            ticks.add(right_at(code(str(v), 20, MUTED), OX - 0.15, py(v)))
        xl = m.VGroup(*[code(str(b), 20, MUTED).move_to([px(b), OY - 0.28, 0]) for b in BATCHES])
        on_baseline(xl)
        for b in BATCHES:
            ticks.add(m.Line([px(b), OY, 0], [px(b), OY - 0.08, 0]).set_stroke(FAINT, 2))
        x_lab = text("batch (students per step)", 22, MUTED).move_to([OX + W / 2, OY - 0.72, 0])
        y_lab = left_at(text("GB read per step", 22, MEMORY), OX - 0.45, py(100) + 0.5)
        chart = m.VGroup(axes_, ticks, xl, x_lab, y_lab)
        top = m.DashedLine([px(1), py(HELD_GB), 0], [px(32), py(HELD_GB), 0], dash_length=0.12).set_stroke(WEIGHTS, 2.5)
        top_lab = right_at(code(f"all 8 experts · {HELD_GB:.1f} GB", 20, WEIGHTS), px(32), py(HELD_GB) + 0.26)
        bot = m.DashedLine([px(1), py(READ_GB), 0], [px(32), py(READ_GB), 0], dash_length=0.12).set_stroke(MUTED, 2)
        bot_lab = right_at(code(f"2 experts · {READ_GB:.1f} GB", 20, MUTED), px(32), py(READ_GB) - 0.26)
        flat = left_at(text("dense: flat (EP 4)", 20, MUTED), px(1) + 0.12, py(HELD_GB) - 0.3)

        def seg(a, b):
            pts = [[px(x), py(step_gb(x)), 0] for x in np.logspace(np.log2(a), np.log2(b), 24, base=2)]
            return m.VMobject().set_points_as_corners(pts).set_stroke(MEMORY, 5)

        segs = [seg(a, b) for a, b in ((1, 2), (2, 4), (4, 16), (16, 32))]
        dot = m.Dot([px(1), py(step_gb(1)), 0], radius=0.1, color=MEMORY).set_z_index(3)

        def move(b, k, run_time=1.0):
            return m.AnimationGroup(m.Create(segs[k]), dot.animate.move_to([px(b), py(step_gb(b)), 0]), run_time=run_time)

        lab16 = right_at(code(f"{step_gb(16):.1f} GB", 22, MEMORY), px(16) + 0.35, py(step_gb(16)) - 0.85)
        r1 = code(f"16 students: {T16:.0f} tok/s in total", 22, INK, t2c={rf"{T16:.0f} tok/s": TOKEN, r"16 students": REQUEST})
        r2 = code(f"×{T16 / CEIL_MOE:.1f} of batch 1, not ×16", 22, MUTED, t2c={rf"×{T16 / CEIL_MOE:.1f}": RED})
        r3 = code(f"a dense 46.7 B: {D47:.0f} tok/s", 22, MUTED, t2c={rf"{D47:.0f} tok/s": WEIGHTS})
        reads = m.VGroup(r1, r2, r3).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.14)
        left_at(reads, OX - 0.45, -2.85)
        moral = pill("the savings move from bytes to math", COMPUTE, AMBER_FILL, 22).move_to([-3.8, -3.05, 0])

        self.say(
            "[Now add students.] [A dense model reads the same bytes at any batch]: [that flat line was episode four's free lunch.]",
            m.AnimationGroup(self.new_header("03 · a batch", "Every student lights two"), m.FadeIn(chart)),
            m.AnimationGroup(m.Create(top), m.FadeIn(top_lab)),
            m.FadeIn(flat, shift=0.1 * m.DOWN),
        )
        self.say(
            "[But here, each student's token picks its own two experts], [in every layer.]",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(h) for h in heads], lag_ratio=0.08), m.FadeIn(grid), m.FadeIn(users),
                             m.FadeIn(sample)),
            m.AnimationGroup(*light_rows([0]), *light_heads(1, 0), m.FadeIn(counts[1]), m.Create(bot), m.FadeIn(bot_lab),
                             m.FadeIn(dot, scale=0.5)),
        )
        self.say(
            "[Two students light four experts]; [four students light six]; [sixteen light all eight.]",
            m.AnimationGroup(*light_rows([1]), *light_heads(2, 1), m.Transform(counts[1], counts[2]), move(2, 0)),
            m.AnimationGroup(*light_rows([2, 3]), *light_heads(4, 2), m.Transform(counts[1], counts[4]), move(4, 1)),
            m.AnimationGroup(m.LaggedStart(*light_rows(range(4, 16)), lag_ratio=0.04), *light_heads(16, 4),
                             m.Transform(counts[1], counts[16]), move(16, 2, run_time=1.6)),
        )
        self.say(
            "[So the bytes per step climb, from about twenty-six gigabytes] [to over ninety-two]: [nearly the whole model, every step.]",
            m.Indicate(dot, color=MEMORY, scale_factor=1.5),
            m.FadeIn(lab16, shift=0.1 * m.UP),
            m.AnimationGroup(m.Create(segs[3]), m.Circumscribe(top_lab, color=WEIGHTS, buff=0.08)),
        )
        self.say(
            "[With even, random routing], [an expert stays dark only if every student skips it]: [three quarters, to the power of the batch.]",
            m.FadeIn(formula, shift=0.1 * m.UP),
            m.Circumscribe(m.VGroup(*[grid[r][7] for r in range(16)]), color=MUTED, buff=0.06),
            m.Indicate(glyphs(formula, "0.75^B"), color=REQUEST, scale_factor=1.15),
        )
        self.say(
            "[At sixteen students, that's about one chance in a hundred]: [so on average, seven point nine of the eight are lit.]",
            m.FadeIn(at16, shift=0.1 * m.UP),
            m.Circumscribe(glyphs(at16, f"{lit(16):.1f} of 8"), color=MEMORY, buff=0.08),
        )
        self.say(
            "[On an H200, sixteen students get eight hundred and thirty tokens a second in total]: "
            "[four and a half times batch one, not sixteen times.]",
            m.AnimationGroup(m.FadeOut(lab16), m.FadeIn(r1, shift=0.1 * m.UP)),
            m.FadeIn(r2, shift=0.1 * m.UP),
        )
        self.say(
            "[That's about what a dense model of the full size gets]: [past a small batch, the savings move from bytes to math.]",
            m.FadeIn(r3, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeOut(m.VGroup(formula, at16)), m.FadeIn(moral, scale=0.9)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 4 · at scale: DeepSeek-V3

    def at_scale(self):
        self.next_section(
            "4 · Two hundred and fifty-six experts",
            notes=f"DeepSeek-V3 (DeepSeek-AI, technical report, Dec 2024): 671 B total, 37 B activated per token; each MoE "
                  f"layer has 1 shared + 256 routed experts, 8 activated per token. Expected experts lit per layer, uniform "
                  f"routing: B = 16 → {lit(16, DS_E, DS_K):.1f}, 64 → {lit(64, DS_E, DS_K):.1f}, 256 → {lit(256, DS_E, DS_K):.1f} "
                  f"(notes §9). The grid is one seeded random routing (illustrative) whose counts match the expectation. "
                  f"Tokens per expert ≈ B × 8 ÷ 256: {PER_EXPERT_64:.0f} at B = 64; {IN_FLIGHT:,} tokens per step give each "
                  "expert 256 (notes §9). Quote: 'the batch size per expert is relatively small (usually within 256 tokens), "
                  f"and the bottleneck is memory access rather than computation.' FP8 weights ≈ {FP8_GB} GB > 8 × 80 GB = "
                  f"{NODE_GB:.0f} GB (one 8×H100 server).")
        G, C, GAP = 16, 0.205, 0.045
        cells = m.VGroup(*[m.Square(C).set_stroke(EDGE, 1.2).set_fill(CARD, 1) for _ in range(G * G)])
        cells.arrange_in_grid(rows=G, cols=G, buff=GAP).move_to([-3.85, -0.12, 0])
        for c in cells:
            c.set_stroke(WEIGHTS, 1.2).set_fill(BLUE_FILL, 1)
        cap = left_at(code("one layer · 256 experts", 20, WEIGHTS), cells.get_left()[0], 2.27)

        def light(b, before):
            new = sorted(set(DS_UNION[b]) - set(DS_UNION.get(before, [])))
            return m.LaggedStart(*[cells[i].animate.set_fill(MEMORY, 1).set_stroke(MEMORY, 1.2) for i in new],
                                 lag_ratio=min(0.08, 2.0 / max(len(new), 1)), run_time=1.2)

        def count_text(b):
            n = len(DS_UNION[b])
            t = code(f"batch {b:>3} · {n:>3} of 256 read", 20, INK, t2c={rf"batch {b:>3}": REQUEST, rf"{n:>3} of 256": MEMORY})
            return left_at(t, cells.get_left()[0], -2.5)

        counts = {b: count_text(b) for b in (1, 16, 64, 256)}
        sample = cite("one random example").next_to(counts[1], m.DOWN, buff=0.16).align_to(counts[1], m.LEFT)

        RX = -0.95
        facts = m.VGroup(
            code(f"holds {DS_TOTAL} B · reads {DS_ACTIVE} B per token", 22, MUTED,
                 t2c={rf"holds {DS_TOTAL} B": WEIGHTS, rf"reads {DS_ACTIVE} B": MEMORY}),
            code("256 routed + 1 shared · 8 per token", 20, MUTED),
            cite("DeepSeek-AI, 2024 · V3 technical report")).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.14)
        left_at(facts, RX, 1.75)
        per = m.VGroup(text("tokens per expert", 22, MUTED),
                       code(f"64 × 8 ÷ 256 = {PER_EXPERT_64:.0f}", 30, INK, t2c={r"64": REQUEST, rf"= {PER_EXPERT_64:.0f}": COMPUTE})
                       ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.14)
        left_at(per, RX, 0.35)
        need = m.VGroup(text("for 256 tokens per expert", 22, MUTED),
                        code(f"256 × 256 ÷ 8 = {IN_FLIGHT:,} tokens", 28, INK, t2c={rf"{IN_FLIGHT:,} tokens": REQUEST})
                        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.14)
        left_at(need, RX, 0.35)
        quote = quote_card(["“the batch size per expert is relatively small",
                            "(usually within 256 tokens), and the bottleneck",
                            "is memory access rather than computation.”"], 6.6, size=22)
        left_at(quote, RX, -1.25)
        q_src = cite("DeepSeek-V3 report, on decoding").next_to(quote, m.DOWN, buff=0.12).align_to(quote, m.LEFT)

        sc = 6.0 / FP8_GB
        bar_w = m.Rectangle(width=FP8_GB * sc, height=0.42).set_stroke(WEIGHTS, 2).set_fill(BLUE_FILL, 1)
        left_at(bar_w, RX, -1.2)
        bar_n = m.Rectangle(width=NODE_GB * sc, height=0.42).set_stroke(MUTED, 2).set_fill(CARD_HI, 1)
        left_at(bar_n, RX, -2.1)
        over = m.Rectangle(width=(FP8_GB - NODE_GB) * sc, height=0.42).set_stroke(RED, 2).set_fill(RED_FILL, 1)
        left_at(over, RX + NODE_GB * sc, -1.2)
        lab_w = left_at(code(f"weights in FP8 · {FP8_GB} GB", 20, WEIGHTS), RX, -0.75)
        lab_n = left_at(code(f"one server · 8 × H100 · {NODE_GB:.0f} GB", 20, MUTED), RX, -1.65)
        mem = m.VGroup(bar_w, bar_n, lab_w, lab_n)

        self.say(
            "[Now scale it up.] [The DeepSeek-V3 model has two hundred and fifty-six experts in each layer], [and every token picks eight.]",
            m.AnimationGroup(self.new_header("04 · at scale", "Two hundred and fifty-six experts"), m.FadeIn(facts[0], shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(cells, lag_ratio=0.002, run_time=1.4), m.FadeIn(cap), m.FadeIn(facts[1:])),
            m.AnimationGroup(light(1, None), m.FadeIn(counts[1]), m.FadeIn(sample)),
        )
        self.say(
            "[At sixteen students, about a hundred light up.] [At sixty-four, about two hundred and twenty.] "
            "[At two hundred and fifty-six, practically all.]",
            m.AnimationGroup(light(16, 1), m.Transform(counts[1], counts[16])),
            m.AnimationGroup(light(64, 16), m.Transform(counts[1], counts[64])),
            m.AnimationGroup(light(256, 64), m.Transform(counts[1], counts[256])),
        )
        self.say(
            "[Yet each expert sees only a sliver of the batch]: [at sixty-four students, just two tokens.]",
            m.FadeIn(per[0], shift=0.1 * m.UP),
            m.FadeIn(per[1], shift=0.1 * m.UP),
        )
        self.say(
            "[Its authors say it plainly]: [the batch per expert is small], [and memory access is the bottleneck, not computation.]",
            m.AnimationGroup(m.FadeIn(quote[:2]), m.FadeIn(q_src)),
            m.FadeIn(quote[2][:2]),
            m.FadeIn(quote[2][2:]),
        )
        self.say(
            "[To give every expert two hundred and fifty-six tokens a step], [you need about eight thousand tokens in flight.]",
            m.FadeOut(per, shift=0.15 * m.UP),
            m.FadeIn(need, shift=0.15 * m.UP),
        )
        self.say(
            "[And at one byte per weight, the model alone is six hundred and seventy-one gigabytes]: "
            "[more than a server of eight H100s can hold.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(quote, q_src), shift=0.15 * m.DOWN),
                             m.FadeIn(lab_w), m.GrowFromEdge(bar_w, m.LEFT, run_time=1.2)),
            m.AnimationGroup(m.FadeIn(lab_n), m.GrowFromEdge(bar_n, m.LEFT), m.FadeIn(over), m.Indicate(over, color=RED)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 5 · expert parallelism: dispatch and combine

    def expert_parallel(self):
        self.next_section(
            "5 · Expert parallelism",
            notes="Schematic (illustrative): 4 GPUs, 8 experts, 2 per GPU; each GPU keeps its own requests' tokens and runs "
                  "their attention (data parallel), the router picks 2 experts per token, an all-to-all DISPATCH sends each "
                  "token to the GPUs holding its experts, each expert runs on the tokens it received (a bigger batch per "
                  "expert), and an all-to-all COMBINE returns the outputs, mixed with the router's gate weights (notes §9, "
                  "§10). Two all-to-alls per MoE layer per step. The routing drawn is balanced on purpose (2 tokens per "
                  "expert); real routing is not, hence load balancing.")
        XS = [-4.8, -1.6, 1.6, 4.8]
        cards = m.VGroup(*[card(2.9, 3.75, EDGE, CARD, radius=0.18, stroke=2).move_to([x, 0.05, 0]) for x in XS])
        titles = m.VGroup()
        for g, x in enumerate(XS):
            row = m.VGroup(icon(GPU_ICON, 0.36), code(f"GPU {g + 1}", 20, MUTED)).arrange(m.RIGHT, buff=0.15)
            titles.add(row.move_to([x, 1.55, 0]))
        EXY, SLOTY, HOMEY = 0.72, -0.05, -1.25
        exps = m.VGroup(*[expert(f"E{e + 1}", 1.15, 0.55).move_to([XS[e // 2] + (-0.65 if e % 2 == 0 else 0.65), EXY, 0])
                          for e in range(8)])
        homes = m.VGroup(*[token_row([f"t{t + 1}"], size=20)[0].move_to([XS[t // 2] + (-0.62 if t % 2 == 0 else 0.62), HOMEY, 0])
                           for t in range(8)])
        picks = m.VGroup(*[code(f"E{a + 1} E{b + 1}", 20, COMPUTE).move_to(homes[t].get_center() + [0, 0.5, 0])
                           for t, (a, b) in enumerate(EP_PICKS)])
        att = m.VGroup(*[m.RoundedRectangle(corner_radius=0.1, width=2.3, height=0.78).set_stroke(COMPUTE, 2.5)
                         .move_to([x, HOMEY, 0]) for x in XS])
        slot_of = {}
        for e in range(8):
            ts = [t for t, p in enumerate(EP_PICKS) if e in p]
            for j, t in enumerate(ts):
                slot_of[(t, e)] = exps[e].get_center() + np.array([-0.29 if j == 0 else 0.29, SLOTY - EXY, 0])
        copies = {(t, e): homes[t].copy().scale(0.92).move_to(p) for (t, e), p in slot_of.items()}
        order = sorted(copies)
        bus = m.Line([-6.2, -2.25, 0], [6.2, -2.25, 0]).set_stroke(MEMORY, 3)
        bus_lab = code("network: NVLink inside a server, InfiniBand between", 20, MEMORY).move_to([0, -2.6, 0])
        links = m.VGroup(*[m.Line([x, cards[0].get_bottom()[1], 0], [x, -2.25, 0]).set_stroke(MEMORY, 3) for x in XS])
        steps = ["1 attention", "2 dispatch", "3 experts", "4 combine"]
        colors = [COMPUTE, MEMORY, COMPUTE, MEMORY]
        dim = m.VGroup(*[pill(s, MUTED, CARD, 20, mono=True) for s in steps]).arrange(m.RIGHT, buff=0.35).move_to([0, -3.2, 0])
        lit_ = [pill(s, c, FILLS[c], 20, mono=True).move_to(d) for s, c, d in zip(steps, colors, dim)]

        def phase(i):
            anims = [m.Transform(dim[i], lit_[i])]
            if i:
                anims.append(m.Transform(dim[i - 1], pill(steps[i - 1], MUTED, CARD, 20, mono=True).move_to(dim[i - 1])))
            return m.AnimationGroup(*anims)

        self.say(
            "[So big deployments spread the experts over many GPUs]: [that's expert parallelism.]",
            m.AnimationGroup(self.new_header("05 · expert parallelism", "Send the tokens to the experts"),
                             m.LaggedStart(*[m.FadeIn(m.VGroup(c, t), shift=0.1 * m.UP) for c, t in zip(cards, titles)], lag_ratio=0.15)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(x, shift=0.1 * m.DOWN) for x in exps], lag_ratio=0.08),
                             m.Create(bus), m.Create(links), m.FadeIn(bus_lab)),
        )
        self.say(
            "[Each GPU keeps its own students' tokens], [runs their attention], [and the router picks two experts for each.]",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(h, shift=0.1 * m.UP) for h in homes], lag_ratio=0.08), m.FadeIn(dim)),
            m.AnimationGroup(phase(0), m.LaggedStart(*[m.Create(a) for a in att], lag_ratio=0.1)),
            m.AnimationGroup(m.FadeOut(att), m.LaggedStart(*[m.FadeIn(p, shift=0.08 * m.UP) for p in picks], lag_ratio=0.08)),
        )
        self.say(
            "[Then comes dispatch]: [an all-to-all, where every GPU sends each token to the GPUs that hold its experts.]",
            m.AnimationGroup(phase(1), pulse(bus, MEMORY, 12)),
            m.LaggedStart(*[m.FadeIn(copies[k], target_position=homes[k[0]], path_arc=-0.9 if slot_of[k][0] > homes[k[0]].get_x() else 0.9)
                            for k in order], lag_ratio=0.06, run_time=2.4),
        )
        self.say(
            "[Now each expert works on a batch gathered from everyone.]",
            m.AnimationGroup(phase(2), *[m.Indicate(x, color=COMPUTE, scale_factor=1.08) for x in exps],
                             *[m.Indicate(copies[k], color=COMPUTE, scale_factor=1.1) for k in order]),
        )
        self.say(
            "[Then combine]: [a second all-to-all brings every result home], [where the router's weights mix them into one.]",
            m.AnimationGroup(phase(3), pulse(bus, MEMORY, 12)),
            m.LaggedStart(*[m.FadeOut(copies[k], target_position=homes[k[0]], path_arc=0.9 if slot_of[k][0] > homes[k[0]].get_x() else -0.9)
                            for k in order], lag_ratio=0.06, run_time=2.2),
            m.AnimationGroup(m.FadeOut(picks), *[m.Indicate(h, color=TOKEN, scale_factor=1.15) for h in homes]),
        )
        self.say(
            "[Two all-to-alls in every expert layer, at every step]: [now the network is part of the speed limit too.]",
            m.AnimationGroup(pulse(bus, MEMORY, 12), m.Circumscribe(m.VGroup(dim[1], dim[3]), color=MEMORY, buff=0.1)),
            m.AnimationGroup(m.Indicate(bus_lab, color=MEMORY, scale_factor=1.05), *[pulse(l, MEMORY, 10) for l in links]),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 6 · in production

    def in_production(self):
        self.next_section(
            "6 · What the wires cost",
            notes=f"DeepEP (DeepSeek, README at commit 458cdcb, 2025-03-05, re-read 2026-10-24), low-latency kernels, H800 with "
                  f"pure RDMA (400 Gb/s InfiniBand), 128 tokens per batch, 7168 hidden, top-8, FP8 dispatch / BF16 combine: EP8 dispatch {DISPATCH_US} µs, "
                  f"combine {COMBINE_US} µs (EP256: 194 / 360 µs). DeepSeek-V3 has {DS_MOE_LAYERS} MoE layers (config.json: 61 "
                  "layers, first 3 dense), each with one dispatch and one combine per step. DeepSeek, 'Day 6: DeepSeek-V3/R1 "
                  "inference system overview' (Feb 2025): batches split into two microbatches so 'the communication cost of one "
                  "microbatch is hide[den] behind the computation of the other' (prefill; decode uses a 5-stage pipeline to the "
                  f"same end). Decode: routed experts EP{DEC_GPUS} over 18 nodes, 'each GPU manages 2 routed experts and 1 shared "
                  f"expert', with {SPARES} redundant routed experts ({DEC_GPUS} × 2 = {DEC_GPUS * PER_GPU} = 256 + {SPARES}). "
                  "Which slots hold the spares is drawn for illustration. Episode 21: tensor, pipeline and context parallelism.")
        # left: DeepEP latencies
        LX = -6.3
        title = left_at(text("all-to-all on eight GPUs", 22, INK, weight=m.BOLD), LX, 2.2)
        sc = 3.3 / COMBINE_US
        y1, y2 = 1.45, 0.75
        l1 = left_at(code("dispatch", 22, MEMORY), LX, y1)
        l2 = left_at(code("combine", 22, MEMORY), LX, y2)
        BX = LX + 1.75
        b1 = m.Rectangle(width=DISPATCH_US * sc, height=0.4).set_stroke(MEMORY, 2).set_fill(CYAN_FILL, 1)
        b2 = m.Rectangle(width=COMBINE_US * sc, height=0.4).set_stroke(MEMORY, 2).set_fill(CYAN_FILL, 1)
        left_at(b1, BX, y1)
        left_at(b2, BX, y2)
        v1 = code(f"{DISPATCH_US} µs", 22, INK).next_to(b1, m.RIGHT, buff=0.2)
        v2 = code(f"{COMBINE_US} µs", 22, INK).next_to(b2, m.RIGHT, buff=0.2)
        src = cite("DeepEP, 2025 · H800 over InfiniBand · 128 tokens · top-8").move_to([0, 0, 0])
        left_at(src, LX, 0.22)
        layers = left_at(code(f"× {DS_MOE_LAYERS} expert layers, every step", 22, COMPUTE), LX, -0.35)

        # overlap: two micro-batches
        OY1, OY2 = -1.55, -2.25
        ol1 = left_at(code("batch A", 20, MUTED), LX, OY1)
        ol2 = left_at(code("batch B", 20, MUTED), LX, OY2)
        TX, U = LX + 1.3, 0.62

        def lane(y, start_compute):
            parts = m.VGroup()
            for i in range(6):
                is_c = (i % 2 == 0) == start_compute
                c = COMPUTE if is_c else MEMORY
                r = m.Rectangle(width=U - 0.06, height=0.36).set_stroke(c, 2).set_fill(FILLS[c], 1)
                parts.add(r.move_to([TX + U * (i + 0.5), y, 0]))
            return parts

        la, lb = lane(OY1, True), lane(OY2, False)
        legend = m.VGroup(code("compute", 20, COMPUTE), code("all-to-all", 20, MEMORY)).arrange(m.RIGHT, buff=0.4)
        left_at(legend, TX, -2.85)
        ov_src = cite("DeepSeek inference overview, 2025").move_to([0, 0, 0])
        left_at(ov_src, LX, -3.3)

        # right: the decode unit, 144 GPUs × 2 routed experts
        RX0 = 0.95
        unit_t = left_at(text("decode unit · 144 GPUs", 22, INK, weight=m.BOLD), RX0, 2.2)
        unit_s = left_at(code("2 routed + 1 shared per GPU", 20, MUTED), RX0, 1.78)
        gpus = m.VGroup()
        for g in range(DEC_GPUS):
            frame = m.Rectangle(width=0.24, height=0.31).set_stroke(EDGE, 1.2).set_fill(CARD, 1)
            halves = m.VGroup(*[m.Rectangle(width=0.16, height=0.1).set_stroke(width=0).set_fill(WEIGHTS, 1) for _ in range(2)])
            halves.arrange(m.DOWN, buff=0.04).move_to(frame)
            gpus.add(m.VGroup(frame, halves))
        gpus.arrange_in_grid(rows=8, cols=18, buff=(0.04, 0.06))
        gpus.move_to([RX0 + gpus.width / 2, -0.15, 0])
        spare_rng = np.random.default_rng(32)
        spare_idx = sorted(spare_rng.choice(DEC_GPUS * PER_GPU, SPARES, replace=False).tolist())
        spares = [gpus[i // 2][1][i % 2] for i in spare_idx]
        ep12 = m.VGroup(pill("EP 12", REQUEST, PINK_FILL, 20, mono=True), cite("the decode unit from episode 12"))
        ep12.arrange(m.RIGHT, buff=0.2)
        left_at(ep12, RX0, -2.05)
        slots = code(f"{DEC_GPUS * PER_GPU} slots = 256 + {SPARES} spares", 22, INK,
                     t2c={rf"{SPARES} spares": COMPUTE, r"256": WEIGHTS})
        left_at(slots, RX0, -2.6)
        hot = left_at(text("spares copy the busiest experts", 20, MUTED), RX0, -3.05)
        ep21 = m.VGroup(pill("EP 21", WEIGHTS, BLUE_FILL, 20, mono=True), text("tensor, pipeline, context parallelism", 20, INK))
        ep21.arrange(m.RIGHT, buff=0.2)
        left_at(ep21, RX0, -3.05)

        self.say(
            "[With DeepSeek's open kernels on eight GPUs], [a dispatch takes about a hundred and sixty microseconds], "
            "[and a combine about three hundred and twenty.]",
            m.AnimationGroup(self.new_header("06 · in production", "What the wires cost"), m.FadeIn(title), m.FadeIn(src)),
            m.AnimationGroup(m.FadeIn(l1), m.GrowFromEdge(b1, m.LEFT), m.FadeIn(v1)),
            m.AnimationGroup(m.FadeIn(l2), m.GrowFromEdge(b2, m.LEFT), m.FadeIn(v2)),
        )
        self.say(
            "[That's in each of its fifty-eight expert layers], [so engines hide it]: [one micro-batch talks while the other computes.]",
            m.FadeIn(layers, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(ol1), m.FadeIn(ol2), m.FadeIn(legend), m.FadeIn(ov_src)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.RIGHT) for pair in zip(la, lb) for r in pair],
                                           lag_ratio=0.12, run_time=2.0)),
        )
        self.say(
            "[In production, the decode unit from episode twelve spreads them over a hundred and forty-four GPUs]: "
            "[two routed experts on each], [plus a shared one.]",
            m.AnimationGroup(m.FadeIn(unit_t), m.FadeIn(ep12), m.FadeIn(m.VGroup(*[g[0] for g in gpus]), lag_ratio=0.004, run_time=1.4)),
            m.FadeIn(m.VGroup(*[g[1] for g in gpus]), lag_ratio=0.003, run_time=1.2),
            m.FadeIn(unit_s),
        )
        self.say(
            "[That's two hundred and eighty-eight slots for two hundred and fifty-six experts]: [the thirty-two spares copy the busiest ones.]",
            m.FadeIn(slots, shift=0.1 * m.UP),
            m.AnimationGroup(m.LaggedStart(*[s.animate.set_fill(COMPUTE, 1) for s in spares], lag_ratio=0.04), m.FadeIn(hot)),
        )
        self.say(
            "[Episode twenty-one covers the other ways to split a model across GPUs], [and what the wires cost.]",
            m.AnimationGroup(m.FadeOut(hot), m.FadeIn(ep21, shift=0.1 * m.UP)),
            m.Circumscribe(m.VGroup(b1, b2, v1, v2), color=MEMORY, buff=0.12),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Hold every expert in memory (93.4 GB for Mixtral in BF16, 671 GB for DeepSeek-V3 in "
                                         "FP8); read only the experts the batch lights up (2 of 8 at batch 1, nearly all by "
                                         "16); route tokens to experts on other GPUs and back (all-to-all dispatch and combine).")
        halo = glow(WEIGHTS, radius=4.0)
        words = m.VGroup(*[title_text(w, 80, c) for w, c in (("Hold", WEIGHTS), ("Read", MEMORY), ("Route", TOKEN))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("every expert", "what the batch lights", "tokens there and back")])
        slots = [max(w.width, s.width) for w, s in zip(words, subs)]
        x, gap = 0.0, 1.0
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
        tagline = text("Store a crowd of experts, read the few each token needs.", 30, MUTED).next_to(row_, m.DOWN, buff=0.75)
        mascot = owl(0.5).next_to(tagline, m.DOWN, buff=0.45)

        self.say(
            "[To sum up]: [hold], [read], [route.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[Hold every expert in memory. Read only the experts a batch lights up. And route each token out to its experts, and home again.]",
            m.AnimationGroup(m.Write(tagline), m.FadeIn(mascot, shift=0.1 * m.UP)),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row_, tagline, mascot)], run_time=0.9)
