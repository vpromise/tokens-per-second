"""Tokens per Second · Episode 17 · The Last Step of a Token (Module 4 · Faster Tokens)

Every pass of Owl-8B ends in 128,256 logits, one score per vocabulary token; episodes 1–16 made that pass fast,
this one is the last step: turning the scores into ONE token. Three parts, one running example (Owl writing a JSON
flashcard about the seasons):
1. Sampling (owl_reference.sample): softmax, temperature (0.5 sharpens, 1.5 flattens, 0 = greedy argmax), top-k,
   top-p (the nucleus), renormalise, one weighted roll; the same steps as an engine's chain of logits processors
   (vLLM's order: grammar mask → penalties → temperature → top-k/top-p → sample).
2. Grammar masks (constrained decoding, notes §12): the schema allows only some tokens next; every other logit is
   set to −∞, the rest renormalised, then Owl picks — the format is guaranteed, the distribution is bent. The mask
   is one bit per token (128,256 bits ≈ 16 KB per request per step); Outlines (FSM index), XGrammar (pushdown,
   prechecked tokens, overlapped with the GPU), llguidance (~50 µs per token ≈ 1% of an H100 step of Owl-8B).
3. Determinism: temperature 0 on this Mac at batch 1 gave 1 answer in 20 runs (twice); a lab's 1,000 runs on a
   busy server gave 80 different answers. Floating-point addition is not associative, kernels split a sum
   differently for different batch sizes (split-K, episode 16's split-KV), continuous batching (episode 10) changes
   the batch every step, and a near-tie flips. Measured here: Owl's 256-token seasons answer changed at token 221
   inside batches of 4 and 8 (132 at 16) — the token where episode 13's 4-bit speculative run drifted too. The fix:
   batch-invariant kernels (1,000 of 1,000 identical; 26 s → 55 s, 42 s with a better attention kernel).

Sections: the last step · sampling · in code · grammar masks · the mask's cost · temperature zero · the order of a
sum · measured here · the fix · recap (shape, mask, pick). Episodes 1, 10, 13, 16 linked by number; "AI, Explained"
Episode 33 (decoding and sampling) linked for the details of temperature, top-k and top-p.

Sources (facts dated October 2026):
- owl_reference.py: OWL (vocab 128,256, d_model 4,096), weight_bytes, step_time (H100 batch-1 floor 4.79 ms), sample
  (the code excerpt is checked against it below), measured().
- research/notes.md §12: constrained decoding (mask = −∞, 128,256-bit mask = 16 KB per sequence per step; Outlines,
  Willard & Louf, arXiv 2307.09702: naive O(N) per token, FSM + vocabulary index → O(1) on average; XGrammar, Dong
  et al., arXiv 2411.15100 / MLSys 2025: pushdown matcher, context-independent tokens prechecked, overlap with GPU
  compute, "up to 100x speedup", "near-zero overhead"; llguidance "on the order of 50μs of CPU time per token (for
  128k tokenizer)"; compilation the first time a schema appears); the worked JSON-mask example (0.50 / 0.30 / 0.20 →
  0.714 / 0.286), relabelled here for Owl's flashcard; determinism (Thinking Machines Lab, Horace He et al.,
  2025-09-10): Qwen3-235B, 1,000 runs at T = 0, "80 unique completions, with the most common of these occuring 78
  times"; batch-invariant kernels: all 1,000 identical; 26 s → 55 s unoptimised, 42 s with a better attention
  kernel (Qwen3-8B, vLLM).
- Thinking Machines Lab blog "Defeating Nondeterminism in LLM Inference" (read 2026-10-24): identical for the first
  102 tokens, first divergence at token 103: "Feynman was born on May 11, 1918, in" → 992 "Queens, New York", 8 "New
  York City"; (0.1 + 1e20) − 1e20 = 0 vs 0.1 + (1e20 − 1e20) = 0.1; split-K matmul and split-KV attention change the
  reduction order with the batch size; "the reduction order for each element must be fixed regardless of the
  batch-size of the kernel".
- vLLM V1 sampler (vllm/v1/sample/sampler.py docstring, read 2026-10-24): allowed tokens / bad words / logit bias →
  penalties → temperature → min-p → top-k/top-p → sample; the structured-output bitmask is applied to the logits
  before the sampler.
- research/measurements.md §12 / .json "determinism" and raw/determinism__llama-3.1-8b__4bit__chat256.json (this Mac
  Studio, M4 Max, mlx-lm 0.32.2, Owl-8B 4-bit, greedy): batch 1, 20 runs → 1 distinct output for the 128-token
  (octopus facts) and the 256-token (Owl's seasons answer) prompts; the same prompt as member 0 of a BatchGenerator
  batch of 2/4/8/16: 128 tokens identical at every size, 256 tokens identical at 2, differing from token 221 at 4 and
  8, from token 132 at 16; mlx_lm.server with 15 other clients: 20 of 20 identical to the solo answer. §9: the 4-bit
  speculative run on the same chat prompt left the greedy text at token 221 too.

Illustrative (said "for example" or marked on screen): the eight next-token scores after "Earth's axis is" (only the
top eight of 128,256 are drawn; the rest are ignored); the first-token candidates ("Sure", "Here", "```json", "{");
the flashcard text; the three "level" probabilities (notes §12's numbers, relabelled); the 80-bar logit strip; the
allowed cells in the mask strip; the batch sizes in the continuous-batching strip; the near-tie scores of " turn" vs
" position" (the measured fact is only that the text changed there). Real: every softmax / top-p / sample result
(computed with owl_reference.sample, default seed), the float64 and float32 sums (NumPy, seed 1, 4,096 products =
Owl-8B's d_model), every measured and cited number.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from owl_reference import *  # noqa: E402,F403

SERIES_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------- 1 · the last step

VOCAB = OWL["vocab"]                                   # 128,256 scores per pass
W_8B = weight_bytes()                                  # 16.06 GB
assert VOCAB == 128_256 and round(W_8B / 1e9) == 16     # "a hundred and twenty-eight thousand", "16 GB"

# ---------------------------------------------------------------------------- 2 · sampling (illustrative scores)

TOKS = [" tilted", " tipped", " slanted", " angled", " leaning", " not", " at", " always"]
LOGITS = np.array([6.2, 4.7, 4.4, 4.1, 3.7, 3.3, 3.0, 2.6])
TOP_K, TOP_P = 3, 0.9


def softmax(z, t=1.0):
    p = np.exp((z - z.max()) / t)
    return p / p.sum()


P1, P05, P15 = softmax(LOGITS), softmax(LOGITS, 0.5), softmax(LOGITS, 1.5)
ORDER = np.argsort(-P1)
CUM = np.cumsum(P1[ORDER])
NUC = int(np.searchsorted(CUM, TOP_P)) + 1             # the nucleus, exactly as sample() cuts it
KEPT_MASS = float(P1[ORDER[:NUC]].sum())
P_NUC = np.where(np.arange(len(P1)) < NUC, P1 / KEPT_MASS, 0.0)
PICK = sample(LOGITS, 1.0, TOP_P)                      # owl_reference.sample, its default seed
GREEDY = sample(LOGITS, 0)

assert list(ORDER) == list(range(len(TOKS)))           # the bars are drawn in rank order
assert round(P1[0] * 100) == 58                         # "fifty-eight percent"
assert round(P05[0] * 100) == 91 and round(P15[0] * 100) == 41   # "ninety-one", "forty-one"
assert NUC == 5 and round(KEPT_MASS, 2) == 0.93         # "that's five"
assert GREEDY == 0 and TOKS[PICK] == " tipped"          # greedy: tilted; the roll: tipped, the second choice
assert round(sum(P1[:TOP_K]) * 100) == 81

SAMPLE_SRC = """def sample(logits, temperature=1.0, top_p=1.0, rng=None):
    if temperature == 0:
        return int(np.argmax(logits))
    p = np.exp((logits - logits.max()) / temperature)
    p /= p.sum()
    order = np.argsort(-p)
    cum = np.cumsum(p[order])
    keep = order[: int(np.searchsorted(cum, top_p)) + 1]
    rng = rng or np.random.default_rng(0)
    return int(rng.choice(keep, p=p[keep] / p[keep].sum()))"""
_ns = {"np": np}
exec(SAMPLE_SRC, _ns)                                   # the excerpt on screen == owl_reference.sample
for _t in (0, 0.5, 1.0, 1.5):
    for _p in (0.5, 0.9, 1.0):
        for _s in range(5):
            assert _ns["sample"](LOGITS, _t, _p, np.random.default_rng(_s)) == sample(LOGITS, _t, _p, np.random.default_rng(_s))
assert len(SAMPLE_SRC.splitlines()) == 10

# ---------------------------------------------------------------------------- 3 · grammar masks (notes §12's example, relabelled)

LEVELS = ["medium", "moderate", "hard"]
P_LEVEL = np.array([0.50, 0.30, 0.20])
ALLOWED = {"easy", "medium", "hard"}
MASK = np.array([w in ALLOWED for w in LEVELS])
LEFT_MASS = float(P_LEVEL[MASK].sum())
P_MASKED = np.where(MASK, P_LEVEL / LEFT_MASS, 0.0)
with np.errstate(divide="ignore"):
    MASKED_LOGITS = np.where(MASK, np.log(P_LEVEL), -np.inf)
G_PICK = sample(MASKED_LOGITS, 1.0)
assert round(LEFT_MASS, 2) == 0.70                                  # "divide by seventy percent"
assert round(P_MASKED[0], 3) == 0.714 and round(P_MASKED[2], 3) == 0.286   # "seventy-one", "twenty-nine"
assert LEVELS[G_PICK] == "medium" and np.allclose(softmax(MASKED_LOGITS), P_MASKED)

CARD_FINAL = {"front": "Why do we have seasons?", "back": "Earth's axis is tipped.", "level": "medium"}
assert CARD_FINAL["back"].endswith(TOKS[PICK].strip() + ".") and CARD_FINAL["level"] == LEVELS[G_PICK]
CHATTY = "Sure! Here's your flashcard:\n" + json.dumps(CARD_FINAL)
try:
    json.loads(CHATTY)
    raise AssertionError("the chatty reply should not parse")
except json.JSONDecodeError as _e:
    PARSE_ERR = str(_e).split(" (char")[0]               # "Expecting value: line 1 column 1"
assert json.loads(json.dumps(CARD_FINAL, indent=2)) == CARD_FINAL

# ---------------------------------------------------------------------------- 4 · the mask's cost

MASK_BYTES = VOCAB / 8                                  # one bit per token
STEP_H100 = step_time()                                 # 4.79 ms: Owl-8B's batch-1 floor on an H100
LLG_US = 50                                             # llguidance, notes §12 (cited)
LLG_SHARE = LLG_US * 1e-6 / STEP_H100
assert MASK_BYTES == 16_032 and round(MASK_BYTES / 1000) == 16      # "about sixteen kilobytes"
assert round(STEP_H100 * 1000, 1) == 4.8 and round(LLG_SHARE * 100) == 1   # "about one percent"

# ---------------------------------------------------------------------------- 5 · temperature zero (measured + cited)

DET = measured("determinism")
D128, D256 = DET["in_process__llama-3.1-8b__4bit"], DET["in_process__llama-3.1-8b__4bit__chat256"]
SOLO, BUSY = DET["server_det_solo"], DET["server_det_underload"]
BATCHES = [1, 2, 4, 8, 16]
assert D128["batch1_runs"] == D256["batch1_runs"] == 20                       # "twenty runs"
assert D128["batch1_distinct_outputs"] == D256["batch1_distinct_outputs"] == 1  # "one answer"
assert (D128["gen_tokens"], D256["gen_tokens"]) == (128, 256)
DIV128 = [D128["same_prompt_inside_batch"][str(b)]["first_divergence_token"] for b in BATCHES]
DIV256 = [D256["same_prompt_inside_batch"][str(b)]["first_divergence_token"] for b in BATCHES]
assert DIV128 == [None] * 5                                                   # "identical every time"
assert DIV256 == [None, None, 221, 221, 132]                                 # "changed at batch four"
assert BUSY["requests"] == 20 and BUSY["distinct_outputs"] == 1 and DET["server_underload_matching_solo_majority"] == 20
assert list(BUSY["hash_counts"]) == list(SOLO["hash_counts"])                # the same answer alone and under load
RAW256 = json.loads((SERIES_DIR / "research" / "measurements" / "raw" / "determinism__llama-3.1-8b__4bit__chat256.json").read_text())
B4 = RAW256["in_batch"]["4"]
LAMP = "Think of it like a big lamp in your room. If you"
REF_AFTER, B4_AFTER = B4["ref_text_after"], B4["batch_text_after"]
assert LAMP + REF_AFTER[len(" your room. If you"):] in RAW256["reference_text"].replace("\n", " ")
assert REF_AFTER.startswith(" your room. If you turn it towards you") and B4_AFTER.startswith(" your room. If you position it directly above you")
SPEC_DIV = measured("speculative", "llama-3.1-8b__4bit", "prompts", "open_chat", "by_num_draft_tokens", "2", "first_divergence_token")
assert SPEC_DIV == 221 == DIV256[2]                                           # "this same token"

TML_RUNS, TML_UNIQUE, TML_MOST = 1000, 80, 78                                 # notes §12 (Thinking Machines, 2025)
TML_SAME, TML_QUEENS, TML_NYC = 102, 992, 8                                   # the blog, read 2026-10-24
T_DEFAULT, T_INVARIANT, T_BETTER = 26, 55, 42                                 # seconds, Qwen3-8B on vLLM (notes §12)
assert TML_QUEENS + TML_NYC == TML_RUNS and TML_UNIQUE == 80 and TML_SAME == 102

# ---------------------------------------------------------------------------- 6 · the order of a sum

LOOSE = (0.1 + 1e20) - 1e20
TIGHT = 0.1 + (1e20 - 1e20)
assert LOOSE == 0.0 and TIGHT == 0.1                                          # "the answer can change"

N_DOT = OWL["d_model"]                                                        # 4,096 products in one dot product
_rng = np.random.default_rng(1)
PRODUCTS = (_rng.standard_normal(N_DOT) * _rng.standard_normal(N_DOT) / 64).astype(np.float32)
RUN_SUM = float(np.cumsum(PRODUCTS, dtype=np.float32)[-1])                    # one long run (a full batch)
PIECES = [np.cumsum(c, dtype=np.float32)[-1] for c in np.split(PRODUCTS, 4)]  # four pieces (a lone request)
SPLIT_SUM = float(np.cumsum(np.array(PIECES, dtype=np.float32), dtype=np.float32)[-1])
S_RUN, S_SPLIT = f"{RUN_SUM:.8f}", f"{SPLIT_SUM:.8f}"
DIFF_AT = next(i for i, (a, b) in enumerate(zip(S_RUN, S_SPLIT)) if a != b)
assert N_DOT == 4096 and RUN_SUM != SPLIT_SUM and abs(RUN_SUM - SPLIT_SUM) < 1e-6   # "differ in the last digits"
assert (S_RUN, S_SPLIT) == ("0.47209933", "0.47209930") and DIFF_AT == len(S_RUN) - 1

NEAR = {1: (18.375, 18.371), 4: (18.369, 18.373)}                             # illustrative near-tie scores
assert NEAR[1][0] > NEAR[1][1] and NEAR[4][1] > NEAR[4][0]

# ---------------------------------------------------------------------------- look

PREFIX = "ep17"
BAR, BAR_FILL = MUTED, CARD_HI          # a token's score / probability
PICKED = (TOKEN, GREEN_FILL)            # the chosen token
MASKED = (RED, RED_FILL)                # forbidden by the grammar
PROD, PROD_FILL = COMPUTE, AMBER_FILL   # products and sums: the math units


def cite(s, size=20):
    return text(s, size, MUTED)


def model_card(name, lines, color, width=3.9, owl_size=0.42):
    words = m.VGroup(text(name, 28, INK, weight=m.BOLD), *[code(t, 20, MUTED) for t in lines])
    words.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.08)
    body = m.VGroup(owl(owl_size), words).arrange(m.RIGHT, buff=0.3)
    box = card(max(width, body.width + 0.7), body.height + 0.45, color, CARD)
    body.move_to(box).align_to(box, m.LEFT).shift(0.35 * m.RIGHT)
    return m.VGroup(box, body)


def tick_mark(color=TOKEN, size=0.3):
    pts = [np.array(p) * size for p in ([-0.5, 0.02, 0], [-0.14, -0.36, 0], [0.5, 0.42, 0])]
    return m.VMobject().set_points_as_corners(pts).set_stroke(color, 6)


def cross_mark(color=RED, size=0.26):
    h = size / 2
    return m.VGroup(m.Line([-h, -h, 0], [h, h, 0]), m.Line([-h, h, 0], [h, -h, 0])).set_stroke(color, 6)


def tchip(word, color=EDGE, fill=CARD, width=None, height=0.6, size=24):
    shown = word.strip() or word
    w = width or max(1.1, code(shown, size - 4).width + 0.5)
    return chip(shown, color, fill, width=w, height=height, size=size, mono=True)


def swap(old, new, shift=0.0):
    """Cross-fade one label into another (both glide a little)."""
    return m.AnimationGroup(m.FadeOut(old, shift=shift * m.UP), m.FadeIn(new, shift=shift * m.UP))


def lines_of(words, src):
    out, k = [], 0
    for line in src.split("\n"):
        n = sum(not c.isspace() for c in line)
        out.append(words[k:k + n])
        k += n
    return out


def line_band(box, glyph_rows, color):
    """A soft highlight band across a code card behind some of its lines."""
    top = max(g.get_top()[1] for row in glyph_rows for g in row) + 0.07
    bot = min(g.get_bottom()[1] for row in glyph_rows for g in row) - 0.09
    left, right = box.get_left()[0] + 0.1, box.get_right()[0] - 0.1
    band = m.Rectangle(width=right - left, height=top - bot, stroke_width=0, fill_color=color, fill_opacity=0.13)
    mark = m.Rectangle(width=0.06, height=top - bot, stroke_width=0, fill_color=color, fill_opacity=1)
    band.move_to([(left + right) / 2, (top + bot) / 2, 0])
    mark.move_to([left + 0.03, (top + bot) / 2, 0])
    return m.VGroup(band, mark).set_z_index(1)


class HBars:
    """Horizontal bars, one row per token: a mono label right-aligned at x0, a bar from x0, a value after it."""

    def __init__(self, tokens, values, x0, y_top, dy, scale, fmt, size=22, height=0.34, val_size=20):
        self.x0, self.scale, self.fmt, self.height, self.val_size = x0, scale, fmt, height, val_size
        self.ys = [y_top - i * dy for i in range(len(tokens))]
        self.labels = m.VGroup(*[code(t.strip(), size, INK).move_to([0, y, 0]).align_to([x0 - 0.22, 0, 0], m.RIGHT)
                                 for t, y in zip(tokens, self.ys)])
        self.colors = [(BAR, BAR_FILL)] * len(tokens)
        self.bars = m.VGroup(*[self.rect(i, v) for i, v in enumerate(values)])
        self.vals = [self.val(i, v) for i, v in enumerate(values)]
        self.values = list(values)

    def rect(self, i, v, color=None, fill=None):
        c, f = self.colors[i]
        w = max(v * self.scale, 0.035)
        r = m.Rectangle(width=w, height=self.height).set_stroke(color or c, 2).set_fill(fill or f, 1)
        return r.move_to([self.x0 + w / 2, self.ys[i], 0])

    def val(self, i, v, label=None, color=None):
        words = code(label if label is not None else self.fmt(v), self.val_size, color or MUTED)
        return words.move_to([0, self.ys[i], 0]).align_to([self.x0 + max(v * self.scale, 0.035) + 0.18, 0, 0], m.LEFT)

    def row(self, i):
        return m.VGroup(self.labels[i], self.bars[i], self.vals[i])

    def to(self, values, fmt=None, rows=None, labels=None, colors=None):
        """Animate the bars (and their value labels) to new values; keeps track of the new labels."""
        fmt = fmt or self.fmt
        anims = []
        for i in rows if rows is not None else range(len(values)):
            if colors and i in colors:
                self.colors[i] = colors[i]
            v = values[i]
            anims.append(m.Transform(self.bars[i], self.rect(i, v)))
            lab = labels.get(i) if labels else None
            col = self.colors[i][0] if self.colors[i][0] != BAR else MUTED
            new = self.val(i, v, lab, col)
            anims.append(swap(self.vals[i], new))
            self.vals[i] = new
            self.values[i] = v
        return m.AnimationGroup(*anims)

    def paint(self, i, color, fill):
        self.colors[i] = (color, fill)
        return m.AnimationGroup(self.bars[i].animate.set_stroke(color).set_fill(fill, 1),
                                self.vals[i].animate.set_color(color))


def pct(v):
    return f"{v * 100:.0f}%"


def cells_row(n, size, gap, color, fill, y, x_left, groups=1, group_gap=0.0):
    """n small squares in a row, optionally in equal groups separated by group_gap."""
    row, x, per = m.VGroup(), x_left, n // groups
    for i in range(n):
        if i and i % per == 0:
            x += group_gap
        row.add(m.Square(size).set_stroke(color, 1.6).set_fill(fill, 1).move_to([x + size / 2, y, 0]))
        x += size + gap
    return row


def brace_under(group, color=PROD, depth=0.18):
    left, right = group.get_left()[0], group.get_right()[0]
    y = group.get_bottom()[1] - 0.1
    mid = (left + right) / 2
    pts = [[left, y, 0], [left, y - depth / 2, 0], [mid - 0.08, y - depth / 2, 0], [mid, y - depth, 0],
           [mid + 0.08, y - depth / 2, 0], [right, y - depth / 2, 0], [right, y, 0]]
    return m.VMobject().set_points_as_corners([np.array(p) for p in pts]).set_stroke(color, 2.5)


# ---------------------------------------------------------------------------- the episode


class Ep17TheLastStepOfAToken(Episode):
    number = 17

    def lesson(self):
        self.last_step()
        self.sampling()
        self.in_code()
        self.grammar()
        self.mask_cost()
        self.temp_zero()
        self.sum_order()
        self.measured_here()
        self.the_fix()
        self.recap()

    # ------------------------------------------------------------------ 1 · the last step

    def last_step(self):
        self.next_section("1 · The last step", notes=(
            f"Every forward pass of Owl-8B (Llama-3.1-8B, {W_8B / 1e9:.2f} GB in BF16) ends in one logit per vocabulary "
            f"token: {VOCAB:,} scores (OWL['vocab']). Episodes 1–16 made the pass itself fast (episode 1: prefill once, then "
            "sample one token per decode step). The strip of bars is schematic (seeded random heights). The picked token "
            "'tipped' is the one section 2's sample() call really returns."))
        big = model_card("Owl-8B", ["8B · BF16", f"{W_8B / 1e9:.0f} GB of weights"], WEIGHTS).move_to([-4.35, 1.05, 0])
        rng = np.random.default_rng(7)
        n, x_left, width, base = 72, -1.15, 5.0, 0.25
        heights = 0.12 + 1.35 * rng.random(n) ** 3
        heights[21], heights[45] = 1.62, 1.5
        bw = width / n
        strip = m.VGroup(*[m.Rectangle(width=bw * 0.62, height=h).set_stroke(width=0).set_fill(BAR, 0.6)
                           .move_to([x_left + (i + 0.5) * bw, base + h / 2, 0]) for i, h in enumerate(heights)])
        floor = m.Line([x_left - 0.1, base, 0], [x_left + width + 0.1, base, 0]).set_stroke(FAINT, 2)
        a_in = arrow(big.get_right() + [0.05, 0, 0], [x_left - 0.2, 1.05, 0], WEIGHTS)
        s_lab = code(f"{VOCAB:,} logits · one score per token", 20, MUTED).next_to(floor, m.DOWN, buff=0.2).align_to(floor, m.LEFT)
        tok = tchip("tipped", TOKEN, GREEN_FILL, width=1.55).move_to([5.45, 1.05, 0])
        a_out = arrow([x_left + width + 0.15, 1.05, 0], tok.get_left(), TOKEN)
        l_fast = text("episodes 1–16: make this pass fast", 22, WEIGHTS).move_to([0, -0.75, 0]).align_to(big, m.LEFT)
        l_last = text("episode 17: scores in, one token out", 22, TOKEN).move_to([0, -0.75, 0]).align_to(tok, m.RIGHT)

        qs = []
        for i, (title, sub) in enumerate((("Pick one", "temperature · top-k · top-p"),
                                          ("Valid JSON", "grammar masks"),
                                          ("Same answer twice?", "batch-dependent sums"))):
            box = card(4.05, 1.45, EDGE, CARD, radius=0.18, stroke=2)
            num = title_text(str(i + 1), 40, TOKEN)
            words = m.VGroup(text(title, 24, INK, weight=m.BOLD), text(sub, 20, MUTED)).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.12)
            body = m.VGroup(num, words).arrange(m.RIGHT, buff=0.3).move_to(box).align_to(box, m.LEFT).shift(0.3 * m.RIGHT)
            qs.append(m.VGroup(box, body).move_to([-4.25 + i * 4.25, -2.3, 0]))

        self.say(
            "[Every pass of Owl-8B ends the same way]: [a hundred and twenty-eight thousand scores], [one per token it knows.]",
            m.AnimationGroup(self.new_header("01 · the last step", "Scores in, one token out"), m.FadeIn(big, shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowArrow(a_in), m.Create(floor),
                             m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in strip], lag_ratio=0.015, run_time=1.6)),
            m.FadeIn(s_lab, shift=0.1 * m.UP),
        )
        self.say(
            "[Episodes one to sixteen made that pass fast.] [Today]: [the last step, turning scores into one token.]",
            m.AnimationGroup(m.FadeIn(l_fast, shift=0.1 * m.UP), m.Circumscribe(big, color=WEIGHTS, buff=0.06)),
            m.FadeIn(l_last, shift=0.1 * m.UP),
            m.AnimationGroup(m.GrowArrow(a_out), m.FadeIn(tok, target_position=strip[45])),
        )
        self.say(
            "[Three questions]: [how to pick it], [how to keep Owl's flashcards valid JSON], [and why temperature zero can give two answers.]",
            m.FadeIn(qs[0], shift=0.2 * m.UP),
            m.Circumscribe(qs[0], color=TOKEN, buff=0.06),
            m.FadeIn(qs[1], shift=0.2 * m.UP),
            m.FadeIn(qs[2], shift=0.2 * m.UP),
        )
        self.wait(0.15)
        self.clear()

    # ------------------------------------------------------------------ 2 · sampling

    def sampling(self):
        self.next_section("2 · Sampling", notes=(
            "Illustrative scores (said 'for example'): the top eight of 128,256 next-token logits after Owl's flashcard text "
            f"'Earth's axis is'; logits {LOGITS.tolist()} (the other tokens ignored). softmax(z / T): T = 1 → tilted "
            f"{P1[0]:.3f}; T = 0.5 → {P05[0]:.3f}; T = 1.5 → {P15[0]:.3f}; T = 0 → argmax (owl_reference.sample returns "
            f"argmax). Top-k = {TOP_K} keeps {sum(P1[:TOP_K]):.2f} of the mass. Top-p = {TOP_P}: running total "
            f"{', '.join(f'{c:.3f}' for c in CUM[:NUC])} → {NUC} tokens (sample() cuts at searchsorted(cum, top_p) + 1), "
            f"renormalised by {KEPT_MASS:.3f}; sample(logits, 1.0, 0.9) with its default seed returns '{TOKS[PICK].strip()}'. "
            "'AI, Explained' Episode 33 covers temperature, top-k, top-p and beam search in depth."))
        ctx = code('"back": "Earth\'s axis is', 24, INK, t2c={r'^"back"': CYAN, r'"Earth.*$': GREEN})
        ctx.move_to([0, 2.1, 0]).align_to([-6.2, 0, 0], m.LEFT)
        ep33 = cite("AI, Explained · Episode 33: decoding and sampling").move_to([0, 2.1, 0]).align_to([6.3, 0, 0], m.RIGHT)
        chart = HBars(TOKS, LOGITS, x0=-3.95, y_top=1.1, dy=0.45, scale=0.7, fmt=lambda v: f"{v:.1f}")
        head_l = code("score", 22, MUTED).next_to(chart.bars[0], m.UP, buff=0.16).align_to(chart.bars[0], m.LEFT)
        head_p = code("probability", 22, MUTED).move_to(head_l).align_to(head_l, m.LEFT)
        topn = cite("top 8 of 128,256 · scores for example").move_to([0, -2.7, 0]).align_to([-6.2, 0, 0], m.LEFT)

        px = 4.35
        f_soft = m.MathTex(r"p_i", "=", r"\frac{e^{z_i/T}}{\sum_j e^{z_j/T}}", font_size=46, color=INK).move_to([px, 1.0, 0])
        f_soft[0].set_color(TOKEN)
        t_lab = {t: title_text(f"T = {t}", 44, INK).move_to([px, -0.45, 0]) for t in ("1.0", "0.5", "1.5", "0")}
        t_note = {
            "0.5": text("sharper: the top token grows", 22, MUTED),
            "1.5": text("flatter: the tail grows", 22, MUTED),
            "0": text("no roll: always the top score", 22, MUTED),
        }
        for t in t_note.values():
            t.next_to(t_lab["1.0"], m.DOWN, buff=0.25).set_x(px)
        greedy = pill("greedy decoding = argmax", TOKEN, GREEN_FILL, 22).move_to([px, -1.95, 0])

        cut_k = m.DashedLine([-5.6, (chart.ys[TOP_K - 1] + chart.ys[TOP_K]) / 2, 0], [2.1, (chart.ys[TOP_K - 1] + chart.ys[TOP_K]) / 2, 0],
                             dash_length=0.1).set_stroke(INK, 2.5)
        k_lab = code(f"top-k = {TOP_K}", 22, INK, weight=m.BOLD).next_to(cut_k, m.RIGHT, buff=0.15)
        y_cut = (chart.ys[NUC - 1] + chart.ys[NUC]) / 2
        cut_p = m.DashedLine([-5.6, y_cut, 0], [2.1, y_cut, 0], dash_length=0.1).set_stroke(TOKEN, 2.5)
        p_lab = code(f"top-p = {TOP_P}", 22, TOKEN, weight=m.BOLD).next_to(cut_p, m.RIGHT, buff=0.15)
        run_x = 5.2
        run_head = code("running total", 20, MUTED).move_to([run_x, 1.7, 0])
        runs = m.VGroup(*[code(f"{c * 100:.0f}%", 20, TOKEN if i < NUC else FAINT).move_to([run_x, chart.ys[i], 0]) for i, c in enumerate(CUM)])
        p_note = text(f"the fewest tokens that add up to {TOP_P}: {NUC} here", 22, MUTED).move_to([0, -2.7, 0]).align_to([-0.6, 0, 0], m.LEFT)
        renorm = code(f"÷ {KEPT_MASS:.2f}", 26, INK, weight=m.BOLD).move_to([3.3, chart.ys[1], 0])
        die = m.Dot(radius=0.1, color=TOKEN).move_to([-5.75, chart.ys[0], 0]).set_z_index(3)
        roll_path = [0, 2, 4, 1, 3, 0, 2, PICK]
        picked = code(TOKS[PICK].strip(), 24, TOKEN, weight=m.BOLD).next_to(ctx, m.RIGHT, buff=0.16)
        picked.shift((ctx[-1].get_bottom()[1] - picked[0].get_bottom()[1]) * m.UP)

        self.say(
            "[Episode thirty-three of AI, Explained covered sampling.] [Here's the short version.]",
            m.AnimationGroup(self.new_header("02 · sampling", "From scores to odds"), m.FadeIn(ep33)),
            m.AddTextLetterByLetter(ctx, time_per_char=0.04),
        )
        self.say(
            "[Owl has written: Earth's axis is], [and these are its top scores for the next token, for example.]",
            m.Circumscribe(ctx, color=GREEN, buff=0.08),
            m.AnimationGroup(m.FadeIn(head_l), m.FadeIn(topn),
                             m.LaggedStart(*[m.AnimationGroup(m.FadeIn(chart.labels[i]), m.GrowFromEdge(chart.bars[i], m.LEFT), m.FadeIn(chart.vals[i]))
                                             for i in range(len(TOKS))], lag_ratio=0.12)),
        )
        chart.scale, chart.fmt = 4.8, pct
        self.say(
            "[Softmax turns scores into probabilities]: [raise each one as a power], [then divide by the total.]",
            m.FadeIn(f_soft, shift=0.1 * m.UP),
            m.AnimationGroup(swap(head_l, head_p), chart.to(P1)),
            m.FadeIn(t_lab["1.0"], scale=0.9),
        )
        self.say(
            "[Temperature divides every score first.] [At zero point five, tilted jumps to ninety-one percent]; [at one point five, it falls to forty-one.]",
            m.Circumscribe(f_soft[2], color=INK, buff=0.06),
            m.AnimationGroup(swap(t_lab["1.0"], t_lab["0.5"]), m.FadeIn(t_note["0.5"]), chart.to(P05)),
            m.AnimationGroup(swap(t_lab["0.5"], t_lab["1.5"]), swap(t_note["0.5"], t_note["1.5"]), chart.to(P15)),
        )
        t_back = title_text("T = 1.0", 44, INK).move_to(t_lab["1.0"])
        self.say(
            "[At temperature zero there's no roll at all]: [Owl takes the top score.] [That's greedy decoding.]",
            m.AnimationGroup(swap(t_lab["1.5"], t_lab["0"]), swap(t_note["1.5"], t_note["0"]), chart.to(P1)),
            chart.paint(0, *PICKED),
            m.FadeIn(greedy, shift=0.1 * m.UP),
        )
        self.say(
            "[Top-k keeps a fixed number of tokens, here three.] [Top-p keeps the fewest whose odds add up to a threshold]: [at zero point nine, five.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(f_soft, t_lab["0"], t_note["0"], greedy)), chart.paint(0, BAR, BAR_FILL),
                             m.Create(cut_k), m.FadeIn(k_lab)),
            m.AnimationGroup(m.FadeOut(m.VGroup(cut_k, k_lab)), m.FadeIn(run_head),
                             m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.LEFT) for r in runs], lag_ratio=0.12)),
            m.AnimationGroup(m.Create(cut_p), m.FadeIn(p_lab), m.FadeIn(p_note),
                             *[part.animate.set_opacity(0.3) for i in range(NUC, len(TOKS)) for part in chart.row(i)]),
        )
        roll = m.Succession(*[die.animate(run_time=0.22).move_to([-5.75, chart.ys[i], 0]) for i in roll_path[1:]])
        self.say(
            "[Renormalise those five], [and roll a weighted die]: [this time it lands on tipped, the second choice.]",
            m.AnimationGroup(m.FadeOut(runs[NUC:]), m.FadeIn(renorm, scale=0.9), chart.to(P_NUC, rows=range(NUC))),
            m.Succession(m.FadeIn(die, scale=0.5, run_time=0.3), roll),
            m.AnimationGroup(chart.paint(PICK, *PICKED), m.FadeIn(picked, target_position=chart.bars[PICK])),
        )
        self.wait(0.15)
        self.clear()

    # ------------------------------------------------------------------ 3 · in code

    def in_code(self):
        self.next_section("3 · In code", notes=(
            "owl_reference.sample, with its docstring dropped and the nucleus line split in two (cum = np.cumsum(...)); the "
            "scene runs this excerpt and checks it returns exactly owl_reference.sample's token for T ∈ {0, 0.5, 1, 1.5}, "
            "top_p ∈ {0.5, 0.9, 1} and five seeds. The chain is vLLM V1's order (sampler.py docstring, read 2026-10-24): "
            "the structured-output bitmask is applied to the logits before the sampler; then allowed-token / bad-word / "
            "logit-bias processors, penalties (repetition, frequency, presence), temperature, min-p, top-k/top-p, sample."))
        blk = code_block(SAMPLE_SRC, size=18, title="owl_reference.py").move_to([0, 0.3, 0]).align_to([-6.3, 0, 0], m.LEFT)
        rows = lines_of(blk[1], SAMPLE_SRC)
        groups = [((1, 2), "greedy at zero", TOKEN), ((3, 4), "softmax with T", INK), ((5, 6, 7), "the nucleus", INK),
                  ((8, 9), "one weighted roll", TOKEN)]
        bands, notes = [], []
        for idx, label, color in groups:
            band = line_band(blk[0], [rows[i] for i in idx], color)
            note = text(label, 24, color, weight=m.BOLD).move_to([0, band.get_y(), 0]).align_to([blk.get_right()[0] + 0.35, 0, 0], m.LEFT)
            bands.append(band)
            notes.append(note)

        chain_spec = [("logits", MUTED, CARD, "128,256 scores"), ("mask", RED, RED_FILL, "grammar: −∞"),
                      ("penalties", INK, CARD, "lower repeats"), ("temperature", INK, CARD, "divide by T"),
                      ("top-k, top-p", INK, CARD, "trim the tail"), ("pick", TOKEN, GREEN_FILL, "one roll")]
        chips_, subs = m.VGroup(), m.VGroup()
        for i, (label, color, fill, sub) in enumerate(chain_spec):
            c = chip(label, color, fill, width=1.86, height=0.66, size=20)
            c.move_to([-5.55 + i * 2.22, -0.15, 0])
            chips_.add(c)
            subs.add(text(sub, 20, MUTED).next_to(c, m.DOWN, buff=0.2))
        arrows = m.VGroup(*[arrow(chips_[i].get_right(), chips_[i + 1].get_left(), MUTED, buff=0.06) for i in range(5)])
        c_title = text("a serving engine's chain of logits processors", 24, INK, weight=m.BOLD).move_to([0, 1.2, 0])
        c_src = cite("order as in vLLM's sampler · every request, every step").move_to([0, -1.85, 0])
        rail = m.Line(chips_[0].get_center(), chips_[-1].get_center())

        self.say(
            "[In Owl's reference code, that's ten lines]: [greedy at zero], [softmax with temperature], [the nucleus], [and one roll.]",
            m.AnimationGroup(self.new_header("03 · in code", "Ten lines, then a chain"), m.FadeIn(blk, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(bands[0]), m.FadeIn(notes[0], shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.FadeIn(bands[1]), m.FadeIn(notes[1], shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.FadeIn(bands[2]), m.FadeIn(notes[2], shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.FadeIn(bands[3]), m.FadeIn(notes[3], shift=0.1 * m.LEFT)),
        )
        code_frame = m.VGroup(blk, *bands, *notes)
        self.say(
            "[Engines run these steps as a chain of logits processors], [penalties included], [for every request, every step.]",
            m.AnimationGroup(m.FadeOut(code_frame, shift=0.2 * m.UP), m.FadeIn(c_title),
                             m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.RIGHT) for c in chips_], lag_ratio=0.15),
                             m.LaggedStart(*[m.GrowArrow(a) for a in arrows], lag_ratio=0.15), m.FadeIn(subs)),
            m.Circumscribe(chips_[2], color=INK, buff=0.06),
            m.AnimationGroup(m.FadeIn(c_src), m.ShowPassingFlash(rail.copy().set_stroke(TOKEN, 8), time_width=0.5, run_time=1.4)),
        )
        self.wait(0.15)
        self.clear()

    # ------------------------------------------------------------------ 4 · grammar masks

    def grammar(self):
        self.next_section("4 · Grammar masks", notes=(
            "Constrained decoding (notes §12): at each step a grammar engine computes which vocabulary tokens keep the output "
            "valid (JSON schema, regex, CFG) and sets every other logit to −∞, then samples. Owl's flashcard schema and the "
            "card are illustrative. The chatty reply really fails json.loads: '" + PARSE_ERR + "' (computed in the scene). "
            "First-token candidates illustrative. The 'level' probabilities are notes §12's worked example (yes 0.50 / maybe "
            f"0.30 / no 0.20 → 0.714 / 0.286), relabelled medium / moderate / hard: kept mass {LEFT_MASS:.2f}, renormalised "
            f"{P_MASKED[0]:.3f} and {P_MASKED[2]:.3f}; sample() on the masked logits (default seed) picks '{LEVELS[G_PICK]}'. "
            "'JSON mode guarantees syntax; schema-constrained decoding guarantees structure' (notes §12). 'The format is "
            "guaranteed, but the distribution was bent' (notes §12)."))
        schema_src = '{\n  "front": string,\n  "back":  string,\n  "level": "easy" | "medium" | "hard"\n}'
        schema = code_block(schema_src, size=18, title="flashcard schema")
        paint(schema[1], schema_src, {r'"(front|back|level)"(?=:)': CYAN, r'"(easy|medium|hard)"': GREEN, r"\bstring\b": VIOLET})
        schema.move_to([-3.3, 1.15, 0])
        example = json_card(CARD_FINAL, size=18, title="one of Owl's cards", accent=MUTED).move_to([3.3, 1.15, 0])
        reply = message("assistant", "Sure! Here's your flashcard: {\"front\": …", width=6.2, size=20).move_to([-3.3, -1.85, 0])
        term = terminal(["$ python add_cards.py", "! JSONDecodeError:", f"! {PARSE_ERR}"], width=6.2, size=18,
                        title="the study app").move_to([3.3, -1.85, 0])
        frame_a = m.VGroup(schema, example, reply, term)

        self.say(
            "[Owl's flashcards go straight into a study app], [so every card must be valid JSON, with exactly these fields.]",
            m.AnimationGroup(self.new_header("04 · grammar masks", "Valid JSON, every time"), m.FadeIn(example, shift=0.1 * m.UP)),
            m.FadeIn(schema, shift=0.1 * m.UP),
        )
        self.say(
            "[Asking nicely in the prompt usually works.] [But one chatty reply], [and the parser fails.]",
            m.FadeIn(reply, shift=0.1 * m.UP),
            m.Circumscribe(reply[2][1], color=RED, buff=0.06),
            m.FadeIn(term, shift=0.1 * m.UP),
        )

        # frame B: the grammar engine at work
        out_box = card(6.3, 2.35, EDGE, CARD, radius=0.16, stroke=2).move_to([-3.25, 1.2, 0])
        out_cap = code("Owl's output so far", 20, MUTED).align_to(out_box, m.UL).shift(0.3 * m.RIGHT + 0.17 * m.DOWN)
        cursor0 = m.Rectangle(width=0.13, height=0.24, stroke_width=0, fill_color=TOKEN, fill_opacity=1)
        cursor0.move_to(out_cap.get_left() + [0.07, -0.55, 0])
        partial_src = '{\n  "front": "Why do we have seasons?",\n  "back": "Earth\'s axis is tipped.",\n  "level": "'
        rules = {r'"(front|back|level)"(?=:)': CYAN, r':\s*"[^"\n]*"': GREEN}
        whole = code(partial_src, 18, INK).next_to(out_cap, m.DOWN, buff=0.16).align_to(out_cap, m.LEFT)
        partial = m.VGroup()                       # one Text per line (so the cursor sits beside the last line's box)
        for line, glyph_row in zip(partial_src.split("\n"), lines_of(whole, partial_src)):
            partial.add(code(line.strip(), 18, INK, t2c=rules).move_to(glyph_row))
        base_y = partial[-1][1].get_bottom()[1]                                  # the "l" of "level": the baseline
        cursor = cursor0.copy().next_to(partial[-1], m.RIGHT, buff=0.06)
        cursor.shift((base_y - 0.04 - cursor.get_bottom()[1]) * m.UP)
        eng_box = card(6.0, 2.35, INK, CARD, radius=0.16, stroke=2).move_to([3.4, 1.2, 0])
        eng_title = text("grammar engine", 26, INK, weight=m.BOLD).align_to(eng_box, m.UL).shift(0.3 * m.RIGHT + 0.2 * m.DOWN)
        eng_sub = text("follows the schema, token by token", 20, MUTED).next_to(eng_title, m.DOWN, buff=0.1).align_to(eng_title, m.LEFT)
        state0 = code("allowed: {", 22, TOKEN).next_to(eng_sub, m.DOWN, buff=0.35).align_to(eng_title, m.LEFT)
        state1 = code('allowed: easy | medium | hard', 20, TOKEN).move_to(state0).align_to(state0, m.LEFT)

        firsts = [("Sure", False), ("Here", False), ("```json", False), ("{", True)]
        f_chips, f_marks = m.VGroup(), m.VGroup()
        for i, (w, ok) in enumerate(firsts):
            c = tchip(w, *(PICKED if ok else (EDGE, CARD)), width=1.75).move_to([-3.525 + i * 2.35, -1.25, 0])
            f_chips.add(c)
            mk = (tick_mark() if ok else cross_mark()).next_to(c, m.DOWN, buff=0.22)
            f_marks.add(mk)
        f_cap = text("the first token: Owl's favourite candidates, for example", 22, MUTED).move_to([0, -2.6, 0])

        lv = HBars(LEVELS, P_LEVEL, x0=-3.55, y_top=-0.75, dy=0.62, scale=6.0, fmt=pct, size=24, height=0.4, val_size=22)
        lv_cap = cite("top 3 of 128,256 · for example").move_to([0, -2.85, 0]).align_to([-6.2, 0, 0], m.LEFT)
        strike = cross_mark(RED, 0.24).next_to(lv.labels[1], m.LEFT, buff=0.25)
        why = text("not in the schema: −∞", 22, RED).move_to([0, lv.ys[1], 0]).align_to([0.9, 0, 0], m.LEFT)
        div = code(f"÷ {LEFT_MASS:.2f}", 28, INK, weight=m.BOLD).move_to([0, lv.ys[2], 0]).align_to([3.9, 0, 0], m.LEFT)
        add_src = 'medium"\n}'
        done = code(add_src, 18, INK, t2c={r'medium"': GREEN})
        done[0:7].next_to(partial[-1], m.RIGHT, buff=0.04)
        done[0:7].shift((base_y - done[0].get_bottom()[1]) * m.UP)
        done[7:].next_to(partial, m.DOWN, buff=0.08).align_to(partial, m.LEFT)
        valid = pill("json.loads ✓  valid", TOKEN, GREEN_FILL, 22, mono=True).move_to([0, lv.ys[0], 0]).align_to([2.3, 0, 0], m.LEFT)
        bent = m.VGroup(pill("format: guaranteed", TOKEN, GREEN_FILL, 22), pill("odds: bent", AMBER, AMBER_FILL, 22))
        bent.arrange(m.RIGHT, buff=0.35).move_to([0, -2.85, 0]).align_to([0.9, 0, 0], m.LEFT)

        self.say(
            "[Constrained decoding makes it certain]: [a grammar engine follows the schema], [and at the start, allows only an opening brace.]",
            m.AnimationGroup(m.FadeOut(frame_a, shift=0.2 * m.UP), m.FadeIn(out_box), m.FadeIn(out_cap), m.FadeIn(cursor0)),
            m.AnimationGroup(m.FadeIn(eng_box), m.FadeIn(eng_title), m.FadeIn(eng_sub)),
            m.AnimationGroup(m.FadeIn(state0, shift=0.1 * m.UP), m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.UP) for c in f_chips], lag_ratio=0.15),
                             m.FadeIn(f_cap)),
        )
        self.say(
            "[So a chatty opening can never be written]: [everything but the brace is masked.]",
            m.LaggedStart(*[m.AnimationGroup(f_chips[i][0].animate.set_stroke(RED).set_fill(RED_FILL, 1), m.Create(f_marks[i])) for i in range(3)], lag_ratio=0.3),
            m.AnimationGroup(m.Create(f_marks[3]), m.Circumscribe(f_chips[3], color=TOKEN, buff=0.06)),
        )
        self.say(
            "[Later, at the level field], [Owl favours medium, at fifty percent], [but moderate is close behind], [and isn't allowed.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(f_chips, f_marks, f_cap)), m.FadeOut(cursor0), m.FadeIn(partial), m.FadeIn(cursor),
                             swap(state0, state1)),
            m.AnimationGroup(m.FadeIn(lv_cap), m.LaggedStart(*[m.AnimationGroup(m.FadeIn(lv.labels[i]), m.GrowFromEdge(lv.bars[i], m.LEFT), m.FadeIn(lv.vals[i]))
                                                               for i in range(3)], lag_ratio=0.2)),
            m.Circumscribe(lv.row(1), color=INK, buff=0.08),
            m.Circumscribe(state1, color=TOKEN, buff=0.06),
        )
        lv.colors[1] = MASKED
        self.say(
            "[So the mask sets moderate's score to minus infinity], [and its bar drops to zero.]",
            m.AnimationGroup(m.Create(strike), lv.labels[1].animate.set_color(RED), m.FadeIn(why, shift=0.1 * m.LEFT)),
            lv.to(P_MASKED, rows=[1], labels={1: "0%"}),
        )
        self.say(
            "[Renormalise]: [divide what's left by seventy percent]: [medium becomes seventy-one percent, hard twenty-nine.]",
            m.Wait(0.3),
            m.FadeIn(div, scale=0.9),
            lv.to(P_MASKED, rows=[0, 2], fmt=lambda v: f"{v * 100:.1f}%"),
        )
        self.say(
            "[Then Owl picks as usual], [and the card stays valid, token by token, to the closing brace.]",
            m.AnimationGroup(lv.paint(G_PICK, *PICKED), m.FadeOut(cursor), m.FadeIn(done[0:7], target_position=lv.bars[G_PICK])),
            m.AnimationGroup(m.FadeIn(done[7:], shift=0.1 * m.UP), m.FadeOut(div), m.FadeIn(valid, scale=0.9)),
        )
        self.say(
            "[The catch]: [the format is guaranteed], [but the odds were bent], [and that can cost quality.]",
            m.FadeOut(lv_cap),
            m.FadeIn(bent[0], shift=0.1 * m.UP),
            m.FadeIn(bent[1], shift=0.1 * m.UP),
            m.Circumscribe(lv.row(2), color=AMBER, buff=0.08),
        )
        self.wait(0.15)
        self.clear()

    # ------------------------------------------------------------------ 5 · the mask's cost

    def mask_cost(self):
        self.next_section("5 · The mask's cost", notes=(
            f"Notes §12: the mask is a {VOCAB:,}-bit vector for Llama 3 = {MASK_BYTES:,.0f} bytes ≈ 16 KB per sequence per step. "
            "Outlines (Willard & Louf, arXiv 2307.09702): the naive approach 'entails a fixed O(N) cost for each token "
            "generated'; precompiling the regex/schema into an FSM with 'an index over a language model's vocabulary' makes "
            "each step 'O(1) on average'. XGrammar (Dong et al., arXiv 2411.15100, MLSys 2025): pushdown (stack) matcher for "
            "nested grammars like JSON; vocabulary split 'into context-independent tokens that can be prechecked and "
            "context-dependent tokens'; grammar work overlapped with GPU compute; 'up to 100x speedup'. llguidance: 'on the "
            f"order of 50μs of CPU time per token (for 128k tokenizer)'; vs Owl-8B's batch-1 floor on an H100, "
            f"{STEP_H100 * 1000:.2f} ms (owl_reference.step_time) → {LLG_SHARE * 100:.1f}% (narrated as 'one engine'). The allowed cells in the strip and "
            "the first timeline (CPU work drawn large) are schematic; the second timeline is to scale."))
        n, size, gap = 96, 0.1, 0.025
        x_left = -6.0
        rng = np.random.default_rng(3)
        allowed = set(rng.choice(n, 4, replace=False).tolist())
        bits = m.VGroup(*[m.Rectangle(width=size, height=0.36).set_stroke(width=0).set_fill(TOKEN if i in allowed else RED_FILL, 1)
                          .move_to([x_left + i * (size + gap) + size / 2, 1.75, 0]) for i in range(n)])
        b_lab = code(f"{VOCAB:,} bits = {MASK_BYTES:,.0f} bytes ≈ 16 KB", 22, INK).next_to(bits, m.DOWN, buff=0.22).align_to(bits, m.LEFT)
        b_sub = text("per request, per step", 22, MUTED).next_to(b_lab, m.RIGHT, buff=0.4)
        legend = m.VGroup(m.Square(0.22).set_stroke(width=0).set_fill(TOKEN, 1), text("allowed", 20, MUTED),
                          m.Square(0.22).set_stroke(width=0).set_fill(RED_FILL, 1).set_stroke(RED, 1.5), text("masked", 20, MUTED))
        legend.arrange(m.RIGHT, buff=0.15).next_to(bits, m.UP, buff=0.2).align_to(bits, m.RIGHT)
        rail = m.Line(bits[0].get_left(), bits[-1].get_right())

        y1, y2 = 0.2, -0.55
        naive = m.VGroup(text("check every token, every step", 24, INK), code(f"{VOCAB:,} checks", 22, RED, weight=m.BOLD))
        naive.arrange(m.RIGHT, buff=0.4).move_to([0, y1, 0]).align_to([-6.0, 0, 0], m.LEFT)
        index = m.VGroup(text("Outlines: schema → state machine + index", 24, INK), code("≈ 1 lookup", 22, TOKEN, weight=m.BOLD))
        index.arrange(m.RIGHT, buff=0.4).move_to([0, y2, 0]).align_to([-6.0, 0, 0], m.LEFT)
        idx_src = cite("Willard & Louf, 2023").next_to(index, m.RIGHT, buff=0.4)

        stack = m.VGroup(*[m.VGroup(card(1.0, 0.4, INK, CARD, radius=0.08, stroke=1.8), code(s, 20, INK)) for s in ("{", "[", "{")])
        for s in stack:
            s[1].move_to(s[0])
        stack.arrange(m.UP, buff=0.06).move_to([-5.6, -2.0, 0])
        st_lab = code("stack", 20, MUTED).next_to(stack, m.DOWN, buff=0.12)
        xg = m.VGroup(text("XGrammar", 26, INK, weight=m.BOLD), text("a stack for nested JSON", 22, INK), text("most tokens prechecked", 22, MUTED))
        xg.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.1).move_to([0, -1.9, 0]).align_to([-4.6, 0, 0], m.LEFT)
        xg_src = cite("Dong et al., 2024").next_to(xg, m.DOWN, buff=0.16).align_to(xg, m.LEFT)

        tx0, tw = 1.0, 5.2
        ty_gpu, ty_cpu = -1.8, -2.6
        gpu_l = code("GPU", 20, MEMORY).move_to([0, ty_gpu, 0]).align_to([tx0 - 0.2, 0, 0], m.RIGHT)
        cpu_l = code("CPU", 20, MUTED).move_to([0, ty_cpu, 0]).align_to([tx0 - 0.2, 0, 0], m.RIGHT)
        gpu_bar = m.Rectangle(width=tw, height=0.42).set_stroke(MEMORY, 2).set_fill(CYAN_FILL, 1).move_to([tx0 + tw / 2, ty_gpu, 0])
        gpu_txt = code(f"one Owl-8B step · {STEP_H100 * 1000:.1f} ms", 20, MEMORY).move_to(gpu_bar)
        cpu_big = m.Rectangle(width=1.9, height=0.42).set_stroke(INK, 2).set_fill(CARD_HI, 1).move_to([tx0 + 0.95, ty_cpu, 0])
        cpu_txt = code("next mask", 20, INK).move_to(cpu_big)
        cpu_true = m.Rectangle(width=max(tw * LLG_SHARE, 0.05), height=0.42).set_stroke(INK, 2).set_fill(INK, 1)
        cpu_true.move_to([tx0 + max(tw * LLG_SHARE, 0.05) / 2, ty_cpu, 0])
        cpu_val = code(f"≈ {LLG_US} µs · about {LLG_SHARE * 100:.0f}%", 20, INK, weight=m.BOLD).next_to(cpu_true, m.RIGHT, buff=0.2)
        llg_src = cite("llguidance · to scale").next_to(gpu_bar, m.UP, buff=0.15).align_to(gpu_bar, m.RIGHT)

        self.say(
            "[The mask is one bit per token]: [about sixteen kilobytes], [per request, per step.]",
            m.AnimationGroup(self.new_header("05 · the cost", "A mask every step"),
                             m.LaggedStart(*[m.FadeIn(b) for b in bits], lag_ratio=0.01, run_time=1.2)),
            m.AnimationGroup(m.FadeIn(b_lab, shift=0.1 * m.UP), m.FadeIn(legend)),
            m.FadeIn(b_sub, shift=0.1 * m.UP),
        )
        self.say(
            "[Checking every token, every step, would be slow.] [Outlines precompiles the schema], [so a step is about one lookup.]",
            m.AnimationGroup(m.FadeIn(naive, shift=0.1 * m.UP), m.ShowPassingFlash(rail.copy().set_stroke(RED, 8), time_width=0.4, run_time=1.4)),
            m.FadeIn(index[0], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(index[1], scale=0.9), m.FadeIn(idx_src)),
        )
        self.say(
            "[XGrammar adds a stack for nested JSON], [and builds the next mask while the GPU runs.]",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(s, shift=0.15 * m.DOWN) for s in stack], lag_ratio=0.25), m.FadeIn(st_lab),
                             m.FadeIn(xg[0], shift=0.1 * m.UP), m.FadeIn(xg[1], shift=0.1 * m.UP), m.FadeIn(xg[2], shift=0.1 * m.UP),
                             m.FadeIn(xg_src)),
            m.AnimationGroup(m.FadeIn(gpu_l), m.FadeIn(cpu_l), m.GrowFromEdge(gpu_bar, m.LEFT, run_time=1.2), m.FadeIn(gpu_txt),
                             m.GrowFromEdge(cpu_big, m.LEFT), m.FadeIn(cpu_txt)),
        )
        self.say(
            "[To scale, that's tiny]: [one engine quotes about fifty microseconds per token], [one percent of an Owl-8B step on an H100.]",
            m.AnimationGroup(m.FadeOut(cpu_txt), m.Transform(cpu_big, cpu_true), m.FadeIn(llg_src)),
            m.FadeIn(cpu_val, shift=0.1 * m.LEFT),
            m.Circumscribe(gpu_txt, color=MEMORY, buff=0.06),
        )
        self.wait(0.15)
        self.clear()

    # ------------------------------------------------------------------ 6 · temperature zero

    def temp_zero(self):
        self.next_section("6 · Temperature zero", notes=(
            "Measured on this Mac (research/measurements.md §12, mlx-lm 0.32.2, Owl-8B 4-bit, greedy): 20 batch-1 runs of "
            f"the {D128['gen_tokens']}-token octopus-facts prompt and of the {D256['gen_tokens']}-token Owl seasons prompt "
            "(chat.txt): 1 distinct output each (hash b669b363… for the seasons answer). Thinking Machines Lab (Horace He et "
            "al., 2025-09-10; notes §12): Qwen3-235B-A22B-Instruct-2507, 'Tell me about Richard Feynman', 1,000 completions "
            "of 1,000 tokens at temperature 0: 80 unique, the most common 78 times; identical for 102 tokens, first "
            "divergence at token 103 ('…born on May 11, 1918, in' → 992 'Queens, New York', 8 'New York City'; the blog, "
            "read 2026-10-24)."))
        prompt = m.VGroup(card(2.6, 0.75, REQUEST, PINK_FILL, radius=0.14), text("same prompt", 24, INK, weight=m.BOLD))
        prompt[1].move_to(prompt[0])
        prompt.move_to([-4.9, 1.5, 0])
        t0 = pill("T = 0", INK, CARD, 22, mono=True).next_to(prompt, m.DOWN, buff=0.2)
        mascot = owl(0.6).move_to([-2.2, 1.5, 0])
        a1 = arrow(prompt.get_right(), mascot.get_left(), MUTED)
        answers = m.VGroup(*[tchip(f"answer {i + 1}", *PICKED, width=2.0, size=22) for i in range(3)])
        answers.arrange(m.RIGHT, buff=0.55).move_to([2.6, 1.5, 0])
        eqs = text("the same, every time?", 22, TOKEN).next_to(answers, m.DOWN, buff=0.22)
        a2 = arrow(mascot.get_right(), answers.get_left(), MUTED)

        rows = []
        for j, (d, lab) in enumerate(((D128, "octopus facts · 128 tokens"), (D256, "Owl's seasons answer · 256 tokens"))):
            y = -0.35 - j * 0.95
            l = text(lab, 22, INK).move_to([0, y + 0.33, 0]).align_to([-6.2, 0, 0], m.LEFT)
            cells = m.VGroup(*[m.Square(0.36).set_stroke(TOKEN, 1.5).set_fill(GREEN_FILL, 1) for _ in range(d["batch1_runs"])])
            cells.arrange(m.RIGHT, buff=0.08).move_to([0, y - 0.12, 0]).align_to([-6.2, 0, 0], m.LEFT)
            res = code(f"{d['batch1_runs']} runs → {d['batch1_distinct_outputs']} answer", 22, TOKEN, weight=m.BOLD).next_to(cells, m.RIGHT, buff=0.4)
            rows.append((l, cells, res))
        mac_src = cite("measured on this Mac · mlx-lm · Owl-8B 4-bit · batch 1").move_to([0, -2.95, 0]).align_to([-6.2, 0, 0], m.LEFT)
        frame_a = m.VGroup(prompt, t0, mascot, a1, a2, answers, eqs, *[m.VGroup(*r) for r in rows], mac_src)

        big = m.VGroup(stat(f"{TML_RUNS:,}", "runs", "", INK, 80), stat(str(TML_UNIQUE), "different answers", "", RED, 80))
        big.arrange(m.RIGHT, buff=1.4).move_to([0, 1.3, 0])
        big_arrow = arrow(big[0].get_right() + [0.15, 0, 0], big[1].get_left() + [-0.15, 0, 0], MUTED)
        most = text(f"the most common one: only {TML_MOST} times", 24, MUTED).next_to(big[1], m.DOWN, buff=0.25).align_to(big[1], m.LEFT)
        tml_src = cite("Thinking Machines Lab, 2025 · Qwen3-235B · temperature 0").move_to([0, 0.05, 0]).align_to(big, m.LEFT)
        stem = code("…Feynman was born on May 11, 1918, in", 22, INK).move_to([0, -1.25, 0]).align_to([-6.2, 0, 0], m.LEFT)
        b1 = m.VGroup(code(" Queens, New York", 22, TOKEN, weight=m.BOLD), code(f"{TML_QUEENS} runs", 20, MUTED)).arrange(m.RIGHT, buff=0.35)
        b2 = m.VGroup(code(" New York City", 22, RED, weight=m.BOLD), code(f"{TML_NYC} runs", 20, MUTED)).arrange(m.RIGHT, buff=0.35)
        bx = stem.get_right()[0] + 0.7
        b1.move_to([0, -0.75, 0]).align_to([bx, 0, 0], m.LEFT)
        b2.move_to([0, -1.75, 0]).align_to([bx, 0, 0], m.LEFT)
        fork = m.VGroup(m.Line(stem.get_right() + [0.1, 0, 0], b1.get_left() + [-0.1, 0, 0]),
                        m.Line(stem.get_right() + [0.1, 0, 0], b2.get_left() + [-0.1, 0, 0])).set_stroke(MUTED, 2.5)
        same_lab = code(f"identical for {TML_SAME} tokens, then token {TML_SAME + 1}:", 20, MUTED).move_to([0, -0.6, 0]).align_to(stem, m.LEFT)

        self.say(
            "[Now the third question.] [At temperature zero, Owl always takes the top score], [so one prompt should give one answer.]",
            m.AnimationGroup(self.new_header("06 · temperature zero", "Same prompt, same answer?"), m.FadeIn(prompt, shift=0.1 * m.RIGHT)),
            m.AnimationGroup(m.FadeIn(t0), m.GrowArrow(a1), m.FadeIn(mascot, shift=0.1 * m.UP)),
            m.AnimationGroup(m.GrowArrow(a2), m.LaggedStart(*[m.FadeIn(a, shift=0.1 * m.RIGHT) for a in answers], lag_ratio=0.3), m.FadeIn(eqs)),
        )
        self.say(
            "[On this Mac, at batch one, it does]: [twenty runs, one answer], [for a short prompt and a long one.]",
            m.AnimationGroup(m.FadeIn(mac_src), m.FadeIn(rows[0][0]), m.FadeIn(rows[1][0])),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c, scale=0.5) for c in rows[0][1]], lag_ratio=0.05), m.FadeIn(rows[0][2])),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c, scale=0.5) for c in rows[1][1]], lag_ratio=0.05), m.FadeIn(rows[1][2])),
        )
        self.say(
            "[But one lab sent a single prompt a thousand times, at temperature zero], [and got eighty different answers.]",
            m.AnimationGroup(m.FadeOut(frame_a, shift=0.2 * m.UP), m.FadeIn(big[0], shift=0.1 * m.UP), m.FadeIn(tml_src)),
            m.AnimationGroup(m.GrowArrow(big_arrow), m.FadeIn(big[1], scale=0.9), m.FadeIn(most, shift=0.1 * m.UP)),
        )
        self.say(
            "[All agreed for a hundred and two tokens], [then Feynman was born in Queens, New York], [or, eight times, New York City.]",
            m.AnimationGroup(m.FadeIn(stem, shift=0.1 * m.RIGHT), m.FadeIn(same_lab)),
            m.AnimationGroup(m.Create(fork[0]), m.FadeIn(b1, shift=0.1 * m.RIGHT)),
            m.AnimationGroup(m.Create(fork[1]), m.FadeIn(b2, shift=0.1 * m.RIGHT)),
        )
        self.wait(0.15)
        self.clear()

    # ------------------------------------------------------------------ 7 · the order of a sum

    def sum_order(self):
        self.next_section("7 · The order of a sum", notes=(
            f"Python floats: (0.1 + 1e20) − 1e20 = {LOOSE!r}, 0.1 + (1e20 − 1e20) = {TIGHT!r} (the Thinking Machines blog's "
            f"example). One dot product of Owl-8B (d_model {N_DOT:,}): {N_DOT:,} float32 products (NumPy, seed 1), summed in "
            f"one sequential run = {S_RUN} vs in four pieces of {N_DOT // 4:,} then added = {S_SPLIT} (the four-piece split stands "
            "for split-K; real kernels choose their own tile sizes). Kernels split a "
            "reduction to keep the GPU busy when the batch is small (split-K matmul; split-KV attention = Flash-Decoding, "
            "episode 16) and not when it is big, so the order depends on the batch size (Thinking Machines, 2025). Episode "
            "10: continuous batching re-forms the batch every step. The batch sizes in the strip and the near-tie scores "
            "(' turn' vs ' position') are illustrative; the measured fact (§12) is that the text changed at token 221."))
        ex = m.VGroup(
            m.VGroup(code("(0.1 + 1e20) - 1e20", 28, INK), code(f"= {LOOSE!r}", 28, RED, weight=m.BOLD)).arrange(m.RIGHT, buff=0.5),
            m.VGroup(code("0.1 + (1e20 - 1e20)", 28, INK), code(f"= {TIGHT!r}", 28, TOKEN, weight=m.BOLD)).arrange(m.RIGHT, buff=0.5),
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.35).move_to([0, 0.6, 0])
        ex_cap = text("same three numbers, different order", 24, MUTED).next_to(ex, m.DOWN, buff=0.5)
        ex_src = cite("Python floats · Thinking Machines Lab, 2025").next_to(ex_cap, m.DOWN, buff=0.2)

        dot_title = text(f"one dot product in Owl-8B: {N_DOT:,} products", 24, INK, weight=m.BOLD).move_to([0, 2.15, 0]).align_to([-6.2, 0, 0], m.LEFT)
        size, gap, xl = 0.28, 0.05, -3.9
        ya, yb = 1.15, -1.3
        row_a = cells_row(16, size, gap, PROD, PROD_FILL, ya, xl, groups=4, group_gap=0.25)
        row_b = cells_row(16, size, gap, PROD, PROD_FILL, yb, xl)
        lab_a = m.VGroup(code("batch 1", 22, REQUEST, weight=m.BOLD), text("4 pieces at once", 20, MUTED)).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.08)
        lab_a.move_to([0, ya, 0]).align_to([-6.2, 0, 0], m.LEFT)
        lab_b = m.VGroup(code("batch 16", 22, REQUEST, weight=m.BOLD), text("one long run", 20, MUTED)).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.08)
        lab_b.move_to([0, yb, 0]).align_to([-6.2, 0, 0], m.LEFT)
        groups = [m.VGroup(*row_a[4 * k:4 * k + 4]) for k in range(4)]
        braces = m.VGroup(*[brace_under(g) for g in groups])
        partials = m.VGroup(*[m.Square(0.3).set_stroke(PROD, 2).set_fill(AMBER, 0.85).move_to([g.get_x(), ya - 0.75, 0]) for g in groups])
        p_line = m.Line(partials[0].get_center(), partials[-1].get_center()).set_stroke(PROD, 2.5).set_z_index(-1)
        res_x = 3.3
        res_a = code(S_SPLIT, 28, INK, weight=m.BOLD).move_to([0, ya - 0.75, 0]).align_to([res_x, 0, 0], m.LEFT)
        a_res = arrow(partials[-1].get_right(), res_a.get_left(), PROD, buff=0.15)
        sweep = m.Line(row_b[0].get_left() + [0, -0.32, 0], row_b[-1].get_right() + [0, -0.32, 0])
        run_arrow = m.Arrow(sweep.get_start(), sweep.get_end(), buff=0, color=PROD, stroke_width=4, max_tip_length_to_length_ratio=0.05)
        res_b = code(S_RUN, 28, INK, weight=m.BOLD).move_to([0, yb, 0]).align_to([res_x, 0, 0], m.LEFT)
        a_res_b = arrow(row_b[-1].get_right(), res_b.get_left(), PROD, buff=0.15)
        red_a = code(S_SPLIT[DIFF_AT:], 28, RED, weight=m.BOLD).move_to(res_a[DIFF_AT:]).align_to(res_a[DIFF_AT:], m.LEFT)
        red_b = code(S_RUN[DIFF_AT:], 28, RED, weight=m.BOLD).move_to(res_b[DIFF_AT:]).align_to(res_b[DIFF_AT:], m.LEFT)
        f32 = cite(f"float32 · {N_DOT:,} real products · each square = {N_DOT // 16} of them").move_to([0, -2.4, 0]).align_to([-6.2, 0, 0], m.LEFT)

        ep16 = m.VGroup(pill("episode 16", MEMORY, CYAN_FILL, 20, mono=True), text("Flash-Decoding splits the KV cache the same way", 24, INK))
        ep16.arrange(m.RIGHT, buff=0.35).move_to([0, 1.9, 0]).align_to([-6.2, 0, 0], m.LEFT)
        ep10 = m.VGroup(pill("episode 10", REQUEST, PINK_FILL, 20, mono=True), text("continuous batching: the batch changes every step", 24, INK))
        ep10.arrange(m.RIGHT, buff=0.35).move_to([0, 1.1, 0]).align_to([-6.2, 0, 0], m.LEFT)
        slots = m.VGroup(*[m.Square(0.42).set_stroke(EDGE, 1.5).set_fill(CARD, 1) for _ in range(16)]).arrange(m.RIGHT, buff=0.08)
        slots.move_to([0, 0.15, 0]).align_to([-6.2, 0, 0], m.LEFT)
        sizes = [3, 9, 14, 6, 11]
        size_labs = [code(f"batch this step: {s}", 22, REQUEST, weight=m.BOLD).next_to(slots, m.RIGHT, buff=0.4) for s in sizes]

        def fill_to(k):
            return m.AnimationGroup(*[s.animate.set_fill(PINK_FILL if i < k else CARD, 1).set_stroke(REQUEST if i < k else EDGE, 1.5)
                                      for i, s in enumerate(slots)], run_time=0.45)

        steps = [m.AnimationGroup(fill_to(sizes[0]), m.FadeIn(size_labs[0]))]
        for a, b in zip(range(len(sizes) - 1), range(1, len(sizes))):
            steps.append(m.AnimationGroup(fill_to(sizes[b]), swap(size_labs[a], size_labs[b])))

        ny = [-1.35, -2.05]
        near_head = text("two tokens nearly tied · scores for example", 20, MUTED).move_to([0, -0.75, 0]).align_to([-6.2, 0, 0], m.LEFT)
        near = HBars([" turn", " position"], NEAR[1], x0=-3.6, y_top=ny[0], dy=ny[0] - ny[1], scale=0.2, fmt=lambda v: f"{v:.3f}",
                     size=24, height=0.42, val_size=22)
        near.colors[0] = PICKED
        near.bars[0].set_stroke(TOKEN).set_fill(GREEN_FILL, 1)
        near.vals[0].set_color(TOKEN)
        nb = {1: code("batch 1", 22, REQUEST, weight=m.BOLD), 4: code("batch 4", 22, REQUEST, weight=m.BOLD)}
        for v in nb.values():
            v.move_to([0, (ny[0] + ny[1]) / 2, 0]).align_to([3.3, 0, 0], m.LEFT)

        self.say(
            "[The cause is floating-point addition]: [change the order of a sum], [and the answer can change.]",
            m.AnimationGroup(self.new_header("07 · batch-dependent sums", "The order of a sum"), m.FadeIn(ex[0][0]), m.FadeIn(ex[1][0])),
            m.AnimationGroup(m.FadeIn(ex[0][1], shift=0.1 * m.LEFT), m.FadeIn(ex_cap)),
            m.AnimationGroup(m.FadeIn(ex[1][1], shift=0.1 * m.LEFT), m.FadeIn(ex_src)),
        )
        self.say(
            "[A GPU sums a dot product in pieces], [cut by batch size]: [four pieces for one request], [one long run for a full batch.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(ex, ex_cap, ex_src), shift=0.2 * m.UP), m.FadeIn(dot_title),
                             m.LaggedStart(*[m.FadeIn(c, scale=0.6) for c in row_a], lag_ratio=0.03), m.FadeIn(f32)),
            m.AnimationGroup(m.FadeIn(row_b), m.FadeIn(lab_a), m.FadeIn(lab_b)),
            m.AnimationGroup(m.Create(braces), m.FadeIn(partials, shift=0.15 * m.DOWN), m.Create(p_line),
                             m.GrowArrow(a_res), m.FadeIn(res_a)),
            m.AnimationGroup(m.GrowArrow(run_arrow, run_time=1.2), m.GrowArrow(a_res_b), m.FadeIn(res_b)),
        )
        self.say(
            "[Same numbers, different order]: [the two sums differ in the last digit.]",
            m.AnimationGroup(m.Circumscribe(res_a, color=INK, buff=0.08), m.Circumscribe(res_b, color=INK, buff=0.08)),
            m.AnimationGroup(m.FadeOut(res_a[DIFF_AT:]), m.FadeIn(red_a), m.FadeOut(res_b[DIFF_AT:]), m.FadeIn(red_b)),
        )
        for whole in (res_a, res_b):              # gotcha 22: drop the faded digits so a later fade can't revive them
            whole.remove(*list(whole[DIFF_AT:]))
        sums = m.VGroup(dot_title, row_a, row_b, lab_a, lab_b, braces, partials, p_line, a_res, res_a, red_a,
                        run_arrow, a_res_b, res_b, red_b, f32)
        self.say(
            "[Episode sixteen's Flash-Decoding splits the KV cache the same way], [and with continuous batching, from episode ten], [the batch changes every step.]",
            m.AnimationGroup(m.FadeOut(sums, shift=0.2 * m.UP), m.FadeIn(ep16, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(ep10, shift=0.1 * m.UP), m.FadeIn(slots)),
            m.Succession(*steps),
        )
        self.say(
            "[Usually that changes nothing.] [But when two tokens are nearly tied], [it can flip the winner], [and the answer takes another path.]",
            m.Wait(0.2),
            m.AnimationGroup(m.FadeIn(near_head), m.FadeIn(nb[1]),
                             *[m.AnimationGroup(m.FadeIn(near.labels[i]), m.GrowFromEdge(near.bars[i], m.LEFT), m.FadeIn(near.vals[i])) for i in range(2)]),
            m.AnimationGroup(swap(nb[1], nb[4]), near.to(NEAR[4], colors={0: (BAR, BAR_FILL), 1: PICKED})),
            m.Circumscribe(near.row(1), color=TOKEN, buff=0.08),
        )
        self.wait(0.15)
        self.clear()

    # ------------------------------------------------------------------ 8 · measured here

    def measured_here(self):
        self.next_section("8 · Measured here", notes=(
            "research/measurements.md §12 (this Mac, mlx-lm 0.32.2, Owl-8B 4-bit, greedy): each prompt as member 0 of a "
            "BatchGenerator batch of B = 1, 2, 4, 8, 16 (other members: different short questions), compared with the "
            f"batch-1 stream. 128 tokens (octopus facts): identical at every B. 256 tokens (Owl's seasons answer): identical "
            f"at B = 2; first divergence at token {DIV256[2]} for B = 4 and 8, {DIV256[4]} for B = 16. The texts are the "
            "raw file's ref_text_after / batch_text_after (raw/determinism__llama-3.1-8b__4bit__chat256.json), with the start "
            "of the sentence from its reference_text. §9: the 4-bit speculative run (k = 2…6) on the same chat prompt "
            f"differs from plain greedy from token {SPEC_DIV}. mlx_lm.server, 20 requests of the octopus prompt alone and 20 "
            "more while 15 other clients kept the batch full: 1 distinct output each, 20 of 20 identical to the solo answer."))
        cols = [-0.55, 0.95, 2.45, 3.95, 5.45]
        y_h, y1, y2, y3 = 2.05, 1.4, 0.75, 0.1
        head = m.VGroup(*[code(f"B = {b}", 22, REQUEST, weight=m.BOLD).move_to([x, y_h, 0]) for x, b in zip(cols, BATCHES)])
        head_l = text("same prompt inside a batch of", 20, MUTED).move_to([0, y_h, 0]).align_to([-6.2, 0, 0], m.LEFT)

        def cell(div, x, y):
            ok = div is None
            c = chip("same" if ok else f"@ {div}", *(PICKED if ok else MASKED), width=1.3, height=0.52, size=26, mono=True)
            return c.move_to([x, y, 0])

        r1l = text("octopus facts · 128 tokens", 22, INK).move_to([0, y1, 0]).align_to([-6.2, 0, 0], m.LEFT)
        r2l = text("Owl's seasons · 256 tokens", 22, INK).move_to([0, y2, 0]).align_to([-6.2, 0, 0], m.LEFT)
        r1 = m.VGroup(*[cell(d, x, y1) for d, x in zip(DIV128, cols)])
        r2 = m.VGroup(*[cell(d, x, y2) for d, x in zip(DIV256, cols)])
        r3l = text("mlx_lm.server, 15 other clients", 22, INK).move_to([0, y3, 0]).align_to([-6.2, 0, 0], m.LEFT)
        r3 = chip(f"{BUSY['distinct_outputs']} answer in {BUSY['requests']} · same as alone", *PICKED, width=7.4, height=0.52, size=26, mono=True)
        r3.move_to([(cols[0] + cols[-1]) / 2, y3, 0])
        src = cite("measured on this Mac · mlx-lm · Owl-8B 4-bit · greedy").move_to([0, -3.38, 0]).align_to([-6.2, 0, 0], m.LEFT)

        lamp = text(LAMP, 24, INK).move_to([0, -0.75, 0]).align_to([-6.2, 0, 0], m.LEFT)
        ref_rest = REF_AFTER[len(" your room. If you "):].split(",")[0]
        b4_rest = B4_AFTER[len(" your room. If you "):].split(",")[0]
        br1 = m.VGroup(code("batch 1", 20, REQUEST, weight=m.BOLD), text("… " + ref_rest, 24, TOKEN, weight=m.BOLD)).arrange(m.RIGHT, buff=0.3)
        br2 = m.VGroup(code("batch 4", 20, REQUEST, weight=m.BOLD), text("… " + b4_rest, 24, RED, weight=m.BOLD)).arrange(m.RIGHT, buff=0.3)
        br1.move_to([0, -1.4, 0]).align_to([-5.4, 0, 0], m.LEFT)
        br2.move_to([0, -2.0, 0]).align_to([-5.4, 0, 0], m.LEFT)
        tok_lab = code(f"token {DIV256[2]}", 22, RED, weight=m.BOLD).move_to([0, -1.7, 0]).align_to([6.3, 0, 0], m.RIGHT)
        ep13 = m.VGroup(pill("episode 13", TOKEN, GREEN_FILL, 20, mono=True),
                        text(f"4-bit speculative decoding left the plain answer at token {SPEC_DIV} too", 22, INK))
        ep13.arrange(m.RIGHT, buff=0.3).move_to([0, -2.72, 0]).align_to([-6.2, 0, 0], m.LEFT)

        self.say(
            "[We tested it on this Mac]: [each prompt alone, then inside batches of two, four, eight and sixteen.]",
            m.AnimationGroup(self.new_header("08 · measured here", "Token 221"), m.FadeIn(src)),
            m.AnimationGroup(m.FadeIn(head_l), m.LaggedStart(*[m.FadeIn(h, shift=0.1 * m.DOWN) for h in head], lag_ratio=0.2),
                             m.FadeIn(r1l), m.FadeIn(r2l)),
        )
        self.say(
            "[The short answer never changed, even through a busy server.] [The long one changed at batch four, eight and sixteen.]",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c, scale=0.8) for c in r1], lag_ratio=0.15), m.FadeIn(r3l, shift=0.1 * m.RIGHT),
                             m.FadeIn(r3, scale=0.9)),
            m.LaggedStart(*[m.FadeIn(c, scale=0.8) for c in r2], lag_ratio=0.25),
        )
        self.say(
            "[At token two hundred and twenty-one], [a lamp went from turn it towards you], [to position it directly above you.]",
            m.AnimationGroup(m.FadeIn(lamp, shift=0.1 * m.UP), m.FadeIn(tok_lab), m.Circumscribe(r2[2], color=RED, buff=0.06)),
            m.FadeIn(br1, shift=0.1 * m.RIGHT),
            m.FadeIn(br2, shift=0.1 * m.RIGHT),
        )
        self.say(
            "[Episode thirteen's speculative decoding, in four bits, drifted at this same token]: [one fragile near-tie.] [Rare, and late, but real.]",
            m.FadeIn(ep13, shift=0.1 * m.UP),
            m.Circumscribe(tok_lab, color=RED, buff=0.08),
            m.Circumscribe(r2[2:], color=RED, buff=0.08),
        )
        self.wait(0.15)
        self.clear()

    # ------------------------------------------------------------------ 9 · the fix

    def the_fix(self):
        self.next_section("9 · The fix", notes=(
            "Thinking Machines Lab (2025; notes §12): 'the reduction order for each element must be fixed regardless of the "
            "batch-size of the kernel' (batch invariance) — e.g. a fixed split size instead of a fixed number of splits. With "
            f"batch-invariant RMSNorm, matmul and attention, all {TML_RUNS:,} completions were identical. Cost (Qwen3-8B, one "
            f"GPU, vLLM, 1,000 sequences): default {T_DEFAULT} s, unoptimised deterministic {T_INVARIANT} s, with an improved "
            f"attention kernel {T_BETTER} s. The two rows here sum the same {N_DOT:,} products in the same four pieces: "
            f"{S_SPLIT} for both."))
        size, gap, xl = 0.28, 0.05, -3.9
        ya, yb = 1.6, 0.55
        row_a = cells_row(16, size, gap, PROD, PROD_FILL, ya, xl, groups=4, group_gap=0.25)
        row_b = cells_row(16, size, gap, PROD, PROD_FILL, yb, xl, groups=4, group_gap=0.25)
        lab_a = code("batch 1", 22, REQUEST, weight=m.BOLD).move_to([0, ya, 0]).align_to([-6.2, 0, 0], m.LEFT)
        lab_b = code("batch 16", 22, REQUEST, weight=m.BOLD).move_to([0, yb, 0]).align_to([-6.2, 0, 0], m.LEFT)
        res_a = code(S_SPLIT, 28, TOKEN, weight=m.BOLD).move_to([0, ya, 0]).align_to([3.3, 0, 0], m.LEFT)
        res_b = code(S_SPLIT, 28, TOKEN, weight=m.BOLD).move_to([0, yb, 0]).align_to([3.3, 0, 0], m.LEFT)
        a_a = arrow(row_a.get_right(), res_a.get_left(), PROD, buff=0.15)
        a_b = arrow(row_b.get_right(), res_b.get_left(), PROD, buff=0.15)
        same = text("same pieces, same order, whatever the batch", 22, MUTED).move_to([0, -0.15, 0]).align_to(row_a, m.LEFT)

        all_same = stat(f"{TML_RUNS:,} of {TML_RUNS:,}", "identical", "", TOKEN, 48).move_to([0, -1.65, 0]).align_to([-6.2, 0, 0], m.LEFT)
        x0, k = 2.6, 0.055
        bars = []
        for i, (lab, s, color, fill) in enumerate((("default", T_DEFAULT, MUTED, CARD_HI), ("batch-invariant", T_INVARIANT, AMBER, AMBER_FILL),
                                                    ("+ better attention", T_BETTER, AMBER, AMBER_FILL))):
            y = -1.05 - i * 0.6
            l = text(lab, 20, INK).move_to([0, y, 0]).align_to([x0 - 0.2, 0, 0], m.RIGHT)
            b = m.Rectangle(width=k * s, height=0.38).set_stroke(color, 2).set_fill(fill, 1).move_to([x0 + k * s / 2, y, 0])
            v = code(f"{s} s", 20, color).next_to(b, m.RIGHT, buff=0.15)
            bars.append(m.VGroup(l, b, v))
        tml = cite("Thinking Machines Lab, 2025 · Qwen3-8B · vLLM · 1,000 sequences").move_to([0, -3.2, 0]).align_to([-6.2, 0, 0], m.LEFT)

        self.say(
            "[The fix is batch-invariant kernels]: [every sum cut the same way, in the same order], [whatever the batch.]",
            m.AnimationGroup(self.new_header("09 · the fix", "Batch-invariant kernels"), m.FadeIn(lab_a), m.FadeIn(lab_b)),
            m.AnimationGroup(m.FadeIn(row_a), m.FadeIn(row_b), m.FadeIn(same)),
            m.AnimationGroup(m.GrowArrow(a_a), m.GrowArrow(a_b), m.FadeIn(res_a), m.FadeIn(res_b)),
        )
        self.say(
            "[With them, all thousand runs matched.] [The price]: [fifty-five seconds instead of twenty-six], [or forty-two with a better attention kernel.]",
            m.AnimationGroup(m.FadeIn(all_same, scale=0.9), m.FadeIn(tml)),
            m.AnimationGroup(m.FadeIn(bars[0][0]), m.GrowFromEdge(bars[0][1], m.LEFT), m.FadeIn(bars[0][2])),
            m.AnimationGroup(m.FadeIn(bars[1][0]), m.GrowFromEdge(bars[1][1], m.LEFT), m.FadeIn(bars[1][2])),
            m.AnimationGroup(m.FadeIn(bars[2][0]), m.GrowFromEdge(bars[2][1], m.LEFT), m.FadeIn(bars[2][2])),
        )
        self.wait(0.15)
        self.clear()

    # ------------------------------------------------------------------ 10 · recap

    def recap(self):
        self.next_section("10 · Recap", notes=(
            "Shape (temperature, top-k, top-p: owl_reference.sample), mask (a grammar sets forbidden tokens to −∞, then "
            "renormalise), pick (one roll, or the argmax at temperature 0). Same pick twice needs the same order of sums: "
            "batch-invariant kernels. Next: episode 18, fewer bits per weight."))
        halo = glow(TOKEN, radius=3.4).move_to([0, 0.7, 0])
        words = m.VGroup(title_text("Shape", 76, MUTED), title_text("Mask", 76, INK), title_text("Pick", 76, TOKEN))
        words.arrange(m.RIGHT, buff=1.3).move_to([0, 0.8, 0])
        on_baseline(words)
        a1 = arrow(words[0].get_right(), words[1].get_left(), MUTED, buff=0.25)
        a2 = arrow(words[1].get_right(), words[2].get_left(), MUTED, buff=0.25)
        subs = m.VGroup(code("÷ T · top-p", 22, MUTED), code("−∞ · renormalise", 22, INK), code("one roll", 22, TOKEN))
        y_base = words[1].get_bottom()[1] - 0.55
        for s, w, k in zip(subs, words, (1, 3, 0)):    # a glyph that sits on the baseline: T, r, o
            s.move_to([w.get_x(), 0, 0]).shift((y_base - s[k].get_bottom()[1]) * m.UP)
        line = text("the same pick twice needs the same order of sums", 26, INK).move_to([0, -1.6, 0])
        mascot = owl(0.75).move_to([0, -2.85, 0])

        self.say(
            "[To sum up]: [shape], [mask], [pick.]",
            m.AnimationGroup(self.new_header("10 · recap", "Shape, mask, pick"), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.2 * m.UP),
            m.AnimationGroup(m.GrowArrow(a1), m.FadeIn(words[1], shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowArrow(a2), m.FadeIn(words[2], shift=0.2 * m.UP)),
        )
        self.say(
            "[Temperature and top-p shape the odds], [a grammar masks what the schema forbids], [and one token is picked.]",
            m.FadeIn(subs[0], shift=0.1 * m.UP),
            m.FadeIn(subs[1], shift=0.1 * m.UP),
            m.FadeIn(subs[2], shift=0.1 * m.UP),
        )
        self.say(
            "[And to pick the same token twice], [add up in the same order.]",
            m.FadeIn(line, shift=0.1 * m.UP),
            m.FadeIn(mascot, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()
