"""Tokens per Second · Episode 14 · Drafting Without a Draft Model (Module 4 · Faster Tokens)

The model can draft for itself. Episode 13 had a small model (Owl-mini) draft γ tokens and Owl verify them in one
pass; here the drafts come from Owl itself: Medusa heads sprouting from its last layer (a tree of candidates), EAGLE
drafting from the feature Owl hands to its output head (EAGLE-2's dynamic tree), prompt lookup copying spans from the
prompt (the running example: Owl quoting the student's notes back), and multi-token prediction built in during
training (Meta; DeepSeek-V3). Then why the gains fade at high batch (the roofline, and EAGLE-3's SGLang batch table).
Owl-8B = Llama-3.1-8B: 8 B parameters, 16 GB of BF16 weights, 128 KiB of KV cache per token.

Sections: hook (Episode 13's draft model is a second model: its own weights and cache; measured on this Mac, an
Owl-mini step is 2.05 ms = a fifth of a 4-bit Owl step) · Medusa (heads sprout, top-2 × top-3 = 6 continuations, tree
attention, longest accepted branch: 3 tokens in one pass; Medusa-1 > 2.2×) · EAGLE (one draft layer on features,
Owl's own head, autoregressive; EAGLE-2 grows the tree where confident; 2.7–3.5×, EAGLE-3 up to 6.5×) · prompt lookup
(the last 3 tokens found in the notes, 10 copied, 8 accepted + Owl's own token = 9 per pass; the code; measured on
this Mac in Episode 13's copy-the-notes run: α = 1.00, 30.7 → 148 tok/s = 4.82×, above the 34 tok/s one-token
ceiling; prompt lookup itself NOT measured here, Saxena reports 2–4×) · multi-token prediction (Meta 4 heads, up to
3×; DeepSeek-V3 one MTP module, 85–90% acceptance → E = 1 + α = 1.85–1.90 tokens per pass, reported 1.8× TPS) · busy
servers (tokens per pass = batch × 4 on the H100 roofline: 4 → 256 → 512, past the 295 ridge the pass gets longer;
EAGLE-3 Table: EAGLE 1.40× → below 1× from batch 24, EAGLE-3 1.81× → 1.38× at 64) · which drafter · recap.

Sources (facts dated October 2026; abstracts and tables re-read 2026-10-24):
- Medusa (Cai, Li, Geng, Peng, Lee, Chen, Dao; ICML 2024, arXiv 2401.10774): "adding extra decoding heads"; each head
  one feed-forward layer with a residual connection on the last hidden state, head k predicts the token k+1 ahead;
  top-2 of head 1 × top-3 of head 2 = 6 candidates verified with tree attention; "longest accepted prefix";
  "Medusa-1 can achieve over 2.2x speedup without compromising generation quality" (frozen backbone, lossless);
  Medusa-2 2.3–3.6× (fine-tunes the backbone; typical acceptance is not distribution-exact) (notes §6).
- EAGLE (Li, Wei, Zhang, Zhang; ICML 2024, arXiv 2401.15077): "Autoregression at the feature (second-to-top-layer)
  level is more straightforward than at the token level"; the draft also gets the token sequence advanced by one
  step; LLaMA2-Chat 70B 2.7–3.5× latency speedup, distribution preserved. EAGLE-2 (arXiv 2406.16858): dynamic draft
  tree, 3.05–4.26×. EAGLE-3 (arXiv 2503.01840, 2025): "abandons feature prediction in favor of direct token
  prediction", fuses low/mid/high-layer features, draft = one Transformer decoder layer, "up to 6.5x"; SGLang
  throughput table (H100, Llama-3.1-8B-Instruct, MT-Bench, "did not use the tree structure", chain length 3) at batch
  2/4/8/16/24/32/48/56/64: EAGLE 1.40/1.38/1.23/1.02/0.93/0.94/0.88/0.99/0.99×, EAGLE-3 1.81/1.82/1.62/1.48/1.39/
  1.32/1.38/1.34/1.38× (the primary table has batch 56, which notes §6 omits).
- Prompt-lookup decoding (A. Saxena, GitHub, Nov 2023): "replace the draft model with simple string matching in the
  prompt"; max_ngram_size = 3, num_pred_tokens = 10; "significant speedups (2x-4x) in input-grounded tasks"
  (summarization, context QA ~2.4× on average; one A100 40GB, Mistral-7B-Instruct, greedy). HF transformers:
  prompt_lookup_num_tokens=10; vLLM: n-gram speculator (notes §6, §13). The scene's lookup_draft() is a simplified
  excerpt (one n-gram size, first match), run on the scene's own tokens and asserted.
- Multi-token prediction: Gloeckle et al. (Meta, ICML 2024, arXiv 2404.19737): "predict the following n tokens using n
  independent output heads, operating on top of a shared model trunk"; "models trained with 4-token prediction are
  up to 3 times faster at inference". DeepSeek-V3 (arXiv 2412.19437): depth D = 1 (one MTP module: shared embedding
  and output head, one Transformer block, a projection); "the acceptance rate of the second token prediction ranges
  between 85% and 90%", "delivering 1.8 times TPS"; E = (1 − α^(γ+1))/(1 − α) at γ = 1 is 1 + α = 1.85–1.90.
- research/measurements.md (this Mac Studio, M4 Max 546 GB/s, mlx-lm 0.32.2, 2026-10-24), via measured(): §1 decode
  ms per token (Llama-3.2-1B 4-bit 2.05 ms, Llama-3.1-8B 4-bit 9.81 ms, BF16 30.7–31.0 tok/s); §9 speculative
  decoding, copy_notes prompt ("Copy out the following study notes exactly…", KV-cache notes), 8B BF16 target with
  Llama-3.2-1B-4bit drafting k = γ = 6: acceptance 1.00, 6.92 tokens per target pass, 30.7 → 148.0 tok/s (4.82×),
  output identical to greedy. Ceiling 546 GB/s ÷ 16.06 GB = 34.0 tok/s (decode_ceiling). Prompt lookup itself was
  NOT measured on this Mac (said on screen and in the narration).
- H100 SXM: 3.35 TB/s, 989 TFLOPS BF16 dense, ridge ≈ 295 FLOPs/byte (NVIDIA, dated Oct 2026; owl_reference).
  A BF16 decode pass of n tokens does 2·P·n FLOPs over 2·P bytes of weights: intensity ≈ n (KV reads ignored).

Illustrative (said "for example" or schematic): the words-as-tokens (real tokenizers split differently), the head
picks and tree shapes, the EAGLE-2 confidences, Owl's own continuation that rejects "and appends", the layer drawings
and feature bars. The student's notes are a sentence in the spirit of the measured copy-the-notes prompt.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import inspect
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from owl_reference import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- the numbers

PREFIX = "ep14"
W_BYTES = weight_bytes()
CEIL_MAC = decode_ceiling(hw=MAC)
assert round(W_BYTES / 1e9) == 16 and round(CEIL_MAC) == 34  # "the one-token ceiling of thirty-four"

# hook: the draft model is a second model with its own steps (measurements.md §1)
MINI_MS = measured("decode", "llama-3.2-1b", "4bit", "ms_per_token")
OWL4_MS = measured("decode", "llama-3.1-8b", "4bit", "ms_per_token")
assert round(MINI_MS, 2) == 2.05 and round(OWL4_MS, 2) == 9.81
assert round(MINI_MS / OWL4_MS, 1) == 0.2  # "a fifth of a four-bit Owl step"

# Medusa: top picks per head (illustrative), the tree they make
NEXT = "keys"
PICKS = {1: ["and", "for"], 2: ["values", "the", "queries"]}
PATHS = len(PICKS[1]) * len(PICKS[2])
TREE_TOKENS = 1 + len(PICKS[1]) + PATHS
assert PATHS == 6 and TREE_TOKENS == 9  # "two picks times three, six possible continuations"
MEDUSA_BRANCH = [NEXT, PICKS[1][0], PICKS[2][0]]
assert len(MEDUSA_BRANCH) == 3  # "three tokens from one pass"
MEDUSA1 = 2.2  # "Medusa-1 can achieve over 2.2x speedup without compromising generation quality"

# EAGLE: a dynamic tree (illustrative confidences): deep where sure, short where not
EAGLE_MAIN = [("and", 0.82), ("values", 0.74), ("only", 0.68)]
EAGLE_ALT = [("for", 0.11), ("the", 0.15), ("that", 0.12)]
EAGLE_X = (2.7, 3.5)  # LLaMA2-Chat 70B, latency speedup
EAGLE2_X = (3.05, 4.26)
EAGLE3_MAX = 6.5
assert round(sum(EAGLE_X) / 2) == 3  # "about three times faster"

# prompt lookup on the running example (words stand in for tokens)
NOTES = "With a cache each step computes the keys and values only for the newest token and appends them to a store in memory".split()
ANSWER = "Because each step computes the".split()
OWL_NEXT = "keys and values only for the newest token instead".split()  # what Owl itself would write (illustrative)
NGRAM, K = 3, 10  # Saxena's defaults: max_ngram_size = 3, num_pred_tokens = 10


def lookup_draft(seq, n=3, k=10):
    tail = seq[-n:]
    for i in range(len(seq) - n):
        if seq[i:i + n] == tail:
            return seq[i + n:i + n + k]
    return []  # no match


LOOKUP_SRC = inspect.getsource(lookup_draft).strip()  # the card shows the code that runs

SEQ = NOTES + ANSWER
MATCH = next(i for i in range(len(SEQ) - NGRAM) if SEQ[i:i + NGRAM] == SEQ[-NGRAM:])
DRAFT = lookup_draft(SEQ, NGRAM, K)
ACCEPTED = next((i for i, (a, b) in enumerate(zip(DRAFT, OWL_NEXT)) if a != b), len(DRAFT))
BONUS = OWL_NEXT[ACCEPTED]
PER_PASS = ACCEPTED + 1
assert NOTES[MATCH:MATCH + NGRAM] == ["step", "computes", "the"] and MATCH == 4
assert DRAFT == "keys and values only for the newest token and appends".split() and len(DRAFT) == K  # "the ten tokens"
assert ACCEPTED == 8 and BONUS == "instead" and PER_PASS == 9  # "eight accepted … nine tokens for one read"

# measured on this Mac (measurements.md §9): Owl copying notes, Owl-mini drafting γ = 6, BF16 target
SPEC = measured("speculative", "llama-3.1-8b__bf16", "prompts", "copy_notes")
GAMMA = 6
RUN = SPEC["by_num_draft_tokens"][str(GAMMA)]
BASE = SPEC["baseline_tok_s"]
FAST, ALPHA, PER_TARGET, SPEEDUP = RUN["tok_s_median"], RUN["acceptance_rate"], RUN["tokens_per_target_pass"], RUN["speedup"]
assert SPEC["best_k"] == GAMMA and ALPHA == 1.0 and RUN["output_identical_to_baseline"]  # "every draft token was accepted"
assert round(BASE) == 31 and round(BASE, 1) == 30.7 and round(FAST) == 148  # "thirty-one … a hundred and forty-eight"
assert round(SPEEDUP, 2) == 4.82 and BASE < CEIL_MAC < FAST  # "far above the one-token ceiling"
assert round(PER_TARGET, 2) == 6.92 and 6.5 < PER_TARGET < 7  # "almost seven tokens"
SAXENA = (2, 4)  # "significant speedups (2x-4x) in input-grounded tasks"

# multi-token prediction
META_HEADS, META_X = 4, 3  # "4-token prediction … up to 3 times faster at inference"
DS_ALPHA = (0.85, 0.90)  # "the acceptance rate of the second token prediction ranges between 85% and 90%"
DS_E = tuple(expected_tokens_per_pass(a, 1) for a in DS_ALPHA)  # Episode 13's formula with γ = 1
DS_TPS = 1.8  # "delivering 1.8 times TPS"
assert tuple(round(e, 2) for e in DS_E) == (1.85, 1.9) and round(DS_E[1], 1) == 1.9  # "about one point nine"
assert all(abs(e - (1 + a)) < 1e-9 for e, a in zip(DS_E, DS_ALPHA))  # E = 1 + α

# busy servers: a pass of n tokens on one H100 (BF16 weights; intensity ≈ tokens per pass; KV reads ignored)
CHAIN = 3  # EAGLE-3's SGLang test drafts a chain of 3: each sequence puts 4 tokens through the pass
PER_SEQ = CHAIN + 1
RIDGE = ridge(GPU)


def pass_ms(tokens):
    return step_time(batch=tokens) * 1e3


assert abs(flops_per_token() / weight_bytes() - 1) < 1e-12  # FLOPs per byte = tokens per pass
assert round(RIDGE) == 295 and round(pass_ms(1), 1) == round(pass_ms(PER_SEQ), 1) == 4.8  # "the pass costs the same"
assert 64 * PER_SEQ == 256 < RIDGE and round(pass_ms(256), 1) == 4.8  # "two hundred and fifty-six … close to the ridge"
assert 128 * PER_SEQ > RIDGE and round(pass_ms(512), 1) == 8.3 and round(pass_ms(512) / pass_ms(128), 1) == 1.7
BATCHES = [2, 4, 8, 16, 24, 32, 48, 56, 64]
EAGLE1_TP = [1.40, 1.38, 1.23, 1.02, 0.93, 0.94, 0.88, 0.99, 0.99]
EAGLE3_TP = [1.81, 1.82, 1.62, 1.48, 1.39, 1.32, 1.38, 1.34, 1.38]
assert EAGLE1_TP[0] == 1.40 and all(x < 1 for x in EAGLE1_TP[BATCHES.index(24):])  # "forty percent … then slower"
assert round(EAGLE3_TP[0], 1) == round(EAGLE3_TP[1], 1) == 1.8 and EAGLE3_TP[-1] == 1.38  # "one point eight … one point three eight"

# ---------------------------------------------------------------------------- look


def cite(s, size=20):
    return text(s, size, MUTED)


def tok(word, color=TOKEN, size=20):
    """A token chip (GREEN = Owl's own tokens, PINK = the prompt)."""
    return token_row([word], color, size)[0]


def draft_tok(word, size=20):
    """A drafted, not yet verified token: grey outline."""
    return token_row([word], MUTED, size, fill=CARD)[0]


def accept(chip):
    return chip[0].animate.set_stroke(TOKEN, 1.8).set_fill(GREEN_FILL, 1)


def reject(chip):
    return chip.animate.set_opacity(0.35)


def mark(ok, size=24):
    return code("✓" if ok else "✗", size, GOOD if ok else BAD, weight=m.BOLD)


def owl_stack(center, w=2.0, h=3.5, n=7, name="Owl-8B", sub="32 layers", name_size=26, pitch=0.28):
    """Owl as a stack of layers (weights BLUE); [0] box, [1] layer bars (top first), [2] label."""
    box = card(w, h, WEIGHTS, BLUE_FILL, radius=0.16, stroke=2.5).move_to([center[0], center[1], 0])
    top = box.get_top()[1]
    bars = m.VGroup(*[
        m.RoundedRectangle(corner_radius=0.05, width=w - 0.5, height=0.15, stroke_width=0,
                           fill_color=WEIGHTS, fill_opacity=0.5).move_to([box.get_x(), top - 0.3 - i * pitch, 0])
        for i in range(n)
    ])
    label = m.VGroup(title_text(name, name_size, INK), code(sub, 20, MUTED)).arrange(m.DOWN, buff=0.08)
    label.move_to([box.get_x(), box.get_bottom()[1] + 0.18 + label.height / 2, 0])
    return m.VGroup(box, bars, label)


def sweep(stack, color=MEMORY, run_time=1.2):
    """One pass through Owl: a flash from the bottom layer to the top (every weight read once)."""
    box = stack[0]
    line = m.Line(box.get_bottom() + [0, 0.12, 0], box.get_top() + [0, -0.12, 0]).set_stroke(color, 16)
    return m.ShowPassingFlash(line, time_width=0.6, run_time=run_time)


def head_box(name, offset, w=1.75, h=0.62, color=WEIGHTS):
    box = card(w, h, color, BLUE_FILL, radius=0.12, stroke=2.2)
    words = m.VGroup(text(name, 20, INK, weight=m.BOLD), code(offset, 20, MUTED)).arrange(m.RIGHT, buff=0.18)
    return m.VGroup(box, words.move_to(box))


def feature(heights, w=0.95, h=0.72, color=INK):
    """A feature vector: a small frame of bars (a hidden state, schematic)."""
    frame = m.RoundedRectangle(corner_radius=0.08, width=w, height=h).set_stroke(EDGE, 2).set_fill(CARD, 1)
    n = len(heights)
    pitch = (w - 0.2) / n
    bars = m.VGroup(*[
        m.Rectangle(width=pitch * 0.62, height=max(0.04, v * (h - 0.22))).set_stroke(width=0).set_fill(color, 0.85)
        .move_to([frame.get_left()[0] + 0.1 + pitch * (i + 0.5), frame.get_bottom()[1] + 0.11 + v * (h - 0.22) / 2, 0])
        for i, v in enumerate(heights)
    ])
    return m.VGroup(frame, bars)


def edge(a, b, color=MUTED, width=2.5):
    return m.Line(a.get_right() + [0.04, 0, 0], b.get_left() + [-0.04, 0, 0]).set_stroke(color, width)


def mac_stamp(lines):
    mac = icon(recolored("mac", MUTED, CARD, PREFIX), 0.5)
    words = m.VGroup(*[cite(s) for s in lines]).arrange(m.DOWN, buff=0.06, aligned_edge=m.LEFT)
    return m.VGroup(mac, words).arrange(m.RIGHT, buff=0.22)


# ---------------------------------------------------------------------------- the episode


class Ep14DraftingWithoutADraftModel(Episode):
    number = 14

    def lesson(self):
        self.hook()
        self.medusa()
        self.eagle()
        self.lookup()
        self.lookup_measured()
        self.mtp()
        self.busy()
        self.choose()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes=f"Episode 13 recap: a draft model guesses γ tokens, Owl verifies them in one pass (accept/reject strip). The draft model is a second model: its own weights and KV cache, its own decode steps, and it must share Owl's tokenizer. Measured on this Mac (measurements.md §1, mlx-lm 0.32.2): an Owl-mini step (Llama-3.2-1B 4-bit) takes {MINI_MS:.2f} ms, a 4-bit Owl-8B step {OWL4_MS:.2f} ms: {MINI_MS / OWL4_MS:.0%} (Episode 13: that is why the same draft helps the BF16 target far more). Today: drafts from Owl itself (Medusa heads, EAGLE features, prompt lookup, multi-token prediction). The four draft tokens and their marks are illustrative.")
        cap = code("Episode 13 · draft, then verify in one pass", 20, MUTED).move_to([-6.2, 2.35, 0], aligned_edge=m.LEFT)
        y = 1.5
        mini = m.VGroup(card(2.0, 1.05, WEIGHTS, BLUE_FILL, radius=0.16, stroke=2.2),
                        m.VGroup(title_text("Owl-mini", 26), code("1B · 4-bit", 20, MUTED)).arrange(m.DOWN, buff=0.06))
        mini[1].move_to(mini[0])
        mini.move_to([-5.2, y, 0])
        words = ["keys", "and", "values", "only"]
        chips = m.VGroup(*[draft_tok(w) for w in words]).arrange(m.RIGHT, buff=0.1).move_to([-1.55, y, 0])
        big = m.VGroup(card(2.3, 1.05, WEIGHTS, BLUE_FILL, radius=0.16, stroke=2.5),
                       m.VGroup(title_text("Owl-8B", 26), code("8B · 16 GB", 20, MUTED)).arrange(m.DOWN, buff=0.06))
        big[1].move_to(big[0])
        big.move_to([2.35, y, 0])
        a1 = arrow(mini.get_right(), chips.get_left(), MUTED)
        a2 = arrow(chips.get_right(), big.get_left(), MUTED)
        oks = [True, True, True, False]
        marks = m.VGroup(*[mark(ok).next_to(c, m.DOWN, buff=0.16) for ok, c in zip(oks, chips)])
        verdict = code(f"{sum(oks)} accepted · 1 pass", 20, TOKEN).next_to(big, m.DOWN, buff=0.2)

        tags = m.VGroup(pill("its own weights", WEIGHTS, BLUE_FILL, 20), pill("its own KV cache", KV, VIOLET_FILL, 20),
                        pill("its own steps", MEMORY, CYAN_FILL, 20)).arrange(m.RIGHT, buff=0.25)
        tags.move_to([-6.2, 0.15, 0], aligned_edge=m.LEFT)

        bx, scale = -2.3, 5.6 / OWL4_MS
        rows = m.VGroup()
        for i, (label, ms) in enumerate((("Owl-mini step", MINI_MS), ("Owl-8B 4-bit step", OWL4_MS))):
            yy = -0.95 - 0.68 * i
            lab = text(label, 22, INK).move_to([-6.2, yy, 0], aligned_edge=m.LEFT)
            bar = m.Rectangle(width=scale * ms, height=0.42).set_stroke(MEMORY, 2).set_fill(CYAN_FILL, 1)
            bar.move_to([bx + scale * ms / 2, yy, 0])
            val = code(f"{ms:.2f} ms", 22, INK, weight=m.BOLD).next_to(bar, m.RIGHT, buff=0.18)
            rows.add(m.VGroup(lab, bar, val))
        fifth = code(f"≈ 1/{OWL4_MS / MINI_MS:.0f}", 24, MEMORY, weight=m.BOLD).next_to(rows[0][2], m.RIGHT, buff=0.45)
        stamp = mac_stamp(["measured on this Mac · M4 Max · mlx-lm"]).move_to([-6.2, -2.3, 0], aligned_edge=m.LEFT)

        ideas = m.VGroup(pill("extra heads", WEIGHTS, BLUE_FILL, 22), pill("its own features", WEIGHTS, BLUE_FILL, 22),
                         pill("copy the prompt", REQUEST, PINK_FILL, 22)).arrange(m.RIGHT, buff=0.35)
        ideas.move_to([-0.6, -3.2, 0])
        mascot = owl(0.6).move_to([5.9, -3.0, 0])

        self.say(
            "[In episode thirteen, a small model drafted a few tokens], [and Owl checked them all in one pass.]",
            m.AnimationGroup(self.new_header("hook", "No second model"), m.FadeIn(cap), m.FadeIn(mini, shift=0.2 * m.RIGHT),
                             m.GrowArrow(a1), m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.RIGHT) for c in chips], lag_ratio=0.2)),
            m.AnimationGroup(m.GrowArrow(a2), m.FadeIn(big), sweep(big, run_time=0.9),
                             m.LaggedStart(*[m.FadeIn(x, scale=0.6) for x in marks], lag_ratio=0.2), m.FadeIn(verdict)),
        )
        self.say(
            "[But that draft is a second model, with its own weights and cache]: [each of its steps costs a fifth of a four-bit Owl step.]",
            m.LaggedStart(*[m.FadeIn(t, shift=0.15 * m.UP) for t in tags], lag_ratio=0.3),
            m.AnimationGroup(m.FadeIn(stamp), m.LaggedStart(*[m.AnimationGroup(m.FadeIn(r[0]), m.GrowFromEdge(r[1], m.LEFT), m.FadeIn(r[2]))
                                                              for r in rows], lag_ratio=0.4), m.FadeIn(fifth, run_time=1.6)),
        )
        self.say(
            "[Today, Owl drafts for itself]: [with extra heads], [with its own features], [and by copying from the prompt.]",
            m.AnimationGroup(mini.animate.set_opacity(0.3), a1.animate.set_opacity(0.3), m.FadeIn(mascot, shift=0.2 * m.UP)),
            m.FadeIn(ideas[0], shift=0.15 * m.UP),
            m.FadeIn(ideas[1], shift=0.15 * m.UP),
            m.FadeIn(ideas[2], shift=0.15 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ Medusa

    def medusa(self):
        self.next_section("1 · Medusa heads", notes=f"Medusa (Cai et al., ICML 2024, arXiv 2401.10774): extra decoding heads on the last hidden state, each a single feed-forward layer with a residual connection; head k guesses the token k+1 positions ahead (the usual LM head guesses the next one). Each head offers its top picks; top-2 of head 1 × top-3 of head 2 = {PATHS} continuations ({TREE_TOKENS} tree tokens), all verified in ONE forward pass with tree attention (a mask: each token attends only to its own branch's ancestors). The step keeps the longest accepted prefix: here '{' '.join(MEDUSA_BRANCH)}' = {len(MEDUSA_BRANCH)} tokens. Medusa-1 (heads trained, backbone frozen): 'over 2.2x speedup without compromising generation quality'; Medusa-2 (backbone fine-tuned too) 2.3–3.6×; its optional typical acceptance is not distribution-exact. Picks and the accepted branch are illustrative.")
        stack = owl_stack((-5.15, -0.4))
        src = np.array([stack[0].get_right()[0], stack[1][0].get_y(), 0])
        hx, hys = -2.4, (1.8, 0.7, -0.4)
        heads = [head_box("LM head", "+1"), head_box("head 1", "+2"), head_box("head 2", "+3")]
        for h, yy in zip(heads, hys):
            h.move_to([hx, yy, 0])
        stems = [m.CubicBezier(src, src + [0.8, 0, 0], h.get_left() + [-0.8, 0, 0], h.get_left()).set_stroke(MUTED, 2.5) for h in heads]
        top_bar = stack[1][0].copy().set_fill(INK, 0.9)
        state = code("hidden state", 20, MUTED).next_to(stack[0], m.UP, buff=0.12).align_to(stack[0], m.LEFT)
        layer_note = code("1 small layer each", 20, MUTED).next_to(heads[2], m.DOWN, buff=0.18)

        gx = hx + 1.25
        rows = []
        for h, words, first in zip(heads, ([NEXT], PICKS[1], PICKS[2]), (True, False, False)):
            chips = m.VGroup(*[(tok(w) if first else draft_tok(w)) for w in words]).arrange(m.RIGHT, buff=0.1)
            chips.move_to([gx, h.get_y(), 0], aligned_edge=m.LEFT)
            rows.append(chips)
        picks_cap = code("top picks", 20, MUTED).move_to([gx, 2.45, 0], aligned_edge=m.LEFT)

        # the tree: next token → head-1 picks → head-2 picks under each
        lx = (0.35, 2.4, 4.75)
        root = tok(NEXT).move_to([lx[0], 0.2, 0])
        l1 = [draft_tok(w).move_to([lx[1], yy, 0]) for w, yy in zip(PICKS[1], (1.17, -0.77))]
        pitch = 0.64
        l2 = []
        for parent in l1:
            for j, w in enumerate(PICKS[2]):
                l2.append(draft_tok(w).move_to([lx[2], parent.get_y() + (1 - j) * pitch, 0]).align_to([lx[2] - 0.55, 0, 0], m.LEFT))
        for c in l1:
            c.align_to([lx[1] - 0.4, 0, 0], m.LEFT)
        root.align_to([lx[0] - 0.45, 0, 0], m.LEFT)
        e1 = [edge(root, c) for c in l1]
        e2 = [edge(l1[i // 3], c) for i, c in enumerate(l2)]
        col_caps = m.VGroup(*[code(s, 20, MUTED).move_to([x, 2.45, 0]) for s, x in
                              zip(("LM head", "head 1", "head 2"), (root.get_x(), l1[0].get_x(), l2[0].get_x() + 0.1))])
        n_paths = code(f"{len(PICKS[1])} × {len(PICKS[2])} = {PATHS} continuations", 22, INK).move_to([2.9, -2.45, 0])
        mask_cap = code("tree attention: each token sees only its branch", 20, MUTED).move_to([1.5, -2.45, 0])
        win = pill(f"{len(MEDUSA_BRANCH)} tokens · 1 pass", TOKEN, GREEN_FILL, 22).move_to([2.9, -2.5, 0])
        med1 = m.VGroup(code(f"Medusa-1: only the heads trained · over {MEDUSA1}× faster", 20, INK),
                        cite("Cai et al., 2024")).arrange(m.RIGHT, buff=0.3).move_to([0.9, -3.3, 0])

        path_nodes = [root, l1[0], l2[0]]
        path_edges = [e1[0], e2[0]]
        others = [c for c in l1 + l2 if c not in path_nodes] + [e for e in e1 + e2 if e not in path_edges]

        self.say(
            "[Idea one: Medusa.] [Owl's top layer feeds one output head, which picks the next token.]",
            m.AnimationGroup(self.new_header("01 · medusa", "Heads that look ahead"), m.FadeIn(stack, shift=0.2 * m.UP)),
            m.AnimationGroup(m.FadeIn(top_bar), m.FadeIn(state), m.Create(stems[0]), m.FadeIn(heads[0], shift=0.15 * m.RIGHT),
                             m.FadeIn(rows[0], shift=0.15 * m.RIGHT)),
        )
        self.say(
            "[Medusa sprouts more heads beside it], [each just one small trained layer.]",
            m.AnimationGroup(m.Create(stems[1]), m.Create(stems[2]), m.GrowFromPoint(heads[1], src), m.GrowFromPoint(heads[2], src),
                             run_time=1.4),
            m.AnimationGroup(m.FadeIn(layer_note, shift=0.1 * m.UP), m.Circumscribe(m.VGroup(heads[1], heads[2]), color=TOKEN, buff=0.08)),
        )
        self.say(
            "[Head one guesses the token after next], [and head two the one after that], [each offering its top few picks.]",
            m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.RIGHT) for c in rows[1]], lag_ratio=0.3),
            m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.RIGHT) for c in rows[2]], lag_ratio=0.3),
            m.FadeIn(picks_cap, shift=0.1 * m.DOWN),
        )
        self.say(
            "[Together they form a tree]: [two picks times three, six possible continuations.]",
            m.AnimationGroup(m.FadeOut(picks_cap), m.FadeIn(col_caps),
                             m.ReplacementTransform(rows[0][0], root),
                             *[m.ReplacementTransform(rows[1][i], l1[i]) for i in range(2)],
                             *[m.ReplacementTransform(rows[2][j], l2[j]) for j in range(3)],
                             run_time=1.4),
            m.AnimationGroup(m.LaggedStart(*[m.Create(e) for e in e1 + e2], lag_ratio=0.1),
                             m.LaggedStart(*[m.FadeIn(c, target_position=l2[i]) for i, c in enumerate(l2[3:])], lag_ratio=0.15),
                             m.FadeIn(n_paths, shift=0.1 * m.UP)),
        )
        self.say(
            "[One pass of Owl checks every branch at once]; [a mask lets each token see only its own branch.]",
            m.AnimationGroup(sweep(stack), m.LaggedStart(*[m.ShowPassingFlash(e.copy().set_stroke(MEMORY, 6), time_width=0.8)
                                                           for e in e1 + e2], lag_ratio=0.05, run_time=1.4)),
            m.AnimationGroup(m.FadeOut(n_paths), m.FadeIn(mask_cap, shift=0.1 * m.UP)),
        )
        self.say(
            "[Owl keeps the longest branch it agrees with]: [three tokens from one pass, and with only the heads trained, the same answers.]",
            m.AnimationGroup(*[accept(c) for c in path_nodes[1:]], *[e.animate.set_stroke(TOKEN, 4) for e in path_edges],
                             *[o.animate.set_opacity(0.25) for o in others]),
            m.AnimationGroup(m.FadeOut(mask_cap), m.FadeIn(win, scale=0.9), m.FadeIn(med1, shift=0.1 * m.UP, run_time=1.4)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ EAGLE

    def eagle(self):
        self.next_section("2 · EAGLE", notes=f"EAGLE (Li, Wei, Zhang, Zhang; ICML 2024, arXiv 2401.15077): 'Autoregression at the feature (second-to-top-layer) level is more straightforward than at the token level' — the draft works on the feature vector Owl's last layer hands to its output (LM) head, plus the token sequence advanced by one step; one small layer predicts the next feature, Owl's own LM head turns it into a token, and it runs again (autoregressive, so each guess knows the previous one; Medusa's heads guess independently). EAGLE-2 (arXiv 2406.16858): a dynamic draft tree, expanded where the draft's confidence is high ({EAGLE2_X[0]}–{EAGLE2_X[1]}×). Verified by Owl in one pass, so the output distribution is preserved. EAGLE: {EAGLE_X[0]}–{EAGLE_X[1]}× on LLaMA2-Chat 70B. EAGLE-3 (arXiv 2503.01840, 2025): direct token prediction from fused low/mid/high-layer features, draft = one Transformer decoder layer, 'up to 6.5x'. The feature bars and confidences are illustrative.")
        stack = owl_stack((-5.5, 1.0), w=1.8, h=2.7, n=5, name="Owl-8B", name_size=24, pitch=0.27)
        ry = 1.05
        f_in = feature([0.35, 0.8, 0.5, 0.95, 0.25, 0.6, 0.7, 0.4]).move_to([-3.55, 1.55, 0])
        f_lab = code("feature", 20, MUTED).next_to(f_in, m.UP, buff=0.12)
        t_in = tok(NEXT).move_to([-3.3, 0.45, 0])
        t_lab = code("latest token", 20, MUTED).next_to(t_in, m.DOWN, buff=0.12)
        layer = m.VGroup(card(1.9, 1.2, WEIGHTS, BLUE_FILL, radius=0.16, stroke=2.2),
                         m.VGroup(text("draft layer", 22, INK, weight=m.BOLD), code("1 layer", 20, MUTED)).arrange(m.DOWN, buff=0.08))
        layer[1].move_to(layer[0])
        layer.move_to([-1.35, ry, 0])
        f_out = feature([0.6, 0.3, 0.85, 0.45, 0.7, 0.2, 0.9, 0.5]).move_to([0.75, ry, 0])
        lm = head_box("Owl's head", "", w=1.75)
        lm[1][1].set_opacity(0)
        lm[1].move_to(lm[0])
        lm.move_to([2.6, ry, 0])
        t_out = draft_tok(EAGLE_MAIN[0][0]).move_to([4.25, ry, 0])
        a0 = arrow(stack[0].get_right() + [0, 0.5, 0], f_in.get_left() + [0, 0.0, 0], MUTED)
        a1 = arrow(f_in.get_right(), layer[0].get_left() + [0, 0.28, 0], MUTED)
        a2 = arrow(t_in.get_right(), layer[0].get_left() + [0, -0.28, 0], MUTED)
        a3 = arrow(layer[0].get_right(), f_out.get_left(), MUTED)
        a4 = arrow(f_out.get_right(), lm[0].get_left(), MUTED)
        a5 = arrow(lm[0].get_right(), t_out.get_left(), MUTED)
        loop = m.ArcBetweenPoints(t_out.get_top() + [0, 0.08, 0], layer[0].get_top() + [0.3, 0.05, 0], angle=0.75)
        loop.set_stroke(TOKEN, 3).add_tip(tip_length=0.18, tip_width=0.18)
        again = code("again", 20, TOKEN).move_to([1.4, 2.45, 0])

        # the dynamic tree (EAGLE-2), below the pipeline
        ty, ay = -1.3, -2.3
        xs = (-5.4, -3.5, -1.6, 0.3)
        t_root = tok(NEXT).move_to([xs[0], ty, 0])
        mains = [draft_tok(w).move_to([x, ty, 0]) for (w, _), x in zip(EAGLE_MAIN, xs[1:])]
        alts = [draft_tok(w).move_to([x, ay, 0]) for (w, _), x in zip(EAGLE_ALT, xs[1:])]
        chain = [t_root] + mains
        m_edges = [edge(a, b) for a, b in zip(chain, mains)]
        a_edges = [edge(p, c) for p, c in zip(chain, alts)]
        m_conf = [code(f"{p:.2f}", 20, TOKEN).next_to(c, m.UP, buff=0.12) for (_, p), c in zip(EAGLE_MAIN, mains)]
        a_conf = [code(f"{p:.2f}", 20, MUTED).next_to(c, m.DOWN, buff=0.12) for (_, p), c in zip(EAGLE_ALT, alts)]
        tree_cap = code("draft tree, for example", 20, MUTED).move_to([xs[0] - 0.45, -3.3, 0], aligned_edge=m.LEFT)
        deep = code("deeper if sure", 20, TOKEN).move_to([xs[3] + 0.8, ty, 0], aligned_edge=m.LEFT)
        short = code("short if not", 20, MUTED).move_to([xs[3] + 0.8, ay, 0], aligned_edge=m.LEFT)

        s1 = m.VGroup(title_text(f"{EAGLE_X[0]}–{EAGLE_X[1]}×", 44, TOKEN), cite("EAGLE · 70B chat"),
                      cite("Li et al., 2024")).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
        s3 = m.VGroup(title_text(f"up to {EAGLE3_MAX}×", 44, TOKEN), cite("EAGLE-3 · fused layers"),
                      cite("Li et al., 2025")).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
        stats = m.VGroup(s1, s3).arrange(m.DOWN, buff=0.4, aligned_edge=m.LEFT).move_to([3.95, -0.6, 0], aligned_edge=m.UL)
        same = pill("verified: same answers", TOKEN, GREEN_FILL, 20).move_to([0.05, -3.3, 0], aligned_edge=m.LEFT)

        self.say(
            "[Idea two: EAGLE.] [It drafts from the feature Owl's top layer hands to its output head.]",
            m.AnimationGroup(self.new_header("02 · eagle", "Drafting from features"), m.FadeIn(stack, shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowArrow(a0), m.FadeIn(f_in, shift=0.2 * m.RIGHT), m.FadeIn(f_lab)),
        )
        self.say(
            "[One small layer takes that feature and the latest token], [and predicts the next feature.]",
            m.AnimationGroup(m.FadeIn(t_in, shift=0.15 * m.UP), m.FadeIn(t_lab), m.GrowArrow(a1), m.GrowArrow(a2),
                             m.FadeIn(layer, scale=0.9)),
            m.AnimationGroup(m.GrowArrow(a3), m.FadeIn(f_out, shift=0.2 * m.RIGHT)),
        )
        self.say(
            "[Owl's own head turns it into a token], [and the small layer runs again, knowing what it just picked.]",
            m.AnimationGroup(m.GrowArrow(a4), m.FadeIn(lm), m.GrowArrow(a5), m.FadeIn(t_out, shift=0.15 * m.RIGHT)),
            m.AnimationGroup(m.Create(loop), m.FadeIn(again),
                             m.Succession(m.Wait(0.5), m.Circumscribe(layer[0], color=TOKEN, buff=0.06))),
        )
        self.say(
            "[Its next version, EAGLE-2, grows the tree deeper where the drafter is confident], [and keeps it short where it's not.]",
            m.AnimationGroup(m.FadeIn(tree_cap), m.FadeIn(t_root, target_position=t_in),
                             m.Succession(*[m.AnimationGroup(m.Create(e), m.FadeIn(c, target_position=t_out), m.FadeIn(p))
                                            for e, c, p in zip(m_edges, mains, m_conf)], run_time=2.4),
                             m.FadeIn(deep)),
            m.AnimationGroup(m.LaggedStart(*[m.AnimationGroup(m.Create(e), m.FadeIn(c), m.FadeIn(p))
                                             for e, c, p in zip(a_edges, alts, a_conf)], lag_ratio=0.3), m.FadeIn(short)),
        )
        self.say(
            "[Owl verifies the tree in one pass, so the answers don't change]: [about three times faster], [and up to six and a half with EAGLE-3.]",
            m.AnimationGroup(sweep(stack), *[accept(c) for c in mains], *[e.animate.set_stroke(TOKEN, 4) for e in m_edges],
                             *[x.animate.set_opacity(0.3) for x in alts + a_edges + a_conf], m.FadeIn(same, scale=0.9)),
            m.FadeIn(s1, shift=0.15 * m.UP),
            m.FadeIn(s3, shift=0.15 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ prompt lookup: the running example

    def lookup(self):
        self.next_section("3 · Prompt lookup", notes=f"Prompt-lookup decoding (A. Saxena, GitHub, Nov 2023): 'replace the draft model with simple string matching in the prompt'. Take the last {NGRAM} tokens (max_ngram_size = 3), find them earlier in the prompt, propose the next {K} tokens (num_pred_tokens = 10), verify in one pass as in Episode 13. Running example: a student's notes on the KV cache (words stand in for tokens; illustrative): the tail 'step computes the' matches the notes at position {MATCH}; lookup_draft() returns '{' '.join(DRAFT)}'. Owl's own continuation (illustrative) agrees with the first {ACCEPTED}, rejects 'and appends' and adds its own '{BONUS}': {PER_PASS} tokens for one pass, the draft cost a string search. Input-grounded tasks (summaries, document QA, code edits, quoting notes) have long matches; open chat has few.")
        y_lab, y_notes = 2.35, 1.95
        user = icon(recolored("user", REQUEST, PINK_FILL, PREFIX), 0.42)
        n_lab = m.VGroup(user, code("the student's notes", 20, REQUEST)).arrange(m.RIGHT, buff=0.18).move_to([-6.3, y_lab, 0], aligned_edge=m.LEFT)
        question = code("Q: how does the cache save work?", 20, REQUEST).move_to([6.4, y_lab, 0], aligned_edge=m.RIGHT)
        notes = token_row(NOTES, REQUEST, 22, max_width=12.7)
        notes.move_to([-6.3, y_notes, 0], aligned_edge=m.UL)
        assert len({round(c.get_y(), 2) for c in notes[MATCH:MATCH + NGRAM]}) == 1  # the match sits on one row

        mascot = owl(0.32)
        a_lab = m.VGroup(mascot, code("Owl's answer", 20, TOKEN)).arrange(m.RIGHT, buff=0.18).move_to([-6.3, 0.42, 0], aligned_edge=m.LEFT)
        answer = token_row(ANSWER, TOKEN, 22).move_to([-6.3, -0.15, 0], aligned_edge=m.LEFT)
        tail = answer[-NGRAM:]

        def box_around(chips, color=TOKEN):
            return m.SurroundingRectangle(m.VGroup(*chips), buff=0.07, corner_radius=0.1).set_stroke(color, 3)

        t_box = box_around(tail)
        t_lab = code(f"last {NGRAM} tokens", 20, TOKEN).next_to(t_box, m.RIGHT, buff=0.25)
        windows = [box_around(notes[i:i + NGRAM], INK) for i in range(MATCH + 1)]
        scan = windows[0].copy()
        found = box_around(notes[MATCH:MATCH + NGRAM], TOKEN)

        d_lab = code("draft", 22, MUTED, weight=m.BOLD).move_to([-6.3, -1.05, 0], aligned_edge=m.LEFT)
        draft = m.VGroup(*[draft_tok(w, 22) for w in DRAFT]).arrange(m.RIGHT, buff=0.08).next_to(d_lab, m.RIGHT, buff=0.3)
        oks = [i < ACCEPTED for i in range(K)]
        marks = m.VGroup(*[mark(ok).next_to(c, m.DOWN, buff=0.14) for ok, c in zip(oks, draft)])
        bonus = tok(BONUS, size=22).next_to(marks[ACCEPTED], m.DOWN, buff=0.14)
        b_lab = code("Owl's own token", 20, TOKEN).next_to(bonus, m.LEFT, buff=0.2)
        stat = m.VGroup(title_text(f"{PER_PASS} tokens", 56, TOKEN), m.VGroup(text("for one read of the weights", 24, INK),
                        code("the draft cost: a string search", 20, MUTED)).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT))
        stat.arrange(m.RIGHT, buff=0.35).move_to([-6.3, -3.15, 0], aligned_edge=m.LEFT)
        flow = m.ShowPassingFlash(m.Line(draft.get_left() + [0, 0.0, 0], draft.get_right()).set_stroke(MEMORY, 12),
                                  time_width=0.5, run_time=1.1)

        self.say(
            "[Idea three needs no training at all: prompt lookup.] [A student pastes notes on the KV cache, and asks how it saves work.]",
            m.AnimationGroup(self.new_header("03 · prompt lookup", "Copy from the notes"), m.FadeIn(n_lab)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.UP) for c in notes], lag_ratio=0.04, run_time=1.8),
                             m.FadeIn(question, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[Owl's answer starts to quote the notes.] [Take its last three tokens.]",
            m.AnimationGroup(m.FadeIn(a_lab), m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.RIGHT) for c in answer], lag_ratio=0.15)),
            m.AnimationGroup(m.Create(t_box), m.FadeIn(t_lab)),
        )
        self.say(
            "[Search the prompt for the same three]... [there they are.]",
            m.AnimationGroup(m.FadeIn(scan), m.Succession(*[m.Transform(scan, w, run_time=0.3) for w in windows[1:]])),
            m.AnimationGroup(m.ReplacementTransform(scan, found), m.Circumscribe(m.VGroup(*notes[MATCH:MATCH + NGRAM]), color=TOKEN, buff=0.12)),
        )
        self.say(
            "[Copy the ten tokens that follow], [as the draft.]",
            m.LaggedStart(*[m.FadeIn(d, target_position=notes[MATCH + NGRAM + i], path_arc=-0.3)
                            for i, d in enumerate(draft)], lag_ratio=0.1, run_time=1.8),
            m.FadeIn(d_lab, shift=0.1 * m.RIGHT),
        )
        self.say(
            "[Owl verifies them in one pass]: [eight are accepted], [and where the copy goes wrong, Owl adds its own next token.]",
            flow,
            m.AnimationGroup(m.LaggedStart(*[m.AnimationGroup(accept(c), m.FadeIn(x, scale=0.6)) for c, x in zip(draft[:ACCEPTED], marks[:ACCEPTED])],
                                           lag_ratio=0.15)),
            m.AnimationGroup(*[reject(c) for c in draft[ACCEPTED:]], *[m.FadeIn(x, scale=0.6) for x in marks[ACCEPTED:]],
                             m.FadeIn(bonus, shift=0.15 * m.UP), m.FadeIn(b_lab)),
        )
        self.say(
            "[Nine tokens for one read of the weights], [and the draft cost only a string search.]",
            m.FadeIn(stat[0], shift=0.15 * m.UP),
            m.FadeIn(stat[1], shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    def lookup_measured(self):
        self.next_section("3b · Copying, measured", notes=f"The code is a simplified excerpt of Saxena's find_candidate_pred_tokens (one n-gram size, first match), and it runs: the scene asserts it returns the draft shown. Measured on this Mac in Episode 13's run (measurements.md §9, mlx-lm 0.32.2, greedy, 256 tokens, 3 trials): 'Copy out the following study notes exactly…' with Owl-8B BF16 as target and Owl-mini (Llama-3.2-1B 4-bit) drafting γ = {GAMMA}: acceptance {ALPHA:.2f}, {PER_TARGET:.2f} tokens per target pass, {BASE:.1f} → {FAST:.1f} tok/s = {SPEEDUP:.2f}×, output identical to greedy. One-token ceiling: 546 GB/s ÷ {W_BYTES / 1e9:.2f} GB = {CEIL_MAC:.1f} tok/s (decode_ceiling); plain decode reaches {BASE / CEIL_MAC:.0%} of it. Prompt lookup itself was NOT measured on this Mac. Saxena (Nov 2023, one A100, Mistral-7B, greedy): '2x-4x in input-grounded tasks', ~2.4× on summarization and context QA. HF transformers: prompt_lookup_num_tokens=10; vLLM: an n-gram speculator.")
        cb = code_block(LOOKUP_SRC, size=20, title="prompt lookup · simplified")
        cb.move_to([-6.4, 0.75, 0], aligned_edge=m.LEFT)
        body = cb[1]
        hl = [glyphs(body, "tail = seq[-n:]"), glyphs(body, "if seq[i:i + n] == tail:"), glyphs(body, "return seq[i + n:i + n + k]")]
        hf = code("transformers: prompt_lookup_num_tokens=10", 20, MUTED).next_to(cb, m.DOWN, buff=0.25).align_to(cb, m.LEFT)

        px = 1.15
        stamp = mac_stamp(["measured on this Mac · Episode 13", "copying notes · Owl-mini drafts 6"]).move_to([px, 2.15, 0], aligned_edge=m.LEFT)
        acc = code(f"accepted: {ALPHA:.0%} of drafts", 22, TOKEN).move_to([px, 1.45, 0], aligned_edge=m.LEFT)
        bw = 3.35
        per = bw / FAST
        b1 = m.Rectangle(width=per * BASE, height=0.42).set_stroke(MEMORY, 2).set_fill(CYAN_FILL, 1).move_to([px + per * BASE / 2, 0.75, 0])
        b2 = m.Rectangle(width=per * FAST, height=0.42).set_stroke(TOKEN, 2).set_fill(GREEN_FILL, 1).move_to([px + per * FAST / 2, -0.05, 0])
        cx = px + per * CEIL_MAC
        ceil = m.DashedLine([cx, 1.08, 0], [cx, -0.38, 0], dash_length=0.1).set_stroke(MEMORY, 2.5)
        l1 = code(f"{BASE:.1f} tok/s plain", 20, INK).next_to(ceil, m.RIGHT, buff=0.15).set_y(0.75)
        l2 = code(f"{FAST:.0f} tok/s", 22, INK, weight=m.BOLD).next_to(b2, m.RIGHT, buff=0.15)
        ceil_lab = m.VGroup(code(f"one-token ceiling: {CEIL_MAC:.0f} tok/s", 20, MEMORY), code(f"= 546 GB/s ÷ {W_BYTES / 1e9:.0f} GB", 20, MUTED))
        ceil_lab.arrange(m.DOWN, buff=0.06, aligned_edge=m.LEFT).move_to([px, -0.6, 0], aligned_edge=m.UL)
        speed = title_text(f"{SPEEDUP:.2f}×", 52, TOKEN).move_to([px, -1.62, 0], aligned_edge=m.LEFT)
        per_pass = code(f"{PER_TARGET:.1f} tokens per weight read", 20, INK).move_to([px, -2.25, 0], aligned_edge=m.LEFT)
        no_meas = pill("prompt lookup itself: not measured here", AMBER, AMBER_FILL, 20).move_to([-6.4, -2.85, 0], aligned_edge=m.LEFT)
        sax = m.VGroup(code(f"{SAXENA[0]}–{SAXENA[1]}× on input-grounded tasks", 20, TOKEN), cite("Saxena, 2023 · one A100 · Mistral-7B"))
        sax.arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT).move_to([px, -2.98, 0], aligned_edge=m.LEFT)
        same = pill("same easy guesses · no second model", REQUEST, PINK_FILL, 20).move_to([-6.4, -2.85, 0], aligned_edge=m.LEFT)

        self.say(
            "[In code, it's a few lines]: [take the last tokens], [find them earlier], [and copy what followed.]",
            m.AnimationGroup(self.new_header("03 · measured", "Copying is easy to guess"), m.FadeIn(cb, shift=0.15 * m.UP), m.FadeIn(hf)),
            m.Circumscribe(hl[0], color=TOKEN, buff=0.06),
            m.Circumscribe(hl[1], color=TOKEN, buff=0.06),
            m.Circumscribe(hl[2], color=TOKEN, buff=0.06),
        )
        self.say(
            "[On this Mac, in episode thirteen's test, Owl copied out a student's notes with Owl-mini drafting]: [every draft token was accepted.]",
            m.FadeIn(stamp, shift=0.1 * m.UP),
            m.FadeIn(acc, shift=0.1 * m.UP),
        )
        self.say(
            "[Thirty-one tokens per second became a hundred and forty-eight], [far above the one-token ceiling of thirty-four.]",
            m.AnimationGroup(m.GrowFromEdge(b1, m.LEFT), m.FadeIn(l1),
                             m.GrowFromEdge(b2, m.LEFT, run_time=1.8), m.FadeIn(l2, run_time=1.8)),
            m.AnimationGroup(m.Create(ceil), m.FadeIn(ceil_lab)),
        )
        self.say(
            "[Each read of the weights now yields almost seven tokens.] [Prompt lookup goes after the same easy guesses, with no second model.]",
            m.AnimationGroup(m.FadeIn(speed, shift=0.1 * m.UP), m.FadeIn(per_pass)),
            m.FadeIn(same, scale=0.9),
        )
        self.say(
            "[We didn't measure prompt lookup on this Mac]; [its author reports two to four times faster on tasks like this.]",
            m.AnimationGroup(m.FadeOut(same), m.FadeIn(no_meas, scale=0.9)),
            m.FadeIn(sax, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ multi-token prediction

    def mtp(self):
        self.next_section("4 · Multi-token prediction", notes=f"Multi-token prediction: the drafter is trained in. Gloeckle et al. (Meta, ICML 2024, arXiv 2404.19737): 'predict the following n tokens using n independent output heads, operating on top of a shared model trunk'; 'models trained with 4-token prediction are up to 3 times faster at inference'. DeepSeek-V3 (arXiv 2412.19437): MTP depth D = 1 — one MTP module (shared embedding and output head, one Transformer block, a projection) chained after the main model; it can be dropped at inference or used to draft: 'the acceptance rate of the second token prediction ranges between 85% and 90%', 'delivering 1.8 times TPS'. Episode 13's formula E = (1 − α^(γ+1))/(1 − α) with γ = 1 is 1 + α = {DS_E[0]:.2f}–{DS_E[1]:.2f} tokens per pass (expected_tokens_per_pass). The trunk and heads drawing is schematic.")
        trunk = owl_stack((-4.75, -0.25), w=2.9, h=3.6, n=7, name="shared trunk", sub="trained together", name_size=24)
        src = np.array([trunk[0].get_right()[0], trunk[1][0].get_y(), 0])
        hys = (1.65, 0.65, -0.35, -1.35)
        heads = [head_box(f"head {i + 1}", f"+{i + 1}", w=1.7).move_to([-1.4, yy, 0]) for i, yy in enumerate(hys)]
        stems = [m.CubicBezier(src, src + [0.7, 0, 0], h.get_left() + [-0.7, 0, 0], h.get_left()).set_stroke(MUTED, 2.5) for h in heads]
        outs = [tok(w).next_to(h, m.RIGHT, buff=0.3) for w, h in zip(("keys", "and", "values", "only"), heads)]
        meta = m.VGroup(title_text(f"up to {META_X}×", 44, TOKEN), cite(f"{META_HEADS}-token prediction · Gloeckle et al., 2024")).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        meta.move_to([-6.2, -3.05, 0], aligned_edge=m.LEFT)

        # DeepSeek-V3: one module chained after the main model
        mod = m.VGroup(card(1.7, 0.62, WEIGHTS, BLUE_FILL, radius=0.12, stroke=2.2),
                       text("MTP module", 20, INK, weight=m.BOLD))
        mod[1].move_to(mod[0])
        mod.move_to(heads[1])
        ds_lab = code("DeepSeek-V3 · one extra module", 20, MUTED).move_to([-1.4, 2.45, 0])

        fx = 4.1
        f_card = card(4.8, 4.15, EDGE, CARD, radius=0.18, stroke=2).move_to([fx, 0.12, 0])
        acc = code(f"α = {DS_ALPHA[0]:.2f}–{DS_ALPHA[1]:.2f}, reported", 20, TOKEN).move_to([fx, 1.72, 0])
        f_title = code("Episode 13's formula", 20, MUTED).move_to([fx, 1.12, 0])
        formula = tex(r"E = \frac{1-\alpha^{\gamma+1}}{1-\alpha}", 40).move_to([fx, 0.42, 0])
        g1 = tex(r"\gamma = 1:\; E = 1 + \alpha", 36).move_to([fx, -0.4, 0])
        e_val = title_text(f"{DS_E[0]:.2f}–{DS_E[1]:.2f}", 48, TOKEN).move_to([fx, -1.08, 0])
        e_sub = code("tokens per pass", 20, MUTED).next_to(e_val, m.DOWN, buff=0.1)
        tps = m.VGroup(code(f"reported: {DS_TPS}× tokens/s", 22, INK), cite("DeepSeek-V3 report, 2024")).arrange(m.DOWN, buff=0.08)
        tps.move_to([fx, -2.72, 0])

        self.say(
            "[Idea four is built in during training]: [multi-token prediction.]",
            m.AnimationGroup(self.new_header("04 · multi-token", "Trained to look ahead"), m.FadeIn(trunk, shift=0.2 * m.UP)),
            m.AnimationGroup(*[m.Create(s) for s in stems], *[m.GrowFromPoint(h, src) for h in heads], run_time=1.4),
        )
        self.say(
            "[Meta trained models to predict four tokens at once from one shared trunk], [and they ran up to three times faster.]",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(o, shift=0.15 * m.RIGHT) for o in outs], lag_ratio=0.25), sweep(trunk)),
            m.FadeIn(meta, shift=0.1 * m.UP),
        )
        self.say(
            "[Even DeepSeek's version three keeps one extra module that predicts the token after next], [and its guess is accepted eighty-five to ninety percent of the time.]",
            m.AnimationGroup(*[m.FadeOut(x) for x in (heads[2], heads[3], stems[2], stems[3], outs[2], outs[3])],
                             m.ReplacementTransform(heads[1], mod), m.FadeIn(ds_lab), outs[1][0].animate.set_stroke(MUTED, 1.8).set_fill(CARD, 1)),
            m.AnimationGroup(m.FadeIn(f_card), m.FadeIn(acc, shift=0.1 * m.UP)),
        )
        self.say(
            "[By episode thirteen's formula, that's about one point nine tokens per pass], [and DeepSeek reports one point eight times the speed.]",
            m.AnimationGroup(m.FadeIn(f_title), m.FadeIn(formula), m.FadeIn(g1, shift=0.1 * m.DOWN, run_time=1.4),
                             m.FadeIn(e_val, shift=0.1 * m.UP, run_time=1.6), m.FadeIn(e_sub)),
            m.FadeIn(tps, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ busy servers

    def busy(self):
        self.next_section("5 · Why the gains fade", notes=f"Speculation spends spare compute to save memory passes (notes §6). A BF16 decode pass of n tokens does 2·P·n FLOPs over 2·P bytes of weights, so its intensity ≈ n FLOPs/byte (KV reads ignored). H100 SXM (3.35 TB/s, 989 TFLOPS BF16 dense; NVIDIA, dated Oct 2026): ridge ≈ {RIDGE:.0f}. Each sequence puts 1 + {CHAIN} tokens through a pass (EAGLE-3's SGLang test drafts a chain of {CHAIN}). Batch 1: 1 → {PER_SEQ} tokens, pass {pass_ms(1):.2f} → {pass_ms(PER_SEQ):.2f} ms. Batch 64: 64 → {64 * PER_SEQ} tokens (just under the ridge), {pass_ms(64):.2f} → {pass_ms(64 * PER_SEQ):.2f} ms. Batch 128: 128 → {128 * PER_SEQ}, past the ridge: {pass_ms(128):.2f} → {pass_ms(128 * PER_SEQ):.2f} ms ({pass_ms(512) / pass_ms(128):.1f}× longer) (owl_reference.step_time; a roofline floor). Real kernels run below peak, the drafter itself costs time and rejected guesses are wasted work, so real gains fade sooner: EAGLE-3 paper (Li et al., 2025), SGLang throughput on one H100, Llama-3.1-8B-Instruct (Owl's model), MT-Bench, chain of 3, no tree: batch {', '.join(map(str, BATCHES))}: EAGLE {', '.join(f'{x:.2f}' for x in EAGLE1_TP)}×; EAGLE-3 {', '.join(f'{x:.2f}' for x in EAGLE3_TP)}× (batch 56 is in the primary table, not in notes §6).")
        roof = Roofline(GPU["bandwidth"], GPU["peak"], width=6.2, height=3.9, x_decades=(-1, 4), y_decades=(11.5, 15.5), center=(-2.15, -0.45, 0))
        ylab = code("attainable FLOP/s · one H100", 20, MUTED).next_to(roof.axes[1].get_end(), m.RIGHT, buff=0.2).align_to(roof.axes[1].get_end(), m.UP)
        r_lab = code(f"ridge ≈ {RIDGE:.0f}", 20, MUTED).move_to(roof.at(roof.ridge_x, 10 ** 11.85)).align_to(roof.ridge, m.LEFT).shift(0.15 * m.RIGHT)
        x_note = code("FLOPs per byte ≈ tokens per pass", 20, MUTED).move_to(roof.x_label).shift(0.0 * m.DOWN)

        def dots(batch):
            plain = roof.dot(batch, MEMORY, radius=0.11)
            spec = roof.dot(batch * PER_SEQ, TOKEN, radius=0.13).set_z_index(4)
            return plain, spec

        px = 2.4

        def readout(batch):
            n = batch * PER_SEQ
            share = n / RIDGE
            g = m.VGroup(
                title_text(f"batch {batch}", 44, INK),
                code(f"tokens/pass: {batch} → {n}", 22, INK),
                code(f"pass: {pass_ms(batch):.1f} → {pass_ms(n):.1f} ms", 22, MEMORY if n < RIDGE else COMPUTE),
                code(f"math needs {share:.0%} of it" if share <= 1 else "math is the limit now", 20, MUTED if share < 0.5 else COMPUTE),
            ).arrange(m.DOWN, buff=0.18, aligned_edge=m.LEFT)
            return g.move_to([px, 0.55, 0], aligned_edge=m.LEFT)

        key = m.VGroup(
            m.VGroup(m.Dot(radius=0.1, color=MEMORY), code("one token per user", 20, INK)).arrange(m.RIGHT, buff=0.15),
            m.VGroup(m.Dot(radius=0.12, color=TOKEN), code(f"+ {CHAIN} drafts each", 20, INK)).arrange(m.RIGHT, buff=0.15),
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.14).move_to([px, -1.95, 0], aligned_edge=m.LEFT)
        p1, s1 = dots(1)
        p64, s64 = dots(64)
        p128, s128 = dots(128)
        r1, r64, r128 = readout(1), readout(64), readout(128)
        spare_arrow = m.DoubleArrow(roof.at(1, 10 ** 14.92), roof.at(1, 10 ** 12.68), buff=0, color=COMPUTE, stroke_width=3,
                                    max_tip_length_to_length_ratio=0.08)
        spare = code("spare\ncompute", 22, COMPUTE, weight=m.BOLD).next_to(roof.at(1, 10 ** 14.35), m.RIGHT, buff=0.2)

        self.say(
            "[But on a busy server, the gains fade.] [Drafting spends spare compute to save trips through memory.]",
            m.AnimationGroup(self.new_header("05 · busy servers", "Why the gains fade"),
                             m.FadeIn(roof.axes), m.FadeIn(roof.ticks), m.FadeIn(x_note), m.FadeIn(ylab),
                             m.Create(roof.memory), m.Create(roof.compute), m.Create(roof.ridge), m.FadeIn(r_lab)),
            m.AnimationGroup(m.FadeIn(spare, shift=0.1 * m.DOWN), m.GrowArrow(spare_arrow)),
        )
        self.say(
            "[At batch one, checking four tokens instead of one barely moves Owl along the memory slope]: [the pass costs the same.]",
            m.AnimationGroup(m.FadeIn(p1, scale=0.4), m.FadeIn(key), m.FadeIn(r1, shift=0.1 * m.UP),
                             m.FadeIn(s1, target_position=p1, run_time=1.4)),
            m.Circumscribe(r1[2], color=MEMORY, buff=0.08),
        )
        self.say(
            "[At batch sixty-four, that's two hundred and fifty-six tokens per pass], [close to the ridge]; [past it, every extra guess costs time.]",
            m.AnimationGroup(m.FadeOut(spare), m.FadeOut(spare_arrow), m.Transform(p1, p64), m.Transform(s1, s64), m.Transform(r1, r64), run_time=1.4),
            m.Circumscribe(r_lab, color=INK, buff=0.08),
            m.AnimationGroup(m.Transform(p1, p128), m.Transform(s1, s128), m.Transform(r1, r128), run_time=1.4),
        )
        self.wait(0.3)
        self.clear()

        # EAGLE-3's batch table
        ox, oy, w, h = -5.55, -2.2, 6.8, 3.9
        y0, y1 = 0.7, 2.1
        xs = [ox + 0.45 + i * (w - 0.9) / (len(BATCHES) - 1) for i in range(len(BATCHES))]

        def p(i, v):
            return np.array([xs[i], oy + h * (v - y0) / (y1 - y0), 0])

        xa = m.Arrow([ox, oy, 0], [ox + w + 0.4, oy, 0], buff=0, color=FAINT, stroke_width=3, max_tip_length_to_length_ratio=0.03)
        ya = m.Arrow([ox, oy, 0], [ox, oy + h + 0.15, 0], buff=0, color=FAINT, stroke_width=3, max_tip_length_to_length_ratio=0.06)
        xt = m.VGroup(*[code(str(b), 20, MUTED).move_to([x, oy - 0.3, 0]) for b, x in zip(BATCHES, xs)])
        x_title = text("batch size", 20, MUTED).move_to([ox + w / 2, oy - 0.7, 0])
        yt = m.VGroup(*[code(f"{v:.1f}×", 20, MUTED).next_to([ox, oy + h * (v - y0) / (y1 - y0), 0], m.LEFT, buff=0.14) for v in (1.0, 1.5, 2.0)])
        y_title = text("throughput vs no drafting", 20, MUTED).move_to([ox + 0.3, oy + h + 0.1, 0], aligned_edge=m.LEFT)
        one = m.DashedLine([ox, p(0, 1.0)[1], 0], [ox + w, p(0, 1.0)[1], 0], dash_length=0.12).set_stroke(INK, 2, opacity=0.6)
        slower = m.Rectangle(width=w, height=p(0, 1.0)[1] - oy).set_stroke(width=0).set_fill(BAD, 0.12)
        slower.move_to([ox + w / 2, (oy + p(0, 1.0)[1]) / 2, 0])
        slow_lab = code("below 1×: slower", 20, BAD).move_to([ox + 0.15, oy + 0.25, 0], aligned_edge=m.LEFT)

        def series(vals, color, r=0.08):
            line = m.VMobject().set_points_as_corners([p(i, v) for i, v in enumerate(vals)]).set_stroke(color, 4)
            pts = m.VGroup(*[m.Dot(p(i, v), radius=r, color=color).set_z_index(3) for i, v in enumerate(vals)])
            return line, pts

        e1_line, e1_dots = series(EAGLE1_TP, INK)
        e3_line, e3_dots = series(EAGLE3_TP, TOKEN)
        e1_first = code(f"{EAGLE1_TP[0]:.2f}×", 20, INK).next_to(p(0, EAGLE1_TP[0]), m.DOWN, buff=0.16)
        e3_first = code(f"{EAGLE3_TP[0]:.2f}×", 20, TOKEN).next_to(p(0, EAGLE3_TP[0]), m.UP, buff=0.16)
        e1_24 = code(f"{EAGLE1_TP[4]:.2f}×", 20, BAD).next_to(p(4, EAGLE1_TP[4]), m.DOWN, buff=0.16)
        e3_last = code(f"{EAGLE3_TP[-1]:.2f}×", 20, TOKEN).next_to(p(8, EAGLE3_TP[-1]), m.UP, buff=0.16)
        lx = 2.05
        legend = m.VGroup(
            m.VGroup(m.Line([0, 0, 0], [0.5, 0, 0]).set_stroke(TOKEN, 4), code("EAGLE-3", 22, TOKEN, weight=m.BOLD)).arrange(m.RIGHT, buff=0.2),
            m.VGroup(m.Line([0, 0, 0], [0.5, 0, 0]).set_stroke(INK, 4), code("EAGLE", 22, INK, weight=m.BOLD)).arrange(m.RIGHT, buff=0.2),
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.2).move_to([lx, 1.5, 0], aligned_edge=m.LEFT)
        setup = m.VGroup(code("SGLang · one H100", 20, INK), code("Llama-3.1-8B = Owl's model", 20, INK),
                         code("MT-Bench · 3 drafts", 20, INK), cite("Li et al., EAGLE-3, 2025")).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
        setup.move_to([lx, -0.3, 0], aligned_edge=m.LEFT)

        self.say(
            "[The EAGLE-3 authors measured this on an H100, with Owl's own model.]",
            m.AnimationGroup(m.FadeIn(xa), m.FadeIn(ya), m.FadeIn(xt), m.FadeIn(x_title), m.FadeIn(yt), m.FadeIn(y_title),
                             m.FadeIn(slower), m.Create(one), m.FadeIn(slow_lab), m.FadeIn(setup, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[The original EAGLE gave forty percent more throughput at batch two], [but from batch twenty-four on, it made things slower.]",
            m.AnimationGroup(m.FadeIn(legend[1]), m.FadeIn(e1_dots[0], scale=0.5), m.FadeIn(e1_first)),
            m.AnimationGroup(m.Create(e1_line, run_time=1.8), m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in e1_dots[1:]], lag_ratio=0.2),
                             m.FadeIn(e1_24, run_time=1.6)),
        )
        self.say(
            "[Its successor, EAGLE-3, holds on]: [about one point eight times at small batches], [and still one point three eight at sixty-four.]",
            m.FadeIn(legend[0]),
            m.AnimationGroup(m.FadeIn(e3_dots[0], scale=0.5), m.FadeIn(e3_dots[1], scale=0.5), m.FadeIn(e3_first)),
            m.AnimationGroup(m.Create(e3_line, run_time=1.6), m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in e3_dots[2:]], lag_ratio=0.15),
                             m.FadeIn(e3_last, run_time=1.4)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ which drafter

    def choose(self):
        self.next_section("6 · Which drafter?", notes=f"Summary of the four self-drafting families (gains as reported by their authors, on their own models and hardware): prompt lookup — nothing to train, {SAXENA[0]}–{SAXENA[1]}× on input-grounded tasks (Saxena 2023); Medusa — a few trained heads, over {MEDUSA1}× (Medusa-1, frozen model; Cai et al. 2024); EAGLE — one trained layer, {EAGLE_X[0]}–{EAGLE_X[1]}× (70B chat), EAGLE-3 up to {EAGLE3_MAX}× (Li et al. 2024, 2025); multi-token prediction — trained into the model, {DS_TPS}× TPS (DeepSeek-V3) / up to {META_X}× (Gloeckle et al. 2024). All are verified by the target model, as in Episode 13 (Medusa-2's typical acceptance is the lossy exception). Engines: vLLM and SGLang ship n-gram and EAGLE speculators (notes §6, §13).")
        data = (
            ("Prompt lookup", "nothing to train · copies the prompt", f"{SAXENA[0]}–{SAXENA[1]}× when answers quote it", "Saxena, 2023", REQUEST),
            ("Medusa", "a few trained heads", f"over {MEDUSA1}× · same answers", "Cai et al., 2024", WEIGHTS),
            ("EAGLE", "one trained draft layer", f"{EAGLE_X[0]}–{EAGLE_X[1]}× · EAGLE-3 up to {EAGLE3_MAX}×", "Li et al., 2024 · 2025", WEIGHTS),
            ("Multi-token", "trained into the model", f"{DS_TPS}× tokens per second", "DeepSeek-V3, 2024", WEIGHTS),
        )
        pos = ((-3.25, 1.05), (3.25, 1.05), (-3.25, -1.45), (3.25, -1.45))
        tiles = []
        for (name, needs, gain, src, color), (x, y) in zip(data, pos):
            box = card(6.1, 2.2, color, CARD, radius=0.18, stroke=2).move_to([x, y, 0])
            words = m.VGroup(title_text(name, 32, color), text(needs, 22, INK), code(gain, 22, TOKEN), cite(src))
            words.arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT).move_to(box).align_to(box, m.LEFT).shift(0.4 * m.RIGHT)
            words.align_to(box, m.UP).shift(0.28 * m.DOWN)
            tiles.append(m.VGroup(box, words))
        foot = cite("gains as reported by each paper, on its own models and hardware").move_to([0, -3.15, 0])

        self.say(
            "[So which drafter?] [Copying needs no training, and shines when answers quote the prompt.]",
            m.AnimationGroup(self.new_header("06 · choose", "Which drafter?"),
                             m.LaggedStart(*[m.FadeIn(t, shift=0.15 * m.UP) for t in tiles], lag_ratio=0.2), m.FadeIn(foot)),
            m.Circumscribe(tiles[0][0], color=REQUEST, buff=0.06),
        )
        self.say(
            "[Heads and feature drafters need some training, but help on any text]; [multi-token heads come built in.]",
            m.Circumscribe(m.VGroup(tiles[1][0], tiles[2][0]), color=WEIGHTS, buff=0.08),
            m.Circumscribe(tiles[3][0], color=WEIGHTS, buff=0.06),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes=f"Sprout (Medusa and multi-token heads on Owl's own layers; EAGLE's one draft layer on its features), branch (a tree of candidates verified in one pass: {PATHS} continuations from top-2 × top-3), copy (prompt lookup: the notes are a free draft; {PER_PASS} tokens in one pass in the example; copy-the-notes accepted {ALPHA:.0%} of drafts on this Mac, {SPEEDUP:.2f}×). Gains fade at high batch (EAGLE-3 table). Next: Episode 15, FlashAttention.")
        words = [("Sprout", WEIGHTS, "heads on Owl's own layers"),
                 ("Branch", TOKEN, "a tree of guesses, one pass"),
                 ("Copy", REQUEST, "the prompt is a free draft")]
        cols = m.VGroup()
        for w, color, sub in words:
            cols.add(m.VGroup(title_text(w, 66, color), text(sub, 22, MUTED)).arrange(m.DOWN, buff=0.25))
        cols.arrange(m.RIGHT, buff=1.15).move_to([0, 0.75, 0])
        base = 0.85
        for c in cols:
            c[0].shift((base - c[0][0].get_bottom()[1]) * m.UP)
            c[1].move_to([c[0].get_x(), base - 0.55, 0])
        mid = base + 0.3
        arrows = m.VGroup(*[
            arrow([a[0].get_right()[0] + 0.12, mid, 0], [b[0].get_left()[0] - 0.12, mid, 0], MUTED) for a, b in zip(cols, cols[1:])
        ])
        tagline = text("The best draft model can be the model itself, or the notes it was given.", 30, INK).move_to([0, -1.35, 0])
        mascot = owl(0.75).move_to([0, -2.75, 0])

        self.say(
            "[So: sprout], [branch], [and copy.]",
            m.AnimationGroup(self.new_header("recap", "Drafting in three words"), m.FadeIn(cols[0], shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowArrow(arrows[0]), m.FadeIn(cols[1], shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowArrow(arrows[1]), m.FadeIn(cols[2], shift=0.2 * m.UP)),
        )
        self.say(
            "[The best draft model can be the model itself], [or the notes it was given.]",
            m.FadeIn(tagline, shift=0.1 * m.UP),
            m.FadeIn(mascot, shift=0.2 * m.UP),
        )
        self.wait(0.4)
        self.clear()
