"""Tokens per Second · Episode 1 · Press Enter (Module 1 · One Token, Two Speed Limits)

The series opener. A student pastes her earth-science notes into Owl, asks "Why does Earth have seasons?"
and presses Enter; the answer streams back. The episode follows that one request through its six stages
(tokenize → queue → prefill → decode loop → sample → detokenize and stream), then puts it on a timeline to
scale: one thick AMBER prefill bar, then 299 thin CYAN decode ticks. It defines time to first token (TTFT)
and time per output token (TPOT / ITL), shows the stream as server-sent events, checks both clocks against
a run measured on this Mac, steps the course's TinyLM through generate() (prefill once, then decode with the
cache), introduces Owl-8B's three numbers, and ends on the course's question: why is decode so slow, and
what makes it fast? (Episode 2).

Every number on screen is computed below from owl_reference.py or loaded with measured(), and asserted.

Sources (facts dated October 2026; research/notes.md):
- §1 worked example: Llama-3.1-8B BF16 on one H100 SXM, batch 1; 1,500-token prompt, 300-token answer;
  prefill 2·P·n = 24.1 TFLOP at 50% MFU ≈ 49 ms; a decode step reads 16.06 GB of weights + ~0.24 GB of KV
  (1,800 tokens × 128 KiB): 4.86 ms floor, ≈ 6.5 ms at 75% MBU; decode ≈ 97% of the time.
- §1 metrics (Databricks, "LLM Inference Performance Engineering: Best Practices", 2023): TTFT, TPOT,
  ITL, latency = TTFT + TPOT × tokens, throughput = output tokens/s across all users.
- §1 stage 6 and §13: incremental detokenization (a token can end inside a multi-byte UTF-8 character;
  servers buffer until the bytes form valid text) and the OpenAI-compatible SSE stream
  (`data: {"choices":[{"delta":{"content":…}}]}` … `data: [DONE]`), shown simplified.
- H100 SXM: 989 TFLOP/s dense BF16, 3.35 TB/s, 80 GB (NVIDIA spec; owl_reference.HARDWARE).
- Token ids: Llama-3.1-8B-Instruct's tokenizer.json (the one in the local HF cache), read with the
  `tokenizers` library on 2026-10-24 (no model run): "Why does Earth have seasons?" → 10445 1587 9420 617
  15956 30; " ☀" → 26182 (bytes 20 E2 98) + 222 (byte 80).
- Measured on this Mac (research/measurements.md §10, mlx-lm 0.32.2): mlx_lm.server with Owl-8B 4-bit,
  one client: TTFT p50 72 ms (64-token prompt), ITL p50 10.6 ms; 16 clients: 290 tok/s in total, 21 tok/s
  per stream. Ceiling: 546 GB/s ÷ 4.52 GB of 4-bit weights = 8.3 ms per token.

Illustrative (said "for example" or labelled schematic): the student, her notes and the 1,500/300 token
counts (the course's worked example); Owl's answer text; the score bars in the six-stage diagram; an idle
GPU with no queue. The TinyLM run is real (seeded random weights, temperature 1.0).

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

# ---------------------------------------------------------------------------- the worked example (notes §1)

N_PROMPT, N_OUT = 1500, 300
MFU, MBU = 0.5, 0.75                                  # assumed utilisations, as in notes §1
P = OWL["params"]
PREFILL_FLOPS = flops_per_token() * N_PROMPT          # 2 · P · n
T_PREFILL = prefill_time(N_PROMPT, mfu=MFU)           # s
CONTEXT = N_PROMPT + N_OUT                            # KV read per step, upper bound (as in §1)
KV_GB = CONTEXT * kv_bytes_per_token() / 1e9
W_GB = weight_bytes() / 1e9
T_FLOOR = step_time(context=CONTEXT)                  # bytes ÷ bandwidth
T_STEP = T_FLOOR / MBU
STEPS = N_OUT - 1                                     # the first token comes from prefill's scores
T_DECODE = STEPS * T_STEP
T_TOTAL = T_PREFILL + T_DECODE
SHARE = T_DECODE / T_TOTAL
TOK_S = 1 / T_STEP
MATH_US = flops_per_token() / GPU["peak"] * 1e6       # the math of one token at peak
VOCAB = OWL["vocab"]

assert f"{PREFILL_FLOPS / 1e12:.1f}" == "24.1"                         # "twenty-four trillion"
assert round(T_PREFILL * 1000) == 49                                    # "forty-nine milliseconds"
assert f"{KV_GB:.2f}" == "0.24" and f"{W_GB:.2f}" == "16.06"            # "sixteen gigabytes"
assert round(T_FLOOR * 1000, 2) == 4.86                                 # "nearly five milliseconds"
assert round(T_STEP * 1000, 1) == 6.5                                   # "six and a half milliseconds"
assert round(T_DECODE, 2) == 1.94 and round(T_TOTAL, 1) == 2.0          # "about two seconds"
assert 0.97 < SHARE < 0.98 and f"{SHARE:.1%}" == "97.5%"                # "more than ninety-seven percent"
assert N_PROMPT / N_OUT == 5                                            # "five times more tokens"
assert round(TOK_S) == 154                                              # "about a hundred and fifty"
assert round(MATH_US) == 16                                             # "sixteen microseconds"
assert round(P / 1e9) == 8 and kv_bytes_per_token() == 128 * 1024       # the three numbers
assert 0.19 < weight_bytes() / GPU["memory"] < 0.21                     # "a fifth of the memory"
assert VOCAB == 128_256                                                 # "a hundred and twenty-eight thousand"

# ---------------------------------------------------------------------------- measured on this Mac (§10)

SRV = measured("server", "llama-3.1-8b__4bit")
TTFT_MAC = SRV["1"]["ttft_ms"]["p50"]
ITL_MAC, ITL_P99 = SRV["1"]["itl_ms"]["p50"], SRV["1"]["itl_ms"]["p99"]
PROMPT_MAC = SRV["1"]["median_prompt_tokens"]
SOLO_TOK_S = SRV["1"]["per_request_decode_tok_s"]["median"]
AGG16 = SRV["16"]["aggregate_tok_s"]
PER16 = SRV["16"]["per_request_decode_tok_s"]["median"]
W4_GB = measured("decode", "llama-3.1-8b", "4bit", "weights_gb")
FLOOR_MAC_MS = W4_GB * 1e9 / MAC["bandwidth"] * 1000

assert round(TTFT_MAC) == 72 and round(PROMPT_MAC) in (64, 65)          # "seventy-two milliseconds"
assert round(ITL_MAC, 1) == 10.6 and round(ITL_P99, 1) == 11.1          # "ten point six"
assert round(FLOOR_MAC_MS, 1) == 8.3                                    # "eight point three"
assert round(SOLO_TOK_S) == 94 and round(AGG16) == 290 and round(PER16) == 21  # "two hundred and ninety", "twenty-one"

# ---------------------------------------------------------------------------- tokens (Llama-3.1 tokenizer)

QUESTION = "Why does Earth have seasons?"
Q_TOKENS = [("Why", 10445), (" does", 1587), (" Earth", 9420), (" have", 617), (" seasons", 15956), ("?", 30)]
ANSWER_TOKENS = ["Season", "s", " come", " from", " Earth", "'s", " tilt"]   # illustrative answer, real pieces
ANSWER = ("Seasons come from Earth's tilt, not its distance\n"
          "from the Sun. In June, the north leans toward it:\n"
          "longer days, and more direct sunlight.")
SUN_SPLIT = [(26182, b" \xe2\x98"), (222, b"\x80")]
assert "".join(t for t, _ in Q_TOKENS) == QUESTION
assert ANSWER.replace("\n", " ").startswith("".join(ANSWER_TOKENS))
assert (SUN_SPLIT[0][1] + SUN_SPLIT[1][1]).decode() == " ☀"
try:
    SUN_SPLIT[0][1].decode()
    raise AssertionError("the first piece should not be valid UTF-8 on its own")
except UnicodeDecodeError:
    pass

# ---------------------------------------------------------------------------- the TinyLM run (owl_reference)

TINY = TinyLM()
T_PROMPT = [3, 14, 15, 9, 2, 6]
TEMP, N_TINY = 1.0, 4


def softmax(z, t=1.0):
    p = np.exp((z - z.max()) / t)
    return p / p.sum()


with np.errstate(all="ignore"):  # macOS Accelerate raises spurious matmul warnings
    _logits, _cache = TINY.prefill(T_PROMPT)
    PROBS, OUT, CACHE_LEN, _seq = [], [], [len(_cache)], list(T_PROMPT)
    for _ in range(N_TINY):
        PROBS.append(softmax(_logits, TEMP))
        _tok = sample(_logits, TEMP)
        OUT.append(_tok)
        _seq.append(_tok)
        _logits = TINY.decode(_tok, _cache)
        CACHE_LEN.append(len(_cache))
        assert np.allclose(_logits, TINY.forward(_seq, KVCache())[-1])   # cache == recompute, every step
    PROBS.append(softmax(_logits, TEMP))
    assert OUT == generate(TINY, T_PROMPT, N_TINY, temperature=TEMP)
assert OUT == [24, 12, 8, 17] and CACHE_LEN == [6, 7, 8, 9, 10]          # "token twenty-four"; six entries

GEN_SRC = """def generate(model, prompt, max_new_tokens,
             temperature=0.0):
    logits, cache = model.prefill(prompt)
    out = []
    for _ in range(max_new_tokens):
        token = sample(logits, temperature)
        out.append(token)
        logits = model.decode(token, cache)
    return out"""
_real = inspect.getsource(generate)
assert all(line.strip() in _real for line in GEN_SRC.split("\n")) and len(GEN_SRC.split("\n")) <= 10

# ---------------------------------------------------------------------------- look

FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPRSTUVWXYZ0123456789")
STAGES = [("tokenize", MUTED), ("queue", REQUEST), ("prefill", COMPUTE),
          ("decode", MEMORY), ("sample", TOKEN), ("stream", TOKEN)]
GPU_ICON = recolored("gpu", MUTED, CARD_HI, prefix="ep01")
MAC_ICON = recolored("mac", MUTED, CARD_HI, prefix="ep01")
DOC_ICON = recolored("doc", PINK, PINK_FILL, prefix="ep01")


def cite(s, size=20):
    return text(s, size, MUTED)


def baseline(words, s):
    """The baseline of a Text made from s: the median bottom of its flat-bottomed letters (median of all
    glyphs if a ligature changed the glyph count)."""
    chars = [c for c in s if not c.isspace()]
    if len(words) == len(chars):
        bottoms = sorted(g.get_bottom()[1] for g, c in zip(words, chars) if c in FLAT)
        if bottoms:
            return bottoms[len(bottoms) // 2]
    return float(np.median([g.get_bottom()[1] for g in words]))


def sit(words, s, y):
    """Put the baseline of Text `words` (made from s) at height y."""
    return words.shift((y - baseline(words, s)) * m.UP)


def part(words, src, key, start=0):
    """The glyphs of `key` (first at or after `start`) in a mono Text made from src."""
    i = src.index(key, start)
    skip = sum(c.isspace() for c in src[:i])
    return words[i - skip : i - skip + sum(not c.isspace() for c in key)]


def lines_of(words, src):
    out, k = [], 0
    for line in src.split("\n"):
        n = sum(not c.isspace() for c in line)
        out.append(words[k : k + n])
        k += n
    return out


def mono(s, size=20, color=INK, rules=None, **kw):
    """code() with literal-substring colouring (rules: {substring: colour}, in priority order)."""
    return code(s, size, color, t2c={re.escape(k): v for k, v in (rules or {}).items()}, **kw)


def stage_chip(i, label, color, lit, width=1.9, height=0.62):
    box = card(width, height, color if lit else EDGE, FILLS.get(color, CARD) if lit else CARD, radius=0.14,
               stroke=2.2 if lit else 1.5)
    num = code(str(i + 1), 20, color if lit else FAINT, weight=m.BOLD)
    words = text(label, 22, INK if lit else MUTED, weight=m.SEMIBOLD if lit else m.NORMAL)
    body = m.VGroup(num, words).arrange(m.RIGHT, buff=0.16).move_to(box)
    base = box.get_y() - 0.09
    sit(num, str(i + 1), base)
    sit(words, label, base)
    return m.VGroup(box, body)


def bracket(x0, x1, y, color=INK, tick=0.12):
    """A flat measuring bracket from x0 to x1 at height y, its end ticks pointing down."""
    return m.VGroup(
        m.Line([x0, y, 0], [x1, y, 0]),
        m.Line([x0, y, 0], [x0, y - tick, 0]),
        m.Line([x1, y, 0], [x1, y - tick, 0]),
    ).set_stroke(color, 2.5)


def score_bars(seed, pick, n=14, left=0.3, base=-1.7, width=1.9, hmax=1.6, plain=False):
    """Schematic next-token scores (illustrative heights); the picked one GREEN, the tallest."""
    rng = np.random.default_rng(seed)
    h = 0.15 + rng.random(n) ** 2
    h[pick] = h.max() + 0.3
    h = h / h.max() * hmax
    bw = width / n * 0.66
    bars = m.VGroup()
    for i, hi in enumerate(h):
        color = TOKEN if (i == pick and not plain) else MUTED
        r = m.Rectangle(width=bw, height=hi).set_stroke(width=0).set_fill(color, 1 if color == TOKEN else 0.55)
        r.move_to([left + (i + 0.5) * width / n, base + hi / 2, 0])
        bars.add(r)
    return bars


def prob_bars(p, pick=None, left=2.05, base=-1.5, width=4.45, scale=None):
    """TinyLM's real next-token probabilities, one bar per vocabulary entry."""
    n = len(p)
    bw = width / n * 0.7
    bars = m.VGroup()
    for i, pi in enumerate(p):
        hi = max(pi * scale, 0.02)
        color = TOKEN if i == pick else MUTED
        r = m.Rectangle(width=bw, height=hi).set_stroke(width=0).set_fill(color, 1 if i == pick else 0.55)
        r.move_to([left + (i + 0.5) * width / n, base + hi / 2, 0])
        bars.add(r)
    return bars


def code_cursor(box, line_glyphs, line_src, color):
    base = baseline(line_glyphs, line_src)
    top, bot = base + 0.26, base - 0.1
    left, right = box.get_left()[0] + 0.12, box.get_right()[0] - 0.12
    rect = m.Rectangle(width=right - left, height=top - bot, stroke_width=0, fill_color=color, fill_opacity=0.16)
    rect.move_to([(left + right) / 2, (top + bot) / 2, 0])
    mark = m.Rectangle(width=0.05, height=top - bot, stroke_width=0, fill_color=color, fill_opacity=1)
    mark.move_to([left + 0.025, (top + bot) / 2, 0])
    return m.VGroup(rect, mark).set_z_index(1)


def flash_copy(group, color, width=None):
    """Copies of a group's shapes in another colour (for a flash that fades in and out)."""
    out = group.copy()
    for s in out:
        s.set_fill(color, 1).set_stroke(color, width=0)
    return out.set_z_index(3)


# ---------------------------------------------------------------------------- the episode


class Ep01PressEnter(Episode):
    number = 1

    def lesson(self):
        self.hook()
        self.stages()
        self.to_scale()
        self.clocks()
        self.streaming()
        self.in_code()
        self.meet_owl()
        self.question()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="A student pastes her earth-science notes into Owl (the course's study-buddy chatbot), asks 'Why does Earth have seasons?' and presses Enter; the answer streams back token by token. The student, her notes and Owl's answer are illustrative; the topic matches the Owl prompt used for the measurements (research/measurements prompts/chat.txt). What happens between Enter and the last word is the course's story.")
        win = card(9.6, 5.5, EDGE, CARD, radius=0.22, stroke=2).move_to([0, -0.3, 0])
        top = win.get_top()[1]
        rule = m.Line([win.get_left()[0], top - 0.66, 0], [win.get_right()[0], top - 0.66, 0]).set_stroke(EDGE, 2)
        face = owl(0.38).move_to([win.get_left()[0] + 0.55, top - 0.33, 0])
        name = title_text("Owl", 28).next_to(face, m.RIGHT, buff=0.22)
        role = text("study buddy", 22, MUTED).next_to(name, m.RIGHT, buff=0.22)
        sit(name, "Owl", top - 0.43)
        sit(role, "study buddy", top - 0.43)
        tagline = text("notes in · answers and flashcards out", 20, MUTED)
        sit(tagline.align_to([win.get_right()[0] - 0.35, 0, 0], m.RIGHT), tagline.text, top - 0.43)

        field_ = card(8.7, 0.72, EDGE, "#0A1013", radius=0.14, stroke=1.5).move_to([0, win.get_bottom()[1] + 0.56, 0])
        key = m.VGroup(card(1.15, 0.48, MUTED, CARD_HI, radius=0.1, stroke=1.5), code("Enter", 20, INK))
        key[1].move_to(key[0])
        key.move_to(field_).align_to(field_, m.RIGHT).shift(0.14 * m.LEFT)
        typed = text(QUESTION, 26, INK)
        typed.move_to(field_).align_to(field_, m.LEFT).shift(0.3 * m.RIGHT)
        sit(typed, QUESTION, field_.get_y() - 0.1)
        attach = m.VGroup(icon(DOC_ICON, 0.36), code("earth-science-notes.txt", 20, MUTED)).arrange(m.RIGHT, buff=0.14)
        attach_box = card(attach.width + 0.4, 0.52, EDGE, CARD_HI, radius=0.12, stroke=1.5)
        attach = m.VGroup(attach_box, attach.move_to(attach_box))
        attach.next_to(field_, m.UP, buff=0.14).align_to(field_, m.LEFT)
        window = m.VGroup(win, rule, face, name, role, field_, key)

        # the student's message (right) and Owl's answer (left)
        q_words = text(QUESTION, 26, INK)
        u_label = code("student", 22, PINK, weight=m.BOLD)
        u_file = code("earth-science-notes.txt", 20, MUTED)
        u_note = m.VGroup(u_label, icon(DOC_ICON, 0.3), u_file).arrange(m.RIGHT, buff=0.16)
        sit(u_file, u_file.text, baseline(u_label, "student"))
        u_inner = m.VGroup(u_note, q_words).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.16)
        u_box = card(u_inner.width + 0.75, u_inner.height + 0.4, EDGE, CARD_HI, radius=0.14, stroke=1.5)
        u_stripe = m.RoundedRectangle(corner_radius=0.04, width=0.08, height=u_box.height - 0.18, stroke_width=0,
                                      fill_color=PINK, fill_opacity=1).move_to(u_box).align_to(u_box, m.LEFT).shift(0.1 * m.RIGHT)
        u_inner.move_to(u_box).align_to(u_box, m.LEFT).shift(0.35 * m.RIGHT)
        user_msg = m.VGroup(u_box, u_stripe, u_inner)
        user_msg.align_to([win.get_right()[0] - 0.35, 0, 0], m.RIGHT).align_to([0, top - 0.86, 0], m.UP)

        answer = text(ANSWER, 23, INK)
        o_label = code("Owl", 22, GREEN, weight=m.BOLD)
        o_box = card(answer.width + 0.75, answer.height + o_label.height + 0.55, EDGE, CARD_HI, radius=0.14, stroke=1.5)
        o_box.align_to([win.get_left()[0] + 0.35, 0, 0], m.LEFT).align_to([0, user_msg.get_bottom()[1] - 0.2, 0], m.UP)
        o_stripe = m.RoundedRectangle(corner_radius=0.04, width=0.08, height=o_box.height - 0.18, stroke_width=0,
                                      fill_color=GREEN, fill_opacity=1).move_to(o_box).align_to(o_box, m.LEFT).shift(0.1 * m.RIGHT)
        o_label.align_to(o_box, m.UL).shift(0.35 * m.RIGHT + 0.2 * m.DOWN)
        answer.next_to(o_label, m.DOWN, buff=0.14, aligned_edge=m.LEFT)
        owl_msg = m.VGroup(o_box, o_stripe, o_label)

        # the zoom-out: the student's screen on the left, a GPU on the right
        gpu = m.VGroup(card(3.4, 2.6, EDGE, CARD, radius=0.2, stroke=2))
        gpu_body = m.VGroup(icon(GPU_ICON, 1.0), text("a GPU", 26, INK, weight=m.BOLD), code("H100 · Owl-8B", 20, MUTED))
        gpu_body.arrange(m.DOWN, buff=0.16).move_to(gpu[0])
        gpu.add(gpu_body).move_to([4.6, -0.3, 0])
        laptop = m.VGroup(card(3.4, 2.6, EDGE, CARD, radius=0.2, stroke=2))
        lap_body = m.VGroup(owl(0.85), text("Owl", 26, INK, weight=m.BOLD), code("the student's chat", 20, MUTED))
        lap_body.arrange(m.DOWN, buff=0.16).move_to(laptop[0])
        laptop.add(lap_body).move_to([-4.6, -0.3, 0])
        ask = title_text("?", 72, AMBER).move_to([0, -0.3, 0])

        self.say(
            "[A student pastes her earth science notes into Owl], [types a question], [and presses Enter.]",
            m.AnimationGroup(self.new_header("00 · hook", "One question, one answer"), m.FadeIn(window, shift=0.2 * m.UP),
                             m.FadeIn(attach, shift=0.15 * m.UP), lag_ratio=0.15),
            m.AddTextLetterByLetter(typed, time_per_char=0.04),
            m.Succession(
                m.Indicate(key, color=INK, scale_factor=0.86, run_time=0.45),
                m.AnimationGroup(m.FadeOut(typed, shift=0.3 * m.UP), m.FadeOut(attach, shift=0.3 * m.UP),
                                 m.FadeIn(user_msg, shift=0.4 * m.UP), run_time=0.7),
            ),
        )
        self.say(
            "[A moment later], [the answer starts streaming back, token by token.]",
            m.FadeIn(owl_msg, shift=0.2 * m.UP),
            m.AddTextLetterByLetter(answer, time_per_char=0.035),
        )
        self.say(
            "[Owl is this course's running example]: [a study buddy chatbot], [and this one request is our story.]",
            m.Circumscribe(m.VGroup(face, name), color=GREEN, buff=0.1),
            m.FadeIn(tagline, shift=0.1 * m.UP),
            m.Circumscribe(user_msg, color=PINK, buff=0.06),
        )
        screen = m.VGroup(window, tagline, user_msg, owl_msg, answer)
        to_gpu = arrow([-2.95, 0.35, 0], [2.95, 0.35, 0], PINK, buff=0.1)
        back = arrow([2.95, -0.95, 0], [-2.95, -0.95, 0], GREEN, buff=0.1)
        req_lab = code("request", 20, PINK).next_to(to_gpu, m.UP, buff=0.12)
        tok_lab = code("tokens", 20, GREEN).next_to(back, m.DOWN, buff=0.12)
        self.say(
            "[So what really happens], [between Enter and the last word?]",
            m.AnimationGroup(m.FadeOut(screen, target_position=laptop, scale=0.3), m.FadeIn(laptop, scale=2.5, target_position=screen),
                             m.FadeIn(gpu, shift=0.2 * m.LEFT)),
            m.AnimationGroup(m.GrowArrow(to_gpu), m.FadeIn(req_lab), m.GrowArrow(back), m.FadeIn(tok_lab),
                             m.FadeIn(ask, scale=0.6), lag_ratio=0.2),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the six stages

    def stages(self):
        self.next_section("1 · Six stages", notes=f"The six stages of one generation (notes §1): tokenize (CPU, cheap), queue (wait for a slot in the running batch), prefill (one parallel forward pass over all prompt tokens: writes their keys and values into the KV cache and yields the first output token's scores; compute-bound), decode loop (one token per step: reads every weight plus the cache, appends one K/V entry), sample (scores → one token id), detokenize and stream. Token ids are real (Llama-3.1 tokenizer). The prompt size (1,500 tokens with her notes) and the idle GPU are illustrative; the score bars are schematic; the scores cover the whole vocabulary ({VOCAB:,} tokens, Llama-3.1 config).")
        xs = [-5.45 + 2.18 * i for i in range(6)]
        dims = m.VGroup(*[stage_chip(i, s, c, False).move_to([x, 2.25, 0]) for i, ((s, c), x) in enumerate(zip(STAGES, xs))])
        lit = [stage_chip(i, s, c, True).move_to([x, 2.25, 0]).set_z_index(1) for i, ((s, c), x) in enumerate(zip(STAGES, xs))]
        light = lambda i: m.AnimationGroup(m.FadeOut(dims[i]), m.FadeIn(lit[i]))
        frame = m.DashedVMobject(m.RoundedRectangle(corner_radius=0.2, width=xs[5] - xs[3] + 1.9 + 0.22, height=0.84),
                                 num_dashes=60).set_stroke(MEMORY, 2)
        frame.move_to([(xs[3] + xs[5]) / 2, 2.25, 0])
        loop_tag = code("repeats once per token", 20, MEMORY).next_to(frame, m.DOWN, buff=0.12).align_to(frame, m.RIGHT)

        # tokenize
        q = text(QUESTION, 28, INK).move_to([-3.1, 1.2, 0])
        toks = token_row([t for t, _ in Q_TOKENS], REQUEST, size=24, gap=0.16).move_to([-3.1, 1.2, 0])
        ids = m.VGroup(*[code(str(i), 20, MUTED).next_to(c, m.DOWN, buff=0.14) for c, (_, i) in zip(toks, Q_TOKENS)])
        tok_src = cite("Llama 3.1 tokenizer").next_to(m.VGroup(toks, ids), m.RIGHT, buff=0.45).align_to(ids, m.DOWN)

        # queue and model
        req = m.VGroup(card(2.5, 0.6, REQUEST, PINK_FILL, radius=0.14, stroke=2), code(f"prompt · {N_PROMPT:,}", 20, INK))
        req[1].move_to(req[0])
        req.move_to([-3.1, 1.05, 0])
        lane = m.DashedVMobject(m.RoundedRectangle(corner_radius=0.16, width=3.55, height=0.9), num_dashes=46).set_stroke(FAINT, 2)
        lane.move_to([-4.62, -0.75, 0])
        lane_lab = code("queue", 20, MUTED).next_to(lane, m.DOWN, buff=0.14).align_to(lane, m.LEFT)
        mcard = card(2.3, 2.3, WEIGHTS, CARD, radius=0.18, stroke=2.5).move_to([-1.55, -0.75, 0])
        mname = title_text("Owl-8B", 26).move_to(mcard.get_top() + 0.38 * m.DOWN)
        stripes = m.VGroup(*[m.RoundedRectangle(corner_radius=0.04, width=1.7, height=0.13, stroke_width=0, fill_color=WEIGHTS,
                                                fill_opacity=0.85) for _ in range(6)]).arrange(m.DOWN, buff=0.13)
        stripes.move_to(mcard).shift(0.22 * m.DOWN)
        model = m.VGroup(mcard, mname, stripes)
        once = code(f"all {N_PROMPT:,} at once", 20, COMPUTE).next_to(lane, m.UP, buff=0.3).align_to(lane, m.LEFT)

        # cache, scores, sample, screen
        kv_lab = code("KV cache", 20, KV).move_to([-3.6, -2.75, 0])
        kv_block = m.VGroup(m.Rectangle(width=1.35, height=0.44).set_stroke(KV, 1.5).set_fill(VIOLET_FILL, 1),
                            code(f"{N_PROMPT:,}", 20, INK))
        kv_block[0].move_to([mcard.get_left()[0] + 0.675, -2.75, 0])
        kv_block[1].move_to(kv_block[0])
        kv_lab.next_to(kv_block, m.LEFT, buff=0.25)
        cells = m.VGroup(*[m.Rectangle(width=0.3, height=0.44).set_stroke(KV, 1.5).set_fill(VIOLET_FILL, 1) for _ in range(5)])
        for k, c in enumerate(cells):
            c.next_to(kv_block[0], m.RIGHT, buff=0.06 + 0.36 * k)
        plain = score_bars(0, 9, plain=True)
        states = [score_bars(s, pk) for s, pk in zip(range(1, 7), (9, 3, 11, 6, 1, 8))]
        bars = plain
        scores_lab = code(f"{VOCAB:,} scores", 20, MUTED).next_to(plain, m.DOWN, buff=0.16)
        spot = np.array([3.05, -0.75, 0])
        gen = [token_row([t], TOKEN, size=22)[0].move_to(spot).set_z_index(2) for t in ANSWER_TOKENS[:5]]
        panel = card(2.75, 2.75, EDGE, CARD_HI, radius=0.16, stroke=1.5).move_to([5.05, -0.75, 0])
        p_stripe = m.RoundedRectangle(corner_radius=0.04, width=0.08, height=2.5, stroke_width=0, fill_color=GREEN,
                                      fill_opacity=1).move_to(panel).align_to(panel, m.LEFT).shift(0.1 * m.RIGHT)
        p_lab = code("student", 20, MUTED).align_to(panel, m.UL).shift(0.32 * m.RIGHT + 0.18 * m.DOWN)
        shown = token_row(ANSWER_TOKENS[:4], TOKEN, size=22, gap=0.08, max_width=2.25)
        shown.next_to(p_lab, m.DOWN, buff=0.22, aligned_edge=m.LEFT)
        screen = m.VGroup(panel, p_stripe, p_lab)
        entry = mcard.get_top() + np.array([0.55, 0.0, 0])
        path = m.ArcBetweenPoints(spot + 0.36 * m.UP, entry + 0.05 * m.UP, angle=1.7)
        loop_arrow = path.copy().set_stroke(MUTED, 3).add_tip(tip_length=0.18, tip_width=0.16)
        path_from_spot = m.ArcBetweenPoints(spot, entry + 0.05 * m.UP, angle=1.7)

        def sweep():
            flashes = [flash_copy(m.VGroup(s), MEMORY)[0] for s in stripes]
            return m.LaggedStart(*[m.Succession(m.FadeIn(f, run_time=0.12), m.FadeOut(f, run_time=0.22)) for f in flashes],
                                 lag_ratio=0.45)

        def step(k, quick=True):
            """Decode step k: token k streams out and loops back; the cache grows; token k+1 is sampled."""
            rt = 0.55 if quick else 0.9
            return m.Succession(
                m.AnimationGroup(m.FadeIn(shown[k], target_position=gen[k]), m.MoveAlongPath(gen[k], path_from_spot), run_time=rt),
                m.AnimationGroup(m.FadeOut(gen[k], scale=0.4), sweep(), m.GrowFromEdge(cells[k], m.LEFT),
                                 m.Transform(bars, states[k + 1]), m.FadeIn(gen[k + 1], scale=0.6), run_time=rt),
            )

        self.say(
            "[First, tokenize]: [the text is cut into tokens], [and each token becomes a number.]",
            m.AnimationGroup(self.new_header("01 · the trip", "Six stages, Enter to answer"),
                             m.LaggedStart(*[m.FadeIn(d, shift=0.1 * m.DOWN) for d in [lit[0], *dims[1:]]], lag_ratio=0.08),
                             m.FadeIn(q, shift=0.15 * m.UP)),
            m.AnimationGroup(m.FadeOut(q, shift=0.15 * m.DOWN),
                             m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.DOWN) for c in toks], lag_ratio=0.12)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(i, shift=0.1 * m.UP) for i in ids], lag_ratio=0.12), m.FadeIn(tok_src)),
        )
        words = m.VGroup(toks, ids, tok_src)
        self.say(
            "[Next, the whole prompt, fifteen hundred tokens with her notes], [waits in a queue for the GPU]. [This one is free, so it goes straight in.]",
            m.AnimationGroup(light(1), m.FadeOut(words, target_position=req, scale=0.5), m.FadeIn(req, target_position=words)),
            m.AnimationGroup(m.FadeIn(lane), m.FadeIn(lane_lab), m.FadeIn(model, shift=0.15 * m.UP),
                             req.animate.move_to(lane.get_center() + 0.35 * m.LEFT)),
            req.animate.move_to(lane.get_center() + 0.35 * m.RIGHT),
        )
        amber = flash_copy(stripes, COMPUTE)
        self.say(
            "[Then prefill]: [all fifteen hundred tokens go through the model at once], [in one parallel pass.]",
            light(2),
            m.AnimationGroup(m.FadeOut(req, target_position=mcard, scale=0.3),
                             m.Succession(m.FadeIn(amber, run_time=0.35), m.Wait(0.3), m.FadeOut(amber, run_time=0.6))),
            m.FadeIn(once, shift=0.1 * m.UP),
        )
        self.say(
            "[It leaves keys and values for every prompt token in a cache], [and a score for each of the hundred and twenty-eight thousand tokens it knows.]",
            m.AnimationGroup(m.FadeIn(kv_lab), m.GrowFromEdge(kv_block[0], m.LEFT), m.FadeIn(kv_block[1])),
            m.AnimationGroup(m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in bars], lag_ratio=0.06), m.FadeIn(scores_lab)),
        )
        self.say(
            "[Now the decode loop.] [Sample picks one token from those scores]: [the first piece of the answer.]",
            m.AnimationGroup(light(3), m.Create(frame), m.FadeIn(loop_tag), m.FadeOut(once)),
            m.AnimationGroup(light(4), m.Transform(bars, states[0])),
            m.FadeIn(gen[0], target_position=states[0][9], scale=0.5),
        )
        self.say(
            "[It's streamed to the student], [then fed back in], [and the model runs again], [for one more token.]",
            m.AnimationGroup(light(5), m.FadeIn(screen), m.FadeIn(shown[0], target_position=gen[0])),
            m.AnimationGroup(m.Create(loop_arrow), m.MoveAlongPath(gen[0], path_from_spot), run_time=1.0),
            m.AnimationGroup(m.FadeOut(gen[0], scale=0.4), sweep(), m.GrowFromEdge(cells[0], m.LEFT)),
            m.AnimationGroup(m.Transform(bars, states[1]), m.FadeIn(gen[1], scale=0.6)),
        )
        self.say(
            "[Round and round, one token per step]: [each step reads every weight], [and adds one entry to the cache.]",
            m.Succession(step(1), step(2), step(3)),
            m.Circumscribe(stripes, color=MEMORY, buff=0.1),
            m.Circumscribe(m.VGroup(kv_block, cells[:4]), color=KV, buff=0.08),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ to scale

    def to_scale(self):
        self.next_section("2 · To scale", notes=f"Notes §1 worked example, computed with owl_reference: prefill = 2 × {P / 1e9:.2f} B × {N_PROMPT:,} = {PREFILL_FLOPS / 1e12:.1f} TFLOP ÷ ({GPU['peak'] / 1e12:.0f} TFLOP/s × {MFU:.0%} MFU) = {T_PREFILL * 1e3:.1f} ms (prefill_time). One decode step reads {W_GB:.2f} GB of weights + {KV_GB:.2f} GB of KV ({CONTEXT:,} tokens × 128 KiB, the end-of-reply size, as in §1) ÷ {GPU['bandwidth'] / 1e12:.2f} TB/s = {T_FLOOR * 1e3:.2f} ms (step_time); at {MBU:.0%} MBU, {T_STEP * 1e3:.2f} ms. {STEPS} more steps = {T_DECODE:.2f} s; total {T_TOTAL:.2f} s; decode = {SHARE:.1%} of the time although prefill handled 5× more tokens. MFU 50% and MBU 75% are assumptions (§1); H100 SXM spec from NVIDIA (dated October 2026). Attention FLOPs (~2% at this length) are ignored.")
        tags = m.VGroup(
            pill("Owl-8B · BF16", WEIGHTS, BLUE_FILL, 22, mono=True),
            pill("one H100", MUTED, CARD, 22, mono=True),
            pill(f"prompt {N_PROMPT:,} tokens", REQUEST, PINK_FILL, 22, mono=True),
            pill(f"answer {N_OUT} tokens", TOKEN, GREEN_FILL, 22, mono=True),
        ).arrange(m.RIGHT, buff=0.3).move_to([0, 2.2, 0])

        p1s = f"prefill  2 × {P / 1e9:.2f} B params × {N_PROMPT:,} tokens = {PREFILL_FLOPS / 1e12:.1f} TFLOP"
        p2s = f"÷ ({GPU['peak'] / 1e12:.0f} TFLOP/s × {MFU:.0%} MFU) = {T_PREFILL * 1e3:.0f} ms"
        d1s = f"decode   ({W_GB:.2f} GB weights + {KV_GB:.2f} GB KV) ÷ {GPU['bandwidth'] / 1e12:.2f} TB/s = {T_FLOOR * 1e3:.2f} ms"
        d2s = f"÷ {MBU:.0%} MBU = {T_STEP * 1e3:.1f} ms per step"
        p1 = mono(p1s, 22, rules={"prefill": COMPUTE, f"= {PREFILL_FLOPS / 1e12:.1f} TFLOP": COMPUTE})
        p2 = mono(p2s, 22, rules={f"= {T_PREFILL * 1e3:.0f} ms": COMPUTE})
        d1 = mono(d1s, 22, rules={"decode": MEMORY, f"{W_GB:.2f} GB weights": WEIGHTS, f"{KV_GB:.2f} GB KV": KV,
                                   f"= {T_FLOOR * 1e3:.2f} ms": MEMORY})
        d2 = mono(d2s, 22, rules={f"= {T_STEP * 1e3:.1f} ms per step": MEMORY})
        left = -6.2
        p1.align_to([left, 0, 0], m.LEFT)
        d1.align_to([left, 0, 0], m.LEFT)
        sit(p1, p1s, 1.15)
        sit(d1, d1s, -0.05)
        p2.align_to(part(p1, p1s, "2 ×"), m.LEFT)
        d2.align_to(part(d1, d1s, "("), m.LEFT)
        sit(p2, p2s, 0.62)
        sit(d2, d2s, -0.58)
        spec = cite("H100 SXM: NVIDIA spec · 50% MFU and 75% MBU assumed").to_corner(m.DR, buff=0.5)

        tl = Timeline(2.0, width=11.0, origin=(-5.5, -2.25, 0), lane_height=0.7, label_every=0.5)
        pre = tl.bar(0, T_PREFILL, COMPUTE)
        ticks = m.VGroup(*[tl.tick(T_PREFILL + k * T_STEP, MEMORY).set_stroke(MEMORY, 1.8) for k in range(1, STEPS + 1)])
        pre_lab = code(f"prefill {T_PREFILL * 1e3:.0f} ms", 20, COMPUTE).next_to(pre, m.UP, buff=0.22).align_to(pre, m.LEFT)
        dec_lab = code(f"{STEPS} decode steps × {T_STEP * 1e3:.1f} ms = {T_DECODE:.2f} s", 20, MEMORY)
        dec_lab.next_to(ticks, m.UP, buff=0.22).align_to(ticks, m.RIGHT)
        sit(dec_lab, dec_lab.text, baseline(pre_lab, pre_lab.text))

        share_p = stat(f"{1 - SHARE:.1%}", "", f"prefill · {N_PROMPT:,} tokens", COMPUTE, 80)
        share_d = stat(f"{SHARE:.1%}", "", f"decode · {N_OUT} tokens", MEMORY, 80)
        shares = m.VGroup(share_p, share_d).arrange(m.RIGHT, buff=2.2, aligned_edge=m.DOWN).move_to([0, 0.35, 0])

        self.say(
            "[Now, to scale.] [Owl's model on one H100 GPU]: [fifteen hundred tokens in], [three hundred out.]",
            m.AnimationGroup(self.new_header("02 · to scale", "One big pass, then many small steps"), m.FadeIn(tl)),
            m.AnimationGroup(m.FadeIn(tags[0], shift=0.1 * m.UP), m.FadeIn(tags[1], shift=0.1 * m.UP)),
            m.FadeIn(tags[2], shift=0.1 * m.UP),
            m.FadeIn(tags[3], shift=0.1 * m.UP),
        )
        self.say(
            "[Prefill does about two operations per parameter, for every prompt token]: [twenty-four trillion in all.]",
            m.FadeIn(p1[: len(p1) - len(part(p1, p1s, f"= {PREFILL_FLOPS / 1e12:.1f} TFLOP"))], shift=0.1 * m.UP),
            m.FadeIn(part(p1, p1s, f"= {PREFILL_FLOPS / 1e12:.1f} TFLOP"), shift=0.1 * m.UP),
        )
        self.say(
            "[At half the chip's peak], [that's forty-nine milliseconds]: [one thick amber bar.]",
            m.AnimationGroup(m.FadeIn(p2[: len(p2) - len(part(p2, p2s, f"= {T_PREFILL * 1e3:.0f} ms"))], shift=0.1 * m.UP), m.FadeIn(spec)),
            m.FadeIn(part(p2, p2s, f"= {T_PREFILL * 1e3:.0f} ms"), shift=0.1 * m.UP),
            m.AnimationGroup(m.GrowFromEdge(pre, m.LEFT), m.FadeIn(pre_lab, shift=0.1 * m.UP)),
        )
        cut1 = d1s.index("+")
        cut2 = d1s.index("÷")
        n1 = sum(not c.isspace() for c in d1s[:cut1])
        n2 = sum(not c.isspace() for c in d1s[:cut2])
        self.say(
            "[Then every decode step reads all sixteen gigabytes of weights], [plus the cache], [to make one token.]",
            m.FadeIn(d1[:n1], shift=0.1 * m.UP),
            m.FadeIn(d1[n1:n2], shift=0.1 * m.UP),
            m.FadeIn(d1[n2:], shift=0.1 * m.UP),
        )
        self.say(
            "[At a realistic three quarters of the memory bandwidth], [that's six and a half milliseconds a step.]",
            m.FadeIn(d2[: len(d2) - len(part(d2, d2s, f"= {T_STEP * 1e3:.1f} ms per step"))], shift=0.1 * m.UP),
            m.FadeIn(part(d2, d2s, f"= {T_STEP * 1e3:.1f} ms per step"), shift=0.1 * m.UP),
        )
        self.say(
            "[Repeat that for every token of the answer]: [a thin cyan tick each], [almost two seconds of them.]",
            m.LaggedStart(*[m.FadeIn(t) for t in ticks], lag_ratio=0.03, run_time=2.6),
            m.FadeIn(dec_lab, shift=0.1 * m.UP),
            m.Circumscribe(ticks, color=MEMORY, buff=0.08),
        )
        math = m.VGroup(p1, p2, d1, d2)
        self.say(
            "[Prefill handled five times more tokens], [yet decode takes more than ninety-seven percent of the time.]",
            m.AnimationGroup(m.FadeOut(math, shift=0.15 * m.UP), m.FadeIn(share_p, shift=0.15 * m.UP), m.Circumscribe(pre, color=COMPUTE, buff=0.06)),
            m.FadeIn(share_d, shift=0.15 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the two clocks

    def clocks(self):
        self.next_section("3 · Two clocks", notes=f"Databricks (2023): TTFT = how quickly users start seeing output (network + queue + tokenize + prefill + first sample); TPOT = time per output token for each user, benchmarks compute (E2E − TTFT)/(N − 1); ITL = each individual gap (its p99 shows hiccups the mean hides); latency = TTFT + TPOT × tokens. Here: TTFT ≈ {T_PREFILL * 1e3:.0f} ms + queue + network; TPOT ≈ {T_STEP * 1e3:.1f} ms; latency ≈ {T_PREFILL * 1e3:.0f} ms + {STEPS} × {T_STEP * 1e3:.1f} ms = {T_TOTAL:.2f} s; one student sees 1/TPOT ≈ {TOK_S:.0f} tokens/s.")
        mini = Timeline(2.0, width=6.0, origin=(-4.6, 1.95, 0), lane_height=0.42)
        m_pre = mini.bar(0, T_PREFILL, COMPUTE)
        m_dec = mini.bar(T_PREFILL, T_TOTAL, MEMORY)
        m_lab = code(f"the whole reply · {T_TOTAL:.2f} s", 20, MUTED).next_to(mini.axis, m.RIGHT, buff=0.35).shift(0.2 * m.UP)
        zoom = m.Rectangle(width=mini.span * 0.1 / 2.0 + 0.08, height=0.5).set_stroke(INK, 2).set_fill(opacity=0)
        zoom.move_to(mini.x(0.05)).shift(0.02 * m.DOWN)

        ms = T_PREFILL * 1e3
        st = T_STEP * 1e3
        big = Timeline(100, width=11.0, origin=(-5.5, -1.45, 0), lane_height=0.9, label_every=20, unit="ms")
        b_pre = big.bar(0, ms, COMPUTE, label="prefill")
        n_steps = int((100 - ms) // st)
        b_dec = m.VGroup(*[big.bar(ms + k * st + 0.35, ms + (k + 1) * st - 0.35, MEMORY) for k in range(n_steps)])
        dots = m.VGroup(*[m.Dot(big.x(ms + k * st) + np.array([0, 0.48, 0]), radius=0.075, color=TOKEN) for k in range(n_steps + 1)])
        dots.set_z_index(3)
        dec_word = code("decode", 20, MEMORY).next_to(b_dec, m.RIGHT, buff=0.25)
        lines = m.VGroup(
            m.DashedLine(zoom.get_corner(m.DL), big.origin + np.array([0, 1.1, 0]), dash_length=0.08),
            m.DashedLine(zoom.get_corner(m.DR), big.origin + np.array([big.span, 1.1, 0]), dash_length=0.08),
        ).set_stroke(FAINT, 2)
        big_parts = m.VGroup(big, b_pre, b_dec, dots)

        y_br = 0.12
        tt_br = bracket(big.x(0)[0], big.x(ms)[0], y_br, AMBER)
        tt_val = mono(f"TTFT ≈ {ms:.0f} ms", 26, rules={f"≈ {ms:.0f} ms": COMPUTE}, weight=m.BOLD)
        tt_sub = text("time to first token", 22, MUTED)
        tt = m.VGroup(tt_val, tt_sub).arrange(m.DOWN, buff=0.12).next_to(tt_br, m.UP, buff=0.18)
        tt_more = code("+ queue + network", 20, MUTED)
        m.VGroup(tt_val, tt_sub, tt_more).arrange(m.DOWN, buff=0.12).next_to(tt_br, m.UP, buff=0.18)
        tp_br = bracket(big.x(ms + st)[0], big.x(ms + 2 * st)[0], y_br, CYAN)
        tp_val = mono(f"TPOT ≈ {st:.1f} ms", 26, rules={f"≈ {st:.1f} ms": MEMORY}, weight=m.BOLD)
        tp_sub = text("time per output token", 22, MUTED)
        itl = code("ITL: each single gap", 20, MUTED)
        tp = m.VGroup(tp_val, tp_sub, itl).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT).next_to(tp_br, m.UP, buff=0.18)
        tp.align_to(tp_br, m.LEFT).shift(0.3 * m.LEFT)

        fs = f"latency ≈ TTFT + TPOT × (N − 1) = {ms:.0f} ms + {st:.1f} ms × {STEPS} ≈ {T_TOTAL:.2f} s"
        f = mono(fs, 22, rules={f"≈ {T_TOTAL:.2f} s": INK, "TTFT": COMPUTE, "TPOT": MEMORY, f"{ms:.0f} ms": COMPUTE,
                                f"{st:.1f} ms": MEMORY})
        f.move_to([0, -2.55, 0])
        i1, i2, i3 = fs.index("TTFT"), fs.index("+ TPOT"), fs.index("=")
        g = lambda a, b: f[sum(not c.isspace() for c in fs[:a]): sum(not c.isspace() for c in fs[:b])]
        speed_s = f"1 ÷ TPOT ≈ {TOK_S:.0f} tokens/s for this one student"
        speed = mono(speed_s, 22, rules={f"≈ {TOK_S:.0f} tokens/s": TOKEN}).next_to(f, m.DOWN, buff=0.3).align_to(f, m.LEFT)
        src = cite("Databricks, 2023").align_to([6.6, 0, 0], m.RIGHT)
        sit(src, "Databricks 2023", baseline(speed, speed_s))

        self.say(
            "[Now zoom in], [on the first tenth of a second.]",
            m.AnimationGroup(self.new_header("03 · two clocks", "How a reply feels"), m.FadeIn(mini), m.FadeIn(m_pre),
                             m.FadeIn(m_dec), m.FadeIn(m_lab), m.Create(zoom)),
            m.AnimationGroup(m.Create(lines), m.FadeIn(big_parts, target_position=zoom, scale=0.2), m.FadeIn(dec_word)),
        )
        self.say(
            "[The first clock is time to first token]: [the wait before anything appears]. [Here, about fifty milliseconds], [plus any queue.]",
            m.AnimationGroup(m.FadeOut(lines), m.Create(tt_br), m.FadeIn(tt_val, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(tt_sub), m.Flash(dots[0], color=TOKEN, line_length=0.18, flash_radius=0.2)),
            m.Circumscribe(tt_val, color=COMPUTE, buff=0.08),
            m.FadeIn(tt_more, shift=0.1 * m.UP),
        )
        self.say(
            "[The second is time per output token]: [the average gap between tokens], [six and a half milliseconds.]",
            m.AnimationGroup(m.Create(tp_br), m.FadeIn(tp_val, shift=0.1 * m.UP), m.FadeIn(tp_sub)),
            m.AnimationGroup(m.LaggedStart(*[m.Flash(d, color=TOKEN, line_length=0.14, flash_radius=0.16) for d in dots[1:]], lag_ratio=0.25),
                             m.FadeIn(itl)),
            m.Circumscribe(tp_val, color=MEMORY, buff=0.08),
        )
        self.say(
            "[Add them up]: [the first token], [plus two hundred and ninety-nine gaps], [makes about two seconds.]",
            m.FadeIn(g(0, i1), shift=0.1 * m.UP),
            m.FadeIn(g(i1, i2), shift=0.1 * m.UP),
            m.FadeIn(g(i2, i3), shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(g(i3, len(fs)), shift=0.1 * m.UP), m.FadeIn(src)),
        )
        self.say(
            "[Flip the gap around], [and this student sees about a hundred and fifty tokens a second.]",
            m.FadeIn(speed[: sum(not c.isspace() for c in speed_s[: speed_s.index("≈")])], shift=0.1 * m.UP),
            m.FadeIn(speed[sum(not c.isspace() for c in speed_s[: speed_s.index("≈")]):], shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ streaming

    def streaming(self):
        self.next_section("4 · Streaming", notes=f"OpenAI-compatible streaming (notes §13): with stream: true the server returns server-sent events, one chunk per token, `data: {{\"choices\":[{{\"delta\":{{\"content\":…}}}}]}}`, then `data: [DONE]` (shown simplified: no id, object, finish_reason). Incremental detokenization (§1): a token can end inside a multi-byte UTF-8 character; real Llama-3.1 pieces: ' ☀' = token 26182 (bytes 20 E2 98) + token 222 (byte 80). Measured on this Mac (measurements §10, mlx_lm.server, Owl-8B 4-bit, mlx-lm 0.32.2): one client, ~{PROMPT_MAC:.0f}-token prompts: TTFT p50 {TTFT_MAC:.0f} ms (measured from the first connection attempt; the server's prompt cache served ~{SRV['1']['cached_prompt_tokens_median']:.0f} of those prompt tokens, the shared chat-template header), ITL p50 {ITL_MAC:.1f} ms (p99 {ITL_P99:.1f}), {SOLO_TOK_S:.0f} tok/s per stream; ceiling 546 GB/s ÷ {W4_GB:.2f} GB = {FLOOR_MAC_MS:.1f} ms per token. 16 clients: {AGG16:.0f} tok/s in total, {PER16:.1f} tok/s per stream (continuous batching in mlx_lm.server). Throughput = output tokens/s across all users (Databricks, 2023); more in Episode 4.")
        # phase A: the event stream and a split character
        evs = [f'data: {{"choices":[{{"delta":{{"content":"{t}"}}}}]}}' for t in ANSWER_TOKENS[:4]]
        rows = [mono(e, 20, rules={"data:": MUTED, '"choices"': CYAN, '"delta"': CYAN, '"content"': CYAN,
                                   f'"{t}"': GREEN}) for e, t in zip(evs, ANSWER_TOKENS)]
        more = code("⋮", 20, MUTED)
        done = mono("data: [DONE]", 20, rules={"data:": MUTED, "[DONE]": VIOLET})
        body = m.VGroup(*rows, more, done).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.16)
        sse_box = card(body.width + 0.7, body.height + 1.0, EDGE, "#0A1013", radius=0.16, stroke=2)
        sse_cap = code("text/event-stream", 20, FAINT).align_to(sse_box, m.UL).shift(0.35 * m.RIGHT + 0.16 * m.DOWN)
        body.next_to(sse_cap, m.DOWN, buff=0.25, aligned_edge=m.LEFT)
        sse = m.VGroup(sse_box, sse_cap, body).move_to([-2.35, 0.75, 0]).align_to([-6.6, 0, 0], m.LEFT)
        sse_src = cite("OpenAI chat API, stream: true (simplified)").next_to(sse_box, m.DOWN, buff=0.12).align_to(sse_box, m.LEFT)

        chat = card(3.3, 2.55, EDGE, CARD_HI, radius=0.16, stroke=1.5).move_to([4.95, 0.75, 0])
        c_stripe = m.RoundedRectangle(corner_radius=0.04, width=0.08, height=2.3, stroke_width=0, fill_color=GREEN,
                                      fill_opacity=1).move_to(chat).align_to(chat, m.LEFT).shift(0.1 * m.RIGHT)
        c_lab = code("Owl", 22, GREEN, weight=m.BOLD).align_to(chat, m.UL).shift(0.35 * m.RIGHT + 0.2 * m.DOWN)
        c_toks = token_row(ANSWER_TOKENS[:4], TOKEN, size=22, gap=0.08, max_width=2.6)
        c_toks.next_to(c_lab, m.DOWN, buff=0.25, aligned_edge=m.LEFT)
        chat_g = m.VGroup(chat, c_stripe, c_lab)

        u_y = -2.35
        u_lab = text("Some tokens are part of a character:", 22, MUTED)
        sit(u_lab.align_to([-6.6, 0, 0], m.LEFT), u_lab.text, u_y + 0.62)
        t_a = chip(str(SUN_SPLIT[0][0]), TOKEN, GREEN_FILL, width=1.3, height=0.6, size=24, mono=True)
        t_b = chip(str(SUN_SPLIT[1][0]), TOKEN, GREEN_FILL, width=1.0, height=0.6, size=24, mono=True)
        b_a = code(" ".join(f"{b:02X}" for b in SUN_SPLIT[0][1]), 20, MUTED)
        b_b = code(" ".join(f"{b:02X}" for b in SUN_SPLIT[1][1]), 20, MUTED)
        t_a.move_to([-5.95, u_y - 0.15, 0])
        b_a.next_to(t_a, m.RIGHT, buff=0.25)
        hold = pill("half a character: hold", MUTED, CARD, 20, mono=True).next_to(b_a, m.RIGHT, buff=0.3)
        t_b.next_to(hold, m.RIGHT, buff=0.55)
        b_b.next_to(t_b, m.RIGHT, buff=0.25)
        whole = m.VGroup(code("→", 22, MUTED), text("☀", 34, TOKEN), code("sent", 22, TOKEN)).arrange(m.RIGHT, buff=0.18)
        whole.next_to(b_b, m.RIGHT, buff=0.3)
        sit(whole[2], "sent", baseline(t_a[1], t_a[1].text))
        whole[0].match_y(whole[2])
        for words in (b_a, b_b):
            sit(words, words.text, baseline(t_a[1], t_a[1].text))

        # phase B: measured on this Mac
        mac_box = card(6.4, 4.5, EDGE, CARD, radius=0.2, stroke=2).move_to([-3.4, -0.35, 0])
        mac_head = m.VGroup(icon(MAC_ICON, 0.62), m.VGroup(text("measured on this Mac", 24, INK, weight=m.BOLD),
                                                            code("mlx_lm.server · Owl-8B 4-bit", 20, MUTED)).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.1))
        mac_head.arrange(m.RIGHT, buff=0.3).align_to(mac_box, m.UL).shift(0.4 * m.RIGHT + 0.32 * m.DOWN)
        mx = mac_box.get_left()[0] + 0.4

        def mac_row(value, color, label, y):
            v = title_text(value, 60, color)
            u = code("ms", 22, MUTED)
            lab = text(label, 22, MUTED)
            v.move_to([0, y, 0]).align_to([mx, 0, 0], m.LEFT)
            u.next_to(v, m.RIGHT, buff=0.12)
            lab.align_to([mx + 2.55, 0, 0], m.LEFT)
            base = v.get_bottom()[1]
            sit(u, "ms", base)
            sit(lab, label, base)
            return m.VGroup(v, u, lab)

        s_ttft = mac_row(f"{TTFT_MAC:.0f}", INK, f"TTFT · {PROMPT_MAC:.0f}-token prompt", 0.35)
        s_itl = mac_row(f"{ITL_MAC:.1f}", MEMORY, f"ITL · p99 {ITL_P99:.1f} ms", -0.65)
        floor_lab = text("bandwidth limit per token:", 22, MUTED)
        floor_s = f"{W4_GB:.2f} GB ÷ {MAC['bandwidth'] / 1e9:.0f} GB/s = {FLOOR_MAC_MS:.1f} ms"
        floor_t = mono(floor_s, 22, rules={f"= {FLOOR_MAC_MS:.1f} ms": MEMORY})
        floor = m.VGroup(floor_lab, floor_t).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.14)
        floor.move_to([0, -1.75, 0]).align_to([mx, 0, 0], m.LEFT)
        mac_src = cite("research/measurements.md §10 · mlx-lm 0.32.2").next_to(mac_box, m.DOWN, buff=0.12).align_to(mac_box, m.LEFT)

        # throughput: one axis, tokens per second
        x0, unit = 0.55, 4.6 / 300
        chart_t = text("tokens per second", 22, MUTED).move_to([3.95, 1.55, 0]).align_to([x0, 0, 0], m.LEFT)
        rows_spec = [("1 student", SOLO_TOK_S, REQUEST, "her speed"), ("16 students", PER16, REQUEST, "each"),
                     ("16 students", AGG16, TOKEN, "in total")]
        bars, labels, values = m.VGroup(), m.VGroup(), m.VGroup()
        for k, (lab, v, col, what) in enumerate(rows_spec):
            y = 0.75 - 0.95 * k
            b = m.RoundedRectangle(corner_radius=0.06, width=v * unit, height=0.42, stroke_width=0, fill_color=col, fill_opacity=1)
            b.move_to([x0 + v * unit / 2, y, 0])
            bars.add(b)
            labels.add(text(f"{lab} · {what}", 20, MUTED).next_to(b, m.UP, buff=0.1).align_to([x0, 0, 0], m.LEFT))
            values.add(code(f"{v:.0f}", 22, INK, weight=m.BOLD).next_to(b, m.RIGHT, buff=0.15))
        ep4 = pill("more in Episode 4", AMBER, AMBER_FILL, 20).move_to([4.0, -2.35, 0]).align_to([x0, 0, 0], m.LEFT)

        self.say(
            "[Each token goes out the moment it's sampled], [as one small server sent event.]",
            m.AnimationGroup(self.new_header("04 · streaming", "Sent the moment it's sampled"),
                             m.FadeIn(m.VGroup(sse_box, sse_cap)), m.FadeIn(chat_g), m.FadeIn(sse_src)),
            m.Succession(*[m.AnimationGroup(m.FadeIn(r, shift=0.1 * m.UP), m.FadeIn(c, target_position=r),
                                            run_time=0.55) for r, c, t in zip(rows, c_toks, ANSWER_TOKENS)],
                         m.FadeIn(more, run_time=0.3), m.FadeIn(done, shift=0.1 * m.UP, run_time=0.5)),
        )
        self.say(
            "[Some tokens are only part of a character]: [this sun takes two], [so the server holds the first], [until the text is whole.]",
            m.FadeIn(u_lab, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(t_a, shift=0.1 * m.UP), m.FadeIn(b_a), m.FadeIn(t_b, shift=0.1 * m.UP), m.FadeIn(b_b), lag_ratio=0.2),
            m.FadeIn(hold, scale=0.8),
            m.FadeIn(whole, shift=0.1 * m.RIGHT),
        )
        phase_a = m.VGroup(sse, sse_src, chat_g, c_toks, u_lab, t_a, b_a, hold, t_b, b_b, whole)
        self.say(
            "[On this Mac, Owl's four bit model streams through a local server]: [first token in seventy-two milliseconds], [then one every ten point six.]",
            m.AnimationGroup(m.FadeOut(phase_a, shift=0.2 * m.UP), m.FadeIn(mac_box), m.FadeIn(mac_head, shift=0.1 * m.UP), m.FadeIn(mac_src)),
            m.FadeIn(s_ttft, shift=0.1 * m.UP),
            m.FadeIn(s_itl, shift=0.1 * m.UP),
        )
        self.say(
            "[Its bandwidth limit is one token every eight point three milliseconds], [so that's close.]",
            m.FadeIn(floor, shift=0.1 * m.UP),
            m.Circumscribe(m.VGroup(s_itl, floor), color=MEMORY, buff=0.12),
        )
        self.say(
            "[A server also counts throughput]: [tokens per second, summed over every user.]",
            m.FadeIn(chart_t, shift=0.1 * m.UP),
            m.AnimationGroup(m.GrowFromEdge(bars[0], m.LEFT), m.FadeIn(labels[0]), m.FadeIn(values[0])),
        )
        self.say(
            "[With sixteen students at once], [this Mac made two hundred and ninety a second in all], [but each student got just twenty-one.]",
            m.AnimationGroup(m.FadeIn(labels[2]), m.FadeIn(labels[1])),
            m.AnimationGroup(m.GrowFromEdge(bars[2], m.LEFT), m.FadeIn(values[2])),
            m.AnimationGroup(m.GrowFromEdge(bars[1], m.LEFT), m.FadeIn(values[1]), m.FadeIn(ep4, shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the loop in code

    def in_code(self):
        self.next_section("5 · In code", notes=f"owl_reference.generate() (signature wrapped, comments dropped) and TinyLM, run in the scene: TinyLM() (seed 0: 32-token vocabulary, 2 layers, d_model 16, GQA 4 query / 2 KV heads), prompt {T_PROMPT}, generate(tiny, prompt, {N_TINY}, temperature={TEMP}) → {OUT}; the cache grows {CACHE_LEN}. Bars = softmax(logits / T) before each sample (sample() draws from them; temperature 0 would take the argmax). After every decode step the logits equal a full recompute without cache (np.allclose), asserted in the scene. The KV cache itself is Episode 5.")
        block = code_block(GEN_SRC, 20, title="owl_reference.py")
        box, body, cap = block[0], block[1], block[2]
        body.set_z_index(2)
        cap.set_z_index(2)
        block.move_to([0, 0, 0]).align_to([-6.6, 0, 0], m.LEFT).align_to([0, 2.45, 0], m.UP)
        src_lines = GEN_SRC.split("\n")
        glines = lines_of(body, GEN_SRC)
        cur = lambda i, col: code_cursor(box, glines[i], src_lines[i], col)
        bar = cur(2, COMPUTE)
        call_s = f"generate(TinyLM(), prompt, {N_TINY}, temperature={TEMP:g})"
        call = mono(call_s, 20, MUTED).next_to(box, m.DOWN, buff=0.2).align_to(box, m.LEFT)
        check = pill("cache == recompute: True", TOKEN, GREEN_FILL, 20, mono=True).next_to(call, m.DOWN, buff=0.3).align_to(box, m.LEFT)
        ep5 = pill("Episode 5", KV, VIOLET_FILL, 20, mono=True).next_to(check, m.RIGHT, buff=0.3)

        lab_x, item_x = 1.75, 3.0
        title = code("TinyLM · a toy model", 20, MUTED).move_to([4.2, 2.25, 0]).align_to([lab_x, 0, 0], m.LEFT)

        def row_label(s, color, y):
            return code(s, 20, color).move_to([0, y, 0]).align_to([lab_x, 0, 0], m.LEFT)

        y_p, y_c, y_o = 1.5, 0.72, -2.65
        lab_p, lab_c, lab_o = row_label("prompt", REQUEST, y_p), row_label("cache", KV, y_c), row_label("out", TOKEN, y_o)
        p_chips = token_row([str(t) for t in T_PROMPT], REQUEST, size=20, gap=0.07)
        p_chips.move_to([0, y_p, 0]).align_to([item_x, 0, 0], m.LEFT)
        cells = m.VGroup(*[m.Rectangle(width=0.3, height=0.42).set_stroke(KV, 1.5).set_fill(VIOLET_FILL, 1) for _ in range(CACHE_LEN[-1])])
        cells.arrange(m.RIGHT, buff=0.045).move_to([0, y_c, 0]).align_to([item_x, 0, 0], m.LEFT)
        o_chips = token_row([str(t) for t in OUT], TOKEN, size=20, gap=0.07)
        o_chips.move_to([0, y_o, 0]).align_to([item_x, 0, 0], m.LEFT)
        scale = 1.35 / max(p.max() for p in PROBS)
        states = []
        for k, p in enumerate(PROBS):
            pick = OUT[k] if k < len(OUT) else None
            states.append((prob_bars(p, None, left=lab_x, scale=scale), prob_bars(p, pick, left=lab_x, scale=scale)))
        bars = states[0][0]
        p_lab = code("probabilities, 32 tokens", 20, MUTED).next_to(bars, m.DOWN, buff=0.16).align_to([lab_x, 0, 0], m.LEFT)
        p_lab.set_y(-1.82)
        picked = lambda k: states[k][1][OUT[k]]
        def tick(k):
            near = [b.get_top()[1] for b in states[k][1][max(OUT[k] - 2, 0): OUT[k] + 3]]
            words = code(str(OUT[k]), 20, TOKEN, weight=m.BOLD)
            return words.move_to([picked(k).get_x(), max(near) + 0.1 + words.height / 2, 0])
        ticks = [tick(k) for k in range(N_TINY)]

        def one(k, rt=0.5):
            """A whole loop turn: sample token k, append it, decode (cache + 1, new scores)."""
            return m.Succession(
                m.AnimationGroup(m.Transform(bar, cur(5, TOKEN)), m.Transform(bars, states[k][1]), m.FadeIn(ticks[k]), run_time=rt),
                m.AnimationGroup(m.Transform(bar, cur(6, TOKEN)), m.FadeIn(o_chips[k], target_position=picked(k)), run_time=rt),
                m.AnimationGroup(m.Transform(bar, cur(7, MEMORY)), m.FadeOut(ticks[k]), m.FadeIn(cells[6 + k], target_position=o_chips[k]),
                                 m.Transform(bars, states[k + 1][0]), run_time=rt),
            )

        self.say(
            "[Here's that loop in code], [straight from the course's reference file.]",
            m.AnimationGroup(self.new_header("05 · in code", "Prefill once, decode many"), m.FadeIn(block, shift=0.15 * m.UP)),
            m.AnimationGroup(m.FadeIn(title), m.FadeIn(m.VGroup(lab_p, lab_c, lab_o)), m.FadeIn(call)),
        )
        self.say(
            "[Prefill runs the whole prompt through once]: [six tokens into a toy model], [six entries in its cache], [and scores for the next token.]",
            m.FadeIn(bar),
            m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.DOWN) for c in p_chips], lag_ratio=0.12),
            m.LaggedStart(*[m.FadeIn(c, target_position=p) for c, p in zip(cells[:6], p_chips)], lag_ratio=0.1),
            m.AnimationGroup(m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in bars], lag_ratio=0.03), m.FadeIn(p_lab)),
        )
        self.say(
            "[Sample draws one token, weighted by those scores]: [token twenty-four.]",
            m.AnimationGroup(m.Transform(bar, cur(5, TOKEN)), m.Transform(bars, states[0][1])),
            m.AnimationGroup(m.FadeIn(ticks[0], shift=0.1 * m.UP), m.Transform(bar, cur(6, TOKEN)),
                             m.FadeIn(o_chips[0], target_position=picked(0))),
        )
        self.say(
            "[Decode feeds it back in], [the cache grows by one], [and fresh scores come out.]",
            m.AnimationGroup(m.Transform(bar, cur(7, MEMORY)), m.FadeOut(ticks[0])),
            m.FadeIn(cells[6], target_position=o_chips[0]),
            m.Transform(bars, states[1][0]),
        )
        self.say(
            "[Round the loop, one token per step], [until the answer is long enough.]",
            m.Succession(one(1), one(2), one(3)),
            m.AnimationGroup(m.Transform(bar, cur(8, TOKEN)), m.Circumscribe(o_chips, color=TOKEN, buff=0.08)),
        )
        self.say(
            "[The cache is a shortcut, not an approximation]: [the scores match recomputing everything], [exactly.]",
            m.FadeIn(check, shift=0.1 * m.UP),
            m.Circumscribe(cells, color=KV, buff=0.08),
            m.FadeIn(ep5, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ meet Owl

    def meet_owl(self):
        self.next_section("6 · Meet Owl", notes=f"Owl-8B = Llama-3.1-8B (config.json; Llama 3 paper): {P / 1e9:.2f} B parameters; {W_GB:.2f} GB in BF16 (weight_bytes); KV per token 2 × 32 layers × 8 KV heads × 128 × 2 bytes = {kv_bytes_per_token():,.0f} B = 128 KiB (kv_bytes_per_token). On an 80 GB H100: weights {weight_bytes() / GPU['memory']:.0%} of memory; this chat's cache ({CONTEXT:,} tokens) = {KV_GB:.2f} GB. Episode 2 (weights per token), Episode 5 (the cache).")
        mascot = owl(1.15).move_to([-5.45, 0.9, 0])
        s1 = stat(f"{P / 1e9:.0f}", "B", "parameters", WEIGHTS, 84)
        s2 = stat(f"{W_GB:.0f}", "GB", "of weights, in BF16", WEIGHTS, 84)
        s3 = stat(f"{kv_bytes_per_token() / 1024:.0f}", "KiB", "of KV cache per token", KV, 84)
        stats = m.VGroup(s1, s2, s3).arrange(m.RIGHT, buff=0.9, aligned_edge=m.DOWN).move_to([0.95, 0.85, 0])
        model = code(f"Owl-8B = {OWL['name']}", 20, MUTED).next_to(stats, m.UP, buff=0.4).align_to(stats, m.LEFT)

        mem = MemoryBar(GPU["memory"], width=11.6, height=0.6, title="one H100 · 80 GB of memory").move_to([0, -1.6, 0])
        free = GPU["memory"] - weight_bytes() - CONTEXT * kv_bytes_per_token()
        self_add = [mem.set("weights", weight_bytes(), WEIGHTS), mem.set("kv", CONTEXT * kv_bytes_per_token(), KV)]
        w_seg = mem.segments["weights"]
        w_lab = code(f"weights {W_GB:.0f} GB", 20, WEIGHTS).next_to(w_seg, m.DOWN, buff=0.18).align_to(w_seg, m.LEFT)
        w_pct = code(f"{weight_bytes() / GPU['memory']:.0%}", 20, BG, weight=m.BOLD).move_to(w_seg).set_z_index(2)
        kv_seg = mem.segments["kv"]
        kv_x = kv_seg.get_x()
        kv_lab = code(f"this chat's KV cache · {KV_GB:.2f} GB", 20, KV)
        kv_lab.move_to([0, -2.85, 0]).align_to([kv_x + 0.18, 0, 0], m.LEFT)
        kv_lead = m.Line([kv_x, kv_seg.get_bottom()[1] - 0.06, 0], [kv_x, kv_lab.get_y(), 0]).set_stroke(KV, 2)
        kv_lead.add(m.Line([kv_x, kv_lab.get_y(), 0], [kv_x + 0.12, kv_lab.get_y(), 0]).set_stroke(KV, 2))
        free_lab = code(f"free · {free / 1e9:.1f} GB", 20, MUTED).move_to(mem.track).shift(2.2 * m.RIGHT)

        self.say(
            "[Meet Owl's model]: [eight billion parameters], [sixteen gigabytes of weights], [and a hundred and twenty-eight kibibytes of cache for every token.]",
            m.AnimationGroup(self.new_header("06 · meet Owl", "Three numbers to remember"), m.FadeIn(mascot, shift=0.2 * m.UP), m.FadeIn(model)),
            m.FadeIn(s1, shift=0.15 * m.UP),
            m.FadeIn(s2, shift=0.15 * m.UP),
            m.FadeIn(s3, shift=0.15 * m.UP),
        )
        self.say(
            "[On an eighty gigabyte H100], [the weights fill a fifth of the memory]; [this whole chat's cache], [just a sliver.]",
            m.AnimationGroup(m.FadeIn(mem.track), m.FadeIn(mem.caption), m.FadeIn(free_lab)),
            m.AnimationGroup(self_add[0], m.FadeIn(w_lab, shift=0.1 * m.UP), m.FadeIn(w_pct)),
            m.AnimationGroup(self_add[1], m.Create(kv_lead), m.FadeIn(kv_lab, shift=0.1 * m.UP)),
            m.Circumscribe(kv_seg, color=KV, buff=0.08),
        )
        blink = mascot[1].animate(rate_func=m.there_and_back).stretch(0.15, 1)
        self.say(
            "[Keep these three numbers in mind]: [they come back in almost every episode.]",
            m.Circumscribe(stats, color=MUTED, buff=0.2),
            blink,
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the question

    def question(self):
        self.next_section("7 · The question", notes=f"One decode step on one H100 at batch 1: reading {W_GB + KV_GB:.1f} GB at 3.35 TB/s takes {T_FLOOR * 1e3:.2f} ms (the floor); the math, 2·P = {flops_per_token() / 1e9:.1f} GFLOP at 989 TFLOP/s, takes {MATH_US:.0f} µs (notes §2, worked example 1: math units idle ~99.7% of the time). Why decode is slow and what makes it fast is the course's question; Episode 2 starts with the two speed limits.")
        cap = text("one decode step, one H100", 24, MUTED).move_to([0, 2.05, 0]).align_to([-5.3, 0, 0], m.LEFT)
        x0, width = -2.6, 8.4
        r1 = m.RoundedRectangle(corner_radius=0.06, width=width, height=0.5, stroke_width=0, fill_color=MEMORY, fill_opacity=1)
        r1.move_to([x0 + width / 2, 1.15, 0])
        r2 = m.RoundedRectangle(corner_radius=0.02, width=max(width * MATH_US / (T_FLOOR * 1e6), 0.04), height=0.5,
                                stroke_width=0, fill_color=COMPUTE, fill_opacity=1)
        r2.move_to([x0 + r2.width / 2, 0.35, 0])
        l1 = text(f"read {W_GB + KV_GB:.1f} GB", 24, INK).next_to([x0, 1.15, 0], m.LEFT, buff=0.3)
        l2 = text(f"the math, {flops_per_token() / 1e9:.0f} GFLOP", 24, INK).next_to([x0, 0.35, 0], m.LEFT, buff=0.3)
        v1 = code(f"{T_FLOOR * 1e3:.2f} ms", 22, MEMORY, weight=m.BOLD).next_to(r1, m.DOWN, buff=0.1).align_to(r1, m.RIGHT)
        v2 = code(f"{MATH_US:.0f} µs", 22, COMPUTE, weight=m.BOLD).next_to(r2, m.RIGHT, buff=0.2)
        q1 = title_text("Why is decode so slow?", 60, INK).move_to([0, -1.35, 0])
        q2 = title_text("And what makes it fast?", 60, TOKEN).next_to(q1, m.DOWN, buff=0.35)
        nxt = cite("Episode 2 starts here").next_to(q2, m.DOWN, buff=0.3)

        self.say(
            "[In each decode step], [the math takes sixteen microseconds], [but reading the weights takes nearly five milliseconds.]",
            m.AnimationGroup(self.new_header("07 · the question", "The course in one question"), m.FadeIn(cap)),
            m.AnimationGroup(m.FadeIn(l2), m.GrowFromEdge(r2, m.LEFT), m.FadeIn(v2)),
            m.AnimationGroup(m.FadeIn(l1), m.GrowFromEdge(r1, m.LEFT, run_time=1.6), m.FadeIn(v1)),
        )
        self.say(
            "[So here's the question for this whole course]: [why is decode so slow], [and what makes it fast?]",
            m.Circumscribe(r1, color=MEMORY, buff=0.08),
            m.FadeIn(q1, shift=0.2 * m.UP),
            m.AnimationGroup(m.FadeIn(q2, shift=0.2 * m.UP), m.FadeIn(nxt)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Prefill (one parallel, compute-bound pass), decode (one token per step, memory-bound), stream (each token sent as it is sampled).")
        halo = glow(AMBER, radius=4.0)
        words = m.VGroup(*[title_text(w, 80, c) for w, c in (("Prefill", COMPUTE), ("Decode", MEMORY), ("Stream", TOKEN))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("one parallel pass", "one token per step", "as each is sampled")])
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
        dots = m.VGroup(*[m.Dot([(a.get_right()[0] + b.get_left()[0]) / 2, mid, 0], radius=0.07, color=MUTED) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, dots, subs).move_to(0.45 * m.UP)
        tagline = text("One big parallel pass, then hundreds of one-token steps.", 30, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [prefill], [decode], [stream.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say("[One big parallel pass], [then hundreds of one token steps.]",
                 m.Write(tagline), m.Circumscribe(words[1], color=MEMORY, buff=0.12))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
