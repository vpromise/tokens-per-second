"""Tokens per Second · Episode 11 · Chunked Prefill (Module 3 · Many Users at Once)

Sixty students are mid-answer with Owl on one H100 when a sixty-first pastes a 10,000-token chapter. Done in
one step, its prefill takes 327 ms, and in that step nobody else gets a token: a decode stall, 27× the usual
12 ms gap. The episode measures how a student feels it (the gaps between their tokens over a busy minute:
median 12 ms, mean 17.5 ms, p99 327 ms; and ITL percentiles measured on this Mac), then animates the fix: a
token budget per step (512 here), filled with the 60 decodes first and a 452-token slice of the chapter after
(chunked prefill, Sarathi-Serve; on by default in vLLM). Side by side on one clock: gaps of 16.6 ms instead of
one 327 ms hole; the newcomer's first token after 378 ms instead of 327. Then the budget as a dial (gaps vs
first token), the notes' worked example B recomputed (131 ms hiccup → 17.6 ms gaps), and a bridge to
Episode 12 (prefill and decode on separate machines). Episode 10 (continuous batching) is linked by number.

Every number on screen is computed below from owl_reference.py or loaded with measured(), and asserted.

The model of a step (notes §5, worked example B): step time = max(compute at 50% MFU, memory at 80% MBU),
compute = 2 · P FLOPs per token in the step (prefill_time; attention ignored), memory = all weights + every
decoding student's KV cache (step_time). A prompt's slice rides in the same step as the decodes.

Sources (facts dated October 2026; research/notes.md):
- §5 chunked prefill: Sarathi (arXiv 2308.16369) and Sarathi-Serve (Agrawal et al., OSDI 2024, arXiv
  2403.02310): "chunked-prefills which splits a prefill request into near equal sized chunks and creates
  stall-free schedules that adds new requests in a batch without pausing ongoing decodes"; a token budget per
  step, decodes first ("decode-maximal batching").
- vLLM docs, Optimization and Tuning (read 2026-10-24): "In V1, chunked prefill is enabled by default whenever
  possible"; "It batches all pending decode requests before scheduling any prefill operations"; on
  max_num_batched_tokens: "Smaller values (e.g., 2048) achieve better ITL because there are fewer prefills
  slowing down decodes" and "Higher values achieve better time to first token (TTFT)".
- §1 metrics (Databricks 2023): ITL = the distribution of individual gaps; its p99 shows hiccups the mean hides.
- §5 worked example B (8B, H100): 30 users at 2K context, a 4,000-token prompt: 131 ms in one step vs 512-token
  chunks of 17.6 ms (recomputed and asserted below).
- H100 SXM: 989 TFLOP/s dense BF16, 3.35 TB/s, 80 GB (NVIDIA spec; owl_reference.HARDWARE).
- Measured on this Mac (research/measurements.md §10, mlx-lm 0.32.2, 2026-10-24): mlx_lm.server, Owl-8B
  4-bit; one client: ITL p50 10.6 ms (ceiling 546 GB/s ÷ 4.52 GB = 8.3 ms); 16 clients: ITL p50 47.1, p99
  67.4, max 395 ms. The measurement notes: the tail spikes are "most likely steps in which a newcomer's
  prompt was processed inside the decode loop" (server internals not instrumented).

Illustrative (said "say" / "for example", or labelled): the 60 students, their 2,048-token contexts, the
chapter, one chapter arriving every second in the busy minute, the 512-token budget.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from owl_reference import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- the server (notes §5 model)

USERS, CTX, CHAPTER, BUDGET = 60, 2048, 10_000, 512   # illustrative: 60 students 2K tokens in, a 10K chapter
MFU, MBU = 0.5, 0.8                                    # as in notes §5 worked example B
W_GB = weight_bytes() / 1e9
KV_USERS_GB = USERS * CTX * kv_bytes_per_token() / 1e9
T_DEC = step_time(batch=USERS, context=CTX) / MBU      # a decode-only step: weights + 60 caches, 80% MBU
GFLOP = flops_per_token() / 1e9
assert USERS * flops_per_token() / GPU["peak"] < step_time(batch=USERS, context=CTX)   # decode is memory-bound


def step_s(tokens, floor=T_DEC):
    """One step: the math for every token in it (50% MFU), or the memory floor, whichever takes longer."""
    return max(prefill_time(tokens, mfu=MFU), floor)


def plan(budget, prompt=CHAPTER, users=USERS, floor=T_DEC):
    """The slices a prompt is cut into when every step has `budget` tokens and decodes go first
    (budget None: the whole prompt in one step), and each step's time."""
    room = prompt if budget is None else budget - users
    sizes, left = [], prompt
    while left > 0:
        sizes.append(min(room, left))
        left -= sizes[-1]
    return sizes, [step_s(users + c, floor) for c in sizes]


T_BIG = step_s(CHAPTER + USERS)                        # the chapter (and the 60 decodes) in one step
BIG_TFLOP = (CHAPTER + USERS) * flops_per_token() / 1e12
SIZES, TIMES = plan(BUDGET)
CHUNK, T_CHUNK, N_STEPS, TTFT_CH = SIZES[0], TIMES[0], len(SIZES), sum(TIMES)
FREE = T_DEC * GPU["peak"] * MFU / flops_per_token()   # tokens per step that still fit under the memory floor
ms = lambda s: s * 1e3

assert round(W_GB, 2) == 16.06 and round(KV_USERS_GB, 2) == 16.11            # "sixteen gigabytes … plus caches"
assert round(ms(T_DEC), 1) == 12.0                                             # "twelve milliseconds"
assert round(GFLOP, 1) == 16.1 and round(BIG_TFLOP) == 162
assert round(ms(T_BIG)) == 327 and round(T_BIG / T_DEC) == 27                 # "327", "twenty-seven times"
assert 0.3 < T_BIG < 0.35                                                      # "a third of a second"
assert CHUNK == 452 and round(ms(T_CHUNK), 1) == 16.6                          # "452", "sixteen point six"
assert N_STEPS == 23 and round(ms(TTFT_CH)) == 378                             # "twenty-three", "378"
assert SIZES[-1] == CHAPTER - 22 * CHUNK == 56 and round(ms(TIMES[-1]), 1) == 12.0   # step 23: a 56-token chunk
assert prefill_time(USERS + SIZES[-1], mfu=MFU) < T_DEC                        # … so the last step is memory-bound
assert round(FREE, -1) == 370                                                  # "about three hundred and seventy"


def busy_minute(budget, seconds=60.0, every=1.0):
    """Every gap one student sees over a busy minute: a chapter arrives every second (for example: 60 students,
    each pasting one a minute); each step serves all 60 decodes plus as much prompt as the budget allows."""
    t, due, waiting, gaps = 0.0, 0.0, 0, []
    while t < seconds:
        while due <= t + 1e-12:
            waiting += CHAPTER
            due += every
        take = waiting if budget is None else min(waiting, budget - USERS)
        waiting -= take
        gaps.append(step_s(USERS + take))
        t += gaps[-1]
    return np.array(gaps)


G_BIG, G_CH = busy_minute(None), busy_minute(BUDGET)
P50_BIG, P99_BIG = np.percentile(G_BIG, [50, 99])
P50_CH, P99_CH = np.percentile(G_CH, [50, 99])
assert round(ms(P50_BIG), 1) == 12.0 and round(ms(G_BIG.mean()), 1) == 17.5  # "twelve", "seventeen and a half"
assert round(ms(P99_BIG)) == 327 and round(ms(P99_CH), 1) == 16.6             # p99: 327 → "under seventeen"
assert round(ms(P50_CH), 1) == 12.0 and round(ms(G_CH.mean()), 1) == 13.4

# the dial: one budget per doubling, 256 … 16K (16K swallows the chapter in one step)
DIAL = [256, 512, 1024, 2048, 4096, 8192, 16384]
DIAL_GAP = [ms(max(plan(b)[1])) for b in DIAL]
DIAL_TTFT = [ms(sum(plan(b)[1])) for b in DIAL]
DIAL_STEPS = [len(plan(b)[0]) for b in DIAL]
assert round(DIAL_GAP[3], 1) == 66.5 and round(DIAL_TTFT[1] - DIAL_TTFT[3]) == 33   # "sixty-six and a half", "33 sooner"
assert DIAL_GAP[0] == ms(T_DEC) and DIAL_STEPS[0] == 52 and round(DIAL_TTFT[0]) == 624   # below ≈370: free; 52 steps, 624 ms
assert round(DIAL_GAP[-1]) == round(DIAL_TTFT[-1]) == 327                    # one giant step: gap = first token
assert all(a < b for a, b in zip(DIAL_GAP, DIAL_GAP[1:])) and all(a >= b for a, b in zip(DIAL_TTFT, DIAL_TTFT[1:]))

# notes §5 worked example B, recomputed: 30 users at 2K context, a 4,000-token prompt, 512-token chunks
B_USERS, B_PROMPT, B_CHUNK = 30, 4000, 512
B_DEC = step_time(batch=B_USERS, context=CTX) / MBU
B_BIG = step_s(B_PROMPT + B_USERS, B_DEC)
B_SIZES, B_TIMES = plan(B_CHUNK + B_USERS, B_PROMPT, B_USERS, B_DEC)
B_STEP = B_TIMES[0]
assert round(ms(B_DEC), 1) == 9.0 and round(ms(B_BIG)) == 131 and round(B_BIG / B_DEC) == 15   # "131 ms", "~15×"
assert B_SIZES[0] == 512 and len(B_SIZES) == 8 and round(ms(B_STEP), 1) == 17.6                 # "8 steps of 17.6"
assert round((B_PROMPT + B_USERS) * flops_per_token() / 1e12, 1) == 64.7         # "64.7 TFLOP"

# ---------------------------------------------------------------------------- measured on this Mac (§10)

SRV = measured("server", "llama-3.1-8b__4bit")
ITL1 = SRV["1"]["itl_ms"]["p50"]
I16 = SRV["16"]["itl_ms"]
W4_GB = measured("decode", "llama-3.1-8b", "4bit", "weights_gb")
FLOOR_MAC = W4_GB * 1e9 / MAC["bandwidth"] * 1000
assert round(ITL1, 1) == 10.6 and round(FLOOR_MAC, 1) == 8.3                  # "ten point six", "eight point three"
assert round(I16["p50"]) == 47 and round(I16["p99"]) == 67 and round(I16["max"], -2) == 400   # "47", "67", "four tenths"

# ---------------------------------------------------------------------------- look

FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPRSTUVWXYZ0123456789")
GPU_ICON = recolored("gpu", MUTED, CARD_HI, prefix="ep11")
DOC_ICON = recolored("doc", PINK, PINK_FILL, prefix="ep11")
USER_ICON = recolored("user", PINK, PINK_FILL, prefix="ep11")
MAC_ICON = recolored("mac", MUTED, CARD_HI, prefix="ep11")


def cite(s, size=20):
    return text(s, size, MUTED)


def baseline(words, s):
    """The baseline of a Text made from s: the median bottom of its flat-bottomed letters."""
    chars = [c for c in s if not c.isspace()]
    if len(words) == len(chars):
        bottoms = sorted(g.get_bottom()[1] for g, c in zip(words, chars) if c in FLAT)
        if bottoms:
            return bottoms[len(bottoms) // 2]
    return float(np.median([g.get_bottom()[1] for g in words]))


def sit(words, s, y):
    """Put the baseline of Text `words` (made from s) at height y."""
    return words.shift((y - baseline(words, s)) * m.UP)


def mono(s, size=20, color=INK, rules=None, **kw):
    """code() with literal-substring colouring (rules: {substring: colour}, in priority order)."""
    return code(s, size, color, t2c={re.escape(k): v for k, v in (rules or {}).items()}, **kw)


def bracket(x0, x1, y, color=INK, tick=0.12, down=True):
    """A flat measuring bracket from x0 to x1 at height y, its end ticks pointing down (or up)."""
    d = -tick if down else tick
    return m.VGroup(
        m.Line([x0, y, 0], [x1, y, 0]),
        m.Line([x0, y, 0], [x0, y + d, 0]),
        m.Line([x1, y, 0], [x1, y + d, 0]),
    ).set_stroke(color, 2.5)


def student_grid(rows, cols, size, gap, color=REQUEST, fill=PINK_FILL, stroke=2.0):
    return m.VGroup(*[m.RoundedRectangle(corner_radius=size * 0.18, width=size, height=size)
                      .set_stroke(color, stroke).set_fill(fill, 1) for _ in range(rows * cols)]
                    ).arrange_in_grid(rows=rows, cols=cols, buff=gap)


def tinted(group, color, fill, stroke=2.0):
    """Copies of a grid's squares in other colours (a flash or a frozen look laid over it)."""
    out = group.copy()
    for s in out:
        s.set_fill(fill, 1).set_stroke(color, stroke)
    return out.set_z_index(2)


def big_stat(value, unit, label, color, size=60):
    """A big DIN number, its unit (mono, readable size) and a MUTED label under it."""
    number = title_text(value, size, color)
    u = code(unit, 26, MUTED).next_to(number, m.RIGHT, buff=0.14).align_to(number, m.DOWN)
    lab = text(label, 22, MUTED).next_to(number, m.DOWN, buff=0.18).align_to(number, m.LEFT)
    return m.VGroup(number, u, lab)


def blink(overlay, on=0.12, off=0.22):
    return m.Succession(m.FadeIn(overlay, run_time=on), m.FadeOut(overlay, run_time=off))


class Clock:
    """A millisecond axis from x0 to x1: .x(t) maps ms to the scene."""

    def __init__(self, span, x0, x1, y, every=100, size=20):
        self.span, self.x0, self.x1, self.y = span, x0, x1, y
        self.axis = m.Line([x0, y, 0], [x1 + 0.15, y, 0]).set_stroke(FAINT, 2)
        marks = m.VGroup()
        for t in range(0, int(span) + 1, every):
            tick = m.Line([self.x(t), y - 0.07, 0], [self.x(t), y + 0.07, 0]).set_stroke(FAINT, 2)
            marks.add(tick, code(f"{t} ms", size, MUTED).next_to(tick, m.DOWN, buff=0.1))
        self.group = m.VGroup(self.axis, marks)

    def x(self, t):
        return self.x0 + (self.x1 - self.x0) * t / self.span


def block(clock, t0, t1, y, h, color, fill=None, inset=0.025):
    w = max(clock.x(t1) - clock.x(t0) - 2 * inset, 0.02)
    r = m.Rectangle(width=w, height=h).set_stroke(color, 1.5).set_fill(fill or FILLS.get(color, CARD), 1)
    return r.move_to([(clock.x(t0) + clock.x(t1)) / 2, y, 0])


def mixed_block(clock, t0, t1, y, h, prompt_tokens, inset=0.025):
    """One budgeted step: a CYAN sliver for the decodes, the rest AMBER for the prompt slice (by token count)."""
    x0, x1 = clock.x(t0) + inset, clock.x(t1) - inset
    share = USERS / (USERS + prompt_tokens)
    xm = x0 + (x1 - x0) * share
    dec = m.Rectangle(width=max(xm - x0, 0.02), height=h).set_stroke(MEMORY, 1.5).set_fill(CYAN_FILL, 1).move_to([(x0 + xm) / 2, y, 0])
    pre = m.Rectangle(width=max(x1 - xm, 0.02), height=h).set_stroke(COMPUTE, 1.5).set_fill(AMBER_FILL, 1)
    pre.move_to([(xm + x1) / 2, y, 0])
    return m.VGroup(pre, dec)


def schedule(budget, arrive, span):
    """Steps on a clock: decode-only steps until `arrive` ms, the chapter (whole or sliced), then decodes again.
    Returns [(t0, t1, prompt_tokens)]."""
    steps, t = [], 0.0
    while t + 1e-9 < arrive:
        steps.append((t, t + ms(T_DEC), 0))
        t += ms(T_DEC)
    sizes, times = plan(budget)
    for c, d in zip(sizes, times):
        steps.append((t, t + ms(d), c))
        t += ms(d)
    while t + ms(T_DEC) <= span:
        steps.append((t, t + ms(T_DEC), 0))
        t += ms(T_DEC)
    return steps


# ---------------------------------------------------------------------------- the episode


class Ep11ChunkedPrefill(Episode):
    number = 11

    def lesson(self):
        self.hook()
        self.stall()
        self.tail()
        self.budget()
        self.trade()
        self.dial()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes=f"Sixty students mid-answer with Owl on one H100 (illustrative); each decode step gives every one of them one token (continuous batching, Episode 10). A sixty-first student pastes a {CHAPTER:,}-token chapter; done in one step, its prefill takes {ms(T_BIG):.0f} ms (computed in section 1) and nobody else gets a token meanwhile: a decode stall ('generation stall', Sarathi-Serve).")
        grid = student_grid(6, 10, 0.34, 0.12).move_to([-3.7, -0.35, 0])
        who = text(f"{USERS} students, mid-answer", 22, MUTED).next_to(grid, m.UP, buff=0.3).align_to(grid, m.LEFT)
        gpu_box = card(2.7, 2.7, EDGE, CARD, radius=0.2, stroke=2).move_to([1.35, -0.35, 0])
        gpu_body = m.VGroup(icon(GPU_ICON, 0.9), text("one GPU", 26, INK, weight=m.BOLD), code("H100 · Owl-8B", 20, MUTED))
        gpu_body.arrange(m.DOWN, buff=0.16).move_to(gpu_box)
        gpu = m.VGroup(gpu_box, gpu_body)
        mascot = owl(0.42).next_to(gpu_box, m.UP, buff=0.22)
        out = arrow([0.0, -0.35, 0], [-1.42, -0.35, 0], TOKEN)
        out_lab = code("tokens", 20, TOKEN).next_to(out, m.UP, buff=0.12)

        face = icon(USER_ICON, 0.85)
        doc = icon(DOC_ICON, 1.0)
        newbie = m.VGroup(face, doc).arrange(m.RIGHT, buff=0.25).move_to([4.95, -0.3, 0])
        newbie_lab = text("one more student", 22, MUTED).next_to(newbie, m.UP, buff=0.3)
        chapter_lab = code(f"{CHAPTER:,} tokens", 22, PINK, weight=m.BOLD).next_to(newbie, m.DOWN, buff=0.22)
        inbound = arrow([3.95, -0.35, 0], [2.75, -0.35, 0], REQUEST)

        flash = tinted(grid, TOKEN, TOKEN)
        frozen = tinted(grid, FAINT, CARD)
        busy_box = card(2.7, 2.7, COMPUTE, AMBER_FILL, radius=0.2, stroke=2.5).move_to(gpu_box)
        busy_words = m.VGroup(text("prefill", 28, COMPUTE, weight=m.BOLD), code(f"{CHAPTER:,} tokens", 20, INK))
        busy_words.arrange(m.DOWN, buff=0.14).move_to(busy_box).shift(0.3 * m.UP)
        m_track = m.RoundedRectangle(corner_radius=0.1, width=2.0, height=0.2).set_stroke(EDGE, 2).set_fill(CARD, 1)
        m_track.move_to(busy_box).shift(0.65 * m.DOWN)
        m_bar = m.RoundedRectangle(corner_radius=0.1, width=2.0, height=0.2).set_stroke(width=0).set_fill(COMPUTE, 1)
        m_bar.move_to(m_track).set_z_index(4)
        busy = m.VGroup(busy_box, busy_words, m_track).set_z_index(3)
        stopped = mono(f"no tokens for {ms(T_BIG):.0f} ms", 24, RED, weight=m.BOLD).next_to(grid, m.DOWN, buff=0.35)
        ask = title_text("Let it in, without freezing anyone?", 40, INK).move_to([0, -2.85, 0])

        self.say(
            "[Sixty students are studying with Owl at once], [all on one GPU.]",
            m.AnimationGroup(self.new_header("00 · hook", "One paste, sixty frozen streams"),
                             m.LaggedStart(*[m.FadeIn(s, scale=0.6) for s in grid], lag_ratio=0.015), m.FadeIn(who)),
            m.AnimationGroup(m.FadeIn(gpu, shift=0.15 * m.UP), m.FadeIn(mascot, shift=0.15 * m.UP), m.GrowArrow(out),
                             m.FadeIn(out_lab)),
        )
        self.say(
            "Every step, each of them gets one more token of their answer.",
            m.Succession(*[m.AnimationGroup(blink(flash), pulse(out, TOKEN, 8)) for _ in range(5)]),
        )
        self.say(
            "[Then one more student pastes a whole chapter]: [ten thousand tokens.]",
            m.AnimationGroup(m.FadeIn(newbie, shift=0.2 * m.LEFT), m.FadeIn(newbie_lab)),
            m.AnimationGroup(m.FadeIn(chapter_lab, shift=0.1 * m.UP), m.GrowArrow(inbound)),
        )
        self.say(
            "[The GPU takes it in one giant step], [and every other stream stops.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(doc, chapter_lab), target_position=gpu_box, scale=0.4),
                             m.FadeOut(gpu_body), m.FadeIn(busy), m.Succession(m.Wait(0.4), m.GrowFromEdge(m_bar, m.LEFT, run_time=3.2))),
            m.AnimationGroup(m.FadeIn(frozen), m.FadeOut(out_lab), m.FadeOut(out)),
        )
        self.say(
            "[For a third of a second], [not one of the sixty gets a token.]",
            m.FadeIn(stopped, shift=0.1 * m.UP),
            m.Circumscribe(grid, color=RED, buff=0.12),
        )
        out2 = arrow([0.0, -0.35, 0], [-1.42, -0.35, 0], TOKEN)
        self.say(
            "[So how do you let a long prompt in], [without freezing everyone else?]",
            m.AnimationGroup(m.FadeOut(busy), m.FadeOut(m_bar), m.FadeIn(gpu_body), m.FadeOut(frozen), m.FadeOut(stopped), m.GrowArrow(out2)),
            m.AnimationGroup(m.FadeIn(ask, shift=0.15 * m.UP), m.Succession(blink(flash), blink(flash))),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the stall, in numbers

    def stall(self):
        self.next_section("1 · The stall", notes=f"Notes §5 worked-example-B model, at this episode's size: Owl-8B (BF16) on one H100 SXM; {USERS} students decoding at {CTX:,} tokens of context; a decode-only step reads {W_GB:.2f} GB of weights + {USERS} × {CTX:,} × 128 KiB = {KV_USERS_GB:.2f} GB of KV at 80% of 3.35 TB/s = {ms(T_DEC):.1f} ms (step_time / MBU; its math, {USERS} × {GFLOP:.1f} GFLOP, takes {ms(prefill_time(USERS, mfu=MFU)):.1f} ms at 50% MFU, so the step is memory-bound). The chapter's step: ({CHAPTER:,} + {USERS}) tokens × {GFLOP:.1f} GFLOP = {BIG_TFLOP:.0f} TFLOP at 50% of 989 TFLOP/s = {ms(T_BIG):.1f} ms (prefill_time; attention ignored — it would add ≈16% at 10K tokens, notes §11 formula). The students' gap: {ms(T_BIG):.0f} ms instead of {ms(T_DEC):.0f} ms, {T_BIG / T_DEC:.1f}×. Student count, contexts and chapter size are illustrative; MFU 50% and MBU 80% are the notes' assumptions.")
        tags = m.VGroup(
            pill("Owl-8B · BF16 · one H100", WEIGHTS, BLUE_FILL, 22),
            pill(f"{USERS} students, {CTX:,} tokens in", REQUEST, PINK_FILL, 22),
            pill(f"+ a {CHAPTER:,}-token chapter", REQUEST, PINK_FILL, 22),
        ).arrange(m.RIGHT, buff=0.3).move_to([0, 2.2, 0])

        lx, rx = -6.3, 0.45
        l1 = text("a decode step", 24, MEMORY, weight=m.BOLD)
        l2s = f"{W_GB:.2f} GB weights + {KV_USERS_GB:.2f} GB KV"
        l2 = mono(l2s, 20, rules={f"{W_GB:.2f} GB weights": WEIGHTS, f"{KV_USERS_GB:.2f} GB KV": KV})
        l3s = f"÷ (80% × 3.35 TB/s) = {ms(T_DEC):.1f} ms"
        l3 = mono(l3s, 20, rules={f"= {ms(T_DEC):.1f} ms": MEMORY}, weight=m.BOLD)
        r1 = text("the chapter's step", 24, COMPUTE, weight=m.BOLD)
        r2s = f"({CHAPTER:,} + {USERS}) tokens × {GFLOP:.1f} GFLOP"
        r2 = mono(r2s, 20, rules={f"{CHAPTER:,}": REQUEST})
        r3s = f"÷ (50% × 989 TFLOP/s) = {ms(T_BIG):.0f} ms"
        r3 = mono(r3s, 20, rules={f"= {ms(T_BIG):.0f} ms": COMPUTE}, weight=m.BOLD)
        for words, x in ((l1, lx), (l2, lx), (l3, lx), (r1, rx), (r2, rx), (r3, rx)):
            words.align_to([x, 0, 0], m.LEFT)
        sit(l1, l1.text, 1.42)
        sit(r1, r1.text, 1.42)
        sit(l2, l2s, 0.98)
        sit(r2, r2s, 0.98)
        sit(l3, l3s, 0.58)
        sit(r3, r3s, 0.58)
        split = l2s.index("+")
        n_split = sum(not c.isspace() for c in l2s[:split])

        clock = Clock(450, -4.75, 6.15, -2.2)
        lanes = [-0.05, -0.45, -0.85, -1.25]
        names = ["student 1", "student 2", "student 3", f"… student {USERS}"]
        lane_labs = m.VGroup(*[text(n, 20, MUTED).move_to([0, y, 0]).align_to([-5.05, 0, 0], m.RIGHT) for n, y in zip(names, lanes)])
        gpu_y = -1.78
        gpu_lab = code("GPU", 20, MUTED).move_to([0, gpu_y, 0]).align_to([-5.05, 0, 0], m.RIGHT)
        arrive = 5 * ms(T_DEC)
        steps = schedule(None, arrive, 450)
        before = [s for s in steps if s[2] == 0 and s[1] <= arrive + 1e-6]
        big = [s for s in steps if s[2] > 0][0]
        after = [s for s in steps if s[2] == 0 and s[0] >= big[1] - 1e-6]
        blocks_before = m.VGroup(*[block(clock, a, b, gpu_y, 0.4, MEMORY) for a, b, _ in before])
        big_block = block(clock, big[0], big[1], gpu_y, 0.4, COMPUTE)
        big_lab = code("prefill · the whole chapter", 20, COMPUTE).move_to(big_block).set_z_index(2)
        blocks_after = m.VGroup(*[block(clock, a, b, gpu_y, 0.4, MEMORY) for a, b, _ in after])
        dots = lambda ss: m.VGroup(*[m.Dot([clock.x(b), y, 0], radius=0.06, color=TOKEN) for _, b, _ in ss for y in lanes])
        dots_before, dots_big, dots_after = dots(before), dots([big]), dots(after)
        x_a, x_b = clock.x(before[-1][1]), clock.x(big[1])
        band = m.Rectangle(width=x_b - x_a - 0.2, height=1.55).set_stroke(RED, 1.5).set_fill(RED, 0.1)
        band.move_to([(x_a + x_b) / 2, (lanes[0] + lanes[-1]) / 2, 0])
        gap_lab = mono(f"no tokens for {ms(T_BIG):.0f} ms", 24, RED, weight=m.BOLD).move_to(band).shift(0.2 * m.UP)
        gap_sub = text(f"{T_BIG / T_DEC:.0f}× the usual {ms(T_DEC):.0f} ms gap", 22, INK).next_to(gap_lab, m.DOWN, buff=0.16)
        stall_tag = pill("decode stall", RED, RED_FILL, 22).move_to([0, -2.95, 0])

        self.say(
            "[Let's put numbers on it.] [Owl's model on one H100], [sixty students, each about two thousand tokens in], [and a ten thousand token chapter.]",
            self.new_header("01 · the stall", "One step, sixty frozen streams"),
            m.FadeIn(tags[0], shift=0.1 * m.UP),
            m.FadeIn(tags[1], shift=0.1 * m.UP),
            m.FadeIn(tags[2], shift=0.1 * m.UP),
        )
        self.say(
            "[A decode step reads all sixteen gigabytes of weights], [plus everyone's cache]: [twelve milliseconds.]",
            m.AnimationGroup(m.FadeIn(l1, shift=0.1 * m.UP), m.FadeIn(l2[:n_split], shift=0.1 * m.UP)),
            m.FadeIn(l2[n_split:], shift=0.1 * m.UP),
            m.FadeIn(l3, shift=0.1 * m.UP),
        )
        self.say(
            "[Each step gives all sixty students one token], [so their streams tick along.]",
            m.AnimationGroup(m.FadeIn(clock.group), m.FadeIn(lane_labs), m.FadeIn(gpu_lab)),
            m.Succession(*[m.AnimationGroup(m.GrowFromEdge(blk, m.LEFT), m.FadeIn(dots_before[4 * i: 4 * i + 4], scale=0.5),
                                            run_time=0.4) for i, blk in enumerate(blocks_before)]),
        )
        self.say(
            "[The chapter's prefill is ten thousand tokens of math, in one step]: [three hundred and twenty-seven milliseconds.]",
            m.AnimationGroup(m.FadeIn(r1, shift=0.1 * m.UP), m.FadeIn(r2, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(r3, shift=0.1 * m.UP), m.GrowFromEdge(big_block, m.LEFT, run_time=2.2),
                             m.FadeIn(big_lab)),
        )
        self.say(
            "[For all that time], [sixty streams get nothing]: [a gap twenty-seven times the usual.]",
            m.AnimationGroup(m.FadeIn(dots_big, scale=0.5), m.LaggedStart(*[m.GrowFromEdge(b, m.LEFT) for b in blocks_after], lag_ratio=0.3),
                             m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in dots_after], lag_ratio=0.02)),
            m.AnimationGroup(m.FadeIn(band), m.FadeIn(gap_lab, scale=0.9)),
            m.FadeIn(gap_sub, shift=0.1 * m.UP),
        )
        self.say(
            "[This is the decode stall]: [the longer the prompt], [the longer everyone waits.]",
            m.FadeIn(stall_tag, shift=0.1 * m.UP),
            m.Circumscribe(big_block, color=COMPUTE, buff=0.06),
            m.Circumscribe(band, color=RED, buff=0.06),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the tail: how it feels

    def tail(self):
        self.next_section("2 · The gap between tokens", notes=f"Each student gets one token per step, so their gaps are the step times. Simulated busy minute (illustrative: one {CHAPTER:,}-token chapter arrives every second, i.e. each of the {USERS} students pastes one a minute; whole chapter in one step): {len(G_BIG):,} gaps; median (p50) {ms(P50_BIG):.1f} ms, mean {ms(G_BIG.mean()):.1f} ms, p99 {ms(P99_BIG):.0f} ms (np.percentile; {np.mean(G_BIG > T_DEC * 1.001):.1%} of gaps are stalls, more than 1%, so p99 lands on them). The chart shows the minute's first 3 s. ITL = the distribution of individual gaps; 'its p99 shows hiccups that the mean TPOT hides' (notes §1, Databricks 2023). Measured on this Mac (research/measurements.md §10, mlx-lm 0.32.2, mlx_lm.server, Owl-8B 4-bit, ~64-token prompts, 128-token answers): one client ITL p50 {ITL1:.1f} ms vs the ceiling 546 GB/s ÷ {W4_GB:.2f} GB = {FLOOR_MAC:.1f} ms; 16 clients ITL p50 {I16['p50']:.1f}, p99 {I16['p99']:.1f}, max {I16['max']:.0f} ms (n = {I16['n']:,}). The measurement notes call the tail spikes 'most likely steps in which a newcomer's prompt was processed inside the decode loop' (not instrumented).")
        x0, x1, y0, y1, span, top = -5.3, 0.45, -2.0, 1.45, 3.0, 350.0
        cx = lambda t: x0 + (x1 - x0) * t / span
        cy = lambda g: y0 + (y1 - y0) * g / top
        ends = np.cumsum(G_BIG)
        shown = [(t, g) for t, g in zip(ends, G_BIG) if t <= span]
        stems = m.VGroup(*[m.Line([cx(t), y0, 0], [cx(t), cy(ms(g)), 0]).set_stroke(RED if g > T_DEC * 1.001 else MEMORY, 2.2)
                           for t, g in shown])
        spikes = [s for s, (_, g) in zip(stems, shown) if g > T_DEC * 1.001]
        x_axis = m.Line([x0 - 0.05, y0, 0], [x1 + 0.15, y0, 0]).set_stroke(FAINT, 2)
        y_axis = m.Line([x0, y0, 0], [x0, y1 + 0.15, 0]).set_stroke(FAINT, 2)
        xt = m.VGroup(*[code(f"{s} s", 20, MUTED).move_to([cx(s), y0 - 0.3, 0]) for s in range(4)])
        yt = m.VGroup(*[code(f"{g} ms", 20, MUTED).move_to([0, cy(g), 0]).align_to([x0 - 0.15, 0, 0], m.RIGHT) for g in (100, 200, 300)])
        frame_ = m.VGroup(x_axis, y_axis, xt, yt)
        title = text("one student's gaps between tokens", 24, INK, weight=m.BOLD).move_to([0, 2.25, 0]).align_to([-6.3, 0, 0], m.LEFT)
        sub = text("first 3 s of a busy minute · for example, one chapter every second", 20, MUTED)
        sub.next_to(title, m.DOWN, buff=0.14).align_to(title, m.LEFT)
        p99_line = m.DashedLine([x0, cy(ms(P99_BIG)), 0], [x1 + 0.1, cy(ms(P99_BIG)), 0], dash_length=0.1).set_stroke(RED, 2.5)
        p99_tag = code("p99", 20, RED, weight=m.BOLD).next_to(p99_line, m.RIGHT, buff=0.15)

        col = 1.95
        s50 = big_stat(f"{ms(P50_BIG):.1f}", "ms", "median gap · p50", MEMORY, 60).move_to([0, 1.05, 0]).align_to([col, 0, 0], m.LEFT)
        smean = big_stat(f"{ms(G_BIG.mean()):.1f}", "ms", "mean gap", INK, 60).move_to([0, -0.3, 0]).align_to([col, 0, 0], m.LEFT)
        s99 = big_stat(f"{ms(P99_BIG):.0f}", "ms", "p99 · 1 token in 100 waits this long", RED, 60).move_to([0, -1.65, 0]).align_to([col, 0, 0], m.LEFT)
        stats = m.VGroup(s50, smean, s99)

        # measured on this Mac
        box = card(4.9, 4.75, EDGE, CARD, radius=0.2, stroke=2).move_to([4.05, -0.4, 0])
        bx = box.get_left()[0] + 0.35
        head = m.VGroup(icon(MAC_ICON, 0.55), m.VGroup(text("measured on this Mac", 24, INK, weight=m.BOLD),
                                                        text("mlx_lm.server · Owl-8B 4-bit", 20, MUTED)).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.08))
        head.arrange(m.RIGHT, buff=0.25).align_to(box, m.UL).shift(0.35 * m.RIGHT + 0.3 * m.DOWN)
        one_lab = text("one student alone", 22, PINK, weight=m.BOLD)
        one_val = mono(f"{ITL1:.1f} ms per token", 22, rules={f"{ITL1:.1f} ms": MEMORY}, weight=m.BOLD)
        one_ceil = text(f"ceiling: {W4_GB:.2f} GB ÷ 546 GB/s = {FLOOR_MAC:.1f} ms", 20, MUTED)
        many_lab = text("sixteen students at once", 22, PINK, weight=m.BOLD)
        for words, y in ((one_lab, 0.75), (one_val, 0.33), (one_ceil, -0.05), (many_lab, -0.6)):
            words.align_to([bx, 0, 0], m.LEFT)
            sit(words, words.text, y)

        def mini(value, label, color, x):
            v = title_text(value, 40, color)
            u = code("ms", 24, MUTED).next_to(v, m.RIGHT, buff=0.08).align_to(v, m.DOWN)
            lab = code(label, 24, MUTED).next_to(v, m.DOWN, buff=0.14).align_to(v, m.LEFT)
            return m.VGroup(v, u, lab).move_to([x, -1.35, 0], aligned_edge=m.LEFT)

        m50 = mini(f"{I16['p50']:.1f}", "p50", MEMORY, bx)
        m99 = mini(f"{I16['p99']:.1f}", "p99", RED, bx + 1.5)
        mmax = mini(f"{I16['max']:.0f}", "max", RED, bx + 3.0)
        why = m.VGroup(text("spikes: most likely prompts", 20, MUTED),
                       text("prefilled inside the decode loop", 20, MUTED)).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.08)
        why.align_to([bx, 0, 0], m.LEFT).set_y(-2.35)
        src = cite("research/measurements.md §10").next_to(box, m.DOWN, buff=0.12).align_to(box, m.LEFT)

        self.say(
            "[How does that feel to a student?] [Plot every gap between their tokens], [over a busy minute.]",
            m.AnimationGroup(self.new_header("02 · the gap between tokens", "The average hides it"), m.FadeIn(title, shift=0.1 * m.UP)),
            m.FadeIn(frame_),
            m.AnimationGroup(m.FadeIn(sub), m.LaggedStart(*[m.GrowFromEdge(s, m.DOWN) for s in stems], lag_ratio=0.02, run_time=2.4)),
        )
        self.say(
            "[Say each student pastes a chapter about once a minute]: [that's one long prompt every second.]",
            m.Circumscribe(sub, color=MUTED, buff=0.08),
            m.LaggedStart(*[m.Flash(s.get_end(), color=RED, line_length=0.16, flash_radius=0.2) for s in spikes], lag_ratio=0.3),
        )
        self.say(
            "[Most gaps are twelve milliseconds, the median.] [Even the average looks fine: seventeen and a half.]",
            m.FadeIn(s50, shift=0.1 * m.UP),
            m.FadeIn(smean, shift=0.1 * m.UP),
        )
        self.say(
            "[But the ninety-ninth percentile], [the gap that one token in a hundred waits for], [is three hundred and twenty-seven.]",
            m.AnimationGroup(m.Create(p99_line), m.FadeIn(p99_tag)),
            m.FadeIn(s99[2], shift=0.1 * m.UP),
            m.FadeIn(s99[:2], shift=0.1 * m.UP),
        )
        self.say(
            "[On this Mac], [one student alone gets a token every ten point six milliseconds], [near the eight point three millisecond ceiling.]",
            m.AnimationGroup(m.FadeOut(stats, shift=0.2 * m.UP), m.FadeIn(box), m.FadeIn(head, shift=0.1 * m.UP), m.FadeIn(src)),
            m.AnimationGroup(m.FadeIn(one_lab), m.FadeIn(one_val, shift=0.1 * m.UP)),
            m.FadeIn(one_ceil, shift=0.1 * m.UP),
        )
        self.say(
            "[With sixteen students], [the median gap was forty-seven milliseconds], [the ninety-ninth percentile sixty-seven], [and the worst, four tenths of a second.]",
            m.FadeIn(many_lab, shift=0.1 * m.UP),
            m.FadeIn(m50, shift=0.1 * m.UP),
            m.FadeIn(m99, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(mmax, shift=0.1 * m.UP), m.FadeIn(why)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the budget: decodes first, then a slice

    def budget(self):
        self.next_section("3 · The token budget", notes=f"Chunked prefill with stall-free scheduling (Sarathi-Serve, Agrawal et al., OSDI 2024): every step has a token budget ({BUDGET}, illustrative); the {USERS} decodes go first (one token each), a chunk of the waiting prompt fills the remaining {CHUNK} ('decode-maximal batching'). The chapter becomes {N_STEPS} chunks ({N_STEPS - 1} × {CHUNK} + {SIZES[-1]}); each slice's keys and values join the cache, and each later chunk attends to the chunks already cached. Each full step: {BUDGET} tokens × {GFLOP:.1f} GFLOP at 50% MFU = {ms(T_CHUNK):.1f} ms (now compute-bound; the decode-only floor is {ms(T_DEC):.1f} ms); the last, smaller step {ms(TIMES[-1]):.1f} ms. All {N_STEPS} steps: {ms(TTFT_CH):.1f} ms, then the chapter's first token. vLLM V1: 'chunked prefill is enabled by default whenever possible'; the scheduler 'batches all pending decode requests before scheduling any prefill operations' (vLLM docs, Optimization and Tuning). Episode 10: continuous batching re-forms the batch every step.")
        L, R = -5.6, 5.6
        W = R - L
        tx = lambda tokens: L + W * tokens / CHAPTER
        edges = np.concatenate([[0], np.cumsum(SIZES)])
        chap_y, kv_y = 1.72, -1.45

        def pieces(y, color, opacity):
            return m.VGroup(*[m.Rectangle(width=max(tx(b) - tx(a) - 0.03, 0.03), height=0.4).set_stroke(width=0)
                              .set_fill(color, opacity).move_to([(tx(a) + tx(b)) / 2, y, 0]) for a, b in zip(edges, edges[1:])])

        def track(y):
            return m.Rectangle(width=W + 0.1, height=0.5).set_stroke(EDGE, 1.5).set_fill(CARD, 1).move_to([0, y, 0])

        chap_track, chap = track(chap_y), pieces(chap_y, REQUEST, 0.9)
        kv_track, kv = track(kv_y), pieces(kv_y, KV, 1)
        chap_lab = text(f"the chapter · {CHAPTER:,} prompt tokens, waiting", 22, MUTED)
        chap_lab.align_to([L, 0, 0], m.LEFT)
        sit(chap_lab, chap_lab.text, 2.15)

        def counter(n, y_base, color, word):
            words = mono(f"{n:,} {word}", 20, color)
            return sit(words.align_to([R, 0, 0], m.RIGHT), f"{n:,} {word}", y_base)

        left_ctr = counter(CHAPTER, 2.15, REQUEST, "left")
        kv_lab = m.VGroup(code("KV cache", 22, KV, weight=m.BOLD), text("· the chapter so far", 22, MUTED)).arrange(m.RIGHT, buff=0.15)
        kv_lab.align_to([L, 0, 0], m.LEFT)
        sit(kv_lab[0], "KV cache", -0.98)
        sit(kv_lab[1], "· the chapter so far", -0.98)
        kv_ctr = counter(0, -0.98, KV, "tokens")

        # one step: the budget track, decodes first, then a slice
        TL, TR, ty, th = L, 3.4, 0.38, 0.68
        TW = TR - TL
        slot_x0 = TL + TW * USERS / BUDGET
        outline = m.Rectangle(width=TW + 0.1, height=th + 0.12).set_stroke(EDGE, 2).set_fill(CARD, 1).move_to([(TL + TR) / 2, ty, 0])
        dec_rect = m.Rectangle(width=slot_x0 - TL - 0.03, height=th).set_stroke(MEMORY, 1.5).set_fill(CYAN_FILL, 1)
        dec_rect.move_to([(TL + slot_x0) / 2, ty, 0])
        dec_num = code(f"{USERS}", 22, INK, weight=m.BOLD).move_to(dec_rect).set_z_index(3)
        slot = m.DashedVMobject(m.Rectangle(width=TR - slot_x0 - 0.03, height=th), num_dashes=70).set_stroke(FAINT, 1.5)
        slot.move_to([(slot_x0 + TR) / 2, ty, 0])
        fills = [m.Rectangle(width=TR - slot_x0 - 0.03, height=th).set_stroke(COMPUTE, 1.5).set_fill(AMBER_FILL, 1)
                 .move_to(slot) for _ in range(2)]
        slot_lab = code(f"chunk · {CHUNK} prompt tokens", 22, INK).move_to(slot).set_z_index(3)
        step_word = code("step 1", 22, INK, weight=m.BOLD)
        budget_word = text(f"· a budget of {BUDGET} tokens", 22, MUTED)
        step_row = m.VGroup(step_word, budget_word).arrange(m.RIGHT, buff=0.18).align_to([L, 0, 0], m.LEFT)
        sit(step_word, "step 1", 0.98)
        sit(budget_word, budget_word.text, 0.98)
        step_words = [code(f"step {k}", 22, INK, weight=m.BOLD).move_to(step_word, aligned_edge=m.LEFT) for k in range(1, N_STEPS + 1)]
        for w_ in step_words:
            sit(w_, w_.text, 0.98)
        budget_word.next_to(step_words[-1], m.RIGHT, buff=0.18)
        sit(budget_word, budget_word.text, 0.98)
        dec_lab = text("decodes first, one per student", 20, MEMORY)
        dec_lab.align_to([L, 0, 0], m.LEFT)
        sit(dec_lab, dec_lab.text, -0.33)
        time_s = f"one step ≈ {ms(T_CHUNK):.1f} ms"
        time_lab = mono(time_s, 20, MUTED, rules={f"{ms(T_CHUNK):.1f} ms": INK}).align_to([TR, 0, 0], m.RIGHT)
        sit(time_lab, time_s, -0.33)

        grid = student_grid(6, 10, 0.12, 0.045, stroke=1.2).move_to([5.45, ty, 0])
        grid_lab = text(f"{USERS} students", 20, MUTED).move_to([5.45, 0, 0])
        sit(grid_lab, grid_lab.text, -0.33)
        flash = tinted(grid, TOKEN, TOKEN, stroke=1.2)
        feed = arrow([TR + 0.1, ty, 0], [grid.get_left()[0] - 0.02, ty, 0], TOKEN)

        ep10 = pill("Episode 10 · re-form the batch every step", REQUEST, PINK_FILL, 20).move_to([0, -2.55, 0]).align_to([L, 0, 0], m.LEFT)
        question = text("now: what goes into each step?", 24, INK).next_to(ep10, m.RIGHT, buff=0.35)
        first = pill(f"its first token · after {ms(TTFT_CH):.0f} ms", TOKEN, GREEN_FILL, 22, mono=True).move_to([0, chap_y, 0])
        name = title_text("chunked prefill", 46, COMPUTE).move_to([0, -2.6, 0]).align_to([L, 0, 0], m.LEFT)
        c1 = cite("Sarathi-Serve · Agrawal et al., OSDI 2024")
        c2 = cite("vLLM: on by default whenever possible")
        cites = m.VGroup(c1, c2).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.1).next_to(name, m.RIGHT, buff=0.6)

        def into_slot(k, fill):
            return m.AnimationGroup(m.FadeOut(chap[k], target_position=fill), m.FadeIn(fill, target_position=chap[k]),
                                    m.Transform(left_ctr, counter(int(CHAPTER - edges[k + 1]), 2.15, REQUEST, "left")))

        def into_cache(k, fill):
            return m.AnimationGroup(m.FadeOut(fill, target_position=kv[k]), m.FadeIn(kv[k], target_position=fill),
                                    m.Transform(kv_ctr, counter(int(edges[k + 1]), -0.98, KV, "tokens")))

        self.say(
            "[Episode ten re-formed the batch at every step.] [The question now is what goes into each step.]",
            m.AnimationGroup(self.new_header("03 · the budget", "Decodes first, then a chunk"), m.FadeIn(ep10, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(question), m.FadeIn(chap_track), m.FadeIn(chap_lab), m.FadeIn(left_ctr),
                             m.LaggedStart(*[m.FadeIn(p) for p in chap], lag_ratio=0.05)),
        )
        self.say(
            "[Give every step a token budget], [say five hundred and twelve tokens.]",
            m.AnimationGroup(m.FadeIn(outline), m.FadeIn(slot), m.FadeIn(step_word)),
            m.AnimationGroup(m.FadeIn(budget_word, shift=0.1 * m.UP), m.Circumscribe(outline, color=MUTED, buff=0.06)),
        )
        self.say(
            "[Decodes go first]: [sixty students], [one token each.]",
            m.AnimationGroup(m.GrowFromEdge(dec_rect, m.LEFT), m.FadeIn(dec_num)),
            m.AnimationGroup(m.FadeIn(grid), m.FadeIn(grid_lab)),
            m.AnimationGroup(m.FadeIn(dec_lab, shift=0.1 * m.UP), m.GrowArrow(feed)),
        )
        self.say(
            "[A chunk of the chapter fills the rest]: [four hundred and fifty-two prompt tokens.]",
            into_slot(0, fills[0]),
            m.AnimationGroup(m.FadeIn(slot_lab), m.FadeIn(time_lab, shift=0.1 * m.UP)),
        )
        self.say(
            "[The step runs]: [every student gets a token], [and the chunk's keys and values land in the cache.]",
            m.ShowPassingFlash(m.Line(outline.get_left(), outline.get_right()).set_stroke(INK, 8), time_width=0.6, run_time=0.9),
            m.AnimationGroup(blink(flash), pulse(feed, TOKEN, 8)),
            m.AnimationGroup(m.FadeIn(kv_track), m.FadeIn(kv_lab), m.FadeIn(kv_ctr), into_cache(0, fills[0])),
        )
        self.say(
            "[Next step, the same again]: [sixty decodes first], [then the next chunk], [which attends to the chunks already cached.]",
            m.Transform(step_word, step_words[1]),
            m.Circumscribe(dec_rect, color=MEMORY, buff=0.05),
            into_slot(1, fills[1]),
            m.Succession(m.Circumscribe(kv[0], color=KV, buff=0.05, run_time=0.8),
                         m.AnimationGroup(blink(flash), into_cache(1, fills[1]))),
        )
        # the last step: the remaining 56 tokens fill only part of the slot, and the step is memory-bound again
        last = N_STEPS - 1
        last_fill = m.Rectangle(width=(TR - slot_x0 - 0.03) * SIZES[-1] / CHUNK, height=th).set_stroke(COMPUTE, 1.5)
        last_fill.set_fill(AMBER_FILL, 1).move_to(slot).align_to(slot, m.LEFT)
        last_lab = code(f"chunk · {SIZES[-1]} prompt tokens", 22, INK).set_z_index(3)
        last_lab.move_to(slot_lab).align_to([last_fill.get_right()[0] + 0.35, 0, 0], m.LEFT)
        last_s = f"≈ {ms(TIMES[-1]):.1f} ms · memory-bound"
        last_time = mono(last_s, 20, MUTED, rules={f"{ms(TIMES[-1]):.1f} ms": INK, "memory-bound": MEMORY}).align_to([TR, 0, 0], m.RIGHT)
        sit(last_time, last_s, -0.33)
        fast = []
        for k in range(2, last):
            fast.append(m.AnimationGroup(m.FadeOut(chap[k], target_position=kv[k]), m.FadeIn(kv[k], target_position=chap[k]),
                                         blink(flash, 0.03, 0.05), m.Transform(step_word, step_words[k]),
                                         m.Transform(left_ctr, counter(int(CHAPTER - edges[k + 1]), 2.15, REQUEST, "left")),
                                         m.Transform(kv_ctr, counter(int(edges[k + 1]), -0.98, KV, "tokens")), run_time=0.08))
        fast.append(m.AnimationGroup(m.FadeOut(chap[last], target_position=last_fill), m.FadeIn(last_fill, target_position=chap[last]),
                                     m.Transform(step_word, step_words[last]), m.Transform(left_ctr, counter(0, 2.15, REQUEST, "left")),
                                     m.Transform(slot_lab, last_lab), m.Transform(time_lab, last_time), run_time=0.5))
        fast.append(m.AnimationGroup(m.FadeIn(kv[last], target_position=last_fill), blink(flash),
                                     m.Transform(kv_ctr, counter(CHAPTER, -0.98, KV, "tokens")), run_time=0.45))
        self.say(
            "[Twenty-three steps later], [the whole chapter is in the cache], [and its first token streams back.]",
            m.Succession(*fast),
            m.Circumscribe(m.VGroup(kv_track), color=KV, buff=0.06),
            m.FadeIn(first, scale=0.8),
        )
        self.say(
            "[This is chunked prefill], [from the Sarathi Serve paper], [and vLLM now does it by default.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(ep10, question)), m.FadeIn(name, shift=0.1 * m.UP)),
            m.FadeIn(c1, shift=0.1 * m.UP),
            m.FadeIn(c2, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the trade, on one clock

    def trade(self):
        self.next_section("4 · The trade", notes=f"Both schedules on one clock (the chapter arrives after {5} decode steps, {ms(5 * T_DEC):.0f} ms). One giant step: {ms(T_BIG):.1f} ms with no token for anyone, then the chapter's first token. Budget {BUDGET}: {N_STEPS - 1} steps of {ms(T_CHUNK):.1f} ms + one of {ms(TIMES[-1]):.1f} ms; every step still gives each of the {USERS} students a token (the CYAN sliver = {USERS} decode tokens, the AMBER part = the slice, widths by token count). Busy minute (section 2's simulation): p99 gap {ms(P99_BIG):.0f} → {ms(P99_CH):.1f} ms (mean {ms(G_BIG.mean()):.1f} → {ms(G_CH.mean()):.1f} ms, median {ms(P50_CH):.1f}). The newcomer's TTFT (from arrival, queue and network aside): {ms(T_BIG):.0f} → {ms(TTFT_CH):.0f} ms, because each step also does the {USERS} decodes' math ({USERS} × {GFLOP:.1f} GFLOP ≈ {ms(prefill_time(USERS, mfu=MFU)):.1f} ms per step) and the last slice waits for the memory floor.")
        clock = Clock(460, -6.0, 5.8, -0.8)
        arrive = 5 * ms(T_DEC)
        ya_dot, ya_blk, yb_dot, yb_blk = 1.88, 1.42, 0.34, -0.12

        def lane(budget, y_dot, y_blk):
            steps = schedule(budget, arrive, 460)
            blocks = m.VGroup()
            for a, b, c in steps:
                if c == 0:
                    blocks.add(block(clock, a, b, y_blk, 0.4, MEMORY))
                elif budget is None:
                    blocks.add(block(clock, a, b, y_blk, 0.4, COMPUTE))
                else:
                    blocks.add(mixed_block(clock, a, b, y_blk, 0.4, c))
            dots = m.VGroup(*[m.Dot([clock.x(b), y_dot, 0], radius=0.065, color=TOKEN) for _, b, _ in steps])
            done = [b for a, b, c in steps if c > 0][-1]
            return steps, blocks, dots, done

        sa, ba, da, done_a = lane(None, ya_dot, ya_blk)
        sb, bb, db, done_b = lane(BUDGET, yb_dot, yb_blk)
        lab_a = m.VGroup(text("one giant step", 24, INK, weight=m.BOLD), text("· no budget", 22, MUTED)).arrange(m.RIGHT, buff=0.15)
        lab_b = m.VGroup(text("chunked", 24, INK, weight=m.BOLD), text(f"· a budget of {BUDGET} tokens", 22, MUTED)).arrange(m.RIGHT, buff=0.15)
        for lab, y in ((lab_a, 2.3), (lab_b, 0.78)):
            lab.align_to([-6.0, 0, 0], m.LEFT)
            sit(lab[0], lab[0].text, y)
            sit(lab[1], lab[1].text, y)
        big_i = [i for i, s in enumerate(sa) if s[2] > 0][0]
        xa0, xa1 = clock.x(sa[big_i - 1][1]), clock.x(sa[big_i][1])
        hole = bracket(xa0 + 0.08, xa1 - 0.08, ya_dot + 0.12, RED, down=False)
        hole_lab = mono(f"{ms(T_BIG):.0f} ms, no tokens", 22, RED, weight=m.BOLD).move_to([(xa0 + xa1) / 2, ya_dot - 0.08, 0])
        k = len([s for s in sb if s[2] == 0 and s[1] <= arrive + 1e-6]) + 9
        xb0, xb1 = clock.x(sb[k - 1][1]), clock.x(sb[k][1])
        small = bracket(xb0, xb1, yb_dot + 0.2, TOKEN, tick=0.1)
        small_lab = code(f"{ms(T_CHUNK):.1f} ms", 20, TOKEN).next_to(small, m.UP, buff=0.08)

        def first_mark(t, y_blk, y_lab):
            x = clock.x(t)
            line = m.Line([x, y_blk - 0.3, 0], [x, y_blk + 0.3, 0]).set_stroke(REQUEST, 4).set_z_index(3)
            lab = text("chapter's first token", 20, REQUEST).move_to([0, y_lab, 0]).align_to([x - 0.12, 0, 0], m.RIGHT)
            return m.VGroup(line, lab)

        mark_a = first_mark(done_a, ya_blk, ya_blk - 0.45)
        mark_b = first_mark(done_b, yb_blk, yb_blk - 0.45)

        def compare(title_s, before, after, after_color, x):
            box = card(5.6, 1.55, EDGE, CARD, radius=0.16, stroke=1.5).move_to([x, -2.55, 0])
            head = text(title_s, 22, MUTED).align_to(box, m.UL).shift(0.3 * m.RIGHT + 0.18 * m.DOWN)
            v0 = title_text(before, 50, RED)
            u0 = code("ms", 26, MUTED)
            to = code("→", 28, MUTED)
            v1 = title_text(after, 50, after_color)
            u1 = code("ms", 26, MUTED)
            row = m.VGroup(v0, u0, to, v1, u1).arrange(m.RIGHT, buff=0.22)
            row.align_to(box, m.LEFT).shift(0.3 * m.RIGHT)
            row.set_y(box.get_y() - 0.22)
            base = v0.get_bottom()[1]
            for w_ in (u0, u1):
                w_.align_to([0, base, 0], m.DOWN)
            to.set_y(v0.get_y())
            return m.VGroup(box, head, m.VGroup(v0, u0, to), m.VGroup(v1, u1))

        p99 = compare("p99 gap, one busy minute", f"{ms(P99_BIG):.0f}", f"{ms(P99_CH):.1f}", TOKEN, -3.15)
        ttft = compare("the chapter's first token", f"{ms(T_BIG):.0f}", f"{ms(TTFT_CH):.0f}", COMPUTE, 3.15)
        ttft[2][0].set_color(INK)

        self.say(
            "[Now both on one clock.] [One giant step]: [a three hundred and twenty-seven millisecond hole in every stream.]",
            m.AnimationGroup(self.new_header("04 · the trade", "Small gaps, a later first token"), m.FadeIn(clock.group)),
            m.AnimationGroup(m.FadeIn(lab_a), m.LaggedStart(*[m.GrowFromEdge(b, m.LEFT) for b in ba], lag_ratio=0.12),
                             m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in da], lag_ratio=0.12)),
            m.AnimationGroup(m.Create(hole), m.FadeIn(hole_lab)),
        )
        self.say(
            "[With a budget of five hundred and twelve], [tokens keep coming], [every sixteen point six milliseconds.]",
            m.AnimationGroup(m.FadeIn(lab_b), m.LaggedStart(*[m.GrowFromEdge(b, m.LEFT) for b in bb], lag_ratio=0.1)),
            m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in db], lag_ratio=0.08),
            m.AnimationGroup(m.Create(small), m.FadeIn(small_lab)),
        )
        self.say(
            "[Over the busy minute], [the ninety-ninth percentile gap falls from three hundred and twenty-seven milliseconds] [to under seventeen.]",
            m.AnimationGroup(m.FadeIn(p99[0]), m.FadeIn(p99[1])),
            m.FadeIn(p99[2], shift=0.1 * m.UP),
            m.FadeIn(p99[3], shift=0.1 * m.UP),
        )
        self.say(
            "[The newcomer pays a little]: [its first token comes after three hundred and seventy-eight milliseconds], [instead of three hundred and twenty-seven.]",
            m.AnimationGroup(m.FadeIn(ttft[0]), m.FadeIn(ttft[1]), m.FadeIn(mark_a)),
            m.AnimationGroup(m.FadeIn(mark_b), m.FadeIn(ttft[3], shift=0.1 * m.UP)),
            m.FadeIn(ttft[2], shift=0.1 * m.UP),
        )
        chunk_dots = m.VGroup(*[d for d, s in zip(db, sb) if s[2] > 0])
        self.say(
            "[That time isn't wasted]: [the GPU kept serving the other sixty along the way.]",
            m.Circumscribe(m.VGroup(*[b for b, s in zip(bb, sb) if s[2] > 0]), color=COMPUTE, buff=0.06),
            m.LaggedStart(*[m.Flash(d, color=TOKEN, line_length=0.12, flash_radius=0.14) for d in chunk_dots], lag_ratio=0.12),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the dial

    def dial(self):
        rows = "; ".join(f"{b:,}: {s} steps, worst gap {g:.1f} ms, first token {t:.0f} ms" for b, s, g, t in zip(DIAL, DIAL_STEPS, DIAL_GAP, DIAL_TTFT))
        self.next_section("5 · The dial", notes=f"The budget trades the others' gaps against the newcomer's TTFT (same model; budgets per doubling): {rows}. Below ≈ {FREE:.0f} tokens per step the step's math ({FREE:.0f} × {GFLOP:.1f} GFLOP at 50% MFU) fits under the {ms(T_DEC):.1f} ms memory floor, so a chunk rides free (Episode 4's free lunch), but the chapter needs more steps. At 16K the whole chapter goes in one step: gap = first token = {ms(T_BIG):.0f} ms. vLLM docs (Optimization and Tuning, max_num_batched_tokens): 'Smaller values (e.g., 2048) achieve better ITL because there are fewer prefills slowing down decodes'; 'Higher values achieve better time to first token (TTFT)'. Notes §5 worked example B, recomputed with the same functions: {B_USERS} users at {CTX:,} tokens → decode step {ms(B_DEC):.1f} ms; a {B_PROMPT:,}-token prompt in one step = ({B_PROMPT:,} + {B_USERS}) × {GFLOP:.1f} GFLOP = {(B_PROMPT + B_USERS) * GFLOP / 1e3:.1f} TFLOP → {ms(B_BIG):.1f} ms (~{B_BIG / B_DEC:.0f}× the normal gap); {B_CHUNK}-token chunks ({len(B_SIZES)} of them) + {B_USERS} decodes → {ms(B_STEP):.1f} ms steps. Episode 12: prefill and decode on separate machines (disaggregation).")
        ox, oy, dx, H, top = -5.2, -2.45, 1.15, 4.3, 700.0
        u0 = np.log2(DIAL[0])
        px = lambda b: ox + dx * (np.log2(b) - u0)
        py = lambda v: oy + H * v / top
        us = [np.log2(b) for b in DIAL]
        gap_at = lambda u: float(np.interp(u, us, DIAL_GAP))
        ttft_at = lambda u: float(np.interp(u, us, DIAL_TTFT))
        x_end = px(DIAL[-1])
        x_axis = m.Line([ox, oy, 0], [x_end + 0.25, oy, 0]).set_stroke(FAINT, 2)
        y_axis = m.Line([ox, oy, 0], [ox, oy + H + 0.1, 0]).set_stroke(FAINT, 2)
        xt = m.VGroup(*[code(lbl, 20, MUTED).move_to([px(b), oy - 0.3, 0]) for b, lbl in
                        zip(DIAL, ["256", "512", "1K", "2K", "4K", "8K", "16K"])])
        yt = m.VGroup(*[code(f"{v}", 20, MUTED).move_to([0, py(v), 0]).align_to([ox - 0.15, 0, 0], m.RIGHT) for v in (200, 400, 600)])
        y_unit = code("ms", 24, MUTED).move_to([0, py(700), 0]).align_to([ox - 0.15, 0, 0], m.RIGHT)
        x_name = text("token budget per step", 20, MUTED).move_to([(ox + x_end) / 2, oy - 0.68, 0])
        axes_ = m.VGroup(x_axis, y_axis, xt, yt, y_unit, x_name)
        gap_curve = m.VMobject().set_points_as_corners([[px(b), py(v), 0] for b, v in zip(DIAL, DIAL_GAP)]).set_stroke(MEMORY, 5)
        ttft_curve = m.VMobject().set_points_as_corners([[px(b), py(v), 0] for b, v in zip(DIAL, DIAL_TTFT)]).set_stroke(COMPUTE, 5)

        def key(color, words):
            swatch = m.Line([0, 0, 0], [0.45, 0, 0]).set_stroke(color, 5)
            lab = text(words, 22, color)
            swatch.next_to(lab, m.LEFT, buff=0.15).set_y(lab.get_y())
            return m.VGroup(swatch, lab)

        ttft_name = key(COMPUTE, "the chapter's first token")
        gap_name = key(MEMORY, "everyone else's worst gap")
        m.VGroup(ttft_name, gap_name).arrange(m.RIGHT, buff=0.6).move_to([0, 2.2, 0]).align_to([ox - 0.6, 0, 0], m.LEFT)
        giant = text("one giant step", 20, MUTED).move_to([0, py(DIAL_TTFT[-1]) + 0.33, 0]).align_to([x_end + 0.05, 0, 0], m.RIGHT)

        def knob_parts(u):
            x = ox + dx * (u - u0)
            g, t = py(gap_at(u)), py(ttft_at(u))
            line = m.DashedLine([x, oy, 0], [x, max(g, t) + 0.35, 0], dash_length=0.1).set_stroke(MUTED, 2)
            d1 = m.Dot([x, g, 0], radius=0.11, color=MEMORY).set_z_index(3)
            d2 = m.Dot([x, t, 0], radius=0.11, color=COMPUTE).set_z_index(3)
            return line, d1, d2

        knob = m.VGroup(*knob_parts(np.log2(BUDGET)))

        def slide(b0, b1, run_time=1.6):
            a, b = np.log2(b0), np.log2(b1)

            def update(mob, alpha):
                line, d1, d2 = knob_parts(a + (b - a) * alpha)
                mob[0].become(line)
                mob[1].move_to(d1.get_center())
                mob[2].move_to(d2.get_center())
            return m.UpdateFromAlphaFunc(knob, update, run_time=run_time)

        # read-out panel
        box = card(3.9, 3.05, EDGE, CARD, radius=0.2, stroke=2).move_to([4.55, 0.65, 0])
        bx = box.get_left()[0] + 0.35

        def readout(b):
            i = DIAL.index(b)
            n = title_text(f"{b:,}", 48, INK)
            nl = text("tokens per step", 20, MUTED)
            g = title_text(f"{DIAL_GAP[i]:.1f}", 40, MEMORY)
            gl = text("ms worst gap", 20, MUTED)
            t = title_text(f"{DIAL_TTFT[i]:.0f}", 40, COMPUTE)
            tl = text("ms first token", 20, MUTED)
            for v, lab, y in ((n, nl, 1.7), (g, gl, 0.88), (t, tl, 0.22)):
                v.align_to([bx, 0, 0], m.LEFT).set_y(y)
                lab.next_to(v, m.RIGHT, buff=0.15)
                sit(lab, lab.text, v.get_bottom()[1])
            n_steps = text(f"the chapter takes {DIAL_STEPS[i]} steps", 20, MUTED).align_to([bx, 0, 0], m.LEFT)
            sit(n_steps, n_steps.text, -0.5)
            return m.VGroup(m.VGroup(n, nl), m.VGroup(g, gl), m.VGroup(t, tl, n_steps))

        shown = readout(512)
        reads = {b: readout(b) for b in (256, 512, 2048)}
        free_x = px(FREE)
        free_line = m.DashedLine([free_x, oy, 0], [free_x, py(300), 0], dash_length=0.08).set_stroke(MEMORY, 2)
        free_lab = code(f"≈ {FREE:.0f}", 20, MEMORY).move_to([free_x, oy - 0.68, 0])
        free_note = m.VGroup(text(f"below ≈ {FREE:.0f} tokens, each step", 20, MEMORY),
                             text("waits on memory anyway", 20, MEMORY)).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.08)
        free_note.next_to(box, m.DOWN, buff=0.3).align_to(box, m.LEFT).shift(0.1 * m.RIGHT)

        quote = m.VGroup(
            text("vLLM tuning guide, on the budget:", 20, MUTED),
            text("“Smaller values (e.g., 2048)", 22, INK),
            text("achieve better ITL”", 22, INK),
            text("“Higher values achieve better", 22, INK),
            text("time to first token (TTFT)”", 22, INK),
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.1)
        quote[3].shift(0.12 * m.DOWN)
        quote[4].shift(0.12 * m.DOWN)
        quote.move_to([0, -2.05, 0]).align_to(box, m.LEFT).shift(0.1 * m.RIGHT)

        ex_box = card(4.4, 4.1, EDGE, CARD, radius=0.2, stroke=2).move_to([4.3, -0.4, 0])
        ex_x = ex_box.get_left()[0] + 0.35
        ex_head = text("the notes' example B", 22, MUTED)
        ex_case = text(f"{B_USERS} students, {B_PROMPT:,}-token prompt", 20, INK)
        ex_one_l = text("one giant step", 22, INK, weight=m.BOLD)
        ex_one = m.VGroup(title_text(f"{ms(B_BIG):.0f}", 50, RED), code("ms hiccup", 24, MUTED)).arrange(m.RIGHT, buff=0.14, aligned_edge=m.DOWN)
        ex_ch_l = text(f"{len(B_SIZES)} chunks of {B_CHUNK}", 22, INK, weight=m.BOLD)
        ex_ch = m.VGroup(title_text(f"{ms(B_STEP):.1f}", 50, TOKEN), code("ms gaps", 24, MUTED)).arrange(m.RIGHT, buff=0.14, aligned_edge=m.DOWN)
        for w_, y in ((ex_head, 1.25), (ex_case, 0.83), (ex_one_l, 0.25), (ex_ch_l, -1.15)):
            w_.align_to([ex_x, 0, 0], m.LEFT)
            sit(w_, w_.text, y)
        ex_one.align_to([ex_x, 0, 0], m.LEFT).set_y(-0.32)
        ex_ch.align_to([ex_x, 0, 0], m.LEFT).set_y(-1.72)
        ex_src = cite("notes §5 · recomputed here").next_to(ex_box, m.DOWN, buff=0.12).align_to(ex_box, m.LEFT)
        ep12 = pill("Episode 12 · split the work", REQUEST, PINK_FILL, 20).next_to(ex_src, m.DOWN, buff=0.2).align_to(ex_box, m.LEFT)

        self.say(
            "[The budget is a dial.] [Bigger budgets bring the chapter's first token sooner], [but stretch everyone else's gaps.]",
            m.AnimationGroup(self.new_header("05 · the dial", "Gaps versus first token"), m.FadeIn(axes_), m.FadeIn(box),
                             m.FadeIn(shown), m.FadeIn(knob)),
            m.AnimationGroup(m.Create(ttft_curve, run_time=1.4), m.FadeIn(ttft_name, shift=0.1 * m.UP)),
            m.AnimationGroup(m.Create(gap_curve, run_time=1.4), m.FadeIn(gap_name, shift=0.1 * m.UP), m.FadeIn(giant)),
        )
        self.say(
            "[At two thousand and forty-eight tokens per step], [the gaps grow to sixty-six and a half milliseconds], [yet the first token comes only thirty-three milliseconds sooner.]",
            m.AnimationGroup(slide(512, 2048), m.Transform(shown[0], reads[2048][0])),
            m.Transform(shown[1], reads[2048][1]),
            m.Transform(shown[2], reads[2048][2]),
        )
        self.say(
            "[Below about three hundred and seventy tokens], [the gaps stop growing]: [each step is waiting on memory anyway.]",
            m.AnimationGroup(slide(2048, 256, 2.0), m.Transform(shown[0], reads[256][0]), m.Create(free_line), m.FadeIn(free_lab)),
            m.AnimationGroup(m.Transform(shown[1], reads[256][1]), m.Transform(shown[2], reads[256][2])),
            m.FadeIn(free_note, shift=0.1 * m.UP),
        )
        self.say(
            "[But then the chapter needs fifty-two steps], [and its first token takes six hundred and twenty-four milliseconds.]",
            m.Circumscribe(shown[2][2], color=MUTED, buff=0.08),
            m.Circumscribe(m.VGroup(shown[2][0], shown[2][1]), color=COMPUTE, buff=0.1),
        )
        self.say(
            "[The vLLM tuning guide says the same]: [smaller budgets for better gaps], [larger ones for a faster first token.]",
            m.AnimationGroup(m.FadeOut(free_note), m.FadeIn(quote[0]), slide(256, 512),
                             *[m.Transform(shown[i], reads[512][i]) for i in range(3)]),
            m.FadeIn(quote[1:3], shift=0.1 * m.UP),
            m.FadeIn(quote[3:], shift=0.1 * m.UP),
        )
        self.say(
            "[The course notes work a smaller case the same way]: [thirty students, and a four thousand token prompt.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(box, shown, quote)), m.FadeIn(ex_box), m.FadeIn(ex_head), m.FadeIn(ex_src)),
            m.FadeIn(ex_case, shift=0.1 * m.UP),
        )
        self.say(
            "[One giant step is a hundred and thirty-one millisecond hiccup]; [chunks of five hundred and twelve make gaps of seventeen point six.]",
            m.AnimationGroup(m.FadeIn(ex_one_l), m.FadeIn(ex_one, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(ex_ch_l), m.FadeIn(ex_ch, shift=0.1 * m.UP)),
        )
        self.say(
            "[Chunking lets one GPU share both kinds of work.] [Episode twelve goes further]: [it gives each its own machines.]",
            m.Circumscribe(m.VGroup(gap_curve, ttft_curve), color=MUTED, buff=0.1),
            m.FadeIn(ep12, shift=0.1 * m.UP),
            m.Circumscribe(ep12, color=REQUEST, buff=0.06),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes=f"A token budget per step ({BUDGET} here); decodes first, one token for each running request; the waiting prompt fills the rest in slices (chunks). Gaps stay near {ms(T_CHUNK):.0f} ms instead of a {ms(T_BIG):.0f} ms stall; the newcomer's first token comes a little later ({ms(TTFT_CH):.0f} vs {ms(T_BIG):.0f} ms).")
        halo = glow(PINK, radius=4.0)
        words = m.VGroup(*[title_text(w, 80, c) for w, c in (("Budget", REQUEST), ("Decodes", MEMORY), ("Chunks", COMPUTE))])
        subs = m.VGroup(*[code(s, 22, MUTED) for s in ("a cap per step", "go first", "fill the rest")])
        slots = [max(w.width, s.width) for w, s in zip(words, subs)]
        x, gap = 0.0, 0.9
        for w, s, slot_w in zip(words, subs, slots):
            w.move_to([x + slot_w / 2, 0, 0])
            s.move_to([x + slot_w / 2, 0, 0])
            x += slot_w + gap
        on_baseline(words)
        line_y = min(w.get_bottom()[1] for w in words) - 0.45
        for s in subs:
            sit(s, s.text, line_y)
        mid = words[0][0].get_center()[1]
        dots = m.VGroup(*[m.Dot([(a.get_right()[0] + b.get_left()[0]) / 2, mid, 0], radius=0.07, color=MUTED) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, dots, subs).move_to(0.45 * m.UP)
        tagline = text("Chunk the long prompt, and nobody's stream stops.", 30, MUTED).next_to(row, m.DOWN, buff=0.7)
        mascot = owl(0.5).next_to(tagline, m.DOWN, buff=0.45)

        self.say(
            "[To sum up]: [a budget], [decodes first], [then chunks.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say("[Chunk the long prompt], [and nobody's stream stops.]",
                 m.Write(tagline), m.FadeIn(mascot, shift=0.15 * m.UP))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline, mascot)], run_time=0.9)
