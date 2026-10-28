"""Tokens per Second · Episode 7 · PagedAttention (Module 2 · The KV Cache)

Fixed-size blocks and a block table end fragmentation. Running example: Owl-8B (Llama-3.1-8B: 8 B parameters,
16 GB of BF16 weights, 128 KiB of KV cache per token) serving students; Episode 5 fit 52 of them on one H100.

Sections: hook (Episode 5's 52 equal chats vs real chats of unknown length) · reserve (a memory map: one
contiguous stretch per request, sized for the longest answer; internal fragmentation when an answer stops
early; external fragmentation when a fifth student finds 40 free squares but no 32 in one piece; 26% of the
map holds tokens) · the bill (Kwon et al.: 20.4–38.2% in earlier systems; Episode 5's 52 students → fewer
than 20) · blocks (virtual memory: blocks = pages, tokens = bytes, requests = processes; a block = 16 tokens
= 2 MiB for Owl; a block table per request; a new block only when the last is full; the kernel follows the
table) · packed (the same students replayed with blocks: 96% of the blocks in use hold tokens, the fifth
student fits, 64% of the map stays free) · the catch (attention kernel 20–26% slower, 2–4x throughput; why
16) · share (n = 4 flashcards share the prompt's blocks; copy-on-write of the half-full last block; 276 vs 90
blocks) · recap.

Sources (facts dated October 2026):
- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023
  (arXiv 2309.06180), read 2026-10-24: "only 20.4% - 38.2% of the KV cache memory is used to store the actual
  token states in the existing systems"; waste = reserved slots, "internal fragmentation due to
  over-provisioning for potential maximum sequence lengths", "external fragmentation from the memory allocator";
  "vLLM sets its default block size as 16"; "one can think of blocks as pages, tokens as bytes, and requests as
  processes"; "vLLM limits all the memory wastes for a request within one block"; abstract: "near-zero waste
  in KV cache memory", "improves the throughput of popular LLMs by 2-4x with the same level of latency" (vs
  FasterTransformer and Orca); §4.4: a reference count per physical block, "copy-on-write mechanism at the
  block granularity" (Fig. 8, samples A1/A2); §6: "6.1% - 9.8% memory saving on parallel sampling and 37.6% -
  55.2% on beam search"; §7.1: "20-26% higher attention kernel latency, compared to the highly-optimized
  FasterTransformer implementation"; §7.2: too small a block underuses the GPU's parallelism, too large a
  block increases internal fragmentation.
- vLLM launch blog (2023-06-20): "a mere waste of under 4%"; parallel sampling memory "up to 55%".
- research/notes.md §3 (Episode 5's H100: 90% of 80 GB minus 16.06 GB of weights = 56 GB = 427,000 token slots
  = 52 chats at 8K tokens) and §4 (mechanism, numbers, worked example).
- owl_reference.py: kv_bytes_per_token (128 KiB), weight_bytes, GPU; the allocator below (Paged) is this
  episode's own miniature of vLLM's bookkeeping, run here to place every block on screen.

Illustrative (said "for example" / "in this example", or schematic): the students' prompt, answer and
reservation lengths (A 90+60 of 576 tokens, B 70+50 of 448, C 200+80 of 704, D 110+60 of 384, E 200 of 512);
the 144-square memory slice; the block-table toy (41 → 49 tokens); the occupied cells in the block-table map;
the flashcard request (a 1,000-token prompt, four 100-token answers; the toy prompt of 40 tokens); the
1/16/256 block-size comparison for a 40-token answer. "52 → fewer than 20" is derived (52 × 38.2% = 19.9).

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from owl_reference import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- the numbers

PREFIX = "ep07"
KIB, MIB, GIB = 2**10, 2**20, 2**30
KV_TOKEN = int(kv_bytes_per_token())
BLOCK = 16  # vLLM's default block size (Kwon et al., 2023)
BLOCK_BYTES = BLOCK * KV_TOKEN
assert KV_TOKEN == 128 * KIB and BLOCK_BYTES == 2 * MIB  # "sixteen tokens each", "2 MiB"
assert 2.0e6 < BLOCK_BYTES < 2.2e6  # "about two megabytes for Owl" (2.1 MB)

# Episode 5's H100 (notes §3 worked example)
KV_ROOM = 0.9 * GPU["memory"] - weight_bytes()
USERS = int(KV_ROOM / KV_TOKEN // 8192)
USED_LO, USED_HI = 0.204, 0.382  # Kwon et al. 2023: share of KV memory holding token states, earlier systems
assert USERS == 52  # "fifty-two students on one H100 … eight thousand tokens"
KEEP = int(USERS * USED_HI)
assert USERS * USED_HI < 20 and KEEP == 19  # "fewer than twenty of its fifty-two students"


class Paged:
    """PagedAttention's bookkeeping in miniature: a free list, a block table per sequence, a reference count
    per physical block. Tokens are appended one at a time; a new block is taken only when the last one is
    full; a shared, partly filled last block is copied before anyone writes into it (copy-on-write)."""

    def __init__(self, n_blocks):
        self.free, self.refs = list(range(n_blocks)), [0] * n_blocks
        self.tables, self.lengths, self.copies = {}, {}, []

    def take(self):
        self.free.sort()  # lowest free address first (any free block would do)
        block = self.free.pop(0)
        self.refs[block] = 1
        return block

    def append(self, seq):
        table, n = self.tables.setdefault(seq, []), self.lengths.get(seq, 0)
        if n % BLOCK == 0:  # the last block is full (or there is none): take any free block
            table.append(self.take())
        elif self.refs[table[-1]] > 1:  # shared and partly filled: copy on write
            old = table[-1]
            self.refs[old] -= 1
            table[-1] = self.take()
            self.copies.append((seq, old, table[-1]))
        self.lengths[seq] = n + 1

    def fork(self, parent, child):
        self.tables[child], self.lengths[child] = list(self.tables[parent]), self.lengths[parent]
        for block in self.tables[child]:
            self.refs[block] += 1

    def release(self, seq):
        for block in self.tables.pop(seq):
            self.refs[block] -= 1
            if self.refs[block] == 0:
                self.free.append(block)
        self.lengths.pop(seq)

    def in_use(self):
        return sum(r > 0 for r in self.refs)


# the memory map: an illustrative slice of GPU memory, one square = 16 token slots
ROWS, COLS = 6, 24
N_CELLS = ROWS * COLS


@dataclass
class Student:
    name: str
    prompt: int   # tokens in the prompt
    tokens: int   # tokens stored at the snapshot (prompt + answer so far)
    reserve: int  # tokens reserved up front (prompt + the longest answer allowed)


STUDENTS = [Student("A", 90, 150, 576), Student("B", 70, 120, 448), Student("C", 200, 280, 704),
            Student("D", 110, 170, 384)]
LATE = Student("E", 200, 200, 512)
assert all(s.reserve % BLOCK == 0 for s in STUDENTS + [LATE])


def first_fit(owner, cells):
    """The first run of `cells` free squares in a row, or None (a contiguous allocator)."""
    run = 0
    for i, who in enumerate(owner):
        run = run + 1 if who is None else 0
        if run == cells:
            return i - cells + 1
    return None


def runs_of_free(owner):
    out, run = [], 0
    for who in owner + ["end"]:
        if who is None:
            run += 1
        elif run:
            out.append(run)
            run = 0
    return out


OWNER = [None] * N_CELLS
START = {}
for s in STUDENTS:
    START[s.name] = first_fit(OWNER, s.reserve // BLOCK)
    OWNER[START[s.name]: START[s.name] + s.reserve // BLOCK] = [s.name] * (s.reserve // BLOCK)
FULL_OWNER = list(OWNER)
OWNER = [None if w == "B" else w for w in OWNER]  # the second student finishes and leaves
HOLES = runs_of_free(OWNER)
E_CELLS = LATE.reserve // BLOCK
assert first_fit(OWNER, E_CELLS) is None  # "not in one piece"
assert HOLES == [28, 12] and sum(HOLES) > E_CELLS > max(HOLES)  # "enough free memory in total"
B = STUDENTS[1]
assert B.reserve - B.tokens == 328  # never used: 20.5 squares, internal fragmentation
STAYING = [s for s in STUDENTS if s.name != "B"]
OLD_TOKENS = sum(s.tokens for s in STAYING)
OLD_USED = OLD_TOKENS / (N_CELLS * BLOCK)
assert 0.24 < OLD_USED < 0.28 and round(100 * OLD_USED) == 26  # "only about a quarter … holds tokens"

# the same students, replayed with blocks: prompts prefilled, then one token each per step; B leaves when done
REPLAY = Paged(N_CELLS)
for s in STUDENTS:
    for _ in range(s.prompt):
        REPLAY.append(s.name)
_left = {s.name: s.tokens - s.prompt for s in STUDENTS}
while any(_left.values()):
    for name in list(_left):
        if _left[name] > 0:
            REPLAY.append(name)
            _left[name] -= 1
            if _left[name] == 0 and name == "B":
                REPLAY.release("B")
for _ in range(LATE.tokens):
    REPLAY.append("E")
NEW_TOKENS = sum(REPLAY.lengths.values())
NEW_USED = NEW_TOKENS / (REPLAY.in_use() * BLOCK)
FREE_SHARE = 1 - REPLAY.in_use() / N_CELLS
assert NEW_TOKENS == OLD_TOKENS + LATE.tokens and REPLAY.in_use() == 52
assert round(100 * NEW_USED) == 96  # "ninety-six percent of the memory in use holds tokens"
assert round(100 * FREE_SHARE) == 64 and 0.6 < FREE_SHARE < 0.7  # "nearly two thirds … still free"
TAILS = {name: len(t) * BLOCK - REPLAY.lengths[name] for name, t in REPLAY.tables.items()}
assert max(TAILS.values()) <= BLOCK - 1  # "at most fifteen slots"
for s in STAYING + [LATE]:
    assert len(REPLAY.tables[s.name]) == math.ceil(s.tokens / BLOCK)

# block table toy (section 3): student A's 41 tokens in 3 blocks among other students' blocks
TOY_CELLS = 35  # 5 x 7
TOY_TABLE = [4, 8, 24]
TOY_FREE = [30, 13, 5, 34, 18, 26]  # the free list, in the order blocks were freed
TOY_OTHERS = sorted(set(range(TOY_CELLS)) - set(TOY_TABLE) - set(TOY_FREE))
TOY_A, TOY_GROW = 41, 8  # 41 tokens; 7 more fill block 2, the 8th takes a new block
assert math.ceil(TOY_A / BLOCK) == 3 and math.ceil((TOY_A + TOY_GROW) / BLOCK) == 4
TOY_NEW = TOY_FREE[0]  # the allocator hands out the next free block, wherever it is

# sharing (section 6): Owl's flashcard button asks for n = 4 answers to one prompt
N_ANS, PROMPT_T, ANSWER_T = 4, 1000, 100
SHARE = Paged(400)
for _ in range(PROMPT_T):
    SHARE.append("a1")
for k in range(2, N_ANS + 1):
    SHARE.fork("a1", f"a{k}")
for _ in range(ANSWER_T):
    for k in range(1, N_ANS + 1):
        SHARE.append(f"a{k}")
SHARED_BLOCKS = SHARE.in_use()
NAIVE_BLOCKS = N_ANS * math.ceil((PROMPT_T + ANSWER_T) / BLOCK)
SAVED = 1 - SHARED_BLOCKS / NAIVE_BLOCKS
assert NAIVE_BLOCKS == 276 and SHARED_BLOCKS == 90  # "ninety blocks instead of two hundred and seventy-six"
assert 0.64 < SAVED < 0.7  # "about two thirds less memory"
assert len(SHARE.copies) == N_ANS - 1  # three copies; the last answer writes in place

TOY_PROMPT = 40  # the toy fork: 16 + 16 + 8 tokens
FORK = Paged(12)
for _ in range(TOY_PROMPT):
    FORK.append("a1")
for k in range(2, N_ANS + 1):
    FORK.fork("a1", f"a{k}")
assert FORK.refs[:3] == [4, 4, 4]  # "each block counts its users"
REF_STEPS = []
for k in range(1, N_ANS + 1):
    FORK.append(f"a{k}")
    REF_STEPS.append(FORK.refs[2])
assert [c[:2] for c in FORK.copies] == [("a1", 2), ("a2", 2), ("a3", 2)]  # first three copy
assert REF_STEPS == [3, 2, 1, 1] and FORK.tables["a4"][-1] == 2  # "the count drops to three" … "writes in place"

# block size (section 5): one 40-token answer in blocks of 1, 16 and 256 tokens
SIZES = [1, 16, 256]
ANS40 = 40
SIZE_BLOCKS = [math.ceil(ANS40 / b) for b in SIZES]
SIZE_SPARE = [n * b - ANS40 for n, b in zip(SIZE_BLOCKS, SIZES)]
assert SIZE_BLOCKS == [40, 3, 1] and SIZE_SPARE == [0, 8, 216]


# ---------------------------------------------------------------------------- look


def cite(s):
    return text(s, 20, MUTED)


def square(size, color=EDGE, fill=CARD, stroke=1.6):
    return m.RoundedRectangle(corner_radius=size * 0.12, width=size, height=size, stroke_color=color,
                              stroke_width=stroke, fill_color=fill, fill_opacity=1)


def slot_block(size, per_side=4, filled=16, color=KV, fill=KV, stroke=2.4, empty=EDGE):
    """A block drawn with its token slots (4 x 4 = 16): the first `filled` slots coloured."""
    box = square(size, color, CARD, stroke)
    pad, gap = size * 0.1, size * (0.04 if per_side <= 4 else 0.012)
    s = (size - 2 * pad - (per_side - 1) * gap) / per_side
    slots = m.VGroup(*[
        m.Square(s).set_stroke(width=0).set_fill(fill if k < filled else empty, 1) for k in range(per_side * per_side)
    ]).arrange_in_grid(rows=per_side, cols=per_side, buff=gap).move_to(box)
    return m.VGroup(box, slots)


def swatch(color, fill, label, stroke=1.6):
    sq = square(0.3, color, fill, stroke)
    words = text(label, 22, INK)
    return m.VGroup(sq, words).arrange(m.RIGHT, buff=0.16)


class MemMap:
    """A grid of squares read row by row like memory addresses. Fills (tokens) are separate rectangles."""

    def __init__(self, rows, cols, size, gap, center):
        self.size = size
        self.cells = m.VGroup(*[square(size) for _ in range(rows * cols)])
        self.cells.arrange_in_grid(rows=rows, cols=cols, buff=gap).move_to(center)

    def fill(self, i, frac, color=KV):
        c, inset = self.cells[i], 0.045
        full = c.width - 2 * inset
        w = max(full * frac, 0.02)
        rect = m.Rectangle(width=w, height=c.height - 2 * inset).set_stroke(width=0).set_fill(color, 1)
        return rect.move_to([c.get_left()[0] + inset + w / 2, c.get_y(), 0]).set_z_index(1)

    def tail(self, i, frac):
        """The empty end of a partly filled square (waste)."""
        c, inset = self.cells[i], 0.045
        full = c.width - 2 * inset
        w = full * (1 - frac)
        rect = m.Rectangle(width=w, height=c.height - 2 * inset).set_stroke(RED, 1.5).set_fill(RED_FILL, 1)
        return rect.move_to([c.get_right()[0] - inset - w / 2, c.get_y(), 0]).set_z_index(1)

    def fills(self, cells, tokens, color=KV):
        """Token fills for `tokens` tokens laid into the squares `cells` (16 per square)."""
        out = []
        for k, i in enumerate(cells):
            left = tokens - k * BLOCK
            if left <= 0:
                break
            out.append(self.fill(i, min(1.0, left / BLOCK), color))
        return out


def reserved(c):
    return c.animate.set_fill(VIOLET_FILL, 1).set_stroke(VIOLET, 1.6, opacity=0.55)


def freed(c):
    return c.animate.set_fill(CARD, 1).set_stroke(EDGE, 1.6, opacity=1)


MAP_SIZE, MAP_GAP, MAP_CENTER = 0.38, 0.07, (0, 0.85, 0)


def frac_of(tokens, k):
    return min(1.0, (tokens - k * BLOCK) / BLOCK)


# ---------------------------------------------------------------------------- the episode


class Ep07PagedAttention(Episode):
    number = 7

    def lesson(self):
        self.hook()
        self.reserve()
        self.bill()
        self.blocks()
        self.packed()
        self.catch()
        self.share()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes=f"Episode 5 fit {USERS} students on one H100 by assuming every chat is 8K tokens (notes §3: 56 GB of KV room ÷ 128 KiB ÷ 8,192). Real chats vary (a quick question, a pasted chapter) and the answer's length is unknown while it is generated. The bar lengths are illustrative. Teaser: Kwon et al. 2023 (the vLLM paper, SOSP): earlier systems used only 20.4–38.2% of KV memory for token states; vLLM: 'near-zero waste'.")
        gpu = icon(recolored("gpu", MUTED, CARD, PREFIX), 0.5).move_to([-5.7, 1.95, 0])
        x0, width = -5.1, 10.7
        track = m.Rectangle(width=width, height=0.44).set_stroke(EDGE, 2).set_fill(CARD, 1).move_to([x0 + width / 2, 1.95, 0])
        seg_w = (width - 0.1) / USERS
        segs = m.VGroup(*[
            m.Rectangle(width=seg_w - 0.035, height=0.32).set_stroke(width=0).set_fill(KV, 1).move_to([x0 + 0.05 + seg_w * (i + 0.5), 1.95, 0])
            for i in range(USERS)
        ])
        cap = code(f"Episode 5 · {USERS} students × 8K tokens · one H100", 20, MUTED).next_to(track, m.DOWN, buff=0.16).align_to(track, m.LEFT)

        user = recolored("user", REQUEST, PINK_FILL, PREFIX)
        prompts = [0.45, 5.6, 1.9, 0.9]  # illustrative lengths (scene units)
        answers = [1.3, 0.9, 2.6, 3.4]
        ys = [0.55, -0.2, -0.95, -1.7]
        rows, p_bars, a_bars, a_full, marks = m.VGroup(), [], [], [], m.VGroup()
        for p, a, y in zip(prompts, answers, ys):
            head = icon(user, 0.42).move_to([-5.75, y, 0])
            bar = m.Rectangle(width=p, height=0.3).set_stroke(REQUEST, 1.6).set_fill(PINK_FILL, 1).move_to([-5.3 + p / 2, y, 0])
            start = -5.3 + p + 0.05
            grow = m.Rectangle(width=a, height=0.3).set_stroke(TOKEN, 1.6).set_fill(GREEN_FILL, 1).move_to([start + a / 2, y, 0])
            stub = m.Rectangle(width=0.06, height=0.3).set_stroke(TOKEN, 1.6).set_fill(GREEN_FILL, 1).move_to([start + 0.03, y, 0])
            q = title_text("?", 30, MUTED).next_to(grow, m.RIGHT, buff=0.18)
            rows.add(m.VGroup(head, bar))
            p_bars.append(bar)
            a_bars.append(stub)
            a_full.append(grow)
            marks.add(q)
        quick = text("a quick question", 22, INK).next_to(p_bars[0], m.RIGHT, buff=0.3)
        chapter = text("a whole chapter", 22, INK).next_to(p_bars[1], m.RIGHT, buff=0.3)
        mascot = owl(0.62).move_to([5.7, -1.0, 0])
        ask = title_text("How do you hand out memory you can't predict?", 38, INK).move_to([0, 1.75, 0])
        tease = m.VGroup(
            pill("the idea behind vLLM", MUTED, CARD, 22),
            pill("PagedAttention", KV, VIOLET_FILL, 24),
            pill("20–38% used → near-zero waste", KV, VIOLET_FILL, 22, mono=True),
        ).arrange(m.RIGHT, buff=0.35).move_to([0, -2.8, 0])
        src = cite("Kwon et al., 2023").next_to(tease, m.DOWN, buff=0.18).align_to(tease, m.RIGHT)

        self.say(
            "[Episode five fit fifty-two students on one H100], [assuming every chat used exactly eight thousand tokens.]",
            m.AnimationGroup(self.new_header("hook", "Memory you can't predict"), m.FadeIn(gpu), m.FadeIn(track), m.FadeIn(cap)),
            m.LaggedStart(*[m.FadeIn(sg, shift=0.1 * m.RIGHT) for sg in segs], lag_ratio=0.04),
        )
        self.say(
            "[Real chats aren't like that.] [One student asks a quick question]; [another pastes in a whole chapter.]",
            m.LaggedStart(*[m.FadeIn(r, shift=0.15 * m.RIGHT) for r in rows], lag_ratio=0.2),
            m.AnimationGroup(m.Circumscribe(p_bars[0], color=REQUEST, buff=0.08), m.FadeIn(quick, shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.Circumscribe(p_bars[1], color=REQUEST, buff=0.08), m.FadeIn(chapter, shift=0.1 * m.LEFT)),
        )
        for stub in a_bars:
            stub.set_z_index(1)
        self.say(
            "[And while Owl is answering], [nobody knows how long each answer will get.]",
            m.AnimationGroup(m.FadeIn(mascot, shift=0.2 * m.UP), m.FadeIn(m.VGroup(*a_bars))),
            m.AnimationGroup(*[m.Transform(st, g, run_time=2.4) for st, g in zip(a_bars, a_full)],
                             m.FadeOut(quick), m.FadeOut(chapter),
                             m.LaggedStart(*[m.FadeIn(q, scale=0.6) for q in marks], lag_ratio=0.25, run_time=2.4)),
        )
        self.say(
            "[So how do you hand out memory you can't predict?] [Today: the trick at the heart of vLLM.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(gpu, track, segs, cap), shift=0.2 * m.UP), m.FadeIn(ask, shift=0.2 * m.UP)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(t, shift=0.15 * m.UP) for t in tease], lag_ratio=0.25), m.FadeIn(src)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ reserve

    def old_map(self):
        mm = MemMap(ROWS, COLS, MAP_SIZE, MAP_GAP, MAP_CENTER)
        cap = code("GPU memory · one square = 16 token slots", 20, MUTED)
        cap.next_to(mm.cells, m.UP, buff=0.22).align_to(mm.cells, m.LEFT)
        note = cite("illustrative requests").next_to(mm.cells, m.UP, buff=0.22).align_to(mm.cells, m.RIGHT)
        return mm, cap, note

    def letter(self, mm, i, name):
        return code(name, 20, BG, weight=m.BOLD).move_to(mm.cells[i]).set_z_index(3)

    def reserve(self):
        self.next_section("1 · One stretch per request", notes=f"Before vLLM, serving systems stored each request's KV cache in one contiguous region, pre-allocated for the maximum possible length (Kwon et al. 2023, §3). Illustrative slice: {N_CELLS} squares of 16 token slots (2 MiB each for Owl-8B). Reservations (prompt + longest allowed answer, first fit): A {STUDENTS[0].reserve}, B {STUDENTS[1].reserve}, C {STUDENTS[2].reserve}, D {STUDENTS[3].reserve} tokens. The second student (B) stops at {B.tokens} tokens: {B.reserve - B.tokens} reserved slots never used (internal fragmentation). B leaves: free runs {HOLES} squares; a fifth student (E) needs {E_CELLS} in one piece: refused although {sum(HOLES)} are free (external fragmentation). Tokens stored: {OLD_TOKENS} of {N_CELLS * BLOCK} slots = {100 * OLD_USED:.1f}% (computed here).")
        mm, cap, note = self.old_map()
        fills, letters = {}, {}
        for s in STUDENTS:
            start, n = START[s.name], s.reserve // BLOCK
            cells = list(range(start, start + n))
            fills[s.name] = (mm.fills(cells, s.prompt), mm.fills(cells, s.tokens))
            letters[s.name] = self.letter(mm, start, s.name)
        slab = {s.name: [mm.cells[i] for i in range(START[s.name], START[s.name] + s.reserve // BLOCK)] for s in STUDENTS}

        def arrive(s):
            pf = fills[s.name][0]
            return m.AnimationGroup(
                m.LaggedStart(*[reserved(c) for c in slab[s.name]], lag_ratio=0.03),
                m.LaggedStart(*[m.GrowFromEdge(r, m.LEFT) for r in pf], lag_ratio=0.15),
                m.FadeIn(letters[s.name]),
            )

        def grow(s):
            pf, ff = fills[s.name]
            anims = [m.Transform(pf[-1], ff[len(pf) - 1])]
            anims += [m.GrowFromEdge(r, m.LEFT) for r in ff[len(pf):]]
            return m.Succession(*anims, lag_ratio=1.0)

        legend = m.VGroup(
            swatch(KV, KV, "tokens"),
            swatch(VIOLET, VIOLET_FILL, "reserved, still empty"),
            swatch(RED, RED_FILL, "never used"),
            swatch(EDGE, CARD, "free"),
        ).arrange(m.RIGHT, buff=0.6).move_to([0, -0.9, 0])
        legend[1][0].set_stroke(VIOLET, 1.6, opacity=0.55)

        # B: its answer stops early; its unused squares never held a token
        b_cells = slab["B"]
        b_used = math.ceil(B.tokens / BLOCK)
        b_tail = mm.tail(START["B"] + b_used - 1, frac_of(B.tokens, b_used - 1))
        b_waste = m.VGroup(*b_cells[b_used:])
        internal = m.VGroup(pill("internal fragmentation", RED, RED_FILL, 22), text("reserved, never used", 22, MUTED))
        internal.arrange(m.DOWN, buff=0.14).move_to([-3.6, -1.85, 0])
        # E: needs 32 squares in one piece; the holes hold 28 and 12
        hole1 = [mm.cells[i] for i in range(N_CELLS) if OWNER[i] is None and FULL_OWNER[i] == "B"]
        hole2 = [mm.cells[i] for i in range(N_CELLS) if FULL_OWNER[i] is None]
        ghost_at = list(range(START["B"], START["B"] + E_CELLS))
        ghost = m.VGroup(*[
            square(MAP_SIZE + 0.08, REQUEST, CARD, 2.2).set_fill(opacity=0).move_to(mm.cells[i]) for i in ghost_at
        ]).set_z_index(4)
        clash = m.VGroup(*[ghost[k] for k, i in enumerate(ghost_at) if OWNER[i] is not None])
        e_tag = m.VGroup(code("E", 22, REQUEST, weight=m.BOLD), text(f"needs {E_CELLS} squares in one piece", 22, INK))
        e_tag.arrange(m.RIGHT, buff=0.2)
        h_labels = m.VGroup(code(f"hole: {HOLES[0]}", 20, RED), code(f"hole: {HOLES[1]}", 20, RED))
        external = m.VGroup(pill("external fragmentation", RED, RED_FILL, 22), text(f"{sum(HOLES)} free, but in pieces", 22, MUTED))
        external.arrange(m.DOWN, buff=0.14).move_to([3.6, -1.85, 0])
        e_tag.move_to([3.6, -1.85, 0])
        h_labels.arrange(m.RIGHT, buff=0.5).next_to(e_tag, m.DOWN, buff=0.2)
        verdict = m.VGroup(title_text(f"{100 * OLD_USED:.0f}%", 64, RED),
                           m.VGroup(text("of this memory holds tokens", 24, INK), text("and no new student fits", 22, MUTED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT))
        verdict.arrange(m.RIGHT, buff=0.35).move_to([0, -3.0, 0])

        self.say(
            "[Before vLLM], [servers gave each request one unbroken stretch of memory], [reserved the moment it arrived.]",
            m.AnimationGroup(self.new_header("01 · reserve", "One stretch per request"), m.FadeIn(mm.cells), m.FadeIn(cap), m.FadeIn(note)),
            arrive(STUDENTS[0]),
            m.Circumscribe(m.VGroup(*slab["A"]), color=REQUEST, buff=0.06),
        )
        self.say(
            "[How long a stretch?] [Enough for the longest answer the request might ever need.]",
            m.FadeIn(legend, shift=0.1 * m.UP),
            m.LaggedStart(*[arrive(s) for s in STUDENTS[1:]], lag_ratio=0.45),
        )
        self.say(
            "[As Owl answers, tokens fill each stretch from the left], [and the rest sits reserved, waiting.]",
            m.AnimationGroup(*[grow(s) for s in STUDENTS], run_time=2.6),
            m.Circumscribe(legend[1], color=VIOLET, buff=0.08),
        )
        self.say(
            "[Then the second student's answer stops early.] [Every slot it never used was wasted all along]: [internal fragmentation.]",
            m.AnimationGroup(m.Circumscribe(letters["B"], color=REQUEST, buff=0.08), m.FadeIn(b_tail)),
            m.LaggedStart(*[c.animate.set_fill(RED_FILL, 1).set_stroke(RED, 1.6, opacity=1) for c in b_waste], lag_ratio=0.04),
            m.FadeIn(internal, shift=0.1 * m.UP),
        )
        b_fills = m.VGroup(*fills["B"][0], *fills["B"][1][len(fills["B"][0]):])
        self.say(
            "[When that student leaves], [its stretch becomes a hole.]",
            m.AnimationGroup(m.FadeOut(b_fills), m.FadeOut(b_tail), m.FadeOut(letters["B"])),
            m.LaggedStart(*[freed(c) for c in b_cells], lag_ratio=0.03),
        )
        self.say(
            "[Now a fifth student arrives], [needing a longer stretch.] [There's enough free memory in total], [but not in one piece]: [external fragmentation.]",
            m.FadeOut(internal),
            m.AnimationGroup(m.FadeIn(e_tag, shift=0.1 * m.UP), m.LaggedStart(*[m.FadeIn(g) for g in ghost], lag_ratio=0.03)),
            m.AnimationGroup(*[c.animate.set_stroke(RED, 2.4) for c in hole1 + hole2], m.FadeIn(h_labels, shift=0.1 * m.UP)),
            m.AnimationGroup(*[g.animate.set_stroke(RED, 3.5) for g in clash], m.Circumscribe(clash, color=RED, buff=0.06)),
            m.AnimationGroup(*[m.FadeOut(g) for g in ghost], m.FadeOut(e_tag), m.FadeOut(h_labels), m.FadeIn(external, shift=0.1 * m.UP)),
        )
        self.say(
            "[Count it up]: [in this example, no new student fits], [yet only about a quarter of the memory holds tokens.]",
            m.Wait(0.3),
            m.FadeIn(verdict[1], shift=0.1 * m.UP),
            m.FadeIn(verdict[0], scale=0.85),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the bill

    def bill(self):
        self.next_section("2 · The bill", notes=f"Kwon et al., SOSP 2023 (arXiv 2309.06180): 'only 20.4% - 38.2% of the KV cache memory is used to store the actual token states in the existing systems' (their Orca baselines reserving for the maximum length, a 2x over-reservation, or the true output length; §6). Our example: {100 * OLD_USED:.0f}%. Derived: at 20.4–38.2% the H100 of Episode 5 (56 GB of KV room, {USERS} students at 8K tokens) holds {USERS * USED_LO:.1f}–{USERS * USED_HI:.1f} students' worth of tokens: fewer than 20.")
        x0, width, y = -5.0, 10.0, 1.15
        xp = lambda f: x0 + width * f
        track = m.Rectangle(width=width, height=0.56).set_stroke(EDGE, 2).set_fill(CARD, 1).move_to([x0 + width / 2, y, 0])
        waste = m.Rectangle(width=width * (1 - USED_LO) - 0.08, height=0.44).set_stroke(width=0).set_fill(RED_FILL, 1)
        waste.move_to([(xp(USED_LO) + xp(1)) / 2 - 0.04, y, 0])
        sure = m.Rectangle(width=width * USED_LO - 0.08, height=0.44).set_stroke(width=0).set_fill(KV, 1).move_to([(xp(0) + xp(USED_LO)) / 2 + 0.04, y, 0])
        maybe = m.Rectangle(width=width * (USED_HI - USED_LO), height=0.44).set_stroke(width=0).set_fill(KV, 0.45).move_to([(xp(USED_LO) + xp(USED_HI)) / 2, y, 0])
        ticks = m.VGroup(*[code(f"{p}%", 20, MUTED).next_to([xp(p / 100), y - 0.28, 0], m.DOWN, buff=0.12) for p in (0, 100)])
        held = code(f"held tokens: {100 * USED_LO:.1f}–{100 * USED_HI:.1f}%", 22, KV, weight=m.BOLD).move_to([x0, y + 0.62, 0], aligned_edge=m.LEFT)
        lost = code("reserved or fragmented", 22, RED).move_to([xp(1), y + 0.62, 0], aligned_edge=m.RIGHT)
        src = cite("Kwon et al., 2023 · earlier serving systems, measured").move_to([xp(1), y - 1.25, 0], aligned_edge=m.RIGHT)
        mark = m.Triangle(fill_color=INK, fill_opacity=1, stroke_width=0).scale(0.09).move_to([xp(OLD_USED), y - 0.4, 0])
        ours = code(f"our example: {100 * OLD_USED:.0f}%", 20, INK).move_to([xp(OLD_USED), y - 0.82, 0])

        dots = m.VGroup(*[m.Dot(radius=0.11, color=REQUEST) for _ in range(USERS)])
        dots.arrange_in_grid(rows=2, cols=USERS // 2, buff=0.17).move_to([0, -1.55, 0])
        gone = m.VGroup(*dots[KEEP:])
        before = code(f"Episode 5: {USERS} students per H100", 20, MUTED).next_to(dots, m.UP, buff=0.22).align_to(dots, m.LEFT)
        after = m.VGroup(title_text("fewer than 20", 48, REQUEST), cite(f"{USERS} × {100 * USED_HI:.1f}% = {USERS * USED_HI:.1f}"))
        after.arrange(m.RIGHT, buff=0.4).move_to([0, -2.95, 0])

        self.say(
            "[That's not just our example.] [In twenty twenty-three, the vLLM team measured earlier servers]: [only twenty to thirty-eight percent of their cache memory held tokens.]",
            m.AnimationGroup(self.new_header("02 · the bill", "Most of the cache, wasted"), m.FadeIn(track), m.FadeIn(ticks)),
            m.AnimationGroup(m.GrowFromEdge(waste, m.RIGHT), m.FadeIn(lost), m.FadeIn(src)),
            m.AnimationGroup(m.GrowFromEdge(sure, m.LEFT), m.FadeIn(maybe), m.FadeIn(held, shift=0.1 * m.UP),
                             m.FadeIn(mark), m.FadeIn(ours)),
        )
        self.say(
            "[At that rate, the H100 from Episode five], [would hold fewer than twenty of its fifty-two students.]",
            m.AnimationGroup(m.FadeIn(before), m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.02), run_time=1.6),
            m.AnimationGroup(*[d.animate.set_fill(FAINT, 0.5) for d in gone], m.FadeIn(after, shift=0.15 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ blocks and the block table

    def blocks(self):
        self.next_section("3 · Blocks and a block table", notes=f"PagedAttention (Kwon et al. 2023) treats the KV cache like OS virtual memory: 'one can think of blocks as pages, tokens as bytes, and requests as processes'. KV is stored in fixed-size blocks, 'vLLM sets its default block size as 16': for Owl-8B 16 × 128 KiB = {BLOCK_BYTES // MIB} MiB ({BLOCK_BYTES / 1e6:.1f} MB). Each request's block table maps logical blocks (in token order) to physical blocks anywhere in GPU memory; a block is allocated only when the last one is full (here the free list's next one, block {TOY_NEW}); the attention kernel gathers keys and values through the table. The toy (A with {TOY_A} → {TOY_A + TOY_GROW} tokens; table {TOY_TABLE} → +{TOY_NEW}; other occupied cells) is illustrative.")
        # analogy
        heads = m.VGroup(text("operating system", 26, MUTED, weight=m.BOLD), text("PagedAttention", 26, KV, weight=m.BOLD))
        pairs = [("page", "block"), ("byte", "token"), ("process", "request")]
        rows = m.VGroup(*[m.VGroup(code(a, 30, INK), code("→", 30, MUTED), code(b, 30, KV, weight=m.BOLD)) for a, b in pairs])
        xs = [-2.4, 0, 2.4]
        for k, r in enumerate(rows):
            for part, x in zip(r, xs):
                part.move_to([x, 0.55 - 0.85 * k, 0])
        heads[0].move_to([xs[0], 1.55, 0])
        heads[1].move_to([xs[2], 1.55, 0])
        rule = m.Line([-4.0, 1.12, 0], [4.0, 1.12, 0]).set_stroke(EDGE, 2)
        vm = pill("virtual memory", INK, CARD, 24).move_to([0, 2.25, 0])
        a_src = cite("Kwon et al., 2023 (the vLLM paper)").move_to([0, -2.35, 0])
        analogy = m.VGroup(heads, rows, rule, vm, a_src)

        # one block, up close
        big = slot_block(2.3, filled=0).move_to([-3.2, -0.1, 0])
        b_title = title_text("1 block = 16 tokens", 48, INK)
        b_math = code(f"16 × 128 KiB = {BLOCK_BYTES // MIB} MiB", 26, KV)
        b_who = text("of Owl's keys and values", 22, MUTED)
        b_src = cite("vLLM's default block size · Kwon et al., 2023")
        b_text = m.VGroup(b_title, b_math, b_who, b_src).arrange(m.DOWN, buff=0.25, aligned_edge=m.LEFT).move_to([2.1, -0.1, 0])

        # the toy: A's logical blocks, its block table, physical memory
        lsize = 0.8
        ly = [1.2, 0.2, -0.8, -1.8]
        lx = -5.3
        logical = [slot_block(lsize, filled=min(BLOCK, max(0, TOY_A - BLOCK * k))).move_to([lx, y, 0]) for k, y in enumerate(ly[:3])]
        new_logical = slot_block(lsize, filled=0).move_to([lx, ly[3], 0])
        l_cap = text("A's tokens, in order", 22, INK).move_to([lx - 0.4, 2.3, 0], aligned_edge=m.LEFT)
        l_nums = m.VGroup(*[code(str(k), 20, MUTED).next_to([lx - lsize / 2, y, 0], m.LEFT, buff=0.15) for k, y in enumerate(ly)])

        tcx = -2.55
        t_box = card(2.3, 4.55, PINK, CARD, radius=0.16, stroke=2).move_to([tcx, -0.18, 0])
        t_title = code("block table", 22, REQUEST, weight=m.BOLD).move_to([tcx, 1.78, 0])
        t_rows = [code(f"{k}  →  {p:>2}", 24, INK, t2c={r"→": MUTED}) for k, p in enumerate(TOY_TABLE + [TOY_NEW])]
        for r, y in zip(t_rows, ly):
            r.move_to([tcx, y, 0])
        t_arrows = [arrow([lx + lsize / 2, y, 0], [tcx - 1.15, y, 0], MUTED, buff=0.08) for y in ly]

        phys = MemMap(5, 7, 0.56, 0.13, (3.55, 0.05, 0))
        p_cap = text("GPU memory: physical blocks", 22, INK).move_to([phys.cells.get_left()[0], 2.3, 0], aligned_edge=m.LEFT)
        for i in TOY_OTHERS:
            phys.cells[i].set_fill(VIOLET_FILL, 1).set_stroke(VIOLET, 1.6, opacity=0.5)
        p_fills = [phys.fill(i, 1.0) for i in TOY_TABLE + [TOY_NEW]]
        p_letters = [code("A", 20, BG, weight=m.BOLD).move_to(phys.cells[i]).set_z_index(3) for i in TOY_TABLE + [TOY_NEW]]
        others = cite("other students' blocks").next_to(phys.cells, m.DOWN, buff=0.25).align_to(phys.cells, m.LEFT)
        o_sw = square(0.26, VIOLET, VIOLET_FILL, 1.6).set_stroke(opacity=0.5).next_to(others, m.LEFT, buff=0.14)
        others = m.VGroup(o_sw, others)

        def link(k):
            start = np.array([tcx + 1.15, ly[k], 0])
            end = phys.cells[(TOY_TABLE + [TOY_NEW])[k]].get_left() + np.array([-0.04, 0, 0])
            return m.Arrow(start, end, buff=0.05, color=REQUEST, stroke_width=3.5, max_tip_length_to_length_ratio=0.06,
                           path_arc=-0.35 if end[1] > start[1] else 0.35)

        links = [link(k) for k in range(4)]

        self.say(
            "[The fix, called PagedAttention], [borrows an idea from operating systems]: [virtual memory.]",
            self.new_header("03 · blocks", "Pages for the KV cache"),
            m.AnimationGroup(m.FadeIn(heads, shift=0.1 * m.DOWN), m.Create(rule)),
            m.AnimationGroup(m.FadeIn(vm, scale=0.85), m.FadeIn(a_src)),
        )
        self.say(
            "[Blocks are pages], [tokens are bytes], [and requests are processes.]",
            m.FadeIn(rows[0], shift=0.1 * m.UP),
            m.FadeIn(rows[1], shift=0.1 * m.UP),
            m.FadeIn(rows[2], shift=0.1 * m.UP),
        )
        self.say(
            "[The cache is cut into small, fixed-size blocks]: [sixteen tokens each], [about two megabytes for Owl.]",
            m.AnimationGroup(m.FadeOut(analogy, shift=0.2 * m.UP), m.FadeIn(big, scale=0.8)),
            m.AnimationGroup(m.LaggedStart(*[sq.animate.set_fill(KV, 1) for sq in big[1]], lag_ratio=0.12, run_time=1.8),
                             m.FadeIn(b_title, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(b_math, shift=0.1 * m.UP), m.FadeIn(b_who), m.FadeIn(b_src)),
        )
        self.say(
            "[Each request keeps a block table]: [its first block can sit here], [its second over there], [anywhere in memory.]",
            m.AnimationGroup(m.FadeOut(b_text), m.ReplacementTransform(big, logical[0]), m.FadeIn(l_cap),
                             m.FadeIn(logical[1], shift=0.1 * m.UP), m.FadeIn(logical[2], shift=0.1 * m.UP), m.FadeIn(l_nums[:3]),
                             m.FadeIn(t_box), m.FadeIn(t_title), m.FadeIn(phys.cells), m.FadeIn(p_cap), m.FadeIn(others)),
            m.AnimationGroup(m.GrowArrow(t_arrows[0]), m.FadeIn(t_rows[0]), m.Create(links[0]), m.FadeIn(p_fills[0]), m.FadeIn(p_letters[0])),
            m.AnimationGroup(m.GrowArrow(t_arrows[1]), m.FadeIn(t_rows[1]), m.Create(links[1]), m.FadeIn(p_fills[1]), m.FadeIn(p_letters[1])),
            m.AnimationGroup(m.GrowArrow(t_arrows[2]), m.FadeIn(t_rows[2]), m.Create(links[2]), m.FadeIn(p_fills[2]), m.FadeIn(p_letters[2])),
        )
        to_fill = [logical[2][1][k] for k in range(TOY_A - 2 * BLOCK, BLOCK)]
        self.say(
            "[A new block is handed out only when the last one is full.]",
            m.Succession(
                m.LaggedStart(*[sq.animate.set_fill(KV, 1) for sq in to_fill], lag_ratio=0.3, run_time=1.6),
                m.AnimationGroup(m.FadeIn(new_logical, shift=0.1 * m.UP), m.FadeIn(l_nums[3]), m.GrowArrow(t_arrows[3]), m.FadeIn(t_rows[3])),
                m.AnimationGroup(m.Create(links[3]), m.FadeIn(p_fills[3]), m.FadeIn(p_letters[3]), new_logical[1][0].animate.set_fill(KV, 1)),
            ),
        )
        reads = [m.ShowPassingFlash(lk.copy().set_stroke(MEMORY, 7), time_width=0.6, run_time=0.9) for lk in links]
        name = text("attention that reads paged memory", 22, KV, weight=m.BOLD).move_to([tcx, -2.85, 0])
        self.say(
            "[To attend, the kernel follows the table], [gathering keys and values block by block]: [hence the name, PagedAttention.]",
            m.Circumscribe(m.VGroup(t_box, t_title), color=REQUEST, buff=0.06),
            m.LaggedStart(*[m.AnimationGroup(r, m.Indicate(m.VGroup(phys.cells[i], p_fills[k], p_letters[k]), color=MEMORY, scale_factor=1.25))
                            for k, (r, i) in enumerate(zip(reads, TOY_TABLE + [TOY_NEW]))], lag_ratio=0.6),
            m.FadeIn(name, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ packed

    def packed(self):
        self.next_section("4 · Same students, packed", notes=f"Replay of section 1's students with blocks, run here with the Paged allocator: prompts prefilled, then one token per student per step; the second student leaves when done and its blocks return to the free list; then the fifth student (E, {LATE.tokens} tokens) arrives. Blocks in use: {REPLAY.in_use()} of {N_CELLS}; tokens {NEW_TOKENS} of {REPLAY.in_use() * BLOCK} slots = {100 * NEW_USED:.1f}% (vs {100 * OLD_USED:.0f}% with reservations); {100 * FREE_SHARE:.0f}% of the map still free. Waste = the unfilled end of each last block: {TAILS} slots (never more than 15). Kwon et al.: 'vLLM limits all the memory wastes for a request within one block'; abstract: 'near-zero waste'; vLLM blog (2023-06-20): 'a mere waste of under 4%'.")
        mm, cap, note = self.old_map()
        # rebuild section 1's final state
        fills, letters = {}, {}
        for s in STAYING:
            start, n = START[s.name], s.reserve // BLOCK
            for i in range(start, start + n):
                mm.cells[i].set_fill(VIOLET_FILL, 1).set_stroke(VIOLET, 1.6, opacity=0.55)
            fills[s.name] = mm.fills(list(range(start, start + n)), s.tokens)
            letters[s.name] = self.letter(mm, start, s.name)
        old_fills = m.VGroup(*[r for s in STAYING for r in fills[s.name]])
        old_letters = m.VGroup(*letters.values())
        refused = pill("E: no room", RED, RED_FILL, 22).move_to([3.6, -1.75, 0])
        legend = m.VGroup(swatch(KV, KV, "tokens"), swatch(RED, RED_FILL, "unused end of a block"), swatch(EDGE, CARD, "free"))
        legend.arrange(m.RIGHT, buff=0.6).move_to([0, -0.9, 0])
        old_stat = m.VGroup(title_text(f"{100 * OLD_USED:.0f}%", 56, RED), text("reserved stretches", 22, MUTED)).arrange(m.DOWN, buff=0.12)
        old_stat.move_to([-3.6, -2.75, 0])

        owner_new = {}
        for name, table in REPLAY.tables.items():
            for b in table:
                owner_new[b] = name
        moves, restyle = [], []
        for s in STAYING:
            for k, r in enumerate(fills[s.name]):
                target = mm.fill(REPLAY.tables[s.name][k], frac_of(s.tokens, k))
                moves.append(r.animate(path_arc=0.5).move_to(target.get_center()))
        for i, c in enumerate(mm.cells):
            if owner_new.get(i) in ("A", "C", "D"):
                restyle.append(c.animate.set_fill(VIOLET_FILL, 1).set_stroke(VIOLET, 1.6, opacity=1))
            else:
                restyle.append(freed(c))
        e_fills = mm.fills(REPLAY.tables["E"], LATE.tokens)
        e_cells = [mm.cells[i] for i in REPLAY.tables["E"]]
        admitted = pill("E: fits", GOOD, GOOD_FILL, 22).move_to(refused)
        tails = m.VGroup(*[
            mm.tail(REPLAY.tables[name][-1], frac_of(REPLAY.lengths[name], len(REPLAY.tables[name]) - 1))
            for name in ("A", "C", "D", "E")
        ])
        tail_cells = m.VGroup(*[mm.cells[REPLAY.tables[name][-1]] for name in ("A", "C", "D", "E")])
        new_stat = m.VGroup(title_text(f"{100 * NEW_USED:.0f}%", 56, GOOD), text("blocks in use that hold tokens", 22, INK)).arrange(m.DOWN, buff=0.12)
        new_stat.move_to([0.6, -2.75, 0])
        free_note = m.VGroup(code(f"{100 * FREE_SHARE:.0f}%", 30, INK, weight=m.BOLD), text("of the map still free", 22, MUTED)).arrange(m.DOWN, buff=0.12)
        free_note.move_to([4.4, -2.75, 0])
        nzw = m.VGroup(pill("near-zero waste", GOOD, GOOD_FILL, 22), cite("vLLM blog, 2023: under 4% wasted in practice"))
        nzw.arrange(m.RIGHT, buff=0.3).move_to([-2.2, -1.75, 0])

        self.say(
            "[Now replay our students with blocks.]",
            m.AnimationGroup(self.new_header("04 · packed", "Same students, no holes"), m.FadeIn(mm.cells), m.FadeIn(cap), m.FadeIn(note),
                             m.FadeIn(old_fills), m.FadeIn(old_letters), m.FadeIn(refused), m.FadeIn(old_stat)),
        )
        self.say(
            "[Each one now holds only the blocks it has filled]: [no reservations, and no holes.]",
            m.AnimationGroup(m.FadeOut(old_letters), *restyle, m.LaggedStart(*moves, lag_ratio=0.03), run_time=2.6),
            m.FadeIn(legend, shift=0.1 * m.UP),
        )
        self.say(
            "[And the fifth student gets in], [its blocks scattered through the free space.]",
            m.AnimationGroup(m.FadeOut(refused), m.FadeIn(admitted, scale=0.9)),
            m.LaggedStart(*[m.AnimationGroup(c.animate.set_fill(VIOLET_FILL, 1).set_stroke(REQUEST, 2.2, opacity=1),
                                             m.GrowFromEdge(r, m.LEFT)) for c, r in zip(e_cells, e_fills)], lag_ratio=0.15),
        )
        self.say(
            "[The only waste is the unfilled end of each request's last block]: [at most fifteen slots.]",
            m.LaggedStart(*[m.FadeIn(t) for t in tails], lag_ratio=0.25),
            m.LaggedStart(*[m.Circumscribe(c, color=RED, buff=0.05) for c in tail_cells], lag_ratio=0.2),
        )
        self.say(
            "[Here, ninety-six percent of the memory in use holds tokens], [and nearly two thirds of the map is still free.]",
            m.FadeIn(new_stat, shift=0.15 * m.UP),
            m.FadeIn(free_note, shift=0.15 * m.UP),
        )
        self.say(
            "[The paper calls this near-zero waste]; [its authors put it at under four percent in practice.]",
            m.FadeIn(nzw[0], scale=0.9),
            m.FadeIn(nzw[1], shift=0.1 * m.LEFT),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the catch

    def catch(self):
        self.next_section("5 · The catch, and why 16", notes="Kwon et al. 2023 §7.1: gathering through the block table 'leads to 20–26% higher attention kernel latency, compared to the highly-optimized FasterTransformer implementation'; the overhead is confined to attention. Abstract: vLLM 'improves the throughput of popular LLMs by 2-4× with the same level of latency' (vs FasterTransformer and Orca), because far more requests fit in each batch (Episode 4: one weight sweep serves the whole batch). §7.2: too small a block 'may not fully utilize the GPU's parallelism for reading and processing KV cache'; too large, 'internal fragmentation increases and the probability of sharing decreases'; 16 is the default. The 40-token comparison (40 / 3 / 1 blocks; 0 / 8 / 216 spare slots) is computed here and illustrative.")
        # contiguous vs blocks
        size, pitch = 0.42, 0.52
        left = m.VGroup(*[square(size, KV, KV) for _ in range(8)]).arrange(m.RIGHT, buff=pitch - size).move_to([-3.6, 1.0, 0])
        l_title = text("one stretch", 24, INK, weight=m.BOLD).move_to([-3.6, 2.0, 0])
        sweep = m.Arrow(left.get_left() + [0, -0.45, 0], left.get_right() + [0, -0.45, 0], buff=0, color=MEMORY, stroke_width=5,
                        max_tip_length_to_length_ratio=0.06)
        grid = m.VGroup(*[square(size, EDGE, CARD) for _ in range(24)]).arrange_in_grid(rows=3, cols=8, buff=pitch - size).move_to([3.6, 0.85, 0])
        picks = [16, 1, 11, 4, 21, 14, 7, 23]
        for i in picks:
            grid[i].set_fill(KV, 1).set_stroke(KV, 1.6)
        r_title = text("in blocks", 24, INK, weight=m.BOLD).move_to([3.6, 2.0, 0])
        hops = m.VGroup(*[
            m.Arrow(grid[a].get_center(), grid[b].get_center(), buff=0.18, color=MEMORY, stroke_width=3.5,
                    max_tip_length_to_length_ratio=0.15, path_arc=0.25)
            for a, b in zip(picks, picks[1:])
        ]).set_z_index(2)
        slower = m.VGroup(pill("attention kernel: 20–26% slower", RED, RED_FILL, 24)).move_to([-3.1, -1.25, 0])
        faster = m.VGroup(pill("whole server: 2–4× throughput", GOOD, GOOD_FILL, 24)).move_to([3.1, -1.25, 0])
        src = cite("Kwon et al., 2023 · same latency, vs earlier systems").move_to([3.1, -1.95, 0])
        ep4 = code("more students per weight sweep → Episode 4", 20, MUTED).move_to([0, -2.75, 0])

        # block sizes for one 40-token answer
        cols = m.VGroup()
        specs = [
            (SIZES[0], SIZE_BLOCKS[0], SIZE_SPARE[0], "more hops, GPU underused", MUTED),
            (SIZES[1], SIZE_BLOCKS[1], SIZE_SPARE[1], "vLLM's default", GOOD),
            (SIZES[2], SIZE_BLOCKS[2], SIZE_SPARE[2], "a big unused end", RED),
        ]
        pictures = []
        for b, n, spare, verdict, color in specs:
            if b == 1:
                pic = m.VGroup(*[m.Square(0.16).set_stroke(KV, 1.2).set_fill(KV, 1) for _ in range(n)]).arrange_in_grid(rows=5, cols=8, buff=0.07)
            elif b == 16:
                pic = m.VGroup(*[slot_block(0.62, filled=min(16, ANS40 - 16 * k), empty=RED_FILL) for k in range(n)]).arrange(m.RIGHT, buff=0.12)
            else:
                pic = slot_block(1.55, per_side=16, filled=ANS40, empty=RED_FILL)
            pictures.append(pic)
            lab = m.VGroup(code(f"{b} token{'s' if b > 1 else ''} per block", 22, INK, weight=m.BOLD),
                           code(f"{n} block{'s' if n > 1 else ''} · {spare} spare", 20, MUTED),
                           text(verdict, 22, color, weight=m.BOLD))
            lab.arrange(m.DOWN, buff=0.14)
            cols.add(m.VGroup(pic, lab))
        for c, x in zip(cols, [-4.6, 0, 4.6]):
            c[0].move_to([x, 0.85, 0])
            c[1].move_to([x, -1.15, 0])
        q_cap = text("one 40-token answer, three block sizes", 22, MUTED).move_to([0, -2.6, 0])

        self.say(
            "[The catch: attention now hops from block to block], [and in the paper, its kernel ran twenty to twenty-six percent slower.]",
            m.AnimationGroup(self.new_header("05 · the catch", "Hops, and why sixteen"), m.FadeIn(left), m.FadeIn(l_title), m.GrowArrow(sweep),
                             m.FadeIn(grid), m.FadeIn(r_title), m.LaggedStart(*[m.Create(h) for h in hops], lag_ratio=0.3)),
            m.FadeIn(slower, shift=0.1 * m.UP),
        )
        self.say(
            "[But attention is only part of each step]: [packing more students into every batch gave two to four times the throughput.]",
            m.Circumscribe(slower, color=RED, buff=0.06),
            m.AnimationGroup(m.FadeIn(faster, shift=0.1 * m.UP), m.FadeIn(src), m.FadeIn(ep4)),
        )
        self.say(
            "[Why sixteen tokens?] [Bigger blocks waste more at the end], [while tiny ones mean more hops], [and a GPU left underused.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(left, l_title, sweep, grid, r_title, hops, slower, faster, src, ep4)),
                             m.FadeIn(cols[1], shift=0.15 * m.UP), m.FadeIn(q_cap)),
            m.FadeIn(cols[2], shift=0.15 * m.UP),
            m.FadeIn(cols[0], shift=0.15 * m.UP),
            m.Circumscribe(cols[1][0], color=GOOD, buff=0.1),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ share

    def share(self):
        self.next_section("6 · Four answers, one prompt", notes=f"Parallel sampling (n = 4): the four answers' block tables point at the prompt's physical blocks, each with a reference count (Kwon et al. 2023 §4.4). The prompt's last block is partly filled; the first answer to write into it gets its own copy and the count drops (copy-on-write, Fig. 8); the last finds the count at 1 and writes in place. Simulated here with Paged: toy prompt {TOY_PROMPT} tokens → counts {REF_STEPS}; copies {[c[:2] for c in FORK.copies]}. Numbers, computed: a {PROMPT_T:,}-token prompt and {N_ANS} answers of {ANSWER_T} tokens: {NAIVE_BLOCKS} blocks if each answer stored the prompt ({NAIVE_BLOCKS * 2} MiB for Owl-8B) vs {SHARED_BLOCKS} shared ({SHARED_BLOCKS * 2} MiB): {100 * SAVED:.0f}% less. Kwon et al. report 6.1–9.8% saved on their parallel-sampling traces (short prompts, long outputs) and 37.6–55.2% on beam search; vLLM blog: up to 55%. Sharing prompts across different requests (prefix caching) is Episode 8. The flashcard request is illustrative.")
        msg = message("user", "Four flashcards from my notes, please", width=4.4, size=20).move_to([-4.15, 1.75, 0])
        n4 = pill("n = 4", REQUEST, PINK_FILL, 22, mono=True).next_to(msg, m.DOWN, buff=0.22).align_to(msg, m.LEFT)
        bsize = 0.95
        trunk = m.VGroup(*[slot_block(bsize, filled=f) for f in (16, 16, TOY_PROMPT - 2 * BLOCK)]).arrange(m.RIGHT, buff=0.22).move_to([1.55, 1.55, 0])
        t_cap = text("the prompt's blocks", 22, INK).next_to(trunk, m.RIGHT, buff=0.35)
        badges = [code("×4", 22, REQUEST, weight=m.BOLD).next_to(b, m.UP, buff=0.14) for b in trunk]
        lane_x = [-4.5, -1.5, 1.5, 4.5]
        lane_y = -0.05
        lanes = m.VGroup(*[text(f"answer {k + 1}", 22, TOKEN, weight=m.BOLD).move_to([x, lane_y, 0]) for k, x in enumerate(lane_x)])
        hub = trunk.get_bottom() + np.array([0, -0.12, 0])
        fan = m.VGroup(*[m.Line(l.get_top() + [0, 0.1, 0], hub).set_stroke(MUTED, 2, opacity=0.7) for l in lanes])
        naive = m.VGroup(*[
            m.VGroup(*[slot_block(0.42, filled=f, color=RED, fill=RED, stroke=1.6) for f in (16, 16, 8)]).arrange(m.RIGHT, buff=0.08).move_to([x, lane_y - 0.75, 0])
            for x in lane_x
        ])
        copy_y = lane_y - 0.78
        copies = [slot_block(0.72, filled=TOY_PROMPT - 2 * BLOCK).move_to([x, copy_y, 0]) for x in lane_x[:3]]
        for c in copies:
            c[1][TOY_PROMPT - 2 * BLOCK].set_fill(TOKEN, 1)
        in_place = trunk[2][1][TOY_PROMPT - 2 * BLOCK]
        cow = pill("copy on write", KV, VIOLET_FILL, 22).move_to([0, -1.62, 0])
        place = text("writes in place", 22, TOKEN).move_to([lane_x[3], copy_y, 0])
        place_line = m.DashedLine(lanes[3].get_top() + [0.25, 0.1, 0], trunk[2].get_bottom() + [0.25, -0.06, 0], dash_length=0.1).set_stroke(TOKEN, 2.5)

        def badge(n):
            return code(f"×{n}", 22, REQUEST if n > 1 else MUTED, weight=m.BOLD).move_to(badges[2])

        x0, full_w = -3.7, 4.2
        w2 = full_w * SHARED_BLOCKS / NAIVE_BLOCKS
        bar_naive = m.Rectangle(width=full_w, height=0.32).set_stroke(width=0).set_fill(RED, 0.85).move_to([x0 + full_w / 2, -2.2, 0])
        bar_shared = m.Rectangle(width=w2, height=0.32).set_stroke(width=0).set_fill(KV, 1).move_to([x0 + w2 / 2, -2.75, 0])
        lab_naive = code(f"{NAIVE_BLOCKS} blocks · {NAIVE_BLOCKS * 2} MiB", 20, INK).next_to(bar_naive, m.RIGHT, buff=0.2)
        lab_shared = code(f"{SHARED_BLOCKS} blocks · {SHARED_BLOCKS * 2} MiB", 20, INK).next_to(bar_shared, m.RIGHT, buff=0.2)
        name_naive = text("four copies", 22, MUTED).next_to(bar_naive, m.LEFT, buff=0.25)
        name_shared = text("shared", 22, MUTED).next_to(bar_shared, m.LEFT, buff=0.25)
        cut = title_text(f"−{100 * SAVED:.0f}%", 56, GOOD).move_to([5.55, -2.47, 0])
        setup = code(f"{PROMPT_T:,}-token prompt + {N_ANS} × {ANSWER_T}-token answers", 20, MUTED).move_to([-5.6, -1.6, 0], aligned_edge=m.LEFT)
        paper = cite("Kwon et al., 2023: 6–10% saved on their sampling traces, up to 55% on beam search").move_to([0, -3.4, 0])
        ep8 = pill("across students → Episode 8", MUTED, CARD, 20).move_to([4.3, -1.6, 0])
        mascot = owl(0.5).next_to(n4, m.RIGHT, buff=0.35)

        self.say(
            "[Blocks can also be shared.] [Owl's flashcard button asks for four different cards], [from the same notes.]",
            m.AnimationGroup(self.new_header("06 · share", "Four answers, one prompt"), m.FadeIn(trunk, shift=0.1 * m.DOWN), m.FadeIn(t_cap)),
            m.AnimationGroup(m.FadeIn(msg, shift=0.1 * m.RIGHT), m.FadeIn(mascot)),
            m.FadeIn(n4, shift=0.1 * m.UP),
        )
        self.say(
            "[Stored naively, the prompt's cache would be copied four times.]",
            m.AnimationGroup(m.FadeIn(lanes), m.LaggedStart(*[m.FadeIn(g, target_position=trunk) for g in naive], lag_ratio=0.2)),
        )
        self.say(
            "[With block tables], [all four answers point at the same prompt blocks], [and each block counts its users.]",
            m.FadeOut(naive, shift=0.2 * m.DOWN),
            m.LaggedStart(*[m.Create(f) for f in fan], lag_ratio=0.15),
            m.LaggedStart(*[m.FadeIn(b, shift=0.1 * m.DOWN) for b in badges], lag_ratio=0.2),
        )
        empty = m.VGroup(*trunk[2][1][TOY_PROMPT - 2 * BLOCK:])
        self.say(
            "[But the last prompt block is only half full], [and every answer wants to write its next token there.]",
            m.Circumscribe(trunk[2], color=INK, buff=0.08),
            m.Indicate(empty, color=TOKEN, scale_factor=1.15),
        )
        self.say(
            "[The first answer to write gets its own copy first]: [copy on write.] [The count drops to three.]",
            m.FadeIn(copies[0], target_position=trunk[2], path_arc=0.6),
            m.FadeIn(cow, scale=0.9),
            m.Transform(badges[2], badge(REF_STEPS[0])),
        )
        self.say(
            "[The next two do the same], [and the last one, now alone, simply writes in place.]",
            m.AnimationGroup(
                m.LaggedStart(*[m.FadeIn(c, target_position=trunk[2], path_arc=0.6) for c in copies[1:]], lag_ratio=0.5),
                m.Succession(m.Wait(0.5), m.Transform(badges[2], badge(REF_STEPS[1])), m.Transform(badges[2], badge(REF_STEPS[2]))),
            ),
            m.AnimationGroup(m.Create(place_line), m.FadeIn(place), in_place.animate.set_fill(TOKEN, 1)),
        )
        self.say(
            "[For example, with a thousand-token prompt and four answers of a hundred tokens], [that's ninety blocks instead of two hundred and seventy-six]: [about two thirds less memory.]",
            m.AnimationGroup(m.FadeOut(cow), m.FadeIn(setup), m.FadeIn(name_naive), m.GrowFromEdge(bar_naive, m.LEFT), m.FadeIn(lab_naive)),
            m.AnimationGroup(m.FadeIn(name_shared), m.GrowFromEdge(bar_shared, m.LEFT), m.FadeIn(lab_shared)),
            m.FadeIn(cut, scale=0.85),
        )
        self.say(
            "[The longer the shared prompt, the bigger the saving]; [and sharing a prompt between different students is next episode's trick.]",
            m.FadeIn(paper),
            m.FadeIn(ep8, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Block (fixed 16-token blocks, anywhere in memory), table (one block table per request, a new block only when the last is full), share (reference counts; copy only on write). Result: waste limited to one partial block per request (Kwon et al. 2023: 'near-zero waste'). Next: Episode 8, prefix caching.")
        words = [("Block", KV, "16 tokens, anywhere in memory"),
                 ("Table", REQUEST, "one per request"),
                 ("Share", TOKEN, "copy only on write")]
        cols = m.VGroup()
        for w, color, sub in words:
            cols.add(m.VGroup(title_text(w, 72, color), text(sub, 22, MUTED)).arrange(m.DOWN, buff=0.25))
        cols.arrange(m.RIGHT, buff=1.4).move_to([0, 0.75, 0])
        base = 0.85
        for c in cols:
            c[0].shift((base - c[0][0].get_bottom()[1]) * m.UP)  # B, T, S sit on one baseline
            c[1].move_to([c[0].get_x(), base - 0.55, 0])
        mid = base + 0.3
        arrows = m.VGroup(*[
            arrow([a[0].get_right()[0] + 0.15, mid, 0], [b[0].get_left()[0] - 0.15, mid, 0], MUTED) for a, b in zip(cols, cols[1:])
        ])
        tagline = text("Memory handed out sixteen tokens at a time.", 32, INK).move_to([0, -1.35, 0])
        mascot = owl(0.75).move_to([0, -2.7, 0])

        self.say(
            "[So: block], [table], [and share.]",
            m.AnimationGroup(self.new_header("recap", "PagedAttention in three words"), m.FadeIn(cols[0], shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowArrow(arrows[0]), m.FadeIn(cols[1], shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowArrow(arrows[1]), m.FadeIn(cols[2], shift=0.2 * m.UP)),
        )
        self.say(
            "[Memory is handed out sixteen tokens at a time], [so almost every byte of Owl's cache holds a token.]",
            m.FadeIn(tagline, shift=0.1 * m.UP),
            m.FadeIn(mascot, shift=0.2 * m.UP),
        )
        self.wait(0.4)
        self.clear()
