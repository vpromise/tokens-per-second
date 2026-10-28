"""Tokens per Second · Episode 10 · Continuous Batching (Module 3 · Many Users at Once).

Episode 4 showed that a batch shares each trip through the weights. But requests don't start or finish
together: static batching holds every seat until the longest request ends (padding), and a latecomer
waits for the whole batch. Continuous (iteration-level) batching, Orca's idea (OSDI 2022), re-forms the
batch after every step. The running example is owl_reference's request set on a 4-slot grid: request E
waits 12 steps under static batching and 2 under continuous. Then a busy stream under three schedulers,
Orca's code, selective batching and its 36.9x, mlx_lm.server measured on this Mac (1 / 4 / 16 clients),
and preemption when the KV cache runs out (Episode 7).

    VOICE=draft manimgx inspect scene.py      # silent layout pass
    manimgx render scene.py                   # with the shared narrator running
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from owl_reference import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- the numbers (all computed)

SLOTS = 4
REQS = [Request("A", 0, 8, 12), Request("B", 0, 8, 3), Request("C", 0, 8, 4), Request("D", 0, 8, 2),
        Request("E", 1, 8, 3)]                     # owl_reference's own check set
BY = {r.rid: r for r in REQS}
STATIC = static_batching(REQS, SLOTS)
CONT = continuous_batching(REQS, SLOTS)
assert STATIC == {"A": (0, 13), "B": (0, 4), "C": (0, 5), "D": (0, 3), "E": (13, 17)}
assert CONT == {"A": (0, 13), "B": (0, 4), "C": (0, 5), "D": (0, 3), "E": (3, 7)}
WAIT_S, WAIT_C = (d["E"][0] - BY["E"].arrival for d in (STATIC, CONT))
assert (WAIT_S, WAIT_C) == (12, 2)                 # "E waits 12 steps vs 2"


def makespan(done):
    return max(end for _, end in done.values())


END_S, END_C = makespan(STATIC), makespan(CONT)
assert (END_S, END_C) == (17, 13)
WORK = sum(1 + r.output for r in REQS)              # slot-steps of real work: a prompt step + one per token
FIRST = [r for r in REQS if STATIC[r.rid][0] == 0]
FIRST_END = max(STATIC[r.rid][1] for r in FIRST)
PAD_S = sum(FIRST_END - STATIC[r.rid][1] for r in FIRST)
FIRST_SEATS = SLOTS * FIRST_END
assert WORK == 29 and (PAD_S, FIRST_SEATS) == (27, 52) and PAD_S / FIRST_SEATS > 0.5 and SLOTS * END_C == 52
# step 3 of the continuous run: three decoding requests plus the latecomer's whole prompt, in one pass
STEP3 = [r for r in REQS if CONT[r.rid][0] <= 3 < CONT[r.rid][1]]
STEP3_TOKENS = sum(r.prompt if CONT[r.rid][0] == 3 else 1 for r in STEP3)
assert [r.rid for r in STEP3] == ["A", "B", "C", "E"] and STEP3_TOKENS == 11 and BY["E"].prompt == 8


def dynamic_batching(requests, slots, window):
    """Static batching that waits up to `window` steps (from the oldest waiting request) for a full batch.
    Not in owl_reference: this episode's own sketch, in its style."""
    queue, t, done = sorted(requests, key=lambda r: r.arrival), 0, {}
    while queue:
        t = max(t, queue[0].arrival)
        deadline = queue[0].arrival + window
        while len([r for r in queue if r.arrival <= t]) < slots and t < deadline:
            t += 1
        batch = [r for r in queue if r.arrival <= t][:slots]
        for r in batch:
            done[r.rid] = (t, t + 1 + r.output)
            queue.remove(r)
        t += 1 + max(r.output for r in batch)
    return done


# a busy server, for example: twelve requests arriving about one per step (illustrative lengths)
BUSY_ARR = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
BUSY_OUT = [7, 2, 4, 3, 3, 5, 2, 4, 6, 2, 4, 3]
BUSY = [Request(f"r{i}", a, 8, o) for i, (a, o) in enumerate(zip(BUSY_ARR, BUSY_OUT))]
WINDOW = 2
RUNS = {"static": static_batching(BUSY, SLOTS), "dynamic": dynamic_batching(BUSY, SLOTS, WINDOW),
        "continuous": continuous_batching(BUSY, SLOTS)}
BUSY_WORK = sum(1 + r.output for r in BUSY)


def busy_share(done):
    return BUSY_WORK / (SLOTS * makespan(done))


def mean_wait(done, requests=BUSY):
    return sum(done[r.rid][0] - r.arrival for r in requests) / len(requests)


assert [makespan(RUNS[k]) for k in RUNS] == [25, 22, 16]
assert [round(100 * busy_share(RUNS[k])) for k in RUNS] == [57, 65, 89]
assert [round(mean_wait(RUNS[k]), 1) for k in RUNS] == [5.3, 4.6, 1.2]

# measured on this Mac (research/measurements.md §10: mlx_lm.server 0.32.2, Owl-8B 4-bit, 128 tokens per request)
SRV = measured("server", "llama-3.1-8b__4bit")
CLIENTS = (1, 4, 16)
AGG = {c: SRV[str(c)]["aggregate_tok_s"] for c in CLIENTS}
EACH = {c: SRV[str(c)]["per_request_decode_tok_s"]["median"] for c in CLIENTS}
TTFT = {c: SRV[str(c)]["ttft_ms"]["p50"] for c in CLIENTS}
CEIL_Q4 = measured("decode", "llama-3.1-8b", "4bit", "ceiling_tok_s")       # 546 GB/s ÷ 4.52 GB
Q4_GB = measured("models", "llama-3.1-8b", "4bit", "weights_gb")
MLX_LM = measured("software", "mlx_lm")
assert measured("server_batches_concurrent_requests", "answer") is True and MLX_LM == "0.32.2"
assert [round(AGG[c]) for c in CLIENTS] == [90, 202, 290]
assert [round(EACH[c]) for c in CLIENTS] == [94, 64, 21]
assert [round(TTFT[c]) for c in CLIENTS] == [72, 551, 842]
assert round(CEIL_Q4) == 121 and round(MAC["bandwidth"] / (Q4_GB * 1e9)) == 121 and round(Q4_GB, 1) == 4.5
assert round(AGG[16] / AGG[1], 1) == 3.2 and EACH[1] < CEIL_Q4

ORCA_X = 36.9   # Yu et al., OSDI 2022 abstract: "36.9× throughput improvement at the same level of latency"

# preemption, schematic: a memory budget of 100 units, weights 30, each running request's KV +2 per step
CAP, W_UNITS, GROW = 100, 30, 2
KV0 = [("A", 12), ("B", 8), ("C", 9), ("D", 5)]


def grown(state, k=1):
    return [(rid, n + k * GROW) for rid, n in state]


KV1 = grown(KV0)
KV2 = KV1 + [("E", 6), ("F", 6)]
KV3 = grown(KV2)
KV4 = grown([s for s in KV3 if s[0] != "F"])


def used(state):
    return W_UNITS + sum(n for _, n in state)


assert used(KV1) == 72 and used(KV3) == 96 and used(KV3) + GROW * len(KV3) > CAP and used(KV4) == 98

USER = recolored("user", REQUEST, PINK_FILL, prefix="ep10")

# ---------------------------------------------------------------------------- helpers

_FLAT = set("0123456789abcdefhiklmnorstuvwxzABCDEFGHIKLMNOPRSTUVWXYZ")


def baseline(t):
    """The baseline of a Text: median bottom of its letters that sit on the line."""
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


def row(label, value, color, y, xl, xr, size=22):
    """A readout row: a MUTED label on the left, a coloured value right-aligned, on one baseline."""
    lab = left_at(code(label, size, MUTED), xl, y)
    val = right_at(code(value, size, color), xr, y)
    sit(val, baseline(lab))
    return lab, val


STYLE = {"prompt": (COMPUTE, AMBER_FILL), "token": (REQUEST, PINK_FILL), "pad": (RED, RED_FILL),
         "wait": (REQUEST, CARD)}


def cell(kind, x, y, size=0.5, letter=None, stroke=2.5):
    """One slot-step: a request's prompt step (AMBER), a token step (PINK), padding (RED) or waiting (hollow)."""
    color, fill = STYLE[kind]
    sq = m.Square(size).set_stroke(color, stroke if kind != "wait" else 2).move_to([x, y, 0])
    sq.set_fill(PINK, 0.42) if kind == "token" else sq.set_fill(fill, 1)
    group = m.VGroup(sq)
    if kind == "pad":  # hatched: a seat computing nothing useful
        h = size / 2
        for k in ((-0.5, 0.0, 0.5) if size > 0.4 else (0.0,)):
            c = k * size
            a, b = ((x - h, y - h + c), (x + h - c, y + h)) if c >= 0 else ((x - h - c, y - h), (x + h, y + h + c))
            group.add(m.Line([*a, 0], [*b, 0]).set_stroke(RED, 1.5, opacity=0.85))
    if letter:
        group.add(code(letter, 20, INK if kind == "prompt" else REQUEST, weight=m.BOLD).move_to(sq))
    group.kind = kind
    return group.set_z_index(2)


def seats(done, requests, slots, batched, slot_of=None):
    """Which slot runs what at every step: {(slot, step): (kind, rid)}. Static/dynamic batches hold their
    seats until the batch's longest request ends (the rest is padding); continuous takes the lowest free seat."""
    slot_of, cells, free_at = slot_of or {}, {}, [0] * slots
    order = sorted(requests, key=lambda r: (done[r.rid][0], r.arrival))
    if batched:
        for t0 in sorted({done[r.rid][0] for r in requests}):
            members = [r for r in order if done[r.rid][0] == t0]
            end = max(done[r.rid][1] for r in members)
            for i, r in enumerate(members):
                s = slot_of.get(r.rid, i)
                assert free_at[s] <= t0
                start, stop = done[r.rid]
                for t in range(start, end):
                    cells[(s, t)] = ("prompt" if t == start else "token" if t < stop else "pad", r.rid)
                free_at[s] = end
    else:
        for r in order:
            start, stop = done[r.rid]
            s = slot_of.get(r.rid, min(i for i in range(slots) if free_at[i] <= start))
            assert free_at[s] <= start
            for t in range(start, stop):
                cells[(s, t)] = ("prompt" if t == start else "token", r.rid)
            free_at[s] = stop
    return cells


class SlotGrid:
    """The 4-slot grid: rows are seats in the batch, columns are steps; a queue lane under it."""

    COLS, PITCH, SIZE = 17, 0.6, 0.5
    X0 = -4.2                            # centre of column 0
    ROWS = (1.75, 1.15, 0.55, -0.05)
    QUEUE = -0.85
    STEP = 2.35

    def __init__(self):
        x = self.x
        self.base = m.VGroup(*[
            m.VGroup(*[m.Square(self.SIZE).set_stroke(EDGE, 1.5).set_fill(CARD, 1).move_to([x(t), y, 0])
                       for t in range(self.COLS)]) for y in self.ROWS])
        self.labels = m.VGroup(*[right_at(code(f"slot {i + 1}", 20, MUTED), x(0) - 0.45, y) for i, y in enumerate(self.ROWS)])
        self.steps = m.VGroup(*[code(str(t), 20, MUTED).move_to([x(t), self.STEP, 0]) for t in range(self.COLS)])
        for s in self.steps[1:]:
            sit(s, baseline(self.steps[0]))
        self.step_cap = right_at(code("step", 20, MUTED), x(0) - 0.45, self.STEP)
        sit(self.step_cap, baseline(self.steps[0]))
        width = self.COLS * self.PITCH + 0.06
        self.track = m.RoundedRectangle(corner_radius=0.12, width=width, height=0.66).set_stroke(EDGE, 1.5)
        self.track.set_fill(CARD, 0.35).move_to([(x(0) + x(self.COLS - 1)) / 2, self.QUEUE, 0])
        self.queue_cap = right_at(code("queue", 20, MUTED), x(0) - 0.45, self.QUEUE)
        top, bottom = self.STEP + 0.3, self.QUEUE - 0.42
        self.marker = m.RoundedRectangle(corner_radius=0.08, width=0.62, height=top - bottom).set_stroke(INK, 2.5)
        self.marker.set_fill(BG, 0).move_to([x(0), (top + bottom) / 2, 0]).set_z_index(5)
        self.marker_y = (top + bottom) / 2

    def x(self, t):
        return self.X0 + self.PITCH * t

    def run(self, done, batched, slot_of=None):
        """Every cell of one schedule, by step: {step: [cell, ...]}, waiting cells {step: cell}, and the
        latecomers' first cells (to glide up from the queue)."""
        spec = seats(done, REQS, SLOTS, batched, slot_of)
        by_t, first = defaultdict(list), {}
        for (s, t), (kind, rid) in sorted(spec.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            c = cell(kind, self.x(t), self.ROWS[s], self.SIZE, rid if kind == "prompt" else None)
            by_t[t].append(c)
            if kind == "prompt" and BY[rid].arrival < t:
                first[rid] = c
        waits = {}
        for r in REQS:
            for t in range(r.arrival, done[r.rid][0]):
                waits[t] = cell("wait", self.x(t), self.QUEUE, self.SIZE, r.rid if t == r.arrival else None)
        return by_t, waits, first

    def step(self, t, by_t, waits, first, run_time=0.32, with_wait=True):
        """One step of the clock: the marker moves to column t and that column's cells appear."""
        parts = [self.marker.animate.move_to([self.x(t), self.marker_y, 0])]
        gliders = [id(c) for c in first.values()]
        for c in by_t.get(t, []):
            if id(c) in gliders:
                parts.append(m.FadeIn(c, target_position=waits_last(waits, t)))
            else:
                parts.append(m.FadeIn(c, scale=0.6))
        if t in waits and with_wait:
            parts.append(m.FadeIn(waits[t], scale=0.6))
        return m.AnimationGroup(*parts, run_time=run_time)


def waits_last(waits, t):
    """Where a latecomer was waiting just before step t (its last queue cell)."""
    return waits[max(k for k in waits if k < t)].get_center()


class Strip:
    """A small slot grid for the busy-server comparison (4 rows × 25 steps)."""

    COLS, PITCH, SIZE = 25, 0.33, 0.26
    X0 = -3.35

    def __init__(self, y, done, batched):
        self.y = y
        self.ys = [y + (1.5 - i) * self.PITCH for i in range(SLOTS)]
        self.base = m.VGroup(*[m.Square(self.SIZE).set_stroke(EDGE, 1.2).set_fill(CARD, 1).move_to([self.x(t), yy, 0])
                               for yy in self.ys for t in range(self.COLS)])
        spec = seats(done, BUSY, SLOTS, batched)
        self.by_t = defaultdict(list)
        for (s, t), (kind, _) in spec.items():
            assert t < self.COLS
            self.by_t[t].append(cell(kind, self.x(t), self.ys[s], self.SIZE, stroke=1.8))
        self.end = makespan(done)
        h = SLOTS * self.PITCH + 0.08
        self.marker = m.RoundedRectangle(corner_radius=0.05, width=0.34, height=h).set_stroke(INK, 2).set_fill(BG, 0)
        self.marker.move_to([self.x(0), y, 0]).set_z_index(5)

    def x(self, t):
        return self.X0 + self.PITCH * t

    def sweep(self, step_time=0.085):
        steps = []
        for t in range(self.end):
            parts = [self.marker.animate.move_to([self.x(t), self.y, 0])]
            parts += [m.FadeIn(c, scale=0.6) for c in self.by_t.get(t, [])]
            steps.append(m.AnimationGroup(*parts, run_time=step_time))
        return m.Succession(m.FadeIn(self.marker, run_time=0.1), *steps)


def line_mark(block, key, color):
    """A gutter mark beside one code line and that line in `color` (mono text: glyph indexing is safe)."""
    body = block[1]
    g = glyphs(body, key)
    mark = m.RoundedRectangle(corner_radius=0.03, width=0.08, height=max(g.height, 0.2)).set_stroke(width=0)
    mark.set_fill(color, 1).move_to([block[0].get_left()[0] + 0.17, g.get_y(), 0])
    return m.AnimationGroup(m.FadeIn(mark, shift=0.1 * m.RIGHT), g.animate.set_color(color)), mark


# ---------------------------------------------------------------------------- the episode


class Ep10ContinuousBatching(Episode):
    number = 10

    def lesson(self):
        self.hook()
        self.static_run()
        self.continuous_run()
        self.busy()
        self.orca()
        self.on_this_mac()
        self.preemption()
        self.recap()

    # ------------------------------------------------------------------ 1 · hook

    def hook(self):
        self.next_section(
            "1 · Nobody finishes together",
            notes="Episode 4 in one line: a batch shares each trip through the weights. Here the batch has four "
                  "seats (slots). The requests are owl_reference's own check set: A, B, C, D arrive at step 0 "
                  "needing 12, 3, 4 and 2 tokens; E arrives at step 1 needing 3. Each request takes one step for "
                  "its prompt (prefilled in one step, as in owl_reference) plus one step per token. Token counts are "
                  f"toy-sized on purpose. static_batching: E starts at step {STATIC['E'][0]} (waits {WAIT_S}); "
                  f"continuous_batching: E starts at step {CONT['E'][0]} (waits {WAIT_C}).")
        G = SlotGrid
        ys = G.ROWS
        x_left = G.X0 - G.SIZE / 2

        def bar(cells_, y, start=0, dashed=False):
            w = cells_ * G.PITCH - (G.PITCH - G.SIZE)
            r = m.RoundedRectangle(corner_radius=0.08, width=w, height=G.SIZE).move_to(
                [x_left + start * G.PITCH + w / 2, y, 0])
            if dashed:
                return m.DashedVMobject(r.set_stroke(REQUEST, 2.5), num_dashes=28)
            return r.set_stroke(REQUEST, 2.5).set_fill(PINK, 0.42)

        users = m.VGroup(*[icon(USER, 0.42).move_to([x_left - 0.55, y, 0]) for y in ys])
        equal = [bar(9, y) for y in ys]
        bars = m.VGroup(*equal)
        batch_box = m.RoundedRectangle(corner_radius=0.15, width=9 * G.PITCH + 0.2, height=ys[0] - ys[3] + G.SIZE + 0.3)
        batch_box = m.DashedVMobject(batch_box.set_stroke(REQUEST, 2), num_dashes=60).move_to(
            [x_left + 9 * G.PITCH / 2 - 0.05, (ys[0] + ys[3]) / 2, 0])
        one = left_at(text("one batch", 28, REQUEST, weight=m.BOLD), 1.3, 1.25)
        shared = m.VGroup(code("16 GB of weights per step,", 20, WEIGHTS), code("shared by all four", 20, WEIGHTS)
                          ).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
        left_at(shared, 1.3, 0.5)
        mascot = owl(0.6).move_to([5.6, -2.55, 0])

        x_end = x_left + 9 * G.PITCH - (G.PITCH - G.SIZE)
        top, bottom = ys[0] + 0.45, ys[3] - 0.45
        start_line = m.DashedLine([x_left, top, 0], [x_left, bottom, 0], dash_length=0.1).set_stroke(MUTED, 2)
        end_line = m.DashedLine([x_end, top, 0], [x_end, bottom, 0], dash_length=0.1).set_stroke(MUTED, 2)
        start_lab = left_at(code("start together", 20, MUTED), x_left, bottom - 0.3)
        end_lab = left_at(code("finish together", 20, MUTED), x_end + 0.15, bottom - 0.3)

        real = [bar(1 + BY[r].output, y) for r, y in zip("ABCD", ys)]
        labels = m.VGroup()
        for r, y, b in zip("ABCD", ys, real):
            n = BY[r].output
            labels.add(code(f"{r} · {n} tokens", 20, INK, t2c={r: REQUEST}).next_to(b, m.RIGHT, buff=0.25))
        e_user = icon(USER, 0.42).move_to([x_left - 0.55, G.QUEUE, 0])
        e_bar = bar(1 + BY["E"].output, G.QUEUE, start=BY["E"].arrival, dashed=True)
        e_lab = code(f"E · {BY['E'].output} tokens · arrives one step later", 20, INK, t2c={"E": REQUEST}).next_to(
            e_bar, m.RIGHT, buff=0.25)

        p_static = pill(f"wait for the whole batch: {WAIT_S} steps", RED, RED_FILL, 22)
        p_cont = pill(f"take the next free seat: {WAIT_C} steps", GREEN, GREEN_FILL, 22)
        left_at(p_static, x_left - 0.75, -1.95)
        left_at(p_cont, x_left - 0.75, -2.75)

        self.say(
            "[In episode four, batching was a free lunch]: [students shared every trip through the weights], "
            "[for about the cost of one.]",
            m.AnimationGroup(self.new_header("01 · the problem", "Nobody finishes together"),
                             m.LaggedStart(*[m.FadeIn(u, shift=0.1 * m.RIGHT) for u in users], lag_ratio=0.15),
                             m.LaggedStart(*[m.GrowFromEdge(b, m.LEFT) for b in equal], lag_ratio=0.1, run_time=1.4)),
            m.AnimationGroup(m.Create(batch_box), m.FadeIn(one, shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.FadeIn(shared, shift=0.1 * m.UP), m.FadeIn(mascot, shift=0.15 * m.UP)),
        )
        self.say(
            "[But there, everyone started together], [and finished together.]",
            m.AnimationGroup(m.FadeOut(batch_box), m.Create(start_line), m.FadeIn(start_lab)),
            m.AnimationGroup(m.Create(end_line), m.FadeIn(end_lab)),
        )
        self.say(
            "[Real students don't.] [Some need a short answer, some a long one], [and new ones keep arriving.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(end_line, end_lab, one, shared)),
                             m.Circumscribe(bars, color=REQUEST, buff=0.12)),
            m.AnimationGroup(*[m.Transform(a, b) for a, b in zip(equal, real)],
                             m.LaggedStart(*[m.FadeIn(lab, shift=0.1 * m.LEFT) for lab in labels], lag_ratio=0.15)),
            m.AnimationGroup(m.FadeOut(start_lab), m.FadeIn(e_user, shift=0.1 * m.RIGHT), m.Create(e_bar),
                             m.FadeIn(e_lab, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[So when does a newcomer get a seat?] [Wait for the whole batch, and it waits twelve steps.] "
            "[Take the next free seat, and it waits two.]",
            m.Circumscribe(m.VGroup(e_user, e_bar), color=REQUEST, buff=0.1),
            m.FadeIn(p_static, shift=0.1 * m.UP),
            m.FadeIn(p_cont, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 2 · static batching

    def static_run(self):
        self.next_section(
            "2 · Static batching",
            notes=f"owl_reference.static_batching(REQS, 4) = {STATIC}. A batch starts together and ends when its "
                  f"longest request ends (t += 1 + max output). Cells: AMBER = the request's prompt step, PINK = one "
                  f"token, RED = padding (a finished request's seat, still computed until the batch ends), hollow = "
                  f"waiting in the queue. First batch: {FIRST_SEATS} slot-steps (4 × {FIRST_END}), of which "
                  f"{PAD_S} are padding ({100 * PAD_S / FIRST_SEATS:.0f}%). E arrives at step 1 and starts at "
                  f"{STATIC['E'][0]}: waits {WAIT_S} steps; everything done at step {END_S}. Orca on request-level "
                  "batching: 'requests that have finished earlier than other requests in a batch cannot return to "
                  "the client, while newly arrived requests have to wait until the current batch completely finishes.'")
        G = self.grid = SlotGrid()
        by_t, waits, first = G.run(STATIC, batched=True, slot_of={"E": 3})
        step = lambda t, rt=0.32: G.step(t, by_t, waits, first, rt)  # noqa: E731

        def steps(ts, rt=0.3):
            return m.Succession(*[step(t, rt) for t in ts])

        sq = lambda kind: cell(kind, 0, 0, 0.28, stroke=2)  # noqa: E731
        legend = m.VGroup()
        for kind, word in (("prompt", "prompt"), ("token", "one token"), ("wait", "waiting"), ("pad", "padding")):
            color = STYLE[kind][0] if kind != "wait" else REQUEST
            legend.add(m.VGroup(sq(kind), code(word, 20, color)).arrange(m.RIGHT, buff=0.15))
        legend.arrange(m.RIGHT, buff=0.6).move_to([0.6, -1.65, 0])

        xs = (-4.0, 0.3, 4.2)
        labs = [sit(text(s, 22, MUTED).move_to([x, -2.3, 0]), -2.4) for s, x in
                zip(("the latecomer waited", "padding in slot-steps", "everything done"), xs)]
        vals = [sit(title_text(s, 44, RED).move_to([x, -2.93, 0]), -3.12) for s, x in
                zip((f"{WAIT_S} steps", f"{PAD_S} of {FIRST_SEATS}", f"step {END_S}"), xs)]
        self.stats = (labs, vals)
        self.legend = legend

        col1 = m.VGroup(*by_t[1])
        pads3 = m.VGroup(*[c for c in by_t[3] if c.kind == "pad"])
        pads5 = m.VGroup(*[c for c in by_t[5] if c.kind == "pad"])
        queue_cells = m.VGroup(*[waits[t] for t in sorted(waits)])
        all_pads = m.VGroup(*[c for t in by_t for c in by_t[t] if c.kind == "pad"])

        self.say(
            "[Here is Owl's server with four seats]: [each row is one seat in the batch], "
            "[and each column is one step of the model.]",
            m.AnimationGroup(self.new_header("02 · static batching", "The batch waits for its longest"),
                             m.LaggedStart(*[m.FadeIn(r_, shift=0.1 * m.UP) for r_ in G.base], lag_ratio=0.15)),
            m.LaggedStart(*[m.FadeIn(lab, shift=0.1 * m.RIGHT) for lab in G.labels], lag_ratio=0.15),
            m.AnimationGroup(m.FadeIn(G.step_cap), m.LaggedStart(*[m.FadeIn(s) for s in G.steps], lag_ratio=0.05),
                             m.Succession(m.Wait(0.8), m.FadeIn(G.marker))),
        )
        self.say(
            "[Static batching takes four requests]: [each spends its first step on the prompt], [then one step per token.]",
            step(0),
            m.AnimationGroup(m.Circumscribe(m.VGroup(*by_t[0]), color=COMPUTE, buff=0.06), m.FadeIn(legend[0])),
            m.AnimationGroup(G.step(1, by_t, waits, first, with_wait=False), m.FadeIn(legend[1])),
        )
        self.say(
            "[Right then, the latecomer arrives], [but all four seats are taken.]",
            m.AnimationGroup(m.FadeIn(G.track), m.FadeIn(G.queue_cap), m.FadeIn(legend[2]),
                             m.FadeIn(waits[1], shift=0.3 * m.RIGHT)),
            m.Circumscribe(col1, color=REQUEST, buff=0.06),
        )
        self.say(
            "[At step three, the shortest request is done.] [But its seat isn't freed]: "
            "[it computes padding until the batch ends.]",
            steps([2, 3]),
            m.Circumscribe(pads3, color=RED, buff=0.06),
            m.FadeIn(legend[3], shift=0.1 * m.UP),
        )
        self.say(
            "[Then the next two finish], [and their seats fill with padding too.]",
            steps([4, 5], 0.35),
            m.Circumscribe(pads5, color=RED, buff=0.06),
        )
        self.say(
            "[The long request keeps going], [and the latecomer keeps waiting], [step after step.]",
            steps([6, 7, 8], 0.28),
            steps([9, 10], 0.28),
            steps([11, 12], 0.28),
        )
        self.say(
            "[Only at step thirteen is the batch done], [and the latecomer finally gets a seat], "
            "[finishing at step seventeen.]",
            m.AnimationGroup(G.marker.animate.move_to([G.x(13), G.marker_y, 0]), run_time=0.5),
            step(13, 0.8),
            m.AnimationGroup(steps([14, 15, 16], 0.3), m.FadeIn(labs[2]), m.FadeIn(vals[2], shift=0.1 * m.UP)),
        )
        self.say(
            "[It waited twelve steps.] [And more than half the first batch was padding]: "
            "[twenty-seven of fifty-two slot-steps.]",
            m.AnimationGroup(m.Circumscribe(queue_cells, color=RED, buff=0.06), m.FadeIn(labs[0]),
                             m.FadeIn(vals[0], shift=0.1 * m.UP)),
            m.Circumscribe(all_pads, color=RED, buff=0.08),
            m.AnimationGroup(m.FadeIn(labs[1]), m.FadeIn(vals[1], shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.static_cells = m.VGroup(*[c for t in by_t for c in by_t[t]], *waits.values())

    # ------------------------------------------------------------------ 3 · continuous batching

    def continuous_run(self):
        self.next_section(
            "3 · Continuous batching",
            notes=f"owl_reference.continuous_batching(REQS, 4) = {CONT}. Iteration-level scheduling: after every "
                  f"step, finished requests leave and waiting ones take the free seats. D finishes at step 3 and E "
                  f"takes its seat at once: waits {WAIT_C} steps (static: {WAIT_S}). No padding. Everything is done "
                  f"at step {END_C} (static: {END_S}) with the same {WORK} slot-steps of work. Seats left empty here "
                  "are free because nobody else is waiting (the busy server comes next).")
        G = self.grid
        by_t, waits, first = G.run(CONT, batched=False)
        step = lambda t, rt=0.32: G.step(t, by_t, waits, first, rt)  # noqa: E731

        def steps(ts, rt=0.3):
            return m.Succession(*[step(t, rt) for t in ts])

        labs, vals = self.stats
        new_vals = [title_text(s, 44, GREEN).move_to(v) for s, v in
                    zip((f"{WAIT_C} steps", f"0 of {SLOTS * END_C}", f"step {END_C}"), vals)]
        for nv, v in zip(new_vals, vals):
            sit(nv, baseline(v))
        olds = [code(f"was {s}", 20, FAINT) for s in (WAIT_S, PAD_S, END_S)]
        for o, nv in zip(olds, new_vals):
            o.next_to(nv, m.RIGHT, buff=0.2)
            sit(o, baseline(nv))
        freed = m.VGroup(*[G.base[s][t] for s, t in ((1, 4), (1, 5), (1, 6), (2, 5), (2, 6))])

        self.say(
            "[Now the same four requests, under continuous batching]: "
            "[after every step, the scheduler re-forms the batch.]",
            m.AnimationGroup(self.new_header("03 · continuous batching", "Re-form the batch every step"),
                             m.FadeOut(self.static_cells), m.FadeOut(m.VGroup(*vals)),
                             G.marker.animate.move_to([G.x(0), G.marker_y, 0])),
            m.Succession(step(0, 0.4), step(1, 0.4), step(2, 0.4)),
        )
        self.say(
            "[At step three, the shortest request finishes], [and its seat goes straight to the latecomer.]",
            m.AnimationGroup(G.marker.animate.move_to([G.x(3), G.marker_y, 0]), m.Circumscribe(m.VGroup(*by_t[2])[-1:], color=GREEN, buff=0.06)),
            step(3, 0.9),
        )
        self.say(
            "[It waited two steps, not twelve.]",
            m.AnimationGroup(m.FadeIn(new_vals[0], shift=0.1 * m.UP), m.FadeIn(olds[0])),
        )
        self.say(
            "[The others leave the moment they're done]: [no padding], [and every freed seat is open to the queue.]",
            steps([4, 5, 6, 7], 0.32),
            m.AnimationGroup(m.FadeIn(new_vals[1], shift=0.1 * m.UP), m.FadeIn(olds[1])),
            m.AnimationGroup(steps([8, 9, 10, 11, 12], 0.25), m.Circumscribe(freed, color=GREEN, buff=0.06)),
        )
        self.say(
            "[The whole batch is done at step thirteen instead of seventeen], [doing exactly the same work.]",
            m.AnimationGroup(G.marker.animate.move_to([G.x(13), G.marker_y, 0]), m.FadeIn(new_vals[2], shift=0.1 * m.UP),
                             m.FadeIn(olds[2])),
            m.Circumscribe(m.VGroup(*[c for t in by_t for c in by_t[t]]), color=REQUEST, buff=0.1),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 4 · a busy server

    def busy(self):
        s, d, c = (RUNS[k] for k in ("static", "dynamic", "continuous"))
        self.next_section(
            "4 · A busy server",
            notes="Illustrative stream (for example): twelve requests arriving at steps "
                  f"{BUSY_ARR} needing {BUSY_OUT} tokens, four seats. Scheduled by owl_reference.static_batching, "
                  f"this episode's dynamic_batching (static batching that waits up to {WINDOW} steps from the oldest "
                  "waiting request to fill the batch) and owl_reference.continuous_batching. Steps to finish: "
                  f"{makespan(s)} / {makespan(d)} / {makespan(c)}; seats busy (work ÷ seats × steps, work = "
                  f"{BUSY_WORK} slot-steps): {100 * busy_share(s):.0f}% / {100 * busy_share(d):.0f}% / "
                  f"{100 * busy_share(c):.0f}%; mean wait {mean_wait(s):.1f} / {mean_wait(d):.1f} / {mean_wait(c):.1f} "
                  "steps. Names vary: llama.cpp's server calls its continuous batching 'a.k.a dynamic batching' (notes §5).")
        strips = [Strip(1.4, s, True), Strip(-0.42, d, True), Strip(-2.24, c, False)]
        names = [("Static", "per request", RED), ("Dynamic", "waits to fill", MUTED), ("Continuous", "every step", GREEN)]
        lefts, rights = [], []
        for st, (name, sub, color), run in zip(strips, names, (s, d, c)):
            n = left_at(title_text(name, 30, INK), -6.5, st.y + 0.2)
            sb = left_at(code(sub, 20, MUTED), -6.5, st.y - 0.3)
            lefts.append(m.VGroup(n, sb))
            a = left_at(code(f"{makespan(run)} steps", 22, INK), 5.0, st.y + 0.2)
            b = left_at(code(f"{100 * busy_share(run):.0f}% busy", 22, color), 5.0, st.y - 0.25)
            rights.append(m.VGroup(a, b))
        counts = defaultdict(int)
        dots = m.VGroup()
        top = strips[0].ys[0] + 0.32
        for a in BUSY_ARR:
            dots.add(m.Dot([strips[0].x(a), top + 0.13 * counts[a], 0], radius=0.05, color=REQUEST))
            counts[a] += 1
        arr_lab = right_at(code("arrivals", 20, REQUEST), strips[0].x(0) - 0.3, top + 0.05)
        example = right_at(cite("for example"), 6.5, top + 0.05)
        names_note = text("llama.cpp's server: “continuous batching (a.k.a dynamic batching)”", 22, MUTED)
        names_note.move_to([0, -3.25, 0])
        names_pill = pill("names vary", MUTED, CARD, 20).next_to(names_note, m.LEFT, buff=0.3)
        m.VGroup(names_pill, names_note).set_x(0)

        self.say(
            "[On a busy server, the difference grows.] [Say twelve requests arrive, about one per step.]",
            m.AnimationGroup(self.new_header("04 · a busy server", "Three ways to batch"),
                             *[m.FadeIn(st.base) for st in strips], *[m.FadeIn(l_, shift=0.1 * m.RIGHT) for l_ in lefts]),
            m.AnimationGroup(m.FadeIn(arr_lab), m.LaggedStart(*[m.FadeIn(dt, scale=0.5) for dt in dots], lag_ratio=0.12),
                             m.FadeIn(example)),
        )
        self.say(
            "[Static batching takes twenty-five steps], [with the seats busy only fifty-seven percent of the time.]",
            strips[0].sweep(),
            m.FadeIn(rights[0], shift=0.1 * m.LEFT),
        )
        self.say(
            "[Dynamic batching waits a moment to fill each batch]: [twenty-two steps, sixty-five percent], "
            "[still stuck behind the longest request.]",
            strips[1].sweep(),
            m.FadeIn(rights[1], shift=0.1 * m.LEFT),
            m.Circumscribe(m.VGroup(*[x for t in strips[1].by_t for x in strips[1].by_t[t] if x.kind == "pad"]),
                           color=RED, buff=0.05),
        )
        self.say(
            "[Continuous batching, also called iteration-level batching], "
            "[needs sixteen steps, with the seats busy eighty-nine percent of the time.]",
            strips[2].sweep(0.11),
            m.AnimationGroup(m.FadeIn(rights[2], shift=0.1 * m.LEFT), m.Circumscribe(rights[2], color=GREEN, buff=0.1)),
        )
        self.say(
            "[Careful with the names, though]: [llama.cpp calls its continuous batching dynamic batching.]",
            m.FadeIn(names_pill, shift=0.1 * m.UP),
            m.FadeIn(names_note, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 5 · Orca

    def orca(self):
        self.next_section(
            "5 · Orca: iteration-level scheduling",
            notes="Yu et al., 'Orca: A Distributed Serving System for Transformer-Based Generative Models', OSDI 2022 "
                  "(usenix.org/conference/osdi22/presentation/yu, read 2026-10-24). Abstract: iteration-level "
                  "scheduling 'schedules execution at the granularity of iteration (instead of request)'; selective "
                  "batching 'applies batching only to a selected set of operations'; 'Our evaluation on a GPT-3 175B "
                  "model shows that ORCA can significantly outperform NVIDIA FasterTransformer in terms of both "
                  "latency and throughput: 36.9× throughput improvement at the same level of latency.' Quote card from "
                  "the paper (notes §5). Code: owl_reference.static_batching / continuous_batching, simplified (the "
                  "waiting list is named; names unchanged). The selective-batching picture is step 3 of the continuous "
                  f"run: A, B, C decode one token each, E's {BY['E'].prompt}-token prompt joins: {STEP3_TOKENS} tokens "
                  "through the matmuls in one pass, attention per request against its own KV cache.")
        name = title_text("Orca", 72, INK).move_to([0, 1.55, 0])
        src = cite("Yu et al., OSDI 2022").next_to(name, m.DOWN, buff=0.22)
        quote_lines = m.VGroup(text("“a newly arrived request can be considered for processing", 28, INK),
                               text("after waiting for only a single iteration of the model.”", 28, INK)
                               ).arrange(m.DOWN, buff=0.16, aligned_edge=m.LEFT)
        q_card = card(quote_lines.width + 0.9, quote_lines.height + 0.8, REQUEST, CARD, radius=0.2, stroke=2)
        quote = m.VGroup(q_card, quote_lines.move_to(q_card)).move_to([0, -0.85, 0])
        idea = pill("iteration-level scheduling", REQUEST, PINK_FILL, 24).move_to([0, -2.6, 0])

        static_src = "t += 1 + max(r.output for r in batch)"
        s_block = code_block(static_src, 24, title="static_batching · owl_reference.py").move_to([0, 0.5, 0])
        s_block[1].set_z_index(2)
        s_note = text("the clock jumps past the longest request", 26, MUTED).next_to(s_block, m.DOWN, buff=0.45)
        cont_src = ("while queue or running:\n"
                    "    waiting = [r for r in queue if r.arrival <= t]\n"
                    "    for r in waiting[: slots - len(running)]:\n"
                    "        running[r.rid] = [t, 1 + r.output]\n"
                    "        queue.remove(r)\n"
                    "    t += 1\n"
                    "    for rid in list(running):\n"
                    "        running[rid][1] -= 1\n"
                    "        if running[rid][1] == 0:\n"
                    "            done[rid] = (running.pop(rid)[0], t)")
        c_block = code_block(cont_src, 20, title="continuous_batching · owl_reference.py, simplified")
        c_block.move_to([0, -0.1, 0])
        c_block[1].set_z_index(2)
        hl_max, mk_max = line_mark(s_block, "max(r.output for r in batch)", RED)
        hl_step, mk_step = line_mark(c_block, "t += 1", COMPUTE)
        hl_join, mk_join = line_mark(c_block, "running[r.rid] = [t, 1 + r.output]", REQUEST)
        hl_leave, mk_leave = line_mark(c_block, "done[rid] = (running.pop(rid)[0], t)", GREEN)

        self.say(
            "[The idea comes from Orca, in twenty twenty-two]: [schedule every iteration of the model, not every request.]",
            m.AnimationGroup(self.new_header("05 · iteration-level", "Orca's idea"), m.FadeIn(name, shift=0.15 * m.UP),
                             m.FadeIn(src)),
            m.AnimationGroup(m.FadeIn(quote, shift=0.1 * m.UP), m.FadeIn(idea, shift=0.1 * m.UP)),
        )
        self.say(
            "[In our reference code, static batching moves the clock past its longest request.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(name, src, quote, idea), shift=0.15 * m.UP),
                             m.FadeIn(s_block, shift=0.15 * m.UP, run_time=0.7),
                             m.Succession(m.Wait(0.9), m.AnimationGroup(hl_max, m.FadeIn(s_note, shift=0.1 * m.UP)))),
        )
        s_group = m.VGroup(s_block, mk_max, s_note)
        self.say(
            "[Continuous batching moves it one step at a time]: [waiting requests join when a seat is free], "
            "[and finished ones leave at once.]",
            m.AnimationGroup(m.FadeOut(s_group, shift=0.15 * m.UP), m.FadeIn(c_block, shift=0.15 * m.UP, run_time=0.7),
                             m.Succession(m.Wait(0.9), hl_step)),
            hl_join,
            hl_leave,
        )
        self.wait(0.3)
        code_group = m.VGroup(c_block, mk_step, mk_join, mk_leave)

        # selective batching: step 3 of the continuous run
        groups = [("A", 1, REQUEST, PINK_FILL), ("B", 1, REQUEST, PINK_FILL), ("C", 1, REQUEST, PINK_FILL),
                  ("E", BY["E"].prompt, COMPUTE, AMBER_FILL)]
        assert [g[0] for g in groups] == [r.rid for r in STEP3] and sum(g[1] for g in groups) == STEP3_TOKENS
        tok_groups = m.VGroup(*[
            m.VGroup(*[m.RoundedRectangle(corner_radius=0.05, width=0.36, height=0.36).set_stroke(c, 2).set_fill(f, 1)
                       for _ in range(n)]).arrange(m.RIGHT, buff=0.08)
            for _, n, c, f in groups]).arrange(m.RIGHT, buff=0.55).move_to([0, 1.65, 0])
        tok_labs = m.VGroup(*[code(rid if n == 1 else f"{rid} · its {n}-token prompt", 20, c).next_to(g, m.DOWN, buff=0.15)
                              for (rid, n, c, _), g in zip(groups, tok_groups)])
        on_baseline(tok_labs)
        caption = cite(f"step 3 of the continuous run: {STEP3_TOKENS} tokens in one pass").move_to([0, 2.35, 0])
        mm = card(10.4, 1.25, WEIGHTS, BLUE_FILL, radius=0.16).move_to([0, -0.1, 0])
        mm_text = m.VGroup(text("matrix multiplies, batched", 26, INK, weight=m.BOLD),
                           code(f"one pass through the weights · all {STEP3_TOKENS} tokens", 20, WEIGHTS)).arrange(m.DOWN, buff=0.14)
        mm_text.move_to(mm)
        downs = m.VGroup(*[arrow([g.get_x(), tok_labs.get_bottom()[1] - 0.05, 0], [g.get_x(), mm.get_top()[1], 0],
                                 groups[i][2]) for i, g in enumerate(tok_groups)])
        att_x = (-4.65, -1.55, 1.55, 4.65)
        atts = m.VGroup()
        for (rid, _, c, _), x in zip(groups, att_x):
            box = card(2.8, 1.2, KV, VIOLET_FILL, radius=0.14).move_to([x, -2.0, 0])
            words = m.VGroup(text(f"attention · {rid}", 22, INK, weight=m.BOLD), code("own KV cache", 20, KV)
                             ).arrange(m.DOWN, buff=0.14).move_to(box)
            atts.add(m.VGroup(box, words))
        splits = m.VGroup(*[arrow([x, mm.get_bottom()[1], 0], [x, atts[i].get_top()[1], 0], KV) for i, x in enumerate(att_x)])
        sel = cite("selective batching · Yu et al., OSDI 2022").move_to([0, -3.2, 0])

        self.say(
            "[But how can requests at different points share a step?] "
            "[Orca batches the big matrix multiplies over all their tokens], "
            "[and runs attention separately for each request.]",
            m.AnimationGroup(m.FadeOut(code_group, shift=0.15 * m.UP),
                             m.LaggedStart(*[m.FadeIn(g, shift=0.1 * m.DOWN) for g in tok_groups], lag_ratio=0.2),
                             m.FadeIn(tok_labs), m.FadeIn(caption)),
            m.AnimationGroup(m.LaggedStart(*[m.GrowArrow(a) for a in downs], lag_ratio=0.1), m.FadeIn(mm), m.FadeIn(mm_text)),
            m.AnimationGroup(m.LaggedStart(*[m.GrowArrow(a) for a in splits], lag_ratio=0.1),
                             m.LaggedStart(*[m.FadeIn(a, shift=0.1 * m.DOWN) for a in atts], lag_ratio=0.15), m.FadeIn(sel)),
        )
        diagram = m.VGroup(tok_groups, tok_labs, caption, mm, mm_text, downs, atts, splits, sel)
        halo = glow(GREEN, radius=3.4).move_to([0, 0.5, 0])
        big = title_text(f"{ORCA_X:g}×", 120, GREEN).move_to([0, 0.75, 0])
        what = text("the throughput of FasterTransformer, at the same latency", 30, INK).move_to([0, -0.75, 0])
        who = cite("Orca vs NVIDIA FasterTransformer · GPT-3 175B · Yu et al., OSDI 2022").move_to([0, -1.5, 0])
        engines = m.VGroup(*[pill(e, MUTED, CARD, 22, mono=True) for e in ("vLLM", "SGLang", "TensorRT-LLM", "llama-server")],
                           pill("mlx_lm.server", REQUEST, PINK_FILL, 22, mono=True)).arrange(m.RIGHT, buff=0.3).move_to([0, -2.75, 0])
        for e in engines[1:]:
            e[1].move_to(e[0])
            sit(e[1], baseline(engines[0][1]))
        today = cite("iteration-level scheduling today").next_to(engines, m.UP, buff=0.25)
        self.say(
            "[On the GPT-3 model, Orca measured thirty-six point nine times the throughput of FasterTransformer], "
            "[at the same latency.]",
            m.AnimationGroup(m.FadeOut(diagram, shift=0.15 * m.UP), m.FadeIn(halo, scale=0.6), m.FadeIn(big, scale=0.85)),
            m.AnimationGroup(m.FadeIn(what, shift=0.1 * m.UP), m.FadeIn(who)),
        )
        self.say(
            "[Today, every major serving engine schedules this way], [including the MLX server on this Mac.]",
            m.AnimationGroup(m.FadeIn(today), m.LaggedStart(*[m.FadeIn(e, shift=0.1 * m.UP) for e in engines[:4]], lag_ratio=0.2)),
            m.AnimationGroup(m.FadeIn(engines[4], shift=0.1 * m.UP), m.Succession(m.Wait(0.6), m.Circumscribe(engines[4], color=REQUEST, buff=0.08))),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 6 · measured on this Mac

    def on_this_mac(self):
        self.next_section(
            "6 · Measured on this Mac",
            notes=f"research/measurements.md §10 (measured 2026-10-24, Mac Studio M4 Max 546 GB/s): mlx_lm.server "
                  f"{MLX_LM} serving mlx-community/Meta-Llama-3.1-8B-Instruct-4bit ({Q4_GB:.2f} GB), --temp 0 "
                  "--max-tokens 128; tools/stream_client.py streams /v1/chat/completions, closed loop, C clients × R "
                  "requests (1 × 10, 4 × 5, 16 × 3), distinct ~64-token prompts. The server routes requests through "
                  "BatchGenerator (continuous batching; --decode-concurrency 32). Aggregate tok/s "
                  f"{', '.join(f'{AGG[c]:.0f}' for c in CLIENTS)}; per-request decode tok/s (median) "
                  f"{', '.join(f'{EACH[c]:.1f}' for c in CLIENTS)}; TTFT p50 "
                  f"{', '.join(f'{TTFT[c]:.0f}' for c in CLIENTS)} ms. Ceiling for one stream: 546 GB/s ÷ "
                  f"{Q4_GB:.2f} GB = {CEIL_Q4:.0f} tok/s. 16 vs 1 client: ×{AGG[16] / AGG[1]:.1f} total. The TTFT "
                  "growth is waiting while the server interleaves newcomers with everyone's decode steps — the "
                  "measurement's reading, server internals not instrumented.")
        term = terminal([
            "$ mlx_lm.server --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
            "$ python stream_client.py --clients 16 --requests 3 --max-tokens 128",
        ], width=11.8, size=20, title=f"this Mac · mlx-lm {MLX_LM}").move_to([0, 1.75, 0])

        base_y, scale = -1.75, 2.0 / AGG[16]
        xs = (-4.25, -2.55, -0.85)
        axis = m.Line([-5.0, base_y, 0], [-0.1, base_y, 0]).set_stroke(FAINT, 2.5)
        bars, vals = [], []
        for c, x in zip(CLIENTS, xs):
            h = AGG[c] * scale
            bars.append(m.Rectangle(width=1.0, height=h).set_stroke(TOKEN, 2.5).set_fill(GREEN_FILL, 1).move_to([x, base_y + h / 2, 0]))
            vals.append(code(f"{AGG[c]:.0f}", 22, TOKEN).next_to(bars[-1], m.UP, buff=0.14))
        y_cap = left_at(text("total tokens per second", 22, TOKEN), -5.0, 0.55)
        rows_y = (-2.15, -2.6, -3.05)
        row_labs = [right_at(code(s, 20, MUTED), -5.15, y) for s, y in zip(("students", "each", "TTFT"), rows_y)]
        n_row = [code(str(c), 22, REQUEST).move_to([x, rows_y[0], 0]) for c, x in zip(CLIENTS, xs)]
        e_row = [code(f"{EACH[c]:.0f} tok/s", 20, REQUEST).move_to([x, rows_y[1], 0]) for c, x in zip(CLIENTS, xs)]
        t_row = [code(f"{TTFT[c]:.0f} ms", 20, INK).move_to([x, rows_y[2], 0]) for c, x in zip(CLIENTS, xs)]
        for r_, lab in zip((n_row, e_row, t_row), row_labs):
            for item in r_:
                sit(item, baseline(lab))

        X, XR = 0.9, 6.4
        name = left_at(text("Owl-8B · 4-bit", 28, INK, weight=m.BOLD), X, 0.45)
        size = left_at(code(f"{Q4_GB:.1f} GB of weights", 20, WEIGHTS), X, 0.0)
        src = left_at(cite("measured on this Mac · mlx-lm"), X, -0.4)
        r1 = row("one stream", f"{EACH[1]:.0f} tok/s", REQUEST, -1.0, X, XR)
        r2 = row("ceiling", f"{CEIL_Q4:.0f} tok/s", MEMORY, -1.45, X, XR)
        r2b = left_at(code(f"546 GB/s ÷ {Q4_GB:.1f} GB", 20, MUTED), X, -1.85)
        r3 = row("16 vs 1 student", f"×{AGG[16] / AGG[1]:.1f} total", TOKEN, -2.45, X, XR)
        wait_note = left_at(text(f"alone: {TTFT[1]:.0f} ms · the rest is waiting", 22, INK), X, -2.95)
        reading = left_at(cite("our reading · server internals not instrumented"), X, -3.35)

        self.say(
            "[So let's measure it]: [MLX's server on this Mac], [with one, four and sixteen students at once.]",
            m.AnimationGroup(self.new_header("06 · measured on this Mac", "Sixteen students, one Mac"),
                             m.FadeIn(term, shift=0.15 * m.DOWN)),
            m.AnimationGroup(m.FadeIn(name, shift=0.1 * m.UP), m.FadeIn(size), m.FadeIn(src)),
            m.AnimationGroup(m.FadeIn(axis), m.FadeIn(y_cap), *[m.FadeIn(lab) for lab in row_labs[:1]],
                             m.LaggedStart(*[m.FadeIn(n) for n in n_row], lag_ratio=0.25)),
        )
        self.say(
            "[One student gets ninety-four tokens a second], [against a ceiling of a hundred and twenty-one.]",
            m.AnimationGroup(m.GrowFromEdge(bars[0], m.DOWN), m.FadeIn(vals[0]), m.FadeIn(row_labs[1]), m.FadeIn(e_row[0]),
                             m.FadeIn(m.VGroup(*r1))),
            m.AnimationGroup(m.FadeIn(m.VGroup(*r2)), m.FadeIn(r2b)),
        )
        self.say(
            "[Four students get two hundred and two in total], [and sixteen get two hundred and ninety]: "
            "[over three times one stream.]",
            m.AnimationGroup(m.GrowFromEdge(bars[1], m.DOWN), m.FadeIn(vals[1]), m.FadeIn(e_row[1])),
            m.AnimationGroup(m.GrowFromEdge(bars[2], m.DOWN), m.FadeIn(vals[2]), m.FadeIn(e_row[2])),
            m.FadeIn(m.VGroup(*r3), shift=0.1 * m.UP),
        )
        self.say(
            "[Each stream slows to twenty-one tokens a second], [still faster than anyone reads], "
            "[but the first token now takes most of a second.]",
            m.Circumscribe(e_row[2], color=REQUEST, buff=0.08),
            m.Circumscribe(m.VGroup(*e_row), color=REQUEST, buff=0.1),
            m.AnimationGroup(m.FadeIn(row_labs[2]), m.LaggedStart(*[m.FadeIn(t) for t in t_row], lag_ratio=0.3)),
        )
        self.say(
            "[Alone, the first token took seventy-two milliseconds]: "
            "[the rest is waiting for a turn among everyone else's steps.]",
            m.Circumscribe(t_row[0], color=INK, buff=0.08),
            m.AnimationGroup(m.FadeIn(wait_note, shift=0.1 * m.UP), m.FadeIn(reading), m.Circumscribe(t_row[2], color=RED, buff=0.08)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 7 · preemption

    def preemption(self):
        self.next_section(
            "7 · Preemption",
            notes="Schematic (for example): a memory budget of 100 units; weights 30; each running request's KV "
                  f"cache grows by {GROW} units per step. Four running ({used(KV1)}% after a step), two more admitted, "
                  f"one more step → {used(KV3)}%: the next step would need {used(KV3) + GROW * len(KV3)}%. The scheduler "
                  f"preempts the newest (F): frees its cache, puts it back in the queue; the others continue "
                  f"({used(KV4)}%). vLLM docs (notes §16): when the KV cache is full and a running sequence needs a new "
                  "block, the scheduler preempts a request and either recomputes it later or swaps it to CPU; V1: 'the "
                  "default preemption mode is RECOMPUTE rather than SWAP'. Episode 7: PagedAttention's 16-token blocks; "
                  "Episode 25 covers preemption in production (KV saturation alerts).")
        y, h, x0, w = 1.0, 0.72, -5.5, 11.0

        def u(n):
            return w * n / CAP

        track = m.Rectangle(width=w, height=h).set_stroke(EDGE, 2).set_fill(CARD, 1).move_to([0, y, 0])
        cap = left_at(code("GPU memory · schematic", 20, MUTED), x0, y + 0.62)
        wrect = m.Rectangle(width=u(W_UNITS) - 0.04, height=h - 0.1).set_stroke(WEIGHTS, 2).set_fill(BLUE_FILL, 1)
        wrect.move_to([x0 + u(W_UNITS) / 2, y, 0])
        wlab = code("weights", 20, INK).move_to(wrect)

        def layout(state):
            x, out = x0 + u(W_UNITS), {}
            for rid, n in state:
                rect = m.Rectangle(width=u(n) - 0.04, height=h - 0.1).set_stroke(KV, 2).set_fill(VIOLET_FILL, 1)
                rect.move_to([x + u(n) / 2, y, 0])
                out[rid] = m.VGroup(rect, code(rid, 20, INK, weight=m.BOLD).move_to(rect)).set_z_index(2)
                x += u(n)
            return out

        L0, L1, L2, L3, L4 = (layout(s) for s in (KV0, KV1, KV2, KV3, KV4))
        seg = dict(L0)
        seg.update({k: L2[k] for k in ("E", "F")})

        def to(layout_, keys):
            return m.AnimationGroup(*[m.Transform(seg[k], layout_[k]) for k in keys], run_time=0.6)

        sq = lambda c, f: m.Square(0.24).set_stroke(c, 2).set_fill(f, 1)  # noqa: E731
        leg_w = m.VGroup(sq(WEIGHTS, BLUE_FILL), code("weights · one copy", 20, WEIGHTS)).arrange(m.RIGHT, buff=0.2)
        leg_k = m.VGroup(sq(KV, VIOLET_FILL), code("KV cache · one per running request", 20, KV)).arrange(m.RIGHT, buff=0.2)
        left_at(leg_w, x0, 0.1)
        left_at(leg_k, x0 + 4.0, 0.1)
        q_y = -1.0
        q_lab = left_at(code("queue", 20, MUTED), x0, q_y)
        chips = {rid: pill(rid, REQUEST, PINK_FILL, 22, mono=True).move_to([x0 + 1.6 + 0.8 * i, q_y, 0]) for i, rid in enumerate("EF")}
        full = right_at(code(f"{used(KV3)}% · no room for the next step", 20, RED), x0 + w, y + 0.62)
        sit(full, baseline(cap))
        back = pill("F", REQUEST, PINK_FILL, 22, mono=True).move_to(chips["E"])
        gone = pill("preempted · recompute later, or swap to CPU", RED, RED_FILL, 20).next_to(back, m.RIGHT, buff=0.3)
        ep7 = pill("episode 7 · PagedAttention", KV, VIOLET_FILL, 22).move_to([0, -2.05, 0])
        ep7_text = text("the cache grows in 16-token blocks: preempt only when no free block is left", 24, INK).move_to([0, -2.75, 0])
        vllm = cite("vLLM docs · V1 recomputes preempted requests by default").move_to([0, -3.3, 0])

        self.say(
            "[One more limit: memory.] [Every running request keeps a KV cache], [and it grows by one token every step.]",
            m.AnimationGroup(self.new_header("07 · preemption", "When the cache runs out"), m.FadeIn(track), m.FadeIn(cap),
                             m.FadeIn(wrect), m.FadeIn(wlab), m.FadeIn(leg_w)),
            m.AnimationGroup(*[m.GrowFromEdge(seg[k], m.LEFT) for k in "ABCD"], m.FadeIn(leg_k)),
            to(L1, "ABCD"),
        )
        self.say(
            "[Admit two more], [and soon there's no room left for the next step's tokens.]",
            m.AnimationGroup(m.FadeIn(q_lab), m.FadeIn(chips["E"], run_time=0.4), m.FadeIn(chips["F"], run_time=0.4),
                             m.Succession(m.Wait(0.55), m.AnimationGroup(
                                 m.FadeOut(chips["E"], target_position=L2["E"]), m.FadeIn(seg["E"], target_position=chips["E"]),
                                 m.FadeOut(chips["F"], target_position=L2["F"]), m.FadeIn(seg["F"], target_position=chips["F"]),
                                 run_time=0.8))),
            m.AnimationGroup(to(L3, "ABCDEF"), m.Succession(m.Wait(0.5), m.AnimationGroup(
                m.FadeIn(full), m.Circumscribe(track, color=RED, buff=0.06)))),
        )
        self.say(
            "[So the scheduler preempts one]: [it frees that request's cache and sends it back to the queue], "
            "[to be recomputed later.]",
            m.Circumscribe(seg["F"], color=RED, buff=0.06),
            m.AnimationGroup(m.FadeOut(seg["F"], target_position=back), m.FadeIn(back, target_position=seg["F"]),
                             m.FadeIn(gone, shift=0.1 * m.LEFT), m.FadeOut(full)),
            to(L4, "ABCDE"),
        )
        self.say(
            "[Episode seven's PagedAttention hands out this memory in small blocks], "
            "[so a request is preempted only when no free block is left.]",
            m.FadeIn(ep7, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(ep7_text, shift=0.1 * m.UP), m.FadeIn(vllm)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Free a seat the step its request finishes (no padding); refill it from the queue "
                                         "at the very next step; repeat every iteration (Orca's iteration-level "
                                         "scheduling). Memory still limits the batch: preempt when the KV cache is full.")
        halo = glow(REQUEST, radius=4.0)
        words = m.VGroup(*[title_text(w, 80, c) for w, c in (("Free", GREEN), ("Refill", REQUEST), ("Repeat", COMPUTE))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("a seat, when done", "from the queue", "every step")])
        words.arrange(m.RIGHT, buff=1.3)
        on_baseline(words)
        for w, s in zip(words, subs):
            s.set_x(w.get_x())
        line_y = min(w.get_bottom()[1] for w in words) - 0.45
        for s in subs:
            s.set_y(line_y)
        mid = words[0][0].get_center()[1]
        dots = m.VGroup(*[m.Dot([(a.get_right()[0] + b.get_left()[0]) / 2, mid, 0], radius=0.07, color=MUTED)
                          for a, b in zip(words, words[1:])])
        row_ = m.VGroup(words, dots, subs).move_to(0.45 * m.UP)
        tagline = text("Re-form the batch every step, not every request.", 30, MUTED).next_to(row_, m.DOWN, buff=0.75)

        self.say(
            "[To sum up]: [free], [refill], [repeat.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[Free a seat the moment its request is done, refill it from the queue, and do it again every single step.]",
            m.Write(tagline),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row_, tagline)], run_time=0.9)

