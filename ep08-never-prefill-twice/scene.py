"""Tokens per Second · Episode 8 · Never Prefill Twice (Module 2 · The KV Cache)

A shared prefix can be cached and reused. Running example: a student (Ana) pastes one chapter of her course
notes into Owl and asks fifty questions about it; every request also starts with Owl's system prompt.

Sections: hook (each question re-sends the chapter: 8 s of prefill each on this Mac, ≈ 6.7 min for fifty) ·
shared prefixes (every request starts with the system prompt; Ana's questions all repeat her chapter; a
token's K and V depend only on the tokens before it, so the same prefix gives the same cache) · keep the
blocks (Episode 7's 16-token blocks outlive the request; the next question's block table points at them;
only its own block is prefilled) · block hashes (vLLM: each full block's hash covers its 16 tokens and its
parent's hash, so it names the whole prefix; lookup hit, hit … first miss; one changed token near the start
breaks every later hash; the agent side is "Agents, Animated" Episode 7) · SGLang's radix tree (edges hold
token runs, a new question lights up its longest match, the rest is prefilled and hung on as a branch; LRU
eviction of leaves; longest-match-first scheduling; hit rates 50–99%) · measured on this Mac (mlx-lm prompt
cache: TTFT 8.01 s → 0.137 s, 58×, same answer; the 906 MB cache file = 6,910 tokens × 128 KiB; fifty
questions 400 s → 15 s) · pricing (a hit skips ≥ 111 TFLOP of prefill and costs only memory; Anthropic
reads 0.1× (0.05× on Opus/Sonnet 5.5), writes 1.25×, 5-minute TTL refreshed on use; OpenAI automatic,
"up to 95%" off; the session bill 50 → 6.15 chapter-prices) · the catch (one H100 next to Owl's weights
holds ≈ 61 cached chapters; cold prefixes are evicted) · recap.

Sources (facts dated October 2026):
- research/notes.md §4 (prefix caching): vLLM Automatic Prefix Caching ("we hash each kv-cache block by the
  tokens in the block and the tokens in the prefix before the block"; "We only cache full blocks"; LRU
  eviction; V1 on by default) — checked against the vLLM design doc (docs.vllm.ai, design/prefix_caching,
  read 2026-10-24: hash(parent hash, token ids, extra keys such as LoRA ids); a request's lookup hashes its
  prompt block by block). SGLang RadixAttention (Zheng et al., NeurIPS 2024; LMSYS blog, Jan 2024: "Each
  tree edge carries a label denoting a substring or a sequence of tokens"; "an LRU eviction policy that
  recursively evicts leaf nodes"; "we sort the requests by matched prefix length and prioritize requests
  with longer matched prefixes"; "the cache hit rate ranges from 50% to 99%"). Owl's 1,024-token system
  prompt = 64 blocks is the notes' worked example (illustrative).
- research/notes.md §0 #14 and §11 (context caching APIs, official docs read 2026-10-23, DATED): Anthropic
  cache reads 0.1× base input (0.05× on Opus 5.5 and Sonnet 5.5), 5-minute cache writes 1.25×, "By default,
  the cache has a 5-minute lifetime. The cache is refreshed for no additional cost each time the cached
  content is used."; OpenAI "Prompt caching is enabled by default", cached input "discounted up to 95%".
- research/measurements.md §7 (this Mac Studio, M4 Max, mlx-lm 0.32.2, 2026-10-24): Llama-3.1-8B-Instruct
  4-bit, chapter prefix 6,910 tokens + 45-token question: TTFT 8.011 s without the cache, 0.137 s with it
  (median of 5; 58×; 0.160 s including loading the file from the OS file cache); same greedy answer; cache
  file 906 MB. (The BF16 model's cached answer drifted in wording after ~16 tokens: different summation
  order; not shown.)
- owl_reference.py: kv_bytes_per_token (128 KiB), flops_per_token (2·P), weight_bytes, GPU (H100 80 GB);
  Episode 5's H100 budget (90% of 80 GB minus 16.06 GB of weights ≈ 56 GB for the cache).

Illustrative (said "for example" or schematic): the student names and questions, the four requests and
their segment widths (not to scale), the block-to-memory placements, the six 16-token blocks and their token
ids (seeded random; the hashes are really computed by block_hashes below), the radix tree's branches, idle
times and queue percentages, the session bill (chapter tokens only, every question within the 5-minute
window), the fifty-question totals (measured TTFTs × 50; the first question builds the cache).

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

PREFIX = "ep08"
KIB, MIB, GIB = 2**10, 2**20, 2**30
BLOCK = 16  # Episode 7: vLLM's default block size

# measured on this Mac (research/measurements.md §7): Owl-8B 4-bit, mlx-lm's prompt cache
PC = measured("prompt_cache", "llama-3.1-8b", "4bit")
CHAPTER, QUESTION = PC["prefix_tokens"], PC["suffix_tokens"]
T_COLD, T_WARM = PC["ttft_no_cache_s"], PC["ttft_with_cache_s"]
SPEEDUP = T_COLD / T_WARM
FILE_MB = PC["cache_file_MB"]
assert CHAPTER == 6910 and QUESTION == 45  # "six thousand nine hundred and ten tokens"
assert T_COLD == measured("headline", "prompt_cache_8b_4bit_ttft_no_cache_s")
assert round(T_COLD) == 8 and round(T_COLD, 2) == 8.01  # "eight seconds"
assert round(T_WARM, 3) == 0.137 and T_WARM < 1 / 7  # "under a seventh of a second"
assert round(SPEEDUP) == 58  # "fifty-eight times sooner"
assert PC["same_greedy_output"] and PC["answer_no_cache"] == PC["answer_with_cache"]  # "same answer, word for word"
ANSWER = PC["answer_with_cache"].split(". ")[0] + "."
assert ANSWER == "Llama-3.1-8B stores 128 KiB of KV cache per token in BF16."
CHAPTER_KV = CHAPTER * kv_bytes_per_token()  # the chapter's keys and values, BF16 cache
assert kv_bytes_per_token() == 128 * KIB
assert round(FILE_MB) == 906 and abs(CHAPTER_KV / 1e6 - FILE_MB) < 0.5  # "906 MB: 6,910 tokens at 128 KiB each"
assert CHAPTER // BLOCK == 431  # full 16-token blocks in the chapter

# fifty questions about the chapter (the first one builds the cache)
QUESTIONS = 50
SESSION_COLD = QUESTIONS * T_COLD
SESSION_WARM = T_COLD + (QUESTIONS - 1) * T_WARM
assert 6.5 < SESSION_COLD / 60 < 7 and round(SESSION_COLD) == 401  # "almost seven minutes"
assert round(SESSION_WARM) == 15  # "about fifteen seconds"

# Owl's system prompt (notes §4 worked example)
SYSTEM = 1024
assert SYSTEM // BLOCK == 64
SYSTEM_MS = prefill_time(SYSTEM) * 1000  # one H100 at 50% MFU (notes §4: "about 33 ms of H100 time")
assert round(SYSTEM_MS) == 33  # "thirty-three milliseconds"

# what a hit skips, and what it keeps
SKIPPED = flops_per_token() * CHAPTER  # 2·P·n, attention not counted
assert 100e12 < SKIPPED and round(SKIPPED / 1e12) == 111  # "over a hundred trillion operations"
assert round(CHAPTER_KV / 1e6) == 906

# provider prices (notes §0 #14, §11; dated October 2026), in units of the base input price
READ, READ_NEWER, WRITE, OPENAI_OFF = 0.1, 0.05, 1.25, 0.95
BILL_COLD = QUESTIONS * 1.0
BILL_WARM = WRITE + (QUESTIONS - 1) * READ
BILL_NEWER = WRITE + (QUESTIONS - 1) * READ_NEWER
assert BILL_COLD == 50 and round(BILL_WARM, 2) == 6.15 and round(BILL_WARM) == 6  # "fifty … about six"
assert round(BILL_NEWER, 2) == 3.7

# the catch: one H100 serving Owl-8B (Episode 5's budget)
KV_ROOM = 0.9 * GPU["memory"] - weight_bytes()
CHAPTERS_FIT = int(KV_ROOM // CHAPTER_KV)
assert round(KV_ROOM / 1e9) == 56 and CHAPTERS_FIT == 61 and round(CHAPTERS_FIT, -1) == 60  # "about sixty"


# ---------------------------------------------------------------------------- the block hashes (vLLM-style)


def block_hashes(tokens, size=16):
    hashes, parent = [], None
    for i in range(size, len(tokens) + 1, size):
        block = tuple(tokens[i - size:i])
        parent = hash((parent, block))  # chained
        hashes.append(parent)
    return hashes


HASH_SRC = inspect.getsource(block_hashes).rstrip()
assert len(HASH_SRC.splitlines()) <= 10


def cached_tokens(tokens, table, size=BLOCK):
    """Tokens a new prompt can skip: its blocks' hashes, looked up in order, up to the first miss."""
    hits = 0
    for h in block_hashes(tokens, size):
        if h not in table:
            break
        hits += 1
    return hits * size


_rng = np.random.default_rng(8)
_ids = lambda n: [int(t) for t in _rng.integers(0, OWL["vocab"], n)]  # illustrative token ids
SYS_IDS, CH_IDS, Q1_IDS, Q2_IDS = _ids(2 * BLOCK), _ids(3 * BLOCK), _ids(BLOCK), _ids(BLOCK)
P1 = SYS_IDS + CH_IDS + Q1_IDS
P2 = SYS_IDS + CH_IDS + Q2_IDS
CHANGED = BLOCK + 4  # one token in the second block (still the system prompt)
P3 = list(P2)
P3[CHANGED] = (P3[CHANGED] + 1) % OWL["vocab"]
TABLE = set(block_hashes(P1))
H1, H2, H3 = block_hashes(P1), block_hashes(P2), block_hashes(P3)
assert len(H1) == len(H2) == len(H3) == 6
assert H1[:5] == H2[:5] and H1[5] != H2[5]
assert cached_tokens(P2, TABLE) == 5 * BLOCK  # "hit, hit, hit, hit, hit … the first miss"
assert H3[0] == H1[0] and all(h not in TABLE for h in H3[1:]) and cached_tokens(P3, TABLE) == BLOCK


def short(h):
    return f"{h & 0xFFFF:04x}"


assert len({short(h) for h in H1 + H2[5:] + H3[1:]}) == 6 + 1 + 5  # every hash on screen looks different

# ---------------------------------------------------------------------------- look


def cite(s, size=20):
    return text(s, size, MUTED)


def seg(w, label, color, fill, h=0.55, size=20):
    """One part of a prompt: a box with a label."""
    box = m.RoundedRectangle(corner_radius=0.08, width=w, height=h, stroke_color=color, stroke_width=2,
                             fill_color=fill, fill_opacity=1)
    words = text(label, size, INK).move_to(box)
    return m.VGroup(box, words)


def strip(parts, x0, y, gap=0.06):
    """A prompt drawn as segments from left to right: parts = [(width, label, colour, fill), …]."""
    segs, x = [], x0
    for w, label, color, fill in parts:
        s = seg(w, label, color, fill)
        s.move_to([x + w / 2, y, 0])
        segs.append(s)
        x += w + gap
    return m.VGroup(*segs)


SYS_LOOK = (MUTED, CARD)
USER_LOOK = (REQUEST, PINK_FILL)


def token_block(label, colors, w=1.55, h=0.95, edge=EDGE):
    """A 16-token block: a label and a 2 × 8 grid of token cells, coloured by where the tokens came from."""
    box = card(w, h, edge, CARD, radius=0.12, stroke=2)
    cells = m.VGroup(*[m.Square(0.12).set_stroke(width=0).set_fill(c, 1) for c in colors])
    cells.arrange_in_grid(rows=2, cols=8, buff=0.04)
    words = text(label, 20, INK).move_to(box).shift(0.2 * m.UP)
    cells.move_to(box).shift(0.2 * m.DOWN)
    return m.VGroup(box, words, cells)


def hash_pill(h, color=KV, fill=VIOLET_FILL):
    return pill(short(h), color, fill, 22, mono=True)


def mark(ok):
    return text("✓ hit", 22, GOOD, weight=m.BOLD) if ok else text("✗ miss", 22, COMPUTE, weight=m.BOLD)


def elbow(p, c, stub=0.35, color=KV, width=4):
    """A tree edge: right from the parent, down (or up) to the child's height, then right to the child."""
    p, c = np.array([*p[:2], 0], float), np.array([*c[:2], 0], float)
    if abs(p[1] - c[1]) < 1e-6:
        return m.Line(p, c).set_stroke(color, width)
    x1 = p[0] + stub
    path = m.VMobject().set_points_as_corners([p, [x1, p[1], 0], [x1, c[1], 0], c])
    return path.set_stroke(color, width)


def tree_node(at, r=0.13):
    return m.Circle(radius=r).set_stroke(MUTED, 2.5).set_fill(CARD, 1).move_to([*at[:2], 0]).set_z_index(2)


def hbar(x0, y, w, color, fill=None, h=0.36):
    rect = m.Rectangle(width=max(w, 0.03), height=h).set_stroke(color, 1.5).set_fill(fill or color, 1)
    return rect.move_to([x0 + max(w, 0.03) / 2, y, 0])


def mac_stamp(lines):
    mac = icon(recolored("mac", MUTED, CARD, PREFIX), 0.55)
    words = m.VGroup(*[cite(s) for s in lines]).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
    return m.VGroup(mac, words).arrange(m.RIGHT, buff=0.25)


# ---------------------------------------------------------------------------- the episode


class Ep08NeverPrefillTwice(Episode):
    number = 8

    def lesson(self):
        self.hook()
        self.prefixes()
        self.keep_blocks()
        self.hashes()
        self.radix()
        self.on_this_mac()
        self.pricing()
        self.catch()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes=f"Running example: Ana pastes one chapter of her course notes ({CHAPTER:,} tokens with the chat template; the measured prompt of measurements.md §7) and asks question after question. Each request re-sends the chapter, so it is prefilled again: {T_COLD:.2f} s to the first token on this Mac (Owl-8B 4-bit, measured), {QUESTIONS} × = {SESSION_COLD:.0f} s ≈ {SESSION_COLD / 60:.1f} min. Teaser: {T_WARM:.3f} s with the chapter cached (measured). The questions are illustrative (the third is the measured one).")
        doc = icon(recolored("doc", REQUEST, PINK_FILL, PREFIX), 1.2)
        box = card(3.9, 3.5, REQUEST, PINK_FILL, radius=0.2)
        doc_words = m.VGroup(text("a chapter of her notes", 24, INK, weight=m.BOLD), code(f"{CHAPTER:,} tokens", 22, MUTED))
        doc_words.arrange(m.DOWN, buff=0.15)
        m.VGroup(doc, doc_words).arrange(m.DOWN, buff=0.35).move_to(box)
        chapter = m.VGroup(box, doc, doc_words).move_to([-4.2, 0.3, 0])
        asks = ["What is a ridge point?", "Why is decode memory-bound?", "How big is the KV cache per token?"]
        msgs = m.VGroup(*[message("user", q, width=5.4) for q in asks]).arrange(m.DOWN, buff=0.25).move_to([1.85, 0.55, 0])
        more = code("… question 50", 20, MUTED).next_to(msgs, m.DOWN, buff=0.3).align_to(msgs, m.LEFT)
        mascot = owl(0.6).move_to([5.75, -2.75, 0])

        ys = [1.45, 0.5, -0.45]
        rows = m.VGroup(*[
            strip([(6.2, f"chapter · {CHAPTER:,} tokens", *USER_LOOK), (1.15, f"Q{i + 1}", *USER_LOOK)], -4.9, y)
            for i, y in enumerate(ys)
        ])
        chapters = m.VGroup(*[r[0] for r in rows])
        qs = m.VGroup(*[r[1] for r in rows])
        times = m.VGroup(*[code(f"{T_COLD:.1f} s", 24, COMPUTE, weight=m.BOLD).next_to(r, m.RIGHT, buff=0.35) for r in rows])
        t_head = cite("time to first token · this Mac").next_to(times, m.UP, buff=0.3).align_to(times, m.LEFT)
        dots = code("… × 50", 22, MUTED).next_to(rows, m.DOWN, buff=0.3).align_to(rows, m.LEFT)
        total = m.VGroup(code(f"{QUESTIONS} × {T_COLD:.1f} s =", 26, INK), title_text(f"{SESSION_COLD / 60:.1f} min", 52, COMPUTE),
                         text("of prefill", 24, MUTED))
        total.arrange(m.RIGHT, buff=0.3).move_to([-0.6, -2.2, 0])
        on_baseline(total)
        warm = m.VGroup(*[code(f"{T_WARM:.2f} s", 24, GOOD, weight=m.BOLD).move_to(t, aligned_edge=m.LEFT) for t in times[1:]])
        tease = pill(f"{T_COLD:.2f} s → {T_WARM:.3f} s · measured on this Mac", GOOD, GOOD_FILL, 22, mono=True).move_to([-0.6, -3.15, 0])

        self.say(
            "[A student pastes a chapter of her notes into Owl], [and asks question after question about it.]",
            m.AnimationGroup(self.new_header("hook", "Fifty questions, one chapter"), m.FadeIn(chapter, shift=0.2 * m.RIGHT),
                             m.FadeIn(mascot, shift=0.2 * m.UP)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(x, shift=0.15 * m.UP) for x in msgs], lag_ratio=0.45), m.FadeIn(more)),
        )
        self.say(
            "[Each question goes out with the whole chapter in front of it], [so Owl prefills the same chapter every time.]",
            m.AnimationGroup(m.FadeTransform(chapter, chapters), m.FadeTransform(m.VGroup(msgs, more), qs), m.FadeIn(dots)),
            m.LaggedStart(*[c[0].animate.set_stroke(COMPUTE).set_fill(AMBER_FILL) for c in chapters], lag_ratio=0.35),
        )
        self.say(
            "[On this Mac, that's eight seconds before each answer even starts]: [almost seven minutes over fifty questions.]",
            m.AnimationGroup(m.FadeIn(t_head), m.LaggedStart(*[m.FadeIn(t, shift=0.15 * m.LEFT) for t in times], lag_ratio=0.3)),
            m.FadeIn(total, shift=0.15 * m.UP),
        )
        self.say(
            "[But the chapter never changed.] [Today: how servers compute a shared prefix once], [and never prefill it twice.]",
            m.LaggedStart(*[c[0].animate.set_stroke(KV).set_fill(VIOLET_FILL) for c in chapters], lag_ratio=0.25),
            m.AnimationGroup(*[m.Transform(t, w) for t, w in zip(times[1:], warm)]),
            m.FadeIn(tease, scale=0.9),
        )
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ shared prefixes

    def prefixes(self):
        self.next_section("1 · Shared prefixes", notes=f"What Owl receives: every request starts with Owl's system prompt ({SYSTEM:,} tokens = {SYSTEM // BLOCK} blocks, the notes' §4 worked example, illustrative); each of Ana's questions also repeats her {CHAPTER:,}-token chapter. Attention is causal, so a token's keys and values depend only on the tokens up to it: an identical prefix gives identical K and V (up to floating-point summation order, measurements §7/§12). On one H100 the 1,024-token system prompt costs prefill_time(1024) = {SYSTEM_MS:.1f} ms at 50% MFU (notes §4 worked example: 'about 33 ms of H100 time'), for every request that does not reuse it. Segment widths are not to scale; names illustrative.")
        ys = [1.5, 0.65, -0.2, -1.05]
        x0 = -3.9
        sys_part = (2.15, f"system · {SYSTEM:,}", *SYS_LOOK)
        rows = m.VGroup(
            strip([sys_part, (5.0, f"Ana's chapter · {CHAPTER:,}", *USER_LOOK), (1.2, "Q1", *USER_LOOK)], x0, ys[0], gap=0.3),
            strip([sys_part, (5.0, f"Ana's chapter · {CHAPTER:,}", *USER_LOOK), (1.2, "Q2", *USER_LOOK)], x0, ys[1], gap=0.3),
            strip([sys_part, (3.4, "Ben's notes", *USER_LOOK), (1.2, "Q1", *USER_LOOK)], x0, ys[2], gap=0.3),
            strip([sys_part, (2.1, "hi Owl!", *USER_LOOK)], x0, ys[3], gap=0.3),
        )
        names = m.VGroup(*[code(s, 20, MUTED).move_to([x0 - 0.3, y, 0], aligned_edge=m.RIGHT)
                           for s, y in zip(["Ana · Q1", "Ana · Q2", "Ben · Q1", "Cleo"], ys)])
        sys_segs = m.VGroup(*[r[0] for r in rows])
        ch_segs = m.VGroup(rows[0][1], rows[1][1])
        band_sys = m.SurroundingRectangle(sys_segs, buff=0.1, corner_radius=0.12).set_stroke(KV, 2.5)
        band_ch = m.SurroundingRectangle(ch_segs, buff=0.1, corner_radius=0.12).set_stroke(KV, 2.5)
        lab_sys = text("every request", 22, KV, weight=m.BOLD).next_to(band_sys, m.UP, buff=0.15)
        lab_ch = text("all fifty of Ana's questions", 22, KV, weight=m.BOLD).next_to(band_ch, m.UP, buff=0.15)
        unit = cite("tokens · widths not to scale").move_to([x0 - 0.3, -1.75, 0], aligned_edge=m.LEFT)
        rule = m.VGroup(code("K, V of token i  =  f(tokens 1 … i)", 24, INK, t2c={r"K, V": KV, r"f\(tokens 1 … i\)": REQUEST}),
                        text("same prefix in, same cache out", 24, KV, weight=m.BOLD)).arrange(m.RIGHT, buff=0.6)
        rule.move_to([0, -2.65, 0])
        why = cite("attention only looks back").next_to(rule, m.DOWN, buff=0.2)
        cost = pill(f"≈ {SYSTEM_MS:.0f} ms of H100 prefill, every request", COMPUTE, AMBER_FILL, 20).move_to([1.0, ys[3], 0], aligned_edge=m.LEFT)

        self.say(
            "[Look at what Owl actually receives.] [Every request starts with the same system prompt.]",
            m.AnimationGroup(self.new_header("01 · prefixes", "Everyone repeats the beginning"),
                             m.LaggedStart(*[m.FadeIn(m.VGroup(n, r), shift=0.15 * m.RIGHT) for n, r in zip(names, rows)], lag_ratio=0.2),
                             m.FadeIn(unit)),
            m.AnimationGroup(m.Create(band_sys), m.FadeIn(lab_sys, shift=0.1 * m.DOWN)),
        )
        self.say(
            "[On an H100, that prompt alone takes thirty-three milliseconds of prefill], [for every request, from every student.]",
            m.FadeIn(cost, shift=0.1 * m.LEFT),
            m.LaggedStart(*[m.Indicate(seg_, color=COMPUTE, scale_factor=1.06) for seg_ in sys_segs], lag_ratio=0.2),
        )
        self.say(
            "[And every one of her questions repeats the same chapter.]",
            m.AnimationGroup(m.Create(band_ch), m.FadeIn(lab_ch, shift=0.1 * m.DOWN)),
        )
        shared = [s[0] for s in sys_segs] + [s[0] for s in ch_segs]
        self.say(
            "[A token's keys and values depend only on the tokens before it], [so the same prefix always yields the same cache.]",
            m.AnimationGroup(m.FadeIn(rule[0], shift=0.1 * m.UP), m.FadeIn(why)),
            m.AnimationGroup(m.FadeIn(rule[1], shift=0.1 * m.UP),
                             m.LaggedStart(*[b.animate.set_stroke(KV).set_fill(VIOLET_FILL) for b in shared], lag_ratio=0.1)),
        )
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ keep the blocks

    def keep_blocks(self):
        self.next_section("2 · Keep the blocks", notes="Episode 7: the cache lives in 16-token blocks anywhere in memory, found through a per-request block table. Prefix caching (vLLM APC, notes §4) keeps a finished request's full blocks instead of freeing them (they are evicted LRU only when memory is needed); a new request whose first blocks match points its block table at the cached blocks (reference count +1) and prefills only the rest. vLLM V1 turned automatic prefix caching on by default, reporting 'near-zero performance degradation, even when the cache hit rate is 0%' (notes §4). Six blocks and their placements are schematic.")
        kinds = [SYS_LOOK, SYS_LOOK, USER_LOOK, USER_LOOK, USER_LOOK, USER_LOOK]

        def row(y, q):
            chips = m.VGroup(*[
                card(0.95, 0.55, c, f, radius=0.1, stroke=2) for c, f in kinds
            ]).arrange(m.RIGHT, buff=0.12).move_to([0.55, y, 0])
            word = code(q, 20, INK).move_to(chips[-1])
            return chips, word

        top, q1 = row(1.75, "Q1")
        bot, q2 = row(-1.75, "Q2")
        top_lab = m.VGroup(text("system prompt", 20, MUTED).next_to(top[:2], m.UP, buff=0.15),
                           text("chapter", 20, REQUEST).next_to(top[2:5], m.UP, buff=0.15))
        name1 = code("question 1", 20, INK).move_to([-3.1, 1.75, 0], aligned_edge=m.RIGHT)
        name2 = code("question 2", 20, INK).move_to([-3.1, -1.75, 0], aligned_edge=m.RIGHT)
        pool = cell_grid(3, 14, size=0.42, gap=0.08).move_to([0.55, 0.0, 0])
        pool_lab = m.VGroup(text("KV blocks", 22, KV, weight=m.BOLD), text("in GPU memory", 20, MUTED)).arrange(m.DOWN, buff=0.1)
        pool_lab.move_to([-4.35, 0.0, 0])
        where = [(2, 1), (0, 4), (2, 6), (1, 9), (0, 12), (2, 11)]
        cells = [pool[r][c] for r, c in where]
        new_cell = pool[1][12]
        lines1 = m.VGroup(*[m.Line(t.get_bottom(), c.get_top()).set_stroke(KV, 2, opacity=0.85) for t, c in zip(top, cells)])
        lines2 = m.VGroup(*[m.Line(b.get_top(), c.get_bottom()).set_stroke(GOOD, 2.5, opacity=0.9) for b, c in zip(bot[:5], cells[:5])])
        line_new = m.Line(bot[5].get_top(), new_cell.get_bottom()).set_stroke(KV, 2, opacity=0.85)
        kept = pill("cached", KV, VIOLET_FILL, 22).move_to([5.3, 0.0, 0])
        refs = cite("block table · Episode 7").move_to([4.05, -1.75, 0], aligned_edge=m.LEFT)
        hit = pill(f"✓ cache hit · 5 of 6 blocks reused, 1 prefilled", GOOD, GOOD_FILL, 22).move_to([0.55, -2.95, 0])
        default = m.VGroup(text("vLLM: on by default", 20, INK), cite("near-zero cost at 0% hits")).arrange(m.DOWN, buff=0.1)
        default.move_to([5.0, -1.08, 0])

        self.say(
            "[Episode seven stored the cache in blocks of sixteen tokens], [anywhere in memory.]",
            m.AnimationGroup(self.new_header("02 · prefix caching", "Keep the blocks"),
                             m.FadeIn(top, shift=0.1 * m.DOWN), m.FadeIn(q1), m.FadeIn(top_lab), m.FadeIn(name1),
                             m.FadeIn(pool), m.FadeIn(pool_lab)),
            m.LaggedStart(*[m.Succession(m.Create(ln, run_time=0.4), c.animate(run_time=0.3).set_fill(KV, 1).set_stroke(KV))
                            for ln, c in zip(lines1, cells)], lag_ratio=0.25),
        )
        self.say(
            "[When the first question is answered], [prefix caching keeps its blocks, instead of freeing them.]",
            m.AnimationGroup(m.FadeOut(lines1), m.VGroup(top, q1, top_lab, name1).animate.set_opacity(0.3)),
            m.FadeIn(kept, shift=0.1 * m.LEFT),
        )
        self.say(
            "[The next question starts the same way], [so its block table points at the cached blocks]:",
            m.AnimationGroup(m.FadeIn(bot, shift=0.1 * m.UP), m.FadeIn(q2), m.FadeIn(name2)),
            m.AnimationGroup(m.LaggedStart(*[m.Create(ln) for ln in lines2], lag_ratio=0.15), m.FadeIn(refs),
                             m.LaggedStart(*[c.animate.set_stroke(GOOD, 3) for c in cells[:5]], lag_ratio=0.15)),
        )
        self.say(
            "[only its own last block is prefilled.] [That's a cache hit.]",
            m.Succession(bot[5].animate(run_time=0.6).set_stroke(COMPUTE).set_fill(AMBER_FILL),
                         m.AnimationGroup(m.Create(line_new), new_cell.animate.set_fill(KV, 1).set_stroke(KV), run_time=0.6)),
            m.FadeIn(hit, scale=0.9),
        )
        self.say(
            "[In vLLM, prefix caching is on by default]: [even when nothing hits, it costs almost nothing.]",
            m.FadeIn(default[0], shift=0.1 * m.UP),
            m.FadeIn(default[1], shift=0.1 * m.UP),
        )
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ block hashes

    def hashes(self):
        self.next_section("3 · Block hashes", notes=f"vLLM's automatic prefix caching (design doc, read 2026-10-24; notes §4): 'we hash each kv-cache block by the tokens in the block and the tokens in the prefix before the block' — hash((parent hash, token ids, extra keys such as LoRA ids)); 'We only cache full blocks'. block_hashes() on screen is the function this scene runs (Python's hash; vLLM can use sha256). Six illustrative 16-token blocks (seeded random token ids): system ×2, chapter ×3, question. Question 2 shares the first five blocks: {cached_tokens(P2, TABLE)} tokens reused, 16 prefilled (computed). Changing token {CHANGED + 1} (in block 2) changes hashes 2–6: {cached_tokens(P3, TABLE)} tokens reused. Shown: last four hex digits. Keeping prompts stable (stable content first, dynamic content last) is 'Agents, Animated' Episode 7.")
        colors = [[MUTED] * 16, [MUTED] * 16, [REQUEST] * 16, [REQUEST] * 16, [REQUEST] * 16, [REQUEST] * 16]
        labels_a = ["system", "system", "chapter", "chapter", "chapter", "question 1"]
        labels_b = labels_a[:5] + ["question 2"]
        pitch, x0 = 2.05, -5.125
        xs = [x0 + i * pitch for i in range(6)]

        def blocks(labels, y):
            return m.VGroup(*[token_block(lab, c).move_to([x, y, 0]) for lab, c, x in zip(labels, colors, xs)])

        def pills(hs, y, color=KV, fill=VIOLET_FILL):
            return m.VGroup(*[hash_pill(h, color, fill).move_to([x, y, 0]) for h, x in zip(hs, xs)])

        def links(cards, ps, color=KV):
            down = m.VGroup(*[arrow(c.get_bottom(), p.get_top(), MUTED, buff=0.05) for c, p in zip(cards, ps)])
            chain = m.VGroup(*[arrow(a.get_right(), b.get_left(), color, buff=0.08) for a, b in zip(ps, ps[1:])])
            return down, chain

        row_a = blocks(labels_a, 1.6)
        pills_a = pills(H1, 0.55)
        down_a, chain_a = links(row_a, pills_a)
        block = code_block(HASH_SRC, 20).move_to([-6.3, -0.3, 0], aligned_edge=m.UL)
        note = m.VGroup(text("each hash covers", 22, INK), text("its block's 16 tokens", 22, INK), text("+ its parent's hash", 22, INK),
                        cite("after vLLM's prefix caching")).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
        note[3].shift(0.08 * m.DOWN)
        note.next_to(block, m.RIGHT, buff=0.4).align_to(block, m.UP).shift(0.1 * m.DOWN)
        whole = m.VGroup(text("so one hash names", 22, KV, weight=m.BOLD), text("the whole prefix", 22, KV, weight=m.BOLD))
        whole.arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT).next_to(note, m.DOWN, buff=0.35).align_to(note, m.LEFT)
        notes = m.VGroup(block, note, whole)

        row_b = blocks(labels_b, -0.95)
        pills_b = pills(H2, -2.0)
        down_b, chain_b = links(row_b, pills_b)
        marks = m.VGroup(*[mark(i < 5).move_to([x, -2.62, 0]) for i, x in enumerate(xs)])
        result = code(f"{cached_tokens(P2, TABLE)} tokens reused · {len(P2) - cached_tokens(P2, TABLE)} prefilled", 22, INK,
                      t2c={r"\d+ tokens reused": GOOD, r"\d+ prefilled": COMPUTE}).move_to([0, -3.25, 0])
        changed_cell = row_b[1][2][CHANGED - BLOCK]
        pills_c = m.VGroup(*[hash_pill(h, COMPUTE, AMBER_FILL).move_to(p) for h, p in zip(H3[1:], pills_b[1:])])
        marks_c = m.VGroup(*[mark(False).move_to(mk) for mk in marks[1:5]])
        result_c = code(f"{cached_tokens(P3, TABLE)} tokens reused · {len(P3) - cached_tokens(P3, TABLE)} prefilled", 22, INK,
                        t2c={r"\d+ tokens reused": GOOD, r"\d+ prefilled": COMPUTE}).move_to(result)
        link = m.VGroup(cite("the agent side:"), pill("Agents, Animated · Ep 7", KV, VIOLET_FILL, 20)).arrange(m.RIGHT, buff=0.2)

        self.say(
            "[How does the server know a block is already cached?] [It names every full block with a hash],",
            m.AnimationGroup(self.new_header("03 · block hashes", "A name for every prefix"),
                             m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.DOWN) for c in row_a], lag_ratio=0.12)),
            m.AnimationGroup(m.GrowArrow(down_a[0]), m.FadeIn(pills_a[0], shift=0.1 * m.DOWN)),
        )
        self.say(
            "[computed from the block's sixteen tokens] [and the hash of the block before it.]",
            m.AnimationGroup(m.FadeIn(block, shift=0.1 * m.UP), m.FadeIn(note[:3])),
            m.LaggedStart(*[m.AnimationGroup(m.GrowArrow(c), m.GrowArrow(d), m.FadeIn(p, shift=0.1 * m.DOWN))
                            for c, d, p in zip(chain_a, down_a[1:], pills_a[1:])], lag_ratio=0.35),
        )
        self.say(
            "[So each hash stands for the whole prefix up to that block], [not just its own sixteen tokens.]",
            m.Succession(*[pulse(a, KV) for a in chain_a], m.Circumscribe(pills_a[-1], color=KV, buff=0.08)),
            m.AnimationGroup(m.FadeIn(note[3]), m.FadeIn(whole, shift=0.1 * m.UP)),
        )
        self.say(
            "[A new prompt is hashed the same way and looked up], [block by block: hit, hit, hit, hit, hit],",
            m.AnimationGroup(m.FadeOut(notes), m.FadeIn(row_b, shift=0.1 * m.UP), m.FadeIn(down_b), m.FadeIn(pills_b), m.FadeIn(chain_b)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(mk, shift=0.1 * m.UP) for mk in marks[:5]], lag_ratio=0.3),
                             m.LaggedStart(*[p[0].animate.set_stroke(GOOD) for p in pills_b[:5]], lag_ratio=0.3)),
        )
        self.say(
            "[until the first miss.] [From there on, the tokens are prefilled.]",
            m.AnimationGroup(m.FadeIn(marks[5], shift=0.1 * m.UP), pills_b[5][0].animate.set_stroke(COMPUTE)),
            m.AnimationGroup(row_b[5][0].animate.set_stroke(COMPUTE), row_b[5][2].animate.set_fill(COMPUTE), m.FadeIn(result)),
        )
        self.say(
            "[Change one token near the start], [and every hash after it changes]: [almost nothing is reused.]",
            changed_cell.animate.set_fill(RED).scale(1.5),
            m.AnimationGroup(m.LaggedStart(*[m.Transform(p, q) for p, q in zip(pills_b[1:], pills_c)], lag_ratio=0.15),
                             m.LaggedStart(*[m.Transform(a, b) for a, b in zip(marks[1:5], marks_c)], lag_ratio=0.15),
                             *[a.animate.set_color(COMPUTE) for a in chain_b]),
            m.AnimationGroup(m.Transform(result, result_c),
                             *[c[0].animate.set_stroke(COMPUTE) for c in row_b[1:5]]),
        )
        link.move_to([3.6, -3.25, 0])
        self.say(
            "[Keeping prompts cache-friendly is the agent side of this story]: [Agents, Animated, episode seven.]",
            m.AnimationGroup(result.animate.move_to([-2.9, -3.25, 0]), m.FadeIn(link[0])),
            m.FadeIn(link[1], scale=0.9),
        )
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ the radix tree

    def radix(self):
        self.next_section("4 · SGLang's radix tree", notes="SGLang's RadixAttention (Zheng et al., NeurIPS 2024; LMSYS blog, Jan 2024): the KV cache of all requests, prompts and answers, is kept in a radix tree; 'Each tree edge carries a label denoting a substring or a sequence of tokens'; a new request is matched against the tree (longest prefix) and only the remainder is computed and inserted; 'an LRU eviction policy that recursively evicts leaf nodes'; cache-aware scheduling: 'we sort the requests by matched prefix length and prioritize requests with longer matched prefixes'; 'the cache hit rate ranges from 50% to 99%' in the paper's benchmarks. The branches, idle times and queue percentages are illustrative.")
        R, S = (-5.9, 0.75), (-2.7, 0.75)
        A, B, C = (0.6, 1.9), (0.6, 0.75), (0.6, -0.4)
        A1, A2, A3 = (3.7, 2.35), (3.7, 1.75), (3.7, 1.15)
        B1 = (3.7, 0.45)
        e_rs, e_sa, e_sb, e_sc = elbow(R, S), elbow(S, A), elbow(S, B), elbow(S, C)
        e_a1, e_a2, e_b1 = elbow(A, A1), elbow(A, A2), elbow(B, B1)
        e_a3 = elbow(A, A3, color=COMPUTE)

        def label(edge_end, s, parent_x, color=INK):
            x1 = parent_x + 0.35
            return text(s, 20, color).move_to([(x1 + edge_end[0]) / 2, edge_end[1] + 0.25, 0])

        l_rs = label(S, "system prompt", R[0])
        l_sa, l_sb, l_sc = label(A, "Ana's chapter", S[0]), label(B, "Ben's notes", S[0]), label(C, "hi Owl!", S[0])
        l_a1, l_a2, l_b1 = label(A1, "Q1", A[0]), label(A2, "Q2", A[0]), label(B1, "Q1", B[0])
        l_a3 = label(A3, "Q3", A[0], COMPUTE)
        nodes = {k: tree_node(p) for k, p in dict(R=R, S=S, A=A, B=B, C=C, A1=A1, A2=A2, B1=B1).items()}
        n_a3 = tree_node(A3)
        root_lab = code("root", 20, MUTED).next_to(nodes["R"], m.DOWN, buff=0.18)
        idle = {k: code(s, 20, MUTED).next_to(nodes[k], m.RIGHT, buff=0.25)
                for k, s in dict(A1="used 5 s ago", A2="used 1 min ago", B1="used 9 min ago", C="used 2 min ago").items()}
        trunk = m.VGroup(nodes["R"], root_lab, e_rs, l_rs, nodes["S"])
        mid = m.VGroup(e_sa, e_sb, e_sc, l_sa, l_sb, l_sc, nodes["A"], nodes["B"], nodes["C"])
        leaves = m.VGroup(e_a1, e_a2, e_b1, l_a1, l_a2, l_b1, nodes["A1"], nodes["A2"], nodes["B1"], *idle.values())

        req = strip([(S[0] - R[0] - 0.06, "system prompt", *USER_LOOK), (A[0] - S[0] - 0.06, "Ana's chapter", *USER_LOOK),
                     (A3[0] - A[0], "Q3", *USER_LOOK)], R[0], -1.6)
        req_lab = code("her next question", 20, REQUEST).next_to(req, m.DOWN, buff=0.15).align_to(req, m.LEFT)
        flash = m.VGroup(e_rs.copy(), e_sa.copy(), e_a1.copy())

        meter_lab = text("KV memory", 20, MUTED).move_to([-5.9, -1.95, 0], aligned_edge=m.LEFT)
        meter = Meter(width=2.4, fraction=0.55, color=KV).move_to([-4.7, -2.55, 0])
        b_branch = m.VGroup(e_b1, l_b1, nodes["B1"], idle["B1"])
        b_node = m.VGroup(e_sb, l_sb, nodes["B"])

        fracs = [(0.13, "Dan"), (0.99, "Ana"), (0.6, "Ben")]
        chips = m.VGroup()
        for f, who in fracs:
            box = card(2.15, 0.72, REQUEST, CARD, radius=0.1, stroke=2)
            words = code(f"{who} · {f:.0%}", 20, INK).move_to(box).shift(0.09 * m.UP)
            track = m.Rectangle(width=1.75, height=0.07).set_stroke(width=0).set_fill(EDGE, 1)
            track.move_to(box).shift(0.2 * m.DOWN)
            fill = m.Rectangle(width=1.75 * f, height=0.07).set_stroke(width=0).set_fill(GOOD, 1)
            fill.align_to(track, m.LEFT).set_y(track.get_y())
            chips.add(m.VGroup(box, track, fill, words))
        slots = [np.array([x, -2.55, 0]) for x in (-1.75, 0.6, 2.95)]
        for c, p in zip(chips, slots):
            c.move_to(p)
        order = sorted(range(3), key=lambda i: -fracs[i][0])
        q_lab = text("waiting · longest match first", 20, MUTED).move_to([slots[0][0] - 1.075, -1.95, 0], aligned_edge=m.LEFT)
        rates = cite("hit rates 50–99% in the paper's benchmarks · Zheng et al., 2024").move_to([0.6, -3.25, 0])

        self.say(
            "[SGLang organises its whole cache as a radix tree.]",
            m.AnimationGroup(self.new_header("04 · radix tree", "SGLang's tree of prefixes"),
                             m.FadeIn(nodes["R"]), m.FadeIn(root_lab), m.Create(e_rs), m.FadeIn(nodes["S"]), m.FadeIn(l_rs)),
        )
        self.say(
            "[Each edge holds a run of tokens], [and each path from the root is a prefix it has seen.]",
            m.LaggedStart(*[m.AnimationGroup(m.Create(e), m.FadeIn(lab), m.FadeIn(n))
                            for e, lab, n in ((e_sa, l_sa, nodes["A"]), (e_sb, l_sb, nodes["B"]), (e_sc, l_sc, nodes["C"]))], lag_ratio=0.3),
            m.AnimationGroup(
                m.LaggedStart(*[m.AnimationGroup(m.Create(e), m.FadeIn(lab), m.FadeIn(n))
                                for e, lab, n in ((e_a1, l_a1, nodes["A1"]), (e_a2, l_a2, nodes["A2"]), (e_b1, l_b1, nodes["B1"]))], lag_ratio=0.3),
                m.FadeIn(m.VGroup(*idle.values())),
                m.Succession(m.Wait(1.0), *[m.ShowPassingFlash(f.set_stroke(INK, 7), time_width=0.8, run_time=0.6) for f in flash])),
        )
        self.say(
            "[Her next question walks down the tree], [lighting up the longest match],",
            m.AnimationGroup(m.FadeIn(req, shift=0.15 * m.UP), m.FadeIn(req_lab)),
            m.AnimationGroup(
                m.Succession(e_rs.animate(run_time=0.5).set_stroke(GOOD, 6), e_sa.animate(run_time=0.5).set_stroke(GOOD, 6)),
                m.LaggedStart(*[n.animate.set_stroke(GOOD, 3.5) for n in (nodes["R"], nodes["S"], nodes["A"])], lag_ratio=0.4),
                m.LaggedStart(req[0][0].animate.set_stroke(GOOD, 3), req[1][0].animate.set_stroke(GOOD, 3), lag_ratio=0.5)),
        )
        self.say(
            "[then only her new tokens are prefilled], [and hung on as a new branch.]",
            req[2][0].animate.set_stroke(COMPUTE).set_fill(AMBER_FILL),
            m.AnimationGroup(m.Create(e_a3), m.FadeIn(n_a3), m.FadeIn(l_a3)),
        )
        self.say(
            "[When memory runs short], [the least recently used leaves are evicted, one after another],",
            m.AnimationGroup(m.FadeOut(req), m.FadeOut(req_lab), m.FadeIn(meter_lab), m.FadeIn(meter),
                             e_a3.animate.set_stroke(KV), l_a3.animate.set_color(INK),
                             m.Succession(m.Wait(0.5), meter.fill_to(0.95, run_time=1.2))),
            m.Succession(m.FadeOut(b_branch, run_time=0.8), m.FadeOut(b_node, run_time=0.8), meter.fill_to(0.72, run_time=0.6)),
        )
        self.say(
            "[and the scheduler runs requests with longer matches first.] [In the paper's benchmarks, hit rates ran from fifty to ninety-nine percent.]",
            m.Succession(m.AnimationGroup(m.FadeIn(chips, shift=0.1 * m.UP), m.FadeIn(q_lab), run_time=0.8),
                         m.AnimationGroup(*[chips[i].animate.move_to(slots[k]) for k, i in enumerate(order)], run_time=1.2)),
            m.FadeIn(rates, shift=0.1 * m.UP),
        )
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ measured on this Mac

    def on_this_mac(self):
        self.next_section("5 · Measured on this Mac", notes=f"research/measurements.md §7 (2026-10-24, M4 Max, mlx-lm 0.32.2): Llama-3.1-8B-Instruct 4-bit; the chapter prefix ({CHAPTER:,} tokens incl. the chat template) cached as mlx_lm.cache_prompt does, then the {QUESTION}-token question: TTFT {T_COLD:.3f} s without the cache vs {T_WARM:.3f} s with it (median of 5; {SPEEDUP:.0f}×; {PC['ttft_with_cache_incl_disk_load_s']:.3f} s including loading the file, from the OS file cache). Same greedy answer (48 tokens). Cache file {FILE_MB:.0f} MB = {CHAPTER:,} × 131,072 B = {CHAPTER_KV / 1e6:.1f} MB (BF16 cache, 128 KiB per token: Episode 5). Fifty questions: {QUESTIONS} × {T_COLD:.2f} = {SESSION_COLD:.0f} s vs {T_COLD:.2f} + 49 × {T_WARM:.3f} = {SESSION_WARM:.1f} s (derived). Stock CLI cross-check: 8.03 s → 0.145 s (55×).")
        term = terminal([
            '$ mlx_lm.cache_prompt --prompt "$(cat chapter.txt)" \\',
            "    --prompt-cache-file chapter.safetensors",
            '$ mlx_lm.generate --prompt "$(cat question.txt)" \\',
            "    --prompt-cache-file chapter.safetensors",
        ], width=8.6, title="this Mac · Llama-3.1-8B, 4-bit")
        for i in (1, 3):
            term[4][i].set_color(INK).shift(0.65 * m.RIGHT)
        term.move_to([-6.4, 2.55, 0], aligned_edge=m.UL)
        stamp = mac_stamp(["measured on this Mac", "M4 Max · mlx-lm 0.32.2"])
        stamp.arrange(m.DOWN, buff=0.2).move_to([5.0, 1.5, 0])

        x0, full = -3.0, 5.6
        ys = (-0.95, -1.65)
        head = cite(f"time to first token (TTFT) · {CHAPTER:,} + {QUESTION} tokens").move_to([x0, -0.4, 0], aligned_edge=m.LEFT)
        names = m.VGroup(code("no cache", 22, MUTED).move_to([x0 - 0.3, ys[0], 0], aligned_edge=m.RIGHT),
                         code("chapter cached", 22, MUTED).move_to([x0 - 0.3, ys[1], 0], aligned_edge=m.RIGHT))
        bar_cold = hbar(x0, ys[0], full, COMPUTE, AMBER_FILL)
        bar_warm = hbar(x0, ys[1], full * T_WARM / T_COLD, COMPUTE, AMBER_FILL)
        v_cold = code(f"{T_COLD:.2f} s", 24, COMPUTE, weight=m.BOLD).next_to(bar_cold, m.RIGHT, buff=0.2)
        v_warm = code(f"{T_WARM:.3f} s", 24, GOOD, weight=m.BOLD).next_to(bar_warm, m.RIGHT, buff=0.2)
        fast = m.VGroup(title_text(f"{SPEEDUP:.0f}×", 72, GOOD), text("sooner", 22, MUTED)).arrange(m.DOWN, buff=0.1)
        fast.move_to([5.55, -1.3, 0])

        same = m.VGroup(pill("✓ same answer, word for word", GOOD, GOOD_FILL, 22),
                        text(f"“{ANSWER}”", 20, MUTED)).arrange(m.DOWN, buff=0.2, aligned_edge=m.LEFT)
        same.move_to([-6.4, -2.75, 0], aligned_edge=m.LEFT)
        size = m.VGroup(code(f"cache file  {FILE_MB:.0f} MB", 22, KV, weight=m.BOLD),
                        code(f"= {CHAPTER:,} × 128 KiB ✓", 22, INK, t2c={"✓": GOOD})).arrange(m.DOWN, buff=0.18, aligned_edge=m.LEFT)
        size.move_to([6.4, -2.75, 0], aligned_edge=m.RIGHT)

        s_head = cite(f"prefill over a {QUESTIONS}-question session").move_to(head, aligned_edge=m.LEFT)
        s_cold = hbar(x0, ys[0], full, COMPUTE, AMBER_FILL)
        s_warm = hbar(x0, ys[1], full * SESSION_WARM / SESSION_COLD, COMPUTE, AMBER_FILL)
        sv_cold = code(f"{SESSION_COLD:.0f} s", 24, COMPUTE, weight=m.BOLD).next_to(s_cold, m.RIGHT, buff=0.2)
        sv_warm = code(f"{SESSION_WARM:.1f} s", 24, GOOD, weight=m.BOLD).next_to(s_warm, m.RIGHT, buff=0.2)
        s_fast = m.VGroup(title_text(f"{SESSION_COLD / SESSION_WARM:.0f}×", 72, GOOD), text("less waiting", 22, MUTED)).arrange(m.DOWN, buff=0.1)
        s_fast.move_to(fast)

        self.say(
            "[We measured it on this Mac]: [Owl's eight-billion-parameter model at four bits, with the prompt cache in MLX.]",
            m.AnimationGroup(self.new_header("05 · measured", "Eight seconds, or a seventh of one"), m.FadeIn(stamp)),
            m.FadeIn(term, shift=0.15 * m.UP),
        )
        self.say(
            "[Without a cache], [the chapter and the question take eight seconds to the first token.]",
            m.AnimationGroup(m.FadeIn(head), m.FadeIn(names[0])),
            m.AnimationGroup(m.GrowFromEdge(bar_cold, m.LEFT, run_time=2.0), m.Succession(m.Wait(1.6), m.FadeIn(v_cold))),
        )
        self.say(
            "[With the chapter's cache loaded], [the first token comes in under a seventh of a second]: [fifty-eight times sooner.]",
            m.FadeIn(names[1]),
            m.AnimationGroup(m.GrowFromEdge(bar_warm, m.LEFT), m.FadeIn(v_warm, shift=0.1 * m.LEFT)),
            m.FadeIn(fast, scale=0.85),
        )
        self.say(
            "[Same answer, word for word], [and the saved cache is nine hundred and six megabytes]:",
            m.FadeIn(same, shift=0.1 * m.UP),
            m.FadeIn(size[0], shift=0.1 * m.UP),
        )
        self.say(
            "[six thousand nine hundred and ten tokens, at a hundred and twenty-eight kibibytes each.]",
            m.FadeIn(size[1], shift=0.1 * m.UP),
        )
        self.say(
            "[Over fifty questions], [almost seven minutes of prefill shrink to about fifteen seconds.]",
            m.AnimationGroup(m.Transform(head, s_head), m.Transform(bar_cold, s_cold), m.Transform(v_cold, sv_cold)),
            m.AnimationGroup(m.Transform(bar_warm, s_warm), m.Transform(v_warm, sv_warm), m.Transform(fast, s_fast)),
        )
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ pricing

    def pricing(self):
        self.next_section("6 · Why cached input is cheap", notes=f"Server side: a hit skips the chapter's prefill, 2·P·n = 2 × 8.03e9 × {CHAPTER:,} = {SKIPPED / 1e12:.0f} TFLOP for Owl-8B (attention adds more; notes §11), and costs memory instead: {CHAPTER_KV / 1e6:.0f} MB of BF16 keys and values. Provider prices [DATED October 2026; notes §0 #14 and §11, official docs read 2026-10-23]: Anthropic cache reads 0.1× base input (0.05× on Opus 5.5 and Sonnet 5.5), 5-minute cache writes 1.25× (1-hour writes 2×), 'The cache is refreshed for no additional cost each time the cached content is used'; OpenAI: 'Prompt caching is enabled by default', cached input 'discounted up to 95%' (per model: 50–95%). Session bill in units of one full-price chapter (chapter tokens only, every question within the 5-minute window): {QUESTIONS} × 1 = {BILL_COLD:.0f}; 1.25 + 49 × 0.1 = {BILL_WARM:.2f}; at 0.05×: {BILL_NEWER:.2f} (derived).")

        def tile_(x, color, fill, kicker, number, unit):
            box = card(6.1, 1.4, color, fill, radius=0.16, stroke=2.5).move_to([x, 1.8, 0])
            k = text(kicker, 22, MUTED).align_to(box, m.UL).shift(0.35 * m.RIGHT + 0.2 * m.DOWN)
            n = title_text(number, 48, color)
            u = text(unit, 22, INK)
            nu = m.VGroup(n, u).arrange(m.RIGHT, buff=0.3)
            u.align_to(n, m.DOWN)
            nu.align_to(box, m.LEFT).shift(0.35 * m.RIGHT).set_y(box.get_y() - 0.2)
            return m.VGroup(box, k, nu)

        skip = tile_(-3.3, COMPUTE, AMBER_FILL, "a cache hit skips", f"{SKIPPED / 1e12:.0f} TFLOP", "prefill per question")
        keep = tile_(3.3, KV, VIOLET_FILL, "and only costs memory", f"{CHAPTER_KV / 1e6:.0f} MB", "of blocks, kept around")

        def price(label, color, fill):
            return pill(label, color, fill, 22, mono=True)

        a_name = text("Anthropic", 26, INK, weight=m.BOLD).move_to([-6.3, 0.45, 0], aligned_edge=m.LEFT)
        a_read = price(f"cache read {READ:g}×", GOOD, GOOD_FILL).move_to([-3.6, 0.45, 0], aligned_edge=m.LEFT)
        a_new = code(f"{READ_NEWER:g}× on Opus & Sonnet 5.5", 20, MUTED).next_to(a_read, m.RIGHT, buff=0.3)
        a_write = price(f"cache write {WRITE:g}×", COMPUTE, AMBER_FILL).move_to([-3.6, -0.15, 0], aligned_edge=m.LEFT)
        a_ttl = text("lives 5 minutes, refreshed by every hit", 22, MUTED).next_to(a_write, m.RIGHT, buff=0.3)
        o_name = text("OpenAI", 26, INK, weight=m.BOLD).move_to([-6.3, -0.8, 0], aligned_edge=m.LEFT)
        o_auto = price("automatic", MUTED, CARD).move_to([-3.6, -0.8, 0], aligned_edge=m.LEFT)
        o_off = price(f"cached input up to {OPENAI_OFF:.0%} off", GOOD, GOOD_FILL).next_to(o_auto, m.RIGHT, buff=0.3)
        unit_note = cite("official pricing docs, October 2026").move_to([-6.3, -1.3, 0], aligned_edge=m.LEFT)

        x0, full = -2.3, 7.4
        ys = (-2.25, -2.8, -3.3)
        b_names = m.VGroup(*[code(s, 20, MUTED).move_to([x0 - 0.3, y, 0], aligned_edge=m.RIGHT)
                             for s, y in zip(["50 × full price", "write, then 49 × 0.1", "… at 0.05"], ys)])
        bills = [BILL_COLD, BILL_WARM, BILL_NEWER]
        b_bars = m.VGroup(*[hbar(x0, y, full * b / BILL_COLD, c, f, h=0.32)
                            for y, b, (c, f) in zip(ys, bills, [(MUTED, EDGE), (GOOD, GOOD_FILL), (GOOD, GOOD_FILL)])])
        b_vals = m.VGroup(*[code(f"{b:g}" if b == int(b) else f"{b:.2f}", 22, INK if i == 0 else GOOD, weight=m.BOLD).next_to(bar, m.RIGHT, buff=0.2)
                            for i, (b, bar) in enumerate(zip(bills, b_bars))])
        b_head = cite("Ana's session, in full-price chapters").move_to([x0, -1.85, 0], aligned_edge=m.LEFT)

        self.say(
            "[A hit skips over a hundred trillion operations of prefill], [and costs only memory, to keep the blocks around.]",
            m.AnimationGroup(self.new_header("06 · pricing", "Why cached input is cheap"), m.FadeIn(skip, shift=0.15 * m.UP)),
            m.FadeIn(keep, shift=0.15 * m.UP),
        )
        self.say(
            "[So providers price cached input low]: [Anthropic bills cache reads at a tenth of the input price], [a twentieth on some newer models],",
            m.AnimationGroup(m.FadeIn(a_name), m.FadeIn(unit_note)),
            m.FadeIn(a_read, shift=0.1 * m.RIGHT),
            m.FadeIn(a_new, shift=0.1 * m.RIGHT),
        )
        self.say(
            "[charges a quarter extra to write the cache], [and keeps it five minutes, refreshed by every hit.]",
            m.FadeIn(a_write, shift=0.1 * m.RIGHT),
            m.FadeIn(a_ttl, shift=0.1 * m.RIGHT),
        )
        self.say(
            "[OpenAI caches automatically], [with cached input up to ninety-five percent off.]",
            m.AnimationGroup(m.FadeIn(o_name), m.FadeIn(o_auto, shift=0.1 * m.RIGHT)),
            m.FadeIn(o_off, shift=0.1 * m.RIGHT),
        )
        self.say(
            "[For her fifty questions, at a tenth], [fifty full-price chapters become about six.]",
            m.AnimationGroup(m.FadeIn(b_head), m.FadeIn(b_names[0]), m.GrowFromEdge(b_bars[0], m.LEFT), m.FadeIn(b_vals[0])),
            m.AnimationGroup(m.FadeIn(b_names[1:]), m.LaggedStart(m.GrowFromEdge(b_bars[1], m.LEFT), m.GrowFromEdge(b_bars[2], m.LEFT), lag_ratio=0.4),
                             m.FadeIn(b_vals[1:])),
        )
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ the catch

    def catch(self):
        self.next_section("7 · The catch: memory", notes=f"One H100 SXM (80 GB, NVIDIA spec, dated) serving Owl-8B in BF16 (Episode 5's budget): 90% of 80 GB minus {weight_bytes() / 1e9:.2f} GB of weights = {KV_ROOM / 1e9:.1f} GB for keys and values; one cached chapter = {CHAPTER_KV / 1e6:.0f} MB, so {CHAPTERS_FIT} chapters fit (with no running requests at all). So engines evict cold prefixes (vLLM: LRU over free blocks; SGLang: LRU leaves), and hosted caches expire (Anthropic: 5 minutes by default, refreshed on use; a 1-hour option costs 2× to write). The evicted slices are schematic.")
        x0, width, y = -5.8, 11.6, 0.75
        per = width / GPU["memory"]
        xb = lambda nbytes: x0 + per * nbytes
        track = m.Rectangle(width=width, height=0.72).set_stroke(EDGE, 2).set_fill(CARD, 1).move_to([x0 + width / 2, y, 0])

        def seg_(a, b, color, h=0.56, opacity=1.0, inset=0.0):
            w = per * (b - a) - inset
            return m.Rectangle(width=w, height=h).set_stroke(width=0).set_fill(color, opacity).move_to([(xb(a) + xb(b)) / 2, y, 0])

        gpu = icon(recolored("gpu", MUTED, CARD, PREFIX), 0.6)
        title = m.VGroup(gpu, code("H100 · 80 GB · Owl-8B in BF16", 22, INK)).arrange(m.RIGHT, buff=0.25).move_to([x0, 1.85, 0], aligned_edge=m.LEFT)
        w_seg = seg_(0, weight_bytes(), WEIGHTS)
        free = seg_(0.9 * GPU["memory"], GPU["memory"], FAINT, opacity=0.6)
        w_lab = code(f"weights {weight_bytes() / 1e9:.0f} GB", 20, WEIGHTS).move_to([(x0 + xb(weight_bytes())) / 2, y - 0.68, 0])
        f_lab = code("10% free", 20, MUTED).move_to([(xb(0.9 * GPU["memory"]) + xb(GPU["memory"])) / 2, y - 0.68, 0])
        slices = m.VGroup(*[seg_(weight_bytes() + i * CHAPTER_KV, weight_bytes() + (i + 1) * CHAPTER_KV, KV, h=0.5, inset=0.025)
                            for i in range(CHAPTERS_FIT)])
        c_lab = code(f"{CHAPTERS_FIT} cached chapters · {CHAPTER_KV / 1e6:.0f} MB each", 20, KV).move_to(
            [(xb(weight_bytes()) + xb(0.9 * GPU["memory"])) / 2, y - 0.68, 0])
        n_cold = 6
        cold = slices[:n_cold]
        clock = icon(recolored("clock", MUTED, CARD, PREFIX), 0.7)
        lines = m.VGroup(
            text("cold prefixes are evicted", 26, INK, weight=m.BOLD),
            code("engines: least recently used first (vLLM, SGLang)", 20, MUTED),
            code("hosted APIs: after 5 minutes unused (Anthropic default)", 20, MUTED),
        ).arrange(m.DOWN, buff=0.18, aligned_edge=m.LEFT)
        evict = m.VGroup(clock, lines).arrange(m.RIGHT, buff=0.4).move_to([0, -1.75, 0])

        self.say(
            "[The catch is memory.] [Next to Owl's weights, a single H100 has room for only about sixty cached chapters],",
            m.AnimationGroup(self.new_header("07 · the catch", "Memory is the price"), m.FadeIn(title), m.FadeIn(track),
                             m.GrowFromEdge(w_seg, m.LEFT), m.FadeIn(w_lab), m.FadeIn(free), m.FadeIn(f_lab)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(s, shift=0.05 * m.RIGHT) for s in slices], lag_ratio=0.05), m.FadeIn(c_lab)),
        )
        self.say(
            "[so cold prefixes are evicted]: [the least recently used go first], [and hosted caches expire after minutes.]",
            m.AnimationGroup(m.FadeIn(clock), m.FadeIn(lines[0], shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(lines[1], shift=0.1 * m.UP), cold.animate.set_fill(FAINT, 0.8)),
            m.AnimationGroup(m.FadeIn(lines[2], shift=0.1 * m.UP), m.FadeOut(cold)),
        )
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes=f"Keep (blocks outlive their request), match (a hash chain over 16-token blocks, or SGLang's radix tree), skip (prefill only what's new). Measured on this Mac: TTFT {T_COLD:.2f} s → {T_WARM:.3f} s. Next: Episode 9, the cost of a long memory.")
        words = [("Keep", KV, "blocks outlive the request"),
                 ("Match", TOKEN, "hash chain or radix tree"),
                 ("Skip", COMPUTE, "prefill only what's new")]
        cols = m.VGroup()
        for w, color, sub in words:
            cols.add(m.VGroup(title_text(w, 72, color), text(sub, 22, MUTED)).arrange(m.DOWN, buff=0.25))
        cols.arrange(m.RIGHT, buff=1.3).move_to([0, 0.75, 0])
        base = 0.85
        for c in cols:
            c[0].shift((base - c[0][0].get_bottom()[1]) * m.UP)  # first letters (K, M, S) sit on the baseline
            c[1].move_to([c[0].get_x(), base - 0.55, 0])
        mid = base + 0.3
        arrows = m.VGroup(*[
            arrow([a[0].get_right()[0] + 0.15, mid, 0], [b[0].get_left()[0] - 0.15, mid, 0], MUTED) for a, b in zip(cols, cols[1:])
        ])
        tagline = text("A shared prefix is computed once.", 32, INK).move_to([0, -1.35, 0])
        mascot = owl(0.75).move_to([0, -2.7, 0])

        self.say(
            "[So: keep], [match], [and skip.]",
            m.AnimationGroup(self.new_header("recap", "Never prefill twice"), m.FadeIn(cols[0], shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowArrow(arrows[0]), m.FadeIn(cols[1], shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowArrow(arrows[1]), m.FadeIn(cols[2], shift=0.2 * m.UP)),
        )
        self.say(
            "[Keep the blocks, match the prefix, skip the prefill]: [and a shared prefix is only ever computed once.]",
            m.Circumscribe(cols, color=MUTED, buff=0.2),
            m.AnimationGroup(m.FadeIn(tagline, shift=0.1 * m.UP), m.FadeIn(mascot, shift=0.2 * m.UP)),
        )
        self.wait(0.4)
        self.clear()
