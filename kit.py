"""The series kit for "Tokens per Second" (how LLM inference really works): design system, inference
components, voice, and the Episode base class with the series intro and next-episode outro. Episodes
import it:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from kit import *  # noqa: F403
    from owl_reference import *  # noqa: F403  (the course's numbers and code)

    class Ep02TheTwoSpeedLimits(Episode):
        number = 2
        def lesson(self): ...

Run from an episode folder: `VOICE=draft manimgx inspect scene.py` (silent draft) or, with the narrator
serving from the series folder, `manimgx render scene.py`. A tour of every component: `kit_demo.py`.
"""

import base64
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

import manimgx as m
from manimgx.audio import Word

SERIES = Path(__file__).resolve().parent
ASSETS = SERIES / "assets"
TITLE_WIDTH = 12.5  # the widest an intro/outro card title may be (the frame is about 14.2 wide)

# ---------------------------------------------------------------------------- engine fixes
# (carried over from "AI, Explained" and "Agents, Animated": each fixes a ManimGX timing fault, see the
# tutorial-video skill's reference/manimgx.md, gotchas 19–29)


def _from_copy(mobject, target_mobject, **kwargs):
    """TransformFromCopy shows its finished target from the start of the line; a FadeIn from the source waits."""
    return m.FadeIn(target_mobject, target_position=mobject, **kwargs)


def _fade_transform(mobject, target_mobject, stretch=True, dim_to_match=1, **kwargs):
    """FadeTransform(Pieces) show the target early too; a cross-fade where each side glides waits for its cue."""
    return m.AnimationGroup(
        m.FadeOut(mobject, target_position=target_mobject),
        m.FadeIn(target_mobject, target_position=mobject),
        **kwargs,
    )


m.TransformFromCopy = _from_copy
m.FadeTransform = _fade_transform
m.FadeTransformPieces = _fade_transform


# A line that introduces a group and also moves or highlights a PART of it would show that part early:
# marking those animations as introducers keeps play() from adding them before their cue.
def _leaves(animation):
    subs = getattr(animation, "animations", None)
    if subs:
        for sub in subs:
            yield from _leaves(sub)
    else:
        yield animation


def _cue_safe(scene, animations):
    on_screen = {id(x) for x in scene.get_mobject_family_members()}
    roots, family = set(), set()
    for animation in animations:
        for leaf in _leaves(animation):
            if leaf.is_introducer() and getattr(leaf, "mobject", None) is not None:
                roots.add(id(leaf.mobject))
                family |= {id(x) for x in leaf.mobject.get_family()}
    for animation in animations:
        for leaf in _leaves(animation):
            mob = getattr(leaf, "mobject", None)
            if mob is None or leaf.is_introducer() or id(mob) in roots or id(mob) in on_screen:
                continue
            if id(mob) in family:
                leaf.introducer = True
    return animations


# Compiled under ManimGX's package path so play labels still point at the episode's own scene.py line.
_SAY = """
def say(self, script, *animations, voice=None):
    return _scene_say(self, script, *_cue_safe(self, animations), voice=voice)
"""
_say_namespace = {"_scene_say": m.Scene.say, "_cue_safe": _cue_safe}
exec(compile(_SAY, str(Path(m.__file__).parent / "_series_kit_say.py"), "exec"), _say_namespace)

# ---------------------------------------------------------------------------- design system
# A deep blue-green "silicon" stage. Colours carry meaning across the whole course — learn them once:
#   compute (FLOPs, prefill) AMBER · memory traffic (bytes, bandwidth, decode) CYAN · KV cache VIOLET
#   weights BLUE · requests / users PINK · generated tokens and "good" GREEN · "bad" RED · structure EDGE/MUTED

BG = "#0D1417"
CARD = "#132026"
CARD_HI = "#192A31"
EDGE = "#2A3E46"
INK = "#E6F0F2"
MUTED = "#91A7AE"
FAINT = "#4E636A"
AMBER, AMBER_FILL = "#FFB547", "#3A2A10"
CYAN, CYAN_FILL = "#3FD5E6", "#0E2F35"
VIOLET, VIOLET_FILL = "#A78BFA", "#261D40"
BLUE, BLUE_FILL = "#5B9DFF", "#13274A"
PINK, PINK_FILL = "#FF7AB8", "#3D1530"
GREEN, GREEN_FILL = "#5EE08A", "#123320"
RED, RED_FILL = "#FF6464", "#3D1717"
GOOD, GOOD_FILL = GREEN, GREEN_FILL
BAD, BAD_FILL = RED, RED_FILL
FILLS = {AMBER: AMBER_FILL, CYAN: CYAN_FILL, VIOLET: VIOLET_FILL, BLUE: BLUE_FILL, PINK: PINK_FILL,
         GREEN: GREEN_FILL, RED: RED_FILL, MUTED: CARD, EDGE: CARD, INK: CARD}

# what each colour means (use these names in scenes so the meaning stays fixed)
COMPUTE, MEMORY, KV, WEIGHTS, REQUEST, TOKEN = AMBER, CYAN, VIOLET, BLUE, PINK, GREEN

ROLES = {"system": (MUTED, CARD), "user": (PINK, PINK_FILL), "assistant": (GREEN, GREEN_FILL)}

HEAD = "DIN Alternate"   # titles, headers, big numbers (bold)
SANS = "Helvetica Neue"  # everything you read
MONO = "Source Code Pro"  # code, numbers in tables, units, labels: no ligatures, so "->", "!=" stay as typed

m.config.background_color = BG

MASCOT = "Owl"  # the course's running chatbot (drawn by owl())

# the course: module → (name, accent, fill); episode → (title, module). See COURSE.md.
MODULES = {
    1: ("ONE TOKEN, TWO SPEED LIMITS", AMBER, AMBER_FILL),
    2: ("THE KV CACHE", VIOLET, VIOLET_FILL),
    3: ("MANY USERS AT ONCE", PINK, PINK_FILL),
    4: ("FASTER TOKENS", GREEN, GREEN_FILL),
    5: ("FEWER BITS, MORE GPUS", BLUE, BLUE_FILL),
    6: ("PRODUCTION & YOUR MAC", CYAN, CYAN_FILL),
}
EPISODES = {
    1: ("Press Enter", 1), 2: ("The Two Speed Limits", 1), 3: ("Prefill vs Decode", 1),
    4: ("The Batching Free Lunch", 1),
    5: ("The KV Cache", 2), 6: ("Shrinking the Cache", 2), 7: ("PagedAttention", 2), 8: ("Never Prefill Twice", 2),
    9: ("The Cost of a Long Memory", 2),
    10: ("Continuous Batching", 3), 11: ("Chunked Prefill", 3), 12: ("Split the Work", 3),
    13: ("Guess and Check", 4), 14: ("Drafting Without a Draft Model", 4), 15: ("FlashAttention", 4),
    16: ("The CPU Tax", 4), 17: ("The Last Step of a Token", 4),
    18: ("Fewer Bits, Same Model", 5), 19: ("Four Bits and Below", 5), 20: ("Mixture of Experts at Serve Time", 5),
    21: ("When One GPU Isn't Enough", 5),
    22: ("The Serving Stack", 6), 23: ("Your Mac as an Inference Server", 6), 24: ("The Price of a Token", 6),
    25: ("Keeping It Alive", 6), 26: ("One Request, Revisited", 6),
}
LEVELS = MODULES  # older kits' name

_SMALL = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()


def spoken_title(title):
    """A title as the narrator should read it: no symbols (the lexicon splits CamelCase names)."""
    return title.rstrip("?.").replace(" & ", " and ").replace(":", ",").replace(" vs ", " versus ")


def say_number(n):
    """A number under a hundred as words, for the narrator."""
    return _SMALL[n] if n < 20 else _TENS[n // 10 - 2] + ("" if n % 10 == 0 else "-" + _SMALL[n % 10])


# ---------------------------------------------------------------------------- the voice

NARRATOR = os.environ.get("NARRATOR_URL", "http://127.0.0.1:8765/speak")
VOICE_REF = ""  # a clean 10–20 s WAV of a voice to copy; "" = the built-in voice


def narrator(text, voice_ref=VOICE_REF, exaggeration=0.5, cfg_weight=0.3, temperature=0.8):
    """Chatterbox, run locally by the tutorial-video skill's tts_server.py, timed by Whisper.
    Start it from the series folder: ~/.claude/skills/tutorial-video/scripts/narrator.sh"""
    settings = dict(voice_ref=voice_ref, exaggeration=exaggeration, cfg_weight=cfg_weight, temperature=temperature)
    request = urllib.request.Request(
        NARRATOR, json.dumps({"text": text, **settings}).encode(), {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=1800) as reply:
            said = json.load(reply)
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"no narrator at {NARRATOR}: start ~/.claude/skills/tutorial-video/scripts/narrator.sh"
            " in the series folder (or render with VOICE=draft)"
        ) from error
    return m.Speech(base64.b64decode(said["audio"]), text=text, words=[Word(*w) for w in said["words"]])


def draft_voice(text, seconds_per_word=0.3):
    """A silent stand-in at about the real voice's pace: for layout and timing passes."""
    words = re.findall(r"\S+", text)
    rate, lead = 24000, 0.15
    timed = [Word(w, lead + i * seconds_per_word, lead + (i + 1) * seconds_per_word) for i, w in enumerate(words)]
    silence = np.zeros(int((2 * lead + seconds_per_word * len(words)) * rate), dtype=np.float32)
    return m.Speech(silence, text=text, words=timed, rate=rate)


# ---------------------------------------------------------------------------- basic components


def text(s, size=32, color=INK, weight=m.NORMAL, **kw):
    """Body text (Helvetica Neue)."""
    return m.Text(s, font=SANS, font_size=size, color=color, weight=weight, **kw)


def title_text(s, size=40, color=INK, **kw):
    """Display text (DIN Alternate Bold): titles, big numbers, recap words."""
    return m.Text(s, font=HEAD, font_size=size, color=color, weight=m.BOLD, **kw)


def code(s, size=22, color=INK, t2c=None, **kw):
    """Monospace text (Source Code Pro). `t2c={"regex": colour}` colours matching parts (ManimGX's own t2c
    takes literal strings only, so the regexes are matched here; a key that matches nothing as a regex is
    tried as a literal). The first rule to claim a character wins."""
    words = m.Text(s, font=MONO, font_size=size, color=color, **kw)
    if t2c:
        paint(words, s, t2c)
    return words


def paint(words, s, rules):
    """Colour the glyphs of Text `words` (made from string `s`) whose characters match `rules`
    ({regex: colour}, in priority order). Whitespace draws no glyph, so indices skip it."""
    glyph_of, n = {}, 0
    for i, ch in enumerate(s):
        if not ch.isspace():
            glyph_of[i] = n
            n += 1
    if n != len(words):  # a ligature or emoji changed the glyph count: leave the text plain
        return words
    claimed = set()
    for pattern, colour in rules.items():
        try:
            spans = [mt.span() for mt in re.finditer(pattern, s, re.MULTILINE)]
        except re.error:
            spans = []
        if not spans:
            spans = [mt.span() for mt in re.finditer(re.escape(pattern), s)]
        for a, b in spans:
            for i in range(a, b):
                if i in glyph_of and i not in claimed:
                    claimed.add(i)
                    words[glyph_of[i]].set_color(colour)
    return words


def tex(s, size=40, color=INK, **kw):
    """A formula (LaTeX math)."""
    return m.MathTex(s, font_size=size, color=color, **kw)


def glyphs(words, key):
    """The glyphs that draw the first `key` in a Text (spaces and newlines draw none). Mono text only:
    ligatures (fi, fl) in the sans font merge two letters into one glyph."""
    start = words.text.index(key)
    skip = sum(c.isspace() for c in words.text[:start])
    width = sum(not c.isspace() for c in key)
    return words[start - skip : start - skip + width]


def icon(name, height):
    return m.SVGMobject(str(ASSETS / f"{name}.svg"), height=height)


def card(width, height, color=EDGE, fill=CARD, radius=0.2, stroke=2.5):
    return m.RoundedRectangle(
        corner_radius=radius, width=width, height=height,
        stroke_color=color, stroke_width=stroke, fill_color=fill, fill_opacity=1,
    )


def pill(label, color, fill, size=22, mono=False):
    words = code(label, size - 2, color, weight=m.BOLD) if mono else text(label, size, color, weight=m.BOLD)
    height = words.height + 0.3
    box = card(words.width + 0.55, height, color, fill, radius=height / 2, stroke=2)
    return m.VGroup(box, words.move_to(box))


def chip(word, color=CYAN, fill=CYAN_FILL, width=1.5, height=0.72, size=28, mono=False):
    """A word in a small rounded box, on the same baseline as its neighbours'."""
    box = card(width, height, color, fill, radius=0.14, stroke=2.5)
    label = (code(word, size - 4, INK) if mono else text(word, size, INK, weight=m.BOLD)).move_to(box)
    chars = word.replace(" ", "")
    if len(label) == len(chars):
        sitting = [g for g, c in zip(label, chars) if c not in "gjpqy,;_()[]{}"] or list(label)
        bottom = min(g.get_bottom()[1] for g in sitting)
    else:  # a ligature (fi, fl) merged two letters into one glyph: most glyphs sit on the baseline
        bottom = float(np.median([g.get_bottom()[1] for g in label]))
    baseline = box.get_center()[1] - 0.449 * size / 48 / 2
    label.shift((baseline - bottom) * m.UP)
    return m.VGroup(box, label)


def header(tag, title, color, fill):
    """The section header: an accent bar, a mono tag, a DIN title (top left)."""
    tag_words = code(tag.upper(), 20, color, weight=m.BOLD)
    title_words = title_text(title, 36)
    bar = m.RoundedRectangle(corner_radius=0.04, width=0.1, height=0.62, stroke_width=0, fill_color=color, fill_opacity=1)
    row = m.VGroup(bar, tag_words, title_words).arrange(m.RIGHT, buff=0.3)
    tag_words.align_to(title_words, m.DOWN)
    return row.to_corner(m.UL, buff=0.5)


def bubble(words, color, fill, tail="left"):
    box = card(words.width + 0.7, words.height + 0.55, color, fill, radius=0.3)
    words.move_to(box)
    side = -1 if tail == "left" else 1
    base = box.get_bottom() + np.array([side * (box.width / 2 - 0.6), 0.15, 0])
    nub = m.Polygon(
        base + [-0.2, 0, 0], base + [0.2, 0, 0], base + [side * 0.5, -0.55, 0],
        stroke_color=color, stroke_width=2.5, fill_color=fill, fill_opacity=1,
    )
    return m.VGroup(nub, box, words)


def node(name, label, sub, color, fill, width=2.1, height=2.1, label_size=26):
    box = card(width, height, color, fill)
    body = m.VGroup(icon(name, 0.8), text(label, label_size, INK, weight=m.BOLD), text(sub, 20, MUTED))
    body.arrange(m.DOWN, buff=0.13).move_to(box)
    return m.VGroup(box, body)


def tile(name, title, desc, color, width=6.3, height=2.1):
    box = card(width, height, color, CARD, stroke=2)
    words = m.VGroup(text(title, 30, INK, weight=m.BOLD), text(desc, 22, MUTED))
    words.arrange(m.DOWN, buff=0.18, aligned_edge=m.LEFT)
    body = m.VGroup(icon(name, 1.05), words).arrange(m.RIGHT, buff=0.45)
    body.move_to(box).align_to(box, m.LEFT).shift(0.45 * m.RIGHT)
    return m.VGroup(box, body)


def arrow(start, end, color=MUTED, buff=0.1):
    return m.Arrow(start, end, buff=buff, color=color, stroke_width=5, max_tip_length_to_length_ratio=0.3)


def pulse(a, color, width=10):
    return m.ShowPassingFlash(m.Line(a.get_start(), a.get_end()).set_stroke(color, width=width), time_width=0.7)


def on_baseline(words):
    for w in words[1:]:
        w.shift((words[0][0].get_bottom()[1] - w[0].get_bottom()[1]) * m.UP)
    return words


def glow(color, radius=3.0, layers=28):
    return m.VGroup(
        *[m.Circle(radius=radius * (1 - i / layers), stroke_width=0).set_fill(color, opacity=0.009) for i in range(layers)]
    )


def backdrop():
    image = m.ImageMobject(str(ASSETS / "backdrop.png"))
    image.height = 8
    return image


def axes(x_range, y_range, x_length, y_length, color=FAINT):
    """Quiet axes for the dark stage: thin, no numbers (label them with text()). Note: a range is cut to
    whole steps, so pick a step that divides it (gotcha 27)."""
    return m.Axes(
        x_range=x_range, y_range=y_range, x_length=x_length, y_length=y_length,
        axis_config={"color": color, "stroke_width": 2.5, "include_tip": True, "tip_length": 0.18},
    )


def recolored(name, color, fill, prefix):
    """An icon in another accent, written as assets/<prefix>-<name>-<hex>.svg (pass your episode's prefix,
    e.g. "ep07", so parallel episodes never write the same file). Returns the new name."""

    def swap(match):
        hexcode = match.group(0)
        if hexcode.upper() in (BG.upper(), CARD.upper()):
            return hexcode
        r, g, b = (int(hexcode[i : i + 2], 16) / 255 for i in (1, 3, 5))
        return color if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.3 else fill

    variant = f"{prefix}-{name}-{color.lstrip('#').lower()}"
    out = re.sub(r"#[0-9A-Fa-f]{6}", swap, (ASSETS / f"{name}.svg").read_text())
    path = ASSETS / f"{variant}.svg"
    if not path.exists() or path.read_text() != out:
        path.write_text(out)
    return variant


# ---------------------------------------------------------------------------- code and data cards

_PY_KEYWORDS = r"\b(def|while|for|in|if|elif|else|return|import|from|as|with|try|except|raise|class|break|continue|not|and|or|is|None|True|False|lambda|yield|await|async)\b"


def code_block(src, size=20, width=None, title=None, accent=EDGE):
    """Python (or pseudo-code) in a dark card, lightly highlighted: keywords VIOLET, strings GREEN, comments
    FAINT, numbers AMBER, called names CYAN. Keep lines short (about 44 characters at size 20 fill 7 units)."""
    t2c = {
        r"#[^\n]*": FAINT,
        r"\"[^\"\n]*\"|'[^'\n]*'": GREEN,
        _PY_KEYWORDS: VIOLET,
        r"\b\d+(\.\d+)?(e\d+)?\b": AMBER,
        r"\b[a-z_][a-z0-9_]*(?=\()": CYAN,
    }
    body = code(src, size, INK, t2c=t2c)
    w = max(width or 0, body.width + 0.7)
    top, parts = 0.0, []
    if title:
        cap = code(title, 20, MUTED)
        top = cap.height + 0.25
        parts.append(cap)
    box = card(w, body.height + 0.6 + top, accent, CARD, radius=0.16, stroke=2)
    body.move_to(box).align_to(box, m.LEFT).shift(0.35 * m.RIGHT)
    if title:
        body.shift(-top / 2 * m.UP)
        parts[0].align_to(box, m.UL).shift(0.35 * m.RIGHT + 0.18 * m.DOWN)
    return m.VGroup(box, body, *parts)


def terminal(lines, width=6.5, size=20, title="terminal"):
    """A terminal window. Lines starting with "$ " are commands (the prompt in GREEN); lines starting
    with "! " are errors (RED); others are output (MUTED)."""
    rows = []
    for line in lines:
        if line.startswith("$ "):
            rows.append(code(line, size, INK, t2c={r"^\$": GREEN}))
        elif line.startswith("! "):
            rows.append(code(line[2:], size, RED))
        else:
            rows.append(code(line or " ", size, MUTED))
    body = m.VGroup(*rows).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.14)
    bar_h = 0.42
    w = max(width, body.width + 0.7)
    box = card(w, body.height + 0.55 + bar_h, EDGE, "#0A1013", radius=0.16, stroke=2)
    bar = m.Line(box.get_corner(m.UL) + [0.0, -bar_h, 0], box.get_corner(m.UR) + [0.0, -bar_h, 0]).set_stroke(EDGE, 2)
    dots = m.VGroup(*[m.Dot(radius=0.06, color=c) for c in (RED, AMBER, GREEN)]).arrange(m.RIGHT, buff=0.12)
    dots.move_to(box.get_corner(m.UL) + [0.4, -bar_h / 2, 0])
    cap = code(title, 20, FAINT).move_to(box.get_top() + [0, -bar_h / 2, 0])
    body.next_to(bar, m.DOWN, buff=0.28).align_to(box, m.LEFT).shift(0.35 * m.RIGHT)
    return m.VGroup(box, bar, dots, cap, body)


def json_card(obj, size=19, accent=CYAN, fill=CARD, width=None, title=None):
    """A JSON object as a card: keys CYAN, strings GREEN, numbers AMBER, true/false/null VIOLET."""
    src = json.dumps(obj, indent=2, ensure_ascii=False)
    t2c = {
        r"\"[^\"\n]*\"(?=\s*:)": CYAN,
        r":\s*\"[^\"\n]*\"": GREEN,
        r"\b-?\d+(\.\d+)?\b": AMBER,
        r"\b(true|false|null)\b": VIOLET,
    }
    body = code(src, size, INK, t2c=t2c)
    top, cap = 0.0, None
    if title:
        cap = code(title, 20, accent, weight=m.BOLD)
        top = cap.height + 0.22
    w = max(width or 0, body.width + 0.7, (cap.width + 0.7) if cap else 0)
    box = card(w, body.height + 0.55 + top, accent, fill, radius=0.16, stroke=2)
    body.move_to(box).align_to(box, m.LEFT).shift(0.35 * m.RIGHT + top / 2 * m.DOWN)
    group = m.VGroup(box, body)
    if cap:
        cap.align_to(box, m.UL).shift(0.35 * m.RIGHT + 0.17 * m.DOWN)
        group.add(cap)
    return group


def message(role, body, width=5.2, size=20, mono=False):
    """One chat message: a role label in its colour (system MUTED, user PINK, assistant = Owl GREEN)."""
    color, fill = ROLES[role]
    label = code(role, 22, color, weight=m.BOLD)
    words = code(body, size - 2, INK) if mono else text(body, size, INK)
    inner = m.VGroup(label, words).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.12)
    if words.width > width - 0.6:
        words.scale((width - 0.6) / words.width)
        inner.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.12)
    box = card(width, inner.height + 0.36, EDGE, CARD, radius=0.12, stroke=1.5)
    stripe = m.RoundedRectangle(corner_radius=0.05, width=0.09, height=box.height - 0.16, stroke_width=0,
                                fill_color=color, fill_opacity=1)
    stripe.move_to(box).align_to(box, m.LEFT).shift(0.1 * m.RIGHT)
    inner.move_to(box).align_to(box, m.LEFT).shift(0.32 * m.RIGHT)
    return m.VGroup(box, stripe, inner)


def loop_ring(labels, radius=1.6, colors=None, size=24, center=(0, 0, 0)):
    """A cycle of steps around a circle with arrows between them. Returns VGroup(nodes, arcs)."""
    n = len(labels)
    colors = colors or [AMBER, CYAN, GREEN, PINK, VIOLET][:n]
    center = np.array(center, dtype=float)
    angles = [np.pi / 2 - 2 * np.pi * i / n for i in range(n)]
    nodes = m.VGroup()
    for label, color, a in zip(labels, colors, angles):
        words = text(label, size, INK, weight=m.BOLD)
        c = chip(label, color, FILLS.get(color, CARD), width=max(1.4, words.width + 0.5), height=0.66, size=size)
        nodes.add(c.move_to(center + radius * np.array([np.cos(a), np.sin(a), 0])))

    def clearance(chip_, a):
        half_w, half_h = chip_.width / 2 + 0.14, chip_.height / 2 + 0.14
        reach = half_w * abs(np.sin(a)) + half_h * abs(np.cos(a))
        return min(reach / radius, 0.45 * 2 * np.pi / n)

    arcs = m.VGroup()
    for i in range(n):
        a0, a1 = angles[i], angles[(i + 1) % n]
        if a1 > a0:
            a1 -= 2 * np.pi
        start, end = a0 - clearance(nodes[i], a0), a1 + clearance(nodes[(i + 1) % n], a1)
        arc = m.Arc(radius=radius, start_angle=start, angle=end - start, arc_center=center)
        arcs.add(arc.set_stroke(MUTED, 4).add_tip(tip_length=0.2))
    return m.VGroup(nodes, arcs)


# ---------------------------------------------------------------------------- inference components


def token_row(tokens, color=TOKEN, size=20, gap=0.08, max_width=12.0, fill=None):
    """Tokens as small mono chips in a row (wrapping at max_width). Returns a VGroup of chips."""
    ref = code("Mx", size)  # every chip has the same height and baseline, whatever its letters
    chips = []
    for tok in tokens:
        shown = tok if tok.strip() else "␣"
        words = code(shown, size, INK if tok.strip() else MUTED)
        box = card(max(words.width + 0.26, 0.44), ref.height + 0.3, color, fill or FILLS.get(color, CARD),
                   radius=0.08, stroke=1.8)
        words.move_to(box)
        sitting = [g for g, ch in zip(words, shown.replace(" ", "")) if ch not in "gjpqy,;_()[]{}"] or list(words)
        if len(words) == len(shown.replace(" ", "")):
            words.shift((box.get_y() - ref.height / 2 - min(g.get_bottom()[1] for g in sitting)) * m.UP)
        chips.append(m.VGroup(box, words))
    rows, row, width = [], [], 0.0
    for c in chips:
        if row and width + c.width > max_width:
            rows.append(m.VGroup(*row).arrange(m.RIGHT, buff=gap))
            row, width = [], 0.0
        row.append(c)
        width += c.width + gap
    if row:
        rows.append(m.VGroup(*row).arrange(m.RIGHT, buff=gap))
    m.VGroup(*rows).arrange(m.DOWN, aligned_edge=m.LEFT, buff=gap)
    return m.VGroup(*chips)


def stat(value, unit="", label="", color=INK, size=72):
    """A big number (DIN) with a small unit and a MUTED label under it."""
    number = title_text(value, size, color)
    parts = [number]
    if unit:
        u = code(unit, max(20, size // 3), MUTED)
        u.next_to(number, m.RIGHT, buff=0.15).align_to(number, m.DOWN)
        parts.append(u)
    group = m.VGroup(*parts)
    if label:
        group.add(text(label, 22, MUTED).next_to(group, m.DOWN, buff=0.18).align_to(number, m.LEFT))
    return group


def cell_grid(rows, cols, size=0.34, gap=0.06, color=EDGE, fill=CARD):
    """A grid of small squares (KV blocks, batch slots × steps …). grid[r][c] is one cell."""
    grid = m.VGroup(*[
        m.VGroup(*[m.Square(size).set_stroke(color, 1.5).set_fill(fill, 1) for _ in range(cols)]).arrange(m.RIGHT, buff=gap)
        for _ in range(rows)
    ]).arrange(m.DOWN, buff=gap)
    return grid


def owl(size=1.0, color=GREEN, fill=GREEN_FILL):
    """Owl, the course's chatbot: a round body, two big eyes, ear tufts and a small beak.
    Parts: [0] body, [1] eyes (VGroup of two rings with pupils), [2] tufts, [3] beak."""
    s = size
    body = m.RoundedRectangle(corner_radius=0.55 * s, width=1.25 * s, height=1.2 * s,
                              stroke_color=color, stroke_width=4 * s, fill_color=fill, fill_opacity=1)
    eye = lambda: m.VGroup(m.Circle(radius=0.22 * s).set_stroke(color, 4 * s).set_fill(BG, 1),
                           m.Dot(radius=0.085 * s, color=INK))
    eyes = m.VGroup(eye(), eye()).arrange(m.RIGHT, buff=0.08 * s).move_to(body).shift(0.12 * s * m.UP)
    top = body.get_top()
    tufts = m.VGroup(*[
        m.Polygon(top + [side * 0.48 * s, -0.12 * s, 0], top + [side * 0.22 * s, -0.02 * s, 0],
                  top + [side * 0.5 * s, 0.24 * s, 0], stroke_color=color, stroke_width=4 * s,
                  fill_color=fill, fill_opacity=1)
        for side in (-1, 1)
    ])
    beak = m.Polygon(*(body.get_center() + np.array(p) * s for p in [(-0.09, -0.12, 0), (0.09, -0.12, 0), (0, -0.28, 0)]),
                     stroke_width=0, fill_color=AMBER, fill_opacity=1)
    return m.VGroup(body, eyes, tufts, beak)


class Meter(m.VGroup):
    """A horizontal gauge (memory used, a budget …). `fill_to(f)` animates it; it turns RED above 90%."""

    def __init__(self, width=4.0, fraction=0.0, color=CYAN, height=0.24):
        super().__init__()
        self.track_width, self.bar_height, self.color = width, height, color
        self.track = m.RoundedRectangle(corner_radius=height / 2, width=width, height=height,
                                        stroke_color=EDGE, stroke_width=2, fill_color=CARD, fill_opacity=1)
        self.bar = self._bar(fraction)
        self.add(self.track, self.bar)

    def _bar(self, fraction):
        f = float(np.clip(fraction, 0.004, 1.0))
        bar = m.RoundedRectangle(corner_radius=self.bar_height / 2, width=self.track_width * f, height=self.bar_height,
                                 stroke_width=0, fill_color=RED if fraction > 0.9 else self.color, fill_opacity=1)
        return bar.align_to(self.track, m.LEFT).set_y(self.track.get_y())

    def fill_to(self, fraction, **kwargs):
        return m.Transform(self.bar, self._bar(fraction).align_to(self.track, m.LEFT), **kwargs)


class MemoryBar(m.VGroup):
    """A memory budget drawn as one long bar of `capacity` bytes, filled left to right by named segments
    (weights BLUE, KV VIOLET …). `set(name, nbytes, color)` returns the animation that adds or resizes a
    segment; later segments slide to stay packed. `.segments[name]` is the drawn rectangle."""

    def __init__(self, capacity, width=10.0, height=0.6, title=None, unit="GB"):
        super().__init__()
        self.capacity, self.bar_width, self.bar_height, self.unit = capacity, width, height, unit
        self.track = m.Rectangle(width=width, height=height).set_stroke(EDGE, 2).set_fill(CARD, 1)
        self.add(self.track)
        self.order, self.sizes, self.colors, self.segments = [], {}, {}, {}
        if title:
            self.caption = code(title, 20, MUTED).next_to(self.track, m.UP, buff=0.14).align_to(self.track, m.LEFT)
            self.add(self.caption)

    def _rect(self, x0, nbytes, color):
        w = max(self.bar_width * nbytes / self.capacity, 0.001)
        rect = m.Rectangle(width=w, height=self.bar_height - 0.08).set_stroke(width=0).set_fill(color, 1)
        return rect.move_to([x0 + w / 2, self.track.get_y(), 0])

    def set(self, name, nbytes, color=None, run_time=0.9):
        if name not in self.sizes:
            self.order.append(name)
        self.sizes[name] = nbytes
        self.colors[name] = color or self.colors.get(name, MUTED)
        x, anims = self.track.get_left()[0], []
        for key in self.order:
            target = self._rect(x, self.sizes[key], self.colors[key])
            if key in self.segments:
                anims.append(m.Transform(self.segments[key], target))
            else:
                self.segments[key] = target
                self.add(target)
                anims.append(m.GrowFromEdge(target, m.LEFT))
            x += target.width
        return m.AnimationGroup(*anims, run_time=run_time)

    def used(self):
        return sum(self.sizes.values())


class Roofline(m.VGroup):
    """The roofline chart on log–log axes: attainable FLOP/s = min(peak, bandwidth × intensity).
    The memory slope is CYAN, the compute roof AMBER, the ridge a dashed line. `.point(I)` is the scene
    point on the roof at intensity I (FLOPs/byte); `.dot(I)` a dot there; `.at(I, flops)` any point."""

    def __init__(self, bandwidth, peak, width=7.0, height=4.2, x_decades=(-1, 4), y_decades=(11, 15.5),
                 center=(0, -0.3, 0), labels=True):
        super().__init__()
        self.bw, self.peak, self.w, self.h = bandwidth, peak, width, height
        self.xd, self.yd = x_decades, y_decades
        self.origin = np.array(center, dtype=float) - np.array([width / 2, height / 2, 0])
        x_axis = m.Arrow(self.origin, self.origin + [width + 0.7, 0, 0], buff=0, color=FAINT, stroke_width=3,
                         max_tip_length_to_length_ratio=0.04)
        y_axis = m.Arrow(self.origin, self.origin + [0, height + 0.5, 0], buff=0, color=FAINT, stroke_width=3,
                         max_tip_length_to_length_ratio=0.06)
        self.axes = m.VGroup(x_axis, y_axis)
        self.ridge_x = peak / bandwidth
        lo, hi = 10 ** x_decades[0], 10 ** x_decades[1]
        self.memory = m.Line(self.point(lo), self.point(self.ridge_x)).set_stroke(MEMORY, 6)
        self.compute = m.Line(self.point(self.ridge_x), self.point(hi)).set_stroke(COMPUTE, 6)
        self.ridge = m.DashedLine(self.at(self.ridge_x, 10 ** y_decades[0]), self.point(self.ridge_x),
                                  dash_length=0.12).set_stroke(MUTED, 2)
        self.add(self.axes, self.memory, self.compute, self.ridge)
        if labels:
            ticks = m.VGroup()
            for d in range(x_decades[0], x_decades[1] + 1):
                ticks.add(code(_si(10 ** d, ""), 20, MUTED).next_to(self.at(10 ** d, 10 ** y_decades[0]), m.DOWN, buff=0.12))
            for d in range(int(np.floor(y_decades[0])) + 1, int(y_decades[1]) + 1):  # none at the corner
                ticks.add(code(_si(10 ** d, "F/s"), 20, MUTED).next_to(self.at(10 ** x_decades[0], 10 ** d), m.LEFT, buff=0.12))
            self.x_label = text("arithmetic intensity (FLOPs per byte)", 22, MUTED).next_to(ticks[0], m.DOWN, buff=0.18)
            self.x_label.set_x(self.origin[0] + width / 2)
            self.ticks = ticks
            self.add(ticks, self.x_label)
        offset = np.array(center, dtype=float) - self.get_center()  # centre the whole chart, labels included
        self.shift(offset)
        self.origin = self.origin + offset

    def ridge_label(self, size=20):
        """'ridge ≈ N FLOPs/byte', just above the bend."""
        return code(f"ridge ≈ {self.ridge_x:.0f} FLOPs/byte", size, MUTED).next_to(self.point(self.ridge_x), m.UP, buff=0.25)

    def at(self, intensity, flops):
        fx = (np.log10(intensity) - self.xd[0]) / (self.xd[1] - self.xd[0])
        fy = (np.log10(flops) - self.yd[0]) / (self.yd[1] - self.yd[0])
        return self.origin + np.array([fx * self.w, fy * self.h, 0])

    def attainable(self, intensity):
        return min(self.peak, self.bw * intensity)

    def point(self, intensity):
        return self.at(intensity, self.attainable(intensity))

    def dot(self, intensity, color=INK, radius=0.11):
        return m.Dot(self.point(intensity), radius=radius, color=color).set_z_index(3)


def _si(x, unit):
    for value, prefix in ((1e15, "P"), (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if x >= value:
            v = x / value
            return f"{v:g} {prefix}{unit}".strip() if unit else f"{v:g}{prefix}"
    return f"{x:g} {unit}".strip()


class Timeline(m.VGroup):
    """A time axis for one request (or many): `.x(t)` maps seconds to the scene; `.bar(t0, t1, color)`
    a block (e.g. prefill AMBER), `.tick(t, color)` a thin mark (e.g. one decode step CYAN)."""

    def __init__(self, seconds, width=11.0, origin=(-5.5, -2.0, 0), lane_height=0.5, label_every=None, unit="s"):
        super().__init__()
        # the axis length is `span`, not `self.width`: a VGroup's width is measured live and grows with the tick labels
        self.seconds, self.span, self.origin, self.lane_height = seconds, width, np.array(origin, float), lane_height
        self.axis = m.Line(self.origin, self.origin + [width, 0, 0]).set_stroke(FAINT, 2)
        self.add(self.axis)
        if label_every:
            marks = m.VGroup()
            t = 0.0
            while t <= seconds + 1e-9:
                at = self.origin + np.array([self.span * t / seconds, 0, 0])
                tick = m.Line(at + [0, -0.07, 0], at + [0, 0.07, 0]).set_stroke(FAINT, 2)
                lab = code(f"{t:g} {unit}", 20, MUTED).next_to(tick, m.DOWN, buff=0.1)
                marks.add(tick, lab)
                t += label_every
            self.marks = marks
            self.add(marks)

    def x(self, t, lane=0):
        return self.origin + np.array([self.span * t / self.seconds, 0.2 + (lane + 0.5) * self.lane_height, 0])

    def bar(self, t0, t1, color=COMPUTE, lane=0, label=None, size=20):
        w = max(self.span * (t1 - t0) / self.seconds, 0.02)
        rect = m.Rectangle(width=w, height=self.lane_height * 0.8).set_stroke(color, 1.5).set_fill(FILLS.get(color, CARD), 1)
        rect.move_to(self.x((t0 + t1) / 2, lane))
        if label:
            words = code(label, size, color)
            if words.width < w - 0.15:
                words.move_to(rect)
            else:
                words.next_to(rect, m.UP, buff=0.08)
            return m.VGroup(rect, words)
        return rect

    def tick(self, t, color=MEMORY, lane=0):
        p = self.x(t, lane)
        return m.Line(p + [0, -0.32 * self.lane_height, 0], p + [0, 0.32 * self.lane_height, 0]).set_stroke(color, 2.5)


# ---------------------------------------------------------------------------- the episode


class Episode(m.Scene):
    """An episode of the course: its intro card, its lesson (`lesson()`), the next-episode card."""

    number = 0
    voice = draft_voice if os.environ.get("VOICE") == "draft" else narrator
    say = _say_namespace["say"]

    def construct(self) -> None:
        self.head = None
        self.add(backdrop().set_z_index(-100))
        self.series_intro()
        self.lesson()
        self.series_outro()

    def lesson(self) -> None:
        raise NotImplementedError

    @property
    def module(self):
        return MODULES[EPISODES[self.number][1]]

    level = module

    def new_header(self, tag, title, color=None, fill=None):
        """The animation that swaps the header: pass it to say() with a short [span]."""
        _, accent, tint = self.module
        new = header(tag, title, color or accent, fill or tint)
        parts = [m.FadeIn(new, shift=0.25 * m.UP)]
        if self.head:
            parts.insert(0, m.FadeOut(self.head, shift=0.25 * m.UP))
        self.head = new
        return m.AnimationGroup(*parts, run_time=0.9)

    def clear(self, *keep, run_time=0.8):
        gone = [mob for mob in self.mobjects[1:] if mob not in keep and mob is not self.head]
        if gone:
            self.play(*[m.FadeOut(mob) for mob in gone], run_time=run_time)

    def drop_header(self):
        old, self.head = self.head, None
        return m.FadeOut(old, shift=0.25 * m.UP) if old else m.Wait(0.01)

    def _fit(self, heading):
        if heading.width > TITLE_WIDTH:
            heading.scale(TITLE_WIDTH / heading.width)
        return heading

    def series_intro(self):
        title, module = EPISODES[self.number]
        name, accent, tint = MODULES[module]
        self.next_section(f"Episode {self.number} · {title}", notes="Series card.")
        halo = glow(accent, radius=4.2).move_to(0.1 * m.DOWN)
        prompt = code(f"$ owl serve --episode {self.number:02d}", 22, MUTED, t2c={r"^\$": GREEN})
        brand = title_text("Tokens per Second", 42, INK, gradient=(CYAN, AMBER))
        badge = pill(f"MODULE {module} · {name}", accent, tint, 20, mono=True)
        heading = self._fit(title_text(title, 76 if len(title) < 22 else 60))
        top = m.VGroup(prompt, brand).arrange(m.DOWN, buff=0.22)
        group = m.VGroup(top, badge, heading).arrange(m.DOWN, buff=0.45).move_to(0.25 * m.UP)
        half = min(heading.width, 6) / 2
        rule = m.Line([-half, 0, 0], [half, 0, 0]).set_stroke(accent, width=5).next_to(heading, m.DOWN, buff=0.35)
        mascot = owl(0.5).next_to(rule, m.DOWN, buff=0.45)

        self.wait(0.3)
        self.say(
            f"[Welcome to Tokens per Second.] [Episode {say_number(self.number)}]: [{spoken_title(title)}.]",
            m.AnimationGroup(m.FadeIn(halo, scale=0.6), m.AddTextLetterByLetter(prompt), m.FadeIn(brand, shift=0.2 * m.DOWN), run_time=1.2),
            m.AnimationGroup(m.FadeIn(badge, shift=0.2 * m.UP), m.FadeIn(mascot, shift=0.2 * m.UP)),
            m.AnimationGroup(m.Write(heading), m.GrowFromCenter(rule)),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, group, rule, mascot)], run_time=0.8)

    def series_outro(self):
        upcoming = self.number + 1
        self.next_section("Next episode", notes="Next-episode card.")
        if self.head:
            self.play(self.drop_header(), run_time=0.6)
        if upcoming not in EPISODES:
            self.say("Thanks for watching!")
            return
        title, module = EPISODES[upcoming]
        name, accent, tint = MODULES[module]
        halo = glow(accent, radius=4.0)
        kicker = code("next episode", 24, MUTED)
        number = pill(f"EP {upcoming:02d} · {name}", accent, tint, 20, mono=True)
        heading = self._fit(title_text(title, 64 if len(title) < 26 else 50))
        group = m.VGroup(kicker, number, heading).arrange(m.DOWN, buff=0.4).move_to(0.2 * m.UP)
        self.say(
            f"[Next time, in episode {say_number(upcoming)}]: [{spoken_title(title)}.] Thanks for watching!",
            m.AnimationGroup(m.FadeIn(halo, scale=0.6), m.FadeIn(kicker), m.FadeIn(number, shift=0.2 * m.UP)),
            m.Write(heading),
        )
        self.wait(1.0)
        self.play(*[m.FadeOut(mob) for mob in (halo, group)], run_time=1.0)
        self.wait(0.5)
