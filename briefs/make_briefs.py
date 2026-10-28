"""Write one subagent brief per episode (briefs/epNN.md) for "Tokens per Second", from the rows of COURSE.md,
the per-episode notes in SPEC (briefs/spec.py), briefs/pronunciation.txt and the series notes below.
Run: python3 briefs/make_briefs.py [N ...]"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERIES_DIR = HERE.parent
VIDEOS = SERIES_DIR.parent
sys.path.insert(0, str(HERE))
from spec import SPEC  # noqa: E402

NARRATOR_PYTHON = "~/rag-video/tts/.venv/bin/python"
MODULE_NAMES = {1: "ONE TOKEN, TWO SPEED LIMITS", 2: "THE KV CACHE", 3: "MANY USERS AT ONCE", 4: "FASTER TOKENS",
                5: "FEWER BITS, MORE GPUS", 6: "PRODUCTION & YOUR MAC"}


def _scene(series, n):
    hit = sorted(VIDEOS.glob(f"{series}/ep{n:02d}-*/scene.py"))
    return str(hit[0]) if hit else None


AG = {2: "the loop stepped beside a growing messages list; asserts tying numbers to a real run",
      6: "a scrolling context column and a token meter, 'every call re-sends the context'",
      7: "KV cache & prompt caching from the agent side: prefix strips, cost bars, ×1.8 bill",
      16: "a protocol traced end to end with real JSON packets", 20: "a backoff bar chart with jitter bands",
      22: "a task board with badges; tests run for real", 23: "a timeline with lanes and a moving clock marker",
      24: "mailboxes, a state machine and a tracker table", 25: "real git worktrees run during render",
      26: "costs (4×/15× tokens), failure taxonomy grid", 27: "a real script run; a 2×2 grid",
      28: "a browser at 1366×768 with pixel coordinates; a robot arm; a voice timeline",
      29: "pass@k / pass^k curves, statistics, intervals", 30: "16 rollouts in sandboxes, advantages, a toy update",
      32: "a trace waterfall, cost and latency ledgers, crash/resume", 33: "the finale: 14 layers plugged into one loop"}
AI = {24: "attention & transformers", 32: "retrieval pipeline", 33: "decoding & sampling", 37: "efficient attention, KV sizes",
      38: "positional encodings & long context", 39: "mixture of experts", 42: "quantization & distillation",
      43: "reasoning & test-time compute"}


def references(keys):
    out = []
    for key in keys:
        kind, n = key[:2], int(key[2:])
        if kind == "ag":
            path, what = _scene("agent-series", n), AG.get(n, "")
            label = "'Agents, Animated' (sister course, OLD palette)"
        elif kind == "ai":
            path, what = _scene("ai-series", n), AI.get(n, "")
            label = "'AI, Explained' (sister course, OLD palette)"
        else:
            path, what = _scene("inference-series", n), "THIS course, NEW palette — match it"
            label = f"this course's Episode {n}"
            if not path:
                out.append(f"{label} (not made yet — skip)")
                continue
        out.append(f"{path} — {label}: {what}")
    return "; ".join(out)


TEMPLATE = """You are producing ONE narrated animated tutorial for the course "Tokens per Second" — how LLM inference really works, from one token to a fleet (Mac, Apple Silicon). Work fully autonomously and finish with a short report.

YOUR EPISODE: Episode {N} · "{TITLE}" (Module {M} · {MODULE_NAME}). Folder: {SERIES_DIR}/ep{NN}-{slug}/ (create it). Scene class: `Ep{NN}{ClassName}(Episode)` with `number = {N}`.
Goal (what the viewer will understand): {GOAL}
Key concepts: {CONCEPTS}
Mechanism to animate: {MECHANISM}
Running example: {EXAMPLE}
EPISODE NOTES: {NOTES}

READ FIRST, in this order:
1. ~/.claude/skills/tutorial-video/SKILL.md, then reference/narration.md, reference/design.md, reference/manimgx.md (gotchas 19–29 matter), reference/series.md.
2. {SERIES_DIR}/COURSE.md (the whole course: what comes before and after yours — link to other episodes by number, never repeat them at length).
3. {SERIES_DIR}/kit.py — the visual system (deep blue-green stage; DIN Alternate titles, Helvetica Neue text, Source Code Pro code) and the inference components: Roofline, MemoryBar, Timeline, token_row, stat, cell_grid, Meter, owl() (the course's chatbot), code_block/terminal/json_card, message, loop_ring; header via self.new_header(tag, title). Look at its storyboards ({SERIES_DIR}/KitDemo.storyboard*.png, already current — do not re-run kit_demo.py, it writes outside your folder).
4. {SERIES_DIR}/owl_reference.py — the canonical numbers and code (MODELS, HARDWARE, weight_bytes, kv_bytes_per_token, flops_per_token, ridge, step_time, decode_ceiling, prefill_time, measured(), TinyLM with prefill/decode and its KV cache, sample, generate, Request/static_batching/continuous_batching, expected_tokens_per_pass/speculative_speedup). Import it in your scene (`from owl_reference import *` after the sys.path line) and COMPUTE every number you show from it; any code you show is an excerpt (≤ 10 lines) in its style and names.
5. {SERIES_DIR}/research/notes.md — sections: {NOTES_SECTIONS}. And {SERIES_DIR}/research/measurements.md (numbers measured on THIS Mac Studio, M4 Max 546 GB/s, 64 GB, with mlx-lm; load them with measured(...) — never re-run the benchmarks yourself). Open the original sources the notes cite when you need more detail (WebFetch).
6. Style references (read their scene.py and look at a few storyboard PNGs; sister courses share the engine and pacing but use an OLD palette — copy structure and care, NOT colours or fonts): {REFERENCE_EPISODES}.

Match the series: one running example carried through the episode (Owl unless a better example teaches better — then say so), a header (mono tag + DIN title via self.new_header) per section, sections via self.next_section(...), ~38–46 narrated self.say(...) lines, 3.5–4.5 minutes, a three-word recap near the end. Show mechanisms MOVING: bytes flowing, bars filling, caches growing, dots sliding on the roofline, tokens streaming. Every number on screen is computed in the scene (from owl_reference / measured()) and asserted, or cited with its source; invented data is said "for example" and recorded as illustrative in the section notes.

HARD RULES (other agents work in the same series folder at the same time):
- Write only inside your episode folder. Do NOT edit kit.py, owl_reference.py, COURSE.md, lexicon.json, research/, other episodes, or existing files in {SERIES_DIR}/assets/. You MAY add NEW icon SVGs to assets/ named `ep{NN}-*.svg` (flat 2-tone, this palette — see kit.py), and recoloured copies via kit's recolored(name, color, fill, prefix="ep{NN}").
- Do not run models or benchmarks (no mlx_lm, no downloads): the measurements are done and shared.
- The narrator is ALREADY RUNNING at http://127.0.0.1:8765/speak, shared by several agents (requests queue). NEVER start, stop or restart it. If a request fails, wait a minute and retry. Its log is {SERIES_DIR}/narrator.log (shared, appended).
- Python with Whisper: {NARRATOR_PYTHON}. ManimGX CLI: manimgx. Keep each response and tool output compact (pipe long logs through tail/grep): long turns time out. In the shell, `ls` is aliased to eza, which can hang in a background shell: use `/bin/ls`.

PROCESS:
1. Write the script + scene. 2. Draft pass: `cd <folder> && VOICE=draft manimgx inspect scene.py` until exit 0 and "layout: no problems in N samples"; READ EVERY storyboard PNG and fix what you see (overlaps, cramped or empty frames, text over lines, misaligned baselines, faint or half-drawn objects, anything closer than 0.5 units to the frame edge). 3. Voice pass: `manimgx inspect scene.py`. For a line below 100% after 4 takes, transcribe the clip (librosa 16 kHz → mlx_whisper, path_or_hf_repo="mlx-community/whisper-large-v3-turbo") to decide: checker false alarm (keep) or real misread (reword, `python3 ~/.claude/skills/tutorial-video/scripts/drop_lines.py voice '<regex>'`, re-run inspect). 4. `python3 ~/.claude/skills/tutorial-video/scripts/drop_lines.py voice . --voice-id draft_voice`. 5. `manimgx render scene.py`, then `manimgx present scene.py --no-open`. 6. From INSIDE your folder: `{NARRATOR_PYTHON} ~/.claude/skills/tutorial-video/scripts/verify_audio.py <Scene>.mp4 --terms AI,<every ALL-CAPS acronym you say>`; and a sync check of 2 cues with `manimgx inspect scene.py -t a,b -o /tmp/ep{NN}_sync.png` (Read it).

FINAL REPORT (short): paths of MP4/SRT/HTML/scene.py; duration; layout result; verify_audio summary; lines reworded or below 100% and why; new icons/helpers; facts and numbers you checked (with sources) and anything you are unsure about.
"""

COMMON = """
SERIES NOTES (apply to every episode):
- Visual language (keep it consistent — viewers learn it once): the colours carry meaning across the course — compute (FLOPs, prefill, math units) AMBER · memory traffic (bytes, bandwidth, decode) CYAN · KV cache VIOLET · weights BLUE · requests / users PINK · generated tokens and good outcomes GREEN · bad RED · structure EDGE/MUTED. kit.py names them COMPUTE, MEMORY, KV, WEIGHTS, REQUEST, TOKEN. Headers use your module's accent (kit picks it). Display text (titles, big numbers, recap words) in title_text (DIN); reading text in text() (Helvetica Neue); code, units, tables of numbers, commands and exact strings in code() / code_block / json_card / terminal (Source Code Pro).
- The running example: Owl, a study-buddy chatbot (students paste notes, ask questions, get answers and JSON flashcards). Owl-8B = Llama-3.1-8B: 8 B parameters, 16 GB of weights (BF16), 128 KiB of KV cache per token — repeat these three numbers when they help. Owl-mini = Llama-3.2-1B 4-bit for live-Mac demos. Owl appears as owl() (small, a corner presence; not in every frame).
- Facts: prefer research/notes.md (each fact is sourced and dated) and primary sources (papers, official docs). Show "Source, Year" in small MUTED text (font_size ≥ 20) near a claim that needs one, e.g. "Kwon et al., 2023", "NVIDIA, H100 spec", "measured on this Mac · mlx-lm". Hardware specs and prices are fast-moving: label them dated (October 2026) in the section notes; avoid prices unless the notes give a sourced number — otherwise say "for example".
- Measured numbers: use measured(...) from owl_reference and say "measured on this Mac" on screen. Always pair a measured decode number with its ceiling (bandwidth ÷ bytes) — the course's signature check.
- No real logos, screenshots or photos; draw GPUs, Macs and servers as simple cards/icons (assets: gpu, mac, server, ram, llm, database, …). Product names (vLLM, SGLang, MLX, llama.cpp, H100) are fine as text.
- Numbers as words in narration ("sixteen gigabytes", "a hundred and twenty-eight kibibytes" — or rephrase: "a hundred and twenty-eight K"); digits are fine on screen. Avoid spoken symbols and single letters where possible ("alpha", "gamma" are fine as words).
{PRONUNCIATION}
- ManimGX: kit.py already fixes TransformFromCopy/FadeTransform early-show and parts animated after their group's intro in the same line. Still: never Indicate/Wiggle on a LATER cue an object whose intro may still be running (use Circumscribe); probe any doubtful line with inspect -t just before/after its cue. Within one line, fade out the SAME group object you faded in (gotcha 24). Chained .animate.shift(v) in one Succession land wrong: use absolute move_to (gotcha 25). Never name a scene method voice, film or say. Axes ranges are cut to whole steps (gotcha 27).
- Text: Text() strips leading spaces of a single line (indent by position; multi-line strings keep indentation). Helvetica Neue and DIN merge "fi"/"fl" into one glyph: glyph-indexing helpers (glyphs(), paint()) are for mono text only. code() / code_block / json_card / terminal colour code by regex already.
- Margins: keep every object ≥ 0.5 units inside the frame edges (the layout check only catches cut-offs, not text flush against the edge), and make sure cards placed late never cover earlier arrows or labels still on screen.
"""


def course_rows():
    rows = {}
    for line in (SERIES_DIR / "COURSE.md").read_text().splitlines():
        cells = [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]
        if len(cells) == 7 and cells[0].isdigit():
            rows[int(cells[0])] = dict(title=cells[1].strip("*"), goal=cells[2], concepts=cells[3], mechanism=cells[4],
                                       example=cells[5])
    return rows


def module_of(n):
    src = (SERIES_DIR / "kit.py").read_text()
    return {int(a): int(c) for a, b, c in re.findall(r"(\d+): \(\"(.+?)\", (\d)\)", src)}[n]


def brief(n, rows):
    slug, cls, refs, sections, notes = SPEC[n]
    row, module = rows[n], module_of(n)
    pron = (HERE / "pronunciation.txt").read_text().strip() if (HERE / "pronunciation.txt").exists() else ""
    fill = {
        "{N}": str(n), "{NN}": f"{n:02d}", "{TITLE}": row["title"], "{M}": str(module),
        "{MODULE_NAME}": MODULE_NAMES[module], "{SERIES_DIR}": str(SERIES_DIR), "{slug}": slug,
        "{ClassName}": cls, "{GOAL}": row["goal"], "{CONCEPTS}": row["concepts"],
        "{MECHANISM}": row["mechanism"], "{EXAMPLE}": row["example"], "{NOTES}": notes,
        "{NOTES_SECTIONS}": sections, "{REFERENCE_EPISODES}": references(refs), "{NARRATOR_PYTHON}": NARRATOR_PYTHON,
    }
    text = TEMPLATE
    for key, value in fill.items():
        text = text.replace(key, value)
    leftover = re.findall(r"\{(?:[A-Z_]+|slug|ClassName)\}", text)
    assert not leftover, (n, leftover)
    return text.rstrip() + "\n" + COMMON.replace("{PRONUNCIATION}", pron)


def main():
    rows = course_rows()
    for n in [int(a) for a in sys.argv[1:]] or sorted(SPEC):
        path = HERE / f"ep{n:02d}.md"
        path.write_text(brief(n, rows))
        print(path.name, len(path.read_text().split()), "words")


if __name__ == "__main__":
    main()
