"""Write README.md and index.html for the series: every episode with its files and length."""
import html, re, subprocess
from pathlib import Path
SERIES = Path(__file__).resolve().parent.parent
import sys; sys.path.insert(0, str(SERIES))
LEVEL_NAMES = {1: "One Token, Two Speed Limits", 2: "The KV Cache", 3: "Many Users at Once", 4: "Faster Tokens", 5: "Fewer Bits, More GPUs", 6: "Production & Your Mac"}
COLORS = {1: "#FFB547", 2: "#A78BFA", 3: "#FF7AB8", 4: "#5EE08A", 5: "#5B9DFF", 6: "#3FD5E6"}
src = (SERIES / "kit.py").read_text()
episodes = {int(n): (t, int(l)) for n, t, l in re.findall(r"(\d+): \(\"(.+?)\", (\d)\)", src)}
held = {int(m) for m in re.findall(r"^\| (\d+) \|.*⏸", (SERIES / "COURSE.md").read_text(), re.M)}
rows, total = [], 0.0
for n, (title, level) in sorted(episodes.items()):
    d = next((p for p in SERIES.glob(f"ep{n:02d}-*") if p.is_dir()), None)
    mp4 = next(d.glob("*.mp4"), None) if d else None
    mp4 = next((p for p in d.glob("Ep*.mp4")), None) if d else None
    secs = None
    if mp4:
        out = subprocess.run(["afinfo", str(mp4)], capture_output=True, text=True).stdout
        m = re.search(r"estimated duration: ([\d.]+)", out)
        secs = float(m.group(1)) if m else None
        total += secs or 0
    rows.append((n, title, level, d, mp4, secs))
def mmss(s):
    if not s: return "—"
    s = int(round(s)); return f"{s // 60}:{s % 60:02d}"
# README
lines = ["# Tokens per Second — the course", "",
         f"{sum(1 for r in rows if r[4])} of {len(rows)} episodes · {mmss(total)} in total ({total / 3600:.1f} hours) · 1080p60, narrated, captioned, with HTML slides.", "",
         "Open `index.html` to browse. Curriculum and goals: `COURSE.md`. Lines worth a human listen: `LISTEN.md`.", ""]
for level in range(1, 7):
    lines += [f"## Module {level} · {LEVEL_NAMES[level]}", "", "| # | Episode | Length | Video | Slides |", "|---|---|---|---|---|"]
    for n, title, lvl, d, mp4, secs in rows:
        if lvl != level: continue
        if mp4:
            lines.append(f"| {n} | {title} | {mmss(secs)} | [{mp4.name}]({d.name}/{mp4.name}) | [slides]({d.name}/{mp4.with_suffix('.html').name}) |")
        else:
            lines.append(f"| {n} | {title} | — | {'on hold' if n in held else 'not produced yet'} | |")
    lines.append("")
lines += ["## Editing an episode", "", "Each folder has `scene.py` (imports `../kit.py`) and `voice/` (cached narration). Start the narrator",
          "(`~/.claude/skills/tutorial-video/scripts/narrator.sh` from this folder), edit, then `manimgx render scene.py`",
          "and `manimgx present scene.py --no-open`: only changed lines are spoken again.", ""]
(SERIES / "README.md").write_text("\n".join(lines))
# index.html
cards = []
for level in range(1, 7):
    items = "".join(
        f'<li><span class="n">{n:02d}</span><a href="{html.escape(d.name)}/{html.escape(mp4.with_suffix(".html").name)}">{html.escape(title)}</a>'
        f'<span class="t">{mmss(secs)}</span><a class="v" href="{html.escape(d.name)}/{html.escape(mp4.name)}">video</a></li>'
        if mp4 else f'<li><span class="n">{n:02d}</span>{html.escape(title)}<span class="t">{"on hold" if n in held else "—"}</span></li>'
        for n, title, lvl, d, mp4, secs in rows if lvl == level)
    cards.append(f'<section style="--c:{COLORS[level]}"><h2>Module {level} · {LEVEL_NAMES[level]}</h2><ol>{items}</ol></section>')
page = f"""<!doctype html><meta charset="utf-8"><title>Tokens per Second</title>
<style>body{{background:#0D1417;color:#E6F0F2;font:16px/1.5 "Helvetica Neue",system-ui,sans-serif;max-width:900px;margin:40px auto;padding:0 20px}}
h1{{background:linear-gradient(90deg,#3FD5E6,#FFB547);-webkit-background-clip:text;color:transparent;margin-bottom:4px}}
.sub{{color:#91A7AE;margin-top:0}} section{{border-left:3px solid var(--c);padding-left:16px;margin:28px 0}}
h2{{color:var(--c);font-size:18px;letter-spacing:.04em}} ol{{list-style:none;padding:0}} li{{display:flex;gap:12px;padding:6px 0;border-bottom:1px solid #1f3038}}
.n{{color:#6E858C;font-family:"Source Code Pro",monospace}} a{{color:#E6F0F2;text-decoration:none;flex:1}} a:hover{{color:var(--c)}}
.t{{color:#91A7AE;font-family:"Source Code Pro",monospace}} a.v{{flex:0;color:#6E858C}}</style>
<h1>Tokens per Second</h1><p class="sub">{sum(1 for r in rows if r[4])} narrated episodes · {mmss(total)} · how LLM inference really works, from one token to a fleet. Titles open the slides; "video" opens the MP4.</p>
{''.join(cards)}"""
(SERIES / "index.html").write_text(page)
print(f"{sum(1 for r in rows if r[4])} episodes, {mmss(total)} total")
