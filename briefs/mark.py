"""Set the status cell of COURSE.md rows: python3 briefs/mark.py STATUS N [N ...]
STATUS: 'run' (🎬) or 'done' (✅ `folder`)."""
import re, sys
from pathlib import Path
course = Path(__file__).resolve().parent.parent / "COURSE.md"
status, numbers = sys.argv[1], {int(a) for a in sys.argv[2:]}
lines = course.read_text().splitlines()
for i, line in enumerate(lines):
    cells = re.split(r"(?<!\\)\|", line)
    if len(cells) == 9 and cells[1].strip().isdigit() and int(cells[1]) in numbers:
        n = int(cells[1])
        if status == "run":
            cells[7] = " 🎬 "
        else:
            folder = next(p.name for p in course.parent.glob(f"ep{n:02d}-*") if p.is_dir())
            cells[7] = f" ✅ `{folder}` "
            cells[2] = f" **{cells[2].strip().strip('*')}** "
        lines[i] = "|".join(cells)
course.write_text("\n".join(lines) + "\n")
