"""Writes assets/ for "Tokens per Second": the backdrop (deep blue-green, a soft vignette, a fine grid) and
the base icons — the "Agents, Animated" set recoloured into this palette, plus gpu, mac, server and ram.
Run once: python3 make_assets.py (needs numpy and Pillow)."""
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
SOURCE = Path.home() / "videos" / "agent-series" / "assets"
ASSETS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------- backdrop
W, H = 1920, 1080
y, x = np.mgrid[0:H, 0:W].astype(np.float32)
base = np.array([13, 20, 23], np.float32)            # #0D1417
lift = np.array([24, 42, 47], np.float32)            # a cool centre lift
r = np.sqrt(((x - W * 0.5) / (W * 0.62)) ** 2 + ((y - H * 0.45) / (H * 0.70)) ** 2)
t = np.clip(1 - r, 0, 1) ** 2
img = base + (lift - base) * t[..., None] * 0.55
img *= (1 - 0.35 * np.clip(r - 0.75, 0, 1))[..., None]
step = 1920 / 14.222 * 0.5                           # a line every half unit
minor = (np.abs((x + 0.5) % step) < 1.0) | (np.abs((y + 0.5) % step) < 1.0)
major = (np.abs((x + 0.5) % (step * 4)) < 1.2) | (np.abs((y + 0.5) % (step * 4)) < 1.2)
img += minor[..., None] * np.array([5, 8, 9], np.float32) * (0.6 + 0.4 * t[..., None])
img += major[..., None] * np.array([4, 7, 8], np.float32) * (0.6 + 0.4 * t[..., None])
Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(ASSETS / "backdrop.png")

# ---------------------------------------------------------------------------- recoloured base icons
REMAP = {
    "#FF5C7A": "#FF6464", "#3A1520": "#3D1717",                                   # rose → red
    "#3DDC97": "#5EE08A", "#12301F": "#123320", "#1B6B47": "#1E6B3E",             # mint → green
    "#8EF0C4": "#A6F2C0", "#1B5A44": "#1B5536",
    "#FFC857": "#FFB547", "#3A2E14": "#3A2A10", "#B7791F": "#B9781C", "#FFD98A": "#FFD493",  # sun → amber
    "#5AB8FF": "#5B9DFF", "#132838": "#13274A", "#D6ECFF": "#D3E4FF", "#4AA8F0": "#4A8CF0",  # azure → blue
    "#C792EA": "#A78BFA", "#2C1F38": "#261D40", "#E7C4F5": "#DCD0FE", "#EEDDFB": "#ECE6FE",  # orchid → violet
    "#261B30": "#211A38",
    "#FF7A59": "#FF7AB8",                                                         # coral → pink (requests)
    "#7A7166": "#6E858C", "#A9A094": "#91A7AE", "#DDD5C9": "#D3E1E4",            # warm greys → cool greys
    "#1E1B17": "#132026", "#14120F": "#0D1417",
}
for svg in sorted(SOURCE.glob("*.svg")):
    if re.match(r"ep\d", svg.name):
        continue
    out = re.sub(r"#[0-9A-Fa-f]{6}", lambda mt: REMAP.get(mt.group(0).upper(), mt.group(0)), svg.read_text())
    left = {c.upper() for c in re.findall(r"#[0-9A-Fa-f]{6}", out)} - {v.upper() for v in REMAP.values()}
    assert not left, (svg.name, left)
    (ASSETS / svg.name).write_text(out)

# ---------------------------------------------------------------------------- new icons (flat, 2-tone, blue base)
B, BF, LT, GOOD = "#5B9DFF", "#13274A", "#D3E4FF", "#5EE08A"
NEW = {
    "gpu": f'''<rect x="8" y="24" width="84" height="54" rx="7" fill="{BF}" stroke="{B}" stroke-width="5"/>
  <rect x="38" y="37" width="24" height="28" rx="3" fill="{B}"/>
  <rect x="19" y="37" width="11" height="28" rx="2" fill="{LT}"/><rect x="70" y="37" width="11" height="28" rx="2" fill="{LT}"/>
  {"".join(f'<rect x="{16 + 9 * i}" y="80" width="5" height="9" rx="1" fill="{B}"/>' for i in range(8))}''',
    "mac": f'''<rect x="14" y="16" width="72" height="48" rx="5" fill="{BF}" stroke="{B}" stroke-width="5"/>
  <rect x="22" y="24" width="56" height="32" rx="2" fill="{LT}" fill-opacity="0.25"/>
  <path d="M6 70 H94 L88 82 H12 Z" fill="{B}"/>''',
    "server": "".join(f'''<rect x="14" y="{12 + 27 * i}" width="72" height="21" rx="4" fill="{BF}" stroke="{B}" stroke-width="5"/>
  <circle cx="74" cy="{22.5 + 27 * i}" r="3.5" fill="{GOOD}"/><rect x="24" y="{20 + 27 * i}" width="30" height="5" rx="2" fill="{B}"/>''' for i in range(3)),
    "ram": f'''<rect x="6" y="28" width="88" height="38" rx="4" fill="{BF}" stroke="{B}" stroke-width="5"/>
  {"".join(f'<rect x="{15 + 19 * i}" y="36" width="13" height="18" rx="2" fill="{LT}"/>' for i in range(4))}
  {"".join(f'<rect x="{12 + 7 * i}" y="70" width="4" height="9" fill="{B}"/>' for i in range(11))}''',
}
for name, body in NEW.items():
    (ASSETS / f"{name}.svg").write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">\n  {body}\n</svg>\n')

print(f"wrote backdrop.png and {len(list(ASSETS.glob('*.svg')))} icons in {ASSETS}", file=sys.stderr)
