"""Tokens per Second · Episode 3 · Prefill vs Decode (Module 1 · One Token, Two Speed Limits)

The same model runs two different workloads. Prefill pushes the whole prompt through each weight it reads
(arithmetic intensity ≈ prompt length: compute-bound); decode pushes one token through every weight it
reads (intensity ≈ batch size: memory-bound). Two dots slide apart on this Mac's roofline, both ceilings
are checked against measurements, TTFT and TPOT are tied to their limits, and the M5 Max (more compute,
little more bandwidth) speeds prefill ×3.4 but decode only ×1.17.

Running example: Owl-8B (Llama-3.1-8B, BF16, 16 GB of weights) measured on this Mac Studio (M4 Max,
546 GB/s) with mlx-lm. It teaches better than Owl-mini here: at 16 bits a weight is 2 bytes and does
2 FLOPs per token, so intensity = tokens exactly, and both of its ceilings check out at 91–92%. Owl-mini
(Llama-3.2-1B 4-bit) appears as the same split measured on the small model.

Sources (facts dated October 2026):
- research/measurements.md (this Mac, mlx-lm 0.32.2, 2026-10-24), loaded with measured(): decode 31.0 tok/s
  (§1), prefill 1,005 tok/s at 2,048 tokens and TTFT per prompt length (§2), decode per context length
  (§3), bf16 matmul 15.2 TFLOPS (§0, a self-measured roof: Apple publishes no GPU FLOPS), 4-bit rows.
  Prefill FLOPs per token = 2 × 6.98 B block parameters ≈ 14 GFLOP (the embedding is a lookup and the
  LM head runs only for the last token), not 2P = 16 (measurements.md §2).
- research/notes.md §1 (TTFT / TPOT, Databricks 2023), §2 (intensity ≈ B for decode, ≈ n for prefill; the
  llama.cpp Apple Silicon table, github.com/ggml-org/llama.cpp/discussions/4167, LLaMA-2-7B F16:
  M4 Max PP512 922.83 / TG128 31.64 tok/s, M5 Max 3158.49 / 37.11, re-read 2026-10-24; Apple ML Research,
  "Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU", 2025-11-19), §14.
- Apple newsroom 2026-03-03 ("Apple debuts M5 Pro and M5 Max …"): M5 Max "up to 614GB/s", a GPU with "a Neural
  Accelerator in each GPU core" (re-read 2026-10-24). Apple ML Research 2025-11-19 quotes re-read 2026-10-24.

Illustrative (said "for example" or shown as schematic): the student's message and the flashcard tokens,
the 64-square prompt grid (one square = 32 tokens), the 7 drawn prompt tokens in the intensity panels,
the 200-token answer on the timeline, and the M5 roof's height (drawn ×3.4 from the prefill speedup,
because Apple publishes no TFLOPS). Asserts below tie every number on screen and in the narration to data.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from owl_reference import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- the numbers

W_BYTES = weight_bytes()                                       # Owl-8B, BF16
BW = MAC["bandwidth"]                                          # 546 GB/s (Apple spec)
PEAK = measured("roofs", "matmul_bf16_TFLOPS", "median") * 1e12  # 15.2 TFLOP/s, measured (no Apple spec)
RIDGE = PEAK / BW
assert round(W_BYTES / 1e9) == 16 and BW == 546e9                    # "sixteen gigabytes"
assert f"{PEAK / 1e12:.1f}" == "15.2" and round(PEAK / 1e12) == 15    # "about fifteen teraflops"
assert round(RIDGE) == 28                                            # "near twenty-eight FLOPs per byte"
assert abs(RIDGE - measured("roofs", "ridge_flops_per_byte_spec_bw")) < 1e-9

M8 = measured("models", "llama-3.1-8b", "bf16")
assert M8["params_total"] == 8_030_261_248
BLOCK_PARAMS = M8["params_matmul_excl_embed_head"]               # 6.98 B: no embedding table, no LM head
PREFILL_FLOPS = 2 * BLOCK_PARAMS                                 # per prompt token
assert round(PREFILL_FLOPS / 1e9) == 14 and round(flops_per_token() / 1e9) == 16  # "fourteen, not sixteen"

DEC_TOK = measured("decode", "llama-3.1-8b", "bf16", "tok_s_median")
DEC_CEIL = decode_ceiling(hw=MAC)
DEC_FRAC = DEC_TOK / DEC_CEIL
assert round(DEC_TOK) == 31 and f"{DEC_TOK:.1f}" == "31.0" and round(DEC_CEIL) == 34  # "thirty-one", "thirty-four"
assert round(100 * DEC_FRAC) == 91                                   # "ninety-one percent"

PROMPT = 2048
PRE_BY = measured("prefill", "llama-3.1-8b", "bf16", "by_prompt_tokens")
PRE_TOK = PRE_BY[str(PROMPT)]["tok_s_median"]
PRE_CEIL = PEAK / PREFILL_FLOPS
PRE_FRAC = PRE_TOK / PRE_CEIL
assert round(PRE_TOK) == 1005 and round(PRE_CEIL, -1) == 1090        # "one thousand and five", "about one thousand and ninety"
assert round(100 * PRE_FRAC) == 92                                   # "ninety-two percent"
RATIO_MEAS = PRE_TOK / DEC_TOK
RATIO_PRED = PRE_CEIL / DEC_CEIL
assert round(RATIO_MEAS) == 32 == round(RATIO_PRED)                  # "predicts thirty-two ... measured thirty-two"
assert abs(RATIO_MEAS - measured("headline", "prefill_over_decode_8b_bf16")) < 1e-9

# Owl-mini (Llama-3.2-1B, 4-bit), the same split
MINI_PRE = measured("prefill", "llama-3.2-1b", "4bit", "by_prompt_tokens", str(PROMPT), "tok_s_median")
MINI_DEC = measured("decode", "llama-3.2-1b", "4bit", "tok_s_median")
MINI_RATIO = MINI_PRE / MINI_DEC
assert round(MINI_PRE) == 6624 and round(MINI_DEC) == 489 and f"{MINI_RATIO:.1f}" == "13.6" and MINI_RATIO > 13  # "more than thirteen"

# Time to first token and time per output token (Owl-8B BF16, this Mac)
TTFT = {n: PRE_BY[str(n)]["ttft_s_median"] for n in (128, PROMPT, 16384)}
CTX = measured("context", "llama-3.1-8b", "bf16", "by_context_tokens")
TPOT_MS = {n: 1000 / CTX[str(n)]["tok_s_median"] for n in (128, PROMPT, 16384)}
ANSWER = 200                                                     # tokens in the answer (for example)
assert round(TTFT[PROMPT]) == 2 and round(TPOT_MS[PROMPT]) == 33     # "two seconds", "every thirty-three milliseconds"
assert round(TTFT[128], 1) == 0.2 and round(TTFT[16384]) == 20       # "a fifth of a second", "twenty seconds"
assert round(TTFT[16384] / TTFT[128], -1) == 100                     # "about a hundred times longer" (prompt ×128)
TPOT_RISE = TPOT_MS[16384] / TPOT_MS[128] - 1
assert round(100 * TPOT_RISE) == 14                                  # "only fourteen percent"

# The M5 Max vs the M4 Max: llama.cpp discussion #4167, LLaMA-2-7B F16 (cited, not computed here)
LCPP = {"M4 Max": dict(bw=546, pp=922.83, tg=31.64), "M5 Max": dict(bw=614, pp=3158.49, tg=37.11)}
PP_X = LCPP["M5 Max"]["pp"] / LCPP["M4 Max"]["pp"]
TG_X = LCPP["M5 Max"]["tg"] / LCPP["M4 Max"]["tg"]
BW_X = LCPP["M5 Max"]["bw"] / LCPP["M4 Max"]["bw"]
assert f"{PP_X:.1f}" == "3.4" and f"{TG_X:.2f}" == "1.17" and round(100 * (BW_X - 1)) == 12  # "three point four", "one point one seven", "twelve percent"

# The mirror image, measured here: 4-bit weights (fewer bytes) vs BF16
Q4_DEC = measured("decode", "llama-3.1-8b", "4bit", "tok_s_median")
Q4_PRE = measured("prefill", "llama-3.1-8b", "4bit", "by_prompt_tokens", str(PROMPT), "tok_s_median")
Q4_GB = measured("models", "llama-3.1-8b", "4bit", "weights_gb")
BYTES_X = M8["weights_gb"] / Q4_GB
Q4_DEC_X = Q4_DEC / DEC_TOK
Q4_PRE_X = Q4_PRE / PRE_TOK
assert f"{BYTES_X:.1f}" == "3.6" and 3.5 <= BYTES_X < 3.6       # "three and a half times fewer bytes"
assert f"{Q4_DEC_X:.1f}" == "3.3" and f"{Q4_PRE_X:.2f}" == "0.94"  # "three point three times faster", "slightly slower"

# ---------------------------------------------------------------------------- look

PREFIX = "ep03"
SRC_MAC = "measured on this Mac · Owl-8B BF16 · mlx-lm"


def cite(s, size=20):
    return text(s, size, MUTED)


def fmt(x):
    return f"{x:,.0f}"


def mac_roofline():
    """This Mac's roofline: Apple's 546 GB/s slope, the measured 15.2 TFLOP/s roof."""
    roof = Roofline(BW, PEAK, width=6.0, height=4.0, x_decades=(-1, 4), y_decades=(10, 14), center=(-2.5, -0.45, 0))
    ylab = code("attainable FLOP/s", 20, MUTED).next_to(roof.axes[1].get_end(), m.RIGHT, buff=0.2)
    ylab.align_to(roof.axes[1].get_end(), m.UP)
    return roof, ylab


def roof_path(roof, a, b):
    """The path along the roof from intensity a to b (it turns the corner at the ridge)."""
    pts = [roof.point(a)]
    if a < roof.ridge_x < b:
        pts.append(roof.point(roof.ridge_x))
    pts.append(roof.point(b))
    return m.VMobject().set_points_as_corners(pts)


def pre_label(roof, n):
    words = code(f"prefill · {n:,}", 20, COMPUTE, weight=m.BOLD)
    at = roof.point(n)
    if n < roof.ridge_x:
        return words.next_to(at, m.UL, buff=0.16)
    return words.next_to(at, m.UP, buff=0.24)


def square(color, fill, size=0.3):
    return m.RoundedRectangle(corner_radius=0.05, width=size, height=size, stroke_color=color, stroke_width=2,
                              fill_color=fill, fill_opacity=1)


def bar_pair(width, frac, color, fill, height=0.38):
    """A ceiling track (outlined) and a measured bar inside it (to animate with GrowFromEdge)."""
    track = m.RoundedRectangle(corner_radius=0.08, width=width, height=height, stroke_color=color, stroke_width=2,
                               fill_color=CARD, fill_opacity=1)
    bar = m.RoundedRectangle(corner_radius=0.08, width=width * frac, height=height, stroke_width=0,
                             fill_color=color, fill_opacity=0.9)
    bar.align_to(track, m.LEFT).set_y(track.get_y())
    return track, bar


def bracket(x0, x1, y, color, down=True):
    d = -0.12 if down else 0.12
    return m.VGroup(m.Line([x0, y, 0], [x1, y, 0]), m.Line([x0, y, 0], [x0, y + d, 0]),
                    m.Line([x1, y, 0], [x1, y + d, 0])).set_stroke(color, 2.5)


# ---------------------------------------------------------------------------- the episode


class Ep03PrefillVsDecode(Episode):
    number = 3

    def lesson(self):
        self.hook()
        self.intensity()
        self.roofline()
        self.checked()
        self.feel()
        self.new_chip()
        self.knobs()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section(
            "1 · Two workloads",
            notes=f"A student pastes a chapter (2,048 tokens; the message and the flashcard tokens are illustrative). "
                  f"Prefill reads the whole prompt in one pass; decode writes one token per step. Measured on this Mac "
                  f"(Owl-8B BF16, mlx-lm): prefill {PRE_TOK:,.1f} tok/s at {PROMPT} tokens vs decode {DEC_TOK:.2f} tok/s, "
                  f"×{RATIO_MEAS:.1f} (measurements.md §1–2). Episode 1 walked the whole request; this one asks why the two phases differ.")
        mascot = owl(0.62).move_to([-6.0, 1.85, 0])
        msg = message("user", "Here's chapter 3 of my biology notes.\nMake me flashcards!", width=4.9)
        msg.move_to([-2.95, 1.85, 0])
        grid = m.VGroup(*[square(REQUEST, PINK_FILL) for _ in range(64)])
        grid.arrange_in_grid(rows=4, cols=16, buff=0.08).move_to([-2.95, -0.25, 0])
        grid_lab = code(f"prompt · {PROMPT:,} tokens", 20, REQUEST).next_to(grid, m.UP, buff=0.16).align_to(grid, m.LEFT)
        pre_lab = m.VGroup(title_text("prefill", 30, COMPUTE), text("all at once", 20, MUTED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        pre_lab.move_to([0.45, grid.get_y(), 0], aligned_edge=m.LEFT)

        toks = token_row(["What", " do", " mito", "chond", "ria", " make", "?"], color=TOKEN, size=20)
        toks.move_to([-2.95, -1.95, 0]).align_to(grid, m.LEFT)
        dec_lab = m.VGroup(title_text("decode", 30, MEMORY), text("one token per step", 20, MUTED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        dec_lab.move_to([0.45, toks.get_y(), 0], aligned_edge=m.LEFT)

        weights = pill("Owl-8B · 16 GB of weights", WEIGHTS, BLUE_FILL, 22).move_to([4.6, 1.85, 0])
        pre_stat = stat(fmt(PRE_TOK), "tok/s", "", COMPUTE, 60)
        pre_stat.move_to([0, grid.get_y(), 0]).align_to([3.35, 0, 0], m.LEFT)
        dec_stat = stat(f"{DEC_TOK:.0f}", "tok/s", "", MEMORY, 60)
        dec_stat.move_to([0, toks.get_y(), 0]).align_to(pre_stat, m.LEFT)
        ratio = m.VGroup(title_text(f"×{RATIO_MEAS:.0f}", 44, INK), text("faster per token", 22, MUTED)).arrange(m.RIGHT, buff=0.25)
        ratio[1].align_to(ratio[0], m.DOWN)
        ratio.move_to([0, -3.05, 0]).align_to(pre_stat, m.LEFT)
        src = cite(SRC_MAC).move_to([0, -3.05, 0]).align_to(grid, m.LEFT)

        self.say(
            "[A student pastes a chapter of biology notes into Owl], [and asks for flashcards.]",
            m.AnimationGroup(self.new_header("01 · two jobs", "One model, two workloads"),
                             m.FadeIn(mascot, shift=0.2 * m.UP), m.FadeIn(msg, shift=0.2 * m.UP)),
            m.AnimationGroup(m.FadeIn(grid_lab), m.LaggedStart(*[m.FadeIn(c, scale=0.6) for c in grid], lag_ratio=0.015)),
        )
        self.say(
            "[Owl reads all two thousand and forty-eight prompt tokens in a single pass]: [that's prefill.]",
            grid.animate(run_time=0.7).set_fill(AMBER_FILL, 1).set_stroke(COMPUTE),
            m.FadeIn(pre_lab, shift=0.15 * m.RIGHT),
        )
        self.say(
            "[Then it writes the answer one token at a time]: [that's decode.]",
            m.LaggedStart(*[m.FadeIn(t, shift=0.12 * m.UP) for t in toks], lag_ratio=0.35, run_time=2.6),
            m.FadeIn(dec_lab, shift=0.15 * m.RIGHT),
        )
        self.say(
            "[Same model, same sixteen gigabytes of weights.] [Yet on this Mac], [prefill handles thirty-two times more tokens per second.]",
            m.FadeIn(weights, shift=0.15 * m.DOWN),
            m.AnimationGroup(m.FadeIn(pre_stat, shift=0.15 * m.LEFT), m.FadeIn(dec_stat, shift=0.15 * m.LEFT), m.FadeIn(src)),
            m.FadeIn(ratio, shift=0.15 * m.UP),
        )
        self.say(
            "[Today: why one model runs two such different workloads], [each with its own speed limit.]",
            m.Circumscribe(m.VGroup(grid, pre_stat), color=COMPUTE, buff=0.12),
            m.Circumscribe(m.VGroup(toks, dec_stat), color=MEMORY, buff=0.12),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ intensity

    def intensity(self):
        self.next_section(
            "2 · Intensity",
            notes="Each forward pass reads every weight once (2 bytes at 16 bits). Decode multiplies it with one token: "
                  "2 FLOPs for 2 bytes, intensity 1. Prefill multiplies it with every prompt token: 2 × 2,048 FLOPs for the "
                  "same 2 bytes, intensity ≈ 2,048. So prefill intensity ≈ prompt length, decode intensity ≈ batch size "
                  "(notes §2). Seven drawn tokens stand for the 2,048.")
        panels = {}
        for key, x, color, n_tok in (("decode", -3.4, MEMORY, 1), ("prefill", 3.4, COMPUTE, 7)):
            p = {}
            p["box"] = card(6.2, 4.6, color, CARD, radius=0.2, stroke=2.5).move_to([x, -0.3, 0])
            top = p["box"].get_top()[1]
            sub = "1 token" if key == "decode" else f"{PROMPT:,} tokens"
            p["title"] = m.VGroup(title_text(key, 30, color), code(f"· {sub}", 22, MUTED)).arrange(m.RIGHT, buff=0.2)
            p["title"].move_to([x, top - 0.38, 0])
            cells = m.VGroup(*[square(WEIGHTS, BLUE_FILL, 0.32) for _ in range(7)]).arrange(m.RIGHT, buff=0.08)
            cells.move_to([x + 0.5, top - 1.05, 0])
            mem = code("memory", 20, MUTED).next_to(cells, m.LEFT, buff=0.3)
            p["mem"] = m.VGroup(mem, cells)
            src_cell = cells[3]
            land = np.array([src_cell.get_x(), -0.62, 0])
            p["arrow"] = arrow(src_cell.get_bottom() + 0.04 * m.DOWN, land + 0.28 * m.UP, MEMORY, buff=0.02)
            p["bytes"] = m.VGroup(code("read once", 20, MEMORY), code("2 bytes", 20, MEMORY)).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
            p["bytes"].next_to(p["arrow"], m.RIGHT, buff=0.22)
            p["w"] = square(WEIGHTS, BLUE_FILL, 0.32).move_to(src_cell).set_z_index(2)
            p["land"] = land
            tok_color, tok_fill = (TOKEN, GREEN_FILL) if key == "decode" else (REQUEST, PINK_FILL)
            xs = [src_cell.get_x()] if n_tok == 1 else list(np.linspace(src_cell.get_x() - 2.1, src_cell.get_x() + 2.1, n_tok))
            p["toks"] = m.VGroup(*[square(tok_color, tok_fill, 0.34).move_to([tx, -1.62, 0]) for tx in xs])
            p["uses"] = m.VGroup(*[m.Line(land + 0.2 * m.DOWN, t.get_top() + 0.04 * m.UP).set_stroke(COMPUTE, 3) for t in p["toks"]])
            if key == "decode":
                s, res = "2 FLOPs ÷ 2 bytes = ", "1"
            else:
                s, res = f"{2 * PROMPT:,} FLOPs ÷ 2 bytes = ", f"{PROMPT:,}"
            p["formula"] = m.VGroup(code(s, 22, INK), title_text(res, 34, color)).arrange(m.RIGHT, buff=0.08)
            p["formula"][1].align_to(p["formula"][0], m.DOWN)
            p["formula"].move_to([x, -2.25, 0])
            cap = "intensity ≈ batch size" if key == "decode" else "intensity ≈ prompt length"
            p["cap"] = text(cap, 26, color, weight=m.BOLD).move_to([x, -3.15, 0])
            panels[key] = p
        d, pr = panels["decode"], panels["prefill"]
        intensity_def = code("arithmetic intensity = FLOPs ÷ bytes", 20, MUTED)
        intensity_def.move_to([0, 2.42, 0]).align_to([6.7, 0, 0], m.RIGHT)

        self.say(
            "[Here's the key: count how many times each weight is used], [once it has been read.]",
            m.AnimationGroup(self.new_header("02 · intensity", "One weight, many uses"),
                             *[m.FadeIn(p[k]) for p in (d, pr) for k in ("box", "title")]),
            m.AnimationGroup(m.FadeIn(d["mem"], shift=0.1 * m.DOWN), m.FadeIn(pr["mem"], shift=0.1 * m.DOWN)),
        )
        self.say(
            "[Every pass reads each weight from memory once]: [two bytes, for sixteen-bit weights.]",
            m.AnimationGroup(*[m.AnimationGroup(m.GrowArrow(p["arrow"]), m.Succession(
                m.FadeIn(p["w"], run_time=0.3), p["w"].animate(run_time=1.0).move_to(p["land"]))) for p in (d, pr)]),
            m.AnimationGroup(m.FadeIn(d["bytes"]), m.FadeIn(pr["bytes"])),
        )
        self.say(
            "[In decode, that weight meets just one token]: [one multiply and one add, two FLOPs for two bytes.] [An intensity of one.]",
            m.AnimationGroup(m.FadeIn(d["toks"], shift=0.1 * m.UP), m.Create(d["uses"])),
            m.FadeIn(d["formula"][0]),
            m.FadeIn(d["formula"][1], scale=0.7),
        )
        self.say(
            "[In prefill, it meets the whole prompt at once]: [two thousand and forty-eight tokens], [so every byte read does two thousand times the work.]",
            m.LaggedStart(*[m.FadeIn(t, shift=0.1 * m.UP) for t in pr["toks"]], lag_ratio=0.12),
            m.LaggedStart(*[m.Create(u) for u in pr["uses"]], lag_ratio=0.12),
            m.AnimationGroup(m.FadeIn(pr["formula"][0]), m.FadeIn(pr["formula"][1], scale=0.7)),
        )
        self.say(
            "[That ratio is the arithmetic intensity.] [For prefill, it's about the prompt length]; [for decode, about the batch size: one, for a single student.]",
            m.AnimationGroup(m.FadeIn(intensity_def), m.Circumscribe(d["formula"], color=MEMORY, buff=0.1),
                             m.Circumscribe(pr["formula"], color=COMPUTE, buff=0.1)),
            m.FadeIn(pr["cap"], shift=0.12 * m.UP),
            m.FadeIn(d["cap"], shift=0.12 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the roofline

    def roofline(self):
        self.next_section(
            "3 · Two dots on the roofline",
            notes=f"This Mac's roofline: slope = 546 GB/s (Apple spec), roof = {PEAK / 1e12:.1f} TFLOP/s (bf16 matmul measured "
                  f"here; Apple publishes no GPU FLOPS), ridge = {RIDGE:.1f} FLOPs/byte (measurements.md §0). Both phases start "
                  f"at intensity 1; the prefill dot slides with the prompt (8 → 128 → 2,048 tokens) past the ridge onto the roof: "
                  f"compute-bound. Decode stays at intensity ≈ batch = 1: memory-bound. Batching moves it right: Episode 4. "
                  f"(The H100's ridge is ≈ 295, Episode 2.)")
        roof, ylab = mac_roofline()
        ridge_lab = code(f"ridge ≈ {RIDGE:.0f} FLOPs/byte", 20, MUTED)
        ridge_lab.move_to(roof.at(roof.ridge_x, 10 ** 10.55)).align_to(roof.ridge, m.LEFT).shift(0.15 * m.RIGHT)
        mem_zone = text("memory-bound", 22, MEMORY, weight=m.BOLD).move_to(roof.at(2.0, 10 ** 10.32))
        cmp_zone = text("compute-bound", 22, COMPUTE, weight=m.BOLD).move_to(roof.at(700, 10 ** 12.35))

        d_dot = roof.dot(1, MEMORY, radius=0.11)
        p_dot = roof.dot(1, COMPUTE, radius=0.13).set_z_index(4)
        p_lab = code("prefill", 20, COMPUTE, weight=m.BOLD).next_to(p_dot, m.DR, buff=0.06)
        d_lab = code("decode", 20, MEMORY, weight=m.BOLD).next_to(p_lab, m.DOWN, buff=0.1).align_to(p_lab, m.LEFT)
        steps = [8, 128, PROMPT]
        p_labs = [pre_label(roof, n) for n in steps]

        def side(name, rule, value, color, y):
            box = card(4.2, 2.25, color, CARD, radius=0.18, stroke=2).move_to([4.4, y, 0])
            head = title_text(name, 32, color)
            sub = text(rule, 22, MUTED)
            words = m.VGroup(head, sub).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
            words.align_to(box, m.UL).shift(0.3 * m.RIGHT + 0.25 * m.DOWN)
            val = code(value, 22, INK).align_to(box, m.LEFT).shift(0.3 * m.RIGHT)
            val.set_y(box.get_bottom()[1] + 0.42)
            return box, words, val

        p_box, p_words, p_val = side("prefill", "intensity ≈ prompt length", "1-token prompt", COMPUTE, 1.0)
        d_box, d_words, d_val = side("decode", "intensity ≈ batch size", "batch of 1", MEMORY, -1.55)
        p_vals = [code(f"{n:,}-token prompt", 22, INK).move_to(p_val, aligned_edge=m.LEFT) for n in steps]
        ep4 = pill("EP 04", MEMORY, CYAN_FILL, 20, mono=True)
        ep4.next_to(d_val, m.RIGHT, buff=0.3)
        src = m.VGroup(cite("slope: 546 GB/s (Apple spec)"), cite("roof: 15.2 TFLOP/s (measured here)"))
        src.arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT).next_to(d_box, m.DOWN, buff=0.22).align_to(d_box, m.LEFT)
        ghost = m.DashedVMobject(m.Circle(radius=0.13).set_stroke(MEMORY, 2.5), num_dashes=10).move_to(roof.point(8))
        hop = m.ArcBetweenPoints(roof.point(1) + [-0.05, 0.2, 0], roof.point(8) + [-0.2, 0.12, 0], angle=-1.9)
        hop.set_stroke(MEMORY, 3).add_tip(tip_length=0.16, tip_width=0.16)

        self.say(
            "[Episode two drew the roofline.] [Here it is for this Mac]: [bandwidth is the cyan slope], [and the amber roof is about fifteen teraflops, measured.]",
            self.new_header("03 · the roofline", "Two dots slide apart"),
            m.AnimationGroup(m.FadeIn(roof.axes), m.FadeIn(roof.ticks), m.FadeIn(roof.x_label), m.FadeIn(ylab)),
            m.Create(roof.memory),
            m.AnimationGroup(m.Create(roof.compute), m.FadeIn(src)),
        )
        self.say(
            "[The ridge sits near twenty-eight FLOPs per byte]: [left of it, memory sets the speed]; [right of it, compute does.]",
            m.AnimationGroup(m.Create(roof.ridge), m.FadeIn(ridge_lab)),
            m.FadeIn(mem_zone, shift=0.1 * m.UP),
            m.FadeIn(cmp_zone, shift=0.1 * m.UP),
        )
        self.say(
            "[Both phases start at the same spot]: [one token, an intensity of one.]",
            m.AnimationGroup(m.FadeIn(d_dot, scale=0.4), m.FadeIn(p_dot, scale=0.4), m.FadeIn(d_lab), m.FadeIn(p_lab)),
            m.AnimationGroup(*[m.FadeIn(x, shift=0.15 * m.LEFT) for x in (p_box, p_words, p_val, d_box, d_words, d_val)]),
        )
        d_lab_home = d_lab.copy().next_to(d_dot, m.DR, buff=0.06)
        moves, a = [], 1
        for n, lab, val in zip(steps, p_labs, p_vals):
            extra = [m.Transform(d_lab, d_lab_home)] if a == 1 else []
            moves.append(m.AnimationGroup(m.MoveAlongPath(p_dot, roof_path(roof, a, n)), m.Transform(p_lab, lab),
                                          m.Transform(p_val, val), *extra, run_time=1.1))
            a = n
        self.say(
            "[Now grow the prompt]: [eight tokens], [a hundred and twenty-eight], [two thousand and forty-eight.]",
            m.Circumscribe(p_val, color=COMPUTE, buff=0.1),
            *moves,
        )
        self.say(
            "[Past the ridge, prefill hits the roof]: [it's compute-bound], [and the math units are the limit.]",
            pulse(roof.compute, COMPUTE, 12),
            m.Circumscribe(cmp_zone, color=COMPUTE, buff=0.1),
            m.Flash(p_dot, color=COMPUTE, flash_radius=0.35),
        )
        self.say(
            "[Decode stays down at one]: [memory-bound, waiting on bytes.]",
            m.Flash(d_dot, color=MEMORY, flash_radius=0.3),
            m.AnimationGroup(pulse(roof.memory, MEMORY, 12), m.Circumscribe(mem_zone, color=MEMORY, buff=0.1)),
        )
        self.say(
            "[A bigger batch would move decode to the right], [and that's episode four.]",
            m.AnimationGroup(m.Create(hop), m.FadeIn(ghost, scale=0.5)),
            m.FadeIn(ep4, scale=0.8),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ both ceilings, checked

    def checked(self):
        self.next_section(
            "4 · Both ceilings, checked",
            notes=f"Decode ceiling = 546 GB/s ÷ 16.06 GB = {DEC_CEIL:.1f} tok/s; measured {DEC_TOK:.2f} = {100 * DEC_FRAC:.0f}%. "
                  f"Prefill: FLOPs per prompt token = 2 × {BLOCK_PARAMS / 1e9:.2f} B block parameters = {PREFILL_FLOPS / 1e9:.2f} GFLOP "
                  f"(the embedding is a lookup; the LM head runs only for the last token, measurements.md §2), so the ceiling is "
                  f"{PEAK / 1e12:.2f} TFLOP/s ÷ {PREFILL_FLOPS / 1e9:.2f} GFLOP = {PRE_CEIL:.0f} tok/s; measured {PRE_TOK:.1f} at "
                  f"{PROMPT} tokens = {100 * PRE_FRAC:.0f}% (attention adds a little). Ratio predicted {RATIO_PRED:.1f}, measured "
                  f"{RATIO_MEAS:.1f}. Owl-mini (1B 4-bit): {MINI_PRE:.0f} vs {MINI_DEC:.1f} tok/s = ×{MINI_RATIO:.1f}.")
        cols = {}
        for key, x, color, fill in (("decode", -3.4, MEMORY, CYAN_FILL), ("prefill", 3.4, COMPUTE, AMBER_FILL)):
            c = {}
            c["box"] = card(6.2, 4.15, color, CARD, radius=0.2, stroke=2.5).move_to([x, 0.2, 0])
            left = c["box"].get_left()[0] + 0.35
            top = c["box"].get_top()[1]
            limit = "limited by bandwidth" if key == "decode" else "limited by compute"
            c["title"] = m.VGroup(title_text(key, 34, color), text(limit, 22, MUTED)).arrange(m.RIGHT, buff=0.3)
            c["title"][1].align_to(c["title"][0], m.DOWN)
            c["title"].move_to([0, top - 0.42, 0]).align_to([left, 0, 0], m.LEFT)
            cols[key] = c
            c["left"], c["top"] = left, top

        d, p = cols["decode"], cols["prefill"]

        def row(c, mob, dy):
            return mob.move_to([0, c["top"] - dy, 0]).align_to([c["left"], 0, 0], m.LEFT)

        d["per"] = row(d, code(f"{W_BYTES / 1e9:.2f} GB read per token", 22, INK), 1.0)
        d["note"] = row(d, text("every weight, every step", 20, MUTED), 1.42)
        p["per"] = row(p, code(f"2 × {BLOCK_PARAMS / 1e9:.2f} B params = {PREFILL_FLOPS / 1e9:.0f} GFLOP", 22, INK), 1.0)
        not16 = text("not 16:", 20, RED, weight=m.BOLD)
        n1 = text("the embedding is a lookup,", 20, MUTED)
        n2 = text("the output layer runs for the last token only", 20, MUTED)
        p["note"] = m.VGroup(not16, n1).arrange(m.RIGHT, buff=0.15)
        row(p, p["note"], 1.42)
        n2.next_to(p["note"], m.DOWN, buff=0.08).align_to(p["note"], m.LEFT)
        for c, s, val in ((d, "546 GB/s ÷ 16.06 GB", f"{DEC_CEIL:.0f}"),
                          (p, f"{PEAK / 1e12:.1f} TFLOP/s ÷ {PREFILL_FLOPS / 1e9:.0f} GFLOP", f"≈ {round(PRE_CEIL, -1):,.0f}")):
            f = row(c, code(s, 22, INK), 2.15)
            lab = text("ceiling", 22, MUTED)
            v = title_text(val, 44, MEMORY if c is d else COMPUTE)
            u = code("tok/s", 22, MUTED)
            grp = m.VGroup(lab, v, u).arrange(m.RIGHT, buff=0.2)
            lab.align_to(v, m.DOWN)
            u.align_to(v, m.DOWN)
            row(c, grp, 2.78)
            c["ceil"] = m.VGroup(f, grp)
        d["track"], d["bar"] = bar_pair(5.5, DEC_FRAC, MEMORY, CYAN_FILL)
        p["track"], p["bar"] = bar_pair(5.5, PRE_FRAC, COMPUTE, AMBER_FILL)
        for c in (d, p):
            row(c, c["track"], 3.42)
            c["bar"].align_to(c["track"], m.LEFT).set_y(c["track"].get_y())
        d["cap"] = code(f"measured {DEC_TOK:.1f} tok/s · {100 * DEC_FRAC:.0f}%", 20, INK)
        p["cap"] = code(f"measured {fmt(PRE_TOK)} tok/s · {100 * PRE_FRAC:.0f}%", 20, INK)
        for c in (d, p):
            c["cap"].next_to(c["track"], m.DOWN, buff=0.12).align_to(c["track"], m.LEFT)

        pred = m.VGroup(code("prefill ÷ decode:", 24, MUTED), code("predicted", 24, INK),
                        title_text(f"×{RATIO_PRED:.0f}", 40, INK))
        meas = m.VGroup(code("· measured", 24, INK), title_text(f"×{RATIO_MEAS:.0f}", 40, GREEN))
        ratio = m.VGroup(*pred, *meas).arrange(m.RIGHT, buff=0.22)
        for w in (pred[0], pred[1], meas[0]):
            w.align_to(pred[2], m.DOWN)
        ratio.move_to([0, -2.45, 0])
        mini = code(f"Owl-mini (1B, 4-bit): prefill {fmt(MINI_PRE)} · decode {MINI_DEC:.0f} tok/s → ×{MINI_RATIO:.1f}", 20, MUTED)
        mini.move_to([0, -3.15, 0])
        src = cite(SRC_MAC).move_to([0, 2.55, 0]).align_to([6.6, 0, 0], m.RIGHT)

        self.say(
            "[Now let's check both on this Mac.] [Decode first]: [bandwidth divided by sixteen gigabytes gives a ceiling of thirty-four tokens per second.]",
            m.AnimationGroup(self.new_header("04 · measured", "Both ceilings, checked"), m.FadeIn(src),
                             *[m.FadeIn(c[k]) for c in (d, p) for k in ("box", "title")]),
            m.AnimationGroup(m.FadeIn(d["per"], shift=0.1 * m.UP), m.FadeIn(d["note"])),
            m.AnimationGroup(m.FadeIn(d["ceil"], shift=0.1 * m.UP), m.FadeIn(d["track"])),
        )
        self.say(
            "[Measured: thirty-one tokens per second], [ninety-one percent of the ceiling.]",
            m.GrowFromEdge(d["bar"], m.LEFT, run_time=1.2),
            m.FadeIn(d["cap"], shift=0.1 * m.UP),
        )
        self.say(
            "[Prefill is limited by compute]: [about fourteen gigaflops per prompt token.]",
            m.Circumscribe(p["title"], color=COMPUTE, buff=0.1),
            m.FadeIn(p["per"], shift=0.1 * m.UP),
        )
        self.say(
            "[Not sixteen]: [the embedding is just a lookup], [and the output layer runs only for the last token.]",
            m.FadeIn(not16, scale=0.8),
            m.FadeIn(n1),
            m.FadeIn(n2),
        )
        self.say(
            "[Fifteen teraflops divided by fourteen gigaflops]: [about one thousand and ninety tokens per second.]",
            m.FadeIn(p["ceil"][0], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(p["ceil"][1:], shift=0.1 * m.UP), m.FadeIn(p["track"])),
        )
        self.say(
            "[Measured: one thousand and five], [ninety-two percent of the ceiling.]",
            m.GrowFromEdge(p["bar"], m.LEFT, run_time=1.2),
            m.FadeIn(p["cap"], shift=0.1 * m.UP),
        )
        self.say(
            "[So the roofline predicts prefill thirty-two times faster per token], [and we measured thirty-two.]",
            m.FadeIn(pred, shift=0.1 * m.UP),
            m.FadeIn(meas, shift=0.1 * m.UP),
        )
        self.say(
            "[And Owl-mini, our small four-bit model, shows the same split]: [more than thirteen times.]",
            m.FadeIn(mini, shift=0.1 * m.UP),
            m.Circumscribe(mini, color=MUTED, buff=0.1),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ what a user feels

    def feel(self):
        self.next_section(
            "5 · TTFT and TPOT",
            notes=f"Episode 1 defined TTFT and TPOT. TTFT is prefill (compute); TPOT is decode (bandwidth). Owl-8B BF16 on this "
                  f"Mac: a {PROMPT}-token prompt → TTFT {TTFT[PROMPT]:.2f} s; decode at {PROMPT} tokens of context → TPOT "
                  f"{TPOT_MS[PROMPT]:.1f} ms (measurements.md §2–3). The 200-token answer is illustrative. Prompt 128 → 16,384 "
                  f"(×128): TTFT {TTFT[128]:.3f} → {TTFT[16384]:.2f} s (×{TTFT[16384] / TTFT[128]:.0f}, attention grows too); "
                  f"TPOT {TPOT_MS[128]:.1f} → {TPOT_MS[16384]:.1f} ms (+{100 * TPOT_RISE:.0f}%, the KV cache read grows).")
        total = TTFT[PROMPT] + ANSWER * TPOT_MS[PROMPT] / 1000
        line = Timeline(9.0, width=11.2, origin=(-5.6, 0.45, 0), lane_height=0.5, label_every=1, unit="s")
        assert total < 9.0
        pre = line.bar(0, TTFT[PROMPT], COMPUTE)
        ticks = m.VGroup(*[line.tick(TTFT[PROMPT] + (k + 0.5) * TPOT_MS[PROMPT] / 1000, MEMORY) for k in range(ANSWER)])
        for t in ticks:
            t.set_stroke(width=2)
        x0, x1 = line.x(0)[0], line.x(TTFT[PROMPT])[0]
        ttft_br = bracket(x0, x1, 1.42, COMPUTE, down=True)
        ttft_lab = m.VGroup(code("TTFT", 24, COMPUTE, weight=m.BOLD), text("set by compute", 22, MUTED)).arrange(m.RIGHT, buff=0.2)
        ttft_lab.next_to(ttft_br, m.UP, buff=0.14).align_to(ttft_br, m.LEFT)
        ttft_val = title_text(f"{TTFT[PROMPT]:.1f} s", 30, COMPUTE)
        tx0, tx1 = ticks[0].get_x(), ticks[-1].get_x()
        tpot_br = bracket(tx0, tx1, 1.42, MEMORY, down=True)
        tpot_lab = m.VGroup(code("TPOT", 24, MEMORY, weight=m.BOLD), text("set by bandwidth", 22, MUTED)).arrange(m.RIGHT, buff=0.2)
        tpot_lab.next_to(tpot_br, m.UP, buff=0.14).align_to(tpot_br, m.LEFT)
        tpot_val = title_text(f"{TPOT_MS[PROMPT]:.0f} ms", 30, MEMORY)
        ttft_val.next_to(ttft_lab, m.RIGHT, buff=0.3).align_to(ttft_lab[0], m.DOWN)
        tpot_grp = m.VGroup(tpot_lab.copy(), tpot_val.copy()).arrange(m.RIGHT, buff=0.3)
        tpot_grp.next_to(tpot_br, m.UP, buff=0.14).align_to(tpot_br, m.RIGHT)
        tpot_lab.move_to(tpot_grp[0])
        tpot_val.move_to(tpot_grp[1]).align_to(tpot_lab[0], m.DOWN)
        caption = text(f"Owl-8B on this Mac · a {PROMPT:,}-token chapter in · {ANSWER} tokens out (for example)", 20, MUTED)
        caption.move_to([0, 2.38, 0]).align_to(line.axis, m.LEFT)

        # TTFT and TPOT against prompt length
        rows = [128, PROMPT, 16384]
        ys = [-1.25, -1.85, -2.45]
        bx, bw = -3.7, 6.2
        head_p = code("prompt", 20, MUTED).move_to([0, -0.68, 0]).align_to([bx - 0.35, 0, 0], m.RIGHT)
        head_t = code("TTFT", 20, COMPUTE, weight=m.BOLD).move_to([0, -0.68, 0]).align_to([bx, 0, 0], m.LEFT)
        head_d = code("TPOT", 20, MEMORY, weight=m.BOLD).move_to([5.6, -0.68, 0])
        table = {}
        for n, y in zip(rows, ys):
            lab = code(f"{n:,} tokens", 22, INK).move_to([0, y, 0]).align_to([bx - 0.35, 0, 0], m.RIGHT)
            w = max(bw * TTFT[n] / TTFT[16384], 0.04)
            bar = m.Rectangle(width=w, height=0.36).set_stroke(COMPUTE, 1.5).set_fill(AMBER_FILL, 1)
            bar.move_to([bx + w / 2, y, 0])
            val = code(f"{TTFT[n]:.2f} s" if TTFT[n] < 10 else f"{TTFT[n]:.1f} s", 22, COMPUTE).next_to(bar, m.RIGHT, buff=0.18)
            tp = code(f"{TPOT_MS[n]:.0f} ms", 22, MEMORY).move_to([5.6, y, 0])
            table[n] = (lab, bar, val, tp)
        src = cite(SRC_MAC + " · TPOT at that context").move_to([0, -3.2, 0])

        self.say(
            "[Episode one named the two numbers a user feels.] [The wait for the first token is prefill]: [set by compute.]",
            m.AnimationGroup(self.new_header("05 · what you feel", "Wait, then stream"), m.FadeIn(line), m.FadeIn(caption)),
            m.AnimationGroup(m.GrowFromEdge(pre, m.LEFT), m.FadeIn(ttft_br)),
            m.FadeIn(ttft_lab, shift=0.1 * m.UP),
        )
        self.say(
            "[After that, the time per output token is decode]: [set by bandwidth.]",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(t) for t in ticks], lag_ratio=0.03, run_time=2.4), m.FadeIn(tpot_br)),
            m.FadeIn(tpot_lab, shift=0.1 * m.UP),
        )
        self.say(
            "[For our student's chapter], [the first token takes two seconds on this Mac], [then a new one arrives every thirty-three milliseconds.]",
            m.Circumscribe(pre, color=COMPUTE, buff=0.08),
            m.FadeIn(ttft_val, shift=0.1 * m.LEFT),
            m.FadeIn(tpot_val, shift=0.1 * m.LEFT),
        )
        lab, bar, val, tp = table[PROMPT]
        lab1, bar1, val1, tp1 = table[128]
        lab3, bar3, val3, tp3 = table[16384]
        self.say(
            "[Paste more, and the first wait grows with the prompt]: [a fifth of a second for a short question], [twenty seconds for sixteen thousand tokens.]",
            m.AnimationGroup(m.FadeIn(head_p), m.FadeIn(head_t), m.FadeIn(lab), m.GrowFromEdge(bar, m.LEFT), m.FadeIn(val), m.FadeIn(src)),
            m.AnimationGroup(m.FadeIn(lab1), m.GrowFromEdge(bar1, m.LEFT), m.FadeIn(val1)),
            m.AnimationGroup(m.FadeIn(lab3), m.GrowFromEdge(bar3, m.LEFT, run_time=1.4), m.FadeIn(val3)),
        )
        self.say(
            "[But each later token slows down by only fourteen percent.]",
            m.AnimationGroup(m.FadeIn(head_d), m.LaggedStart(m.FadeIn(tp1), m.FadeIn(tp), m.FadeIn(tp3), lag_ratio=0.3)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the M5

    def new_chip(self):
        self.next_section(
            "6 · The M5 test",
            notes=f"llama.cpp Apple Silicon benchmark (github.com/ggml-org/llama.cpp/discussions/4167, LLaMA-2-7B F16, "
                  f"llama-bench -p 512 -n 128; M5 rows use a newer llama.cpp commit): M4 Max → M5 Max PP512 922.83 → 3158.49 "
                  f"tok/s (×{PP_X:.2f}), TG128 31.64 → 37.11 (×{TG_X:.3f}), bandwidth 546 → 614 GB/s (×{BW_X:.3f}). The M5 Max GPU "
                  f"has 'a Neural Accelerator in each GPU core' (Apple newsroom, 2026-03-03). Apple ML Research (2025-11-19): 'generating the first token is "
                  f"compute-bound' / 'Generating subsequent tokens is bounded by memory bandwidth'. The M5 roof is drawn ×{PP_X:.1f} "
                  f"from the prefill speedup (illustrative: Apple publishes no TFLOPS). Hardware dated October 2026.")
        roof, ylab = mac_roofline()
        d_dot = roof.dot(1, MEMORY, radius=0.11)
        p_dot = roof.dot(PROMPT, COMPUTE, radius=0.13)
        d_lab = code("decode", 20, MEMORY, weight=m.BOLD).next_to(d_dot, m.DR, buff=0.06)
        p_lab = code("prefill", 20, COMPUTE, weight=m.BOLD).next_to(p_dot, m.UP, buff=0.12)
        m4_lab = code("M4 Max", 20, MUTED).next_to(roof.point(600), m.DOWN, buff=0.24)

        bw2, peak2 = BW * BW_X, PEAK * PP_X
        r2 = peak2 / bw2
        lo, hi = 10 ** roof.xd[0], 10 ** roof.xd[1]
        new_mem = m.Line(roof.at(lo, bw2 * lo), roof.at(r2, peak2)).set_stroke(MEMORY, 6)
        new_cmp = m.Line(roof.at(r2, peak2), roof.at(hi, peak2)).set_stroke(COMPUTE, 6)
        new_ridge = m.DashedLine(roof.at(r2, 10 ** roof.yd[0]), roof.at(r2, peak2), dash_length=0.12).set_stroke(MUTED, 2)
        ghost = m.VGroup(m.DashedLine(roof.memory.get_start(), roof.memory.get_end(), dash_length=0.1),
                         m.DashedLine(roof.compute.get_start(), roof.compute.get_end(), dash_length=0.1)).set_stroke(FAINT, 3)
        m5_lab = code("M5 Max", 20, INK, weight=m.BOLD).next_to(roof.at(60, peak2), m.UP, buff=0.18)
        p_new = roof.at(PROMPT, peak2)
        d_new = roof.at(1, bw2)
        p_lab_new = p_lab.copy().next_to(m.Dot(p_new, radius=0.13), m.UP, buff=0.12)
        up_arrow = arrow(roof.point(PROMPT) + [0.36, 0.0, 0], p_new + [0.36, 0.0, 0], COMPUTE, buff=0.0)
        up_lab = title_text(f"×{PP_X:.1f}", 30, COMPUTE).next_to(up_arrow, m.RIGHT, buff=0.12)
        d_x = title_text(f"×{TG_X:.2f}", 26, MEMORY).next_to(m.Dot(d_new, radius=0.11), m.UL, buff=0.1)

        # the llama.cpp table
        tb = card(4.4, 3.45, EDGE, CARD, radius=0.18, stroke=2)
        tb.move_to([4.35, 2.45 - tb.height / 2, 0])
        top, left, right = tb.get_top()[1], tb.get_left()[0] + 0.28, tb.get_right()[0] - 0.28
        t_head = m.VGroup(code("M4 Max → M5 Max", 22, INK, weight=m.BOLD), cite("llama.cpp · LLaMA-2-7B · F16"))
        t_head.arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT).move_to([0, top - 0.5, 0]).align_to([left, 0, 0], m.LEFT)
        na = text("+ Neural Accelerators", 22, COMPUTE, weight=m.BOLD)
        na.move_to(roof.at(1.9, 10 ** 13.68))

        def trow(name, unit, detail, ratio, color, dy):
            a = m.VGroup(text(name, 22, color, weight=m.BOLD), code(unit, 20, MUTED)).arrange(m.RIGHT, buff=0.15)
            a[1].align_to(a[0], m.DOWN)
            b = code(detail, 20, MUTED)
            lft = m.VGroup(a, b).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
            lft.move_to([0, top - dy, 0]).align_to([left, 0, 0], m.LEFT)
            r = title_text(ratio, 34, color).move_to([0, top - dy, 0]).align_to([right, 0, 0], m.RIGHT)
            return m.VGroup(lft, r)

        r_bw = trow("bandwidth", "GB/s", "546 → 614", f"×{BW_X:.2f}", MEMORY, 1.4)
        r_pp = trow("prefill", "tok/s", "923 → 3,158", f"×{PP_X:.1f}", COMPUTE, 2.15)
        r_tg = trow("decode", "tok/s", "31.6 → 37.1", f"×{TG_X:.2f}", MEMORY, 2.9)
        # Apple's own words
        q1 = m.VGroup(text("“… generating the first", 20, INK), text("token is compute-bound …”", 20, COMPUTE))
        q2 = m.VGroup(text("“Generating subsequent", 20, INK), text("tokens is bounded by", 20, INK),
                      text("memory bandwidth …”", 20, MEMORY))
        quote = m.VGroup(*q1, *q2).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
        for q in q2:
            q.shift(0.1 * m.DOWN)
        q_src = cite("Apple ML Research, Nov 2025")
        q_body = m.VGroup(quote, q_src)
        q_src.next_to(quote, m.DOWN, buff=0.14).align_to(quote, m.LEFT)
        qb = card(4.4, q_body.height + 0.5, EDGE, CARD, radius=0.18, stroke=2)
        qb.move_to([4.35, tb.get_bottom()[1] - 0.3 - qb.height / 2, 0])
        q_body.move_to(qb).align_to([left, 0, 0], m.LEFT)

        self.say(
            "[Here's the idea, tested on new hardware.] [The M5 Max put Neural Accelerators in every GPU core]: [far more compute], [but only twelve percent more bandwidth.]",
            m.AnimationGroup(self.new_header("06 · new hardware", "More compute lifts the roof"),
                             m.FadeIn(roof.axes), m.FadeIn(roof.ticks), m.FadeIn(roof.x_label), m.FadeIn(ylab),
                             m.FadeIn(roof.memory), m.FadeIn(roof.compute), m.FadeIn(roof.ridge), m.FadeIn(m4_lab),
                             m.FadeIn(d_dot), m.FadeIn(p_dot), m.FadeIn(d_lab), m.FadeIn(p_lab)),
            m.AnimationGroup(m.FadeIn(tb), m.FadeIn(t_head)),
            m.FadeIn(na, shift=0.1 * m.DOWN),
            m.FadeIn(r_bw, shift=0.1 * m.UP),
        )
        self.say(
            "[In the llama.cpp benchmark for Macs], [prompt processing got three point four times faster], [while generation got just one point one seven.]",
            m.Circumscribe(t_head, color=MUTED, buff=0.08),
            m.FadeIn(r_pp, shift=0.1 * m.UP),
            m.FadeIn(r_tg, shift=0.1 * m.UP),
        )
        self.say(
            "[More compute lifts the roof]; [the prefill dot rides up with it], [while the decode dot barely moves.]",
            m.AnimationGroup(m.FadeIn(ghost), m.Transform(roof.compute, new_cmp), m.Transform(roof.memory, new_mem),
                             m.Transform(roof.ridge, new_ridge), m.FadeIn(m5_lab), run_time=1.4),
            m.AnimationGroup(p_dot.animate.move_to(p_new), p_lab.animate.move_to(p_lab_new),
                             m.GrowArrow(up_arrow), m.FadeIn(up_lab)),
            m.AnimationGroup(d_dot.animate.move_to(d_new), d_lab.animate.shift(d_new - roof.point(1)), m.FadeIn(d_x)),
        )
        self.say(
            "[In Apple's own words]: [the first token is compute-bound], [and the tokens after it are bound by memory bandwidth.]",
            m.AnimationGroup(m.FadeIn(qb), m.FadeIn(q_src)),
            m.FadeIn(q1, shift=0.1 * m.UP),
            m.FadeIn(q2, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ two knobs

    def knobs(self):
        self.next_section(
            "7 · Two knobs",
            notes=f"More compute (M5 Max vs M4 Max, llama.cpp #4167): prefill ×{PP_X:.2f}, decode ×{TG_X:.2f}. Fewer bytes "
                  f"(Owl-8B 4-bit vs BF16, measured on this Mac): weights {M8['weights_gb']:.2f} → {Q4_GB:.2f} GB (×{BYTES_X:.2f} "
                  f"fewer), decode {DEC_TOK:.1f} → {Q4_DEC:.1f} tok/s (×{Q4_DEC_X:.2f}), prefill at {PROMPT} tokens {PRE_TOK:.0f} → "
                  f"{Q4_PRE:.0f} tok/s (×{Q4_PRE_X:.2f}: dequantizing costs compute, measurements.md §1–2). Quantization is Episodes 18–19.")
        cx = {"prefill": 0.9, "decode": 4.6}
        ry = {"compute": 0.55, "bytes": -1.25}
        heads = m.VGroup(*[title_text(k, 36, c).move_to([cx[k], 1.85, 0]) for k, c in (("prefill", COMPUTE), ("decode", MEMORY))])

        def rlabel(title, sub, y):
            g = m.VGroup(title_text(title, 32, INK), text(sub, 20, MUTED)).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
            return g.move_to([0, y, 0]).align_to([-6.3, 0, 0], m.LEFT)

        lab_c = rlabel("more compute", "M5 Max vs M4 Max · llama.cpp", ry["compute"])
        lab_b = rlabel("fewer bytes", "4-bit vs 16-bit weights · this Mac", ry["bytes"])

        def cell(value, y, x, color, hot):
            box = card(3.0, 1.45, color if hot else EDGE, FILLS[color] if hot else CARD, radius=0.18, stroke=2.5 if hot else 2)
            box.move_to([x, y, 0])
            v = title_text(value, 60, color if hot else MUTED).move_to(box)
            return m.VGroup(box, v)

        c_pp = cell(f"×{PP_X:.1f}", ry["compute"], cx["prefill"], COMPUTE, True)
        c_tg = cell(f"×{TG_X:.2f}", ry["compute"], cx["decode"], MEMORY, False)
        b_pp = cell(f"×{Q4_PRE_X:.2f}", ry["bytes"], cx["prefill"], COMPUTE, False)
        b_tg = cell(f"×{Q4_DEC_X:.1f}", ry["bytes"], cx["decode"], MEMORY, True)
        bytes_note = code(f"weights {M8['weights_gb']:.2f} → {Q4_GB:.2f} GB · Owl-8B", 20, BLUE)
        bytes_note.move_to([0, ry["bytes"] - 1.12, 0]).align_to(lab_b, m.LEFT)
        math_note = text("unpacking 4-bit weights costs extra math", 20, MUTED).move_to([cx["prefill"], ry["bytes"] - 1.12, 0])
        math_note.align_to(b_pp, m.LEFT)
        src = cite(SRC_MAC.replace("Owl-8B BF16", "Owl-8B") + " · llama.cpp discussion 4167").move_to([0, -3.25, 0])
        speed = code("speed-up", 20, MUTED).move_to([2.75, 2.42, 0])

        self.say(
            "[The mirror image is measured right here]: [four-bit weights move three and a half times fewer bytes.]",
            m.AnimationGroup(self.new_header("07 · two knobs", "Which knob moves which dot"), m.FadeIn(heads), m.FadeIn(speed),
                             m.FadeIn(lab_c), m.FadeIn(c_pp), m.FadeIn(c_tg)),
            m.AnimationGroup(m.FadeIn(lab_b, shift=0.1 * m.UP), m.FadeIn(bytes_note), m.FadeIn(src)),
        )
        self.say(
            "[Decode got three point three times faster], [and prefill got slightly slower], [because unpacking four-bit weights costs extra math.]",
            m.FadeIn(b_tg, scale=0.85),
            m.FadeIn(b_pp, scale=0.85),
            m.FadeIn(math_note, shift=0.1 * m.UP),
        )
        self.say(
            "[So add compute, and prefill speeds up.] [Remove bytes, and decode does.]",
            m.Circumscribe(c_pp, color=COMPUTE, buff=0.1),
            m.Circumscribe(b_tg, color=MEMORY, buff=0.1),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section(
            "Recap",
            notes="Intensity decides the limit: prefill reuses each weight across the whole prompt (compute-bound, sets TTFT); "
                  "decode uses each weight once per token (memory-bound, sets TPOT). Batching moves decode right: Episode 4.")
        halo = glow(AMBER, radius=4.0)
        words = m.VGroup(*[title_text(w, 66, c) for w, c in (("Intensity", INK), ("Compute", COMPUTE), ("Bandwidth", MEMORY))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("uses per byte read", "prefill · TTFT", "decode · TPOT")])
        slots = [max(w.width, s.width) for w, s in zip(words, subs)]
        x, gap = 0.0, 0.9
        for w, s, slot in zip(words, subs, slots):
            w.move_to([x + slot / 2, 0, 0])
            s.move_to([x + slot / 2, 0, 0])
            x += slot + gap
        on_baseline(words)
        line_y = min(w.get_bottom()[1] for w in words) - 0.45
        for s in subs:
            s.set_y(line_y).align_to(subs[0], m.UP)
        mid = words[0][0].get_center()[1]
        dots = m.VGroup(*[m.Dot([(a.get_right()[0] + b.get_left()[0]) / 2, mid, 0], radius=0.07, color=MUTED) for a, b in zip(words, words[1:])])
        row_ = m.VGroup(words, dots, subs).move_to(0.55 * m.UP)
        tag1 = text("A long prompt reuses every weight: prefill hits the compute roof.", 28, INK)
        tag2 = text("One token per weight read: decode waits on bandwidth.", 28, MUTED)
        tagline = m.VGroup(tag1, tag2).arrange(m.DOWN, buff=0.2).next_to(row_, m.DOWN, buff=0.7)
        mascot = owl(0.55).move_to([5.9, -2.9, 0])

        self.say(
            "[To sum up]: [intensity], [compute], [bandwidth.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[A long prompt reuses every weight, so prefill hits the compute roof]; [one token per read leaves decode waiting on bandwidth.]",
            m.FadeIn(tag1, shift=0.1 * m.UP),
            m.FadeIn(tag2, shift=0.1 * m.UP),
        )
        self.say(
            "[Decode only climbs with a bigger batch], [and that's where we go next.]",
            m.Circumscribe(words[2], color=MEMORY, buff=0.12),
            m.FadeIn(mascot, shift=0.2 * m.UP),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row_, tagline, mascot)], run_time=0.9)
