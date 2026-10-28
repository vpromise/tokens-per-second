"""Tokens per Second · Episode 23 · Your Mac as an Inference Server (Module 6 · Production & Your Mac)

Unified memory makes big models local; bandwidth sets the speed. Running example: everything measured on this
Mac Studio (Apple M4 Max, 546 GB/s, 64 GB, mlx-lm 0.32.2), side by side with the ceilings from owl_reference.

Sections: hook (every measurement of the course came from this desk) · one pool (a GPU server: HBM 80 GB beside
the GPU, the CPU's RAM across PCIe 128 GB/s; a Mac: one unified pool for CPU, GPU and Neural Engine; MLX's GPU
working set 55.7 GB of 64, 8.3 GB left to macOS; 546 GB/s ≈ 1/6 of an H100's 3.35 TB/s) · what fits (Owl-8B BF16
16.1 GB fits; Owl-70B 141 GB BF16 and 75 GB 8-bit do not; 4-bit 39.7 GB fits; its KV is 320 KiB per token, so
32K tokens add 10 GiB (50.4 GB in all) and the limit is ≈ 48,750 tokens; 128K would need 40 GiB of KV) · bandwidth
÷ size (this Mac's ceiling as one line on log–log axes; six models measured here sit at 62–91% of it; Episodes
2/18/19's 31.0 / 58.4 / 101.9 tok/s; Owl-70B 4-bit ≤ 13.8 tok/s, computed) · the ladder (M4 120 → M4 Pro 273 →
M4 Max 546 → M5 Max 614 → M5 Ultra 1,200 GB/s; four-bit Owl-8B ceilings 27 → 266 tok/s; Owl-70B does not fit
a 32 GB M4, 7–30 tok/s on the others) · a live run (the stock mlx_lm.generate command and its real output:
95-token prompt at 623 tok/s, 256 tokens at 102.7 tok/s, peak 4.78 GB) · serving (mlx_lm.server, an
OpenAI-compatible endpoint as in Episode 22; TTFT and ITL percentiles at 1 / 4 / 16 clients; the 8.3 ms gap floor)
· the board (ten measurements next to their limits: 78–100%, none above) · energy (GPU joules per token:
0.61 BF16, 0.32 4-bit, 0.17 BF16 batched 16; a prompt token ≈ 1/11 of an output token) · engines (MLX and
llama.cpp, not ranked) · recap (fit, divide, measure).

Sources (facts dated October 2026):
- research/measurements.md / .json (this Mac Studio, 2026-10-24, mlx-lm 0.32.2 / MLX 0.32.4), loaded with
  measured(): machine (64 GB unified, 546 GB/s spec, MLX recommended working set 55.66 GB), §0 roofs (read
  482.8 GB/s, bf16 matmul 15.2 TFLOPS), §1 decode (median of 5, 256 tokens, greedy; weights = file bytes), §2
  prefill at 2,048 tokens (8B BF16 1,005 tok/s, 14.6 TFLOPS = 96% of the measured matmul), §3 context (8B 4-bit
  at 32K: 51.0 tok/s), §4 memory (KV slope 131,072 B/token), §8 batching (8B BF16, B = 8: 235 tok/s), §9
  speculative (8B BF16 target, 1B 4-bit draft, copy prompt, k = 6: 148 tok/s, 6.92 tokens per pass), §10 serving
  (mlx_lm.server, 8B 4-bit, 1 / 4 / 16 closed-loop clients, 128 tokens per request), §11 energy (GPU-only IOReport
  counters via zeus-apple-silicon; DRAM not counted), cross-check (stock mlx_lm.generate, median of 3).
  The terminal's output lines are read from research/measurements/logs/cli_crosscheck.log (the run whose speed is
  the median); the answer's first words are the same greedy run's text from the harness (raw decode JSON).
- owl_reference.py: MODELS (owl-8b, owl-mini, owl-70b configs), HARDWARE (H100 SXM 3.35 TB/s and 80 GB; M4 120,
  M4 Pro 273, M4 Max 546 GB/s; M4 32 GB and M4 Pro 64 GB maximum memory), weight_bytes, kv_bytes_per_token,
  decode_ceiling, step_time.
- research/notes.md §2 hardware table [DATED Oct 2026; Apple newsroom]: M4 Pro/Max 2024-10-30, M5 Max 614 GB/s
  (2026-03-03, "up to 614GB/s"), M5 Ultra 1.2 TB/s (2026-08-25); maximum memory M4 Max/M5 Max 128 GB, M5 Ultra
  512 GB; NVIDIA H100 page: "PCIe Gen5: 128GB/s". §14: unified memory (CPU, GPU and Neural Engine share one
  LPDDR pool), MLX / mlx-lm (mlx-community models, 4-bit = group 64 + scale + bias), llama.cpp (GGUF k-quants,
  Metal backend, mmap; Ollama and LM Studio wrap these engines), llama.cpp README: "Apple silicon is a
  first-class citizen".

Computed, not measured: everything about Owl-70B (weights at 16 / 8.5 / 4.5 bits per weight, its KV, the
48,750-token bound, which ignores activations, so it is an upper bound; its 13.8 tok/s ceiling) and the ladder's
ceilings for other Macs. Illustrative: the server and Mac block diagrams, the dot clusters for clients, the
"halve the bytes" staircase.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from owl_reference import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- the numbers

PREFIX = "ep23"
GB, GiB = 1e9, 2 ** 30
MEAS = Path(__file__).resolve().parent.parent / "research" / "measurements"

# this Mac, and the H100 it is compared with
BW = MAC["bandwidth"]
RAM = MAC["memory"]
WORKSET = measured("machine", "mlx_max_recommended_working_set_GB") * GB   # what MLX lets the GPU use
OS_SPARE = RAM - WORKSET
H100 = HARDWARE["H100"]
PCIE = 128e9                                                                # NVIDIA H100 page, notes §2
HBM_X = H100["bandwidth"] / BW
assert measured("machine", "unified_memory_GB") * GB == RAM and measured("machine", "memory_bandwidth_spec_GBps") * GB == BW
assert round(RAM / GB) == 64 and round(BW / GB) == 546                      # "sixty-four", "five hundred and forty-six"
assert round(WORKSET / GB) == 56 and round(OS_SPARE / GB) == 8              # "about fifty-six", "about eight"
assert H100["memory"] == 80e9 and 5.5 < HBM_X < 6.5                          # "eighty gigabytes", "about a sixth"

# what fits: Owl-8B and Owl-70B at 16, 8.5 and 4.5 bits per weight (MLX's 8- and 4-bit carry a scale + offset per 64)
O70 = MODELS["owl-70b"]
W8 = weight_bytes()
W8_4 = weight_bytes(bits=4.5)
W70 = {b: weight_bytes(O70, bits=b) for b in (16, 8.5, 4.5)}
assert round(W8 / GB) == 16                                                  # "sixteen gigabytes"
assert round(W70[16] / GB) == 141 and round(W70[8.5] / GB) == 75             # "a hundred and forty-one", "seventy-five"
assert 39 < W70[4.5] / GB < 40 and W70[4.5] < WORKSET < W70[8.5]            # "under forty", fits; 8-bit does not
KV70 = kv_bytes_per_token(O70)
assert KV70 == 320 * 1024                                                    # "three hundred and twenty K per token"
CTX = 32_768
KV32 = CTX * KV70
assert KV32 == 10 * GiB and W70[4.5] + KV32 < WORKSET                        # "adds ten gibibytes", still fits
MAXCTX = (WORKSET - W70[4.5]) / KV70
assert 45_000 < MAXCTX < 50_000                                              # "just under fifty thousand tokens"
KV128 = 131_072 * KV70
assert KV128 == 40 * GiB and W70[4.5] + KV128 > RAM                          # "forty gibibytes of cache", no fit
assert abs(measured("models", "llama-3.1-8b", "4bit", "weights_file_bytes") - W8_4) / W8_4 < 0.001

# bandwidth ÷ size: the six models measured here against this Mac's line
KEYS = [(name, bits) for name in ("llama-3.2-1b", "llama-3.1-8b") for bits in ("4bit", "8bit", "bf16")]
SIZE = {k: measured("models", *k, "weights_file_bytes") for k in KEYS}
TOK = {k: measured("decode", *k, "tok_s_median") for k in KEYS}
LINE = {k: BW / SIZE[k] for k in KEYS}
SHARE = {k: TOK[k] / LINE[k] for k in KEYS}
for k in KEYS:
    assert abs(LINE[k] - measured("decode", *k, "ceiling_tok_s")) / LINE[k] < 0.002 and SHARE[k] < 1
assert round(min(SHARE.values()) * 100) == 62 and round(max(SHARE.values()) * 100) == 91   # "sixty-two … ninety-one"
T8 = {b: TOK[("llama-3.1-8b", b)] for b in ("bf16", "8bit", "4bit")}
assert [f"{T8[b]:.1f}" for b in ("bf16", "8bit", "4bit")] == ["31.0", "58.4", "101.9"]  # "thirty-one, fifty-eight, a hundred and two"
CEIL70 = decode_ceiling(O70, MAC, bits=4.5)
assert round(CEIL70) == 14                                                   # "about fourteen tokens a second"

# the ladder (HARDWARE for the M4 family; notes §2 for the M5 Max and M5 Ultra, Apple newsroom, dated)
LADDER = [
    ("M4", HARDWARE["M4"]["bandwidth"], HARDWARE["M4"]["memory"], "2024 · up to 32 GB"),
    ("M4 Pro", HARDWARE["M4 Pro"]["bandwidth"], HARDWARE["M4 Pro"]["memory"], "2024 · up to 64 GB"),
    ("M4 Max", BW, 128e9, "this Mac · 64 GB"),
    ("M5 Max", 614e9, 128e9, "Mar 2026 · up to 128 GB"),
    ("M5 Ultra", 1200e9, 512e9, "Aug 2026 · up to 512 GB"),
]
L8 = {n: bw / W8_4 for n, bw, _, _ in LADDER}
L70 = {n: (bw / W70[4.5] if mem > W70[4.5] else None) for n, bw, mem, _ in LADDER}
assert [round(b / GB) for _, b, _, _ in LADDER] == [120, 273, 546, 614, 1200]
assert round(L8["M4"]) == 27 and round(L8["M5 Ultra"]) == 266 and round(L8["M5 Ultra"] / L8["M4"]) == 10  # "ten times"
assert L70["M4"] is None and round(L70["M4 Pro"]) == 7 and round(L70["M5 Ultra"]) == 30   # "between seven and thirty"
assert abs(L8["M4 Max"] - LINE[("llama-3.1-8b", "4bit")]) < 0.1

# a live run: the stock command and its real output (cross-check log; the run with the median speed)
REPO = measured("models", "llama-3.1-8b", "4bit", "repo")
CLI = measured("cli_crosscheck_mlx_lm_generate", "Meta-Llama-3.1-8B-Instruct-4bit", "tok_s_median")
RUNS = re.findall(r"## run \d+: mlx_lm\.generate --model (\S+) .*\nPrompt: (\d+) tokens, ([\d.]+) tokens-per-sec\n"
                  r"Generation: (\d+) tokens, ([\d.]+) tokens-per-sec\nPeak memory: ([\d.]+) GB",
                  (MEAS / "logs" / "cli_crosscheck.log").read_text())
RUN = [r for r in RUNS if r[0] == REPO and abs(float(r[4]) - CLI) < 1e-6]
assert len(RUN) == 1
_, P_TOK, P_TPS, G_TOK, G_TPS, PEAK = RUN[0]
assert (int(P_TOK), int(G_TOK)) == (95, 256) and float(P_TPS) > 600         # "ninety-five tokens at over six hundred"
assert round(float(G_TPS), -1) == 100 and float(PEAK) < 5                   # "about a hundred", "under five gigabytes"
SAMPLE = json.loads((MEAS / "raw" / "decode__llama-3.1-8b__4bit.json").read_text())["sample_text_first120tok"]
ANSWER = "Hello there, young friend. I'm Owl, your study buddy."
assert SAMPLE.startswith(ANSWER)

# serving through mlx_lm.server (Owl-8B 4-bit)
SRV = {c: measured("server", "llama-3.1-8b__4bit", str(c)) for c in (1, 4, 16)}
FLOOR = W8_4 / BW                                                            # the gap between tokens can't beat this
assert round(SRV[1]["ttft_ms"]["p50"]) == 72 and round(SRV[1]["itl_ms"]["p50"], 1) == 10.6   # "seventy-two", "ten and a half"
assert round(SRV[16]["aggregate_tok_s"]) == 290 and round(SRV[16]["per_request_decode_tok_s"]["median"]) == 21
assert 0.5 < SRV[16]["ttft_ms"]["p50"] / 1000 < 1                            # "most of a second"
assert round(FLOOR * 1e3, 1) == 8.3

# the board: every measurement next to its limit
READ = measured("roofs", "read_GBps", "median") * GB
PEAK_MM = measured("roofs", "matmul_bf16_TFLOPS", "median")
PRE = measured("prefill", "llama-3.1-8b", "bf16", "by_prompt_tokens", "2048")
KV_SLOPE = measured("memory", "llama-3.1-8b", "4bit", "kv16", "slope_active_bytes_per_token")
CTX_TOK = measured("context", "llama-3.1-8b", "4bit", "by_context_tokens", str(CTX), "tok_s_median")
CTX_CEIL = decode_ceiling(OWL, MAC, bits=4.5, context=CTX)
B8 = measured("batching", "llama-3.1-8b", "bf16", "by_batch", "8", "aggregate_tok_s")
B8_CEIL = 8 / step_time(OWL, MAC, batch=8)                                   # one weight read per step, shared by 8
SPEC = measured("speculative", "llama-3.1-8b__bf16", "prompts", "copy_notes", "by_num_draft_tokens", "6")
DRAFT = measured("models", "llama-3.2-1b", "4bit", "weights_file_bytes")
SPEC_CEIL = SPEC["tokens_per_target_pass"] * BW / (W8 + 6 * DRAFT)          # bytes per pass: target once, draft 6 times
SRV1 = SRV[1]["per_request_decode_tok_s"]["median"]
BOARD = [  # label, measured, limit, share, limit colour, episode
    ("memory read", f"{READ / GB:.0f} GB/s", f"{BW / GB:.0f} GB/s", READ / BW, MEMORY, 2),
    ("decode · BF16", f"{T8['bf16']:.1f} tok/s", f"{LINE[('llama-3.1-8b', 'bf16')]:.1f} tok/s", SHARE[("llama-3.1-8b", "bf16")], MEMORY, 2),
    ("decode · 8-bit", f"{T8['8bit']:.1f} tok/s", f"{LINE[('llama-3.1-8b', '8bit')]:.1f} tok/s", SHARE[("llama-3.1-8b", "8bit")], MEMORY, 18),
    ("decode · 4-bit", f"{T8['4bit']:.1f} tok/s", f"{LINE[('llama-3.1-8b', '4bit')]:.1f} tok/s", SHARE[("llama-3.1-8b", "4bit")], MEMORY, 19),
    ("prefill · 2K prompt", f"{PRE['achieved_TFLOPS']:.1f} TFLOPS", f"{PEAK_MM:.1f} TFLOPS", PRE["fraction_of_measured_matmul_peak"], COMPUTE, 3),
    ("KV per token", f"{KV_SLOPE:,.0f} B", f"{kv_bytes_per_token():,.0f} B", KV_SLOPE / kv_bytes_per_token(), KV, 5),
    ("decode · 32K context", f"{CTX_TOK:.1f} tok/s", f"{CTX_CEIL:.1f} tok/s", CTX_TOK / CTX_CEIL, MEMORY, 9),
    ("batch of 8 · BF16", f"{B8:.0f} tok/s", f"{B8_CEIL:.0f} tok/s", B8 / B8_CEIL, MEMORY, 4),
    ("speculative · 6 drafts", f"{SPEC['tok_s_median']:.0f} tok/s", f"{SPEC_CEIL:.0f} tok/s", SPEC["tok_s_median"] / SPEC_CEIL, MEMORY, 13),
    ("server · 1 stream", f"{SRV1:.1f} tok/s", f"{L8['M4 Max']:.1f} tok/s", SRV1 / L8["M4 Max"], MEMORY, 22),
]
PCT = [round(r[3] * 100) for r in BOARD]
assert PCT == [88, 91, 91, 84, 96, 100, 82, 86, 79, 78], PCT
assert max(r[3] for r in BOARD) <= 1 and min(r[3] for r in BOARD) >= 0.75    # "not one beats it … at least three quarters"
assert PCT[0] == 88 and (min(PCT[1:4]), max(PCT[1:4])) == (84, 91) and PCT[4] == 96   # B2, B3
assert (min(PCT[6:]), max(PCT[6:])) == (78, 86)                              # "between seventy-eight and eighty-six"
assert round(B8) == 235 and round(B8_CEIL) == 272 and round(SPEC_CEIL) == 187

# energy per token (GPU only)
EN = {k: measured("energy", "llama-3.1-8b", k) for k in ("bf16", "4bit")}
E16, E4 = EN["bf16"]["decode_b1_J_per_token"], EN["4bit"]["decode_b1_J_per_token"]
E16B = EN["bf16"]["decode_b16_J_per_token_minus_prefill_est"]
EIN = EN["bf16"]["prefill_mJ_per_prompt_token"] / 1000
assert round(E16, 1) == 0.6 and 0.45 < E4 / E16 < 0.55 and E16B < 0.2       # "six tenths", "roughly halve", "under a fifth"
assert 0.08 < EIN / E16 < 0.12                                               # "about a tenth"
SPEC_X = SPEC["speedup"]
PRE16 = PRE["tok_s_median"]

# ---------------------------------------------------------------------------- look


def cite(s, size=20):
    return text(s, size, MUTED)


def hbar(x0, y, w, h, color, fill, stroke=2):
    return m.Rectangle(width=max(w, 0.02), height=h).set_stroke(color, stroke).set_fill(fill, 1).move_to([x0 + max(w, 0.02) / 2, y, 0])


def block(w, h, label, color=EDGE, fill=CARD, size=22, mono=True, label_color=INK):
    box = card(w, h, color, fill, radius=0.12, stroke=2.2)
    words = code(label, size, label_color) if mono else text(label, size, label_color)
    return m.VGroup(box, words.move_to(box))


def term_card(lines, indent=(), colors=None, size=20, title="terminal"):
    """kit.terminal with continuation lines indented (Text drops leading spaces) and some rows recoloured."""
    probe = terminal(lines, size=size, title=title)
    width = probe[4].width + 0.7 + (0.6 if indent else 0)
    t = terminal(lines, width=width, size=size, title=title)
    for i in indent:
        t[4][i].set_color(INK).shift(0.6 * m.RIGHT)
    for i, c in (colors or {}).items():
        t[4][i].set_color(c)
    return t


class LogChart:
    """Log–log axes drawn by hand: .p(x, y) maps data to the scene."""

    def __init__(self, x0, y0, w, h, xr, yr):
        self.x0, self.y0, self.w, self.h = x0, y0, w, h
        self.lx, self.ly = np.log10(xr), np.log10(yr)

    def p(self, x, y):
        fx = (np.log10(x) - self.lx[0]) / (self.lx[1] - self.lx[0])
        fy = (np.log10(y) - self.ly[0]) / (self.ly[1] - self.ly[0])
        return np.array([self.x0 + fx * self.w, self.y0 + fy * self.h, 0.0])


# ---------------------------------------------------------------------------- the episode


class Ep23YourMacAsAnInferenceServer(Episode):
    number = 23

    def lesson(self):
        self.hook()
        self.one_pool()
        self.what_fits()
        self.speed_line()
        self.ladder()
        self.live_run()
        self.serving()
        self.board()
        self.energy()
        self.engines()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes=f"Every measurement in this course came from this Mac Studio (2025): Apple M4 Max (16-core "
                          f"CPU, 40-core GPU), {RAM / GB:.0f} GB unified memory, {BW / GB:.0f} GB/s (Apple, Mac Studio tech specs). The "
                          f"floating numbers are measured values from research/measurements.md: decode {T8['bf16']:.1f} and {T8['4bit']:.1f} "
                          f"tok/s, prefill {PRE16:,.0f} tok/s, KV slope {KV_SLOPE:,.0f} B/token, streaming read {READ / GB:.0f} GB/s, "
                          f"{SRV[16]['aggregate_tok_s']:.0f} tok/s through the server at 16 clients, speculative decoding "
                          f"{SPEC_X:.1f}x, {E16:.2f} J per token.")
        values = [f"{T8['bf16']:.1f} tok/s", f"{T8['4bit']:.1f} tok/s", f"{PRE16:,.0f} tok/s", f"{KV_SLOPE:,.0f} B",
                  f"{READ / GB:.0f} GB/s", f"{SRV[16]['aggregate_tok_s']:.0f} tok/s", f"{SPEC_X:.1f}×", f"{E16:.2f} J"]
        spots = [(-5.2, 1.95), (-1.3, 2.1), (2.2, 1.95), (5.1, 1.75), (-5.2, -1.0), (-0.9, -1.35), (2.6, -0.45), (5.1, -1.6)]
        chips = m.VGroup()
        for v, (x, y) in zip(values, spots):
            w = code(v, 22, TOKEN)
            b = card(w.width + 0.45, 0.58, EDGE, CARD, radius=0.12, stroke=1.6)
            chips.add(m.VGroup(b, w.move_to(b)).move_to([x, y, 0]))
        studio = icon("ep23-studio", 1.7).move_to([-3.3, 0.75, 0])
        studio_t = m.VGroup(text("Mac Studio", 28, INK, weight=m.BOLD), cite("the desk behind every measurement")).arrange(m.DOWN, buff=0.1)
        studio_t.next_to(studio, m.DOWN, buff=0.3)
        specs = m.VGroup()
        for val, sub, colr in (("M4 Max", "16-core CPU · 40-core GPU", INK), (f"{RAM / GB:.0f} GB", "unified memory", INK),
                               (f"{BW / GB:.0f} GB/s", "memory bandwidth", MEMORY)):
            specs.add(m.VGroup(title_text(val, 50, colr), text(sub, 22, MUTED)).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT))
        specs.arrange(m.DOWN, buff=0.38, aligned_edge=m.LEFT).move_to([1.0, 0.6, 0], aligned_edge=m.LEFT)
        src = cite("Apple · Mac Studio (2025) tech specs").next_to(specs, m.DOWN, buff=0.3).align_to(specs, m.LEFT)
        agenda = m.VGroup(pill("what fits", WEIGHTS, BLUE_FILL, 22), pill("how fast", MEMORY, CYAN_FILL, 22),
                          pill("measured vs ceiling", TOKEN, GREEN_FILL, 22)).arrange(m.RIGHT, buff=0.4).move_to([-0.7, -2.8, 0])
        mascot = owl(0.6).move_to([5.6, -2.8, 0])

        self.say(
            "[Every number we measured in this course] [came from one desk]: [this Mac Studio.]",
            m.AnimationGroup(self.new_header("hook", "Everything on one desk"),
                             m.LaggedStart(*[m.FadeIn(c, scale=0.8) for c in chips], lag_ratio=0.12)),
            m.AnimationGroup(m.FadeIn(studio, scale=0.85),
                             m.LaggedStart(*[m.FadeOut(c, target_position=studio.get_center(), scale=0.3) for c in chips], lag_ratio=0.05)),
            m.FadeIn(studio_t, shift=0.1 * m.UP),
        )
        self.say(
            "[An M4 Max chip], [sixty-four gigabytes of unified memory], [and five hundred and forty-six gigabytes a second of bandwidth.]",
            m.FadeIn(specs[0], shift=0.15 * m.LEFT),
            m.FadeIn(specs[1], shift=0.15 * m.LEFT),
            m.AnimationGroup(m.FadeIn(specs[2], shift=0.15 * m.LEFT), m.FadeIn(src)),
        )
        self.say(
            "[Today: what fits in that memory], [how fast it runs], [and every measurement next to its ceiling.]",
            m.AnimationGroup(m.FadeIn(agenda[0], shift=0.1 * m.UP), m.FadeIn(mascot, shift=0.2 * m.UP)),
            m.FadeIn(agenda[1], shift=0.1 * m.UP),
            m.FadeIn(agenda[2], shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 1 · unified memory

    def one_pool(self):
        self.next_section("1 · One pool", notes=f"A GPU server: the H100's {H100['memory'] / GB:.0f} GB of HBM sits beside the GPU "
                          f"({H100['bandwidth'] / 1e12:.2f} TB/s); the CPU's RAM is across PCIe Gen5 ({PCIE / GB:.0f} GB/s, NVIDIA H100 "
                          "page). Apple silicon: CPU, GPU and Neural Engine share one LPDDR pool (notes §14), so nothing is copied "
                          f"between them. MLX's recommended GPU working set on this Mac is {WORKSET / GB:.2f} GB of {RAM / GB:.0f} "
                          f"(measured: mx.metal device info), leaving {OS_SPARE / GB:.1f} GB to macOS and apps (the mlx-lm README "
                          "documents raising it with sysctl iogpu.wired_limit_mb). Bandwidth: 546 GB/s vs 3.35 TB/s = "
                          f"1/{HBM_X:.1f}. Block diagrams are schematic.")
        lp = card(6.2, 4.85, EDGE, CARD, radius=0.2, stroke=2).move_to([-3.4, -0.15, 0])
        rp = card(6.2, 4.85, EDGE, CARD, radius=0.2, stroke=2).move_to([3.4, -0.15, 0])
        lt = text("a GPU server", 24, MUTED, weight=m.BOLD).align_to(lp, m.UL).shift(0.35 * m.RIGHT + 0.25 * m.DOWN)
        rt = text("this Mac", 24, MUTED, weight=m.BOLD).align_to(rp, m.UL).shift(0.35 * m.RIGHT + 0.25 * m.DOWN)

        my, uy = 0.85, -0.5
        hbm = block(2.4, 0.8, f"HBM · {H100['memory'] / GB:.0f} GB", MEMORY, CYAN_FILL).move_to([-1.75, my, 0])
        gpu = block(2.4, 0.8, "H100 GPU", COMPUTE, AMBER_FILL).move_to([-1.75, uy, 0])
        pipe = m.Line(hbm.get_bottom(), gpu.get_top()).set_stroke(MEMORY, 12)
        ram = block(1.6, 0.8, "RAM", EDGE, CARD_HI).move_to([-5.45, my, 0])
        cpu = block(1.6, 0.8, "CPU", EDGE, CARD_HI).move_to([-5.45, uy, 0])
        ram_l = m.Line(ram.get_bottom(), cpu.get_top()).set_stroke(EDGE, 5)
        pcie = m.DashedLine(cpu.get_right() + [0.08, 0, 0], gpu.get_left() - [0.08, 0, 0], dash_length=0.1).set_stroke(MUTED, 2.5)
        pcie_t = m.VGroup(code("PCIe", 20, MUTED), code(f"{PCIE / GB:.0f} GB/s", 20, MUTED)).arrange(m.DOWN, buff=0.06)
        pcie_t.move_to([pcie.get_center()[0], uy - 0.5, 0])

        pool = card(5.5, 0.8, MEMORY, CYAN_FILL, radius=0.12, stroke=2.2).move_to([3.4, my, 0])
        pool_t = code(f"unified memory · {RAM / GB:.0f} GB", 22, INK).move_to(pool)
        units = m.VGroup(block(1.3, 0.75, "CPU", EDGE, CARD_HI), block(1.3, 0.75, "GPU", COMPUTE, AMBER_FILL),
                         block(2.25, 0.75, "Neural Engine", EDGE, CARD_HI, 22, mono=False))
        for u, x in zip(units, (1.3, 3.0, 5.0)):
            u.move_to([x, uy, 0])
        wires = m.VGroup(*[m.Line([u.get_x(), pool.get_bottom()[1], 0], [u.get_x(), u.get_top()[1], 0]).set_stroke(MEMORY, 6) for u in units])
        split = pool.get_left()[0] + pool.width * WORKSET / RAM
        spare = m.Rectangle(width=pool.get_right()[0] - split - 0.06, height=pool.height - 0.14).set_stroke(width=0).set_fill(EDGE, 1)
        spare.move_to([(split + pool.get_right()[0] - 0.06) / 2, my, 0]).set_z_index(1)
        gpu_t = code(f"GPU: up to {WORKSET / GB:.1f} GB", 22, MEMORY, weight=m.BOLD).move_to([(pool.get_left()[0] + split) / 2, my, 0])
        spare_t = code(f"macOS {OS_SPARE / GB:.1f} GB", 20, MUTED).next_to(pool, m.UP, buff=0.12).align_to(pool, m.RIGHT)
        mlx_t = cite("MLX's GPU limit").next_to(pool, m.UP, buff=0.12).align_to(pool, m.LEFT)

        by = -1.95
        bw_lab_l = cite("memory bandwidth").move_to([-6.15, by + 0.45, 0], aligned_edge=m.LEFT)
        bw_lab_r = cite("memory bandwidth").move_to([0.65, by + 0.45, 0], aligned_edge=m.LEFT)
        full = 4.0
        bar_h = hbar(-6.15, by, full, 0.3, MEMORY, MEMORY)
        val_h = code(f"{H100['bandwidth'] / 1e12:.2f} TB/s", 20, MEMORY).next_to(bar_h, m.RIGHT, buff=0.15)
        bar_m = hbar(0.65, by, full / HBM_X, 0.3, MEMORY, MEMORY)
        val_m = code(f"{BW / GB:.0f} GB/s", 20, MEMORY).next_to(bar_m, m.RIGHT, buff=0.15)
        ratio = code(f"≈ 1/{HBM_X:.0f} of an H100", 20, MUTED).next_to(val_m, m.RIGHT, buff=0.3)
        src = cite("NVIDIA H100 spec · Apple Mac Studio (2025) spec · MLX on this Mac").move_to([0, -3.2, 0])
        assert ratio.get_right()[0] < rp.get_right()[0] - 0.2 and val_h.get_right()[0] < lp.get_right()[0] - 0.1

        self.say(
            "[On a GPU server, the GPU keeps its own memory]: [eighty gigabytes on an H100], [with the CPU's memory across a narrow link.]",
            m.AnimationGroup(self.new_header("01 · unified memory", "One pool for everything"), m.FadeIn(lp), m.FadeIn(lt),
                             m.FadeIn(gpu, shift=0.1 * m.UP), m.FadeIn(hbm, shift=0.1 * m.DOWN), m.Create(pipe)),
            m.Circumscribe(hbm, color=MEMORY, buff=0.08),
            m.AnimationGroup(m.FadeIn(cpu), m.FadeIn(ram), m.Create(ram_l), m.Create(pcie), m.FadeIn(pcie_t, shift=0.1 * m.UP)),
        )
        self.say(
            "[A Mac has one pool of unified memory]: [the CPU, the GPU and the Neural Engine all read the same bytes], [with nothing to copy.]",
            m.AnimationGroup(m.FadeIn(rp), m.FadeIn(rt), m.FadeIn(pool, shift=0.1 * m.DOWN), m.FadeIn(pool_t)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(u, shift=0.1 * m.UP) for u in units], lag_ratio=0.25),
                             m.LaggedStart(*[m.Create(w) for w in wires], lag_ratio=0.25)),
            m.LaggedStart(*[pulse(w, INK, 8) for w in wires], *[pulse(m.Line(w.get_end(), w.get_start()), INK, 8) for w in wires],
                          lag_ratio=0.15),
        )
        self.say(
            "[So the GPU can use most of the sixty-four gigabytes]: [MLX allows about fifty-six here], [and leaves the rest to macOS.]",
            m.Circumscribe(pool, color=MEMORY, buff=0.08),
            m.AnimationGroup(m.FadeOut(pool_t), m.FadeIn(gpu_t), m.FadeIn(mlx_t, shift=0.1 * m.DOWN)),
            m.AnimationGroup(m.GrowFromEdge(spare, m.RIGHT), m.FadeIn(spare_t, shift=0.1 * m.DOWN)),
        )
        self.say(
            "[The catch is speed]: [five hundred and forty-six gigabytes a second], [about a sixth of the H100's.]",
            m.AnimationGroup(m.FadeIn(bw_lab_l), m.GrowFromEdge(bar_h, m.LEFT), m.FadeIn(val_h, shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.FadeIn(bw_lab_r), m.GrowFromEdge(bar_m, m.LEFT), m.FadeIn(val_m, shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.FadeIn(ratio, shift=0.1 * m.LEFT), m.FadeIn(src)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 2 · what fits

    def what_fits(self):
        self.next_section("2 · What fits", notes=f"64 GB of unified memory; MLX's GPU limit {WORKSET / GB:.2f} GB (measured), the rest "
                          f"({OS_SPARE / GB:.2f} GB) for macOS. Owl-8B BF16 = {W8 / GB:.2f} GB. Owl-70B (Llama-3.1-70B, {O70['params'] / 1e9:.2f} B "
                          f"parameters): BF16 {W70[16] / GB:.1f} GB, 8-bit with scales (8.5 bits/weight) {W70[8.5] / GB:.1f} GB, 4-bit with "
                          f"scales (4.5 bits/weight) {W70[4.5] / GB:.2f} GB (owl_reference weight_bytes). KV per token = 2 × 80 layers × 8 "
                          f"KV heads × 128 × 2 B = {KV70:,.0f} B = 320 KiB; 32,768 tokens = {KV32 / GiB:.0f} GiB ({KV32 / GB:.2f} GB), total "
                          f"{(W70[4.5] + KV32) / GB:.1f} GB. Largest context under the limit: ({WORKSET / GB:.2f} − {W70[4.5] / GB:.2f}) GB ÷ "
                          f"{KV70:,} B = {MAXCTX:,.0f} tokens (an upper bound: activations and prefill buffers also need memory). 128K tokens "
                          f"= {KV128 / GiB:.0f} GiB of KV: {(W70[4.5] + KV128) / GB:.1f} GB in all. Computed, not measured (no 70B model was "
                          "downloaded or run).")
        x0, span = -3.3, 7.6
        per = span / (RAM / GB)
        gx = lambda gb: x0 + per * gb  # noqa: E731
        ys = [1.55, 0.6, -0.35, -1.6]
        h = 0.44
        tracks = m.VGroup(*[m.Rectangle(width=span, height=h).set_stroke(EDGE, 2).set_fill(CARD, 1).move_to([x0 + span / 2, y, 0]) for y in ys])
        lim = WORKSET / GB
        reserve = m.VGroup(*[m.Rectangle(width=gx(64) - gx(lim) - 0.04, height=h - 0.1).set_stroke(width=0).set_fill(EDGE, 1)
                             .move_to([(gx(lim) + gx(64)) / 2, y, 0]) for y in ys])
        limit = m.DashedLine([gx(lim), ys[0] + 0.42, 0], [gx(lim), ys[-1] - 0.32, 0], dash_length=0.1).set_stroke(MUTED, 2).set_z_index(2)
        ticks = m.VGroup(*[code(f"{v}" if v < 64 else "64 GB", 20, MUTED).move_to([gx(v), ys[0] + 0.65, 0]) for v in (0, 16, 32, 48, 64)])
        ticks[-1].shift(0.2 * m.RIGHT)
        lim_t = code(f"GPU limit {lim:.1f} GB", 20, MUTED).move_to([gx(lim) - 0.12, ys[-1] - 0.5, 0], aligned_edge=m.RIGHT)
        os_t = code(f"macOS {OS_SPARE / GB:.1f}", 20, MUTED).move_to([gx(lim) + 0.12, ys[-1] - 0.5, 0], aligned_edge=m.LEFT)

        def label(name, sub, y):
            g = m.VGroup(text(name, 22, INK, weight=m.BOLD), code(sub, 20, MUTED)).arrange(m.DOWN, buff=0.06, aligned_edge=m.LEFT)
            return g.move_to([-6.3, y, 0], aligned_edge=m.LEFT)

        labs = [label("Owl-8B", f"BF16 · {W8 / GB:.1f} GB", ys[0]), label("Owl-70B", f"BF16 · {W70[16] / GB:.0f} GB", ys[1]),
                label("Owl-70B", f"8-bit · {W70[8.5] / GB:.0f} GB", ys[2]), label("Owl-70B", f"4-bit · {W70[4.5] / GB:.1f} GB", ys[3])]

        def seg(a, b, y, color, fill, z=1):
            return m.Rectangle(width=gx(b) - gx(a), height=h - 0.1).set_stroke(color, 1.5).set_fill(fill, 1).move_to([(gx(a) + gx(b)) / 2, y, 0]).set_z_index(z)

        a_bar = seg(0, W8 / GB, ys[0], WEIGHTS, WEIGHTS)
        a_ok = m.VGroup(code("✓", 24, GREEN, weight=m.BOLD), text("plenty of room", 20, GREEN)).arrange(m.RIGHT, buff=0.12)
        a_ok.next_to(a_bar, m.RIGHT, buff=0.2)

        def spill(y, gb):
            fit = seg(0, lim, y, WEIGHTS, WEIGHTS)
            over = seg(lim, 64, y, RED, RED_FILL, 2)
            tail = m.DashedVMobject(m.Rectangle(width=0.6, height=h - 0.1), num_dashes=12).set_stroke(RED, 2).move_to([gx(64) + 0.4, y, 0])
            val = code(f"{gb:.0f} GB ✗", 20, RED, weight=m.BOLD).next_to(tail, m.RIGHT, buff=0.15)
            return fit, over, tail, val

        b_fit, b_over, b_tail, b_val = spill(ys[1], W70[16] / GB)
        c_fit, c_over, c_tail, c_val = spill(ys[2], W70[8.5] / GB)
        assert b_val.get_right()[0] < 6.6

        d_w = seg(0, W70[4.5] / GB, ys[3], WEIGHTS, WEIGHTS)
        d_ok = code("✓", 26, GREEN, weight=m.BOLD).next_to(d_w, m.RIGHT, buff=0.15)
        wa = W70[4.5] / GB
        kv0 = seg(wa, wa + 0.02, ys[3], KV, KV)
        kv32 = seg(wa, wa + KV32 / GB, ys[3], KV, KV)
        kvmax = seg(wa, lim, ys[3], KV, KV)
        w_lab = code(f"weights {wa:.1f} GB", 20, WEIGHTS).move_to([x0, ys[3] - 0.5, 0], aligned_edge=m.LEFT)
        kv_a = code(f"KV {int(KV70) // 1024} KiB per token", 20, KV).move_to([gx(lim) - 0.12, ys[3] + 0.5, 0], aligned_edge=m.RIGHT)
        kv_b = code(f"{CTX:,} tokens = {KV32 / GiB:.0f} GiB of KV", 20, KV).move_to(kv_a, aligned_edge=m.RIGHT)
        kv_c = code(f"up to ≈ {MAXCTX:,.0f} tokens", 20, KV, weight=m.BOLD).move_to(kv_a, aligned_edge=m.RIGHT)
        ghost = m.DashedVMobject(m.Rectangle(width=gx(64) + 2.0 - gx(lim), height=h - 0.1), num_dashes=40).set_stroke(RED, 2)
        ghost.move_to([(gx(lim) + gx(64) + 2.0) / 2, ys[3], 0]).set_z_index(3)
        g_t = code(f"128K: {KV128 / GiB:.0f} GiB ✗", 20, RED).move_to([6.4, ys[3] + 0.5, 0], aligned_edge=m.RIGHT)
        src = cite("computed from the model configs, not measured · GPU limit measured on this Mac").move_to([0, -3.2, 0])
        assert g_t.get_left()[0] > gx(lim) + 0.1 and kv_b.get_left()[0] > x0 - 0.2 and ghost.get_right()[0] < 6.55

        self.say(
            "[Let's fill it.] [Sixty-four gigabytes, minus about eight for macOS.]",
            m.AnimationGroup(self.new_header("02 · what fits", "Weights, cache, and macOS"), m.FadeIn(ticks),
                             m.LaggedStart(*[m.FadeIn(t) for t in tracks], lag_ratio=0.15)),
            m.AnimationGroup(m.LaggedStart(*[m.GrowFromEdge(r, m.RIGHT) for r in reserve], lag_ratio=0.1), m.Create(limit),
                             m.FadeIn(lim_t), m.FadeIn(os_t), m.FadeIn(src)),
        )
        self.say(
            "[In sixteen bits, Owl-8B takes sixteen gigabytes]: [plenty of room.]",
            m.AnimationGroup(m.FadeIn(labs[0], shift=0.1 * m.RIGHT), m.GrowFromEdge(a_bar, m.LEFT)),
            m.FadeIn(a_ok, shift=0.1 * m.LEFT),
        )
        self.say(
            "[Its big sibling, Owl-70B, needs a hundred and forty-one in sixteen bits], [and seventy-five in eight]: [neither fits.]",
            m.AnimationGroup(m.FadeIn(labs[1], shift=0.1 * m.RIGHT), m.GrowFromEdge(b_fit, m.LEFT),
                             m.Succession(m.Wait(0.6), m.AnimationGroup(m.GrowFromEdge(b_over, m.LEFT), m.Create(b_tail)))),
            m.AnimationGroup(m.FadeIn(labs[2], shift=0.1 * m.RIGHT), m.GrowFromEdge(c_fit, m.LEFT),
                             m.Succession(m.Wait(0.5), m.AnimationGroup(m.GrowFromEdge(c_over, m.LEFT), m.Create(c_tail)))),
            m.AnimationGroup(m.FadeIn(b_val, shift=0.1 * m.LEFT), m.FadeIn(c_val, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[In four bits, it's under forty gigabytes], [and it fits.]",
            m.AnimationGroup(m.FadeIn(labs[3], shift=0.1 * m.RIGHT), m.GrowFromEdge(d_w, m.LEFT), m.FadeIn(w_lab)),
            m.FadeIn(d_ok, scale=1.5),
        )
        self.say(
            "[Then the cache grows: three hundred and twenty K per token.] [A thirty-two-thousand-token chat adds ten gibibytes.]",
            m.AnimationGroup(m.FadeOut(d_ok), m.FadeIn(kv0), m.FadeIn(kv_a, shift=0.1 * m.DOWN)),
            m.AnimationGroup(m.Transform(kv0, kv32, run_time=2.0), m.FadeOut(kv_a), m.FadeIn(kv_b)),
        )
        self.say(
            "[The limit is just under fifty thousand tokens]; [a full hundred and twenty-eight K would need forty gibibytes of cache.]",
            m.AnimationGroup(m.Transform(kv0, kvmax, run_time=1.4), m.FadeOut(kv_b), m.FadeIn(kv_c)),
            m.AnimationGroup(m.Create(ghost), m.FadeIn(g_t, shift=0.1 * m.LEFT)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 3 · bandwidth ÷ size

    def speed_line(self):
        self.next_section("3 · Bandwidth ÷ size", notes="Episode 2's rule: batch-1 decode tok/s ≈ bandwidth ÷ bytes per token. On log–log "
                          "axes, tok/s = 546 GB/s ÷ size is a straight line of slope −1. Measured on this Mac (research/measurements.md §1, "
                          "median of 5, 256 tokens, greedy), size = the weight files: " + "; ".join(
                              f"{k[0]} {k[1]} {SIZE[k] / GB:.3f} GB {TOK[k]:.1f} tok/s ({SHARE[k] * 100:.0f}% of {LINE[k]:.1f})" for k in KEYS)
                          + f". Owl-70B 4-bit: {W70[4.5] / GB:.2f} GB → ceiling {CEIL70:.1f} tok/s (decode_ceiling(owl-70b, MAC, bits=4.5)); "
                          "computed, not measured. The staircase (50 → 25 GB, 10.9 → 21.8 tok/s) is illustrative.")
        ch = LogChart(-5.5, -2.55, 6.3, 4.5, (0.4, 100), (5, 1000))
        o = ch.p(0.4, 5)
        x_axis = m.Arrow(o, o + [ch.w + 0.8, 0, 0], buff=0, color=FAINT, stroke_width=3, max_tip_length_to_length_ratio=0.035)
        y_axis = m.Arrow(o, o + [0, ch.h + 0.6, 0], buff=0, color=FAINT, stroke_width=3, max_tip_length_to_length_ratio=0.05)
        xt = m.VGroup(*[code(f"{v:g}", 20, MUTED).next_to(ch.p(v, 5), m.DOWN, buff=0.12) for v in (0.5, 1, 2, 5, 10, 20, 50, 100)])
        yt = m.VGroup(*[code(f"{v:g}", 20, MUTED).next_to(ch.p(0.4, v), m.LEFT, buff=0.12) for v in (10, 30, 100, 300, 1000)])
        x_lab = text("weights read per token (GB)", 22, MUTED).next_to(xt, m.DOWN, buff=0.12).set_x(o[0] + ch.w / 2)
        y_lab = text("tokens per second", 22, MUTED).move_to(o + [0.3, ch.h + 0.45, 0], aligned_edge=m.LEFT)
        axes_g = m.VGroup(x_axis, y_axis, xt, yt, x_lab, y_lab)
        line = m.Line(ch.p(BW / 1000 / GB, 1000), ch.p(100, BW / 100 / GB)).set_stroke(MEMORY, 5)
        hi_x, lo_x = 80, 40  # halve the bytes (80 → 40 GB): the line's tokens double (6.8 → 13.7 tok/s)
        stair = m.VGroup(m.DashedLine(ch.p(hi_x, BW / hi_x / GB), ch.p(hi_x, BW / lo_x / GB), dash_length=0.07),
                         m.DashedLine(ch.p(hi_x, BW / lo_x / GB), ch.p(lo_x, BW / lo_x / GB), dash_length=0.07)).set_stroke(INK, 2.5)
        st_a = code("½ bytes", 20, INK).next_to(stair[1], m.UP, buff=0.1).shift(0.25 * m.RIGHT)
        st_b = code("2× tok/s", 20, INK).next_to(stair[0], m.RIGHT, buff=0.12)

        dots = {k: m.Dot(ch.p(SIZE[k] / GB, TOK[k]), radius=0.09, color=TOKEN).set_z_index(3) for k in KEYS}
        bits_t = m.VGroup(*[code({"4bit": "4", "8bit": "8", "bf16": "16"}[k[1]], 20, MUTED).next_to(dots[k], m.LEFT, buff=0.12) for k in KEYS])
        mini_t = text("Owl-mini", 22, TOKEN, weight=m.BOLD).next_to(dots[KEYS[1]], m.UR, buff=0.25).shift(0.15 * m.UP)
        own = text("Owl-8B", 22, TOKEN, weight=m.BOLD).next_to(dots[KEYS[4]], m.UR, buff=0.25).shift(0.15 * m.UP)

        rx = 1.5
        formula = code("tok/s ≈ bandwidth ÷ bytes", 20, INK, t2c={r"tok/s": TOKEN, r"bandwidth": MEMORY, r"bytes": WEIGHTS})
        f_box = card(formula.width + 0.5, formula.height + 0.45, EDGE, CARD, radius=0.14, stroke=2)
        f_g = m.VGroup(f_box, formula.move_to(f_box)).move_to([rx, 2.0, 0], aligned_edge=m.LEFT)
        ep2 = cite("Episode 2").next_to(f_g, m.DOWN, buff=0.12).align_to(f_g, m.LEFT)
        legend = m.VGroup(
            m.VGroup(m.Line([0, 0, 0], [0.4, 0, 0]).set_stroke(MEMORY, 5), code(f"{BW / GB:.0f} GB/s ÷ size", 20, MEMORY)).arrange(m.RIGHT, buff=0.15),
            m.VGroup(m.Dot(radius=0.09, color=TOKEN), text("measured on this Mac", 20, TOKEN)).arrange(m.RIGHT, buff=0.25),
        ).arrange(m.DOWN, buff=0.18, aligned_edge=m.LEFT).move_to([rx, 0.78, 0], aligned_edge=m.LEFT)
        names = {"bf16": ("BF16", 2), "8bit": ("8-bit", 18), "4bit": ("4-bit", 19)}
        table = m.VGroup()
        t_head = cite("Owl-8B, measured here").move_to([rx, 0.12, 0], aligned_edge=m.LEFT)
        for i, b in enumerate(("bf16", "8bit", "4bit")):
            nm, ep = names[b]
            y = -0.25 - 0.34 * i
            table.add(m.VGroup(code(nm, 20, INK).move_to([rx, y, 0], aligned_edge=m.LEFT),
                               code(f"{T8[b]:.1f} tok/s", 20, TOKEN).move_to([rx + 3.3, y, 0], aligned_edge=m.RIGHT),
                               code(f"Ep {ep}", 20, MUTED).move_to([6.4, y, 0], aligned_edge=m.RIGHT)))
        lo, hi = min(SHARE.values()), max(SHARE.values())
        band = pill(f"all six: {lo * 100:.0f}–{hi * 100:.0f}% of the line", TOKEN, GREEN_FILL, 20).move_to([rx, -2.62, 0], aligned_edge=m.LEFT)
        p70 = ch.p(W70[4.5] / GB, CEIL70)
        drop = m.DashedLine(ch.p(W70[4.5] / GB, 5), p70, dash_length=0.08).set_stroke(WEIGHTS, 2.5)
        ring = m.Circle(radius=0.12).set_stroke(WEIGHTS, 3).set_fill(BG, 1).move_to(p70).set_z_index(3)
        r70 = m.VGroup(text("Owl-70B · 4-bit", 22, WEIGHTS, weight=m.BOLD), code(f"≤ {CEIL70:.1f} tok/s · computed", 20, INK))
        r70.arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT).move_to([rx, 0, 0], aligned_edge=m.LEFT)
        r70.shift((p70[1] - r70[0].get_y()) * m.UP)
        pointer = m.DashedLine(ring.get_right() + [0.05, 0, 0], [rx - 0.15, p70[1], 0], dash_length=0.08).set_stroke(WEIGHTS, 2.5)
        assert r70.get_right()[0] < 6.6 and band.get_right()[0] < 6.6 and f_g.get_right()[0] < 6.6
        assert table[0][1].get_left()[0] > table[0][0].get_right()[0] + 0.2 and table[0][2].get_left()[0] > table[0][1].get_right()[0] + 0.2
        eight = [dots[("llama-3.1-8b", b)] for b in ("bf16", "8bit", "4bit")]

        self.say(
            "[Fitting is half the story.] [Speed is bandwidth divided by size], [as in episode two.]",
            m.AnimationGroup(self.new_header("03 · speed", "Bandwidth ÷ size"), m.FadeIn(axes_g)),
            m.FadeIn(f_g, shift=0.1 * m.DOWN),
            m.FadeIn(ep2),
        )
        self.say(
            "[On log scales, this Mac's ceiling is one straight line]: [halve the bytes, and the tokens double.]",
            m.AnimationGroup(m.Create(line, run_time=1.6), m.FadeIn(legend[0], shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.Create(stair), m.FadeIn(st_a), m.FadeIn(st_b)),
        )
        self.say(
            "[Six models measured here sit just under it]: [Owl-mini and Owl-8B, at sixty-two to ninety-one percent of the line.]",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(dots[k], scale=0.4) for k in KEYS], lag_ratio=0.15), m.FadeIn(legend[1]), m.FadeIn(bits_t)),
            m.AnimationGroup(m.FadeIn(mini_t, shift=0.1 * m.UP), m.FadeIn(own, shift=0.1 * m.UP), m.FadeIn(band, shift=0.1 * m.UP)),
        )
        self.say(
            "[Episode two measured thirty-one tokens a second in sixteen bits]; [episodes eighteen and nineteen, fifty-eight and a hundred and two.]",
            m.AnimationGroup(m.FadeIn(t_head), m.FadeIn(table[0], shift=0.1 * m.LEFT), m.Circumscribe(eight[0], color=TOKEN, buff=0.06)),
            m.LaggedStart(*[m.AnimationGroup(m.FadeIn(t, shift=0.1 * m.LEFT), m.Circumscribe(d, color=TOKEN, buff=0.06))
                            for t, d in zip(table[1:], eight[1:])], lag_ratio=0.5),
        )
        self.say(
            "[And the four-bit Owl-70B would sit here]: [at most about fourteen tokens a second], [computed, not measured.]",
            m.AnimationGroup(m.FadeOut(stair), m.FadeOut(st_a), m.FadeOut(st_b), m.Create(drop), m.FadeIn(ring, scale=0.5)),
            m.AnimationGroup(m.FadeIn(r70[0], shift=0.1 * m.LEFT), m.Create(pointer)),
            m.FadeIn(r70[1], shift=0.1 * m.LEFT),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 4 · the ladder

    def ladder(self):
        self.next_section("4 · The ladder", notes="Apple bandwidth ladder [DATED Oct 2026]: M4 120 GB/s (≤ 32 GB), M4 Pro 273 (≤ 64 GB), "
                          "M4 Max 546 (40-core GPU; ≤ 128 GB; this Mac has 64 GB), M5 Max 614 (newsroom 2026-03-03, ≤ 128 GB), M5 Ultra "
                          "1.2 TB/s (newsroom 2026-08-25, ≤ 512 GB) — notes §2. Ceilings = bandwidth ÷ weights: four-bit Owl-8B "
                          f"({W8_4 / GB:.2f} GB) " + ", ".join(f"{n} {L8[n]:.1f}" for n, *_ in LADDER) + f" tok/s; Owl-70B 4-bit ({W70[4.5] / GB:.1f} GB) "
                          + ", ".join(f"{n} {'does not fit' if L70[n] is None else f'{L70[n]:.1f}'}" for n, *_ in LADDER)
                          + f". Measured on this Mac: {T8['4bit']:.1f} tok/s for four-bit Owl-8B. Expect 55–85% of a ceiling (notes §14).")
        ys = [-1.9, -0.98, -0.02, 0.92, 1.8]
        x0, scale = -3.2, 2.8 / 1200e9
        c1, c2 = 3.6, 6.35
        heads = m.VGroup(
            cite("memory bandwidth").move_to([x0, 2.45, 0], aligned_edge=m.LEFT),
            m.VGroup(code("Owl-8B 4-bit", 20, INK), code(f"{W8_4 / GB:.2f} GB", 20, MUTED)).arrange(m.DOWN, buff=0.04, aligned_edge=m.RIGHT)
            .move_to([c1, 2.45, 0], aligned_edge=m.RIGHT),
            m.VGroup(code("Owl-70B 4-bit", 20, INK), code(f"{W70[4.5] / GB:.1f} GB", 20, MUTED)).arrange(m.DOWN, buff=0.04, aligned_edge=m.RIGHT)
            .move_to([c2, 2.45, 0], aligned_edge=m.RIGHT),
        )
        rows = []
        for (name, bw, mem, sub), y in zip(LADDER, ys):
            here = name == "M4 Max"
            lab = m.VGroup(code(name, 24, MEMORY if here else INK, weight=m.BOLD), text(sub, 20, MEMORY if here else MUTED))
            lab.arrange(m.DOWN, buff=0.05, aligned_edge=m.LEFT).move_to([-6.3, y, 0], aligned_edge=m.LEFT)
            bar = hbar(x0, y, scale * bw, 0.34, MEMORY, MEMORY if here else CYAN_FILL)
            val = code(f"{bw / GB:,.0f} GB/s", 20, INK).next_to(bar, m.RIGHT, buff=0.15)
            v8 = code(f"{L8[name]:.1f}", 24, INK).move_to([c1, y, 0], aligned_edge=m.RIGHT)
            v70 = (code("won't fit", 22, RED) if L70[name] is None else code(f"{L70[name]:.1f}", 24, INK)).move_to([c2, y, 0], aligned_edge=m.RIGHT)
            rows.append((lab, bar, val, v8, v70))
        here_box = m.RoundedRectangle(corner_radius=0.12, width=13.0, height=0.9).set_stroke(MEMORY, 2).set_fill(CYAN_FILL, 0.0)
        here_box.move_to([0, ys[2] - 0.06, 0])
        meas = code(f"measured {T8['4bit']:.1f}", 20, TOKEN).next_to(rows[2][3], m.DOWN, buff=0.07).align_to(rows[2][3], m.RIGHT)
        ten = pill(f"M4 → M5 Ultra: {LADDER[-1][1] / LADDER[0][1]:.0f}× the bandwidth, {L8['M5 Ultra'] / L8['M4']:.0f}× the tokens", MEMORY, CYAN_FILL, 20)
        ten.move_to([0, -2.6, 0])
        src = cite("Apple newsroom (Oct 2024 · Mar 2026 · Aug 2026) · ceilings in tok/s = bandwidth ÷ weights").move_to([0, -3.2, 0])
        assert rows[-1][2].get_right()[0] < heads[1].get_left()[0] and meas.get_bottom()[1] > rows[1][3].get_top()[1] + 0.05

        def show(i):
            lab, bar, val, _, _ = rows[i]
            return m.AnimationGroup(m.FadeIn(lab, shift=0.1 * m.RIGHT), m.GrowFromEdge(bar, m.LEFT), m.FadeIn(val, shift=0.1 * m.LEFT))

        self.say(
            "[Other Macs have other pipes.] [The base M4 moves a hundred and twenty gigabytes a second]; [the M4 Pro, two hundred and seventy-three.]",
            m.AnimationGroup(self.new_header("04 · the ladder", "The Mac bandwidth ladder"), m.FadeIn(heads[0])),
            show(0),
            show(1),
        )
        self.say(
            "[This M4 Max: five hundred and forty-six.] [The M5 Max, from this year: six hundred and fourteen.] [And the M5 Ultra: twelve hundred.]",
            m.AnimationGroup(show(2), m.Create(here_box)),
            show(3),
            m.AnimationGroup(show(4), m.FadeIn(src)),
        )
        self.say(
            "[Divide by the size of four-bit Owl-8B]: [from twenty-seven tokens a second on the M4], [to two hundred and sixty-six on the Ultra.]",
            m.AnimationGroup(m.FadeIn(heads[1], shift=0.1 * m.DOWN), m.LaggedStart(*[m.FadeIn(r[3], shift=0.1 * m.UP) for r in rows], lag_ratio=0.2),
                             m.FadeIn(meas)),
            m.Circumscribe(rows[0][3], color=MEMORY, buff=0.08),
            m.AnimationGroup(m.Circumscribe(rows[4][3], color=MEMORY, buff=0.08), m.FadeIn(ten, shift=0.1 * m.UP)),
        )
        self.say(
            "[The big Owl-70B doesn't even fit in the base M4's thirty-two gigabytes]; [on the others, it tops out between seven and thirty.]",
            m.AnimationGroup(m.FadeIn(heads[2], shift=0.1 * m.DOWN), m.FadeIn(rows[0][4], scale=1.3)),
            m.LaggedStart(*[m.FadeIn(r[4], shift=0.1 * m.UP) for r in rows[1:]], lag_ratio=0.2),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 5 · a live run

    def live_run(self):
        self.next_section("5 · A live run", notes=f"The stock command, as run on this Mac (research/measurements.md cross-check): "
                          f"mlx_lm.generate --model {REPO} --prompt \"$(cat prompts/chat.txt)\" --max-tokens 256 --temp 0. Output lines "
                          f"from logs/cli_crosscheck.log, the run with the median speed: Prompt: {P_TOK} tokens, {P_TPS} tokens-per-sec; "
                          f"Generation: {G_TOK} tokens, {G_TPS} tokens-per-sec; Peak memory: {PEAK} GB. (The stock tool divides N by the "
                          f"decode interval, ≈0.4% above the harness's {T8['4bit']:.1f}.) The answer's first words are the same greedy "
                          "run's text (harness, same model, prompt and temperature 0). Prefill is compute-bound, decode bandwidth-bound "
                          f"(Episode 3); peak memory = {W8_4 / GB:.2f} GB of weights + KV for {int(P_TOK) + int(G_TOK)} tokens + working buffers.")
        lines = ["$ mlx_lm.generate \\", f"--model {REPO} \\", '--prompt "$(cat prompts/chat.txt)" --max-tokens 256 --temp 0',
                 ANSWER + " …",
                 f"Prompt: {P_TOK} tokens, {P_TPS} tokens-per-sec", f"Generation: {G_TOK} tokens, {G_TPS} tokens-per-sec",
                 f"Peak memory: {PEAK} GB"]
        term = term_card(lines, indent=(1, 2), colors={3: TOKEN}, title="terminal · this Mac")
        term.move_to([0, 0.35, 0]).align_to(m.Point([0, 2.55, 0]), m.UP)
        body = term[4]
        chrome = m.VGroup(*term[:4])

        def tile(head, value, sub, color, fill):
            words = m.VGroup(m.VGroup(code(head, 22, color, weight=m.BOLD), code(value, 22, INK)).arrange(m.RIGHT, buff=0.25),
                             text(sub, 20, MUTED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
            box = card(words.width + 0.6, words.height + 0.4, color, fill, radius=0.14, stroke=2)
            words.move_to(box).align_to(box, m.LEFT).shift(0.3 * m.RIGHT)
            return m.VGroup(box, words)

        t1 = tile("prefill", f"{float(P_TPS):.0f} tok/s", f"{P_TOK} tokens · math-bound", COMPUTE, AMBER_FILL)
        t2 = tile("decode", f"{float(G_TPS):.1f} tok/s", f"{G_TOK} tokens · bandwidth-bound", MEMORY, CYAN_FILL)
        t3 = tile("memory", f"{float(PEAK):.2f} GB", f"{W8_4 / GB:.2f} GB weights + cache", WEIGHTS, BLUE_FILL)
        tiles = m.VGroup(t1, t2, t3).arrange(m.RIGHT, buff=0.35).move_to([0, -2.55, 0])
        assert tiles.width < 13.2 and term.get_bottom()[1] > tiles.get_top()[1] + 0.15 and term.width < 13.0

        self.say(
            "[Here is one real run on this Mac]: [one command with MLX], [and four-bit Owl-8B answers, token by token.]",
            m.AnimationGroup(self.new_header("05 · a live run", "One command, one Owl"), m.FadeIn(chrome)),
            m.Succession(*[m.AddTextLetterByLetter(body[i], time_per_char=0.009) for i in range(3)]),
            m.AddTextLetterByLetter(body[3], time_per_char=0.03),
        )
        self.say(
            "[The prompt: ninety-five tokens at over six hundred a second]. [That's prefill, limited by math.]",
            m.LaggedStart(*[m.FadeIn(body[i], shift=0.1 * m.UP) for i in (4, 5, 6)], lag_ratio=0.3),
            m.AnimationGroup(body[4].animate.set_color(COMPUTE), m.Circumscribe(body[4], color=COMPUTE, buff=0.06), m.FadeIn(t1, shift=0.1 * m.UP)),
        )
        self.say(
            "[The answer: two hundred and fifty-six tokens at about a hundred a second]: [decode, limited by bandwidth], [in under five gigabytes.]",
            m.AnimationGroup(body[5].animate.set_color(MEMORY), m.Circumscribe(body[5], color=MEMORY, buff=0.06)),
            m.FadeIn(t2, shift=0.1 * m.UP),
            m.AnimationGroup(body[6].animate.set_color(WEIGHTS), m.FadeIn(t3, shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 6 · serving

    def serving(self):
        self.next_section("6 · Serving", notes="research/measurements.md §10: mlx_lm.server (mlx-lm 0.32.2) with Owl-8B 4-bit, --temp 0 "
                          "--max-tokens 128, streamed through /v1/chat/completions (SSE) by a stdlib client; closed loop with C = 1 / 4 / "
                          "16 clients (10 / 20 / 48 requests), ~64-token prompts, 128 output tokens each. " + "; ".join(
                              f"C={c}: TTFT p50 {SRV[c]['ttft_ms']['p50']:.0f} / p99 {SRV[c]['ttft_ms']['p99']:.0f} ms, ITL p50 "
                              f"{SRV[c]['itl_ms']['p50']:.1f} / p99 {SRV[c]['itl_ms']['p99']:.1f} ms, aggregate {SRV[c]['aggregate_tok_s']:.0f} "
                              f"tok/s, per stream {SRV[c]['per_request_decode_tok_s']['median']:.1f}" for c in (1, 4, 16))
                          + f". The server batches concurrent requests (BatchGenerator, continuous batching). Gap floor = {W8_4 / GB:.2f} GB ÷ "
                          f"546 GB/s = {FLOOR * 1e3:.2f} ms. OpenAI-compatible APIs: Episode 22. The client dots are schematic.")
        lines = ["$ mlx_lm.server \\", f"--model {REPO} \\", "--port 8090 --temp 0 --max-tokens 128",
                 "$ python3 stream_client.py --clients 16 --requests 3"]
        term = term_card(lines, indent=(1, 2), title="terminal · this Mac")
        term.move_to([-0.95, 0, 0]).align_to(m.Point([0, 2.55, 0]), m.UP)
        body, chrome = term[4], m.VGroup(*term[:4])
        api = m.VGroup(code("POST /v1/chat/completions", 20, REQUEST), cite("OpenAI-compatible · Episode 22"))
        api.arrange(m.RIGHT, buff=0.35).next_to(term, m.DOWN, buff=0.3).align_to(term, m.LEFT)

        cols = [(-6.3, m.LEFT), (-0.9, m.RIGHT), (2.4, m.RIGHT), (4.4, m.RIGHT), (6.4, m.RIGHT)]
        hy = -0.95
        heads = m.VGroup(*[cite(h).move_to([x, hy, 0], aligned_edge=a) for h, (x, a) in
                           zip(("clients", "TTFT p50 / p99", "ITL p50 / p99", "total", "per stream"), cols)])
        rule = m.Line([-6.3, hy - 0.25, 0], [6.4, hy - 0.25, 0]).set_stroke(EDGE, 2)
        rows = {}
        for c, y in zip((1, 4, 16), (-1.52, -2.06, -2.62)):
            s = SRV[c]
            dots = m.VGroup(*[m.Dot(radius=0.065, color=REQUEST) for _ in range(c)])
            dots.arrange_in_grid(rows=1 if c < 8 else 2, buff=0.07).move_to([-5.0, y, 0], aligned_edge=m.LEFT)
            n = code(f"{c}", 22, INK, weight=m.BOLD).move_to([-6.3, y, 0], aligned_edge=m.LEFT)
            red = c == 16
            rows[c] = m.VGroup(
                m.VGroup(n, dots),
                code(f"{s['ttft_ms']['p50']:,.0f} / {s['ttft_ms']['p99']:,.0f} ms", 20, RED if red else INK).move_to([cols[1][0], y, 0], aligned_edge=m.RIGHT),
                code(f"{s['itl_ms']['p50']:.1f} / {s['itl_ms']['p99']:.1f} ms", 20, INK).move_to([cols[2][0], y, 0], aligned_edge=m.RIGHT),
                code(f"{s['aggregate_tok_s']:.0f} tok/s", 20, TOKEN).move_to([cols[3][0], y, 0], aligned_edge=m.RIGHT),
                code(f"{s['per_request_decode_tok_s']['median']:.1f}", 20, INK).move_to([cols[4][0], y, 0], aligned_edge=m.RIGHT),
            )
        floor = code(f"gap floor: {W8_4 / GB:.2f} GB ÷ 546 GB/s = {FLOOR * 1e3:.1f} ms", 20, MEMORY).move_to([-6.3, -3.25, 0], aligned_edge=m.LEFT)
        src = cite("measured on this Mac · mlx-lm 0.32.2").move_to([6.4, -3.25, 0], aligned_edge=m.RIGHT)
        mascot = owl(0.55).move_to([5.5, 1.4, 0])
        assert term.get_right()[0] < mascot.get_left()[0] - 0.2 and floor.get_right()[0] < src.get_left()[0] - 0.3
        assert api.get_bottom()[1] > heads.get_top()[1] + 0.1

        self.say(
            "[The MLX package also ships a server]: [the OpenAI-compatible endpoint from episode twenty-two.]",
            m.AnimationGroup(self.new_header("06 · serving", "A server on your desk"), m.FadeIn(chrome), m.FadeIn(mascot, shift=0.2 * m.UP),
                             m.Succession(*[m.AddTextLetterByLetter(body[i], time_per_char=0.018) for i in range(3)])),
            m.AnimationGroup(m.FadeIn(api, shift=0.1 * m.UP), m.Circumscribe(api[0], color=REQUEST, buff=0.06)),
        )
        r1 = rows[1]
        self.say(
            "[With one user, the first token arrives in seventy-two milliseconds], [then one every ten and a half.]",
            m.AnimationGroup(m.FadeIn(heads), m.Create(rule), m.FadeIn(r1[0]), m.FadeIn(r1[1], shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.FadeIn(r1[2], shift=0.1 * m.LEFT), m.FadeIn(r1[3]), m.FadeIn(r1[4]), m.FadeIn(floor)),
        )
        self.say(
            "[With sixteen users, it batches them]: [two hundred and ninety tokens a second in total], [though each stream slows to about twenty-one.]",
            m.AnimationGroup(m.AddTextLetterByLetter(body[3], time_per_char=0.018), m.FadeIn(rows[4][0]), m.FadeIn(rows[16][0]), m.FadeIn(src)),
            m.AnimationGroup(m.FadeIn(rows[4][3]), m.FadeIn(rows[16][3]), m.Circumscribe(rows[16][3], color=TOKEN, buff=0.06)),
            m.AnimationGroup(m.FadeIn(rows[4][4]), m.FadeIn(rows[16][4])),
        )
        self.say(
            "[And a newcomer waits most of a second for its first token], [while the server fits it in between everyone else's steps.]",
            m.AnimationGroup(m.FadeIn(rows[4][1]), m.FadeIn(rows[16][1]), m.Circumscribe(rows[16][1], color=RED, buff=0.06)),
            m.AnimationGroup(m.FadeIn(rows[4][2]), m.FadeIn(rows[16][2])),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 7 · the board

    def board(self):
        self.next_section("7 · The board", notes="Every measurement next to its limit (research/measurements.md, this Mac, 2026-10-24): "
                          + "; ".join(f"{r[0]} {r[1]} vs {r[2]} = {r[3] * 100:.1f}% (Episode {r[5]})" for r in BOARD)
                          + f". Limits: memory read vs Apple's 546 GB/s spec (§0); decode vs 546 GB/s ÷ the weight file (§1); prefill vs the "
                          f"measured {PEAK_MM:.1f} TFLOPS bf16 matmul (Apple publishes none, §2); KV slope vs 2·L·n_kv·d_h·2 B (§4); 32K "
                          f"context vs decode_ceiling(bits=4.5, context=32,768) = {CTX_CEIL:.1f} (§3); batch of 8 vs 8 ÷ step_time = "
                          f"{B8_CEIL:.0f} (§8); speculative decoding (BF16 target, 1B 4-bit draft, copy prompt, 6 drafts, "
                          f"{SPEC['tokens_per_target_pass']:.2f} tokens per pass) vs tokens per pass × 546 GB/s ÷ (16.06 GB + 6 × "
                          f"{DRAFT / GB:.3f} GB) = {SPEC_CEIL:.0f} (§9); one server stream vs the 4-bit ceiling (§10).")
        top, pitch = 2.35, 0.44
        cx = {"label": -6.3, "meas": -0.6, "limit": 1.95, "bar": 2.2, "pct": 5.3, "ep": 6.4}
        heads = m.VGroup(cite("measurement").move_to([cx["label"], top, 0], aligned_edge=m.LEFT),
                         code("measured", 20, TOKEN).move_to([cx["meas"], top, 0], aligned_edge=m.RIGHT),
                         code("limit", 20, MEMORY).move_to([cx["limit"], top, 0], aligned_edge=m.RIGHT),
                         cite("share of limit").move_to([cx["bar"] + 1.075, top, 0]),
                         cite("ep").move_to([cx["ep"], top, 0], aligned_edge=m.RIGHT))
        rule = m.Line([-6.3, top - 0.24, 0], [6.4, top - 0.24, 0]).set_stroke(EDGE, 2)
        bw = 2.15
        rows = []
        for i, (lab, meas, lim, share, colr, ep) in enumerate(BOARD):
            y = top - 0.55 - pitch * i
            track = m.Rectangle(width=bw, height=0.2).set_stroke(EDGE, 1.5).set_fill(CARD, 1).move_to([cx["bar"] + bw / 2, y, 0])
            fill = m.Rectangle(width=bw * share, height=0.2).set_stroke(width=0).set_fill(TOKEN, 1).move_to([cx["bar"] + bw * share / 2, y, 0])
            rows.append(m.VGroup(
                code(lab, 20, INK).move_to([cx["label"], y, 0], aligned_edge=m.LEFT),
                code(meas, 20, TOKEN).move_to([cx["meas"], y, 0], aligned_edge=m.RIGHT),
                code(lim, 20, colr).move_to([cx["limit"], y, 0], aligned_edge=m.RIGHT),
                m.VGroup(track, fill),
                code(f"{share * 100:.0f}%", 20, INK, weight=m.BOLD).move_to([cx["pct"], y, 0], aligned_edge=m.RIGHT),
                code(f"Ep {ep}", 20, MUTED).move_to([cx["ep"], y, 0], aligned_edge=m.RIGHT),
            ))
        for r in rows:
            assert r[0].get_right()[0] < r[1].get_left()[0] - 0.25 and r[2].get_right()[0] < cx["bar"] - 0.1
            assert r[4].get_left()[0] > cx["bar"] + bw + 0.1 and r[5].get_left()[0] > r[4].get_right()[0] + 0.15
        verdict = pill(f"all {len(BOARD)}: {min(PCT)}–{max(PCT)}% of their limits · none above", TOKEN, GREEN_FILL, 22).move_to([0, -2.72, 0])
        src = cite("measured on this Mac · M4 Max · mlx-lm 0.32.2 · 2026-10-24").move_to([0, -3.3, 0])
        pct_col = m.VGroup(*[r[4] for r in rows])

        def grow(r):
            return m.AnimationGroup(m.FadeIn(r[0]), m.FadeIn(r[1]), m.FadeIn(r[2]), m.FadeIn(r[3][0]), m.GrowFromEdge(r[3][1], m.LEFT),
                                    m.FadeIn(r[4]), m.FadeIn(r[5]))

        self.say(
            "[Now every measurement in this course], [on one board, next to its limit.]",
            m.AnimationGroup(self.new_header("07 · the board", "Measured vs ceiling"), m.FadeIn(src)),
            m.AnimationGroup(m.FadeIn(heads), m.Create(rule)),
        )
        self.say(
            "[Memory reads reach eighty-eight percent of Apple's spec]; [decode, eighty-four to ninety-one percent of bandwidth over bytes.]",
            grow(rows[0]),
            m.LaggedStart(*[grow(r) for r in rows[1:4]], lag_ratio=0.35),
        )
        self.say(
            "[Prefill uses ninety-six percent of the measured math roof]; [the cache grows exactly as episode five's formula says.]",
            grow(rows[4]),
            grow(rows[5]),
        )
        self.say(
            "[Long context, batching, speculative decoding and the server] [land between seventy-eight and eighty-six percent.]",
            m.LaggedStart(*[grow(r) for r in rows[6:]], lag_ratio=0.3),
            m.Circumscribe(m.VGroup(*[r[4] for r in rows[6:]]), color=TOKEN, buff=0.08),
        )
        self.say(
            "[Not one beats its limit], [and every one reaches at least three quarters of it.]",
            m.Circumscribe(pct_col, color=TOKEN, buff=0.1),
            m.FadeIn(verdict, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 8 · energy

    def energy(self):
        self.next_section("8 · Energy", notes="research/measurements.md §11: GPU-only energy from IOReport 'Energy Model' counters via "
                          "zeus-apple-silicon 1.1.0 (no sudo); CPU and DRAM channels did not update on macOS 27.0.1, so DRAM energy is not "
                          f"counted (model-based estimates, not a wall meter). Owl-8B: BF16 batch 1 {E16:.3f} J/token at "
                          f"{EN['bf16']['decode_b1_gpu_W']:.1f} W; 4-bit batch 1 {E4:.3f} J at {EN['4bit']['decode_b1_gpu_W']:.1f} W; BF16 "
                          f"batch 16 {E16B:.3f} J per generated token (prefill share removed) at {EN['bf16']['decode_b16_gpu_W']:.1f} W; "
                          f"prefill {EIN * 1000:.1f} mJ per prompt token ({E16 / EIN:.1f}x less than an output token). API prices charge "
                          "output 4–6x input: Episode 24.")
        x0, per = -2.6, 5.6 / 0.65
        ys = [1.4, 0.6, -0.2]
        rows = []
        for (lab, sub, j, w, best), y in zip(
                (("BF16", "1 user", E16, EN["bf16"]["decode_b1_gpu_W"], False), ("4-bit", "1 user", E4, EN["4bit"]["decode_b1_gpu_W"], False),
                 ("BF16", "16 users", E16B, EN["bf16"]["decode_b16_gpu_W"], True)), ys):
            name = m.VGroup(code(lab, 22, WEIGHTS, weight=m.BOLD), code(sub, 20, MUTED)).arrange(m.RIGHT, buff=0.2)
            name.move_to([-6.3, y, 0], aligned_edge=m.LEFT)
            bar = hbar(x0, y, per * j, 0.38, MEMORY, MEMORY if best else CYAN_FILL)
            val = m.VGroup(code(f"{j:.2f} J", 24, GREEN if best else INK, weight=m.BOLD), code(f"GPU {w:.0f} W", 20, MUTED)).arrange(m.RIGHT, buff=0.25)
            val.next_to(bar, m.RIGHT, buff=0.2)
            rows.append((name, bar, val))
        cap = cite("joules per generated token · decode").move_to([x0, 2.1, 0], aligned_edge=m.LEFT)

        iy = -1.65
        in_t = cite("BF16 · per token").move_to([x0, iy + 0.6, 0], aligned_edge=m.LEFT)
        rd = m.VGroup(code("read", 22, COMPUTE, weight=m.BOLD), code("prompt", 20, MUTED)).arrange(m.RIGHT, buff=0.2).move_to([-6.3, iy, 0], aligned_edge=m.LEFT)
        rd_bar = hbar(x0, iy, per * EIN, 0.38, COMPUTE, AMBER_FILL)
        rd_val = code(f"{EIN * 1000:.0f} mJ", 22, COMPUTE).next_to(rd_bar, m.RIGHT, buff=0.2)
        wr = m.VGroup(code("write", 22, MEMORY, weight=m.BOLD), code("output", 20, MUTED)).arrange(m.RIGHT, buff=0.2).move_to([-6.3, iy - 0.7, 0], aligned_edge=m.LEFT)
        wr_bar = hbar(x0, iy - 0.7, per * E16, 0.38, MEMORY, CYAN_FILL)
        wr_val = code(f"{E16 * 1000:.0f} mJ", 22, MEMORY).next_to(wr_bar, m.RIGHT, buff=0.2)
        ratio = pill(f"≈ 1/{E16 / EIN:.0f} · prices point the same way · Episode 24", COMPUTE, AMBER_FILL, 20)
        ratio.next_to(rd_val, m.RIGHT, buff=0.35)
        caveat = cite("GPU only (IOReport counters via zeus-apple-silicon) · memory chips not counted · measured on this Mac").move_to([0, -3.2, 0])
        assert ratio.get_right()[0] < 6.6 and rows[0][2].get_right()[0] < 6.6

        self.say(
            "[One last number: energy.] [On this GPU, a sixteen-bit Owl token costs about six tenths of a joule.]",
            m.AnimationGroup(self.new_header("08 · energy", "Joules per token"), m.FadeIn(cap)),
            m.AnimationGroup(m.FadeIn(rows[0][0]), m.GrowFromEdge(rows[0][1], m.LEFT), m.FadeIn(rows[0][2], shift=0.1 * m.LEFT), m.FadeIn(caveat)),
        )
        self.say(
            "[Four bits roughly halve it]; [sixteen users batched together bring it under a fifth.]",
            m.AnimationGroup(m.FadeIn(rows[1][0]), m.GrowFromEdge(rows[1][1], m.LEFT), m.FadeIn(rows[1][2], shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.FadeIn(rows[2][0]), m.GrowFromEdge(rows[2][1], m.LEFT), m.FadeIn(rows[2][2], shift=0.1 * m.LEFT)),
        )
        self.say(
            "[Reading a prompt token costs about a tenth of writing one]: [the same direction as the prices in episode twenty-four.]",
            m.AnimationGroup(m.FadeIn(in_t), m.FadeIn(wr), m.GrowFromEdge(wr_bar, m.LEFT), m.FadeIn(wr_val),
                             m.Succession(m.Wait(0.5), m.AnimationGroup(m.FadeIn(rd), m.GrowFromEdge(rd_bar, m.LEFT), m.FadeIn(rd_val)))),
            m.FadeIn(ratio, shift=0.1 * m.LEFT),
        )
        self.say(
            "[These count the GPU alone, not the memory chips], [so the true cost per token is higher.]",
            m.Circumscribe(caveat, color=MUTED, buff=0.08),
            m.Indicate(caveat, color=INK, scale_factor=1.05),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 9 · engines

    def engines(self):
        self.next_section("9 · Two engines", notes="MLX: Apple's array framework (lazy evaluation, unified-memory native); mlx-lm adds generate, "
                          "a server and batching; models from the mlx-community Hugging Face org, 4-bit = group 64 with a scale and bias "
                          "(notes §14). llama.cpp: 'LLM inference in C/C++' with a Metal backend, GGUF files and k-quants (Episode 19), "
                          "memory-mapped weights; Ollama and LM Studio build on these engines; its README: 'Apple silicon is a "
                          "first-class citizen'. No llama.cpp numbers were measured on this Mac, so the engines are not ranked; "
                          "measurements.md notes that other engines will differ, especially for batching, quantized KV and long context.")

        def eng(cx, title, sub, bullets, tag):
            box = card(6.15, 4.1, EDGE, CARD, radius=0.2, stroke=2).move_to([cx, 0.1, 0])
            head = m.VGroup(title_text(title, 38, INK), cite(sub)).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
            head.align_to(box, m.UL).shift(0.35 * m.RIGHT + 0.3 * m.DOWN)
            items = m.VGroup(*[text(b, 22, INK) for b in bullets]).arrange(m.DOWN, buff=0.22, aligned_edge=m.LEFT)
            items.next_to(head, m.DOWN, buff=0.35).align_to(head, m.LEFT)
            tag.next_to(items, m.DOWN, buff=0.35).align_to(head, m.LEFT)
            return m.VGroup(box, head, items, tag)

        mlx = eng(-3.3, "MLX", "Apple's array framework",
                  ["native to unified memory", "mlx-lm: generate, server, batching", "4-bit models in groups of 64"],
                  pill("every number in this course", TOKEN, GREEN_FILL, 20))
        cpp = eng(3.3, "llama.cpp", "LLM inference in C/C++",
                  ["GPU kernels for Apple's Metal", "GGUF files and k-quants · Episode 19", "under apps like Ollama"],
                  m.VGroup(cite("“Apple silicon is a first-class citizen”"), cite("llama.cpp README")).arrange(m.DOWN, buff=0.06, aligned_edge=m.LEFT))
        fair = pill("not measured here: compare them yourself, same model, same bits", INK, CARD_HI, 20).move_to([0, -2.95, 0])
        for g in (mlx, cpp):
            assert g[3].get_bottom()[1] > g[0].get_bottom()[1] + 0.15 and g[2].get_right()[0] < g[0].get_right()[0] - 0.2

        self.say(
            "[Two engines run most local models on a Mac.] [MLX is Apple's own array framework], [and every number here came from it.]",
            m.AnimationGroup(self.new_header("09 · engines", "MLX and llama.cpp"), m.FadeIn(mlx[0]), m.FadeIn(cpp[0])),
            m.AnimationGroup(m.FadeIn(mlx[1], shift=0.1 * m.UP), m.LaggedStart(*[m.FadeIn(b, shift=0.1 * m.LEFT) for b in mlx[2]], lag_ratio=0.3)),
            m.FadeIn(mlx[3], shift=0.1 * m.UP),
        )
        self.say(
            "[The other is llama.cpp]: [it runs the GGUF files from episode nineteen on the Mac's GPU], [and sits under apps like Ollama.]",
            m.FadeIn(cpp[1], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(cpp[2][0], shift=0.1 * m.LEFT), m.FadeIn(cpp[2][1], shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.FadeIn(cpp[2][2], shift=0.1 * m.LEFT), m.FadeIn(cpp[3])),
        )
        self.say(
            "[We only measured MLX, so we won't rank them]: [both live under the same bandwidth line.]",
            m.FadeIn(fair, shift=0.1 * m.UP),
            m.AnimationGroup(m.Circumscribe(mlx[0], color=MEMORY, buff=0.06), m.Circumscribe(cpp[0], color=MEMORY, buff=0.06)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes=f"Fit: weights + KV + macOS within {RAM / GB:.0f} GB (MLX's GPU limit {WORKSET / GB:.1f} GB): Owl-70B "
                          f"4-bit + 32K tokens = {(W70[4.5] + KV32) / GB:.1f} GB. Divide: tok/s ≈ bandwidth ÷ bytes ({BW / GB:.0f} GB/s ÷ "
                          f"{W8_4 / GB:.2f} GB = {L8['M4 Max']:.1f}). Measure: {T8['4bit']:.1f} tok/s measured; every measurement on this Mac "
                          f"at {min(PCT)}–{max(PCT)}% of its limit. Next: Episode 24, The Price of a Token.")
        words = [("Fit", "weights + KV + macOS", WEIGHTS), ("Divide", "bandwidth ÷ bytes", MEMORY), ("Measure", "against the ceiling", TOKEN)]
        cols = m.VGroup()
        for w, sub, colr in words:
            cols.add(m.VGroup(title_text(w, 72, colr), code(sub, 20, MUTED)).arrange(m.DOWN, buff=0.25))
        cols.arrange(m.RIGHT, buff=1.3).move_to([0, 0.75, 0])
        base = 0.85
        for c in cols:
            c[0].shift((base - c[0].get_bottom()[1]) * m.UP)
            c[1].move_to([c[0].get_x(), base - 0.5, 0])
        mid = base + 0.3
        arrows = m.VGroup(*[arrow([a[0].get_right()[0] + 0.15, mid, 0], [b[0].get_left()[0] - 0.15, mid, 0], MUTED) for a, b in zip(cols, cols[1:])])
        tagline = text("Memory decides what runs. Bandwidth decides how fast.", 30, INK,
                       t2c={"Memory decides what runs.": WEIGHTS, "Bandwidth decides how fast.": MEMORY}).move_to([0, -1.35, 0])
        mascot = owl(0.75).move_to([0, -2.7, 0])

        self.say(
            "[So: fit], [divide], [measure.]",
            m.AnimationGroup(self.new_header("recap", "Your Mac in three words"), m.FadeIn(cols[0], shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowArrow(arrows[0]), m.FadeIn(cols[1], shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowArrow(arrows[1]), m.FadeIn(cols[2], shift=0.2 * m.UP)),
        )
        self.say(
            "[Memory decides what runs on your Mac]; [bandwidth decides how fast.]",
            m.AnimationGroup(m.FadeIn(tagline, shift=0.1 * m.UP), m.FadeIn(mascot, shift=0.2 * m.UP)),
            m.Circumscribe(tagline, color=MEMORY, buff=0.1),
        )
        self.wait(0.4)
        self.clear()
