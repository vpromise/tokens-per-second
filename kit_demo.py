"""A tour of the "Tokens per Second" kit: every component on screen once, with the course's real numbers.
VOICE=draft manimgx inspect kit_demo.py   → KitDemo.storyboard*.png"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kit import *  # noqa: E402,F403
from owl_reference import *  # noqa: E402,F403


class KitDemo(Episode):
    number = 2

    def lesson(self):
        # 1 · the roofline, with Owl's decode and prefill on it
        roof = Roofline(GPU["bandwidth"], GPU["peak"], width=6.6, height=4.0, center=(-1.7, -0.5, 0))
        decode = roof.dot(1, MEMORY)
        prefill = roof.dot(1500, COMPUTE)
        d_lab = code("decode · batch 1", 20, MEMORY).next_to(decode, m.DR, buff=0.12)
        p_lab = code("prefill\n1,500 tokens", 20, COMPUTE).next_to(prefill, m.DOWN, buff=0.25).align_to(prefill, m.LEFT)
        ridge_lab = roof.ridge_label()
        side = m.VGroup(
            stat(f"{decode_ceiling(hw=GPU):.0f}", "tok/s", "H100 · 8B BF16 ceiling", MEMORY, 64),
            stat(f"{decode_ceiling(hw=MAC):.0f}", "tok/s", "M4 Max · same model", MEMORY, 64),
        ).arrange(m.DOWN, buff=0.55, aligned_edge=m.LEFT).move_to([4.75, -0.4, 0])
        self.say(
            "[The roofline]: [memory slope in cyan, compute roof in amber], [and two workloads on it.]",
            self.new_header("kit · 1", "Roofline, stats"),
            m.AnimationGroup(m.FadeIn(roof.axes), m.Create(roof.memory), m.Create(roof.compute), m.FadeIn(roof.ticks),
                             m.FadeIn(roof.x_label), m.Create(roof.ridge), m.FadeIn(ridge_lab)),
            m.AnimationGroup(m.FadeIn(decode, scale=0.5), m.FadeIn(d_lab), m.FadeIn(prefill, scale=0.5), m.FadeIn(p_lab),
                             m.FadeIn(side, shift=0.2 * m.LEFT)),
        )
        self.clear()

        # 2 · memory bar, timeline, tokens
        bar = MemoryBar(GPU["memory"], width=11.0, title="H100 · 80 GB of HBM").move_to([0, 2.0, 0])
        w_lab = code("weights 16 GB", 20, WEIGHTS)
        kv_lab = code("KV cache", 20, KV)
        line = Timeline(2.0, width=11.0, origin=(-5.5, -0.9, 0), label_every=0.5)
        pre = line.bar(0, 0.049, COMPUTE)
        pre_lab = code("prefill 49 ms", 20, COMPUTE).next_to(pre, m.UP, buff=0.3).align_to(pre, m.LEFT)
        ticks = m.VGroup(*[line.tick(0.049 + 0.0065 * i, MEMORY) for i in range(0, 300, 6)])
        dec_lab = code("300 decode steps · 6.5 ms each", 20, MEMORY).next_to(ticks, m.UP, buff=0.3).align_to(line.axis, m.RIGHT)
        toks = token_row(["The", " KV", " cache", " stores", " keys", " and", " values"], size=20)
        toks.move_to([0, -2.85, 0])
        self.say(
            "[A memory bar], [a timeline], [and tokens.]",
            m.AnimationGroup(self.new_header("kit · 2", "Memory, time, tokens"), m.FadeIn(bar.track), m.FadeIn(bar.caption)),
            m.AnimationGroup(bar.set("weights", weight_bytes(), WEIGHTS), bar.set("kv", 30e9, KV),
                             m.FadeIn(line), m.FadeIn(pre), m.FadeIn(pre_lab), m.LaggedStart(*[m.FadeIn(t) for t in ticks], lag_ratio=0.02),
                             m.FadeIn(dec_lab)),
            m.LaggedStart(*[m.FadeIn(t, shift=0.15 * m.UP) for t in toks], lag_ratio=0.1),
        )
        w_lab.next_to(bar.segments["weights"], m.DOWN, buff=0.15)
        kv_lab.next_to(bar.segments["kv"], m.DOWN, buff=0.15)
        self.say("[Labels under each segment.]", m.AnimationGroup(m.FadeIn(w_lab), m.FadeIn(kv_lab)))
        self.clear()

        # 3 · cards, grid, owl
        grid = cell_grid(4, 12, size=0.3).move_to([-4.4, 1.35, 0])
        for cell in grid[0][:7]:  # one sequence's KV blocks
            cell.set_fill(VIOLET_FILL, 1).set_stroke(VIOLET, 1.5)
        src = ("logits, cache = model.prefill(prompt)\n"
               "for _ in range(max_new_tokens):\n"
               "    token = sample(logits)\n"
               "    logits = model.decode(token, cache)")
        block = code_block(src, 20, title="owl_reference.py").move_to([2.75, 1.35, 0])
        term = terminal(["$ mlx_lm.generate --model owl-mini", "Generation: 256 tokens, 312 tok/s", "! for example only"],
                        width=5.6, size=18).move_to([-3.6, -2.0, 0])
        jc = json_card({"front": "What is in the KV cache?", "back": "keys and values", "tokens": 6}, 18,
                       title="Owl's flashcard").move_to([3.2, -2.0, 0])
        mascot = owl(0.8).move_to([-1.55, 1.35, 0])
        self.say(
            "[A grid], [code and a terminal], [JSON], [and Owl.]",
            m.AnimationGroup(self.new_header("kit · 3", "Cards and Owl"), m.FadeIn(grid), m.FadeIn(block)),
            m.FadeIn(term),
            m.FadeIn(jc),
            m.FadeIn(mascot, shift=0.2 * m.UP),
        )
        self.clear()
