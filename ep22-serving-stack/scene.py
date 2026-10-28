"""Tokens per Second · Episode 22 · The Serving Stack (Module 6 · Production & Your Mac).

Module 6 opens by zooming out. Episode 1 followed one request inside the engine; this episode follows a
request down the whole stack between Owl's student and the GPU — client → router → OpenAI-compatible API
server → scheduler → engine → GPU — and back up as server-sent events. Each layer is where an earlier
episode's idea lives, linked by number and never re-taught: Episode 1 (tokenize and stream), 8 (the prefix
cache, hence prefix-aware routing), 10 (continuous batching), 17 (sampling and grammar masks), 12 and 21
(deployment shapes). Then what only the fleet adds: routing policies, the engines, autoscaling, cold
starts (16 GB of weights ÷ disk bandwidth), and multi-LoRA serving.

The running example is a real request: the one tools/stream_client.py sent to mlx_lm.server on this Mac
(research/measurements.md §10, Owl-8B 4-bit; its tracking tag "[client 0 request 0]" dropped), with the
real first pieces of its streamed answer and its real usage chunk.

Sources (facts dated October 2026; research/notes.md §13 and §17, plus the primary pages read 2026-10-24):
- OpenAI-compatible streaming (notes §13, OpenAI OpenAPI spec): stream: true → text/event-stream, one
  chat.completion.chunk per piece with a delta, a final chunk with finish_reason, an optional usage chunk
  (stream_options.include_usage) with empty choices, then `data: [DONE]`. The chunk fields shown follow
  mlx_lm/server.py 0.32.2 (generate_response / completion_usage_response; system_fingerprint, model and
  created omitted, the uuid shortened to "chatcmpl-…").
- Routing (notes §13): OpenAI's prompt-caching docs route by "A hash of the initial tokens"; llm-d and
  NVIDIA Dynamo do prefix-aware / KV-aware routing. Stake measured on this Mac (Episode 8, measurements §7):
  Owl-8B 4-bit, 6,910-token chapter prefix: TTFT 8.01 s without the cache, 0.137 s with it.
- Engines and status (notes §13, GitHub releases 2026-10-23): vLLM v0.32.0, SGLang v0.5.22, TensorRT-LLM
  v1.2.1, llama.cpp, mlx-lm v0.32.2; Hugging Face TGI "is now in maintenance mode" (2025-12-11), archived
  2026-03-21.
- Server vs engine on this Mac (measurements §8, §10): BatchGenerator (the engine behind mlx_lm.server) at
  batch 1: 101.3 tok/s (9.87 ms per step, 256-token prompts); through mlx_lm.server, one client: 94.0
  tok/s per stream, ITL p50 10.6 ms (64-token prompts); ceiling 546 GB/s ÷ 4.52 GB = 121 tok/s.
- Autoscaling (Google Cloud, GKE "Best practices for autoscaling LLM inference workloads with GPUs",
  updated 2026-10-22, notes §13): queue size / batch size; GPU utilization "Does not measure how much work
  is being done while the GPU is active."
- Cold start (NVIDIA technical blog, "Reducing Cold Start Latency for LLM Inference with NVIDIA Run:ai Model
  Streamer", 2025-09-16, read 2026-10-24): Llama 3 8B, 15 GB, one A10G of a g5.12xlarge, GP3 SSD
  (1,000 MiB/s): HF safetensors loader 47.99 s, streamer 14.34 s (concurrency 16); total until the vLLM
  engine is ready 66.13 s vs 35.08 s.
- Sleep mode (vLLM blog, 2025-10-26, read 2026-10-24; A100, Qwen3-0.6B and Phi-3-vision): level 1 wake
  0.26 / 0.82 s, level 2 0.85 / 2.58 s; without sleep 37.6 / 58.1 s per switch ("~48s" average).
- Multi-LoRA (notes §17): S-LoRA, Sheng et al. 2023 (arXiv 2311.03285): "thousands of LoRA adapters on a
  single GPU", throughput "up to 4 times" vs HuggingFace PEFT and vLLM with naive LoRA support.

    VOICE=draft manimgx inspect scene.py      # silent layout pass
    manimgx render scene.py                   # with the shared narrator running
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from owl_reference import *  # noqa: E402,F403

PREFIX = "ep22"
RESEARCH = SERIES / "research" / "measurements"

# ---------------------------------------------------------------------------- the numbers (all computed)

# the request, exactly as the stream test sent it to mlx_lm.server on this Mac (measurements §10)
SRV = measured("server", "llama-3.1-8b__4bit")
ONE = SRV["1"]
RAW = json.loads((RESEARCH / "raw" / "server__llama-3.1-8b__4bit__c1.json").read_text())
REQ0 = RAW["requests"][0]
USAGE = REQ0["usage"]
assert round(RAW["ttft_ms"]["p50"], 6) == round(ONE["ttft_ms"]["p50"], 6)          # the same run
assert USAGE == {"prompt_tokens": 64, "completion_tokens": 128, "total_tokens": 192,
                 "prompt_tokens_details": {"cached_tokens": 63}}
assert REQ0["chunks"] == 128 == ONE["max_tokens"] and REQ0["finish_reason"] == "length"
CLIENT_SRC = (RESEARCH / "scripts" / "stream_client.py").read_text()
QUESTION = "Write a detailed, 400-word explanation of how volcanoes form for a curious 12-year-old."
assert all(s in CLIENT_SRC for s in ('"how volcanoes form"', "Write a detailed, 400-word explanation of ",
                                     "for a curious 12-year-old.", '"stream": True', '"include_usage": True',
                                     '"temperature": 0.0'))
REQ_BODY = {"messages": [{"role": "user", "content": QUESTION}], "max_tokens": 128, "temperature": 0.0,
           "stream": True, "stream_options": {"include_usage": True}}
WRAP = ["Write a detailed,", "400-word explanation of how", "volcanoes form for a curious", "12-year-old."]
assert " ".join(WRAP) == QUESTION
REQ_SRC = ('{\n'
           '  "messages": [{"role": "user",\n'
           f'    "content": "{WRAP[0]}\n'
           f'      {WRAP[1]}\n'
           f'      {WRAP[2]}\n'
           f'      {WRAP[3]}"}}],\n'
           '  "max_tokens": 128,\n'
           '  "temperature": 0.0,\n'
           '  "stream": true,\n'
           '  "stream_options":\n'
           '    {"include_usage": true}\n'
           '}')
assert json.loads(REQ_SRC.replace("\n      ", " ")) == REQ_BODY                    # it is the real request

# the first pieces of its streamed answer: one SSE chunk per token. Token split from Llama-3.1's
# tokenizer.json in the local HF cache, read with the `tokenizers` library (no model run), 2026-10-24.
PIECES = ["Hey", " there", ",", " young", " adventurer", "!"]
assert REQ0["text"].startswith("".join(PIECES))
USAGE_SRC = ('{"prompt_tokens": 64,\n'
             ' "completion_tokens": 128,\n'
             ' "prompt_tokens_details":\n'
             '   {"cached_tokens": 63}}')
assert all(USAGE[k] == v for k, v in json.loads(USAGE_SRC).items())                # total_tokens left out

# through the server vs the engine alone, one stream, Owl-8B 4-bit, this Mac
PER1 = ONE["per_request_decode_tok_s"]["median"]
ITL1 = ONE["itl_ms"]["p50"]
ENG1 = measured("batching", "llama-3.1-8b", "4bit", "by_batch", "1", "aggregate_tok_s")
ENG_STEP = measured("batching", "llama-3.1-8b", "4bit", "by_batch", "1", "step_ms")
Q4_GB = measured("models", "llama-3.1-8b", "4bit", "weights_gb")
CEIL = MAC["bandwidth"] / (Q4_GB * 1e9)
COST = 1 - PER1 / ENG1
MLX_LM = measured("software", "mlx_lm")
assert round(CEIL) == 121 == round(measured("decode", "llama-3.1-8b", "4bit", "ceiling_tok_s"))
assert (round(PER1), round(ENG1), round(100 * COST)) == (94, 101, 7)   # "ninety-four", "a hundred and one", "seven percent"
assert round(ITL1, 1) == 10.6 and round(ENG_STEP, 1) == 9.9 and PER1 < ENG1 < CEIL and MLX_LM == "0.32.2"

# Episode 8's prompt cache, measured on this Mac: the stake of routing to the right replica
PC = measured("prompt_cache", "llama-3.1-8b", "4bit")
assert PC["prefix_tokens"] == 6910 and round(PC["ttft_no_cache_s"], 1) == 8.0     # "eight seconds"
assert round(PC["ttft_with_cache_s"] * 1000) == 137                                # "a hundred and thirty-seven milliseconds"

# cold starts: weights ÷ disk bandwidth
W_BYTES = weight_bytes()                                    # Owl-8B, BF16: 16.06 GB
NVME = 5e9                                                  # for example: a fast local NVMe drive (notes §13)
T_NVME = W_BYTES / NVME
assert round(W_BYTES / 1e9) == 16 and round(T_NVME, 1) == 3.2                      # "three point two seconds"
RUNAI = dict(gb=15e9, disk=1000 * 2 ** 20, hf_load=47.99, stream_load=14.34, hf_ready=66.13, stream_ready=35.08)
T_DISK = RUNAI["gb"] / RUNAI["disk"]                        # what the GP3 disk allows
assert round(T_DISK, 1) == 14.3 and abs(T_DISK - RUNAI["stream_load"]) < 0.05      # "just what the disk allows"
assert round(RUNAI["hf_ready"]) == 66 and round(RUNAI["hf_load"]) == 48 and round(RUNAI["disk"] / 1e9, 2) == 1.05
SLEEP = dict(l1=(0.26, 0.82), l2=(0.85, 2.58), reload=(37.6, 58.1))
assert max(SLEEP["l1"] + SLEEP["l2"]) < 3 and round(sum(SLEEP["reload"]) / 2) == 48  # "under three seconds, not forty-eight"

# multi-LoRA: a rank-16 adapter on q, k, v, o of every layer (for example)
RANK = 16


def lora_params(model=OWL, r=RANK):
    """A LoRA adapter adds B·A (out × r and r × in) beside each adapted matrix: r × (in + out) numbers."""
    d, q_out, kv = model["d_model"], model["heads"] * model["head_dim"], model["kv_heads"] * model["head_dim"]
    shapes = [(d, q_out), (d, kv), (d, kv), (q_out, d)]     # q, k, v, o projections
    return model["layers"] * sum(r * (a + b) for a, b in shapes)


LORA_P = lora_params()
LORA_BYTES = 2 * LORA_P                                     # BF16
assert LORA_P == 13_631_488 and round(LORA_BYTES / 1e6) == 27                      # "about twenty-seven megabytes"
COPIES = 4
assert round(COPIES * W_BYTES / 1e9) == 64 and GPU["memory"] == 80e9                # "sixty-four of eighty"
PER_COPY = W_BYTES / LORA_BYTES
assert 580 < PER_COPY < 600
SLORA_X = 4

# icons
USER = recolored("user", REQUEST, PINK_FILL, PREFIX)
GPU_ICON = recolored("gpu", MUTED, CARD, PREFIX)
SERVER_ICON = recolored("server", MUTED, CARD, PREFIX)
GEAR = recolored("gear", MUTED, CARD, PREFIX)
LLM = recolored("llm", MUTED, CARD, PREFIX)
ROUTER = "ep22-router"

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


def ep_pill(n, size=20):
    """An earlier episode, by number, in its module's colour."""
    _, accent, fill = MODULES[EPISODES[n][1]]
    return pill(f"EP {n:02d}", accent, fill, size, mono=True)


def ep_link(n, words, x, y, size=22):
    p = left_at(ep_pill(n), x, y)
    w = text(words, size, INK).next_to(p, m.RIGHT, buff=0.25)
    return m.VGroup(p, w)


JSON_RULES = {
    r'"Write[^\n]*|^\s+400-word[^\n]*|^\s+volcanoes form[^\n]*|12-year-old\."': GREEN,  # the soft-wrapped question
    r"^data:": MUTED,
    r"…": MUTED,
    r"\[DONE\]": VIOLET,
    r'"[^"\n]*"(?=\s*:)': CYAN,
    r'"[^"\n]*"': GREEN,
    r"\b-?\d+(\.\d+)?\b": AMBER,
    r"\b(true|false|null)\b": VIOLET,
}


def jcard(src, title, accent, size=20, width=None):
    """JSON (or SSE lines) in a card with a mono title: keys CYAN, strings GREEN, numbers AMBER."""
    body = code(src, size, INK, t2c=JSON_RULES)
    cap = code(title, 20, accent, weight=m.BOLD)
    top = cap.height + 0.22
    w = max(width or 0, body.width + 0.7, cap.width + 0.7)
    box = card(w, body.height + 0.55 + top, accent, CARD, radius=0.16, stroke=2)
    body.move_to(box).align_to(box, m.LEFT).shift(0.35 * m.RIGHT + top / 2 * m.DOWN)
    cap.align_to(box, m.UL).shift(0.35 * m.RIGHT + 0.17 * m.DOWN)
    return m.VGroup(box, body, cap)


def line_of(words, src, k):
    """The glyphs of line k of a mono Text made from `src`."""
    lines = src.split("\n")
    start = sum(len(s) + 1 for s in lines[:k])
    skip = sum(c.isspace() for c in src[:start])
    width = sum(not c.isspace() for c in lines[k])
    return words[start - skip: start - skip + width]


def gutter(box, part, color):
    """A bar in a card's left margin beside some lines."""
    return m.RoundedRectangle(corner_radius=0.03, width=0.08, height=part.height + 0.06, stroke_width=0,
                              fill_color=color, fill_opacity=1).move_to([box.get_left()[0] + 0.17, part.get_y(), 0])


def packet(start, end, color=REQUEST, run_time=0.8, radius=0.09):
    """A dot flying from start to end, vanishing there (one cue)."""
    dot = m.Dot(radius=radius, color=color).move_to(end).set_z_index(6)
    out = m.FadeOut(dot, scale=0.5, run_time=0.25)
    out.introducer = True  # the dot is introduced by the FadeIn just before, in the same cue
    return m.Succession(m.FadeIn(dot, target_position=start, run_time=run_time), out)


def chip_up(start, end, color=TOKEN, run_time=0.9):
    """A small token square travelling up a rail."""
    sq = m.Square(0.2).set_stroke(color, 2).set_fill(GREEN_FILL, 1).move_to(end).set_z_index(6)
    out = m.FadeOut(sq, scale=0.5, run_time=0.2)
    out.introducer = True
    return m.Succession(m.FadeIn(sq, target_position=start, run_time=run_time), out)


def window(width, height, title, center):
    """Owl's app: a card with a title bar (a small Owl and a caption)."""
    box = card(width, height, EDGE, "#0F191D", radius=0.18, stroke=2).move_to(center)
    bar_y = box.get_top()[1] - 0.46
    bar = m.Line([box.get_left()[0], bar_y, 0], [box.get_right()[0], bar_y, 0]).set_stroke(EDGE, 2)
    mascot = owl(0.26).move_to([box.get_left()[0] + 0.38, bar_y + 0.23, 0])
    name = text(title, 20, MUTED).next_to(mascot, m.RIGHT, buff=0.2)
    name.set_y(bar_y + 0.23)
    return m.VGroup(box, bar, mascot, name)


# ---------------------------------------------------------------------------- the stack

LAYERS = [("client", "Owl's app", USER), ("router", "picks a replica", ROUTER),
          ("API server", "OpenAI-compatible", SERVER_ICON), ("scheduler", "forms the batch", GEAR),
          ("engine", "runs the model", LLM), ("GPU", "weights + KV cache", GPU_ICON)]
L_Y = [2.05, 1.05, 0.05, -0.95, -1.95, -2.95]
CARD_X, CARD_W, CARD_H = -3.15, 5.3, 0.8
RAIL_DOWN, RAIL_UP = -6.15, -0.15


def layer_card(i):
    name, sub, ic = LAYERS[i]
    y = L_Y[i]
    color = REQUEST if i == 0 else EDGE
    box = card(CARD_W, CARD_H, color, CARD, radius=0.14, stroke=2).move_to([CARD_X, y, 0])
    pic = icon(ic, 0.5).move_to([CARD_X - CARD_W / 2 + 0.45, y, 0])
    nm = left_at(text(name, 24, INK, weight=m.BOLD), CARD_X - CARD_W / 2 + 0.88, y)
    sb = right_at(text(sub, 20, MUTED), CARD_X + CARD_W / 2 - 0.25, y)
    sit(sb, baseline(nm))
    return m.VGroup(box, pic, nm, sb)


class MiniStack(m.VGroup):
    """The stack in miniature (top right): six small layers and a dot where the request is now."""

    X, TOP, PITCH = 6.3, 2.3, 0.32

    def __init__(self):
        super().__init__()
        self.rects = m.VGroup(*[
            m.RoundedRectangle(corner_radius=0.05, width=0.5, height=0.22).set_stroke(EDGE, 2).set_fill(CARD, 1)
            .move_to([self.X, self.y(i), 0]) for i in range(6)])
        self.marker = m.Dot(radius=0.075, color=REQUEST).move_to([self.X - 0.42, self.y(0), 0]).set_z_index(3)
        self.add(self.rects, self.marker)

    def y(self, i):
        return self.TOP - self.PITCH * i

    def goto(self, i, color=REQUEST, run_time=0.6):
        anims = [self.marker.animate.move_to([self.X - 0.42, self.y(i), 0]).set_color(color)]
        anims += [r.animate.set_stroke(color if k == i else EDGE, 2) for k, r in enumerate(self.rects)]
        return m.AnimationGroup(*anims, run_time=run_time)


# ---------------------------------------------------------------------------- the episode


class Ep22TheServingStack(Episode):
    number = 22

    def lesson(self):
        self.stack()
        self.request()
        self.router()
        self.replica()
        self.streaming()
        self.autoscaling()
        self.cold_start()
        self.multi_lora()
        self.recap()

    # ------------------------------------------------------------------ 1 · the stack

    def stack(self):
        self.next_section(
            "1 · The stack",
            notes="The map of the episode (notes §13 'Layers'): client SDK → (gateway) → router / load balancer → "
                  "engine replicas (API server: tokenizer and detokenizer; scheduler; model runner; sampler) → GPU. "
                  "The gateway (auth, rate limits) is folded into the router here. Earlier episodes, linked by "
                  "number only: 1 (tokenize, prefill, decode, SSE stream), 8 (prefix caching), 10 (continuous "
                  "batching), 17 (sampling and grammar masks). The stack then shrinks into the small map at the top "
                  "right, whose dot shows where the request is in each section.")
        cards = [layer_card(i) for i in range(6)]
        query = title_text("?", 96, MUTED).move_to([CARD_X, (L_Y[1] + L_Y[4]) / 2, 0])
        bracket = m.DashedLine([RAIL_DOWN, L_Y[0] - 0.5, 0], [RAIL_DOWN, L_Y[5] + 0.5, 0], dash_length=0.12)
        bracket.set_stroke(MUTED, 2)
        dot = m.Dot(radius=0.12, color=REQUEST).move_to([RAIL_DOWN, L_Y[0], 0]).set_z_index(4)
        segs = [m.Line([RAIL_DOWN, L_Y[i], 0], [RAIL_DOWN, L_Y[i + 1], 0]).set_stroke(REQUEST, 4) for i in range(5)]
        req_lab = left_at(code("request", 20, REQUEST), RAIL_DOWN - 0.12, L_Y[0] + 0.58)
        up = m.Arrow([RAIL_UP, L_Y[5], 0], [RAIL_UP, L_Y[0] + 0.1, 0], buff=0, color=TOKEN, stroke_width=4,
                     max_tip_length_to_length_ratio=0.05)
        tok_lab = code("tokens", 20, TOKEN).move_to([RAIL_UP, L_Y[0] + 0.58, 0])
        sit(tok_lab, baseline(req_lab))
        x0 = 0.3
        links = {1: ep_link(8, "the prefix cache picks the replica", x0, L_Y[1]),
                 3: ep_link(10, "continuous batching", x0, L_Y[3]),
                 4: ep_link(17, "sampling and grammar masks", x0, L_Y[4]),
                 2: ep_link(1, "tokenize, then stream the reply", x0, L_Y[2])}
        for g in links.values():
            sit(g[1], baseline(g[0][1]))

        def down(i):
            return m.AnimationGroup(m.FadeIn(cards[i], shift=0.15 * m.DOWN), m.Create(segs[i - 1]),
                                    dot.animate.move_to([RAIL_DOWN, L_Y[i], 0]))

        self.say(
            "[In episode one, we followed one request inside the engine.] [Today, we zoom out]: "
            "[what sits between the student and the GPU?]",
            m.AnimationGroup(self.new_header("01 · the stack", "Between the student and the GPU"),
                             m.FadeIn(cards[0], shift=0.15 * m.DOWN), m.FadeIn(cards[5], shift=0.15 * m.UP)),
            m.Create(bracket),
            m.FadeIn(query, scale=0.7),
        )
        self.say(
            "[The request leaves the student's app], [meets a router], [reaches an API server], "
            "[joins the scheduler's batch], [runs in the engine], [on the GPU.]",
            m.AnimationGroup(m.FadeOut(query, scale=0.7), m.FadeOut(bracket), m.FadeIn(dot, scale=0.5),
                             m.FadeIn(req_lab)),
            down(1), down(2), down(3), down(4),
            m.AnimationGroup(m.Create(segs[4]), dot.animate.move_to([RAIL_DOWN, L_Y[5], 0]),
                             m.Circumscribe(cards[5], color=REQUEST, buff=0.06)),
        )
        self.say(
            "[And the answer travels back up the same way], [one token at a time.]",
            m.AnimationGroup(m.GrowArrow(up), m.FadeIn(tok_lab)),
            m.LaggedStart(*[chip_up([RAIL_UP, L_Y[5], 0], [RAIL_UP, L_Y[0], 0]) for _ in range(5)], lag_ratio=0.3),
        )
        self.say(
            "[Almost every layer is an earlier episode.] [The router uses episode eight's prefix cache], "
            "[and the scheduler is episode ten's continuous batching.]",
            m.Circumscribe(m.VGroup(*cards[1:5]), color=MEMORY, buff=0.1),
            m.FadeIn(links[1], shift=0.15 * m.LEFT),
            m.FadeIn(links[3], shift=0.15 * m.LEFT),
        )
        self.say(
            "[The engine samples with episode seventeen's grammar masks], "
            "[and the API server streams, as in episode one.]",
            m.FadeIn(links[4], shift=0.15 * m.LEFT),
            m.FadeIn(links[2], shift=0.15 * m.LEFT),
        )
        self.wait(0.5)
        # zoom out: the stack becomes the small map in the corner
        self.mini = MiniStack()
        rest = m.VGroup(*[m.VGroup(*c[1:]) for c in cards], *segs, dot, req_lab, up, tok_lab, *links.values())
        self.play(m.FadeOut(rest), *[m.ReplacementTransform(c[0], r) for c, r in zip(cards, self.mini.rects)],
                  run_time=1.1)
        self.play(m.FadeIn(self.mini.marker, scale=0.5), run_time=0.3)
        self.remove(*self.mini.rects, self.mini.marker)
        self.add(self.mini)

    # ------------------------------------------------------------------ 2 · the request

    def request(self):
        self.next_section(
            "2 · The request",
            notes="The exact body tools/stream_client.py sent to mlx_lm.server on this Mac (measurements §10, "
                  "research/measurements/scripts/stream_client.py; the content's tracking prefix '[client 0 request 0]' "
                  "dropped), shown soft-wrapped; the scene asserts it parses to the real request. POST "
                  "/v1/chat/completions is the OpenAI-compatible endpoint served by vLLM, SGLang, TensorRT-LLM "
                  "(trtllm-serve), llama.cpp's llama-server, mlx_lm.server and LM Studio (localhost:1234/v1) — notes "
                  "§13. Misconception (notes §13): 'OpenAI-compatible means identical' — engines differ in which "
                  "parameters they honour (logprobs, tool calling, n, grammar extensions).")
        req = jcard(REQ_SRC, "POST /v1/chat/completions", REQUEST)
        req.move_to([0, 2.45 - req.height / 2, 0]).align_to([5.92, 0, 0], m.RIGHT)
        req[1].set_z_index(2)
        body = req[1]
        g_msg = gutter(req[0], m.VGroup(*[line_of(body, REQ_SRC, k) for k in (1, 2, 3, 4, 5)]), REQUEST)
        g_max = gutter(req[0], line_of(body, REQ_SRC, 6), COMPUTE)
        g_str = gutter(req[0], line_of(body, REQ_SRC, 8), TOKEN)
        XL, XR = -6.55, req.get_left()[0] - 0.45
        app = window(XR - XL, 1.5, "Owl", ((XL + XR) / 2, 1.7, 0))
        ask = m.VGroup(icon(USER, 0.45), text("“Write a detailed, 400-word…”", 22, INK)).arrange(m.RIGHT, buff=0.25)
        ask.move_to(app[0]).shift(0.22 * m.DOWN).align_to(app[0], m.LEFT).shift(0.3 * m.RIGHT)
        send = arrow(app[0].get_right(), [req.get_left()[0], app[0].get_y(), 0], REQUEST, buff=0.06)
        src = m.VGroup(cite("the request our stream test sent"), cite("to mlx_lm.server on this Mac"))
        src.arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
        left_at(src, XL, 0.45)
        names = ["vLLM", "SGLang", "TensorRT-LLM", "llama-server", "mlx_lm.server"]
        engines = m.VGroup(*[pill(n, MUTED, CARD, 20, mono=True) for n in names])
        rows_ = [m.VGroup(*engines[:3]).arrange(m.RIGHT, buff=0.15), m.VGroup(*engines[3:]).arrange(m.RIGHT, buff=0.15)]
        left_at(rows_[0], XL, -0.85)
        left_at(rows_[1], XL, -1.45)
        for e in engines[1:3]:
            sit(e[1], baseline(engines[0][1]))
        sit(engines[4][1], baseline(engines[3][1]))
        same = left_at(text("the same request works on:", 22, INK), XL, -0.3)
        caveat = left_at(pill("compatible ≠ identical", RED, RED_FILL, 22), XL, -2.15)
        extras = m.VGroup(text("the extra options they honour differ:", 22, INK),
                          code("logprobs · tool calls · grammars", 20, MUTED)).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
        left_at(extras, XL, -2.95)
        assert max(r.get_right()[0] for r in rows_) < XR and extras.get_right()[0] < XR and extras.get_bottom()[1] > -3.5

        self.say(
            "[Owl's app talks to its server in one format]: [an OpenAI-compatible chat completions request.]",
            m.AnimationGroup(self.new_header("02 · the request", "One JSON request"), self.mini.goto(0),
                             m.FadeIn(app, shift=0.15 * m.UP), m.FadeIn(ask, shift=0.15 * m.UP)),
            m.AnimationGroup(m.GrowArrow(send), m.FadeIn(req, shift=0.2 * m.LEFT)),
        )
        self.say(
            "[This is the very request our stream test sent to this Mac]: [a list of messages], [a token limit], "
            "[and stream true.]",
            m.AnimationGroup(m.FadeIn(src), m.Circumscribe(req, color=REQUEST, buff=0.08)),
            m.FadeIn(g_msg, scale=0.6),
            m.FadeIn(g_max, scale=0.6),
            m.FadeIn(g_str, scale=0.6),
        )
        self.say(
            "[The same request works on vLLM, SGLang, llama.cpp and MLX], "
            "[though each one honours different extra options.]",
            m.AnimationGroup(m.FadeIn(same), m.LaggedStart(*[m.FadeIn(e, shift=0.1 * m.UP) for e in engines], lag_ratio=0.15)),
            m.LaggedStart(m.FadeIn(caveat, shift=0.1 * m.UP), m.FadeIn(extras[0], shift=0.1 * m.UP),
                          m.FadeIn(extras[1], shift=0.1 * m.UP), lag_ratio=0.35),
        )
        self.wait(0.4)
        self.clear(self.mini)

    # ------------------------------------------------------------------ 3 · the router

    def router(self):
        self.next_section(
            "3 · The router",
            notes="Routing policies (notes §13): round-robin, least-loaded, prefix/KV-aware. OpenAI's prompt-caching "
                  "docs: routing by 'A hash of the initial tokens' (prompt_cache_key influences it); llm-d and NVIDIA "
                  "Dynamo do prefix-aware / KV-aware routing. Three replicas and their queue lengths (3, 2, 1) are "
                  "illustrative. The stake is Episode 8's measurement on this Mac (measurements §7): Owl-8B 4-bit, a "
                  f"{PC['prefix_tokens']:,}-token chapter prefix + a 45-token question: TTFT {PC['ttft_no_cache_s']:.2f} s "
                  f"without the cache, {PC['ttft_with_cache_s'] * 1000:.0f} ms with it.")
        ys = (1.55, 0.0, -1.55)
        RX, RW, RH = 0.15, 3.5, 1.25
        queues = (3, 2, 1)
        reps = []
        for i, (y, q) in enumerate(zip(ys, queues)):
            box = card(RW, RH, EDGE, CARD, radius=0.16, stroke=2).move_to([RX, y, 0])
            pic = icon(GPU_ICON, 0.5).move_to([RX - RW / 2 + 0.42, y + 0.27, 0])
            nm = left_at(text(f"replica {i + 1}", 22, INK, weight=m.BOLD), RX - RW / 2 + 0.85, y + 0.27)
            sq = m.VGroup(*[m.Square(0.17).set_stroke(REQUEST, 1.8).set_fill(PINK_FILL, 1) for _ in range(q)])
            sq.arrange(m.RIGHT, buff=0.07)
            right_at(sq, RX + RW / 2 - 0.25, y + 0.27)
            reps.append(m.VGroup(box, pic, nm, sq))
        q_lab = code("queue", 20, MUTED).next_to(reps[0][3], m.UP, buff=0.12).align_to(reps[0][3], m.RIGHT)
        q_lab.set_y(ys[0] + RH / 2 + 0.25)
        rbox = card(1.9, 1.5, MUTED, CARD, radius=0.18, stroke=2.5).move_to([-3.55, 0, 0])
        rbody = m.VGroup(icon(ROUTER, 0.62), text("router", 24, INK, weight=m.BOLD)).arrange(m.DOWN, buff=0.14).move_to(rbox)
        rcard = m.VGroup(rbox, rbody)
        stud_y = (1.35, 0.15, -1.05)
        studs = m.VGroup(*[icon(USER, 0.5).move_to([-5.95, y, 0]) for y in stud_y])
        ins = m.VGroup(*[arrow(s.get_right(), rbox.get_left() + [0, 0.3 * (1 - k), 0], REQUEST, buff=0.12)
                         for k, s in enumerate(studs)])
        outs = m.VGroup(*[arrow(rbox.get_right(), reps[k][0].get_left(), MUTED, buff=0.08) for k in range(3)])
        policy = [pill(p, MUTED, CARD, 22, mono=True) for p in ("round robin", "least loaded")]
        policy.append(pill("prefix-aware", KV, VIOLET_FILL, 22, mono=True))
        for p in policy:
            p.move_to([-3.55, -1.25, 0])
        ep8 = ep_pill(8).move_to([-3.55, 1.25, 0])
        hashed = code("hash(first tokens)", 20, KV).move_to([-3.55, 1.85, 0])
        kv = pill("KV · chapter 7", KV, VIOLET_FILL, 20, mono=True)
        left_at(kv, RX - RW / 2 + 0.85, ys[1] - 0.27)
        her = studs[2]
        her_chip = pill("chapter 7", REQUEST, PINK_FILL, 20, mono=True).next_to(her, m.DOWN, buff=0.22)
        her_chip.set_x(-5.55)
        bar_x0 = RX - RW / 2 + 0.85
        prefill = m.Rectangle(width=RX + RW / 2 - 0.25 - bar_x0, height=0.34).set_stroke(COMPUTE, 2).set_fill(AMBER_FILL, 1)
        left_at(prefill, bar_x0, ys[2] - 0.27)
        pre_lab = code("prefill", 20, COMPUTE).move_to(prefill).set_z_index(2)
        OX = 2.3
        miss = m.VGroup(title_text(f"{PC['ttft_no_cache_s']:.1f} s", 44, RED),
                        code("TTFT · cache miss", 20, MUTED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        left_at(miss, OX, ys[2])
        hit = m.VGroup(title_text(f"{PC['ttft_with_cache_s']:.2f} s", 44, GREEN),
                       code("TTFT · cache hit", 20, MUTED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        left_at(hit, OX, ys[1])
        src8 = cite(f"measured on this Mac · Owl-8B 4-bit · {PC['prefix_tokens']:,}-token chapter · Episode 8").move_to([0.5, -2.75, 0])
        src_ai = cite("OpenAI: “a hash of the initial tokens” · llm-d, NVIDIA Dynamo: prefix- and KV-aware routing")
        src_ai.move_to([0, -3.25, 0])
        a_rr = [packet(rbox.get_right(), reps[k][0].get_left(), run_time=0.5) for k in range(3)]

        self.say(
            "[At scale, Owl runs many copies of the model, called replicas], "
            "[and a router in front picks one for each request.]",
            m.AnimationGroup(self.new_header("03 · the router", "Send it where the cache is"), self.mini.goto(1),
                             m.LaggedStart(*[m.FadeIn(r, shift=0.15 * m.LEFT) for r in reps], lag_ratio=0.2),
                             m.FadeIn(q_lab)),
            m.AnimationGroup(m.FadeIn(rcard, scale=0.9), m.FadeIn(studs, shift=0.1 * m.RIGHT), m.Create(ins),
                             m.Create(outs)),
        )
        self.say(
            "[Round robin simply takes turns]; [least loaded picks the shortest queue.]",
            m.AnimationGroup(m.FadeIn(policy[0], shift=0.1 * m.UP), m.Succession(*a_rr)),
            m.AnimationGroup(m.FadeOut(policy[0], shift=0.1 * m.UP), m.FadeIn(policy[1], shift=0.1 * m.UP),
                             m.Circumscribe(reps[2][3], color=REQUEST, buff=0.08),
                             packet(rbox.get_right(), reps[2][0].get_left(), run_time=0.7)),
        )
        self.say(
            "[But episode eight's prefix cache lives on one replica]: [this student's chapter is cached only here.]",
            m.AnimationGroup(m.FadeIn(ep8, shift=0.1 * m.DOWN), m.FadeIn(kv, shift=0.1 * m.RIGHT)),
            m.AnimationGroup(m.FadeIn(her_chip, shift=0.1 * m.UP), m.Circumscribe(kv, color=KV, buff=0.08)),
        )
        self.say(
            "[Send her next question elsewhere], [and the whole chapter is prefilled again]: [eight seconds on this Mac.]",
            m.Succession(packet(her.get_right(), rbox.get_left(), run_time=0.5),
                         packet(rbox.get_right(), reps[2][0].get_left(), run_time=0.6)),
            m.AnimationGroup(m.GrowFromEdge(prefill, m.LEFT, run_time=1.6), m.FadeIn(pre_lab)),
            m.AnimationGroup(m.FadeIn(miss, shift=0.1 * m.LEFT), m.FadeIn(src8)),
        )
        self.say(
            "[A prefix-aware router hashes the prompt's first tokens], [and finds the replica with the cache]: "
            "[first token in a hundred and thirty-seven milliseconds.]",
            m.AnimationGroup(m.FadeOut(policy[1], shift=0.1 * m.UP), m.FadeIn(policy[2], shift=0.1 * m.UP),
                             m.FadeIn(hashed, shift=0.1 * m.UP)),
            m.AnimationGroup(packet(rbox.get_right(), reps[1][0].get_left(), run_time=0.7),
                             m.Succession(m.Wait(0.5), m.Circumscribe(kv, color=GREEN, buff=0.08))),
            m.AnimationGroup(m.FadeIn(hit, shift=0.1 * m.LEFT), m.FadeIn(src_ai)),
        )
        self.wait(0.4)
        self.clear(self.mini)

    # ------------------------------------------------------------------ 4 · inside a replica

    def replica(self):
        self.next_section(
            "4 · Inside a replica",
            notes="One replica (notes §13): API server (OpenAI-compatible HTTP, chat template, tokenizer and "
                  "detokenizer) → scheduler (continuous batching, Episode 10; chunked prefill, Episode 11) → engine "
                  "(model runner with its kernels; sampler, Episode 17). The request's 64 prompt tokens are its real "
                  "usage.prompt_tokens (chat template included). The batch grid is schematic (Episode 10's picture). "
                  "Engines and status, dated (GitHub releases 2026-10-23, notes §13): vLLM v0.32.0 (PagedAttention, "
                  "prefix caching, continuous batching); SGLang v0.5.22 (RadixAttention; large-scale serving); "
                  "TensorRT-LLM v1.2.1 (NVIDIA; now a PyTorch-based executor [inferred from docs]); llama.cpp / "
                  "llama-server (GGUF; 'Apple silicon is a first-class citizen'); MLX / mlx-lm v0.32.2 (Apple; every "
                  "measurement in this course). Hugging Face TGI: README 'text-generation-inference is now in maintenance "
                  "mode' (notes date it 2025-12-11), repository archived 2026-03-21 (GitHub, read 2026-10-24); it recommends "
                  "vLLM and SGLang, or llama.cpp and MLX locally. Deployment shapes: one replica across several GPUs (Episode 21), prefill and "
                  "decode in separate pools (Episode 12).")
        xs, Y, W, H = (-4.5, -0.9, 2.7), 1.55, 3.0, 1.3
        names = [("API server", "JSON ⇄ tokens", SERVER_ICON), ("scheduler", "forms the batch", GEAR),
                 ("engine", "model + sampler", LLM)]
        pipe = []
        for x, (nm, sb, ic) in zip(xs, names):
            box = card(W, H, EDGE, CARD, radius=0.16, stroke=2).move_to([x, Y, 0])
            words = m.VGroup(text(nm, 24, INK, weight=m.BOLD), text(sb, 20, MUTED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
            body = m.VGroup(icon(ic, 0.55), words).arrange(m.RIGHT, buff=0.25).move_to(box)
            pipe.append(m.VGroup(box, body))
        links = [arrow(pipe[k][0].get_right(), pipe[k + 1][0].get_left(), MUTED, buff=0.06) for k in range(2)]

        # API server: the request's 64 prompt tokens
        n_prompt = USAGE["prompt_tokens"]
        grid = m.VGroup(*[m.Square(0.13).set_stroke(REQUEST, 1.2).set_fill(PINK_FILL, 1) for _ in range(n_prompt)])
        grid.arrange_in_grid(rows=4, cols=16, buff=0.05).move_to([xs[0], -0.5, 0])
        g_cap = code(f"{n_prompt} prompt tokens", 20, REQUEST).move_to([xs[0], 0.35, 0])
        # scheduler: the request joins a running batch at the next step
        cols, size, gap = 6, 0.3, 0.07
        cells = m.VGroup()
        for r in range(4):
            for c in range(cols):
                sq = m.Square(size).set_stroke(EDGE, 1.5).set_fill(CARD, 1)
                sq.move_to([xs[1] - (cols - 1) * (size + gap) / 2 + c * (size + gap), -0.15 - r * (size + gap), 0])
                cells.add(sq)
        busy = m.VGroup()
        for r in range(3):
            for c in range(4):
                busy.add(cells[r * cols + c].copy().set_stroke(REQUEST, 1.8).set_fill(PINK, 0.42))
        nxt = 4
        step_col = m.VGroup(*[cells[r * cols + nxt].copy().set_stroke(REQUEST, 1.8).set_fill(PINK, 0.42) for r in range(3)])
        joiner = cells[3 * cols + nxt].copy().set_stroke(COMPUTE, 2).set_fill(AMBER_FILL, 1)
        marker = m.RoundedRectangle(corner_radius=0.06, width=size + 0.14, height=4 * size + 3 * gap + 0.14)
        marker.set_stroke(INK, 2).move_to([cells[nxt].get_x(), (cells[nxt].get_y() + cells[3 * cols + nxt].get_y()) / 2, 0])
        s_cap = code("joins the batch", 20, MUTED).move_to([xs[1], 0.35, 0])
        # engine: weights → sampler → a token
        wbox = card(1.45, 0.75, WEIGHTS, BLUE_FILL, radius=0.12, stroke=2)
        wblk = m.VGroup(wbox, code("weights", 20, INK).move_to(wbox))
        sbox = card(1.35, 0.75, TOKEN, GREEN_FILL, radius=0.12, stroke=2)
        samp = m.VGroup(sbox, code("sample", 20, INK).move_to(sbox))
        hey = token_row([PIECES[0]])
        flow = m.VGroup(wblk, samp, hey).arrange(m.RIGHT, buff=0.42).move_to([xs[2], -0.55, 0])
        e_arrows = [arrow(flow[k].get_right(), flow[k + 1].get_left(), MUTED, buff=0.05) for k in range(2)]
        e_cap = m.VGroup(ep_pill(17), code("sampling", 20, MUTED)).arrange(m.RIGHT, buff=0.2)
        e_cap.move_to([xs[2], 0.35, 0])
        sit(e_cap[1], baseline(e_cap[0][1]))
        details = m.VGroup(grid, g_cap, cells, busy, step_col, joiner, marker, s_cap, flow, *e_arrows, e_cap)

        # the engines
        rows = [("vLLM", "PagedAttention", "prefix caching"), ("SGLang", "RadixAttention", "agentic workloads"),
                ("TensorRT-LLM", "NVIDIA's engine", "for NVIDIA GPUs"), ("llama.cpp", "GGUF models", "CPUs and Macs"),
                ("MLX", "mlx-lm · Apple", "this course's runs")]
        ecards = []
        for nm, a, b in rows:
            box = card(2.45, 1.45, EDGE if nm != "MLX" else REQUEST, CARD, radius=0.16, stroke=2)
            words = m.VGroup(code(nm, 20, INK, weight=m.BOLD), text(a, 20, MUTED), text(b, 20, MUTED))
            words.arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT).move_to(box).align_to(box, m.LEFT).shift(0.22 * m.RIGHT)
            ecards.append(m.VGroup(box, words))
        roster = m.VGroup(*ecards).arrange(m.RIGHT, buff=0.15).move_to([0, -0.75, 0])
        for e in ecards[1:]:
            e[1].align_to(ecards[0][1], m.UP)
        cap_big = code("GPU servers", 20, MUTED).next_to(m.VGroup(*ecards[:3]), m.UP, buff=0.18)
        cap_small = code("laptops and Macs", 20, MUTED).next_to(m.VGroup(*ecards[3:]), m.UP, buff=0.18)
        sit(cap_small, baseline(cap_big))
        tgi = pill("Hugging Face TGI · maintenance mode · archived March 2026", MUTED, CARD, 22).move_to([0, -2.1, 0])
        dated = cite("releases and status: GitHub, October 2026").move_to([0, -3.35, 0])
        ep21 = ep_link(21, "one replica, several GPUs", 0, -2.8)
        ep12 = ep_link(12, "prefill and decode in separate pools", 0, -2.8)
        m.VGroup(ep21, ep12).arrange(m.RIGHT, buff=0.7).move_to([0, -2.8, 0])
        sit(ep12[1], baseline(ep21[1]))
        assert roster.get_right()[0] < 6.5 and all(e[1].width < e[0].width - 0.3 for e in ecards)

        self.say(
            "[Inside a replica, the API server applies the chat template], "
            "[and turns the request into sixty-four tokens.]",
            m.AnimationGroup(self.new_header("04 · inside a replica", "Server, scheduler, engine"), self.mini.goto(2),
                             m.LaggedStart(*[m.FadeIn(p, shift=0.15 * m.DOWN) for p in pipe], lag_ratio=0.2),
                             m.Create(m.VGroup(*links))),
            m.AnimationGroup(m.FadeIn(g_cap), m.LaggedStart(*[m.FadeIn(s, scale=0.5) for s in grid], lag_ratio=0.02, run_time=1.4)),
        )
        self.say(
            "[The scheduler slips them into the batch that's already running], [at the very next step.]",
            m.AnimationGroup(self.mini.goto(3), pulse(links[0], REQUEST), m.FadeIn(cells), m.FadeIn(busy), m.FadeIn(s_cap)),
            m.AnimationGroup(m.FadeIn(marker), m.FadeIn(step_col, scale=0.6),
                             m.FadeIn(joiner, target_position=pipe[0][0].get_right())),
        )
        self.say(
            "[The engine runs the model], [and its sampler picks each token.]",
            m.AnimationGroup(self.mini.goto(4), pulse(links[1], REQUEST), m.FadeIn(wblk), m.FadeIn(e_cap)),
            m.AnimationGroup(m.GrowArrow(e_arrows[0]), m.FadeIn(samp), m.Succession(m.Wait(0.4), m.AnimationGroup(
                m.GrowArrow(e_arrows[1]), m.FadeIn(hey, shift=0.15 * m.RIGHT)))),
        )
        self.say(
            "[Several open engines fill this box]: [for GPU servers, vLLM], [SGLang], [and NVIDIA's TensorRT-LLM.]",
            m.AnimationGroup(m.FadeOut(details, shift=0.15 * m.DOWN), m.Circumscribe(pipe[2], color=MEMORY, buff=0.08)),
            m.AnimationGroup(m.FadeIn(cap_big), m.FadeIn(ecards[0], shift=0.15 * m.UP)),
            m.FadeIn(ecards[1], shift=0.15 * m.UP),
            m.FadeIn(ecards[2], shift=0.15 * m.UP),
        )
        self.say(
            "[For laptops and Macs], [llama.cpp], [and Apple's MLX], [which ran every measurement in this course.]",
            m.FadeIn(cap_small),
            m.FadeIn(ecards[3], shift=0.15 * m.UP),
            m.FadeIn(ecards[4], shift=0.15 * m.UP),
            m.Circumscribe(ecards[4], color=REQUEST, buff=0.08),
        )
        self.say(
            "[One name has left the list]: [Hugging Face's Text Generation Inference, now archived.]",
            m.FadeIn(tgi, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(dated), m.Circumscribe(tgi, color=MUTED, buff=0.08)),
        )
        self.say(
            "[A replica can span several GPUs, as in episode twenty-one], "
            "[or split into prefill and decode pools, as in episode twelve.]",
            m.FadeIn(ep21, shift=0.1 * m.UP),
            m.FadeIn(ep12, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear(self.mini)

    # ------------------------------------------------------------------ 5 · streaming

    def streaming(self):
        self.next_section(
            "5 · Streaming",
            notes="Server-sent events from mlx_lm.server 0.32.2 (source: generate_response / completion_usage_response): "
                  "each chunk is `data: {json}` + a blank line; object 'chat.completion.chunk'; delta "
                  "{'role': 'assistant', 'content': piece}; the last chunk carries the finish_reason (its delta has no "
                  "content); with stream_options.include_usage a usage chunk (empty choices) follows, then "
                  "`data: [DONE]`. Shown without system_fingerprint, model and created; 'chatcmpl-…' shortens a uuid; "
                  "later chunks are elided with '…'. Pieces: the real first tokens of this request's answer ('Hey "
                  f"there, young adventurer! …', Llama-3.1 tokenizer). Usage is this request's real one: {USAGE} "
                  "(total_tokens left out on screen). 63 of 64 prompt tokens were cached because the run's warm-up had "
                  "sent the same prompt (mlx_lm.server keeps a prompt cache). Server vs engine (this Mac, Owl-8B "
                  f"4-bit, batch 1): BatchGenerator {ENG1:.1f} tok/s ({ENG_STEP:.2f} ms per step, 256-token prompts, "
                  f"measurements §8) vs mlx_lm.server one client {PER1:.1f} tok/s per stream (ITL p50 {ITL1:.2f} ms, "
                  f"~64-token prompts, §10): {100 * COST:.1f}% slower; ceiling 546 GB/s ÷ {Q4_GB:.2f} GB = {CEIL:.0f} "
                  "tok/s. Reading (ours, not instrumented): the server's per-token work — detokenize, one JSON chunk "
                  "and one HTTP write per token, Python threads. The engine run's longer prompts (more KV read per "
                  "step) make the true gap, if anything, slightly larger.")
        first_src = ('data: {"id": "chatcmpl-…",\n'
                     '  "object": "chat.completion.chunk",\n'
                     '  "choices": [{"index": 0,\n'
                     '    "delta": {"role": "assistant",\n'
                     f'              "content": "{PIECES[0]}"}},\n'
                     '    "finish_reason": null}]}')
        assert json.loads(first_src[6:])["choices"][0]["delta"]["content"] == PIECES[0]
        rest_src = [f'data: {{… "content": "{p}"}}' for p in PIECES[1:4]]
        tail_src = ['data: {… "finish_reason": "length"}',
                    'data: {… "choices": [], "usage": {…}}',
                    "data: [DONE]"]
        first = code(first_src, 20, INK, t2c=JSON_RULES)
        rest = [code(s, 20, INK, t2c=JSON_RULES) for s in rest_src]
        more = code(f"… {REQ0['chunks'] - 4} more chunks, one per token", 20, MUTED)
        tail = [code(s, 20, INK, t2c=JSON_RULES) for s in tail_src]
        lines = m.VGroup(first, *rest, more, *tail).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        bar_h = 0.42
        sbox = card(lines.width + 0.7, lines.height + 0.5 + bar_h, EDGE, "#0A1013", radius=0.16, stroke=2)
        sbox.move_to([-6.5 + sbox.width / 2, 2.45 - sbox.height / 2, 0])
        sbar = m.Line(sbox.get_corner(m.UL) + [0, -bar_h, 0], sbox.get_corner(m.UR) + [0, -bar_h, 0]).set_stroke(EDGE, 2)
        scap = code("text/event-stream", 20, FAINT).move_to(sbox.get_top() + [0, -bar_h / 2, 0])
        lines.next_to(sbar, m.DOWN, buff=0.22).align_to(sbox, m.LEFT).shift(0.35 * m.RIGHT)
        lines.set_z_index(2)
        frame_ = m.VGroup(sbox, sbar, scap)
        g_delta = gutter(sbox, m.VGroup(line_of(first, first_src, 3), line_of(first, first_src, 4)), TOKEN)
        g_fin = gutter(sbox, tail[0], COMPUTE)
        g_use = gutter(sbox, tail[1], MEMORY)
        assert sbox.get_bottom()[1] > -3.5 and sbox.get_right()[0] < 0.9

        x_r = sbox.get_right()[0] + 0.4
        screen = window(5.9 - x_r, 2.2, "Owl · the student's screen", ((x_r + 5.9) / 2, 1.35, 0))
        chips = token_row(PIECES, size=20, max_width=screen[0].width - 0.6)
        chips.move_to(screen[0]).shift(0.2 * m.DOWN).align_to(screen[0], m.LEFT).shift(0.3 * m.RIGHT)
        comma = chips[2]                                   # a lone comma is too small to read at size 20
        big = code(",", 28, INK).move_to(comma[1]).align_to(comma[1], m.UP)
        comma.remove(comma[1])
        comma.add(big)
        groups = [chips[0:1], chips[1:3], chips[3:4], chips[4:6]]
        line_groups = [m.VGroup(first), m.VGroup(rest[0], rest[1]), m.VGroup(rest[2]), m.VGroup(more)]
        limit = pill(f"max_tokens: {REQ_BODY['max_tokens']} reached", COMPUTE, AMBER_FILL, 20, mono=True)
        limit.next_to(screen[0], m.DOWN, buff=0.25).align_to(screen[0], m.LEFT)
        usage = jcard(USAGE_SRC, "usage chunk", MEMORY)
        usage.move_to([x_r + usage.width / 2, 0, 0]).align_to([0, -0.75, 0], m.UP)
        usage[1].set_z_index(2)
        cached = glyphs(usage[1], '"cached_tokens": 63')
        assert usage.get_right()[0] < 6.5

        def show(k):
            return m.AnimationGroup(m.FadeIn(line_groups[k], shift=0.1 * m.UP),
                                    m.Succession(m.Wait(0.25), m.LaggedStart(*[m.FadeIn(c, target_position=line_groups[k][0])
                                                                               for c in groups[k]], lag_ratio=0.3)))

        self.say(
            "[Now the answer flows back up.] [With stream true, it arrives as server-sent events]: "
            "[one small JSON chunk per token.]",
            m.AnimationGroup(self.new_header("05 · streaming", "One event per token"),
                             m.Succession(*[self.mini.goto(i, TOKEN, run_time=0.3) for i in (3, 2, 1, 0)])),
            m.AnimationGroup(m.FadeIn(frame_, shift=0.15 * m.UP), m.FadeIn(screen, shift=0.15 * m.UP)),
            show(0),
        )
        self.say(
            "[Each chunk carries a delta, just the new text]: [hey], [there], [young], [adventurer.]",
            m.AnimationGroup(m.FadeIn(g_delta, scale=0.6), m.Circumscribe(line_of(first, first_src, 4), color=TOKEN, buff=0.06)),
            m.Circumscribe(chips[0], color=TOKEN, buff=0.06),
            show(1), show(2), show(3),
        )
        self.say(
            "[The last chunk gives the finish reason]: [length], "
            "[because Owl hit its limit of a hundred and twenty-eight tokens.]",
            m.FadeIn(tail[0], shift=0.1 * m.UP),
            m.FadeIn(g_fin, scale=0.6),
            m.FadeIn(limit, shift=0.1 * m.UP),
        )
        self.say(
            "[Then a usage chunk counts the tokens]: [sixty-three of the sixty-four prompt tokens came from the "
            "server's cache], [and data done closes the stream.]",
            m.AnimationGroup(m.FadeIn(tail[1], shift=0.1 * m.UP), m.FadeIn(g_use, scale=0.6), m.FadeIn(usage, shift=0.15 * m.UP)),
            m.Circumscribe(cached, color=KV, buff=0.06),
            m.FadeIn(tail[2], shift=0.1 * m.UP),
        )
        self.wait(0.3)
        stream_group = m.VGroup(frame_, lines, g_delta, g_fin, g_use, screen, chips, limit, usage)

        # the server's own cost, measured
        X0, SCALE = -1.45, 5.0 / CEIL
        VX = X0 + CEIL * SCALE + 0.3
        rows = [("ceiling", f"546 GB/s ÷ {Q4_GB:.2f} GB", CEIL, MEMORY, CYAN_FILL, 1.15),
                ("engine alone", "BatchGenerator · batch 1", ENG1, TOKEN, GREEN_FILL, 0.05),
                ("through the server", "mlx_lm.server · one stream", PER1, TOKEN, GREEN_FILL, -1.05)]
        bars, labs, vals = [], [], []
        for name, sub, v, col, fill, y in rows:
            b = m.Rectangle(width=v * SCALE, height=0.55).set_stroke(col, 2.5).set_fill(fill, 1)
            left_at(b, X0, y)
            bars.append(b)
            lab = m.VGroup(text(name, 24, INK, weight=m.BOLD), code(sub, 20, MUTED)).arrange(m.DOWN, buff=0.08, aligned_edge=m.RIGHT)
            right_at(lab, X0 - 0.3, y)
            labs.append(lab)
            vals.append(left_at(code(f"{v:.0f} tok/s", 22, col), VX, y))
        assert min(lab.get_left()[0] for lab in labs) > -6.6 and max(v.get_right()[0] for v in vals) < 5.95
        src = cite(f"measured on this Mac · mlx-lm {MLX_LM} · Owl-8B 4-bit ({Q4_GB:.2f} GB)").move_to([0, -1.95, 0])
        lost = m.Rectangle(width=(ENG1 - PER1) * SCALE, height=0.55).set_stroke(RED, 2).set_fill(RED_FILL, 1)
        left_at(lost, bars[2].get_right()[0], rows[2][5])
        cost = m.VGroup(title_text(f"−{100 * COST:.0f}%", 44, RED),
                        code(f"{ENG_STEP:.1f} → {ITL1:.1f} ms per token", 20, MUTED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        reading = m.VGroup(text("the server's own work, per token:", 22, INK),
                           cite("one JSON chunk, one HTTP write · our reading")).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        m.VGroup(cost, reading).arrange(m.RIGHT, buff=0.6).move_to([0, -2.85, 0])

        self.say(
            "[Through the server, one stream got ninety-four tokens a second]; [the engine alone, a hundred and one], "
            "[under a ceiling of a hundred and twenty-one.]",
            m.AnimationGroup(m.FadeOut(stream_group, shift=0.15 * m.UP), m.FadeIn(labs[2]), m.FadeIn(src),
                             m.Succession(m.Wait(0.4), m.AnimationGroup(m.GrowFromEdge(bars[2], m.LEFT), m.FadeIn(vals[2])))),
            m.AnimationGroup(m.FadeIn(labs[1]), m.GrowFromEdge(bars[1], m.LEFT), m.FadeIn(vals[1])),
            m.AnimationGroup(m.FadeIn(labs[0]), m.GrowFromEdge(bars[0], m.LEFT), m.FadeIn(vals[0])),
        )
        self.say(
            "[So the server's own work], [a JSON chunk for every token], [costs about seven percent here.]",
            m.GrowFromEdge(lost, m.LEFT),
            m.FadeIn(reading, shift=0.1 * m.UP),
            m.FadeIn(cost, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear(self.mini)

    # ------------------------------------------------------------------ 6 · autoscaling

    def autoscaling(self):
        self.next_section(
            "6 · Autoscaling",
            notes="The exam-week queue curve and the replica counts are illustrative (for example). Google Cloud, GKE "
                  "'Best practices for autoscaling LLM inference workloads with GPUs' (updated 2026-10-22, notes §13): "
                  "'Use queue size to maximize throughput and minimize cost within a certain target latency "
                  "threshold'; 'We recommend choosing batch size-based autoscaling if you have latency-sensitive "
                  "workloads'; GPU utilization 'Does not measure how much work is being done while the GPU is active.' "
                  "Cold start = provision the node + pull the container + load the weights + allocate KV and capture "
                  "CUDA graphs / compile (notes §13).")
        ox, oy, w, h = -6.1, -0.15, 5.6, 2.2
        axes_ = m.VGroup(
            m.Arrow([ox, oy, 0], [ox + w + 0.3, oy, 0], buff=0, color=FAINT, stroke_width=3, max_tip_length_to_length_ratio=0.04),
            m.Arrow([ox, oy, 0], [ox, oy + h + 0.25, 0], buff=0, color=FAINT, stroke_width=3, max_tip_length_to_length_ratio=0.07))
        days = m.VGroup(*[code(d, 20, MUTED).move_to([ox + 0.55 + 1.1 * k, oy - 0.3, 0])
                          for k, d in enumerate(("Mon", "Tue", "Wed", "Thu", "Fri"))])
        for d in days[1:]:
            sit(d, baseline(days[0]))
        y_lab = left_at(code("queue · exam week", 20, REQUEST), ox + 0.2, oy + h + 0.1)
        pts = [(0.0, 0.25), (0.8, 0.3), (1.6, 0.55), (2.2, 1.2), (2.8, 1.85), (3.4, 2.05), (4.0, 1.4), (4.6, 0.75), (5.4, 0.45)]
        curve = m.VMobject().set_points_smoothly([np.array([ox + 0.1 + px, oy + py, 0]) for px, py in pts]).set_stroke(REQUEST, 4)
        thr_y = oy + 1.0
        thr = m.DashedLine([ox, thr_y, 0], [ox + w, thr_y, 0], dash_length=0.12).set_stroke(MEMORY, 2)
        thr_lab = left_at(code("scale up", 20, MEMORY), ox + 0.2, thr_y + 0.22)

        xs = (1.0, 2.35, 3.7, 5.05)
        reps = m.VGroup(*[icon(GPU_ICON, 0.62).move_to([x, 1.35, 0]) for x in xs])
        r_cap = left_at(code("replicas", 20, MUTED), 0.55, 2.2)
        plus = m.VGroup(*[code("+1", 22, MEMORY, weight=m.BOLD).move_to([xs[k], 0.6, 0]) for k in (2, 3)])
        frames = m.VGroup(*[m.DashedVMobject(m.RoundedRectangle(corner_radius=0.12, width=1.15, height=1.0)
                                             .set_stroke(MEMORY, 2), num_dashes=24).move_to(reps[k]) for k in (2, 3)])
        cold = pill("cold", MEMORY, CYAN_FILL, 20, mono=True).move_to(plus[1])

        def quote(title, color, lines_):
            words = m.VGroup(text(title, 24, color, weight=m.BOLD), *[text(s, 20, INK) for s in lines_])
            words.arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
            box = card(words.width + 0.6, words.height + 0.45, color, CARD, radius=0.16, stroke=2)
            return m.VGroup(box, words.move_to(box).align_to(box, m.LEFT).shift(0.3 * m.RIGHT))

        q_ok = quote("scale on queue size or batch size", GREEN,
                     ["“Use queue size to maximize throughput", "and minimize cost within a certain",
                      "target latency threshold”"])
        q_no = quote("not on GPU utilization", RED,
                     ["“Does not measure how much work", "is being done while the GPU", "is active.”"])
        quotes = m.VGroup(q_ok, q_no).arrange(m.RIGHT, buff=0.4, aligned_edge=m.UP).move_to([0, -2.0, 0])
        assert quotes.get_width() < 12.9 and quotes.get_top()[1] < -0.85
        src = cite("Google Cloud · GKE inference autoscaling guide · October 2026").move_to([0, -3.35, 0])

        steps = [pill("machine", MUTED, CARD, 22), pill("container", MUTED, CARD, 22),
                 pill(f"{W_BYTES / 1e9:.0f} GB of weights", WEIGHTS, BLUE_FILL, 22), pill("warm-up", COMPUTE, AMBER_FILL, 22)]
        row = m.VGroup(*steps).arrange(m.RIGHT, buff=0.55).move_to([0, -2.1, 0])
        for s in steps[1:]:
            sit(s[1], baseline(steps[0][1]))
        joins = [arrow(steps[k].get_right(), steps[k + 1].get_left(), MUTED, buff=0.06) for k in range(3)]
        row_cap = code("a new replica, before its first token:", 20, MEMORY).next_to(row, m.UP, buff=0.3)

        self.say(
            "[In exam week, requests pile up in the queue], [and an autoscaler adds replicas to drain it.]",
            m.AnimationGroup(self.new_header("06 · autoscaling", "More replicas when the queue grows"), self.mini.goto(1),
                             m.FadeIn(axes_), m.FadeIn(days), m.FadeIn(y_lab), m.FadeIn(r_cap), m.FadeIn(reps[:2]),
                             m.FadeIn(thr), m.FadeIn(thr_lab), m.Create(curve, run_time=2.4)),
            m.LaggedStart(m.AnimationGroup(m.FadeIn(reps[2], scale=0.7), m.FadeIn(plus[0], shift=0.1 * m.UP)),
                          m.AnimationGroup(m.FadeIn(reps[3], scale=0.7), m.FadeIn(plus[1], shift=0.1 * m.UP)), lag_ratio=0.6),
        )
        self.say(
            "[Google's guide says to scale on queue size or batch size], [not on GPU utilization], "
            "[which counts busy time, not work done.]",
            m.AnimationGroup(m.FadeIn(q_ok, shift=0.1 * m.UP), m.FadeIn(src)),
            m.FadeIn(q_no, shift=0.1 * m.UP),
            m.Circumscribe(q_no[1][1:], color=RED, buff=0.06),
        )
        self.say(
            "[But a new replica starts cold]: [it needs a machine], [a container], [sixteen gigabytes of weights], "
            "[and a warm-up.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(quotes, src), shift=0.15 * m.DOWN), m.Create(frames),
                             m.FadeOut(plus[1]), m.FadeIn(cold, shift=0.1 * m.UP), m.FadeIn(row_cap)),
            m.FadeIn(steps[0], shift=0.1 * m.UP),
            m.AnimationGroup(m.GrowArrow(joins[0]), m.FadeIn(steps[1], shift=0.1 * m.UP)),
            m.AnimationGroup(m.GrowArrow(joins[1]), m.FadeIn(steps[2], shift=0.1 * m.UP)),
            m.AnimationGroup(m.GrowArrow(joins[2]), m.FadeIn(steps[3], shift=0.1 * m.UP)),
        )
        self.wait(0.4)
        self.clear(self.mini)

    # ------------------------------------------------------------------ 7 · cold starts

    def cold_start(self):
        self.next_section(
            "7 · Cold starts",
            notes="NVIDIA technical blog, 'Reducing Cold Start Latency for LLM Inference with NVIDIA Run:ai Model "
                  "Streamer' (2025-09-16, read 2026-10-24): Llama 3 8B in one 15 GB safetensors file, one A10G of an "
                  "AWS g5.12xlarge, GP3 SSD (750 GB, 16K IOPS, 1,000 MiB/s). Load to GPU: HF safetensors loader "
                  f"{RUNAI['hf_load']} s; streamer {RUNAI['stream_load']} s (concurrency 16). Total until the vLLM engine "
                  f"is ready: {RUNAI['hf_ready']} s vs {RUNAI['stream_ready']} s. The 'warm-up' segment is total − "
                  f"load (our split: engine start, KV allocation, CUDA graphs). Check: 15 GB ÷ 1,000 MiB/s = {T_DISK:.2f} s, the streamer's measured "
                  f"{RUNAI['stream_load']} s: the disk is the limit. Owl-8B: {W_BYTES / 1e9:.2f} GB ÷ 5 GB/s (for example, a "
                  f"fast local NVMe drive; notes §13) = {T_NVME:.1f} s for the weights alone. Sleep mode: vLLM blog "
                  "(2025-10-26, read 2026-10-24), A100, Qwen3-0.6B and Phi-3-vision: level 1 wake 0.26 / 0.82 s, level 2 "
                  "0.85 / 2.58 s, vs 37.6 / 58.1 s per model switch without it ('~48s' average). Larger models wake "
                  "slower (the blog: '~3-6s for large models' at level 1). A warm spare is an idle replica that is "
                  "already loaded.")
        # kit's Timeline maps time with self.width, which becomes the group's width once tick labels are added:
        # build it bare (width = the axis) and add the ticks here
        tl = Timeline(70, width=8.7, origin=(-4.0, -2.6, 0), lane_height=0.8)
        assert abs(tl.width - 8.7) < 1e-6
        ticks = m.VGroup()
        for t in range(0, 71, 10):
            at = tl.origin + np.array([8.7 * t / 70, 0, 0])
            tick = m.Line(at + [0, -0.07, 0], at + [0, 0.07, 0]).set_stroke(FAINT, 2)
            ticks.add(tick, code(f"{t} s", 20, MUTED).next_to(tick, m.DOWN, buff=0.1))
        lanes = {"A": 3, "B": 2, "C": 1, "D": 0}
        names = [("default loader", "NVIDIA · 15 GB", "A"), ("parallel streamer", "NVIDIA · 15 GB", "B"),
                 ("Owl, local drive", f"{W_BYTES / 1e9:.0f} GB at 5 GB/s", "C"), ("sleeping replica", "vLLM sleep mode", "D")]
        lane_labs = {}
        for nm, sub, key in names:
            y = tl.x(0, lanes[key])[1]
            g = m.VGroup(text(nm, 22, INK, weight=m.BOLD), text(sub, 20, MUTED)).arrange(m.DOWN, buff=0.06, aligned_edge=m.RIGHT)
            lane_labs[key] = right_at(g, -4.2, y)
            assert g.get_left()[0] > -6.6
        a_load = tl.bar(0, RUNAI["hf_load"], MEMORY, lanes["A"], label=f"load weights {RUNAI['hf_load']:.1f} s")
        a_rest = tl.bar(RUNAI["hf_load"], RUNAI["hf_ready"], COMPUTE, lanes["A"], label="warm-up")
        b_load = tl.bar(0, RUNAI["stream_load"], MEMORY, lanes["B"], label=f"{RUNAI['stream_load']:.1f} s")
        b_rest = tl.bar(RUNAI["stream_load"], RUNAI["stream_ready"], COMPUTE, lanes["B"], label="warm-up")
        c_load = tl.bar(0, T_NVME, MEMORY, lanes["C"])
        c_lab = code(f"{T_NVME:.1f} s · weights only", 20, MEMORY).next_to(c_load, m.RIGHT, buff=0.2)
        d_wake = tl.bar(0, max(SLEEP["l2"]), TOKEN, lanes["D"])
        d_lab = code(f"wakes in {min(SLEEP['l1']):.1f}–{max(SLEEP['l2']):.1f} s · not ~48 s", 20, TOKEN).next_to(d_wake, m.RIGHT, buff=0.2)
        ready_a = code(f"ready {RUNAI['hf_ready']:.0f} s", 20, INK).next_to(a_rest[0], m.RIGHT, buff=0.15)
        ready_b = code(f"ready {RUNAI['stream_ready']:.0f} s", 20, INK).next_to(b_rest[0], m.RIGHT, buff=0.15)
        spare = pill("warm spare: already loaded", TOKEN, GREEN_FILL, 20).next_to(d_lab, m.RIGHT, buff=0.5)
        assert ready_a.get_right()[0] < 6.1 and ready_a.get_top()[1] < 0.57 and spare.get_right()[0] < 6.5
        src = cite("NVIDIA, Run:ai Model Streamer, 2025 · Llama 3 8B on one A10G, cloud SSD · vLLM blog, Oct 2025 · A100")
        src.move_to([0, -3.4, 0])

        f1 = m.VGroup(code(f"15 GB ÷ 1,000 MiB/s = {T_DISK:.1f} s", 22, MEMORY),
                      code(f"the disk's limit · measured {RUNAI['stream_load']:.2f} s", 20, MUTED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        left_at(f1, -6.4, 1.85)
        f2 = code(f"{W_BYTES / 1e9:.2f} GB ÷ 5 GB/s = {T_NVME:.1f} s", 22, MEMORY)
        left_at(f2, f1.get_right()[0] + 0.6, f1[0].get_y())
        sit(f2, baseline(f1[0]))
        assert f2.get_right()[0] < 5.9

        A = 3.0 / RUNAI["hf_ready"]   # seconds of film per second of cold start
        self.say(
            "[Here is a real cold start, measured by NVIDIA]: "
            "[an eight-billion model on a cloud SSD, ready after sixty-six seconds], "
            "[forty-eight of them spent loading the weights.]",
            m.AnimationGroup(self.new_header("07 · cold starts", "Sixty-six seconds to ready"),
                             self.mini.goto(5), m.FadeIn(tl), m.FadeIn(ticks), m.FadeIn(src),
                             m.FadeIn(lane_labs["A"], shift=0.1 * m.RIGHT)),
            m.Succession(m.GrowFromEdge(a_load[0], m.LEFT, run_time=A * RUNAI["hf_load"], rate_func=m.linear),
                         m.GrowFromEdge(a_rest[0], m.LEFT, run_time=A * (RUNAI["hf_ready"] - RUNAI["hf_load"]),
                                        rate_func=m.linear),
                         m.AnimationGroup(m.FadeIn(ready_a, shift=0.1 * m.LEFT), m.FadeIn(a_rest[1]), run_time=0.4)),
            m.AnimationGroup(m.FadeIn(a_load[1]), m.Circumscribe(a_load[0], color=MEMORY, buff=0.06)),
        )
        self.say(
            "[A streamer that reads in parallel took fourteen point three]: [just what the disk allows], "
            "[fifteen gigabytes at about a gigabyte a second.]",
            m.AnimationGroup(m.FadeIn(lane_labs["B"], shift=0.1 * m.RIGHT),
                             m.Succession(m.GrowFromEdge(b_load[0], m.LEFT, run_time=A * RUNAI["stream_load"], rate_func=m.linear),
                                          m.AnimationGroup(m.FadeIn(b_load[1]), m.GrowFromEdge(
                                              b_rest[0], m.LEFT, run_time=A * (RUNAI["stream_ready"] - RUNAI["stream_load"]),
                                              rate_func=m.linear)),
                                          m.AnimationGroup(m.FadeIn(b_rest[1]), m.FadeIn(ready_b, shift=0.1 * m.LEFT),
                                                           run_time=0.4))),
            m.FadeIn(f1[0], shift=0.1 * m.UP),
            m.FadeIn(f1[1], shift=0.1 * m.UP),
        )
        self.say(
            "[From a fast local drive at five gigabytes a second], "
            "[Owl's sixteen gigabytes would load in three point two seconds.]",
            m.AnimationGroup(m.FadeIn(lane_labs["C"], shift=0.1 * m.RIGHT), m.FadeIn(f2, shift=0.1 * m.UP)),
            m.Succession(m.GrowFromEdge(c_load, m.LEFT, run_time=0.6), m.FadeIn(c_lab, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[Or skip the cold start]: [keep a warm spare], [or a sleeping one]: "
            "[vLLM's sleep mode swaps models in under three seconds, not forty-eight.]",
            m.Circumscribe(m.VGroup(a_load, a_rest), color=RED, buff=0.08),
            m.FadeIn(spare, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(lane_labs["D"], shift=0.1 * m.RIGHT), m.GrowFromEdge(d_wake, m.LEFT, run_time=0.5)),
            m.FadeIn(d_lab, shift=0.1 * m.LEFT),
        )
        self.wait(0.4)
        self.clear(self.mini)

    # ------------------------------------------------------------------ 8 · multi-LoRA

    def multi_lora(self):
        self.next_section(
            "8 · Many fine-tunes, one base",
            notes="Owl's subject fine-tunes (chemistry, history, French) and the adapter names are illustrative. "
                  f"Full copies: {COPIES} × {W_BYTES / 1e9:.2f} GB = {COPIES * W_BYTES / 1e9:.1f} GB of an H100's 80 GB. "
                  f"A LoRA adapter (for example rank {RANK} on q, k, v, o of all 32 layers of Llama-3.1-8B) adds "
                  f"r × (in + out) per matrix: {LORA_P:,} parameters, {LORA_BYTES / 1e6:.1f} MB in BF16; about "
                  f"{PER_COPY:.0f} such adapters fit in the memory of one more copy of the base. Serving: one base, "
                  "adapters selected by the request's model field (vLLM serves LoRA adapters as model names). "
                  "S-LoRA (Sheng et al., 2023, arXiv 2311.03285): 'Unified Paging' for adapter weights and KV; "
                  "'thousands of LoRA adapters on a single GPU'; 'up to 4 times' the throughput of HuggingFace PEFT "
                  "and vLLM with naive LoRA support. LoRA itself: Hu et al., 2021. vLLM's prefix-cache hash includes "
                  "the LoRA ID (notes §17).")
        base = card(4.2, 1.1, WEIGHTS, BLUE_FILL, radius=0.16, stroke=2.5).move_to([-3.95, 1.75, 0])
        base_t = m.VGroup(text("Owl-8B base", 26, INK, weight=m.BOLD), code(f"{W_BYTES / 1e9:.0f} GB · BF16", 20, WEIGHTS))
        base_t.arrange(m.DOWN, buff=0.1).move_to(base)
        base_g = m.VGroup(base, base_t)
        subjects = ["chemistry", "history", "French"]
        subj = m.VGroup(*[pill(s, WEIGHTS, BLUE_FILL, 22) for s in subjects]).arrange(m.RIGHT, buff=0.3)
        subj.move_to([2.2, 1.75, 0])
        for s in subj[1:]:
            sit(s[1], baseline(subj[0][1]))
        ex = cite("fine-tunes, for example").next_to(subj, m.UP, buff=0.15).align_to(subj, m.LEFT)

        mem = MemoryBar(GPU["memory"], width=11.0, height=0.7, title="H100 · 80 GB of HBM")
        mem.move_to([-0.55, 0.05, 0])
        x_left = mem.track.get_left()[0]
        lab_y = mem.track.get_bottom()[1] - 0.3
        seg_w = mem.bar_width * W_BYTES / GPU["memory"]
        seg_labs = m.VGroup(*[code(lab, 20, WEIGHTS).move_to([x_left + seg_w * (k + 0.5), lab_y, 0])
                              for k, lab in enumerate(["base", *subjects])])
        for s in seg_labs[1:]:
            sit(s, baseline(seg_labs[0]))
        full = code(f"{COPIES * W_BYTES / 1e9:.0f} of 80 GB · little room left for the KV cache", 20, RED)
        left_at(full, x_left, lab_y - 0.5)
        free = code("free for the KV cache", 20, KV).move_to([x_left + seg_w + (mem.bar_width - seg_w) / 2, lab_y, 0])
        sit(free, baseline(seg_labs[0]))
        zoom = card(5.6, 1.05, WEIGHTS, CARD, radius=0.14, stroke=2).move_to([-3.25, -1.85, 0])
        zoom_t = m.VGroup(text("3 adapters × 27 MB", 24, INK, weight=m.BOLD),
                          code(f"each ≈ 1/{PER_COPY:.0f} of the base", 20, WEIGHTS)).arrange(m.DOWN, buff=0.08).move_to(zoom)
        zoom_g = m.VGroup(zoom, zoom_t)
        tick = m.DashedLine([x_left + seg_w, mem.track.get_bottom()[1] - 0.02, 0], [zoom.get_top()[0] + 1.2, zoom.get_top()[1], 0],
                            dash_length=0.08)
        tick.set_stroke(WEIGHTS, 2)
        formula = code(f"rank {RANK} × (in + out) · q, k, v, o · 32 layers = {LORA_P / 1e6:.1f} M numbers", 20, MUTED)
        formula.move_to([0, -2.8, 0])
        b_src = cite("LoRA: two thin matrices beside each adapted weight · Hu et al., 2021").move_to([0, -3.3, 0])

        # frame B: one base, many adapters, one batch
        req_src = '{"model": "owl-chem",\n "messages": [ … ],\n "stream": true}'
        req = jcard(req_src, "POST /v1/chat/completions", REQUEST)
        req.move_to([-6.45 + req.width / 2, 1.6, 0])
        req[1].set_z_index(2)
        g_model = gutter(req[0], line_of(req[1], req_src, 0), REQUEST)
        b2 = card(3.0, 0.8, WEIGHTS, BLUE_FILL, radius=0.14, stroke=2.5)
        ads = m.VGroup(*[pill(f"owl-{s}", WEIGHTS, CARD, 20, mono=True) for s in ("chem", "history", "french")])
        ads.arrange(m.RIGHT, buff=0.2)
        ads.move_to([0, 1.3, 0]).align_to([req.get_right()[0] + 0.8, 0, 0], m.LEFT)
        base2 = m.VGroup(b2, text("Owl-8B base", 22, INK, weight=m.BOLD).move_to(b2)).move_to([ads.get_x(), 2.15, 0])
        assert ads.get_right()[0] < 5.9
        for a in ads[1:]:
            sit(a[1], baseline(ads[0][1]))
        ad_cap = code(f"adapters · {LORA_BYTES / 1e6:.0f} MB each", 20, WEIGHTS).next_to(ads, m.DOWN, buff=0.15)
        pick = arrow([req[0].get_right()[0], line_of(req[1], req_src, 0).get_y(), 0], ads[0].get_left(), REQUEST, buff=0.1)

        who = ["chem", "history", "french", "chem"]
        ry = [-0.75, -1.3, -1.85, -2.4]
        r_labs = m.VGroup(*[m.VGroup(icon(USER, 0.36).move_to([-5.9, y, 0]), right_at(code(f"owl-{w}", 20, REQUEST), -2.75, y))
                            for w, y in zip(who, ry)])
        shared = m.Rectangle(width=2.0, height=ry[0] - ry[-1] + 0.45).set_stroke(WEIGHTS, 2.5).set_fill(BLUE_FILL, 1)
        shared.move_to([-1.45, (ry[0] + ry[-1]) / 2, 0])
        shared_t = m.VGroup(text("base weights", 22, INK, weight=m.BOLD), code("read once", 20, WEIGHTS),
                            code("per step", 20, WEIGHTS)).arrange(m.DOWN, buff=0.08).move_to(shared)
        adds = m.VGroup(*[m.VGroup(m.RoundedRectangle(corner_radius=0.06, width=1.95, height=0.42).set_stroke(WEIGHTS, 2)
                                   .set_fill(CARD, 1).move_to([0.75, y, 0]), code(f"+ {w}", 20, WEIGHTS).move_to([0.75, y, 0]))
                          for w, y in zip(who, ry)])
        outs = m.VGroup(*[m.Square(0.3).set_stroke(TOKEN, 2).set_fill(GREEN_FILL, 1).move_to([2.1, y, 0]) for y in ry])
        slora = m.VGroup(title_text(f"up to {SLORA_X}×", 44, GREEN), text("the throughput, and", 22, INK),
                         text("thousands of adapters", 22, INK), text("on one GPU", 22, INK),
                         cite("S-LoRA · Sheng et al., 2023")).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        left_at(slora, 2.75, -1.6)
        assert slora.get_right()[0] < 6.5 and all(a[1].width < a[0].width - 0.15 for a in adds)

        self.say(
            "[Owl also has fine-tuned versions]: [one for chemistry], [one for history], [one for French.]",
            m.AnimationGroup(self.new_header("08 · multi-LoRA", "Many fine-tunes, one base"), self.mini.goto(4),
                             m.FadeIn(base_g, shift=0.1 * m.UP), m.FadeIn(ex)),
            m.FadeIn(subj[0], shift=0.1 * m.UP),
            m.FadeIn(subj[1], shift=0.1 * m.UP),
            m.FadeIn(subj[2], shift=0.1 * m.UP),
        )
        self.say(
            "[Full copies cost sixteen gigabytes each]: [four of them take sixty-four of the eighty gigabytes on an H100.]",
            m.AnimationGroup(m.FadeIn(mem.track), m.FadeIn(mem.caption), mem.set("base", W_BYTES, WEIGHTS),
                             m.FadeIn(seg_labs[0])),
            m.AnimationGroup(m.Succession(*[mem.set(s, W_BYTES, WEIGHTS, run_time=0.45) for s in subjects]),
                             m.LaggedStart(*[m.FadeIn(s) for s in seg_labs[1:]], lag_ratio=0.4), m.FadeIn(full)),
        )
        for s in subjects[:-1]:
            mem.sizes[s] = LORA_BYTES
        shrink = mem.set(subjects[-1], LORA_BYTES, WEIGHTS, run_time=1.0)
        self.say(
            "[Each LoRA adapter stores only small low-rank updates]: [about twenty-seven megabytes at rank sixteen], "
            "[so the copies shrink to slivers.]",
            m.AnimationGroup(m.FadeOut(full), m.FadeIn(formula, shift=0.1 * m.UP), m.FadeIn(b_src)),
            m.AnimationGroup(m.FadeIn(zoom_g, shift=0.1 * m.UP), m.Create(tick)),
            m.AnimationGroup(shrink, m.FadeOut(seg_labs[1:]), m.FadeIn(free)),
        )
        self.wait(0.3)
        frame_a = m.VGroup(base_g, subj, ex, mem, seg_labs[0], free, zoom_g, tick, formula, b_src)
        self.say(
            "[So one server keeps one base and many adapters], [and the request's model field picks one.]",
            m.AnimationGroup(m.FadeOut(frame_a, shift=0.15 * m.UP), m.FadeIn(base2, shift=0.15 * m.UP),
                             m.LaggedStart(*[m.FadeIn(a, shift=0.1 * m.UP) for a in ads], lag_ratio=0.2), m.FadeIn(ad_cap)),
            m.AnimationGroup(m.FadeIn(req, shift=0.15 * m.RIGHT), m.FadeIn(g_model, scale=0.6),
                             m.Succession(m.Wait(0.5), m.AnimationGroup(m.GrowArrow(pick),
                                                                        m.Circumscribe(ads[0], color=REQUEST, buff=0.08)))),
        )
        self.say(
            "[One batch can mix them]: [every row shares the base weights], [and adds its own adapter's small multiply.]",
            m.LaggedStart(*[m.FadeIn(r_, shift=0.1 * m.RIGHT) for r_ in r_labs], lag_ratio=0.2),
            m.AnimationGroup(m.GrowFromEdge(shared, m.LEFT), m.FadeIn(shared_t)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(a, shift=0.1 * m.RIGHT) for a in adds], lag_ratio=0.15),
                             m.Succession(m.Wait(0.6), m.LaggedStart(*[m.FadeIn(o, shift=0.1 * m.RIGHT) for o in outs],
                                                                     lag_ratio=0.15))),
        )
        self.say(
            "[The S-LoRA system serves thousands of adapters on one GPU], "
            "[with up to four times the throughput of simpler setups.]",
            m.FadeIn(slora[1:], shift=0.1 * m.UP),
            m.FadeIn(slora[0], scale=0.85),
        )
        self.wait(0.4)
        self.clear(self.mini)

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Route each request to the replica that holds its prefix; stream every token "
                                         "as a server-sent event the moment it is sampled; scale on queue size, before "
                                         "a cold start (weights ÷ disk bandwidth, plus engine start) bites.")
        halo = glow(MEMORY, radius=4.0)
        words = m.VGroup(*[title_text(w, 80, c) for w, c in (("Route", REQUEST), ("Stream", TOKEN), ("Scale", MEMORY))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("to the cache", "every token", "on the queue")])
        words.arrange(m.RIGHT, buff=1.3)
        on_baseline(words)
        for w_, s in zip(words, subs):
            s.set_x(w_.get_x())
        line_y = min(w_.get_bottom()[1] for w_ in words) - 0.45
        for s in subs:
            s.set_y(line_y)
        mid = words[0][0].get_center()[1]
        dots = m.VGroup(*[m.Dot([(a.get_right()[0] + b.get_left()[0]) / 2, mid, 0], radius=0.07, color=MUTED)
                          for a, b in zip(words, words[1:])])
        row_ = m.VGroup(words, dots, subs).move_to(0.45 * m.UP)
        tagline = text("Between the student and the GPU: a router, a server, an engine.", 30, MUTED).next_to(row_, m.DOWN, buff=0.75)

        self.say(
            "[To sum up]: [route], [stream], [scale.]",
            m.AnimationGroup(self.drop_header(), m.FadeOut(self.mini), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[Route each request to its cache, stream every token as it's made, and scale on the queue before it overflows.]",
            m.Write(tagline),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row_, tagline)], run_time=0.9)
