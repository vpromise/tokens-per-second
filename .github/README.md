# Tokens per Second — how LLM inference really works

A video course in 26 short episodes (3.5–4.5 minutes each), narrated and animated. It follows one request — a student pressing Enter in **Owl**, a study-buddy chatbot — from the keyboard to the last token, and every technique that makes that trip faster, cheaper and possible for thousands of users at once: the KV cache, batching, paging, speculative decoding, FlashAttention, quantization, MoE, parallelism, the serving stack, and the Mac on your desk.

Part of a family of narrated, animated video courses, rendered with [ManimGX](https://pypi.org/project/manimgx/) and voiced by a local narrator ([tutorial-video](https://github.com/vpromise/tutorial-video)):

- **AI, Explained** — 46 episodes on modern AI, from first principles to frontier research · [ai-explained](https://github.com/vpromise/ai-explained)
- **Agents, Animated** — how AI agents work, from one loop to a team · [agents-animated](https://github.com/vpromise/agents-animated)
- **Tokens per Second** — how LLM inference really works *(this repository)*
- **Trial and Reward** — reinforcement learning, from slot machines to reasoning models · [trial-and-reward](https://github.com/vpromise/trial-and-reward)
- **Out of the Noise** — how diffusion models make pictures and video · [out-of-the-noise](https://github.com/vpromise/out-of-the-noise)

This repository holds the **source code** of the course: one scene per episode, the series kit, reference code, research notes and the production tools. The rendered videos are not included.

## Episodes


### Module 1 · One Token, Two Speed Limits

| # | Episode | What you'll understand | Length | Source |
|---|---|---|---|---|
| 1 | Press Enter | A reply is one parallel prefill plus hundreds of one-token decode steps | 3:55 | [`ep01-press-enter/scene.py`](../ep01-press-enter/scene.py) |
| 2 | The Two Speed Limits | Every step is limited by math or by memory traffic; for chat it's memory | 3:46 | [`ep02-two-speed-limits/scene.py`](../ep02-two-speed-limits/scene.py) |
| 3 | Prefill vs Decode | The same model runs two different workloads | 3:49 | [`ep03-prefill-vs-decode/scene.py`](../ep03-prefill-vs-decode/scene.py) |
| 4 | The Batching Free Lunch | Serving 64 users costs about the same weight traffic as serving one | 4:19 | [`ep04-batching-free-lunch/scene.py`](../ep04-batching-free-lunch/scene.py) |

### Module 2 · The KV Cache

| # | Episode | What you'll understand | Length | Source |
|---|---|---|---|---|
| 5 | The KV Cache | The model remembers past tokens as keys and values, and that memory has a size | 3:39 | [`ep05-kv-cache/scene.py`](../ep05-kv-cache/scene.py) |
| 6 | Shrinking the Cache | Five tricks make the cache smaller | 3:44 | [`ep06-shrinking-the-cache/scene.py`](../ep06-shrinking-the-cache/scene.py) |
| 7 | PagedAttention | Fixed-size blocks and a block table end fragmentation | 3:41 | [`ep07-paged-attention/scene.py`](../ep07-paged-attention/scene.py) |
| 8 | Never Prefill Twice | A shared prefix can be cached and reused | 3:44 | [`ep08-never-prefill-twice/scene.py`](../ep08-never-prefill-twice/scene.py) |
| 9 | The Cost of a Long Memory | Long contexts cost quadratically in prefill and linearly in decode | 3:38 | [`ep09-long-memory/scene.py`](../ep09-long-memory/scene.py) |

### Module 3 · Many Users at Once

| # | Episode | What you'll understand | Length | Source |
|---|---|---|---|---|
| 10 | Continuous Batching | Re-form the batch every step, not every request | 3:45 | [`ep10-continuous-batching/scene.py`](../ep10-continuous-batching/scene.py) |
| 11 | Chunked Prefill | One long prompt must not freeze everyone else's tokens | 3:50 | [`ep11-chunked-prefill/scene.py`](../ep11-chunked-prefill/scene.py) |
| 12 | Split the Work | Prefill and decode can run on different machines | 3:56 | [`ep12-split-the-work/scene.py`](../ep12-split-the-work/scene.py) |

### Module 4 · Faster Tokens

| # | Episode | What you'll understand | Length | Source |
|---|---|---|---|---|
| 13 | Guess and Check | A small model drafts, the big one verifies — exactly lossless | 4:30 | [`ep13-guess-and-check/scene.py`](../ep13-guess-and-check/scene.py) |
| 14 | Drafting Without a Draft Model | The model can draft for itself | 3:50 | [`ep14-drafting-without-a-draft/scene.py`](../ep14-drafting-without-a-draft/scene.py) |
| 15 | FlashAttention | Exact attention, reordered to respect the memory hierarchy | 4:01 | [`ep15-flashattention/scene.py`](../ep15-flashattention/scene.py) |
| 16 | The CPU Tax | At batch 1, launching work can cost more than doing it | 3:58 | [`ep16-cpu-tax/scene.py`](../ep16-cpu-tax/scene.py) |
| 17 | The Last Step of a Token | Turning logits into one token: sampling, grammars, and determinism | 4:30 | [`ep17-last-step-of-a-token/scene.py`](../ep17-last-step-of-a-token/scene.py) |

### Module 5 · Fewer Bits, More GPUs

| # | Episode | What you'll understand | Length | Source |
|---|---|---|---|---|
| 18 | Fewer Bits, Same Model | Fewer bits per weight means fewer bytes per token | 4:20 | [`ep18-fewer-bits/scene.py`](../ep18-fewer-bits/scene.py) |
| 19 | Four Bits and Below | 4-bit formats keep quality with clever scales | 4:28 | [`ep19-four-bits/scene.py`](../ep19-four-bits/scene.py) |
| 20 | Mixture of Experts at Serve Time | MoE reads few experts per token but must hold all of them | 4:10 | [`ep20-moe-at-serve-time/scene.py`](../ep20-moe-at-serve-time/scene.py) |
| 21 | When One GPU Isn't Enough | Splitting a model across GPUs, and what the wires cost | 4:19 | [`ep21-one-gpu-isnt-enough/scene.py`](../ep21-one-gpu-isnt-enough/scene.py) |

### Module 6 · Production & Your Mac

| # | Episode | What you'll understand | Length | Source |
|---|---|---|---|---|
| 22 | The Serving Stack | What sits between the user and the GPU | 4:26 | [`ep22-serving-stack/scene.py`](../ep22-serving-stack/scene.py) |
| 23 | Your Mac as an Inference Server | Unified memory makes big models local; bandwidth sets the speed | 4:32 | [`ep23-your-mac/scene.py`](../ep23-your-mac/scene.py) |
| 24 | The Price of a Token | Cost per million tokens from first principles | 4:05 | [`ep24-price-of-a-token/scene.py`](../ep24-price-of-a-token/scene.py) |
| 25 | Keeping It Alive | What to watch and what to do when load spikes | 4:15 | [`ep25-keeping-it-alive/scene.py`](../ep25-keeping-it-alive/scene.py) |
| 26 | One Request, Revisited | How every piece fits into one fast, cheap request | 4:28 | [`ep26-one-request-revisited/scene.py`](../ep26-one-request-revisited/scene.py) |

## Repository layout

| Path | What |
|---|---|
| `COURSE.md` | the full syllabus: key concepts, what each episode animates, the running example |
| `kit.py` | the series kit: palette, fonts, components, intro and outro cards |
| `owl_reference.py` | canonical numbers and the cost model for Owl, the running example |
| `epNN-*/scene.py` | one narrated scene per episode (`self.say("…[cue]…", animation)`: the narration drives the timing) |
| `assets/` | icons and the backdrop |
| `lexicon.json` | pronunciation overrides for the narrator |
| `research/` | research notes (sources, dated facts) and the measurement or experiment scripts |
| `briefs/` | the production briefs each episode was made from, and the QA tools (`check.sh`, `make_index.py`) |
| `LISTEN.md` | narration lines flagged for a human listen |

## Rendering an episode

Requirements: macOS on Apple silicon (the narrator runs Chatterbox on MPS and Whisper on MLX), ManimGX (`uv tool install --python 3.13 manimgx`) and the [tutorial-video](https://github.com/vpromise/tutorial-video) toolkit (set it up with its `scripts/setup_tts.sh`).

```sh
cd epNN-*/
VOICE=draft manimgx inspect scene.py      # silent draft: layout check + storyboards
# in the course folder, start the narrator (it reads lexicon.json):
#   <tutorial-video>/scripts/narrator.sh
manimgx render scene.py                    # MP4 + SRT captions (speaks new lines once, then caches them)
manimgx present scene.py --no-open         # HTML slides
```

## Not in this repository

Rendered MP4s, captions, HTML slides, the narration cache (`voice/`), storyboards and logs.

Measurements were made with MLX on an Apple M4 Max (64 GB). Their summaries are included (`research/measurements.md`, `research/measurements.json`); raw outputs and logs (`research/measurements/raw/`, `logs/`) are not — `research/measurements/scripts/` regenerates them. A few scenes read those logs.
