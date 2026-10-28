"""The canonical numbers and code for "Tokens per Second" — every episode that shows a formula, a number
or code quotes from here, so names, shapes and values match across the course. Episodes show SHORT
excerpts (≤ 10 lines), may simplify, but never rename things. `python3 owl_reference.py` runs the checks.

The running example is Owl, a study-buddy chatbot (students paste notes, ask questions, get answers and
JSON flashcards). Owl runs Llama-3.1-8B-Instruct ("Owl-8B"): 8 B parameters, 16 GB of weights in BF16,
128 KiB of KV cache per token. Owl-mini (Llama-3.2-1B, 4-bit) runs the live demos on the Mac.
Numbers measured on this Mac live in research/measurements.json (load them with `measured()`).
Facts as of October 2026; sources in research/notes.md.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------- models (config.json)

MODELS = {
    "owl-8b": dict(name="Llama-3.1-8B", params=8.03e9, layers=32, d_model=4096, heads=32, kv_heads=8,
                   head_dim=128, ffn=14336, vocab=128_256),
    "owl-mini": dict(name="Llama-3.2-1B", params=1.24e9, layers=16, d_model=2048, heads=32, kv_heads=8,
                     head_dim=64, ffn=8192, vocab=128_256),
    "owl-70b": dict(name="Llama-3.1-70B", params=70.55e9, layers=80, d_model=8192, heads=64, kv_heads=8,
                    head_dim=128, ffn=28672, vocab=128_256),
}
OWL = MODELS["owl-8b"]

# ---------------------------------------------------------------------------- hardware (vendor pages, Oct 2026)
# bandwidth in bytes/s, peak = dense BF16 FLOP/s (None where the vendor publishes none), memory in bytes

HARDWARE = {
    "H100": dict(name="H100 SXM", bandwidth=3.35e12, peak=989e12, memory=80e9),
    "A100": dict(name="A100 80GB SXM", bandwidth=2.039e12, peak=312e12, memory=80e9),
    "H200": dict(name="H200 SXM", bandwidth=4.8e12, peak=989e12, memory=141e9),
    "M4 Max": dict(name="Apple M4 Max (this Mac)", bandwidth=546e9, peak=None, memory=64e9),
    "M4 Pro": dict(name="Apple M4 Pro", bandwidth=273e9, peak=None, memory=64e9),
    "M4": dict(name="Apple M4", bandwidth=120e9, peak=None, memory=32e9),
}
GPU = HARDWARE["H100"]
MAC = HARDWARE["M4 Max"]

# ---------------------------------------------------------------------------- the arithmetic (episodes 2–9)


def weight_bytes(model=OWL, bits=16):
    """Bytes of weights read on every decode step."""
    return model["params"] * bits / 8


def kv_bytes_per_token(model=OWL, bits=16):
    """One token's keys and values, in every layer: 2 × layers × kv_heads × head_dim × bytes."""
    return 2 * model["layers"] * model["kv_heads"] * model["head_dim"] * bits / 8


def flops_per_token(model=OWL):
    """A forward pass does about one multiply-add (2 FLOPs) per parameter per token."""
    return 2 * model["params"]


def ridge(hw=GPU):
    """FLOPs per byte where the roofline bends: below it a kernel is memory-bound."""
    return hw["peak"] / hw["bandwidth"]


def step_time(model=OWL, hw=GPU, batch=1, context=0, bits=16, kv_bits=16):
    """Roofline time of one decode step (s): read weights + every sequence's KV, or do the math."""
    memory = (weight_bytes(model, bits) + batch * context * kv_bytes_per_token(model, kv_bits)) / hw["bandwidth"]
    compute = batch * flops_per_token(model) / hw["peak"] if hw["peak"] else 0.0
    return max(memory, compute)


def decode_ceiling(model=OWL, hw=GPU, bits=16, context=0):
    """The batch-1 speed limit in tokens/s: bandwidth ÷ bytes per token."""
    return 1 / step_time(model, hw, 1, context, bits)


def prefill_time(n_prompt, model=OWL, hw=GPU, mfu=0.5):
    """Prompt tokens go through in one parallel pass: compute-bound (attention ignored)."""
    return flops_per_token(model) * n_prompt / (hw["peak"] * mfu)


def measured(*keys):
    """A number measured on this Mac: measured("decode", "llama-3.1-8b", "bf16", "tok_s_median")."""
    value = json.loads((HERE / "research" / "measurements.json").read_text())
    for key in keys:
        value = value[key]
    return value


# ---------------------------------------------------------------------------- a tiny transformer (runs in NumPy)
# Small enough to step through on screen, real enough that its KV cache really works: decoding with the
# cache gives exactly the logits of recomputing the whole sequence (checked below).


@dataclass
class KVCache:
    keys: list = field(default_factory=list)    # one (tokens, kv_heads, head_dim) array per layer
    values: list = field(default_factory=list)

    def __len__(self):
        return 0 if not self.keys else self.keys[0].shape[0]


class TinyLM:
    def __init__(self, vocab=32, d_model=16, layers=2, heads=4, kv_heads=2, seed=0):
        rng = np.random.default_rng(seed)
        self.vocab, self.d, self.n_layers = vocab, d_model, layers
        self.heads, self.kv_heads, self.hd = heads, kv_heads, d_model // heads
        w = lambda *shape: rng.normal(0, shape[0] ** -0.5, shape)
        self.embed = w(vocab, d_model)
        self.blocks = [dict(q=w(d_model, heads * self.hd), k=w(d_model, kv_heads * self.hd),
                            v=w(d_model, kv_heads * self.hd), o=w(heads * self.hd, d_model),
                            up=w(d_model, 4 * d_model), down=w(4 * d_model, d_model)) for _ in range(layers)]

    def forward(self, tokens, cache):
        """Run new tokens through the model, appending their K/V to `cache`; returns logits per token."""
        x = self.embed[np.asarray(tokens)]
        start, n = len(cache), len(tokens)
        for i, b in enumerate(self.blocks):
            h = x / np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-6)            # RMS norm
            q = (h @ b["q"]).reshape(n, self.heads, self.hd)
            k = (h @ b["k"]).reshape(n, self.kv_heads, self.hd)
            v = (h @ b["v"]).reshape(n, self.kv_heads, self.hd)
            if len(cache.keys) <= i:
                cache.keys.append(k), cache.values.append(v)
            else:
                cache.keys[i] = np.concatenate([cache.keys[i], k])
                cache.values[i] = np.concatenate([cache.values[i], v])
            K = np.repeat(cache.keys[i], self.heads // self.kv_heads, axis=1)     # GQA: heads share K/V
            V = np.repeat(cache.values[i], self.heads // self.kv_heads, axis=1)
            scores = np.einsum("nhd,thd->hnt", q, K) / np.sqrt(self.hd)
            later = np.arange(start + n)[None, :] > (start + np.arange(n))[:, None]  # causal mask
            scores[:, later] = -np.inf
            att = np.exp(scores - scores.max(-1, keepdims=True))
            att /= att.sum(-1, keepdims=True)
            x = x + np.einsum("hnt,thd->nhd", att, V).reshape(n, -1) @ b["o"]
            h = x / np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-6)
            x = x + np.maximum(h @ b["up"], 0) @ b["down"]
        return x @ self.embed.T

    def prefill(self, prompt):
        """One parallel pass over the whole prompt: fills the cache, returns the next-token logits."""
        cache = KVCache()
        return self.forward(prompt, cache)[-1], cache

    def decode(self, token, cache):
        """One step: one new token in, the cache grows by one, the next logits out."""
        return self.forward([token], cache)[-1]


def sample(logits, temperature=1.0, top_p=1.0, rng=None):
    """Logits → one token id. temperature 0 means greedy (argmax)."""
    if temperature == 0:
        return int(np.argmax(logits))
    p = np.exp((logits - logits.max()) / temperature)
    p /= p.sum()
    order = np.argsort(-p)
    keep = order[: int(np.searchsorted(np.cumsum(p[order]), top_p)) + 1]          # nucleus
    rng = rng or np.random.default_rng(0)
    return int(rng.choice(keep, p=p[keep] / p[keep].sum()))


def generate(model, prompt, max_new_tokens, temperature=0.0):
    logits, cache = model.prefill(prompt)        # one big parallel pass
    out = []
    for _ in range(max_new_tokens):              # then one token per step
        token = sample(logits, temperature)
        out.append(token)
        logits = model.decode(token, cache)       # reads every weight once
    return out


# ---------------------------------------------------------------------------- scheduling (episodes 10–12)


@dataclass
class Request:
    rid: str
    arrival: int        # the step it arrives
    prompt: int         # prompt tokens (prefilled in one step here)
    output: int         # tokens to generate


def static_batching(requests, slots):
    """A batch starts together and ends when its longest request ends. Returns {rid: (start, end)}."""
    queue, t, done = sorted(requests, key=lambda r: r.arrival), 0, {}
    while queue:
        t = max(t, queue[0].arrival)
        batch = [r for r in queue if r.arrival <= t][:slots]
        for r in batch:
            done[r.rid] = (t, t + 1 + r.output)
            queue.remove(r)
        t += 1 + max(r.output for r in batch)
    return done


def continuous_batching(requests, slots):
    """Iteration-level scheduling: a finished request's slot is refilled at the very next step."""
    queue, running, t, done = sorted(requests, key=lambda r: r.arrival), {}, 0, {}
    while queue or running:
        for r in [r for r in queue if r.arrival <= t][: slots - len(running)]:
            running[r.rid] = [t, 1 + r.output]
            queue.remove(r)
        t += 1
        for rid in list(running):
            running[rid][1] -= 1
            if running[rid][1] == 0:
                done[rid] = (running.pop(rid)[0], t)
    return done


# ---------------------------------------------------------------------------- speculative decoding (episode 13)


def expected_tokens_per_pass(alpha, gamma):
    """Leviathan et al. 2023: tokens produced per target pass with γ drafted tokens, acceptance rate α."""
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


def speculative_speedup(alpha, gamma, c):
    """…and the wall-clock speedup when one draft step costs c target steps."""
    return expected_tokens_per_pass(alpha, gamma) / (gamma * c + 1)


# ---------------------------------------------------------------------------- self-checks


if __name__ == "__main__":
    assert weight_bytes() == 16.06e9                                      # "16 GB of weights"
    assert kv_bytes_per_token() == 131_072                                 # "128 KiB per token"
    assert kv_bytes_per_token(MODELS["owl-mini"]) == 32_768                # Owl-mini: 32 KiB
    assert kv_bytes_per_token() * 131_072 == 16 * 2 ** 30                  # a 128K chat = 16 GiB
    assert round(decode_ceiling(hw=GPU)) == 209 and round(decode_ceiling(hw=MAC)) == 34
    assert round(ridge(GPU)) == 295
    assert round(prefill_time(1500) * 1000) == 49                          # §1: ≈ 49 ms at 50% MFU
    assert round(step_time(context=1800) * 1000, 2) == 4.86                # §1: 4.86 ms floor
    tiny = TinyLM()
    prompt = [3, 14, 15, 9, 2, 6]
    logits, cache = tiny.prefill(prompt)
    seq = list(prompt)
    for _ in range(5):                                                     # cache == recompute, every step
        tok = sample(logits, 0)
        seq.append(tok)
        logits = tiny.decode(tok, cache)
        assert np.allclose(logits, tiny.forward(seq, KVCache())[-1])
    assert len(cache) == len(prompt) + 5
    assert generate(tiny, prompt, 5) == seq[len(prompt):]
    reqs = [Request("A", 0, 8, 12), Request("B", 0, 8, 3), Request("C", 0, 8, 4), Request("D", 0, 8, 2),
            Request("E", 1, 8, 3)]
    s, c = static_batching(reqs, 4), continuous_batching(reqs, 4)
    assert s["E"][0] == 13 and c["E"][0] == 3                              # E waits 12 steps vs 2
    assert round(expected_tokens_per_pass(0.8, 4), 2) == 3.36              # §6
    print("owl_reference: all checks passed")
    print(f"  8B BF16 ceiling: H100 {decode_ceiling(hw=GPU):.0f} tok/s, M4 Max {decode_ceiling(hw=MAC):.1f} tok/s")
    print(f"  static vs continuous, E starts at step {s['E'][0]} vs {c['E'][0]}")
