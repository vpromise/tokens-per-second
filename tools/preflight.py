"""Pronunciation pre-flight for "Tokens per Second": speak one sentence per risky term through the running
narrator (which reads lexicon.json), then transcribe each clip with Whisper large and small, and report
which terms both listeners heard. Run with the narrator env, from the series folder:
    ~/rag-video/tts/.venv/bin/python tools/preflight.py [out_dir]"""

import base64
import io
import json
import re
import sys
import urllib.request
from pathlib import Path

import librosa
import mlx_whisper
import soundfile

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "research/preflight")
OUT.mkdir(parents=True, exist_ok=True)

# (term as written in the script, sentence, regex a correct transcript must match)
TESTS = [
    ("KV", "The KV cache grows by one entry per token.", r"\bkv\b|\bk\.?v\.?\b|\bkay vee\b"),
    ("GPU", "One GPU reads every weight for every token.", r"\bgpu\b"),
    ("H100", "On an H100, the ceiling is about two hundred tokens per second.", r"h ?100|h one hundred|h-100"),
    ("MLX", "We measured it with MLX on this Mac.", r"\bmlx\b|m l x"),
    ("vLLM", "vLLM made paged attention famous.", r"\bv ?llm\b|v l l m"),
    ("SGLang", "SGLang keeps a radix tree of prefixes.", r"sg ?lang|s g lang"),
    ("PagedAttention", "PagedAttention splits the cache into blocks.", r"paged attention"),
    ("FlashAttention", "FlashAttention never stores the full score matrix.", r"flash ?attention"),
    ("llama.cpp", "llama.cpp runs the same model on a laptop.", r"llama ?(\.|dot )?c ?p ?p|llama cpp"),
    ("FLOPs", "Each token costs about sixteen billion FLOPs.", r"\bflops\b"),
    ("FP8", "FP8 halves the bytes of every weight.", r"\bfp ?8\b|fp eight"),
    ("INT4", "INT4 weights shrink the model to a quarter.", r"\bint ?4\b|int four"),
    ("GQA", "GQA lets query heads share key and value heads.", r"\bgqa\b|g q a"),
    ("MLA", "MLA compresses keys and values into a small latent.", r"\bmla\b|m l a"),
    ("MoE", "An MoE model reads only a few experts per token.", r"\bmoe\b|m o e|\bmo e\b"),
    ("HBM", "HBM is fast, but the chip's SRAM is faster.", r"\bhbm\b|h b m"),
    ("SRAM", "Tiles of keys and values move through SRAM.", r"\bs ?ram\b|s ram"),
    ("CUDA", "CUDA graphs replay hundreds of kernels at once.", r"\bcuda\b"),
    ("NVLink", "NVLink connects the GPUs at nine hundred gigabytes per second.", r"nv ?link|nvlink"),
    ("PCIe", "PCIe is several times slower.", r"pci ?e|pcie|p c i e"),
    ("TTFT", "TTFT is the time to the first token.", r"\bttft\b|t t f t"),
    ("DeepSeek", "DeepSeek splits prefill and decode onto different machines.", r"deep ?seek"),
    ("Mixtral", "Mixtral has eight experts in every layer.", r"mixtral"),
    ("GPTQ", "GPTQ corrects the error of each rounded weight.", r"\bgptq\b|g p t q"),
    ("AWQ", "AWQ protects the most important channels.", r"\bawq\b|a w q"),
    ("GGUF", "GGUF files pack weights and scales together.", r"\bgguf\b|g g u f"),
    ("EAGLE", "EAGLE drafts tokens from the model's own features.", r"\beagle\b"),
    ("Medusa", "Medusa adds extra heads that guess ahead.", r"\bmedusa\b"),
    ("Owl", "Owl answers the student in two seconds.", r"\bowl\b"),
    ("prefill", "The prefill runs all prompt tokens at once.", r"\bpre-?fill\b"),
    ("roofline", "The roofline shows both speed limits.", r"roof ?line"),
    ("kibibytes", "Each token needs a hundred and twenty-eight kibibytes.", r"kibibytes?"),
    ("LoRA", "One base model can serve a thousand LoRA adapters.", r"\blora\b|\blaura\b|\blora\b"),
]

LISTENERS = {"large": "mlx-community/whisper-large-v3-turbo", "small": "mlx-community/whisper-small-mlx"}


def speak(sentence):
    settings = dict(voice_ref="", exaggeration=0.5, cfg_weight=0.3, temperature=0.8)  # the same voice as kit.narrator
    req = urllib.request.Request("http://127.0.0.1:8765/speak", json.dumps({"text": sentence, **settings}).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as reply:
        return base64.b64decode(json.load(reply)["audio"])


rows = []
for i, (term, sentence, pattern) in enumerate(TESTS):
    wav = OUT / f"{i:02d}-{re.sub(r'[^A-Za-z0-9]+', '_', term)}.wav"
    if not wav.exists():
        wav.write_bytes(speak(sentence))
    audio, _ = librosa.load(io.BytesIO(wav.read_bytes()), sr=16000)
    heard = {name: mlx_whisper.transcribe(audio, path_or_hf_repo=repo, language="en")["text"].strip()
             for name, repo in LISTENERS.items()}
    ok = {name: bool(re.search(pattern, t.lower())) for name, t in heard.items()}
    rows.append(dict(term=term, sentence=sentence, heard=heard, ok=ok))
    mark = "ok  " if all(ok.values()) else ("half" if any(ok.values()) else "FAIL")
    print(f"{mark} {term:15} large: {heard['large'][:70]!r} | small: {heard['small'][:70]!r}", flush=True)

(OUT / "preflight.json").write_text(json.dumps(rows, indent=1))
print(f"{sum(all(r['ok'].values()) for r in rows)}/{len(rows)} terms heard by both listeners")
