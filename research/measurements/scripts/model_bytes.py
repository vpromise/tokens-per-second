"""Weight bytes and KV bytes/token for an MLX model, from the files on disk.

Reads only the safetensors JSON headers (no tensors loaded) and config.json.

Outputs, per model:
  file_bytes          sum of *.safetensors file sizes (what the course calls "size")
  tensor_bytes        sum of tensor payload bytes from the headers
  embed_bytes         bytes of model.embed_tokens.* (weight + scales + biases)
  tied                tie_word_embeddings
  decode_bytes_per_token
                      bytes a batch-1 decode step must read from weights:
                      tensor_bytes, minus the input embedding table if it is
                      NOT tied (an untied input embedding is a row lookup:
                      one row is read, not the whole table). If tied, the
                      same matrix is the LM head and is read in full.
  kv_bytes_per_token  2 * layers * kv_heads * head_dim * 2 (bf16 cache)

Usage: python -I model_bytes.py <repo_id> [<repo_id> ...]  (prints JSON)
"""
import glob
import json
import os
import struct
import sys

from huggingface_hub import snapshot_download


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def analyse(repo):
    d = snapshot_download(repo, local_files_only=True)
    cfg = json.load(open(os.path.join(d, "config.json")))
    files = sorted(glob.glob(os.path.join(d, "*.safetensors")))
    file_bytes = sum(os.path.getsize(os.path.realpath(f)) for f in files)
    tensor_bytes = 0
    embed_bytes = 0
    n_params_stored = 0
    for f in files:
        for name, t in header(f).items():
            if name == "__metadata__":
                continue
            b = t["data_offsets"][1] - t["data_offsets"][0]
            tensor_bytes += b
            if "embed_tokens" in name:
                embed_bytes += b
    tied = bool(cfg.get("tie_word_embeddings", False))
    L = cfg["num_hidden_layers"]
    n_kv = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
    hd = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    vocab = cfg["vocab_size"]
    hidden = cfg["hidden_size"]
    row_bytes = embed_bytes / vocab  # one embedding row (incl. its scales)
    decode_bytes = tensor_bytes if tied else tensor_bytes - embed_bytes + row_bytes
    q = cfg.get("quantization")
    return {
        "repo": repo,
        "snapshot": d,
        "files": [os.path.basename(f) for f in files],
        "file_bytes": file_bytes,
        "tensor_bytes": tensor_bytes,
        "embed_bytes": embed_bytes,
        "tied_embeddings": tied,
        "decode_bytes_per_token": int(decode_bytes),
        "quantization": q,
        "layers": L,
        "hidden_size": hidden,
        "attention_heads": cfg["num_attention_heads"],
        "kv_heads": n_kv,
        "head_dim": hd,
        "vocab_size": vocab,
        "kv_bytes_per_token_bf16": 2 * L * n_kv * hd * 2,
    }


if __name__ == "__main__":
    print(json.dumps([analyse(r) for r in sys.argv[1:]], indent=1))
