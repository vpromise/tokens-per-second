"""Parameter counts from safetensors headers of an unquantized (bf16) MLX repo.
Prints total, embedding, lm_head, and 'matmul' params (those used per token in a
forward pass, excluding the input-embedding lookup and the LM head).
Usage: python -I param_counts.py <repo> ..."""
import glob, json, math, os, struct, sys
from huggingface_hub import snapshot_download
out = []
for repo in sys.argv[1:]:
    d = snapshot_download(repo, local_files_only=True)
    cfg = json.load(open(os.path.join(d, "config.json")))
    tot = emb = head = 0
    for f in glob.glob(os.path.join(d, "*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; h = json.loads(fh.read(n))
        for k, t in h.items():
            if k == "__metadata__": continue
            p = math.prod(t["shape"]); tot += p
            if "embed_tokens" in k: emb += p
            if "lm_head" in k: head += p
    tied = cfg.get("tie_word_embeddings", False)
    head_eff = emb if tied else head
    out.append({"repo": repo, "params_total": tot, "embed_params": emb, "lm_head_params": head_eff,
                "tied": tied, "matmul_params_excl_embed_and_head": tot - emb - (0 if tied else head)})
print(json.dumps(out, indent=1))
