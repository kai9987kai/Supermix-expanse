"""Which graft moves Archimedes' replay loss? CE on replay rows under masks/ablations."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
import expanse_core as ec  # noqa: E402
import train_expanse as te  # noqa: E402

torch.set_num_threads(8)
ckpt = sys.argv[1] if len(sys.argv) > 1 else str(ec.PATHS["grafted"])
rows = ec.load_replay()[::78][:48]
E, tokE, _ = ec.load_expanse(ckpt)
A, tokA, _ = ec.load_archimedes(str(ec.PATHS["arch_final"]))
E.eval(); A.eval()
base = tokA.vocab_size


@torch.no_grad()
def ce(model, tok, mask_new=False, **kw):
    x, y, pl, meta, _ = te.encode_rows(rows, tok, 128)
    tot = n = 0.0
    for b in range(0, x.shape[0], 8):
        xb, yb = x[b:b + 8], y[b:b + 8]
        feats = ec.arch.OmniCore.featurize([r["user"] for r in meta[b:b + 8]])
        extra = {}
        if hasattr(model, "omni7") and model.omni7 is not None:
            extra["omni7_state"] = model.omni7_state_for([r["user"] for r in meta[b:b + 8]])
        lg = model(xb, omni_features=feats, return_mtp=False, **extra, **kw).logits
        if mask_new:
            lg = lg.clone(); lg[..., base:] = float("-inf")
        c = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), yb[:, 1:].reshape(-1), reduction="sum")
        tot += float(c); n += int((yb[:, 1:] != -100).sum())
    return tot / n


print(f"archimedes                 {ce(A, tokA):.4f}")
print(f"expanse as built           {ce(E, tokE):.4f}")
print(f"expanse, new-vocab masked  {ce(E, tokE, mask_new=True):.4f}")
print(f"expanse, fly off           {ce(E, tokE, use_fly=False):.4f}")
print(f"expanse, masked + fly off  {ce(E, tokE, mask_new=True, use_fly=False):.4f}")
emb = E.embed_tokens.weight.detach()
print(f"embedding row norm: old mean {emb[:base].norm(dim=1).mean():.3f}  new mean {emb[base:].norm(dim=1).mean():.3f}")
