"""Stage 3: consolidate Supermix Expanse's heterogeneous grafts into native v2 blocks.

The v1 model already contains useful but partly separate source systems:
Archimedes, Qwen/BioMed-derived donor experts, Omni branches, FlyCore and the
full male-CNS connectome.  This trainer makes them supervise a shared latent
space, then bakes that supervision into causal latent-MoE residual blocks that
remain in the final checkpoint after the teacher-fusion machinery is discarded.

Typical CPU run (start from the trained v1 checkpoint):

    python expanse/train_consolidation_v2.py --steps 800 --batch 4 --threads 8

Quick structural/training smoke run:

    python expanse/train_consolidation_v2.py --smoke --threads 2

Training phases
---------------
1. alignment: heterogeneous source projectors learn a common 512-d geometry;
   native latent blocks start matching it while their zero output gates protect
   the v1 function at initialisation;
2. joint: language loss + representation consolidation + router regularisation;
3. bake: the teacher-fusion bank is frozen, forcing the student to chase a fixed
   target rather than co-adapting the target indefinitely;
4. teacher-free finish: teacher losses are removed and only the self-contained
   v2 model is optimised on the mixed corpus.

The final ''supermix_expanse_v2.pt'' contains no Qwen/BioMed/teacher projector
runtime dependency beyond the donor weights that were already physically grafted
into Expanse v1.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import consolidation_v2 as cv2  # noqa: E402
import expanse_core as ec  # noqa: E402
import train_expanse as te  # noqa: E402

PARTIAL_SCHEMA = "supermix-expanse-v2-partial-v1"


def log(*args: Any) -> None:
    print("[v2]", *args, flush=True)


def parse_layers(text: str) -> Tuple[int, ...]:
    vals = tuple(sorted({int(x.strip()) for x in text.split(",") if x.strip()}))
    if not vals:
        raise argparse.ArgumentTypeError("--layers needs at least one layer index")
    return vals


def atomic_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def encode_omni7(model, rows: Sequence[Mapping[str, Any]], batch: int = 64) -> Optional[torch.Tensor]:
    if getattr(model, "omni7", None) is None:
        return None
    out: List[torch.Tensor] = []
    users = [str(r["user"]) for r in rows]
    for i in range(0, len(users), batch):
        out.append(model.omni7_state_for(users[i:i + batch]).cpu())
    return torch.cat(out, 0) if out else torch.zeros(0, int(model.omni7.state_dim))


def dynamic_source_dims(model, omni_features: torch.Tensor, o7: Optional[torch.Tensor]) -> Dict[str, int]:
    h = int(model.config.hidden_size)
    dims: Dict[str, int] = {"arch": h}
    rec = getattr(model, "donor_receipts", {}) or {}
    if "qwen" in rec:
        dims["qwen"] = h
    if "biomedlm" in rec:
        dims["biomedlm"] = h
    if o7 is not None:
        dims["omni7"] = int(o7.shape[-1])
    if omni_features.numel():
        dims["omni"] = int(omni_features.shape[-1])
    if getattr(model, "fly_core", None) is not None:
        dims["fly"] = h
    if getattr(model, "cns_full", None) is not None:
        dims["cns"] = h
    return dims


def phase_weights(progress: float, bake_start: float, teacher_free_start: float) -> Dict[str, float]:
    return cv2.consolidation_phase_weights(progress, bake_start, teacher_free_start)


def set_bank_trainable(bank: cv2.TeacherFusionBank, trainable: bool) -> None:
    bank.train(trainable)
    for p in bank.parameters():
        p.requires_grad_(trainable)


def make_optimizer(model, bank, *, lr_trunk: float, lr_graft: float, lr_v2: float, lr_align: float):
    trunk, graft, native = [], [], []
    frozen_names: List[str] = []
    for name, p in model.named_parameters():
        if name.startswith("consolidation_v2."):
            p.requires_grad_(True)
            native.append(p)
            continue
        if name.startswith("omni7.net.") or (
            name.startswith("omni_core.")
            and not name.startswith(("omni_core.to_trunk", "omni_core.gate", "omni_core.bridge_norm"))
        ):
            p.requires_grad_(False)
            frozen_names.append(name)
            continue
        is_graft = name.startswith(("fly_core.", "omni_core.", "omni7.", "cns_full."))
        if not is_graft and ".mlp.experts." in name:
            parts = name.split(".")
            try:
                li, slot = int(parts[1]), int(parts[4])
                donor = set(model.donor_slots().get(li, [])) if hasattr(model, "donor_slots") else set()
                is_graft = slot in donor
            except (ValueError, IndexError):
                pass
        (graft if is_graft else trunk).append(p)
    groups = [
        {"params": trunk, "lr": lr_trunk, "weight_decay": 0.01, "name": "trunk"},
        {"params": graft, "lr": lr_graft, "weight_decay": 0.0, "name": "graft"},
        {"params": native, "lr": lr_v2, "weight_decay": 0.0, "name": "v2"},
        {"params": list(bank.parameters()), "lr": lr_align, "weight_decay": 1e-4, "name": "fusion"},
    ]
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95))
    counts = {g["name"]: sum(p.numel() for p in g["params"]) for g in groups}
    return opt, counts, frozen_names


def cosine_lr(step: int, steps: int, warm_frac: float = 0.05) -> float:
    warm = max(1, int(steps * warm_frac))
    if step < warm:
        return (step + 1) / warm
    q = (step - warm) / max(1, steps - warm)
    return 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * q))


def source_balance_loss(weights: torch.Tensor) -> torch.Tensor:
    mean = weights.float().mean(0)
    return mean.numel() * mean.pow(2).sum() - 1.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 3: heterogeneous latent consolidation for Supermix Expanse")
    ap.add_argument("--inp", default=str(ec.PATHS["final"]), help="trained v1 Expanse checkpoint")
    ap.add_argument("--out", default=str(ec.PATHS["checkpoints"] / "supermix_expanse_v2.pt"))
    ap.add_argument("--layers", type=parse_layers, default=(1, 2, 4), help="comma-separated transformer layers")
    ap.add_argument("--shared_dim", type=int, default=512)
    ap.add_argument("--latent_experts", type=int, default=8)
    ap.add_argument("--latent_rank", type=int, default=128)
    ap.add_argument("--latent_top_k", type=int, default=2)
    ap.add_argument("--memory_slots", type=int, default=24)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--fly_rows", type=int, default=2500)
    ap.add_argument("--max_rows_per_source", type=int, default=5000)
    ap.add_argument("--dev_cap", type=int, default=160)
    ap.add_argument("--lr_trunk", type=float, default=1e-5)
    ap.add_argument("--lr_graft", type=float, default=5e-5)
    ap.add_argument("--lr_v2", type=float, default=2e-4)
    ap.add_argument("--lr_align", type=float, default=3e-4)
    ap.add_argument("--distill_weight", type=float, default=0.35)
    ap.add_argument("--align_weight", type=float, default=0.10)
    ap.add_argument("--native_balance_weight", type=float, default=0.01)
    ap.add_argument("--source_balance_weight", type=float, default=0.003)
    ap.add_argument("--bake_start", type=float, default=0.72)
    ap.add_argument("--teacher_free_start", type=float, default=0.90)
    ap.add_argument("--skip_slow_sources", action="store_true", help="omit one-token Fly/CNS source probes")
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--resume", default="auto", help="auto | none | partial checkpoint path")
    ap.add_argument("--max_minutes", type=float, default=0.0)
    ap.add_argument("--stop_after", type=int, default=0)
    ap.add_argument("--seed", type=int, default=260923)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if not (0.0 < args.bake_start < args.teacher_free_start < 1.0):
        raise ValueError("need 0 < --bake_start < --teacher_free_start < 1")
    if args.smoke:
        args.steps = min(args.steps, 3)
        args.fly_rows = min(args.fly_rows, 64)
        args.max_rows_per_source = min(args.max_rows_per_source, 64)
        args.dev_cap = min(args.dev_cap, 8)
        args.eval_every = max(args.eval_every, args.steps + 1)
        if args.out == str(ec.PATHS["checkpoints"] / "supermix_expanse_v2.pt"):
            args.out = str(ec.PATHS["checkpoints"] / "smoke" / "supermix_expanse_v2_smoke.pt")

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed % (2 ** 32))
    out_path = Path(args.out)
    partial_path = out_path.with_name(out_path.stem + ".partial.pt")
    t0 = time.time()

    # v1 model -----------------------------------------------------------------
    model, tok, payload = ec.load_expanse(args.inp)
    receipt_v1 = dict(payload.get("expanse") or {})
    payload["state_dict"] = None
    model.train()
    log(f"loaded v1 {args.inp}: {sum(p.numel() for p in model.parameters()):,} params, vocab {tok.vocab_size}")

    # Corpus is exactly the same evidence families as stage 2.  Reusing it is
    # important: v2 should consolidate, not silently replace the training task.
    fly_teacher = ec.FlyCore(int(model.config.hidden_size), config=model.fly_config)
    fly_teacher.load_state_dict(model.fly_core.state_dict())
    fly_teacher.requires_grad_(False).eval()
    corpus = te.load_corpus(fly_teacher, fly_rows=args.fly_rows, seed=args.seed, dev_frac=0.05,
                            dev_cap=args.dev_cap, max_rows_per_source=args.max_rows_per_source)
    if args.smoke:
        train_rows, dev_rows = te.smoke_subset(corpus, 20, 2, args.seed)
    else:
        train_rows = [r for s in te.SOURCES for r in corpus[s]["train"]]
        dev_rows = [r for s in te.SOURCES for r in corpus[s]["dev"]]
    x_tr, y_tr, pl_tr, m_tr, drop_tr = te.encode_rows(train_rows, tok, args.seq)
    x_dv, y_dv, pl_dv, m_dv, drop_dv = te.encode_rows(dev_rows, tok, args.seq)
    if not len(m_tr):
        raise RuntimeError("no train rows survived tokenisation/sequence packing")
    src_tr = [r.get("source", "replay") for r in m_tr]
    src_dv = [r.get("source", "replay") for r in m_dv]
    log(f"packed train {tuple(x_tr.shape)} dev {tuple(x_dv.shape)}; dropped {drop_tr}+{drop_dv}")

    feats_tr = ec.arch.OmniCore.featurize([r["user"] for r in m_tr])
    feats_dv = ec.arch.OmniCore.featurize([r["user"] for r in m_dv])
    log("encoding Omni v7 prompt states")
    o7_tr = encode_omni7(model, m_tr)
    o7_dv = encode_omni7(model, m_dv)

    source_dims = dynamic_source_dims(model, feats_tr, o7_tr)
    cfg = cv2.V2Config(
        hidden_size=int(model.config.hidden_size), shared_dim=args.shared_dim, layers=tuple(args.layers),
        latent_experts=args.latent_experts, latent_rank=args.latent_rank, latent_top_k=args.latent_top_k,
        memory_slots=args.memory_slots, source_dims=source_dims, domains=tuple(te.SOURCES),
    ).validate()
    stack = cv2.attach_consolidation_v2(model, cfg)
    bank = cv2.TeacherFusionBank(cfg)
    extractor = cv2.InternalSourceExtractor(model)
    opt, param_counts, frozen_names = make_optimizer(
        model, bank, lr_trunk=args.lr_trunk, lr_graft=args.lr_graft, lr_v2=args.lr_v2, lr_align=args.lr_align
    )
    base_lrs = [float(g["lr"]) for g in opt.param_groups]
    log(f"v2 config {cfg.to_dict()}")
    log(f"trainable parameter groups {param_counts}; frozen tensors {len(frozen_names)}")

    @torch.no_grad()
    def evaluate(tag: str) -> Dict[str, Any]:
        model.eval()
        totals: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        total = n_tok = 0.0
        for i in range(0, x_dv.shape[0], args.batch):
            xb, yb = x_dv[i:i + args.batch], y_dv[i:i + args.batch]
            kw: Dict[str, Any] = {"omni_features": feats_dv[i:i + args.batch]}
            if o7_dv is not None:
                kw["omni7_state"] = o7_dv[i:i + args.batch]
            out = model(xb, labels=yb, return_mtp=False, **kw)
            ce = F.cross_entropy(out.logits[:, :-1].reshape(-1, out.logits.shape[-1]), yb[:, 1:].reshape(-1),
                                 reduction="none").view(xb.shape[0], -1)
            mask = yb[:, 1:] != -100
            for j in range(xb.shape[0]):
                val = float(ce[j][mask[j]].sum())
                n = int(mask[j].sum())
                s = src_dv[i + j]
                totals[s] = totals.get(s, 0.0) + val
                counts[s] = counts.get(s, 0) + n
                total += val
                n_tok += n
        model.train()
        rep: Dict[str, Any] = {"dev_loss": total / max(1.0, n_tok)}
        for s in te.SOURCES:
            if counts.get(s):
                rep[f"dev_{s}"] = totals[s] / counts[s]
        rep.update(model.gate_report())
        rep.update(stack.report())
        log(f"[eval {tag}] " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rep.items()))
        return rep

    start_eval = evaluate("start")

    # resume -------------------------------------------------------------------
    step = ptr = 0
    order_gen = torch.Generator().manual_seed(args.seed + 17)
    order = torch.randperm(x_tr.shape[0], generator=order_gen)
    history: List[Dict[str, Any]] = [{"step": 0, **start_eval}]
    rp = None if args.resume == "none" else (partial_path if args.resume == "auto" else Path(args.resume))
    if rp is not None and rp.exists():
        part = torch.load(rp, map_location="cpu", weights_only=False)
        if part.get("schema") != PARTIAL_SCHEMA:
            raise ValueError(f"{rp} is not a {PARTIAL_SCHEMA} checkpoint")
        if part.get("v2_config") != cfg.to_dict():
            raise ValueError("partial was created with a different v2 architecture")
        model.load_state_dict(part["model"], strict=True)
        bank.load_state_dict(part["bank"], strict=True)
        opt.load_state_dict(part["optimizer"])
        step, ptr, order = int(part["step"]), int(part["ptr"]), part["order"]
        order_gen.set_state(part["order_gen"])
        history = list(part.get("history", history))
        torch.set_rng_state(part["rng_torch"])
        random.setstate(part["rng_python"])
        np.random.set_state(part["rng_numpy"])
        log(f"resumed {rp} at step {step}/{args.steps}")
        del part

    baked = False
    last_router: Dict[str, float] = {}
    run_start = time.time()
    steps_this_run = 0

    def save_partial(reason: str) -> None:
        atomic_save({
            "schema": PARTIAL_SCHEMA,
            "step": step,
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "bank": {k: v.detach().cpu() for k, v in bank.state_dict().items()},
            "optimizer": opt.state_dict(),
            "ptr": ptr,
            "order": order,
            "order_gen": order_gen.get_state(),
            "history": history,
            "v2_config": cfg.to_dict(),
            "args": vars(args),
            "reason": reason,
            "rng_torch": torch.get_rng_state(),
            "rng_python": random.getstate(),
            "rng_numpy": np.random.get_state(),
        }, partial_path)
        log(f"partial ({reason}) step {step} -> {partial_path}")

    while step < args.steps:
        if args.max_minutes and (time.time() - t0) / 60.0 >= args.max_minutes:
            save_partial("wall_budget")
            return 0
        if args.stop_after and steps_this_run >= args.stop_after:
            save_partial("stop_after")
            return 0
        if ptr + args.batch > order.numel():
            order = torch.randperm(x_tr.shape[0], generator=order_gen)
            ptr = 0
        bi = order[ptr:ptr + args.batch]
        ptr += args.batch
        xb, yb, plb = x_tr[bi], y_tr[bi], pl_tr[bi]
        feats = feats_tr[bi]
        o7 = o7_tr[bi] if o7_tr is not None else None
        domains = [src_tr[i] for i in bi.tolist()]
        progress = step / max(1, args.steps - 1)
        pw = phase_weights(progress, args.bake_start, args.teacher_free_start)
        if progress >= args.bake_start and not baked:
            set_bank_trainable(bank, False)
            baked = True
            log(f"teacher fusion frozen at step {step} ({progress:.1%}); targets are now fixed")

        scale = cosine_lr(step, args.steps)
        for g, base in zip(opt.param_groups, base_lrs):
            g["lr"] = base * scale

        kw: Dict[str, Any] = {"omni_features": feats}
        if o7 is not None:
            kw["omni7_state"] = o7
        out = model(xb, labels=yb, **kw)
        lm_loss = out.loss
        distill_terms: List[torch.Tensor] = []
        align_terms: List[torch.Tensor] = []
        source_balance_terms: List[torch.Tensor] = []
        router_summaries: Dict[str, List[float]] = {}

        if pw["teacher"] > 0 or pw["distill"] > 0:
            for li in cfg.layers:
                rec = stack.last.get(li)
                if rec is None:
                    continue
                with torch.no_grad():
                    sources = extractor.collect(
                        li, rec["hidden"].detach(), plb, omni_features=feats, omni7_state=o7,
                        include_slow=not args.skip_slow_sources,
                    )
                # The query is detached so the student cannot move the teacher
                # mixture simply by changing its own representation.
                fused = bank(sources, rec["latent"].detach(), lengths=plb, domains=domains)
                if pw["align"] > 0 and not baked:
                    align_terms.append(bank.alignment_loss(fused["projected"]))
                if pw["distill"] > 0:
                    sl = bank.student_loss(rec["latent"], fused["fused"], lengths=plb)
                    distill_terms.append(sl["total"])
                source_balance_terms.append(source_balance_loss(fused["weights"]))
                mw = fused["weights"].detach().mean(0)
                for n, w in zip(fused["names"], mw.tolist()):
                    router_summaries.setdefault(n, []).append(float(w))

        zero = lm_loss.new_zeros(())
        distill = torch.stack(distill_terms).mean() if distill_terms else zero
        align = torch.stack(align_terms).mean() if align_terms else zero
        native_balance = stack.aux_loss()
        src_balance = torch.stack(source_balance_terms).mean() if source_balance_terms else zero
        loss = (
            lm_loss
            + args.distill_weight * pw["distill"] * distill
            + args.align_weight * pw["align"] * align
            + args.native_balance_weight * native_balance
            + args.source_balance_weight * src_balance
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        trainable = [p for g in opt.param_groups for p in g["params"] if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        step += 1
        steps_this_run += 1
        last_router = {n: float(sum(v) / len(v)) for n, v in router_summaries.items() if v}

        if step == 1 or step % 10 == 0 or args.smoke:
            gate = sum(float(stack.blocks[str(i)].out_gate.detach().abs().mean()) for i in cfg.layers) / len(cfg.layers)
            log(
                f"[step {step}/{args.steps}] total={float(loss.detach()):.4f} lm={float(lm_loss.detach()):.4f} "
                f"distill={float(distill.detach()):.4f} align={float(align.detach()):.4f} "
                f"native_bal={float(native_balance.detach()):.4f} gate={gate:.5f} "
                f"phase(d={pw['distill']:.2f},a={pw['align']:.2f},t={pw['teacher']:.0f}) "
                f"router={json.dumps(last_router, sort_keys=True)}"
            )
        if args.eval_every and step % args.eval_every == 0 and step < args.steps:
            history.append({"step": step, **evaluate(str(step))})
        if args.save_every and step % args.save_every == 0 and step < args.steps:
            save_partial("rolling")

    # Final teacher-free checkpoint -------------------------------------------
    set_bank_trainable(bank, False)
    final_eval = evaluate("final")
    history.append({"step": step, **final_eval})
    receipt = dict(receipt_v1)
    receipt["stage"] = "consolidated-v2"
    receipt["consolidation_v2"] = {
        "architecture": cfg.to_dict(),
        "training": {
            "steps": step,
            "batch": args.batch,
            "seq": args.seq,
            "seed": args.seed,
            "parameter_groups": param_counts,
            "loss_weights": {
                "distill": args.distill_weight,
                "alignment": args.align_weight,
                "native_router_balance": args.native_balance_weight,
                "source_router_balance": args.source_balance_weight,
            },
            "phase_boundaries": {"bake_start": args.bake_start, "teacher_free_start": args.teacher_free_start},
            "source_dims": source_dims,
            "slow_sources": not args.skip_slow_sources,
            "start_eval": start_eval,
            "final_eval": final_eval,
            "last_source_router_mean": last_router,
            "history": history,
            "wall_minutes": round((time.time() - t0) / 60.0, 2),
        },
        "runtime": "teacher fusion/projectors removed; native latent stack only",
    }
    extra = {
        "note": "Expanse v2 heterogeneous multi-source latent consolidation",
        "base_checkpoint": str(args.inp),
        "teacher_free_runtime": True,
    }
    cv2.save_v2(out_path, model, tok, extra, receipt)
    out_path.with_suffix(".receipt.json").write_text(json.dumps(ec.jsonable(receipt), indent=1), encoding="utf-8")
    if partial_path.exists():
        partial_path.unlink()
    log(f"saved self-contained v2 -> {out_path} ({(time.time() - t0) / 60.0:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
