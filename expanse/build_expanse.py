"""Stage 1: build the grafted Supermix Expanse checkpoint (no training).

    python expanse/build_expanse.py                       # full build
    python expanse/build_expanse.py --smoke               # quick structural build, no teachers

Steps (DESIGN.md "build_expanse.py")
    1. load Archimedes final; record its logits on 32 replay rows (fly on and
       off) as the function-preservation baseline;
    2. grow 16 MoE slots (8 code + 8 bio donors per layer). The slot count is
       one config field shared by every MoE layer, so layers 1-4 all go
       72 -> 88; donors fill layers 1 and 2, the new slots of 3 and 4 stay dead;
    3. donor experts: Qwen2.5-Coder-7B L0 -> student L1, L1 -> L2 (slots
       72-79), BioMedLM L0 -> L1, L1 -> L2 (slots 80-87), installed dormant;
    4. attach FullConnectomeCore (full male-CNS type graph, 4 steps) and
       OmniV7Branch (v7 weights loaded, frozen); FlyCore made causal;
    5. extend the vocabulary over the teacher corpora (<= 1,500 tokens seen
       >= 3 times), new embedding rows lifted from the teachers' tables
       (fallback: mean + 0.1 std noise for words no teacher covers);
    6. function preservation: every new gate 0, donors dormant, fly per
       position. The only expected difference from Archimedes is the fly
       causality fix; with the fly off on both sides the logits must agree
       to < 1e-4 (they agree bitwise, see tests/test_expanse_core.py);
    7. save ``checkpoints/supermix_expanse_grafted.pt`` + receipt json.

Qwen3-Coder has no 7B/8B release; Qwen2.5-Coder-7B-Instruct is the official
Qwen coder at that size and is used instead (recorded in the receipt).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import expanse_core as ec  # noqa: E402
from expanse_core import PATHS, text_utils  # noqa: E402

QWEN_NOTE = ("Qwen3-Coder has no 7B/8B release; Qwen2.5-Coder-7B-Instruct, the official Qwen coder at that size, "
             "is the code teacher (substitution disclosed).")


def log(*a) -> None:
    print("[build]", *a, flush=True)


def ensure_npz(path: Path) -> Path:
    """The connectome table lives in data/; copy it (read-only source) if absent."""
    if not path.exists():
        src = PATHS["npz_source"]
        if not src.exists():
            raise FileNotFoundError(f"{path} missing and no source at {src}")
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, path)
        log(f"copied {src} -> {path}")
    return path


def baseline_rows(n: int, seed: int) -> List[Dict[str, Any]]:
    rows = ec.load_replay()
    random.Random(seed).shuffle(rows)
    return rows[:n]


def encode_batch(tok, rows):
    enc = [tok.encode_turn(r["user"], r["assistant"])[0] for r in rows]
    L = max(len(e) for e in enc)
    x = torch.tensor([e + [text_utils.PAD] * (L - len(e)) for e in enc])
    return x, [len(e) for e in enc]


@torch.no_grad()
def batched_logits(model, x, lens, feats, batch: int = 8, **kw) -> List[torch.Tensor]:
    """Per-row logits trimmed to the row length (float32, cpu)."""
    out = []
    for i in range(0, x.shape[0], batch):
        lg = model(x[i:i + batch], omni_features=feats[i:i + batch], return_mtp=False,
                   **{k: (v[i:i + batch] if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == x.shape[0] else v)
                      for k, v in kw.items()}).logits
        for j in range(lg.shape[0]):
            out.append(lg[j, : lens[i + j]].clone())
    return out


def max_diff(a: List[torch.Tensor], b: List[torch.Tensor], vocab: int) -> float:
    return max(float((x[:, :vocab] - y[:, :vocab]).abs().max()) for x, y in zip(a, b))


def domain_ids(tok, rows: List[Dict[str, Any]]) -> Optional[Set[int]]:
    """Student ids appearing in a teacher corpus (the donor's domain tokens)."""
    if not rows:
        return None
    ids: Set[int] = set()
    for r in rows:
        ids.update(tok.encode(r["user"] + " " + r["assistant"]))
    ids -= set(range(len(text_utils.SPECIAL_TOKENS)))
    return ids


def build_donors(model, tok, teacher, *, name: str, slots: List[int], n_experts: int, dom: Optional[Set[int]],
                 fit_steps: int, seed: int, max_tokens: Optional[int]) -> Dict[str, Any]:
    import donor_graft as dg

    receipts, summaries = [], {}
    for src, dst in ((0, 1), (1, 2)):
        log(f"{name}: teacher L{src} -> student L{dst}, {n_experts} experts into slots {slots[0]}..{slots[-1]}")
        experts = dg.build_donor_experts(model, tok, teacher, src_layer=src, dst_layer=dst, n_experts=n_experts,
                                         domain_ids=dom, fit_steps=fit_steps, seed=seed, max_tokens=max_tokens)
        receipts.append(dg.install_experts(model, dst, experts, slots=slots[:n_experts]))
        summaries[str(dst)] = {"src_layer": src, "experts": dg.experts_summary(experts),
                               "fit": ec.jsonable(experts[0].get("fit", {})) if experts else {}}
        del experts
    merged = dg.merge_install_receipts(*receipts)
    return {"install": ec.jsonable(merged), "summary": summaries}


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 1: grafted Supermix Expanse")
    ap.add_argument("--arch", default=str(PATHS["arch_final"]))
    ap.add_argument("--out", default=None, help="default checkpoints/supermix_expanse_grafted.pt (smoke: checkpoints/smoke/...)")
    ap.add_argument("--n_new", type=int, default=16, help="new MoE slots per MoE layer")
    ap.add_argument("--code_experts", type=int, default=8)
    ap.add_argument("--bio_experts", type=int, default=8)
    ap.add_argument("--fit_steps", type=int, default=400)
    ap.add_argument("--donor_max_tokens", type=int, default=None, help="subsample matched tokens (speed)")
    ap.add_argument("--npz", default=str(PATHS["npz"]))
    ap.add_argument("--cns_min_input_fraction", type=float, default=0.0)
    ap.add_argument("--cns_radius", type=float, default=0.9)
    ap.add_argument("--cns_steps", type=int, default=4)
    ap.add_argument("--vocab_max_new", type=int, default=1500)
    ap.add_argument("--vocab_min_count", type=int, default=3)
    ap.add_argument("--rows", type=int, default=32, help="function-preservation rows")
    ap.add_argument("--skip_donors", action="store_true")
    ap.add_argument("--skip_vocab", action="store_true")
    ap.add_argument("--skip_omni7", action="store_true")
    ap.add_argument("--skip_cns", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="8 rows, no teachers (no donors, fallback vocab rows)")
    ap.add_argument("--allow_mismatch", action="store_true", help="save even if fly-off preservation fails")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.smoke:
        args.rows = min(args.rows, 8)
        args.skip_donors = True
    out = Path(args.out) if args.out else (PATHS["checkpoints"] / "smoke" / "supermix_expanse_grafted_smoke.pt"
                                           if args.smoke else PATHS["grafted"])
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    t0 = time.time()
    receipt: Dict[str, Any] = {"stage": "grafted", "schema": ec.EXP_SCHEMA, "smoke": bool(args.smoke),
                               "args": vars(args), "teacher_substitution": QWEN_NOTE, "sources": {}, "grafts": {}}

    # 1. Archimedes + baseline -------------------------------------------------
    log(f"loading Archimedes final {args.arch}")
    A, tok, pay = ec.load_archimedes(args.arch)
    A.eval()
    receipt["sources"]["archimedes"] = {"path": str(args.arch), "sha256": ec.sha256_file(args.arch),
                                        "schema": pay["schema"], "params": sum(p.numel() for p in A.parameters())}
    rows = baseline_rows(args.rows, args.seed)
    x, lens = encode_batch(tok, rows)
    feats = ec.arch.OmniCore.featurize([r["user"] for r in rows])
    base_fly = batched_logits(A, x, lens, feats)
    base_nofly = batched_logits(A, x, lens, feats, use_fly=False)
    base_vocab = tok.vocab_size
    log(f"baseline logits on {len(rows)} rows (vocab {base_vocab})")
    del A

    # 4a. connectome graph (built before the model so the core is constructed once)
    graph = None
    if not args.skip_cns:
        import connectome_full as cf
        npz = ensure_npz(Path(args.npz))
        log(f"building the full male-CNS graph from {npz}")
        graph = cf.build_full_graph(npz, min_input_fraction=args.cns_min_input_fraction, radius=args.cns_radius)
        r = graph["receipt"]
        log(f"   {r['n_types']} types, {r['n_edges']} edges, |W| spectral radius {r['abs_spectral_radius']:.4f}, "
            f"n_in {r['n_in']} n_out {r['n_out']}")
        receipt["sources"]["connectome_npz"] = {"path": str(npz), "sha256": r.get("npz_sha256")}
    omni7_meta = None
    if not args.skip_omni7:
        import omni_v7_branch as ob
        omni7_meta = ob.load_v7_meta()

    # 2. grow + build the Expanse model ------------------------------------------
    model, grow = ec.expanse_from_archimedes(pay, n_new=args.n_new, graph=graph, cns_steps=args.cns_steps,
                                             omni7_meta=omni7_meta, fly_causal=True)
    receipt["grafts"]["moe_growth"] = {**grow, "note": "moe_spare_experts is shared by every MoE layer: all grow; "
                                                       "donors fill layers 1-2, layers 3-4 keep dead spares"}
    if graph is not None:
        receipt["grafts"]["cns_full"] = {"meta": model.cns_meta, "params": sum(p.numel() for p in model.cns_full.parameters())}
    del graph
    pay["state_dict"] = None  # the model holds the weights now
    slots_before = grow["slots_before"]
    log(f"Expanse built: MoE {grow['slots_before']} -> {grow['slots_after']} slots, new graft tensors {grow['new_graft_tensors']}")

    # 4b. omni v7 weights ------------------------------------------------------------
    if model.omni7 is not None:
        log("loading Omni Collective v7 Frontier into the frozen branch")
        o7 = model.omni7.load_v7()
        receipt["grafts"]["omni7"] = {k: v for k, v in o7.items() if k not in ("intent_labels", "domain_labels")}
        receipt["grafts"]["omni7"]["labels"] = {"intent": o7.get("intent_labels"), "domain": o7.get("domain_labels")}
        receipt["sources"]["omni_v7"] = {"path": o7.get("path"), "sha256": o7.get("sha256")}

    # 3. donor experts ------------------------------------------------------------------
    code_rows = ec.read_jsonl(PATHS["code_rows"])
    bio_rows = ec.read_jsonl(PATHS["bio_rows"])
    conn_rows = ec.read_jsonl(PATHS["connectome_rows"])
    teachers = []
    if not args.skip_donors or not args.skip_vocab:
        import donor_graft as dg
        qdir, bdir = PATHS["qwen_dir"], PATHS["biomedlm_dir"]
        if not args.smoke and (qdir / "graft_slices.safetensors").exists():
            teachers.append(("qwen", dg.load_qwen_slices(qdir / "graft_slices.safetensors", qdir / "tokenizer.json",
                                                         qdir / "config.json")))
            receipt["sources"]["qwen_slices"] = {"path": str(qdir / "graft_slices.safetensors"),
                                                 "sha256": ec.sha256_file(qdir / "graft_slices.safetensors")}
        if not args.smoke and (bdir / "config.json").exists():
            teachers.append(("biomedlm", dg.load_biomedlm_slices(bdir, layers=(0, 1))))
            receipt["sources"]["biomedlm"] = {"path": str(bdir)}
    if not args.skip_donors:
        tmap = dict(teachers)
        donors: Dict[str, Any] = {}
        plan = [("qwen", list(range(slots_before, slots_before + args.code_experts)), args.code_experts, code_rows),
                ("biomedlm", list(range(slots_before + args.code_experts, slots_before + args.code_experts + args.bio_experts)),
                 args.bio_experts, bio_rows)]
        for name, slots, n_exp, drows in plan:
            if name not in tmap or n_exp <= 0:
                log(f"donor {name}: skipped (teacher missing or 0 experts)")
                continue
            if slots[-1] >= grow["slots_after"]:
                raise ValueError(f"donor slots {slots} exceed {grow['slots_after']} slots; raise --n_new")
            dom = domain_ids(tok, drows)
            res = build_donors(model, tok, tmap[name], name=name, slots=slots, n_experts=n_exp, dom=dom,
                               fit_steps=args.fit_steps, seed=args.seed, max_tokens=args.donor_max_tokens)
            res["domain_tokens"] = len(dom) if dom else None
            donors[name] = res
        model.donor_receipts = {k: v["install"] for k, v in donors.items()}
        receipt["grafts"]["donor_experts"] = {k: {kk: vv for kk, vv in v.items() if kk != "install"} for k, v in donors.items()}

    # 5. vocabulary ------------------------------------------------------------------------
    new_tok = tok
    if not args.skip_vocab:
        texts = [r["user"] + " " + r["assistant"] for r in code_rows + bio_rows + conn_rows]
        new_tok = text_utils.WordTokenizer.extend(tok, texts, max_new=args.vocab_max_new, min_count=args.vocab_min_count)
        new_strings = new_tok.tokens[tok.vocab_size:]
        emb = model.embed_tokens.weight.detach()
        lift: Dict[str, Any] = {"teachers": {}}
        rows_new = torch.zeros(len(new_strings), emb.shape[1])
        uncovered = list(range(len(new_strings)))
        if new_strings and teachers:
            import donor_graft as dg
            rows_new, lift = dg.lifted_embedding_rows(new_strings, emb, tok, [t for _, t in teachers])
            uncovered = list(lift.get("uncovered", []))
        if uncovered:
            g = torch.Generator().manual_seed(args.seed)
            mean, std = emb.mean(0), emb.std(0)
            rows_new[uncovered] = mean + 0.1 * std * torch.randn(len(uncovered), emb.shape[1], generator=g)
        added = ec.append_embedding_rows(model, rows_new) if new_strings else 0
        receipt["vocab"] = {"base": tok.vocab_size, "added": added, "size": new_tok.vocab_size,
                            "corpus_rows": {"code": len(code_rows), "bio": len(bio_rows), "connectome": len(conn_rows)},
                            "max_new": args.vocab_max_new, "min_count": args.vocab_min_count,
                            "lifted": added - len(uncovered), "fallback_mean_noise": len(uncovered),
                            "lift": ec.jsonable({k: v for k, v in lift.items() if k != "uncovered"}),
                            "sample_new_tokens": new_strings[:40]}
        log(f"vocabulary {tok.vocab_size} -> {new_tok.vocab_size} ({added} added, {len(uncovered)} fallback rows)")
    model.vocab_base = int(base_vocab)
    del teachers

    # 6. function preservation ------------------------------------------------------------
    model.eval()
    state = model.omni7_state_for([r["user"] for r in rows])
    kw = {"omni7_state": state} if state is not None else {}
    got_fly = batched_logits(model, x, lens, feats, **kw)
    got_nofly = batched_logits(model, x, lens, feats, use_fly=False, **kw)
    model.fly_causal = False
    got_legacy = batched_logits(model, x, lens, feats, **kw)
    model.fly_causal = True
    fp = {"rows": len(rows), "compared_vocab": base_vocab,
          "max_abs_logit_diff_fly_causal": max_diff(got_fly, base_fly, base_vocab),
          "max_abs_logit_diff_fly_off": max_diff(got_nofly, base_nofly, base_vocab),
          "max_abs_logit_diff_legacy_fly": max_diff(got_legacy, base_fly, base_vocab),
          "note": "fly_causal differs by design (per-position sense); fly_off and legacy_fly must be ~0"}
    fp["passed"] = fp["max_abs_logit_diff_fly_off"] < 1e-4
    receipt["function_preservation"] = fp
    log(f"function preservation: fly causal {fp['max_abs_logit_diff_fly_causal']:.3e}  fly off "
        f"{fp['max_abs_logit_diff_fly_off']:.3e}  legacy fly {fp['max_abs_logit_diff_legacy_fly']:.3e}")
    if not fp["passed"] and not args.allow_mismatch:
        raise RuntimeError(f"function preservation failed with the fly off: {fp}")

    # 7. save ----------------------------------------------------------------------------------
    total = sum(p.numel() for p in model.parameters())
    frozen = sum(p.numel() for n, p in model.named_parameters() if n.startswith("omni7.net."))
    receipt["params"] = {
        "total": total, "frozen_omni7_net": frozen,
        "fly_core": sum(p.numel() for p in model.fly_core.parameters()),
        "omni_core": sum(p.numel() for p in model.omni_core.parameters()) if model.omni_core is not None else 0,
        "omni7_bridge": sum(p.numel() for n, p in model.named_parameters() if n.startswith("omni7.") and not n.startswith("omni7.net.")),
        "cns_full": sum(p.numel() for p in model.cns_full.parameters()) if model.cns_full is not None else 0,
    }
    receipt["gate_report"] = model.gate_report()
    receipt["built_seconds"] = round(time.time() - t0, 1)
    extra = dict(pay.get("extra") or {})
    extra.update({"run_name": "supermix_expanse", "note": "stage 1: grafted, untrained" + (" (smoke)" if args.smoke else ""),
                  "warm_start": "supermix-archimedes final"})
    ec.save_expanse(out, model, new_tok, extra, receipt)
    rpath = out.with_suffix(".receipt.json")
    rpath.write_text(json.dumps(ec.jsonable({**receipt, **{k: v for k, v in model.expanse_spec().items() if k != "omni7_meta"}}),
                                indent=1), encoding="utf-8")
    log(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB), params {total:,} ({frozen:,} frozen v7) in {time.time() - t0:.1f}s")
    log(f"receipt {rpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
