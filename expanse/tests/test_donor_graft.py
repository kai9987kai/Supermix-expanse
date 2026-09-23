"""Tests for src/donor_graft.py.

Fast tests (default) use tiny *random* teachers built with ``transformers`` and
saved in the real on-disk formats (a bf16 Qwen2 ``graft_slices.safetensors``;
a bf16, sharded GPT-2 HF dir with an index), so the loaders, the lazy bf16->fp32
reads and the exact single-token forwards are checked against the reference
implementations:

* single-token states (resid_pre / resid_in / mlp_in / mlp_out) == HF modules;
* ``teacher_mlp_acts`` reproduces the MLP output through the down projection;
* ``LazyTensor`` row/column gathers == dense indexing; ``ridge`` recovers a map;
* ``match_tokens`` pairs decode back to the student strings;
* ``build_donor_experts`` on the real Archimedes student + tiny GPT-2 teacher
  (src L1 -> dst L1): shapes, rms targets, refinement improves held-out R2;
* ``install_experts`` into in-memory-grown dead slots is bit-for-bit
  function-preserving, and ``wake_grafted_experts`` then routes tokens to them;
* ``lifted_embedding_rows`` scales rows and reports coverage.

Slow demos on the real teachers (``--real`` or ``EXPANSE_REAL=1``):
``test_real_qwen_demo`` (Qwen L0 -> student L1, 2 experts, 100 fit steps) and
``test_real_biomedlm_smoke``; both write receipts to ``expanse/data/receipts/``.
Nothing is ever written into a checkpoint on disk.

Run: ``python -m pytest -q -s tests/test_donor_graft.py`` (``python tests/test_donor_graft.py --real``).
"""

from __future__ import annotations

import builtins
import copy
import json
import keyword
import os
import shutil
import sys
import time
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
EXPANSE = HERE.parent
EXP = EXPANSE.parent
sys.path.insert(0, str(EXPANSE / "src"))
sys.path.insert(0, str(EXP / "supermix-archimedes" / "archimedes" / "src"))

import donor_graft as dg  # noqa: E402

STUDENT_PT = EXP / "external" / "base" / "supermix_archimedes.pt"
QWEN_DIR = EXP / "external" / "teachers" / "qwen2.5-coder-7b-instruct"
BIO_DIR = EXP / "external" / "teachers" / "biomedlm"
RECEIPTS = EXPANSE / "data" / "receipts"
REAL = os.environ.get("EXPANSE_REAL", "") == "1"

torch.set_num_threads(int(os.environ.get("EXPANSE_THREADS", "2")))


# ---------------------------------------------------------------------------
# fixtures: tiny teachers in the real file formats, the real student
# ---------------------------------------------------------------------------
def _randomise(module: torch.nn.Module, seed: int) -> None:
    """Non-trivial norms and biases so every bias / scale path is exercised."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in module.named_parameters():
            if name.endswith("bias") or "norm" in name or "ln_" in name:
                p.copy_(torch.randn(p.shape, generator=g) * 0.3 + (1.0 if name.endswith("weight") else 0.0))
            else:
                p.copy_(torch.randn(p.shape, generator=g) * 0.08)


@pytest.fixture(scope="module")
def tiny_gpt2(tmp_path_factory):
    transformers = pytest.importorskip("transformers")
    if not (BIO_DIR / "tokenizer.json").exists():
        pytest.skip("BioMedLM tokenizer not present")
    out = tmp_path_factory.mktemp("tiny_gpt2")
    cfg = transformers.GPT2Config(n_embd=64, n_layer=2, n_head=4, vocab_size=28896, n_positions=32,
                                  activation_function="gelu_new", layer_norm_epsilon=1e-5,
                                  scale_attn_by_inverse_layer_idx=True)
    model = transformers.GPT2LMHeadModel(cfg).eval()
    _randomise(model, 1)
    model.to(torch.bfloat16).save_pretrained(out, safe_serialization=True, max_shard_size="1MB")
    shutil.copy(BIO_DIR / "tokenizer.json", out / "tokenizer.json")
    ref = transformers.GPT2Model.from_pretrained(out, dtype=torch.float32).eval()  # bf16-rounded weights in fp32
    teacher = dg.load_biomedlm_slices(out, layers=(0, 1))
    return teacher, ref, out


@pytest.fixture(scope="module")
def tiny_qwen(tmp_path_factory):
    transformers = pytest.importorskip("transformers")
    from safetensors.torch import save_file
    if not (QWEN_DIR / "tokenizer.json").exists():
        pytest.skip("Qwen tokenizer not present")
    out = tmp_path_factory.mktemp("tiny_qwen")
    cfg = transformers.Qwen2Config(hidden_size=64, intermediate_size=160, num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, vocab_size=151936, rms_norm_eps=1e-6,
                                   max_position_embeddings=64, tie_word_embeddings=False)
    model = transformers.Qwen2Model(cfg).eval()
    _randomise(model, 2)
    sd = {k: v.detach().to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()}
    keep = {}
    for k, v in sd.items():
        if k == "embed_tokens.weight" or (k.startswith("layers.") and any(
                s in k for s in ("input_layernorm", "post_attention_layernorm", "v_proj", "o_proj", "mlp."))):
            keep["model." + k] = v
    save_file(keep, str(out / "graft_slices.safetensors"))
    (out / "config.json").write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    model.load_state_dict({k: v.float() for k, v in sd.items()})  # reference sees the bf16-rounded weights
    teacher = dg.load_qwen_slices(out / "graft_slices.safetensors", QWEN_DIR / "tokenizer.json", out / "config.json")
    return teacher, model, out


@pytest.fixture(scope="module")
def student():
    if not STUDENT_PT.exists():
        pytest.skip("Archimedes checkpoint not present")
    from archimedes_core import load_archimedes
    model, tok, _payload = load_archimedes(str(STUDENT_PT))
    del _payload
    model.eval()
    return model, tok


def _hook_states(blocks, attr_norm2, attr_mlp, ids_input, run):
    """Capture (resid_pre, resid_in, mlp_in, mlp_out) per block of an HF model."""
    caps = {}
    handles = []
    for L, blk in enumerate(blocks):
        def pre_block(m, args, kwargs=None, L=L):
            x = args[0] if args else kwargs["hidden_states"]
            caps[("pre", L)] = x.detach()
        handles.append(blk.register_forward_pre_hook(lambda m, a, k, L=L: pre_block(m, a, k, L), with_kwargs=True))
        n2 = getattr(blk, attr_norm2)
        handles.append(n2.register_forward_hook(lambda m, i, o, L=L: caps.update({("rin", L): i[0].detach(), ("min", L): o.detach()})))
        handles.append(getattr(blk, attr_mlp).register_forward_hook(lambda m, i, o, L=L: caps.update({("mout", L): o.detach()})))
    try:
        with torch.no_grad():
            run(ids_input)
    finally:
        for h in handles:
            h.remove()
    return caps


def _check_states(teacher, caps, ids, n_layers, tol=2e-5):
    st = dg.teacher_single_token_states(teacher, ids, n_layers - 1)
    worst = 0.0
    for L in range(n_layers):
        for key, cap in (("resid_pre", "pre"), ("resid_in", "rin"), ("mlp_in", "min"), ("mlp_out", "mout")):
            ref = caps[(cap, L)].reshape(len(ids), -1).float()
            got = st[key][L]
            rel = float((got - ref).abs().max() / ref.abs().max().clamp_min(1e-12))
            worst = max(worst, rel)
            assert rel < tol, (key, L, rel)
    return st, worst


# ---------------------------------------------------------------------------
# fast tests
# ---------------------------------------------------------------------------
def test_gpt2_single_token_states_match_hf(tiny_gpt2):
    teacher, ref, _ = tiny_gpt2
    assert teacher.arch == "gpt2" and teacher.d_model == 64 and teacher.d_ff(0) == 256
    assert isinstance(teacher.embed, dg.LazyTensor) and isinstance(teacher.layers[0]["fc_w"], dg.LazyTensor)
    ids = torch.randint(0, 28896, (40,), generator=torch.Generator().manual_seed(0))
    caps = _hook_states(ref.h, "ln_2", "mlp", ids.view(-1, 1), lambda x: ref(input_ids=x))
    _, worst = _check_states(teacher, caps, ids, 2)
    print(f"gpt2 single-token states vs HF: worst relative max-diff {worst:.2e}")


def test_qwen_single_token_states_match_hf(tiny_qwen):
    teacher, ref, _ = tiny_qwen
    assert teacher.arch == "qwen2" and teacher.extra["n_kv_heads"] == 2 and teacher.d_ff(1) == 160
    ids = torch.randint(0, 151000, (40,), generator=torch.Generator().manual_seed(1))
    caps = _hook_states(ref.layers, "post_attention_layernorm", "mlp", ids.view(-1, 1), lambda x: ref(input_ids=x))
    _, worst = _check_states(teacher, caps, ids, 2)
    print(f"qwen2 single-token states vs HF (GQA fold): worst relative max-diff {worst:.2e}")


@pytest.mark.parametrize("which", ["tiny_gpt2", "tiny_qwen"])
def test_mlp_acts_reproduce_mlp_out(which, request):
    teacher = request.getfixturevalue(which)[0]
    ids = torch.arange(100, 160)
    st = dg.teacher_single_token_states(teacher, ids, 1)
    for L in (0, 1):
        A = dg.teacher_mlp_acts(teacher, L, st["mlp_in"][L], ff_chunk=48)
        lay = teacher.layers[L]
        down = dg._full(lay["down_w"]).T if teacher.arch == "qwen2" else dg._full(lay["proj_w"])
        out = A @ down + (lay["proj_b"] if teacher.arch == "gpt2" else 0)
        assert torch.allclose(out, st["mlp_out"][L], rtol=1e-4, atol=1e-5)
        cols = torch.tensor([5, 0, 77, 5, 31])
        assert torch.allclose(dg.teacher_mlp_acts(teacher, L, st["mlp_in"][L], cols=cols), A[:, cols], atol=1e-6)
    # stop_before_mlp / keep_layers
    st1 = dg.teacher_single_token_states(teacher, ids, 1, keep_layers=[1], stop_before_mlp=True)
    assert list(st1["mlp_in"]) == [1] and st1["mlp_out"] == {}
    assert torch.allclose(st1["mlp_in"][1], st["mlp_in"][1])


def test_lazy_tensor_gathers(tiny_qwen):
    teacher = tiny_qwen[0]
    lz = teacher.layers[0]["down_w"]  # (64, 160) bf16 on disk
    dense = lz.full()
    ids = torch.tensor([159, 3, 3, 64, 0, 100, 101])
    assert torch.equal(lz.take_cols(ids, block=16), dense[:, ids])
    assert torch.equal(lz.take_rows(torch.tensor([63, 1, 1, 20]), block=8), dense[[63, 1, 1, 20]])
    assert torch.equal(lz.rows(2, 9), dense[2:9]) and torch.equal(lz.cols(10, 40), dense[:, 10:40])
    emb = teacher.embed
    big = torch.tensor([151935, 0, 70000, 70001, 12])
    rows = emb.take_rows(big)
    assert rows.dtype == torch.float32 and rows.shape == (5, 64)
    assert torch.equal(rows[2], emb.rows(70000, 70001)[0])


def test_ridge_recovers_linear_map():
    g = torch.Generator().manual_seed(0)
    A = torch.randn(2000, 30, generator=g)
    W = torch.randn(30, 7, generator=g)
    W_hat, r2 = dg.ridge(A, A @ W, lam=1e-8)
    assert r2 > 0.999999 and torch.allclose(W_hat, W, atol=1e-4)
    W_n, r2_n = dg.ridge(A, A @ W + 3.0 * torch.randn(2000, 7, generator=g))
    assert 0.3 < r2_n < 0.95  # noisy target: held-out R2 is honest


def test_match_tokens_roundtrip(student, tiny_gpt2, tiny_qwen):
    _, tok = student
    for teacher in (tiny_gpt2[0], tiny_qwen[0]):
        s_ids, t_ids = dg.match_tokens(tok, teacher)
        assert s_ids.numel() > 1000 and s_ids.numel() == t_ids.numel()
        bad = sum(teacher.tokenizer.decode([int(t)]) != tok.tokens[int(s)] for s, t in zip(s_ids, t_ids))
        print(f"match_tokens {teacher.name}: {s_ids.numel()} pairs, {bad} decode mismatches")
        assert bad <= 0.01 * s_ids.numel()
        assert int(s_ids.min()) >= 6


def _grow_layer_inplace(model, layer: int, n_new: int):
    """Test helper (expanse_core.grow_moe_slots is the real one): append n_new
    dead expert slots to one MoE layer of an in-memory model."""
    from mimomix_core import SparseMoEFeedForward
    old = model.layers[layer].mlp
    cfg = copy.deepcopy(model.config)
    cfg.moe_spare_experts = int(cfg.moe_spare_experts) + n_new
    new = SparseMoEFeedForward(cfg)
    sd = old.state_dict()
    n_old = old.n_routed
    alive_bias = old.expert_bias[old.expert_alive.bool()]
    sd["gate.weight"] = torch.cat([sd["gate.weight"], torch.zeros(n_new, sd["gate.weight"].shape[1])])
    sd["expert_bias"] = torch.cat([sd["expert_bias"], torch.full((n_new,), float(alive_bias.min()) - 10.0)])
    sd["expert_alive"] = torch.cat([sd["expert_alive"], torch.zeros(n_new, dtype=torch.uint8)])
    for i in range(n_old, n_old + n_new):
        for k in ("gate_proj", "up_proj", "down_proj"):
            sd[f"experts.{i}.{k}.weight"] = torch.zeros_like(sd[f"experts.0.{k}.weight"])
    new.load_state_dict(sd, strict=True)
    new.train(old.training)
    model.layers[layer].mlp = new
    return list(range(n_old, n_old + n_new))


def _logits(model, tok, prompts):
    outs = []
    with torch.no_grad():
        for p in prompts:
            ids, _ = tok.encode_turn(p, None)
            outs.append(model(torch.tensor([ids]), use_fly=False).logits)
    return outs


def _count_routed(model, tok, layer, slots, prompts):
    mlp = model.layers[layer].mlp
    got = {"hits": 0, "tokens": 0}

    def hook(m, inputs):
        x = inputs[0].reshape(-1, inputs[0].shape[-1])
        alive = m.expert_alive.bool()
        logits = (x @ m.gate.weight.T).masked_fill(~alive, float("-inf"))
        sel = (torch.softmax(logits, -1) + m.expert_bias).masked_fill(~alive, float("-inf"))
        idx = torch.topk(sel, m.top_k, dim=-1).indices
        got["hits"] += int(torch.isin(idx, torch.tensor(slots)).any(dim=1).sum())
        got["tokens"] += x.shape[0]

    h = mlp.register_forward_pre_hook(hook)
    try:
        _logits(model, tok, prompts)
    finally:
        h.remove()
    return got


PROMPTS = ["What is 47 x 6?", "def add(a, b): return a + b. What does add(2, 3) return?",
           "What does the enzyme lactase break down?", "Explain what a for loop does in Python."]


def test_build_install_wake_tiny_gpt2(student, tiny_gpt2):
    model, tok = student
    model = copy.deepcopy(model)  # never touch the shared fixture
    teacher = tiny_gpt2[0]
    t0 = time.time()
    experts = dg.build_donor_experts(model, tok, teacher, src_layer=1, dst_layer=1, n_experts=2,
                                     domain_ids=None, fit_steps=80, seed=0, max_tokens=1500)
    assert len(experts) == 2
    fit = experts[0]["fit"]
    assert fit["tokens_used"] == 1500 and fit["heldout_tokens"] == 150 and fit["selected_neurons"] == 192
    assert fit["seconds"]["total"] > 0 and "ones_read_rmse_heldout" in fit["maps"]
    for e in experts:
        assert e["gate_proj"].shape == (96, 320) and e["up_proj"].shape == (96, 320)
        assert e["down_proj"].shape == (320, 96) and e["router"].shape == (320,)
        assert len(set(e["neurons"])) == 96
        # refinement keeps the best train fit seen (held-out R2 on a random tiny teacher may not improve)
        assert e["fit_mse_last"] <= e["fit_mse_first"] + 1e-9, (e["fit_mse_first"], e["fit_mse_last"])
        # load-matched wake: a woken donor takes roughly a typical expert's share of held-out tokens
        assert e["write_scale"] in (0.1, 1.0)
        assert e["wake_share_heldout"] <= max(0.2, 4 * e["wake_share_target"]), (e["wake_share_heldout"], e["wake_share_target"])
        assert abs(float(e["router"].norm()) - fit["router_norm"]) < 1e-3 * fit["router_norm"]
    assert not set(experts[0]["neurons"]) & set(experts[1]["neurons"])
    json.dumps(dg.experts_summary(experts))
    print(f"tiny gpt2 build: {time.time() - t0:.1f}s, ridge {fit['ridge_r2']}, "
          f"R2 {[(round(e['r2_before'], 3), round(e['r2_after'], 3)) for e in experts]}")

    # function preservation: grow (in memory) -> install dormant -> bit-identical
    base = _logits(model, tok, PROMPTS)
    slots = _grow_layer_inplace(model, 1, 2)
    grown = _logits(model, tok, PROMPTS)
    grow_diff = max(float((a - b).abs().max()) for a, b in zip(base, grown))
    receipt = dg.install_experts(model, 1, experts, slots)
    installed = _logits(model, tok, PROMPTS)
    assert all(torch.equal(a, b) for a, b in zip(grown, installed)), "dormant install must be bit-for-bit"
    info = receipt["layers"][1]
    assert info["slots"] == slots and info["dormant_bias"] < info["target_bias"]
    mlp = model.layers[1].mlp
    assert int(mlp.expert_alive[slots].sum()) == 0
    assert torch.equal(mlp.experts[slots[0]].down_proj.weight, experts[0]["down_proj"])
    with pytest.raises(ValueError):
        dg.install_experts(model, 1, experts[:1], [0])  # alive slot refused

    from archimedes_core import wake_grafted_experts
    json_receipt = json.loads(json.dumps(receipt))  # survives a JSON round trip (string layer keys)
    before = _count_routed(model, tok, 1, slots, PROMPTS)
    wake_grafted_experts(model, json_receipt, 1.0)
    after = _count_routed(model, tok, 1, slots, PROMPTS)
    woke = _logits(model, tok, PROMPTS)
    change = max(float((a - b).abs().max()) for a, b in zip(installed, woke))
    print(f"grow max|dlogit| {grow_diff:.2e}; install bit-identical; routed to donors: dormant "
          f"{before['hits']}/{before['tokens']} -> awake {after['hits']}/{after['tokens']} tokens; "
          f"awake max|dlogit| {change:.3f}")
    assert before["hits"] == 0 and after["hits"] > 0 and change > 0
    with pytest.raises(ValueError):
        dg.install_experts(model, 1, experts, slots)  # now alive -> refused

    merged = dg.merge_install_receipts(receipt, {"layers": {"1": {**info, "slots": [99], "placed": [{}]}}})
    assert merged["layers"][1]["slots"] == slots + [99]


def test_lifted_embedding_rows_tiny(student, tiny_gpt2, tiny_qwen):
    model, tok = student
    E = model.embed_tokens.weight.detach()
    words = [" hydrophobic", " lactase", "zzqxjvwk" * 3, " def"]
    rows, rec = dg.lifted_embedding_rows(words, E, tok, [tiny_gpt2[0], tiny_qwen[0]])
    assert rows.shape == (4, 320)
    assert 2 in rec["uncovered"] and rec["covered"] == 3
    norms = rows.norm(dim=1)
    assert torch.allclose(norms[[0, 1, 3]], torch.full((3,), rec["target_norm"]), rtol=1e-4)
    assert float(norms[2]) == 0.0
    assert set(rec["teachers"]) == {tiny_gpt2[0].name, tiny_qwen[0].name}
    json.dumps(rec)


# ---------------------------------------------------------------------------
# slow demos on the real teachers
# ---------------------------------------------------------------------------
def code_domain_ids(tok) -> set:
    """Student ids that look like code: Python keywords/builtins, operator and
    bracket tokens, and every token of the replay code corpus user prompts."""
    names = set(keyword.kwlist) | {n for n in dir(builtins) if n.islower() and not n.startswith("_")}
    names |= {"append", "join", "split", "self", "items", "keys", "values", "lambda", "def", "return", "range"}
    ids = {i for i, t in enumerate(tok.tokens) if t.strip() in names or (t.strip() and all(not c.isalnum() for c in t.strip()))}
    corpus = EXP / "supermix-archimedes" / "corpus" / "code.jsonl"
    if corpus.exists():
        for line in corpus.read_text(encoding="utf-8").splitlines():
            ids.update(tok.encode(json.loads(line)["user"]))
    return {i for i in ids if i >= 6}


def _peak_mb():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return round(getattr(mi, "peak_wset", mi.rss) / 2**20, 1), round(mi.rss / 2**20, 1)
    except Exception:
        return None, None


def _write_receipt(name, payload):
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    path = RECEIPTS / name
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    print("receipt ->", path)


@pytest.mark.skipif(not (REAL and (QWEN_DIR / "graft_slices.safetensors").exists()), reason="set EXPANSE_REAL=1")
def test_real_qwen_demo(student):
    model, tok = student
    model = copy.deepcopy(model)
    peak0, rss0 = _peak_mb()
    t0 = time.time()
    qwen = dg.load_qwen_slices(QWEN_DIR / "graft_slices.safetensors", QWEN_DIR / "tokenizer.json", QWEN_DIR / "config.json")
    t_load = time.time() - t0
    dom = code_domain_ids(tok)
    experts = dg.build_donor_experts(model, tok, qwen, src_layer=0, dst_layer=1, n_experts=2, domain_ids=dom,
                                     fit_steps=100, seed=0)
    wall = time.time() - t0
    peak1, rss1 = _peak_mb()
    fit = experts[0]["fit"]
    for e in experts:
        assert e["r2_after"] > e["r2_before"]
    # lifted rows sanity on 300 existing student tokens treated as "new" (their true rows are known)
    s_ids, _ = dg.match_tokens(tok, qwen)
    g = torch.Generator().manual_seed(1)
    probe = s_ids[torch.randperm(s_ids.numel(), generator=g)[:300]]
    rows, lift = dg.lifted_embedding_rows([tok.tokens[int(i)] for i in probe], model.embed_tokens.weight.detach(), tok, [qwen])
    true = model.embed_tokens.weight.detach()[probe]
    cos = torch.nn.functional.cosine_similarity(rows, true, dim=1)
    rand_cos = torch.nn.functional.cosine_similarity(rows, true[torch.randperm(300, generator=g)], dim=1)
    # dormant install in memory + wake check
    base = _logits(model, tok, PROMPTS)
    slots = _grow_layer_inplace(model, 1, 2)
    grown = _logits(model, tok, PROMPTS)
    receipt = dg.install_experts(model, 1, experts, slots)
    installed = _logits(model, tok, PROMPTS)
    bit_identical = all(torch.equal(a, b) for a, b in zip(grown, installed))
    from archimedes_core import wake_grafted_experts
    wake_grafted_experts(model, receipt, 1.0)
    routed = _count_routed(model, tok, 1, slots, PROMPTS)
    out = {
        "what": "donor_graft demo: Qwen2.5-Coder-7B-Instruct layer 0 MLP -> Archimedes MoE layer 1, 2 experts, "
                "installed only into an in-memory copy (checkpoint on disk untouched)",
        "teacher": qwen.describe(), "load_seconds": round(t_load, 2), "wall_seconds_total": round(wall, 2),
        "threads": torch.get_num_threads(), "domain_ids": len(dom), "fit": fit,
        "experts": dg.experts_summary(experts),
        "ram_mb": {"rss_before": rss0, "peak_wset_before": peak0, "rss_after": rss1, "peak_wset_after": peak1},
        "lifted_rows_probe": {"n": 300, "map_r2_heldout": lift["teachers"][qwen.name]["map_r2_heldout"],
                              "cos_to_true_row_mean": float(cos.mean()), "cos_to_random_row_mean": float(rand_cos.mean())},
        "install": {"receipt": json.loads(json.dumps(receipt)), "dormant_bit_identical": bit_identical,
                    "grow_max_abs_dlogit": max(float((a - b).abs().max()) for a, b in zip(base, grown)),
                    "awake_routed_tokens": routed},
    }
    _write_receipt("donor_demo_qwen_L0_to_L1.json", out)
    print(json.dumps({k: out[k] for k in ("wall_seconds_total", "ram_mb", "lifted_rows_probe")}, indent=1))
    print("fit:", json.dumps({k: fit[k] for k in ("matched_tokens", "tokens_used", "ridge_r2", "target_rms", "seconds")}))
    for e in out["experts"]:
        print(f"  expert {e['group']}: R2 {e['r2_before']:.3f} -> {e['r2_after']:.3f}, top tokens {e['router_top_tokens'][:8]}")
    assert bit_identical and routed["hits"] > 0


@pytest.mark.skipif(not (REAL and (BIO_DIR / "fetch.receipt.json").exists()), reason="set EXPANSE_REAL=1 (and fetch BioMedLM)")
def test_real_biomedlm_smoke(student):
    model, tok = student
    peak0, _ = _peak_mb()
    t0 = time.time()
    bio = dg.load_biomedlm_slices(BIO_DIR, layers=(0, 1))
    experts = dg.build_donor_experts(model, tok, bio, src_layer=0, dst_layer=1, n_experts=1, fit_steps=60, seed=0)
    wall = time.time() - t0
    peak1, _ = _peak_mb()
    fit = experts[0]["fit"]
    assert experts[0]["r2_after"] > experts[0]["r2_before"]
    _write_receipt("donor_smoke_biomedlm_L0_to_L1.json", {
        "what": "donor_graft smoke: BioMedLM layer 0 MLP -> Archimedes MoE layer 1, 1 expert (in memory only)",
        "teacher": bio.describe(), "wall_seconds_total": round(wall, 2), "fit": fit,
        "experts": dg.experts_summary(experts), "ram_mb": {"peak_wset_before": peak0, "peak_wset_after": peak1}})
    print(f"biomedlm: {fit['matched_tokens']} matched, ridge {fit['ridge_r2']}, "
          f"R2 {experts[0]['r2_before']:.3f} -> {experts[0]['r2_after']:.3f}, {wall:.1f}s")


if __name__ == "__main__":
    if "--real" in sys.argv:
        os.environ["EXPANSE_REAL"] = "1"
        sys.argv.remove("--real")
    sys.exit(pytest.main([__file__, "-q", "-s", *sys.argv[1:]]))
