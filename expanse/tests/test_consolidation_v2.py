import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import consolidation_v2 as cv2  # noqa: E402


def test_native_block_identity_and_route():
    torch.manual_seed(0)
    cfg = cv2.V2Config(hidden_size=16, shared_dim=24, layers=(0,), latent_experts=4,
                       latent_rank=8, latent_top_k=2, memory_slots=5)
    b = cv2.NativeLatentBlock(cfg)
    x = torch.randn(2, 3, 16)
    y, aux = b(x)
    assert torch.equal(x, y)
    assert aux["latent"].shape == (2, 3, 24)
    assert aux["route"].shape == (2, 3, 4)
    assert torch.allclose(aux["route"].sum(-1), torch.ones(2, 3))
    assert (aux["route"] > 0).sum(-1).max().item() <= 2
    with torch.no_grad():
        b.out_gate.fill_(0.03)
    y2, _ = b(x)
    assert not torch.equal(x, y2)


class TupleLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x):
        return self.lin(x), None


class Dummy(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.layers = nn.ModuleList([TupleLayer(d) for _ in range(3)])

    def forward(self, x):
        for l in self.layers:
            x, _ = l(x)
        return x


def test_stack_exact_identity_when_attached():
    torch.manual_seed(1)
    m = Dummy()
    x = torch.randn(2, 4, 16)
    ref = m(x)
    cfg = cv2.V2Config(hidden_size=16, shared_dim=24, layers=(0, 2), latent_experts=4,
                       latent_rank=8, latent_top_k=2, memory_slots=4)
    stack = cv2.attach_consolidation_v2(m, cfg)
    got = m(x)
    assert torch.equal(ref, got)
    assert set(stack.last) == {0, 2}
    assert "consolidation_v2.blocks.0.out_gate" in m.state_dict()


def test_fusion_bank():
    torch.manual_seed(2)
    cfg = cv2.V2Config(hidden_size=16, shared_dim=24, layers=(0,), latent_experts=4,
                       latent_rank=8, latent_top_k=2, memory_slots=0,
                       source_dims={"arch": 16, "qwen": 12, "omni7": 20},
                       domains=("replay", "code"))
    bank = cv2.TeacherFusionBank(cfg, projector_rank=8)
    sources = {
        "arch": torch.randn(5, 16),
        "qwen": torch.randn(5, 12),
        "omni7": torch.randn(5, 20),
    }
    student = torch.randn(5, 24)
    out = bank(sources, student, domains=["replay", "code", "code", "replay", "code"])
    assert out["fused"].shape == (5, 24)
    assert out["weights"].shape == (5, 3)
    assert torch.allclose(out["weights"].sum(-1), torch.ones(5), atol=1e-6)
    al = bank.alignment_loss(out["projected"])
    sl = bank.student_loss(student, out["fused"])["total"]
    assert torch.isfinite(al) and torch.isfinite(sl)
    (al + sl).backward()
    assert any(p.grad is not None for p in bank.parameters())


def test_config_roundtrip():
    c = cv2.V2Config().validate()
    d = c.to_dict()
    c2 = cv2.V2Config.from_dict(d)
    assert c2 == c


class DenseExpert(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.lin(x)


class DummyMLP(nn.Module):
    def __init__(self, d, n=4):
        super().__init__()
        self.experts = nn.ModuleList([DenseExpert(d) for _ in range(n)])


class DonorLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mlp = DummyMLP(d)


class DonorModel(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.layers = nn.ModuleList([DonorLayer(d) for _ in range(2)])
        self.donor_receipts = {
            "qwen": {"layers": {"1": {"slots": [0, 1]}}},
            "biomedlm": {"layers": {"1": {"slots": [2, 3]}}},
        }
        self.fly_core = None
        self.cns_full = None


def test_internal_source_extractor_donors():
    torch.manual_seed(3)
    m = DonorModel()
    ex = cv2.InternalSourceExtractor(m)
    h = torch.randn(3, 5, 16)
    lengths = torch.tensor([2, 4, 5])
    src = ex.collect(1, h, lengths, include_slow=False)
    assert set(src) == {"arch", "qwen", "biomedlm"}
    assert src["arch"].shape == (3, 16)
    assert src["qwen"].shape == (3, 16)
    assert src["biomedlm"].shape == (3, 16)


def test_phase_schedule_has_strict_teacher_free_tail():
    early = cv2.consolidation_phase_weights(0.0, 0.72, 0.90)
    bake = cv2.consolidation_phase_weights(0.80, 0.72, 0.90)
    final = cv2.consolidation_phase_weights(0.90, 0.72, 0.90)
    assert early["teacher"] == 1.0 and early["align"] == 1.0
    assert 0.35 < bake["distill"] < 1.0 and bake["teacher"] == 1.0
    assert final == {"distill": 0.0, "align": 0.0, "teacher": 0.0}


def test_closed_gate_is_exact_but_trainable():
    """Regression: a zero out_gate must still receive gradient in training, or it never opens."""
    torch.manual_seed(0)
    cfg = cv2.V2Config(hidden_size=16, shared_dim=24, layers=(0,), latent_experts=4,
                       latent_rank=8, latent_top_k=2, memory_slots=5)
    b = cv2.NativeLatentBlock(cfg).train()
    x = torch.randn(2, 3, 16)
    y, _ = b(x)
    assert torch.equal(x, y)                       # still exact at birth, in training mode too
    y.pow(2).sum().backward()
    assert b.out_gate.grad is not None and float(b.out_gate.grad.abs().sum()) > 0
    b.eval()
    with torch.no_grad():
        y2, _ = b(x)
    assert torch.equal(x, y2)                      # inference with a closed gate skips the write


def test_student_loss_is_scale_bounded():
    """Regression: cosine/relational terms use unit vectors, so the loss stays O(1) at any width."""
    torch.manual_seed(0)
    for dim in (16, 512):
        s, t = torch.randn(4, 1, dim), torch.randn(4, dim)
        parts = cv2.TeacherFusionBank.student_loss(s, t)
        assert 0.0 <= float(parts["cosine"]) <= 2.0
        assert float(parts["relational"]) <= 4.0
        assert float(parts["total"]) < 10.0
        same = cv2.TeacherFusionBank.student_loss(t.unsqueeze(1), t)
        assert float(same["total"]) < 1e-4
