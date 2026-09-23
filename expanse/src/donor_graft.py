"""Teacher -> student weight grafting: donor MoE experts and lifted embedding rows.

Expanse gives Archimedes (320-d, word tokenizer) two much larger teachers --
Qwen2.5-Coder-7B-Instruct (3584-d, byte BPE, SwiGLU 18944) and BioMedLM 2.7B
(2560-d GPT-2, gelu_new 10240). Distillation on generated text transfers what a
teacher *says*; this module additionally transfers what its early MLPs *compute*,
by carving a few hundred of their neurons into new SwiGLU experts of the student
MoE, and seeds new vocabulary rows from the teachers' embedding tables.

Why single-token contexts
    The two models share no coordinate system and no tokenizer, so the only
    place where "the same input" is well defined is a token both vocabularies
    spell identically (``match_tokens``: a student token string that the teacher
    encodes to exactly one token). With that token alone at position 0,
    attention over one position is the identity on values -- softmax over one key
    is 1, RoPE and the 1/(layer+1) scaling never matter -- so the teacher's
    forward up to any early MLP is *exact* and cheap without the rest of the
    network: ``o(v(norm(x)))`` (+ biases; GQA heads folded for Qwen). The student
    is run the same way (input_ids (n, 1)) and its MoE block is hooked. Paired
    rows give every map below a supervised fit.

Maps (row-vector convention, ``y ~ x @ W``; ``ridge`` = fp64 closed form)
    ``P`` (d -> 320): teacher residual at its MLP input (``resid_in``, the stream
        the MLP output is *added* to) -> student residual at the MoE input (the
        ``post_attn_norm`` input, which the MoE output is added to). This is the
        "capture the residual too" option of the spec: residual-to-residual is
        the space where a contribution lives, so a teacher neuron's write
        direction ``W_down[:, j]`` maps to ``W_down[:, j] @ P`` in the student.
    ``Q`` (320 -> d): student MoE input ``X_s`` (post-norm) -> teacher MLP input
        (post-norm). Folding it into the teacher's input weights lets an expert
        read the student stream as if it were the teacher's.

Expert construction (``build_donor_experts``)
    1. rows: matched tokens -> ``X_s``, student residual, teacher ``resid_in``/``mlp_in``;
       10% of tokens are held out of *every* fit and only used for R2.
    2. score every teacher neuron: ``mean_domain |A_j| * ||W_down[:, j] @ P||``
       (streamed over d_ff chunks, never holding the (n, 18944) activation matrix).
    3. top ``96 * n_experts`` neurons -> ``n_experts`` groups of 96 by balanced
       spherical k-means on their activation profiles over tokens (neurons that
       fire together live in one expert, so the router can pick them together).
    4. init by folding: qwen2 ``gate = Wg_sel Q^T``, ``up = Wu_sel Q^T``; gpt2
       ``gate = 1.702 Z``, ``up = Z`` with ``Z = (Q W_fc_sel + beta b_sel)^T``
       (``beta`` is a least-squares "constant-one" read of ``X_s`` so the c_fc
       bias survives the bias-free student), ``up`` rescaled so the output rms
       matches the target; ``down = s * (W_down_sel^T P)^T``. The target
       ``Y = s * (group's teacher contribution) @ P`` with ``s`` making its rms the
       median rms of the layer's existing routed-expert outputs (so a woken
       donor speaks at the volume of its neighbours). Then Adam refines the
       three matrices by local function matching (MSE of ``expert(X_s)`` vs ``Y``).
    5. router row = normalised mean ``X_s`` over the tokens where the group's
       contribution is in the top 10%, scaled to the median alive router norm.

Installation (``install_experts``) writes weights + router rows into *dead* slots
and leaves them dead (``expert_alive`` 0) at ``dormant_bias``: the MoE masks dead
slots to -inf before its softmax, so routing -- and the model -- is bit-for-bit
unchanged until ``archimedes_core.wake_grafted_experts`` anneals them in (see its
docstring for why a live graft at the median bias wrecks a trained router).

Numerics: teacher slices are bf16 at rest and fp32 the moment they are read
(``LazyTensor``); nothing here multiplies in bf16 (1 GFLOP/s on this CPU). The
Qwen embedding (152064 x 3584) is only ever read by row blocks.
"""

from __future__ import annotations

import inspect
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import torch
import torch.nn.functional as F
from safetensors import safe_open
from tokenizers import Tokenizer

# Same strings as mimomix_text.SPECIAL_TOKENS (ids 0..5); never matched.
STUDENT_SPECIALS = ("<pad>", "<bos>", "<eos>", "<unk>", "<user>", "<assistant>")
GELU_SIGMOID_K = 1.702  # gelu(z) ~ z * sigmoid(1.702 z)


# ---------------------------------------------------------------------------
# 1. Teacher tensors: bf16 on disk, fp32 on read
# ---------------------------------------------------------------------------
class LazyTensor:
    """A tensor inside a safetensors file, read on demand and returned as fp32.

    Big teacher matrices (Qwen MLP 18944 x 3584, the 152064-row embedding) stay
    on disk / in the page cache in bf16; callers pull row or column blocks and
    get fp32 copies, so peak RAM is one block, not the matrix."""

    def __init__(self, path: Union[str, Path], key: str, shape: Sequence[int]):
        self.path, self.key, self.shape = str(path), key, tuple(int(s) for s in shape)

    def __repr__(self) -> str:
        return f"LazyTensor({Path(self.path).name}:{self.key}, shape={self.shape})"

    def _slice(self, *index):
        with safe_open(self.path, framework="pt") as f:
            return f.get_slice(self.key)[index if len(index) > 1 else index[0]].float()

    def full(self) -> torch.Tensor:
        with safe_open(self.path, framework="pt") as f:
            return f.get_tensor(self.key).float()

    def rows(self, start: int, stop: int) -> torch.Tensor:
        return self._slice(slice(int(start), int(stop)))

    def cols(self, start: int, stop: int) -> torch.Tensor:
        return self._slice(slice(None), slice(int(start), int(stop)))

    def _take(self, ids: torch.Tensor, dim: int, block: int) -> torch.Tensor:
        ids = torch.as_tensor(ids, dtype=torch.long).flatten()
        if ids.numel() == 0:
            shape = list(self.shape)
            shape[dim] = 0
            return torch.zeros(shape)
        uniq, inverse = torch.unique(ids, sorted=True, return_inverse=True)
        other = self.shape[1 - dim]
        out = torch.empty((uniq.numel(), other) if dim == 0 else (other, uniq.numel()))
        blocks = torch.div(uniq, block, rounding_mode="floor")
        with safe_open(self.path, framework="pt") as f:
            sl = f.get_slice(self.key)
            for b in torch.unique(blocks).tolist():
                where = (blocks == b).nonzero().flatten()
                members = uniq[where]
                lo, hi = int(members[0]), int(members[-1]) + 1
                if dim == 0:
                    chunk = sl[lo:hi].float()
                    out[where] = chunk[members - lo]
                else:
                    chunk = sl[:, lo:hi].float()
                    out[:, where] = chunk[:, members - lo]
        return out[inverse] if dim == 0 else out[:, inverse]

    def take_rows(self, ids, block: int = 2048) -> torch.Tensor:
        return self._take(ids, 0, block)

    def take_cols(self, ids, block: int = 2048) -> torch.Tensor:
        return self._take(ids, 1, block)


TensorLike = Union[torch.Tensor, LazyTensor]


def _full(t: TensorLike) -> torch.Tensor:
    return t.full() if isinstance(t, LazyTensor) else t.float()


def _rows(t: TensorLike, a: int, b: int) -> torch.Tensor:
    return t.rows(a, b) if isinstance(t, LazyTensor) else t[a:b].float()


def _cols(t: TensorLike, a: int, b: int) -> torch.Tensor:
    return t.cols(a, b) if isinstance(t, LazyTensor) else t[:, a:b].float()


def _take_rows(t: TensorLike, ids) -> torch.Tensor:
    ids = torch.as_tensor(ids, dtype=torch.long)
    return t.take_rows(ids) if isinstance(t, LazyTensor) else t.index_select(0, ids).float()


def _take_cols(t: TensorLike, ids) -> torch.Tensor:
    ids = torch.as_tensor(ids, dtype=torch.long)
    return t.take_cols(ids) if isinstance(t, LazyTensor) else t.index_select(1, ids).float()


@dataclass
class TeacherSlices:
    """The first layers of a teacher, enough for exact single-token forwards.

    ``embed`` is (V, d) -- a ``LazyTensor`` for real teachers (index it with
    ``_take_rows``), a plain tensor for tiny test teachers. ``layers[L]`` holds
    layer L's tensors under canonical names (fp32 tensors or ``LazyTensor``):

    * qwen2: ``norm1`` (d), ``v_w`` (kv, d), ``v_b`` (kv), ``o_w_kv`` (d, kv) =
      o_proj with the query heads that share a kv head summed (GQA fold; exact
      for one position), ``norm2`` (d), ``gate_w``/``up_w`` (d_ff, d), ``down_w`` (d, d_ff).
    * gpt2 (Conv1D weights are (in, out)): ``ln1_w``/``ln1_b``, ``v_w`` (d, d) =
      c_attn[:, 2d:3d], ``v_b``, ``attn_proj_w`` (d, d), ``attn_proj_b``,
      ``ln2_w``/``ln2_b``, ``fc_w`` (d, d_ff), ``fc_b``, ``proj_w`` (d_ff, d), ``proj_b``;
      ``extra["wpe0"]`` is the position-0 embedding.
    """

    name: str
    arch: str  # 'qwen2' | 'gpt2'
    d_model: int
    embed: TensorLike
    layers: List[Dict[str, Any]]
    tokenizer: Tokenizer
    eps: float
    extra: Dict[str, Any] = field(default_factory=dict)

    def d_ff(self, layer: int = 0) -> int:
        lay = self.layers[layer]
        return int(lay["gate_w"].shape[0] if self.arch == "qwen2" else lay["fc_w"].shape[1])

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "arch": self.arch, "d_model": self.d_model, "d_ff": self.d_ff(0),
                "vocab": int(self.embed.shape[0]), "layers": len(self.layers), "eps": self.eps,
                "source": self.extra.get("source")}


def _contiguous_layers(keys: Iterable[str], pattern_prefix: str) -> List[int]:
    found = set()
    for k in keys:
        if k.startswith(pattern_prefix):
            rest = k[len(pattern_prefix):]
            head = rest.split(".", 1)[0]
            if head.isdigit():
                found.add(int(head))
    layers = sorted(found)
    if layers != list(range(len(layers))):
        raise ValueError(f"teacher layers must be 0..k contiguous, found {layers}")
    return layers


def load_qwen_slices(safetensors_path, tokenizer_json, config_json) -> TeacherSlices:
    """Qwen2 graft slices (embeddings + first layers' norms, v/o attention, MLP)."""

    path = str(safetensors_path)
    cfg = json.loads(Path(config_json).read_text(encoding="utf-8"))
    d = int(cfg["hidden_size"])
    n_heads, n_kv = int(cfg["num_attention_heads"]), int(cfg.get("num_key_value_heads", cfg["num_attention_heads"]))
    head_dim = int(cfg.get("head_dim") or d // n_heads)
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        shapes = {k: tuple(f.get_slice(k).get_shape()) for k in keys}
        layer_ids = _contiguous_layers(keys, "model.layers.")
        layers: List[Dict[str, Any]] = []
        for L in layer_ids:
            p = f"model.layers.{L}."
            o = f.get_tensor(p + "self_attn.o_proj.weight").float()  # (d, n_heads*head_dim)
            o_kv = o.view(d, n_kv, n_heads // n_kv, head_dim).sum(2).reshape(d, n_kv * head_dim)
            del o
            layers.append({
                "norm1": f.get_tensor(p + "input_layernorm.weight").float(),
                "v_w": f.get_tensor(p + "self_attn.v_proj.weight").float(),
                "v_b": f.get_tensor(p + "self_attn.v_proj.bias").float() if p + "self_attn.v_proj.bias" in shapes
                else torch.zeros(shapes[p + "self_attn.v_proj.weight"][0]),
                "o_w_kv": o_kv.contiguous(),
                "norm2": f.get_tensor(p + "post_attention_layernorm.weight").float(),
                "gate_w": LazyTensor(path, p + "mlp.gate_proj.weight", shapes[p + "mlp.gate_proj.weight"]),
                "up_w": LazyTensor(path, p + "mlp.up_proj.weight", shapes[p + "mlp.up_proj.weight"]),
                "down_w": LazyTensor(path, p + "mlp.down_proj.weight", shapes[p + "mlp.down_proj.weight"]),
            })
    embed = LazyTensor(path, "model.embed_tokens.weight", shapes["model.embed_tokens.weight"])
    return TeacherSlices(
        name=str(cfg.get("_name_or_path") or "qwen2.5-coder-7b-instruct"), arch="qwen2", d_model=d, embed=embed,
        layers=layers, tokenizer=Tokenizer.from_file(str(tokenizer_json)), eps=float(cfg.get("rms_norm_eps", 1e-6)),
        extra={"n_heads": n_heads, "n_kv_heads": n_kv, "head_dim": head_dim, "source": path,
               "vocab_size": int(cfg.get("vocab_size", embed.shape[0]))},
    )


def load_biomedlm_slices(hf_dir, layers=(0, 1)) -> TeacherSlices:
    """GPT-2 (BioMedLM) wte, wpe[0] and layers ``0..max(layers)`` from a (sharded) HF dir.

    Only those tensors are read (``safe_open`` per shard). The 1/(layer+1)
    attention scaling is irrelevant here: over one position softmax is 1."""

    hf = Path(hf_dir)
    cfg = json.loads((hf / "config.json").read_text(encoding="utf-8"))
    d = int(cfg["n_embd"])
    want = list(range(max(int(x) for x in layers) + 1))
    index_path = hf / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    else:
        single = hf / "model.safetensors"
        with safe_open(str(single), framework="pt") as f:
            weight_map = {k: single.name for k in f.keys()}
    prefix = "transformer." if any(k.startswith("transformer.") for k in weight_map) else ""

    def where(name: str) -> str:
        return str(hf / weight_map[prefix + name])

    def get(name: str) -> torch.Tensor:
        with safe_open(where(name), framework="pt") as f:
            return f.get_tensor(prefix + name).float()

    def lazy(name: str) -> LazyTensor:
        with safe_open(where(name), framework="pt") as f:
            shape = f.get_slice(prefix + name).get_shape()
        return LazyTensor(where(name), prefix + name, shape)

    out_layers: List[Dict[str, Any]] = []
    for L in want:
        p = f"h.{L}."
        with safe_open(where(p + "attn.c_attn.weight"), framework="pt") as f:
            v_w = f.get_slice(prefix + p + "attn.c_attn.weight")[:, 2 * d:3 * d].float()
        out_layers.append({
            "ln1_w": get(p + "ln_1.weight"), "ln1_b": get(p + "ln_1.bias"),
            "v_w": v_w.contiguous(), "v_b": get(p + "attn.c_attn.bias")[2 * d:3 * d].contiguous(),
            "attn_proj_w": get(p + "attn.c_proj.weight"), "attn_proj_b": get(p + "attn.c_proj.bias"),
            "ln2_w": get(p + "ln_2.weight"), "ln2_b": get(p + "ln_2.bias"),
            "fc_w": lazy(p + "mlp.c_fc.weight"), "fc_b": get(p + "mlp.c_fc.bias"),
            "proj_w": lazy(p + "mlp.c_proj.weight"), "proj_b": get(p + "mlp.c_proj.bias"),
        })
    with safe_open(where("wpe.weight"), framework="pt") as f:
        wpe0 = f.get_slice(prefix + "wpe.weight")[0:1].float()[0]
    return TeacherSlices(
        name=str(cfg.get("_name_or_path") or hf.name or "biomedlm"), arch="gpt2", d_model=d, embed=lazy("wte.weight"),
        layers=out_layers, tokenizer=Tokenizer.from_file(str(hf / "tokenizer.json")),
        eps=float(cfg.get("layer_norm_epsilon", 1e-5)),
        extra={"wpe0": wpe0, "n_head": int(cfg.get("n_head", 0)), "source": str(hf),
               "activation": cfg.get("activation_function", "gelu_new")},
    )


# ---------------------------------------------------------------------------
# 2. Token matching and exact single-token teacher / student forwards
# ---------------------------------------------------------------------------
def match_tokens(student_tok, teacher: TeacherSlices) -> Tuple[torch.Tensor, torch.Tensor]:
    """Student ids <-> teacher ids for student tokens the teacher spells as ONE token.

    Leading whitespace is part of both vocabularies' tokens (``" add"`` ->
    ``"Ġadd"``), so the pairing keeps position-in-word information. Specials and
    pure-whitespace tokens are skipped."""

    tokens = list(student_tok.tokens)
    cand = [i for i, s in enumerate(tokens) if s not in STUDENT_SPECIALS and s.strip() != ""]
    enc = teacher.tokenizer.encode_batch([tokens[i] for i in cand], add_special_tokens=False)
    special_ids = {int(i) for i in teacher.tokenizer.get_added_tokens_decoder().keys()}
    vmax = int(teacher.embed.shape[0])
    s_ids, t_ids = [], []
    for i, e in zip(cand, enc):
        if len(e.ids) == 1 and e.ids[0] not in special_ids and 0 <= e.ids[0] < vmax:
            s_ids.append(i)
            t_ids.append(int(e.ids[0]))
    return torch.tensor(s_ids, dtype=torch.long), torch.tensor(t_ids, dtype=torch.long)


def _rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def _teacher_embed(teacher: TeacherSlices, teacher_ids: torch.Tensor) -> torch.Tensor:
    x = _take_rows(teacher.embed, teacher_ids)
    if teacher.arch == "gpt2":
        x = x + teacher.extra["wpe0"]
    return x


def _attention_one_position(teacher: TeacherSlices, lay: Dict[str, Any], x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (normed input, attention output) for single-position inputs."""
    if teacher.arch == "qwen2":
        h = _rms_norm(x, lay["norm1"], teacher.eps)
        v = h @ lay["v_w"].T + lay["v_b"]
        return h, v @ lay["o_w_kv"].T
    h = F.layer_norm(x, (x.shape[-1],), lay["ln1_w"], lay["ln1_b"], teacher.eps)
    v = h @ lay["v_w"] + lay["v_b"]
    return h, v @ lay["attn_proj_w"] + lay["attn_proj_b"]


def _mlp_norm(teacher: TeacherSlices, lay: Dict[str, Any], x: torch.Tensor) -> torch.Tensor:
    if teacher.arch == "qwen2":
        return _rms_norm(x, lay["norm2"], teacher.eps)
    return F.layer_norm(x, (x.shape[-1],), lay["ln2_w"], lay["ln2_b"], teacher.eps)


def _acts_block(teacher: TeacherSlices, lay: Dict[str, Any], x: torch.Tensor, a: int, b: int) -> torch.Tensor:
    if teacher.arch == "qwen2":
        return F.silu(x @ _rows(lay["gate_w"], a, b).T) * (x @ _rows(lay["up_w"], a, b).T)
    return F.gelu(x @ _cols(lay["fc_w"], a, b) + lay["fc_b"][a:b], approximate="tanh")


def _acts_take(teacher: TeacherSlices, lay: Dict[str, Any], x: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    if teacher.arch == "qwen2":
        return F.silu(x @ _take_rows(lay["gate_w"], cols).T) * (x @ _take_rows(lay["up_w"], cols).T)
    return F.gelu(x @ _take_cols(lay["fc_w"], cols) + lay["fc_b"][cols], approximate="tanh")


def _down_block(teacher: TeacherSlices, lay: Dict[str, Any], a: int, b: int) -> torch.Tensor:
    """(b-a, d): each neuron's write vector into the teacher residual."""
    if teacher.arch == "qwen2":
        return _cols(lay["down_w"], a, b).T
    return _rows(lay["proj_w"], a, b)


def _down_take(teacher: TeacherSlices, lay: Dict[str, Any], cols: torch.Tensor) -> torch.Tensor:
    if teacher.arch == "qwen2":
        return _take_cols(lay["down_w"], cols).T.contiguous()
    return _take_rows(lay["proj_w"], cols)


def _mlp_forward(teacher: TeacherSlices, lay: Dict[str, Any], x: torch.Tensor, ff_chunk: int = 2048) -> torch.Tensor:
    d_ff = int(lay["gate_w"].shape[0] if teacher.arch == "qwen2" else lay["fc_w"].shape[1])
    out = torch.zeros_like(x)
    for a in range(0, d_ff, ff_chunk):
        b = min(d_ff, a + ff_chunk)
        out += _acts_block(teacher, lay, x, a, b) @ _down_block(teacher, lay, a, b)
    if teacher.arch == "gpt2":
        out += lay["proj_b"]
    return out


@torch.no_grad()
def teacher_single_token_states(teacher: TeacherSlices, teacher_ids, upto_layer: int, *,
                                keep_layers: Optional[Sequence[int]] = None, stop_before_mlp: bool = False,
                                ff_chunk: int = 2048) -> Dict[str, Dict[int, torch.Tensor]]:
    """Exact teacher forward of each token alone at position 0, through ``upto_layer``.

    Returns ``{"resid_pre", "resid_in", "mlp_in", "mlp_out"}``, each ``{L: (n, d)}``:
    ``resid_pre[L]`` enters layer L, ``resid_in[L]`` = resid_pre + attention = the
    residual the MLP output is added to, ``mlp_in[L]`` = its post-norm MLP input,
    ``mlp_out[L]`` = the MLP output (GPT-2: incl. c_proj bias). ``keep_layers``
    limits what is kept; ``stop_before_mlp`` skips the last layer's MLP (so no
    ``mlp_out`` for it) when only its inputs are needed."""

    ids = torch.as_tensor(teacher_ids, dtype=torch.long).flatten()
    if upto_layer >= len(teacher.layers):
        raise ValueError(f"teacher {teacher.name} has {len(teacher.layers)} layers loaded, need {upto_layer + 1}")
    keep = set(range(upto_layer + 1)) if keep_layers is None else {int(k) for k in keep_layers}
    out: Dict[str, Dict[int, torch.Tensor]] = {"resid_pre": {}, "resid_in": {}, "mlp_in": {}, "mlp_out": {}}
    x = _teacher_embed(teacher, ids)
    for L in range(upto_layer + 1):
        lay = teacher.layers[L]
        if L in keep:
            out["resid_pre"][L] = x
        _, attn = _attention_one_position(teacher, lay, x)
        x = x + attn
        m_in = _mlp_norm(teacher, lay, x)
        if L in keep:
            out["resid_in"][L] = x
            out["mlp_in"][L] = m_in
        if L == upto_layer and stop_before_mlp:
            break
        m_out = _mlp_forward(teacher, lay, m_in, ff_chunk)
        if L in keep:
            out["mlp_out"][L] = m_out
        x = x + m_out
    return out


@torch.no_grad()
def teacher_mlp_acts(teacher: TeacherSlices, layer: int, mlp_in: torch.Tensor, *, cols=None,
                     ff_chunk: int = 2048) -> torch.Tensor:
    """Neuron activations (n, d_ff) -- or (n, len(cols)) -- of teacher MLP ``layer``.

    qwen2: ``silu(x Wg^T) * (x Wu^T)``; gpt2: ``gelu_new(x c_fc + b)``. Computed in
    fp32, in blocks of ``ff_chunk`` neurons."""

    lay = teacher.layers[layer]
    x = mlp_in.float()
    if cols is not None:
        return _acts_take(teacher, lay, x, torch.as_tensor(cols, dtype=torch.long))
    d_ff = teacher.d_ff(layer)
    out = torch.empty(x.shape[0], d_ff)
    for a in range(0, d_ff, ff_chunk):
        b = min(d_ff, a + ff_chunk)
        out[:, a:b] = _acts_block(teacher, lay, x, a, b)
    return out


class _StopForward(Exception):
    pass


@torch.no_grad()
def student_single_token_moe_inputs(model, student_ids, layer: int, *, return_residual: bool = False,
                                    batch: int = 2048):
    """Student MoE input ``X_s`` (n, 320) at ``layer`` for each token alone at position 0.

    A forward pre-hook on ``model.layers[layer].mlp`` captures its input and stops
    the forward (nothing past the block is needed); with ``return_residual`` a
    pre-hook on ``post_attn_norm`` also captures the residual stream the MoE
    output is added to, returning ``(X_s, R_s)``. Grafts are skipped, no cache,
    eval mode; the model's mode is restored afterwards."""

    block = model.layers[layer]
    ids = torch.as_tensor(student_ids, dtype=torch.long).flatten()
    device = next(model.parameters()).device
    got: Dict[str, torch.Tensor] = {}

    def grab_mlp(module, inputs):
        got["x"] = inputs[0].detach()
        raise _StopForward

    def grab_res(module, inputs):
        got["r"] = inputs[0].detach()

    kwargs: Dict[str, Any] = {"use_cache": False}
    if "skip_grafts" in inspect.signature(type(model).forward).parameters:
        kwargs["skip_grafts"] = True
    was_training = model.training
    stats = [(m, m.collect_stats) for m in model.modules() if hasattr(m, "collect_stats")]
    handles = [block.mlp.register_forward_pre_hook(grab_mlp)]
    if return_residual:
        handles.append(block.post_attn_norm.register_forward_pre_hook(grab_res))
    xs, rs = [], []
    try:
        model.eval()
        for m, _ in stats:
            m.collect_stats = False
        for i in range(0, ids.numel(), batch):
            chunk = ids[i:i + batch].view(-1, 1).to(device)
            got.clear()
            try:
                model(chunk, **kwargs)
            except _StopForward:
                pass
            if "x" not in got:
                raise RuntimeError(f"layer {layer} MoE input was not captured")
            xs.append(got["x"].reshape(-1, got["x"].shape[-1]).float().cpu())
            if return_residual:
                rs.append(got["r"].reshape(-1, got["r"].shape[-1]).float().cpu())
    finally:
        for h in handles:
            h.remove()
        for m, flag in stats:
            m.collect_stats = flag
        if hasattr(model, "_ctx"):
            model._ctx = {}
        model.train(was_training)
    X = torch.cat(xs) if xs else torch.zeros(0, block.post_attn_norm.weight.shape[0])
    if return_residual:
        return X, torch.cat(rs) if rs else torch.zeros_like(X)
    return X


# ---------------------------------------------------------------------------
# 3. Least squares
# ---------------------------------------------------------------------------
def _r2(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred, target = pred.double(), target.double()
    ss_res = float(((target - pred) ** 2).sum())
    ss_tot = float(((target - target.mean(0, keepdim=True)) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-300)


def _ridge_solve(A: torch.Tensor, B: torch.Tensor, lam: float) -> torch.Tensor:
    gram = A.T @ A
    reg = float(lam) * float(gram.diagonal().mean().clamp_min(1e-300))
    gram.diagonal().add_(reg)
    return torch.linalg.solve(gram, A.T @ B)


def ridge(A: torch.Tensor, B: torch.Tensor, lam: Union[float, str] = 1e-3, *, holdout: float = 0.1,
          seed: int = 0) -> Tuple[torch.Tensor, float]:
    """``W`` minimising ``||A W - B||^2 + lam' ||W||^2`` in fp64 closed form.

    ``lam`` is relative (``lam' = lam * mean diag(A^T A)``) so one default works
    for embeddings and residual streams of any scale. ``r2`` is measured on a
    random ``holdout`` fraction of rows with ``W`` fitted on the rest; the
    returned ``W`` (fp32) is then refitted on all rows. No intercept (see
    ``fit_map`` for the centred, validated variant ``build_donor_experts``
    uses); ``lam="auto"`` picks the strength from ``RIDGE_GRID`` on a validation split."""

    if lam == "auto":
        m = fit_map(A, B, "auto", center=False, holdout=holdout, seed=seed)
        return m.W, m.r2
    A64, B64 = A.double(), B.double()
    n = A64.shape[0]
    r2 = float("nan")
    n_ho = int(round(n * holdout))
    if n_ho >= 2 and n - n_ho >= 2:
        g = torch.Generator().manual_seed(int(seed))
        perm = torch.randperm(n, generator=g)
        ho, tr = perm[:n_ho], perm[n_ho:]
        W_tr = _ridge_solve(A64[tr], B64[tr], lam)
        r2 = _r2(A64[ho] @ W_tr, B64[ho])
    W = _ridge_solve(A64, B64, lam)
    return W.float(), r2


#: Relative ridge strengths tried by ``lam="auto"``. The student/teacher spaces
#: share little linear structure (top-10 nearest-neighbour overlap of the two
#: embedding tables is ~3%), so an unvalidated small ridge on a 3584-d input
#: fits noise: measured on Qwen L0 -> student L1, held-out R2 -0.19 at 1e-3
#: vs +0.08 at 10.
RIDGE_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def _ridge_path(A: torch.Tensor, B: torch.Tensor, lams: Sequence[float]) -> List[torch.Tensor]:
    """``W(lam)`` for every relative ``lam`` from one eigendecomposition of ``A^T A`` (fp64)."""
    gram = A.T @ A
    scale = float(gram.diagonal().mean().clamp_min(1e-300))
    evals, V = torch.linalg.eigh(gram)
    proj = V.T @ (A.T @ B)
    return [V @ (proj / (evals.clamp_min(0) + float(lam) * scale).unsqueeze(1)) for lam in lams]


@dataclass
class LinearMap:
    """``y ~ (x - mu_in) @ W + mu_out`` (``mu_*`` zero when fitted without intercept)."""

    W: torch.Tensor
    mu_in: torch.Tensor
    mu_out: torch.Tensor
    lam: float
    r2: float
    r2_by_lam: Dict[str, float]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mu_in) @ self.W + self.mu_out

    @property
    def intercept(self) -> torch.Tensor:
        """``c`` with ``y ~ x @ W + c``."""
        return self.mu_out - self.mu_in @ self.W

    def receipt(self) -> Dict[str, Any]:
        return {"r2_heldout": self.r2, "lam_rel": self.lam, "centered": bool(self.mu_in.abs().sum() > 0),
                "r2_val_by_lam": self.r2_by_lam}


def fit_map(A: torch.Tensor, B: torch.Tensor, lam: Union[float, str] = "auto", *, center: bool = True,
            holdout: float = 0.1, seed: int = 0, grid: Sequence[float] = RIDGE_GRID) -> LinearMap:
    """Ridge map with optional intercept and validated strength (fp64).

    Rows split into ``holdout`` (R2 only) and the rest; with ``lam="auto"`` the
    rest is split again (90/10) and the grid value with the best validation R2
    wins. The reported R2 is for a fit on the rest; the returned map is refitted
    on all rows."""

    A64, B64 = A.double(), B.double()
    n = A64.shape[0]
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g)
    n_ho = int(round(n * holdout)) if n >= 20 else 0
    ho, rest = perm[:n_ho], perm[n_ho:]

    def centred(rows):
        a, b = A64[rows], B64[rows]
        if not center:
            return a, b, torch.zeros(A64.shape[1], dtype=torch.float64), torch.zeros(B64.shape[1], dtype=torch.float64)
        ma, mb = a.mean(0), b.mean(0)
        return a - ma, b - mb, ma, mb

    r2_by_lam: Dict[str, float] = {}
    if lam == "auto":
        n_va = max(2, int(round(rest.numel() * 0.1)))
        va, inner = rest[:n_va], rest[n_va:]
        a, b, ma, mb = centred(inner)
        for l, W in zip(grid, _ridge_path(a, b, grid)):
            r2_by_lam[f"{l:g}"] = _r2((A64[va] - ma) @ W + mb, B64[va])
        lam = float(max(grid, key=lambda l: r2_by_lam[f"{l:g}"]))
    lam = float(lam)
    r2 = float("nan")
    if n_ho >= 2:
        a, b, ma, mb = centred(rest)
        W = _ridge_solve(a, b, lam)
        r2 = _r2((A64[ho] - ma) @ W + mb, B64[ho])
    a, b, ma, mb = centred(torch.arange(n))
    W = _ridge_solve(a, b, lam)
    return LinearMap(W.float(), ma.float(), mb.float(), lam, r2, r2_by_lam)


# ---------------------------------------------------------------------------
# 4. Donor experts
# ---------------------------------------------------------------------------
def _balanced_assign(sim: torch.Tensor, cap: int) -> torch.Tensor:
    """Greedy capacity-constrained assignment: best (item, cluster) pairs first."""
    n, k = sim.shape
    assign = torch.full((n,), -1, dtype=torch.long)
    counts = [0] * k
    done = 0
    for flat in torch.argsort(sim.flatten(), descending=True).tolist():
        i, c = divmod(flat, k)
        if assign[i] >= 0 or counts[c] >= cap:
            continue
        assign[i] = c
        counts[c] += 1
        done += 1
        if done == n:
            break
    return assign


def _balanced_kmeans(profiles: torch.Tensor, k: int, gen: torch.Generator, iters: int = 30) -> List[torch.Tensor]:
    """Split rows of ``profiles`` into ``k`` equal groups by spherical k-means."""
    n = profiles.shape[0]
    if k == 1:
        return [torch.arange(n)]
    cap = math.ceil(n / k)
    X = profiles / profiles.norm(dim=1, keepdim=True).clamp_min(1e-12)
    centers = [X[int(torch.randint(n, (1,), generator=gen))]]
    for _ in range(1, k):  # k-means++ on cosine distance
        dist = (1 - (X @ torch.stack(centers).T).max(dim=1).values).clamp_min(0) ** 2
        probs = dist / dist.sum() if float(dist.sum()) > 0 else torch.full((n,), 1.0 / n)
        centers.append(X[int(torch.multinomial(probs, 1, generator=gen))])
    C = torch.stack(centers)
    assign = None
    for _ in range(iters):
        new = _balanced_assign(X @ C.T, cap)
        if assign is not None and torch.equal(new, assign):
            break
        assign = new
        C = torch.stack([X[assign == c].mean(0) for c in range(k)])
        C = C / C.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return [torch.nonzero(assign == c).flatten() for c in range(k)]


@torch.no_grad()
def _routed_expert_rms(mlp, X: torch.Tensor) -> float:
    """Median per-token rms of the outputs of the experts the router actually picks on X."""
    alive = mlp.expert_alive.bool()
    logits = (X @ mlp.gate.weight.T).masked_fill(~alive.unsqueeze(0), float("-inf"))
    scores = torch.sigmoid(logits) if mlp.score_function == "sigmoid" else torch.softmax(logits, dim=-1)
    sel = (scores + mlp.expert_bias.unsqueeze(0)).masked_fill(~alive.unsqueeze(0), float("-inf"))
    idx = torch.topk(sel, mlp.top_k, dim=-1).indices
    vals = []
    for e in torch.unique(idx).tolist():
        rows = (idx == e).any(dim=1).nonzero().flatten()
        ex = mlp.experts[e]
        x = X[rows]
        y = ex.down_proj(F.silu(ex.gate_proj(x)) * ex.up_proj(x))  # no dropout
        vals.append(y.pow(2).mean(dim=1).sqrt())
    return float(torch.cat(vals).median())


def _swiglu(X: torch.Tensor, G: torch.Tensor, U: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    return (F.silu(X @ G.T) * (X @ U.T)) @ D.T


def _rms(x: torch.Tensor) -> float:
    return float(x.pow(2).mean().sqrt())


@torch.no_grad()
def _route_share(mlp, X: torch.Tensor, row: torch.Tensor, bias: float) -> float:
    """Fraction of tokens whose top-k would include one extra expert with router
    ``row`` awake at ``bias`` (every other slot as it is) -- the load a fully
    woken donor would take on ``X``."""
    alive = mlp.expert_alive.bool()
    logits = torch.cat([(X @ mlp.gate.weight.T).masked_fill(~alive.unsqueeze(0), float("-inf")),
                        (X @ row).unsqueeze(1)], dim=1)
    scores = torch.sigmoid(logits) if mlp.score_function == "sigmoid" else torch.softmax(logits, dim=-1)
    dead = torch.cat([~alive, torch.zeros(1, dtype=torch.bool)])
    bias_all = torch.cat([mlp.expert_bias.float(), torch.tensor([float(bias)])])
    sel = (scores + bias_all.unsqueeze(0)).masked_fill(dead.unsqueeze(0), float("-inf"))
    idx = torch.topk(sel, mlp.top_k, dim=-1).indices
    return float((idx == logits.shape[1] - 1).any(dim=1).float().mean())


def build_donor_experts(model, student_tok, teacher: TeacherSlices, *, src_layer: int, dst_layer: int,
                        n_experts: int, domain_ids: Optional[Set[int]] = None, fit_steps: int = 400,
                        seed: int = 0, max_tokens: Optional[int] = None, lam: Union[float, str] = "auto",
                        center: bool = True, router: str = "centered", lr: float = 3e-3,
                        verbose: bool = True) -> List[Dict[str, Any]]:
    """Carve ``n_experts`` SwiGLU experts for student MoE ``dst_layer`` out of
    teacher MLP ``src_layer`` (see module docstring). Returns one dict per expert
    with ``gate_proj`` (w, H), ``up_proj`` (w, H), ``down_proj`` (H, w), ``router``
    (H,) tensors ready for ``install_experts``, plus its fit statistics; every
    dict also carries the shared ``fit`` receipt (token counts, ridge R2s, time).

    Options beyond the spec (defaults = the measured-better choices; the spec's
    literal recipe is ``lam=1e-3, center=False, router="mean"``):
    ``lam`` relative ridge strength or ``"auto"`` (validated on ``RIDGE_GRID``);
    ``center`` fits P/Q with intercepts, Q's constant reaching the bias-free
    expert through the ones-read ``beta``; ``router`` ``"mean"`` (normalised
    mean ``X_s`` of the group's top-10% tokens) or ``"centered"`` (that mean
    minus the mean ``X_s`` of all tokens, so the row points at what is
    *specific* to those tokens instead of the direction every token shares);
    ``max_tokens`` subsamples matched tokens (deterministic in ``seed``)."""

    t0 = time.time()
    log = (lambda *a: print("[donor]", *a, flush=True)) if verbose else (lambda *a: None)
    if router not in ("mean", "centered"):
        raise ValueError(f"router must be 'mean' or 'centered', got {router!r}")
    mlp = model.layers[dst_layer].mlp
    if not hasattr(mlp, "experts"):
        raise ValueError(f"student layer {dst_layer} is not an MoE layer")
    width = int(mlp.experts[0].gate_proj.out_features)
    gen = torch.Generator().manual_seed(int(seed))

    # 1. paired single-token rows
    s_ids, t_ids = match_tokens(student_tok, teacher)
    n_matched = int(s_ids.numel())
    if max_tokens is not None and n_matched > int(max_tokens):
        pick = torch.randperm(n_matched, generator=gen)[: int(max_tokens)].sort().values
        s_ids, t_ids = s_ids[pick], t_ids[pick]
    n = int(s_ids.numel())
    if n < 50:
        raise ValueError(f"only {n} matched tokens between student and {teacher.name}")
    perm = torch.randperm(n, generator=gen)
    n_ho = max(2, int(round(0.1 * n)))
    ho, tr = perm[:n_ho].sort().values, perm[n_ho:].sort().values
    log(f"{teacher.name} L{src_layer} -> student L{dst_layer}: {n_matched} matched tokens, using {n} "
        f"({tr.numel()} fit / {ho.numel()} held out)")

    X_s, R_s = student_single_token_moe_inputs(model, s_ids, dst_layer, return_residual=True)
    st = teacher_single_token_states(teacher, t_ids, src_layer, keep_layers=[src_layer], stop_before_mlp=True)
    T_res, T_in = st["resid_in"][src_layer], st["mlp_in"][src_layer]
    del st
    t_rows = time.time() - t0

    # 2. maps, fitted on the fit split only. P's intercept is irrelevant (it maps
    # contributions, i.e. differences); Q's intercept c is a constant teacher
    # input that reaches the expert through beta, a read of X_s that is ~1 on
    # every token (it also carries GPT-2's c_fc bias).
    Pm = fit_map(T_res[tr], R_s[tr], lam, center=center, seed=seed)
    Qm = fit_map(X_s[tr], T_in[tr], lam, center=center, seed=seed)
    P, Q = Pm.W, Qm.W                                   # (d, H), (H, d)
    c_in = Qm.intercept if center else torch.zeros(T_in.shape[1])
    beta = _ridge_solve(X_s[tr].double(), torch.ones(tr.numel(), 1, dtype=torch.float64), 1e-3).float()[:, 0]
    ones_rmse = float(((X_s[ho] @ beta - 1) ** 2).mean().sqrt())
    log(f"maps (held-out R2): P teacher-resid->student-resid {Pm.r2:.3f} (lam {Pm.lam:g}), "
        f"Q student-in->teacher-mlp_in {Qm.r2:.3f} (lam {Qm.lam:g}); ones-read rmse {ones_rmse:.4f}")

    # 3. neuron scores, streamed over d_ff
    lay = teacher.layers[src_layer]
    d_ff = teacher.d_ff(src_layer)
    dom = torch.tensor([int(i) in domain_ids for i in s_ids.tolist()], dtype=torch.bool) if domain_ids else None
    dom_rows = tr[dom[tr]] if dom is not None else tr
    domain_fallback = dom_rows.numel() < 16
    if domain_fallback:
        dom_rows = tr
    x_dom = T_in[dom_rows]
    scores = torch.empty(d_ff)
    for a in range(0, d_ff, 2048):
        b = min(d_ff, a + 2048)
        acts = _acts_block(teacher, lay, x_dom, a, b)
        scores[a:b] = acts.abs().mean(0) * (_down_block(teacher, lay, a, b) @ P).norm(dim=1)
    K = width * int(n_experts)
    sel = torch.topk(scores, K).indices
    A = _acts_take(teacher, lay, T_in, sel)             # (n, K)
    U = _down_take(teacher, lay, sel) @ P               # (K, H): mapped write direction per neuron
    t_score = time.time() - t0
    log(f"scored {d_ff} neurons on {dom_rows.numel()} {'domain' if dom is not None and not domain_fallback else 'all'} "
        f"tokens; top {K} carry {float(scores[sel].sum() / scores.sum()):.1%} of the score mass")

    # 4. groups, targets, init, refinement
    groups = _balanced_kmeans(A[tr].T.contiguous(), int(n_experts), gen)
    groups.sort(key=lambda g: -float(scores[sel[g]].sum()))
    target_rms = _routed_expert_rms(mlp, X_s)
    alive = mlp.expert_alive.bool()
    router_norm = float(mlp.gate.weight.detach()[alive].norm(dim=1).median())
    target_bias = float(mlp.expert_bias[alive].median())
    fit_info = {
        "teacher": teacher.name, "arch": teacher.arch, "src_layer": int(src_layer), "dst_layer": int(dst_layer),
        "matched_tokens": n_matched, "tokens_used": n, "fit_tokens": int(tr.numel()), "heldout_tokens": int(ho.numel()),
        "domain_tokens": int(dom.sum()) if dom is not None else None, "domain_fallback_to_all": bool(domain_fallback),
        "ridge_r2": {"P_resid_to_resid": Pm.r2, "Q_in_to_mlp_in": Qm.r2},
        "maps": {"P": Pm.receipt(), "Q": Qm.receipt(), "ones_read_rmse_heldout": ones_rmse},
        "options": {"lam": lam, "center": bool(center), "router": router},
        "d_model": teacher.d_model, "d_ff": d_ff, "selected_neurons": K,
        "score_mass_selected": float(scores[sel].sum() / scores.sum()), "target_rms": target_rms,
        "router_norm": router_norm, "fit_steps": int(fit_steps), "lr": lr, "seed": int(seed),
    }
    experts: List[Dict[str, Any]] = []
    x_mean = X_s[tr].mean(0)
    for g_index, members in enumerate(groups):
        neurons = sel[members]
        Y_raw = A[:, members] @ U[members]              # (n, H) group's contribution in student space
        # Winsorise by token: GPT-2-family residual streams carry a few massive
        # outlier tokens/dims, which otherwise dominate both the scale and the
        # MSE (BioMedLM R2 went to -22 without this). Rows above 4x the median
        # fit-token norm are shrunk to that norm, direction kept.
        row_norm = Y_raw.norm(dim=1)
        cap = 4.0 * float(row_norm[tr].median().clamp_min(1e-12))
        Y_raw = Y_raw * (cap / row_norm.clamp_min(cap)).unsqueeze(1)
        scale = target_rms / max(_rms(Y_raw[tr]), 1e-12)
        Y = Y_raw * scale
        if teacher.arch == "qwen2":
            Wg, Wu = _take_rows(lay["gate_w"], neurons), _take_rows(lay["up_w"], neurons)  # (w, d)
            G = Wg @ Q.T + (Wg @ c_in)[:, None] * beta[None, :]                           # (w, H)
            Up = Wu @ Q.T + (Wu @ c_in)[:, None] * beta[None, :]
        else:
            # gelu(z) ~ z*sigmoid(k z) = silu(k z) / k, so a SwiGLU emulates it with
            # gate = k z and a CONSTANT up of 1/k, read through beta (X_s @ beta ~ 1).
            # (gate = k z, up = z would grow like z^2 and blow up on large inputs.)
            Wf = _take_cols(lay["fc_w"], neurons)                                           # (d, w)
            Z = (Q @ Wf + beta[:, None] * (c_in @ Wf + lay["fc_b"][neurons])[None, :]).T
            G = GELU_SIGMOID_K * Z
            Up = (beta / GELU_SIGMOID_K).unsqueeze(0).expand(Z.shape[0], -1).clone()
        D = (U[members] * scale).T.contiguous()          # (H, w)
        r2_before = _r2(_swiglu(X_s[ho], G, Up, D), Y[ho])
        params = [G.clone().requires_grad_(True), Up.clone().requires_grad_(True), D.clone().requires_grad_(True)]
        opt = torch.optim.Adam(params, lr=lr)
        Xtr, Ytr = X_s[tr], Y[tr]
        first_loss = last_loss = float("nan")
        best_loss, best = float("inf"), None
        with torch.enable_grad():
            for step in range(int(fit_steps)):
                opt.zero_grad(set_to_none=True)
                loss = F.mse_loss(_swiglu(Xtr, *params), Ytr)
                if float(loss.detach()) < best_loss:        # loss of the params *before* this step
                    best_loss, best = float(loss.detach()), [p.detach().clone() for p in params]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                if step == 0:
                    first_loss = float(loss.detach())
                last_loss = float(loss.detach())
        final = [p.detach() for p in params]
        if best is not None and float(F.mse_loss(_swiglu(Xtr, *final), Ytr)) > best_loss:
            final = best                                    # Adam wandered off: keep the best fit seen
        G, Up, D = final
        r2_after = _r2(_swiglu(X_s[ho], G, Up, D), Y[ho])
        r2_train = _r2(_swiglu(Xtr, G, Up, D), Ytr)
        # router row: where this group matters most
        contrib = Y_raw[tr].norm(dim=1)
        top = tr[torch.topk(contrib, max(1, int(round(0.1 * tr.numel())))).indices]
        rows = {"mean": X_s[top].mean(0), "centered": X_s[top].mean(0) - x_mean}
        rows = {k: v / v.norm().clamp_min(1e-12) * router_norm for k, v in rows.items()}
        share = {k: _route_share(mlp, X_s[ho], v, target_bias) for k, v in rows.items()}
        # does the row find the group's own tokens? share of held-out top-10% tokens routed to it
        ho_norm = Y_raw[ho].norm(dim=1)
        ho_top = ho[ho_norm >= ho_norm.quantile(0.9)]
        share_top = {k: _route_share(mlp, X_s[ho_top], v, target_bias) for k, v in rows.items()}
        r = rows[router]
        # Load-matched wake bias: at the alive-median bias some rows would take
        # almost every token (a BioMedLM row routed 95% of held-out tokens), which
        # would hijack the layer once woken. Bisect the bias so the woken expert
        # takes the load of a typical alive expert (top_k / n_alive) on fit
        # tokens, never above the alive median.
        want_share = float(mlp.top_k) / max(1, int(alive.sum()))
        lo, hi = target_bias - 8.0, target_bias
        if _route_share(mlp, X_s[tr], r, hi) <= want_share:
            wake_bias = hi
        else:
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if _route_share(mlp, X_s[tr], r, mid) < want_share else (lo, mid)
            wake_bias = lo
        wake_share_ho = _route_share(mlp, X_s[ho], r, wake_bias)
        # Weak graft: when the refined expert cannot beat the mean on held-out
        # tokens its write is mostly error, so it starts quiet (x0.1) and keeps
        # the teacher-derived read directions; training decides the rest.
        write_scale = 1.0 if r2_after >= 0.05 else 0.1
        if write_scale != 1.0:
            D = D * write_scale
        top_tokens = [student_tok.tokens[int(s_ids[i])] for i in top[:12].tolist()]  # topk is sorted
        log(f"expert {g_index}: {members.numel()} neurons, score share {float(scores[neurons].sum() / scores[sel].sum()):.2f}, "
            f"held-out R2 fold {r2_before:.3f} -> refined {r2_after:.3f} (train {r2_train:.3f}); awake route share "
            f"mean-row {share['mean']:.2f} / centred-row {share['centered']:.2f} (on its top-10% tokens "
            f"{share_top['mean']:.2f} / {share_top['centered']:.2f}); wake bias {wake_bias:.3f} -> held-out share "
            f"{wake_share_ho:.3f} (target {want_share:.3f}); write x{write_scale:g}; top tokens {top_tokens[:8]}")
        experts.append({
            "gate_proj": G.contiguous(), "up_proj": Up.contiguous(), "down_proj": D.contiguous(), "router": r.contiguous(),
            "teacher": teacher.name, "arch": teacher.arch, "src_layer": int(src_layer), "dst_layer": int(dst_layer),
            "group": g_index, "neurons": neurons.tolist(), "score_share": float(scores[neurons].sum() / scores[sel].sum()),
            "target_scale": scale, "r2_before": r2_before, "r2_after": r2_after, "r2_train_after": r2_train,
            "fit_mse_first": first_loss, "fit_mse_last": last_loss, "router_mode": router,
            "awake_route_share": share, "awake_route_share_top_tokens": share_top,
            "wake_bias": wake_bias, "wake_share_heldout": wake_share_ho, "wake_share_target": want_share,
            "write_scale": write_scale,
            "router_top_tokens": top_tokens, "fit": fit_info,
        })
    fit_info["seconds"] = {"rows_and_states": round(t_rows, 2), "maps_and_scores": round(t_score - t_rows, 2),
                           "total": round(time.time() - t0, 2)}
    log(f"done in {fit_info['seconds']['total']:.1f}s")
    return experts


def experts_summary(experts: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """JSON-safe view of ``build_donor_experts`` output (tensors -> norms)."""
    out = []
    for e in experts:
        row = {k: v for k, v in e.items() if not isinstance(v, torch.Tensor) and k not in ("neurons", "fit")}
        row["n_neurons"] = len(e["neurons"])
        row["neurons_head"] = e["neurons"][:16]
        row["weight_norms"] = {k: float(e[k].norm()) for k in ("gate_proj", "up_proj", "down_proj", "router")}
        out.append(row)
    return out


@torch.no_grad()
def install_experts(model, dst_layer: int, experts: Sequence[Dict[str, Any]], slots: List[int],
                    dormant_margin: float = 1.5) -> Dict[str, Any]:
    """Copy donor experts into DEAD slots of ``model.layers[dst_layer].mlp``, dormant.

    Weights and router rows go in, ``expert_bias`` = ``min(alive bias) - dormant_margin``,
    ``expert_alive`` stays 0 -- routing is bit-for-bit unchanged. The receipt is
    what ``archimedes_core.wake_grafted_experts`` reads (``target_bias`` = alive median)."""

    mlp = model.layers[dst_layer].mlp
    slots = [int(s) for s in slots]
    if len(slots) != len(experts):
        raise ValueError(f"{len(experts)} experts for {len(slots)} slots")
    if len(set(slots)) != len(slots):
        raise ValueError("duplicate slots")
    alive = mlp.expert_alive.bool()
    for s in slots:
        if not 0 <= s < alive.numel():
            raise ValueError(f"slot {s} out of range (layer has {alive.numel()})")
        if bool(alive[s]):
            raise ValueError(f"slot {s} of layer {dst_layer} is alive; donor experts only go into dead slots")
    alive_bias = mlp.expert_bias[alive]
    dormant = float(alive_bias.min()) - float(dormant_margin)
    target = float(alive_bias.median())
    placed = []
    for slot, e in zip(slots, experts):
        ex = mlp.experts[slot]
        ex.gate_proj.weight.copy_(e["gate_proj"])
        ex.up_proj.weight.copy_(e["up_proj"])
        ex.down_proj.weight.copy_(e["down_proj"])
        mlp.gate.weight[slot].copy_(e["router"])
        mlp.expert_bias[slot] = dormant
        mlp.expert_alive[slot] = 0
        placed.append({"slot": slot, "teacher": e.get("teacher"), "src_layer": e.get("src_layer"),
                       "group": e.get("group"), "n_neurons": len(e.get("neurons", [])),
                       "r2_before": e.get("r2_before"), "r2_after": e.get("r2_after"),
                       "wake_bias": e.get("wake_bias"), "write_scale": e.get("write_scale"),
                       "router_top_tokens": e.get("router_top_tokens", [])[:8]})
    # per-slot wake targets (load-matched, see build_donor_experts); `target_bias`
    # stays the alive median for archimedes_core.wake_grafted_experts compatibility
    by_slot = {int(slot): float(min(e["wake_bias"], target)) if e.get("wake_bias") is not None else target
               for slot, e in zip(slots, experts)}
    return {"layers": {int(dst_layer): {"slots": slots, "dormant_bias": dormant, "target_bias": target,
                                        "target_bias_by_slot": by_slot,
                                        "placed": placed, "dormant": len(placed), "alive_after": int(mlp.expert_alive.sum()),
                                        "dormant_margin": float(dormant_margin)}}}


@torch.no_grad()
def wake_donor_slots(model, receipt: Dict[str, Any], progress: float) -> Dict[str, float]:
    """Like ``archimedes_core.wake_grafted_experts`` but anneals each slot to its own
    load-matched ``target_bias_by_slot`` (falls back to the layer ``target_bias``)."""

    progress = float(min(1.0, max(0.0, progress)))
    out = {}
    for li, info in receipt["layers"].items():
        mlp = model.layers[int(li)].mlp
        by_slot = {int(k): float(v) for k, v in (info.get("target_bias_by_slot") or {}).items()}
        for slot in info["slots"]:
            target = by_slot.get(int(slot), info["target_bias"])
            mlp.expert_alive[slot] = 1
            mlp.expert_bias[slot] = info["dormant_bias"] + (target - info["dormant_bias"]) * progress
        out[str(li)] = float(mlp.expert_bias[info["slots"]].mean()) if info["slots"] else 0.0
    return out


def merge_install_receipts(*receipts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge several ``install_experts`` receipts (e.g. code + bio into one layer)."""
    merged: Dict[str, Any] = {"layers": {}}
    for rec in receipts:
        for li, info in rec["layers"].items():
            li = int(li)
            if li not in merged["layers"]:
                merged["layers"][li] = {**info, "slots": list(info["slots"]), "placed": list(info["placed"])}
                continue
            cur = merged["layers"][li]
            cur["slots"] += list(info["slots"])
            cur["placed"] += list(info["placed"])
            cur["dormant"] = len(cur["slots"])
            # the later install saw the same alive set; keep the lower dormant bias
            cur["dormant_bias"] = min(cur["dormant_bias"], info["dormant_bias"])
    return merged


# ---------------------------------------------------------------------------
# 5. New vocabulary rows
# ---------------------------------------------------------------------------
@torch.no_grad()
def lifted_embedding_rows(new_token_strings: Sequence[str], student_embed: torch.Tensor, student_tok,
                          teachers: List[TeacherSlices], *, max_pieces: int = 4,
                          lam: Union[float, str] = "auto", center: bool = True,
                          exclude_ids: Optional[Sequence[int]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Embedding rows for new student tokens, lifted from the teachers' tables.

    Per teacher: ridge map teacher embedding -> student embedding fitted on the
    matched single tokens (``fit_map``: intercept + validated strength by default;
    ``exclude_ids`` keeps given student ids out of the fit, for probes on known
    rows); a new token = mean of its teacher pieces' embeddings,
    mapped (a teacher *covers* it when it splits into 1..``max_pieces`` non-special
    pieces). Rows average over covering teachers, then are rescaled to the mean
    norm of the existing student rows. Uncovered rows are returned as zeros and
    listed in ``receipt["uncovered"]`` (the caller picks the fallback)."""

    E_s = student_embed.detach().float()
    n_new = len(new_token_strings)
    H = E_s.shape[1]
    acc = torch.zeros(n_new, H, dtype=torch.float64)
    cover = torch.zeros(n_new, dtype=torch.long)
    receipt: Dict[str, Any] = {"teachers": {}, "n_new": n_new, "max_pieces": max_pieces}
    for T in teachers:
        s_ids, t_ids = match_tokens(student_tok, T)
        if exclude_ids is not None:
            keep = ~torch.isin(s_ids, torch.as_tensor(list(exclude_ids), dtype=torch.long))
            s_ids, t_ids = s_ids[keep], t_ids[keep]
        M = fit_map(_take_rows(T.embed, t_ids), E_s[s_ids], lam, center=center)
        specials = {int(i) for i in T.tokenizer.get_added_tokens_decoder().keys()}
        enc = T.tokenizer.encode_batch([str(s) for s in new_token_strings], add_special_tokens=False)
        pieces: List[Optional[List[int]]] = []
        for e in enc:
            ok = 1 <= len(e.ids) <= max_pieces and not any(int(i) in specials for i in e.ids)
            pieces.append(list(e.ids) if ok else None)
        need = sorted({i for p in pieces if p for i in p})
        covered = sum(p is not None for p in pieces)
        if need:
            rows = _take_rows(T.embed, torch.tensor(need))
            pos = {t: k for k, t in enumerate(need)}
            for w, p in enumerate(pieces):
                if p is None:
                    continue
                v = M(rows[[pos[i] for i in p]].mean(0, keepdim=True))[0]
                acc[w] += v.double()
                cover[w] += 1
        receipt["teachers"][T.name] = {"matched_tokens": int(s_ids.numel()), "map_r2_heldout": M.r2,
                                       "map_lam_rel": M.lam, "covered": covered}
    target_norm = float(E_s.norm(dim=1).mean())
    rows = torch.zeros(n_new, H)
    have = cover > 0
    if bool(have.any()):
        mean = (acc[have] / cover[have].unsqueeze(1)).float()
        rows[have] = mean / mean.norm(dim=1, keepdim=True).clamp_min(1e-12) * target_norm
    receipt.update({"covered": int(have.sum()), "uncovered": torch.nonzero(~have).flatten().tolist(),
                    "target_norm": target_norm, "teachers_per_row_mean": float(cover.float().mean()) if n_new else 0.0})
    return rows, receipt


__all__ = [
    "LazyTensor", "TeacherSlices", "load_qwen_slices", "load_biomedlm_slices", "match_tokens",
    "teacher_single_token_states", "teacher_mlp_acts", "student_single_token_moe_inputs", "ridge", "fit_map",
    "LinearMap", "RIDGE_GRID",
    "build_donor_experts", "experts_summary", "install_experts", "merge_install_receipts", "lifted_embedding_rows",
]
