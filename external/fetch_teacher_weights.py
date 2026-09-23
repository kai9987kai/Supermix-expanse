"""Fetch teacher weights by HTTP range reads, without storing the full fp32 originals.

biomedlm  stanford-crfm/BioMedLM ships only a 10.7 GB fp32 zip-pickle. Each tensor's
          storage is read by range out of the remote zip, cast to bf16 and written as
          safetensors shards into teachers/biomedlm (next to its config/tokenizer).
          GPT-2's scale_attn_by_inverse_layer_idx (logits / (layer+1)) is folded into
          the query slice of c_attn and the flag is switched off in config.json, so the
          model is numerically the same in transformers and convertible by llama.cpp,
          whose gpt2 graph does not implement that flag.
qwen      Qwen/Qwen2.5-Coder-7B-Instruct: only the tensors the expert graft needs
          (embeddings, layers 0-1 norms, v/o attention, MLP) -> one safetensors file.

Usage: python fetch_teacher_weights.py biomedlm|qwen
"""
import io, json, os, pickle, struct, sys, time, zipfile

import numpy as np
import torch
from huggingface_hub import HfFileSystem
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.abspath(__file__))
fs = HfFileSystem()


class _Storage:
    def __init__(self, key, dtype, numel):
        self.key, self.dtype, self.numel = key, dtype, numel


_DT = {"FloatStorage": (np.float32, 4), "HalfStorage": (np.float16, 2), "BFloat16Storage": (None, 2),
       "LongStorage": (np.int64, 8), "IntStorage": (np.int32, 4), "BoolStorage": (np.bool_, 1),
       "ByteStorage": (np.uint8, 1), "UntypedStorage": (np.uint8, 1)}


def _rebuild(storage, offset, size, stride, *_):
    return {"storage": storage, "offset": offset, "size": tuple(size), "stride": tuple(stride)}


class _Unpickler(pickle.Unpickler):
    def find_class(self, mod, name):
        if name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return _rebuild
        if mod == "collections" and name == "OrderedDict":
            import collections
            return collections.OrderedDict
        if name.endswith("Storage"):
            return name
        if mod == "torch._utils" and name == "_rebuild_parameter":
            return lambda data, requires_grad, hooks: data
        return super().find_class(mod, name)

    def persistent_load(self, pid):
        # ('storage', storage_type, key, location, numel)
        _, stype, key, _loc, numel = pid
        return _Storage(key, stype if isinstance(stype, str) else getattr(stype, "__name__", str(stype)), numel)


def biomedlm():
    dest = os.path.join(ROOT, "teachers", "biomedlm")
    os.makedirs(dest, exist_ok=True)
    done_path = os.path.join(dest, "fetch.receipt.json")
    if os.path.exists(done_path):
        print("already fetched:", done_path); return
    f = fs.open("stanford-crfm/BioMedLM/pytorch_model.bin", "rb", block_size=64 * 2**20)
    zf = zipfile.ZipFile(f)
    prefix = zf.namelist()[0].split("/")[0]
    meta = _Unpickler(io.BytesIO(zf.read(f"{prefix}/data.pkl"))).load()
    cfg = json.load(open(os.path.join(dest, "config.json")))
    assert cfg.get("scale_attn_by_inverse_layer_idx") in (True, False)
    fold = bool(cfg.get("scale_attn_by_inverse_layer_idx"))
    n_embd = cfg["n_embd"]
    keys = [k for k in meta if not (k.endswith(".attn.bias") or k.endswith(".attn.masked_bias"))]
    skipped = [k for k in meta if k not in keys]
    shard, shard_bytes, shards, index, t0 = {}, 0, [], {}, time.time()
    cache = {}

    def flush():
        nonlocal shard, shard_bytes
        if not shard:
            return
        name = f"model-{len(shards)+1:05d}.safetensors"
        save_file(shard, os.path.join(dest, name), metadata={"format": "pt"})
        for k in shard:
            index[k] = name
        shards.append(name); print(f"  wrote {name} ({shard_bytes/2**30:.2f} GiB) {time.time()-t0:.0f}s", flush=True)
        shard, shard_bytes = {}, 0

    for i, k in enumerate(keys):
        m = meta[k]; st = m["storage"]
        np_dt, isz = _DT[st.dtype]
        if st.key not in cache:
            raw = zf.read(f"{prefix}/data/{st.key}")
            cache = {st.key: raw}          # storages are not shared except tied wte/lm_head
        raw = cache[st.key]
        n = int(np.prod(m["size"])) if m["size"] else 1
        arr = np.frombuffer(raw, dtype=np_dt, count=n, offset=m["offset"] * isz).reshape(m["size"])
        exp_stride = tuple(int(np.prod(m["size"][j + 1:])) for j in range(len(m["size"])))
        assert m["stride"] == exp_stride, (k, m["stride"], exp_stride)
        t = torch.from_numpy(arr.copy()).float()
        if fold and k.endswith("attn.c_attn.weight"):
            layer = int(k.split(".h.")[1].split(".")[0])
            t[:, :n_embd] /= float(layer + 1)          # Conv1D weight is (in, 3*out): q is the first slice
        if fold and k.endswith("attn.c_attn.bias"):
            layer = int(k.split(".h.")[1].split(".")[0])
            t[:n_embd] /= float(layer + 1)
        if k == "lm_head.weight" and "transformer.wte.weight" in meta and \
                meta["transformer.wte.weight"]["storage"].key == st.key:
            continue                         # tied; tie_word_embeddings handles it
        shard[k] = t.to(torch.bfloat16).contiguous(); shard_bytes += shard[k].numel() * 2
        print(f"[{i+1}/{len(keys)}] {k} {tuple(t.shape)}", flush=True)
        if shard_bytes > 2 * 2**30:
            flush()
    flush()
    json.dump({"metadata": {"total_size": 0}, "weight_map": index},
              open(os.path.join(dest, "model.safetensors.index.json"), "w"), indent=1)
    if fold:
        cfg["scale_attn_by_inverse_layer_idx"] = False
        cfg["_expanse_note"] = "attention 1/(layer_idx+1) scaling folded into c_attn query weights/bias"
    cfg["torch_dtype"] = "bfloat16"
    json.dump(cfg, open(os.path.join(dest, "config.json"), "w"), indent=2)
    json.dump({"source": "stanford-crfm/BioMedLM/pytorch_model.bin", "tensors": len(index), "skipped_buffers": skipped,
               "folded_inverse_layer_scaling": fold, "dtype": "bfloat16", "shards": shards,
               "seconds": round(time.time() - t0)}, open(done_path, "w"), indent=2)
    print("DONE biomedlm", flush=True)


def qwen():
    repo = "Qwen/Qwen2.5-Coder-7B-Instruct"
    dest = os.path.join(ROOT, "teachers", "qwen2.5-coder-7b-instruct", "graft_slices.safetensors")
    if os.path.exists(dest):
        print("already fetched:", dest); return
    idx = json.load(open(os.path.join(ROOT, "teachers", "qwen2.5-coder-7b-instruct", "model.safetensors.index.json")))
    want = ["model.embed_tokens.weight"]
    for L in (0, 1):
        want += [f"model.layers.{L}.input_layernorm.weight", f"model.layers.{L}.post_attention_layernorm.weight",
                 f"model.layers.{L}.self_attn.v_proj.weight", f"model.layers.{L}.self_attn.v_proj.bias",
                 f"model.layers.{L}.self_attn.o_proj.weight",
                 f"model.layers.{L}.mlp.gate_proj.weight", f"model.layers.{L}.mlp.up_proj.weight",
                 f"model.layers.{L}.mlp.down_proj.weight"]
    out, headers = {}, {}
    for k in want:
        shard = idx["weight_map"][k]
        g = fs.open(f"{repo}/{shard}", "rb", block_size=64 * 2**20)
        if shard not in headers:
            n = struct.unpack("<Q", g.read(8))[0]; headers[shard] = (8 + n, json.loads(g.read(n)))
        base, hdr = headers[shard]
        info = hdr[k]; a, b = info["data_offsets"]
        g.seek(base + a); raw = g.read(b - a)
        assert info["dtype"] == "BF16", info
        t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(info["shape"]).clone()
        out[k] = t; print(k, tuple(t.shape), flush=True)
    save_file(out, dest, metadata={"source": repo, "note": "graft slices only"})
    print("DONE qwen", flush=True)


if __name__ == "__main__":
    {"biomedlm": biomedlm, "qwen": qwen}[sys.argv[1]]()
