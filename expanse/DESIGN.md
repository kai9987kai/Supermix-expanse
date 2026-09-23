# Supermix Expanse — design spec (v1)

Expanse = **Archimedes final** (47.6M, v93 trunk + v87 experts + FlyCore + OmniCore v48/v38)
**+ Omni Collective v7 Frontier** (77.5M, frozen branch + intent/domain distillation)
**+ Qwen2.5-Coder-7B-Instruct** (code teacher; layer-0/1 MLP graft into new MoE experts; embedding-lifted vocab)
**+ BioMedLM 2.7B** (biomedical teacher; same graft recipe)
**+ the full Janelia male-CNS v1.0 connectome** (all 11,751 cell types, all 3,830,931 type→type edges, fixed measured weights)

Qwen3-Coder has no 7B/8B release; Qwen2.5-Coder-7B-Instruct is the official Qwen coder at that size (user asked for "Qwen3-Coder 7B/8B"; this substitution is disclosed in the README and receipt).

Hardware: Windows 11 ARM64 Snapdragon X Plus, 8 cores, 16 GB RAM, **no CUDA**. torch 2.11 CPU. **bf16 matmul is ~1 GFLOP/s on this CPU — never compute in bf16; always fp32.** fp32 dense ≈110 GFLOP/s; CSR spmm ≈13 GFLOP/s.
While developing/testing: `torch.set_num_threads(2)` unless timing something, keep peak RAM < 3 GB per process, never run two heavy jobs at once.

## Paths

| name | path |
|---|---|
| EXP | `C:/Users/kai99/Desktop/supermix-expanse` |
| REPO (read-only) | `EXP/supermix-archimedes` — import from `REPO/archimedes/src` (it adds `champion/` to sys.path itself) |
| code | `EXP/expanse/src/*.py`, CLIs `EXP/expanse/{build_expanse,train_expanse,eval_expanse,make_teacher_data}.py`, tests `EXP/expanse/tests/` |
| data | `EXP/expanse/data/` |
| checkpoints | `EXP/expanse/checkpoints/` |
| Archimedes final | `EXP/external/base/supermix_archimedes.pt` (schema `supermix-archimedes-v1`, load with `archimedes_core.load_archimedes`) |
| Omni v7 | `EXP/external/omni_v7/` (`omni_collective_v7_frontier.pth`, `..._meta.json`, model .py files) |
| Qwen slices | `EXP/external/teachers/qwen2.5-coder-7b-instruct/graft_slices.safetensors` (bf16: `model.embed_tokens.weight` (152064,3584); for L in 0,1: `model.layers.L.{input_layernorm,post_attention_layernorm}.weight`, `self_attn.v_proj.{weight,bias}` (512,3584)/(512), `self_attn.o_proj.weight` (3584,3584), `mlp.{gate,up}_proj.weight` (18944,3584), `mlp.down_proj.weight` (3584,18944)) + `tokenizer.json`, `config.json` (rms_norm_eps 1e-6, 28 heads, 4 kv heads, head_dim 128) |
| BioMedLM | `EXP/external/teachers/biomedlm/` HF dir: bf16 safetensors shards + `model.safetensors.index.json`, `config.json` (gpt2, n_embd 2560, n_layer 32, n_head 20, vocab 28896, gelu_new, ln eps 1e-5; the 1/(layer+1) attention scaling is ALREADY FOLDED into c_attn q-slice and the flag set false), `tokenizer.json`. GPT-2 Conv1D weights are (in, out). Tied lm_head. |
| Coder GGUF | `EXP/external/teachers/gguf/qwen2.5-coder-7b-instruct-q4_k_m.gguf` |
| BioMedLM GGUF | `EXP/external/teachers/gguf/biomedlm-q8_0.gguf` (produced later by the conversion step) |
| llama.cpp | `EXP/external/llama.cpp/llama-server.exe` (b11115, win-cpu-arm64) |
| connectome | copy of `C:/Users/kai99/Desktop/New folder (9)/Supermix/datasets/v91_malecns/malecns_types.npz` → `EXP/expanse/data/malecns_types.npz` (arrays: `type_names`,`superclass`,`cell_class`,`nt` (object, 11751), `sign` int8, `sign_is_modulatory` bool, `n_neurons` int32, `has_flywire`,`has_manc` bool, `pre`,`post` int32 (3,830,931), `weight` int64 synapse counts). Never modify anything under `New folder (9)`. |

## Trunk facts (Archimedes final)

hidden 320, 6 layers, vocab 9451 (word tokenizer `mimomix_text.WordTokenizer`, `digit_tokens=True`; pieces regex `\s*[A-Za-z]+(?:'[A-Za-z]+)?|\s*\d|\s*[^\sA-Za-z\d]|\s+` — leading whitespace is part of a token, e.g. `" add"`, `"\n    return"`). Specials PAD0 BOS1 EOS2 UNK3 USER4 ASSISTANT5; `encode_turn(user, assistant)` → `[1,4]+enc(user)+[5]+enc(assistant)+[2]`, returns (ids, prompt_len). Tied embeddings (`embed_tokens.weight is lm_head.weight`). MoE on layers 1–4: `model.layers[L].mlp` is `SparseMoEFeedForward` with 72 slots (`experts[i]` = `DenseFeedForward` SwiGLU 320→96→320, no biases), `gate` Linear(320,72), buffers `expert_bias` (72), `expert_alive` uint8 (72, all 1 in final). Selection = softmax score + expert_bias; dead slots masked to −inf. Graft hook: `ArchimedesModel.__init__` registers `self.layers[2].register_forward_hook(self._graft_hook)`; `_graft_hook(module, inputs, output)` receives `(hidden, present)`. `forward(input_ids, *args, fly_obs=None, omni_features=None, use_fly=True, skip_grafts=False, **kw)` passes the rest to `MiMoMixModel.forward(input_ids, labels, attention_mask, past_key_values, use_cache, thinking_cycles, adaptive_thinking, return_mtp, cache_slack, past_length)`. `OmniCore.featurize(prompts)` → (B,128) prompt features (user text only).

**Known bug fixed in Expanse:** `FlyCore.sense` mean-pools hidden over all T (future tokens + PAD) and broadcasts one write to every position → causal leak (≈3e-3 logits at trained gate) and KV-decode ≠ full forward. Expanse senses **per position**.

## Modules

### `src/connectome_full.py` (agent A)

```python
SUPERCLASS_IN  = {"cb_sensory","vnc_sensory","ol_sensory","sensory_ascending","sensory_descending","visual_projection","ascending_neuron"}   # 1,277 types
SUPERCLASS_OUT = {"descending_neuron","cb_motor","vnc_motor","vnc_efferent","cb_efferent","efferent_ascending","efferent_descending","cb_endocrine","vnc_endocrine"}  # 713 types
def build_full_graph(npz_path, min_input_fraction: float = 0.0, radius: float = 0.9) -> dict
```
Returns numpy: `n`, `post`,`pre` (int32, sorted by post then pre), `value` float32 = `input_fraction(post,pre) * sign[pre] * radius` where input_fraction = synapses / total synapses onto `post` (over ALL edges, before any threshold), `sign` int8, `in_idx`,`out_idx` int64, `type_names`,`superclass`,`nt` (object), and `receipt` dict: n_types, n_edges, synapses_total, synapse_mass_kept, n_in, n_out, sign counts (exc/inh/modulatory types), `abs_spectral_radius` (power iteration on |W|, ≤ radius), `signed_spectral_radius_est`, npz sha256, threshold, radius. Default = **full**: all 3,830,931 edges.

```python
class CSRMatmul(torch.autograd.Function)  # y = W @ x ; grad_x = W^T @ grad_y  (W constant, both CSR prebuilt; no grad to W)
class FullConnectomeCore(nn.Module):
    def __init__(self, hidden_size: int, graph: dict, steps: int = 4)
    def forward(self, hidden: Tensor[B,T,H], token_mask: Optional[BoolTensor[B,T]] = None) -> Tuple[Tensor[B,T,H], dict]
```
* Buffers (persistent, so the checkpoint is self-contained): `crow`,`col` (int32), `val` (fp32) for W and `crow_t`,`col_t`,`val_t` for Wᵀ; `in_idx`,`out_idx` (int64); `sign` (int8). Build the torch sparse CSR tensors lazily from buffers (cache; rebuild after `load_state_dict`/`.to()`).
* Parameters: `in_norm` = `nn.RMSNorm(H)`; `in_proj` Linear(H, n_in, bias=False); `out_proj` Linear(n_out, H, bias=False); per-type `log_gain_in`, `log_gain_out`, `bias` (zeros N); `leak_logit` (zeros N → leak a=0.5); `gate` zeros(H).
* Forward, per token (causal — no mixing across positions): select columns = positions with token_mask (default all), M of them. `U` (N,M) zeros with `U[in_idx] = in_proj(in_norm(h_sel)).T`. `r0 = 0`; for k in range(steps): `drive = exp(log_gain_in)[:,None] * CSRMatmul(W, exp(log_gain_out)[:,None] * r) + U + bias[:,None]`; `r = (1-a)*r + a*relu(drive)` (skip the spmm when r is exactly zero on k=0). Readout `e = r[out_idx].T` (M,n_out), RMS-normalise per row (eps 1e-6), `w = out_proj(e)`, `out = hidden` with `out[sel] = h_sel + gate * w`. Info: mean rate, efferent rms (pre-norm), fraction active, M.
* Must be **exactly** identity when `gate == 0` (bitwise: return `hidden + 0*…` is NOT ok if it changes values; use `hidden.clone()` + index_put of `h_sel + gate*w` — with gate 0 that is h_sel exactly).
* `def rewired_graph(graph, seed=0) -> dict`: degree-preserving null that swaps targets only between edges whose source types share (sign, in/out/other role) and keeps each edge's value with its target (as `malecns_connectome.stratified_rewire` does); used only as an eval ablation.
* Tests (`tests/test_connectome_full.py`): zero-gate exactness, causality (changing token t never changes positions < t), CSRMatmul gradcheck on a tiny random graph (double), receipt numbers on the real npz, timing at M=512 columns with 8 threads (report s/forward and s/backward).

### `src/connectome_text.py` (agent A)
`build_connectome_rows(graph, seed, n_rows) -> List[dict]` — **language distillation of the full connectome**, exact facts from the npz, short rows in the house style (see `supermix-archimedes/corpus/*.jsonl`: `{"user","assistant","domain":"connectome","task"}`; replies one line, ≤ 60 words, end with a short `total N` style tag where numeric). Tasks over ALL types whose name matches `[A-Za-z][A-Za-z0-9_\-]{0,15}` (not just the 1,000 largest as v93 did): `cns_nt` (transmitter + excitatory/inhibitory/modulatory), `cns_superclass`, `cns_top_input` (strongest presynaptic type by synapses, with count), `cns_top_output`, `cns_type_count` (n_neurons), `cns_in_degree` (number of presynaptic types), `cns_path_role` (is it sensory/motor/descending/ascending/intrinsic). Several paraphrased prompt templates per task. Hold out 10% of **types** (not rows) for eval: `split` field "train"/"heldout". Write `EXP/expanse/data/connectome_rows.jsonl` + report json (counts per task, distinct types).

### `src/omni_v7_branch.py` (agent B)
```python
class OmniV7Branch(nn.Module):
    def __init__(self, hidden_size: int, meta: dict)   # meta = the small parts of omni_collective_v7_frontier_meta.json (all keys except response_bank)
    def load_v7(self, pth_path)                        # strict load into self.net (OmniCollectiveNetV4 built from meta dims exactly as OmniCollectiveEngineV7 does); freeze
    def train(self, mode=True)                          # self.net ALWAYS stays in eval()
    def featurize(self, prompts: List[str]) -> Dict[str, Tensor]     # token_ids, word_ids, prompt_features (+ zero image, has_image 0) via omni_collective_model.encode_text / encode_word_hashes / prompt_feature_vector
    @torch.no_grad() def encode(self, feats) -> Tensor[B, 988]      # [fused (960, output of net.response_refiner via forward hook), softmax(intent) 15, softmax(domain) 13]
    def forward(self, hidden, state: Tensor[B,988]) -> Tuple[hidden, info]   # hidden + (gate * to_trunk(bridge_norm(state))).unsqueeze(1)
    def aux_logits(self, h: Tensor[B,H]) -> Tuple[intent_logits, domain_logits]   # small Linear heads (H->15, H->13) for v7 distillation
```
`bridge_norm` LayerNorm(988), `to_trunk` Linear(988,H,bias=False), `gate` zeros(H). The v7 python files live in `EXP/external/omni_v7` (add to sys.path; they need sympy, installed). Prompt-only conditioning ⇒ causal-safe. Test: exact reproduction of `OmniCollectiveEngineV7._run_prompt` intent/domain logits for 3 prompts; zero-gate exactness.

### `src/donor_graft.py` (agent B)
Teacher→student weight grafting. Maths must be fp32 (convert bf16 slices on load).
```python
@dataclass class TeacherSlices: name, arch ('qwen2'|'gpt2'), d_model, embed (V,d) fp32, layers: List[dict], tokenizer (tokenizers.Tokenizer), eps, extra
def load_qwen_slices(safetensors_path, tokenizer_json, config_json) -> TeacherSlices
def load_biomedlm_slices(hf_dir, layers=(0,1)) -> TeacherSlices     # reads only wte, wpe, h.0/h.1 tensors via safetensors.safe_open
def match_tokens(student_tok, teacher) -> Tuple[LongTensor, LongTensor]   # student ids ↔ teacher ids where the student token string encodes to EXACTLY one teacher token (use teacher.tokenizer.encode(s, add_special_tokens=False)); skip specials, pure-whitespace tokens
def teacher_single_token_states(teacher, teacher_ids, upto_layer: int) -> dict   # exact single-position forward: attention over one position = o(v(norm(x))) (+biases; GQA repeat for qwen; GPT-2: e = wte + wpe[0], ln_1, c_attn v-slice, c_proj); returns per layer L: mlp_in[L] (post-norm MLP input), mlp_out[L], resid_in[L]
def teacher_mlp_acts(teacher, layer, mlp_in) -> Tensor[n, d_ff]           # qwen2: silu(x@Wg.T)*(x@Wu.T); gpt2: gelu_new(x@c_fc + b)  (chunk over rows)
def student_single_token_moe_inputs(model, student_ids, layer) -> Tensor[n,320]   # forward_hook on model.layers[layer].mlp capturing its input, model run on input_ids of shape (n,1), skip_grafts=True, use_cache=False
def ridge(A, B, lam=1e-3) -> Tuple[W, r2]                                  # fp64 closed form, r2 on a 10% held-out split
def build_donor_experts(model, student_tok, teacher, *, src_layer, dst_layer, n_experts, domain_ids: Optional[set], fit_steps=400, seed=0) -> List[dict]
def install_experts(model, dst_layer, experts, slots: List[int], dormant_margin=1.5) -> dict   # copies weights into DEAD slots, router rows, dormant bias = min(alive bias) - margin, target bias = median(alive bias); returns receipt compatible with archimedes_core.wake_grafted_experts ({"layers": {L: {"slots", "dormant_bias", "target_bias", "placed": [...]}}})
def lifted_embedding_rows(new_token_strings, student_embed, student_tok, teachers: List[TeacherSlices]) -> Tuple[Tensor[n_new,320], dict]   # ridge map teacher-embedding→student-embedding fitted on matched single tokens; a new word = mean of its teacher token embeddings mapped; average over teachers that cover it; rescale rows to the mean norm of existing student rows; report R² per teacher
```
**Expert construction** (per teacher, per (src_layer→dst_layer)):
1. matched tokens → teacher `mlp_in`, neuron acts `A` (n, d_ff), teacher `mlp_out`; student MoE input `X_s` (n,320) for the same tokens (single-token contexts on both sides).
2. Output map `P` (d→320): ridge from teacher `resid_in[src_layer]` to student residual at the MoE input of dst_layer (capture the MoE block's residual stream input too, or use `X_s` if simpler — document which).
3. Rank neurons by domain-weighted contribution: `score_j = mean_{domain tokens} |A_j| * ||W_down[:, j] @ P||` (domain tokens = student ids in `domain_ids`, else all). Take the top `96*n_experts`, split into `n_experts` groups of 96 by k-means on their normalised activation profiles over tokens (balanced assignment).
4. Init each SwiGLU expert: qwen2 → fold (`gate = Wg_sel @ Q`, `up = Wu_sel @ Q`, `down = Pᵀ-mapped Wd_sel`, with `Q` = ridge map student input → teacher `mlp_in`); gpt2 → `gate = 1.702*(c_fc_sel@Q)`, `up = c_fc_sel@Q` scaled so the init output rms matches, `down` as above. Then **refine by local function matching**: Adam (lr 3e-3, `fit_steps`) minimising MSE between `expert(X_s)` and target `Y = (group's teacher contribution) @ P` rescaled so target rms = median rms of the layer's existing routed-expert outputs on `X_s`. Report held-out R² (10% tokens) before/after refinement.
5. Router row = the normalised mean `X_s` of the tokens where that group's contribution norm is in the top 10%, scaled to the median norm of existing `gate.weight` rows.
Budget: ≤ 2.5 GB RAM (Qwen embed is 152064×3584 — never materialise it all in fp32 at once; index the rows you need from the bf16 tensor then convert).

### `src/expanse_core.py` (agent C)
```python
EXP_SCHEMA = "supermix-expanse-v1"
def grow_moe_slots(state_dict, config: MiMoMixConfig, n_new: int, layers=(1,2,3,4)) -> Tuple[state_dict, config]
    # config.moe_spare_experts += n_new ; pad gate.weight rows (zeros), expert_bias (min(alive)-10), expert_alive (0), experts.{72..} (zeros) for every MoE layer
class ExpanseModel(ArchimedesModel):
    def __init__(self, config, fly_config=None, with_omni=True, expanse: dict = None)   # expanse = {"graph": dict from build_full_graph (or None when loading: buffers come from state_dict via graph shapes stored in expanse), "cns_steps": 4, "omni7_meta": {...}, "fly_causal": True}
    def _graft_hook(self, module, inputs, output)   # order: fly (per-position, causal) -> omni v48/v38 -> omni7 bridge -> full connectome core
    def forward(self, input_ids, *args, fly_obs=None, omni_features=None, omni7_state=None, use_fly=True, use_omni7=True, use_cns_full=True, skip_grafts=False, token_mask=None, **kwargs)
    def gate_report(self) -> dict   # adds omni7 / cns_full gates + donor expert slot stats
def fly_forward_causal(fly_core, hidden, fly_obs=None) -> Tuple[hidden, info]
    # obs_t = tanh(sensory_proj(norm(h_t))) per position (B,T,14); brains_forward on (B*T,14); write gate*to_trunk([desc.flatten(1); consensus]) per position.
    # If fly_obs (B,14) given: broadcast as before (no leak). info['obs'] is (B,T,14); aux losses use the obs at the last prompt position.
def save_expanse(path, model, tok, extra, receipt) ; def load_expanse(path) -> (model, tok, payload)   # atomic .tmp + os.replace; payload: schema, base_schema, config, state_dict, tokenizer, extra, expanse (receipt incl. graph metadata needed to rebuild module shapes: n, n_in, n_out, nnz, steps, omni7 meta)
```
The frozen v7 net is included in the state dict (self-contained checkpoint). `token_mask` = `input_ids != PAD` by default so PAD columns are skipped by the connectome core. KV-cache decode must work: with `use_cache`, the hook sees only new positions; all Expanse grafts are per-position or prompt-constant, so no extra cache state is needed — verify decode == full forward (max |Δlogit| < 1e-4) with every graft gate set to small random values.

### `build_expanse.py` (agent C) — Stage 1, structural graft
1. load Archimedes final; record function-preservation baseline logits on 32 corpus rows.
2. grow 16 slots in MoE layers 1 and 2 (8 code + 8 bio experts per layer) → 88 slots there (other MoE layers unchanged unless simpler to grow all — document).
3. donor experts: Qwen L0→student L1, Qwen L1→student L2; BioMedLM L0→L1, BioMedLM L1→L2; installed dormant.
4. attach FullConnectomeCore (full graph, steps 4) and OmniV7Branch (v7 loaded, frozen); FlyCore causal.
5. vocabulary extension: `WordTokenizer.extend(tok, texts, max_new=1500, min_count=3)` over the teacher corpora (`data/code_rows.jsonl`, `data/bio_rows.jsonl`, `data/connectome_rows.jsonl`), new rows from `lifted_embedding_rows` (fallback mean+0.1·std·randn when no teacher covers a word).
6. function preservation check (all new gates 0, donor experts dormant, fly per-position): report max |Δlogit| vs Archimedes on the 32 rows (expected ≠ 0 only because of the fly-causality fix; also report with `use_fly=False` on both → must be < 1e-4).
7. save `checkpoints/supermix_expanse_grafted.pt` + receipt json.

### `train_expanse.py` (agent C) — Stage 2, distil + fine-tune
Based on `supermix-archimedes/archimedes/train_archimedes.py`. Data: replay (`REPO/corpus/{omni,code,math}.jsonl`, 3,743 rows), fly rows **self-distilled** from the frozen original FlyCore (sample obs ~ U-shaped ranges matching `fly_row`, teacher probs = original `fly_core.brains_forward`; the Fly Lab experience log is not available), code teacher rows, bio teacher rows, connectome rows (train split). Dev = 5% per source (connectome dev = heldout types). Losses: model loss; `kd_arch` 0.5 = KL(T=2) to the **frozen Archimedes** on replay rows using **cached top-32 teacher logits** (precomputed once to `data/kd_arch_top32.pt`); `omni7_kd` 0.2 = KL of `aux_logits(h at last prompt position)` to cached v7 intent/domain distributions (all rows); `fly_aux` 1.0 = KL(brains_forward(obs) ‖ teacher probs) + MSE(sense at last prompt position, obs) on fly rows. Param groups: trunk 3e-5 (wd 0.01); grafts 2e-4 (wd 0): fly_core, omni_core bridge, omni7 bridge+aux heads, full_cns params, donor expert slots; new embedding rows and donor router rows get their gradients scaled ×(lr_graft/lr_trunk) by hooks. Frozen: omni v48/v38 encoders, omni7 net. Wake donor experts over progress 0.1→0.5 (`wake_grafted_experts`). Cosine schedule, 5% warmup, clip 1.0. `--max_minutes`, eval every N steps (dev loss per source + gate report), rolling resumable partial checkpoint (model + optimizer + step + rng) overwritten in place, final save `checkpoints/supermix_expanse.pt` + receipt.

### `eval_expanse.py` (agent C)
Compare Archimedes final vs Expanse: per-source dev loss (on rows the model's tokenizer covers without UNK; report coverage), `probe_accuracy`-style exact answers on replay-style held-out generated problems with greedy (non-speculative) decoding, code pass-rate on held-out code prompts (same sandbox as teacher filtering), bio token-F1 vs teacher answers + PubMedQA yes/no accuracy on held-out items, connectome exact-match on held-out types. Ablations on Expanse: full_cns gate 0; full_cns with `rewired_graph` W; omni7 gate 0; donor experts dead; fly off. Writes `checkpoints/eval_report.json` + markdown table.

### `src/teacher_data.py` + `make_teacher_data.py` (agent D)
* `LlamaServer(model_path, port, threads=6, ctx=2048)` context manager: starts `llama-server.exe` (subprocess, no shell), waits for `/health`, `complete(prompt, n_predict, temperature, stop, seed)` via `/completion`, `chat(messages, ...)` via `/v1/chat/completions`, terminates on exit.
* **Code (Qwen2.5-Coder-7B-Instruct)**: ≥ 40 task families of small Python functions with randomised names/arguments and **our own test cases** (e.g. sum of list, count vowels, reverse words, is_prime, gcd, fibonacci n, max/min, remove duplicates keep order, factorial, palindrome, running total, clamp, celsius→fahrenheit, second largest, word lengths, flatten one level, ...). System prompt: answer on ONE line, ≤ 60 words: a short plain-English explanation, then the function as a single line `def name(args): return ...` (no newlines, no imports unless `math`). Verify: extract the `def` line, AST whitelist (no import except math, no open/exec/eval/compile/__import__/globals/getattr/os/sys/subprocess/while-True, no dunder attributes), run in a **subprocess** (`python -I -c`, 3 s timeout, `-X utf8`, empty env except PATH/SYSTEMROOT) against the tests; keep only passing rows. Target ≥ 1,500 verified train rows + 150 held-out prompts (by family-disjoint seeds). Row: `{"user","assistant","domain":"programming","task":"coder_<family>","source":"qwen2.5-coder-7b","split"}`.
* **Bio (BioMedLM, base LM)**: (a) term definitions — mine biomedical terms from BioMedLM's own vocab (single `Ġ`-tokens, alphabetic, ≥ 7 chars, lowercase, NOT tokens of the student vocab and not in a common-English list you build from the student vocab + a small stoplist) and prompt few-shot `"{Term} is"`-style completions (temperature 0.3, ≤ 48 tokens, stop at end of 1–2 sentences); (b) PubMedQA `pqa_labeled` questions (HF dataset `qiaojin/PubMedQA`, config `pqa_labeled`; small) answered as `"yes/no/maybe, because ..."` in one line, keep only rows whose yes/no matches `final_decision`. **Cross-teacher filter**: Qwen2.5-Coder-7B judges each definition "Is this statement accurate? yes/no" (1 token) — keep yes. Target ≥ 1,500 train rows + 150 held-out terms/questions. Row format as above with `"domain":"biomedical"`, `"source":"biomedlm"`.
* BioMedLM GGUF conversion: sparse-checkout `ggml-org/llama.cpp` at tag `b11115` (`convert_hf_to_gguf.py`, `gguf-py/`, needed helpers) into `EXP/external/llama.cpp-src`, run with `--outtype q8_0 --outfile EXP/external/teachers/gguf/biomedlm-q8_0.gguf` on the BioMedLM HF dir; verify with `llama-perplexity` on ~2k tokens of biomedical text (report PPL; expect < 30) and 3 sample completions.
* Replies must fit the student: user ≤ 40 words, assistant ≤ 60 words, one line, ASCII-only (transliterate/drop other chars), no markdown/backticks.
