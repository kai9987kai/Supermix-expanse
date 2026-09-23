"""v93 neurogenesis controller: the model grows, prunes and rewires while it
trains (docs/V93_NEUROGENESIS_TWO_HEMISPHERES.md, design contract D6).

The controller runs inside the trainer's eval branch every ``grow_every``
steps, after the dev eval and before the checkpoint write. It decides from
the statistics the model accumulated during that dev pass (mean module rate,
rate covariance, covariance with the trunk residual, dev expert load) and
from the weights themselves (edge strengths) -- never from wall-clock time
or an unsaved random stream -- so a crash resume that repeats the dev pass
reproduces the same event bit for bit. The one random ingredient, the
symmetry-breaking noise on a newborn expert, comes from a ``torch.Generator``
seeded from ``settings.seed`` and the step, and that seed is written into the
event record.

One event, in this order:

1. **Apoptosis.** Edges whose strength ``softplus(edge_logit)`` sat below
   ``prune_threshold`` on two consecutive events are pruned (grown edges and
   real connectome edges are counted separately); grown modules left with no
   output path at all (no outgoing edge, no efferent tap) are killed, which is
   exact because nothing reads them; experts whose dev load share fell below
   ``expert_dead_load_fraction / n_alive`` on two consecutive events are
   killed, never below two alive experts per layer and never the last alive
   expert that carries load.
2. **Mitosis.** The ``grow_modules`` alive modules with the highest
   ``mean_rate * (1 + out_share)`` are split into free slots (only modules
   that fired during the dev pass and have an output path are candidates).
3. **Synaptogenesis.** The ``grow_edges`` unconnected alive pairs with the
   largest positive rate covariance are opened at ``edge_logit`` (strength
   9.1e-4 at -7); ``ceil(cross_quota * grow_edges)`` of them are reserved for
   cross-hemisphere pairs when the core has two sides.
4. **Taps.** ``grow_taps`` afferent taps go to the modules without one whose
   rate covaries most with the trunk residual norm; ``grow_taps`` efferent
   taps to the modules without one with the highest mean rate.
5. **Expert birth.** Per MoE layer, the highest-load expert is copied into a
   spare slot when its dev load share exceeds ``expert_split_load_fraction /
   n_alive``.

Every parameter slice an event writes has its AdamW moments zeroed in place
(no tensor changes shape, so the optimiser, scheduler, checkpoint, resume
and strict-load machinery are untouched). The witness-batch loss is measured
before and after each event, the event is appended as one JSON line to the
log, and an event whose |delta| exceeds 1e-3 nats is flagged.

Expert load. ``SparseMoEFeedForward.growth_statistics()['mean_load']`` is the
fraction of dev tokens routed to each slot and sums to ``top_k`` (2 for
v89), so the fair share of an expert is ``top_k / n_alive``, not
``1 / n_alive``. The controller therefore normalises the load to a *share*
that sums to 1 over alive experts before comparing it with
``fraction / n_alive``: ``expert_split_load_fraction = 2`` then means "twice
the fair share", which is what D6 says, rather than "the fair share", which
would fire a birth at nearly every event.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))

from mimomix_core import ConnectomeCore, SparseMoEFeedForward  # noqa: E402

__all__ = [
    "GrowthSettings",
    "NeurogenesisController",
    "build_settings_from_args",
    "zero_moments",
    "STATE_SCHEMA",
    "FLAG_THRESHOLD_NATS",
]

STATE_SCHEMA = "supermix-v93-neurogenesis-state-v1"
#: An event that moves the witness-batch loss by more than this is flagged
#: (docs/V93_NEUROGENESIS_TWO_HEMISPHERES.md, "Neurogenesis log").
FLAG_THRESHOLD_NATS = 1e-3
#: Edges (and experts) are pruned on their second consecutive weak reading.
STRIKES_TO_PRUNE = 2


@dataclass
class GrowthSettings:
    """Per-event quotas and thresholds. Every default disables growth, so a
    trainer that never sets ``grow_every`` behaves exactly as v89/v91 did."""

    grow_every: int = 0            # 0 disables; the trainer validates it is a multiple of --eval_every
    grow_modules: int = 0          # module splits per event
    grow_edges: int = 0            # synapses opened per event (>= half cross-hemisphere when the core has two sides)
    grow_taps: int = 0             # afferent taps AND efferent taps opened per event
    grow_experts: bool = False     # allow expert births
    prune_threshold: float = 1e-4  # softplus strength below which an edge is pruned after 2 consecutive events
    expert_dead_load_fraction: float = 0.1    # x (1 / n_alive), for 2 consecutive events -> kill
    #: Deaths per MoE layer per event, weakest first. v89 arrives with ~56
    #: starved experts (routing report) and the first v93 preflight event put
    #: 80 of 260 on first strike; an uncapped second event would re-route
    #: every token those 80 carried at once. Two per layer per event turns
    #: the cull into eight gradual steps whose witness deltas are readable.
    max_expert_kills_per_layer: int = 2
    expert_split_load_fraction: float = 2.0   # x (1 / n_alive) -> birth from that expert
    cross_quota: float = 0.5
    edge_logit: float = -7.0
    expert_noise: float = 0.01
    witness_rows: int = 8
    seed: int = 93


def build_settings_from_args(args: Any) -> GrowthSettings:
    """Settings from the trainer's argparse namespace.

    Reads ``grow_every``, ``grow_modules``, ``grow_edges``, ``grow_taps``,
    ``grow_experts``, ``prune_threshold`` and ``witness_rows``; an attribute
    the namespace lacks keeps its default, so an older trainer (or a test
    namespace) builds a controller that never grows. The remaining fields are
    deliberately not read from ``args``: ``seed`` in particular would
    otherwise pick up the run's ``--seed`` and silently couple the expert
    noise to it.
    """

    defaults = GrowthSettings()
    values: Dict[str, Any] = {}
    for name in ("grow_every", "grow_modules", "grow_edges", "grow_taps", "prune_threshold", "witness_rows"):
        value = getattr(args, name, None)
        if value is not None:
            values[name] = type(getattr(defaults, name))(value)
    grow_experts = getattr(args, "grow_experts", None)
    if grow_experts is not None:
        values["grow_experts"] = bool(grow_experts)
    return GrowthSettings(**values)


IndexSpec = Union[None, torch.Tensor, Tuple[int, Union[torch.Tensor, Sequence[int]]]]


def zero_moments(optimiser: Optional[torch.optim.Optimizer], param: Optional[torch.Tensor], index_spec: IndexSpec = None) -> int:
    """Zero the AdamW moments (``exp_avg``, ``exp_avg_sq``) of ``param`` on the
    slice ``index_spec`` describes, in place, and return how many moment
    entries were zeroed (summed over the two moments).

    ``index_spec`` is ``None`` for the whole tensor, a boolean tensor of the
    parameter's shape for an entry-wise selection, or ``(dim, index)`` for
    whole slices along ``dim`` (``index`` a boolean vector over that dimension
    or a sequence of indices). A parameter the optimiser has no state for yet
    (never had a gradient) is left alone, as is a moment whose shape does not
    match the parameter. ``step`` is kept: the moments restart from zero under
    the run's existing bias correction, which is what a freshly-added
    parameter would see anyway.
    """

    if optimiser is None or param is None:
        return 0
    state = optimiser.state
    if param not in state:  # membership never creates the defaultdict entry
        return 0
    entry = state[param]
    zeroed = 0
    with torch.no_grad():
        for key in ("exp_avg", "exp_avg_sq"):
            moment = entry.get(key)
            if not isinstance(moment, torch.Tensor) or tuple(moment.shape) != tuple(param.shape):
                continue
            if index_spec is None:
                zeroed += moment.numel()
                moment.zero_()
            elif isinstance(index_spec, torch.Tensor):
                selection = index_spec.to(device=moment.device, dtype=torch.bool)
                if tuple(selection.shape) != tuple(moment.shape):
                    raise ValueError(
                        f"boolean index_spec has shape {tuple(selection.shape)}, parameter {tuple(moment.shape)}"
                    )
                count = int(selection.sum())
                if count:
                    moment[selection] = 0.0
                    zeroed += count
            else:
                dim, index = index_spec
                index = torch.as_tensor(index)
                if index.dtype == torch.bool:
                    index = torch.nonzero(index, as_tuple=False).flatten()
                index = index.to(device=moment.device, dtype=torch.long)
                if index.numel():
                    moment.index_fill_(int(dim), index, 0.0)
                    zeroed += int(index.numel()) * (moment.numel() // max(1, int(moment.shape[int(dim)])))
    return zeroed


def _ranked(score: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    """Indices of ``candidates`` ordered by ``score`` descending; ties keep
    index order (stable sort), so the ranking is a pure function of the
    statistics."""

    masked = torch.where(candidates, score.to(torch.float32), torch.full_like(score, float("-inf"), dtype=torch.float32))
    order = torch.argsort(masked, descending=True, stable=True)
    return order[: int(candidates.sum())]


class _Written:
    """Which slices of the core's parameters an event wrote, for moment zeroing."""

    def __init__(self, n: int, device: torch.device):
        self.edges = torch.zeros(n, n, dtype=torch.bool, device=device)
        self.nodes = torch.zeros(n, dtype=torch.bool, device=device)
        self.in_rows = torch.zeros(n, dtype=torch.bool, device=device)
        self.out_cols = torch.zeros(n, dtype=torch.bool, device=device)

    def split(self, parent: int, child: int) -> None:
        # split_module: child row and column, parent column halved, child
        # bias/leak/read-in rows copied, parent and child read-out columns.
        self.edges[child, :] = True
        self.edges[:, parent] = True
        self.edges[:, child] = True
        self.nodes[child] = True
        self.in_rows[child] = True
        self.out_cols[parent] = True
        self.out_cols[child] = True

    def kill(self, index: int) -> None:
        self.edges[index, :] = True
        self.edges[:, index] = True
        self.nodes[index] = True
        self.in_rows[index] = True
        self.out_cols[index] = True

    def any(self) -> bool:
        return bool(self.edges.any() or self.nodes.any() or self.in_rows.any() or self.out_cols.any())


class NeurogenesisController:
    """Runs D6 events on a live model. See the module docstring.

    ``witness`` is ``(input_ids, labels)`` of a few dev rows, fixed for the
    run (the trainer passes ``dev_x[:witness_rows], dev_y[:witness_rows]``);
    the loss on it before and after every event is the only measurement of
    what the non-exact operations (splits, efferent taps, expert births) did.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        settings: GrowthSettings,
        log_path: str,
        witness: Tuple[torch.Tensor, torch.Tensor],
    ):
        self.model = model
        self.settings = settings
        self.log_path = str(log_path)
        device = self._device()
        input_ids, labels = witness
        self.witness_x = torch.as_tensor(input_ids).to(device=device, dtype=torch.long)
        self.witness_y = torch.as_tensor(labels).to(device=device, dtype=torch.long)
        if self.witness_x.shape != self.witness_y.shape:
            raise ValueError(
                f"witness input_ids {tuple(self.witness_x.shape)} and labels {tuple(self.witness_y.shape)} differ"
            )
        # consecutive-weak counters (the apoptosis memory), restored on resume
        self._edge_weak: Optional[torch.Tensor] = None     # (n, n) uint8, lazily sized to the core
        self._expert_weak: Dict[Tuple[int, int], int] = {}  # (layer, slot) -> strikes
        self.n_events = 0
        self.events: List[Dict[str, Any]] = []
        self.flagged_events: List[int] = []
        self.last_record: Optional[Dict[str, Any]] = None
        #: The slices the last event wrote (bool masks over the core plus the
        #: expert slots per layer); tests and diagnostics read it.
        self.last_written: Dict[str, Any] = {}

    # -- model access ---------------------------------------------------------

    def _device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def core(self) -> Optional[ConnectomeCore]:
        core = getattr(self.model, "cns_core", None)
        return core if isinstance(core, ConnectomeCore) else None

    def _moe_layers(self) -> List[Tuple[int, SparseMoEFeedForward]]:
        """``(layer_index, module)`` for every trunk MoE layer, in depth order."""

        found: List[Tuple[int, SparseMoEFeedForward]] = []
        layers = getattr(self.model, "layers", None)
        if layers is not None:
            for index, layer in enumerate(layers):
                mlp = getattr(layer, "mlp", None)
                if isinstance(mlp, SparseMoEFeedForward):
                    found.append((int(index), mlp))
        return found

    def _all_moes(self) -> List[SparseMoEFeedForward]:
        return [m for m in self.model.modules() if isinstance(m, SparseMoEFeedForward)]

    # -- dev-pass statistics --------------------------------------------------

    def begin_dev_pass(self) -> None:
        """Start accumulating growth statistics (eval-mode forwards only)."""

        core = self.core
        if core is not None:
            core.reset_stats()
            core.collect_stats = True
        for moe in self._all_moes():
            moe.reset_stats()
            moe.collect_stats = True

    def end_dev_pass(self) -> None:
        """Stop accumulating; the statistics stay until the next ``begin``."""

        core = self.core
        if core is not None:
            core.collect_stats = False
        for moe in self._all_moes():
            moe.collect_stats = False

    # -- the witness measurement ---------------------------------------------

    def witness_loss(self) -> float:
        """LM loss on the witness rows in eval mode (MTP off), training mode restored."""

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                out = self.model(self.witness_x, labels=self.witness_y, return_mtp=False)
            loss = out.lm_loss if out.lm_loss is not None else out.loss
            return float(loss)
        finally:
            self.model.train(was_training)

    # -- the event --------------------------------------------------------------

    def maybe_grow(self, step: int, optimiser: Optional[torch.optim.Optimizer]) -> Optional[Dict[str, Any]]:
        """Run one D6 event at ``step`` if it is a growth step, else ``None``."""

        every = int(self.settings.grow_every)
        if every <= 0 or int(step) % every != 0:
            return None
        step = int(step)
        started = time.perf_counter()
        before = self.witness_loss()
        core = self.core
        core_stats = core.growth_statistics() if core is not None else None
        moe_layers = self._moe_layers()
        moe_stats = [(layer, moe, moe.growth_statistics()) for layer, moe in moe_layers]
        written = _Written(core.n_nodes, core.mask.device) if core is not None else None
        expert_written: Dict[int, set] = {}

        # 1. apoptosis
        pruned_grown, pruned_real, pruned_now = 0, 0, None
        modules_killed: List[int] = []
        if core is not None:
            pruned_grown, pruned_real, pruned_now = self._edge_apoptosis(core, written)
            modules_killed = self._module_apoptosis(core, written)
        experts_killed = self._expert_apoptosis(moe_stats, expert_written)
        # 2. mitosis
        modules_split: List[List[int]] = []
        if core is not None and core_stats is not None:
            modules_split = self._mitosis(core, core_stats, step, written)
        # 3. synaptogenesis
        edges_opened = {"count": 0, "by_block": {"LL": 0, "RR": 0, "LR": 0, "RL": 0}}
        if core is not None and core_stats is not None:
            edges_opened = self._synaptogenesis(core, core_stats, step, pruned_now, written)
        # 4. taps
        taps_opened = {"in": 0, "out": 0}
        if core is not None and core_stats is not None:
            taps_opened = self._taps(core, core_stats, written)
        # 5. expert births
        noise_seed = int(self.settings.seed) * 1_000_003 + step
        experts_born = self._expert_births(moe_stats, noise_seed, expert_written)

        after = self.witness_loss()
        delta = after - before
        moments_zeroed = self._zero_written(optimiser, core, written, moe_layers, expert_written)
        self.last_written = {
            "edges": written.edges.clone() if written is not None else None,
            "nodes": written.nodes.clone() if written is not None else None,
            "in_rows": written.in_rows.clone() if written is not None else None,
            "out_cols": written.out_cols.clone() if written is not None else None,
            "experts": {layer: sorted(slots) for layer, slots in expert_written.items()},
        }
        record: Dict[str, Any] = {
            "step": step,
            "witness_loss_before": before,
            "witness_loss_after": after,
            "witness_delta": delta,
            "flagged": bool(abs(delta) > FLAG_THRESHOLD_NATS),
            "modules_split": modules_split,
            "edges_opened": edges_opened,
            "edges_pruned": {"grown": int(pruned_grown), "real": int(pruned_real)},
            "taps_opened": taps_opened,
            "modules_killed": modules_killed,
            "experts_born": experts_born,
            "experts_killed": experts_killed,
            "alive_modules": int(core.alive.sum()) if core is not None else 0,
            "alive_edges": int(core.mask.sum()) if core is not None else 0,
            "alive_experts_per_layer": [moe.alive_count() for _, moe in moe_layers],
            "seconds": round(time.perf_counter() - started, 3),
            # provenance beyond the contract's keys
            "noise_seed": noise_seed,
            "dev_tokens": int(core_stats["tokens"]) if core_stats is not None else 0,
            "dev_batches": [int(stats["batches"]) for _, _, stats in moe_stats],
            "edges_on_first_strike": int((self._edge_weak == 1).sum()) if self._edge_weak is not None else 0,
            "experts_on_first_strike": sum(1 for count in self._expert_weak.values() if count == 1),
            "moments_zeroed": int(moments_zeroed),
            "witness_rows": int(self.witness_x.shape[0]),
        }
        self.n_events += 1
        self.events.append(self._compact(record))
        if record["flagged"]:
            self.flagged_events.append(step)
        self.last_record = record
        self._append_log(record)
        return record

    # -- 1. apoptosis -----------------------------------------------------------

    def _edge_apoptosis(self, core: ConnectomeCore, written: _Written) -> Tuple[int, int, torch.Tensor]:
        """Prune edges weak on two consecutive events. Returns
        ``(grown_pruned, real_pruned, pruned_now)``."""

        n = core.n_nodes
        device = core.mask.device
        if self._edge_weak is None or tuple(self._edge_weak.shape) != (n, n):
            self._edge_weak = torch.zeros(n, n, dtype=torch.uint8, device=device)
        self._edge_weak = self._edge_weak.to(device)
        threshold = float(self.settings.prune_threshold)
        with torch.no_grad():
            strength = F.softplus(core.edge_logit.detach())
            installed = core.mask > 0
            weak_now = installed & (strength < threshold)
            strikes = torch.where(
                weak_now, (self._edge_weak.to(torch.int32) + 1).clamp_max(255), torch.zeros_like(self._edge_weak, dtype=torch.int32)
            ).to(torch.uint8)
            self._edge_weak = strikes
            due = weak_now & (strikes >= STRIKES_TO_PRUNE)
            mask_before = core.mask.clone()
            grown_pruned = real_pruned = 0
            if bool(due.any()):
                # prune_edges removes EVERY installed edge below the threshold,
                # so edges on their first strike are hidden from it by lifting
                # their mask bit for the call and putting it back (mask holds
                # exact 0/1, so the restore is exact and no parameter is touched).
                protect = weak_now & ~due
                if bool(protect.any()):
                    core.mask[protect] = 0.0
                core.prune_edges(threshold, grown_only=False)
                if bool(protect.any()):
                    core.mask[protect] = 1.0
                grown_pruned = int(core.last_event.get("grown_pruned", 0))
                real_pruned = int(core.last_event.get("real_pruned", 0))
            pruned_now = (mask_before > 0) & (core.mask == 0)
            self._edge_weak[pruned_now] = 0
            written.edges |= pruned_now
        return grown_pruned, real_pruned, pruned_now

    def _module_apoptosis(self, core: ConnectomeCore, written: _Written) -> List[int]:
        """Kill grown modules that nothing reads any more (no outgoing edge, no
        efferent tap): exactly inert, so the kill is function-preserving and
        frees the slot for a later split. Real modules are never killed."""

        with torch.no_grad():
            alive = core.alive.bool()
            grown = core.born_step > 0
            no_out_edges = (core.mask > 0).sum(0) == 0
            no_out_tap = core.out_mask == 0
            inert = alive & grown & no_out_edges & no_out_tap
        killed: List[int] = []
        for index in torch.nonzero(inert, as_tuple=False).flatten().tolist():
            if core.kill_module(int(index)):
                killed.append(int(index))
                written.kill(int(index))
        return killed

    @staticmethod
    def _load_share(stats: Dict[str, Any], alive: torch.Tensor) -> torch.Tensor:
        """Dev load as a share summing to 1 over alive experts (see the module
        docstring on why the raw load, which sums to ``top_k``, is not used)."""

        load = stats["mean_load"].to(torch.float32)
        total = float(load[alive].sum())
        if total <= 0:
            return torch.zeros_like(load)
        return torch.where(alive, load / total, torch.zeros_like(load))

    def _expert_apoptosis(
        self, moe_stats: List[Tuple[int, SparseMoEFeedForward, Dict[str, Any]]], expert_written: Dict[int, set]
    ) -> List[List[int]]:
        killed: List[List[int]] = []
        fraction = float(self.settings.expert_dead_load_fraction)
        for layer, moe, stats in moe_stats:
            if int(stats["batches"]) <= 0:
                continue  # no dev evidence this event: counters neither advance nor reset
            alive = moe.expert_alive.bool()
            n_alive = int(alive.sum())
            share = self._load_share(stats, alive)
            threshold = fraction / max(1, n_alive)
            weak_now = alive & (share < threshold)
            for slot in range(int(alive.numel())):
                key = (layer, slot)
                if bool(weak_now[slot]):
                    self._expert_weak[key] = self._expert_weak.get(key, 0) + 1
                else:
                    self._expert_weak.pop(key, None)
            due = [slot for slot in range(int(alive.numel())) if self._expert_weak.get((layer, slot), 0) >= STRIKES_TO_PRUNE]
            if not due:
                continue
            due.sort(key=lambda slot: (float(share[slot]), slot))  # weakest first, index breaks ties
            floor = max(2, int(moe.top_k))
            with_load = int((alive & (share > 0)).sum())
            cap = int(self.settings.max_expert_kills_per_layer)
            killed_here = 0
            for slot in due:
                if cap > 0 and killed_here >= cap:
                    break  # the rest keep their strikes and go next event
                if moe.alive_count() - 1 < floor:
                    break
                if float(share[slot]) > 0 and with_load <= 1:
                    continue  # never the last alive expert that carries load
                if moe.kill_expert(slot):
                    killed.append([layer, slot])
                    killed_here += 1
                    expert_written.setdefault(layer, set()).add(slot)
                    self._expert_weak.pop((layer, slot), None)
                    if float(share[slot]) > 0:
                        with_load -= 1
        return killed

    # -- 2. mitosis ---------------------------------------------------------------

    def _mitosis(self, core: ConnectomeCore, stats: Dict[str, Any], step: int, written: _Written) -> List[List[int]]:
        """Split the top-scoring modules. ``out_share`` is each module's share
        of the row-normalised strength mass (every post-synaptic module's
        input distribution sums to 1; a module's share is how much of all
        those inputs it supplies). A module that fired during the dev pass,
        was alive for it, and has an output path (an outgoing edge or an
        efferent tap) is a candidate; one with no output path is exactly
        inert, so its child would be too, and :meth:`_module_apoptosis`
        would only kill it again at the next event."""

        quota = int(self.settings.grow_modules)
        if quota <= 0 or int(stats["tokens"]) <= 0 or not core.free_slots():
            return []
        with torch.no_grad():
            mean_rate = stats["mean_rate"].to(torch.float32)
            strength = core.weight().detach().abs().to(torch.float32)
            row_sum = strength.sum(1, keepdim=True)
            normalised = torch.where(row_sum > 0, strength / row_sum.clamp_min(1e-12), torch.zeros_like(strength))
            out_mass = normalised.sum(0)
            out_share = out_mass / out_mass.sum().clamp_min(1e-12)
            score = mean_rate * (1.0 + out_share)
            has_output = ((core.mask > 0).sum(0) > 0) | (core.out_mask > 0)
            candidates = core.alive.bool() & stats["alive"].to(core.alive.device) & (mean_rate > 0) & has_output
            ranked = _ranked(score, candidates)[:quota].tolist()
        splits: List[List[int]] = []
        for parent in ranked:
            if not core.free_slots():
                break
            child = core.split_module(int(parent), step)
            if child is None:
                continue
            splits.append([int(parent), int(child)])
            written.split(int(parent), int(child))
        return splits

    # -- 3. synaptogenesis ---------------------------------------------------------

    def _synaptogenesis(
        self,
        core: ConnectomeCore,
        stats: Dict[str, Any],
        step: int,
        pruned_now: Optional[torch.Tensor],
        written: _Written,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {"count": 0, "by_block": {"LL": 0, "RR": 0, "LR": 0, "RL": 0}}
        quota = int(self.settings.grow_edges)
        if quota <= 0 or int(stats["tokens"]) <= 0:
            return result
        n = core.n_nodes
        with torch.no_grad():
            cov = stats["rate_cov"].to(torch.float32)
            alive = core.alive.bool()
            eligible = (
                alive.unsqueeze(1) & alive.unsqueeze(0)
                & (core.mask == 0)
                & ~torch.eye(n, dtype=torch.bool, device=alive.device)
                & (cov > 0)
            )
            if pruned_now is not None:
                eligible &= ~pruned_now
            side = core.hemisphere.to(torch.int64)
            two_sides = bool(((side == 0) & alive).any()) and bool(((side == 1) & alive).any())
            cross = side.unsqueeze(1) != side.unsqueeze(0)
            ranked = _ranked(cov.flatten(), eligible.flatten())  # flat indices, best first
            chosen: List[int] = []
            if two_sides:
                reserve = min(quota, int(math.ceil(float(self.settings.cross_quota) * quota)))
                cross_ranked = ranked[cross.flatten()[ranked]]
                chosen = cross_ranked[:reserve].tolist()
            taken = torch.zeros(n * n, dtype=torch.bool, device=alive.device)
            if chosen:
                taken[torch.tensor(chosen, device=alive.device)] = True
            rest = ranked[~taken[ranked]][: max(0, quota - len(chosen))].tolist()
            chosen.extend(rest)
        for flat in chosen:
            post, pre = divmod(int(flat), n)
            if core.grow_edge(post, pre, step, float(self.settings.edge_logit)):
                result["count"] += 1
                block = str(core.last_event.get("block", "LL"))
                result["by_block"][block] = result["by_block"].get(block, 0) + 1
                written.edges[post, pre] = True
        return result

    # -- 4. taps ---------------------------------------------------------------------

    def _taps(self, core: ConnectomeCore, stats: Dict[str, Any], written: _Written) -> Dict[str, int]:
        result = {"in": 0, "out": 0}
        quota = int(self.settings.grow_taps)
        if quota <= 0 or int(stats["tokens"]) <= 0:
            return result
        with torch.no_grad():
            alive = core.alive.bool()
            in_candidates = alive & (core.in_mask == 0)
            out_candidates = alive & (core.out_mask == 0)
            in_ranked = _ranked(stats["resid_cov"].to(torch.float32), in_candidates)[:quota].tolist()
            out_ranked = _ranked(stats["mean_rate"].to(torch.float32), out_candidates)[:quota].tolist()
        for module in in_ranked:
            if core.open_tap(int(module), "in"):
                result["in"] += 1
                written.in_rows[int(module)] = True
        for module in out_ranked:
            if core.open_tap(int(module), "out"):
                result["out"] += 1
                written.out_cols[int(module)] = True
        return result

    # -- 5. expert births --------------------------------------------------------------

    def _expert_births(
        self,
        moe_stats: List[Tuple[int, SparseMoEFeedForward, Dict[str, Any]]],
        noise_seed: int,
        expert_written: Dict[int, set],
    ) -> List[List[int]]:
        born: List[List[int]] = []
        if not self.settings.grow_experts:
            return born
        fraction = float(self.settings.expert_split_load_fraction)
        noise = float(self.settings.expert_noise)
        generator = torch.Generator().manual_seed(int(noise_seed))
        for layer, moe, stats in moe_stats:
            if int(stats["batches"]) <= 0:
                continue
            alive = moe.expert_alive.bool()
            n_alive = int(alive.sum())
            if n_alive == 0 or n_alive >= moe.n_routed:
                continue  # no spare slot
            share = self._load_share(stats, alive)
            parent = int(torch.argmax(torch.where(alive, share, torch.full_like(share, float("-inf")))))
            if float(share[parent]) <= fraction / n_alive:
                continue
            # birth_expert's own noise draws from the global RNG; an exact copy
            # plus noise from the controller's seeded generator is what makes a
            # resumed run reproduce the same child.
            child = moe.birth_expert(parent, noise=0.0)
            if child is None:
                continue
            with torch.no_grad():
                if noise > 0:
                    source = list(moe.experts[parent].parameters())
                    target = list(moe.experts[child].parameters())
                    for p_src, p_dst in zip(source, target):
                        draw = torch.randn(p_dst.shape, generator=generator, dtype=torch.float32)
                        p_dst.add_((draw * (noise * float(p_src.std()))).to(device=p_dst.device, dtype=p_dst.dtype))
                    row = moe.gate.weight[parent]
                    draw = torch.randn(row.shape, generator=generator, dtype=torch.float32)
                    moe.gate.weight[child].add_((draw * (noise * float(row.std()))).to(device=row.device, dtype=row.dtype))
            born.append([layer, parent, int(child)])
            expert_written.setdefault(layer, set()).add(int(child))
        return born

    # -- optimiser moments ------------------------------------------------------------

    def _zero_written(
        self,
        optimiser: Optional[torch.optim.Optimizer],
        core: Optional[ConnectomeCore],
        written: Optional[_Written],
        moe_layers: List[Tuple[int, SparseMoEFeedForward]],
        expert_written: Dict[int, set],
    ) -> int:
        if optimiser is None:
            return 0
        zeroed = 0
        if core is not None and written is not None and written.any():
            if bool(written.edges.any()):
                zeroed += zero_moments(optimiser, core.edge_logit, written.edges)
            if bool(written.nodes.any()):
                zeroed += zero_moments(optimiser, core.node_bias, written.nodes)
                zeroed += zero_moments(optimiser, core.leak_logit, written.nodes)
            if bool(written.in_rows.any()):
                zeroed += zero_moments(optimiser, core.read_in.weight, (0, written.in_rows))
                for tap in core.extra_read_in:
                    zeroed += zero_moments(optimiser, tap.weight, (0, written.in_rows))
            if bool(written.out_cols.any()):
                zeroed += zero_moments(optimiser, core.read_out.weight, (1, written.out_cols))
                for tap in core.extra_read_out:
                    zeroed += zero_moments(optimiser, tap.weight, (1, written.out_cols))
                if core.to_thinking is not None:
                    zeroed += zero_moments(optimiser, core.to_thinking.weight, (1, written.out_cols))
        by_layer = {layer: moe for layer, moe in moe_layers}
        for layer, slots in expert_written.items():
            moe = by_layer.get(layer)
            if moe is None or not slots:
                continue
            for slot in sorted(slots):
                for parameter in moe.experts[slot].parameters():
                    zeroed += zero_moments(optimiser, parameter, None)
            zeroed += zero_moments(optimiser, moe.gate.weight, (0, sorted(slots)))
        return zeroed

    # -- log, summary, state -----------------------------------------------------------

    def _append_log(self, record: Dict[str, Any]) -> None:
        parent = os.path.dirname(self.log_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    @staticmethod
    def _compact(record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "step": int(record["step"]),
            "witness_loss_before": float(record["witness_loss_before"]),
            "witness_delta": float(record["witness_delta"]),
            "flagged": bool(record["flagged"]),
            "modules_split": len(record["modules_split"]),
            "edges_opened": int(record["edges_opened"]["count"]),
            "edges_opened_by_block": dict(record["edges_opened"]["by_block"]),
            "edges_pruned_grown": int(record["edges_pruned"]["grown"]),
            "edges_pruned_real": int(record["edges_pruned"]["real"]),
            "taps_in": int(record["taps_opened"]["in"]),
            "taps_out": int(record["taps_opened"]["out"]),
            "modules_killed": len(record["modules_killed"]),
            "experts_born": len(record["experts_born"]),
            "experts_killed": len(record["experts_killed"]),
            "alive_modules": int(record["alive_modules"]),
            "alive_edges": int(record["alive_edges"]),
            "alive_experts_per_layer": list(record["alive_experts_per_layer"]),
            "seconds": float(record["seconds"]),
        }

    def summary(self) -> Dict[str, Any]:
        """The receipt block: settings, event count, totals, flagged steps and
        the compact per-event list."""

        totals = {
            "modules_split": 0, "edges_opened": 0,
            "edges_opened_by_block": {"LL": 0, "RR": 0, "LR": 0, "RL": 0},
            "edges_pruned_grown": 0, "edges_pruned_real": 0,
            "taps_in": 0, "taps_out": 0, "modules_killed": 0,
            "experts_born": 0, "experts_killed": 0, "seconds": 0.0,
        }
        for event in self.events:
            for key in totals:
                if key == "edges_opened_by_block":
                    for block, count in event.get(key, {}).items():
                        totals[key][block] = totals[key].get(block, 0) + int(count)
                else:
                    totals[key] += event.get(key, 0)
        totals["seconds"] = round(float(totals["seconds"]), 3)
        last = self.events[-1] if self.events else None
        return {
            "settings": asdict(self.settings),
            "log_path": self.log_path,
            "n_events": int(self.n_events),
            "totals": totals,
            "flagged_events": list(self.flagged_events),
            "flag_threshold_nats": FLAG_THRESHOLD_NATS,
            "alive_modules": int(last["alive_modules"]) if last else None,
            "alive_edges": int(last["alive_edges"]) if last else None,
            "alive_experts_per_layer": list(last["alive_experts_per_layer"]) if last else None,
            "events": [dict(event) for event in self.events],
        }

    def state_dict(self) -> Dict[str, Any]:
        """The apoptosis memory (consecutive-weak counters) plus the event
        history, all JSON-safe, for ``extra['neurogenesis_state']``."""

        edge_weak: List[List[int]] = []
        if self._edge_weak is not None:
            entries = torch.nonzero(self._edge_weak > 0, as_tuple=False)
            for post, pre in entries.tolist():
                edge_weak.append([int(post), int(pre), int(self._edge_weak[post, pre])])
        return {
            "schema": STATE_SCHEMA,
            "seed": int(self.settings.seed),
            "n_events": int(self.n_events),
            "edge_weak": edge_weak,
            "expert_weak": [[int(layer), int(slot), int(count)] for (layer, slot), count in sorted(self._expert_weak.items())],
            "flagged_events": list(self.flagged_events),
            "events": [dict(event) for event in self.events],
        }

    def load_state_dict(self, state: Optional[Dict[str, Any]]) -> None:
        if not state:
            return
        core = self.core
        self._edge_weak = None
        if core is not None:
            n = core.n_nodes
            weak = torch.zeros(n, n, dtype=torch.uint8, device=core.mask.device)
            for post, pre, count in state.get("edge_weak", []):
                if 0 <= int(post) < n and 0 <= int(pre) < n:
                    weak[int(post), int(pre)] = min(255, int(count))
            self._edge_weak = weak
        self._expert_weak = {
            (int(layer), int(slot)): int(count) for layer, slot, count in state.get("expert_weak", [])
        }
        self.n_events = int(state.get("n_events", 0))
        self.events = [dict(event) for event in state.get("events", [])]
        self.flagged_events = [int(step) for step in state.get("flagged_events", [])]
