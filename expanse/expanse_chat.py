"""Local chat UI for Supermix Expanse (Gradio).

    python expanse/expanse_chat.py [--checkpoint PATH] [--port 7861] [--threads 6]

Each message is answered on its own: Expanse was trained on single-turn rows
with a 128-token context, so earlier turns are not fed back in. The side panel
switches individual grafts off at inference (male-CNS core, Omni v7 branch,
FlyCore, the Qwen/BioMedLM donor experts) and can show Archimedes' answer to the
same prompt for comparison. Decoding is greedy by default (temperature 0), the
setting the model was trained and evaluated for.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import expanse_core as ec  # noqa: E402
from expanse_core import text_utils  # noqa: E402

LOCK = threading.Lock()
CTX = 128


@contextmanager
def gate_closed(module):
    """Temporarily close a graft's residual gate (exact 'off')."""
    if module is None:
        yield
        return
    old = module.gate.detach().clone()
    with torch.no_grad():
        module.gate.zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            module.gate.copy_(old)


@contextmanager
def donors_off(model):
    slots = model.donor_slots()
    alive = {li: model.layers[li].mlp.expert_alive[s].clone() for li, s in slots.items()}
    model.set_donor_alive(False)
    try:
        yield
    finally:
        for li, s in slots.items():
            model.layers[li].mlp.expert_alive[s] = alive[li]


@torch.no_grad()
def stream_reply(model, tok, prompt: str, max_new: int, temperature: float, is_expanse: bool, use_fly: bool = True):
    """Yield the reply text as it grows (KV-cached decode, same kwargs every step)."""
    ids, _ = tok.encode_turn(prompt, None)
    ids = ids[-(CTX - 8):]
    max_new = max(1, min(int(max_new), CTX - len(ids)))
    kw = {"omni_features": ec.arch.OmniCore.featurize([prompt]), "use_fly": use_fly}
    if is_expanse and model.omni7 is not None:
        kw["omni7_state"] = model.omni7_state_for([prompt])
    model.eval()
    x = torch.tensor([ids])
    out = model(x, use_cache=True, return_mtp=False, past_length=0, **kw)
    past, pos, emitted = out.past_key_values, len(ids), []
    gen = torch.Generator().manual_seed(0)
    for _ in range(max_new):
        logits = out.logits[0, -1].float()
        if temperature > 0:
            probs = torch.softmax(logits / temperature, -1)
            token = int(torch.multinomial(probs, 1, generator=gen))
        else:
            token = int(logits.argmax())
        if token == text_utils.EOS:
            break
        emitted.append(token)
        yield tok.decode(emitted).strip()
        out = model(torch.tensor([[token]]), past_key_values=past, use_cache=True, return_mtp=False,
                    past_length=pos, **kw)
        past, pos = out.past_key_values, pos + 1
    if not emitted:
        yield ""


def build_app(E, tokE, A, tokA, receipt):
    import gradio as gr

    def respond(message, history, max_new, temperature, cns_on, omni7_on, fly_on, donors_on, compare):
        message = (message or "").strip()
        if not message:
            yield "Type a question first."
            return
        with LOCK:
            t0 = time.time()
            with ExitStack() as st:
                if not cns_on:
                    st.enter_context(gate_closed(E.cns_full))
                if not omni7_on:
                    st.enter_context(gate_closed(E.omni7))
                if not donors_on:
                    st.enter_context(donors_off(E))
                text = ""
                for text in stream_reply(E, tokE, message, max_new, temperature, True, use_fly=fly_on):
                    yield text
            off = [n for n, on in (("CNS core", cns_on), ("Omni v7", omni7_on), ("FlyCore", fly_on),
                                   ("donor experts", donors_on)) if not on]
            footer = f"\n\n<sub>Expanse · {time.time() - t0:.1f}s" + (f" · off: {', '.join(off)}" if off else "") + "</sub>"
            final = (text or "(empty reply)") + footer
            yield final
            if compare and A is not None:
                t1 = time.time()
                a_text = ""
                for a_text in stream_reply(A, tokA, message, max_new, temperature, False):
                    yield final + "\n\n**Archimedes:** " + a_text
                yield final + "\n\n**Archimedes:** " + (a_text or "(empty reply)") + f"\n<sub>Archimedes · {time.time() - t1:.1f}s</sub>"

    tr = (receipt or {}).get("training", {})
    fin = tr.get("final", {})
    desc = (
        "Supermix Expanse: Archimedes + Omni v7 + Qwen2.5-Coder-7B + BioMedLM + the full male-CNS connectome "
        f"(132.6M params, {tr.get('steps', '?')} training steps). Single-turn: each message is answered on its own, "
        "short answers in the Supermix house style (128-token context). "
        f"Final dev loss: replay {fin.get('dev_replay', float('nan')):.3f}, code {fin.get('dev_code', float('nan')):.3f}, "
        f"bio {fin.get('dev_bio', float('nan')):.3f}, connectome {fin.get('dev_connectome', float('nan')):.3f}."
    )
    examples = [
        ["Write a Python function that returns the sum of a list."],
        ["Python: r = sum(range(1, 6)) What is r?"],
        ["What is a hepatocyte?"],
        ["Which neurotransmitter does the cell type LC4 use in the male CNS?"],
        ["whats the impulse from 40 N acting for 6 s"],
        ["What is the strongest input to the cell type EPG in the male CNS?"],
    ]
    return gr.ChatInterface(
        respond,
        title="Supermix Expanse",
        description=desc,
        examples=examples,
        cache_examples=False,
        additional_inputs=[
            gr.Slider(8, 120, value=80, step=1, label="Max new tokens"),
            gr.Slider(0.0, 1.2, value=0.0, step=0.05, label="Temperature (0 = greedy)"),
            gr.Checkbox(True, label="Male CNS connectome core"),
            gr.Checkbox(True, label="Omni v7 branch"),
            gr.Checkbox(True, label="FlyCore"),
            gr.Checkbox(True, label="Qwen / BioMedLM donor experts"),
            gr.Checkbox(False, label="Also answer with Archimedes (comparison)"),
        ],
        additional_inputs_accordion=gr.Accordion("Model controls", open=False),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ec.PATHS["final"]))
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--no_archimedes", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    E, tokE, payload = ec.load_expanse(args.checkpoint)
    A = tokA = None
    if not args.no_archimedes and Path(ec.PATHS["arch_final"]).exists():
        A, tokA, _ = ec.load_archimedes(str(ec.PATHS["arch_final"]))
    elif not args.no_archimedes:
        print(f"[chat] {ec.PATHS['arch_final']} not found: Archimedes comparison disabled", flush=True)
    print(f"[chat] loaded {args.checkpoint} in {time.time() - t0:.1f}s", flush=True)
    app = build_app(E, tokE, A, tokA, payload.get("expanse"))
    app.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
