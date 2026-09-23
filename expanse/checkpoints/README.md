# Checkpoints

The trained model (`supermix_expanse.pt`, 599 MB) is on Hugging Face:
<https://huggingface.co/Kai9987kai/supermix-expanse>

    hf download Kai9987kai/supermix-expanse supermix_expanse.pt --local-dir expanse/checkpoints

This folder keeps the receipts: `supermix_expanse_grafted.receipt.json` (stage 1 build: per-expert graft fits,
vocabulary lift, function-preservation check) and `supermix_expanse.receipt.json` (stage 2 training history).
`build_expanse.py` / `train_expanse.py` write their checkpoints here.
