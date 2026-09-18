"""Export the trained sequence (GRU) model to ONNX for browser inference.

Usage::

    python -m training.export_onnx

Writes ``web/sequence.onnx`` (input ``x`` (1, 30, 142), ``mask`` (1, 30),
output ``logits`` (1, n_classes)) plus ``web/classes.json`` with the
class labels in order. The browser runs the exact same model via
``onnxruntime-web``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from app.config import load_config
from app.sequence_model import SequenceClassifier


def main() -> None:
    cfg = load_config()
    ckpt_path = cfg.paths["sequence_model"]
    if not ckpt_path.exists():
        raise SystemExit(f"sequence model not found: {ckpt_path} - train first")
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except Exception:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    state = ckpt["state_dict"]
    classes = [str(c) for c in ckpt["classes"]]
    feat_dim = int(ckpt.get("feature_dim", 142))
    seq_len = int(ckpt.get("seq_len", cfg.sequence_window))
    hidden = int(ckpt.get("hidden", 128))

    model = SequenceClassifier(feat_dim, len(classes), hidden=hidden)
    model.load_state_dict(state)
    model.eval()

    dummy_x = torch.zeros(1, seq_len, feat_dim, dtype=torch.float32)
    dummy_mask = torch.ones(1, seq_len, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            model, (dummy_x, dummy_mask),
            cfg.paths["web_dir"] / "sequence.onnx",
            input_names=["x", "mask"],
            output_names=["logits"],
            opset_version=17,
            do_constant_folding=True,
        )
    (cfg.paths["web_dir"] / "classes.json").write_text(
        json.dumps({"classes": classes, "seq_len": seq_len, "feature_dim": feat_dim}),
        encoding="utf-8",
    )
    print(f"  exported -> {cfg.paths['web_dir'] / 'sequence.onnx'}")
    print(f"  classes ({len(classes)}): {classes}")
    print(f"  shapes: x (1,{seq_len},{feat_dim}) mask (1,{seq_len}) -> logits")


if __name__ == "__main__":
    main()