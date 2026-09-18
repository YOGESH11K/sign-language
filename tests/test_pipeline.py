"""End-to-end pipeline test fixture: synthetic data -> preprocess -> train ->
evaluate -> predictor emits real predictions.

This proves the whole ML chain works (webcam-free). Real deployments should use
``training/collect_data.py`` with a camera.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import RecognitionConfig
from app.feature_extractor import frame_features
from app.predictor import SignPredictor
from training.preprocess import preprocess
from training.synthetic_data import make_pose_rel, rel_to_hand


def make_cfg(tmp_path) -> RecognitionConfig:
    cfg = RecognitionConfig()
    cfg.raw_data_dir = str(tmp_path / "raw")
    cfg.processed_data_dir = str(tmp_path / "processed")
    cfg.static_model_path = str(tmp_path / "static.joblib")
    cfg.sequence_model_path = str(tmp_path / "sequence.pt")
    cfg.sign_index_path = str(tmp_path / "sign_index.json")
    cfg.eval_output_dir = str(tmp_path / "eval")
    cfg.hand_model_path = str(tmp_path / "hand_landmarker.task")
    return cfg


def generate_synthetic(cfg, samples=18):
    from training.synthetic_data import generate

    generate(cfg.raw_data_dir, "synthetic", samples,
             static_labels=["A", "B", "C", "D", "E"],
             dynamic_labels=["HELLO", "THANK_YOU", "YES", "NO", "HELP", "HOSPITAL"])


def test_full_pipeline(tmp_path):
    cfg = make_cfg(tmp_path)
    generate_synthetic(cfg)

    # 1) preprocess
    meta = preprocess(cfg)
    assert meta["n_static_samples"] >= 90          # 5 classes
    assert meta["n_sequence_samples"] >= 100       # 6 classes
    assert meta["feature_dim"] == 142

    # 2) train both models
    from training.train import train_sequence, train_static

    static = train_static(cfg, "rf")
    assert static is not None and static["test_acc"] > 0.5
    seq = train_sequence(cfg)
    assert seq is not None and seq["test_acc"] > 0.5
    assert cfg.paths["static_model"].exists()
    assert cfg.paths["sequence_model"].exists()

    # 3) evaluate produces reports
    from training.evaluate import evaluate_static, evaluate_sequence

    evaluate_static(cfg)
    evaluate_sequence(cfg)
    metrics = cfg.paths["eval_output"] / "static_metrics.json"
    assert metrics.exists()
    assert cfg.paths["eval_output"] / "static_confusion_matrix.csv"

    # 4) the real-time predictor loads the trained models and emits predictions
    pred = SignPredictor(cfg)
    pred.load_models()
    assert pred.static_loaded and pred.sequence_loaded

    # a static prediction from live landmarks
    emitted = None
    for _ in range(cfg.stable_frames):
        emitted = pred.update([rel_to_hand(make_pose_rel("open"), np.random.default_rng(5))])
    assert emitted is not None
    assert emitted.label in {"A", "B", "C", "D", "E"}
    assert pred.consume() is not None


def test_static_model_discriminates_letters(tmp_path):
    """Every letter must be consistently predicted as itself on its training pose."""
    cfg = make_cfg(tmp_path)
    generate_synthetic(cfg, samples=24)
    preprocess(cfg)

    from training.train import train_static

    train_static(cfg, "rf")

    pred = SignPredictor(cfg)
    pred.load_models()
    rng = np.random.default_rng(0)
    for letter, pose in {"A": "thumb_up", "B": "flat", "C": "open",
                         "D": "point", "E": "fist"}.items():
        best = {}
        for _ in range(12):
            feat = frame_features([rel_to_hand(make_pose_rel(pose), rng)])
            proba = pred._static.predict_proba(feat.reshape(1, -1))[0]
            lbl = pred.status.static_classes[int(np.argmax(proba))]
            best[lbl] = max(best.get(lbl, 0.0), float(proba.max()))
        top = max(best, key=best.get)
        assert top == letter, f"expected {letter}, predicted {top}"