"""Tests for the streaming predictor: models, smoothing, threshold, cooldown."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import RecognitionConfig
from app.feature_extractor import frame_features
from app.predictor import SignPredictor
from training.synthetic_data import make_pose_rel, rel_to_hand


def _make_cfg(tmp_path, stable=3, cooldown=30, threshold=0.5) -> RecognitionConfig:
    cfg = RecognitionConfig()
    cfg.stable_frames = stable
    cfg.prediction_cooldown = cooldown
    cfg.confidence_threshold = threshold
    cfg.static_model_path = str(tmp_path / "static.joblib")
    cfg.sequence_model_path = str(tmp_path / "sequence.pt")
    cfg.sign_index_path = str(tmp_path / "sign_index.json")
    return cfg


def _train_rf(cfg, rng=None) -> None:
    rng = rng or np.random.default_rng(42)
    open_ = frame_features([rel_to_hand(make_pose_rel("open"), rng)])
    fist_ = frame_features([rel_to_hand(make_pose_rel("fist"), rng)])
    X, y = [], []
    for _ in range(120):
        o = frame_features([rel_to_hand(make_pose_rel("open"), rng)])
        f = frame_features([rel_to_hand(make_pose_rel("fist"), rng)])
        X.append(o)
        y.append(0)
        X.append(f)
        y.append(1)
    X = np.asarray(X)
    y = np.asarray(y)
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=50, random_state=0).fit(X, y)
    joblib.dump({"model": model, "classes": ["OPEN", "FIST"],
                 "feature_dim": X.shape[1]}, cfg.paths["static_model"])


def _hand(pose, rng=None):
    return rel_to_hand(make_pose_rel(pose), rng or np.random.default_rng(99))


def test_predictor_emits_after_stable_frames(tmp_path):
    cfg = _make_cfg(tmp_path, stable=3)
    _train_rf(cfg)
    pred = SignPredictor(cfg)
    pred.load_models()
    assert pred.static_loaded

    new = None
    for i in range(3):
        new = pred.update([_hand("open")])
    assert new is not None and new.label == "OPEN"
    # consume marks it as read exactly once
    got = pred.consume()
    assert got is not None and got.is_new is False
    assert pred.consume() is None


def test_predictor_cooldown_suppresses_repeats(tmp_path):
    cfg = _make_cfg(tmp_path, stable=3, cooldown=50)
    _train_rf(cfg)
    pred = SignPredictor(cfg)
    pred.load_models()

    emitted = []
    for i in range(60):
        m = pred.update([_hand("open")])
        if m is not None:
            emitted.append(m.label)
    # even with 25 stable frames available, only the first emission survives
    # because the cooldown blocks the same label afterwards
    assert emitted.count("OPEN") == 1
    assert len(emitted) <= 1


def test_predictor_handles_label_change(tmp_path):
    cfg = _make_cfg(tmp_path, stable=3, cooldown=20)
    _train_rf(cfg)
    pred = SignPredictor(cfg)
    pred.load_models()

    for _ in range(3):
        pred.update([_hand("open")])
    assert pred.consume() is not None          # OPEN emitted
    emitted = None
    for _ in range(60):                        # switch to FIST; must emit once
        emitted = pred.update([_hand("fist")])
        if emitted is not None:
            break
    assert emitted is not None and emitted.label == "FIST"
    # holding FIST still must not re-emit FIST
    for _ in range(60):
        assert pred.update([_hand("fist")]) is None


def test_predictor_resets_on_no_hands(tmp_path):
    cfg = _make_cfg(tmp_path, stable=2)
    _train_rf(cfg)
    pred = SignPredictor(cfg)
    pred.load_models()
    pred.update([_hand("open")])
    assert pred.update([]) is None
    assert pred.consume() is None


def test_predictor_confidence_gate(tmp_path):
    cfg = _make_cfg(tmp_path, stable=3)
    _train_rf(cfg)
    pred = SignPredictor(cfg)
    pred.load_models()
    # force a low-confidence frame prediction; it must not build a streak
    with patch.object(pred, "_frame_prediction", return_value=("OPEN", 0.3, "static")):
        for _ in range(10):
            assert pred.update([_hand("open")]) is None


def test_predictor_alternating_frames_never_stabilises(tmp_path):
    cfg = _make_cfg(tmp_path, stable=4)
    _train_rf(cfg)
    pred = SignPredictor(cfg)
    pred.load_models()
    for i in range(40):
        pose = "open" if i % 2 == 0 else "fist"
        assert pred.update([_hand(pose)]) is None


def test_predictor_no_model_returns_none(tmp_path):
    cfg = _make_cfg(tmp_path)
    pred = SignPredictor(cfg)
    pred.load_models()
    assert pred.update([_hand("open")]) is None