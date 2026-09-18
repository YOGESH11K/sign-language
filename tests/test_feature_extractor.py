"""Tests for landmark feature extraction / normalisation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.feature_extractor import (
    FEATURE_DIM,
    PER_HAND_DIM,
    canonical_static_frame,
    frame_features,
    hand_feature,
    normalize_hand_landmarks,
    pad_or_truncate_sequence,
    sample_to_sequence,
)
from training.synthetic_data import make_pose_rel, rel_to_hand


def _hand(pose: str, rng=None):
    rng = rng or np.random.default_rng(7)
    return rel_to_hand(make_pose_rel(pose), rng)


def test_feature_dimensions():
    one = frame_features([_hand("open")])
    assert one.shape == (FEATURE_DIM,)
    two = frame_features([_hand("open"), _hand("fist")])
    assert two.shape == (FEATURE_DIM,)
    assert not np.allclose(one, two)


def test_single_hand_pads_second_block_with_zeros():
    feats = frame_features([_hand("open")])
    zero_block = feats[PER_HAND_DIM:]
    assert np.allclose(zero_block, 0.0)


def test_normalisation_position_invariance():
    """Moving/scaling a hand must not change the relative features."""
    rng = np.random.default_rng(11)
    lm = np.asarray(_hand("open")["landmarks"], dtype=np.float32)
    a = normalize_hand_landmarks(lm)
    b = normalize_hand_landmarks(lm * 1.8)
    c = normalize_hand_landmarks(lm + np.array([0.4, -0.3, 0]))
    assert np.allclose(a, b, atol=0.02)
    assert np.allclose(a, c, atol=0.02)


def test_handedness_ordering_is_deterministic():
    h_left = dict(_hand("open", np.random.default_rng(3)))
    h_left["handedness"] = "Left"
    h_right = dict(_hand("open", np.random.default_rng(3)))
    h_right["handedness"] = "Right"
    f1 = frame_features([h_right, h_left])
    f2 = frame_features([h_left, h_right])
    assert np.allclose(f1, f2)


def test_canonical_static_frame_picks_median_like():
    seq = np.stack([
        np.eye(FEATURE_DIM)[0] + 5,
        np.eye(FEATURE_DIM)[0] + 0.5,
        np.eye(FEATURE_DIM)[0] + 1.0,
    ]).astype(np.float32)
    idx = np.argmin(np.linalg.norm(seq - seq.mean(axis=0), axis=1))
    assert np.allclose(canonical_static_frame(seq), seq[idx])


def test_pad_truncate():
    seq = np.ones((5, FEATURE_DIM), dtype=np.float32)
    padded, mask = pad_or_truncate_sequence(seq, 10)
    assert padded.shape == (10, FEATURE_DIM)
    assert np.allclose(mask[:5], 1) and np.allclose(mask[5:], 0)
    short, m2 = pad_or_truncate_sequence(seq, 3)
    assert short.shape == (3, FEATURE_DIM) and np.allclose(m2, 1)


def test_sample_to_sequence_schema():
    sample = {
        "label": "T",
        "frames": [
            {"t": 0.0, "hands": [_hand("open", np.random.default_rng(1))]},
            {"t": 0.04, "hands": [_hand("open", np.random.default_rng(2))]},
        ],
    }
    seq = sample_to_sequence(sample)
    assert seq.shape == (2, FEATURE_DIM)


def test_poses_are_separable():
    """The synthetic pose space must be separable so ML has signal to learn."""
    rng = np.random.default_rng(4)
    a = frame_features([_hand("open", rng)])
    b = frame_features([_hand("fist", rng)])
    dist = np.linalg.norm(a - b)
    assert dist > 1.0