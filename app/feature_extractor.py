"""Feature extraction from hand landmarks.

The model never sees raw pixels; it classifies a normalised feature vector.

Normalisation strategy (feature 10 - position/orientation invariance):
  * hand landmarks are re-centred on the wrist (landmark 0)
  * coordinates are scaled by the hand size (distance wrist -> middle-finger MCP)
  * the two hands are ordered deterministically (Left first, then Right)
  * a missing hand contributes an all-zero block, so one- vs two-hand signs
    are represented cleanly

Layout of the per-hand vector (71 floats):
  0..62  : 21 landmarks (x, y, z) relative to wrist / hand size
  63..67 : distance wrist->fingertip for thumb..pinky, normalised
  68     : distance index-tip -> ring-tip, normalised (finger spread)
  69..70 : handedness one-hot (Right, Left)

Both hands are concatenated -> frame feature vector of 142 floats.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Union

import numpy as np

from .hand_detector import Hand

NUM_LANDMARKS = 21
PER_HAND_DIM = 71
FEATURE_DIM = PER_HAND_DIM * 2
TIP_IDS = (4, 8, 12, 16, 20)
WRIST = 0
MIDDLE_MCP = 9
INDEX_TIP = 8
RING_TIP = 16

_HANDEDNESS_ORDER = {"Left": 0, "Right": 1, "Unknown": 2}


def hand_sort_key(hand: Union[Hand, dict]) -> int:
    name = getattr(hand, "handedness", None)
    if name is None:
        name = hand.get("handedness", "Unknown") if isinstance(hand, dict) else "Unknown"
    return _HANDEDNESS_ORDER.get(name, 2)


def normalize_hand_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Return (63,) landmark coords translated+scaled so the hand is canonical."""
    lm = np.asarray(landmarks, dtype=np.float32).reshape(NUM_LANDMARKS, 3)
    wrist = lm[WRIST]
    scale = float(np.linalg.norm(lm[MIDDLE_MCP] - wrist))
    if scale < 1e-6:
        return np.zeros(NUM_LANDMARKS * 3, dtype=np.float32)
    return ((lm - wrist) / scale).astype(np.float32).ravel()


def hand_feature(hand: Union[Hand, dict]) -> np.ndarray:
    """Build the 71-dim feature vector for a single hand."""
    if isinstance(hand, Hand):
        landmarks = hand.landmarks
        handedness = hand.handedness
    else:
        landmarks = np.asarray(hand.get("landmarks", np.zeros((21, 3))), dtype=np.float32)
        handedness = hand.get("handedness", "Unknown")

    lm = np.asarray(landmarks, dtype=np.float32).reshape(NUM_LANDMARKS, 3)
    wrist = lm[WRIST]
    scale = float(np.linalg.norm(lm[MIDDLE_MCP] - wrist))
    if scale < 1e-6:
        return np.zeros(PER_HAND_DIM, dtype=np.float32)

    rel = ((lm - wrist) / scale).astype(np.float32).ravel()          # 63

    tips = lm[list(TIP_IDS)] - wrist
    tip_dists = (np.linalg.norm(tips, axis=1) / scale).astype(np.float32)   # 5

    spread = (np.linalg.norm(lm[INDEX_TIP] - lm[RING_TIP]) / scale).astype(np.float32)  # 1

    onehot = np.zeros(2, dtype=np.float32)
    if handedness == "Right":
        onehot[0] = 1.0
    elif handedness == "Left":
        onehot[1] = 1.0

    return np.concatenate([rel, tip_dists, spread.reshape(1), onehot]).astype(np.float32)


def frame_features(hands: Sequence[Union[Hand, dict]]) -> np.ndarray:
    """Build the 142-dim feature vector representing all hands in one frame."""
    ordered = sorted(hands, key=hand_sort_key)
    vecs = [hand_feature(h) for h in ordered]
    vecs = (vecs + [np.zeros(PER_HAND_DIM, dtype=np.float32)])[:2]
    if len(vecs) > 2:
        # ignore more than two hands (feature 14: unexpected hands)
        vecs = vecs[:2]
    return np.concatenate(vecs).astype(np.float32)


def raw_frame_to_hands(frame_entry: dict) -> List[Hand]:
    """Convert a stored JSON frame entry into Hand objects."""
    hands = []
    for raw in frame_entry.get("hands", []):
        hands.append(Hand.from_dict(raw))
    return hands


def sample_to_sequence(sample: dict, frame_key: str = "frames") -> np.ndarray:
    """Convert a stored collection sample into a (T, FEATURE_DIM) feature matrix."""
    frames = sample.get(frame_key, [])
    seq = [frame_features(raw_frame_to_hands(fr) if isinstance(fr, dict) else fr)
           for fr in frames]
    if not seq:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)
    return np.stack(seq).astype(np.float32)


def canonical_static_frame(sequence: np.ndarray) -> np.ndarray:
    """Pick the most 'average' frame of a static sample for training.

    Chooses the frame closest to the mean pose, which is robust against a few
    noisy frames at the start/end of the capture burst.
    """
    if len(sequence) == 1:
        return sequence[0]
    mean = sequence.mean(axis=0)
    dists = np.linalg.norm(sequence - mean, axis=1)
    return sequence[int(np.argmin(dists))]


def pad_or_truncate_sequence(
    seq: np.ndarray, length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pad/truncate a (T, D) sequence to exactly ``length`` frames.

    Returns ``(padded (L, D), mask (L,))`` where mask is 1 for real frames.
    """
    t = len(seq)
    if t >= length:
        return seq[:length].astype(np.float32), np.ones(length, dtype=np.float32)
    pad = np.zeros((length - t, FEATURE_DIM), dtype=np.float32)
    out = np.concatenate([seq, pad], axis=0).astype(np.float32)
    mask = np.zeros(length, dtype=np.float32)
    mask[:t] = 1.0
    return out, mask