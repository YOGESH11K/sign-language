"""Generate a synthetic landmark dataset for pipeline tests / CI.

This is a TEST FIXTURE, not part of the real recognition path. Real data comes
from ``training/collect_data.py`` with a webcam. This module writes samples in
the exact same JSON schema so that ``preprocess``, ``train`` and ``evaluate`` can
be exercised end-to-end without a camera, with clearly separable hand shapes.

Usage::

    python -m training.synthetic_data --outdir data/raw --user synthetic
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Canonical OPEN palm: 21 landmark coordinates relative to the wrist (wrist=origin).
_BASE = np.array(
    [
        [0.00, 0.00],   # 0 wrist
        [-0.10, -0.02], [-0.15, -0.08], [-0.18, -0.14], [-0.20, -0.20],  # thumb 1..4
        [0.02, -0.12], [0.03, -0.21], [0.04, -0.28], [0.04, -0.35],       # index 5..8
        [0.07, -0.13], [0.08, -0.22], [0.09, -0.29], [0.09, -0.36],       # middle 9..12
        [0.12, -0.12], [0.13, -0.21], [0.14, -0.28], [0.14, -0.34],       # ring 13..16
        [0.16, -0.10], [0.17, -0.18], [0.18, -0.24], [0.19, -0.29],       # pinky 17..20
    ],
    dtype=float,
)

_MCP = {5: (0.02, -0.12), 9: (0.07, -0.13), 13: (0.12, -0.12), 17: (0.16, -0.10)}
_FINGERS = (5, 9, 13, 17)


def _chain_up(mcp, tip_len) -> np.ndarray:
    """Return [[mcp, pip_a, pip_b, tip]] for one extended finger."""
    (mx, my) = mcp
    return np.array([
        [mx, my],
        [mx + 0.01, my - 0.09],
        [mx + 0.01, my - 0.16],
        [mx + 0.01, my - tip_len],
    ])


def _chain_folded(mcp) -> np.ndarray:
    (mx, my) = mcp
    return np.array([
        [mx, my],
        [mx + 0.03, my - 0.01],
        [mx + 0.07, my + 0.02],
        [mx + 0.11, my + 0.05],
    ])


def _chain_flat(mcp) -> np.ndarray:
    (mx, my) = mcp
    return np.array([
        [mx, my],
        [mx + 0.05, my - 0.005],
        [mx + 0.10, my],
        [mx + 0.14, my],
    ])


def _layout(up: set[int], flat: set[int], thumb_mode: str = "open") -> np.ndarray:
    pts = {0: [0.0, 0.0]}
    # thumb
    if thumb_mode == "open":
        pts.update({1: [-0.10, -0.02], 2: [-0.15, -0.08],
                    3: [-0.18, -0.14], 4: [-0.20, -0.20]})
    elif thumb_mode == "across":       # folded over the palm
        pts.update({1: [-0.06, 0.01], 2: [-0.02, 0.02],
                    3: [0.02, -0.03], 4: [0.05, -0.10]})
    else:                              # tucked beside the palm
        pts.update({1: [-0.08, -0.01], 2: [-0.05, -0.00],
                    3: [-0.02, 0.02], 4: [0.01, 0.04]})
    for f in _FINGERS:
        mcp = _MCP[f]
        if f in up:
            chain = _chain_up(mcp, 0.23)
        elif f in flat:
            chain = _chain_flat(mcp)
        else:
            chain = _chain_folded(mcp)
        for i, pt in enumerate(chain):
            pts[f + i] = pt
    return np.array([pts[i] for i in range(21)])


def make_pose_rel(pose: str) -> np.ndarray:
    """Return the 21x2 relative layout for a named pose."""
    p = pose.lower()
    if p == "open":
        return _BASE.copy()
    if p == "fist":
        return _layout(set(), set(), thumb_mode="tucked")
    if p == "flat":
        return _layout({5, 9, 13, 17}, set(), thumb_mode="across")
    if p == "point":
        return _layout({5}, set(), thumb_mode="across")
    if p == "peace":
        return _layout({5, 9}, set(), thumb_mode="across")
    if p == "thumb_up":
        return _layout(set(), set(), thumb_mode="open")
    raise ValueError(f"unknown pose {pose}")


def rel_to_hand(rel: np.ndarray, rng: np.random.Generator) -> dict:
    """Make a single hand dict: relative layout -> image-space landmarks."""
    scale = float(rng.uniform(0.20, 0.30))
    wrist = (float(rng.uniform(0.35, 0.65)), float(rng.uniform(0.55, 0.80)))
    rot = rng.uniform(-0.2, 0.2)
    c, s = np.cos(rot), np.sin(rot)
    pts = []
    for (x, y) in rel:
        rx, ry = c * x - s * y, s * x + c * y
        pts.append([wrist[0] + rx * scale, wrist[1] + ry * scale,
                    float(rng.normal(0.0, 0.01))])
    lm = np.asarray(pts, dtype=float)
    lm[:3, :2] += rng.normal(0, 0.004, lm[:3, :2].shape)
    return {
        "handedness": rng.choice(["Right", "Left"]),
        "score": float(rng.uniform(0.8, 1.0)),
        "landmarks": lm.tolist(),
        "world_landmarks": np.zeros((21, 3)).tolist(),
    }


POSES = {
    "A": "thumb_up",
    "B": "flat",
    "C": "open",
    "D": "point",
    "E": "fist",
}

DYNAMIC = {
    "HELLO": ("fist", "open"),
    "THANK_YOU": ("open", "fist"),
    "YES": ("fist", "thumb_up"),
    "NO": ("fist", "flat"),
    "HELP": ("point", "open"),
    "HOSPITAL": ("open", "point"),
}


def make_sample(label: str, frames_per_sample: int, rng: np.random.Generator) -> dict:
    static_frames = []
    if label in DYNAMIC:
        start, end = DYNAMIC[label]
        p0 = make_pose_rel(start)
        p1 = make_pose_rel(end)
        hand0, hand1 = rng.choice(["Right", "Left"], size=2, replace=False)
        for i in range(frames_per_sample):
            t = i / (frames_per_sample - 1)
            eased = 0.5 - 0.5 * np.cos(np.pi * t)
            rel = (1 - eased) * p0 + eased * p1
            rel += rng.normal(0.0, 0.004, rel.shape)
            hands = [rel_to_hand(rel, rng)]
            if label in ("YES", "HELP"):   # a few two-hand fixtures
                hands.append(rel_to_hand(rel, rng))
            static_frames.append({"t": i * 0.04, "hands": hands})
    else:
        rel = make_pose_rel(POSES.get(label, "open"))
        for i in range(frames_per_sample):
            noisy = rel + rng.normal(0.0, 0.004, rel.shape)
            hands = [rel_to_hand(noisy, rng)]
            static_frames.append({"t": i * 0.04, "hands": hands})

    sign_type = "dynamic" if label in DYNAMIC else "static"
    return {
        "label": label,
        "type": sign_type,
        "user": "synthetic",
        "session": "synthetic",
        "collected_at": "fixture",
        "fps": 25,
        "frames": static_frames,
        "remote": False,
    }


def generate(outdir: str | Path, user: str, samples: int,
             static_labels: list[str] | None = None,
             dynamic_labels: list[str] | None = None,
             frames: int = 16) -> list[Path]:
    outdir = Path(outdir)
    user_dir = outdir / user
    user_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20240901)
    labels = list(static_labels or list(POSES)) + list(dynamic_labels or list(DYNAMIC))
    saved = []
    for label in labels:
        n = samples
        for k in range(n):
            sample = make_sample(label, frames, rng)
            path = user_dir / f"{label}_{k:03d}_{uuid.uuid4().hex[:6]}.json"
            path.write_text(json.dumps(sample, indent=1), encoding="utf-8")
            saved.append(path)
    return saved


def run_cli(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="data/raw")
    parser.add_argument("--user", default="synthetic")
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--frames", type=int, default=16)
    args = parser.parse_args(argv)
    saved = generate(args.outdir, args.user, args.samples, frames=args.frames)
    print(f"Wrote {len(saved)} synthetic samples to {Path(args.outdir) / args.user}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())