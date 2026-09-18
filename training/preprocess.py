"""Convert raw collected samples into model-ready feature datasets.

Reads everything under ``data/raw/`` (each sample is a JSON with raw landmarks),
extracts normalised features using :mod:`app.feature_extractor`, splits samples
into static and dynamic sets, and writes ``data/processed/dataset.npz`` plus a
human-readable meta JSON.

A sample is used as:

  * static   -> one canonical, most-representative frame of the capture burst
  * dynamic  -> the full frame sequence, zero-padded to ``sequence_window``
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import RecognitionConfig, load_config, ensure_dirs
from app.feature_extractor import (
    FEATURE_DIM,
    canonical_static_frame,
    pad_or_truncate_sequence,
    sample_to_sequence,
)


def iter_samples(raw_dir: Path):
    for path in sorted(raw_dir.rglob("*.json")):
        try:
            sample = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            print(f"  ! skipping unreadable sample {path}")
            continue
        yield path, sample


def load_label_types(cfg: RecognitionConfig) -> dict[str, str]:
    labels_path = cfg.paths["labels"]
    if not labels_path.exists():
        return {}
    data = json.loads(labels_path.read_text(encoding="utf-8"))
    return {name: meta.get("type", "static") for name, meta in data.get("signs", {}).items()}


def preprocess(cfg: RecognitionConfig | None = None, verbose: bool = True) -> dict:
    cfg = cfg or load_config()
    ensure_dirs(cfg)
    raw_dir = cfg.paths["raw_data"]
    label_types = load_label_types(cfg)

    static_seqs: defaultdict = defaultdict(list)
    static_users: defaultdict = defaultdict(list)
    seq_seqs: defaultdict = defaultdict(list)
    seq_users: defaultdict = defaultdict(list)

    skipped = 0
    for path, sample in iter_samples(raw_dir):
        label = sample.get("label")
        if not label:
            skipped += 1
            continue
        sign_type = sample.get("type") or label_types.get(label, "static")
        try:
            seq = sample_to_sequence(sample)
        except Exception as exc:
            print(f"  ! could not extract features from {path}: {exc}")
            skipped += 1
            continue
        if seq.shape[0] == 0:
            skipped += 1
            continue
        if sign_type == "dynamic":
            if seq.shape[0] >= cfg.dynamic_min_frames:
                seq_seqs[label].append(seq)
                seq_users[label].append(sample.get("user") or sample.get("session", "?"))
            else:
                skipped += 1
        else:
            static_seqs[label].append(canonical_static_frame(seq))
            static_users[label].append(sample.get("user") or sample.get("session", "?"))

    if not (static_seqs or seq_seqs):
        raise SystemExit(
            "No usable samples found under data/raw. Collect data first:\n"
            "  python -m training.collect_data --label HELLO --samples 20"
        )

    static_labels = sorted(static_seqs)
    seq_labels = sorted(seq_seqs)

    X_static, y_static, users_static = [], [], []
    for i, lab in enumerate(static_labels):
        for feat, usr in zip(static_seqs[lab], static_users[lab]):
            X_static.append(feat)
            y_static.append(i)
            users_static.append(usr)

    X_seq, mask_seq, y_seq, users_seq = [], [], [], []
    L = cfg.sequence_window
    for i, lab in enumerate(seq_labels):
        for s, usr in zip(seq_seqs[lab], seq_users[lab]):
            padded, mask = pad_or_truncate_sequence(s, L)
            X_seq.append(padded)
            mask_seq.append(mask)
            y_seq.append(i)
            users_seq.append(usr)

    data = {}
    if static_labels:
        data.update(
            X_static=np.stack(X_static),
            y_static=np.asarray(y_static, dtype=np.int64),
            users_static=np.asarray(users_static, dtype="U64"),
            static_labels=np.asarray(static_labels, dtype="U64"),
        )
    if seq_labels:
        data.update(
            X_seq=np.stack(X_seq),
            mask_seq=np.stack(mask_seq),
            y_seq=np.asarray(y_seq, dtype=np.int64),
            users_seq=np.asarray(users_seq, dtype="U64"),
            seq_labels=np.asarray(seq_labels, dtype="U64"),
        )
    data["feature_dim"] = np.asarray(FEATURE_DIM)
    data["seq_len"] = np.asarray(L)

    out = cfg.paths["processed_data"] / "dataset.npz"
    np.savez(out, **data)

    meta = {
        "feature_dim": FEATURE_DIM,
        "seq_len": L,
        "static_classes": static_labels,
        "sequence_classes": seq_labels,
        "n_static_samples": int(len(y_static)),
        "n_sequence_samples": int(len(y_seq)),
        "static_per_class": {lab: len(static_seqs[lab]) for lab in static_labels},
        "sequence_per_class": {lab: len(seq_seqs[lab]) for lab in seq_labels},
    }
    (cfg.paths["processed_data"] / "dataset_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    if verbose:
        print(f"Saved dataset -> {out}")
        print(f"  static:    {meta['n_static_samples']} samples, "
              f"{len(static_labels)} classes {static_labels}")
        print(f"  sequence:  {meta['n_sequence_samples']} samples, "
              f"{len(seq_labels)} classes {seq_labels}")
        print(f"  skipped invalid samples: {skipped}")
    return meta


def summary(cfg: RecognitionConfig | None = None) -> None:
    cfg = cfg or load_config()
    counts = Counter()
    users = set()
    for path, sample in iter_samples(cfg.paths["raw_data"]):
        counts[sample.get("label")] += 1
        users.add(sample.get("user") or sample.get("session", "?"))
    print(f"Raw samples by label: {dict(counts)}")
    print(f"Users present: {sorted(users)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess raw samples into a dataset.")
    parser.add_argument("--summary", action="store_true", help="Only summarise raw data")
    args = parser.parse_args()
    if args.summary:
        summary()
    else:
        preprocess()