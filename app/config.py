"""Central configuration for the Sign Language Communication system.

All tunable knobs live here. Values can be overridden with a JSON config file
``config.json`` next to the project root (simple key/value merge).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def project_path(*parts: str) -> Path:
    return ROOT_DIR.joinpath(*parts)


@dataclass
class RecognitionConfig:
    # --- model / paths -------------------------------------------------
    hand_model_path: str = "models/sign_model/hand_landmarker.task"
    static_model_path: str = "models/sign_model/static.joblib"
    sequence_model_path: str = "models/sign_model/sequence.pt"
    sign_index_path: str = "models/sign_model/sign_index.json"
    labels_path: str = "data/labels.json"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    reference_signs_dir: str = "data/reference"
    eval_output_dir: str = "models/eval"

    # --- recognition stability (feature 9) ----------------------------
    confidence_threshold: float = 0.60
    stable_frames: int = 4
    prediction_cooldown: int = 30          # frames to ignore the same label after emitting
    sequence_window: int = 30              # frames window for dynamic signs
    dynamic_min_frames: int = 10           # min recorded frames to treat a sample as usable dynamic

    # --- camera --------------------------------------------------------
    camera_index: int = 0
    camera_width: int = 960
    camera_height: int = 540
    fps_target: int = 30

    # --- TTS -----------------------------------------------------------
    speech_rate: int = 0                   # -10 .. 10 (Windows SAPI scale)
    tts_muted: bool = False

    # --- misc ----------------------------------------------------------
    hand_model_url: str = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    )
    user_default: str = "user1"

    @property
    def paths(self) -> dict[str, Path]:
        return {
            "hand_model": project_path(self.hand_model_path),
            "static_model": project_path(self.static_model_path),
            "sequence_model": project_path(self.sequence_model_path),
            "sign_index": project_path(self.sign_index_path),
            "labels": project_path(self.labels_path),
            "raw_data": project_path(self.raw_data_dir),
            "processed_data": project_path(self.processed_data_dir),
            "reference_signs": project_path(self.reference_signs_dir),
            "eval_output": project_path(self.eval_output_dir),
        }


def load_config(path: str | Path | None = None) -> RecognitionConfig:
    """Merge a JSON config file over the defaults, if present."""
    cfg = RecognitionConfig()
    if path is None:
        path = ROOT_DIR / "config.json"
    path = Path(path)
    if path.exists():
        try:
            overrides = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"Invalid config file {path}: {exc}") from exc
        for key, value in overrides.items():
            if hasattr(cfg, key) and isinstance(value, (int, float, bool, str)):
                setattr(cfg, key, value)
    return cfg


def save_default_config(path: str | Path | None = None) -> Path:
    if path is None:
        path = ROOT_DIR / "config.json"
    path = Path(path)
    path.write_text(
        json.dumps(asdict(RecognitionConfig()), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return path


def ensure_dirs(cfg: RecognitionConfig) -> None:
    # parent directories of model / log / label files
    for key in ("hand_model", "static_model", "sequence_model", "sign_index",
                "labels", "eval_output"):
        cfg.paths[key].parent.mkdir(parents=True, exist_ok=True)
    # the data directories themselves
    for key in ("raw_data", "processed_data", "reference_signs"):
        cfg.paths[key].mkdir(parents=True, exist_ok=True)
    (cfg.paths["raw_data"] / cfg.user_default).mkdir(parents=True, exist_ok=True)


__all__ = [
    "RecognitionConfig",
    "load_config",
    "save_default_config",
    "ensure_dirs",
    "project_path",
    "ROOT_DIR",
]