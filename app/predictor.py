"""Real-time sign prediction with temporal stabilisation.

Pipeline per frame::

    hand landmarks -> feature vector -> static classifier (posture)
                                   --> sequence model (movement window)
    -> confidence filter -> consecutive-frame smoothing -> stable sign -> emitted word

Configurable knobs (see app/config.py):
  * CONFIDENCE_THRESHOLD  - frame predictions below this confidence are ignored
  * STABLE_FRAMES         - matching frames required before a sign is emitted
  * PREDICTION_COOLDOWN   - after an emission the same sign is ignored for N frames
"""

from __future__ import annotations

import dataclasses
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .config import RecognitionConfig
from .feature_extractor import frame_features


@dataclass
class Prediction:
    label: str
    confidence: float
    source: str = "static"   # "static" | "sequence"
    is_new: bool = True      # True only on the first read after emission


@dataclass
class ModelStatus:
    static_loaded: bool = False
    sequence_loaded: bool = False
    static_classes: list = dataclasses.field(default_factory=list)
    sequence_classes: list = dataclasses.field(default_factory=list)
    static_message: str = "static model not loaded"
    sequence_message: str = "sequence model not loaded"


class SignPredictor:
    """Combines models, smoothing and cooldown into one streaming interface."""

    def __init__(self, cfg: RecognitionConfig):
        self.cfg = cfg
        self.status = ModelStatus()
        self._static = None
        self._seq = None
        self._history: deque = deque(maxlen=cfg.sequence_window)
        self._window_filled = False
        self._reset_stream()

    # ------------------------------------------------------------------ state
    def _reset_stream(self) -> None:
        self._last_label: Optional[str] = None
        self._last_conf = 0.0
        self._streak = 0
        self._cooldown = 0
        self._last_source_val = "static"
        self._current: Optional[Prediction] = None
        self._history.clear()
        self._window_filled = False

    @property
    def static_loaded(self) -> bool:
        return self._static is not None

    @property
    def sequence_loaded(self) -> bool:
        return self._seq is not None

    @property
    def current(self) -> Optional[Prediction]:
        return self._current

    # ----------------------------------------------------------------- models
    def load_models(self, static_path: Optional[Path] = None,
                    sequence_path: Optional[Path] = None) -> ModelStatus:
        self._load_static(Path(static_path) if static_path else self.cfg.paths["static_model"])
        self._load_sequence(Path(sequence_path) if sequence_path else self.cfg.paths["sequence_model"])
        return self.status

    def _load_static(self, path: Path) -> None:
        if not path.exists():
            self.status.static_message = f"static model missing: {path.name}"
            return
        try:
            import joblib
        except ImportError:
            self.status.static_message = "joblib (scikit-learn) not installed"
            return
        try:
            payload = joblib.load(path)
            if isinstance(payload, dict) and "model" in payload:
                self._static = payload["model"]
                classes = payload.get("classes", [])
            elif hasattr(payload, "predict_proba"):
                self._static = payload
                classes = list(payload.classes_)
            else:
                raise ValueError("unexpected static model format")
            self.status.static_classes = [str(c) for c in classes]
            self.status.static_loaded = True
            self.status.static_message = (
                f"static model ready ({len(self.status.static_classes)} classes)"
            )
        except Exception as exc:
            self.status.static_message = f"static model failed to load: {exc}"

    def _load_sequence(self, path: Path) -> None:
        if not path.exists():
            self.status.sequence_message = f"sequence model missing: {path.name}"
            return
        try:
            from .sequence_model import SequenceClassifier
            import torch
        except ImportError as exc:
            self.status.sequence_message = f"pytorch not installed: {exc}"
            return
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            classes = [str(c) for c in ckpt["classes"]]
            model = SequenceClassifier(
                input_dim=int(ckpt["feature_dim"]), n_classes=len(classes)
            )
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            self._seq = model
            self.status.sequence_classes = classes
            self.status.sequence_loaded = True
            self._seq_window = int(ckpt.get("seq_len", self.cfg.sequence_window))
            self._history = deque(maxlen=self._seq_window)
            self.status.sequence_message = (
                f"sequence model ready ({len(classes)} classes)"
            )
        except Exception as exc:
            self.status.sequence_message = f"sequence model failed to load: {exc}"

    # ----------------------------------------------------------------- stream
    def update(self, hands: list) -> Optional[Prediction]:
        """Feed one frame's detections. Returns a NEW emission, or None."""
        if not hands:
            self._reset_stream()
            return None
        feat = frame_features(hands)
        if self._static is None and self._seq is None:
            return None

        self._history.append(feat)
        if self._seq is not None and len(self._history) == self._history.maxlen:
            self._window_filled = True

        label, conf, source = self._frame_prediction()
        self._last_source_val = source
        if label is None or conf < self.cfg.confidence_threshold:
            self._streak = 0
            self._last_label = None
            return None

        if label == self._last_label:
            self._streak += 1
            self._last_conf = max(self._last_conf, conf)
        else:
            self._last_label = label
            self._last_conf = conf
            self._streak = 1

        if self._cooldown > 0:
            self._cooldown -= 1

        current_label = self._current.label if self._current is not None else None
        if (self._streak >= self.cfg.stable_frames
                and label != current_label
                and self._cooldown <= 0):
            self._cooldown = self.cfg.prediction_cooldown
            self._current = Prediction(label=label, confidence=self._last_conf,
                                       source=self._last_source_val)
            return self._current
        return None

    def _frame_prediction(self) -> tuple[Optional[str], float, str]:
        """Best-scored candidate from the appropriate model(s).

        Returns ``(label, confidence, source)``.
        """
        scores: dict[str, float] = {}
        source = "static"
        if self._static is not None and self.status.static_classes:
            proba = self._static.predict_proba(np.asarray(self._history[-1])
                                               .reshape(1, -1))[0]
            for i, cls in enumerate(self.status.static_classes):
                if self._seq is not None and cls in set(self.status.sequence_classes):
                    continue  # dynamic signs: sequence model owns them
                scores[cls] = float(proba[i])

        if self._seq is not None and self._window_filled:
            proba = self._sequence_proba()
            scores.update({k: max(scores.get(k, 0.0), v) for k, v in proba.items()})
            source = "sequence"

        if not scores:
            return None, 0.0, source
        winner = max(scores, key=scores.get)
        return winner, scores[winner], source

    def _sequence_proba(self) -> dict[str, float]:
        import torch

        seq = np.stack(list(self._history))          # (L, D)
        mask = np.ones(len(seq), dtype=np.float32)
        x = torch.from_numpy(seq.astype(np.float32)).unsqueeze(0)
        m = torch.from_numpy(mask).unsqueeze(0)
        with torch.no_grad():
            logits = self._seq(x, m)[0]
            proba = torch.softmax(logits, dim=0).numpy()
        return {cls: float(p) for cls, p in zip(self.status.sequence_classes, proba)}

    def consume(self) -> Optional[Prediction]:
        """Return the newest emission once, with is_new set to False afterwards."""
        if self._current is not None and self._current.is_new:
            self._current.is_new = False
            return self._current
        return None

    def reset(self) -> None:
        self._reset_stream()

    @staticmethod
    def load_sign_index(path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}