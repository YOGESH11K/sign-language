"""Reverse communication: typed (or spoken) text -> sign language representation.

The signed vocabulary is exactly the set of glosses in ``data/labels.json``.
This is intentionally NOT a full text->ISL translation engine: it maps known
words/phrases to the glosses we actually support, and fingerspells unknown words
using the letter glosses if those are available. See README limitations.

Sign visuals: the app replays a *recorded reference take* of a sign (a real
landmark sample saved during dataset collection). If no take exists for a gloss,
a text card is shown instead - nothing is faked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import RecognitionConfig
from .hand_detector import _HAND_CONNECTIONS

PUNCTUATION = "!?.,;:'\"" + "“”‘’()-"


@dataclass
class SignToken:
    gloss: str          # canonical label, e.g. "THANK_YOU"
    matched_text: str   # the source word/phrase, e.g. "thank you"
    letters: bool = False  # True when produced by fingerspelling

    def display(self) -> str:
        if self.letters:
            return " / ".join(list(self.gloss.replace("_", " ").upper()))
        return self.gloss.replace("_", " ").upper()


class SignDictionary:
    """Vocabulary and phrase lookup built from data/labels.json."""

    def __init__(self, labels_path: str | Path):
        labels_path = Path(labels_path)
        self.signs: dict[str, dict] = {}
        self.alias: dict[str, str] = {}
        self.phrases: dict[str, str] = {}
        self.letters: list[str] = []
        if labels_path.exists():
            data = json.loads(labels_path.read_text(encoding="utf-8"))
            for name, meta in data.get("signs", {}).items():
                self.signs[name] = meta
                self.alias[name.lower()] = name
                en = str(meta.get("en", name.lower())).lower()
                if en != name.lower():
                    self.alias[en] = name
                if len(name) == 1 and name.isalpha():
                    self.letters.append(name)
        self.letters.sort()

    @property
    def glosses(self) -> list[str]:
        return list(self.signs)

    def lookup_word(self, word: str) -> Optional[str]:
        """Map a word to a gloss, or None."""
        return self.alias.get(word.lower().strip(PUNCTUATION))

    def unknown_fingerspell(self, word: str) -> list[SignToken]:
        """Spell an unknown word letter-by-letter if letter glosses exist."""
        upper = word.upper()
        if not self.letters or not upper.isalpha():
            return [SignToken(gloss=str(upper), matched_text=word, letters=True)]
        return [SignToken(gloss=ch, matched_text=word, letters=True) for ch in upper]


class SignDictionaryService:
    def __init__(self, dict_: SignDictionary):
        self.dict = dict_

    def to_sign_sequence(self, sentence: str) -> list[SignToken]:
        """Convert free text into an ordered list of sign gloss tokens."""
        if not sentence or not sentence.strip():
            return []
        words = sentence.upper().split()
        # phrase matching first: try to join words into known phrases
        tokens: list[SignToken] = []
        i = 0
        while i < len(words):
            matched = None
            end = i
            for j in range(len(words), i, -1):
                phrase = " ".join(words[i:j]).lower().strip(PUNCTUATION)
                if phrase in self.dict.alias:
                    matched = self.dict.alias[phrase]
                    end = j - 1
                    break
            if matched:
                tokens.append(SignToken(gloss=matched,
                                        matched_text=" ".join(words[i:end + 1])))
                i = end + 1
            elif i == end:
                word = words[i]
                gloss = self.dict.lookup_word(word)
                if gloss:
                    tokens.append(SignToken(gloss=gloss, matched_text=word))
                else:
                    tokens.extend(self.dict.unknown_fingerspell(word.strip(PUNCTUATION)))
                i += 1
            else:  # pragma: no cover
                i = end + 1
        return tokens


class SignRenderer:
    """Renders a recorded sign take as 2D landmark line drawings (cv2)."""

    def __init__(self, canvas_size: int = 380, margin: float = 0.15):
        self.size = canvas_size
        self.margin = margin

    def render_hands(self, hands: list[dict], frame_bgr=None) -> np.ndarray:
        if frame_bgr is None:
            canvas = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        else:
            canvas = frame_bgr.copy()
            canvas = cv2.resize(canvas, (self.size, self.size))
        for raw in hands:
            lm = np.asarray(raw["landmarks"], dtype=np.float32).reshape(21, 3)
            pts = []
            for x, y in lm[:, :2]:
                px = int(x * self.size)
                py = int(y * self.size)
                pts.append((px, py))
                cv2.circle(canvas, (px, py), 5, (80, 200, 255), -1)
            for a, b in _HAND_CONNECTIONS:
                cv2.line(canvas, pts[a], pts[b], (80, 200, 255), 3)
        return canvas

    def render_sample(self, sample: Optional[dict], frame_index: int = 0) -> np.ndarray:
        if sample is None:
            canvas = np.zeros((self.size, self.size, 3), dtype=np.uint8)
            cv2.putText(canvas, "no recorded take", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return canvas
        frames = sample.get("frames", [])
        if not frames:
            canvas = np.zeros((self.size, self.size, 3), dtype=np.uint8)
            cv2.putText(canvas, "no take", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return canvas
        idx = min(int(frame_index), len(frames) - 1)
        hands = frames[idx].get("hands", [])
        canvas = self.render_hands(hands)
        label = sample.get("label", "")
        cv2.putText(canvas, label, (10, self.size - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        return canvas


def load_reference_sample(cfg: RecognitionConfig, gloss: str) -> Optional[dict]:
    """Return the recorded reference take for ``gloss``, if we have one."""
    path = cfg.paths["reference_signs"] / f"{gloss}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def reference_frame_count(sample: Optional[dict]) -> int:
    if not sample:
        return 1
    return max(1, len(sample.get("frames", [])))