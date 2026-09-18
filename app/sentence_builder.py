"""Sentence building from recognised sign labels.

Signs are recognised as gloss labels (e.g. "HELLO", "I", "RAHUL"). Raw glosses
are not spoken one at a time - they are accumulated into a sentence using a
configurable word map (the ``en`` field of ``data/labels.json``).

``HELLO MY NAME IS RAHUL``  ->  "Hello, my name is Rahul."
"""

from __future__ import annotations

import json
from pathlib import Path

PROPER_NOUNS = {"RAHUL"}

# words that keep their capital form in the final sentence
ALWAYS_TITLE = {"I"}
# greeting words that take a comma when the sentence starts with them
GREETING_COMMA = {"hello", "hi", "hey"}


class SentenceBuilder:
    def __init__(self, labels_path: str | Path) -> None:
        self.labels_path = Path(labels_path)
        self._words: list[str] = []
        self._labels: list[str] = []
        self._display: dict[str, str] = self._load_display_words()

    def _load_display_words(self) -> dict[str, str]:
        if not self.labels_path.exists():
            return {}
        try:
            data = json.loads(self.labels_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        out = {}
        for name, meta in data.get("signs", {}).items():
            en = meta.get("en", name.lower())
            if name in PROPER_NOUNS:
                en = name.title()
            elif name in ALWAYS_TITLE:
                en = name
            else:
                en = en.lower()
            out[name] = en
        return out

    def display_word(self, label: str) -> str:
        return self._display.get(label, label.lower())

    def add_word(self, label: str, suppress_duplicates: bool = True) -> bool:
        """Append a sign. Returns False if it was a duplicate and rejected."""
        if suppress_duplicates and self._labels and self._labels[-1] == label:
            return False
        self._words.append(self.display_word(label))
        self._labels.append(label)
        return True

    def undo(self) -> bool:
        if not self._words:
            return False
        self._words.pop()
        self._labels.pop()
        return True

    def clear(self) -> None:
        self._words.clear()
        self._labels.clear()

    @property
    def labels(self) -> list[str]:
        return list(self._labels)

    @property
    def word_count(self) -> int:
        return len(self._words)

    def raw_text(self) -> str:
        return " ".join(self._words)

    def to_sentence(self) -> str:
        """Grammar-lite assembly: capitalise first word, append a full stop."""
        words = list(self._words)
        if not words:
            return ""
        text = words[0][0].upper() + words[0][1:] if words[0] else ""
        rest = " ".join(words[1:])
        if words[0].lower() in GREETING_COMMA and rest:
            text += ","
        sentence = (text + (" " + rest if rest else "")).strip()
        if sentence and not sentence.endswith((".", "?", "!")):
            sentence += "."
        return sentence

    def confirm(self) -> str:
        """Return the finished sentence (caller decides destination)."""
        return self.to_sentence()

    def __len__(self) -> int:
        return len(self._words)