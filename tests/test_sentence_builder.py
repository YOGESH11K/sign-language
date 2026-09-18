"""Tests for sentence building / grammar-lite assembly."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.sentence_builder import SentenceBuilder

REPO = Path(__file__).resolve().parent.parent
LABELS = REPO / "data" / "labels.json"


def make_builder() -> SentenceBuilder:
    return SentenceBuilder(LABELS)


def test_add_and_duplicate_suppression():
    b = make_builder()
    assert b.add_word("HELLO") is True
    assert b.add_word("HELLO") is False   # duplicate rejected
    assert b.add_word("MY") is True
    assert b.word_count == 2


def test_undo_and_clear():
    b = make_builder()
    b.add_word("HELLO")
    b.add_word("MY")
    assert b.undo() is True
    assert b.word_count == 1
    b.clear()
    assert b.word_count == 0
    assert b.undo() is False


def test_sentence_capitalisation():
    b = make_builder()
    for w in ["HELLO", "MY", "NAME", "IS", "RAHUL"]:
        b.add_word(w)
    assert b.to_sentence() == "Hello, my name is Rahul."


def test_I_keeps_capital():
    b = make_builder()
    b.add_word("I")
    b.add_word("YOU")
    assert b.to_sentence() == "I you."


def test_empty_sentence():
    b = make_builder()
    assert b.to_sentence() == ""


def test_unknown_label_falls_back_to_lowercase():
    b = make_builder()
    b.add_word("SOMETHING_NEW")
    assert b.raw_text() == "something_new"


def test_confirm_returns_sentence():
    b = make_builder()
    b.add_word("HELLO")
    assert b.confirm() == "Hello."