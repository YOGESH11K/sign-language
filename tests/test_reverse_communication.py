"""Tests for reverse communication: text -> sign gloss tokenisation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.reverse_communication import SignDictionary, SignDictionaryService

REPO = Path(__file__).resolve().parent.parent


def service() -> SignDictionaryService:
    return SignDictionaryService(SignDictionary(REPO / "data" / "labels.json"))


def test_known_words_map_to_glosses():
    svc = service()
    tokens = svc.to_sign_sequence("HELLO HOW ARE YOU")
    assert [t.gloss for t in tokens] == ["HELLO", "HOW", "ARE", "YOU"]


def test_lowercase_and_punctuation():
    svc = service()
    tokens = svc.to_sign_sequence("hello, my name is rahul!")
    assert [t.gloss for t in tokens] == ["HELLO", "MY", "NAME", "IS", "RAHUL"]


def test_phrase_greedy_matching():
    svc = service()
    tokens = svc.to_sign_sequence("thank you")
    assert [t.gloss for t in tokens] == ["THANK_YOU"]
    assert len(tokens) == 1


def test_unknown_words_fingerspell_with_letters():
    svc = service()
    tokens = svc.to_sign_sequence("HELLO XYZ")
    glosses = [t.gloss for t in tokens]
    assert glosses[0] == "HELLO"
    assert glosses[1:] == ["X", "Y", "Z"]
    assert tokens[1].letters is True


def test_empty_input():
    svc = service()
    assert svc.to_sign_sequence("   ") == []
    assert svc.to_sign_sequence("") == []


def test_display_output():
    svc = service()
    tokens = svc.to_sign_sequence("HELLO YOU")
    assert tokens[0].display() == "HELLO"
    assert tokens[1].display() == "YOU"


def test_dictionary_words_in_labels_are_aliased():
    d = SignDictionary(REPO / "data" / "labels.json")
    assert d.lookup_word("Water") == "WATER"
    assert d.lookup_word("hospital") == "HOSPITAL"
    assert d.lookup_word("coffee") is None