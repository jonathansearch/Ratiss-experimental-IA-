"""tests/test_neural_speaker.py — Le cerveau parle (reconstruction neuronale).

Vérifie que le NeuralSpeaker reconstruit des phrases à partir du chemin de
concepts activé par le Scalpel, pas en recitant : la couverture IDF mesure
la part des mots de contenu reconnus par le réseau.
"""
from __future__ import annotations

import pytest

from ratis_net.neural_speaker import NeuralSpeaker


def _toy_graph() -> dict:
    # petit graphe : protein relié à kinase, structure, synthesis
    def n(w=0.5, p=0.4, r=10):
        return (w, p, r)
    return {
        "protein": [("kinase", *n(0.9)), ("structure", *n(0.8)),
                    ("synthesis", *n(0.7)), ("cell", *n(0.5))],
        "kinase": [("protein", *n(0.9)), ("enzyme", *n(0.6))],
        "structure": [("protein", *n(0.8)), ("folding", *n(0.5))],
        "synthesis": [("protein", *n(0.7))],
        "cell": [("protein", *n(0.5))],
        "enzyme": [("kinase", *n(0.6))],
        "folding": [("structure", *n(0.5))],
        "dog": [("bark", *n(0.9))],
        "bark": [("dog", *n(0.9))],
    }


def _toy_corpus() -> list[str]:
    return [
        "a protein is a chain of amino acids folded into a working structure",
        "the kinase phosphorylates its target protein during cell signaling",
        "a dog is a loyal animal that barks at strangers",
        "the sun rises every morning over the quiet hills",
    ]


@pytest.fixture
def speaker():
    return NeuralSpeaker(_toy_graph(), _toy_corpus())


def test_concept_path_uses_graph(speaker):
    path = speaker.concept_path(["protein"], n=5)
    assert "kinase" in path or "structure" in path
    assert "the" not in path and "a" not in path


def test_speak_reconstructs_related_sentence(speaker):
    r = speaker.speak("what is a protein")
    assert r["ok"], r.get("reason")
    # la phrase choisie parle de protéine (reconnue par le chemin)
    assert "protein" in r["sentence"].lower()
    # éligible par couverture OU par définition du sujet
    assert r["coverage"] >= 0.15


def test_speak_honest_when_unknown(speaker):
    r = speaker.speak("what is a zorglub")
    assert not r["ok"]
    assert "absent" in r["reason"] or "couverture" in r["reason"]


def test_speak_prefers_definition(speaker):
    # « a protein is ... » doit battre « the kinase phosphorylates ... »
    r = speaker.speak("what is a protein")
    assert r["ok"]
    assert "is a chain" in r["sentence"]


def test_sentence_ends_with_punctuation(speaker):
    r = speaker.speak("what is a protein")
    assert r["sentence"][-1] in ".!?"
