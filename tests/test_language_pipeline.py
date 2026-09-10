"""tests/test_language_pipeline.py — Pipeline de langage v2 (unitaire).

Ces tests ne chargent PAS le checkpoint Scalpel (294 MB) : ils utilisent un
index synthétique et les vrais fichiers de grammaire du dépôt. Le test de
bout en bout sur le vrai checkpoint est dans test_language_quality.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from ratis_net.query_analyzer import analyze, detect_language
from ratis_net.intent_router import route
from ratis_net.concept_ranker import ConceptRanker
from ratis_net.chain_reasoning import find_chains
from ratis_net.integrity_proof import prove, verify


# ── query_analyzer ──────────────────────────────────────────────────────────

def test_detect_language_french():
    assert detect_language("qu'est-ce que la photosynthèse ?") == "fr"
    assert detect_language("bonjour, comment ça va ?") == "fr"


def test_detect_language_english():
    assert detect_language("what is quantum mechanics?") == "en"
    assert detect_language("hello, how are you?") == "en"


def test_qtype_greeting():
    assert analyze("hello, how are you?").qtype == "greeting"
    assert analyze("bonjour !").qtype == "greeting"


def test_qtype_identity():
    assert analyze("what is your name?").qtype == "identity"
    assert analyze("qui es-tu ?").qtype == "identity"


def test_qtype_definition_and_explanation():
    assert analyze("what is a black hole").qtype == "definition"
    assert analyze("why does the sun shine").qtype == "explanation"
    assert analyze("how does DNA work").qtype == "explanation"


def test_qtype_gratitude_and_capability():
    assert analyze("thank you very much").qtype == "gratitude"
    assert analyze("can you help me?").qtype == "capability"


def test_compound_extraction():
    a = analyze("what is a black hole")
    assert "black hole" in a.compounds
    assert "black" in a.keywords and "hole" in a.keywords


def test_social_words_not_keywords():
    a = analyze("hello, how are you?")
    assert "hello" not in a.keywords
    a2 = analyze("thank you very much")
    assert "very" not in a2.keywords and "thank" not in a2.keywords


# ── intent_router ───────────────────────────────────────────────────────────

def test_route_social():
    r = route(analyze("hello, how are you?"))
    assert r.social and r.context == "casual_chat" and r.conv_intent == "greet"


def test_route_scientific_domain():
    r = route(analyze("what is quantum mechanics"))
    assert not r.social
    assert r.domain == "quantum_information"
    assert r.intention == "define"


def test_route_biology_to_medicine():
    r = route(analyze("what is a protein"))
    assert r.domain == "medicine"


def test_route_explanation_intent():
    r = route(analyze("how does photosynthesis work"))
    assert r.intention == "explain"


# ── concept_ranker ──────────────────────────────────────────────────────────

def _synthetic_index() -> dict:
    # "protein" relié à des termes bio (forts) et à "the" (très fort mais ubiquitaire)
    return {
        "protein": [("structure", 0.9, 0.5), ("kinase", 0.8, 0.4),
                    ("binding", 0.7, 0.3), ("the", 5.0, 0.9)],
        "structure": [("protein", 0.9, 0.5)],
        "kinase": [("protein", 0.8, 0.4)],
        "binding": [("protein", 0.7, 0.3)],
        "the": [("protein", 5.0, 0.9)] + [(f"w{i}", 0.1, 0.1) for i in range(200)],
    }


def test_ranker_excludes_stopwords_and_ubiquitous():
    ranker = ConceptRanker(_synthetic_index())
    concepts = ranker.rank(["protein"], n=5, glove_n=0)
    assert "the" not in concepts
    assert concepts[0] in {"structure", "kinase", "binding"}


def test_ranker_idf_prefers_specific_terms():
    idx = _synthetic_index()
    # "ubiq" co-occurrent fort mais connecté à tout (degré élevé)
    idx["protein"].append(("ubiq", 10.0, 0.9))
    idx["ubiq"] = [(f"x{i}", 0.1, 0.1) for i in range(500)]
    ranker = ConceptRanker(idx)
    concepts = ranker.rank(["protein"], n=5, glove_n=0)
    assert "ubiq" not in concepts[:3]


def test_ranker_shared_neighbor_bonus():
    idx = _synthetic_index()
    idx["dna"] = [("structure", 0.4, 0.2)]
    idx["structure"].append(("dna", 0.4, 0.2))
    ranker = ConceptRanker(idx)
    concepts = ranker.rank(["protein", "dna"], n=5, glove_n=0)
    assert concepts[0] == "structure"  # voisin partagé des deux mots-clés


# ── chain_reasoning ─────────────────────────────────────────────────────────

def test_chain_finds_path():
    idx = {
        "a": [("b", 1.0, 0.5)],
        "b": [("a", 1.0, 0.5), ("c", 0.8, 0.4)],
        "c": [("b", 0.8, 0.4)],
    }
    chains = find_chains("a", "c", idx)
    assert chains and chains[0].path == ["a", "b", "c"]
    assert chains[0].kind == "association_chain"


def test_chain_skips_stopword_links():
    idx = {
        "a": [("of", 5.0, 0.9), ("b", 0.5, 0.3)],
        "of": [("a", 5.0, 0.9), ("c", 5.0, 0.9)],
        "b": [("a", 0.5, 0.3), ("c", 0.5, 0.3)],
        "c": [("of", 5.0, 0.9), ("b", 0.5, 0.3)],
    }
    chains = find_chains("a", "c", idx)
    assert chains
    assert "of" not in chains[0].path


def test_chain_empty_when_unreachable():
    assert find_chains("x", "y", {"x": []}) == []


# ── integrity_proof ─────────────────────────────────────────────────────────

class _FakeNeuron:
    def __init__(self, w, p, c):
        self.weight, self.p_sig, self.coherence = w, p, c


class _FakeScalpel:
    neurons = {("quantum", "theory"): _FakeNeuron(0.5, 0.3, 0.8),
               ("gravity", "quantum"): _FakeNeuron(0.2, 0.1, 0.6)}


def test_proof_roundtrip():
    scalpel = _FakeScalpel()
    p = prove(["quantum"], scalpel)
    assert p.n_edges == 2
    assert verify(p, scalpel)
    assert p.proof_type == "integrity_commitment"
    assert p.to_dict()["zk_stark"] is False


def test_proof_changes_with_concepts():
    scalpel = _FakeScalpel()
    p1 = prove(["quantum"], scalpel)
    p2 = prove(["gravity"], scalpel)
    assert p1.digest != p2.digest
