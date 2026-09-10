"""tests/test_language_quality.py — Tests de langage bout en bout.

Charge le vrai checkpoint Scalpel (294 MB, Git LFS) et vérifie la qualité
des réponses sur trois registres : conversation, science, biologie. C'est le
test que le projet appelle "test de langue" : il ne mesure pas la vérité
scientifique (rôle des knowledge packs), il mesure la PERTINENCE du routage
et du classement des concepts.

Le test est sauté si le checkpoint n'est pas présent (ex: CI sans LFS).
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts" / "scalpel_wikipedia.pkl"

pytestmark = pytest.mark.skipif(
    not CHECKPOINT.exists() or CHECKPOINT.stat().st_size < 1_000_000,
    reason="checkpoint Scalpel absent (git lfs pull requis)")


@pytest.fixture(scope="module")
def net():
    from ratis_net import RatisNet
    n = RatisNet()
    n.load_scalpel(CHECKPOINT, verbose=False)
    n.load_grammar(verbose=False)
    n.load_knowledge_packs(verbose=False)
    n.build_index(verbose=False)
    return n


# ── Conversation ────────────────────────────────────────────────────────────

def test_greeting_routes_to_conversation(net):
    r = net.speaker.generate_response("hello, how are you?")
    assert r["route"] == "conversation"
    assert r["skeleton_intent"] == "greet"
    # la réponse ne doit pas être un gabarit scientifique
    assert "scientific frame" not in r["sentence"]


def test_french_greeting_stays_french(net):
    r = net.speaker.generate_response("bonjour !")
    assert r["language"] == "fr"
    assert any(w in r["sentence"].lower()
               for w in ("ravi", "bonjour", "merci", "heureux"))


def test_identity_is_fixed_and_honest(net):
    s = net.respond("what is your name?")
    assert "RATIS-Net" in s
    assert "Jonathan Evina" in s and "JOHNKING0" in s
    assert "Topological Coherence" in s


def test_capability_is_honest(net):
    s = net.respond("can you help me?")
    assert "correlations" in s or "corrélations" in s


# ── Science ─────────────────────────────────────────────────────────────────

def test_black_hole_gives_verified_fact(net):
    r = net.respond_with_science("what is a black hole")
    assert any("black hole" in f["text"].lower() and "event horizon" in f["text"].lower()
               for f in r["knowledge_facts"])


def test_dna_gives_verified_fact(net):
    r = net.respond_with_science("how does DNA work")
    assert any("double helix" in f["text"].lower()
               for f in r["knowledge_facts"])


def test_quantum_concepts_are_physics(net):
    concepts = net.concepts("quantum", n=8)
    physics = {"mechanical", "theory", "mechanics", "physics", "relativity",
               "electrodynamics", "computing", "gravity", "statistical",
               "applied", "modeling", "loop", "molecular", "information"}
    assert any(c in physics for c in concepts), concepts


def test_protein_concepts_are_biology(net):
    concepts = net.concepts("protein", n=8)
    bio = {"structure", "kinase", "binding", "synthesis", "folding",
           "domain", "domains", "sequences", "enzyme", "amino"}
    assert any(c in bio for c in concepts), concepts


def test_virus_gives_verified_fact(net):
    r = net.respond_with_science("what is a virus")
    assert any("host cell" in f["text"].lower()
               for f in r["knowledge_facts"])


# ── Chaînes et preuves ──────────────────────────────────────────────────────

def test_chain_between_related_concepts(net):
    chains = net.chain("quantum", "gravity", max_hops=3)
    assert chains
    assert chains[0]["kind"] == "association_chain"
    assert "of" not in chains[0]["path"]


def test_proof_roundtrip_on_checkpoint(net):
    p = net.prove(["quantum", "mechanics"])
    assert p["n_edges"] > 0
    assert net.verify_proof(p)


# ── Statistiques ────────────────────────────────────────────────────────────

def test_stats_report_full_ecosystem(net):
    s = net.stats()
    assert s["scalpel_neurons"] > 3_000_000
    assert s["knowledge_packs"] >= 7
    assert s["grammar_loaded"] and s["conversation_loaded"]
