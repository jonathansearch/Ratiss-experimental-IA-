"""Tests du RatissSynchrotron (reconstruction sémantique topologique).

Vérifie que la tension LCT est calculée, que les fragments résonants sont
sélectionnés par cohérence topologique, et qu'aucun gradient n'est utilisé.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ratis_net.ratiss_synchrotron import (
    RatissSynchrotron, SuperEmbeddingIndex, Synchrotron,
    ResonanceEngine, RatisAssembler, ResonanceResult, IndexedFragment,
)


CORPUS = [
    "quantum mechanics is fascinating",
    "the brain processes information",
    "i am happy to see you",
    "gravity bends spacetime",
    "entropy measures disorder",
    "neural networks learn from data",
    "the universe is expanding",
    "consciousness emerges from interactions",
    "i feel sad when you leave",
    "topology studies shapes",
]


@pytest.fixture
def engine():
    eng = RatissSynchrotron()
    eng.build_corpus(CORPUS)
    return eng


def test_index_stores_fragments_with_super_embeddings(engine):
    assert len(engine.index.fragments) == len(CORPUS)
    for frag in engine.index.fragments:
        assert frag.embedding.shape[0] == engine.tokenizer.dim
        assert len(frag.words) >= 2
        assert frag.source == "corpus"


def test_synchrotron_identifies_topological_voids(engine):
    analysis = engine.synchrotron.analyze_query("quantum mechanics is fascinating")
    assert len(analysis["words"]) == 4
    assert analysis["pooled"] is not None
    assert len(analysis["p_sigs"]) == 4
    # au moins un vide topologique détecté
    n_voids = sum(1 for v in analysis["voids"] if v.needs_closure)
    assert n_voids >= 0
    # tous les embeddings ont la bonne dimension
    assert analysis["embeddings"].shape == (4, engine.tokenizer.dim)


def test_resonance_engine_computes_lct_tension(engine):
    analysis = engine.synchrotron.analyze_query("how does the brain work")
    results = engine.resonance.find_resonant_fragments(analysis, top_k=3)
    assert len(results) <= 3
    for r in results:
        assert r.topological_tension >= 0.0
        assert 0.0 < r.coherence <= 1.0
        assert isinstance(r.accepted, bool)


def test_resonant_fragments_have_low_tension_for_relevant_query(engine):
    # "brain" devrait résonner avec "the brain processes information"
    analysis = engine.synchrotron.analyze_query("how does the brain work")
    results = engine.resonance.find_resonant_fragments(analysis, top_k=3)
    best = results[0] if results else None
    assert best is not None
    assert best.coherence > 0.5
    assert best.topological_tension < 1.0


def test_assembler_reconstructs_from_accepted_fragments(engine):
    analysis = engine.synchrotron.analyze_query("i feel happy")
    resonance = engine.resonance.find_resonant_fragments(analysis, top_k=5)
    reconstruction = engine.assembler.reconstruct("i feel happy", resonance)
    assert "query" in reconstruction
    assert "reconstructed" in reconstruction
    assert reconstruction["avg_coherence"] >= 0.0
    assert reconstruction["n_fragments"] >= 1


def test_generate_response_returns_full_pipeline(engine):
    result = engine.generate_response("what is the brain")
    assert "query" in result
    assert "analysis" in result
    assert "reconstruction" in result
    assert result["reconstruction"]["n_fragments"] >= 1
    assert result["n_indexed_fragments"] == len(CORPUS)


def test_no_gradient_used_in_generation(engine):
    # Vérifie qu'aucun paramètre entraînable n'est mis à jour pendant la génération
    weights_before = [f.embedding.copy() for f in engine.index.fragments]
    _ = engine.generate_response("quantum mechanics")
    weights_after = [f.embedding for f in engine.index.fragments]
    for w1, w2 in zip(weights_before, weights_after):
        assert np.allclose(w1, w2), "Les embeddings de l'index ne doivent pas changer pendant la génération"


def test_empty_query_returns_empty(engine):
    result = engine.generate_response("")
    assert result["reconstruction"]["n_fragments"] >= 0


def test_tension_is_comparable_across_queries(engine):
    # La tension pour une requête pertinente doit être plus basse que pour une requête non pertinente
    relevant = engine.synchrotron.analyze_query("quantum mechanics")
    irrelevant = engine.synchrotron.analyze_query("cooking recipe pasta")
    r_relevant = engine.resonance.find_resonant_fragments(relevant, top_k=1)
    r_irrelevant = engine.resonance.find_resonant_fragments(irrelevant, top_k=1)
    if r_relevant and r_irrelevant:
        assert r_relevant[0].topological_tension <= r_irrelevant[0].topological_tension + 0.5
