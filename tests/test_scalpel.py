"""Tests de la couche Scalpel (neurogenesis et renforcement LCT).

Vérifie que le Scalpel génère des neurones, les renforce par LCT, et que la base
de données de mots est séparée du réseau de neurones.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ratis_net.scalpel import ScalpelLayer, ScalpelNeuron
from ratis_net.glove_tokenizer import GloveTokenizer


@pytest.fixture
def tokenizer():
    return GloveTokenizer(dim=12, n_glove=8)


@pytest.fixture
def scalpel(tokenizer):
    return ScalpelLayer(tokenizer, eta=0.1, coherence_threshold=0.3)


def test_neurogenesis_creates_new_neurons(scalpel):
    """La première fois qu'une paire cohérente est vue, un neurone est généré."""
    r = scalpel.process_phrase("the brain processes information")
    assert r["generated"] > 0
    assert scalpel.network_size() > 0


def test_reinforcement_increases_weight_not_neurons(scalpel):
    """La 2e fois, le neurone est renforcé, pas de neurogenesis."""
    phrase = "the brain processes information"
    scalpel.process_phrase(phrase)
    size_before = scalpel.network_size()
    r = scalpel.process_phrase(phrase)
    size_after = scalpel.network_size()
    assert r["reinforced"] > 0
    assert r["generated"] == 0
    assert size_before == size_after


def test_lct_weight_grows_with_reinforcement(scalpel):
    """Le poids d'une corrélation augmente avec le nombre de renforcements (LCT)."""
    # happy ↔ love a un cosinus positif (cohérent)
    phrase = "i love happy days"
    scalpel.process_phrase(phrase, t_step=0)
    corrs = scalpel.get_correlations("happy")
    assert len(corrs) > 0
    weight_1 = corrs[0][1]
    # renforcer encore
    for t in range(1, 5):
        scalpel.process_phrase(phrase, t_step=t)
    corrs_2 = scalpel.get_correlations("happy")
    weight_2 = corrs_2[0][1]
    assert weight_2 > weight_1


def test_low_coherence_pairs_are_not_stored(scalpel):
    """Les paires non cohérentes (cos < threshold) ne génèrent pas de neurone."""
    r = scalpel.process_phrase("xyzzy qwerty asdfgh")
    # ces mots n'existent pas dans GloVe → embeddings nuls → cohérence 0
    assert r["generated"] == 0


def test_correlations_are_symmetric(scalpel):
    """Si A↔B existe, get_correlations(A) contient B et vice-versa."""
    scalpel.process_phrase("happy love today")
    a = [c[0] for c in scalpel.get_correlations("happy")]
    b = [c[0] for c in scalpel.get_correlations("love")]
    assert "love" in a
    assert "happy" in b


def test_database_is_separate_from_network(scalpel):
    """La base de données de mots (tokenizer) n'est pas le réseau (neurons)."""
    assert hasattr(scalpel, "tokenizer")
    assert hasattr(scalpel, "neurons")
    assert scalpel.tokenizer is not scalpel.neurons
    assert isinstance(scalpel.neurons, dict)
    assert scalpel.network_size() == len(scalpel.neurons)


def test_strongest_correlations_sorted(scalpel):
    """Les corrélations les plus fortes sont triées par poids décroissant."""
    phrases = ["i am happy", "i am very happy", "i am so happy today"]
    for p in phrases:
        scalpel.process_phrase(p)
    strongest = scalpel.strongest_correlations(top_k=5)
    weights = [w for _, _, w in strongest]
    assert weights == sorted(weights, reverse=True)


def test_save_and_load_preserves_network(scalpel, tmp_path):
    """Le réseau Scalpel peut être sauvegardé et rechargé."""
    scalpel.process_phrase("the brain processes information")
    scalpel.process_phrase("the brain processes information")
    path = tmp_path / "scalpel_test.pkl"
    scalpel.save(path)
    new_scalpel = ScalpelLayer(scalpel.tokenizer)
    new_scalpel.load(path)
    assert new_scalpel.network_size() == scalpel.network_size()
    assert new_scalpel.total_reinforcements == scalpel.total_reinforcements
    assert new_scalpel.total_neurogenesis == scalpel.total_neurogenesis
