"""ratis_net.scalpel — Couche de découpe, neurogenesis et corrélation.

La couche Scalpel opère pendant la formation : pour chaque phrase cohérente
entrée dans le système, elle :

  1. Découpe la phrase en paires de mots cohérents (scalpel = découpe fine).
  2. Renforce les poids des corrélations existantes (LCT : ΔW = η·φ·P_sig·C).
  3. Génère de nouveaux neurones quand une corrélation inconnue apparaît
     (neurogenesis — le réseau grandit, ne se contente pas de s'entraîner).
  4. Stocke les corrélations dans une base personnelle locale (pas le réseau
     de neurones — les deux sont séparés, comme Jonathan l'a spécifié).

La base de données de mots n'est PAS le réseau de neurones :
  - Base de données : réservoir passif d'embeddings (GloVe + Topo).
  - Réseau Scalpel  : graphe actif de neurones-corrélations qui grandit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    from ratis_net.glove_tokenizer import GloveTokenizer, glove_topo_signature
except ImportError:
    from glove_tokenizer import GloveTokenizer, glove_topo_signature


# ─────────────────────────────────────────────────────────────────────────────
# Le neurone Scalpel (une corrélation = un neurone)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScalpelNeuron:
    """Un neurone-corrélation : relie deux mots (ou fragments) par un poids LCT.

    Chaque neurone stocke :
      - word_a, word_b : les deux mots corrélés
      - weight : le poids de la corrélation (renforcé par LCT)
      - p_sig : persistance topologique de la paire (signal LCT)
      - coherence : cohérence mesurée (C, modulée par φ)
      - n_reinforcements : nombre de fois que cette corrélation a été renforcée
    """
    word_a: str
    word_b: str
    weight: float = 0.0
    p_sig: float = 0.0
    coherence: float = 0.0
    n_reinforcements: int = 0

    def reinforce(self, eta: float, phi: float, p_sig: float, c: float) -> None:
        """Renforce le poids par la loi LCT : ΔW = η · φ · P_sig · C."""
        delta = eta * phi * p_sig * c
        self.weight += delta
        self.p_sig = max(self.p_sig, p_sig)
        self.coherence = max(self.coherence, c)
        self.n_reinforcements += 1


# ─────────────────────────────────────────────────────────────────────────────
# La couche Scalpel
# ─────────────────────────────────────────────────────────────────────────────

class ScalpelLayer:
    """Couche de découpe, renforcement et neurogenesis.

    Pendant la formation, pour chaque phrase cohérente :
      1. Découpe en paires de mots adjacents (le scalpel).
      2. Pour chaque paire, calcule P_sig et C (cohérence topologique).
      3. Si la paire existe déjà → renforce le poids (LCT).
      4. Si la paire est nouvelle → génère un nouveau neurone (neurogenesis).
      5. Stocke les corrélations dans la base (séparée du réseau).
    """

    def __init__(self, tokenizer: GloveTokenizer, eta: float = 0.1,
                 omega: float = math.pi / 2, coherence_threshold: float = 0.3,
                 seed: int = 42):
        self.tokenizer = tokenizer
        self.eta = eta
        self.omega = omega
        self.coherence_threshold = coherence_threshold
        self.rng = np.random.default_rng(seed)
        # le réseau de neurones-corrélations (grandit, pas statique)
        self.neurons: dict[tuple[str, str], ScalpelNeuron] = {}
        # statistiques de neurogenesis
        self.total_reinforcements = 0
        self.total_neurogenesis = 0

    def _pair_key(self, a: str, b: str) -> tuple[str, str]:
        """Clé canonique pour une paire (ordre indépendant)."""
        return (a, b) if a <= b else (b, a)

    def _compute_p_sig(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """P_sig d'une paire = norme de la composante topo de l'embedding moyen."""
        combined = (emb_a + emb_b) / 2.0
        topo = combined[self.tokenizer.n_glove:]
        return float(np.linalg.norm(topo))

    def _compute_coherence(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """Cohérence = similarité cosinus entre les composantes GloVe."""
        glove_a = emb_a[:self.tokenizer.n_glove]
        glove_b = emb_b[:self.tokenizer.n_glove]
        na = np.linalg.norm(glove_a) + 1e-9
        nb = np.linalg.norm(glove_b) + 1e-9
        return float(np.dot(glove_a, glove_b) / (na * nb))

    def process_phrase(self, phrase: str, t_step: int = 0) -> dict[str, Any]:
        """Découpe une phrase, renforce ou génère les neurones-corrélations.

        Retourne les statistiques de cette découpe (scalpel).
        """
        words = phrase.lower().strip().split()
        if len(words) < 2:
            return {"n_pairs": 0, "reinforced": 0, "generated": 0, "phrase": phrase}

        # embeddings de tous les mots
        embeddings = [self.tokenizer(w, self.tokenizer.dim) for w in words]
        phi = abs(math.cos(self.omega * t_step))

        reinforced = 0
        generated = 0
        # le scalpel découpe en paires adjacentes
        for i in range(len(words) - 1):
            a, b = words[i], words[i + 1]
            key = self._pair_key(a, b)
            emb_a, emb_b = embeddings[i], embeddings[i + 1]
            p_sig = self._compute_p_sig(emb_a, emb_b)
            c = self._compute_coherence(emb_a, emb_b)

            # seulement les paires cohérentes (au-dessus du seuil)
            if c < self.coherence_threshold:
                continue

            if key in self.neurons:
                # renforcement (LCT)
                self.neurons[key].reinforce(self.eta, phi, p_sig, c)
                reinforced += 1
                self.total_reinforcements += 1
            else:
                # neurogenesis : nouveau neurone-corrélation
                neuron = ScalpelNeuron(word_a=key[0], word_b=key[1],
                                        weight=self.eta * phi * p_sig * c,
                                        p_sig=p_sig, coherence=c, n_reinforcements=1)
                self.neurons[key] = neuron
                generated += 1
                self.total_neurogenesis += 1

        return {"n_pairs": len(words) - 1, "reinforced": reinforced,
                "generated": generated, "phrase": phrase}

    def train_corpus(self, corpus: list[str], epochs: int = 1,
                     verbose: bool = True) -> dict[str, Any]:
        """Traite un corpus entier : le scalpel découpe chaque phrase."""
        for ep in range(epochs):
            n_reinforced = 0
            n_generated = 0
            for phrase in corpus:
                r = self.process_phrase(phrase, t_step=ep)
                n_reinforced += r["reinforced"]
                n_generated += r["generated"]
            if verbose:
                print(f"  Epoch {ep}: {n_reinforced} renforcements, "
                      f"{n_generated} neurogenesis, "
                      f"{len(self.neurons)} neurones totaux")
        return {"total_neurons": len(self.neurons),
                "total_reinforcements": self.total_reinforcements,
                "total_neurogenesis": self.total_neurogenesis}

    def get_correlations(self, word: str) -> list[tuple[str, float, float]]:
        """Retourne les corrélations d'un mot : (autre_mot, poids, p_sig)."""
        results = []
        for (a, b), neuron in self.neurons.items():
            if a == word:
                results.append((b, neuron.weight, neuron.p_sig))
            elif b == word:
                results.append((a, neuron.weight, neuron.p_sig))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def strongest_correlations(self, top_k: int = 10) -> list[tuple[str, str, float]]:
        """Les corrélations les plus fortes du réseau."""
        results = [(n.word_a, n.word_b, n.weight)
                    for n in self.neurons.values()]
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]

    def network_size(self) -> int:
        """Nombre de neurones-corrélations dans le réseau."""
        return len(self.neurons)

    def save(self, path: Path) -> None:
        """Sauvegarde le réseau Scalpel (base de données séparée du modèle)."""
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"neurons": self.neurons, "eta": self.eta,
                         "omega": self.omega,
                         "coherence_threshold": self.coherence_threshold,
                         "total_reinforcements": self.total_reinforcements,
                         "total_neurogenesis": self.total_neurogenesis}, f)

    def load(self, path: Path) -> None:
        """Charge le réseau Scalpel."""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.neurons = data["neurons"]
        self.eta = data["eta"]
        self.omega = data["omega"]
        self.coherence_threshold = data["coherence_threshold"]
        self.total_reinforcements = data["total_reinforcements"]
        self.total_neurogenesis = data["total_neurogenesis"]


# ─────────────────────────────────────────────────────────────────────────────
# Test rapide
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tok = GloveTokenizer(dim=12, n_glove=8)
    scalpel = ScalpelLayer(tok, eta=0.1, coherence_threshold=0.3)

    corpus = [
        "the quantum mechanics is fascinating",
        "quantum entanglement is a quantum phenomenon",
        "the brain processes information through neurons",
        "neural networks learn by adjusting weights",
        "i am happy to see you today",
        "the weather is beautiful and i feel happy",
        "gravity is a fundamental force of nature",
        "the universe is expanding and gravity matters",
    ]
    print("=== Scalpel : découpe et neurogenesis ===")
    scalpel.train_corpus(corpus, epochs=3)

    print(f"\nRéseau: {scalpel.network_size()} neurones-corrélations")
    print(f"Renforcements: {scalpel.total_reinforcements}")
    print(f"Neurogenesis: {scalpel.total_neurogenesis}")

    print("\n=== Corrélations les plus fortes ===")
    for a, b, w in scalpel.strongest_correlations(top_k=5):
        print(f"  {a:12s} ↔ {b:12s}  poids={w:.4f}")

    print("\n=== Corrélations de 'quantum' ===")
    for word, weight, p_sig in scalpel.get_correlations("quantum"):
        print(f"  quantum ↔ {word:12s}  poids={weight:.4f}  P_sig={p_sig:.4f}")
