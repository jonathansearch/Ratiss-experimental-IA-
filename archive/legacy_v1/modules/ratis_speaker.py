"""ratis_net.ratis_speaker — Génération de texte mot par mot via le Scalpel.

Au lieu d'assembler des phrases entières (Synchrotron), le speaker suit les
corrélations apprises par le Scalpel pour choisir le mot suivant. C'est un
random walk biaisé sur le graphe de neurones-corrélations, sans gradient ni
Transformer.

Algorithme :
  1. Partir d'un mot de départ (ou d'une requête).
  2. Récupérer les corrélations du mot courant (get_correlations).
  3. Choisir le mot suivant = poids LCT le plus fort (avec une part d'aléa
     pour la diversité, et une pénalité de répétition pour éviter les boucles).
  4. Répéter jusqu'à N mots ou fin des corrélations.

Honnête : on ne génère rien de nouveau. On suit les chemins appris par le
Scalpel. C'est l'équivalent topologique d'un language model, sans gradient.
"""
from __future__ import annotations

import math
import random
from typing import Any

import numpy as np

try:
    from ratis_net.scalpel import ScalpelLayer
except ImportError:
    from scalpel import ScalpelLayer


class RatisSpeaker:
    """Générateur de texte mot par mot basé sur les corrélations Scalpel.

    Le speaker parcourt le graphe de neurones-corrélations. À chaque pas, il
    choisit le mot suivant parmi les corrélations du mot courant, biaisé par le
    poids LCT. Un mécanisme anti-boucle pénalise les mots déjà utilisés.
    """

    def __init__(self, scalpel: ScalpelLayer, temperature: float = 0.3,
                 repetition_penalty: float = 0.5, max_words: int = 100,
                 seed: int = 42):
        self.scalpel = scalpel
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.max_words = max_words
        self.rng = random.Random(seed)
        # Index inversé : mot → liste de (corrélation, poids, p_sig)
        # Construit une fois pour éviter le scan de 3.78M neurones à chaque appel.
        self._index: dict[str, list[tuple[str, float, float]]] = {}
        self._indexed = False

    def build_index(self, verbose: bool = True) -> None:
        """Pré-construit l'index inversé mot→corrélations (une seule fois).

        Pour 3.78M neurones, ça prend ~30s mais ensuite chaque get_correlations
        est O(1) au lieu de O(n_neurons).
        """
        import time as _time
        t0 = _time.time()
        for (a, b), neuron in self.scalpel.neurons.items():
            entry = (b, neuron.weight, neuron.p_sig)
            if a in self._index:
                self._index[a].append(entry)
            else:
                self._index[a] = [entry]
            entry_b = (a, neuron.weight, neuron.p_sig)
            if b in self._index:
                self._index[b].append(entry_b)
            else:
                self._index[b] = [entry_b]
        # Trier chaque entrée par poids décroissant
        for k in self._index:
            self._index[k].sort(key=lambda x: x[1], reverse=True)
        self._indexed = True
        if verbose:
            print(f"  Index inversé: {len(self._index):,} mots en {_time.time()-t0:.1f}s")

    def _get_correlations(self, word: str) -> list[tuple[str, float, float]]:
        """Récupère les corrélations via l'index (O(1)) ou le scan (O(n))."""
        if self._indexed:
            return self._index.get(word, [])
        return self.scalpel.get_correlations(word)

    def _next_word(self, current: str, used: set[str],
                   topic_bias: str | None = None) -> str | None:
        """Choisit le mot suivant à partir des corrélations du mot courant.

        Sélection par poids LCT avec température (0 = déterministe, 1 = aléa
        complet). Les mots déjà utilisés sont pénalisés.
        """
        corrs = self._get_correlations(current)
        if not corrs:
            return None

        candidates = []
        for word, weight, p_sig in corrs:
            if word == current:
                continue
            # pénalité de répétition
            penalty = self.repetition_penalty if word in used else 1.0
            # biais thématique : si un mot-thème est donné, boost les corrélations
            # qui contiennent ce mot ou qui sont corrélées avec lui
            score = weight * penalty
            if topic_bias and topic_bias != current:
                topic_corrs = {w: wt for w, wt, _ in self._get_correlations(topic_bias)}
                if word in topic_corrs:
                    score *= 1.0 + topic_corrs[word] * 0.01
            candidates.append((word, max(score, 1e-9)))

        if not candidates:
            return None

        # softmax avec température
        weights = np.array([c[1] for c in candidates])
        if self.temperature > 0:
            logits = weights / (self.temperature + 1e-9)
            logits = np.clip(logits, -50, 50)  # éviter overflow
            exp = np.exp(logits - logits.max())
            probs = exp / exp.sum()
        else:
            probs = np.zeros(len(candidates))
            probs[weights.argmax()] = 1.0

        idx = self.rng.choices(range(len(candidates)), weights=probs, k=1)[0]
        return candidates[idx][0]

    def generate(self, seed_word: str, n_words: int = 50,
                 topic: str | None = None) -> str:
        """Génère n_words mots à partir de seed_word.

        Si topic est fourni, biais les choix vers les mots corrélés avec le topic.
        """
        used = {seed_word}
        current = seed_word
        words = [current]

        for _ in range(n_words - 1):
            nxt = self._next_word(current, used, topic_bias=topic)
            if nxt is None:
                break
            words.append(nxt)
            used.add(nxt)
            current = nxt

        return " ".join(words)

    def generate_paragraph(self, seed_words: list[str], n_words: int = 100,
                           topic: str | None = None) -> str:
        """Génère un paragraphe en relançant depuis plusieurs mots de départ.

        Évite la boucle en relançant depuis un nouveau mot quand le speaker
        s'arrête (fin des corrélations) ou après ~20 mots.
        """
        all_words = []
        used = set()
        n_generated = 0

        for seed in seed_words:
            if n_generated >= n_words:
                break
            remaining = n_words - n_generated
            current = seed
            all_words.append(current)
            used.add(current)
            n_generated += 1

            chunk_len = min(20, remaining - 1)
            for _ in range(chunk_len):
                nxt = self._next_word(current, used, topic_bias=topic)
                if nxt is None:
                    break
                all_words.append(nxt)
                used.add(nxt)
                current = nxt
                n_generated += 1
            if n_generated < n_words:
                all_words.append(".")

        # capitaliser le début des phrases
        text = " ".join(all_words)
        # nettoyer les espaces avant ponctuation
        text = text.replace(" .", ".").replace(" ,", ",")
        return text


if __name__ == "__main__":
    from pathlib import Path
    from ratis_net.glove_tokenizer import GloveTokenizer

    tok = GloveTokenizer(dim=12, n_glove=8)
    scalpel = ScalpelLayer(tok, eta=0.1, coherence_threshold=0.3)
    scalpel.load(Path("artifacts/scalpel_wikipedia.pkl"))
    print(f"Scalpel: {scalpel.network_size():,} neurons\n")

    speaker = RatisSpeaker(scalpel, temperature=0.2, max_words=100)
    speaker.build_index()

    print("=== Generation from single words ===")
    for seed in ["quantum", "science", "love", "gravity", "brain"]:
        text = speaker.generate(seed, n_words=30)
        print(f"\n[{seed}]")
        print(f"  {text}")

    print("\n\n=== Paragraph generation (multiple seeds) ===")
    para = speaker.generate_paragraph(["the", "quantum", "science"], n_words=80, topic="science")
    print(f"\n{para}")

    print("\n\n=== Paragraph with topic: love ===")
    para2 = speaker.generate_paragraph(["love", "the", "heart"], n_words=60, topic="love")
    print(f"\n{para2}")
