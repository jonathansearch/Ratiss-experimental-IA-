"""ratis_net.trigrammar — Couche de tri-grammaire pour le speaker.

Le Scalpel capture les bigrammes (paires de mots). Le speaker bigramme génère
du texte fragmenté. La couche tri-grammaire améliore la cohérence en regardant
les DEUX derniers mots au lieu d'un seul, sans stocker de trigrammes.

Principe : quand le speaker choisit le mot suivant, il repondère les candidats
en combinant :
  - score_direct = poids(prev → candidat)        [bigramme normal]
  - score_contexte = poids(prev_prev → candidat) [co-occurrence à distance 2]

Score final = score_direct + context_weight * score_contexte

Ça ne nécessite PAS de ré-entraîner le Scalpel. On utilise l'index bigramme
existant mais avec une fenêtre de contexte de 2 mots. Le mot suivant est
choisi en fonction de ce qui précède, pas seulement du mot immédiat.
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


# Mots vides fréquents qui ne portent pas de sens mais structurent la phrase
STOPWORDS = {"the", "a", "an", "of", "in", "and", "to", "is", "was", "were",
             "that", "this", "it", "for", "on", "with", "as", "by", "at",
             "from", "or", "be", "has", "have", "had", "but", "not", "he",
             "she", "his", "her", "its", "their", "they", "which", "who",
             "are", "been", "also", "than", "then", "so", "if", "can", "will",
             "would", "could", "should", "may", "might", "must", "do", "does",
             "did", "no", "yes", "all", "any", "some", "each", "every", "such"}


class TriGrammarSpeaker:
    """Speaker avec fenêtre de contexte de 2 mots (tri-grammaire).

    Le mot suivant est choisi en combinant le bigramme direct (prev → cand)
    et la co-occurrence à distance 2 (prev_prev → cand). Ça produit des
    séquences plus cohérentes car le choix dépend de 2 mots de contexte.
    """

    def __init__(self, scalpel: ScalpelLayer, temperature: float = 0.3,
                 repetition_penalty: float = 0.3, context_weight: float = 0.4,
                 max_words: int = 100, seed: int = 42,
                 sentence_length: int = 20):
        self.scalpel = scalpel
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.context_weight = context_weight
        self.max_words = max_words
        self.sentence_length = sentence_length
        self.rng = random.Random(seed)
        self._index: dict[str, list[tuple[str, float, float]]] = {}
        self._indexed = False

    def build_index(self, verbose: bool = True) -> None:
        """Pré-construit l'index inversé mot→corrélations (une seule fois)."""
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
        for k in self._index:
            self._index[k].sort(key=lambda x: x[1], reverse=True)
        self._indexed = True
        if verbose:
            print(f"  Index inversé: {len(self._index):,} mots en {_time.time()-t0:.1f}s")

    def _corrs(self, word: str) -> list[tuple[str, float, float]]:
        if self._indexed:
            return self._index.get(word, [])
        return self.scalpel.get_correlations(word)

    def _score_candidates(self, prev: str, prev_prev: str | None,
                          used: set[str], topic: str | None = None) -> list[tuple[str, float]]:
        """Score les candidats en combinant bigramme direct + contexte à distance 2."""
        direct_corrs = self._corrs(prev)
        if not direct_corrs:
            return []

        # Pré-charger les co-occurrences du mot précédent-précédent (distance 2)
        context_corrs = {}
        if prev_prev and prev_prev != prev:
            context_corrs = {w: wt for w, wt, _ in self._corrs(prev_prev)}

        # Pré-charger les co-occurrences du topic
        topic_corrs = {}
        if topic and topic != prev:
            topic_corrs = {w: wt for w, wt, _ in self._corrs(topic)}

        candidates = []
        for word, direct_weight, p_sig in direct_corrs:
            if word == prev or word == prev_prev:
                continue
            # pénalité de répétition
            penalty = self.repetition_penalty if word in used else 1.0
            score = direct_weight * penalty
            # bonus de contexte à distance 2
            if word in context_corrs:
                score += self.context_weight * context_corrs[word] * penalty
            # bonus thématique
            if word in topic_corrs:
                score += self.context_weight * 0.5 * topic_corrs[word] * penalty
            candidates.append((word, max(score, 1e-9)))
        return candidates

    def _next_word(self, prev: str, prev_prev: str | None,
                   used: set[str], topic: str | None = None) -> str | None:
        candidates = self._score_candidates(prev, prev_prev, used, topic)
        if not candidates:
            return None

        weights = np.array([c[1] for c in candidates])
        if self.temperature > 0:
            logits = weights / (self.temperature + 1e-9)
            logits = np.clip(logits, -50, 50)
            exp = np.exp(logits - logits.max())
            probs = exp / exp.sum()
        else:
            probs = np.zeros(len(candidates))
            probs[weights.argmax()] = 1.0

        idx = self.rng.choices(range(len(candidates)), weights=probs, k=1)[0]
        return candidates[idx][0]

    def generate(self, seed_word: str, n_words: int = 50,
                 topic: str | None = None) -> str:
        """Génère n_words mots avec fenêtre de contexte de 2 mots."""
        used = {seed_word}
        words = [seed_word]
        prev = seed_word
        prev_prev = None
        word_count = 0

        for i in range(n_words - 1):
            # insérer une ponctuation de phrase tous les N mots
            if word_count > 0 and word_count % self.sentence_length == 0:
                words.append(".")
                # relancer depuis un mot fort (le top du précédent)
                corrs = self._corrs(prev)
                if corrs:
                    prev = corrs[0][0]
                    if prev not in used:
                        words.append(prev)
                        used.add(prev)
                        prev_prev = None
                        word_count = 1
                        continue

            nxt = self._next_word(prev, prev_prev, used, topic)
            if nxt is None:
                # relancer depuis un stopword
                relaunch = self._corrs(prev)
                if relaunch:
                    nxt = relaunch[0][0]
                else:
                    break
            words.append(nxt)
            used.add(nxt)
            prev_prev = prev
            prev = nxt
            word_count += 1

        text = " ".join(words)
        text = text.replace(" .", ".").replace(" ,", ",")
        # capitaliser après les points
        parts = text.split(". ")
        text = ". ".join(p.capitalize() for p in parts)
        return text

    def generate_paragraph(self, seed_words: list[str], n_words: int = 100,
                           topic: str | None = None) -> str:
        """Génère un paragraphe long en relançant depuis plusieurs seeds."""
        all_words = []
        used = set()
        n_generated = 0

        for seed in seed_words:
            if n_generated >= n_words:
                break
            remaining = min(self.sentence_length, n_words - n_generated)
            current = seed
            all_words.append(current)
            used.add(current)
            n_generated += 1
            prev_prev = None

            for _ in range(remaining - 1):
                nxt = self._next_word(current, prev_prev, used, topic)
                if nxt is None:
                    break
                all_words.append(nxt)
                used.add(nxt)
                prev_prev = current
                current = nxt
                n_generated += 1
            if n_generated < n_words:
                all_words.append(".")

        text = " ".join(all_words)
        text = text.replace(" .", ".")
        parts = text.split(". ")
        text = ". ".join(p.capitalize() for p in parts)
        return text


if __name__ == "__main__":
    from pathlib import Path
    from ratis_net.glove_tokenizer import GloveTokenizer

    tok = GloveTokenizer(dim=12, n_glove=8)
    scalpel = ScalpelLayer(tok, eta=0.1, coherence_threshold=0.3)
    scalpel.load(Path("artifacts/scalpel_wikipedia.pkl"))
    print(f"Scalpel: {scalpel.network_size():,} neurons\n")

    speaker = TriGrammarSpeaker(scalpel, temperature=0.15, context_weight=0.5,
                                 sentence_length=15)
    speaker.build_index()

    print("=== Tri-grammaire : generation mot par mot ===\n")
    for seed in ["quantum", "science", "love", "gravity", "brain"]:
        text = speaker.generate(seed, n_words=40, topic=seed)
        print(f"[{seed}]")
        print(f"  {text}\n")

    print("=== Paragraphe long (topic: science) ===\n")
    para = speaker.generate_paragraph(["the", "science", "of"], n_words=120, topic="science")
    print(para)

    print("\n\n=== Paragraphe long (topic: love) ===\n")
    para2 = speaker.generate_paragraph(["love", "is", "a"], n_words=120, topic="love")
    print(para2)
