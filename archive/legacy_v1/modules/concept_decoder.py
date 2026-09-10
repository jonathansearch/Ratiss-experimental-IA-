"""ratis_net.concept_decoder — Pont Scalpel (concepts) → Décodeur (syntaxe).

Le Scalpel sait quels mots sont corrélés (le fond : "de quoi on parle").
Le décodeur sait construire des phrases grammaticales (la forme : "comment
on le dit"). Ce module connecte les deux :

  1. Le Scalpel extrait un CLUSTER de concepts à partir d'un mot-thème.
     Ex: "quantum" → [quantum, mechanics, physics, particle, wave, theory, ...]
  2. Le décodeur prend ce cluster comme VOCABULAIRE restreint et génère une
     phrase fluide en utilisant les bigrammes (vraisemblance de transition).
  3. Le résultat est une phrase grammaticalement structurée qui parle du sujet
     demandé, sans hallucination (les mots viennent du Scalpel, pas du hasard).

Architecture : Scalpel décide du fond, Décodeur décide de la forme.
"""
from __future__ import annotations

import math
import random
from typing import Any

import numpy as np

try:
    from ratis_net.scalpel import ScalpelLayer
    from ratis_net.decoder import BigramModel
except ImportError:
    from scalpel import ScalpelLayer
    from decoder import BigramModel


class ConceptExtractor:
    """Extrait un cluster de concepts à partir d'un mot-thème via le Scalpel.

    Le cluster = les N mots les plus corrélés (poids LCT le plus élevé) au
    mot-thème, plus les mots corrélés à ces corrélations (expansion à 2 niveaux).
    """

    def __init__(self, scalpel: ScalpelLayer, cluster_size: int = 15,
                 expansion_depth: int = 1):
        self.scalpel = scalpel
        self.cluster_size = cluster_size
        self.expansion_depth = expansion_depth
        self._index: dict[str, list[tuple[str, float, float]]] = {}
        self._indexed = False

    def build_index(self, verbose: bool = True) -> None:
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
            print(f"  Index: {len(self._index):,} mots en {_time.time()-t0:.1f}s")

    def _corrs(self, word: str) -> list[tuple[str, float, float]]:
        if self._indexed:
            return self._index.get(word, [])
        return self.scalpel.get_correlations(word)

    def extract_cluster(self, theme: str, size: int | None = None) -> list[str]:
        """Extrait un cluster de concepts autour d'un mot-thème.

        Niveau 0 : les corrélations directes du thème.
        Niveau 1 : les corrélations des corrélations (expansion).
        """
        n = size or self.cluster_size
        cluster = {theme}
        # Niveau 0 : corrélations directes
        direct = self._corrs(theme)[:n]
        for word, _, _ in direct:
            cluster.add(word)
        # Niveau 1 : expansion (corrélations des corrélations)
        if self.expansion_depth >= 1:
            for word, _, _ in direct[:5]:  # top 5 seulement pour l'expansion
                for w2, _, _ in self._corrs(word)[:3]:
                    cluster.add(w2)
        return list(cluster)


class ConceptDecoder:
    """Décodeur qui tisse un cluster de concepts en phrases grammaticales.

    Utilise les bigrammes du décodeur existant pour la vraisemblance de
    transition, mais restreint le vocabulaire au cluster extrait par le Scalpel.
    Le résultat : une phrase qui parle du sujet (concepts du Scalpel) avec une
    structure grammaticale (bigrammes du décodeur).
    """

    # Mots fonctionnels pour la structure grammaticale
    STRUCTURE_WORDS = {"the", "a", "an", "is", "was", "are", "were", "of", "in",
                       "and", "to", "that", "this", "for", "on", "with", "as",
                       "by", "at", "from", "or", "be", "has", "have", "had",
                       "not", "but", "he", "she", "it", "they", "which", "who",
                       "can", "will", "would", "could", "its", "his", "her",
                       "their", "also", "than", "then", "so", "if"}

    def __init__(self, scalpel: ScalpelLayer, bigram_model: BigramModel | None = None,
                 temperature: float = 0.5, repetition_penalty: float = 0.3,
                 seed: int = 42):
        self.scalpel = scalpel
        self.bigram = bigram_model
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.rng = random.Random(seed)
        self._index: dict[str, list[tuple[str, float, float]]] = {}
        self._indexed = False

    def build_index(self, verbose: bool = True) -> None:
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
            print(f"  Index: {len(self._index):,} mots en {_time.time()-t0:.1f}s")

    def _corrs(self, word: str) -> list[tuple[str, float, float]]:
        if self._indexed:
            return self._index.get(word, [])
        return self.scalpel.get_correlations(word)

    def _build_vocab(self, theme: str, cluster_size: int = 15) -> tuple[list[str], set[str]]:
        """Construit le vocabulaire = cluster Scalpel + mots structurels.

        Retourne (vocab, concept_words) où concept_words sont les mots issus
        du Scalpel (à booster) et vocab inclut aussi les mots structurels.
        """
        extractor = ConceptExtractor(self.scalpel, cluster_size=cluster_size)
        if self._indexed:
            extractor._index = self._index
            extractor._indexed = True
        concepts = set(extractor.extract_cluster(theme))
        vocab = list(concepts | self.STRUCTURE_WORDS)
        return vocab, concepts

    def _transition_score(self, prev: str, word: str, theme: str) -> float:
        """Score de transition : bigramme Scalpel + bigramme émotionnel si dispo."""
        # 1. bigramme Scalpel (poids LCT de la paire)
        scalpel_score = 0.0
        corrs = self._corrs(prev)
        for w, weight, _ in corrs:
            if w == word:
                scalpel_score = weight
                break
        # 2. bigramme émotionnel (vraisemblance de transition) si disponible
        emo_score = 1.0
        if self.bigram:
            emo_score = self.bigram.prob("others", prev, word)
        return scalpel_score + emo_score

    def _theme_boost(self, word: str, theme: str) -> float:
        """Boost pour les mots directement corrélés au thème."""
        corrs = self._corrs(theme)
        for w, weight, _ in corrs:
            if w == word:
                return 1.0 + weight * 0.01
        return 1.0

    def generate_sentence(self, theme: str, length: int = 12,
                          cluster_size: int = 15) -> str:
        """Génère une phrase sur un thème, avec structure grammaticale.

        1. Extrait le cluster de concepts (Scalpel).
        2. Génère mot par mot en combinant transition (bigramme) + boost thème.
        Les mots-concepts sont fortement boostés ; les mots structurels sont
        limités à 2 consécutifs pour éviter les séquences "the of a is".
        """
        vocab, concepts = self._build_vocab(theme, cluster_size)
        if not vocab:
            return theme

        used = set()
        structure_streak = 0  # compteur de mots structurels consécutifs
        words = []
        # Commencer par le thème
        if theme in vocab:
            words.append(theme)
            used.add(theme)
        else:
            words.append("the")
            used.add("the")
            structure_streak = 1

        prev = words[0]

        for _ in range(length - 1):
            candidates = []
            for w in vocab:
                if w == prev or w in used:
                    continue
                # Limiter les mots structurels consécutifs (max 2)
                if w in self.STRUCTURE_WORDS and structure_streak >= 2:
                    continue
                trans = self._transition_score(prev, w, theme)
                # Boost massif pour les concepts (10x), léger pour les structurels
                if w in concepts:
                    boost = 10.0 + self._theme_boost(w, theme) * 5.0
                else:
                    boost = 1.0
                penalty = self.repetition_penalty if w in used else 1.0
                score = trans * boost * penalty
                candidates.append((w, max(score, 1e-9)))

            if not candidates:
                # relâcher la contrainte structurelle si aucun candidat
                structure_streak = 0
                for w in vocab:
                    if w == prev or w in used:
                        continue
                    trans = self._transition_score(prev, w, theme)
                    boost = 5.0 if w in concepts else 1.0
                    score = trans * boost
                    candidates.append((w, max(score, 1e-9)))
                if not candidates:
                    break

            # softmax avec température
            weights = np.array([c[1] for c in candidates])
            if self.temperature > 0:
                logits = np.log(weights) / max(self.temperature, 1e-3)
                logits = np.clip(logits, -50, 50)
                exp = np.exp(logits - logits.max())
                probs = exp / exp.sum()
            else:
                probs = np.zeros(len(candidates))
                probs[weights.argmax()] = 1.0

            idx = self.rng.choices(range(len(candidates)), weights=probs, k=1)[0]
            chosen = candidates[idx][0]
            words.append(chosen)
            used.add(chosen)
            prev = chosen
            if chosen in self.STRUCTURE_WORDS:
                structure_streak += 1
            else:
                structure_streak = 0

        # Capitaliser et ponctuer
        text = " ".join(words)
        text = text.capitalize() + "."
        return text

    def generate_paragraph(self, theme: str, n_sentences: int = 5,
                           sentence_length: int = 12,
                           cluster_size: int = 20) -> str:
        """Génère un paragraphe de plusieurs phrases sur le même thème."""
        sentences = []
        for i in range(n_sentences):
            # varier la seed pour la diversité
            self.rng = random.Random(42 + i)
            sent = self.generate_sentence(theme, length=sentence_length,
                                          cluster_size=cluster_size)
            sentences.append(sent)
        return " ".join(sentences)


if __name__ == "__main__":
    from pathlib import Path
    from ratis_net.glove_tokenizer import GloveTokenizer

    tok = GloveTokenizer(dim=12, n_glove=8)
    scalpel = ScalpelLayer(tok, eta=0.1, coherence_threshold=0.3)
    scalpel.load(Path("artifacts/scalpel_wikipedia.pkl"))
    print(f"Scalpel: {scalpel.network_size():,} neurons\n")

    decoder = ConceptDecoder(scalpel, temperature=0.3, seed=42)
    decoder.build_index()

    print("=== Concept Decoder : Scalpel (concepts) + Decodeur (syntaxe) ===\n")

    themes = ["quantum", "science", "love", "gravity", "brain", "music", "history"]
    for theme in themes:
        print(f"[{theme}]")
        # Montrer le cluster extrait
        extractor = ConceptExtractor(scalpel, cluster_size=10)
        extractor._index = decoder._index
        extractor._indexed = True
        cluster = extractor.extract_cluster(theme, size=8)
        print(f"  Cluster: {cluster[:8]}")
        # Générer la phrase
        sentence = decoder.generate_sentence(theme, length=12)
        print(f"  Phrase: {sentence}")
        print()

    print("=== Paragraphes longs ===\n")
    for theme in ["quantum", "love"]:
        print(f"[{theme}]")
        para = decoder.generate_paragraph(theme, n_sentences=4, sentence_length=14)
        print(f"  {para}\n")
