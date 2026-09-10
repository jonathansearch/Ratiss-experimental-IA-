"""ratis_net.concept_ranker — Classement des concepts par pertinence.

Le Scalpel relie les mots par poids LCT bruts ; les termes les plus renforcés
sont souvent des mots ubiquitaires ("to", "american") qui co-occurrent avec
tout. Ce module corrige ce biais par trois signaux combinés :

  1. IDF de degré : un mot relié à 20 000 voisins dilue l'information ;
     un mot relié à 200 voisins est spécifique. score_idf = log(N / (1+deg)).
  2. Voisinage partagé : un candidat adjacent à PLUSIEURS mots de la requête
     (ex: voisin de "black" ET de "hole") reçoit un bonus — c'est le seul
     signal compositionnel disponible après le filtrage cos≥0.3 du Scalpel.
  3. Proximité GloVe : les k-plus-proches-voisins sémantiques du vecteur
     requête complètent (sans remplacer) le voisinage de corpus.

Le résultat : pour "protein" → structure/kinase/binding/folding ; pour
"quantum mechanics" → theory/relativity/electrodynamics ; les artefacts de
corpus ("black metal" pour "black hole") sont rétrogradés mais resteront
visibles tant que le checkpoint n'est pas ré-entraîné : c'est documenté dans
LIMITES_HONNETES.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ratis_net.glove_tokenizer import _load_glove
from ratis_net.query_analyzer import STOPWORDS_EN, STOPWORDS_FR

STOP = STOPWORDS_EN | STOPWORDS_FR | {
    "up", "down", "out", "off", "over", "again", "further", "once", "here",
    "there", "very", "just", "only", "now", "new", "old", "well", "way",
    "get", "got", "make", "made", "take", "time", "year", "years", "day",
    "people", "world", "life", "work", "part", "place", "case", "point",
    "between", "while", "until", "against", "own", "too", "about", "because",
    "being", "having", "doing", "going", "went", "come", "came", "know",
    "known", "think", "thought", "see", "seen", "say", "said", "use", "used",
    "using", "like", "called", "often", "form", "forms", "found", "including",
    "include", "based", "example", "term", "terms", "referred", "typically",
    "usually", "generally", "sometimes", "however", "although", "thus",
    "therefore", "among", "within", "without", "along", "across", "towards",
    "upon", "per", "via", "etc", "also", "may", "many", "much", "one", "two",
    # adjectifs génériques de taille/degré : jamais des concepts
    "small", "large", "big", "little", "long", "short", "high", "low",
    "wide", "narrow", "deep", "full", "empty", "early", "late",
    # nombres : jamais des concepts
    "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "hundred", "thousand", "million", "billion",
}


class ConceptRanker:
    """Classe les candidats-concepts autour d'un groupe de mots-clés.

    index : dict mot -> [(autre, poids, p_sig), ...] (index inversé du Scalpel,
            construit par SkeletonSpeaker.build_index).
    """

    def __init__(self, index: dict[str, list[tuple[str, float, float]]]):
        self.index = index
        self._neighbors: dict[str, set[str]] = {
            k: {t for (t, _, _) in v} for k, v in index.items()
        }
        self._n_words = max(len(index), 1)
        self._glove_words: list[str] | None = None
        self._glove_matrix: np.ndarray | None = None
        self._glove_pos: dict[str, int] | None = None

    # ── IDF de degré ────────────────────────────────────────────────────────
    def idf(self, word: str) -> float:
        deg = len(self._neighbors.get(word, ()))
        return math.log(self._n_words / (1 + deg) + 1)

    # ── GloVe : matrice normalisée, chargée paresseusement ─────────────────
    def _ensure_glove(self) -> None:
        if self._glove_matrix is not None:
            return
        glove = _load_glove()
        if not glove:
            self._glove_words, self._glove_matrix, self._glove_pos = [], None, {}
            return
        words = list(glove.keys())
        matrix = np.stack([glove[w] for w in words]).astype(np.float32)
        matrix /= (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
        self._glove_words = words
        self._glove_matrix = matrix
        self._glove_pos = {w: i for i, w in enumerate(words)}

    def glove_knn(self, keywords: list[str], n: int = 10) -> list[str]:
        """k plus-proches-voisins GloVe du vecteur requête (hors mots-clés)."""
        self._ensure_glove()
        if self._glove_matrix is None or not keywords:
            return []
        pos = self._glove_pos or {}
        vecs = [self._glove_matrix[pos[k]] for k in keywords if k in pos]
        if not vecs:
            return []
        q = np.mean(vecs, axis=0)
        q = q / (np.linalg.norm(q) + 1e-9)
        sims = self._glove_matrix @ q
        kws = set(keywords)
        top = np.argpartition(-sims, min(n + len(kws), len(sims) - 1))[:n + len(kws)]
        top = top[np.argsort(-sims[top])]
        words = self._glove_words or []
        out = []
        for i in top:
            w = words[int(i)]
            if w not in kws and w not in STOP and len(w) > 2:
                out.append(w)
            if len(out) >= n:
                break
        return out

    # ── Classement principal ────────────────────────────────────────────────
    def rank(self, keywords: list[str], n: int = 10,
             glove_n: int = 4, shared_bonus: float = 10.0) -> list[str]:
        """Retourne les n concepts les plus pertinents autour des mots-clés.

        score(c) = max_k poids(k→c) × IDF(c)  +  bonus voisinage partagé
        Complété par les kNN GloVe (sémantique pure) en fin de liste.
        """
        kws = [k for k in keywords if k not in STOP]
        if not kws:
            return []
        scores: dict[str, float] = {}
        for kw in kws:
            for cand, weight, _p_sig in self.index.get(kw, []):
                if cand in STOP or cand in kws or len(cand) < 2:
                    continue
                sc = weight * self.idf(cand)
                if sc > scores.get(cand, 0.0):
                    scores[cand] = sc
        # Bonus : candidat relié à plusieurs mots-clés de la requête
        if len(kws) > 1:
            for cand in list(scores):
                shared = sum(1 for kw in kws[1:]
                             if cand in self._neighbors.get(kw, ()))
                if shared:
                    scores[cand] += shared * shared_bonus
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        concepts = [w for w, _ in ranked[:n]]
        # Complément sémantique GloVe (les composés échappent au corpus)
        if len(concepts) < n:
            for w in self.glove_knn(kws, n=glove_n):
                if w not in concepts and w not in kws:
                    concepts.append(w)
                if len(concepts) >= n:
                    break
        return concepts[:n]
