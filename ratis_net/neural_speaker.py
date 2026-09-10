"""ratis_net.neural_speaker — Reconstruction neuronale de phrases.

Le réseau ne recite pas : il RECONSTRUIT. Pour une requête :

  1. Le Scalpel (v1, 3.78M neurones) active un chemin de concepts autour du
     sujet (IDF de degré + LCT + voisinage partagé).
  2. Le corpus v3 (phrases ordonnées apprises) est scoré par RECOUVREMENT
     avec ce chemin : une phrase est candidate si ses mots de contenu sont
     activés par le réseau — pas seulement si elle contient le mot-sujet.
  3. La phrase la mieux couverte par le chemin est retournée, avec le score
     de reconstruction (combien de ses concepts viennent du réseau).

C'est le principe du Synchrotron appliqué au langage : le réseau reconnaît
les phrases qui lui ressemblent, il ne génère pas mot à mot au hasard.
Si rien ne dépasse le seuil de couverture, le speaker le dit honnêtement.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from ratis_net.query_analyzer import analyze

_FUNCTION = {
    "the", "a", "an", "of", "in", "and", "to", "is", "are", "was", "that",
    "this", "it", "for", "on", "with", "as", "by", "at", "from", "or", "be",
    "has", "have", "but", "not", "its", "their", "which", "who", "been",
    "than", "so", "if", "can", "will", "would", "could", "should", "may",
    "do", "does", "no", "all", "some", "into", "over", "after", "between",
    "through", "during", "before", "under", "while", "when", "how", "what",
    "they", "them", "these", "those", "he", "she", "we", "you", "his", "her",
    "also", "known", "used", "many", "most", "other", "such", "first", "two",
}


class NeuralSpeaker:
    """Reconstruit la phrase la plus couverte par le chemin neuronal."""

    def __init__(self,
                 graph: dict[str, list[tuple[str, float, float, int]]],
                 corpus: list[str] | None = None,
                 seed: int = 42):
        self.graph = graph
        self.corpus = corpus or []
        self._n = max(len(graph), 1)
        self._neighbors = {k: {t for (t, *_r) in v} for k, v in graph.items()}
        # index mot → phrases du corpus (recherche inversée)
        self._by_word: dict[str, list[int]] = defaultdict(list)
        self._content_words: list[set[str]] = []
        for i, sent in enumerate(self.corpus):
            words = set(sent.split()) - _FUNCTION
            self._content_words.append(words)
            for w in words:
                self._by_word[w].append(i)

    def idf(self, word: str) -> float:
        return math.log(self._n / (1 + len(self._neighbors.get(word, ()))) + 1)

    def concept_path(self, keywords: list[str], n: int = 8) -> list[str]:
        """Chemin de concepts activés par le réseau autour des mots-clés."""
        scores: dict[str, float] = {}
        for kw in keywords:
            for cand, weight, p_sig, renf in self.graph.get(kw, []):
                if cand in _FUNCTION or cand in keywords or len(cand) < 3:
                    continue
                sc = weight * (0.5 + p_sig) * math.log1p(renf) * self.idf(cand)
                if len(keywords) > 1:
                    shared = sum(1 for k2 in keywords
                                 if k2 != kw and cand in self._neighbors.get(k2, ()))
                    sc += shared * 5.0
                if sc > scores.get(cand, 0.0):
                    scores[cand] = sc
        return [w for w, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:n]]

    def _coverage(self, sentence_words: set[str], path: set[str]) -> float:
        """Couverture IDF-pondérée : les mots rares (informatifs) comptent
        plus que les mots fréquents. Une phrase est « reconnue » si ses mots
        les plus spécifiques sont activés par le chemin neuronal."""
        if not sentence_words:
            return 0.0
        total = sum(self.idf(w) for w in sentence_words)
        if total <= 0:
            return 0.0
        hit = sum(self.idf(w) for w in sentence_words & path)
        return hit / total

    def speak(self, query: str, min_coverage: float = 0.25) -> dict[str, Any]:
        """Reconstruit la meilleure phrase apprise pour la requête."""
        analysis = analyze(query)
        kws = [k for k in analysis.all_concepts
               if k in self.graph or k in self._by_word]
        if not kws:
            return {"mode": "neural", "ok": False, "sentence": "",
                    "concepts": [], "reason": "concept absent du corpus appris"}

        path = set(kws) | set(self.concept_path(kws, n=8))

        # phrases candidates : celles qui contiennent au moins un mot-clé
        cand_ids: dict[int, int] = defaultdict(int)
        for kw in kws:
            for i in self._by_word.get(kw, []):
                cand_ids[i] += 1

        best, best_score, best_cov, best_defines = None, -1.0, 0.0, False
        for i, hits in cand_ids.items():
            words = self._content_words[i]
            cov = self._coverage(words, path)
            sent_words = self.corpus[i].split()
            # bonus si la phrase DÉFINIT le sujet : « [the/a] <sujet> is/are »
            # en tête — c'est le patron appris le plus informatif. Le sujet
            # doit être le premier mot de contenu (pas "gw was black hole…").
            head = sent_words[:4]
            defines = (len(sent_words) > 3
                       and any(kw in head[:2] for kw in kws)
                       and any(w in {"is", "are", "was", "were"} for w in head))
            # sujet exact = premier mot de la phrase (pas juste présent)
            subject_first = sent_words and sent_words[0] in kws
            score = cov + 0.05 * hits + (0.50 if defines else 0.0) \
                + (0.15 if subject_first else 0.0)
            # Pénalité : les phrases trop courtes ou trop génériques ne
            # doivent pas gagner sur la seule couverture.
            if len(words) < 5:
                score -= 0.15
            if score > best_score:
                best, best_score, best_cov, best_defines = i, score, cov, defines

        # éligibilité : couverture suffisante, OU phrase-définition du sujet
        # (même à couverture modeste, « X is a Y » est la réponse attendue)
        if best is None or (best_cov < min_coverage and not best_defines):
            return {"mode": "neural", "ok": False, "sentence": "",
                    "concepts": sorted(path)[:8],
                    "coverage": round(best_cov, 3),
                    "reason": f"couverture {best_cov:.0%} < {min_coverage:.0%}"}

        sent = self.corpus[best].strip()
        sent = sent[0].upper() + sent[1:]
        if sent[-1] not in ".!?":
            sent += "."
        return {"mode": "neural", "ok": True, "sentence": sent,
                "concepts": sorted(path)[:8], "coverage": round(best_cov, 3),
                "source": "reconstruction du corpus appris"}
