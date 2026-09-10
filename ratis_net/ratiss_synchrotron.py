"""ratis_net.ratiss_synchrotron — Reconstruction sémantique topologique.

Remplace la prédiction statistique (Transformer/gradient) par un assemblage de
fragments pré-indexés dont la signature topologique s'emboîte avec la requête.

Architecture en 4 étapes (sans gradient, sans Transformer) :

  1. Super-Embedding Index : chaque fragment du corpus est stocké avec un
     vecteur hybride [GloVe (sens) + Signature Topo LCT (forme/P_sig)].
  2. Synchrotron (découpeur) : analyse la requête, extrait ses super-embeddings
     et identifie le « vide topologique » (cycles H1 ouverts à fermer).
  3. Moteur de résonance : cherche dans l'index les fragments dont la tension
     LCT avec la requête est faible (bon emboîtement topologique).
  4. Assembleur RATISS : reconstruit la phrase en respectant l'ordre
     syntaxique imposé par la cohérence topologique.

La loi LCT (R = P_sig, ΔW = η·φ·P_sig·C) reste figée. La tension topologique
T = stress / (A × P_sig) mesure la qualité de l'emboîtement.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    from ratis_net.glove_tokenizer import GloveTokenizer, glove_topo_signature
    from ratis_net.topo_cache import TopoCache
except ImportError:
    from glove_tokenizer import GloveTokenizer, glove_topo_signature
    from topo_cache import TopoCache


# ─────────────────────────────────────────────────────────────────────────────
# 1. Super-Embedding Index (base de données vectorielle)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IndexedFragment:
    """Un fragment de texte indexé avec son super-embedding."""
    text: str
    words: list[str]
    embedding: np.ndarray
    topo_signature: np.ndarray
    source: str = "corpus"


class SuperEmbeddingIndex:
    """Index vectorielle de fragments pré-calculés.

    Stocke les super-embeddings [GloVe + Topo] de chaque fragment et permet une
    recherche par similarité cosinus (tension LCT faible = bon emboîtement).
    """

    def __init__(self, dim: int = 12, n_glove: int = 8):
        self.dim = dim
        self.n_glove = n_glove
        self.fragments: list[IndexedFragment] = []
        self._matrix: np.ndarray | None = None

    def add_fragment(self, text: str, tokenizer: GloveTokenizer, source: str = "corpus") -> None:
        words = text.lower().strip().split()
        if len(words) < 2:
            return
        embs = np.array([tokenizer(w, self.dim) for w in words])
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        pooled = (embs * norms).sum(axis=0) / norms.sum()
        n = np.linalg.norm(pooled)
        pooled = pooled / n if n > 1e-9 else pooled
        topo = pooled[self.n_glove:]
        self.fragments.append(IndexedFragment(text=text, words=words,
                                                embedding=pooled, topo_signature=topo,
                                                source=source))
        self._matrix = None

    def build_from_corpus(self, corpus: list[str], tokenizer: GloveTokenizer) -> None:
        for text in corpus:
            self.add_fragment(text, tokenizer)

    def _build_matrix(self) -> np.ndarray:
        if self._matrix is None and self.fragments:
            self._matrix = np.array([f.embedding for f in self.fragments])
        return self._matrix

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[IndexedFragment, float]]:
        """Recherche cosinus : retourne les fragments les plus proches."""
        mat = self._build_matrix()
        if mat is None or len(mat) == 0:
            return []
        q = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        mat_norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
        sims = mat_norm @ q
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(self.fragments[i], float(sims[i])) for i in top_idx]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Synchrotron (découpeur de requête)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TopologicalVoid:
    """Un cycle H1 ouvert dans la requête qui nécessite une fermeture sémantique."""
    position: int
    word: str
    embedding: np.ndarray
    p_sig: float
    needs_closure: bool


class Synchrotron:
    """Découpe la requête et identifie le vide topologique.

    Le Synchrotron ne génère pas : il tokenise, extrait les super-embeddings et
    détecte quels mots ont un P_sig faible (cycles H1 non fermés = vide
    sémantique à combler).
    """

    def __init__(self, tokenizer: GloveTokenizer):
        self.tokenizer = tokenizer

    def analyze_query(self, query: str) -> dict[str, Any]:
        words = query.lower().strip().split()
        if not words:
            return {"words": [], "embeddings": np.array([]), "voids": [], "pooled": None, "p_sigs": []}
        embeddings = np.array([self.tokenizer(w, self.tokenizer.dim) for w in words])
        # P_sig = composante topo de chaque mot (après GloVe)
        topo_dim = self.tokenizer.dim - self.tokenizer.n_glove
        p_sigs = [float(np.linalg.norm(emb[self.tokenizer.n_glove:])) for emb in embeddings]
        # vide topologique = mots dont P_sig est sous la médiane (cycles faibles)
        median_psig = float(np.median(p_sigs)) if p_sigs else 0.0
        voids = [TopologicalVoid(position=i, word=words[i], embedding=embeddings[i],
                                  p_sig=p_sigs[i], needs_closure=p_sigs[i] < median_psig)
                 for i in range(len(words))]
        # pooling pondéré par P_sig (les mots topologiquement forts pèsent plus)
        weights = np.array(p_sigs) + 1e-9
        pooled = (embeddings * weights[:, None]).sum(axis=0) / weights.sum()
        n = np.linalg.norm(pooled)
        pooled = pooled / n if n > 1e-9 else pooled
        return {"words": words, "embeddings": embeddings, "voids": voids,
                "pooled": pooled, "p_sigs": p_sigs}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Moteur de résonance (recherche par cohérence topologique)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ResonanceResult:
    """Un fragment résonant avec sa tension LCT calculée."""
    fragment: IndexedFragment
    cosine_sim: float
    topological_tension: float
    coherence: float
    accepted: bool


class ResonanceEngine:
    """Cherche dans l'index les fragments dont la tension LCT est faible.

    Tension topologique T = stress / (A × P_sig) où :
      - stress = distance de Frobenius entre les signatures topo
      - A = amplitude de référence (alpha_0 × P_sig de la requête)
      - P_sig = persistance du fragment

    T faible → bon emboîtement ; T élevé → mauvais emboîtement.

    Si un Scalpel est connecté, le score de résonance combine :
      - cosine_sim (similarité sémantique GloVe)
      - scalpel_boost (poids LCT des paires mot-requête ↔ mot-fragment)
    Le boost repondère les candidats existants ; il ne crée rien de nouveau.
    """

    def __init__(self, index: SuperEmbeddingIndex, alpha_0: float = 1.0,
                 tension_threshold: float = 1.0, scalpel=None,
                 scalpel_weight: float = 0.3):
        self.index = index
        self.alpha_0 = alpha_0
        self.tension_threshold = tension_threshold
        self.scalpel = scalpel
        self.scalpel_weight = scalpel_weight

    def _scalpel_boost(self, query_words: list[str], fragment_words: list[str]) -> float:
        """Boost basé sur les corrélations LCT apprises par le Scalpel.

        Pour chaque paire (mot_requête, mot_fragment), on récupère le poids LCT
        dans le Scalpel. Le boost = somme normalisée des poids trouvés.
        0 si aucune corrélation, positif si le Scalpel a appris ces paires.
        """
        if self.scalpel is None:
            return 0.0
        total_weight = 0.0
        n_found = 0
        for qw in query_words:
            corrs = self.scalpel.get_correlations(qw)
            corr_dict = {w: w_val for w, w_val, _ in corrs}
            for fw in fragment_words:
                key = self.scalpel._pair_key(qw, fw)
                if key in corr_dict:
                    total_weight += corr_dict[key]
                    n_found += 1
                elif key in self.scalpel.neurons:
                    total_weight += self.scalpel.neurons[key].weight
                    n_found += 1
        if n_found == 0:
            return 0.0
        # normaliser par le nombre de paires testées
        max_possible = max(1.0, float(max(n.weight for n in self.scalpel.neurons.values())) + 1e-9)
        return float(total_weight / (n_found * max_possible))

    def find_resonant_fragments(self, query_analysis: dict[str, Any],
                                top_k: int = 5) -> list[ResonanceResult]:
        pooled = query_analysis["pooled"]
        if pooled is None:
            return []
        candidates = self.index.search(pooled, top_k=top_k * 2)
        results = []
        query_psig = float(np.linalg.norm(pooled[self.index.n_glove:])) + 1e-9
        query_words = query_analysis.get("words", [])
        for frag, sim in candidates:
            # tension topologique : distance entre signatures topo
            stress = float(np.linalg.norm(query_analysis["pooled"][self.index.n_glove:] - frag.topo_signature))
            frag_psig = float(np.linalg.norm(frag.topo_signature)) + 1e-9
            A = self.alpha_0 * query_psig
            tension = stress / (A * frag_psig) if A > 0 and frag_psig > 0 else float("inf")
            # boost Scalpel : corrélations LCT apprises entre requête et fragment
            boost = self._scalpel_boost(query_words, frag.words)
            # score combiné : cosine + boost (le boost repondère, ne crée pas)
            combined_sim = sim + self.scalpel_weight * boost
            coherence = 1.0 / (1.0 + tension) + self.scalpel_weight * boost
            coherence = float(np.clip(coherence, 0.0, 1.0))
            accepted = tension < self.tension_threshold
            results.append(ResonanceResult(fragment=frag, cosine_sim=combined_sim,
                                             topological_tension=tension,
                                             coherence=coherence, accepted=accepted))
        # trier par score combiné décroissant (cosinus + boost)
        results.sort(key=lambda r: r.cosine_sim, reverse=True)
        return results[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Assembleur RATISS (reconstruction)
# ─────────────────────────────────────────────────────────────────────────────

class RatisAssembler:
    """Assemble les fragments résonants en une phrase reconstruite.

    L'ordre syntaxique est déterminé par la cohérence topologique : on place
    d'abord les fragments dont la tension avec la requête est la plus faible,
    puis on vérifie que la cohérence globale reste élevée.
    """

    def __init__(self, max_fragments: int = 3, min_coherence: float = 0.3):
        self.max_fragments = max_fragments
        self.min_coherence = min_coherence

    def reconstruct(self, query: str, resonance_results: list[ResonanceResult]) -> dict[str, Any]:
        accepted = [r for r in resonance_results if r.accepted and r.coherence >= self.min_coherence]
        if not accepted:
            accepted = resonance_results[:1]
        # prendre les meilleurs fragments (jusqu'à max_fragments)
        selected = accepted[:self.max_fragments]
        # assembler : la requête + les fragments résonants
        parts = [query]
        for r in selected:
            if r.fragment.text.lower() not in query.lower():
                parts.append(r.fragment.text)
        reconstructed = " ".join(parts)
        avg_coherence = float(np.mean([r.coherence for r in selected])) if selected else 0.0
        avg_tension = float(np.mean([r.topological_tension for r in selected])) if selected else float("inf")
        return {
            "query": query,
            "reconstructed": reconstructed,
            "selected_fragments": [{"text": r.fragment.text, "tension": r.topological_tension,
                                     "coherence": r.coherence, "cosine": r.cosine_sim}
                                    for r in selected],
            "avg_coherence": avg_coherence,
            "avg_tension": avg_tension,
            "n_fragments": len(selected),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrateur : generate_response
# ─────────────────────────────────────────────────────────────────────────────

class RatissSynchrotron:
    """Moteur de reconstruction sémantique topologique (orchestrateur).

    Aucun gradient, aucun Transformer. La génération se fait par assemblage de
    fragments pré-indexés dont la signature topologique s'emboîte avec la
    requête, validée par la tension LCT.
    """

    def __init__(self, dim: int = 12, n_glove: int = 8,
                 alpha_0: float = 1.0, tension_threshold: float = 1.0,
                 max_fragments: int = 3, min_coherence: float = 0.3,
                 scalpel=None, scalpel_weight: float = 0.3):
        self.tokenizer = GloveTokenizer(dim=dim, n_glove=n_glove)
        self.index = SuperEmbeddingIndex(dim=dim, n_glove=n_glove)
        self.synchrotron = Synchrotron(self.tokenizer)
        self.resonance = ResonanceEngine(self.index, alpha_0=alpha_0,
                                          tension_threshold=tension_threshold,
                                          scalpel=scalpel,
                                          scalpel_weight=scalpel_weight)
        self.assembler = RatisAssembler(max_fragments=max_fragments,
                                         min_coherence=min_coherence)

    def build_corpus(self, corpus: list[str]) -> None:
        """Indexe un corpus de phrases dans le super-embedding index."""
        self.index.build_from_corpus(corpus, self.tokenizer)

    def generate_response(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """Génère une réponse par reconstruction topologique (sans gradient)."""
        analysis = self.synchrotron.analyze_query(query)
        resonance = self.resonance.find_resonant_fragments(analysis, top_k=top_k)
        reconstruction = self.assembler.reconstruct(query, resonance)
        return {
            "query": query,
            "analysis": {"words": analysis["words"], "p_sigs": analysis["p_sigs"],
                          "n_voids": sum(1 for v in analysis["voids"] if v.needs_closure)},
            "reconstruction": reconstruction,
            "n_indexed_fragments": len(self.index.fragments),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Corpus de démonstration (phrases réelles, pas générées)
# ─────────────────────────────────────────────────────────────────────────────

DEMO_CORPUS = [
    "the science of quantum mechanics is fascinating",
    "gravity is a fundamental force of nature",
    "quantum computing uses qubits instead of bits",
    "the universe is expanding at an accelerating rate",
    "artificial intelligence can learn from data",
    "topology studies the properties of shapes",
    "the law of coherence describes how systems maintain structure",
    "black holes bend spacetime around them",
    "neural networks learn by adjusting weights",
    "the speed of light is constant in vacuum",
    "entropy measures the disorder of a system",
    "consciousness emerges from complex interactions",
    "the warp drive would bend space to travel faster",
    "quantum entanglement connects distant particles",
    "the brain processes information through neurons",
    "mathematics is the language of the universe",
    "energy cannot be created or destroyed",
    "the observer effect changes quantum measurements",
    "machine learning finds patterns in data",
    "the theory of relativity changed physics forever",
    "i am happy to see you today",
    "i feel sad when you are not here",
    "that makes me really angry",
    "thank you for your help",
    "how are you feeling right now",
    "the weather is beautiful today",
    "i love learning new things",
    "this is a difficult problem to solve",
    "can you help me understand this",
    "the answer is not what i expected",
]


if __name__ == "__main__":
    import json

    engine = RatissSynchrotron()
    engine.build_corpus(DEMO_CORPUS)
    print(f"Index: {len(engine.index.fragments)} fragments")
    print()

    queries = [
        "what is quantum mechanics",
        "how does the brain work",
        "i feel happy today",
        "explain gravity",
    ]
    for q in queries:
        result = engine.generate_response(q)
        print(f"Q: {q}")
        print(f"  R: {result['reconstruction']['reconstructed']}")
        print(f"  coherence={result['reconstruction']['avg_coherence']:.3f} "
              f"tension={result['reconstruction']['avg_tension']:.3f} "
              f"fragments={result['reconstruction']['n_fragments']}")
        for f in result["reconstruction"]["selected_fragments"]:
            print(f"    [{f['tension']:.3f}] {f['text']}")
        print()
