"""ratis_net.chain_reasoning — Chaînes d'association dans le graphe Scalpel.

Le Scalpel ne fait pas d'inférence logique. Ce module explore le graphe de
corrélations en largeur (BFS) pour trouver des CHAÎNES D'ASSOCIATION entre
deux concepts : A ↔ B ↔ C signifie "A est corrélé à B qui est corrélé à C"
dans le corpus appris — pas "A implique B implique C".

Chaque chaîne est retournée avec ses poids LCT et une étiquette honnête
("association_chain", jamais "inference"). C'est la réponse de RATIS-Net à
la limite "pas de raisonnement multi-sauts" : on peut TRACER les chemins
associatifs, en déclarant clairement leur nature.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any

# Les mots-outils ne forment jamais un maillon intermédiaire : une chaîne
# "quantum ↔ of ↔ gravity" est un artefact de co-occurrence, pas une
# association de sens.
_SKIP_LINKS = {
    "the", "a", "an", "of", "in", "and", "to", "is", "was", "were", "that",
    "this", "it", "for", "on", "with", "as", "by", "at", "from", "or", "be",
    "has", "have", "had", "but", "not", "he", "she", "his", "her", "its",
    "their", "they", "which", "who", "are", "been", "also", "than", "then",
}


@dataclass
class Chain:
    """Une chaîne d'association entre deux concepts."""
    path: list[str]
    weights: list[float]
    min_weight: float
    kind: str = "association_chain"  # PAS une inférence logique

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "hops": len(self.path) - 1,
            "weights": [round(w, 4) for w in self.weights],
            "min_weight": round(self.min_weight, 4),
            "kind": self.kind,
        }


def find_chains(source: str, target: str,
                index: dict[str, list[tuple[str, float, float]]],
                max_hops: int = 3, max_chains: int = 3) -> list[Chain]:
    """Cherche les meilleures chaînes source → ... → target (BFS par poids).

    index : index inversé du Scalpel (mot -> [(autre, poids, p_sig)]).
    Le coût d'un chemin est la somme des -log(poids) ; on cherche les
    chemins les plus forts (poids minimal élevé privilégié).
    """
    source, target = source.lower(), target.lower()
    if source not in index or target not in index:
        return []
    # File de priorité : (coût négatif du pire maillon, chemin, poids)
    heap: list[tuple[float, list[str], list[float]]] = [(-0.0, [source], [])]
    found: list[Chain] = []
    best_seen: dict[str, float] = {source: 0.0}

    while heap and len(found) < max_chains:
        neg_min, path, weights = heapq.heappop(heap)
        node = path[-1]
        if len(path) - 1 >= max_hops + 1:
            continue
        if node == target and len(path) > 1:
            found.append(Chain(path=path, weights=weights,
                               min_weight=min(weights)))
            continue
        if len(path) - 1 == max_hops:
            continue
        for neighbor, weight, _p in index.get(node, []):
            if neighbor in path or weight <= 0:
                continue
            # mots-outils : jamais intermédiaires (le nœud cible peut l'être)
            if neighbor in _SKIP_LINKS and neighbor != target:
                continue
            new_min = min(-neg_min, weight) if weights else weight
            # élagage : on ne revisite un nœud que si on y arrive plus fort
            if neighbor in best_seen and best_seen[neighbor] >= new_min:
                continue
            best_seen[neighbor] = new_min
            heapq.heappush(heap, (-new_min, path + [neighbor],
                                  weights + [weight]))
    found.sort(key=lambda c: -c.min_weight)
    return found


def explain_chain(chain: Chain, language: str = "en") -> str:
    """Traduit une chaîne en phrase honnête."""
    if language == "fr":
        steps = " ↔ ".join(chain.path)
        return (f"Chaîne d'association apprise : {steps} "
                f"(poids minimal {chain.min_weight:.4f} — corrélation, "
                f"pas causalité).")
    steps = " <-> ".join(chain.path)
    return (f"Learned association chain: {steps} "
            f"(minimum weight {chain.min_weight:.4f} — correlation, "
            f"not causation).")
