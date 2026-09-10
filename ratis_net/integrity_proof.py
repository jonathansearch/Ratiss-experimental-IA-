"""ratis_net.integrity_proof — Preuve d'intégrité du graphe Scalpel.

Ce que ce module FAIT :
  Pour une liste de concepts, il calcule une empreinte SHA-256 déterministe
  du sous-graphe de corrélations (paires, poids, P_sig, cohérences) plus les
  métriques LCT (P_sig du cluster, monotonicité). Même concepts + même
  checkpoint → même empreinte. On peut donc vérifier qu'une réponse repose
  sur un état précis du réseau.

Ce que ce module NE FAIT PAS :
  Ce n'est PAS une preuve cryptographique à divulgation nulle (ZK-STARK).
  Un STARK prouve qu'un calcul a été exécuté correctement SANS révéler les
  données, avec une vérification sous-linéaire. Ici, la vérification exige
  de recalculer l'empreinte depuis le checkpoint complet. C'est une preuve
  d'INTÉGRITÉ (engagement sur les données), pas de CORRECTION CALCULATOIRE
  privée. Un vrai module STARK nécessiterait un backend d'arithmétisation
  (AIR/FRI) — voir PISTES OUVERTES.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class IntegrityProof:
    """Empreinte déterministe d'un sous-graphe de concepts."""
    concepts: list[str]
    n_edges: int
    total_weight: float
    p_sig_max: float
    digest: str
    algorithm: str = "sha256-subgraph-v1"
    proof_type: str = "integrity_commitment"  # PAS un ZK-STARK

    def to_dict(self) -> dict[str, Any]:
        return {
            "concepts": self.concepts,
            "n_edges": self.n_edges,
            "total_weight": round(self.total_weight, 6),
            "p_sig_max": round(self.p_sig_max, 6),
            "digest": self.digest,
            "algorithm": self.algorithm,
            "proof_type": self.proof_type,
            "zk_stark": False,
        }


def _canonical_edge(a: str, b: str, weight: float, p_sig: float,
                    coherence: float) -> str:
    lo, hi = (a, b) if a <= b else (b, a)
    return f"{lo}|{hi}|{weight:.8f}|{p_sig:.8f}|{coherence:.8f}"


def prove(concepts: list[str], scalpel: Any) -> IntegrityProof:
    """Calcule l'empreinte du sous-graphe induit par `concepts`.

    scalpel : objet avec attribut `neurons` (dict (a,b) -> ScalpelNeuron).
    """
    concept_set = {c.lower() for c in concepts}
    edges: list[str] = []
    total_weight = 0.0
    p_sig_max = 0.0
    for (a, b), neuron in scalpel.neurons.items():
        if a in concept_set or b in concept_set:
            edges.append(_canonical_edge(a, b, neuron.weight, neuron.p_sig,
                                         neuron.coherence))
            total_weight += neuron.weight
            p_sig_max = max(p_sig_max, neuron.p_sig)
    edges.sort()
    h = hashlib.sha256()
    h.update(json.dumps(sorted(concept_set), ensure_ascii=False).encode())
    for e in edges:
        h.update(e.encode())
    return IntegrityProof(
        concepts=sorted(concept_set),
        n_edges=len(edges),
        total_weight=total_weight,
        p_sig_max=p_sig_max,
        digest=h.hexdigest(),
    )


def verify(proof: IntegrityProof, scalpel: Any) -> bool:
    """Vérifie qu'un recalcul depuis le checkpoint reproduit l'empreinte."""
    recomputed = prove(proof.concepts, scalpel)
    return recomputed.digest == proof.digest
