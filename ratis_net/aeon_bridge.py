"""ratis_net.aeon_bridge — Pont entre RATIS-Net (langage) et AEON (science).

Super RATISS : fusion des deux cerveaux.
  - RATIS-Net : comprend la question, extrait les concepts, formule la réponse.
  - AEON ODV  : calcule les invariants scientifiques (LCT, P_sig, topologie).

Protocole de communication :
  RATIS-Net envoie une requête structurée {concepts, query}
  AEON renvoie un objet {fact, proof, confidence, source}

Le bridge ne modifie ni la loi LCT ni le Scalpel. Il lit les deux et fait
l'interface. Si AEON n'est pas disponible (pas installé), le bridge retombe
sur le Scalpel seul (mode dégradé honnête).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


class AeonFact:
    """Un fait scientifique calculé par AEON, avec preuve et confiance."""

    def __init__(self, fact: str, proof: str, confidence: float,
                 source: str = "aeon_odv", raw_data: dict | None = None):
        self.fact = fact
        self.proof = proof
        self.confidence = confidence
        self.source = source
        self.raw_data = raw_data or {}

    def to_dict(self) -> dict:
        return {"fact": self.fact, "proof": self.proof,
                "confidence": self.confidence, "source": self.source,
                "raw_data": self.raw_data}

    def __repr__(self) -> str:
        return f"AeonFact({self.fact[:60]}... conf={self.confidence:.2f})"


class AeonBridge:
    """Pont entre RATIS-Net et le cœur scientifique intégré (science_core).

    TOUT est dans un seul package. Aucune dépendance externe, aucun sys.path
    vers un autre dépôt. Les fonctions d'AEON ODV (P_sig, LCT, Vietoris-Rips)
    sont directement dans ratis_net/science_core.py.

    Le bridge est l'interface qui permet à RATIS-Net d'appeler les fonctions
    scientifiques : compute_p_sig, measure_lct, scan_monotonicity, etc.
    """

    def __init__(self, aeon_path: str | Path | None = None,
                 engine_path: str | Path | None = None):
        # aeon_path et engine_path sont ignorés — tout est intégré.
        # Conservés pour compatibilité API mais non utilisés.
        self._available = True  # toujours disponible (intégré)

    @property
    def available(self) -> bool:
        return True

    @property
    def backend_name(self) -> str:
        return "integrated_science_core"

    def compute_p_sig(self, points: np.ndarray, max_edge: float = 2.0) -> float:
        """Calcule P_sig via science_core (Vietoris-Rips GF(2) intégré)."""
        from ratis_net.science_core import compute_p_sig
        return compute_p_sig(points, max_edge=max_edge)

    def _local_p_sig(self, distance: np.ndarray, max_edge: float) -> float:
        """Calcul local de P_sig si aucun backend n'est disponible."""
        from itertools import combinations
        n = distance.shape[0]
        if n < 4:
            return 0.0
        # H1 simple : chercher des cycles de longueur 3+
        h1_persistences = []
        for i, j, k in combinations(range(n), 3):
            birth = max(distance[i, j], distance[i, k], distance[j, k])
            if birth <= max_edge:
                # persistence approximative = max_edge - birth
                h1_persistences.append(max_edge - birth)
        return float(max(h1_persistences, default=0.0))

    def query(self, concepts: list[str], scalpel=None) -> AeonFact:
        """Interroge AEON pour valider des concepts scientifiques.

        Retourne un AeonFact avec :
          - fact : description du concept
          - proof : preuve (P_sig calculé ou corrélation LCT)
          - confidence : 0-1
        """
        if not concepts:
            return AeonFact("No concepts provided", "none", 0.0, source="empty")

        # Construire un nuage de points depuis les embeddings des concepts
        # (si le Scalpel est connecté, utiliser ses poids comme coordonnées)
        if scalpel is not None and scalpel._indexed if hasattr(scalpel, '_indexed') else False:
            points = []
            for concept in concepts[:10]:
                corrs = scalpel._corrs(concept) if hasattr(scalpel, '_corrs') else []
                if corrs:
                    # Utiliser les 3 premiers poids comme coordonnées 3D
                    row = [corrs[0][1], corrs[1][1] if len(corrs) > 1 else 0,
                           corrs[2][1] if len(corrs) > 2 else 0]
                    points.append(row)
            if len(points) >= 4:
                points = np.array(points)
                p_sig = self.compute_p_sig(points)
                fact = f"The topological persistence P_sig of the concept cluster [{', '.join(concepts[:5])}] is {p_sig:.4f}."
                proof = f"Computed via {self.backend_name}: Vietoris-Rips H1 persistence = {p_sig:.6f}."
                confidence = min(1.0, p_sig * 2.0)
                return AeonFact(fact, proof, confidence, source=self.backend_name,
                                raw_data={"p_sig": p_sig, "n_concepts": len(concepts),
                                          "n_points": len(points)})

        # Fallback : pas assez de données
        fact = f"Concepts identified: {', '.join(concepts[:5])}. No topological computation available."
        proof = "Scalpel index not built or insufficient correlations."
        confidence = 0.3
        return AeonFact(fact, proof, confidence, source="fallback")

    def validate_monotonicity(self, coords: np.ndarray) -> dict:
        """Valide la monotonicité R(C) de la loi LCT sur des coordonnées."""
        from ratis_net.science_core import scan_monotonicity, evaluate_monotonicity
        measurements = scan_monotonicity(coords)
        result = evaluate_monotonicity(measurements)
        return {"valid": result.get("monotone", False),
                "spearman": result.get("spearman", 0.0),
                "source": "integrated_science_core"}


if __name__ == "__main__":
    bridge = AeonBridge(
        aeon_path=Path("/workspace/aeon"),
        engine_path=Path("/workspace/engine/src"),
    )
    print(f"Backend: {bridge.backend_name} (available: {bridge.available})")

    # Test : P_sig d'un nuage de points
    rng = np.random.default_rng(42)
    points = rng.normal(0, 1, (20, 3))
    p_sig = bridge.compute_p_sig(points)
    print(f"P_sig (random 20 points): {p_sig:.6f}")

    # Test : query avec concepts
    fact = bridge.query(["quantum", "mechanics", "physics", "theory", "energy"])
    print(f"\nFact: {fact.fact}")
    print(f"Proof: {fact.proof}")
    print(f"Confidence: {fact.confidence:.2f}")
