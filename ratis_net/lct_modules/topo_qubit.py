"""ratis_net.lct_modules.topo_qubit — Simulation algorithmique d'un qubit topologique.

Modélise en logiciel un qubit dont l'information est portée par un invariant
topologique (résidu H1) plutôt que par l'amplitude d'un état. L'état logique
= signature (P_sig, phase θ, cohérence C). Les portes agissent sur les résidus ;
la protection topologique = une erreur locale ne change pas l'invariant tant que
le cycle persiste (persistance > seuil de bruit).

Aucune prétention matérielle : c'est l'algorithme prêt AVANT la puce.
"""
from __future__ import annotations

import math
import numpy as np

from .grav_measure import GravitationalTopoMeasure


class TopologicalQubit:
    """Qubit simulé dont le bit logique est encodé dans la topologie du réseau.

    Encodage : réseau de nœuds (anneau + liens d'intrication). |0> = topologie
    triviale du réseau, |1> = un cycle H1 persistant (résidu topologique).
    Mesurer = extraire P_sig du réseau ; le bit est lu par seuil.
    """

    def __init__(self, n_nodes: int = 12, protection: float = 0.15, seed: int = 42):
        self.n_nodes = n_nodes
        self.protection = protection  # seuil de persistance = protection topologique
        self.rng = np.random.RandomState(seed)
        self.measure = GravitationalTopoMeasure(max_edge=2.5)
        # réseau : anneau de nœuds (base), état logique initial |0>
        self._theta = 0.0        # phase (superposition analogue)
        self._twist = 0.0        # torsion du réseau : 0 = |0>, π = |1>
        self._coherence = 1.0

    def _network(self) -> np.ndarray:
        """Réseau de nœuds pour l'état courant : anneau tordu (twist).
        twist=0 → anneau fermé lisse ; twist→π → défaut local (cycle H1 marqué)."""
        pts = []
        for i in range(self.n_nodes):
            a = 2 * math.pi * i / self.n_nodes
            r = 1.0 + (self._twist / math.pi) * 0.4 * math.sin(3 * a)
            pts.append([r * math.cos(a), r * math.sin(a),
                        (self._twist / math.pi) * 0.3 * math.cos(3 * a)])
        return np.array(pts)

    # ── Portes logiques (agissent sur les résidus, pas sur les amplitudes) ──

    def x_gate(self) -> "TopologicalQubit":
        """NOT topologique : bascule la torsion 0 ↔ π (crée/annule le résidu)."""
        self._twist = math.pi - self._twist
        return self

    def h_gate(self) -> "TopologicalQubit":
        """Hadamard topologique : superposition = torsion intermédiaire π/2."""
        self._twist = math.pi / 2
        self._theta = math.pi / 4
        return self

    def phase_gate(self, dtheta: float) -> "TopologicalQubit":
        """Déphase la cohérence du réseau (analogue rotation Z)."""
        self._theta = (self._theta + dtheta) % (2 * math.pi)
        return self

    def noise(self, strength: float) -> None:
        """Bruit local : dégrade la cohérence. La protection topologique fait
        que le bit logique survit tant que la persistance > protection."""
        self._coherence = max(0.0, self._coherence - strength)

    # ── Lecture ──

    def measure_state(self) -> dict:
        """Mesure non destructive de la signature topologique du réseau."""
        pts = self._network()
        # bruit de mesure proportionnel à la décohérence
        pts = pts + self.rng.normal(0, (1 - self._coherence) * 0.05, pts.shape)
        m = self.measure.measure_density(pts)
        protected = m["P_sig"] > self.protection
        return {
            "P_sig": m["P_sig"],
            "n_cycles": m["n_cycles"],
            "betti": m["betti"],
            "twist": self._twist,
            "coherence": self._coherence,
            "protected": bool(protected),
            "logical_bit": int(m["P_sig"] > self.protection),
        }

    def fidelity_vs_ideal(self) -> float:
        """Fidélité = cohérence restante (le bit logique tient si protected)."""
        return self._coherence
