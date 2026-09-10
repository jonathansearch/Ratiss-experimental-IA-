"""ratis_net.lct_modules.grav_measure — Mesure gravitationnelle topologique.

Extrait la "forme" de la densité gravitationnelle / des oscillations
informationnelles : cycles H1 de persistance mesurés sur la structure
d'intrication (cohérence → décohérence), comme une oscillation topologique
stable. Réutilise le backend persistance du dépôt (topo_tokenizer/_PERS_FN).
"""
from __future__ import annotations

import math
import numpy as np

from ..topo_tokenizer import _PERS_FN


def _rips_diagrams(points: np.ndarray, max_edge: float = 2.5) -> dict:
    """Diagrammes de persistance sur un nuage de points (backend du dépôt)."""
    if _PERS_FN is None:
        raise RuntimeError("backend de persistance indisponible")
    diagrams, _ = _PERS_FN(points, max_edge)
    return diagrams


class GravitationalTopoMeasure:
    """Mesure les corrélations topologiques (cycles H1) d'une densité/oscillation.

    Pipeline : densité → échantillonnage de niveaux (isosurfaces) → persistance
    H1 par niveau → oscillation C(θ) = |cos θ| vs P_sig(θ).
    """

    def __init__(self, max_edge: float = 2.5):
        self.max_edge = max_edge

    def measure_density(self, points: np.ndarray) -> dict:
        """P_sig + betti d'un nuage de densité (points N×d)."""
        diagrams = _rips_diagrams(points, self.max_edge)
        h1 = [d - b for b, d in diagrams.get(1, []) if d != float("inf") and d > b]
        return {
            "P_sig": float(max(h1)) if h1 else 0.0,
            "n_cycles": len(h1),
            "betti": [sum(1 for b, d in diagrams.get(k, []) if d == float("inf"))
                      for k in (0, 1, 2)],
        }

    def oscillation_profile(self, points: np.ndarray, n_steps: int = 8) -> list[dict]:
        """Oscillation cohérence/décohérence : θ balaye [0, π/2], C = |cos θ|.

        À chaque pas, on filtre les points par leur norme quantile q = min(0.5, 0.5·C)
        (compression TTF : haute cohérence = filtre strict) puis on mesure P_sig.
        Retourne la courbe (θ, C, P_sig) — la signature oscillationnelle.
        """
        norms = np.linalg.norm(points, axis=1)
        curve = []
        for k in range(n_steps):
            theta = (math.pi / 2) * k / (n_steps - 1)
            C = abs(math.cos(theta))
            q = min(0.5, 0.5 * C)
            thr = np.quantile(norms, 1 - q) if q > 0 else norms.min()
            kept = points[norms <= thr] if q > 0 else points
            if len(kept) < 4:
                curve.append({"theta": theta, "C": C, "P_sig": 0.0, "n_kept": len(kept)})
                continue
            m = self.measure_density(kept)
            curve.append({"theta": theta, "C": C, "P_sig": m["P_sig"],
                          "betti": m["betti"], "n_kept": len(kept)})
        return curve

    def density_field(self, n_shell: int = 40, n_bulk: int = 20,
                      curvature: float = 1.0, seed: int = 42) -> np.ndarray:
        """Génère une densité gravitationnelle de test : coquille + bulk courbé.

        curvature > 1 → coquille déformée (analogue gradient f'(r) du mur warp).
        """
        rng = np.random.RandomState(seed)
        shell = []
        for _ in range(n_shell):
            th = rng.uniform(0, 2 * math.pi)
            ph = rng.uniform(0, math.pi)
            r = 1.0 + curvature * 0.1 * math.sin(2 * ph)
            shell.append([r * math.sin(ph) * math.cos(th),
                          r * math.sin(ph) * math.sin(th),
                          r * math.cos(ph)])
        bulk = rng.uniform(-0.3, 0.3, size=(n_bulk, 3))
        return np.array(shell + bulk.tolist())
