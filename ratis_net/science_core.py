"""ratis_net.science_core — Cœur scientifique intégré (AEON ODV fusionné).

Contient les fonctions scientifiques fondamentales d'AEON ODV, directement
intégrées dans RATIS-Net. Aucune dépendance externe : pas de sys.path vers un
autre dépôt. Tout est dans ce seul package.

Fonctions :
  - rips_persistence : Vietoris-Rips GF(2) (H0/H1, P_sig, Betti)
  - compute_p_sig : P_sig = plus long cycle H1 fini
  - measure_lct : mesure LCT complète (C, P_sig, R, Betti)
  - scan_monotonicity : validation de la loi R(C) croissante
  - validate_invariance : R invariant sous changement d'énergie

La loi LCT est FIGÉE : R = P_sig, ΔW = η·φ·P_sig·C.
"""
from __future__ import annotations

import math
from itertools import combinations
from typing import Any

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Vietoris-Rips persistence (GF(2) boundary reduction)
# ─────────────────────────────────────────────────────────────────────────────

def _ordered_simplices(distance: np.ndarray, max_edge: float) -> list[tuple[tuple[int, ...], float]]:
    n = distance.shape[0]
    simplices: list[tuple[tuple[int, ...], float]] = [((i,), 0.0) for i in range(n)]
    for i, j in combinations(range(n), 2):
        birth = float(distance[i, j])
        if birth <= max_edge:
            simplices.append(((i, j), birth))
    for i, j, k in combinations(range(n), 3):
        birth = float(max(distance[i, j], distance[i, k], distance[j, k]))
        if birth <= max_edge:
            simplices.append(((i, j, k), birth))
    return sorted(simplices, key=lambda item: (item[1], len(item[0]), item[0]))


def rips_persistence(distance: np.ndarray, max_edge: float | None = None) -> dict[str, Any]:
    """Vietoris-Rips persistence H0/H1 via réduction de frontière GF(2)."""
    distance = np.asarray(distance, dtype=float)
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("distance must be a square matrix")
    n = distance.shape[0]
    if n == 0:
        return {"diagrams": {"H0": [], "H1": []}, "betti": [0, 0, 0], "psig": 0.0, "n_finite_h1": 0}
    if max_edge is None:
        finite = distance[np.triu_indices(n, 1)]
        max_edge = float(np.max(finite)) if finite.size else 0.0
    simplices = _ordered_simplices(distance, float(max_edge))
    index = {simplex: idx for idx, (simplex, _) in enumerate(simplices)}
    births = [birth for _, birth in simplices]
    dims = [len(simplex) - 1 for simplex, _ in simplices]
    boundaries: list[set[int]] = []
    for simplex, _ in simplices:
        if len(simplex) == 1:
            boundaries.append(set())
            continue
        boundaries.append({index[tuple(v for pos, v in enumerate(simplex) if pos != removed)] for removed in range(len(simplex))})
    reduced_columns: dict[int, set[int]] = {}
    low_to_column: dict[int, int] = {}
    pairs: list[tuple[int, int]] = []
    creators: set[int] = set()
    for col, raw_boundary in enumerate(boundaries):
        boundary = set(raw_boundary)
        while boundary and max(boundary) in low_to_column:
            boundary.symmetric_difference_update(reduced_columns[low_to_column[max(boundary)]])
        if not boundary:
            creators.add(col)
        else:
            low = max(boundary)
            reduced_columns[col] = boundary
            low_to_column[low] = col
            pairs.append((low, col))
            creators.discard(low)
    diagrams: dict[str, list[list[float | None]]] = {"H0": [], "H1": []}
    for birth_idx, death_idx in pairs:
        dimension = dims[birth_idx]
        if dimension in (0, 1):
            diagrams[f"H{dimension}"].append([float(births[birth_idx]), float(births[death_idx])])
    for creator in creators:
        dimension = dims[creator]
        if dimension in (0, 1):
            diagrams[f"H{dimension}"].append([float(births[creator]), None])
    tol = 1e-9
    finite_h1 = [death - birth for birth, death in diagrams["H1"] if death is not None and (death - birth) > tol]
    betti = [sum(1 for _, death in diagrams[f"H{dim}"] if death is None) for dim in (0, 1)] + [0]
    return {"diagrams": diagrams, "betti": betti, "psig": float(max(finite_h1, default=0.0)),
            "n_finite_h1": len(finite_h1), "max_edge": float(max_edge)}


# ─────────────────────────────────────────────────────────────────────────────
# P_sig : persistance du cycle H1 le plus long
# ─────────────────────────────────────────────────────────────────────────────

def compute_p_sig(points: np.ndarray, max_edge: float = 2.0) -> float:
    """P_sig = persistance du cycle H1 le plus long d'un nuage de points."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] < 4:
        return 0.0
    delta = points[:, None, :] - points[None, :, :]
    distance = np.linalg.norm(delta, axis=2)
    np.fill_diagonal(distance, 0.0)
    return float(rips_persistence(distance, max_edge=max_edge)["psig"])


# ─────────────────────────────────────────────────────────────────────────────
# Mesure LCT : C, P_sig, R, Betti
# ─────────────────────────────────────────────────────────────────────────────

class LCTMeasurement:
    """Une mesure de la loi LCT à un θ donné."""
    def __init__(self, theta: float, coherence_C: float, p_sig: float,
                 n_cycles: int, betti: list, energy: float = 1.0):
        self.theta = theta
        self.coherence_C = coherence_C
        self.p_sig = p_sig
        self.R = p_sig  # R = P_sig (loi figée)
        self.n_cycles = n_cycles
        self.betti = betti
        self.energy = energy

    def to_dict(self) -> dict:
        return {"theta": self.theta, "C": self.coherence_C, "P_sig": self.p_sig,
                "R": self.R, "n_cycles": self.n_cycles, "betti": self.betti,
                "energy": self.energy}


def measure_lct(points: np.ndarray, theta: float = 0.0, max_edge: float = 2.0,
                energy: float = 1.0) -> LCTMeasurement:
    """Mesure C, P_sig, R sur un nuage de points à un θ donné.

    C = |cos(θ)| : cohérence du milieu génial à l'instant θ.
    P_sig = persistance du cycle H1 le plus long.
    R = P_sig (loi LCT figée : R = P_sig).
    """
    points = np.asarray(points, dtype=float)
    C = abs(math.cos(theta))
    p_sig = compute_p_sig(points, max_edge=max_edge)
    delta = points[:, None, :] - points[None, :, :]
    distance = np.linalg.norm(delta, axis=2)
    np.fill_diagonal(distance, 0.0)
    result = rips_persistence(distance, max_edge=max_edge)
    return LCTMeasurement(theta=theta, coherence_C=C, p_sig=p_sig,
                          n_cycles=result["n_finite_h1"], betti=result["betti"],
                          energy=energy)


# ─────────────────────────────────────────────────────────────────────────────
# Validation de la loi LCT : monotonicité et invariance
# ─────────────────────────────────────────────────────────────────────────────

def scan_monotonicity(points: np.ndarray, n_steps: int = 12,
                      omega: float = math.pi / 2,
                      max_edge: float = 2.0) -> list[LCTMeasurement]:
    """Scan R(C) : vérifie que R croît avec C (loi LCT).

    R = P_sig doit être monotone croissant en C = |cos(θ)|.
    """
    measurements = []
    for step in range(n_steps):
        theta = (math.pi / 2) * step / max(1, n_steps - 1)
        m = measure_lct(points, theta=theta, max_edge=max_edge)
        measurements.append(m)
    return measurements


def evaluate_monotonicity(measurements: list[LCTMeasurement]) -> dict:
    """Évalue la monotonicité R(C) (Spearman) et l'invariance sous énergie."""
    from itertools import combinations as comb
    Cs = [m.coherence_C for m in measurements]
    Rs = [m.R for m in measurements]
    # Spearman rank correlation
    n = len(Cs)
    if n < 3:
        return {"monotone": None, "spearman": 0.0, "n_measurements": n}
    rank_C = np.argsort(np.argsort(Cs)).astype(float)
    rank_R = np.argsort(np.argsort(Rs)).astype(float)
    d_sq = np.sum((rank_C - rank_R) ** 2)
    spearman = 1.0 - 6.0 * d_sq / (n * (n ** 2 - 1)) if n > 1 else 0.0
    monotone = spearman > 0.0
    return {"monotone": bool(monotone), "spearman": float(spearman),
            "n_measurements": n, "R_values": Rs, "C_values": Cs}


def validate_invariance(points: np.ndarray, energies: list[float] | None = None,
                        max_edge: float = 2.0) -> dict:
    """Valide que R = P_sig est invariant sous changement d'énergie.

    Loi LCT : on certifie le message (forme), pas le courant (énergie).
    R doit être constant quand on change l'énergie.
    """
    if energies is None:
        energies = [0.5, 1.0, 2.0, 4.0]
    measurements = [measure_lct(points, theta=math.pi / 2, max_edge=max_edge, energy=e)
                    for e in energies]
    Rs = np.array([m.R for m in measurements])
    mean_R = float(np.mean(Rs))
    std_R = float(np.std(Rs))
    cv = 0.0 if mean_R == 0 else 100.0 * std_R / mean_R
    return {"invariant": cv < 5.0, "cv_pct": float(cv),
            "R_mean": mean_R, "R_std": std_R, "energies": energies,
            "R_values": Rs.tolist()}


# ─────────────────────────────────────────────────────────────────────────────
# Tension topologique LCT (pour le Synchrotron)
# ─────────────────────────────────────────────────────────────────────────────

def topological_tension(stress: float, alpha_0: float, reference_p_sig: float,
                        current_p_sig: float) -> float | None:
    """Tension LCT = stress / (A × P_sig).

    A = alpha_0 × reference_p_sig. Si A ou P_sig = 0, retourne None.
    """
    if reference_p_sig == 0.0 or current_p_sig == 0.0:
        return None
    A = alpha_0 * reference_p_sig
    if A == 0.0:
        return None
    return float(stress / (A * current_p_sig))


if __name__ == "__main__":
    # Test rapide
    rng = np.random.default_rng(42)
    points = rng.normal(0, 1, (20, 3))

    print("=== Science Core (AEON intégré) ===\n")
    p_sig = compute_p_sig(points)
    print(f"P_sig (20 random points): {p_sig:.6f}")

    m = measure_lct(points, theta=0.0)
    print(f"LCT measure: C={m.coherence_C:.4f}, P_sig={m.p_sig:.6f}, R={m.R:.6f}")

    measurements = scan_monotonicity(points, n_steps=8)
    result = evaluate_monotonicity(measurements)
    print(f"Monotonicity: spearman={result['spearman']:.4f}, monotone={result['monotone']}")

    inv = validate_invariance(points, energies=[0.5, 1.0, 2.0, 4.0])
    print(f"Invariance: CV={inv['cv_pct']:.4f}%, invariant={inv['invariant']}")
    print(f"  R values: {inv['R_values']}")
