"""ratis_net.topo_proxy — Proxy différentiable de la robustesse topologique.

P_sig (persistance H1) n'est PAS différentiable : c'est un max de différences
de distances qui changent brusquement quand les arêtes du complexe de Rips
changent. Le "gradient par différence finie" est instable → détruit le cycle.

Solution : un PROXY DIFFERENTIABLE de la robustesse topologique.

Proxy = variance des distances inter-neurones.
  - Topologie robuste (P_sig élevé) = neurones bien répartis = variance élevée.
  - Topologie fragile (P_sig bas) = neurones concentrés = variance faible.

La variance est LISSE et DIFFERENTIABLE. Maximiser la variance = distribuer
les neurones = rendre la topologie robuste. C'est l'analogue différentiable
de "maximiser P_sig".

ΔW = η·φ·P_sig·C + η2·∇_W(variance(distances))

Le gradient de la variance est analytique :
  variance = (1/n) Σ d_ij² - (mean d)²
  ∂var/∂w = (chaque poids influence les distances d_ij)

On calcule ∇_W(variance) par autograd numpy (différence finie sur la variance,
qui EST lisse → stable, contrairement à P_sig).
"""
from __future__ import annotations

import numpy as np

from ratis_net.lct_network import _persistence_diagrams_lite


def weight_distances(W: np.ndarray) -> np.ndarray:
    """Distances entre les lignes (neurones) de la matrice de poids."""
    n = W.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(W[i] - W[j]))
            D[i, j] = D[j, i] = d
    return D


def topo_robustness_proxy(W: np.ndarray) -> float:
    """Proxy différentiable de la robustesse topologique.

    = variance des distances inter-neurones. élevée = topologie robuste.
    """
    D = weight_distances(W)
    # variance des distances hors-diagonale
    triu = D[np.triu_indices_from(D, k=1)]
    if len(triu) == 0:
        return 0.0
    return float(np.var(triu))


def topo_proxy_gradient(W: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
    """∇_W(proxy) : gradient de la variance des distances.

    La variance est LISSE → la différence finie est STABLE (contrairement à P_sig).
    On maximise la variance → le gradient pousse vers une distribution des neurones.
    """
    n_rows, n_cols = W.shape
    grad = np.zeros_like(W)
    base = topo_robustness_proxy(W)
    for i in range(n_rows):
        for j in range(n_cols):
            W_plus = W.copy(); W_plus[i, j] += epsilon
            W_minus = W.copy(); W_minus[i, j] -= epsilon
            grad[i, j] = (topo_robustness_proxy(W_plus) - topo_robustness_proxy(W_minus)) / (2 * epsilon)
    return grad


def compute_P_sig(W: np.ndarray, max_edge: float = 2.0) -> float:
    """P_sig (le vrai, non-différentiable) — pour le monitoring."""
    return _persistence_diagrams_lite(W, max_edge=max_edge)
