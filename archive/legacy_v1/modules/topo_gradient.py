"""ratis_net.topo_gradient — Le gradient topologique ∇_W(P_sig).

La clé du saut : au lieu de faire un gradient de LOSS (cross-entropy), on
fait un gradient de TOPOLOGIE. On pousse explicitement les poids à rendre le
cycle H1 plus long.

∇_W(P_sig) = comment bouger chaque poids pour augmenter la persistance
topologique de la matrice de poids.

Méthode (différence finie, approximable sur hardware) :
  Pour chaque poids w_ij, on calcule :
    ∂P_sig/∂w_ij ≈ [P_sig(W + ε·e_ij) - P_sig(W - ε·e_ij)] / (2ε)

On maximise P_sig : l'update devient
    ΔW = η · φ · P_sig · C  +  η2 · ∇_W(P_sig)

Le 1er terme (LCT) signe la direction (supervision).
Le 2e terme (gradient topo) pousse la robustesse topologique.
Le réseau apprend EN devenant topologiquement robuste.
"""
from __future__ import annotations

import numpy as np

from ratis_net.lct_network import _persistence_diagrams_lite


def compute_P_sig(weights_matrix: np.ndarray, max_edge: float = 2.0) -> float:
    """Calcule P_sig d'une matrice de poids (chaque ligne = un neurone)."""
    return _persistence_diagrams_lite(weights_matrix, max_edge=max_edge)


def topo_gradient(weights_matrix: np.ndarray, epsilon: float = 0.05,
                  max_edge: float = 2.0) -> np.ndarray:
    """Calcule ∇_W(P_sig) par différence finie.

    Pour chaque poids w_ij, on mesure comment P_sig change quand on bouge
    w_ij de ±epsilon. C'est le gradient de topologie.

    Args:
        weights_matrix: matrice de poids (n_neurons × n_inputs).
        epsilon: pas de différence finie.
        max_edge: seuil du complexe de Rips.

    Returns:
        Matrice (n_neurons × n_inputs) des dérivées ∂P_sig/∂w_ij.
    """
    n_rows, n_cols = weights_matrix.shape
    grad = np.zeros_like(weights_matrix)
    P_base = compute_P_sig(weights_matrix, max_edge)
    for i in range(n_rows):
        for j in range(n_cols):
            # W + ε·e_ij
            W_plus = weights_matrix.copy()
            W_plus[i, j] += epsilon
            P_plus = compute_P_sig(W_plus, max_edge)
            # W - ε·e_ij
            W_minus = weights_matrix.copy()
            W_minus[i, j] -= epsilon
            P_minus = compute_P_sig(W_minus, max_edge)
            # dérivée centrée
            grad[i, j] = (P_plus - P_minus) / (2 * epsilon)
    return grad


def lct_plus_topo_update(weights: np.ndarray, error_grad: np.ndarray,
                         P_sig: float, phi: float, C: float,
                         eta: float = 0.1, eta2: float = 0.05,
                         max_edge: float = 2.0) -> np.ndarray:
    """Update combiné LCT + gradient topologique.

    ΔW = η · φ · P_sig · C · error_grad  +  η2 · ∇_W(P_sig)

    Le 1er terme : loi LCT (supervision signée par error_grad).
    Le 2e terme : gradient de topologie (pousse P_sig à croître).

    On MAXIMISE P_sig → le 2e terme est ajouté (pas soustrait).

    Args:
        weights: matrice de poids (n_neurons × n_inputs).
        error_grad: gradient d'erreur (signe la direction, LCT).
        P_sig: persistance topologique actuelle.
        phi: phase du milieu génial.
        C: cohérence du signal.
        eta: taux LCT.
        eta2: taux gradient topo.
    """
    # terme 1 : LCT
    lct_term = eta * phi * P_sig * C * error_grad
    # terme 2 : gradient topo (maximise P_sig)
    topo_grad = topo_gradient(weights, max_edge=max_edge)
    topo_term = eta2 * topo_grad
    # update combiné
    return weights + lct_term + topo_term
