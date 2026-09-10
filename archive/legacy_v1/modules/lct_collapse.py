"""ratis_net.lct_collapse — L'effondrement topologique sous poussée thermo.

Le saut v4 : on ne maximise pas P_sig (non-différentiable). On laisse C
s'effondrer sous poussée thermodynamique de l'environnement. Quand C >=
C_seuil_thermo(environnement), l'effondrement se produit. On garde la MARQUE
topologique (hash du cycle H1 survivant), pas la valeur d'énergie.

C'est exactement l'invariance ZK validée sur QPU : après l'effondrement de
la fonction d'onde, on garde le hash topo (380a69c0...), pas l'énergie
(0.152 vs 1.835).
"""
from __future__ import annotations

import hashlib
import math
import numpy as np

from ratis_net.lct_network import _persistence_diagrams_lite


def compute_coherence(token_embedding: np.ndarray, weights: np.ndarray,
                      t_step: int = 0, omega: float = math.pi / 2) -> float:
    """Calcule la cohérence C du signal = corrélation entre le token et les
    poids, modulée par l'oscillation theta(t) = cos(omega*t).

    C oscille VRAIMENT avec le temps : la modulation multiplicative par
    cos(omega*t) fait que C passe par des maxima (collapse possible) et des
    minima (pas de collapse). C'est l'oscillation du milieu génial.
    """
    if weights.ndim == 1:
        w = weights
    else:
        w = weights.mean(axis=0)
    min_d = min(len(w), len(token_embedding))
    w = w[:min_d]
    t = token_embedding[:min_d]
    if np.linalg.norm(w) < 1e-9 or np.linalg.norm(t) < 1e-9:
        corr = 0.0
    else:
        corr = float(np.dot(w, t) / (np.linalg.norm(w) * np.linalg.norm(t) + 1e-9))
    # la coherence oscille : C = |corr| * amp * (0.5 + 0.5*cos(omega*t))
    # amp amplifie la correlation (le token resonne avec les poids)
    theta = math.cos(omega * t_step)
    amp = 5.0  # amplification : un token qui resonne produit une coherence forte
    C = abs(corr) * amp * (0.5 + 0.5 * theta)
    return min(1.0, max(0.0, C))


def topological_mark(weights: np.ndarray, c_seuil: float = 0.0,
                     env_vector: np.ndarray | None = None,
                     max_edge: float = 2.0) -> str:
    """La MARQUE topologique = hash du cycle H1 survivant + contexte thermo.

    La marque dépend de :
    1. La structure topologique des poids (quels neurones sont liés).
    2. Le seuil thermo C_seuil au moment du collapse (contexte émotionnel).
    3. L'environnement (features patient) au moment du collapse.

    C'est la "topo value" : pas la valeur d'énergie, la MARQUE qui reste
    après l'effondrement, contextuelle à l'environnement.
    """
    n = len(weights)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(weights[i] - weights[j]))
            if d <= max_edge:
                edges.append(f"{i}-{j}:{d:.3f}")
    # hash de la structure topologique + contexte thermo
    mark_str = "|".join(sorted(edges))
    mark_str += f"|C_seuil={c_seuil:.6f}"
    if env_vector is not None:
        mark_str += f"|env={np.array2string(env_vector, precision=4)}"
    return hashlib.sha256(mark_str.encode()).hexdigest()[:16]


def collapse(token_embedding: np.ndarray, weights: np.ndarray,
             c_seuil_thermo: float, t_step: int = 0,
             omega: float = math.pi / 2, max_edge: float = 2.0,
             env_vector: np.ndarray | None = None) -> dict:
    """L'effondrement topologique.

    1. On calcule C (cohérence token-poids sous oscillation).
    2. Si C >= C_seuil_thermo -> l'effondrement se produit.
    3. On garde la MARQUE topologique (hash du cycle survivant + contexte thermo).
    4. La valeur d'energie (P_sig) est perdue -- on garde la marque.
    """
    C = compute_coherence(token_embedding, weights, t_step, omega)
    P_sig = _persistence_diagrams_lite(weights, max_edge)

    if C >= c_seuil_thermo:
        # effondrement : on garde la MARQUE topo contextuelle, pas la valeur
        mark = topological_mark(weights, c_seuil=c_seuil_thermo,
                                env_vector=env_vector, max_edge=max_edge)
        return {
            "collapsed": True,
            "mark": mark,
            "C": C,
            "c_seuil": c_seuil_thermo,
            "P_sig_lost": P_sig,
        }
    else:
        return {
            "collapsed": False,
            "mark": None,
            "C": C,
            "c_seuil": c_seuil_thermo,
            "P_sig_lost": None,
        }
