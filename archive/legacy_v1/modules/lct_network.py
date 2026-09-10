"""ratis_net.lct_network — Le réseau LCT.

Un réseau de neurones où l'apprentissage se fait par LCT (pas de gradient).
À chaque step d'entraînement :
  1. Forward pass (activation modulée par C).
  2. Calcul de P_sig = persistance topologique de la matrice de poids globale.
  3. Update des poids par ΔW = η · φ · P_sig · C (loi LCT).

P_sig est calculé à chaque step → l'apprentissage s'auto-régule :
quand la topologie des poids devient robuste (P_sig élevé), les updates
deviennent plus amples → le réseau "accélère" sur les concepts robustes.
"""
from __future__ import annotations

import math
import numpy as np

from ratis_net.lct_neuron import LCTNeuron


def _persistence_diagrams_lite(points: np.ndarray, max_edge: float = 2.0):
    """Persistance H1 allégée (numpy pur) — version compacte pour l'entraînement.

    Sur un petit nombre de points (≤ 20), on calcule les cycles H1 par
    réduction de matrice de bordure simplifiée. Pour la vitesse, on utilise
    une approximation : H1 ≈ nombre de cycles indépendants au seuil max_edge.
    """
    n = len(points)
    if n < 3:
        return 0.0
    # distances
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = float(np.linalg.norm(points[i] - points[j]))
    # arêtes sous le seuil
    edges = [(D[i, j], i, j) for i in range(n) for j in range(i + 1, n) if D[i, j] <= max_edge]
    edges.sort()
    # union-find pour H0
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    n_components = n
    for d, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            n_components -= 1
    # H1 ≈ (arêtes - n + composantes) = cycles indépendants
    n_cycles = max(0, len(edges) - n + n_components)
    # P_sig ≈ persistance du cycle le plus long = (max_edge - min_edge) si cycles
    if n_cycles > 0 and len(edges) > 1:
        return float(edges[-1][0] - edges[0][0])  # persistance approximée
    return 0.0


class LCTNetwork:
    """Réseau LCT : entraîne par la loi LCT, pas par gradient descendant.

    Architecture : n_in → n_hidden → n_out (MLP).
    L'apprentissage maximise P_sig (persistance topologique des poids).
    """

    def __init__(self, n_in: int, n_hidden: int, n_out: int,
                 eta: float = 0.1, omega: float = math.pi / 2, seed: int = 42):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.eta = eta
        self.omega = omega
        # couche cachée : n_hidden neurones LCT
        self.hidden = [LCTNeuron(n_in, eta=eta, omega=omega, seed=seed + i)
                       for i in range(n_hidden)]
        # couche de sortie : n_out neurones LCT
        self.output = [LCTNeuron(n_hidden, eta=eta, omega=omega, seed=seed + 100 + i)
                        for i in range(n_out)]
        # historique P_sig pendant l'entraînement
        self.p_sig_history = []
        self.acc_history = []

    def _weight_matrix(self) -> np.ndarray:
        """Matrice de poids globale (tous les neurones concaténés, padés)."""
        rows = []
        for neuron in self.hidden + self.output:
            rows.append(neuron.weights)
        max_dim = max(r.shape[0] for r in rows)
        return np.array([np.pad(r, (0, max_dim - len(r))) for r in rows])

    def _compute_P_sig(self, max_edge: float = 2.0) -> float:
        """Calcule P_sig = persistance topologique de la matrice de poids."""
        W = self._weight_matrix()
        return _persistence_diagrams_lite(W, max_edge=max_edge)

    def forward(self, x: np.ndarray, t_step: int = 0) -> np.ndarray:
        """Forward pass : retourne les activations de sortie."""
        # couche cachée
        h = np.array([neuron.forward(x, t_step) for neuron in self.hidden])
        # couche de sortie
        out = np.array([neuron.forward(h, t_step) for neuron in self.output])
        return out

    def train_step(self, X: np.ndarray, y: np.ndarray, t_step: int = 0) -> float:
        """Un pas d'entraînement LCT sur tout le batch.

        Returns: accuracy sur le batch.
        """
        # calculer P_sig de la matrice de poids AVANT l'update
        P_sig = self._compute_P_sig(max_edge=2.0)
        self.p_sig_history.append(P_sig)

        correct = 0
        for i in range(len(X)):
            x = X[i]
            target = y[i]
            # forward
            h = np.array([neuron.forward(x, t_step) for neuron in self.hidden])
            out = np.array([neuron.forward(h, t_step) for neuron in self.output])
            # update de la couche de sortie
            for k, neuron in enumerate(self.output):
                neuron.update(h, target[k], P_sig, t_step)
            # update de la couche cachée (erreur propagée simplifiée)
            for j, neuron in enumerate(self.hidden):
                # erreur propagée = somme des erreurs de sortie pondérée
                err = sum((target[k] - out[k]) * self.output[k].weights[j]
                          for k in range(self.n_out))
                neuron.update(x, h[j] + err * 0.1, P_sig, t_step)
            # accuracy
            pred = np.argmax(out)
            if pred == np.argmax(target):
                correct += 1

        acc = correct / len(X)
        self.acc_history.append(acc)
        return acc

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, verbose: bool = True):
        """Entraîne le réseau par LCT sur plusieurs epochs."""
        for ep in range(epochs):
            t_step = ep  # la phase φ avance avec l'epoch
            acc = self.train_step(X, y, t_step=t_step)
            P_sig = self.p_sig_history[-1]
            if verbose and (ep % 10 == 0 or ep == epochs - 1):
                print(f"  Epoch {ep:3d} | acc = {acc:.3f} | P_sig = {P_sig:.4f} | "
                      f"φ = {math.cos(self.omega * ep):+.3f}")
        return self.acc_history, self.p_sig_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prédiction sur un batch."""
        return np.array([np.argmax(self.forward(x)) for x in X])
