"""ratis_net.lct_network_v2 — Réseau LCT v2 : maximise P_sig (gradient topo).

Différence avec v1 :
  v1 : ΔW = η·φ·P_sig·C → apprend mais P_sig est passager (oscille).
  v2 : ΔW = η·φ·P_sig·C + η2·∇_W(P_sig) → P_sig est MAXIMISÉ explicitement.

La loss n'est plus la cross-entropy. La loss = −P_sig.
Le réseau apprend EN devenant topologiquement robuste.
L'accuracy Iris n'est qu'un effet secondaire de cette robustesse.
"""
from __future__ import annotations

import math
import numpy as np

from ratis_net.lct_neuron import LCTNeuron
from ratis_net.lct_network import _persistence_diagrams_lite
from ratis_net.topo_gradient import compute_P_sig, topo_gradient


class LCTNetworkV2:
    """Réseau LCT v2 : maximise P_sig (gradient topologique).

    Update : ΔW = η·φ·P_sig·C·error + η2·∇_W(P_sig)
    Loss = −P_sig (on veut que la topologie devienne robuste).
    """

    def __init__(self, n_in: int, n_hidden: int, n_out: int,
                 eta: float = 0.05, eta2: float = 0.03, omega: float = math.pi / 2,
                 seed: int = 42):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.eta = eta
        self.eta2 = eta2
        self.omega = omega
        self.hidden = [LCTNeuron(n_in, eta=eta, omega=omega, seed=seed + i)
                       for i in range(n_hidden)]
        self.output = [LCTNeuron(n_hidden, eta=eta, omega=omega, seed=seed + 100 + i)
                        for i in range(n_out)]
        self.p_sig_history = []
        self.acc_history = []
        self.loss_history = []  # loss = -P_sig

    def _weight_matrix(self) -> np.ndarray:
        rows = [n.weights for n in self.hidden + self.output]
        max_dim = max(r.shape[0] for r in rows)
        return np.array([np.pad(r, (0, max_dim - len(r))) for r in rows])

    def _compute_P_sig(self, max_edge: float = 2.0) -> float:
        return compute_P_sig(self._weight_matrix(), max_edge=max_edge)

    def forward(self, x: np.ndarray, t_step: int = 0) -> np.ndarray:
        h = np.array([n.forward(x, t_step) for n in self.hidden])
        out = np.array([n.forward(h, t_step) for n in self.output])
        return out

    def train_step(self, X: np.ndarray, y: np.ndarray, t_step: int = 0) -> dict:
        """Un pas d'entraînement v2 : LCT + gradient topo (normalisé)."""
        W = self._weight_matrix()
        P_sig = compute_P_sig(W)
        self.p_sig_history.append(P_sig)
        self.loss_history.append(-P_sig)

        # gradient topo : ∇_W(P_sig), NORMALISÉ pour ne pas écraser le cycle
        topo_grad = topo_gradient(W, max_edge=2.0)
        grad_norm = np.linalg.norm(topo_grad)
        if grad_norm > 1e-9:
            topo_grad = topo_grad / grad_norm  # normalisé (amplitude contrôlée)
        elif P_sig < 1e-6:
            # bootstrap : P_sig=0, on perturbe légèrement pour recréer un cycle
            topo_grad = np.random.default_rng(t_step).normal(0, 0.01, W.shape)

        phi = math.cos(self.omega * t_step)

        correct = 0
        for i in range(len(X)):
            x = X[i]
            target = y[i]
            h = np.array([n.forward(x, t_step) for n in self.hidden])
            out = np.array([n.forward(h, t_step) for n in self.output])

            # update couche sortie (LCT + topo normalisé)
            W_out = np.array([n.weights for n in self.output])
            max_d = W_out.shape[1]
            topo_out = topo_grad[self.n_hidden:, :max_d]
            for k, n in enumerate(self.output):
                error = target[k] - out[k]
                C_k = n.C
                dW_lct = self.eta * phi * P_sig * C_k * error * h
                dW_topo = self.eta2 * topo_out[k]
                n.weights += dW_lct + dW_topo
                n.bias += self.eta * phi * P_sig * C_k * error

            # update couche cachée
            W_hid = np.array([n.weights for n in self.hidden])
            max_dh = W_hid.shape[1]
            topo_hid = topo_grad[:self.n_hidden, :max_dh]
            for j, n in enumerate(self.hidden):
                err = sum((target[k] - out[k]) * self.output[k].weights[j]
                          for k in range(self.n_out))
                C_j = n.C
                dW_lct = self.eta * phi * P_sig * C_j * err * x
                dW_topo = self.eta2 * topo_hid[j]
                n.weights += dW_lct + dW_topo
                n.bias += self.eta * phi * P_sig * C_j * err

            pred = np.argmax(out)
            if pred == np.argmax(target):
                correct += 1

        acc = correct / len(X)
        self.acc_history.append(acc)
        return {"acc": acc, "P_sig": P_sig, "loss": -P_sig}

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, verbose: bool = True):
        for ep in range(epochs):
            t_step = ep
            r = self.train_step(X, y, t_step=t_step)
            if verbose and (ep % 10 == 0 or ep == epochs - 1):
                print(f"  Epoch {ep:3d} | acc={r['acc']:.3f} | P_sig={r['P_sig']:.4f} | "
                      f"loss={r['loss']:.4f}")
        return self.acc_history, self.p_sig_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([np.argmax(self.forward(x)) for x in X])
