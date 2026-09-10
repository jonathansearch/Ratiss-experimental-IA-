"""ratis_net.lct_network_v3 — Réseau LCT v3 : proxy topo différentiable.

v1 : ΔW = η·φ·P_sig·C → apprend, P_sig passager (oscille).
v2 : + η2·∇_W(P_sig) → P_sig NON-différentiable → effondrement.
v3 : + η2·∇_W(variance distances) → proxy DIFFERENTIABLE → stable.

ΔW = η·φ·P_sig·C + η2·∇_W(variance)
loss = -P_sig (monitoring) + -variance (proxy optimisé)

Le réseau apprend (LCT) EN maximisant la robustesse topologique (proxy).
"""
from __future__ import annotations

import math
import numpy as np

from ratis_net.lct_neuron import LCTNeuron
from ratis_net.topo_proxy import compute_P_sig, topo_robustness_proxy, topo_proxy_gradient


class LCTNetworkV3:
    """Réseau LCT v3 : LCT + proxy topo différentiable (variance des distances)."""

    def __init__(self, n_in: int, n_hidden: int, n_out: int,
                 eta: float = 0.05, eta2: float = 0.001, omega: float = math.pi / 2,
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
        self.variance_history = []
        self.acc_history = []

    def _weight_matrix(self) -> np.ndarray:
        rows = [n.weights for n in self.hidden + self.output]
        max_dim = max(r.shape[0] for r in rows)
        return np.array([np.pad(r, (0, max_dim - len(r))) for r in rows])

    def forward(self, x: np.ndarray, t_step: int = 0) -> np.ndarray:
        h = np.array([n.forward(x, t_step) for n in self.hidden])
        out = np.array([n.forward(h, t_step) for n in self.output])
        return out

    def train_step(self, X: np.ndarray, y: np.ndarray, t_step: int = 0) -> dict:
        W = self._weight_matrix()
        P_sig = compute_P_sig(W)
        variance = topo_robustness_proxy(W)
        self.p_sig_history.append(P_sig)
        self.variance_history.append(variance)

        # gradient du proxy (DIFFERENTIABLE → stable)
        topo_grad = topo_proxy_gradient(W)
        grad_norm = np.linalg.norm(topo_grad)
        if grad_norm > 1e-9:
            topo_grad = topo_grad / grad_norm  # normalisé

        phi = math.cos(self.omega * t_step)

        correct = 0
        for i in range(len(X)):
            x = X[i]; target = y[i]
            h = np.array([n.forward(x, t_step) for n in self.hidden])
            out = np.array([n.forward(h, t_step) for n in self.output])

            # update couche sortie
            max_d = len(self.output[0].weights)
            topo_out = topo_grad[self.n_hidden:, :max_d]
            for k, n in enumerate(self.output):
                error = target[k] - out[k]
                C_k = n.C
                dW_lct = self.eta * phi * P_sig * C_k * error * h
                dW_topo = self.eta2 * topo_out[k]
                n.weights += dW_lct + dW_topo
                n.bias += self.eta * phi * P_sig * C_k * error

            # update couche cachée
            max_dh = len(self.hidden[0].weights)
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
        return {"acc": acc, "P_sig": P_sig, "variance": variance}

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, verbose: bool = True):
        for ep in range(epochs):
            r = self.train_step(X, y, t_step=ep)
            if verbose and (ep % 10 == 0 or ep == epochs - 1):
                print(f"  Epoch {ep:3d} | acc={r['acc']:.3f} | P_sig={r['P_sig']:.4f} | "
                      f"var={r['variance']:.4f}")
        return self.acc_history, self.p_sig_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([np.argmax(self.forward(x)) for x in X])
