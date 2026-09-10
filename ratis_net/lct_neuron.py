"""ratis_net.lct_neuron — Le neurone LCT.

Contrairement à un neurone standard (sigmoid + gradient), le neurone LCT :
  - Utilise φ (phase du milieu génial) comme activation modulée par C
  - Met à jour ses poids par la loi LCT : ΔW = η · φ · P_sig · C
  - N'a PAS de loss function ni de gradient descendant

L'apprentissage se fait en MAXIMISANT P_sig (persistance topologique des
poids) — le neurone "apprend" en devenant topologiquement robuste.
"""
from __future__ import annotations

import math
import numpy as np


class LCTNeuron:
    """Neurone LCT : apprend par la loi ΔW = η · φ · P_sig · C.

    Args:
        n_inputs: nombre d'entrées (dimension du vecteur de poids).
        eta: taux d'apprentissage constitutif (adimensionné).
        omega: pulsation du milieu génial (par défaut π/2, validé sur QPU).
        seed: graine aléatoire.
    """

    def __init__(self, n_inputs: int, eta: float = 0.1, omega: float = math.pi / 2, seed: int = 42):
        self.n_inputs = n_inputs
        self.eta = eta
        self.omega = omega
        self.rng = np.random.default_rng(seed)
        # poids initiaux (Glorot-like)
        self.weights = self.rng.normal(0, 0.5, n_inputs)
        self.bias = 0.0
        # phase du milieu géniel (oscille dans le temps)
        self.phi = 1.0
        # cohérence du signal d'entrée (mise à jour à chaque forward)
        self.C = 1.0
        # P_sig injecté par la couche (persistance topologique des poids)
        self.P_sig = 1.0
        # historique pour monitoring
        self.history = []

    def forward(self, x: np.ndarray, t_step: int = 0) -> float:
        """Forward pass : activation modulée par la cohérence C.

        φ = |cos(ωt)| : amplitude de cohérence du milieu génial (toujours ≥ 0,
        jamais n'inverse l'apprentissage). La phase signée cos(ωt) a un sens
        physique dans le cerveau TTF (l'effondrement aux maxima), mais comme
        coefficient d'apprentissage elle désapprend la moitié du temps → on
        garde l'amplitude |cos(ωt)|.

        C = cohérence structurelle du signal d'entrée : la polarité dominante
        des composantes (un signal cohérent a une majorité de composantes du
        même signe). Borne [0.5, 1]. Contrairement à |mean|/std (≈ 0 pour un
        signal centré), ce proxy reste non nul → l'update ΔW a une amplitude
        réelle et les poids bougent effectivement.
        """
        # amplitude de cohérence du milieu génial (toujours ≥ 0)
        self.phi = abs(math.cos(self.omega * t_step))
        # cohérence structurelle = polarité dominante des composantes
        if len(x) > 1 and x.std() > 1e-9:
            majority_sign = np.sign(np.mean(x))
            if majority_sign != 0:
                # fraction des composantes du signe majoritaire, recentrée [0.5, 1]
                frac = float(np.mean(np.sign(x) == majority_sign))
                self.C = min(1.0, max(0.0, 0.5 + 0.5 * frac))
            else:
                self.C = 0.5
        else:
            self.C = 0.5
        # activation = tanh(w·x + b) modulée par C
        z = np.dot(self.weights, x) + self.bias
        # modulation par la cohérence : un signal incohérent (C bas) amortit
        activation = math.tanh(z) * self.C
        return activation

    def update(self, x: np.ndarray, target: float, P_sig: float, t_step: int = 0):
        """Mise à jour des poids par la loi LCT : ΔW = η · φ · P_sig · C.

        Pas de gradient descendant. L'erreur (target - output) signe la
        DIRECTION, mais l'AMPLITUDE de l'update est gouvernée par LCT.

        Args:
            x: vecteur d'entrée.
            target: valeur cible (supervision).
            P_sig: persistance topologique de la matrice de poids (injectée
                   par la couche LCT — voir lct_layer.py).
            t_step: pas de temps (pour la phase φ).
        """
        # forward
        output = self.forward(x, t_step)
        # erreur (signe la direction, comme dans toute supervision)
        error = target - output
        # mise à jour par LCT : ΔW = η · φ · P_sig · C · error · x
        # P_sig module l'amplitude (cycle long = concept robuste = update fort)
        # φ = |cos(ωt)| module l'amplitude (cohérence du milieu génial)
        # C module la confiance (signal cohérent = apprentissage autorisé)
        delta_w = self.eta * self.phi * P_sig * self.C * error * x
        self.weights += delta_w
        self.bias += self.eta * self.phi * P_sig * self.C * error
        self.P_sig = P_sig
        # historique
        self.history.append({
            "t": t_step,
            "phi": self.phi,
            "C": self.C,
            "P_sig": P_sig,
            "error": float(error),
            "weights_norm": float(np.linalg.norm(self.weights)),
        })

    def predict(self, x: np.ndarray) -> float:
        """Prédiction (sans mise à jour). Recalcule C pour la modulation."""
        self.forward(x, 0)
        z = np.dot(self.weights, x) + self.bias
        return math.tanh(z) * self.C
