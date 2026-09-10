"""ratis_net.eth_thermo_fixer — Le fixeur thermodynamique ETH.

Le saut conceptuel de la v4 : on ne maximise pas P_sig (non-différentiable).
On laisse C s'effondrer sous poussée thermodynamique de l'environnement, et
on garde la MARQUE topologique (hash du cycle survivant), pas la valeur.

ETH = fixeur thermodynamique. Pour chaque token + environnement, il apprend
C_seuil_thermo = f(environnement). Pas un seuil global 0.8, mais un seuil
CONTEXTUEL. "Bonjour colère" = C_seuil bas (effondrement rapide, marque
agressive). "Bonjour joie" = C_seuil haut (effondrement lent, marque ouverte).

L'entraînement : on montre des exemples (token + environnement → C_seuil).
ETH apprend le DIFFERENTIEL thermo, pas la valeur fixe. C'est l'émotion qui
émerge : la différence de C_seuil entre "bonjour colère" et "bonjour joie".
"""
from __future__ import annotations

import math
import numpy as np


class ThermoEnvironment:
    """Environnement thermodynamique d'un token (mesures patient simulées).

    Dans le cerveau humain, l'environnement = rythme cardiaque, tension,
    chaleur de la peau. Ici on simule ces features. L'émotion émerge comme
    différentiel de C_seuil entre environnements.
    """

    def __init__(self, heart_rate: float = 70.0, tension: float = 0.3,
                 warmth: float = 0.5, arousal: float = 0.2):
        self.heart_rate = heart_rate    # bpm (60 = calme, 120 = colère/stress)
        self.tension = tension          # 0 = relaxé, 1 = crispé
        self.warmth = warmth            # 0 = froid, 1 = chaud
        self.arousal = arousal          # 0 = endormi, 1 = excité

    def to_vector(self) -> np.ndarray:
        """Vecteur de features de l'environnement."""
        return np.array([
            self.heart_rate / 120.0,   # normalisé
            self.tension,
            self.warmth,
            self.arousal,
        ])

    @staticmethod
    def anger() -> "ThermoEnvironment":
        """Environnement colère : cœur rapide, tendu, chaud, excité."""
        return ThermoEnvironment(heart_rate=110, tension=0.9, warmth=0.8, arousal=0.9)

    @staticmethod
    def joy() -> "ThermoEnvironment":
        """Environnement joie : cœur modéré, détendu, chaud, excité (positif)."""
        return ThermoEnvironment(heart_rate=85, tension=0.1, warmth=0.7, arousal=0.6)

    @staticmethod
    def calm() -> "ThermoEnvironment":
        """Environnement calme : cœur lent, relaxé, neutre."""
        return ThermoEnvironment(heart_rate=65, tension=0.05, warmth=0.5, arousal=0.1)

    @staticmethod
    def fear() -> "ThermoEnvironment":
        """Environnement peur : cœur rapide, tendu, froid, excité."""
        return ThermoEnvironment(heart_rate=115, tension=0.8, warmth=0.2, arousal=0.85)


class ETHThermoFixer:
    """Fixeur thermodynamique : apprend C_seuil = f(token, environnement).

    Pour chaque (token, environnement), ETH prédit le seuil de cohérence
    auquel l'effondrement topologique doit se produire. C_seuil est
    CONTEXTUEL : "bonjour" en colère a un C_seuil différent de "bonjour"
    en joie.

    L'émotion émerge comme le DIFFERENTIEL de C_seuil entre environnements
    pour le même token.
    """

    def __init__(self, token_dim: int = 8, env_dim: int = 4, hidden: int = 16, seed: int = 42):
        self.token_dim = token_dim
        self.env_dim = env_dim
        self.hidden = hidden
        self.rng = np.random.default_rng(seed)
        # petit MLP : (token + env) -> C_seuil ∈ [0, 1]
        input_dim = token_dim + env_dim
        self.W1 = self.rng.normal(0, 0.3, (input_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = self.rng.normal(0, 0.3, (hidden, 1))
        self.b2 = np.zeros(1)
        # historique
        self.history = []

    def predict_c_seuil(self, token_embedding: np.ndarray, env: ThermoEnvironment) -> float:
        """Prédit C_seuil_thermo pour (token, environnement). ∈ [0, 1]."""
        x = np.concatenate([token_embedding, env.to_vector()])
        h = np.tanh(x @ self.W1 + self.b1)
        out = 1.0 / (1.0 + math.exp(-float((h @ self.W2 + self.b2)[0])))  # sigmoid
        return out

    def train_step(self, token_embedding: np.ndarray, env: ThermoEnvironment,
                   target_c_seuil: float, lr: float = 0.1):
        """Entraîne ETH sur un exemple : (token, env) → C_seuil cible.

        L'entraînement se fait par gradient (ici ETH EST différentiable :
        C_seuil est une fonction lisse des features, contrairement à P_sig).
        """
        x = np.concatenate([token_embedding, env.to_vector()])
        # forward
        h = np.tanh(x @ self.W1 + self.b1)
        out = 1.0 / (1.0 + math.exp(-float((h @ self.W2 + self.b2)[0])))
        # erreur
        error = target_c_seuil - out
        # gradient (ETH est lisse → gradient stable)
        d_out = error * out * (1 - out)  # sigmoid derivative
        dW2 = h * d_out
        db2 = d_out
        dh = d_out * self.W2[:, 0]
        dh_pre = dh * (1 - h ** 2)  # tanh derivative
        dW1 = np.outer(x, dh_pre)
        db1 = dh_pre
        # update
        self.W2 += lr * dW2.reshape(-1, 1)
        self.b2 += lr * db2
        self.W1 += lr * dW1
        self.b1 += lr * db1
        # historique
        self.history.append({"c_seuil_pred": out, "target": target_c_seuil, "error": float(error)})
        return float(error)

    def emotional_differential(self, token_embedding: np.ndarray,
                                env1: ThermoEnvironment, env2: ThermoEnvironment) -> float:
        """Le différentiel émotionnel = ΔC_seuil entre deux environnements.

        C'est l'émotion qui émerge : la différence de seuil thermo pour le
        même token dans deux contextes (ex: colère vs joie).
        """
        c1 = self.predict_c_seuil(token_embedding, env1)
        c2 = self.predict_c_seuil(token_embedding, env2)
        return float(c1 - c2)
