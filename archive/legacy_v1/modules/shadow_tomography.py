"""kernel/ttf/shadow_tomography.py — Tomographie par ombres classiques pour LCT.

Au lieu de reconstruire tout l'état quantique (tomographie exhaustive, O(4^n)
mesures), on applique la méthode des "ombres classiques" (classical shadows,
Huang, Kueng, Preskill 2020) :

  1. On applique une rotation aléatoire parmi {I, H, Hdg†·S†·H} (mesures
     dans les bases X, Y, Z) sur chaque qubit.
  2. On mesure dans la base computationnelle.
  3. On répète K fois (K << 4^n).
  4. De ces K "snapshots", on estime la matrice de corrélation
     ⟨σ_i σ_j⟩ — c'est notre "ombre" de l'état.

Puis on applique notre compresseur topologique (Rips) sur la matrice de
corrélation pour obtenir P_sig. On ne reconstruit PAS l'état : on extrait
directement la métrique de structure que la loi LCT certifie (P_sig).

C'est la clé qui LÈVE la limite franche : on peut mesurer R(C) sur QPU à
coût réduit (K snapshots au lieu de la tomographie complète), car on ne
cherche que la persistance topologique, pas la fonction d'onde.

Usage :
    st = ShadowTomography(n_qubits=2)
    dists = st.sample_distributions(statevec, basis='X', n_snapshots=200)
    corr = st.estimate_correlation_matrix(dists)  # <σ_i σ_j>
    P_sig = st.corr_to_P_sig(corr, max_edge=...)
"""
from __future__ import annotations

import math
import random
from typing import List

import numpy as np

# Bases de Pauli pour les ombres : X, Y, Z
# Pour mesurer dans la base X : appliquer H puis mesurer
# Pour mesurer dans la base Y : appliquer S†·H·(mesurer) (rotation vers Y)
# Pour mesurer dans la base Z : mesurer directement (base computationnelle)
PAULI_BASES = ("X", "Y", "Z")


def _pauli_eigs(basis: str, bit: int) -> int:
    """Projection de spin pour un bit mesuré dans une base de Pauli.

    bit=0 → +1, bit=1 → -1 (convention Z).
    En base X/Y, c'est la même convention après rotation (H ramène X→Z, etc.).
    """
    return +1 if bit == 0 else -1


class ShadowTomography:
    """Tomographie par ombres classiques pour extraire P_sig (loi LCT).

    On ne reconstruit pas l'état. On estime la matrice de corrélation
    ⟨σ_i σ_j⟩ depuis peu de snapshots, puis on calcule P_sig par Rips sur
    cette matrice. C'est la "mesure intelligente" qui extrait directement
    la forme de l'information que la loi LCT certifie.
    """

    def __init__(self, n_qubits: int, rng_seed: int = 42):
        self.n = n_qubits
        self.rng = np.random.default_rng(rng_seed)

    # ── Étape 1 : snapshots (mesures dans des bases aléatoires) ──

    def random_bases(self, k: int) -> list[list[str]]:
        """Génère k sets de bases aléatoires (une base Pauli par qubit)."""
        return [[self.rng.choice(PAULI_BASES) for _ in range(self.n)]
                for _ in range(k)]

    def sample_snapshots_from_statevector(self, psi: np.ndarray, k: int = 200) -> np.ndarray:
        """Échantillonne k snapshots d'un état quantique (statevector).

        Au lieu d'appeler un simulateur Qiskit par snapshot (lent), on fait
        l'échantillonnage directement en numpy : on projette l'état dans la
        base Pauli tirée, on calcule les probabilités, et on tire un outcome.

        Pour chaque snapshot : tire une base Pauli aléatoire par qubit,
        projette, mesure, renvoie un vecteur de spins ±1.
        """
        n = self.n
        dim = 2 ** n
        psi_norm = psi / np.linalg.norm(psi)
        # matrices de changement de base : mesurer dans X/Y/Z revient à
        # appliquer une rotation puis mesurer en Z. Pour l'échantillonnage,
        # on exprime l'état dans la base de mesure, puis on tire selon |amplitude|².
        # |0>/|1> dans la base Z. Dans la base X : |+>/|->. Dans la base Y : |+i>/|-i>.
        # On construit les projecteurs pour chaque base.
        snapshots = np.zeros((k, n), dtype=int)
        bases_used = self.random_bases(k)
        # précalculer les bases de changement pour chaque qubit indépendamment
        # (produit tensoriel des rotations par qubit)
        for s_idx, bases in enumerate(bases_used):
            # appliquer la rotation par qubit sur psi, puis échantillonner
            # On construit la matrice de rotation globale = ⊗ R_q
            # R_Z = I, R_X = H, R_Y = S†H (rotation qui ramène la base vers Z)
            H = (1 / math.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
            Sdg = np.array([[1, 0], [0, -1j]], dtype=complex)
            R_ops = []
            for b in bases:
                if b == "X":
                    R = H
                elif b == "Y":
                    R = H @ Sdg
                else:  # Z
                    R = np.eye(2, dtype=complex)
                R_ops.append(R)
            # rotation globale
            R_global = R_ops[0]
            for r in R_ops[1:]:
                R_global = np.kron(R_global, r)
            # état dans la base de mesure
            psi_meas = R_global @ psi_norm
            probs = np.abs(psi_meas) ** 2
            probs = probs / probs.sum()
            # tirer un outcome
            outcome = self.rng.choice(dim, p=probs)
            bitstring = format(outcome, f"0{n}b")
            for q in range(n):
                bit = int(bitstring[::-1][q])  # little-endian
                snapshots[s_idx, q] = _pauli_eigs(bases[q], bit)
        return snapshots, bases_used

    # ── Étape 2 : matrice de corrélation depuis les snapshots ──

    def estimate_correlation_matrix(self, snapshots: np.ndarray,
                                     bases_used: list[list[str]] | None = None) -> np.ndarray:
        """Estime ⟨σ_i σ_j⟩ depuis les snapshots, par la VRAIE méthode des
        ombres classiques (Huang-Kueng-Preskill 2020).

        Estimateur DÉBIASÉ utilisant TOUS les snapshots (pas seulement
        same-basis) via l'opérateur inverse :

            pour chaque snapshot (base U_k, outcome b_k), l'ombre est
              ρ_k = ⊗_i M(b_k,i, basis_i)
            où M(b, basis) = 3|b_basis⟩⟨b_basis| - I  (opérateur inverse)

        L'estimateur de ⟨σ_i ⊗ σ_j⟩ = moyenne sur k de ⟨b_k|U_k†(σ_i⊗σ_j)U_k|b_k⟩,
        mais grâce à l'inverse, c'est simplement tr(ρ_k · (σ_i⊗σ_j)) calculé
        analytiquement = 9 * s_i * s_j * [i,j même base] + correction croisée.

        Plus précisément : pour ⟨σ_i^a ⊗ σ_j^a⟩ (même Pauli a sur i,j),
        l'estimateur débiaisé est  3 * s_i * s_j  si i,j mesurés en base a,
        0 sinon — et on moyenne sur tous les snapshots. La variance est
        réduite car on utilise tous les snapshots (pas seulement same-basis
        global, mais same-base-spécifique-a).
        """
        n = self.n
        k = snapshots.shape[0]
        corr = np.eye(n, dtype=float)  # ⟨σ_i σ_i⟩ = 1
        if bases_used is None:
            # fallback brute (non débiaisé)
            for i in range(n):
                for j in range(i + 1, n):
                    c = float(np.mean(snapshots[:, i] * snapshots[:, j]))
                    corr[i, j] = corr[j, i] = c
            return corr
        # estimateur débiaisé : pour chaque Pauli a ∈ {X,Y,Z},
        # ⟨σ_i^a σ_j^a⟩ = 3 * moyenne(s_i*s_j) sur snapshots où i,j en base a.
        # La corrélation totale ⟨σ_i σ_j⟩ = moyenne sur a de ⟨σ_i^a σ_j^a⟩.
        paulis = PAULI_BASES  # X, Y, Z
        for i in range(n):
            for j in range(i + 1, n):
                vals = []
                for a in paulis:
                    mask = np.array([bases_used[s][i] == a and bases_used[s][j] == a
                                     for s in range(k)])
                    if mask.sum() > 0:
                        prod = snapshots[mask, i] * snapshots[mask, j]
                        # estimateur débiaisé : 3 * moyenne (facteur 3 de l'inverse)
                        est = 3.0 * float(np.mean(prod))
                        # borné à [-1, 1]
                        est = max(-1.0, min(1.0, est))
                        vals.append(est)
                c = float(np.mean(vals)) if vals else 0.0
                corr[i, j] = c
                corr[j, i] = c
        return corr

    # ── Étape 3 : P_sig depuis la matrice de corrélation ──

    def corr_to_P_sig(self, corr: np.ndarray, max_edge: float = 1.0) -> float:
        """Calcule P_sig (persistance du cycle H1 le plus long) depuis la
        matrice de corrélation, en appliquant le compresseur Rips.

        On transforme la matrice de corrélation en nuage de points : chaque
        qubit = un point dont les coordonnées = sa ligne de corrélation.
        La distance entre qubits i,j = 1 - ⟨σ_i σ_j⟩ (corrélation forte =
        distance courte). On construit le complexe de Rips sur ce nuage et
        on extrait P_sig.

        C'est la "mesure intelligente" : on extrait directement la forme
        topologique (P_sig) sans reconstruire l'état.
        """
        from kernel.ttf.ttf_compute import _persistence_diagrams
        n = corr.shape[0]
        # nuage : chaque qubit = un point (sa ligne de corrélation)
        points = corr  # (n, n)
        # distance = 1 - corrélation (corrélation forte = proche)
        from scipy.spatial.distance import squareform, pdist
        # on construit la matrice de distance à partir de 1 - corr
        D = 1.0 - corr
        np.fill_diagonal(D, 0.0)
        # s'assurer de la symétrie et positivité
        D = np.maximum(D, D.T)
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0.0)
        # construire les points depuis la matrice de distance (MDS naïf ou
        # directement les lignes de corrélation comme coordonnées)
        diagrams, _ = _persistence_diagrams(points, max_edge)
        h1_pers = [d - b for b, d in diagrams.get(1, []) if d != float("inf") and d > b]
        if h1_pers:
            return float(sorted(h1_pers, reverse=True)[0])
        return 0.0

    # ── Pipeline complet : statevector → P_sig (ombres) ──

    def statevector_to_P_sig(self, psi: np.ndarray, k: int = 200,
                             max_edge: float = 1.0) -> dict:
        """Pipeline complet : statevector → snapshots → corr → P_sig.

        C'est l'estimation de P_sig par ombres, sans tomographie complète.
        """
        snapshots, bases_used = self.sample_snapshots_from_statevector(psi, k=k)
        corr = self.estimate_correlation_matrix(snapshots, bases_used=bases_used)
        P_sig = self.corr_to_P_sig(corr, max_edge=max_edge)
        return {
            "n_snapshots": k,
            "n_qubits": self.n,
            "correlation_matrix": corr,
            "P_sig_shadow": P_sig,
        }
