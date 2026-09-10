"""ratis_net.ttf_bridge — Pont entre le cerveau TTF-Compute et RATIS-Net.

Piste 2 : au lieu de nourrir le réseau avec un embedding arbitraire (hash →
N(0,1)), on utilise les MCB (Mémoire de Corrélation Bit) du cerveau TTF comme
représentation d'entrée. Le réseau « pense » alors avec la topologie réelle
de la donnée, pas avec un vecteur aléatoire.

Flux :
  mot → nuage de points (forme topologique du mot)
      → cerveau TTF (oscillation + Rips + MCB)
      → triplets (src, dst, φ) = la pensée sans mots
      → embedding topologique (flatten + pool) → entrée de RATIS-Net

Le cerveau TTF-Compute vit dans le dépôt RATISS-ODV-AEON. Ce bridge le
référence par chemin (le dépôt doit être cloné à côté). Si absent, fallback
sur l'embedding hash (piste 1) — le réseau reste fonctionnel.
"""
from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import numpy as np

# tentative de chargement du cerveau TTF-Compute (dépôt voisin)
_AEON_PATH = Path(__file__).resolve().parents[2] / "RATISS-ODV-AEON"
_TTF_AVAILABLE = False
try:
    if _AEON_PATH.is_dir():
        # AEON contient un dossier ratis_net/ (vieille copie v1) qui cacherait
        # le nôtre si mis en tête : on l'ajoute en FIN de path pour ne résoudre
        # que kernel.* (qui n'existe pas dans notre repo).
        sys.path.append(str(_AEON_PATH))
        from kernel.ttf.ttf_compute import TTFBrain  # noqa: F401
        _TTF_AVAILABLE = True
except Exception:
    _TTF_AVAILABLE = False


def _word_to_coords(word: str, n_points: int = 40, seed: int = 42) -> np.ndarray:
    """Transforme un mot en un nuage de points 3D dont la topologie encode le mot.

    On construit une structure en anneaux (cycles H1) : chaque caractère
    distinct du mot engendre un anneau de points. Le NOMBRE d'anneaux et leurs
    rayons dépendent du mot → la topologie (betti_1) encode le mot. Deux mots
    avec des caractères différents ont des topologies différentes. C'est
    l'analogue d'une protéine : la séquence définit la forme, la forme définit
    la topologie.
    """
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big") + seed)
    coords = []
    # un anneau par caractère unique du mot ; le rayon dépend du code du caractère
    unique_chars = list(dict.fromkeys(word))  # ordre préservé, doublons retirés
    n_rings = max(1, len(unique_chars))
    pts_per_ring = max(4, n_points // n_rings)
    for ci, ch in enumerate(unique_chars):
        code = ord(ch)
        radius = 0.8 + 0.4 * (code % 11) / 11.0
        # décalage du centre de l'anneau selon la position du caractère
        cx = 2.0 * ci
        for k in range(pts_per_ring):
            theta = 2 * math.pi * k / pts_per_ring
            x = cx + radius * math.cos(theta)
            y = radius * math.sin(theta)
            z = 0.3 * math.sin(theta * (1 + code % 3))
            coords.append([x, y, z])
    coords = np.array(coords)
    # bruit déterministe propre au mot (signature fine)
    coords += rng.normal(0, 0.05, coords.shape)
    return coords


def _mcb_to_embedding(mcb_buffer, dim: int = 8, seed: int = 42) -> np.ndarray:
    """Flatten les triplets MCB (src, dst, φ) en un embedding de dimension `dim`.

    Les MCB sont la « pensée sans mots » du cerveau : des bits de corrélation
    topologique. On les agrège en un vecteur fixe par pooling (somme pondérée
    par φ, puis projection déterministe en dim). Deux données topologiquement
    proches produisent des MCB proches → des embeddings proches.
    """
    if not mcb_buffer:
        rng = np.random.default_rng(seed)
        return rng.normal(0, 1, dim)
    # pooling : on accumule la corrélation par paire (src, dst)
    # projection déterministe : hash de (src, dst) → dimension
    vec = np.zeros(dim)
    total = 0.0
    for triplet in mcb_buffer:
        src, dst, phi = triplet.src, triplet.dst, triplet.correlation_bit
        h = int.from_bytes(hashlib.sha256(f"{src}-{dst}".encode()).digest()[:4], "big")
        vec[h % dim] += abs(phi)
        total += abs(phi)
    if total > 1e-9:
        vec /= total  # normaliser
    n = np.linalg.norm(vec)
    if n > 1e-9:
        vec = vec / n
    # si la MCB est trop pauvre (peu de dims activées), on complète avec un
    # hash du contenu pour garder l'information de structure
    if np.count_nonzero(vec) < dim // 2:
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(repr([(t.src, t.dst, round(t.correlation_bit, 3)) for t in mcb_buffer]).encode()).digest()[:4], "big") + seed)
        vec = 0.5 * vec + 0.5 * (rng.normal(0, 1, dim) / (np.linalg.norm(rng.normal(0, 1, dim)) + 1e-9))
    return vec


def ttf_embedding(word: str, dim: int = 8, seed: int = 42,
                  n_points: int = 40, n_steps: int = 12) -> np.ndarray:
    """Embedding topologique d'un mot via le cerveau TTF-Compute.

    Si le cerveau TTF n'est pas disponible (dépôt AEON absent), fallback sur
    l'embedding hash (piste 1) — le réseau reste fonctionnel.
    """
    if not _TTF_AVAILABLE:
        return _hash_embedding(word, dim, seed)
    coords = _word_to_coords(word, n_points=n_points, seed=seed)
    brain = TTFBrain(coords=coords, omega=math.pi / 2, max_edge=2.5, Dc=0.3, seed=seed)
    # force la production de MCB en poussant la décohérence au cours du temps
    for k in range(n_steps):
        # décohérence croissante : force le cerveau à transmettre + traduire
        deco = 0.1 + 0.15 * k
        brain.step(t_sec=k * 0.5, force_decoherence=deco)
    # Les MCB sont collectées par le puits d'effondrement (well.collected),
    # qui les accumule au fil des effondrements. C'est la « pensée sans mots »
    # accumulée — on l'utilise comme embedding topologique du mot.
    collected = brain.well.collected
    return _mcb_to_embedding(collected, dim=dim, seed=seed)


def _hash_embedding(word: str, dim: int = 8, seed: int = 42) -> np.ndarray:
    """Fallback : embedding hash orthogonal (piste 1)."""
    h = hashlib.sha256(word.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:4], "big") + seed)
    return rng.normal(0, 1, dim)


def is_ttf_available() -> bool:
    """True si le cerveau TTF-Compute est connecté (dépôt AEON présent)."""
    return _TTF_AVAILABLE


if __name__ == "__main__":
    # diagnostic rapide
    import numpy as np
    print("TTF available:", _TTF_AVAILABLE)
    print("AEON path:", _AEON_PATH, "exists:", _AEON_PATH.is_dir())
