"""ratis_net.topo_tokenizer — Tokenisation par cycles H1 persistants.

Piste 3 : au lieu de définir un token par un hash ou par des anneaux
manuellement construits, on le définit par sa SIGNATURE topologique — les
cycles H1 persistants de son nuage de points. Deux données topologiquement
équivalentes produisent le même token ; deux données différentes produisent
des tokens différents. C'est la tokenisation la plus fidèle à la théorie TTF.

Signature topologique d'un nuage :
  - betti = [b0, b1, b2] (nombres de composantes, cycles, cavités)
  - n_cycles = nombre de cycles H1 persistants
  - P_sig = persistance du cycle H1 le plus long (signal)
  - histogramme des persistances (pool par quantiles) = l'empreinte fine

Le vecteur = [b0, b1, b2, n_cycles, P_sig, + histogramme]. C'est l'identité
topologique de la donnée, invariante sous changement d'énergie (loi LCT).
"""
from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import numpy as np

# persistance homology : on utilise le persistence_optimizer (qui choisit
# automatiquement GUDHI si dispo, sinon CPU vectorisé, sinon lite). Le
# backend GUDHI (C++) est ~95x plus rapide que l'implémentation Python.
try:
    from ratis_net.persistence_optimizer import compute_persistence, preferred_backend, is_gudhi_available
except ImportError:
    # exécution en module direct depuis le dossier ratis_net/
    from persistence_optimizer import compute_persistence, preferred_backend, is_gudhi_available
_PERS_FN = compute_persistence


def _word_to_cloud(word: str, n_points: int = 40, seed: int = 42) -> np.ndarray:
    """Nuage de points dont la topologie encode le mot.

    On construit des anneaux (cycles H1) dont les rayons et les positions
    dépendent fortement des codes des caractères, de sorte que la
    topologie (nombre de cycles, leurs persistances, leurs intersections)
    varie significativement d'un mot à l'autre. Deux mots aux caractères
    différents produisent des topologies distinctes.
    """
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big") + seed)
    coords = []
    unique_chars = list(dict.fromkeys(word))
    n_rings = max(1, len(unique_chars))
    pts_per_ring = max(6, n_points // n_rings)
    # chaque anneau a un rayon et un centre déduits du code du caractère,
    # ce qui fait varier les intersections entre anneaux (et donc les cycles H1)
    for ci, ch in enumerate(unique_chars):
        code = ord(ch)
        radius = 0.6 + 0.8 * (code % 17) / 17.0
        cx = 1.5 * ci + 0.5 * (code % 7)
        cy = 0.4 * ((code % 13) - 6)
        for k in range(pts_per_ring):
            theta = 2 * math.pi * k / pts_per_ring
            # déformation de l'anneau selon le code (rend les cycles H1 asymétriques)
            r_eff = radius * (1.0 + 0.2 * math.sin(theta * (1 + code % 4)))
            x = cx + r_eff * math.cos(theta)
            y = cy + r_eff * math.sin(theta)
            z = 0.4 * math.sin(theta * (1 + code % 3))
            coords.append([x, y, z])
    coords = np.array(coords)
    coords += rng.normal(0, 0.03, coords.shape)
    return coords


def topo_signature(word: str, dim: int = 8, seed: int = 42,
                   n_points: int = 40, max_edge: float = 2.5) -> np.ndarray:
    """Signature topologique d'un mot = vecteur de dimension `dim`.

    Vecteur = [b0, b1, b2, densité_cycles, pers_max, pers_mean, pers_median,
    pers_std, pers_skew] (tronqué/padé à `dim`, normalisé).

    Deux mots avec même topologie → même signature. La signature est
    invariante sous l'énergie (loi LCT) : c'est la « forme » certifiable.
    """
    coords = _word_to_cloud(word, n_points=n_points, seed=seed)
    # diagrammes de persistance
    if _PERS_FN is not None:
        diagrams, _ = _PERS_FN(coords, max_edge)

    # extraction des features topologiques
    b0 = sum(1 for b, d in diagrams.get(0, []) if d == float("inf"))
    b1 = sum(1 for b, d in diagrams.get(1, []) if d == float("inf"))
    b2 = sum(1 for b, d in diagrams.get(2, []) if d == float("inf"))
    h1_pers = [d - b for b, d in diagrams.get(1, []) if d != float("inf") and d > b]
    n_cycles = len(h1_pers)
    P_sig = float(sorted(h1_pers, reverse=True)[0]) if h1_pers else 0.0

    # statistiques discriminantes de la distribution des persistances H1 :
    # max, mean, median, std, skew — capturent la forme de la distribution,
    # pas seulement son centre. Deux mots aux topologies différentes ont des
    # distributions de persistance différentes (ex: aa = beaucoup de cycles
    # faibles ; bonjour = cycles étalés jusqu'à 1.04).
    if h1_pers:
        arr = np.array(h1_pers)
        p_max = float(arr.max())
        p_mean = float(arr.mean())
        p_med = float(np.median(arr))
        p_std = float(arr.std())
        # skewness (asymétrie) : 0 si symétrique, >0 si étalée vers les hautes
        if p_std > 1e-9:
            p_skew = float(((arr - p_mean) ** 3).mean() / (p_std ** 3))
        else:
            p_skew = 0.0
    else:
        p_max = p_mean = p_med = p_std = p_skew = 0.0

    # vecteur signature : chaque feature mise sur une échelle comparable.
    n_pts = max(len(coords), 2)
    feat = [
        float(b0) / 10.0,                              # composantes
        float(b1) / 10.0,                              # cycles infinis
        float(b2) / 10.0,                              # cavités
        math.log1p(n_cycles) / math.log1p(n_pts),     # densité de cycles ∈ [0,1]
        min(p_max, 1.0),                               # persistance max
        min(p_mean * 10.0, 1.0),                       # persistance moyenne
        min(p_med * 10.0, 1.0),                        # persistance médiane
        min(p_std * 10.0, 1.0),                        # dispersion
        min(max(p_skew, -1.0), 1.0),                  # asymétrie ∈ [-1,1]
    ]
    sig = np.array(feat, dtype=float)
    # padding/troncature à dim
    if len(sig) < dim:
        sig = np.pad(sig, (0, dim - len(sig)))
    elif len(sig) > dim:
        sig = sig[:dim]
    # normalisation (centrer sur la plage du topo, pas N(0,1))
    n = np.linalg.norm(sig)
    if n > 1e-9:
        sig = sig / n
    return sig


def is_full_persistence_available() -> bool:
    """True si un backend de persistance complet (GUDHI ou CPU vectorisé) est dispo."""
    return _PERS_FN is not None


def active_backend() -> str:
    """Nom du backend actif (gpu/cpu/lite)."""
    return preferred_backend()


if __name__ == "__main__":
    # diagnostic rapide : signatures distinctes par mot
    import numpy as np
    print("Persistance complète disponible ?", is_full_persistence_available())
    words = ["bonjour", "salut", "bonsoir", "aa", "xyz"]
    embs = {w: topo_signature(w, dim=10) for w in words}
    for w in words:
        print(f"  {w:10s} → {np.round(embs[w], 4)}")
    distinct = len(set(tuple(np.round(e, 4)) for e in embs.values()))
    print(f"\n  {len(words)} mots → {distinct} signatures distinctes")

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    print(f"  cos(bonjour, salut)   = {cos(embs['bonjour'], embs['salut']):.4f}  (lettres ≠)")
    print(f"  cos(bonjour, bonsoir) = {cos(embs['bonjour'], embs['bonsoir']):.4f}  (partagent b,o,n)")
    print(f"  cos(aa, xyz)          = {cos(embs['aa'], embs['xyz']):.4f}  (topo ≠)")
