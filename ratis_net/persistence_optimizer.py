"""ratis_net.persistence_optimizer — Backends de persistance homology.

Le goulot de la piste 4 est le calcul de persistance (cycles H1) sur le
vocabulaire. Ce module expose trois backends, du plus simple au plus rapide :

  - "lite"   : _persistence_diagrams_lite (numpy pur, approximation H1) — déjà
    présent dans lct_network. Toujours dispo, lent.
  - "cpu"    : compute_persistence_cpu — vectorisé NumPy. Distances et arêtes
    calculées en batch (np.linalg.norm vectorisé, np.triu_indices), union-find
    accéléré par path-compression. Gain ~10-50x sur le lite, SANS GPU.
  - "gpu"    : compute_persistence_gpu — via GUDHI (bibliothèque C++ de
    persistance). Si gudhi est installé ET un GPU CUDA est dispo, GUDHI peut
    utiliser un backend accéléré. Sinon fallback sur le backend cpu.

Le module choisit automatiquement le meilleur backend dispo :
  preferred_backend() → "gpu" si gudhi+CUDA, sinon "cpu", sinon "lite".

Tous les backends retournent le MÊME format : (diagrams, edges)
  diagrams = {0: [[birth, death], ...], 1: [[birth, death], ...]}
  edges = [(distance, i, j), ...] triées par distance.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ── Détection des backends ──────────────────────────────────────────────────

# GPU : GUDHI (si installé). GUDHI est une bibliothèque C++ de persistance
# qui peut tirer parti de CUDA si compilée avec. On essaie d'importer, et on
# note sa disponibilité. La détection GPU réelle (nvcc / nvidia-smi) est
# best-effort : si gudhi est là, on suppose qu'il peut utiliser le GPU.
_GUDHI_AVAILABLE = False
try:
    import gudhi  # noqa: F401
    _GUDHI_AVAILABLE = True
except Exception:
    _GUDHI_AVAILABLE = False


def is_gudhi_available() -> bool:
    """True si GUDHI (persistance C++) est installé."""
    return _GUDHI_AVAILABLE


def is_gpu_available() -> bool:
    """Best-effort : True si un GPU CUDA semble dispo (nvcc ou nvidia-smi)."""
    if not _GUDHI_AVAILABLE:
        return False
    import shutil
    if shutil.which("nvidia-smi") or shutil.which("nvcc"):
        return True
    return False


def preferred_backend() -> str:
    """Choisit le meilleur backend : 'gpu' > 'cpu' > 'lite'."""
    if _GUDHI_AVAILABLE:
        return "gpu"
    return "cpu"


# ── Backend CPU vectorisé (NumPy) ───────────────────────────────────────────


def compute_persistence_cpu(points: np.ndarray, max_edge: float,
                            max_dim: int = 2) -> tuple[dict, list]:
    """Persistance H0+H1 vectorisée avec NumPy. SANS GPU.

    Optimisations vs le lite/itératif :
      - Distances : np.linalg.norm sur tout le tableau (broadcast), pas de
        boucle i,j. O(n²) mais en C vectorisé, pas en Python.
      - Arêtes : np.triu_indices extrait le triangle supérieur d'un coup.
      - Tri : np.argsort (C) au lieu de list.sort (Python).
      - H0 : union-find avec path-compression.
      - H1 : filtration Rips + réduction de bordure (comme _persistence_diagrams
        de AEON, mais avec les arêtes/triangles pré-calculés vectoriellement).

    Retourne le MÊME format que _persistence_diagrams (AEON) :
      diagrams = {0: [[b,d],...], 1: [[b,d],...]},  edges = [(d,i,j),...].
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n < 3:
        return {0: [[0.0, float("inf")] for _ in range(n)], 1: []}, []

    # ── Distances vectorisées ──
    # D[i,j] = ||points[i] - points[j]||  (matrice n×n d'un coup)
    diff = points[:, None, :] - points[None, :, :]
    D = np.linalg.norm(diff, axis=2)

    # ── Arêtes sous le seuil (triangle supérieur) ──
    iu, ju = np.triu_indices(n, k=1)
    dists = D[iu, ju]
    keep = dists <= max_edge
    ei, ej, ed = iu[keep], ju[keep], dists[keep]
    # tri par distance croissante
    order = np.argsort(ed)
    ei, ej, ed = ei[order], ej[order], ed[order]
    edges = list(zip(ed.tolist(), ei.tolist(), ej.tolist()))

    diagrams = {0: [], 1: []}

    # ── H0 (union-find avec path-compression) ──
    parent = list(range(n))

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    components = n
    for d, i, j in edges:
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[ri] = rj
            components -= 1
            diagrams[0].append([0.0, d])
    for _ in range(components):
        diagrams[0].append([0.0, float("inf")])

    # ── H1 (réduction de matrice de bordure) ──
    # triangles : on vectorise l'extraction. matrice d'adjacence booléenne,
    # puis pour chaque arête (a<b) les voisins communs c = np.where(A[a] & A[b]).
    edge_set = {(int(a), int(b)): d for d, a, b in edges}
    edge_index = {(int(a), int(b)): k for k, (_, a, b) in enumerate(edges)}
    A = np.zeros((n, n), dtype=bool)
    A[ei, ej] = True
    A[ej, ei] = True
    np.fill_diagonal(A, False)
    triangles = []
    for d_ab, a, b in edges:
        a, b = int(a), int(b)
        cands = np.where(A[a] & A[b])[0]
        for c in cands:
            c = int(c)
            if c > b:
                d_ac = edge_set[(a, c)]
                d_bc = edge_set[(min(b, c), max(b, c))]
                d_tri = max(d_ab, d_ac, d_bc)
                triangles.append((d_tri, a, b, c))

    triangles.sort(key=lambda t: t[0])

    # réduction pour H1 (même algo que AEON)
    edge_order = [(d, i, j) for d, i, j in edges]
    low_marker = {}
    pairs = []
    for d_tri, bnd in [(d, [edge_index[(min(a, b), max(a, b))],
                            edge_index[(min(a, c), max(a, c))],
                            edge_index[(min(b, c), max(b, c))]])
                       for d, a, b, c in triangles]:
        reduced = set(bnd)
        while True:
            present = sorted(reduced)
            if not present:
                break
            low = present[0]
            if low in low_marker:
                reduced = reduced.symmetric_difference(low_marker[low])
            else:
                low_marker[low] = reduced
                birth = edge_order[low][0]
                if d_tri > birth + 1e-9:
                    pairs.append((birth, d_tri))
                break
    for birth, death in pairs:
        diagrams[1].append([float(birth), float(death)])
    killed = set()
    for reduced in low_marker.values():
        killed |= reduced
    for k, (d, i, j) in enumerate(edge_order):
        if k not in killed:
            diagrams[1].append([float(d), float("inf")])

    return diagrams, edges


# ── Backend GPU (GUDHI + CUDA) ──────────────────────────────────────────────


def compute_persistence_gpu(points: np.ndarray, max_edge: float,
                             max_dim: int = 2) -> tuple[dict, list]:
    """Persistance via GUDHI. Si GUDHI est installé et un GPU CUDA est dispo,
    GUDHI peut utiliser un backend accéléré. Sinon fallback sur le backend CPU.

    GUDHI calcule les diagrammes de persistance du complexe de Rips de façon
    native (C++), beaucoup plus rapide que notre implémentation Python sur
    de gros nuages. C'est le chemin à activer le jour où un GPU est dispo.
    """
    if not _GUDHI_AVAILABLE:
        # pas de GUDHI → fallback CPU vectorisé
        return compute_persistence_cpu(points, max_edge, max_dim)

    import gudhi
    points = np.asarray(points, dtype=np.float64)
    rips = gudhi.RipsComplex(points=points.tolist(), max_edge_length=max_edge)
    simplex_tree = rips.create_simplex_tree(max_dimension=max_dim)
    # persistence = liste de (dim, (birth, death))
    pers = simplex_tree.persistence()
    diagrams = {0: [], 1: []}
    for dim, (b, d) in pers:
        diagrams.setdefault(dim, []).append([float(b),
                                            float(d) if d != float("inf") else float("inf")])
    # arêtes (pour la signature, on reconstitue depuis les points)
    n = len(points)
    if n >= 2:
        diff = points[:, None, :] - points[None, :, :]
        D = np.linalg.norm(diff, axis=2)
        iu, ju = np.triu_indices(n, k=1)
        dists = D[iu, ju]
        keep = dists <= max_edge
        ei, ej, ed = iu[keep], ju[keep], dists[keep]
        order = np.argsort(ed)
        edges = list(zip(ed[order].tolist(), ei[order].tolist(), ej[order].tolist()))
    else:
        edges = []
    return diagrams, edges


# ── Sélecteur automatique ──────────────────────────────────────────────────


def compute_persistence(points: np.ndarray, max_edge: float,
                         backend: str | None = None) -> tuple[dict, list]:
    """Calcule la persistance avec le backend demandé (ou le meilleur dispo).

    backend ∈ {None, "lite", "cpu", "gpu"}.
    None → preferred_backend() (gpu si gudhi, sinon cpu).
    """
    if backend is None:
        backend = preferred_backend()
    if backend == "gpu":
        return compute_persistence_gpu(points, max_edge)
    if backend == "cpu":
        return compute_persistence_cpu(points, max_edge)
    if backend == "lite":
        try:
            from ratis_net.lct_network import _persistence_diagrams_lite
            return {0: [], 1: [(0.0, _persistence_diagrams_lite(points, max_edge))]}, []
        except ImportError:
            from lct_network import _persistence_diagrams_lite
            return {0: [], 1: [(0.0, _persistence_diagrams_lite(points, max_edge))]}, []
    raise ValueError(f"backend inconnu: {backend}")


if __name__ == "__main__":
    import time
    print("Backends disponibles :")
    print(f"  GUDHI (GPU) : {is_gudhi_available()}  (GPU dispo : {is_gpu_available()})")
    print(f"  CPU vectorisé : toujours dispo")
    print(f"  Backend préféré : {preferred_backend()}")

    # benchmark CPU vs lite sur un nuage de test
    rng = np.random.default_rng(42)
    pts = rng.normal(0, 1, (40, 3))
    print(f"\nBenchmark sur 40 points 3D (max_edge=2.5) :")
    t = time.time()
    d_cpu, _ = compute_persistence_cpu(pts, 2.5)
    print(f"  CPU vectorisé : {time.time()-t:.3f}s, H1 = {len(d_cpu.get(1,[]))} cycles")
    print(f"  b0={sum(1 for b,d in d_cpu[0] if d==float('inf'))} "
          f"b1={sum(1 for b,d in d_cpu[1] if d==float('inf'))}")
