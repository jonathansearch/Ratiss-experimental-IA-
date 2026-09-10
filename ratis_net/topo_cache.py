"""Cache des signatures topologiques (P_sig) — calcul unique par mot, lookup O(1).

Le Snapshot Topologique (GUDHI 1×/dialogue) a déjà réduit le coût ~36×.
Le cache élimine totalement le recalcul : les signatures sont déterministes
(même seed, mêmes paramètres), donc un mot se calcule UNE SEULE FOIS puis
se sert du disque. Nouveau module : n'altère ni topo_tokenizer, ni le loader.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .topo_tokenizer import topo_signature, active_backend

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "cache" / "topo_signatures"


class TopoCache:
    """dict mot → signature, avec persistance disque (npz + meta json)."""

    def __init__(self, dim: int = 8, n_points: int = 40, max_edge: float = 2.5,
                 path: str | Path | None = None):
        self.dim = dim
        self.n_points = n_points
        self.max_edge = max_edge
        self.path = Path(path) if path else _DEFAULT_PATH
        self._mem: dict[str, np.ndarray] = {}
        self.meta = {"dim": dim, "n_points": n_points, "max_edge": max_edge,
                     "backend": active_backend(), "created": None, "entries": 0}

    def get(self, word: str) -> np.ndarray:
        """Retourne la signature, en la calculant si absente."""
        if word not in self._mem:
            self._mem[word] = topo_signature(word, dim=self.dim,
                                             n_points=self.n_points,
                                             max_edge=self.max_edge)
        return self._mem[word]

    def warmup(self, words: list[str], progress_every: int = 1000) -> int:
        """Pré-calcule le vocabulaire. Retourne le nombre de nouvelles entrées."""
        new = 0
        for i, w in enumerate(words):
            if w not in self._mem:
                self.get(w)
                new += 1
            if progress_every and (i + 1) % progress_every == 0:
                print(f"  {i + 1}/{len(words)} mots ({new} nouveaux)")
        return new

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.meta["created"] = time.time()
        self.meta["entries"] = len(self._mem)
        np.savez_compressed(
            self.path.with_suffix(".npz"),
            words=list(self._mem.keys()),
            signatures=np.array(list(self._mem.values())))
        with open(self.path.with_suffix(".json"), "w") as f:
            json.dump(self.meta, f, indent=2)
        return self.path.with_suffix(".npz")

    def load(self) -> "TopoCache":
        with open(self.path.with_suffix(".json")) as f:
            self.meta = json.load(f)
        data = np.load(self.path.with_suffix(".npz"), allow_pickle=True)
        for w, s in zip(data["words"], data["signatures"]):
            self._mem[str(w)] = np.asarray(s, dtype=float)
        return self

    def __len__(self) -> int:
        return len(self._mem)

    def __contains__(self, word: str) -> bool:
        return word in self._mem
