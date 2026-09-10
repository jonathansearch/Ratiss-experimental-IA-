"""ratis_net.cached_tokenizer — Tokenizer branché sur le TopoCache (lookup O(1)).

Compatible Pipeline (interface Tokenizer : embed / name / dim). Nouveau module,
aucun fichier existant modifié. Les embeddings proviennent du cache pré-calculé
(data/cache/topo_signatures.npz) ; les mots inconnus sont calculés à la volée
(parez) puis servis du dict mémoire.
"""
from __future__ import annotations

import numpy as np

from .pipeline import Tokenizer
from .topo_cache import TopoCache


class CachedTokenizer(Tokenizer):
    """topo_signature via cache disque → les gros runs n'attendent plus la
    persistance, juste le chargement npz (~0.03s pour 15k mots)."""

    def __init__(self, dim: int = 10, path: str | None = None):
        self.cache = TopoCache(dim=dim, path=path) if path else TopoCache(dim=dim)
        try:
            self.cache.load()
        except FileNotFoundError:
            pass  # le calcul paresseux remplira la mémoire
        self._dim = dim

    def embed(self, word: str, dim: int) -> np.ndarray:
        if dim != self._dim:
            # recontruit si un autre dim est exigé (rare)
            import numpy as _np
            sig = self.cache.get(word)
            return _np.pad(sig, (0, dim - len(sig)))[:dim]
        return self.cache.get(word)

    def name(self) -> str:
        return f"CACHE[topo/{len(self.cache)}]"

    def dim(self) -> int:
        return self._dim

    def is_available(self) -> bool:
        return True