"""ratis_net.glove_tokenizer — Tokenisation sémantique + topologique.

Hybride GloVe (sémantique, 400K mots, Stanford NLP) + signature topologique
(P_sig, Betti, cycles H1) du topo_tokenizer. GloVe apporte la variance lexicale
qui manquait au topo_tokenizer pur (les lettres produisaient des signatures
quasi constantes) ; la composante topologique conserve le signal LCT.

Le vecteur final = concat(glove[:n_glove], topo_sig[:n_topo]) puis normalisé.
Deux mots sémantiquement différents produisent des vecteurs différents ; deux
mots topologiquement équivalents partagent la même composante topo.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    from ratis_net.topo_tokenizer import topo_signature
except ImportError:
    from topo_tokenizer import topo_signature

_GLOVE_PATH = Path(__file__).resolve().parents[1] / "data" / "glove" / "glove.6B.50d.txt"
_glove_cache: dict[str, np.ndarray] | None = None


def _load_glove() -> dict[str, np.ndarray]:
    global _glove_cache
    if _glove_cache is not None:
        return _glove_cache
    if not _GLOVE_PATH.exists():
        _glove_cache = {}
        return _glove_cache
    cache: dict[str, np.ndarray] = {}
    with open(_GLOVE_PATH, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            cache[parts[0]] = np.array(parts[1:], dtype=float)
    _glove_cache = cache
    return cache


def glove_embedding(word: str, dim: int = 50) -> np.ndarray:
    """Retourne l'embedding GloVe du mot, ou un vecteur nul si absent."""
    glove = _load_glove()
    w = word.lower().strip()
    if w in glove:
        v = glove[w][:dim]
        if len(v) < dim:
            v = np.pad(v, (0, dim - len(v)))
        return v
    return np.zeros(dim, dtype=float)


def glove_topo_signature(word: str, dim: int = 12, n_glove: int = 8,
                          n_topo: int = 4, seed: int = 42,
                          topo_cache: Any = None) -> np.ndarray:
    """Signature hybride GloVe + topologique, de dimension `dim`.

    Les `n_glove` premières composantes viennent de GloVe (variance sémantique) ;
    les `n_topo` suivantes viennent du topo_tokenizer (signal LCT). Le vecteur
    est normalisé. Si un cache topo est fourni, le lookup est O(1) ; sinon on
    calcule topo_signature à la volée (lent). Si GloVe n'est pas disponible,
    retombe sur topo seul.
    """
    n_glove = min(n_glove, dim)
    n_topo = dim - n_glove
    glove_vec = glove_embedding(word, dim=50)[:n_glove]
    if topo_cache is not None:
        topo_vec = topo_cache.get(word)[:n_topo]
    else:
        topo_vec = topo_signature(word, dim=max(n_topo * 2, 8), seed=seed)[:n_topo]
    sig = np.concatenate([glove_vec, topo_vec])
    n = np.linalg.norm(sig)
    if n > 1e-9:
        sig = sig / n
    return sig


class GloveTokenizer:
    """Tokenizer branchable sur le pipeline : mot → embedding hybride.

    Utilise le cache topo existant (15 122 mots, O(1)) + GloVe (400K mots).
    Retombe sur topo_signature à la volée si le mot n'est pas en cache.
    """

    def __init__(self, dim: int = 12, n_glove: int = 8, seed: int = 42,
                 use_topo_cache: bool = True):
        self.dim = dim
        self.n_glove = n_glove
        self.seed = seed
        self.name = "glove_topo"
        self._topo_cache = None
        if use_topo_cache:
            try:
                from ratis_net.topo_cache import TopoCache
                self._topo_cache = TopoCache(dim=8)
                self._topo_cache.load()
            except Exception:
                self._topo_cache = None

    def __call__(self, word: str, dim: int | None = None) -> np.ndarray:
        return glove_topo_signature(word, dim=dim or self.dim,
                                     n_glove=self.n_glove, seed=self.seed,
                                     topo_cache=self._topo_cache)

    def backend(self) -> str:
        glove = _load_glove()
        return f"glove({len(glove)}_words)+topo" if glove else "topo_fallback"


def is_glove_available() -> bool:
    return len(_load_glove()) > 0


if __name__ == "__main__":
    import numpy as np
    print("GloVe disponible ?", is_glove_available())
    words = ["happy", "sad", "angry", "sorry", "love", "hate"]
    embs = {w: glove_topo_signature(w, dim=12, n_glove=8) for w in words}
    for w in words:
        print(f"  {w:8s} → {np.round(embs[w], 4)}")
    distinct = len(set(tuple(np.round(e, 4)) for e in embs.values()))
    print(f"\n  {len(words)} mots → {distinct} signatures distinctes")

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    print(f"  cos(happy, sad)   = {cos(embs['happy'], embs['sad']):.4f}")
    print(f"  cos(happy, angry)  = {cos(embs['happy'], embs['angry']):.4f}")
    print(f"  cos(happy, love)   = {cos(embs['happy'], embs['love']):.4f}")
    print(f"  cos(happy, hate)   = {cos(embs['happy'], embs['hate']):.4f}")
