"""Tests du cache de signatures topo (chemins réels, pas de mock).
Script exécutable : python tests/test_topo_cache.py (convention du dépôt)."""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ratis_net import topo_tokenizer
from ratis_net.topo_cache import TopoCache


def main() -> None:
    tmp = Path(tempfile.mkdtemp())

    cache = TopoCache(dim=10, path=tmp / "sig")
    vocab = ["bonjour", "salut", "aa", "xyz"]
    cache.warmup(vocab, progress_every=0)
    assert len(cache) == 4, "warmup crée 4 entrées"
    ref = cache.get("bonjour")

    cache.save()

    loaded = TopoCache(dim=10, path=tmp / "sig").load()
    assert len(loaded) == 4, "load restitue 4 entrées"
    np.testing.assert_allclose(loaded.get("bonjour"), ref, atol=1e-10)

    np.testing.assert_allclose(loaded.get("salut"),
                               topo_tokenizer.topo_signature("salut", dim=10),
                               atol=1e-10)
    print("lookup_and_persistence OK")

    c2 = TopoCache(dim=8, path=tmp / "sig2")
    sig = c2.get("nouveau")
    assert "nouveau" in c2
    assert sig.shape == (8,)
    assert c2.get("nouveau") is sig, "second acces = meme objet (pas de recalcul)"
    print("missing_word_is_computed OK")

    c3 = TopoCache(dim=8, path=tmp / "sig3")
    first = c3.warmup(["a", "b"], progress_every=0)
    second = c3.warmup(["a", "b"], progress_every=0)
    assert first == 2
    assert second == 0, "warmup idempotent"
    print("warmup_idempotent OK")

    print("TOUT OK - cache topo valide")


if __name__ == "__main__":
    main()
