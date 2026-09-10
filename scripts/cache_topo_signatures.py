"""CLI : pré-calcule les signatures topo du vocabulaire EmoContext, une seule fois.

Usage : python scripts/cache_topo_signatures.py [--min-len 2] [--top-k 50000]
Spyder la signature via `TopoCache(...).get(w)` — une fois le cache sauvé,
les entraînements deviennent pur lookup (O(1), + gudhi requis que pour les mots
jamais vus).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ratis_net.emocontext_loader import load_emocontext, vocabulary
from ratis_net.topo_cache import TopoCache

DATA = Path(__file__).resolve().parent.parent / "data" / "emocontext"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-len", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--max-examples", type=int, default=None)
    args = ap.parse_args()

    samples = []
    for split in ("train.txt", "dev.txt"):
        p = DATA / split
        if p.exists():
            samples.extend(load_emocontext(p, max_examples=args.max_examples))
    vocab = vocabulary(samples, min_len=args.min_len, top_k=args.top_k)
    print(f"Vocabulaire : {len(vocab)} mots (min_len={args.min_len})")

    cache = TopoCache()
    t0 = __import__("time").time()
    cache.warmup(vocab, progress_every=500)
    out = cache.save()
    dt = __import__("time").time() - t0
    print(f"Cache sauvé → {out} ({dt:.1f}s pour {len(cache)} signatures, "
          f"backend={cache.meta['backend']})")


if __name__ == "__main__":
    main()
