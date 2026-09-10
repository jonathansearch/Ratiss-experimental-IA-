"""Démo : décodeur LCT branché sur le cache topo (génération conditionnée par émotion).

Chaîne : cache signatures (lookup O(1)) → embeddings mots → BigramModel (EmoContext)
→ LCTDecoder.generate (glouton + beam) pour une émotion cible. Aucun fichier
existant modifié ; la démo est un script autonome.

Usage : python scripts/decode_with_cache.py [--emotion happy] [--n-words 8]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ratis_net.decoder import BigramModel, LCTDecoder
from ratis_net.emocontext_loader import load_emocontext, vocabulary, EMO_MAP
from ratis_net.eth_thermo_fixer import ThermoEnvironment
from ratis_net.topo_cache import TopoCache

DATA = Path(__file__).resolve().parent.parent / "data" / "emocontext"


class _ProbeLearner:
    """Learner minimal pour le décodeur : scores par similarité cos à la
    signature moyenne de l'émotion cible (construite du vocab du cache)."""

    def __init__(self, cache: TopoCache, vocab: list[str]):
        self.cache = cache
        self.vocab = vocab
        sigs = np.array([cache.get(w) for w in vocab])
        self.centroid = sigs.mean(axis=0)
        self.centroid /= (np.linalg.norm(self.centroid) + 1e-9)

    def scores(self, embedding: np.ndarray, env: ThermoEnvironment) -> np.ndarray:
        """Confiance par émotion : cos(embedding, centroid) comme vote émotion."""
        v = embedding / (np.linalg.norm(embedding) + 1e-9)
        c = float(np.dot(v, self.centroid))
        base = np.ones(4) * 0.25
        # léger biais vers une classe pour que generate soit déterministe
        return base + np.array([c, -c * 0.3, 0.0, 0.0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emotion", default="happy", choices=list(EMO_MAP.keys()))
    ap.add_argument("--n-words", type=int, default=8)
    ap.add_argument("--max-examples", type=int, default=4000)
    ap.add_argument("--top-k", type=int, default=4000)
    args = ap.parse_args()

    cache = TopoCache().load()
    examples = load_emocontext(DATA / "train.txt", max_examples=args.max_examples)
    vocab = [w for w in vocabulary(examples, min_len=2, top_k=args.top_k) if w in cache]
    print(f"cache: {len(cache)} sigs | vocab filtré: {len(vocab)} mots")

    bigram = BigramModel()
    bigram.fit(examples)

    learner = _ProbeLearner(cache, vocab)
    env = ThermoEnvironment()
    decoder = LCTDecoder(learner, {w: cache.get(w) for w in vocab}, vocab, bigram)

    out = decoder.generate_greedy(args.emotion, env, length=args.n_words)
    print(f"[{args.emotion}] greedy :", " ".join(out))

    out_beam = decoder.generate_beam(args.emotion, env, length=args.n_words,
                                     beam_width=4)
    print(f"[{args.emotion}] beam   :", " ".join(out_beam))


if __name__ == "__main__":
    main()
