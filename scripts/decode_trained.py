"""Génération : décodeur LCT branché sur un learner MESURÉ (centroïdes, 4 classes).

Bluff heureux = espéré dans la console. Le learner n'est pas snn : c'est le
meilleur classifieur que le pipeline (cache) puisse produire honnêtement —
proto-centroides par émotion, cos-sim scores. Les poids de RatisNetV4 n'ont
pas appris (documenté) ; on ne feinte pas.

Usage : python scripts/decode_trained.py [--emotion happy] [--n-words 8]
"""
import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ratis_net.decoder import BigramModel, LCTDecoder
from ratis_net.emocontext_loader import (
    load_emocontext, build_sequence_samples, vocabulary, tokenize, EMO_MAP,
)
from ratis_net.eth_thermo_fixer import ThermoEnvironment
from ratis_net.topo_cache import TopoCache

DATA = Path(__file__).resolve().parent.parent / "data" / "emocontext"


class CentroidLearner:
    """Learner mesuré : centroïde par émotion, scores = cos sim. Honnête : c'est
    un proto-classifieur, pas le v4 (ses poids n'ont pas appris — cf. doc)."""

    def __init__(self, samples: list[tuple]):
        sums, counts = defaultdict(np.zeros), defaultdict(int)
        dim = len(samples[0][0])
        sums = {c: np.zeros(dim) for c in range(4)}
        for emb, _env, label, _cs in samples:
            sums[label] = sums[label] + emb
            counts[label] += 1
        self.centroids = {c: v / max(counts[c], 1) for c, v in sums.items()}
        for c in self.centroids:
            n = np.linalg.norm(self.centroids[c])
            if n > 1e-9:
                self.centroids[c] = self.centroids[c] / n
        self.dim = dim

    def scores(self, token: np.ndarray, env: ThermoEnvironment) -> np.ndarray:
        t = token / (np.linalg.norm(token) + 1e-9)
        return np.array([float(np.dot(t, self.centroids[c]))
                         for c in range(4)]) + 0.5

    def predict(self, token: np.ndarray, env: ThermoEnvironment) -> int:
        return int(np.argmax(self.scores(token, env)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emotion", default="happy", choices=list(EMO_MAP.keys()))
    ap.add_argument("--n-words", type=int, default=8)
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=6000)
    args = ap.parse_args()

    cache = TopoCache(); cache.load()
    def embed(w, dim=8):
        return cache.get(w)
    examples = []
    for split in ("train.txt", "dev.txt"):
        p = DATA / split
        if p.exists():
            examples.extend(load_emocontext(p, max_examples=args.max_examples))
    samples = build_sequence_samples(
        [{"turn1": e["turn1"], "turn2": e["turn2"], "turn3": e["turn3"],
          "label": e["label"], "env": e["env"], "c_seuil": e["c_seuil"],
          "label_num": e["label_num"],
          "seq": " ".join([e["turn1"], e["turn2"], e["turn3"]])} for e in examples],
        embed, dim=8, turn="turn3", min_words=2)

    rng = np.random.RandomState(42)
    idx = rng.permutation(len(examples))
    ntr = int(0.8 * len(examples))
    tr = [examples[i] for i in idx[:ntr]]
    te = [examples[i] for i in idx[ntr:]]
    samples = build_sequence_samples(tr, embed, dim=8, turn="turn3", min_words=2)

    learner = CentroidLearner(samples)
    bigram = BigramModel()
    # bigram.fit utilise un lambda local -> non picklable. On le neutralise.
    bigram.probs = None
    bigram.fit(examples)
    vocab = [w for w in vocabulary(examples, min_len=2, top_k=args.top_k) if w in cache]
    env = ThermoEnvironment.calm()

    # métrique vraie du learner mesuré (voisins naturels par poses de classes)
    y_true, y_pred = [], []
    for ex in te:
        words = tokenize(ex["turn3"])
        if not words: continue
        embs = np.array([cache.get(w) for w in words])
        norms = np.linalg.norm(embs, axis=1, keepdims=True); norms[norms < 1e-9] = 1.0
        s = (embs * norms).sum(axis=0) / norms.sum()
        s = s / (np.linalg.norm(s) + 1e-9)
        y_true.append(ex["label_num"])
        y_pred.append(learner.predict(s, env))
    acc = float(np.mean([int(t == p) for t, p in zip(y_true, y_pred)]))
    from collections import Counter as C
    cm = C(zip(y_true, y_pred))
    print(f"learner mesuré (centroïdes, env NEUTRE): acc={acc:.3f} (n={len(y_true)})")
    print("confusion (true,pred):", {k: v for k, v in cm.items() if k[0] == k[1]})

    decoder = LCTDecoder(learner, {w: cache.get(w) for w in vocab}, vocab, bigram)
    for target in list(EMO_MAP.keys()):
        out = decoder.generate_greedy(target, env, length=args.n_words)
        print(f"[{target:7s}] greedy : {' '.join(out)}")
        out_b = decoder.generate_beam(target, env, length=args.n_words, beam_width=3)
        print(f"[{target:7s}] beam   : {' '.join(out_b)}")

    out_dir = Path("trained")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "decoder_learner.pkl", "wb") as f:
        pickle.dump({"centroids": learner.centroids, "vocab": vocab}, f)
    print("decoder_learner.pkl saved (centroïdes+vocab)")


if __name__ == "__main__":
    main()
