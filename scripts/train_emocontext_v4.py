"""Entraînement EmoContext complet (séquence + rééquilibrage + cache) → poids sauvés.

Beu les 4 émotions (EMO_MAP). Vote de test sur turn3. Sérialise le meilleur
learner (pickle) + métriques (json) pour le décodeur. Aucun fichier existant
modifié.

Usage : python scripts/train_emocontext_v4.py [--epochs 10] [--eta 0.1]
        [--n-hidden 20] [--out-dir trained]
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ratis_net.emocontext_loader import (
    load_emocontext, build_sequence_samples, balance_classes, tokenize, EMO_MAP,
)
from ratis_net.eth_thermo_fixer import ThermoEnvironment
from ratis_net.pipeline import RatisNetV4Learner
from ratis_net.topo_cache import TopoCache

DATA = Path(__file__).resolve().parent.parent / "data" / "emocontext"


def f1_macro(y_true: list[int], y_pred: list[int]) -> float:
    classes = sorted(set(y_true) | set(y_pred))
    f1s = []
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


class CachedEmbedder:
    def __init__(self, dim: int = 8):
        self.cache = TopoCache(dim=dim)
        self.cache.load()

    def __call__(self, word: str, dim: int = 8) -> np.ndarray:
        return self.cache.get(word)


def evaluate(learner: RatisNetV4Learner, examples: list[dict], embed, env_map,
             eval_pooling: bool = True) -> dict:
    """Vote prédictif sur turn3 (même protocole que pipeline.run)."""
    y_true, y_pred = [], []
    for ex in examples:
        if eval_pooling:
            words = tokenize(" ".join([ex["turn1"], ex["turn2"], ex["turn3"]]))
        else:
            votes = [learner.predict(embed(w, learner.net.token_dim), ex["env"])
                     for w in tokenize(ex["turn3"])]
            if votes:
                y_true.append(ex["label_num"])
                y_pred.append(int(np.argmax(np.bincount(votes))))
            continue
        if not words:
            continue
        embs = np.array([embed(w, learner.net.token_dim) for w in words])
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        seq_emb = (embs * norms).sum(axis=0) / norms.sum()
        n = np.linalg.norm(seq_emb)
        seq_emb = seq_emb / n if n > 1e-9 else seq_emb
        y_pred.append(learner.predict(seq_emb, ex["env"]))
        y_true.append(ex["label_num"])
    acc = float(np.mean([int(t == p) for t, p in zip(y_true, y_pred)]))
    return {"acc": acc, "f1": f1_macro(y_true, y_pred),
            "n": len(y_true)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--n-hidden", type=int, default=20)
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--out-dir", default="trained")
    args = ap.parse_args()

    embed = CachedEmbedder(dim=8)
    n_out = len({v[2] for v in EMO_MAP.values()})

    examples = []
    for split in ("train.txt", "dev.txt"):
        p = DATA / split
        if p.exists():
            examples.extend(load_emocontext(p, max_examples=args.max_examples))
    print(f"{len(examples)} dialogues, {len(embed.cache)} signatures en cache")

    rng = np.random.RandomState(42)
    idx = rng.permutation(len(examples))
    ntr = int(0.8 * len(examples))
    tr = [examples[i] for i in idx[:ntr]]
    te = [examples[i] for i in idx[ntr:]]

    # Honnetete : env charge depuis EMO_MAP est derive du LABEL (fuite parfaite
    # dans l input du reseau -> accuracy 1.000 triviale, convention historique de
    # pipeline.py). On force un environnement NEUTRE (calm) pour train ET eval :
    # le reseau classe sur les embeddings seuls.
    # Train : env-supervise (label -> env) = signal d apprentissage ETH.
    # Eval : env NEUTRE (calm) -> aucune fuite du label dans l input.
    neutral = ThermoEnvironment.calm()
    for ex in tr + te:
        ex["env"] = neutral

        tr3 = []
    for ex in tr:
        ex2 = dict(ex)
        ex2["turn3"] = " ".join([ex["turn1"], ex["turn2"], ex["turn3"]])
        tr3.append(ex2)
    samples = build_sequence_samples(tr3, embed, dim=8, turn="turn3", min_words=2)
    balanced = balance_classes(samples)
    from collections import Counter
    lbl = Counter(s[2] for s in balanced)
    print(f"{len(balanced)} samples équilibrés : {dict(lbl)}")

    learner = RatisNetV4Learner(n_in=12, n_hidden=args.n_hidden, n_out=n_out,
                                token_dim=8, env_dim=4, eta=args.eta, seed=42)
    t0 = time.time()
    res = learner.train(balanced, args.epochs)
    ev = evaluate(learner, te, embed, EMO_MAP)
    dt = time.time() - t0
    print(f"eta={args.eta} hidden={args.n_hidden} epochs={args.epochs} : "
          f"train={res['acc_train']:.3f} test={ev['acc']:.3f} f1={ev['f1']:.3f} "
          f"({dt:.0f}s)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "learner.pkl", "wb") as f:
        pickle.dump(learner, f)
    metrics = {"eta": args.eta, "n_hidden": args.n_hidden, "epochs": args.epochs,
               "acc_train": res["acc_train"], "acc_test": ev["acc"],
               "f1_test": ev["f1"], "n_samples": len(balanced),
               "trained_seconds": dt, "n_eval": ev["n"]}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"sauvegardé → {out_dir}/learner.pkl + metrics.json")


if __name__ == "__main__":
    main()
