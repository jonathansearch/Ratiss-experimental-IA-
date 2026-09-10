"""Entraînement EmoContext avec le tokenizer hybride GloVe+topo.

Le topo_tokenizer pur produisait des signatures quasi constantes (std < 0.02)
→ le learner plafonnait à 0.501 (hasard 0.33). Le tokenizer hybride apporte la
variance sémantique de GloVe (400K mots, Stanford NLP) tout en conservant la
signature topologique (P_sig, Betti) fidèle à la loi LCT.

Usage : python scripts/train_emocontext_glove.py [--epochs 10] [--eta 0.1]
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ratis_net.emocontext_loader import (
    load_emocontext, build_sequence_samples, balance_classes, tokenize, EMO_MAP,
)
from ratis_net.eth_thermo_fixer import ThermoEnvironment
from ratis_net.pipeline import RatisNetV4Learner
from ratis_net.glove_tokenizer import GloveTokenizer


DATA = Path(__file__).resolve().parent.parent / "data" / "emocontext"


def f1_macro(y_true, y_pred):
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


def evaluate(learner, examples, embed, eval_pooling=True):
    y_true, y_pred = [], []
    neutral = ThermoEnvironment.calm()
    for ex in examples:
        if eval_pooling:
            words = tokenize(" ".join([ex["turn1"], ex["turn2"], ex["turn3"]]))
        else:
            words = tokenize(ex["turn3"])
        if not words:
            continue
        embs = np.array([embed(w, learner.net.token_dim) for w in words])
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        seq_emb = (embs * norms).sum(axis=0) / norms.sum()
        n = np.linalg.norm(seq_emb)
        seq_emb = seq_emb / n if n > 1e-9 else seq_emb
        y_pred.append(learner.predict(seq_emb, neutral))
        y_true.append(ex["label_num"])
    acc = float(np.mean([int(t == p) for t, p in zip(y_true, y_pred)]))
    return {"acc": acc, "f1": f1_macro(y_true, y_pred), "n": len(y_true)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--n-hidden", type=int, default=20)
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--dim", type=int, default=12)
    ap.add_argument("--n-glove", type=int, default=8)
    args = ap.parse_args()

    embed = GloveTokenizer(dim=args.dim, n_glove=args.n_glove)
    print(f"Tokenizer: {embed.name} | backend: {embed.backend()}")
    n_out = len({v[2] for v in EMO_MAP.values()})

    examples = []
    for split in ("train.txt", "dev.txt"):
        p = DATA / split
        if p.exists():
            examples.extend(load_emocontext(p, max_examples=args.max_examples))
    print(f"{len(examples)} dialogues chargés")

    rng = np.random.RandomState(42)
    idx = rng.permutation(len(examples))
    ntr = int(0.8 * len(examples))
    tr = [examples[i] for i in idx[:ntr]]
    te = [examples[i] for i in idx[ntr:]]

    neutral = ThermoEnvironment.calm()
    for ex in tr + te:
        ex["env"] = neutral

    tr3 = []
    for ex in tr:
        ex2 = dict(ex)
        ex2["turn3"] = " ".join([ex["turn1"], ex["turn2"], ex["turn3"]])
        tr3.append(ex2)
    samples = build_sequence_samples(tr3, embed, dim=args.dim, turn="turn3", min_words=2)
    balanced = balance_classes(samples)
    lbl = Counter(s[2] for s in balanced)
    print(f"{len(balanced)} samples équilibrés : {dict(lbl)}")

    learner = RatisNetV4Learner(n_in=args.dim + 4, n_hidden=args.n_hidden,
                                n_out=n_out, token_dim=args.dim, env_dim=4,
                                eta=args.eta, seed=42)
    t0 = time.time()
    res = learner.train(balanced, args.epochs)
    ev = evaluate(learner, te, embed)
    dt = time.time() - t0
    print(f"eta={args.eta} hidden={args.n_hidden} epochs={args.epochs} : "
          f"train={res['acc_train']:.3f} test={ev['acc']:.3f} f1={ev['f1']:.3f} ({dt:.0f}s)")

    metrics = {"tokenizer": embed.name, "eta": args.eta, "n_hidden": args.n_hidden,
               "epochs": args.epochs, "acc_train": res["acc_train"],
               "acc_test": ev["acc"], "f1_test": ev["f1"],
               "n_samples": len(balanced), "trained_seconds": dt, "n_eval": ev["n"]}
    Path("trained").mkdir(exist_ok=True)
    with open("trained/metrics_glove.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"sauvegardé → trained/metrics_glove.json")


if __name__ == "__main__":
    main()
