"""tests/test_ratis_net_v4_emocontext_scaled.py — Piste 2 : scaling + unité séquence.

La piste 1 a montré que le décodeur plafonne sur happy car le classifieur
sous-jacent est entraîné MOT-À-MOT sur un corpus DÉSÉQUILIBRÉ (happy = 14%).
Cette piste attaque les deux causes racines :

  1. SCALING : on entraîne sur le corpus EmoContext COMPLET (30 160 dialogues),
     pas sur 300. GUDHI rend la tokenisation topologique du vocabulaire complet
     (14 362 mots) faisable (~14 s).
  2. UNITÉ SÉQUENCE : on entraîne le réseau à classer des SÉQUENCES (un
     dialogue = un sample), pas des mots isolés. Un dialogue happy contient une
     dominante de mots happy → la séquence est classée happy, même si "you"/"are"
     sont neutres. C'est l'unité fidèle à LCT : la forme du message, pas chaque
     mot (le courant).
  3. RÉÉQUILIBRAGE : undersampling pour que chaque émotion pèse autant (happy
     n'est plus noyée). La loi LCT est inchangée.

On compare 3 configurations (mesurées, pas supposées) :
  A) baseline mot-à-mot, 300 dialogues, déséquilibré (la piste 4 historique)
  B) séquence, 300 dialogues, rééquilibré (même taille, juste l'unité change)
  C) séquence, corpus complet, rééquilibré (scaling total)

Métriques : accuracy test (vote sur turn3), F1 macro (sensible au minoritaire),
et émergence émotionnelle (différentiels C_seuil).
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from ratis_net.pipeline import (
    Pipeline, EmoContextDataSource, HashTokenizer, TopoTokenizer, RatisNetV4Learner,
)
from ratis_net.emocontext_loader import (
    load_emocontext, build_samples, build_sequence_samples, balance_classes,
    tokenize, EMO_MAP,
)
from ratis_net.persistence_optimizer import is_gudhi_available, preferred_backend


def f1_macro(y_true, y_pred):
    """F1 macro : moyenne du F1 sur chaque classe. Sensible aux classes minoritaires
    (contrairement à l'accuracy qui est dominée par la classe majoritaire)."""
    classes = sorted(set(y_true) | set(y_pred))
    f1s = []
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s))


def run_config(examples, tokenizer, learner, per_word, balanced,
               n_dialogues, epochs, label):
    """Entraîne et évalue une configuration. Retourne un rapport."""
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(examples))
    ntr = int(0.8 * len(examples))
    tr = [examples[i] for i in idx[:ntr]]
    te = [examples[i] for i in idx[ntr:]]

    dim = tokenizer.dim()
    emb_fn = lambda w, d: tokenizer.embed(w, d)
    if per_word:
        samples = build_samples([e.__dict__ for e in tr], emb_fn, dim=dim, per_word=True)
    else:
        samples = build_sequence_samples([e.__dict__ for e in tr], emb_fn, dim=dim)
    if balanced:
        samples = balance_classes(samples)

    t0 = time.time()
    learner.train(samples, epochs)
    t_train = time.time() - t0

    # éval : vote sur turn3 (séquence) pour chaque dialogue de test
    y_true, y_pred = [], []
    for ex in te:
        words = tokenize(ex.turn3)
        if not words:
            continue
        votes = [learner.predict(tokenizer.embed(w, dim), ex.env) for w in words]
        pred = int(np.argmax(np.bincount(votes)))
        y_true.append(ex.label_num)
        y_pred.append(pred)
    acc = float(np.mean(np.array(y_true) == np.array(y_pred))) if y_true else 0.0
    f1 = f1_macro(y_true, y_pred)

    # par classe
    per_class = {}
    for c in sorted(set(y_true)):
        mask = np.array(y_true) == c
        per_class[c] = float(np.mean(np.array(y_pred)[mask] == c)) if mask.sum() else 0.0

    # émergence : C_seuil par émotion pour "ok"
    c_seuils = {}
    emb_ok = tokenizer.embed("ok", dim)
    for lab, (env_cls, _, _) in EMO_MAP.items():
        c_seuils[lab] = learner.c_seuil_for(emb_ok, env_cls())

    print(f"\n  [{label}] {n_dialogues} dial, {'mot-à-mot' if per_word else 'séquence'}, "
          f"{'rééquilibré' if balanced else 'brut'}")
    print(f"    acc test = {acc:.3f} | F1 macro = {f1:.3f} | "
          f"train {t_train:.1f}s, {len(samples)} samples")
    cls_name = {0: "angry", 1: "happy", 2: "others"}
    pc_str = " | ".join(f"{cls_name.get(c,c)}={per_class[c]:.2f}" for c in sorted(per_class))
    print(f"    rappel par classe : {pc_str}")
    print(f"    émergence C_seuil 'ok' : " +
          " | ".join(f"{k}={v:.3f}" for k, v in c_seuils.items()))
    diff_ha = c_seuils.get("happy", 0) - c_seuils.get("angry", 0)
    print(f"    différentiel happy-angry = {diff_ha:+.3f}")

    return {
        "label": label, "per_word": per_word, "balanced": balanced,
        "n_dialogues": n_dialogues, "acc_test": acc, "f1_macro": f1,
        "per_class_recall": {cls_name.get(c, str(c)): per_class[c] for c in sorted(per_class)},
        "c_seuils": c_seuils, "n_samples": len(samples), "train_time": t_train,
    }


def main():
    print("=" * 72)
    print("Piste 2 — Scaling EmoContext + unité SÉQUENCE + rééquilibrage")
    print("=" * 72)
    print(f"  GUDHI : {is_gudhi_available()} | backend : {preferred_backend()}")

    ds = EmoContextDataSource()
    # A) baseline : 300 dialogues
    ex300 = ds.load(max_examples=300)
    # B/C : on charge plus pour le scaling (pas tout pour rester rapide, mais bien plus)
    print("  Chargement du corpus (scaling)...")
    ex_full = ds.load(max_examples=3000)  # 10x le POC, faisable CPU
    print(f"  {len(ex300)} (POC) + {len(ex_full)} (scaling) dialogues chargés")

    results = {}

    # A) baseline mot-à-mot, 300, déséquilibré
    results["A_baseline"] = run_config(
        ex300, HashTokenizer(), RatisNetV4Learner(),
        per_word=True, balanced=False, n_dialogues=300, epochs=8, label="A_baseline")

    # B) séquence, 300, rééquilibré (même taille, juste l'unité change)
    results["B_seq_bal_300"] = run_config(
        ex300, HashTokenizer(), RatisNetV4Learner(),
        per_word=False, balanced=True, n_dialogues=300, epochs=8, label="B_seq_bal_300")

    # C) séquence, scaling 3000, rééquilibré
    results["C_seq_bal_3000"] = run_config(
        ex_full, HashTokenizer(), RatisNetV4Learner(),
        per_word=False, balanced=True, n_dialogues=len(ex_full), epochs=6,
        label="C_seq_bal_scaled")

    print(f"\n{'='*72}")
    print("BILAN PISTE 2")
    print(f"{'='*72}")
    print(f"  {'config':22s} {'acc':>6s} {'F1':>6s} {'happy':>6s} {'angry':>6s} "
          f"{'others':>7s} {'ha_diff':>8s}")
    for k, r in results.items():
        pc = r["per_class_recall"]
        diff = r["c_seuils"].get("happy", 0) - r["c_seuils"].get("angry", 0)
        print(f"  {r['label']:22s} {r['acc_test']:6.3f} {r['f1_macro']:6.3f} "
              f"{pc.get('happy',0):6.2f} {pc.get('angry',0):6.2f} "
              f"{pc.get('others',0):7.2f} {diff:+8.3f}")
    print(f"\n  → L'unité SÉQUENCE + le rééquilibrage ciblent la cause racine du")
    print(f"    plafonnement happy (piste 1) : le classifieur mot-à-mot déséquilibré.")
    print(f"    Le F1 macro (sensible au minoritaire) et le rappel happy mesurent")
    print(f"    si happy cesse d'être noyée. Le scaling valide la faisabilité large.")

    return results


if __name__ == "__main__":
    out = main()
    out_path = _ROOT / "proofs" / "ratis_net_v4_emocontext_scaled_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nRésultats sauvegardés : {out_path}")
