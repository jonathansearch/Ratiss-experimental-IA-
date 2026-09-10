"""tests/test_ratis_net_v2.py — Preuve de concept v2 : maximise P_sig.

v1 : le réseau apprend par LCT mais P_sig oscille (passager).
v2 : ΔW = η·φ·P_sig·C + η2·∇_W(P_sig) → P_sig est MAXIMISÉ explicitement.

On vérifie :
  1. Le réseau apprend (accuracy augmente)
  2. P_sig CROÎT de façon monotone (la topologie devient robuste)
  3. La courbe P_sig a une monotonie Spearman > 0.6 (comme R(C) sur 4MZI)

Si oui → boucle fermée : le réseau apprend EN devenant topologiquement robuste.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from ratis_net.lct_network_v2 import LCTNetworkV2


def load_iris():
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data.astype(float)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    y = iris.target
    y_oh = np.zeros((len(y), 3))
    y_oh[np.arange(len(y)), y] = 1.0
    return X, y_oh, y


def main():
    print("=" * 72)
    print("RATIS-Net v2 — Maximise P_sig (gradient topologique)")
    print("Delta W = eta*phi*P_sig*C + eta2*grad_W(P_sig) | loss = -P_sig")
    print("=" * 72)

    try:
        X, y_oh, y = load_iris()
    except ImportError:
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (150, 4))
        y = rng.integers(0, 3, 150)
        y_oh = np.zeros((150, 3)); y_oh[np.arange(150), y] = 1.0

    idx = np.random.RandomState(42).permutation(len(X))
    n_train = int(0.8 * len(X))
    X_train, X_test = X[idx[:n_train]], X[idx[n_train:]]
    y_train, y_test = y_oh[idx[:n_train]], y[idx[n_train:]]

    print(f"Dataset : {len(X)} samples, {X.shape[1]} features, 3 classes")
    print(f"RATIS-Net v2 : 4->10->3, eta=0.05, eta2=0.03 (gradient topo)")

    net = LCTNetworkV2(n_in=4, n_hidden=10, n_out=3, eta=0.05, eta2=0.03, seed=42)
    acc_hist, psig_hist = net.train(X_train, y_train, epochs=40, verbose=True)

    pred_test = net.predict(X_test)
    test_acc = float(np.mean(pred_test == y_test))

    # monotonie de P_sig
    epochs_arr = np.arange(len(psig_hist))
    ra = np.argsort(np.argsort(epochs_arr))
    rb = np.argsort(np.argsort(psig_hist))
    spearman_psig = float(np.corrcoef(ra, rb)[0, 1]) if len(psig_hist) > 2 else 0.0
    psig_init = psig_hist[0] if psig_hist else 0
    psig_final = psig_hist[-1] if psig_hist else 0
    psig_growth = psig_final - psig_init

    print(f"\n-- Validation v2 --")
    print(f"  Accuracy test     = {test_acc:.3f}")
    print(f"  P_sig initial      = {psig_init:.4f}")
    print(f"  P_sig final        = {psig_final:.4f}")
    print(f"  Croissance P_sig   = {psig_growth:+.4f}")
    print(f"  Spearman(P_sig,t)  = {spearman_psig:+.4f}  (monotonie de P_sig)")

    learns = acc_hist[-1] > acc_hist[0] + 0.1
    psig_monotone = spearman_psig > 0.6
    psig_grows = psig_growth > 0

    print(f"\n  Le reseau APPREND ?      : {'OUI' if learns else 'NON'}")
    print(f"  P_sig CROIT ?            : {'OUI' if psig_grows else 'NON'}")
    print(f"  P_sig MONOTONE (Spearman)? : {'OUI' if psig_monotone else 'NON'}")

    if learns and psig_monotone:
        print(f"\n  -> BOUCLE FERMEE : le reseau apprend EN devenant topologiquement robuste.")
        print(f"    P_sig croit de facon monotone (Spearman {spearman_psig:+.2f}).")
        print(f"    L'accuracy est un effet secondaire de la robustesse topologique.")
        print(f"    La loi LCT gouverne l'apprentissage ET l'auto-regulation.")
        verdict = "PASS"
    elif learns and psig_grows:
        print(f"\n  -> P_sig croit mais pas monotone strict. Presque la boucle fermee.")
        verdict = "PARTIAL"
    else:
        print(f"\n  -> Ajuster hyperparametres (eta2, epochs).")
        verdict = "FAIL"

    return {
        "test_accuracy": test_acc,
        "acc_history": [round(float(a), 4) for a in acc_hist],
        "psig_history": [round(float(p), 4) for p in psig_hist],
        "psig_growth": round(float(psig_growth), 4),
        "spearman_psig_monotonicity": spearman_psig,
        "learns": learns,
        "psig_grows": psig_grows,
        "psig_monotone": psig_monotone,
        "verdict": verdict,
    }


if __name__ == "__main__":
    out = main()
    out_path = _ROOT / "proofs" / "ratis_net_v2_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResultats sauvegardes : {out_path}")
