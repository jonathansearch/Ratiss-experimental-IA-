"""tests/test_ratis_net_v3.py — Preuve de concept v3 : proxy topo differentiable.

v3 : ΔW = η·φ·P_sig·C + η2·∇_W(variance distances)
Le proxy (variance) est DIFFERENTIABLE → stable.
On verifie : (1) accuracy, (2) P_sig croit, (3) variance croit.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from ratis_net.lct_network_v3 import LCTNetworkV3

def load_iris():
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data.astype(float)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    y = iris.target
    y_oh = np.zeros((len(y), 3)); y_oh[np.arange(len(y)), y] = 1.0
    return X, y_oh, y

def main():
    print("=" * 72)
    print("RATIS-Net v3 — Proxy topo differentiable (variance des distances)")
    print("Delta W = eta*phi*P_sig*C + eta2*grad_W(variance)")
    print("=" * 72)

    try:
        X, y_oh, y = load_iris()
    except ImportError:
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (150, 4)); y = rng.integers(0, 3, 150)
        y_oh = np.zeros((150, 3)); y_oh[np.arange(150), y] = 1.0

    idx = np.random.RandomState(42).permutation(len(X))
    n_train = int(0.8 * len(X))
    X_train, X_test = X[idx[:n_train]], X[idx[n_train:]]
    y_train, y_test = y_oh[idx[:n_train]], y[idx[n_train:]]

    print(f"Dataset : {len(X)} samples, {X.shape[1]} features, 3 classes")
    print(f"RATIS-Net v3 : 4->10->3, eta=0.05, eta2=0.001")

    net = LCTNetworkV3(n_in=4, n_hidden=10, n_out=3, eta=0.05, eta2=0.001, seed=42)
    acc_hist, psig_hist = net.train(X_train, y_train, epochs=40, verbose=True)

    pred_test = net.predict(X_test)
    test_acc = float(np.mean(pred_test == y_test))

    # monotonie de P_sig et variance
    epochs_arr = np.arange(len(psig_hist))
    ra = np.argsort(np.argsort(epochs_arr)); rb = np.argsort(np.argsort(psig_hist))
    sp_psig = float(np.corrcoef(ra, rb)[0, 1]) if len(psig_hist) > 2 else 0.0
    var_hist = net.variance_history
    rb_v = np.argsort(np.argsort(var_hist))
    sp_var = float(np.corrcoef(ra, rb_v)[0, 1]) if len(var_hist) > 2 else 0.0

    psig_init = psig_hist[0] if psig_hist else 0
    psig_final = psig_hist[-1] if psig_hist else 0
    psig_growth = psig_final - psig_init
    var_init = var_hist[0] if var_hist else 0
    var_final = var_hist[-1] if var_hist else 0
    var_growth = var_final - var_init

    print(f"\n-- Validation v3 --")
    print(f"  Accuracy test     = {test_acc:.3f}")
    print(f"  P_sig : {psig_init:.4f} -> {psig_final:.4f} (growth {psig_growth:+.4f}, Spearman {sp_psig:+.3f})")
    print(f"  Variance : {var_init:.4f} -> {var_final:.4f} (growth {var_growth:+.4f}, Spearman {sp_var:+.3f})")

    learns = acc_hist[-1] > acc_hist[0] + 0.05
    var_grows = var_growth > 0
    psig_grows = psig_growth > 0

    print(f"\n  APPREND ?      : {'OUI' if learns else 'NON'}")
    print(f"  Variance CROIT?: {'OUI' if var_grows else 'NON'}")
    print(f"  P_sig CROIT ?  : {'OUI' if psig_grows else 'NON'}")

    if learns and var_grows:
        print(f"\n  -> BOUCLE FERMEE (proxy) : le reseau apprend ET distribue ses neurones.")
        print(f"    Le proxy differentiable (variance) stabilise l'optimisation topo.")
        verdict = "PASS"
    elif learns:
        print(f"\n  -> Le reseau apprend. Proxy a stabiliser.")
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    return {
        "test_accuracy": test_acc,
        "acc_history": [round(float(a), 4) for a in acc_hist],
        "psig_history": [round(float(p), 4) for p in psig_hist],
        "variance_history": [round(float(v), 4) for v in var_hist],
        "psig_growth": round(float(psig_growth), 4),
        "variance_growth": round(float(var_growth), 4),
        "spearman_psig": sp_psig, "spearman_variance": sp_var,
        "learns": learns, "var_grows": var_grows, "psig_grows": psig_grows,
        "verdict": verdict,
    }

if __name__ == "__main__":
    out = main()
    out_path = _ROOT / "proofs" / "ratis_net_v3_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResultats sauvegardes : {out_path}")
