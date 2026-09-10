"""tests/test_pipeline.py — Valide le pipeline branchable (connecteurs).

On vérifie que le pipeline assemblé en 3 lignes reproduit les résultats de la
piste 4 (accuracy + émergence d'émotion), avec les 2 tokenizers (Hash et
Topo). C'est la fondation branchable : un partenaire instancie le Pipeline
avec sa source/tokenizer/learner sans toucher au cœur.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from ratis_net.pipeline import (
    Pipeline, EmoContextDataSource, HashTokenizer, TopoTokenizer,
    RatisNetV4Learner,
)


def run_pipeline(name, tokenizer, n_dialogues=300, epochs=6):
    p = Pipeline(EmoContextDataSource(), tokenizer, RatisNetV4Learner(),
                 top_k_vocab=80)
    report = p.run(n_dialogues=n_dialogues, epochs=epochs, verbose=True)
    return report


def main():
    print("=" * 72)
    print("Test du pipeline branchable (connecteurs)")
    print("3 lignes : Pipeline(DataSource, Tokenizer, Learner)")
    print("=" * 72)

    reports = {}
    # 1. Hash (rapide, toujours dispo)
    reports["HASH"] = run_pipeline("HASH", HashTokenizer())
    # 2. Topo (GUDHI si dispo)
    topo = TopoTokenizer()
    if topo.is_available():
        reports["TOPO"] = run_pipeline("TOPO", topo)
    else:
        print("\n  TOPO non disponible (GUDHI absent) — skip")

    print(f"\n{'='*72}")
    print(f"BILAN : branchabilité")
    print(f"{'='*72}")
    for name, r in reports.items():
        emerges = max(abs(r.c_seuils.get("happy", 0) - v)
                      for k, v in r.c_seuils.items() if k != "happy") > 0.05
        print(f"  {r.tokenizer_name:14s} | acc_test={r.acc_test_vote:.3f} | "
              f"acc_train={r.acc_train:.3f} | émotion={'OUI' if emerges else 'non'} | "
              f"backend={r.backend}")

    # vérifications : le pipeline reproduit la piste 4
    ok = all(r.acc_test_vote > 0.70 for r in reports.values())  # bien au-dessus du hasard
    emerges_all = all(
        max(abs(r.c_seuils.get("happy", 0) - v) for k, v in r.c_seuils.items() if k != "happy") > 0.05
        for r in reports.values()
    )

    print(f"\n  Pipeline apprend (acc > 0.70) ? : {'OUI' if ok else 'NON'}")
    print(f"  Émotion émerge dans tous les tokenizers ? : {'OUI' if emerges_all else 'NON'}")
    print(f"  → Le pipeline est branchable : on change 1 mot pour changer de tokenizer.")
    verdict = "PASS" if (ok and emerges_all) else "FAIL"

    return {
        "verdict": verdict,
        "reports": {
            name: {
                "tokenizer": r.tokenizer_name, "backend": r.backend,
                "acc_train": r.acc_train, "acc_test_vote": r.acc_test_vote,
                "c_seuils": r.c_seuils, "n_dialogues": r.n_dialogues,
                "n_samples": r.n_samples,
            }
            for name, r in reports.items()
        },
    }


if __name__ == "__main__":
    out = main()
    out_path = _ROOT / "proofs" / "pipeline_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nRésultats sauvegardés : {out_path}")
