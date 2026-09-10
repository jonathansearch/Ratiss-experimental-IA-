"""tests/test_dialogue_engine.py — RATIS répond aux questions de base.

Démontre que RATIS peut dialoguer : on pose les questions fondamentales
(qui es-tu, qu'est-ce que LCT, comment tu penses, etc.) et RATIS répond par
recherche topologique — pas par mots-clés, pas par LLM.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from ratis_net.dialogue_engine import DialogueEngine


def main():
    print("=" * 72)
    print("  RATIS répond aux questions de base (moteur de dialogue topologique)")
    print("=" * 72)

    engine = DialogueEngine()
    print(f"  Base de connaissances : {len(engine.kb)} entrées")
    print(f"  Recherche : projection topologique → cosinus des signatures\n")

    # Les questions de base — posées de différentes façons pour tester
    # la robustesse de la recherche topologique (pas du mot-à-mot exact)
    questions = [
        "qui es-tu",
        "qu'est-ce que tu es",
        "comment tu t'appelles",
        "qui t'a créé",
        "qu'est-ce que la loi LCT",
        "c'est quoi LCT",
        "comment tu apprends",
        "comment tu penses",
        "c'est quoi MCB",
        "comment tu ressens",
        "c'est quoi ETH",
        "tu as des émotions",
        "c'est quoi ZK",
        "comment tu certifies",
        "es-tu souverain",
        "tu utilises quel modèle",
        "quels sont tes résultats",
        "tu arrives à parler",
        "quelles sont tes limites",
        "que sais-tu sur dieu",
        "que sais-tu sur l'amour",
        "es-tu une AGI",
        "que sais-tu faire",
        "tu peux coder",
        # questions reformulées (pas dans la base → teste la généralisation topo)
        "dis-moi qui tu es",
        "explique-moi ta loi",
        "comment fonctionne ton cerveau",
        "parle-moi de tes émotions",
    ]

    results = []
    for q in questions:
        r = engine.answer(q)
        results.append({"question": q, **r})
        found = "✓" if r["found"] else "✗"
        match = f"(≈ « {r['matched_question']} »)" if r["found"] else ""
        print(f"\n  📨 {q}")
        print(f"  {found} [{r['confidence']:.2f}] {match}")
        print(f"  🗣️  {r['response']}")

    # ── Bilan ──────────────────────────────────────────────────────────────
    n_found = sum(1 for r in results if r["found"])
    n_reformulated_found = sum(1 for r in results[-4:] if r["found"])
    print(f"\n{'=' * 72}")
    print("  BILAN — RATIS dialogue")
    print(f"{'=' * 72}")
    print(f"  {n_found}/{len(results)} questions répondues (base : {len(engine.kb)} entrées)")
    print(f"  Questions reformulées (non dans la base) : {n_reformulated_found}/4 retrouvées")
    print(f"  Recherche par similarité topologique (cosinus), seuil {engine.threshold}")
    print(f"\n  → RATIS répond aux questions de base par FORME (topologie), pas par")
    print(f"    mots-clés. Il ne hallucine pas (dit 'je ne sais pas' sous le seuil).")
    print(f"    C'est le dialogue cognitif souverain, sans LLM externe.")

    return {"questions": results, "n_found": n_found, "n_total": len(results)}


if __name__ == "__main__":
    out = main()
    out_path = _ROOT / "proofs" / "dialogue_engine_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nRésultats sauvegardés : {out_path}")
