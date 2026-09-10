"""tests/test_ratis_net_v4.py — Preuve de concept v4 : le fixeur thermodynamique.

On teste l'architecture v4 : ETH apprend le DIFFERENTIEL thermo (pas la valeur
fixe). On utilise l'exemple "bonjour colère" vs "bonjour joie".

On vérifie :
  1. ETH apprend-il des C_seuil DIFFERENTS pour colère vs joie ?
  2. Le différentiel émotionnel émerge-t-il (ΔC_seuil ≠ 0) ?
  3. Le réseau LCT apprend-il (accuracy) ?
  4. Les effondrements produisent-ils des MARQUES différentes (colère vs joie) ?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from ratis_net.ratis_net_v4 import RatisNetV4
from ratis_net.eth_thermo_fixer import ThermoEnvironment


def make_token_embedding(word: str, dim: int = 8, seed: int = 42) -> np.ndarray:
    """Embedding simple d'un mot (hash → vecteur)."""
    import hashlib
    h = hashlib.sha256(word.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:4], "big") + seed)
    return rng.normal(0, 1, dim)


def main():
    print("=" * 72)
    print("RATIS-Net v4 — Le fixeur thermodynamique ETH")
    print("ΔW = η·φ·P_sig·C (LCT) + ETH = f(token, env) (différentiel thermo)")
    print("On garde la MARQUE topo après collapse, pas la valeur d'énergie.")
    print("=" * 72)

    # dataset : "bonjour" en colère (label 0) vs joie (label 1)
    # C_seuil cibles : colère = 0.3 (effondrement rapide), joie = 0.7 (lent)
    token_bonjour = make_token_embedding("bonjour", dim=8)
    token_merci = make_token_embedding("merci", dim=8)

    samples = []
    for _ in range(20):
        samples.append((token_bonjour, ThermoEnvironment.anger(), 0, 0.3))  # colère → seuil bas
        samples.append((token_bonjour, ThermoEnvironment.joy(), 1, 0.7))   # joie → seuil haut
        samples.append((token_merci, ThermoEnvironment.calm(), 2, 0.5))    # calme → seuil moyen
        samples.append((token_merci, ThermoEnvironment.fear(), 0, 0.2))     # peur → seuil très bas

    print(f"\nDataset : 4 contextes (colère, joie, calme, peur) × 2 mots (bonjour, merci)")
    print(f"  colère → C_seuil=0.3 (effondrement rapide, marque agressive)")
    print(f"  joie   → C_seuil=0.7 (effondrement lent, marque ouverte)")
    print(f"  calme  → C_seuil=0.5 (neutre)")
    print(f"  peur   → C_seuil=0.2 (effondrement très rapide)")

    net = RatisNetV4(n_in=12, n_hidden=10, n_out=3, token_dim=8, env_dim=4, eta=0.1, seed=42)
    print(f"\nRATIS-Net v4 : 12(=8 token ⊕ 4 env)→10→3 + ETH(token_dim=8, env_dim=4)")
    print(f"  entrée réseau = token ⊕ environnement (le label dépend du couple)")

    print(f"\nEntraînement (30 epochs) :")
    results = net.train(samples, epochs=30, lr_eth=0.1, verbose=True)

    # vérifications
    print(f"\n{'='*72}")
    print(f"VALIDATION v4")
    print(f"{'='*72}")

    # 1. ETH a-t-il appris des C_seuil différents ?
    c_anger = net.eth.predict_c_seuil(token_bonjour, ThermoEnvironment.anger())
    c_joy = net.eth.predict_c_seuil(token_bonjour, ThermoEnvironment.joy())
    c_calm = net.eth.predict_c_seuil(token_bonjour, ThermoEnvironment.calm())
    c_fear = net.eth.predict_c_seuil(token_bonjour, ThermoEnvironment.fear())

    print(f"\n  C_seuil 'bonjour' par contexte :")
    print(f"    colère = {c_anger:.3f}  (cible 0.3)")
    print(f"    joie   = {c_joy:.3f}  (cible 0.7)")
    print(f"    calme  = {c_calm:.3f}  (cible 0.5)")
    print(f"    peur   = {c_fear:.3f}  (cible 0.2)")

    # 2. différentiel émotionnel
    diff_anger_joy = net.emotional_differential(token_bonjour,
                                                  ThermoEnvironment.anger(),
                                                  ThermoEnvironment.joy())
    print(f"\n  Différentiel émotionnel 'bonjour' (colère - joie) = {diff_anger_joy:.4f}")
    print(f"  -> {'Emotion EMERGE' if abs(diff_anger_joy) > 0.05 else 'Pas de differentiel'} : "
          f"le meme mot a des seuils thermo differents selon le contexte.")

    # 3. accuracy
    final_acc = results["acc_history"][-1]
    print(f"\n  Accuracy finale = {final_acc:.3f}")

    # 4. marques topo différentes (colère vs joie)
    r_anger = net.forward(token_bonjour, ThermoEnvironment.anger(), t_step=0)
    r_joy = net.forward(token_bonjour, ThermoEnvironment.joy(), t_step=0)
    mark_anger = r_anger.get("mark")
    mark_joy = r_joy.get("mark")
    print(f"\n  Marque topo 'bonjour' colère = {mark_anger}")
    print(f"  Marque topo 'bonjour' joie   = {mark_joy}")
    marks_differ = mark_anger != mark_joy
    print(f"  Marques DIFFÉRENTES ? : {marks_differ}  "
          f"-> marques contextuelles if marks_differ else meme marque")

    # verdict
    eth_learns = abs(c_anger - c_joy) > 0.05
    emotion_emerges = abs(diff_anger_joy) > 0.05

    print(f"\n  ETH apprend des seuils contextuels ? : {'OUI' if eth_learns else 'NON'}")
    print(f"  Emotion emerge (differentiel) ?    : {'OUI' if emotion_emerges else 'NON'}")
    print(f"  Le réseau LCT apprend (accuracy) ?    : {'OUI' if final_acc > 0.5 else 'PARTIAL'}")
    print(f"  Marques topo contextuelles ?         : {'OUI' if marks_differ else 'NON'}")

    if eth_learns and emotion_emerges:
        print(f"\n  → PREUVE v4 : le fixeur thermodynamique apprend le différentiel thermo.")
        print(f"    Le meme mot (bonjour) a des seuils d effondrement differents selon")
        print(f"    l'environnement (colère vs joie). L'émotion ÉMERGE comme différentiel.")
        print(f"    L'effondrement garde la MARQUE topo, pas la valeur.")
        verdict = "PASS"
    elif eth_learns:
        print(f"\n  → ETH apprend les seuils. Émotion partiellement émergente.")
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    return {
        "c_seuil_anger": c_anger, "c_seuil_joy": c_joy,
        "c_seuil_calm": c_calm, "c_seuil_fear": c_fear,
        "emotional_differential_anger_joy": diff_anger_joy,
        "eth_learns": eth_learns, "emotion_emerges": emotion_emerges,
        "marks_differ": marks_differ,
        "mark_anger": mark_anger, "mark_joy": mark_joy,
        "final_accuracy": final_acc,
        "acc_history": [round(float(a), 4) for a in results["acc_history"]],
        "total_collapses": results["total_collapses"],
        "verdict": verdict,
    }


if __name__ == "__main__":
    out = main()
    out_path = _ROOT / "proofs" / "ratis_net_v4_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nRésultats sauvegardés : {out_path}")
