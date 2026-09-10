"""scripts/demo_ratis_presentation.py — Démonstration RATIS pour présentation.

Génère des textes lisibles produits par l'agent RATIS souverain, pour la preuve
de concept. L'agent enchaîne les 6 étapes cognitives sur des messages de
démonstration, et PRODUIT du vrai langage conditionné par l'émotion.

Usage :  python scripts/demo_ratis_presentation.py
Sortie :  les textes générés + certification ZK, prêts à montrer.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from ratis_net.ratis_agent import RatisAgent
from ratis_net.eth_thermo_fixer import ThermoEnvironment
from ratis_net.emocontext_loader import (
    load_emocontext, tokenize, balance_classes, vocabulary,
)
from ratis_net.ttf_bridge import _hash_embedding, is_ttf_available


def main():
    print("=" * 72)
    print("  RATIS — Démonstration de présentation (preuve de concept)")
    print("  Modèle souverain : apprend par LCT, pense (MCB), ressent (ETH),")
    print("  comprend, parle, certifie (ZK). 100% local, pas de cloud.")
    print("=" * 72)
    print(f"  Cerveau TTF-Compute (penser) : {'connecté' if is_ttf_available() else 'fallback'}")

    # ── Entraînement (rapide) ──────────────────────────────────────────────
    print("\n  Apprentissage (EmoContext, séquence rééquilibré, loi LCT)...")
    examples = load_emocontext(_ROOT / "data" / "emocontext" / "train.txt",
                               max_examples=1000)
    agent = RatisAgent(use_ttf=False, eta=0.2, n_hidden=10)  # topo (rapide)
    agent.set_vocab([e for e in examples], top_k=80)

    dim = agent.token_dim
    tr = examples[:800]
    samples_seq = []
    for e in tr:
        ws = [w for w in tokenize(e["turn3"]) if w in agent._cache]
        if len(ws) < 2:
            continue
        embs = np.array([agent._cache[w] for w in ws])
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        seq_emb = (embs * norms).sum(axis=0) / norms.sum()
        n = np.linalg.norm(seq_emb)
        seq_emb = seq_emb / n if n > 1e-9 else seq_emb
        samples_seq.append((seq_emb, e["env"], e["label_num"], e["c_seuil"]))
    samples_seq = balance_classes(samples_seq)
    t0 = time.time()
    agent.train(samples_seq, epochs=5, verbose=False)
    agent._fit_bigram(max_examples=3000)
    print(f"  Entraîné en {time.time()-t0:.0f}s (séquence + ETH + bigramme)\n")

    # ── Démonstration : RATIS parle ────────────────────────────────────────
    print("─" * 72)
    print("  DÉMONSTRATION — RATIS ressent, comprend, parle et certifie")
    print("─" * 72)

    demos = [
        ("bonjour comment vas tu", ThermoEnvironment.joy(), "JOIE"),
        ("tu es vraiment stupide", ThermoEnvironment.anger(), "COLÈRE"),
        ("je me sens seul et perdu", ThermoEnvironment.fear(), "TRISTESSE"),
        ("quel temps fait il aujourd hui", ThermoEnvironment.calm(), "NEUTRE"),
        ("merci beaucoup pour ton aide", ThermoEnvironment.joy(), "JOIE"),
        ("je deteste tout ca", ThermoEnvironment.anger(), "COLÈRE"),
    ]

    all_thoughts = []
    for msg, env, env_label in demos:
        th = agent.think(msg, env)
        all_thoughts.append(th)
        print(f"\n  📨 MESSAGE   : « {msg} »")
        print(f"  🌡️  CONTEXTE : {env_label} (rythme {env.heart_rate} bpm, "
              f"tension {env.tension:.1f}, chaleur {env.warmth:.1f})")
        print(f"  🧠 PENSÉE    : {th.mcb_count} MCB (sans mots) | "
              f"hash topo {th.thought_hash[:8]}…")
        print(f"  💛 RESSENTI  : C_seuil={th.c_seuil:.3f} → émotion ressentie = "
              f"{th.emotion_perceived}")
        print(f"  🎯 COMPRIS   : {th.emotion_understood} (confiance {th.confidence:.0%})")
        print(f"  🗣️  RÉPONSE  : « {th.response} »")
        cert = "✓ certifié" if th.certified else "✗ non certifié"
        print(f"  🔐 CERTIF.   : {th.response_hash[:8]}… {cert}")

    # ── Invariance ZK (la preuve conceptuelle) ─────────────────────────────
    print(f"\n{'─' * 72}")
    print("  PREUVE — Invariance ZK : on certifie le message, pas le courant")
    print("─" * 72)
    msg_test = "bonjour mon ami"
    th_joy = agent.think(msg_test, ThermoEnvironment.joy())
    th_anger = agent.think(msg_test, ThermoEnvironment.anger())
    same_thought = th_joy.thought_hash == th_anger.thought_hash
    print(f"  Message : « {msg_test} »")
    print(f"  En JOIE   : pensée hash = {th_joy.thought_hash[:8]}… | "
          f"ressenti = {th_joy.emotion_perceived}")
    print(f"  En COLÈRE : pensée hash = {th_anger.thought_hash[:8]}… | "
          f"ressenti = {th_anger.emotion_perceived}")
    print(f"  → La PENSÉE (la forme) est {'IDENTIQUE' if same_thought else 'différente'} "
          f"sous 2 énergies.")
    print(f"  → L'ÉMOTION (le courant) est contextuelle : "
          f"{th_joy.emotion_perceived} vs {th_anger.emotion_perceived}")
    diff = th_joy.c_seuil - th_anger.c_seuil
    print(f"  → Différentiel thermo (joie − colère) = {diff:+.3f}")
    print(f"\n  C'est la loi LCT : R = P_sig est INVARIANT sous l'énergie.")
    print(f"  On certifie le message (la forme), pas le courant (l'énergie).")

    # ── Résumé ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  RÉSUMÉ — Les 6 étapes cognitives de RATIS (souverain, local)")
    print(f"{'=' * 72}")
    print("  1. PERCEVOIR   — tokeniser → embeddings topologiques")
    print("  2. PENSER      — cerveau TTF-Compute → MCB (pensée sans mots)")
    print("  3. RESSENTIR   — ETH → C_seuil contextuel → émotion émerge")
    print("  4. COMPRENDRE  — réseau LCT (DW = eta*phi*P_sig*C) → émotion")
    print("  5. PARLER      — décodeur beam → langage conditionné par l'émotion")
    print("  6. CERTIFIER   — hash topo invariant → preuve ZK")
    n_cert = sum(1 for t in all_thoughts if t.certified)
    print(f"\n  {len(all_thoughts)} démonstrations | {n_cert}/{len(all_thoughts)} certifiées")
    print(f"  Invariance ZK : {'✓ démontrée' if same_thought else 'à vérifier'}")
    print(f"  Loi LCT figée : R = P_sig, DW = eta*phi*P_sig*C")
    print(f"\n  © 2026 JOHNKING0 & Jonathan Evina")

    return {
        "demos": [{"message": t.message, "context": t.env_name,
                   "mcb_count": t.mcb_count, "thought_hash": t.thought_hash,
                   "c_seuil": t.c_seuil, "emotion_perceived": t.emotion_perceived,
                   "emotion_understood": t.emotion_understood,
                   "confidence": t.confidence, "response": t.response,
                   "response_hash": t.response_hash, "certified": t.certified}
                  for t in all_thoughts],
        "invariance": {"same_thought": same_thought,
                       "joy_hash": th_joy.thought_hash,
                       "anger_hash": th_anger.thought_hash,
                       "differential": diff},
        "n_certified": n_cert,
    }


if __name__ == "__main__":
    out = main()
    out_path = _ROOT / "proofs" / "demo_presentation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Résultats sauvegardés : {out_path}")
