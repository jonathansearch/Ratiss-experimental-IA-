"""tests/test_ratis_net_v4_ttf_bridge.py — Piste 2 : RATIS-Net ← cerveau TTF.

Au lieu d'un embedding arbitraire (hash), le réseau est nourri par les MCB du
cerveau TTF-Compute : le réseau « pense » avec la topologie réelle du mot.

On vérifie :
  1. Le bridge se connecte-t-il au cerveau TTF (dépôt AEON présent) ?
  2. Les embeddings TTF sont-ils distincts par mot ?
  3. RATIS-Net alimenté par TTF apprend-il (accuracy) ?
  4. Généralise-t-il à des tokens non vus (split TOKENS) ?

Compare avec l'embedding hash (piste 1) pour isoler l'apport de TTF.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from ratis_net.ratis_net_v4 import RatisNetV4
from ratis_net.eth_thermo_fixer import ThermoEnvironment
from ratis_net.ttf_bridge import ttf_embedding, is_ttf_available, _hash_embedding


def build_vocab(n_words):
    words = [
        "bonjour", "merci", "salut", "coucou", "bonsoir", "adieu", "hello",
        "pardon", "bravo", "super", "genial", "ok", "oui", "non", "peutetre",
        "toujours", "jamais", "vite", "lent", "fort", "doux", "chaud",
        "froid", "grand", "petit", "beau", "laid", "vrai", "faux", "neuf",
        "vieux", "riche", "pauvre", "joyeux", "triste", "fou", "sage",
        "blanc", "noir", "rouge", "vert", "bleu", "jaune", "gris", "rose",
    ]
    return words[:n_words]


def rule_label(env):
    v = env.to_vector()
    if v[1] > 0.7 and v[3] > 0.7 and v[2] > 0.5:
        return 0, 0.3
    if v[1] > 0.7 and v[3] > 0.7 and v[2] < 0.5:
        return 0, 0.2
    if v[1] < 0.4 and v[2] > 0.5:
        return 1, 0.7
    return 2, 0.5


def run_experiment(embedding_fn, name, n_words=30, n_train_words=24,
                   reps=8, epochs=30, verbose=True):
    rng_state = np.random.RandomState(42)
    accs_seen, accs_unseen = [], []
    words = build_vocab(n_words)
    envs = [ThermoEnvironment.anger(), ThermoEnvironment.joy(),
            ThermoEnvironment.calm(), ThermoEnvironment.fear()]

    for rep in range(reps):
        perm = rng_state.permutation(len(words))
        train_words = [words[i] for i in perm[:n_train_words]]
        test_words = [words[i] for i in perm[n_train_words:]]
        train_embs = {w: embedding_fn(w) for w in train_words}
        test_embs = {w: embedding_fn(w) for w in test_words}

        samples = []
        for w in train_words:
            for env in envs:
                label, c_seuil = rule_label(env)
                samples.append((train_embs[w], env, label, c_seuil))

        net = RatisNetV4(n_in=12, n_hidden=20, n_out=3, token_dim=8, env_dim=4,
                         eta=0.1, seed=42 + rep)
        for ep in range(epochs):
            for tok, env, label, cs in samples:
                net.train_step(tok, env, label, cs, t_step=ep, lr_eth=0.1)

        def predict(net, tok, env):
            x = net._build_input(tok, env)
            h = np.array([n.forward(x, 0) for n in net.hidden])
            out = np.array([n.forward(h, 0) for n in net.output])
            return int(np.argmax(out))

        cs_ = sum(1 for w in train_words for env in envs
                  if predict(net, train_embs[w], env) == rule_label(env)[0])
        accs_seen.append(cs_ / (len(train_words) * len(envs)))
        cu_ = sum(1 for w in test_words for env in envs
                  if predict(net, test_embs[w], env) == rule_label(env)[0])
        accs_unseen.append(cu_ / (len(test_words) * len(envs)))

    if verbose:
        print(f"\n=== {name} ({n_words} tokens, {n_train_words} train / {n_words-n_train_words} unseen) ===")
        print(f"  tokens VUS    : acc = {np.mean(accs_seen):.3f} ± {np.std(accs_seen):.3f}")
        print(f"  tokens NON VUS: acc = {np.mean(accs_unseen):.3f} ± {np.std(accs_unseen):.3f}")
        print(f"  hasard (3 classes) = 0.333 | classe majoritaire = 0.500")
        gen = np.mean(accs_unseen)
        verdict = "GÉNÉRALISATION" if gen > 0.55 else ("PARTIELLE" if gen > 0.40 else "PAS DE GÉNÉRALISATION")
        print(f"  → {verdict}")
    return float(np.mean(accs_seen)), float(np.mean(accs_unseen))


def main():
    print("=" * 72)
    print("Piste 2 — RATIS-Net nourri par le cerveau TTF-Compute (MCB)")
    print("Le réseau pense avec la topologie réelle du mot, pas un hash.")
    print("=" * 72)

    connected = is_ttf_available()
    print(f"\nCerveau TTF-Compute connecté ? : {'OUI' if connected else 'NON (fallback hash)'}")

    # 1. embeddings distincts ?
    if connected:
        print("\n  Embeddings TTF par mot :")
        words = ["bonjour", "salut", "bonsoir", "xyz"]
        embs = {w: ttf_embedding(w, dim=8) for w in words}
        distinct = len(set(tuple(np.round(e, 4)) for e in embs.values()))
        print(f"    {len(words)} mots → {distinct} embeddings distincts")
        def cos(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        print(f"    cos(bonjour, bonsoir) = {cos(embs['bonjour'], embs['bonsoir']):.3f} (partagent b,o,n)")

    # 2. apprentissage + généralisation : hash vs TTF
    print(f"\n{'='*72}")
    print(f"COMPARAISON : hash (piste 1) vs TTF (piste 2)")
    print(f"{'='*72}")
    seen_h, unseen_h = run_experiment(lambda w: _hash_embedding(w, 8), "Embedding HASH (piste 1)",
                                       reps=5, epochs=20)
    if connected:
        seen_t, unseen_t = run_experiment(lambda w: ttf_embedding(w, 8), "Embedding TTF (piste 2)",
                                           reps=5, epochs=20)
    else:
        seen_t, unseen_t = None, None

    print(f"\n{'='*72}")
    print(f"BILAN PISTE 2")
    print(f"{'='*72}")
    print(f"  HASH : vu={seen_h:.3f}, non vu={unseen_h:.3f}")
    if connected:
        print(f"  TTF  : vu={seen_t:.3f}, non vu={unseen_t:.3f}")
        delta_unseen = unseen_t - unseen_h
        print(f"\n  Δ généralisation (TTF − HASH) = {delta_unseen:+.3f}")
        if delta_unseen > 0.02:
            print(f"  → Le cerveau TTF AMÉLIORE la généralisation : la topologie aide le réseau.")
        elif delta_unseen > -0.02:
            print(f"  → Le cerveau TTF est ÉQUIVALENT au hash pour la généralisation.")
            print(f"    (le réseau apprend la règle contexte→label via l'env, l'embedding du token importe peu)")
        else:
            print(f"  → Le cerveau TTF est moins bon ici (limite du pooling MCB → embedding).")
        print(f"\n  Le réseau « pense » maintenant avec la topologie réelle du cerveau TTF.")
    else:
        print(f"  (TTF non disponible — dépôt AEON absent — test non exécuté)")

    return {
        "ttf_connected": connected,
        "hash_seen": seen_h, "hash_unseen": unseen_h,
        "ttf_seen": seen_t, "ttf_unseen": unseen_t,
    }


if __name__ == "__main__":
    out = main()
    out_path = _ROOT / "proofs" / "ratis_net_v4_ttf_bridge_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nRésultats sauvegardés : {out_path}")
