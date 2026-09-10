"""tests/test_ratis_net_v4_topo_tokenizer.py — Piste 3 : tokenizer topologique.

Au lieu d'un hash, le token est défini par sa SIGNATURE topologique (cycles
H1 persistants : betti, densité de cycles, persistance max/moyenne/médiane/
std/skew). C'est l'identité topologique de la donnée, invariante sous l'énergie.

On compare 3 tokenizers pour la généralisation à des tokens non vus :
  - hash (piste 1)
  - TTF/MCB (piste 2)
  - topo signature (piste 3)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
# IMPORTANT : _ROOT (ce repo, qui a ratis_net_v4) doit être AVANT AEON, car
# AEON contient aussi un dossier ratis_net/ (vieille copie v1) qui cacherait
# le nôtre. On insert AEON en dernier, _ROOT en tête.
sys.path.append(str(_ROOT.parent / "RATISS-ODV-AEON"))
sys.path.insert(0, str(_ROOT))

from ratis_net.ratis_net_v4 import RatisNetV4
from ratis_net.eth_thermo_fixer import ThermoEnvironment
from ratis_net.ttf_bridge import ttf_embedding, is_ttf_available, _hash_embedding
from ratis_net.topo_tokenizer import topo_signature, is_full_persistence_available


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
                   reps=5, epochs=20, dim=8, verbose=True):
    rng_state = np.random.RandomState(42)
    accs_seen, accs_unseen = [], []
    words = build_vocab(n_words)
    envs = [ThermoEnvironment.anger(), ThermoEnvironment.joy(),
            ThermoEnvironment.calm(), ThermoEnvironment.fear()]

    for rep in range(reps):
        perm = rng_state.permutation(len(words))
        train_words = [words[i] for i in perm[:n_train_words]]
        test_words = [words[i] for i in perm[n_train_words:]]
        train_embs = {w: embedding_fn(w, dim) for w in train_words}
        test_embs = {w: embedding_fn(w, dim) for w in test_words}

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
    return float(np.mean(accs_seen)), float(np.mean(accs_unseen))


def main():
    print("=" * 72)
    print("Piste 3 — Tokenizer topologique (cycles H1 persistants)")
    print("Le token = signature topologique de la donnée (invariante sous énergie).")
    print("=" * 72)
    print(f"\n  Persistance complète disponible ? {is_full_persistence_available()}")
    print(f"  Cerveau TTF connecté ? {is_ttf_available()}")

    print(f"\n{'='*72}")
    print(f"COMPARAISON 3 tokenizers (généralisation à tokens non vus)")
    print(f"{'='*72}")
    h_emb = lambda w, d: _hash_embedding(w, d)
    s_h, u_h = run_experiment(h_emb, "HASH (piste 1)", dim=8)

    ttf_ok = is_ttf_available()
    if ttf_ok:
        s_t, u_t = run_experiment(lambda w, d: ttf_embedding(w, d), "TTF/MCB (piste 2)", dim=8)
    else:
        s_t, u_t = None, None

    s_g, u_g = run_experiment(lambda w, d: topo_signature(w, d), "TOPO signature (piste 3)", dim=10)

    print(f"\n{'='*72}")
    print(f"BILAN PISTE 3")
    print(f"{'='*72}")
    print(f"  HASH (piste 1)            : vu={s_h:.3f}, non vu={u_h:.3f}")
    if ttf_ok:
        print(f"  TTF/MCB (piste 2)         : vu={s_t:.3f}, non vu={u_t:.3f}")
    print(f"  TOPO signature (piste 3)  : vu={s_g:.3f}, non vu={u_g:.3f}")

    best_unseen = max(u_h, u_g, u_t if ttf_ok else -1)
    winner = "TOPO signature" if u_g == best_unseen else ("TTF/MCB" if ttf_ok and u_t == best_unseen else "HASH")
    print(f"\n  Meilleure généralisation (non vu) : {winner} ({best_unseen:.3f})")
    print(f"\n  → Le tokenizer topologique définit le token par sa TOPOLOGIE (cycles H1),")
    print(f"    pas par un hash. C'est l'identité certifiable (invariante sous énergie, loi LCT).")

    return {
        "full_persistence": is_full_persistence_available(),
        "ttf_connected": ttf_ok,
        "hash_seen": s_h, "hash_unseen": u_h,
        "ttf_seen": s_t, "ttf_unseen": u_t,
        "topo_seen": s_g, "topo_unseen": u_g,
        "best_unseen": best_unseen, "winner": winner,
    }


if __name__ == "__main__":
    out = main()
    out_path = _ROOT / "proofs" / "ratis_net_v4_topo_tokenizer_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nRésultats sauvegardés : {out_path}")
