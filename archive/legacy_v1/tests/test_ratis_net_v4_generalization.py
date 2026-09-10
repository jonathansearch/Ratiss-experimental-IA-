"""tests/test_ratis_net_v4_generalization.py — Piste 1 : généralisation.

Mesure la vraie généralisation de v4 sur un vocabulaire plus large. On
sépare les TOKENS (pas les instances) : entraîne sur 80% des mots, teste sur
20% de mots JAMAIS vus. Question : le réseau a-t-il appris une RÈGLE
(contexte → label) ou mémorisé des tokens ?

Hypothèses :
  - Embedding orthogonal (hash → N(0,1)) : un token non vu est orthogonal
    aux tokens vus. Si le réseau généralise quand même, c'est qu'il a appris
    la règle via le vecteur d'environnement.
  - Embedding structuré (char-n-grammes) : les mots proches en sens ont des
    vecteurs proches → la généralisation devrait être meilleure.

La règle d'étiquetage : le label dépend du CONTEXTE (émotion), pas du token.
Colère→0, joie→1, calme→2, peur→0. Le même mot dans 4 contextes donne 4 labels
potentiels. Donc mémoriser un token ne suffit pas : il faut apprendre le
couple (token, env) → label, et idéalement la règle env → label.
"""
import hashlib
from pathlib import Path
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from ratis_net.ratis_net_v4 import RatisNetV4
from ratis_net.eth_thermo_fixer import ThermoEnvironment


def hash_embedding(word: str, dim: int = 8, seed: int = 42) -> np.ndarray:
    """Embedding orthogonal : hash → N(0,1). Deux mots sont décorrélés."""
    h = hashlib.sha256(word.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:4], "big") + seed)
    return rng.normal(0, 1, dim)


def structured_embedding(word: str, dim: int = 8, seed: int = 42, _bases={}) -> np.ndarray:
    """Embedding structuré : les mots qui partagent des n-grammes ont des
    vecteurs proches (approche type char-n-gram). Deux mots proches en sens
    sont proches dans l'espace."""
    rng = np.random.default_rng(seed)
    vec = np.zeros(dim)
    grams = ["#" + word[0]] + [word[i:i+2] for i in range(len(word)-1)] + [word[-1] + "#"]
    for g in grams:
        if g not in _bases:
            _bases[g] = rng.normal(0, 1, dim)
        vec += _bases[g]
    n = np.linalg.norm(vec)
    return vec / n if n > 1e-9 else rng.normal(0, 1, dim)


def build_vocab(n_words: int) -> list[str]:
    words = [
        "bonjour", "merci", "salut", "coucou", "bonsoir", "adieu", "hello",
        "pardon", "bravo", "super", "genial", "ok", "oui", "non", "peutetre",
        "toujours", "jamais", "vite", "lent", "fort", "doux", "chaud",
        "froid", "grand", "petit", "beau", "laid", "vrai", "faux", "neuf",
        "vieux", "riche", "pauvre", "joyeux", "triste", "fou", "sage",
        "blanc", "noir", "rouge", "vert", "bleu", "jaune", "gris", "rose",
    ]
    return words[:n_words]


def rule_label(env: ThermoEnvironment) -> tuple[int, float]:
    """Le label dépend du contexte (émotion), pas du token."""
    v = env.to_vector()  # [hr/120, tension, warmth, arousal]
    if v[1] > 0.7 and v[3] > 0.7 and v[2] > 0.5:
        return 0, 0.3   # colère
    if v[1] > 0.7 and v[3] > 0.7 and v[2] < 0.5:
        return 0, 0.2   # peur
    if v[1] < 0.4 and v[2] > 0.5:
        return 1, 0.7   # joie
    return 2, 0.5       # calme


def run_experiment(embedding_fn, name, n_words=30, n_train_words=24,
                   reps=10, epochs=30, verbose=True):
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
        if gen > 0.55:
            verdict = "GÉNÉRALISATION"
        elif gen > 0.40:
            verdict = "GÉNÉRALISATION PARTIELLE"
        else:
            verdict = "PAS DE GÉNÉRALISATION"
        print(f"  → {verdict}")
    return float(np.mean(accs_seen)), float(np.mean(accs_unseen)), verdict


def main():
    print("=" * 72)
    print("RATIS-Net v4 — Piste 1 : généralisation sur vocabulaire large")
    print("Question : le réseau a-t-il appris une RÈGLE ou mémorisé des tokens ?")
    print("=" * 72)

    seen_o, unseen_o, v_o = run_experiment(hash_embedding, "Embedding ORTHOGONAL (hash→N(0,1))")
    seen_s, unseen_s, v_s = run_experiment(structured_embedding, "Embedding STRUCTURÉ (char-n-grammes)")

    print(f"\n{'='*72}")
    print(f"BILAN")
    print(f"{'='*72}")
    print(f"  Embedding orthogonal : vu={seen_o:.3f}, non vu={unseen_o:.3f} ({v_o})")
    print(f"  Embedding structuré : vu={seen_s:.3f}, non vu={unseen_s:.3f} ({v_s})")
    generalizes = unseen_o > 0.55 or unseen_s > 0.55
    print(f"\n  Le réseau généralise-t-il à des tokens non vus ? {'OUI' if generalizes else 'NON'}")
    if generalizes:
        print(f"  → Le réseau a appris la RÈGLE (contexte → label), pas mémorisé des tokens.")
        print(f"  → La brique apprentissage LCT est solide : elle transfère à du non-vu.")

    return {
        "orthogonal_seen": seen_o, "orthogonal_unseen": unseen_o, "orthogonal_verdict": v_o,
        "structured_seen": seen_s, "structured_unseen": unseen_s, "structured_verdict": v_s,
        "generalizes": generalizes,
    }


if __name__ == "__main__":
    out = main()
    out_path = _ROOT / "proofs" / "ratis_net_v4_generalization_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nRésultats sauvegardés : {out_path}")
