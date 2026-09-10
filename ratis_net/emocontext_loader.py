"""ratis_net.emocontext_loader — Charge EmoContext et mappe → ThermoEnvironment.

Piste 4 : on nourrit RATIS-Net avec de vrais dialogues humains annotés en
émotion (EmoContext, SemEval-2019 Task 3 : 30 160 dialogues 3-tours, 4 labels
happy/sad/angry/others).

Mapping émotion annotée → ThermoEnvironment (le contexte thermo du dialogue) :
  - happy  → ThermoEnvironment.joy()   (cœur modéré, détendu, chaud, excité positif)
  - angry  → ThermoEnvironment.anger() (cœur rapide, tendu, chaud, excité)
  - sad    → ThermoEnvironment.fear()   (cœur rapide, tendu, FROID, excité)
    (sad partage l'arousal/tension de la peur, mais froid = retrait)
  - others → ThermoEnvironment.calm()   (cœur lent, relaxé, neutre)

Chaque mot unique d'un dialogue devient un token ; son étiquette = l'émotion
annotée du dialogue. Le réseau apprend à associer (mot, contexte thermo) →
émotion. C'est la « thermodynamique du langage ».
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

try:
    from ratis_net.eth_thermo_fixer import ThermoEnvironment
except ImportError:
    # exécution en module direct depuis le dossier ratis_net/
    from eth_thermo_fixer import ThermoEnvironment

# mapping label EmoContext → (ThermoEnvironment, c_seuil_cible, label_num)
EMO_MAP = {
    "happy":  (ThermoEnvironment.joy,    0.7, 1),  # joie
    "angry":  (ThermoEnvironment.anger, 0.3, 0),  # colère
    "sad":    (ThermoEnvironment.fear,  0.2, 0),  # tristesse (froid = retrait)
    "others": (ThermoEnvironment.calm, 0.5, 2),  # neutre
}

_TOKEN_RE = re.compile(r"[a-zà-ÿ']+")


def tokenize(text: str) -> list[str]:
    """Tokenise un tour de dialogue : mots en minuscules, ponctuation/emojis hors."""
    return _TOKEN_RE.findall(text.lower())


def load_emocontext(path: str | Path, max_examples: int | None = None) -> list[dict]:
    """Charge EmoContext (train.txt ou dev.txt).

    Format TSV 5 colonnes : id, turn1, turn2, turn3, label.
    Retourne une liste de {turn1, turn2, turn3, label, env, c_seuil, label_num}.
    """
    path = Path(path)
    examples = []
    with open(path, encoding="utf-8") as f:
        next(f, None)  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 5:
                continue
            _, t1, t2, t3, label = parts
            label = label.strip().lower()
            if label not in EMO_MAP:
                continue
            env_cls, c_seuil, label_num = EMO_MAP[label]
            examples.append({
                "turn1": t1, "turn2": t2, "turn3": t3,
                "label": label, "env": env_cls(),
                "c_seuil": c_seuil, "label_num": label_num,
            })
            if max_examples and len(examples) >= max_examples:
                break
    return examples


def build_samples(examples: list[dict], embedding_fn, dim: int = 8,
                  per_word: bool = True) -> list[tuple]:
    """Construit les samples d'entraînement.

    Chaque sample = (token_embedding, env, label_num, c_seuil).
    Si per_word : un sample par mot unique (tous les tours concaténés) par
    dialogue. Sinon : un sample par dialogue (token = tour3, l'émotion cible).

    embedding_fn(word, dim) → np.ndarray (hash, TTF, ou topo signature).
    """
    samples = []
    for ex in examples:
        if per_word:
            words = tokenize(ex["turn1"] + " " + ex["turn2"] + " " + ex["turn3"])
            seen = set()
            for w in words:
                if w in seen or len(w) < 2:
                    continue
                seen.add(w)
                emb = embedding_fn(w, dim)
                samples.append((emb, ex["env"], ex["label_num"], ex["c_seuil"]))
        else:
            emb = embedding_fn(ex["turn3"], dim)
            samples.append((emb, ex["env"], ex["label_num"], ex["c_seuil"]))
    return samples


def build_sequence_samples(examples: list[dict], embedding_fn, dim: int = 8,
                           turn: str = "turn3", min_words: int = 2) -> list[tuple]:
    """Construit des samples d'entraînement par SÉQUENCE (piste 2).

    Chaque sample = un DIALOGUE (pas un mot). Le token = l'embedding de la
    SÉQUENCE entière du tour demandé, obtenu par pool des embeddings de ses
    mots (moyenne pondérée par la norme). L'émotion = l'émotion annotée du
    dialogue. Le réseau apprend alors à classer une SÉQUENCE, pas des mots
    isolés — c'est l'unité d'apprentissage fidèle à LCT (la forme du message,
    pas chaque mot = le courant).

    C'est la clé pour débloquer happy : le classifieur mot-à-mot noie happy
    (minoritaire, 14% du corpus) car les mots neutres sont classés par
    fréquence de classe. En classant la SÉQUENCE, un dialogue happy contient
    une dominante de mots happy → la séquence est classée happy.

    Args:
        turn: tour à utiliser ("turn3" = réponse finale, la plus annotée).
        min_words: ignore les séquences trop courtes (bruit).
    """
    samples = []
    for ex in examples:
        words = tokenize(ex[turn])
        if len(words) < min_words:
            continue
        embs = np.array([embedding_fn(w, dim) for w in words])
        # pool : moyenne pondérée par la norme (les mots saillants pèsent plus)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        seq_emb = (embs * norms).sum(axis=0) / norms.sum()
        n = np.linalg.norm(seq_emb)
        seq_emb = seq_emb / n if n > 1e-9 else seq_emb
        samples.append((seq_emb, ex["env"], ex["label_num"], ex["c_seuil"]))
    return samples


def balance_classes(samples: list[tuple], seed: int = 42) -> list[tuple]:
    """Rééquilibre les classes par undersampling (piste 2).

    Le corpus EmoContext est déséquilibré (others 50%, happy 14%). Un réseau
    entraîné tel quel apprend la classe majoritaire. L'undersampling ramène
    toutes les classes au cardinal de la classe minoritaire — chaque émotion
    pèse autant dans l'apprentissage. La loi LCT (ΔW = η·φ·P_sig·C) est
    inchangée ; on agit sur les données, pas sur la règle.
    """
    from collections import defaultdict
    by_class = defaultdict(list)
    for s in samples:
        by_class[s[2]].append(s)
    n_min = min(len(v) for v in by_class.values())
    rng = np.random.RandomState(seed)
    balanced = []
    for cls, items in by_class.items():
        idx = rng.choice(len(items), size=n_min, replace=False)
        balanced.extend(items[i] for i in idx)
    rng.shuffle(balanced)
    return balanced


def vocabulary(examples: list[dict], min_len: int = 2, top_k: int | None = None) -> list[str]:
    """Extrait le vocabulaire (mots uniques) des exemples, trié par fréquence."""
    from collections import Counter
    c = Counter()
    for ex in examples:
        for w in tokenize(ex["turn1"] + " " + ex["turn2"] + " " + ex["turn3"]):
            if len(w) >= min_len:
                c[w] += 1
    if top_k:
        return [w for w, _ in c.most_common(top_k)]
    return [w for w, _ in c.most_common()]


if __name__ == "__main__":
    import sys
    from pathlib import Path
    _ROOT = Path(__file__).resolve().parents[1]
    train = load_emocontext(_ROOT / "data" / "emocontext" / "train.txt", max_examples=1000)
    print(f"Chargé {len(train)} exemples (sur 1000 max)")
    from collections import Counter
    print("Labels:", dict(Counter(e["label"] for e in train)))
    vocab = vocabulary(train, top_k=20)
    print(f"Top-20 mots: {vocab}")
    print(f"\nExemple : {train[0]}")
    print(f"Tokens turn3 : {tokenize(train[0]['turn3'])}")
