"""ratis_net.pipeline — Pipeline branchable (connecteurs).

Découple le pipeline d'entraînement en 4 interfaces claires pour qu'un
partenaire puisse brancher sa source de données, son tokenizer, ou son
backend d'exécution (CPU/GPU) SANS toucher au cœur RATIS-Net.

    [DataSource] → [Tokenizer] → [Learner (RATIS-Net LCT)] → [Pipeline.run]
       (corpus)      (embeddings)    (apprentissage LCT)        (orchestre)

Interfaces :
  - DataSource : fournit des exemples étiquetés (texte + émotion + env thermo)
  - Tokenizer  : transforme un mot en embedding (hash / topo / TTF)
  - Learner    : entraîne et prédit (RATIS-Net v4)
  - Pipeline   : orchestre les 3 + évalue + observe l'émergence

Un partenaire crée :
    p = Pipeline(EmoContextDataSource(), TopoTokenizer(), RatisNetV4Learner())
    report = p.run(n_dialogues=300, epochs=8)
et ne connaît que ces 3 noms. Le cœur (LCT, ETH, collapse) reste encapsulé.
"""
from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

try:
    from ratis_net.ratis_net_v4 import RatisNetV4
    from ratis_net.eth_thermo_fixer import ThermoEnvironment
    from ratis_net.emocontext_loader import (
        load_emocontext, build_samples, vocabulary, tokenize, EMO_MAP,
    )
    from ratis_net.ttf_bridge import _hash_embedding, ttf_embedding, is_ttf_available
    from ratis_net.topo_tokenizer import (
        topo_signature, is_full_persistence_available, active_backend,
    )
except ImportError:
    # exécution en module direct
    from ratis_net_v4 import RatisNetV4
    from eth_thermo_fixer import ThermoEnvironment
    from emocontext_loader import (
        load_emocontext, build_samples, vocabulary, tokenize, EMO_MAP,
    )
    from ttf_bridge import _hash_embedding, ttf_embedding, is_ttf_available
    from topo_tokenizer import (
        topo_signature, is_full_persistence_available, active_backend,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Types partagés
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Example:
    """Un exemple étiqueté : un dialogue (3 tours) + son émotion."""
    turn1: str
    turn2: str
    turn3: str
    label: str              # "happy" / "angry" / "sad" / "others" / ...
    env: ThermoEnvironment  # contexte thermo du dialogue
    c_seuil: float          # seuil cible pour ETH
    label_num: int          # étiquette numérique (pour le réseau)


@dataclass
class TrainReport:
    """Rapport d'entraînement."""
    n_dialogues: int
    labels: dict
    tokenizer_name: str
    backend: str
    n_samples: int
    acc_train: float
    acc_test_vote: float
    c_seuils: dict = field(default_factory=dict)  # émergence : mot → {emotion: c_seuil}
    n_epochs: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Interface 1 : DataSource (la source de données)
# ─────────────────────────────────────────────────────────────────────────────

class DataSource:
    """Interface : fournit des exemples étiquetés."""

    def load(self, max_examples: int | None = None) -> list[Example]:
        raise NotImplementedError

    def name(self) -> str:
        return self.__class__.__name__


class EmoContextDataSource(DataSource):
    """Source : EmoContext (SemEval-2019 Task 3). Données incluses dans le repo."""

    def __init__(self, data_path: str | Path | None = None):
        if data_path is None:
            # chemin par défaut relatif au repo
            repo = Path(__file__).resolve().parents[1]
            data_path = repo / "data" / "emocontext" / "train.txt"
        self.data_path = Path(data_path)

    def load(self, max_examples: int | None = None) -> list[Example]:
        examples = load_emocontext(self.data_path, max_examples=max_examples)
        return [Example(turn1=e["turn1"], turn2=e["turn2"], turn3=e["turn3"],
                        label=e["label"], env=e["env"], c_seuil=e["c_seuil"],
                        label_num=e["label_num"]) for e in examples]


# ─────────────────────────────────────────────────────────────────────────────
# Interface 2 : Tokenizer (mot → embedding)
# ─────────────────────────────────────────────────────────────────────────────

class Tokenizer:
    """Interface : transforme un mot en embedding."""

    def embed(self, word: str, dim: int) -> np.ndarray:
        raise NotImplementedError

    def name(self) -> str:
        return self.__class__.__name__

    def dim(self) -> int:
        """Dimension d'embedding par défaut."""
        return 8


class HashTokenizer(Tokenizer):
    """Tokenizer hash orthogonal (piste 1). Toujours dispo, rapide."""

    def embed(self, word: str, dim: int) -> np.ndarray:
        return _hash_embedding(word, dim)

    def name(self) -> str:
        return "HASH"

    def dim(self) -> int:
        return 8


class TopoTokenizer(Tokenizer):
    """Tokenizer topologique (piste 3) : signature de cycles H1 persistants.
    Utilise persistence_optimizer (GUDHI si dispo, ~95x plus rapide)."""

    def __init__(self, n_points: int = 40):
        self.n_points = n_points
        self._available = is_full_persistence_available()

    def embed(self, word: str, dim: int) -> np.ndarray:
        return topo_signature(word, dim=dim, n_points=self.n_points)

    def name(self) -> str:
        return f"TOPO[{active_backend()}]"

    def dim(self) -> int:
        return 10

    def is_available(self) -> bool:
        return self._available


class TTFTokenizer(Tokenizer):
    """Tokenizer TTF (piste 2) : MCB du cerveau TTF-Compute. Nécessite AEON."""

    def embed(self, word: str, dim: int) -> np.ndarray:
        return ttf_embedding(word, dim)

    def name(self) -> str:
        return f"TTF[{'on' if is_ttf_available() else 'off'}]"

    def dim(self) -> int:
        return 8

    def is_available(self) -> bool:
        return is_ttf_available()


# ─────────────────────────────────────────────────────────────────────────────
# Interface 3 : Learner (le réseau + l'entraînement)
# ─────────────────────────────────────────────────────────────────────────────

class Learner:
    """Interface : entraîne et prédit."""

    def train(self, samples: list, epochs: int) -> dict:
        raise NotImplementedError

    def predict(self, token: np.ndarray, env: ThermoEnvironment) -> int:
        raise NotImplementedError

    def c_seuil_for(self, token: np.ndarray, env: ThermoEnvironment) -> float:
        """C_seuil prédit par ETH (pour observer l'émergence)."""
        raise NotImplementedError


class RatisNetV4Learner(Learner):
    """Implémentation : RATIS-Net v4 (LCT + ETH + collapse). Cœur encapsulé."""

    def __init__(self, n_in: int = 12, n_hidden: int = 20, n_out: int = 3,
                 token_dim: int = 8, env_dim: int = 4, eta: float = 0.1,
                 seed: int = 42):
        self.net = RatisNetV4(n_in=n_in, n_hidden=n_hidden, n_out=n_out,
                              token_dim=token_dim, env_dim=env_dim, eta=eta, seed=seed)
        self.last_acc = 0.0

    def train(self, samples: list, epochs: int) -> dict:
        for ep in range(epochs):
            correct = 0
            for tok, env, label, cs in samples:
                r = self.net.train_step(tok, env, label, cs, t_step=ep, lr_eth=0.1)
                correct += r["acc"]
            self.last_acc = correct / len(samples)
        return {"acc_train": self.last_acc, "n_samples": len(samples), "epochs": epochs}

    def predict(self, token: np.ndarray, env: ThermoEnvironment) -> int:
        x = self.net._build_input(token, env)
        h = np.array([n.forward(x, 0) for n in self.net.hidden])
        out = np.array([n.forward(h, 0) for n in self.net.output])
        return int(np.argmax(out))

    def scores(self, token: np.ndarray, env: ThermoEnvironment) -> np.ndarray:
        """Vecteur de confiance brut de la couche de sortie (pour le décodeur).

        La confiance pour l'émotion i = out[i]. Plus out[i] est élevé, plus le
        réseau « croit » que (token, env) exprime l'émotion i. Le décodeur
        utilise ça pour chercher le mot qui maintient une émotion cible.
        """
        x = self.net._build_input(token, env)
        h = np.array([n.forward(x, 0) for n in self.net.hidden])
        out = np.array([n.forward(h, 0) for n in self.net.output])
        return out

    def c_seuil_for(self, token: np.ndarray, env: ThermoEnvironment) -> float:
        return self.net.eth.predict_c_seuil(self.net._token_for_eth(token), env)


# ─────────────────────────────────────────────────────────────────────────────
# Interface 4 : Pipeline (orchestre)
# ─────────────────────────────────────────────────────────────────────────────

class Pipeline:
    """Orchestre DataSource → Tokenizer (cache) → Learner → éval + émergence.

    C'est le point d'entrée unique pour un partenaire :
        p = Pipeline(EmoContextDataSource(), TopoTokenizer(), RatisNetV4Learner())
        report = p.run(n_dialogues=300, epochs=8)
    """

    def __init__(self, data_source: DataSource, tokenizer: Tokenizer,
                 learner: Learner, top_k_vocab: int = 80):
        self.data_source = data_source
        self.tokenizer = tokenizer
        self.learner = learner
        self.top_k_vocab = top_k_vocab
        self._cache: dict[str, np.ndarray] = {}

    def _build_cache(self, examples: list[Example]) -> dict[str, np.ndarray]:
        """Calcule et cache l'embedding de chaque mot unique."""
        words = vocabulary([e.__dict__ for e in examples], min_len=2,
                           top_k=self.top_k_vocab)
        dim = self.tokenizer.dim()
        cache = {}
        for w in words:
            cache[w] = self.tokenizer.embed(w, dim)
        return cache

    def _cached_embed(self, word: str) -> np.ndarray:
        if word in self._cache:
            return self._cache[word]
        emb = self.tokenizer.embed(word, self.tokenizer.dim())
        self._cache[word] = emb
        return emb

    def run(self, n_dialogues: int = 300, epochs: int = 8,
            split: float = 0.8, verbose: bool = True) -> TrainReport:
        """Entraîne et évalue. Retourne un rapport complet."""
        examples = self.data_source.load(max_examples=n_dialogues)
        labels = Counter(e.label for e in examples)
        self._cache = self._build_cache(examples)
        dim = self.tokenizer.dim()

        if verbose:
            print(f"Pipeline : {self.data_source.name()} × {self.tokenizer.name()} "
                  f"× {self.learner.__class__.__name__}")
            print(f"  {len(examples)} dialogues, {len(self._cache)} mots, "
                  f"dim={dim}, labels={dict(labels)}")

        # split
        rng = np.random.RandomState(42)
        idx = rng.permutation(len(examples))
        ntr = int(split * len(examples))
        tr = [examples[i] for i in idx[:ntr]]
        te = [examples[i] for i in idx[ntr:]]

        # samples d'entraînement
        tr_dicts = [{"turn1": e.turn1, "turn2": e.turn2, "turn3": e.turn3,
                     "label": e.label, "env": e.env, "c_seuil": e.c_seuil,
                     "label_num": e.label_num} for e in tr]
        samples = build_samples(tr_dicts, lambda w, d: self._cached_embed(w),
                                dim=dim, per_word=True)

        # entraînement
        train_res = self.learner.train(samples, epochs)

        # évaluation : vote des mots du tour3
        cor = wr = 0
        for ex in te:
            words = tokenize(ex.turn3)
            if not words:
                continue
            votes = [self.learner.predict(self._cached_embed(w), ex.env) for w in words]
            pred = int(np.argmax(np.bincount(votes)))
            cor += 1 if pred == ex.label_num else 0
            wr += 0 if pred == ex.label_num else 1
        acc_test = cor / (cor + wr)

        # émergence : C_seuil par émotion pour un mot neutre
        c_seuils = {}
        test_word = "ok"
        emb_ok = self._cached_embed(test_word)
        for label, (env_cls, _, _) in EMO_MAP.items():
            c_seuils[label] = self.learner.c_seuil_for(emb_ok, env_cls())

        if verbose:
            print(f"\n  acc train = {train_res['acc_train']:.3f}")
            print(f"  acc test  = {acc_test:.3f}  (vote turn3, hasard=0.333)")
            print(f"  émergence C_seuil '{test_word}' : {c_seuils}")
            diffs = {f"happy-{k}": c_seuils["happy"] - v for k, v in c_seuils.items() if k != "happy"}
            print(f"  différentiels : {diffs}")

        return TrainReport(
            n_dialogues=len(examples), labels=dict(labels),
            tokenizer_name=self.tokenizer.name(),
            backend=active_backend(),
            n_samples=len(samples),
            acc_train=train_res["acc_train"],
            acc_test_vote=acc_test,
            c_seuils=c_seuils,
            n_epochs=epochs,
        )


if __name__ == "__main__":
    # démonstration : le pipeline assemblé en 3 lignes
    print("=" * 72)
    print("Pipeline branchable — démo (3 lignes)")
    print("=" * 72)
    # 1. Hash (rapide, toujours dispo)
    p = Pipeline(EmoContextDataSource(), HashTokenizer(), RatisNetV4Learner())
    p.run(n_dialogues=300, epochs=6)
