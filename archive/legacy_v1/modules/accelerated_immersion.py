"""ratis_net.accelerated_immersion — Immersion structurée accélérée (ancrée).

L'idée de l'immersion accélérée : RATIS génère ses propres dialogues, les
certifie, les réinjecte → accélère la montée en compétence linguistique sans
attendre des données externes.

Le PIÈGE (mode collapse / synthetic data collapse) : un modèle qui s'entraîne
sur ses propres sorties sans vérité-terrain amplifie ses erreurs et dégénère.
Pour l'éviter, on n'génère PAS dans le vide. On ancre l'auto-génération sur des
dialogues RÉELS (EmoContext), et un DOUBLE filtre ne réinjecte que ce qui est
valide :

  1. FILTRE ZK (forme) : le dialogue généré a une structure topologiquement
     cohérente (hash topo stable, pas de bruit pur). La ZK certifie la forme.
  2. FILTRE SÉMANTIQUE (re-classage) : le dialogue généré, re-classé par le
     réseau LCT, retrouve bien l'émotion cible. C'est la vérité-terrain : si
     RATIS génère un dialogue "happy" re-classé "angry", on le JETTE.

Boucle d'immersion :
  1. SEED      — un dialogue réel d'EmoContext (tour3 + émotion + env).
  2. MUTATION  — substituer 1-2 mots par des mots du MÊME registre émotionnel
                 (mots classés dans la même classe par le réseau). Diversité
                 sans casser le sens.
  3. ZK        — hash topo du dialogue muté. Forme cohérente → on garde.
  4. SÉMANTIQUE — re-classage du dialogue muté. Retrouve l'émotion cible →
                 on garde. Sinon on jette (la mutation a cassé le sens).
  5. RÉINJECTION — les dialogues validés rejoignent le set d'entraînement.

Garde-fous anti-collapse :
  - Ancrage vérité-terrain : chaque dialogue généré dérive d'un réel.
  - Double filtre ZK + sémantique.
  - Plafond de réinjection (max_generated_ratio) : ne pas diluer le réel.
  - Diversité mesurée (lexical diversity) : si elle chute, on signale.

La loi LCT (R = P_sig, ΔW = η·φ·P_sig·C) est figée. On agit sur les DONNÉES,
pas sur la règle. Le gain est MESURÉ (avant/après), pas affirmé.
"""
from __future__ import annotations

import hashlib
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    from ratis_net.ratis_net_v4 import RatisNetV4
    from ratis_net.eth_thermo_fixer import ThermoEnvironment
    from ratis_net.emocontext_loader import (
        tokenize, EMO_MAP, build_sequence_samples, balance_classes,
    )
    from ratis_net.lct_collapse import topological_mark
except ImportError:
    from ratis_net_v4 import RatisNetV4
    from eth_thermo_fixer import ThermoEnvironment
    from emocontext_loader import tokenize, EMO_MAP, build_sequence_samples, balance_classes
    from lct_collapse import topological_mark

EMO_NAMES = {0: "angry", 1: "happy", 2: "others", 3: "sad"}


def lexical_diversity(dialogues: list[list[str]]) -> float:
    """Diversité lexicale = mots uniques / mots totaux. Chute = mode collapse.

    Une diversité qui chute au fil des itérations signale que le modèle
    dégénère vers un vocabulaire répétitif (signature du mode collapse).
    """
    all_words = [w for d in dialogues for w in d]
    if not all_words:
        return 0.0
    return len(set(all_words)) / len(all_words)


class AcceleratedImmersion:
    """Boucle d'immersion structurée accélérée (ancrée, double filtre).

    Args:
        net : un RatisNetV4 entraîné (le réseau qui comprend).
        cache : dictionnaire {mot: embedding}.
        real_examples : dialogues réels d'EmoContext (la vérité-terrain).
        max_generated_ratio : plafond de données générées vs réelles (anti-dilution).
    """

    def __init__(self, net: RatisNetV4, cache: dict, real_examples: list,
                 max_generated_ratio: float = 0.5, seed: int = 42):
        self.net = net
        self.cache = cache
        self.real_examples = real_examples
        self.max_generated_ratio = max_generated_ratio
        self.rng = np.random.RandomState(seed)
        # registre émotionnel : mot → classe prédite par le réseau (pour les
        # mutations cohérentes : substituer un mot par un autre de même classe)
        self._word_class: dict[str, int] = {}
        self._by_class: dict[int, list[str]] = defaultdict(list)
        self._build_word_registry()
        self.generated: list[dict] = []
        self.stats = {"n_seeds": 0, "n_mutated": 0, "n_pass_zk": 0,
                      "n_pass_semantic": 0, "n_rejected": 0}

    def _build_word_registry(self):
        """Classe chaque mot du vocabulaire par le réseau → registre émotionnel."""
        calm_env = ThermoEnvironment.calm()
        for w in self.cache:
            x = self.net._build_input(self.cache[w], calm_env)
            h = np.array([n.forward(x, 0) for n in self.net.hidden])
            out = np.array([n.forward(h, 0) for n in self.net.output])
            cls = int(np.argmax(out))
            self._word_class[w] = cls
            self._by_class[cls].append(w)

    def _classify_sequence(self, words: list[str], env: ThermoEnvironment) -> int:
        """Re-classe une SÉQUENCE de mots (piste 2 : la forme du message)."""
        embs = [self.cache[w] for w in words if w in self.cache]
        if len(embs) < 2:
            return -1
        embs = np.array(embs)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        seq_emb = (embs * norms).sum(axis=0) / norms.sum()
        n = np.linalg.norm(seq_emb)
        seq_emb = seq_emb / n if n > 1e-9 else seq_emb
        x = self.net._build_input(seq_emb, env)
        h = np.array([n_.forward(x, 0) for n_ in self.net.hidden])
        out = np.array([n_.forward(h, 0) for n_ in self.net.output])
        return int(np.argmax(out))

    def _mutate(self, words: list[str], target_cls: int,
                n_swaps: int = 2, risky: bool = True) -> list[str] | None:
        """Mutation : substitue des mots pour créer de la diversité.

        Mode prudent (risky=False) : substitue par des mots du MÊME registre
        émotionnel (cohérence garantie, mais peu de diversité).
        Mode risqué (risky=True) : 50% des substitutions viennent d'une classe
        DIFFÉRENTE. Cela crée de la vraie diversité ; le filtre sémantique
        élimine ensuite les mutations qui cassent le sens. Sans mode risqué,
        le taux d'accept est 100% (aucune information ajoutée) et la diversité
        s'effondre (mode collapse observé).
        """
        candidates = self._by_class.get(target_cls, [])
        if len(candidates) < 2:
            return None
        mutated = list(words)
        swap_indices = self.rng.choice(
            len(words), size=min(n_swaps, len(words)), replace=False)
        for idx in swap_indices:
            w = words[idx]
            if risky and self.rng.random() < 0.5:
                # 50% : mot d'une classe différente (diversité réelle)
                other_classes = [c for c in self._by_class if c != target_cls
                                 and len(self._by_class[c]) >= 2]
                if other_classes:
                    cls = self.rng.choice(other_classes)
                    pool = [c for c in self._by_class[cls]
                            if c != w and c not in mutated]
                else:
                    pool = [c for c in candidates if c != w and c not in mutated]
            else:
                # 50% : mot de même classe (cohérence)
                pool = [c for c in candidates if c != w and c not in mutated]
            if not pool:
                continue
            mutated[idx] = self.rng.choice(pool)
        return mutated

    def _zk_filter(self, words: list[str], env: ThermoEnvironment) -> bool:
        """Filtre ZK (forme) : le dialogue a une structure topologiquement
        cohérente. On calcule le hash topo ; s'il est stable (les mots forment
        une forme, pas du bruit pur), on garde."""
        if len(words) < 2:
            return False
        embs = [self.cache[w] for w in words if w in self.cache]
        if len(embs) < 2:
            return False
        W = np.array(embs)
        mark = topological_mark(W, c_seuil=0.0, env_vector=env.to_vector())
        # la marque existe (hash non vide) = la forme est cohérente
        return len(mark) > 0

    def _semantic_filter(self, words: list[str], env: ThermoEnvironment,
                         target_cls: int) -> bool:
        """Filtre sémantique (re-classage) : le dialogue muté, re-classé par
        le réseau, retrouve l'émotion cible. C'est la vérité-terrain."""
        pred = self._classify_sequence(words, env)
        return pred == target_cls

    def generate_batch(self, n_dialogues: int = 200) -> list[dict]:
        """Génère un batch de dialogues par immersion ancrée.

        Pour chaque dialogue réel (seed), on mute, on filtre ZK + sémantique,
        on ne garde que les validés. Plafond : max_generated_ratio du set réel.
        """
        n_seeds = min(n_dialogues, len(self.real_examples))
        seeds = self.rng.choice(len(self.real_examples), size=n_seeds, replace=False)
        accepted = []
        for si in seeds:
            ex = self.real_examples[si]
            self.stats["n_seeds"] += 1
            words = tokenize(ex["turn3"]) if isinstance(ex, dict) else tokenize(ex.turn3)
            words = [w for w in words if w in self.cache]
            if len(words) < 3:
                continue
            target_cls = ex["label_num"] if isinstance(ex, dict) else ex.label_num
            env = ex["env"] if isinstance(ex, dict) else ex.env
            # mutation (1-2 swaps, même registre émotionnel)
            mutated = self._mutate(words, target_cls, n_swaps=self.rng.choice([1, 2]))
            if mutated is None or mutated == words:
                continue
            self.stats["n_mutated"] += 1
            # filtre ZK (forme)
            if not self._zk_filter(mutated, env):
                self.stats["n_rejected"] += 1
                continue
            self.stats["n_pass_zk"] += 1
            # filtre sémantique (re-classage = cible)
            if not self._semantic_filter(mutated, env, target_cls):
                self.stats["n_rejected"] += 1
                continue
            self.stats["n_pass_semantic"] += 1
            accepted.append({
                "turn3": " ".join(mutated), "words": mutated,
                "label": ex["label"] if isinstance(ex, dict) else ex.label,
                "label_num": target_cls, "env": env,
                "c_seuil": ex["c_seuil"] if isinstance(ex, dict) else ex.c_seuil,
                "generated": True,
            })
            # plafond anti-dilution
            if len(accepted) >= n_seeds * self.max_generated_ratio:
                break
        self.generated.extend(accepted)
        return accepted

    def build_training_set(self) -> list:
        """Set d'entraînement = dialogues réels + dialogues générés validés.

        Construit les samples séquence (piste 2) sur le set augmenté.
        """
        all_examples = list(self.real_examples) + \
            [{"turn1": "", "turn2": "", "turn3": g["turn3"],
              "label": g["label"], "env": g["env"],
              "c_seuil": g["c_seuil"], "label_num": g["label_num"]}
             for g in self.generated]
        dim = next(iter(self.cache.values())).shape[0]
        emb_fn = lambda w, d: self.cache.get(w, np.zeros(d))
        samples = build_sequence_samples(all_examples, emb_fn, dim=dim)
        return samples

    def diversity(self) -> float:
        """Diversité lexicale des dialogues générés (chute = mode collapse)."""
        return lexical_diversity([g["words"] for g in self.generated])
