"""ratis_net.decoder — Le décodeur LCT (génération de langage).

La brique qui fait passer RATIS-Net de classifieur (comprendre) à générateur
(parler). On génère une séquence de mots conditionnée par une émotion cible +
un environnement thermo.

Décodage par cohérence topologique (fidèle à la loi LCT) :
  À chaque position, on cherche le mot candidat w tel que :
      score(w) = confiance_réseau(émotion cible | w, env) × vraisemblance(w | mot_précédent, émotion)
  On pénalise la répétition (diversité) et on garde le top-1 (glouton) ou
  top-k stochastique.

La confiance vient du réseau LCT entraîné (learner.scores). La vraisemblance
de transition est apprise des dialogues EmoContext : pour chaque émotion,
on compte les bigrammes (mot1, mot2) dans les dialogues de cette émotion. C'est
un modèle de transition léger (matrice bigramme par émotion), pas un LLM.

Le résultat n'est pas un LLM (pas de grammaire, pas d'auto-régression avec
état caché qui accumule le contexte). C'est un POC de génération : le réseau
PRODUIT du langage conditionné par l'émotion, pas seulement comprend. C'est la
brique manquante vers « parler ».
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    from ratis_net.emocontext_loader import load_emocontext, tokenize, EMO_MAP
    from ratis_net.eth_thermo_fixer import ThermoEnvironment
except ImportError:
    from emocontext_loader import load_emocontext, tokenize, EMO_MAP
    from eth_thermo_fixer import ThermoEnvironment


class BigramModel:
    """Modèle de transition léger : P(mot_suivant | mot_précédent, émotion).

    Appris des dialogues EmoContext (comptage des bigrammes par émotion). C'est
    ce qui donne la vraisemblance linguistique : on privilégie les transitions
    qui existent réellement dans les dialogues humains, pas seulement la
    cohérence LCT seule (qui sinon génère des mots répétés).
    """

    def __init__(self):
        # bigrams[emotion][mot1] = Counter(mot2 -> count)
        self.bigrams: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
        # normalize lazily
        self._norm: dict[str, dict[str, dict[str, float]]] = {}

    def fit(self, examples: list[dict]):
        """Apprend les bigrammes à partir des dialogues (tour3 concaténé)."""
        for ex in examples:
            label = ex["label"]
            words = tokenize(ex["turn3"])
            # pseudo-début et fin
            words = ["<start>"] + words + ["<end>"]
            for w1, w2 in zip(words, words[1:]):
                self.bigrams[label][w1][w2] += 1

    def prob(self, emotion: str, prev: str, nxt: str) -> float:
        """P(nxt | prev, emotion). Fallback lissage 1/N."""
        if emotion not in self._norm:
            self._norm[emotion] = {}
        norm = self._norm[emotion]
        if prev not in norm:
            c = self.bigrams.get(emotion, {}).get(prev, Counter())
            total = sum(c.values()) or 1
            norm[prev] = {w: cnt / total for w, cnt in c.items()}
        return norm[prev].get(nxt, 1e-3)  # lissage : évite les 0 absolus

    def candidates(self, emotion: str, prev: str, vocab: list[str]) -> list[tuple[str, float]]:
        """Top candidats après prev, parmi le vocabulaire, triés par proba."""
        c = self.bigrams.get(emotion, {}).get(prev, Counter())
        # on restreint au vocabulaire disponible
        return [(w, self.prob(emotion, prev, w)) for w in vocab if w in c or True]


class LCTDecoder:
    """Décodeur LCT : génère une séquence de mots conditionnée par une émotion.

    Args:
        learner : un Learner entraîné (expose scores(token, env) -> ndarray).
        cache : dictionnaire {mot: embedding} (les signatures du tokenizer).
        vocab : liste des mots disponibles pour la génération.
        bigram_model : modèle de transition (vraisemblance linguistique).
    """

    def __init__(self, learner, cache: dict[str, np.ndarray], vocab: list[str],
                 bigram_model: BigramModel | None = None):
        self.learner = learner
        self.cache = cache
        self.vocab = [w for w in vocab if w in cache]
        self.bigram = bigram_model

    def _score_word(self, word: str, target_emotion_num: int,
                    env: ThermoEnvironment) -> float:
        """Confiance du réseau que (word, env) exprime l'émotion cible."""
        emb = self.cache[word]
        scores = self.learner.scores(emb, env)
        # confiance pour l'émotion cible, normalisée par softmax sur les scores
        # (les scores peuvent être négatifs → on shift)
        s = scores - scores.min()
        expv = np.exp(s)
        probs = expv / (expv.sum() + 1e-9)
        return float(probs[target_emotion_num])

    def generate(self, target_emotion: str, env: ThermoEnvironment,
                 length: int = 8, temperature: float = 0.7,
                 repeat_penalty: float = 0.5, verbose: bool = False) -> list[str]:
        """Génère une séquence de `length` mots exprimant l'émotion cible.

        Décodage : à chaque pas, score(w) = confiance_LCT(w, cible) ×
        vraisemblance_transition(prev→w). On applique une température pour
        stochastiser (pas du glouton pur) et une pénalité de répétition.
        """
        target_num = EMO_MAP[target_emotion][2]
        words = []
        prev = "<start>"
        for step in range(length):
            scores = []
            for w in self.vocab:
                conf = self._score_word(w, target_num, env)
                # vraisemblance de transition (si bigram dispo)
                trans = self.bigram.prob(target_emotion, prev, w) if self.bigram else 1.0
                # pénalité de répétition
                rep = repeat_penalty if w in words[-3:] else 1.0
                score = conf * trans * rep
                scores.append((w, score, conf))
            # température : on boost les écarts pour rendre le top plus tranché
            s_arr = np.array([s[1] for s in scores])
            s_arr = np.log(s_arr + 1e-9) / max(temperature, 1e-3)
            w_arr = np.exp(s_arr - s_arr.max())
            w_arr /= w_arr.sum() + 1e-9
            # tirage stochastique selon les poids
            idx = int(np.random.RandomState(step).choice(len(self.vocab), p=w_arr))
            chosen = self.vocab[idx]
            words.append(chosen)
            prev = chosen
            if verbose:
                top3 = sorted(scores, key=lambda x: -x[1])[:3]
                print(f"  step {step}: choisi='{chosen}' (conf={scores[idx][2]:.3f}) "
                      f"top3={[t[0] for t in top3]}")
        return words

    def generate_greedy(self, target_emotion: str, env: ThermoEnvironment,
                        length: int = 8, repeat_penalty: float = 0.3) -> list[str]:
        """Variante déterministe (glouton) : le mot de plus haut score à chaque pas.

        Limite honnête : optimise le score LOCAL (mot par mot), pas la cohérence
        GLOBALE de la séquence. L'émotion cible peut être noyée par les mots
        dominants d'une autre classe (ex: "you are" → classe négative, même pour
        une cible happy). Voir generate_autoregressive / generate_beam pour la
        cohérence de séquence.
        """
        target_num = EMO_MAP[target_emotion][2]
        words = []
        prev = "<start>"
        for step in range(length):
            best, best_score = None, -1.0
            for w in self.vocab:
                conf = self._score_word(w, target_num, env)
                trans = self.bigram.prob(target_emotion, prev, w) if self.bigram else 1.0
                rep = repeat_penalty if w in words[-3:] else 1.0
                s = conf * trans * rep
                if s > best_score:
                    best, best_score = w, s
            words.append(best)
            prev = best
        return words

    # ── Piste 1 : décodage auto-régressif avec ÉTAT CACHÉ ─────────────────────
    #
    # Le glouton classe chaque mot individuellement et vote à la fin : l'émotion
    # cible peut être noyée. L'auto-régressif maintient un ÉTAT CACHÉ = la
    # distribution d'émotions accumulée de la séquence en cours. Quand la
    # séquence dévie de la cible (trop de mots classés non-cible), un feedback
    # BOOSTE les mots cible : la cohérence topologique de la séquence ENTIÈRE
    # gouverne le choix, pas le score local. C'est l'analogue d'un LLM avec état
    # caché, mais piloté par LCT : on certifie le message (la forme de la
    # séquence), pas le courant (chaque mot).

    def _word_class(self, word: str, env: ThermoEnvironment) -> int:
        """Classe prédite du mot seul (pour nourrir l'état caché)."""
        return self.learner.predict(self.cache[word], env)

    def _seq_embedding(self, words: list[str]) -> np.ndarray:
        """Embedding d'une SÉQUENCE = moyenne des embeddings de ses mots.

        C'est l'analogue topologique : la séquence entière est une forme, pas
        la somme de mots isolés. La moyenne des embeddings capture la dominante
        sémantique de la phrase (un mot happy + des mots neutres penche vers
        happy, contrairement au vote qui compte chaque mot 1/1).
        """
        embs = [self.cache[w] for w in words if w in self.cache]
        if not embs:
            rng = np.random.default_rng(0)
            return rng.normal(0, 1, next(iter(self.cache.values())).shape[0])
        # moyenne pondérée par la norme (les mots saillants pèsent plus)
        arr = np.array(embs)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        weighted = (arr * norms).sum(axis=0) / norms.sum()
        n = np.linalg.norm(weighted)
        return weighted / n if n > 1e-9 else weighted

    def generate_autoregressive(self, target_emotion: str, env: ThermoEnvironment,
                                length: int = 8, repeat_penalty: float = 0.3,
                                feedback_strength: float = 2.0,
                                min_target_frac: float = 0.5) -> list[str]:
        """Décodage auto-régressif : un état caché VECTEUR maintient la cohérence.

        État caché = embedding de la séquence partielle en cours (moyenne des
        mots générés). À chaque pas :
          1. on classe l'état caché courant (la séquence partielle vue comme
             un tout — c'est la métrique LCT : la forme, pas chaque mot).
          2. si la séquence partielle dévie de la cible, un feedback BOOSTE les
             candidats qui ramènent l'état caché vers la cible : G(w) = confiance
             que (état_caché ⊕ w) soit classé cible.

        La cohérence topologique de la séquence ENTIÈRE gouverne le choix —
        l'émotion cible ne peut plus être noyée par les mots neutres fréquents
        (le défaut du glouton sur happy, où "you"/"are" sont classés angry par
        fréquence du corpus). On certifie le message (la forme de la séquence),
        pas le courant (chaque mot).

        Args:
            feedback_strength: poids du feedback quand l'état caché dévie. 0
                = glouton (état caché désactivé).
            min_target_frac: seuil de confiance (softmax) de la cible en
                dessous duquel le feedback s'active.
        """
        target_num = EMO_MAP[target_emotion][2]
        words = []
        prev = "<start>"
        for step in range(length):
            # état caché = séquence partielle actuelle
            state = self._seq_embedding(words) if words else None
            # confiance de l'état caché courant dans la cible
            if state is not None:
                state_scores = self.learner.scores(state, env)
                state_scores = state_scores - state_scores.min()
                expv = np.exp(state_scores)
                state_probs = expv / (expv.sum() + 1e-9)
                state_conf_target = float(state_probs[target_num])
            else:
                state_conf_target = 0.0
            # besoin de feedback : élevé si l'état caché dévie (confiance cible basse)
            besoin = max(0.0, min_target_frac - state_conf_target)

            best, best_score = None, -1.0
            for w in self.vocab:
                conf = self._score_word(w, target_num, env)
                trans = self.bigram.prob(target_emotion, prev, w) if self.bigram else 1.0
                rep = repeat_penalty if w in words[-3:] else 1.0
                # feedback : confiance que l'état caché ÉTENDU par w soit cible
                if words:
                    ext_state = self._seq_embedding(words + [w])
                    ext_scores = self.learner.scores(ext_state, env)
                    ext_scores = ext_scores - ext_scores.min()
                    expv = np.exp(ext_scores)
                    ext_probs = expv / (expv.sum() + 1e-9)
                    ext_conf = float(ext_probs[target_num])
                    gate = 1.0 + feedback_strength * besoin * (ext_conf - state_conf_target)
                else:
                    gate = 1.0
                s = conf * trans * rep * max(gate, 1e-6)
                if s > best_score:
                    best, best_score = w, s
            words.append(best)
            prev = best
        return words

    def generate_beam(self, target_emotion: str, env: ThermoEnvironment,
                      length: int = 8, beam_width: int = 4,
                      repeat_penalty: float = 0.3,
                      coherence_weight: float = 3.0) -> list[str]:
        """Beam search : explore K séquences, garde la plus cohérente GLOBALEMENT.

        À chaque pas, on étend chaque beam avec chaque mot candidat et on score
        le beam étendu. Le score = somme des confiances LCT (cible, par mot) +
        vraisemblance bigramme + cohérence de la SÉQUENCE partielle (confiance
        que l'embedding de la séquence partielle soit classée cible). On garde
        les K beams de score total maximal.

        La cohérence de séquence est mesurée sur l'ÉTAT CACHÉ (l'embedding moyen
        de la séquence partielle), pas sur un vote de mots individuels : on
        certifie la forme de la séquence (le message), pas chaque mot (le courant).
        """
        target_num = EMO_MAP[target_emotion][2]
        # un beam = (mots, prev, score_total, state_embedding)
        beams = [([], "<start>", 0.0, None)]
        for step in range(length):
            expanded = []
            for words, prev, score, state in beams:
                for w in self.vocab:
                    conf = self._score_word(w, target_num, env)
                    trans = self.bigram.prob(target_emotion, prev, w) if self.bigram else 1.0
                    rep = repeat_penalty if w in words[-3:] else 1.0
                    # état caché étendu = séquence partielle + w
                    ext_state = self._seq_embedding(words + [w])
                    ext_scores = self.learner.scores(ext_state, env)
                    ext_scores = ext_scores - ext_scores.min()
                    expv = np.exp(ext_scores)
                    ext_probs = expv / (expv.sum() + 1e-9)
                    seq_conf = float(ext_probs[target_num])
                    local = conf * trans * rep
                    coherence_bonus = coherence_weight * seq_conf
                    new_score = score + local + coherence_bonus
                    expanded.append((words + [w], w, new_score, ext_state))
            expanded.sort(key=lambda x: -x[2])
            beams = expanded[:beam_width]
        return beams[0][0]


def fit_bigram_from_emocontext(data_path: str | Path | None = None,
                                max_examples: int = 3000) -> BigramModel:
    """Construit le modèle de transition bigramme à partir d'EmoContext."""
    repo = Path(__file__).resolve().parents[1]
    if data_path is None:
        data_path = repo / "data" / "emocontext" / "train.txt"
    examples = load_emocontext(data_path, max_examples=max_examples)
    bm = BigramModel()
    bm.fit(examples)
    return bm


if __name__ == "__main__":
    # démonstration : génère pour chaque émotion
    import sys
    from pathlib import Path
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_ROOT))
    from ratis_net.pipeline import (
        Pipeline, EmoContextDataSource, HashTokenizer, RatisNetV4Learner,
    )
    from ratis_net.emocontext_loader import vocabulary

    print("Entraînement du pipeline (pour le décodeur)...")
    p = Pipeline(EmoContextDataSource(), HashTokenizer(), RatisNetV4Learner())
    examples = p.data_source.load(max_examples=500)
    p._cache = p._build_cache(examples)
    dim = p.tokenizer.dim()
    from ratis_net.emocontext_loader import build_samples
    samples = build_samples([e.__dict__ for e in examples],
                            lambda w, d: p._cached_embed(w), dim=dim, per_word=True)
    p.learner.train(samples, epochs=6)
    vocab = list(p._cache.keys())

    bm = fit_bigram_from_emocontext(max_examples=3000)
    decoder = LCTDecoder(p.learner, p._cache, vocab, bm)

    print("\nGénération (glouton) pour chaque émotion :")
    for emo in ["happy", "angry", "sad", "others"]:
        env_cls = EMO_MAP[emo][0]
        seq = decoder.generate_greedy(emo, env_cls(), length=6)
        print(f"  {emo:7s}: {' '.join(seq)}")

    # validation : re-classer la séquence générée doit retrouver l'émotion cible
    print("\nValidation (re-classage de la séquence générée) :")
    for emo in ["happy", "angry", "sad"]:
        env_cls = EMO_MAP[emo][0]
        env = env_cls()
        seq = decoder.generate_greedy(emo, env, length=6)
        votes = [p.learner.predict(p._cached_embed(w), env) for w in seq]
        pred = int(np.argmax(np.bincount(votes)))
        target = EMO_MAP[emo][2]
        ok = "✓" if pred == target else "✗"
        print(f"  {emo:7s} → généré '{' '.join(seq)}' → reclassé {pred} (cible {target}) {ok}")
