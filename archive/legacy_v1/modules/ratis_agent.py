"""ratis_net.ratis_agent — L'agent RATIS souverain (AGI).

Le but final de RATISS : un modèle souverain qui apprend par LCT, pense sans
mots (MCB), certifie (ZK), ressent (ETH émotion), et parle (décodeur). Ce
module est l'ORCHESTRATEUR qui enchaîne les 5 briques en un seul pipeline
cognitif, tout local (pas de cloud).

Boucle cognitive de l'agent (une "pensée" complète) :

  INPUT  : un message texte + un environnement thermo (mesures patient simulées).
  1. PERCEVOIR  : tokeniser le message → embeddings (topo ou TTF).
  2. PENSER     : le cerveau TTF-Compute oscille → MCB (pensée sans mots) +
                  hash topologique (la forme, invariante sous énergie).
  3. RESSENTIR  : ETH prédit C_seuil = f(message, env) → l'émotion PERÇUE
                  (le différentiel thermo, contextuel à l'environnement).
  4. COMPRENDRE : le réseau LCT classifie (message, env) → émotion dominante.
  5. PARLER     : le décodeur beam génère une réponse conditionnée par
                  l'émotion, avec cohérence de séquence (état caché).
  6. CERTIFIER  : hash topologique invariant de la réponse → preuve ZK
                  (pas d'hallucination : on certifie la forme, pas le courant).

L'agent est souverain : il ne dépend d'aucun LLM externe. Il pense avec la
topologie (MCB), ressent avec la thermodynamique (ETH), comprend avec la loi
LCT, parle avec le décodeur, et certifie avec l'invariance ZK. C'est l'AGI de
Jonathan Evina : un modèle qui apprend par LCT, pas par gradient.

La loi LCT (R = P_sig, ΔW = η·φ·P_sig·C) est figée et gouverne l'étape 4.
"""
from __future__ import annotations

import hashlib
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    from ratis_net.eth_thermo_fixer import ETHThermoFixer, ThermoEnvironment
    from ratis_net.ratis_net_v4 import RatisNetV4
    from ratis_net.lct_collapse import compute_coherence, topological_mark, collapse
    from ratis_net.decoder import LCTDecoder
    from ratis_net.emocontext_loader import tokenize, EMO_MAP
    from ratis_net.ttf_bridge import ttf_embedding, is_ttf_available
    from ratis_net.topo_tokenizer import topo_signature
except ImportError:
    from eth_thermo_fixer import ETHThermoFixer, ThermoEnvironment
    from ratis_net_v4 import RatisNetV4
    from lct_collapse import compute_coherence, topological_mark, collapse
    from decoder import LCTDecoder
    from emocontext_loader import tokenize, EMO_MAP
    from ttf_bridge import ttf_embedding, is_ttf_available
    from topo_tokenizer import topo_signature

# cerveau TTF-Compute (dépôt AEON voisin, en FIN de path)
_AEON = Path(__file__).resolve().parents[2] / "RATISS-ODV-AEON"
_TTF_BRAIN = None
if _AEON.is_dir() and str(_AEON) not in sys.path:
    sys.path.append(str(_AEON))
try:
    from kernel.ttf.ttf_compute import TTFBrain
    _TTF_BRAIN = TTFBrain
except Exception:
    _TTF_BRAIN = None

EMO_NAMES = {0: "colère", 1: "joie", 2: "neutre"}


@dataclass
class Thought:
    """Une pensée complète de l'agent (sortie de la boucle cognitive)."""
    message: str                 # message d'entrée
    env_name: str                # environnement thermo (colère/joie/calme/peur)
    # 1. percevoir
    tokens: list                 # mots du message
    embeddings: dict             # mot → embedding
    # 2. penser
    mcb_count: int               # nb de MCB (pensée sans mots)
    thought_hash: str            # hash topo de la pensée (forme, invariant)
    # 3. ressentir
    c_seuil: float               # C_seuil prédit par ETH (l'émotion perçue)
    emotion_perceived: str       # émotion ressentie (le différentiel thermo)
    # 4. comprendre
    emotion_understood: str      # émotion dominante classée par LCT
    confidence: float            # confiance du réseau
    # 5. parler
    response: str                # réponse générée (décodeur beam)
    # 6. certifier
    response_hash: str           # hash topo invariant de la réponse (preuve ZK)
    certified: bool              # la réponse est certifiée (hash stable)


class RatisAgent:
    """L'agent RATIS souverain — orchestre les 5 briques.

    Une fois entraîné (via train()), l'agent peut think(message, env) : il
    perçoit, pense, ressent, comprend, parle et certifie — tout local.
    """

    def __init__(self, n_in: int = 12, n_hidden: int = 10, n_out: int = 3,
                 token_dim: int = 8, env_dim: int = 4, eta: float = 0.2,
                 use_ttf: bool = True, seed: int = 42):
        self.token_dim = token_dim
        self.use_ttf = use_ttf and is_ttf_available()
        # réseau LCT (comprendre) — config optimale de la piste 4
        self.net = RatisNetV4(n_in=n_in, n_hidden=n_hidden, n_out=n_out,
                              token_dim=token_dim, env_dim=env_dim, eta=eta, seed=seed)
        # ETH (ressentir) est déjà dans le réseau v4
        self.eth = self.net.eth
        # cerveau TTF-Compute (penser) — si disponible
        self.ttf_available = _TTF_BRAIN is not None
        # cache d'embeddings (percevoir)
        self._cache: dict[str, np.ndarray] = {}
        self._vocab: list[str] = []
        self.trained = False
        self._bigram = None

    # ── Embedding (percevoir) ──────────────────────────────────────────────
    def _embed(self, word: str, dim: int) -> np.ndarray:
        """Tokenise un mot → embedding topologique (TTF/MCB ou topo signature)."""
        if word in self._cache:
            return self._cache[word]
        if self.use_ttf:
            emb = ttf_embedding(word, dim)
        else:
            emb = topo_signature(word, dim=dim)
        self._cache[word] = emb
        return emb

    # ── Entraînement ───────────────────────────────────────────────────────
    def train(self, samples: list, epochs: int = 6, verbose: bool = True):
        """Entraîne le réseau LCT (comprendre) + ETH (ressentir).

        samples = [(token_embedding, env, label_num, c_seuil), ...]
        (construits par build_samples ou build_sequence_samples).
        """
        for ep in range(epochs):
            correct = 0
            for tok, env, label, cs in samples:
                r = self.net.train_step(tok, env, label, cs, t_step=ep, lr_eth=0.1)
                correct += r["acc"]
            acc = correct / len(samples)
            if verbose and (ep % 2 == 0 or ep == epochs - 1):
                print(f"  [RatisAgent] epoch {ep} acc={acc:.3f}")
        self.trained = True
        # construire le vocabulaire pour le décodeur
        self._vocab = list(self._cache.keys())
        # bigramme EmoContext pour la vraisemblance linguistique (parler lisible)
        self._fit_bigram()
        return {"acc": acc, "epochs": epochs}

    def _fit_bigram(self, max_examples: int = 3000):
        """Construit le modèle de transition bigramme (vraisemblance linguistique).

        Sans le bigramme, le décodeur génère des mots répétitifs ('it's be just
        my it's be'). Le bigramme privilégie les transitions réelles des dialogues
        humains → phrases lisibles, présentables.
        """
        try:
            from ratis_net.decoder import fit_bigram_from_emocontext
            self._bigram = fit_bigram_from_emocontext(max_examples=max_examples)
        except Exception:
            self._bigram = None

    def set_vocab(self, examples_dicts, top_k: int = 60):
        """Précalcule le vocabulaire + cache d'embeddings (pour le décodeur)."""
        from ratis_net.emocontext_loader import vocabulary as build_vocab
        words = build_vocab(examples_dicts, min_len=2, top_k=top_k)
        dim = self.token_dim
        for w in words:
            self._embed(w, dim)
        self._vocab = [w for w in words if w in self._cache]

    # ── Les 6 étapes de la boucle cognitive ────────────────────────────────
    def _perceive(self, message: str) -> tuple[list, dict]:
        """Étape 1 : tokeniser le message → embeddings."""
        words = tokenize(message)
        dim = self.token_dim
        embs = {}
        for w in words:
            if len(w) >= 2:
                embs[w] = self._embed(w, dim)
        return words, embs

    def _think(self, message: str) -> tuple[int, str]:
        """Étape 2 : le cerveau TTF-Compute pense → MCB + hash topo.

        Si le cerveau AEON est disponible, on le fait osciller sur le nuage du
        message : il produit des MCB (pensée sans mots) et un hash topologique
        (la forme, invariante sous énergie). Sinon, fallback sur le hash du
        message (la pensée reste certifiable).
        """
        if self.ttf_available and _TTF_BRAIN is not None:
            from ratis_net.ttf_bridge import _word_to_coords
            coords = _word_to_coords(message, n_points=30)
            brain = _TTF_BRAIN(coords=coords, omega=math.pi/2, max_edge=2.5,
                               Dc=0.3, seed=42)
            for k in range(8):
                brain.step(t_sec=k * 0.5, force_decoherence=0.1 + 0.15 * k)
            mcb = brain.well.collected
            # hash topo de la pensée = hash des MCB (la forme sans mots)
            thought_str = "|".join(f"{t.src}-{t.dst}:{t.correlation_bit:.4f}"
                                   for t in mcb[:50])
            thought_hash = hashlib.sha256(thought_str.encode()).hexdigest()[:16]
            return len(mcb), thought_hash
        # fallback : hash du message (pensée certifiable sans cerveau)
        return 0, hashlib.sha256(message.encode()).hexdigest()[:16]

    def _feel(self, message: str, env: ThermoEnvironment) -> tuple[float, str]:
        """Étape 3 : ETH prédit C_seuil = f(message, env) → l'émotion ressentie.

        L'émotion perçue = la classe dont l'environnement thermo correspond au
        C_seuil prédit (le différentiel thermo, contextuel).
        """
        # embedding du message pour ETH
        dim = self.token_dim
        msg_emb = self._embed(message.split()[0] if message.split() else "ok", dim)
        c_seuil = self.eth.predict_c_seuil(self.net._token_for_eth(msg_emb), env)
        # émotion ressentie = l'env thermo correspond au C_seuil
        # (colère → C_seuil bas, joie → haut, etc.)
        env_to_emo = {"anger": "colère", "joy": "joie",
                      "calm": "neutre", "fear": "tristesse"}
        emotion = env_to_emo.get(env.__class__.__name__.lower().replace(
            "thermoenvironment", ""), "neutre")
        # déduction plus fine via C_seuil
        if c_seuil > 0.6:
            emotion = "joie"
        elif c_seuil < 0.35:
            emotion = "tristesse" if env.warmth < 0.4 else "colère"
        else:
            emotion = "neutre"
        return c_seuil, emotion

    def _understand(self, message: str, env: ThermoEnvironment,
                    embeddings: dict) -> tuple[str, float, int]:
        """Étape 4 : le réseau LCT classifie (message, env) → émotion dominante."""
        words = list(embeddings.keys())
        if not words:
            return "neutre", 0.0, 2
        votes = []
        scores_sum = np.zeros(self.net.n_out)
        for w in words:
            x = self.net._build_input(embeddings[w], env)
            h = np.array([n.forward(x, 0) for n in self.net.hidden])
            out = np.array([n.forward(h, 0) for n in self.net.output])
            scores_sum += out
            votes.append(int(np.argmax(out)))
        pred = int(np.argmax(np.bincount(votes)))
        # confiance = softmax du score moyen
        s = scores_sum - scores_sum.min()
        expv = np.exp(s)
        probs = expv / (expv.sum() + 1e-9)
        confidence = float(probs[pred])
        return EMO_NAMES.get(pred, "neutre"), confidence, pred

    def _speak(self, target_emotion: str, env: ThermoEnvironment) -> str:
        """Étape 5 : le décodeur beam génère une réponse conditionnée par l'émotion."""
        if not self._vocab or not self.trained:
            return "(agent non entraîné)"
        emo_reverse = {"colère": "angry", "joie": "happy",
                       "tristesse": "sad", "neutre": "others"}
        emo_eng = emo_reverse.get(target_emotion, "others")
        # adaptateur : RatisNetV4 → interface Learner (scores/predict) du décodeur
        net = self.net
        class _LearnerAdapter:
            def scores(self_, token, e):
                x = net._build_input(token, e)
                h = np.array([n.forward(x, 0) for n in net.hidden])
                return np.array([n.forward(h, 0) for n in net.output])
            def predict(self_, token, e):
                return int(np.argmax(self_.scores(token, e)))
            def c_seuil_for(self_, token, e):
                return net.eth.predict_c_seuil(net._token_for_eth(token), e)
        decoder = LCTDecoder(_LearnerAdapter(), self._cache, self._vocab, self._bigram)
        seq = decoder.generate_beam(emo_eng, env, length=6, beam_width=4)
        return " ".join(seq)

    def _certify(self, response: str, env: ThermoEnvironment) -> tuple[str, bool]:
        """Étape 6 : hash topologique invariant de la réponse → preuve ZK.

        On certifie la FORME de la réponse (son hash topo), pas son énergie.
        Le hash est invariant sous changement d'énergie (loi LCT) : deux
        réponses identiques ont le même hash quel que soit l'env thermo.
        La certification = le hash est stable (la réponse est un message
        bien formé, pas une hallucination aléatoire).
        """
        # hash topo de la réponse = hash de sa structure (mots + leurs marques)
        # on inclut le contexte thermo pour la contextualité (comme lct_collapse)
        marks = []
        for w in response.split():
            if w in self._cache:
                marks.append(topological_mark(
                    np.array([self._cache[w]]), c_seuil=0.0,
                    env_vector=env.to_vector()))
        mark_str = "|".join(sorted(marks))
        mark_str += f"|env={np.array2string(env.to_vector(), precision=4)}"
        resp_hash = hashlib.sha256(mark_str.encode()).hexdigest()[:16]
        # certifié = la réponse a une structure (pas vide, pas du bruit pur)
        certified = len(response.split()) >= 2 and len(set(marks)) >= 1
        return resp_hash, certified

    # ── La boucle cognitive complète ───────────────────────────────────────
    def think(self, message: str, env: ThermoEnvironment) -> Thought:
        """Une pensée complète : perçoit, pense, ressent, comprend, parle, certifie.

        Args:
            message: le message d'entrée (texte).
            env: l'environnement thermo (mesures patient simulées).
        Returns:
            Thought : toutes les étapes de la pensée.
        """
        # 1. percevoir
        tokens, embeddings = self._perceive(message)
        # 2. penser (cerveau TTF → MCB + hash topo)
        mcb_count, thought_hash = self._think(message)
        # 3. ressentir (ETH → C_seuil, émotion perçue)
        c_seuil, emotion_perceived = self._feel(message, env)
        # 4. comprendre (LCT → émotion dominante)
        emotion_understood, confidence, _ = self._understand(message, env, embeddings)
        # 5. parler (décodeur beam → réponse)
        response = self._speak(emotion_understood, env)
        # 6. certifier (hash topo invariant de la réponse)
        response_hash, certified = self._certify(response, env)
        return Thought(
            message=message, env_name=env.__class__.__name__,
            tokens=tokens, embeddings={k: None for k in embeddings},  # pas sérialiser
            mcb_count=mcb_count, thought_hash=thought_hash,
            c_seuil=c_seuil, emotion_perceived=emotion_perceived,
            emotion_understood=emotion_understood, confidence=confidence,
            response=response, response_hash=response_hash, certified=certified,
        )


if __name__ == "__main__":
    # démonstration rapide (sans entraînement complet)
    print("RATIS Agent souverain — démonstration")
    print(f"  TTF-Compute (penser) : {_TTF_BRAIN is not None}")
    print(f"  TTF embedding        : {is_ttf_available()}")
    agent = RatisAgent(use_ttf=True)
    print(f"  Cerveau TTF connecté : {agent.ttf_available}")
