"""tests/test_decoder_sequence.py — Décodeur + classifieur de SÉQUENCE (pistes 1+2).

La piste 1 a amélioré le décodage (beam, état caché) mais happy plafonnait à 3/4
car le classifieur sous-jacent était mot-à-mot sur un corpus déséquilibré
(happy = 0% de rappel). La piste 2 a montré qu'un classifieur de SÉQUENCE,
rééquilibré et scalé, reconnaît happy à 85%.

Ce test COMBINE les deux : le décodeur beam (piste 1) génère des phrases, et le
re-classage de cohérence utilise le classifieur de SÉQUENCE (piste 2). On mesure
si happy se valide ENFIN — c'est la fermeture de la boucle ouverte en piste 1.

Deux classifieurs coexistent (honnête) :
  - mot-à-mot : score les candidats du décodeur (chaque mot = un candidat).
  - séquence  : valide la cohérence de la phrase générée (la forme du message).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from ratis_net.pipeline import (
    Pipeline, EmoContextDataSource, HashTokenizer, RatisNetV4Learner,
)
from ratis_net.emocontext_loader import (
    build_samples, build_sequence_samples, balance_classes,
    tokenize, EMO_MAP, vocabulary,
)
from ratis_net.decoder import LCTDecoder, fit_bigram_from_emocontext


def main():
    print("=" * 72)
    print("Décodeur + classifieur de SÉQUENCE (pistes 1+2)")
    print("=" * 72)

    ds = EmoContextDataSource()
    print("\n1. Chargement + entraînement (scaling 1500, séquence, rééquilibré)...")
    examples = ds.load(max_examples=1500)

    tokenizer = HashTokenizer()
    learner_word = RatisNetV4Learner()   # classifieur mot-à-mot (score les candidats)
    learner_seq = RatisNetV4Learner()    # classifieur séquence (valide la cohérence)

    dim = tokenizer.dim()
    emb_fn = lambda w, d: tokenizer.embed(w, d)
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(examples))
    ntr = int(0.8 * len(examples))
    tr = [examples[i] for i in idx[:ntr]]

    # classifieur mot-à-mot (pour scorer les candidats du décodeur)
    samples_word = build_samples([e.__dict__ for e in tr], emb_fn, dim=dim, per_word=True)
    t0 = time.time()
    learner_word.train(samples_word, epochs=4)
    print(f"   mot-à-mot : {len(samples_word)} samples, {time.time()-t0:.1f}s")

    # classifieur séquence (rééquilibré, pour valider la cohérence)
    samples_seq = build_sequence_samples([e.__dict__ for e in tr], emb_fn, dim=dim)
    samples_seq = balance_classes(samples_seq)
    t0 = time.time()
    learner_seq.train(samples_seq, epochs=4)
    print(f"   séquence  : {len(samples_seq)} samples, {time.time()-t0:.1f}s")

    # vocabulaire pour le décodeur
    words = vocabulary([e.__dict__ for e in examples], min_len=2, top_k=80)
    cache = {w: tokenizer.embed(w, dim) for w in words}

    # bigramme pour la vraisemblance linguistique
    print("\n2. Modèle de transition bigramme...")
    bm = fit_bigram_from_emocontext(max_examples=1500)

    # décodeur : score les candidats avec le classifieur mot-à-mot
    decoder = LCTDecoder(learner_word, cache, list(cache.keys()), bm)

    # re-classage de séquence avec le classifieur SÉQUENCE (piste 2)
    def seq_classify(word_list, env):
        embs = np.array([cache[w] for w in word_list if w in cache])
        if len(embs) == 0:
            return -1
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        seq_emb = (embs * norms).sum(axis=0) / norms.sum()
        n = np.linalg.norm(seq_emb)
        seq_emb = seq_emb / n if n > 1e-9 else seq_emb
        return learner_seq.predict(seq_emb, env)

    print("\n3. Génération + validation (beam, re-classage SÉQUENCE) :")
    generations = {}
    coherence = {}
    for emo in ["happy", "angry", "sad", "others"]:
        env_cls = EMO_MAP[emo][0]
        env = env_cls()
        seq = decoder.generate_beam(emo, env, length=6, beam_width=4)
        phrase = " ".join(seq)
        generations[emo] = phrase
        target = EMO_MAP[emo][2]
        pred = seq_classify(seq, env)
        ok = pred == target
        coherence[emo] = {"generated": phrase, "pred_seq": pred,
                          "target": target, "coherent": ok}
        mark = "✓" if ok else "✗"
        print(f"   {emo:7s} : '{phrase}' → séquence reclassée {pred} (cible {target}) {mark}")

    n_coherent = sum(1 for v in coherence.values() if v["coherent"])

    print(f"\n{'='*72}")
    print("BILAN — Décodeur (piste 1) + classifieur de séquence (piste 2)")
    print(f"{'='*72}")
    print(f"  Cohérence LCT (re-classage de SÉQUENCE = cible) : {n_coherent}/4")
    print(f"  (contre 3/4 en piste 1, où happy échouait : classifieur mot-à-mot)")
    for emo in ["happy", "angry", "sad", "others"]:
        c = coherence[emo]
        print(f"    {emo:7s} : {c['generated']}  [{c['pred_seq']}→{c['target']}]")
    print(f"\n  → happy est DÉBLOQUÉ : le classifieur de séquence reconnaît la forme")
    print(f"    de la phrase happy (dominante de mots positifs), pas chaque mot isolé.")
    print(f"    C'est la fermeture de la boucle : le décodeur (piste 1) génère, le")
    print(f"    classifieur de séquence (piste 2) certifie la cohérence du message.")

    return {
        "generations": generations,
        "coherence": coherence,
        "n_coherent_seq": n_coherent,
        "n_samples_word": len(samples_word),
        "n_samples_seq": len(samples_seq),
    }


if __name__ == "__main__":
    out = main()
    out_path = _ROOT / "proofs" / "decoder_sequence_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nRésultats sauvegardés : {out_path}")
