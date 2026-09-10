"""tests/test_decoder_ttf.py — Piste 3 : décodeur qui pense avec les MCB (TTF).

Le décodeur (pistes 1+2) scorait les mots candidats via un embedding HASH.
La piste 3 le branche sur le cerveau TTF-Compute : chaque mot est représenté
par les MCB (Mémoire de Corrélation Bit) du cerveau, pas par un hash
aléatoire. Le réseau "pense" alors avec la topologie RÉELLE de la donnée —
la pensée sans mots — et génère du langage à partir de cette pensée.

On compare le décodeur TTF (MCB) au décodeur HASH, sur la cohérence LCT
(re-classage de séquence = cible). La piste 2 avait montré que le tokenizer
TTF généralise mieux que le hash (0.983 vs 0.758 sur le non-vu). Ici on
valide que cette supériorité topologique se traduit en GÉNÉRATION : les
phrases produites à partir des MCB sont plus cohérentes.

Flux : mot → cerveau TTF (oscillation + Rips + MCB) → embedding → réseau LCT
       (classifieur de séquence) → décodeur beam → phrase → re-classage.
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
    Pipeline, EmoContextDataSource, HashTokenizer, TTFTokenizer, RatisNetV4Learner,
)
from ratis_net.emocontext_loader import (
    build_samples, build_sequence_samples, balance_classes,
    tokenize, EMO_MAP, vocabulary,
)
from ratis_net.ttf_bridge import is_ttf_available, ttf_embedding
from ratis_net.decoder import LCTDecoder, fit_bigram_from_emocontext


def train_and_decode(tokenizer, examples, label, epochs=3, n_vocab=40):
    """Entraîne un classifieur de séquence (piste 2) + génère (piste 1).

    Le TTF est ~500ms/mot : on précalcule un CACHE d'embeddings une fois (par
    mot unique du vocabulaire), puis build_samples puise dans le cache. Sans
    ça, le mot-à-mot recalcule l'embedding à chaque occurrence → des milliers
    d'appels TTF.
    """
    dim = tokenizer.dim()
    # vocabulaire + cache d'embeddings précalculés (TTF coûteux → une fois)
    words = vocabulary([e.__dict__ for e in examples], min_len=2, top_k=n_vocab)
    cache = {w: tokenizer.embed(w, dim) for w in words}
    # emb_fn puise le cache ; fallback (mot hors top-k) calcule à la demande
    emb_fn = lambda w, d: cache.get(w, tokenizer.embed(w, d))

    rng = np.random.RandomState(42)
    idx = rng.permutation(len(examples))
    ntr = int(0.8 * len(examples))
    tr = [examples[i] for i in idx[:ntr]]

    # classifieur mot-à-mot (score les candidats du décodeur) — puise le cache.
    # On ne garde que les mots du vocabulaire (top-k) : le mot-à-mot parcourt
    # tous les mots des dialogues, dont beaucoup hors top-k → fallback TTF lent.
    learner_word = RatisNetV4Learner()
    samples_word = []
    for e in tr:
        ws = tokenize(e.turn1 + " " + e.turn2 + " " + e.turn3)
        seen = set()
        for w in ws:
            if w in seen or len(w) < 2 or w not in cache:
                continue
            seen.add(w)
            samples_word.append((cache[w], e.env, e.label_num, e.c_seuil))
    t0 = time.time()
    learner_word.train(samples_word, epochs=epochs)
    t_word = time.time() - t0

    # classifieur séquence (rééquilibré, valide la cohérence) — puise le cache
    learner_seq = RatisNetV4Learner()
    samples_seq = []
    for e in tr:
        ws = [w for w in tokenize(e.turn3) if w in cache]
        if len(ws) < 2:
            continue
        embs = np.array([cache[w] for w in ws])
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        seq_emb = (embs * norms).sum(axis=0) / norms.sum()
        n = np.linalg.norm(seq_emb)
        seq_emb = seq_emb / n if n > 1e-9 else seq_emb
        samples_seq.append((seq_emb, e.env, e.label_num, e.c_seuil))
    samples_seq = balance_classes(samples_seq)
    t0 = time.time()
    learner_seq.train(samples_seq, epochs=epochs)
    t_seq = time.time() - t0
    bm = fit_bigram_from_emocontext(max_examples=3000)

    decoder = LCTDecoder(learner_word, cache, list(cache.keys()), bm)

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

    generations, coherence = {}, {}
    for emo in ["happy", "angry", "sad", "others"]:
        env_cls = EMO_MAP[emo][0]
        env = env_cls()
        seq = decoder.generate_beam(emo, env, length=6, beam_width=4)
        phrase = " ".join(seq)
        generations[emo] = phrase
        target = EMO_MAP[emo][2]
        pred = seq_classify(seq, env)
        coherence[emo] = {"generated": phrase, "pred_seq": pred,
                          "target": target, "coherent": pred == target}

    n_coh = sum(1 for v in coherence.values() if v["coherent"])
    print(f"\n  [{label}] train mot={t_word:.0f}s seq={t_seq:.1f}s | "
          f"cohérence {n_coh}/4")
    for emo in ["happy", "angry", "sad", "others"]:
        c = coherence[emo]
        mark = "✓" if c["coherent"] else "✗"
        print(f"    {emo:7s} : {c['generated']}  [{c['pred_seq']}→{c['target']}] {mark}")
    return {"label": label, "generations": generations, "coherence": coherence,
            "n_coherent": n_coh, "t_word": t_word, "t_seq": t_seq,
            "n_vocab": len(cache)}


def main():
    print("=" * 72)
    print("Piste 3 — Décodeur qui pense avec les MCB du cerveau TTF")
    print("=" * 72)
    print(f"  TTF-Compute disponible : {is_ttf_available()}")

    ds = EmoContextDataSource()
    n = 400  # taille réduite pour rester rapide (le TTF est ~500ms/mot)
    print(f"\n  Chargement de {n} dialogues...")
    examples = ds.load(max_examples=n)

    # A) HASH (baseline des pistes 1+2)
    print("\n1. Décodeur HASH (baseline pistes 1+2)...")
    res_hash = train_and_decode(HashTokenizer(), examples, "HASH", epochs=3, n_vocab=40)

    # B) TTF (MCB du cerveau) — piste 3
    print("\n2. Décodeur TTF/MCB (piste 3)...")
    if not is_ttf_available():
        print("  TTF-Compute non disponible — piste 3 sautée (le cerveau AEON doit")
        print("  être cloné à côté).")
        res_ttf = None
    else:
        res_ttf = train_and_decode(TTFTokenizer(), examples, "TTF/MCB", epochs=3, n_vocab=30)

    print(f"\n{'='*72}")
    print("BILAN PISTE 3")
    print(f"{'='*72}")
    print(f"  Cohérence LCT (re-classage de séquence = cible) :")
    print(f"    HASH     : {res_hash['n_coherent']}/4")
    if res_ttf:
        print(f"    TTF/MCB  : {res_ttf['n_coherent']}/4")
    print(f"\n  Phrases générées :")
    print(f"    [HASH]")
    for emo in ["happy", "angry", "sad", "others"]:
        print(f"      {emo:7s} : {res_hash['generations'][emo]}")
    if res_ttf:
        print(f"    [TTF/MCB]")
        for emo in ["happy", "angry", "sad", "others"]:
            print(f"      {emo:7s} : {res_ttf['generations'][emo]}")
    print(f"\n  → La piste 2 avait montré que le tokenizer TTF généralise mieux que")
    print(f"    le hash (0.983 vs 0.758 non-vu). La piste 3 vérifie si cette")
    print(f"    supériorité topologique se traduit en GÉNÉRATION cohérente : le")
    print(f"    décodeur nourri par les MCB (pensée sans mots) du cerveau TTF.")

    return {"hash": res_hash, "ttf": res_ttf}


if __name__ == "__main__":
    out = main()
    out_path = _ROOT / "proofs" / "decoder_ttf_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nRésultats sauvegardés : {out_path}")
