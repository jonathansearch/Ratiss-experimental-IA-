"""ratis_net.data_loader — Streaming de corpus massifs vers le Scalpel.

Utilise Hugging Face Datasets en mode streaming pour alimenter le Scalpel
phrase par phrase, sans saturer le disque. Au lieu de télécharger 20 GB de
Wikipedia, on stream : chaque phrase est traitée puis jetée.

Datasets supportés :
  - wikipedia (20220301.en) : langage encyclopédique, neutre
  - c4 (en) : web nettoyé, culture générale
  - oscar (fr) : multilingue

Usage :
  loader = StreamingDataLoader(dataset="wikipedia", max_phrases=50000)
  scalpel.train_corpus_stream(loader, verbose=True)
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def split_sentences(text: str, min_words: int = 4) -> list[str]:
    """Découpe un texte brut en phrases d'au moins min_words mots."""
    if not text:
        return []
    # split sur les points, points d'exclamation, points d'interrogation
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if len(s.split()) >= min_words]


class StreamingDataLoader:
    """Stream un dataset Hugging Face phrase par phrase.

    Ne télécharge rien sur le disque : chaque exemple est traité puis jeté.
    """

    def __init__(self, dataset: str = "wikipedia", config: str = "20220301.en",
                 max_phrases: int = 100000, split: str = "train",
                 min_words: int = 4):
        self.dataset_name = dataset
        self.config = config
        self.max_phrases = max_phrases
        self.split = split
        self.min_words = min_words

    def __iter__(self) -> Iterator[str]:
        """Yield des phrases une par une depuis le dataset streamé."""
        from datasets import load_dataset

        # HF a migré wikipedia vers wikimedia/wikipedia (nouveau format parquet)
        hf_name = f"wikimedia/{self.dataset_name}" if self.dataset_name == "wikipedia" else self.dataset_name
        ds = load_dataset(hf_name, self.config,
                          split=self.split, streaming=True, trust_remote_code=True)
        count = 0
        for example in ds:
            text = example.get("text", "")
            sentences = split_sentences(text, min_words=self.min_words)
            for sent in sentences:
                if count >= self.max_phrases:
                    return
                yield sent
                count += 1

    def estimate_size(self) -> str:
        """Estime la taille approximative du dataset."""
        sizes = {"wikipedia": "~20 GB", "c4": "~300 GB",
                 "oscar": "~1 TB", "the_pile": "~800 GB"}
        return sizes.get(self.dataset_name, "unknown")


class ScalpelStreamingTrainer:
    """Entraîne le Scalpel en streaming depuis un dataset massif.

    La neurogenesis se fait phrase par phrase : pas de fichier à stocker,
    pas de batch. Chaque phrase est découpée, corrélée, puis jetée.
    """

    def __init__(self, scalpel, loader: StreamingDataLoader):
        self.scalpel = scalpel
        self.loader = loader

    def train(self, checkpoint_every: int = 5000, verbose: bool = True) -> dict:
        """Entraîne le Scalpel en streaming.

        Sauvegarde un checkpoint tous les `checkpoint_every` phrases pour
        permettre la reprise.
        """
        t0 = time.time()
        n_phrases = 0
        for phrase in self.loader:
            self.scalpel.process_phrase(phrase, t_step=n_phrases)
            n_phrases += 1
            if verbose and n_phrases % 1000 == 0:
                dt = time.time() - t0
                rate = n_phrases / dt if dt > 0 else 0
                print(f"  {n_phrases:>8d} phrases | {self.scalpel.network_size():>7d} neurones "
                      f"| {rate:.0f} ph/s | {dt:.0f}s")
            if checkpoint_every and n_phrases % checkpoint_every == 0:
                self.scalpel.save(Path("data/scalpel_checkpoint.pkl"))
        dt = time.time() - t0
        return {
            "n_phrases": n_phrases,
            "n_neurons": self.scalpel.network_size(),
            "n_reinforcements": self.scalpel.total_reinforcements,
            "n_neurogenesis": self.scalpel.total_neurogenesis,
            "time_seconds": dt,
            "phrases_per_second": n_phrases / dt if dt > 0 else 0,
        }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Stream a dataset into the Scalpel.")
    ap.add_argument("--dataset", default="wikipedia", help="wikipedia, c4, oscar")
    ap.add_argument("--config", default="20220301.en", help="dataset config")
    ap.add_argument("--max-phrases", type=int, default=10000,
                    help="max phrases to process")
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--coherence-threshold", type=float, default=0.3)
    ap.add_argument("--checkpoint-every", type=int, default=5000)
    args = ap.parse_args()

    from ratis_net.glove_tokenizer import GloveTokenizer
    from ratis_net.scalpel import ScalpelLayer

    tok = GloveTokenizer(dim=12, n_glove=8)
    scalpel = ScalpelLayer(tok, eta=args.eta,
                           coherence_threshold=args.coherence_threshold)

    loader = StreamingDataLoader(dataset=args.dataset, config=args.config,
                                 max_phrases=args.max_phrases)
    print(f"Dataset: {args.dataset} ({loader.estimate_size()})")
    print(f"Max phrases: {args.max_phrases}")
    print(f"Streaming vers le Scalpel...")
    print()

    trainer = ScalpelStreamingTrainer(scalpel, loader)
    result = trainer.train(checkpoint_every=args.checkpoint_every, verbose=True)

    print(f"\n=== Résultat ===")
    print(f"Phrases traitées : {result['n_phrases']}")
    print(f"Neurones générés : {result['n_neurons']}")
    print(f"Renforcements LCT : {result['n_reinforcements']}")
    print(f"Neurogenesis      : {result['n_neurogenesis']}")
    print(f"Temps             : {result['time_seconds']:.0f}s")
    print(f"Vitesse           : {result['phrases_per_second']:.0f} ph/s")

    scalpel.save(Path("data/scalpel_streamed.pkl"))
    import os
    print(f"Réseau sauvegardé : {os.path.getsize('data/scalpel_streamed.pkl')/1024:.0f} KB")

    print(f"\n=== Top 10 corrélations ===")
    for a, b, w in scalpel.strongest_correlations(top_k=10):
        print(f"  {a:15s} <-> {b:15s}  poids={w:.4f}")
