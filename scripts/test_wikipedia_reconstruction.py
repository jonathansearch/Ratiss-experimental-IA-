"""Test reconstruction with the full Wikipedia-trained Scalpel (3.78M neurons).

Loads artifacts/scalpel_wikipedia.pkl and runs Synchrotron reconstruction
queries to verify the network produces coherent responses.

Usage: PYTHONPATH=. python scripts/test_wikipedia_reconstruction.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ratis_net.glove_tokenizer import GloveTokenizer
from ratis_net.scalpel import ScalpelLayer
from ratis_net.ratiss_synchrotron import RatissSynchrotron

CHECKPOINT = Path(__file__).resolve().parents[1] / "artifacts" / "scalpel_wikipedia.pkl"

DEMO_CORPUS = [
    "quantum mechanics describes the behavior of matter at atomic scale",
    "the brain processes information through neural networks",
    "gravity is a fundamental force that bends spacetime",
    "artificial intelligence learns patterns from data",
    "i am happy to see you today",
    "the weather is beautiful and i feel good",
    "entropy measures the disorder of a physical system",
    "consciousness emerges from complex neural interactions",
    "the universe is expanding at an accelerating rate",
    "topology studies the properties of shapes under deformation",
    "neural networks learn by adjusting their weights",
    "the speed of light is constant in vacuum",
    "love is a complex emotion that connects people",
    "science seeks to understand the laws of nature",
    "music evokes deep emotions through rhythm and melody",
]


def main():
    tok = GloveTokenizer(dim=12, n_glove=8)

    print("Loading Wikipedia Scalpel checkpoint...")
    scalpel = ScalpelLayer(tok, eta=0.1, coherence_threshold=0.3)
    scalpel.load(CHECKPOINT)
    print(f"  {scalpel.network_size():,} neurons, {scalpel.total_reinforcements:,} reinforcements")

    # Show correlations for key words
    print("\n=== Correlations for key words ===")
    for w in ["quantum", "science", "love", "gravity", "brain", "happy"]:
        corrs = scalpel.get_correlations(w)[:3]
        print(f"\n  {w}:")
        for word, weight, p_sig in corrs:
            print(f"    -> {word:15s} weight={weight:.2f} P_sig={p_sig:.4f}")

    # Build Synchrotron index with the Scalpel connected
    print("\n\n=== Building Synchrotron index ===")
    engine = RatissSynchrotron(scalpel=scalpel, scalpel_weight=0.4)
    t0 = time.time()
    engine.build_corpus(DEMO_CORPUS)
    print(f"  {len(engine.index.fragments)} fragments indexed in {time.time()-t0:.1f}s")

    # Run reconstruction queries
    queries = [
        "what is quantum mechanics",
        "how does the brain work",
        "i feel happy today",
        "explain gravity",
        "what is consciousness",
        "how does love work",
        "what is science",
    ]

    print("\n=== Reconstruction (Scalpel-boosted) ===")
    for q in queries:
        result = engine.generate_response(q)
        rec = result["reconstruction"]
        print(f"\nQ: {q}")
        print(f"  R: {rec['reconstructed'][:150]}")
        print(f"  coherence={rec['avg_coherence']:.3f} tension={rec['avg_tension']:.3f} "
              f"fragments={rec['n_fragments']}")
        for f in rec["selected_fragments"][:2]:
            print(f"    [{f['tension']:.3f}] {f['text'][:80]}")


if __name__ == "__main__":
    main()
