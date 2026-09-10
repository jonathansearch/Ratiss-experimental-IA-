"""Generate the Colab/Kaggle training notebook for RATIS-Net Scalpel."""
import json
from pathlib import Path

nb = {"nbformat": 4, "nbformat_minor": 0,
      "metadata": {"colab": {"provenance": []},
                    "kernelspec": {"name": "python3", "display_name": "Python 3"},
                    "language_info": {"name": "python"}},
      "cells": []}

def md(text):
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in text.split("\n")]})

def code(src):
    nb["cells"].append({"cell_type": "code", "metadata": {}, "source": [l + "\n" for l in src.split("\n")], "execution_count": None, "outputs": []})

md("# RATIS-Net — Scalpel Training (Wikipedia Streaming)\n\nStreams 5M+ Wikipedia phrases into the Scalpel via Hugging Face Datasets. CPU-only, no GPU needed. Saves checkpoints to Google Drive for resumption across Colab sessions.")

md("## 1. Install dependencies")
code("!pip install -q numpy datasets")

md("## 2. Clone RATIS-Net repo")
code("!git clone https://github.com/evinajonathan13-max/Ratiss-experimental-IA-.git ratisnet_repo\nimport sys; sys.path.insert(0, '/content/ratisnet_repo')")

md("## 3. Download GloVe (171 MB, one-time)")
code("import os, zipfile, urllib.request\nos.makedirs('/content/data/glove', exist_ok=True)\nzip_path = '/content/data/glove/glove.6B.zip'\nif not os.path.exists('/content/data/glove/glove.6B.50d.txt'):\n    print('Downloading GloVe...')\n    urllib.request.urlretrieve('https://nlp.stanford.edu/data/glove.6B.zip', zip_path)\n    with zipfile.ZipFile(zip_path) as z:\n        z.extract('glove.6B.50d.txt', '/content/data/glove/')\n    os.remove(zip_path)\nprint('GloVe ready')")

md("## 4. Mount Google Drive (for checkpoint persistence)")
code("from google.colab import drive\ndrive.mount('/content/drive')\nos.makedirs('/content/drive/MyDrive/ratisnet', exist_ok=True)\nprint('Drive mounted')")

md("## 5. Run Wikipedia streaming to Scalpel")
code("""import sys, time, os
from pathlib import Path
sys.path.insert(0, '/content/ratisnet_repo')

from ratis_net.glove_tokenizer import GloveTokenizer
from ratis_net.scalpel import ScalpelLayer
from ratis_net.data_loader import StreamingDataLoader
from ratis_net.topo_cache import TopoCache

MAX_PHRASES = 5000000
CHECKPOINT_EVERY = 10000
DRIVE_PATH = Path('/content/drive/MyDrive/ratisnet/scalpel_wikipedia.pkl')

# Tokenizer (GloVe + topo cache from repo)
tok = GloveTokenizer(dim=12, n_glove=8)
tok._topo_cache = TopoCache(dim=8)
# Load topo cache from repo data
import numpy as np
cache_repo = Path('/content/ratisnet_repo/data/cache/topo_signatures.npz')
if cache_repo.exists():
    data = np.load(cache_repo, allow_pickle=True)
    tok._topo_cache._mem = {str(data['words'][i]): data['embeddings'][i] for i in range(len(data['words']))}
    print(f'Topo cache: {len(tok._topo_cache._mem)} words')

# Scalpel (resume if checkpoint exists)
scalpel = ScalpelLayer(tok, eta=0.1, coherence_threshold=0.3)
if DRIVE_PATH.exists():
    scalpel.load(DRIVE_PATH)
    print(f'Resuming: {scalpel.network_size()} neurons already learned')
else:
    print('Starting from scratch')

# Stream Wikipedia
loader = StreamingDataLoader(dataset='wikipedia', config='20231101.en', max_phrases=MAX_PHRASES)
print(f'Streaming {MAX_PHRASES} Wikipedia phrases to Scalpel...')
print()

t0 = time.time()
n = 0
for phrase in loader:
    scalpel.process_phrase(phrase, t_step=n)
    n += 1
    if n % 1000 == 0:
        dt = time.time() - t0
        rate = n / dt if dt > 0 else 0
        eta_h = (MAX_PHRASES - n) / rate / 3600 if rate > 0 else 0
        print(f'  {n:>8d} | {scalpel.network_size():>8d} neurons | {rate:.0f} ph/s | ETA {eta_h:.1f}h')
    if n % CHECKPOINT_EVERY == 0:
        scalpel.save(DRIVE_PATH)
        print(f'  >>> Checkpoint saved ({scalpel.network_size()} neurons)')

scalpel.save(DRIVE_PATH)
dt = time.time() - t0
print(f'\\n=== DONE ===')
print(f'Phrases: {n}')
print(f'Neurons: {scalpel.network_size()}')
print(f'Reinforcements: {scalpel.total_reinforcements}')
print(f'Time: {dt/3600:.1f}h')
print(f'Size: {os.path.getsize(DRIVE_PATH)/1024/1024:.1f} MB')
""")

md("## 6. Test reconstruction (Synchrotron)")
code("""from ratis_net.ratiss_synchrotron import RatissSynchrotron

# Build a small index for testing
test_corpus = []
for i, phrase in enumerate(loader):
    if i >= 5000: break
    test_corpus.append(phrase)

engine = RatissSynchrotron(scalpel=scalpel, scalpel_weight=0.4)
engine.build_corpus(test_corpus)

queries = ['what is quantum mechanics', 'i feel happy today', 'explain gravity']
for q in queries:
    r = engine.generate_response(q)
    rec = r['reconstruction']
    print(f'Q: {q}')
    print(f'  R: {rec[\"reconstructed\"][:120]}')
    print(f'  coh={rec[\"avg_coherence\"]:.3f}')
    print()
""")

md("## Notes\n- Colab free: 12h max per session. Checkpoints save to Drive every 10K phrases.\n- On timeout: re-run cell 5, Scalpel resumes from last checkpoint.\n- For Kaggle: replace `/content/drive/MyDrive/` with `/kaggle/working/`.\n- For Lightning AI: replace with `/teamspace/studios/this_studio/`.")

out = Path(__file__).resolve().parent / "ratisnet_colab_training.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Notebook: {out} ({out.stat().st_size} bytes)")
