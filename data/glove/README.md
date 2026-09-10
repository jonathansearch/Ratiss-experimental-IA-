# GloVe Embeddings — Download Instructions

RATIS-Net uses pre-trained GloVe 50d embeddings (Stanford NLP, 400K words) as
its local semantic reservoir. The file is NOT committed (164 MB); download it
separately.

```bash
mkdir -p data/glove
curl -L -o data/glove/glove.6B.zip "https://nlp.stanford.edu/data/glove.6B.zip"
python3 -c "import zipfile; zipfile.ZipFile('data/glove/glove.6B.zip').extract('glove.6B.50d.txt', 'data/glove/')"
rm data/glove/glove.6B.zip
```

Source: https://nlp.stanford.edu/projects/glove/

The `glove_tokenizer.py` loads this file lazily on first use. If absent, it
falls back to topological signatures only (less accurate but functional).
