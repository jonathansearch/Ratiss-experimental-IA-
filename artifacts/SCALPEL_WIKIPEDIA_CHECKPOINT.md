# Scalpel Wikipedia Checkpoint

## Artifact identity

| Field | Value |
|---|---|
| File | `scalpel_wikipedia.pkl` |
| Storage | Git LFS |
| Exact byte size | `308,259,151` bytes |
| SHA-256 | `59d1aafda9da9f7b3a27276225ed04cbb5db834d1d286806926debc85ae06081` |
| Original Drive modified time | `2026-08-25T00:52:53.024Z` |

## Executed run record

This checkpoint was produced by the public `ratisnet_colab_training.ipynb` notebook. The notebook output recorded 5,000,000 streamed Wikipedia phrases, 3,782,801 displayed Scalpel neurons, 43,260,980 reinforcements, a 5.2-hour elapsed time, and a displayed 294.0 MB checkpoint size.

These values describe one recorded execution. They are not a benchmark claim, model-quality claim, or external validation.

## Reproduction and loading

After cloning the repository, retrieve the binary artifact with Git LFS before loading it:

```bash
git lfs pull
ls -lh artifacts/scalpel_wikipedia.pkl
```

The Colab notebook saves and resumes from the corresponding Drive location:

```text
/content/drive/MyDrive/ratisnet/scalpel_wikipedia.pkl
```

The SHA-256 value above can be used to verify that a downloaded artifact has not changed.
