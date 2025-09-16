
# Amazon Reviews NLP — Hyperpersonalization Mini-Study

This repo demonstrates two algorithmic approaches on Amazon-style product reviews:

1) **Classic NLP Baseline** — TF‑IDF + Logistic Regression for sentiment and micro‑persona cues.  
2) **Transformer Fine‑Tune** — DistilBERT (Hugging Face) for sentiment. *(Runs when internet is available.)*

**Optional Analysis**
- Customer language **clustering** with K‑Means on TF‑IDF vectors.
- 2D visualization via PCA.

> This project is structured so you can **run end‑to‑end on a small sample locally** and then **swap in the full Amazon dataset** (e.g., `amazon_polarity` on Hugging Face) for your paper‐grade results.

## Quickstart (Local / Colab)

```bash
# (Recommended) Python 3.10+
pip install -U scikit-learn pandas numpy matplotlib tqdm datasets transformers accelerate torch

python run_baseline.py           # runs TF‑IDF + Logistic Regression + clustering on a sample dataset
python run_transformer.py        # fine‑tunes DistilBERT on sentiment (requires internet to download model/data)
python make_report.py            # compiles metrics artifacts into a markdown report
```

## Swap in Real Amazon Data

Edit `data_loader.py` to set `USE_HF = True` to fetch `amazon_polarity` via `datasets`:
```python
USE_HF = True
SPLIT_TRAIN = "train[:50%]"
SPLIT_TEST  = "test[:10%]"
```

## Outputs
- `artifacts/baseline_metrics.json` — accuracy/F1 for sentiment & persona.
- `artifacts/confusion_sentiment.png` — Confusion matrix for sentiment.
- `artifacts/clusters_pca.png` — 2D PCA visualization of K‑Means clusters.
- `artifacts/transformer_metrics.json` — metrics from DistilBERT (if run).
- `artifacts/report.md` — publication‑style report draft (auto‑generated).

## Suggested Paper Outline
See `paper_outline.md`.
