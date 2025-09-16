
import os
import json
from datetime import datetime

ARTIFACTS = "artifacts"
os.makedirs(ARTIFACTS, exist_ok=True)

def main():
    baseline_path = os.path.join(ARTIFACTS, "baseline_metrics.json")
    transformer_path = os.path.join(ARTIFACTS, "transformer_metrics.json")

    baseline = {}
    transformer = {}

    if os.path.exists(baseline_path):
        with open(baseline_path, "r") as f:
            baseline = json.load(f)
    if os.path.exists(transformer_path):
        with open(transformer_path, "r") as f:
            transformer = json.load(f)

    lines = []
    lines.append(f"# Hyperpersonalization from Text — Amazon Reviews Mini‑Study")
    lines.append(f"_Generated: {datetime.utcnow().isoformat()}Z_")
    lines.append("")
    lines.append("## Setup")
    lines.append("- **Baseline**: TF‑IDF + Logistic Regression")
    lines.append("- **Transformer**: DistilBERT fine‑tune (Amazon Polarity)")
    lines.append("- **Extras**: K‑Means clustering on TF‑IDF + PCA visualization")
    lines.append("")

    if baseline:
        lines.append("## Baseline Results")
        lines.append(f"- Sentiment: acc={baseline['sentiment']['acc']:.3f}, macro‑F1={baseline['sentiment']['macro_f1']:.3f}")
        lines.append(f"- Persona:   acc={baseline['persona']['acc']:.3f}, macro‑F1={baseline['persona']['macro_f1']:.3f}")
        lines.append("![Confusion Matrix](confusion_sentiment.png)")
        lines.append("![Clusters (PCA)](clusters_pca.png)")
        lines.append("")
    else:
        lines.append("## Baseline Results")
        lines.append("_Not found. Run `python run_baseline.py` first._")
        lines.append("")

    if transformer:
        lines.append("## Transformer Results")
        lines.append(f"- DistilBERT: accuracy={transformer.get('eval_accuracy','?')}, macro‑F1={transformer.get('eval_f1','?')}")
        lines.append("")
    else:
        lines.append("## Transformer Results")
        lines.append("_Not found. Run `python run_transformer.py` (internet required)._")
        lines.append("")

    lines.append("## Discussion")
    lines.append("- Baseline provides interpretable features for marketing teams; Transformers improve context‑sensitivity for nuanced sentiment/intent.")
    lines.append("- Clusters suggest micro‑segments by language style (e.g., eco vs luxury vs value), enabling creative + offer personalization.")
    lines.append("")
    lines.append("## Future Work")
    lines.append("- Replace weak persona labels with supervised labels (surveys or purchase metadata).")
    lines.append("- Add BERTopic for robust topic discovery over categories/brands.")
    lines.append("- Evaluate uplift from email/ad personalization in online A/B tests.")
    lines.append("")

    with open(os.path.join(ARTIFACTS, "report.md"), "w") as f:
        f.write("\n".join(lines))

    print("Report written to artifacts/report.md")

    # At the bottom of make_report.py, after other sections
    bertopic_path = os.path.join(ARTIFACTS, "bertopic_topic_info.csv")
    if os.path.exists(bertopic_path):
        lines.append("## BERTopic — Topic Discovery")
        lines.append("")
        lines.append("Top discovered topics with keywords:")
        import pandas as pd
        df = pd.read_csv(bertopic_path)
        for _, row in df.head(5).iterrows():
            lines.append(f"- **Topic {row['Topic']}**: {row['Name']} (Size={row['Count']})")
        lines.append("")

if __name__ == "__main__":
    main()
