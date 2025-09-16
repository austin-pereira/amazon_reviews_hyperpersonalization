
import os
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from data_loader import load_datasets

ARTIFACTS = "artifacts"
os.makedirs(ARTIFACTS, exist_ok=True)

def train_and_eval(train_df, test_df, target_col: str, name: str):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(min_df=1, max_df=0.95, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=200))
    ])
    pipe.fit(train_df["text"], train_df[target_col])
    preds = pipe.predict(test_df["text"])
    acc = accuracy_score(test_df[target_col], preds)
    macro_f1 = f1_score(test_df[target_col], preds, average="macro")
    report = classification_report(test_df[target_col], preds, output_dict=True)
    cm = confusion_matrix(test_df[target_col], preds)

    # Save confusion matrix if it's sentiment
    if name == "sentiment":
        plt.figure()
        import itertools
        # basic plot (no specified colors per instructions)
        plt.imshow(cm, interpolation='nearest')
        plt.title("Confusion Matrix — Sentiment")
        plt.colorbar()
        tick_marks = np.arange(len(np.unique(test_df[target_col])))
        classes = sorted(np.unique(test_df[target_col]).tolist())
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(os.path.join(ARTIFACTS, "confusion_sentiment.png"), dpi=160)
        plt.close()

    return {
        "task": name,
        "acc": acc,
        "macro_f1": macro_f1,
        "report": report
    }, pipe

def cluster_and_plot(tfidf_matrix, labels_text):
    k = 3
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    clusters = km.fit_predict(tfidf_matrix)

    pca = PCA(n_components=2, random_state=42)
    XY = pca.fit_transform(tfidf_matrix.toarray())

    plt.figure()
    plt.scatter(XY[:,0], XY[:,1], s=30)  # no color specification
    for i, label in enumerate(labels_text):
        # keep annotation minimal to avoid clutter—plot a few
        if i < 20:
            plt.annotate(str(i), (XY[i,0], XY[i,1]))
    plt.title("K‑Means clusters (PCA projection)")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS, "clusters_pca.png"), dpi=160)
    plt.close()

def main():
    train_df, test_df = load_datasets()

    # Sentiment: binary (0/1)
    sentiment_metrics, sentiment_model = train_and_eval(train_df, test_df, "label_sentiment", "sentiment")

    # Persona: multi‑class ("eco","value","luxury")
    persona_metrics, persona_model = train_and_eval(train_df, test_df, "persona", "persona")

    # Clustering on TF‑IDF space over all texts (train + test)
    from sklearn.feature_extraction.text import TfidfVectorizer
    vect = TfidfVectorizer(min_df=1, max_df=0.95, ngram_range=(1,2))
    tfidf = vect.fit_transform(pd.concat([train_df["text"], test_df["text"]], axis=0))
    cluster_and_plot(tfidf, pd.concat([train_df["text"], test_df["text"]], axis=0).tolist())

    metrics = {
        "sentiment": {"acc": sentiment_metrics["acc"], "macro_f1": sentiment_metrics["macro_f1"]},
        "persona": {"acc": persona_metrics["acc"], "macro_f1": persona_metrics["macro_f1"]}
    }
    with open(os.path.join(ARTIFACTS, "baseline_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Baseline run complete. Metrics:", json.dumps(metrics, indent=2))
    print("Artifacts saved to:", os.path.abspath(ARTIFACTS))

if __name__ == "__main__":
    main()
