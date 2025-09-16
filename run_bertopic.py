# run_bertopic.py
import os
import pandas as pd
from data_loader import load_datasets
from bertopic import BERTopic
import json

ARTIFACTS = "artifacts"
os.makedirs(ARTIFACTS, exist_ok=True)

def main():
    train_df, test_df = load_datasets()
    texts = pd.concat([train_df["text"], test_df["text"]], axis=0).tolist()

    topic_model = BERTopic(verbose=True)
    topics, probs = topic_model.fit_transform(texts)

    # Save topic info
    topic_info = topic_model.get_topic_info()
    topic_terms = {}
    for topic_id in topic_info["Topic"]:
        if topic_id == -1:
            continue
        topic_terms[str(topic_id)] = topic_model.get_topic(topic_id)

    # Write JSON + CSV
    with open(os.path.join(ARTIFACTS, "bertopic_topics.json"), "w") as f:
        json.dump(topic_terms, f, indent=2)

    topic_info.to_csv(os.path.join(ARTIFACTS, "bertopic_topic_info.csv"), index=False)

    print("Saved BERTopic results to artifacts/")

if __name__ == "__main__":
    main()
