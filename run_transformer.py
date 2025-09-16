
import os
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate

ARTIFACTS = "artifacts"
os.makedirs(ARTIFACTS, exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"

def main():
    # Amazon Polarity is a standard benchmark for sentiment (binary labels 0/1)
    ds_train = load_dataset("amazon_polarity", split="train[:2%]")
    ds_test  = load_dataset("amazon_polarity", split="test[:2%]")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tok(batch):
        return tokenizer(batch["content"], truncation=True, padding="max_length", max_length=128)

    train_enc = ds_train.map(tok, batched=True)
    test_enc  = ds_test.map(tok, batched=True)

    train_enc = train_enc.rename_column("label", "labels")
    test_enc  = test_enc.rename_column("label", "labels")
    cols = ["input_ids", "attention_mask", "labels"]
    train_enc.set_format(type="torch", columns=cols)
    test_enc.set_format(type="torch", columns=cols)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    metric = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": metric.compute(predictions=preds, references=labels)["accuracy"],
                "f1": f1.compute(predictions=preds, references=labels, average="macro")["f1"]}

    args = TrainingArguments(
        output_dir="hf_out",
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=5e-5,
        weight_decay=0.01,
        save_strategy="no",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_enc,
        eval_dataset=test_enc,
        compute_metrics=compute_metrics
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    with open(os.path.join(ARTIFACTS, "transformer_metrics.json"), "w") as f:
        json.dump(eval_metrics, f, indent=2)

    print("Transformer eval metrics:", json.dumps(eval_metrics, indent=2))
    print("Artifacts saved to:", os.path.abspath(ARTIFACTS))

if __name__ == "__main__":
    main()
