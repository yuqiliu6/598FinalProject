import os
from typing import Dict

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np


# ------------------------------
# 1. Config
# ------------------------------
MODEL_NAME = "distilbert-base-uncased"  # or roberta-base, etc.
DATA_DIR = "data"               # where musique_*.json live
OUTPUT_DIR = "musique-hop-classifier"

# Label mapping: 2-hop, 3-hop, 4-hop
HOP_LABELS = {2: 0, 3: 1, 4: 2}
ID2LABEL = {0: "2-hop", 1: "3-hop", 2: "4-hop"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


# ------------------------------
# 2. Load MuSiQue data
# ------------------------------
def load_musique() -> DatasetDict:
    """
    Assumes you have JSON files like:
      - musique_ans_train.json
      - musique_ans_dev.json

    You can adjust file names / splits as needed.
    """
    data_files = {
        "train": "D:/CS598JH/finalproject/data/musique_ans_train.jsonl",
        "validation": "D:/CS598JH/finalproject/data/musique_ans_dev.jsonl",
    }
    ds = load_dataset("json", data_files=data_files)
    return ds


# ------------------------------
# 3. Add hop labels
# ------------------------------
def add_labels(example: Dict) -> Dict:
    """
    Turn example['num_hops'] into a classification label.

    If your JSON does not have num_hops directly, replace this with:
      - len(example['question_decomposition'])   OR
      - len(example['reasoning_graph'])         OR
      - some metadata field provided by MuSiQue
    """
    num_hops = example.get("id", None)
    num_hops = int(num_hops[0])

    if num_hops is None:
        # Example: if hop count is implicit in a list of sub-questions:
        # num_hops = len(example["sub_questions"])
        raise ValueError("id field not found; please adapt add_labels()")

    if num_hops not in HOP_LABELS:
        # Map 4+ hops to 4, or bucket them however you like
        if num_hops >= 4:
            label = HOP_LABELS[4]
        else:
            raise ValueError(f"Unexpected num_hops: {num_hops}")
    else:
        label = HOP_LABELS[num_hops]

    example["labels"] = label
    return example


# ------------------------------
# 4. Tokenization
# ------------------------------
def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["question"],        # adjust if field name differs
        truncation=True,
        padding="max_length",
        max_length=128,
    )


# ------------------------------
# 5. Metrics
# ------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = (preds == labels).mean()
    return {
        "accuracy": float(accuracy),
    }


def main():
    # 1) Load dataset
    ds = load_musique()
    dev = ds["validation"]
    for key in dev.features:
        print(key)
    print(dev.features["question"])
    return


    # 2) Add hop labels
    ds = ds.map(add_labels)
    for key in ds["train"].features:
        print(key)

    # 3) Load tokenizer and tokenize
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # 4) Set format for PyTorch
    ds = ds.remove_columns(
        [col for col in ds["train"].column_names if col not in ["input_ids", "attention_mask", "labels"]]
    )
    ds.set_format("torch")

    # 5) Create model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(HOP_LABELS),
        id2label=ID2LABEL,
        label2id={v: k for k, v in ID2LABEL.items()},
    )

    # 6) Training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=100,
        save_total_limit=2,
    )

    # 7) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 8) Train
    trainer.train()

    # 9) Save final model + tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Training finished. Model saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
