import os
import json
from datasets import load_dataset

raw = "data/raw"
train = os.path.join(raw, "wiki_train.jsonl")
validation = os.path.join(raw, "wiki_val.jsonl")

def prep_data():
    os.makedirs(raw, exist_ok=True)

    train_stream = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True
    ).filter(lambda x: len(x["text"].strip()) > 0)
    val_stream = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="validation", streaming=True
    ).filter(lambda x: len(x["text"].strip()) > 0)

    # Take ~25MB worth of training examples (~50k lines)
    target_mb = 25
    train_count = 0
    with open(train, "w") as f:
        for example in train_stream:
            f.write(json.dumps({"text": example["text"]}) + "\n")
            train_count += 1
            if os.path.getsize(train) >= target_mb * 1e6:
                break

    val_count = 0
    with open(validation, "w") as f:
        for example in val_stream:
            f.write(json.dumps({"text": example["text"]}) + "\n")
            val_count += 1

    train_mb = os.path.getsize(train) / 1e6
    val_mb = os.path.getsize(validation) / 1e6
    print(f"Train examples: {train_count}, size: {train_mb:.1f}MB")
    print(f"Val examples: {val_count}, size: {val_mb:.1f}MB")
    with open(train) as f:
        sample = json.loads(f.readline())["text"]
    print(f"Sample: {sample[:200]!r}")


if __name__ == "__main__":
    prep_data()