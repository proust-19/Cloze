import os
import json
from datasets import load_dataset

raw = "data/raw"
train = os.path.join(raw, "wiki_train.jsonl")
validation = os.path.join(raw, "wiki_val.jsonl")
tar_mb = 25

def prep_data():
    os.makedirs(raw, exist_ok=True)

    train_stream = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True
    ).filter(lambda x: len(x["text"].strip()) > 0)
    val_stream = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="validation", streaming=True
    ).filter(lambda x: len(x["text"].strip()) > 0)

    # Take ~25MB worth of training examples (~50k lines)
    tar_mb = 25
    train_count = 0
    bytes_wri = 0

    with open(train, "w", encoding="utf-8") as f:
        for example in train_stream:
            line = json.dumps({"text": example["text"]}) + "\n"
            f.write(line)
            bytes_wri += len(line.encode("utf-8"))
            train_count += 1
            if bytes_wri >= tar_mb * 1e6:
                break

    val_count = 0
    with open(validation, "w", encoding="utf-8") as f:
        for example in val_stream:
            f.write(json.dumps({"text": example["text"]}) + "\n")
            val_count += 1

    print(f"Train examples: {train_count}, size: {os.path.getsize(train)/1e6:.1f}MB")
    print(f"Val examples: {val_count}, size: {os.path.getsize(validation)/1e6:.1f}MB")
    with open(train, encoding="utf-8") as f:
        sample = json.loads(f.readline())["text"]
    print(f"Sample: {sample[:200]!r}")


if __name__ == "__main__":
    prep_data()