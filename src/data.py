import os
from datasets import load_dataset

raw = "data/raw"
train = os.path.join(raw, "wiki_train.jsonl")
validation = os.path.join(raw, "wiki_val.jsonl")
split = "train[:5%]"

def prep_data():
    os.makedirs(raw, exist_ok=True)

    train_set = load_dataset("wikitext", "wikitext-103-raw-v1", split= split)
    val_set = load_dataset("wikitext", "wikitext-103-raw-v1", split = "validation")

    train_set = train_set.filter(lambda x: len(x["text"].strip()) > 0)
    val_set = val_set.filter(lambda x: len(x["text"].strip()) > 0)

    train_set.to_json(train, lines=True)
    val_set.to_json(validation, lines=True)

    print(f"Train examples: {len(train_set)}")
    print(f"Validation examples: {len(val_set)}")
    print(f"Sample text: {train_set[0]['text'][:200]!r}")

if __name__ == "__main__":
    prep_data()