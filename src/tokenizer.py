import os
import json
from tokenizers import Tokenizer,models, processors, pre_tokenizers, trainers

with open("data/raw/wiki_train.jsonl", "r", encoding="utf-8") as fin, \
    open("data/raw/wiki_train.txt", "w", encoding="utf-8") as fout:
    for line in fin:
        fout.write(json.loads(line)["text"] + "\n")

token = Tokenizer(models.WordPiece(unk_token="[UNK]"))
token.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

trainer = trainers.WordPieceTrainer(
    vocab_size=12000,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    min_frequency=2
)

token.train(["data/raw/wiki_train.txt"], trainer)

token.post_processor = processors.TemplateProcessing(
    single= "[CLS]:0 $A:0 [SEP]:0",
    pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", token.token_to_id("[CLS]")), 
                    ("[SEP]", token.token_to_id("[SEP]"))],
)

print(f"Actual vocab size: {token.get_vocab_size()}")
test_prompt = "the transformer architecture changed natural language processing and whole TTS, SST verse." # input test prompt to verify and continue the without any crashout
encode = token.encode(test_prompt)

print(f"Input text:  '{test_prompt}'")
print(f"Tokens:      {encode.tokens}")
print(f"Token IDs:   {encode.ids}")

os.makedirs("tokenizer", exist_ok=True)
token.save("tokenizer/tokenizer.json")
print("Done! Tokenizer trained from saved text file.")