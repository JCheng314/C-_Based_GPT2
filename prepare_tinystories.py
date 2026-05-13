import os
import struct
import tiktoken
from tqdm import tqdm

DATA_DIR = "data/tinystories"

train_txt = os.path.join(DATA_DIR, "TinyStories-train.txt")
val_txt = os.path.join(DATA_DIR, "TinyStories-valid.txt")

train_bin = os.path.join(DATA_DIR, "train.bin")
val_bin = os.path.join(DATA_DIR, "val.bin")

enc = tiktoken.get_encoding("gpt2")


def write_bin(txt_path, bin_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    # GPT-style tokenization.
    # The endoftext token helps separate stories/documents.
    tokens = enc.encode_ordinary(text)
    eot = enc.eot_token

    # Optional: append EOT once at the end.
    tokens.append(eot)

    print(f"{txt_path}: {len(tokens):,} tokens")

    with open(bin_path, "wb") as f:
        for token in tqdm(tokens):
            f.write(struct.pack("<I", token))  # uint32 little-endian


write_bin(train_txt, train_bin)
write_bin(val_txt, val_bin)