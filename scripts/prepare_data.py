import argparse
import hashlib
import json
import os
import torch
from datasets import load_dataset
from src.tokenizer.bpe import NexoraTokenizer

def is_valid_document(text: str) -> bool:
    s = text.strip()
    if len(s) < 50:
        return False
    lines = s.split("\n")
    if len(lines) > 5 and len(set(lines)) / len(lines) < 0.5:
        return False
    return True

def pack_tokens(token_ids_list, seq_len: int = 2048):
    packed = []
    current_chunk = []
    for doc in token_ids_list:
        current_chunk.extend(doc)
        while len(current_chunk) >= seq_len:
            packed.append(current_chunk[:seq_len])
            current_chunk = current_chunk[seq_len:]
    return packed

def prepare_tinystories(output_dir: str = "data/tinystories", max_train_samples: int = 5000, max_val_samples: int = 500):
    os.makedirs(output_dir, exist_ok=True)
    print("Loading TinyStories train split...")
    train_ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    val_ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)

    train_texts = []
    for i, item in enumerate(train_ds):
        t = item.get("text", "").strip()
        if len(t) >= 50:
            train_texts.append(t)
        if len(train_texts) >= max_train_samples:
            break

    val_texts = []
    for i, item in enumerate(val_ds):
        t = item.get("text", "").strip()
        if len(t) >= 50:
            val_texts.append(t)
        if len(val_texts) >= max_val_samples:
            break

    train_path = os.path.join(output_dir, "train.json")
    val_path = os.path.join(output_dir, "val.json")
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_texts, f)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_texts, f)

    print(f"TinyStories prepared: {len(train_texts)} train, {len(val_texts)} val samples")

def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for NexoraLM")
    parser.add_argument("--task", choices=["tinystories", "fineweb_sample"], default="tinystories")
    parser.add_argument("--out_dir", default="data/tinystories")
    args = parser.parse_args()

    if args.task == "tinystories":
        prepare_tinystories(args.out_dir)

if __name__ == "__main__":
    main()
