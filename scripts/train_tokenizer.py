import argparse
import os
from datasets import load_dataset
from src.tokenizer.bpe import NexoraTokenizer

def stream_training_text(dataset_name: str, subset: str, max_chars: int = 60_000_000):
    dataset = load_dataset(dataset_name, subset, split="train", streaming=True)
    total_chars = 0
    for sample in dataset:
        text = sample.get("text", "")
        if len(text.strip()) < 50:
            continue
        total_chars += len(text)
        yield text
        if total_chars >= max_chars:
            break

def main():
    parser = argparse.ArgumentParser(description="Train NexoraLM Byte-level BPE Tokenizer")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu", help="Hugging Face dataset repo")
    parser.add_argument("--subset", default="sample-10BT", help="Dataset subset/split")
    parser.add_argument("--vocab_size", type=int, default=32768, help="Total vocabulary size")
    parser.add_argument("--output_path", default="checkpoints/tokenizer.json", help="Path to save tokenizer.json")
    parser.add_argument("--max_chars", type=int, default=60_000_000, help="Target character volume (~15M tokens)")
    args = parser.parse_args()

    print(f"Sampling training corpus from {args.dataset} ({args.subset})...")
    text_iter = stream_training_text(args.dataset, args.subset, max_chars=args.max_chars)
    
    print(f"Training Byte-level BPE tokenizer to exact vocabulary size {args.vocab_size}...")
    tokenizer = NexoraTokenizer.train(text_iter, vocab_size=args.vocab_size)

    assert tokenizer.vocab_size == args.vocab_size, f"Vocab size mismatch: {tokenizer.vocab_size} vs {args.vocab_size}"
    tokenizer.save(args.output_path)
    print(f"Tokenizer successfully trained and saved to {args.output_path} with vocab size {tokenizer.vocab_size}")

if __name__ == "__main__":
    main()
