import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.inference.generator import NexoraGenerator
from src.model.transformer import NexoraConfig, NexoraLM
from src.tokenizer.bpe import NexoraTokenizer

BENCHMARK_PROMPTS = [
    "The fundamental principle of gradient descent is",
    "In computer systems, a cache improves performance by",
    "Photosynthesis is the biological process where",
    "A Transformer uses self-attention to"
]

def evaluate_generation(generator: NexoraGenerator):
    results = []
    for prompt in BENCHMARK_PROMPTS:
        start_t = time.time()
        output = generator.generate(prompt, max_new_tokens=64, temperature=0.0)
        elapsed = time.time() - start_t
        results.append({
            "prompt": prompt,
            "completion": output,
            "elapsed_seconds": elapsed
        })
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate NexoraLM checkpoints")
    parser.add_argument("--model_path", default="checkpoints/nexoralm_base.pt")
    parser.add_argument("--tokenizer_path", default="checkpoints/tokenizer.json")
    parser.add_argument("--out_path", default="outputs/evaluation_report.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    config = NexoraConfig()
    model = NexoraLM(config)
    if os.path.exists(args.model_path):
        ckpt = torch.load(args.model_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    tokenizer = NexoraTokenizer.load(args.tokenizer_path) if os.path.exists(args.tokenizer_path) else None
    if tokenizer is None:
        print("Tokenizer not found. Skipping generation evaluation.")
        return

    generator = NexoraGenerator(model, tokenizer, device=device)
    print("Running benchmark prompts...")
    eval_results = evaluate_generation(generator)

    os.makedirs(os.path.dirname(args.out_path) if os.path.dirname(args.out_path) else ".", exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Evaluation report written to {args.out_path}")

if __name__ == "__main__":
    main()
