import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from datasets import load_dataset
from src.model.transformer import NexoraConfig, NexoraLM
from src.tokenizer.bpe import NexoraTokenizer
from src.training.trainer import NexoraTrainer

def stream_and_pack_fineweb(
    tokenizer: NexoraTokenizer,
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    subset: str = "sample-10BT",
    seq_len: int = 2048
):
    dataset = load_dataset(dataset_name, subset, split="train", streaming=True)
    current_tokens = []
    
    for sample in dataset:
        text = sample.get("text", "").strip()
        if len(text) < 50:
            continue
        
        doc_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        current_tokens.extend(doc_ids)

        while len(current_tokens) >= seq_len:
            yield current_tokens[:seq_len]
            current_tokens = current_tokens[seq_len:]

def main():
    parser = argparse.ArgumentParser(description="NexoraLM Pretraining Runner")
    parser.add_argument("--config", default="configs/pretraining.yaml", help="Path to pretraining YAML")
    parser.add_argument("--model_config", default="configs/base.yaml", help="Path to base model YAML")
    parser.add_argument("--tokenizer_path", default="checkpoints/tokenizer.json", help="Path to tokenizer")
    parser.add_argument("--mode", choices=["throughput_gate", "full"], default="throughput_gate")
    parser.add_argument("--gate_steps", type=int, default=200, help="Steps for throughput measurement")
    parser.add_argument("--out_dir", default="checkpoints", help="Output directory for checkpoints")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        train_cfg = yaml.safe_load(f)
    with open(args.model_config, "r") as f:
        m_cfg = yaml.safe_load(f)["model"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing NexoraLM-126M on {device}...")

    config = NexoraConfig(
        vocab_size=m_cfg["vocab_size"],
        max_seq_len=m_cfg["max_seq_len"],
        hidden_size=m_cfg["hidden_size"],
        num_layers=m_cfg["num_layers"],
        num_query_heads=m_cfg["num_query_heads"],
        num_kv_heads=m_cfg["num_kv_heads"],
        head_dim=m_cfg["head_dim"],
        ffn_hidden_size=m_cfg["ffn_hidden_size"],
        tie_word_embeddings=m_cfg["tie_word_embeddings"],
        bias=m_cfg["bias"]
    )
    model = NexoraLM(config)
    print(f"Model instantiated. Total parameters: {model.count_parameters():,}")

    tokenizer = NexoraTokenizer.load(args.tokenizer_path) if os.path.exists(args.tokenizer_path) else None

    tc = train_cfg["training"]
    trainer = NexoraTrainer(
        model=model,
        learning_rate=float(tc["learning_rate"]),
        min_learning_rate=float(tc["min_learning_rate"]),
        weight_decay=float(tc["weight_decay"]),
        beta1=float(tc["beta1"]),
        beta2=float(tc["beta2"]),
        eps=float(tc["eps"]),
        grad_clip=float(tc["grad_clip"]),
        gradient_accumulation_steps=int(tc["gradient_accumulation_steps"]),
        mixed_precision=tc["mixed_precision"],
        device=device
    )

    if args.mode == "throughput_gate":
        print(f"Running throughput gate test for {args.gate_steps} steps...")
        micro_bs = int(tc["micro_batch_size"])
        seq_len = int(train_cfg["dataset"]["seq_len"])
        
        dummy_x = torch.randint(0, config.vocab_size, (micro_bs, seq_len))
        dummy_y = dummy_x.clone()

        start_time = time.time()
        for step in range(args.gate_steps):
            metrics = trainer.train_step(dummy_x, dummy_y)
            if (step + 1) % 20 == 0:
                elapsed = time.time() - start_time
                tok_per_sec = (step + 1) * micro_bs * seq_len / elapsed
                print(f"Step {step+1}/{args.gate_steps} | Loss: {metrics['loss']:.4f} | Tok/s: {tok_per_sec:.1f}")

        total_time = time.time() - start_time
        total_tokens = args.gate_steps * micro_bs * seq_len
        avg_tok_s = total_tokens / total_time
        
        target_tokens = int(train_cfg["dataset"]["train_tokens"])
        est_seconds = target_tokens / avg_tok_s
        est_hours = est_seconds / 3600.0

        gate_report = {
            "accelerator": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "measured_throughput_tokens_sec": avg_tok_s,
            "target_tokens": target_tokens,
            "estimated_runtime_hours": est_hours,
            "decision": "GO" if est_hours < 30.0 else "MODIFY",
            "timestamp": time.time()
        }
        os.makedirs(args.out_dir, exist_ok=True)
        report_path = os.path.join(args.out_dir, "throughput_gate.json")
        with open(report_path, "w") as f:
            json.dump(gate_report, f, indent=2)
        print(f"Throughput gate completed. Result: {gate_report['decision']} ({est_hours:.2f} hours estimated). Saved to {report_path}")

if __name__ == "__main__":
    main()
