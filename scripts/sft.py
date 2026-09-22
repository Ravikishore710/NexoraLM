import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml
from datasets import load_dataset
from src.model.transformer import NexoraConfig, NexoraLM
from src.tokenizer.bpe import NexoraTokenizer
from src.training.trainer import get_cosine_schedule_with_warmup

class SFTDataset(Dataset):
    def __init__(self, items: List[Dict[str, List[int]]], max_seq_len: int = 2048, pad_id: int = 3):
        self.items = items
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        input_ids = item["input_ids"][:self.max_seq_len]
        labels = item["labels"][:self.max_seq_len]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def sft_collate_fn(batch, pad_id: int = 3):
    max_len = max(len(b["input_ids"]) for b in batch)
    b_inputs = []
    b_labels = []

    for b in batch:
        inp = b["input_ids"]
        lbl = b["labels"]
        pad_len = max_len - len(inp)

        if pad_len > 0:
            padded_inp = torch.cat([inp, torch.full((pad_len,), pad_id, dtype=torch.long)])
            padded_lbl = torch.cat([lbl, torch.full((pad_len,), -100, dtype=torch.long)])
        else:
            padded_inp = inp
            padded_lbl = lbl

        b_inputs.append(padded_inp)
        b_labels.append(padded_lbl)

    return {
        "input_ids": torch.stack(b_inputs),
        "labels": torch.stack(b_labels)
    }

def load_sft_mixture(tokenizer: NexoraTokenizer, max_oasst: int = 400, max_reasoning: int = 200):
    formatted_data = []

    print(f"Sampling OpenAssistant (OASST1) instruction data...")
    try:
        ds = load_dataset("OpenAssistant/oasst1", split="train", streaming=True)
        count = 0
        for sample in ds:
            if sample.get("lang") == "en" and sample.get("role") == "assistant":
                text = sample.get("text", "").strip()
                if 20 <= len(text) <= 1500:
                    messages = [
                        {"role": "system", "content": "You are NexoraLM, an accurate and helpful assistant."},
                        {"role": "user", "content": "Help me with this question."},
                        {"role": "assistant", "content": text}
                    ]
                    formatted = tokenizer.apply_chat_template(messages)
                    formatted_data.append(formatted)
                    count += 1
                    if count >= max_oasst:
                        break
    except Exception as e:
        print(f"Notice: Streaming OASST1 skipped ({e}). Using synthetic instruction sample.")

    try:
        print(f"Sampling reasoning instruction data...")
        ds_r = load_dataset("HuggingFaceH4/Bespoke-Stratos-17k", split="train", streaming=True)
        count_r = 0
        for sample in ds_r:
            conversations = sample.get("conversations", [])
            if len(conversations) >= 2:
                user_content = conversations[0].get("value", "")
                asst_content = conversations[1].get("value", "")
                if user_content and asst_content:
                    messages = [
                        {"role": "system", "content": "You are NexoraLM, a concise reasoning assistant."},
                        {"role": "user", "content": user_content[:500]},
                        {"role": "assistant", "content": asst_content[:1000]}
                    ]
                    formatted = tokenizer.apply_chat_template(messages)
                    formatted_data.append(formatted)
                    count_r += 1
                    if count_r >= max_reasoning:
                        break
    except Exception as e:
        print(f"Notice: Streaming reasoning data skipped ({e}).")

    if len(formatted_data) == 0:
        for i in range(100):
            messages = [
                {"role": "system", "content": "You are NexoraLM, a helpful AI assistant."},
                {"role": "user", "content": f"Explain key concept #{i} in computer science."},
                {"role": "assistant", "content": f"Concept #{i} involves modular abstractions and caching algorithms."}
            ]
            formatted_data.append(tokenizer.apply_chat_template(messages))

    return formatted_data

def train_sft(
    model: NexoraLM,
    tokenizer: NexoraTokenizer,
    data: List[Dict[str, List[int]]],
    epochs: int = 1,
    batch_size: int = 2,
    lr: float = 2e-5,
    grad_accum_steps: int = 4,
    device: str = "cpu"
):
    model.to(device)
    model.train()

    split_idx = int(0.9 * len(data))
    train_items = data[:split_idx]
    val_items = data[split_idx:]

    train_ds = SFTDataset(train_items, max_seq_len=512, pad_id=tokenizer.pad_id)
    val_ds = SFTDataset(val_items, max_seq_len=512, pad_id=tokenizer.pad_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: sft_collate_fn(b, pad_id=tokenizer.pad_id)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(train_loader) // grad_accum_steps) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps=max(1, int(0.05 * total_steps)), total_steps=max(1, total_steps))

    print(f"Starting SFT training ({len(train_items)} train pairs, {len(val_items)} val pairs, {total_steps} steps)...")
    opt_step = 0
    accum_loss = 0.0

    for epoch in range(epochs):
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            _, loss, _ = model(input_ids, labels=labels)
            loss = loss / grad_accum_steps
            loss.backward()
            accum_loss += loss.item() * grad_accum_steps

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                opt_step += 1

                if opt_step % 5 == 0 or opt_step == total_steps:
                    mean_loss = accum_loss / grad_accum_steps
                    ppl = math.exp(min(mean_loss, 20.0))
                    print(f"Epoch {epoch+1} | Step {opt_step}/{total_steps} | SFT Loss: {mean_loss:.4f} | Assistant PPL: {ppl:.2f}")
                accum_loss = 0.0

    model.eval()
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: sft_collate_fn(b, pad_id=tokenizer.pad_id)
    )
    val_loss_total = 0.0
    with torch.no_grad():
        for batch in val_loader:
            inp = batch["input_ids"].to(device)
            lbl = batch["labels"].to(device)
            _, v_loss, _ = model(inp, labels=lbl)
            val_loss_total += v_loss.item()

    mean_val_loss = val_loss_total / max(1, len(val_loader))
    val_ppl = math.exp(min(mean_val_loss, 20.0))
    print(f"SFT Validation Complete. Val Loss: {mean_val_loss:.4f} | Val PPL: {val_ppl:.2f}")

    return {
        "final_train_loss": mean_loss if 'mean_loss' in locals() else mean_val_loss,
        "val_loss": mean_val_loss,
        "val_ppl": val_ppl,
        "train_samples": len(train_items),
        "val_samples": len(val_items)
    }

def main():
    parser = argparse.ArgumentParser(description="NexoraLM Supervised Instruction Tuning (SFT)")
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--base_model_path", default="checkpoints/nexoralm_base.pt")
    parser.add_argument("--tokenizer_path", default="checkpoints/tokenizer.json")
    parser.add_argument("--out_path", default="checkpoints/nexoralm_sft.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_samples", type=int, default=20)
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = NexoraTokenizer.load(args.tokenizer_path)

    print(f"Loading model checkpoint from {args.base_model_path}...")
    config = NexoraConfig()
    model = NexoraLM(config)
    if os.path.exists(args.base_model_path):
        ckpt = torch.load(args.base_model_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print("Base model weights loaded.")

    data = load_sft_mixture(tokenizer, max_oasst=args.max_samples, max_reasoning=args.max_samples // 2)
    metrics = train_sft(
        model=model,
        tokenizer=tokenizer,
        data=data,
        epochs=1,
        batch_size=2,
        lr=2e-5,
        grad_accum_steps=2,
        device=args.device
    )

    os.makedirs(os.path.dirname(args.out_path) if os.path.dirname(args.out_path) else ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "metrics": metrics
        },
        args.out_path
    )
    print(f"SFT model checkpoint saved to {args.out_path}")

if __name__ == "__main__":
    main()
