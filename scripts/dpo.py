import argparse
import copy
import json
import os
import sys
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dpo.loss import DPOLoss
from src.model.transformer import NexoraConfig, NexoraLM
from src.tokenizer.bpe import NexoraTokenizer

SAMPLE_PAIRS = [
    {
        "prompt": "What is the primary role of a Transformer attention layer?",
        "chosen": "The primary role is to compute dynamic pairwise token interactions via queries, keys, and values, allowing the model to route information globally across the sequence.",
        "rejected": "It just multiplies numbers and doesn't do much else."
    },
    {
        "prompt": "Why do we use RoPE instead of absolute positional embeddings?",
        "chosen": "Rotary Positional Embeddings (RoPE) encode relative token distances directly into query-key inner products, which enables better length extrapolation and preserves shift invariance.",
        "rejected": "RoPE is used because it has a catchy name and looks modern in papers."
    },
    {
        "prompt": "What is the main benefit of Grouped-Query Attention (GQA)?",
        "chosen": "GQA reduces the memory footprint and bandwidth demands of the KV cache by sharing key-value heads across multiple query heads, while retaining near-MHA representational quality.",
        "rejected": "GQA makes the model larger so it takes more GPU memory to train."
    },
    {
        "prompt": "Explain the role of gradient clipping during LLM training.",
        "chosen": "Gradient clipping caps the L2 norm of parameter gradients, preventing catastrophic optimization steps and exploding gradients caused by high-loss outliers.",
        "rejected": "Gradient clipping deletes half the weights to save disk space."
    }
]

def prepare_dpo_batch(tokenizer: NexoraTokenizer, pairs, device: str = "cpu"):
    chosen_inputs = []
    chosen_labels = []
    rejected_inputs = []
    rejected_labels = []

    for item in pairs:
        chosen_msgs = [
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["chosen"]}
        ]
        rejected_msgs = [
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["rejected"]}
        ]

        c_fmt = tokenizer.apply_chat_template(chosen_msgs)
        r_fmt = tokenizer.apply_chat_template(rejected_msgs)

        chosen_inputs.append(c_fmt["input_ids"])
        chosen_labels.append(c_fmt["labels"])
        rejected_inputs.append(r_fmt["input_ids"])
        rejected_labels.append(r_fmt["labels"])

    def pad_tensors(id_lists, label_lists):
        max_len = max(len(x) for x in id_lists)
        b_ids, b_lbls = [], []
        for ids, lbls in zip(id_lists, label_lists):
            pad_len = max_len - len(ids)
            b_ids.append(ids + [tokenizer.pad_id] * pad_len)
            b_lbls.append(lbls + [-100] * pad_len)
        return (
            torch.tensor(b_ids, dtype=torch.long, device=device),
            torch.tensor(b_lbls, dtype=torch.long, device=device)
        )

    c_x, c_y = pad_tensors(chosen_inputs, chosen_labels)
    r_x, r_y = pad_tensors(rejected_inputs, rejected_labels)
    return c_x, c_y, r_x, r_y

def main():
    parser = argparse.ArgumentParser(description="NexoraLM Direct Preference Optimization (DPO)")
    parser.add_argument("--config", default="configs/dpo.yaml")
    parser.add_argument("--sft_checkpoint", default="checkpoints/nexoralm_sft.pt")
    parser.add_argument("--tokenizer_path", default="checkpoints/tokenizer.json")
    parser.add_argument("--out_path", default="checkpoints/nexoralm_aligned.pt")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    beta = 0.1
    lr = 5e-6
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
            if "dpo" in cfg:
                beta = float(cfg["dpo"].get("beta", beta))
                lr = float(cfg["dpo"].get("lr", lr))

    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = NexoraTokenizer.load(args.tokenizer_path)

    print(f"Initializing DPO on {args.device} with beta={beta}...")
    dpo_loss_fn = DPOLoss(beta=beta)

    config = NexoraConfig()
    policy_model = NexoraLM(config).to(args.device)

    source_ckpt = args.sft_checkpoint if os.path.exists(args.sft_checkpoint) else "checkpoints/nexoralm_base.pt"
    if os.path.exists(source_ckpt):
        print(f"Loading weights from {source_ckpt}...")
        ckpt = torch.load(source_ckpt, map_location=args.device, weights_only=False)
        policy_model.load_state_dict(ckpt["model_state_dict"])
    else:
        print("Notice: Checkpoint not found. Running with initialized weights.")

    print("Cloning NexoraLM into frozen reference model...")
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    policy_model.train()
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=lr)

    c_x, c_y, r_x, r_y = prepare_dpo_batch(tokenizer, SAMPLE_PAIRS, device=args.device)
    print(f"Prepared DPO batch: chosen shape {c_x.shape}, rejected shape {r_x.shape}")

    print(f"Starting DPO preference optimization ({args.steps} steps)...")
    for step in range(1, args.steps + 1):
        optimizer.zero_grad()

        with torch.no_grad():
            ref_c_logits, _, _ = ref_model(c_x)
            ref_r_logits, _, _ = ref_model(r_x)

        pol_c_logits, _, _ = policy_model(c_x)
        pol_r_logits, _, _ = policy_model(r_x)

        loss, chosen_r, rejected_r, reward_acc = dpo_loss_fn(
            policy_chosen_logits=pol_c_logits,
            policy_rejected_logits=pol_r_logits,
            reference_chosen_logits=ref_c_logits,
            reference_rejected_logits=ref_r_logits,
            chosen_labels=c_y,
            rejected_labels=r_y
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        optimizer.step()

        margin = chosen_r.item() - rejected_r.item()
        print(f"DPO Step {step}/{args.steps} | Loss: {loss.item():.4f} | Margin: {margin:+.4f} | Chosen Reward: {chosen_r.item():.4f} | Acc: {reward_acc.item()*100:.1f}%")

    os.makedirs(os.path.dirname(args.out_path) if os.path.dirname(args.out_path) else ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": policy_model.state_dict(),
            "config": config,
            "dpo_final_loss": loss.item(),
            "dpo_final_margin": margin,
            "dpo_reward_accuracy": reward_acc.item()
        },
        args.out_path
    )
    print(f"Aligned model checkpoint saved to {args.out_path}")

if __name__ == "__main__":
    main()
