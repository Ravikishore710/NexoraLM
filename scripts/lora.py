import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from src.lora.lora import apply_lora, get_trainable_parameters
from src.model.transformer import NexoraConfig, NexoraLM
from src.training.trainer import NexoraTrainer

def main():
    parser = argparse.ArgumentParser(description="NexoraLM LoRA Fine-Tuning")
    parser.add_argument("--config", default="configs/lora.yaml")
    parser.add_argument("--base_model_path", default="checkpoints/nexoralm_base.pt")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Instantiating NexoraLM-126M on {device}...")
    config = NexoraConfig()
    model = NexoraLM(config).to(device)

    lc = cfg["lora"]
    print(f"Applying LoRA: r={lc['r']}, alpha={lc['lora_alpha']} to {lc['target_modules']}...")
    apply_lora(
        model,
        r=int(lc["r"]),
        lora_alpha=int(lc["lora_alpha"]),
        lora_dropout=float(lc["lora_dropout"]),
        target_modules=lc["target_modules"],
        freeze_base=True
    )

    stats = get_trainable_parameters(model)
    print(f"LoRA applied. Trainable: {stats['trainable']:,} / {stats['total']:,} ({stats['ratio']*100:.2f}%)")

if __name__ == "__main__":
    main()
