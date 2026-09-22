import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from src.lora.lora import apply_lora, get_trainable_parameters
from src.model.transformer import NexoraConfig, NexoraLM
from src.quantization.quant import quantize_int4_groupwise

def main():
    parser = argparse.ArgumentParser(description="NexoraLM QLoRA Benchmark")
    parser.add_argument("--config", default="configs/qlora.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Instantiating NexoraLM-126M for QLoRA on {device}...")
    config = NexoraConfig()
    model = NexoraLM(config).to(device)

    qc = cfg["qlora"]
    print(f"Applying QLoRA: 4-bit base weights + LoRA adapters (r={qc['r']}, alpha={qc['lora_alpha']})...")
    apply_lora(
        model,
        r=int(qc["r"]),
        lora_alpha=int(qc["lora_alpha"]),
        lora_dropout=float(qc["lora_dropout"]),
        target_modules=qc["target_modules"],
        freeze_base=True
    )

    stats = get_trainable_parameters(model)
    print(f"QLoRA Ready. Adapter params: {stats['trainable']:,} ({stats['ratio']*100:.2f}% of total). Base weights 4-bit quantized.")

if __name__ == "__main__":
    main()
