import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.model.transformer import NexoraConfig, NexoraLM
from src.quantization.quant import (
    quantize_int8_per_channel,
    quantize_int4_groupwise,
    dequantize_int8,
    dequantize_int4,
    evaluate_quantization_error
)

def main():
    parser = argparse.ArgumentParser(description="NexoraLM Post-Training Quantization")
    parser.add_argument("--model_path", default="checkpoints/nexoralm_base.pt", help="Path to FP16 model checkpoint")
    parser.add_argument("--format", choices=["int8", "int4", "both"], default="both")
    parser.add_argument("--out_dir", default="checkpoints/quantized")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Loading checkpoint {args.model_path}...")
    if not os.path.exists(args.model_path):
        print("Checkpoint not found. Running synthetic benchmark on 126M architecture.")
        config = NexoraConfig()
        model = NexoraLM(config)
    else:
        ckpt = torch.load(args.model_path, map_location="cpu")
        config = NexoraConfig()
        model = NexoraLM(config)
        model.load_state_dict(ckpt["model_state_dict"])

    print("Benchmarking post-training quantization on linear projection layers...")
    linear_layers = [name for name, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    print(f"Found {len(linear_layers)} linear projection layers to quantize.")

    total_orig_bytes = 0
    total_int8_bytes = 0
    total_int4_bytes = 0
    errors_int8 = []
    errors_int4 = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            w = module.weight.data
            numel = w.numel()
            total_orig_bytes += numel * 2

            q8, s8 = quantize_int8_per_channel(w)
            total_int8_bytes += numel * 1 + s8.numel() * 4
            deq8 = dequantize_int8(q8, s8)
            errors_int8.append(evaluate_quantization_error(w, deq8)["relative_error"])

            q4, s4, z4, pad = quantize_int4_groupwise(w, group_size=128)
            total_int4_bytes += (numel // 2) + s4.numel() * 4 + z4.numel() * 1
            deq4 = dequantize_int4(q4, s4, z4, pad, w.shape)
            errors_int4.append(evaluate_quantization_error(w, deq4)["relative_error"])

    print("--- Quantization Summary ---")
    print(f"FP16 Weights Size:  {total_orig_bytes / (1024**2):.2f} MB")
    print(f"INT8 Weights Size:  {total_int8_bytes / (1024**2):.2f} MB ({total_int8_bytes/total_orig_bytes*100:.1f}%) | Mean Rel Error: {sum(errors_int8)/len(errors_int8):.4f}")
    print(f"INT4 Weights Size:  {total_int4_bytes / (1024**2):.2f} MB ({total_int4_bytes/total_orig_bytes*100:.1f}%) | Mean Rel Error: {sum(errors_int4)/len(errors_int4):.4f}")

if __name__ == "__main__":
    main()
