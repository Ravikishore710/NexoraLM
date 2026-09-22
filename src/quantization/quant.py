from typing import Dict, Tuple
import torch
import torch.nn as nn

def quantize_int8_per_channel(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    max_val = weight.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    scale = max_val / 127.0
    qweight = (weight / scale).round().clamp(-128, 127).to(torch.int8)
    return qweight, scale

def dequantize_int8(qweight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return qweight.float() * scale

def quantize_int4_groupwise(
    weight: torch.Tensor,
    group_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    orig_shape = weight.shape
    flat = weight.flatten()
    numel = flat.numel()
    pad = (group_size - (numel % group_size)) % group_size
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, dtype=weight.dtype, device=weight.device)])

    groups = flat.view(-1, group_size)
    w_min = groups.min(dim=-1, keepdim=True).values
    w_max = groups.max(dim=-1, keepdim=True).values
    scale = (w_max - w_min).clamp(min=1e-5) / 15.0
    zero_point = (-w_min / scale).round().clamp(0, 15)

    qgroups = ((groups / scale) + zero_point).round().clamp(0, 15).to(torch.uint8)
    return qgroups, scale, zero_point, pad

def dequantize_int4(
    qgroups: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    pad: int,
    orig_shape: Tuple[int, ...]
) -> torch.Tensor:
    dequant = (qgroups.float() - zero_point) * scale
    flat = dequant.flatten()
    if pad > 0:
        flat = flat[:-pad]
    return flat.view(orig_shape)

def evaluate_quantization_error(orig: torch.Tensor, dequant: torch.Tensor) -> Dict[str, float]:
    mse = torch.mean((orig - dequant) ** 2).item()
    orig_norm = orig.norm()
    diff_norm = (orig - dequant).norm()
    rel_error = (diff_norm / orig_norm.clamp(min=1e-5)).item()
    return {"mse": mse, "relative_error": rel_error}
