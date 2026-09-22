import torch
import pytest
from src.quantization.quant import (
    quantize_int8_per_channel,
    dequantize_int8,
    quantize_int4_groupwise,
    dequantize_int4,
    evaluate_quantization_error
)

def test_int8_per_channel_quantization():
    w = torch.randn(128, 256)
    qweight, scale = quantize_int8_per_channel(w)
    
    assert qweight.dtype == torch.int8
    assert qweight.shape == w.shape
    assert scale.shape == (128, 1)

    dequant = dequantize_int8(qweight, scale)
    metrics = evaluate_quantization_error(w, dequant)
    assert metrics["mse"] < 0.01
    assert metrics["relative_error"] < 0.08

def test_int4_groupwise_quantization():
    w = torch.randn(128, 256)
    qgroups, scale, zero_point, pad = quantize_int4_groupwise(w, group_size=128)

    assert qgroups.dtype == torch.uint8
    assert qgroups.shape == (256, 128)

    dequant = dequantize_int4(qgroups, scale, zero_point, pad, w.shape)
    metrics = evaluate_quantization_error(w, dequant)
    assert metrics["mse"] < 0.05
    assert metrics["relative_error"] < 0.25
