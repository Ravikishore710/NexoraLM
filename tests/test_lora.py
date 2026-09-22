import torch
import torch.nn as nn
import pytest
from src.lora.lora import LoRALinear, apply_lora, get_trainable_parameters
from src.model.transformer import NexoraConfig, NexoraLM

def test_lora_linear_forward_and_gradient():
    base = nn.Linear(64, 64, bias=False)
    lora_layer = LoRALinear(base, r=8, lora_alpha=16)

    assert not lora_layer.base_layer.weight.requires_grad
    assert lora_layer.lora_a.requires_grad
    assert lora_layer.lora_b.requires_grad

    x = torch.randn(2, 64)
    out = lora_layer(x)
    loss = out.sum()
    loss.backward()

    assert lora_layer.lora_a.grad is not None
    assert lora_layer.lora_b.grad is not None
    assert lora_layer.base_layer.weight.grad is None

def test_lora_merge_and_unmerge():
    base = nn.Linear(32, 32, bias=False)
    orig_weight = base.weight.data.clone()
    lora_layer = LoRALinear(base, r=4, lora_alpha=8, lora_dropout=0.0)
    lora_layer.eval()

    lora_layer.lora_b.data.normal_()
    x = torch.randn(2, 32)
    out_unmerged = lora_layer(x)

    lora_layer.merge()
    assert not torch.equal(base.weight.data, orig_weight)
    out_merged = lora_layer(x)
    assert torch.allclose(out_unmerged, out_merged, atol=1e-5)

    lora_layer.unmerge()
    assert torch.allclose(base.weight.data, orig_weight, atol=1e-5)

def test_lora_parameter_count():
    config = NexoraConfig(
        vocab_size=1000,
        max_seq_len=64,
        hidden_size=768,
        num_layers=16,
        num_query_heads=12,
        num_kv_heads=4,
        head_dim=64,
        ffn_hidden_size=2048
    )
    model = NexoraLM(config)
    apply_lora(model, r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], freeze_base=True)
    stats = get_trainable_parameters(model)

    assert stats["trainable"] == 1_310_720
    assert stats["trainable"] < stats["total"] * 0.02
