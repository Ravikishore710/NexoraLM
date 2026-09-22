import torch
import pytest
from src.model.transformer import NexoraConfig, NexoraLM

def test_kv_cache_numerical_equivalence():
    config = NexoraConfig(
        vocab_size=100,
        max_seq_len=64,
        hidden_size=64,
        num_layers=2,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=16,
        ffn_hidden_size=128
    )
    model = NexoraLM(config)
    model.eval()

    input_ids = torch.tensor([[12, 34, 56, 78, 90]])

    with torch.no_grad():
        full_logits, _, _ = model(input_ids)

    kv_caches = None
    cached_logits = []
    with torch.no_grad():
        for i in range(input_ids.shape[1]):
            token = input_ids[:, i:i+1]
            out, _, kv_caches = model(token, kv_caches=kv_caches, use_cache=True)
            cached_logits.append(out)

    cached_logits_tensor = torch.cat(cached_logits, dim=1)

    assert torch.allclose(full_logits, cached_logits_tensor, atol=1e-4), "KV-cache output diverged from full forward pass"
