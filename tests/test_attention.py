import torch
import pytest
from src.model.transformer import NexoraConfig, NexoraLM

def test_causal_masking_no_future_leakage():
    config = NexoraConfig(
        vocab_size=1000,
        max_seq_len=64,
        hidden_size=128,
        num_layers=2,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=32,
        ffn_hidden_size=256
    )
    model = NexoraLM(config)
    model.eval()

    seq1 = torch.tensor([[10, 20, 30, 40, 50]])
    seq2 = torch.tensor([[10, 20, 30, 99, 88]])

    with torch.no_grad():
        logits1, _, _ = model(seq1)
        logits2, _, _ = model(seq2)

    assert torch.allclose(logits1[:, :3, :], logits2[:, :3, :], atol=1e-5)
