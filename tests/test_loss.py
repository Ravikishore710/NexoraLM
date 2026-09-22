import torch
import pytest
from src.model.transformer import NexoraConfig, NexoraLM

def test_next_token_cross_entropy():
    config = NexoraConfig(
        vocab_size=100,
        max_seq_len=32,
        hidden_size=64,
        num_layers=2,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=16,
        ffn_hidden_size=128
    )
    model = NexoraLM(config)

    input_ids = torch.randint(0, 100, (2, 8))
    labels = input_ids.clone()
    _, loss, _ = model(input_ids, labels=labels)
    assert loss is not None
    assert loss.item() > 0

def test_assistant_only_loss_masking():
    config = NexoraConfig(
        vocab_size=100,
        max_seq_len=32,
        hidden_size=64,
        num_layers=2,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=16,
        ffn_hidden_size=128
    )
    model = NexoraLM(config)

    input_ids = torch.randint(0, 100, (1, 6))
    labels = torch.tensor([[-100, -100, -100, 50, 60, 70]])
    _, loss, _ = model(input_ids, labels=labels)
    assert loss is not None
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
