import torch
import pytest
from src.attention.gqa import GroupedQueryAttention
from src.ffn.swiglu import SwiGLU
from src.normalization.rmsnorm import RMSNorm
from src.rope.rotary import RotaryEmbedding
from src.model.transformer import NexoraConfig, NexoraLM

def test_rmsnorm_shape():
    norm = RMSNorm(768)
    x = torch.randn(2, 16, 768)
    out = norm(x)
    assert out.shape == (2, 16, 768)

def test_swiglu_shape():
    ffn = SwiGLU(768, 2048)
    x = torch.randn(2, 16, 768)
    out = ffn(x)
    assert out.shape == (2, 16, 768)

def test_rope_shape():
    rope = RotaryEmbedding(dim=64, max_seq_len=2048)
    x = torch.randn(2, 16, 768)
    cos, sin = rope(x, 16)
    assert cos.shape == (16, 64)
    assert sin.shape == (16, 64)

def test_gqa_shape():
    attn = GroupedQueryAttention(hidden_size=768, num_query_heads=12, num_kv_heads=4, head_dim=64)
    rope = RotaryEmbedding(dim=64, max_seq_len=2048)
    x = torch.randn(2, 16, 768)
    cos, sin = rope(x, 16)
    out, cache = attn(x, cos=cos, sin=sin, use_cache=True)
    assert out.shape == (2, 16, 768)
    assert cache[0].shape == (2, 4, 16, 64)
    assert cache[1].shape == (2, 4, 16, 64)

def test_nexoralm_126m_parameters_and_shapes():
    config = NexoraConfig(
        vocab_size=32768,
        max_seq_len=2048,
        hidden_size=768,
        num_layers=16,
        num_query_heads=12,
        num_kv_heads=4,
        head_dim=64,
        ffn_hidden_size=2048,
        tie_word_embeddings=True
    )
    model = NexoraLM(config)
    params = model.count_parameters()
    
    assert 125_000_000 <= params <= 126_500_000
    assert params == 125_854_464

    input_ids = torch.randint(0, 32768, (2, 32))
    labels = torch.randint(0, 32768, (2, 32))
    logits, loss, _ = model(input_ids, labels=labels)
    assert logits.shape == (2, 32, 32768)
    assert loss is not None
    assert loss.item() > 0
