from fastapi.testclient import TestClient
import pytest
from src.model.transformer import NexoraConfig, NexoraLM
from src.serving.app import app, init_app
from src.tokenizer.bpe import NexoraTokenizer

SAMPLE_CORPUS = [
    "Language models generate text sequentially.",
    "FastAPI provides asynchronous HTTP endpoints."
]

def test_api_serving_endpoints():
    tokenizer = NexoraTokenizer.train(SAMPLE_CORPUS, vocab_size=300)
    config = NexoraConfig(
        vocab_size=300,
        max_seq_len=64,
        hidden_size=64,
        num_layers=2,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=16,
        ffn_hidden_size=128
    )
    model = NexoraLM(config)
    init_app(model, tokenizer, device="cpu")

    client = TestClient(app)

    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

    res = client.get("/model")
    assert res.status_code == 200
    assert res.json()["vocab_size"] == 300

    res = client.post("/v1/completions", json={"prompt": "Language", "max_tokens": 5})
    assert res.status_code == 200
    assert "choices" in res.json()

    res = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 5
        }
    )
    assert res.status_code == 200
    assert res.json()["choices"][0]["message"]["role"] == "assistant"
