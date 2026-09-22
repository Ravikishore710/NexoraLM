import os
import yaml

def test_base_config_integrity():
    config_path = os.path.join("configs", "base.yaml")
    assert os.path.exists(config_path)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    m = cfg["model"]
    assert m["vocab_size"] == 32768
    assert m["max_seq_len"] == 2048
    assert m["hidden_size"] == 768
    assert m["num_layers"] == 16
    assert m["num_query_heads"] == 12
    assert m["num_kv_heads"] == 4
    assert m["head_dim"] == 64
    assert m["ffn_hidden_size"] == 2048
    assert m["tie_word_embeddings"] is True
    assert m["bias"] is False

def test_pretraining_config_integrity():
    config_path = os.path.join("configs", "pretraining.yaml")
    assert os.path.exists(config_path)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    assert cfg["dataset"]["train_tokens"] == 300000000
    assert cfg["dataset"]["seq_len"] == 2048
    assert cfg["training"]["micro_batch_size"] == 2
    assert cfg["training"]["gradient_accumulation_steps"] == 32
    assert cfg["training"]["mixed_precision"] == "fp16"
    assert cfg["training"]["beta2"] == 0.95
