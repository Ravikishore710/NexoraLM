import os
import torch
import pytest
from src.model.transformer import NexoraConfig, NexoraLM
from src.training.trainer import NexoraTrainer

def test_checkpoint_save_and_resume(tmp_path):
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
    trainer = NexoraTrainer(
        model=model,
        learning_rate=1e-3,
        gradient_accumulation_steps=2,
        device="cpu"
    )

    batch_x = torch.randint(0, 100, (2, 16))
    batch_y = torch.randint(0, 100, (2, 16))

    for _ in range(4):
        metrics = trainer.train_step(batch_x, batch_y)

    assert trainer.global_step == 4
    assert trainer.tokens_seen == 2 * 16 * 4

    ckpt_path = os.path.join(tmp_path, "test_ckpt.pt")
    trainer.save_checkpoint(ckpt_path)

    new_model = NexoraLM(config)
    new_trainer = NexoraTrainer(
        model=new_model,
        learning_rate=1e-3,
        gradient_accumulation_steps=2,
        device="cpu"
    )
    new_trainer.load_checkpoint(ckpt_path)

    assert new_trainer.global_step == 4
    assert new_trainer.tokens_seen == trainer.tokens_seen
    
    for p1, p2 in zip(trainer.model.parameters(), new_trainer.model.parameters()):
        assert torch.equal(p1, p2)
