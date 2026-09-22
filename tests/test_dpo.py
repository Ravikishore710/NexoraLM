import torch
import pytest
from src.dpo.loss import DPOLoss, get_batch_logps

def test_batch_logps_masking():
    logits = torch.randn(2, 5, 20)
    labels = torch.tensor([
        [-100, -100, 3, 7, 9],
        [-100, 4, 5, -100, -100]
    ])
    logps = get_batch_logps(logits, labels)
    assert logps.shape == (2,)
    assert not torch.isnan(logps).any()

def test_dpo_loss_computation():
    dpo = DPOLoss(beta=0.1)

    batch_size = 2
    seq_len = 8
    vocab_size = 50

    policy_chosen = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    policy_rejected = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    ref_chosen = torch.randn(batch_size, seq_len, vocab_size)
    ref_rejected = torch.randn(batch_size, seq_len, vocab_size)

    chosen_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    rejected_labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    chosen_labels[:, :3] = -100
    rejected_labels[:, :3] = -100

    loss, chosen_r, rejected_r, acc = dpo(
        policy_chosen,
        policy_rejected,
        ref_chosen,
        ref_rejected,
        chosen_labels,
        rejected_labels
    )

    assert loss.item() > 0
    assert not torch.isnan(loss)
    assert 0.0 <= acc.item() <= 1.0

    loss.backward()
    assert policy_chosen.grad is not None
    assert policy_rejected.grad is not None
