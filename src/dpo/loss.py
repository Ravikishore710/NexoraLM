from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_batch_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    average_log_prob: bool = False
) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_mask = shift_labels != -100

    log_probs = F.log_softmax(shift_logits, dim=-1)
    per_token_logps = torch.gather(
        log_probs,
        dim=2,
        index=shift_labels.clamp(min=0).unsqueeze(2)
    ).squeeze(2)

    masked_logps = per_token_logps * loss_mask
    if average_log_prob:
        return masked_logps.sum(-1) / loss_mask.sum(-1).clamp(min=1)
    return masked_logps.sum(-1)

class DPOLoss(nn.Module):
    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing

    def forward(
        self,
        policy_chosen_logits: torch.Tensor,
        policy_rejected_logits: torch.Tensor,
        reference_chosen_logits: torch.Tensor,
        reference_rejected_logits: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_chosen_logps = get_batch_logps(policy_chosen_logits, chosen_labels)
        pi_rejected_logps = get_batch_logps(policy_rejected_logits, rejected_labels)
        ref_chosen_logps = get_batch_logps(reference_chosen_logits, chosen_labels)
        ref_rejected_logps = get_batch_logps(reference_rejected_logits, rejected_labels)

        pi_logratios = pi_chosen_logps - pi_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps

        logits = self.beta * (pi_logratios - ref_logratios)

        losses = (
            -F.logsigmoid(logits) * (1 - self.label_smoothing)
            - F.logsigmoid(-logits) * self.label_smoothing
        )

        chosen_rewards = self.beta * (pi_chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = self.beta * (pi_rejected_logps - ref_rejected_logps).detach()
        reward_acc = (chosen_rewards > rejected_rewards).float().mean()

        return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean(), reward_acc
