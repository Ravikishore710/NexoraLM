import math
import os
import time
from typing import Dict, Optional
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    return LambdaLR(optimizer, lr_lambda)

class NexoraTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 3e-4,
        min_learning_rate: float = 3e-5,
        weight_decay: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.95,
        eps: float = 1e-8,
        grad_clip: float = 1.0,
        warmup_steps: int = 100,
        total_steps: int = 2500,
        gradient_accumulation_steps: int = 16,
        mixed_precision: str = "fp16",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.device = device
        self.grad_clip = grad_clip
        self.grad_accum_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision and device == "cuda"
        self.global_step = 0
        self.tokens_seen = 0

        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2 and not name.endswith(".weight"):
                decay_params.append(param)
            elif param.ndim >= 2 and "tok_embeddings" not in name and "norm" not in name:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]
        self.optimizer = AdamW(optim_groups, lr=learning_rate, betas=(beta1, beta2), eps=eps)
        min_ratio = min_learning_rate / learning_rate
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, warmup_steps, total_steps, min_lr_ratio=min_ratio)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.mixed_precision)

    def train_step(self, input_ids: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        self.model.train()
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)

        with torch.amp.autocast("cuda", enabled=self.mixed_precision):
            _, loss, _ = self.model(input_ids, labels=labels)
            loss = loss / self.grad_accum_steps

        self.scaler.scale(loss).backward()
        step_loss = loss.item() * self.grad_accum_steps
        self.tokens_seen += input_ids.numel()

        grad_norm = 0.0
        if (self.global_step + 1) % self.grad_accum_steps == 0:
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip).item()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()

        self.global_step += 1
        return {
            "loss": step_loss,
            "grad_norm": grad_norm,
            "lr": self.scheduler.get_last_lr()[0],
            "tokens_seen": self.tokens_seen
        }

    def save_checkpoint(self, path: str, extra_meta: Optional[Dict] = None):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "global_step": self.global_step,
            "tokens_seen": self.tokens_seen,
            "rng_state": torch.get_rng_state()
        }
        if extra_meta:
            payload["extra_meta"] = extra_meta
        torch.save(payload, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.global_step = ckpt["global_step"]
        self.tokens_seen = ckpt["tokens_seen"]
        torch.set_rng_state(ckpt["rng_state"])
