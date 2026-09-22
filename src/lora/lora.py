import math
from typing import List, Optional
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.merged = False

        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.lora_a = nn.Parameter(torch.empty((r, in_features)))
        self.lora_b = nn.Parameter(torch.zeros((out_features, r)))
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.base_layer(x)
        if not self.merged:
            lora_out = (self.dropout(x) @ self.lora_a.T) @ self.lora_b.T
            res = res + lora_out * self.scaling
        return res

    def merge(self):
        if not self.merged:
            delta = (self.lora_b @ self.lora_a) * self.scaling
            self.base_layer.weight.data += delta
            self.merged = True

    def unmerge(self):
        if self.merged:
            delta = (self.lora_b @ self.lora_a) * self.scaling
            self.base_layer.weight.data -= delta
            self.merged = False

def _replace_modules(
    module: nn.Module,
    r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: List[str]
):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and any(target in name for target in target_modules):
            setattr(module, name, LoRALinear(child, r, lora_alpha, lora_dropout))
        else:
            _replace_modules(child, r, lora_alpha, lora_dropout, target_modules)

def apply_lora(
    model: nn.Module,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    freeze_base: bool = True
):
    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    _replace_modules(model, r, lora_alpha, lora_dropout, target_modules)

def get_trainable_parameters(model: nn.Module):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total, "ratio": trainable / total}
