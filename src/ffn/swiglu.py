import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int, bias: bool = False):
        super().__init__()
        self.w_gate = nn.Linear(hidden_size, ffn_hidden_size, bias=bias)
        self.w_up = nn.Linear(hidden_size, ffn_hidden_size, bias=bias)
        self.w_down = nn.Linear(ffn_hidden_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
