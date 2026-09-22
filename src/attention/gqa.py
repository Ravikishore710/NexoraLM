import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.rope.rotary import apply_rotary_pos_emb

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, n_kv, t, d = x.shape
    return x[:, :, None, :, :].expand(b, n_kv, n_rep, t, d).reshape(b, n_kv * n_rep, t, d)

class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        num_query_heads: int = 12,
        num_kv_heads: int = 4,
        head_dim: int = 64,
        bias: bool = False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.n_rep = num_query_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_query_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_query_heads * head_dim, hidden_size, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        b, t, _ = x.shape

        q = self.q_proj(x).view(b, t, self.num_query_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rotary_pos_emb(q, cos, sin, position_ids)
        k = apply_rotary_pos_emb(k, cos, sin, position_ids)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_cache = (k, v) if use_cache else None

        k_rep = repeat_kv(k, self.n_rep)
        v_rep = repeat_kv(v, self.n_rep)

        if kv_cache is not None or t == 1:
            scores = torch.matmul(q, k_rep.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if t > 1:
                k_len = k_rep.shape[2]
                causal_mask = torch.triu(torch.full((t, k_len), float("-inf"), device=x.device), diagonal=k_len - t + 1)
                scores = scores + causal_mask.unsqueeze(0).unsqueeze(1)
            attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(q)
            output = torch.matmul(attn_weights, v_rep)
        else:
            output = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)

        output = output.transpose(1, 2).contiguous().view(b, t, -1)
        return self.o_proj(output), new_cache
