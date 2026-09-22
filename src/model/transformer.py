from dataclasses import dataclass
import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.attention.gqa import GroupedQueryAttention
from src.ffn.swiglu import SwiGLU
from src.normalization.rmsnorm import RMSNorm
from src.rope.rotary import RotaryEmbedding

@dataclass
class NexoraConfig:
    vocab_size: int = 32768
    max_seq_len: int = 2048
    hidden_size: int = 768
    num_layers: int = 16
    num_query_heads: int = 12
    num_kv_heads: int = 4
    head_dim: int = 64
    ffn_hidden_size: int = 2048
    norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    bias: bool = False
    rope_base: float = 10000.0

class TransformerBlock(nn.Module):
    def __init__(self, config: NexoraConfig):
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = GroupedQueryAttention(
            hidden_size=config.hidden_size,
            num_query_heads=config.num_query_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            bias=config.bias
        )
        self.post_attention_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn = SwiGLU(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.ffn_hidden_size,
            bias=config.bias
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        normed_x = self.input_norm(x)
        attn_out, new_cache = self.attn(
            normed_x,
            cos=cos,
            sin=sin,
            position_ids=position_ids,
            kv_cache=kv_cache,
            use_cache=use_cache
        )
        x = x + attn_out
        x = x + self.ffn(self.post_attention_norm(x))
        return x, new_cache

class NexoraLM(nn.Module):
    def __init__(self, config: NexoraConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

        self.rotary = RotaryEmbedding(
            dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_base
        )

        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * self.config.num_layers))

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def count_parameters(self) -> int:
        if self.config.tie_word_embeddings:
            unique_params = set()
            count = 0
            for p in self.parameters():
                if p not in unique_params:
                    unique_params.add(p)
                    count += p.numel()
            return count
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        b, t = input_ids.shape

        past_len = kv_caches[0][0].shape[2] if kv_caches is not None and kv_caches[0] is not None else 0
        total_len = past_len + t

        cos, sin = self.rotary(self.tok_embeddings.weight, total_len)

        if position_ids is None and past_len > 0:
            position_ids = torch.arange(past_len, total_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)

        h = self.tok_embeddings(input_ids)

        new_caches = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            h, updated_cache = layer(
                h,
                cos=cos,
                sin=sin,
                position_ids=position_ids,
                kv_cache=layer_cache,
                use_cache=use_cache
            )
            if use_cache:
                new_caches.append(updated_cache)

        h = self.norm(h)
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return logits, loss, new_caches
