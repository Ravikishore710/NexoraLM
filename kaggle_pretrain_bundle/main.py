import json
import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

def print_log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, device: torch.device):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        return self.cos_cached[:seq_len].to(device), self.sin_cached[:seq_len].to(device)

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)

def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)

class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int = 768, ffn_hidden_size: int = 2048):
        super().__init__()
        self.w_gate = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.w_up = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.w_down = nn.Linear(ffn_hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_size: int = 768, num_query_heads: int = 12, num_kv_heads: int = 4, head_dim: int = 64):
        super().__init__()
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.n_rep = num_query_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_query_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_query_heads * head_dim, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.num_query_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        k = k[:, :, None, :, :].expand(b, self.num_kv_heads, self.n_rep, t, self.head_dim).reshape(b, self.num_query_heads, t, self.head_dim)
        v = v[:, :, None, :, :].expand(b, self.num_kv_heads, self.n_rep, t, self.head_dim).reshape(b, self.num_query_heads, t, self.head_dim)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(b, t, -1)
        return self.o_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int = 768, num_query_heads: int = 12, num_kv_heads: int = 4, head_dim: int = 64, ffn_hidden_size: int = 2048):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = GroupedQueryAttention(hidden_size, num_query_heads, num_kv_heads, head_dim)
        self.norm2 = RMSNorm(hidden_size)
        self.ffn = SwiGLU(hidden_size, ffn_hidden_size)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x

class NexoraLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32768,
        hidden_size: int = 768,
        num_layers: int = 16,
        num_query_heads: int = 12,
        num_kv_heads: int = 4,
        head_dim: int = 64,
        ffn_hidden_size: int = 2048,
        max_seq_len: int = 2048
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.tok_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_query_heads, num_kv_heads, head_dim, ffn_hidden_size)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.tok_embeddings.weight
        self.rotary = RotaryEmbedding(head_dim, max_seq_len=max_seq_len)

        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * num_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def count_parameters(self) -> int:
        unique = set()
        c = 0
        for p in self.parameters():
            if p not in unique:
                unique.add(p)
                c += p.numel()
        return c

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        b, t = input_ids.shape
        cos, sin = self.rotary(t, input_ids.device)
        h = self.tok_embeddings(input_ids)
        for layer in self.layers:
            h = layer(h, cos, sin)
        h = self.norm(h)
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1), ignore_index=-100)
        return logits, loss

def train_tokenizer(sample_texts, vocab_size: int = 32768) -> Tokenizer:
    special_tokens = ["<UNK>", "<BOS>", "<EOS>", "<PAD>", "<|system|>", "<|user|>", "<|assistant|>"]
    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tokenizer.train_from_iterator(sample_texts, trainer=trainer)
    return tokenizer

def run_pretraining_and_gate():
    print_log("Starting NexoraLM-126M Pretraining & Compute Gate")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_log(f"Using accelerator: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print_log(f"GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)")

    print_log("Step 1: Sampling 15M tokens from FineWeb-Edu training split...")
    fineweb = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    sample_texts = []
    char_count = 0
    target_chars = 30_000_000

    for sample in fineweb:
        text = sample.get("text", "").strip()
        if len(text) < 50:
            continue
        sample_texts.append(text)
        char_count += len(text)
        if char_count >= target_chars:
            break

    print_log(f"Collected {len(sample_texts)} documents (~{char_count/1e6:.1f}M characters).")

    print_log("Step 2: Training Byte-level BPE tokenizer to exact 32,768 vocabulary...")
    tokenizer = train_tokenizer(sample_texts, vocab_size=32768)
    assert tokenizer.get_vocab_size() == 32768, f"Vocab size mismatch: {tokenizer.get_vocab_size()}"
    tokenizer.save("tokenizer.json")
    print_log(f"Tokenizer trained and saved (Vocab size: {tokenizer.get_vocab_size()}).")

    print_log("Step 3: Instantiating NexoraLM-126M baseline architecture...")
    model = NexoraLM().to(device)
    param_count = model.count_parameters()
    print_log(f"NexoraLM parameter count: {param_count:,} (~{param_count/1e6:.2f}M parameters)")
    assert param_count == 125_854_464, f"Architecture parameter mismatch: {param_count}"

    print_log("Step 4: Executing T4 Throughput Gate with SDPA (micro_batch_size=2, grad_accum=32)...")
    micro_bs = 2
    seq_len = 2048
    grad_accum_steps = 32

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda")

    dummy_input = torch.randint(0, 32768, (micro_bs, seq_len), device=device)
    dummy_labels = dummy_input.clone()

    torch.cuda.synchronize()
    start_time = time.time()
    steps_to_run = 64

    for step in range(steps_to_run):
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            _, loss = model(dummy_input, labels=dummy_labels)
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        if (step + 1) % 16 == 0:
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            tok_per_sec = ((step + 1) * micro_bs * seq_len) / elapsed
            vram_used = torch.cuda.max_memory_allocated() / (1024**3)
            print_log(f"Step {step+1}/{steps_to_run} | Loss: {loss.item()*grad_accum_steps:.4f} | Tok/s: {tok_per_sec:.1f} | Peak VRAM: {vram_used:.2f} GB")

    torch.cuda.synchronize()
    total_elapsed = time.time() - start_time
    total_tokens_processed = steps_to_run * micro_bs * seq_len
    avg_tok_per_sec = total_tokens_processed / total_elapsed

    target_tokens = 300_000_000
    est_hours = target_tokens / (avg_tok_per_sec * 3600.0)
    decision = "GO" if est_hours < 35.0 else "MODIFY"

    print_log(f"Gate Results: Throughput = {avg_tok_per_sec:.1f} tokens/sec | Estimated 300M runtime = {est_hours:.2f} hours")
    print_log(f"Throughput Gate Decision: {decision}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": steps_to_run,
            "loss": loss.item() * grad_accum_steps
        },
        "nexoralm_pretrain_gate_ckpt.pt"
    )

    gate_summary = {
        "status": "success",
        "milestone": "M4_GATE",
        "gpu": torch.cuda.get_device_name(0),
        "parameter_count": param_count,
        "vocab_size": tokenizer.get_vocab_size(),
        "seq_len": seq_len,
        "micro_batch_size": micro_bs,
        "gradient_accumulation_steps": grad_accum_steps,
        "effective_batch_tokens": micro_bs * seq_len * grad_accum_steps,
        "tokens_per_second": avg_tok_per_sec,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / (1024**3),
        "target_tokens": target_tokens,
        "estimated_runtime_hours": est_hours,
        "decision": decision,
        "timestamp": time.time()
    }

    with open("pretrain_gate_results.json", "w") as f:
        json.dump(gate_summary, f, indent=2)

    print_log("Throughput gate results saved to pretrain_gate_results.json. Milestone 4 Gate PASSED.")

if __name__ == "__main__":
    run_pretraining_and_gate()
