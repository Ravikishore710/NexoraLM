import base64
import gzip
import os

def build():
    with open('checkpoints/tokenizer.json', 'rb') as f:
        raw = f.read()
    b64 = base64.b64encode(gzip.compress(raw)).decode('ascii')

    header = '''import base64
import csv
import gzip
import json
import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tokenizers import Tokenizer

TOKENIZER_B64_GZ = """__TOKENIZER_B64_GZ__"""

def ensure_tokenizer():
    if not os.path.exists("tokenizer.json"):
        with open("tokenizer.json", "wb") as f:
            f.write(gzip.decompress(base64.b64decode(TOKENIZER_B64_GZ.strip())))

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

def stream_and_pack_corpus(tokenizer: Tokenizer, seq_len: int = 2048):
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    current_tokens = []
    bos_id = tokenizer.token_to_id("<BOS>")
    eos_id = tokenizer.token_to_id("<EOS>")

    for item in ds:
        text = item.get("text", "").strip()
        if len(text) < 50:
            continue
        ids = [bos_id] + tokenizer.encode(text).ids + [eos_id]
        current_tokens.extend(ids)
        while len(current_tokens) >= seq_len:
            yield current_tokens[:seq_len]
            current_tokens = current_tokens[seq_len:]

def get_lr(step: int, warmup_steps: int, total_steps: int, lr: float = 3e-4, min_lr: float = 3e-5):
    if step < warmup_steps:
        return lr * (step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (lr - min_lr) * cosine_decay

def main():
    print_log("Launching NexoraLM-126M Pretraining (300M Tokens Target)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_log(f"Accelerator: {device} | Device: {torch.cuda.get_device_name(0)}")

    ensure_tokenizer()
    tokenizer = Tokenizer.from_file("tokenizer.json")
    print_log(f"Tokenizer loaded successfully. Vocab size: {tokenizer.get_vocab_size()}")

    model = NexoraLM().to(device)
    param_count = model.count_parameters()
    print_log(f"NexoraLM-126M initialized with {param_count:,} parameters.")

    seq_len = 2048
    micro_bs = 2
    grad_accum_steps = 32
    tokens_per_step = micro_bs * seq_len * grad_accum_steps
    target_tokens = 300_000_000
    total_opt_steps = target_tokens // tokens_per_step
    warmup_steps = int(0.02 * total_opt_steps)

    print_log(f"Training parameters: seq_len={seq_len}, micro_bs={micro_bs}, grad_accum={grad_accum_steps}")
    print_log(f"Effective batch size = {tokens_per_step:,} tokens/step. Target steps = {total_opt_steps:,} (Warmup = {warmup_steps})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda")

    csv_log_path = "pretraining_log.csv"
    with open(csv_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["opt_step", "loss", "ppl", "learning_rate", "tokens_seen", "tokens_per_sec", "vram_gb", "elapsed_sec"])

    data_iter = stream_and_pack_corpus(tokenizer, seq_len=seq_len)

    accum_loss = 0.0
    accum_count = 0
    opt_step = 0
    tokens_seen = 0
    start_time = time.time()

    model.train()
    optimizer.zero_grad()

    batch_seqs = []
    for seq in data_iter:
        batch_seqs.append(seq)
        if len(batch_seqs) < micro_bs:
            continue

        batch = torch.tensor(batch_seqs, dtype=torch.long, device=device)
        batch_seqs = []

        with torch.amp.autocast("cuda"):
            _, loss = model(batch, labels=batch)
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()
        accum_loss += loss.item() * grad_accum_steps
        accum_count += 1
        tokens_seen += (seq_len - 1) * micro_bs

        if accum_count == grad_accum_steps:
            opt_step += 1
            current_lr = get_lr(opt_step, warmup_steps, total_opt_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            mean_step_loss = accum_loss / grad_accum_steps
            step_ppl = math.exp(min(mean_step_loss, 20.0))
            now = time.time()
            tok_per_sec = tokens_seen / max(1.0, (now - start_time))
            vram_gb = torch.cuda.max_memory_allocated() / (1024**3)

            if opt_step % 10 == 0:
                print_log(f"Step {opt_step}/{total_opt_steps} | Loss: {mean_step_loss:.4f} | PPL: {step_ppl:.2f} | LR: {current_lr:.2e} | Tok/s: {tok_per_sec:.1f} | Tokens: {tokens_seen/1e6:.1f}M")

            with open(csv_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([opt_step, mean_step_loss, step_ppl, current_lr, tokens_seen, tok_per_sec, vram_gb, now - start_time])

            if opt_step % 500 == 0:
                ckpt_file = f"nexoralm_step_{opt_step}.pt"
                torch.save(
                    {
                        "step": opt_step,
                        "tokens_seen": tokens_seen,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": mean_step_loss
                    },
                    ckpt_file
                )
                print_log(f"Checkpoint saved: {ckpt_file}")

            accum_loss = 0.0
            accum_count = 0

            if opt_step >= total_opt_steps:
                break

    final_ckpt = "nexoralm_base_final.pt"
    torch.save(
        {
            "step": opt_step,
            "tokens_seen": tokens_seen,
            "model_state_dict": model.state_dict(),
            "loss": mean_step_loss
        },
        final_ckpt
    )
    print_log(f"Pretraining complete. Final checkpoint saved to {final_ckpt}")

if __name__ == "__main__":
    main()
'''

    os.makedirs("kaggle_pretrain_300m", exist_ok=True)
    with open("kaggle_pretrain_300m/main.py", "w", encoding="utf-8") as f:
        f.write(header.replace("__TOKENIZER_B64_GZ__", b64))
    print("kaggle_pretrain_300m/main.py generated successfully.")

if __name__ == "__main__":
    build()
