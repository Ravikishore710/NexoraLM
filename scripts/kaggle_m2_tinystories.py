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
    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, device: torch.device):
        return self.cos_cached[:seq_len].to(device), self.sin_cached[:seq_len].to(device)

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)

def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)

class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int):
        super().__init__()
        self.w_gate = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.w_up = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.w_down = nn.Linear(ffn_hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_size: int = 384, num_query_heads: int = 6, num_kv_heads: int = 2, head_dim: int = 64):
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

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.full((t, t), float("-inf"), device=x.device), diagonal=1)
        scores = scores + mask.unsqueeze(0).unsqueeze(1)
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(q)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, t, -1)
        return self.o_proj(out)

class MicroBlock(nn.Module):
    def __init__(self, hidden_size: int, num_query_heads: int, num_kv_heads: int, head_dim: int, ffn_hidden_size: int):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = GroupedQueryAttention(hidden_size, num_query_heads, num_kv_heads, head_dim)
        self.norm2 = RMSNorm(hidden_size)
        self.ffn = SwiGLU(hidden_size, ffn_hidden_size)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x

class NexoraMicroLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 4096,
        hidden_size: int = 384,
        num_layers: int = 8,
        num_query_heads: int = 6,
        num_kv_heads: int = 2,
        head_dim: int = 64,
        ffn_hidden_size: int = 1536,
        max_seq_len: int = 512
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.tok_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            MicroBlock(hidden_size, num_query_heads, num_kv_heads, head_dim, ffn_hidden_size)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.tok_embeddings.weight
        self.rotary = RotaryEmbedding(head_dim, max_seq_len=max_seq_len)

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

def run_m2_validation():
    print_log("Starting Milestone 2: NexoraMicro-20M Sanity Validation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_log(f"Using accelerator: {device}")
    if device.type == "cuda":
        print_log(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print_log(f"VRAM Allocated: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    print_log("Step 1: Training compact tokenizer on TinyStories train sample...")
    train_stream = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    sample_texts = []
    for item in train_stream:
        t = item.get("text", "").strip()
        if len(t) > 40:
            sample_texts.append(t)
        if len(sample_texts) >= 3000:
            break

    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=4096,
        special_tokens=["<UNK>", "<BOS>", "<EOS>", "<PAD>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tokenizer.train_from_iterator(sample_texts, trainer=trainer)
    print_log(f"Tokenizer trained. Vocab size: {tokenizer.get_vocab_size()}")

    print_log("Step 2: Initializing NexoraMicro architecture...")
    model = NexoraMicroLM(vocab_size=4096).to(device)
    param_count = model.count_parameters()
    print_log(f"NexoraMicro parameter count: {param_count:,} (~{param_count/1e6:.2f}M)")

    print_log("Step 3: Overfit sanity test on single batch...")
    overfit_ids = torch.tensor([tokenizer.encode(sample_texts[0][:256]).ids[:64]], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(50):
        optimizer.zero_grad()
        _, loss = model(overfit_ids, labels=overfit_ids)
        loss.backward()
        optimizer.step()
    print_log(f"Single batch final loss after 50 steps: {loss.item():.4f}")
    assert loss.item() < 0.2, "Overfit sanity test failed: loss did not drop below 0.2"
    print_log("Overfit test PASSED.")

    print_log("Step 4: Checkpoint and resume test...")
    ckpt_path = "m2_checkpoint.pt"
    torch.save({"model_state": model.state_dict(), "step": 50, "loss": loss.item()}, ckpt_path)
    fresh_model = NexoraMicroLM(vocab_size=4096).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    fresh_model.load_state_dict(ckpt["model_state"])

    for p1, p2 in zip(model.parameters(), fresh_model.parameters()):
        assert torch.equal(p1, p2), "Model parameter tensor mismatch after reload"

    _, fresh_loss = fresh_model(overfit_ids, labels=overfit_ids)
    assert abs(fresh_loss.item() - loss.item()) < 1e-3, "Loss mismatch on checkpoint reload"
    print_log("Checkpoint save and resume test PASSED.")

    print_log("Step 5: Evaluating validation split...")
    val_stream = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    val_texts = []
    for item in val_stream:
        t = item.get("text", "").strip()
        if len(t) > 40:
            val_texts.append(t)
        if len(val_texts) >= 100:
            break

    val_losses = []
    fresh_model.eval()
    with torch.no_grad():
        for vt in val_texts[:30]:
            v_ids = torch.tensor([tokenizer.encode(vt[:256]).ids[:64]], device=device)
            if v_ids.shape[1] > 2:
                _, v_loss = fresh_model(v_ids, labels=v_ids)
                val_losses.append(v_loss.item())

    mean_val_loss = sum(val_losses) / len(val_losses)
    val_ppl = math.exp(mean_val_loss)
    print_log(f"Validation Loss: {mean_val_loss:.4f}, Val PPL: {val_ppl:.2f}")

    results = {
        "status": "success",
        "milestone": "M2",
        "parameter_count": param_count,
        "overfit_loss": loss.item(),
        "mean_val_loss": mean_val_loss,
        "val_ppl": val_ppl,
        "timestamp": time.time()
    }
    with open("m2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print_log("Milestone 2 completed successfully. Report saved to m2_results.json")

if __name__ == "__main__":
    run_m2_validation()
