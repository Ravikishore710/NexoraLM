from typing import Generator, List, Optional
import torch
import torch.nn.functional as F
from src.model.transformer import NexoraLM
from src.tokenizer.bpe import NexoraTokenizer

def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    repetition_penalty: float = 1.0,
    generated_tokens: Optional[List[int]] = None
) -> int:
    logits = logits.clone()

    if repetition_penalty != 1.0 and generated_tokens:
        for prev_token in set(generated_tokens):
            if logits[prev_token] < 0:
                logits[prev_token] *= repetition_penalty
            else:
                logits[prev_token] /= repetition_penalty

    if temperature <= 0.0:
        return torch.argmax(logits).item()

    logits = logits / temperature

    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[-1]] = float("-inf")

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).item()
    return next_token

class NexoraGenerator:
    def __init__(self, model: NexoraLM, tokenizer: NexoraTokenizer, device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        use_cache: bool = True
    ) -> str:
        input_ids = self.tokenizer.encode(prompt, add_bos=True)
        tokens = list(input_ids)
        curr_input = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        kv_caches = None
        for _ in range(max_new_tokens):
            if use_cache:
                logits, _, kv_caches = self.model(curr_input, kv_caches=kv_caches, use_cache=True)
                next_token_logits = logits[0, -1, :]
            else:
                curr_all = torch.tensor([tokens], dtype=torch.long, device=self.device)
                logits, _, _ = self.model(curr_all, use_cache=False)
                next_token_logits = logits[0, -1, :]

            next_token = sample_next_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                generated_tokens=tokens
            )

            if next_token == self.tokenizer.eos_id:
                break

            tokens.append(next_token)
            curr_input = torch.tensor([[next_token]], dtype=torch.long, device=self.device)

        return self.tokenizer.decode(tokens[len(input_ids):], skip_special_tokens=True)

    @torch.no_grad()
    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1
    ) -> Generator[str, None, None]:
        input_ids = self.tokenizer.encode(prompt, add_bos=True)
        tokens = list(input_ids)
        curr_input = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        kv_caches = None
        for _ in range(max_new_tokens):
            logits, _, kv_caches = self.model(curr_input, kv_caches=kv_caches, use_cache=True)
            next_token_logits = logits[0, -1, :]

            next_token = sample_next_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                generated_tokens=tokens
            )

            if next_token == self.tokenizer.eos_id:
                break

            tokens.append(next_token)
            token_text = self.tokenizer.decode([next_token], skip_special_tokens=True)
            yield token_text
            curr_input = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
