import json
import os
from typing import Dict, List, Optional
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

class NexoraTokenizer:
    SPECIAL_TOKENS = [
        "<UNK>",
        "<BOS>",
        "<EOS>",
        "<PAD>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>"
    ]

    def __init__(self, tokenizer: Optional[Tokenizer] = None):
        self.tokenizer = tokenizer

    @classmethod
    def train(cls, text_iterator, vocab_size: int = 32768) -> "NexoraTokenizer":
        tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=cls.SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )
        tokenizer.train_from_iterator(text_iterator, trainer=trainer)
        return cls(tokenizer)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = self.tokenizer.encode(text).ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    @property
    def bos_id(self) -> int:
        return self.tokenizer.token_to_id("<BOS>")

    @property
    def eos_id(self) -> int:
        return self.tokenizer.token_to_id("<EOS>")

    @property
    def pad_id(self) -> int:
        return self.tokenizer.token_to_id("<PAD>")

    @property
    def unk_id(self) -> int:
        return self.tokenizer.token_to_id("<UNK>")

    def apply_chat_template(self, messages: List[Dict[str, str]], is_training: bool = False) -> Dict[str, List[int]]:
        input_ids = [self.bos_id]
        labels = [-100]

        for i, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"]
            header = f"<|{role}|>\n"
            body = f"{content}\n"
            
            h_ids = self.tokenizer.encode(header).ids
            b_ids = self.tokenizer.encode(body).ids
            
            input_ids.extend(h_ids + b_ids)
            if role == "assistant":
                labels.extend([-100] * len(h_ids) + b_ids)
            else:
                labels.extend([-100] * (len(h_ids) + len(b_ids)))

        if is_training:
            input_ids.append(self.eos_id)
            labels.append(self.eos_id)
        return {"input_ids": input_ids, "labels": labels}


    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self.tokenizer.save(path)

    @classmethod
    def load(cls, path: str) -> "NexoraTokenizer":
        tokenizer = Tokenizer.from_file(path)
        return cls(tokenizer)
