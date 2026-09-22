import os
import pytest
from src.tokenizer.bpe import NexoraTokenizer

SAMPLE_CORPUS = [
    "Machine learning algorithms build a model based on sample data.",
    "A Transformer decoder utilizes causal self-attention mechanisms.",
    "FineWeb-Edu contains curated high-quality educational material.",
    "In deep neural networks, residual connections enable training of deep layers.",
    "The attention mechanism maps a query and a set of key-value pairs to an output."
]

def test_tokenizer_training_and_special_tokens(tmp_path):
    target_vocab = 300
    tokenizer = NexoraTokenizer.train(SAMPLE_CORPUS, vocab_size=target_vocab)
    
    assert tokenizer.vocab_size == target_vocab
    assert tokenizer.bos_id is not None
    assert tokenizer.eos_id is not None
    assert tokenizer.pad_id is not None
    assert tokenizer.unk_id is not None
    
    test_text = "Machine learning algorithms"
    ids = tokenizer.encode(test_text, add_bos=True, add_eos=True)
    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id
    
    decoded = tokenizer.decode(ids[1:-1])
    assert decoded.strip() == test_text.strip()
    
    save_path = os.path.join(tmp_path, "tokenizer.json")
    tokenizer.save(save_path)
    loaded = NexoraTokenizer.load(save_path)
    assert loaded.vocab_size == target_vocab
    assert loaded.encode(test_text) == tokenizer.encode(test_text)

def test_chat_template_loss_masking():
    tokenizer = NexoraTokenizer.train(SAMPLE_CORPUS, vocab_size=300)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is attention?"},
        {"role": "assistant", "content": "Attention is a routing mechanism."}
    ]
    
    formatted = tokenizer.apply_chat_template(messages)
    input_ids = formatted["input_ids"]
    labels = formatted["labels"]
    
    assert len(input_ids) == len(labels)
    assert input_ids[0] == tokenizer.bos_id
    assert labels[0] == -100
    assert input_ids[-1] == tokenizer.eos_id
    assert labels[-1] == tokenizer.eos_id
    
    assistant_labels = [lbl for lbl in labels if lbl != -100]
    assert len(assistant_labels) > 0
