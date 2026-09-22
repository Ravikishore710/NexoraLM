# Data and Inputs Specification

## 1. Tokenizer Training Data
- Source: FineWeb-Edu training split pool.
- Volume: Exactly 15M tokens deterministically sampled.
- Partition Integrity: Sampled only after held-out validation and test sets are isolated.
- Output: Byte-level BPE with exact vocabulary size of 32,768 tokens.

## 2. Pretraining Data (M4)
- Source: HuggingFaceFW/fineweb-edu (10BT sample shards).
- Format: Parquet streaming reader.
- Budget:
  - Training: 300,000,000 tokens (primary), 500,000,000 tokens (stretch).
  - Validation: 10,000,000 tokens (held out with deterministic hash).
  - Test: 10,000,000 tokens (held out with deterministic hash).
- Quality Filters:
  - Discard empty documents or text < 50 characters.
  - Filter malformed UTF-8 characters.
  - Deduplicate based on MinHash/MD5 signature.
  - Reject documents with repetitive character ratios > 0.2.
- Packing Format:
  - Concatenated documents: `<BOS> doc1 <EOS> doc2 <EOS> ...`
  - Fixed context window chunks: $T = 2048$.
  - Inputs: tokens[0 : T-1], Targets: tokens[1 : T].

## 3. Micro-Model Validation Data (M2)
- Source: roneneldan/TinyStories.
- Training Subset: Deterministic subset of TinyStories train split (~5M tokens).
- Validation Subset: TinyStories validation split (evaluation only).

## 4. Instruction Fine-Tuning Data (M5)
- OpenAssistant Conversations (OASST1): Filtered English assistant trees (~8,000 examples).
- Reasoning Conversations: Bespoke-Stratos-17k reasoning subset (~4,000 examples).
- Format: Canonical chat schema:
  `<|system|>\n{system}\n<|user|>\n{prompt}\n<|assistant|>\n{response}`
- Loss Masking: Target tokens for system and user are masked to index -100. Only assistant tokens contribute to the cross-entropy loss.

## 5. Preference Alignment Data (M8)
- Source: argilla/ultrafeedback-binarized-preferences-cleaned.
- Train: 8,000 preference pairs (prompt, chosen, rejected).
- Eval: 1,000 held-out preference pairs.
