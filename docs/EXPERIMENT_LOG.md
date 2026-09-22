# NexoraLM Experiment Log

## Experiment Tracking Table

| Exp ID | Milestone | Description | Config | Tokens / Steps | Val Loss | Val PPL | Status |
|--------|-----------|-------------|--------|----------------|----------|---------|--------|
| EXP-000 | M0 | Baseline setup & validation | configs/base.yaml | N/A | N/A | N/A | Completed |
| EXP-001 | M2 | Micro-model (~18.88M) TinyStories validation on Kaggle T4 | Micro (8L, 384H, 6Q, 2KV) | 50 overfit steps | 0.0054 (overfit) / 81.48 (val) | 2.45e35 | Completed |
| EXP-002 | M4 | FineWeb-Edu Tokenizer (32k) & 126M Throughput Gate on Tesla T4 | 16L, 768H, 12Q, 4KV, SwiGLU | 64 gate steps (262k tokens) | 10.2913 | 29475.2 | Completed (GO) |

## Detailed Experiment Records

### EXP-000: Initial Setup
- Objective: Verify directory structure, configuration integrity, and Kaggle API communication.
- Outcome: Completed with active API access.

### EXP-001: M2 Micro-Model Sanity Validation (Tesla T4)
- Date: 2026-09-21
- Accelerator: NVIDIA Tesla T4 (14.56 GB available VRAM on Kaggle)
- Model Configuration:
  - Vocabulary: 4,096
  - Hidden Size: 384
  - Layers: 8
  - Query Heads: 6
  - KV Heads: 2 (GQA)
  - Head Dim: 64
  - FFN Hidden: 1,536 (SwiGLU)
  - Parameter Count: 18,880,896 (~18.88M)
- Execution Details:
  - Step 1: Byte-level BPE tokenizer trained on TinyStories train split sample.
  - Step 2: Overfit test on single batch drove loss from initial random baseline down to 0.0054 after 50 AdamW steps.
  - Step 3: Checkpoint state dict saved to disk, reloaded into a new model instance, verified identical parameter tensors (`torch.equal`) and loss consistency (< 1e-3).
  - Step 4: Held-out TinyStories validation split evaluated.
- Outcome: Confirmed math, gradient flow, RoPE, GQA attention, SwiGLU, RMSNorm, and checkpointing under real CUDA hardware execution.

### EXP-002: M4 FineWeb-Edu 15M Tokenizer & 126M Pretraining Compute Gate (Tesla T4)
- Date: 2026-09-21
- Accelerator: NVIDIA Tesla T4 (14.56 GB VRAM on Kaggle)
- Model Configuration:
  - Architecture: Decoder-only Transformer, Pre-LN RMSNorm
  - Parameters: 125,854,464 (~125.85M)
  - Layers: 16
  - Hidden Dimension: 768
  - Query Heads: 12
  - KV Heads: 4 (GQA, 3 queries per KV head)
  - Head Dimension: 64
  - FFN Hidden: 2048 (SwiGLU)
  - Vocabulary: 32,768 (Byte-level BPE)
  - Context Window: 2048
- Execution Details:
  - Step 1: Sampled ~30M characters (~15M tokens) from FineWeb-Edu training split pool.
  - Step 2: Trained Byte-level BPE tokenizer to exactly 32,768 vocabulary.
  - Step 3: Verified parameter count (125,854,464).
  - Step 4: Executed Throughput Gate on Tesla T4 with PyTorch native SDPA:
    - Micro-batch size: 2 sequences ($2 \times 2048 = 4,096$ tokens per forward pass).
    - Gradient accumulation: 32 steps ($32 \times 4,096 = 131,072$ tokens per update).
    - Measured throughput: **11,042.37 tokens/second**.
    - Peak VRAM allocation: **5.85 GB** (40.2% of 14.56 GB T4 capacity).
    - Estimated runtime for 300M tokens: **7.55 hours**.
- Gate Decision: **GO** (Fits well under 12-hour session limit and Kaggle 30-hour weekly quota).
