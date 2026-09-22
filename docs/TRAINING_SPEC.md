# NexoraLM Training Specification

## Pretraining Hardware & Precision
- Target Accelerator: NVIDIA T4 (16 GB GDDR6 VRAM).
- Numerical Precision: Mixed precision FP16 with torch.cuda.amp GradScaler.
- Numerical Stability: FP32 used for RMSNorm reductions, RoPE frequency calculations, and softmax operations.

## Batch Size Dynamics
- Context Length: 2048 tokens.
- Micro-Batch Size: 4 sequences ($4 \times 2048 = 8,192$ tokens per forward pass).
- Gradient Accumulation Steps: 16.
- Global Batch Size: $16 \times 8,192 = 131,072$ tokens per optimizer step (~131k tokens).
- Steps for 300M Tokens: $\frac{300,000,000}{131,072} \approx 2,289$ optimizer steps.

## Optimizer & Schedule
- Optimizer: AdamW.
  - Learning Rate: $\eta_{\max} = 3.0\text{e}-4$.
  - Minimum Learning Rate: $\eta_{\min} = 3.0\text{e}-5$ (10% of peak).
  - Betas: $\beta_1 = 0.9, \beta_2 = 0.95$.
  - Weight Decay: $0.1$ applied to non-normalization, non-embedding 2D weights.
  - Epsilon: $1.0\text{e}-8$.
- Gradient Clipping: Maximum gradient L2 norm = $1.0$.
- Schedule: Cosine decay with linear warmup for the first 2% of total steps (~46 steps).

## Pre-Flight Compute Gate
- Before launching the 300M token run:
  - Run a 500-step throughput verification run on the Kaggle T4 session.
  - Compute tokens/second, memory consumption, and estimated completion time.
  - Record the formal GO / MODIFY decision in EXPERIMENT_LOG.md.

## Checkpointing & State Persistence
- Save interval: Every 500 optimizer steps.
- Checkpoint content:
  - model_state_dict
  - optimizer_state_dict
  - scheduler_state_dict
  - scaler_state_dict
  - global_step
  - tokens_seen
  - config
  - random_seed_states (torch, numpy, python)
