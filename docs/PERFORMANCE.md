# NexoraLM Performance & Benchmark Records

## Benchmark Matrix

| Model Variant | Precision | Base Parameters | Trainable Parameters | Memory / Weights Size | Throughput / Latency | Training / Val Loss | Quality / Error Metric |
|---|---|---|---|---|---|---|---|
| NexoraMicro-Val | FP32/CUDA | 18.88M | 18.88M (100%) | 2.1 GB VRAM | N/A | Overfit: 0.0054 | Val Loss: 81.48 (PPL: 2.4e35) |
| NexoraLM-Gate (T4) | FP16/SDPA | 125.85M | 125.85M (100%) | 5.85 GB Peak VRAM | 11,042.4 tok/s | Gate Loss: 10.29 | 7.55 hrs projected 300M |
| NexoraLM-300M-Base | FP16/CUDA | 125.85M | 125.85M (100%) | 503.5 MB Checkpoint | 11,042.4 tok/s (T4) | Final Loss: 3.8744 | Tokens Seen: 299,746,304 (~300M) |
| NexoraLM-SFT | FP32/FP16 | 125.85M | 125.85M (100%) | 240.0 MB checkpoint | ~10.4 tok/s (CPU) | Step Loss: 10.43 -> Val: 10.30 | Assistant PPL: 29,745.9 |
| NexoraLM-LoRA | FP16+Adapter | 125.85M | 1,310,720 (1.03%) | ~5.2 MB Adapter | N/A | Target Modules: Q, K, V, O | Rank r=16, Alpha=32 |
| NexoraLM-QLoRA | 4-bit+Adapter | 125.85M | 1,310,720 (1.03%) | 64.69 MB Base + 5.2 MB Adapter | N/A | Base Weights 4-bit Quantized | Group Size 128 Symmetric |
| NexoraLM-DPO | FP16/CPU | 125.85M | 125.85M (Policy) | 240.0 MB checkpoint | Policy + Frozen Reference | DPO Loss: 0.6931 -> 0.3956 | Reward Margin: +0.7260, Acc: 100.0% |
| NexoraLM-INT8 | INT8 Per-Channel | 125.85M | 0 (Static Quantized) | 120.55 MB (50.2% of FP16) | Fast Linear GEMM | N/A | Mean Relative Error: 0.0078 (< 0.8%) |
| NexoraLM-INT4 | INT4 Group-Wise | 125.85M | 0 (Static Quantized) | 64.69 MB (27.0% of FP16) | Ultra-Compact Inference | N/A | Mean Relative Error: 0.1006 (~10.0%) |

## Hardware & System Profiling Summary

### 1. Training Hardware Gate (NVIDIA Tesla T4 GPU, 14.56 GB Usable VRAM)
- Micro Batch Size: 2
- Context Length: 2048 tokens
- Gradient Accumulation Steps: 32
- Effective Batch Size: 131,072 tokens per update
- Memory Consumption: 5.85 GB allocated (well below 14.56 GB ceiling)
- Hardware Throughput: 11,042.37 tokens/second
- Gate Recommendation: GO (Approved for 300M token pretraining)

### 2. Post-Training Quantization (113 Linear Projection Layers)
- Total FP16 Weights Size: 240.00 MB
- INT8 Per-Channel Symmetric: 120.55 MB (49.8% memory reduction, 0.0078 relative distortion)
- INT4 Group-Wise (Group Size = 128): 64.69 MB (73.0% memory reduction, 0.1005 relative distortion)

### 3. Direct Preference Optimization (DPO) Progression
- Step 1: Loss = 0.6931 (ln 2 baseline), Implicit Reward Margin = +0.0000, Reward Accuracy = 0.0%
- Step 2: Loss = 0.5258, Implicit Reward Margin = +0.3694, Reward Accuracy = 100.0%
- Step 3: Loss = 0.3956, Implicit Reward Margin = +0.7260, Reward Accuracy = 100.0%

### 4. Inference Serving (FastAPI OpenAI-Compatible Engine)
- Endpoints: `/health`, `/model`, `/v1/chat/completions` (JSON batch & SSE token streaming)
- Time-to-First-Token (TTFT): Monitored via iterative KV cache reuse
- KV Cache Architecture: Pre-allocated past key and value tensors per layer across 4 KV heads
