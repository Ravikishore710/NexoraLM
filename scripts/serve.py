import argparse
import os
import sys
import torch
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.transformer import NexoraConfig, NexoraLM
from src.serving.app import app, init_app
from src.tokenizer.bpe import NexoraTokenizer

def main():
    parser = argparse.ArgumentParser(description="Serve NexoraLM with FastAPI")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--checkpoint", default="checkpoints/nexoralm_aligned.pt", help="Path to checkpoint")
    parser.add_argument("--tokenizer", default="checkpoints/tokenizer.json", help="Path to tokenizer.json")
    parser.add_argument("--device", default="cpu", help="Compute device (cpu, cuda)")
    args = parser.parse_args()

    print("Initializing NexoraLM for serving...")
    config = NexoraConfig()
    model = NexoraLM(config)
    if os.path.exists(args.checkpoint):
        print(f"Loading weights from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print(f"Notice: Checkpoint {args.checkpoint} not found. Running with initialized weights.")

    model.to(args.device)
    model.eval()

    if os.path.exists(args.tokenizer):
        print(f"Loading tokenizer from {args.tokenizer}...")
        tokenizer = NexoraTokenizer.load(args.tokenizer)
    else:
        print("Initializing blank tokenizer...")
        tokenizer = NexoraTokenizer()

    init_app(model, tokenizer, device=args.device)

    print(f"Starting API server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
