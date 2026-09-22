import json
import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from fastapi.middleware.cors import CORSMiddleware

from src.inference.generator import NexoraGenerator
from src.model.transformer import NexoraConfig, NexoraLM
from src.tokenizer.bpe import NexoraTokenizer

app = FastAPI(title="NexoraLM API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model_instance: Optional[NexoraLM] = None
tokenizer_instance: Optional[NexoraTokenizer] = None
generator_instance: Optional[NexoraGenerator] = None

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    stream: bool = False

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    stream: bool = False

def init_app(model: NexoraLM, tokenizer: NexoraTokenizer, device: str = "cpu"):
    global model_instance, tokenizer_instance, generator_instance
    model_instance = model
    tokenizer_instance = tokenizer
    generator_instance = NexoraGenerator(model, tokenizer, device=device)

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model_instance is not None}

@app.get("/model")
def get_model_info():
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return {
        "model": "NexoraLM",
        "parameters": model_instance.count_parameters(),
        "vocab_size": model_instance.config.vocab_size,
        "max_seq_len": model_instance.config.max_seq_len,
        "num_layers": model_instance.config.num_layers,
        "hidden_size": model_instance.config.hidden_size
    }

@app.post("/v1/completions")
def create_completion(req: CompletionRequest):
    if generator_instance is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    if req.stream:
        def stream_tokens():
            for token_text in generator_instance.stream_generate(
                prompt=req.prompt,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k
            ):
                payload = {
                    "id": f"cmpl-{int(time.time()*1000)}",
                    "choices": [{"text": token_text, "finish_reason": None}]
                }
                yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_tokens(), media_type="text/event-stream")

    text = generator_instance.generate(
        prompt=req.prompt,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k
    )
    return {
        "id": f"cmpl-{int(time.time()*1000)}",
        "choices": [{"text": text, "finish_reason": "stop"}]
    }

@app.post("/v1/chat/completions")
def create_chat_completion(req: ChatCompletionRequest):
    if generator_instance is None or tokenizer_instance is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    formatted = tokenizer_instance.apply_chat_template([m.model_dump() for m in req.messages])
    prompt = tokenizer_instance.decode(formatted["input_ids"], skip_special_tokens=False)

    if req.stream:
        def stream_chat():
            for token_text in generator_instance.stream_generate(
                prompt=prompt,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k
            ):
                payload = {
                    "id": f"chatcmpl-{int(time.time()*1000)}",
                    "choices": [{"delta": {"content": token_text}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_chat(), media_type="text/event-stream")

    text = generator_instance.generate(
        prompt=prompt,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k
    )
    return {
        "id": f"chatcmpl-{int(time.time()*1000)}",
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]
    }
