import os
from pathlib import Path
from typing import List, Dict, Optional

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tokenizers import Tokenizer
from safetensors.torch import load_model

from zyluncpt_model import ZylunCPT, cfg


MODEL_DIR = Path("./model")
MODEL_PATH = MODEL_DIR / "model.safetensors"
TOKENIZER_PATH = MODEL_DIR / "zyluncpt_tokenizer.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="ZylunCPT 0.2 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer = None
model = None
EOS_ID = None


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = []
    model: Optional[str] = "zyluncpt-0.2"


def build_prompt(message: str, history: List[Dict[str, str]]) -> str:
    prompt = (
        "<bos><system>"
        "Você é o ZylunCPT 0.2, uma IA brasileira de chat, código e ideias. "
        "Responda sempre em português do Brasil, exceto se o usuário pedir outro idioma. "
        "Seja direto, útil, natural e claro."
        "</system>\n"
    )

    for item in history[-10:]:
        role = item.get("role", "")
        content = str(item.get("content", "")).strip()

        if not content:
            continue

        if role == "user":
            prompt += f"<user>{content}</user>\n"
        elif role == "assistant":
            prompt += f"<assistant>{content}</assistant>\n"

    prompt += f"<user>{message}</user>\n<assistant>"
    return prompt


def clean_answer(text: str) -> str:
    stops = ["<eos>", "<user>", "</assistant>", "</user>", "<system>"]
    for s in stops:
        if s in text:
            text = text.split(s)[0]

    text = text.replace("<assistant>", "")
    text = text.strip()

    return text or "Não consegui gerar uma resposta boa."


def generate_answer(message: str, history: List[Dict[str, str]]) -> str:
    global model, tokenizer, EOS_ID

    prompt = build_prompt(message, history)

    ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    with torch.no_grad():
        out = raw_model.generate(
            input_ids,
            max_new_tokens=180,
            temperature=0.35,
            top_k=40,
            eos_id=EOS_ID,
        )

    new_ids = out[0, input_ids.shape[1]:].tolist()
    text = tokenizer.decode([int(i) for i in new_ids])

    return clean_answer(text)


@app.on_event("startup")
def load_everything():
    global tokenizer, model, EOS_ID

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Tokenizer não encontrado: {TOKENIZER_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}")

    print("Carregando tokenizer...")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    EOS_ID = tokenizer.token_to_id("<eos>")

    print("Carregando modelo...")
    model = ZylunCPT(cfg).to(DEVICE)
    load_model(model, str(MODEL_PATH))
    model.eval()

    print("ZylunCPT 0.2 carregado em:", DEVICE)


@app.get("/")
def root():
    return {
        "ok": True,
        "name": "ZylunCPT 0.2 API",
        "device": DEVICE,
    }


@app.get("/health")
def health():
    return {
        "ok": model is not None and tokenizer is not None,
        "model": "zyluncpt-0.2",
        "device": DEVICE,
    }


@app.post("/chat")
def chat(req: ChatRequest):
    answer = generate_answer(req.message, req.history or [])

    return {
        "answer": answer,
        "model": "zyluncpt-0.2",
    }
