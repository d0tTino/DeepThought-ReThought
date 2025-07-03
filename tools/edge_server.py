#!/usr/bin/env python3
"""Lightweight FastAPI server hosting a 4-bit quantized model."""

from __future__ import annotations

import os

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.getenv("MODEL_PATH", "./model")
PORT = int(os.getenv("PORT", "8000"))

app = FastAPI()

print("Loading model from", MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", load_in_4bit=True)


class Prompt(BaseModel):
    text: str
    max_new_tokens: int = 64


@app.post("/generate")
def generate(prompt: Prompt) -> dict[str, str]:
    inputs = tokenizer(prompt.text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=prompt.max_new_tokens)
    return {"text": tokenizer.decode(ids[0], skip_special_tokens=True)}


if __name__ == "__main__":  # pragma: no cover - CLI
    import uvicorn

    uvicorn.run("edge_server:app", host="0.0.0.0", port=PORT)
