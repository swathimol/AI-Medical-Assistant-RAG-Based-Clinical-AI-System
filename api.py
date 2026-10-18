"""FastAPI service for the AI Medical Assistant (educational use only).

Run:  uvicorn api:app --reload
Docs: http://127.0.0.1:8000/docs
"""
import os
from fastapi import FastAPI
from pydantic import BaseModel
from rag_core import (RAG, generate, DISCLAIMER, EMERGENCY_WORDS, NO_EVIDENCE,
                      DEFAULT_EMBED, DEFAULT_RERANKER)

EMBED_MODEL = os.getenv("EMBED_MODEL", DEFAULT_EMBED)
USE_RERANKER = os.getenv("USE_RERANKER", "0") == "1"
MIN_SIM = float(os.getenv("MIN_SIM", "0.30"))
USE_LOCAL = os.getenv("ENABLE_LOCAL_LLM", "1") == "1"
API_KEY = os.getenv("ANTHROPIC_API_KEY") or None

app = FastAPI(title="AI Medical Assistant API", version="1.0")
rag = RAG(embed_model=EMBED_MODEL,
          reranker=DEFAULT_RERANKER if USE_RERANKER else None)


class Question(BaseModel):
    question: str
    top_k: int = 4


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(rag.chunks),
            "reranker": USE_RERANKER, "threshold": MIN_SIM}


@app.post("/ask")
def ask(q: Question):
    emergency = any(w in q.question.lower() for w in EMERGENCY_WORDS)
    hits, top_sim = rag.retrieve(q.question, k=q.top_k)
    sources = [{"source": h["source"], "similarity": round(h["sim"], 3),
                "text": h["text"]} for h in hits]
    if top_sim < MIN_SIM:
        return {"answer": NO_EVIDENCE, "mode": "refused", "emergency_warning": emergency,
                "top_similarity": round(top_sim, 3), "sources": sources,
                "disclaimer": DISCLAIMER}
    text, mode, err = generate(rag, q.question, hits, api_key=API_KEY, use_local=USE_LOCAL)
    return {"answer": text, "mode": mode, "emergency_warning": emergency,
            "top_similarity": round(top_sim, 3), "sources": sources,
            "note": err, "disclaimer": DISCLAIMER}
