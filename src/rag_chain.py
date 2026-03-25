"""
rag_chain.py
Orchestrator RAG v2:
  - Routing berdasarkan query_type: semantic | numeric
  - numeric  → retrieve_numeric (ranking metadata, no embedding)
  - semantic → retrieve_semantic / by_product / by_category
"""

import os
import sys
import json
import requests
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OLLAMA_BASE_URL, LLM_MODEL, TOP_K_RESULTS

from src.intent_detector import (
    detect_intent,
    INTENT_QA, INTENT_ANALYTIC, INTENT_UNKNOWN,
    QTYPE_NUMERIC, QTYPE_SEMANTIC,
)
from src.retriever import (
    retrieve_by_product,
    retrieve_by_category,
    retrieve_general,
    retrieve_numeric,
)
from src.prompts import (
    SYSTEM_PROMPT_QA,
    SYSTEM_PROMPT_ANALYTIC,
    SYSTEM_PROMPT_NUMERIC,
    build_qa_prompt,
    build_analytic_prompt,
    build_numeric_prompt,
    build_unknown_prompt,
)


# ─── LLM Call ─────────────────────────────────────────────────────────────────

def call_ollama(system_prompt: str, user_prompt: str) -> str:
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,   # rendah → lebih deterministik, kurangi halusinasi
            "top_p": 0.9,
            "num_predict": 512,
        },
    }
    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        return (
            "❌ Gagal terhubung ke Ollama. "
            "Pastikan Ollama sudah berjalan: `ollama serve`"
        )
    except requests.exceptions.Timeout:
        return "❌ Timeout: model terlalu lama merespons. Coba lagi."
    except Exception as e:
        return f"❌ Error saat memanggil LLM: {e}"


def call_ollama_stream(system_prompt: str, user_prompt: str):
    """Generator — yield token satu per satu untuk SSE."""
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "stream": True,
        "options": {"temperature": 0.2, "top_p": 0.9, "num_predict": 512},
    }
    try:
        with requests.post(url, json=payload, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    chunk  = json.loads(line.decode("utf-8"))
                    token  = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
    except Exception as e:
        yield f"\n❌ Error streaming: {e}"


# ─── Main RAG Pipeline ────────────────────────────────────────────────────────

def run_rag(query: str, product_id: str = None) -> dict:
    """
    Pipeline lengkap:
    1. Detect intent + query_type
    2. Route ke retrieval yang tepat:
       - numeric  → retrieve_numeric  (rank by sold / avg_rating)
       - semantic QA       → retrieve_by_product
       - semantic analytic → retrieve_by_category / retrieve_general
    3. Build prompt sesuai mode
    4. Call LLM
    5. Return result dict
    """
    ir = detect_intent(query)
    intent       = ir["intent"]
    query_type   = ir["query_type"]
    rank_by      = ir["rank_by"]
    category     = ir["category"]
    product_hint = ir["product_hint"]
    sent_filter  = ir["sentiment_filter"]

    # ── Routing ───────────────────────────────────────────────────────────────

    if query_type == QTYPE_NUMERIC:
        # Numeric ranking — tidak pakai embedding
        chunks = retrieve_numeric(
            rank_by=rank_by,
            category=category,
            top_k=TOP_K_RESULTS,
        )
        system_prompt = SYSTEM_PROMPT_NUMERIC
        user_prompt   = build_numeric_prompt(
            query=query,
            ranked_chunks=chunks,
            rank_by=rank_by,
            category=category,
        )

    elif intent == INTENT_QA:
        chunks = retrieve_by_product(
            query=query,
            product_id=product_id,
            product_name_hint=product_hint,
            top_k=TOP_K_RESULTS,
        )
        system_prompt = SYSTEM_PROMPT_QA
        user_prompt   = build_qa_prompt(query, chunks, product_hint)

    elif intent == INTENT_ANALYTIC:
        if category:
            chunks = retrieve_by_category(query, category, sent_filter, TOP_K_RESULTS)
        else:
            chunks = retrieve_general(query, TOP_K_RESULTS)
        system_prompt = SYSTEM_PROMPT_ANALYTIC
        user_prompt   = build_analytic_prompt(query, chunks, category)

    else:
        chunks        = []
        system_prompt = SYSTEM_PROMPT_QA
        user_prompt   = build_unknown_prompt(query)

    # ── LLM Call ──────────────────────────────────────────────────────────────
    answer = call_ollama(system_prompt, user_prompt)

    return {
        "answer":          answer,
        "intent":          intent,
        "query_type":      query_type,
        "rank_by":         rank_by,
        "category":        category,
        "retrieved_count": len(chunks),
        "sources": [
            {
                "product_name": c["metadata"].get("product_name", ""),
                "avg_rating":   c["metadata"].get("avg_rating", ""),
                "sold":         c["metadata"].get("sold", ""),
                "relevance":    c.get("relevance_score"),
            }
            for c in chunks[:5]
        ],
    }


# ─── Streaming RAG ────────────────────────────────────────────────────────────

def run_rag_stream(query: str, product_id: str = None):
    """Versi streaming — yield SSE chunks."""
    ir = detect_intent(query)
    intent       = ir["intent"]
    query_type   = ir["query_type"]
    rank_by      = ir["rank_by"]
    category     = ir["category"]
    product_hint = ir["product_hint"]
    sent_filter  = ir["sentiment_filter"]

    if query_type == QTYPE_NUMERIC:
        chunks        = retrieve_numeric(rank_by=rank_by, category=category, top_k=TOP_K_RESULTS)
        system_prompt = SYSTEM_PROMPT_NUMERIC
        user_prompt   = build_numeric_prompt(query, chunks, rank_by, category)

    elif intent == INTENT_QA:
        chunks        = retrieve_by_product(query, product_id, product_hint, TOP_K_RESULTS)
        system_prompt = SYSTEM_PROMPT_QA
        user_prompt   = build_qa_prompt(query, chunks, product_hint)

    elif intent == INTENT_ANALYTIC:
        chunks        = retrieve_by_category(query, category, sent_filter, TOP_K_RESULTS) \
                        if category else retrieve_general(query, TOP_K_RESULTS)
        system_prompt = SYSTEM_PROMPT_ANALYTIC
        user_prompt   = build_analytic_prompt(query, chunks, category)

    else:
        chunks        = []
        system_prompt = SYSTEM_PROMPT_QA
        user_prompt   = build_unknown_prompt(query)

    # Kirim metadata dulu
    meta = {
        "type":            "meta",
        "intent":          intent,
        "query_type":      query_type,
        "rank_by":         rank_by,
        "category":        category,
        "retrieved_count": len(chunks),
    }
    yield f"data: {json.dumps(meta)}\n\n"

    for token in call_ollama_stream(system_prompt, user_prompt):
        yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"
