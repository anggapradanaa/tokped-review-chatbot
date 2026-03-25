"""
retriever.py
Hybrid retrieval v2:
  - SEMANTIC  → embedding similarity (query umum, review-based)
  - NUMERIC   → ranking metadata (sold / avg_rating) per product_id tanpa embedding
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    VECTORSTORE_PATH,
    EMBEDDING_MODEL,
    CHROMA_COLLECTION_NAME,
    TOP_K_RESULTS,
)

import chromadb
from sentence_transformers import SentenceTransformer


# ─── Singleton ────────────────────────────────────────────────────────────────
_embedding_model = None
_chroma_client   = None
_collection      = None


def get_resources():
    global _embedding_model, _chroma_client, _collection
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=VECTORSTORE_PATH)
    if _collection is None:
        _collection = _chroma_client.get_collection(CHROMA_COLLECTION_NAME)
    return _embedding_model, _collection


# ─── SEMANTIC Retrieval ───────────────────────────────────────────────────────

def retrieve_semantic(
    query: str,
    category: str = None,
    product_id: str = None,
    top_k: int = TOP_K_RESULTS,
) -> list[dict]:
    """
    Standard vector similarity search.
    Opsional filter by category atau product_id.
    """
    model, collection = get_resources()
    query_embedding = model.encode(query).tolist()

    where = None
    if product_id:
        where = {"product_id": {"$eq": product_id}}
    elif category:
        where = {"category": {"$eq": category}}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return _format_results(results)


# ─── NUMERIC Retrieval ────────────────────────────────────────────────────────

def retrieve_numeric(
    rank_by: str,           # "sold" | "avg_rating"
    category: str = None,
    top_k: int = TOP_K_RESULTS,
) -> list[dict]:
    """
    Ranking murni berbasis metadata numerik (sold / avg_rating).

    Langkah:
    1. Ambil semua metadata dari collection (tanpa embedding)
    2. Deduplikasi per product_id (ambil nilai terbesar)
    3. Sort descending by rank_by
    4. Return top_k produk beserta chunk teks-nya
    """
    _, collection = get_resources()

    # Ambil semua metadata (ChromaDB get() tanpa filter embedding)
    where = {"category": {"$eq": category}} if category else None

    raw = collection.get(
        where=where,
        include=["metadatas", "documents"],
        limit=50_000,   # ambil semua; ChromaDB lokal tidak masalah
    )

    if not raw or not raw.get("metadatas"):
        return []

    # ── Deduplikasi per product_id: ambil nilai numerik tertinggi ─────────────
    product_map: dict[str, dict] = {}
    for meta, doc in zip(raw["metadatas"], raw["documents"]):
        pid = meta.get("product_id", "")
        if not pid:
            continue

        score = float(meta.get(rank_by, 0) or 0)

        if pid not in product_map or score > float(product_map[pid]["metadata"].get(rank_by, 0)):
            product_map[pid] = {
                "text":     doc,
                "metadata": meta,
                "relevance_score": None,   # tidak ada cosine di sini
            }

    # ── Sort descending ───────────────────────────────────────────────────────
    ranked = sorted(
        product_map.values(),
        key=lambda x: float(x["metadata"].get(rank_by, 0)),
        reverse=True,
    )

    return ranked[:top_k]


# ─── Retrieve by Product (Q&A mode) ──────────────────────────────────────────

def retrieve_by_product(
    query: str,
    product_id: str = None,
    product_name_hint: str = None,
    top_k: int = TOP_K_RESULTS,
) -> list[dict]:
    """Semantic search terfokus pada satu produk."""
    return retrieve_semantic(
        query=query,
        product_id=product_id,
        top_k=top_k,
    )


# ─── Retrieve by Category (analytic/semantic mode) ───────────────────────────

def retrieve_by_category(
    query: str,
    category: str,
    sentiment_filter: str = None,
    top_k: int = TOP_K_RESULTS,
) -> list[dict]:
    """Semantic search dalam satu kategori, opsional filter sentimen."""
    results = retrieve_semantic(query=query, category=category, top_k=top_k)

    if sentiment_filter:
        keyword = f"[{sentiment_filter.upper()}]"
        results = [r for r in results if keyword in r["text"]]

    return results


# ─── Retrieve General ─────────────────────────────────────────────────────────

def retrieve_general(query: str, top_k: int = TOP_K_RESULTS) -> list[dict]:
    """Semantic search tanpa filter."""
    return retrieve_semantic(query=query, top_k=top_k)


# ─── Helper ───────────────────────────────────────────────────────────────────

def _format_results(results: dict) -> list[dict]:
    formatted = []
    if not results or not results.get("documents"):
        return formatted

    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, distances):
        formatted.append({
            "text":            doc,
            "metadata":        meta,
            "relevance_score": round(1 - dist, 4),
        })
    return formatted


def get_product_list(category: str = None, top_n: int = 20) -> list[dict]:
    """Ambil daftar produk unik — berguna untuk intent detector / debug."""
    _, collection = get_resources()
    where = {"category": {"$eq": category}} if category else None

    results = collection.get(
        where=where,
        include=["metadatas"],
        limit=5000,
    )

    if not results or not results.get("metadatas"):
        return []

    seen, products = set(), []
    for meta in results["metadatas"]:
        pid = meta.get("product_id", "")
        if pid and pid not in seen:
            seen.add(pid)
            products.append({
                "product_id":   pid,
                "product_name": meta.get("product_name", ""),
                "category":     meta.get("category", ""),
                "avg_rating":   meta.get("avg_rating", 0),
                "sold":         meta.get("sold", 0),
            })
        if len(products) >= top_n:
            break
    return products
