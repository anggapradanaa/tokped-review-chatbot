"""
indexing.py
Melakukan embedding review dan menyimpannya ke ChromaDB.

Strategi chunking v2:
- Satu dokumen = satu produk (group by product_id)
- Teks dokumen menyertakan info struktural (nama, kategori, sold, rating, ulasan)
- Metadata lengkap dan konsisten untuk numeric retrieval
"""

import os
import sys
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATA_PROCESSED_PATH,
    VECTORSTORE_PATH,
    EMBEDDING_MODEL,
    CHROMA_COLLECTION_NAME,
    MAX_REVIEWS_PER_CHUNK,
)

import chromadb
from sentence_transformers import SentenceTransformer


# ─── Init ─────────────────────────────────────────────────────────────────────

def get_chroma_client():
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    return chromadb.PersistentClient(path=VECTORSTORE_PATH)


def get_embedding_model():
    print(f"[INFO] Memuat embedding model: {EMBEDDING_MODEL}")
    return SentenceTransformer(EMBEDDING_MODEL)


# ─── Chunking: Per Product ID ─────────────────────────────────────────────────

def build_chunks(df: pd.DataFrame) -> list[dict]:
    """
    Satu chunk = satu produk (bukan random batch).

    Jika review suatu produk terlalu banyak, dipecah menjadi beberapa sub-chunk
    per MAX_REVIEWS_PER_CHUNK, tapi tetap membawa metadata produk yang sama.

    Format teks dokumen:
        Produk: <nama>
        Kategori: <kategori>
        Terjual: <sold> unit
        Rating Rata-rata: <avg_rating>/5
        Ulasan:
        [POSITIF] teks review... | [NEGATIF] teks review... | ...
    """
    chunks = []
    chunk_id = 0
    grouped = df.groupby("product_id")

    for product_id, group in tqdm(grouped, desc="Building chunks per produk"):
        product_name = str(group["product_name"].iloc[0])
        category     = str(group["category"].iloc[0])
        shop_id      = str(group["shop_id"].iloc[0])
        sold         = int(pd.to_numeric(group["sold"].iloc[0], errors="coerce") or 0)
        avg_rating   = round(float(group["rating"].mean()), 2)
        review_count = len(group)

        reviews = group[["text_clean", "rating", "sentiment"]].to_dict("records")

        # Pecah jika review terlalu banyak
        for i in range(0, len(reviews), MAX_REVIEWS_PER_CHUNK):
            batch = reviews[i : i + MAX_REVIEWS_PER_CHUNK]

            ulasan_text = " | ".join(
                f"[{r['sentiment'].upper()}] {r['text_clean']}" for r in batch
            )

            # ── Teks dokumen yang informatif ──────────────────────────────
            doc_text = (
                f"Produk: {product_name}\n"
                f"Kategori: {category}\n"
                f"Terjual: {sold} unit\n"
                f"Rating Rata-rata: {avg_rating}/5\n"
                f"Jumlah Ulasan: {review_count}\n"
                f"Ulasan:\n{ulasan_text}"
            )

            # ── Metadata (primitif saja: str / int / float) ───────────────
            metadata = {
                "product_id":   str(product_id),
                "product_name": product_name,
                "category":     category,
                "shop_id":      shop_id,
                "sold":         sold,
                "avg_rating":   avg_rating,
                "review_count": review_count,
                "chunk_index":  i // MAX_REVIEWS_PER_CHUNK,
            }

            chunks.append({
                "id":       f"prod_{product_id}_chunk_{chunk_id}",
                "text":     doc_text,
                "metadata": metadata,
            })
            chunk_id += 1

    print(f"[INFO] Total chunks dibuat: {len(chunks)}")
    return chunks


# ─── Indexing ke ChromaDB ─────────────────────────────────────────────────────

def index_chunks(
    chunks: list[dict],
    model: SentenceTransformer,
    client: chromadb.PersistentClient,
):
    existing = [c.name for c in client.list_collections()]
    if CHROMA_COLLECTION_NAME in existing:
        print(f"[INFO] Menghapus collection lama: {CHROMA_COLLECTION_NAME}")
        client.delete_collection(CHROMA_COLLECTION_NAME)

    collection = client.create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    BATCH_SIZE = 500
    total = len(chunks)
    print(f"[INFO] Mulai indexing {total} chunks ke ChromaDB...")

    for start in tqdm(range(0, total, BATCH_SIZE), desc="Indexing"):
        batch     = chunks[start : start + BATCH_SIZE]
        ids       = [c["id"]       for c in batch]
        texts     = [c["text"]     for c in batch]
        metadatas = [c["metadata"] for c in batch]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    print(f"[INFO] Indexing selesai. Total dokumen: {collection.count()}")
    return collection


# ─── Entry Point ──────────────────────────────────────────────────────────────

def run():
    print(f"[INFO] Membaca data bersih dari: {DATA_PROCESSED_PATH}")
    df = pd.read_csv(DATA_PROCESSED_PATH, low_memory=False)
    print(f"[INFO] Total baris: {len(df)}")

    chunks = build_chunks(df)
    model  = get_embedding_model()
    client = get_chroma_client()
    index_chunks(chunks, model, client)

    print("\n[DONE] Vectorstore siap digunakan!")


if __name__ == "__main__":
    run()
