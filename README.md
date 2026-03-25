# 🛒 Tokped Review Chatbot

Chatbot berbasis RAG (Retrieval-Augmented Generation) yang menjawab pertanyaan berdasarkan **40.607 ulasan produk Tokopedia 2019**.

Ditenagai oleh **Llama 3** (via Ollama) + **ChromaDB** + **Flask**.

---

## ✨ Fitur

| Mode | Contoh Pertanyaan | Retrieval |
|---|---|---|
| **Q&A Produk** | *"Apakah produk ini awet?"* | Semantic (embedding) |
| **Insight Kategori** | *"Apa keluhan terbanyak di handphone?"* | Semantic (embedding) |
| **Ranking Terlaris** | *"Handphone apa yang paling laku?"* | Numeric (sort by `sold`) |
| **Ranking Terbaik** | *"Produk fashion terbaik?"* | Numeric (sort by `avg_rating`) |

---

## 🏗️ Arsitektur v2 — Hybrid Retrieval

```
User Query
    ↓
intent_detector.py
    ├── query_type = "numeric"  → retrieve_numeric()  ← rank by sold/avg_rating
    ├── intent = "qa_product"   → retrieve_by_product() ← semantic (embedding)
    └── intent = "analytic"     → retrieve_by_category() ← semantic (embedding)
         ↓
    prompts.py  →  LLM (Llama 3 via Ollama)
         ↓
    Flask Response
```

**Perbedaan utama dari v1:**
- Chunking berubah dari random batch → **per product_id** (satu produk = satu dokumen)
- Teks dokumen menyertakan metadata terstruktur (`Terjual: X unit`, `Rating: Y/5`)
- Query "terlaris" dan "terbaik" tidak lagi bergantung pada embedding — murni numeric sort
- LLM diwajibkan menyebut angka konkret, dilarang mengarang

---

## 🗂️ Struktur Proyek

```
tokped-review-chatbot/
├── data/
│   ├── raw/                    # Dataset asli CSV
│   └── processed/              # Hasil preprocessing
├── vectorstore/chroma_db/      # ChromaDB (auto-generated)
├── src/
│   ├── preprocessing.py        # Cleaning & normalisasi teks
│   ├── indexing.py             # Chunking per product_id + embed ke ChromaDB
│   ├── retriever.py            # Hybrid: semantic + numeric retrieval
│   ├── intent_detector.py      # Deteksi intent + query_type (semantic/numeric)
│   ├── prompts.py              # 3 template prompt (QA / analytic / numeric)
│   └── rag_chain.py            # Orchestrator: routing + LLM call
├── app/
│   ├── routes.py               # Flask API endpoints
│   └── templates/index.html    # UI chat
├── scripts/run_indexing.py     # Script build vectorstore
├── config.py                   # Konfigurasi global
├── run.py                      # Entry point Flask
└── requirements.txt
```

---

## 🚀 Cara Menjalankan

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Pastikan Ollama & Llama 3 Sudah Siap

```bash
ollama pull llama3
ollama serve   # biarkan berjalan di background
```

### 3. Build Vectorstore (SEKALI SAJA)

Estimasi waktu: **30–60 menit** tergantung spesifikasi PC.

```bash
python scripts/run_indexing.py
```

> ⚠️ Jika sudah pernah build sebelumnya dan ada perubahan di `indexing.py`, jalankan ulang untuk rebuild collection.

### 4. Jalankan Flask

```bash
python run.py
```

### 5. Buka Browser

```
http://localhost:5000
```

---

## ⚙️ Konfigurasi (`config.py`)

| Parameter | Default | Keterangan |
|---|---|---|
| `DATA_RAW_PATH` | Path ke CSV | Lokasi dataset mentah |
| `LLM_MODEL` | `llama3` | Nama model Ollama |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Support Bahasa Indonesia |
| `TOP_K_RESULTS` | `10` | Jumlah dokumen yang diambil |
| `MAX_REVIEWS_PER_CHUNK` | `5` | Review per chunk per produk |
| `FLASK_PORT` | `5000` | Port Flask |

---

## 🔌 API

### `POST /api/chat`
```json
// Request
{ "query": "Handphone apa yang paling laku?", "product_id": null }

// Response
{
  "answer": "...",
  "intent": "analytic",
  "query_type": "numeric",
  "rank_by": "sold",
  "category": "handphone",
  "retrieved_count": 10,
  "sources": [{ "product_name": "...", "avg_rating": 4.5, "sold": 1200 }]
}
```

### `GET /api/health`
Cek status server.

---

## 📦 Tech Stack

- **LLM**: Llama 3 via Ollama (lokal, gratis)
- **Embedding**: `paraphrase-multilingual-MiniLM-L12-v2` (Bahasa Indonesia)
- **Vector DB**: ChromaDB (lokal, persistent)
- **Retrieval**: Hybrid — semantic (embedding) + numeric (metadata sort)
- **Backend**: Flask 3
- **Data**: Tokopedia Product Reviews 2019 (40.607 ulasan, 5 kategori)
