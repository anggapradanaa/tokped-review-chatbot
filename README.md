# 🛒 Tokped Review Chatbot

Chatbot berbasis **RAG (Retrieval-Augmented Generation)** yang memungkinkan pengguna bertanya dalam bahasa natural tentang produk-produk Tokopedia — dijawab berdasarkan **40.607 ulasan nyata pembeli**, bukan pengetahuan umum dari AI.

Ditenagai oleh **Llama 3** (lokal via Ollama) + **ChromaDB** + **Flask**.

---

## 🧠 Apa Itu Proyek Ini?

Proyek ini adalah prototipe sistem tanya-jawab produk e-commerce yang menggabungkan dua teknologi utama:

- **RAG (Retrieval-Augmented Generation)** — sebelum menjawab, sistem terlebih dahulu mencari ulasan yang relevan dari database, lalu memberikan teks ulasan tersebut ke LLM sebagai konteks. Hasilnya, jawaban LLM berbasis data nyata, bukan hasil imajinasi model.
- **Hybrid Retrieval** — sistem punya dua jalur pencarian: semantic search (berbasis kemiripan makna) untuk pertanyaan umum, dan numeric sort (berbasis metadata angka) untuk pertanyaan ranking seperti "terlaris" atau "terbaik".

Data yang digunakan adalah **Tokopedia Product Reviews 2019**, mencakup 5 kategori produk dan lebih dari 3.600 produk unik.

---

## ✨ Fitur Utama

### 1. 📦 Ranking Produk Terlaris
Tanya produk mana yang paling banyak terjual di suatu kategori. Sistem menggunakan data `sold` langsung dari metadata — tidak pakai embedding, murni urut angka.

> Contoh: *"Handphone apa yang paling laku?"*, *"Produk olahraga terlaris apa saja?"*

### 2. ⭐ Ranking Produk Terbaik
Tanya produk mana yang punya rating tertinggi. Sistem mengurutkan berdasarkan `avg_rating` dari seluruh ulasan produk tersebut.

> Contoh: *"Produk fashion terbaik berdasarkan rating?"*, *"Elektronik top rated?"*

### 3. 📊 Insight & Analitik Kategori
Tanya pola atau tren dari ulasan di suatu kategori — keluhan terbanyak, kelebihan yang sering disebut, kondisi umum pasar.

> Contoh: *"Apa keluhan terbanyak di kategori handphone?"*, *"Apa yang paling dipuji dari produk olahraga?"*

### 4. 🔍 Q&A Produk Spesifik
Tanya tentang produk tertentu berdasarkan ulasan pembelinya — kualitas, keawetan, pengiriman, ukuran, bahan, dll.

> Contoh: *"Oppo F11 bagus ga?"*, *"Apakah produk ini awet untuk pemakaian sehari-hari?"*

### 5. 🗂️ Filter Kategori (Sidebar)
Pilih kategori di sidebar kiri sebelum bertanya untuk mempersempit pencarian. Semua query yang dikirim akan otomatis dibatasi hanya pada kategori yang dipilih.

- Jika filter aktif **Handphone** dan kamu tanya *"produk terlaris"* → hasil hanya dari handphone
- Jika kamu tanya kategori yang **berbeda** dari filter aktif → sistem akan menolak dan meminta kamu menyesuaikan

---

## 🏗️ Cara Kerja Sistem

```
User kirim pertanyaan
        ↓
[intent_detector.py]
Deteksi jenis query:
  ├── "paling laku / terlaris"  → query_type: numeric, rank_by: sold
  ├── "terbaik / rating tinggi" → query_type: numeric, rank_by: avg_rating
  ├── "keluhan / insight"       → intent: analytic, query_type: semantic
  └── "review produk X"        → intent: qa_product, query_type: semantic
        ↓
[Conflict Check]
Jika filter sidebar aktif tapi query menyebut kategori lain → tolak, minta klarifikasi
        ↓
[retriever.py]
  ├── numeric  → ambil metadata dari ChromaDB, sort by sold/avg_rating
  └── semantic → vector similarity search di ChromaDB
        ↓
[prompts.py]
Bangun prompt + konteks ulasan untuk LLM
        ↓
[Llama 3 via Ollama]
Generate jawaban berbahasa Indonesia
        ↓
Tampil di UI chat
```

---

## 💡 Tips Penggunaan

**Query yang efektif:**
- Pendek dan spesifik lebih baik daripada panjang dan bertele-tele
- Sertakan nama produk dengan jelas: *"Oppo F11 bagus ga?"* lebih akurat daripada *"bagaimana review terhadap produk oppo f11 apakah kualitasnya baik atau buruk?"*
- Gunakan filter sidebar sebelum bertanya untuk hasil yang lebih terfokus

**Batasan yang perlu diketahui:**
- Data hanya sampai tahun 2019 — tidak mencerminkan kondisi pasar saat ini
- Produk dengan sedikit ulasan cenderung menghasilkan jawaban yang kurang akurat
- Chatbot tidak bisa menjawab pertanyaan di luar konteks ulasan (harga terkini, stok, dll)

---

## 🗂️ Struktur Proyek

```
tokped-review-chatbot/
├── data/
│   ├── raw/                        # Dataset asli CSV
│   └── processed/                  # Hasil preprocessing (auto-generated)
├── vectorstore/
│   └── chroma_db/                  # ChromaDB vector store (auto-generated)
├── src/
│   ├── preprocessing.py            # Cleaning teks & normalisasi slang Indonesia
│   ├── indexing.py                 # Chunking per product_id + embed ke ChromaDB
│   ├── retriever.py                # Hybrid retrieval: semantic + numeric
│   ├── intent_detector.py          # Deteksi intent & query_type
│   ├── prompts.py                  # Template prompt untuk 3 mode
│   └── rag_chain.py                # Orchestrator pipeline + conflict detection
├── app/
│   ├── __init__.py
│   ├── routes.py                   # Flask API endpoints
│   └── templates/
│       └── index.html              # UI chat
├── scripts/
│   └── run_indexing.py             # Script build vectorstore (jalankan sekali)
├── config.py                       # Konfigurasi global
├── run.py                          # Entry point Flask
└── requirements.txt
```

---

## 🚀 Cara Menjalankan

### Prasyarat
- Python 3.10+
- [Ollama](https://ollama.com) terinstall
- Model Llama 3 sudah di-pull

### Langkah 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Langkah 2 — Pull & jalankan Ollama
```bash
ollama pull llama3:latest
ollama serve        # biarkan berjalan di background
```

### Langkah 3 — Build vectorstore *(sekali saja)*
Proses ini melakukan preprocessing data dan indexing ke ChromaDB.
Estimasi waktu: **30–60 menit** tergantung spesifikasi PC.

```bash
python scripts/run_indexing.py
```

> ⚠️ Jalankan ulang hanya jika `indexing.py` atau `preprocessing.py` diubah.

### Langkah 4 — Jalankan Flask
```bash
python run.py
```

### Langkah 5 — Buka browser
```
http://localhost:5000
```

---

## ⚙️ Konfigurasi (`config.py`)

| Parameter | Default | Keterangan |
|---|---|---|
| `DATA_RAW_PATH` | Path ke CSV | Lokasi dataset mentah |
| `LLM_MODEL` | `llama3:latest` | Nama model Ollama (sesuaikan dengan `ollama list`) |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Model embedding, support Bahasa Indonesia |
| `TOP_K_RESULTS` | `5` | Jumlah dokumen yang diambil saat retrieval |
| `MAX_REVIEWS_PER_CHUNK` | `3` | Jumlah review per chunk per produk |
| `FLASK_PORT` | `5000` | Port Flask server |

---

## 🔌 API Endpoints

### `POST /api/chat`
```json
// Request
{
  "query": "Handphone apa yang paling laku?",
  "product_id": null,
  "category": "handphone"
}

// Response
{
  "answer": "...",
  "intent": "analytic",
  "query_type": "numeric",
  "rank_by": "sold",
  "category": "handphone",
  "retrieved_count": 10,
  "sources": [
    { "product_name": "...", "avg_rating": 4.5, "sold": 1200 }
  ]
}
```

### `GET /api/health`
Cek status server.

---

## 📦 Tech Stack

| Komponen | Teknologi |
|---|---|
| LLM | Llama 3 via Ollama (lokal, gratis) |
| Embedding | `paraphrase-multilingual-MiniLM-L12-v2` |
| Vector DB | ChromaDB (lokal, persistent) |
| Retrieval | Hybrid — semantic (embedding) + numeric (metadata sort) |
| Backend | Flask 3 |
| Frontend | HTML + CSS + Vanilla JS |
| Data | Tokopedia Product Reviews 2019 (40.607 ulasan, 5 kategori) |