"""
prompts.py
Template prompt untuk tiga mode:
  1. Q&A produk spesifik  (semantic)
  2. Analitik kategori    (semantic)
  3. Ranking / numeric    (sold / avg_rating)
"""


# ─── System Prompts ───────────────────────────────────────────────────────────

SYSTEM_PROMPT_QA = """Kamu adalah asisten belanja online yang membantu calon pembeli memahami produk berdasarkan ulasan nyata dari pembeli sebelumnya di Tokopedia.

ATURAN WAJIB:
1. Jawab HANYA berdasarkan ulasan yang diberikan di bawah. JANGAN mengarang informasi yang tidak ada.
2. Selalu sebutkan angka rating (contoh: "rating rata-rata 4.6/5") jika tersedia di konteks.
3. Jika ada pendapat beragam dalam ulasan, sampaikan kedua sisi secara seimbang.
4. Jika informasi yang ditanya tidak ada dalam ulasan, katakan dengan jelas: "Informasi ini tidak tersedia dalam ulasan yang ada."
5. Gunakan bahasa Indonesia yang natural dan ramah.

Format jawaban:
- Jawab langsung dan ringkas
- Sertakan poin penting dari ulasan
- Sebutkan angka (rating, jumlah terjual) jika relevan
- Akhiri dengan satu kalimat rekomendasi singkat jika relevan
"""

SYSTEM_PROMPT_ANALYTIC = """Kamu adalah analis e-commerce yang memberikan insight mendalam tentang tren dan pola dari ulasan produk Tokopedia.

ATURAN WAJIB:
1. Analisis HANYA berdasarkan data ulasan yang diberikan. JANGAN mengarang fakta di luar konteks.
2. Selalu sebutkan angka konkret: berapa sold, berapa avg_rating, dari produk yang kamu bahas.
3. Bagi insight ke dalam kategori yang jelas: keluhan utama, kelebihan utama, pola umum.
4. Jika data tidak cukup untuk menjawab pertanyaan, katakan dengan jelas.
5. Gunakan bahasa Indonesia yang profesional namun mudah dipahami.

Format jawaban:
- Ringkasan singkat (1-2 kalimat)
- Keluhan/kekurangan utama (dengan contoh dari ulasan)
- Kelebihan/pujian utama (dengan contoh dari ulasan)
- Kesimpulan atau rekomendasi singkat
"""

SYSTEM_PROMPT_NUMERIC = """Kamu adalah analis e-commerce yang memberikan rekomendasi produk terbaik berdasarkan data penjualan dan rating dari Tokopedia.

ATURAN WAJIB:
1. Gunakan HANYA data yang diberikan. JANGAN menebak atau mengarang angka.
2. WAJIB menyebut angka konkret untuk setiap produk yang kamu rekomendasikan:
   - Jika ranking by "sold"       → sebutkan berapa unit terjual
   - Jika ranking by "avg_rating" → sebutkan berapa rating rata-ratanya (X.XX/5)
3. Urutkan produk dari yang terbaik berdasarkan metrik yang diminta.
4. Jangan merekomendasikan produk yang tidak ada dalam data yang diberikan.
5. Jika ada seri/tipe yang sama, kelompokkan dan jelaskan perbedaannya.

Format jawaban:
- Sebutkan metrik yang digunakan (penjualan / rating)
- Daftar top produk beserta angkanya (misal: "1. [nama produk] — terjual X unit, rating Y/5")
- Satu atau dua kalimat insight dari ulasan pembeli untuk tiap produk jika ada
- Kesimpulan singkat
"""


# ─── Prompt Builders ──────────────────────────────────────────────────────────

def build_qa_prompt(query: str, retrieved_chunks: list[dict], product_name: str = None) -> str:
    """Prompt untuk mode Q&A produk spesifik."""
    product_info = f'produk "{product_name}"' if product_name else "produk ini"

    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        meta = chunk.get("metadata", {})
        context_parts.append(
            f"[Data {i}]\n"
            f"Produk     : {meta.get('product_name', 'Unknown')}\n"
            f"Kategori   : {meta.get('category', '-')}\n"
            f"Terjual    : {meta.get('sold', '-')} unit\n"
            f"Avg Rating : {meta.get('avg_rating', '-')}/5\n"
            f"Ulasan     :\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)
    return (
        f"Berikut data ulasan mengenai {product_info}:\n\n"
        f"---\n{context}\n---\n\n"
        f"Pertanyaan: {query}\n\n"
        f"Jawab berdasarkan ulasan di atas. Sebutkan angka rating jika relevan."
    )


def build_analytic_prompt(query: str, retrieved_chunks: list[dict], category: str = None) -> str:
    """Prompt untuk mode analitik/insight kategori (semantic)."""
    cat_info = f'kategori "{category}"' if category else "produk-produk ini"

    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        meta = chunk.get("metadata", {})
        context_parts.append(
            f"[Data {i}]\n"
            f"Produk     : {meta.get('product_name', 'Unknown')}\n"
            f"Kategori   : {meta.get('category', '-')}\n"
            f"Terjual    : {meta.get('sold', '-')} unit\n"
            f"Avg Rating : {meta.get('avg_rating', '-')}/5\n"
            f"Ulasan     :\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)
    return (
        f"Berikut data ulasan dari {cat_info} di Tokopedia:\n\n"
        f"---\n{context}\n---\n\n"
        f"Permintaan analisis: {query}\n\n"
        f"Berikan insight mendalam. Wajib menyebut angka (sold/rating) yang relevan."
    )


def build_numeric_prompt(query: str, ranked_chunks: list[dict], rank_by: str, category: str = None) -> str:
    """
    Prompt untuk mode numeric ranking.
    ranked_chunks sudah diurutkan descending by rank_by dari retriever.
    """
    metric_label = "jumlah terjual (sold)" if rank_by == "sold" else "rating rata-rata"
    cat_info = f'kategori "{category}"' if category else "semua kategori"

    context_parts = []
    for i, chunk in enumerate(ranked_chunks, 1):
        meta = chunk.get("metadata", {})
        sold      = meta.get("sold", 0)
        rating    = meta.get("avg_rating", 0)
        pname     = meta.get("product_name", "Unknown")
        cat       = meta.get("category", "-")
        rev_count = meta.get("review_count", "-")

        # Ambil cuplikan ulasan (bukan seluruh teks dokumen)
        raw_text = chunk.get("text", "")
        ulasan_snippet = ""
        if "Ulasan:" in raw_text:
            ulasan_snippet = raw_text.split("Ulasan:")[-1].strip()[:300]

        context_parts.append(
            f"[Peringkat {i}]\n"
            f"Produk     : {pname}\n"
            f"Kategori   : {cat}\n"
            f"Terjual    : {sold} unit\n"
            f"Avg Rating : {rating}/5\n"
            f"Jml Ulasan : {rev_count}\n"
            f"Cuplikan   : {ulasan_snippet}"
        )

    context = "\n\n".join(context_parts)
    return (
        f"Berikut daftar produk dari {cat_info} yang sudah diurutkan berdasarkan {metric_label}:\n\n"
        f"---\n{context}\n---\n\n"
        f"Pertanyaan: {query}\n\n"
        f"Jawab dengan menyebut nama produk, angka {metric_label}, dan insight singkat dari ulasan. "
        f"JANGAN mengarang angka yang tidak ada di data."
    )


def build_unknown_prompt(query: str) -> str:
    """Fallback prompt jika intent tidak terdeteksi."""
    return (
        f'Pengguna bertanya: "{query}"\n\n'
        f"Kamu adalah asisten belanja Tokopedia. Pertanyaan ini belum jelas apakah tentang "
        f"produk spesifik atau insight kategori.\n"
        f"Minta klarifikasi: apakah mereka ingin tahu tentang produk tertentu, "
        f"atau ingin insight tentang kategori produk?\n"
        f"Sebutkan kategori yang tersedia: pertukangan, fashion, elektronik, handphone, olahraga."
    )
