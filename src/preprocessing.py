"""
preprocessing.py
Membersihkan dataset Tokopedia reviews dan menyiapkannya untuk indexing.
"""

import re
import os
import pandas as pd
from config import DATA_RAW_PATH, DATA_PROCESSED_PATH, VALID_CATEGORIES


# ─── Text Cleaning ────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Bersihkan teks review dari noise."""
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Hapus URL
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Hapus karakter non-alfanumerik kecuali spasi, titik, koma, tanda tanya, tanda seru
    text = re.sub(r"[^\w\s.,?!]", " ", text)

    # Hapus angka yang berdiri sendiri (bukan bagian kata)
    text = re.sub(r"\b\d+\b", "", text)

    # Normalisasi spasi berlebih
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_slang(text: str) -> str:
    """
    Normalisasi singkatan/slang umum bahasa Indonesia.
    Daftar ini bisa diperluas sesuai kebutuhan.
    """
    slang_dict = {
        r"\byg\b": "yang",
        r"\bdgn\b": "dengan",
        r"\bdg\b": "dengan",
        r"\bkrn\b": "karena",
        r"\bkarna\b": "karena",
        r"\bbs\b": "bisa",
        r"\bbgt\b": "banget",
        r"\bbngt\b": "banget",
        r"\bsdh\b": "sudah",
        r"\budh\b": "sudah",
        r"\bblm\b": "belum",
        r"\blg\b": "lagi",
        r"\bga\b": "tidak",
        r"\bgak\b": "tidak",
        r"\bgk\b": "tidak",
        r"\bga\b": "tidak",
        r"\btdk\b": "tidak",
        r"\btp\b": "tapi",
        r"\btpi\b": "tapi",
        r"\bkl\b": "kalau",
        r"\bklo\b": "kalau",
        r"\bklw\b": "kalau",
        r"\bpke\b": "pakai",
        r"\bpk\b": "pakai",
        r"\bmksh\b": "makasih",
        r"\bmkasi\b": "makasih",
        r"\btrims\b": "terima kasih",
        r"\bkeren\b": "bagus",
        r"\boceh\b": "oke",
        r"\bokeh\b": "oke",
        r"\brecomend\b": "rekomendasi",
        r"\brecomended\b": "direkomendasikan",
        r"\bpengiriman\b": "pengiriman",
        r"\bpengirimann\b": "pengiriman",
        r"\bsgt\b": "sangat",
        r"\bkualitas\b": "kualitas",
        r"\bkualitass\b": "kualitas",
    }

    for pattern, replacement in slang_dict.items():
        text = re.sub(pattern, replacement, text)

    return text


# ─── Main Preprocessing ───────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Jalankan semua langkah preprocessing pada dataframe."""

    print(f"[INFO] Data awal: {len(df)} baris")

    # 1. Drop baris dengan text atau rating kosong
    df = df.dropna(subset=["text", "rating"])
    print(f"[INFO] Setelah drop null text/rating: {len(df)} baris")

    # 2. Filter kategori yang valid
    df = df[df["category"].isin(VALID_CATEGORIES)]
    print(f"[INFO] Setelah filter kategori valid: {len(df)} baris")

    # 3. Filter rating valid (1–5)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df[df["rating"].between(1, 5)]
    print(f"[INFO] Setelah filter rating 1-5: {len(df)} baris")

    # 4. Hapus duplikasi teks yang persis sama
    df = df.drop_duplicates(subset=["text"])
    print(f"[INFO] Setelah drop duplicate text: {len(df)} baris")

    # 5. Bersihkan teks
    print("[INFO] Membersihkan teks...")
    df["text_clean"] = df["text"].apply(clean_text)
    df["text_clean"] = df["text_clean"].apply(normalize_slang)

    # 6. Hapus baris dengan teks terlalu pendek setelah cleaning (< 5 karakter)
    df = df[df["text_clean"].str.len() >= 5]
    print(f"[INFO] Setelah filter teks pendek: {len(df)} baris")

    # 7. Tambah kolom sentiment label sederhana
    def sentiment_label(rating):
        if rating >= 4:
            return "positif"
        elif rating == 3:
            return "netral"
        else:
            return "negatif"

    df["sentiment"] = df["rating"].apply(sentiment_label)

    # 8. Pastikan kolom penting tidak null
    for col in ["product_id", "product_name", "category", "shop_id"]:
        df[col] = df[col].fillna("unknown")

    df["sold"] = pd.to_numeric(df["sold"], errors="coerce").fillna(0).astype(int)

    # Reset index
    df = df.reset_index(drop=True)

    print(f"[INFO] Preprocessing selesai. Total baris bersih: {len(df)}")
    return df


def run():
    """Entry point: load, preprocess, dan simpan ke processed."""
    print(f"[INFO] Membaca dataset dari: {DATA_RAW_PATH}")
    df = pd.read_csv(DATA_RAW_PATH, low_memory=False)

    df_clean = preprocess(df)

    # Pastikan direktori output ada
    os.makedirs(os.path.dirname(DATA_PROCESSED_PATH), exist_ok=True)

    df_clean.to_csv(DATA_PROCESSED_PATH, index=False)
    print(f"[INFO] Data bersih disimpan ke: {DATA_PROCESSED_PATH}")

    # Tampilkan ringkasan
    print("\n=== Ringkasan Dataset ===")
    print(f"Total reviews   : {len(df_clean)}")
    print(f"Kategori        : {df_clean['category'].value_counts().to_dict()}")
    print(f"Distribusi rating:\n{df_clean['rating'].value_counts().sort_index()}")
    print(f"Distribusi sentiment:\n{df_clean['sentiment'].value_counts()}")

    return df_clean


if __name__ == "__main__":
    run()
