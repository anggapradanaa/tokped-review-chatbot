"""
intent_detector.py
Deteksi intent user:
  - intent      : qa_product | analytic | unknown
  - query_type  : semantic | numeric
  - rank_by     : sold | avg_rating | None
  - category    : nama kategori jika terdeteksi
  - sentiment_filter: positif | negatif | None
"""

import re
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import VALID_CATEGORIES


# ─── Intent Labels ────────────────────────────────────────────────────────────

INTENT_QA       = "qa_product"
INTENT_ANALYTIC = "analytic"
INTENT_UNKNOWN  = "unknown"

QTYPE_SEMANTIC = "semantic"
QTYPE_NUMERIC  = "numeric"


# ─── Numeric Query Keywords ───────────────────────────────────────────────────
# "paling laku" → rank by sold
SOLD_KEYWORDS = [
    "paling laku", "terlaris", "paling banyak terjual", "penjualan tertinggi",
    "paling diminati", "best seller", "bestseller", "paling laris",
    "terbanyak terjual", "paling sering dibeli", "paling populer",
]

# "terbaik" → rank by avg_rating
RATING_KEYWORDS = [
    "terbaik", "paling bagus", "rating tertinggi", "nilai tertinggi",
    "paling direkomendasikan", "paling memuaskan", "top rated",
    "paling disukai", "nilai terbaik", "rated tertinggi",
    "paling worth it", "paling berkualitas",
]

# ─── Analytic / Insight Keywords ──────────────────────────────────────────────
ANALYTIC_KEYWORDS = [
    "keluhan", "masalah terbanyak", "paling banyak dikeluhkan",
    "kekurangan", "kelemahan", "paling dipuji", "kelebihan",
    "trend", "tren", "insight", "ringkasan", "analisis", "analisa",
    "rata-rata rating", "overview", "summary", "rangkum", "statistik",
    "apa yang", "bagaimana kondisi", "secara umum", "perbandingan",
    "bandingkan",
]

# ─── Q&A Keywords ─────────────────────────────────────────────────────────────
QA_KEYWORDS = [
    "apakah", "bagaimana", "gimana", "kualitas", "awet",
    "worth it", "layak", "cocok untuk", "bagus ga", "bagus gak",
    "recommended", "rekomen", "pengiriman", "ukuran", "warna",
    "bahan", "material", "fungsi", "spesifikasi", "performa",
    "produk ini", "barang ini", "item ini",
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def extract_category(query: str) -> str | None:
    q = query.lower()
    for cat in VALID_CATEGORIES:
        if cat in q:
            return cat
    alias_map = {
        "hp": "handphone", "ponsel": "handphone", "smartphone": "handphone",
        "baju": "fashion", "pakaian": "fashion", "sepatu": "fashion", "tas": "fashion",
        "gadget": "elektronik",
        "alat": "pertukangan", "perkakas": "pertukangan", "tukang": "pertukangan",
        "sport": "olahraga", "fitness": "olahraga", "gym": "olahraga",
    }
    for alias, cat in alias_map.items():
        if alias in q:
            return cat
    return None


def extract_product_hint(query: str) -> str | None:
    q = query.lower()
    patterns = [
        r"produk\s+(.+?)(?:\s+ini|\s+itu|\s+apakah|\?|$)",
        r"barang\s+(.+?)(?:\s+ini|\s+itu|\s+apakah|\?|$)",
        r"tentang\s+(.+?)(?:\s+ini|\s+itu|\s+apakah|\?|$)",
        r"review\s+(.+?)(?:\s+ini|\s+itu|\s+apakah|\?|$)",
        r"ulasan\s+(.+?)(?:\s+ini|\s+itu|\s+apakah|\?|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            hint = match.group(1).strip()
            if len(hint) > 2:
                return hint
    return None


def extract_sentiment_filter(query: str) -> str | None:
    q = query.lower()
    if any(kw in q for kw in ["keluhan", "negatif", "buruk", "jelek", "kecewa", "masalah"]):
        return "negatif"
    if any(kw in q for kw in ["pujian", "positif", "bagus", "kelebihan", "dipuji"]):
        return "positif"
    return None


# ─── Main Detector ────────────────────────────────────────────────────────────

def detect_intent(query: str) -> dict:
    """
    Returns:
        {
            intent       : "qa_product" | "analytic" | "unknown"
            query_type   : "semantic" | "numeric"
            rank_by      : "sold" | "avg_rating" | None
            category     : str | None
            product_hint : str | None
            sentiment_filter: str | None
        }
    """
    q = query.lower()

    category         = extract_category(q)
    product_hint     = extract_product_hint(q)
    sentiment_filter = extract_sentiment_filter(q)

    # ── 1. Deteksi numeric query ──────────────────────────────────────────────
    is_sold_query   = any(kw in q for kw in SOLD_KEYWORDS)
    is_rating_query = any(kw in q for kw in RATING_KEYWORDS)

    if is_sold_query or is_rating_query:
        rank_by = "sold" if is_sold_query else "avg_rating"
        return {
            "intent":          INTENT_ANALYTIC,
            "query_type":      QTYPE_NUMERIC,
            "rank_by":         rank_by,
            "category":        category,
            "product_hint":    None,
            "sentiment_filter": None,
        }

    # ── 2. Deteksi analytic semantic ─────────────────────────────────────────
    analytic_score = sum(1 for kw in ANALYTIC_KEYWORDS if kw in q)
    qa_score       = sum(1 for kw in QA_KEYWORDS if kw in q)

    if analytic_score > qa_score:
        intent = INTENT_ANALYTIC
    elif qa_score > 0 or product_hint:
        intent = INTENT_QA
    elif category and analytic_score == 0 and qa_score == 0:
        intent = INTENT_ANALYTIC
    else:
        intent = INTENT_UNKNOWN

    return {
        "intent":           intent,
        "query_type":       QTYPE_SEMANTIC,
        "rank_by":          None,
        "category":         category,
        "product_hint":     product_hint,
        "sentiment_filter": sentiment_filter,
    }


# ─── Debug ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        "Handphone apa yang paling laku?",
        "Produk fashion terbaik berdasarkan rating?",
        "Apakah produk ini awet untuk pemakaian sehari-hari?",
        "Apa keluhan terbanyak di kategori elektronik?",
        "Rekomendasikan produk olahraga terlaris",
        "Gimana kualitas bahan baju fashion ini?",
    ]
    for q in tests:
        r = detect_intent(q)
        print(f"Q : {q}")
        print(f"   → intent={r['intent']} | type={r['query_type']} | rank_by={r['rank_by']} | cat={r['category']}\n")
