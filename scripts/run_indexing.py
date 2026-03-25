"""
scripts/run_indexing.py
Script sekali jalan untuk:
1. Preprocessing dataset mentah
2. Indexing ke ChromaDB

Jalankan SEKALI sebelum menjalankan Flask app:
    python scripts/run_indexing.py
"""

import os
import sys

# Tambahkan root project ke path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import time
from src.preprocessing import run as run_preprocessing
from src.indexing import run as run_indexing


def main():
    print("=" * 55)
    print("  Tokped Review Chatbot — Build Pipeline")
    print("=" * 55)

    # ── Step 1: Preprocessing ──────────────────────────────
    print("\n[STEP 1/2] Preprocessing dataset...")
    t0 = time.time()
    run_preprocessing()
    t1 = time.time()
    print(f"[DONE] Preprocessing selesai dalam {t1 - t0:.1f} detik\n")

    # ── Step 2: Indexing ───────────────────────────────────
    print("[STEP 2/2] Indexing ke ChromaDB...")
    t2 = time.time()
    run_indexing()
    t3 = time.time()
    print(f"[DONE] Indexing selesai dalam {t3 - t2:.1f} detik\n")

    # ── Summary ────────────────────────────────────────────
    print("=" * 55)
    print("  BUILD SELESAI!")
    print("=" * 55)
    print(f"\nTotal waktu: {t3 - t0:.1f} detik")
    print("\nLangkah selanjutnya:")
    print("  1. Pastikan Ollama berjalan:  ollama serve")
    print("  2. Jalankan Flask:            python run.py")
    print("  3. Buka browser:              http://localhost:5000")
    print()


if __name__ == "__main__":
    main()
