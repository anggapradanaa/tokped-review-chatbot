import os

# ─── Base Paths ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_RAW_PATH = r"D:\Perkuliahan\Data Science and Machine Learning\tokped-review-chatbot\data\raw\tokopedia-product-reviews-2019.csv"
DATA_PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_reviews.csv")
VECTORSTORE_PATH = os.path.join(BASE_DIR, "vectorstore", "chroma_db")

# ─── Model Config ─────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "llama3.2:3b"  # nama model Ollama (pastikan sudah di-pull: ollama pull llama3)

# ─── Embedding Config ─────────────────────────────────────────────────────────
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # support Bahasa Indonesia

# ─── ChromaDB Config ──────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "tokped_reviews"

# ─── Retrieval Config ─────────────────────────────────────────────────────────
TOP_K_RESULTS = 5          # jumlah dokumen yang diambil saat retrieval
MAX_REVIEWS_PER_CHUNK = 3   # jumlah review yang digabung per chunk

# ─── Category Labels ──────────────────────────────────────────────────────────
VALID_CATEGORIES = ["pertukangan", "fashion", "elektronik", "handphone", "olahraga"]

# ─── Flask Config ─────────────────────────────────────────────────────────────
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True
