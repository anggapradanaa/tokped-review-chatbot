"""
run.py
Entry point untuk menjalankan Flask development server.

Jalankan dengan:
    python run.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG

app = create_app()

if __name__ == "__main__":
    print("=" * 50)
    print("  Tokped Review Chatbot — Flask Server")
    print("=" * 50)
    print(f"\n  URL: http://localhost:{FLASK_PORT}")
    print("  Pastikan Ollama sudah berjalan: ollama serve\n")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
