"""
routes.py
Flask routes/endpoints untuk chatbot Tokopedia Review.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Blueprint, render_template, request, jsonify, Response, stream_with_context
from src.rag_chain import run_rag, run_rag_stream

main = Blueprint("main", __name__)


# ─── Pages ────────────────────────────────────────────────────────────────────

@main.route("/")
def index():
    return render_template("index.html")


# ─── API Endpoints ────────────────────────────────────────────────────────────

@main.route("/api/chat", methods=["POST"])
def chat():
    """
    Endpoint utama chat (non-streaming).
    Body JSON: { query, product_id? }
    """
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Field 'query' diperlukan"}), 400

    query      = data.get("query", "").strip()
    product_id = data.get("product_id", None)

    if not query:
        return jsonify({"error": "Query tidak boleh kosong"}), 400
    if len(query) > 500:
        return jsonify({"error": "Query terlalu panjang (maks 500 karakter)"}), 400

    result = run_rag(query=query, product_id=product_id)

    return jsonify({
        "answer":          result["answer"],
        "intent":          result["intent"],
        "query_type":      result.get("query_type", "semantic"),
        "rank_by":         result.get("rank_by"),
        "category":        result["category"],
        "retrieved_count": result["retrieved_count"],
        "sources":         result["sources"],
    })


@main.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    """Streaming endpoint (Server-Sent Events)."""
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Field 'query' diperlukan"}), 400

    query      = data.get("query", "").strip()
    product_id = data.get("product_id", None)

    if not query:
        return jsonify({"error": "Query tidak boleh kosong"}), 400

    def generate():
        for chunk in run_rag_stream(query=query, product_id=product_id):
            yield chunk

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@main.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Tokped Review Chatbot is running"})
