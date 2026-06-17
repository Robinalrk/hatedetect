"""
Flask API for content moderation.
POST /api/v1/moderate - Classify text as threat/safe.
"""
import pickle
import numpy as np
from flask import Blueprint, request, jsonify
from src.preprocessing.cleaner import TextCleaner

api_bp = Blueprint("api", __name__)

_model = None
_vectorizer = None
_cleaner = None
_CLASSES = {0: "Hate Speech", 1: "Offensive Language", 2: "No Hate or Offensive"}


def load_models():
    global _model, _vectorizer, _cleaner
    if _model is None:
        with open('logistic_regression_model.pkl', 'rb') as f:
            _model = pickle.load(f)
    if _vectorizer is None:
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            _vectorizer = pickle.load(f)
    if _cleaner is None:
        _cleaner = TextCleaner()


@api_bp.route("/api/v1/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "content-moderation", "version": "1.0.0"})


@api_bp.route("/api/v1/moderate", methods=["POST"])
def moderate():
    """
    Moderate text content.
    Accepts JSON: {"text": "string to check"}
    Returns: {"is_threat": bool, "confidence": float, "label": str}
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    raw_text = data["text"]
    if not raw_text.strip():
        return jsonify({"error": "Empty text provided"}), 400

    load_models()
    cleaned = _cleaner.clean(raw_text)
    vectorized = _vectorizer.transform([cleaned])
    pred = _model.predict(vectorized)[0]
    probas = _model.predict_proba(vectorized)[0]
    confidence = float(np.max(probas))
    is_threat = bool(pred < 2)

    return jsonify({
        "is_threat": is_threat,
        "confidence": round(confidence, 4),
        "label": _CLASSES.get(int(pred), "Unknown"),
        "text": raw_text,
    })