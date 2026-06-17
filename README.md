# Advanced Content Moderation & Threat Detection Pipeline

An automated machine learning classification pipeline designed to parse, sanitize, and flag unsafe or toxic content before it hits community or enterprise applications.

## Architecture

```
Raw Text → Clean → Vectorize (TF-IDF) → Classifier → {"is_threat": bool, "confidence": float}
```

| Layer | Technology |
|-------|-----------|
| Raw Text Sanitization | `re`, NLTK (lemmatization, stopword removal, URL/HTML/punctuation stripping) |
| Text Vectorization | `TfidfVectorizer` (5000 max features) |
| Handling Class Imbalance | `SMOTE` (Synthetic Minority Over-sampling) |
| Classifiers | `ComplementNB`, `LogisticRegression`, `RandomForestClassifier` |
| Evaluation | Confusion Matrix, Precision, Recall, F1-Score |
| API | Flask REST API (`/api/v1/moderate`) |

## Quick Start

```bash
# Setup
python -m venv venv
.\venv\Scripts\pip install -r requirements.txt

# Train models (uses labeled_data.csv)
.\venv\Scripts\python app.py --train

# Start API server
.\venv\Scripts\python app.py
```

## API

```bash
# Moderate text content
curl -X POST http://localhost:5000/api/v1/moderate \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a sample text to check"}'

# Response: {"is_threat": false, "confidence": 0.95, "label": "No Hate or Offensive"}
```

## Features

- **SMOTE** oversampling to handle highly imbalanced datasets (95% safe, 5% toxic)
- **Multi-model training** with diagnostic validation (accuracy, precision, recall, F1)
- **TF-IDF vectorization** with custom preprocessing pipeline
- **Low-latency inference** via pre-loaded models in Flask
- **Web interface** for interactive testing (available at `/`)