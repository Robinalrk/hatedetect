"""
Advanced Content Moderation & Threat Detection Pipeline.
Flask API for classifying toxic/hateful content.

Usage:
    python app.py            # Start Flask server
    python app.py --train    # Train models
    python app.py --help     # Show help
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template
from flask_cors import CORS
from src.api.routes import api_bp


def create_app():
    app = Flask(__name__)
    CORS(app)
    app.register_blueprint(api_bp)

    @app.route('/')
    def home():
        return render_template('index.html')

    return app


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "--train":
            print("Training content moderation models...")
            from src.models.trainer import ModelTrainer
            trainer = ModelTrainer()
            results = trainer.train_all()
            best = trainer.save_best_model()
            print(f"\nBest model: {best} (accuracy: {results[best]['accuracy']:.4f})")
            for name, r in results.items():
                print(f"  {name}: {r['accuracy']:.4f}")
            return
        elif sys.argv[1] in ("--help", "-h"):
            print("Usage:")
            print("  python app.py              Start Flask server")
            print("  python app.py --train      Train and save models")
            return

    app = create_app()
    print("=" * 60)
    print("CONTENT MODERATION & THREAT DETECTION API")
    print("=" * 60)
    print("  Endpoints:")
    print("    POST /api/v1/moderate   Classify text")
    print("    GET  /api/v1/health     Health check")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    main()