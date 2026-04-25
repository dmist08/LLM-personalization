import os
from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

from routes.authors  import authors_bp
from routes.generate import generate_bp
from routes.history  import history_bp

def create_app():
    app = Flask(__name__)

    # ── CORS ─────────────────────────────────────────────────────────────
    allowed_origins = os.environ.get(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://localhost:3000"
    ).split(",")

    CORS(app, resources={r"/api/*": {"origins": allowed_origins}})

    # ── Register blueprints under /api ────────────────────────────────────
    app.register_blueprint(authors_bp,  url_prefix="/api")
    app.register_blueprint(generate_bp, url_prefix="/api")
    app.register_blueprint(history_bp,  url_prefix="/api")

    # ── Health check ──────────────────────────────────────────────────────
    @app.route("/api/health")
    def health():
        return jsonify({"status": "ok", "service": "stylevector-backend"})

    # ── Global error handlers ─────────────────────────────────────────────
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not found"}), 404

    @app.errorhandler(500)
    def server_error(e):
        return jsonify({"error": "Internal server error"}), 500

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG", "true") == "true")
