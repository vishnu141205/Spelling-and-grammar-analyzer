from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from analyzer import analyze_text

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"txt", "md"}

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024


def _is_allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.post("/api/analyze")
def analyze_payload():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Please provide non-empty English text for analysis."}), 400

    result = analyze_text(text)
    return jsonify(result)


@app.post("/api/analyze-file")
def analyze_file():
    uploaded_file = request.files.get("file")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({"error": "No file uploaded."}), 400

    if not _is_allowed_file(uploaded_file.filename):
        return jsonify({"error": "Only .txt or .md files are supported."}), 400

    safe_name = secure_filename(uploaded_file.filename)
    file_path = UPLOAD_DIR / safe_name
    uploaded_file.save(file_path)

    content = file_path.read_text(encoding="utf-8", errors="ignore")
    if not content.strip():
        return jsonify({"error": "Uploaded file is empty."}), 400

    result = analyze_text(content)
    result["source_file"] = safe_name
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
