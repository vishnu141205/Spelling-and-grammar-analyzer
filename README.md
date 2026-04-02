# Open-Source English Spelling Analyzer (Compiler Technique)

A web-based application that identifies and corrects English spelling using a compiler-style pipeline:

1. Lexical analysis (token stream generation)
2. Spelling validation and suggestion generation
3. Basic grammar checks
4. Structured output with corrected text
5. File upload support for `.txt` and `.md`

## Tech Stack

- Backend: Python, Flask
- NLP Support: `wordfreq` (lexicon + frequency)
- Frontend: HTML, CSS, Vanilla JavaScript

## Features Implemented

- Compiler-inspired lexical analyzer with token metadata (type, line, column)
- Spelling checker with ranked suggestions
- Basic grammar checks:
  - repeated words
  - article agreement (`a` / `an`)
  - sentence capitalization
  - ending punctuation
- Text correction preview using top suggestion per misspelled word
- File upload analysis (`.txt`, `.md`)
- Attractive, responsive UI with structured report tables

## Project Structure

```
compiler/
├── app.py
├── analyzer.py
├── requirements.txt
├── templates/
│   └── index.html
├── static/
│   ├── styles.css
│   └── app.js
└── uploads/
    └── .gitkeep
```

## Run Locally

```bash
# 1) Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Start application
python app.py
```

Open: `http://127.0.0.1:5000`

## API Endpoints

- `POST /api/analyze`
  - Body: `{ "text": "your content" }`
- `POST /api/analyze-file`
  - FormData field: `file` (txt/md)

## Deploy to GitHub

```bash
# Initialize repository
git init

# Add files
git add .

# Commit
git commit -m "Initial commit: English spelling analyzer using compiler techniques"

# Add your remote repository URL
git remote add origin https://github.com/<your-username>/<your-repo>.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Notes

- This project is intentionally rule-based for grammar checks and can be extended with deeper parsing.
- To improve spelling quality further, integrate a larger domain-specific dictionary or transformer-based checker.
