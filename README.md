# 📈 RAG System - Stock Market & Investment Analysis

A full-stack, local-first Retrieval-Augmented Generation (RAG) web application designed specifically for answering complex questions about stock market fundamentals and investment strategies based on uploaded PDF textbooks (like Benjamin Graham's *The Intelligent Investor*).

## ✨ Features

* **PDF Text Extraction & Smart Chunking**: Easily upload financial PDFs. The system automatically extracts text using `PyMuPDF` and intelligently splits it into overlapping chunks optimized for semantic search.
* **Vector Embeddings via Google Gemini**: Converts text chunks into high-dimensional vector embeddings using Google's `gemini-embedding-001` model.
* **Local Persistent Vector Database**: Stores and retrieves text embeddings locally using **ChromaDB**.
* **Intelligent Q&A Assistant**: Queries the database to find the most relevant context and generates highly accurate, cited answers using **Gemini 2.5 Flash**.
* **Built-in Database Viewer**: A dedicated UI section to inspect stored text chunks and visualize their 3072-dimensional vector embeddings safely in the browser.
* **API Key Rotation**: Automatically falls back to secondary API keys if free-tier rate limits are hit.

## 🛠️ Tech Stack

* **Backend**: Python, Flask, Flask-CORS
* **AI/Models**: Google Gemini API (`gemini-2.5-flash`, `gemini-embedding-001`)
* **Vector Database**: ChromaDB (Local Persistent)
* **Frontend**: Vanilla HTML/CSS/JavaScript with responsive, clean UI
* **PDF Processing**: PyMuPDF (`fitz`)

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* A Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/app/apikey)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Nivedita-Chhokar/RAG-System.git
   cd RAG-System
   ```

2. **Set up a virtual environment (Recommended):**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory and add your Gemini API keys:
   ```env
   GEMINI_API_KEY_1=your_first_key_here
   GEMINI_API_KEY_2=your_second_key_here
   ```

### Running the Application

Start the Flask server:
```bash
python3 app.py
```
Open your browser and navigate to `http://localhost:5000`.

### Backend Verification (CLI)
To inspect your vector database directly from the terminal (e.g., to verify embedding logic), you can run the included verification script:
```bash
python3 inspect_db.py
```
This will print the total chunks stored, metadata, text previews, and the exact numerical vector arrays straight to the console.

## 📜 License
This project is intended for educational and evaluation purposes.
