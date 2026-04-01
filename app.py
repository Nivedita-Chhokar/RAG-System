import os
import json
import fitz  # PyMuPDF
import chromadb
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import time

# ─── Load environment ───────────────────────────────────────────────
load_dotenv()


# ─── API Key Manager (auto-rotates on rate limit) ────────────────────
class KeyManager:
    def __init__(self):
        self.keys = []
        # Load all available keys
        for key_name in ["GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY"]:
            key = os.getenv(key_name, "")
            if key and key != "your_second_key_here" and key != "your_gemini_api_key_here" and key not in self.keys:
                self.keys.append(key)
        self.current_index = 0
        print(f"  🔑 Loaded {len(self.keys)} API key(s)")

    @property
    def current_key(self):
        if not self.keys:
            return ""
        return self.keys[self.current_index]

    def rotate(self):
        """Switch to next key. Returns True if rotated, False if no more keys."""
        if len(self.keys) <= 1:
            return False
        old_index = self.current_index
        self.current_index = (self.current_index + 1) % len(self.keys)
        if self.current_index == old_index:
            return False
        print(f"  🔄 Rotated to API key #{self.current_index + 1}")
        return True

    @property
    def has_keys(self):
        return len(self.keys) > 0


key_manager = KeyManager()

# ─── Flask app ───────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static")
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

# ─── ChromaDB setup ─────────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

# Global state
system_state = {
    "pdf_loaded": False,
    "pdf_name": "",
    "chunk_count": 0,
    "collection_name": "investment_book"
}

# ─── Gemini API base URL ────────────────────────────────────────────
GEMINI_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
GEMINI_BATCH_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents"
GEMINI_GENERATE_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"


# ═══════════════════════════════════════════════════════════════════════
#  PDF TEXT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════
def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using PyMuPDF, page by page."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            pages.append({
                "page_number": page_num + 1,
                "text": text.strip()
            })
    doc.close()
    return pages


# ═══════════════════════════════════════════════════════════════════════
#  TEXT CHUNKING
# ═══════════════════════════════════════════════════════════════════════
def chunk_text(pages, chunk_size=3000, chunk_overlap=150):
    """
    Split extracted pages into overlapping text chunks.
    Each chunk is ~3000 characters with 150 character overlap.
    """
    chunks = []
    chunk_id = 0

    for page_data in pages:
        text = page_data["text"]
        page_num = page_data["page_number"]
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                for boundary in ['. ', '.\n', '! ', '?\n', '? ', '!\n']:
                    last_boundary = text[start:end].rfind(boundary)
                    if last_boundary > chunk_size * 0.5:
                        end = start + last_boundary + len(boundary)
                        break

            chunk_text_content = text[start:end].strip()

            if chunk_text_content:
                chunks.append({
                    "id": f"chunk_{chunk_id:04d}",
                    "text": chunk_text_content,
                    "metadata": {
                        "page_number": page_num,
                        "chunk_index": chunk_id,
                        "char_start": start,
                        "char_end": end
                    }
                })
                chunk_id += 1

            start = end - chunk_overlap if end < len(text) else len(text)

    return chunks


# ═══════════════════════════════════════════════════════════════════════
#  EMBEDDING GENERATION (via REST API)
# ═══════════════════════════════════════════════════════════════════════
def generate_embedding(text, task_type="RETRIEVAL_DOCUMENT"):
    """Generate embedding using Gemini REST API (single text) with retry."""
    for attempt in range(3):
        try:
            payload = {
                "model": "models/gemini-embedding-001",
                "content": {
                    "parts": [{"text": text}]
                },
                "taskType": task_type
            }
            resp = requests.post(
                f"{GEMINI_EMBED_URL}?key={key_manager.current_key}",
                json=payload,
                timeout=30
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["embedding"]["values"]
            elif resp.status_code == 429:
                if key_manager.rotate():
                    continue
                time.sleep(10)
                continue
            else:
                print(f"Embedding API error {resp.status_code}: {resp.text[:300]}")
                return None
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    return None


def generate_batch_embeddings(texts, task_type="RETRIEVAL_DOCUMENT", max_retries=3):
    """Generate embeddings for multiple texts using batch API with retry for rate limits."""
    for attempt in range(max_retries):
        try:
            requests_list = []
            for text in texts:
                requests_list.append({
                    "model": "models/gemini-embedding-001",
                    "content": {
                        "parts": [{"text": text}]
                    },
                    "taskType": task_type
                })

            payload = {"requests": requests_list}
            resp = requests.post(
                f"{GEMINI_BATCH_EMBED_URL}?key={key_manager.current_key}",
                json=payload,
                timeout=120
            )
            if resp.status_code == 200:
                data = resp.json()
                return [emb["values"] for emb in data["embeddings"]]
            elif resp.status_code == 429:
                # Try rotating to another key first
                if key_manager.rotate():
                    print(f"  🔄 Switching API key and retrying immediately...")
                    continue
                wait_time = 30 * (attempt + 1)  # 30s, 60s, 90s
                print(f"  ⏳ Rate limited (429). Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
                continue
            else:
                print(f"Batch embedding API error {resp.status_code}: {resp.text[:300]}")
                return None
        except Exception as e:
            print(f"Error in batch embedding: {e}")
            return None
    print("  ❌ Max retries reached for batch embedding")
    return None


def generate_query_embedding(text):
    """Generate embedding for a query."""
    return generate_embedding(text, task_type="RETRIEVAL_QUERY")


# ═══════════════════════════════════════════════════════════════════════
#  LLM RESPONSE GENERATION (via REST API)
# ═══════════════════════════════════════════════════════════════════════
def generate_response(query, context_chunks):
    """Generate a response using Gemini with retrieved context."""
    context = "\n\n---\n\n".join([
        f"[Source: Page {c['metadata'].get('page_number', 'N/A')}]\n{c['text']}"
        for c in context_chunks
    ])

    prompt = f"""You are an expert investment analyst and educator. Answer the following question 
based ONLY on the provided context from the investment textbook. Be thorough, accurate, and 
cite the relevant page numbers when possible. If the context doesn't contain enough information 
to fully answer the question, state what you can answer and note what's missing.

CONTEXT FROM TEXTBOOK:
{context}

QUESTION: {query}

ANSWER (be detailed and reference the source material):"""

    for attempt in range(3):
        try:
            payload = {
                "contents": [{"parts": [{"text": prompt}]}]
            }
            resp = requests.post(
                f"{GEMINI_GENERATE_URL}?key={key_manager.current_key}",
                json=payload,
                timeout=60
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            elif resp.status_code == 429:
                if key_manager.rotate():
                    continue
                time.sleep(15)
                continue
            else:
                return f"Error generating response: {resp.status_code} - {resp.text[:300]}"
        except Exception as e:
            return f"Error generating response: {str(e)}"
    return "Error: All API keys are rate limited. Please wait a minute and try again."


# ═══════════════════════════════════════════════════════════════════════
#  API ROUTES
# ═══════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/status")
def get_status():
    """Return current system status."""
    if system_state["chunk_count"] == 0:
        try:
            collection = chroma_client.get_collection(system_state["collection_name"])
            count = collection.count()
            if count > 0:
                system_state["pdf_loaded"] = True
                system_state["chunk_count"] = count
                if not system_state["pdf_name"]:
                    system_state["pdf_name"] = "Stored Document"
        except Exception:
            pass

    return jsonify({
        "pdf_loaded": system_state["pdf_loaded"],
        "pdf_name": system_state["pdf_name"],
        "chunk_count": system_state["chunk_count"],
        "api_configured": key_manager.has_keys
    })


@app.route("/api/validate-key", methods=["POST"])
def validate_key():
    """Validate the Gemini API key with a test embedding."""
    emb = generate_embedding("test")
    if emb:
        return jsonify({"valid": True, "message": "API key is working!"})
    return jsonify({"valid": False, "message": "API key validation failed. Check your key."}), 400


@app.route("/api/upload", methods=["POST"])
def upload_pdf():
    """Upload PDF, extract text, chunk it, generate embeddings, store in ChromaDB."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    # Save file
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Validate API key first with a test embedding
    test_emb = generate_embedding("test")
    if not test_emb:
        return jsonify({"error": "Gemini API key is not working. Please update .env with a valid key and restart the server."}), 400

    try:
        # Step 1: Extract text
        pages = extract_text_from_pdf(filepath)
        if not pages:
            return jsonify({"error": "No text could be extracted from the PDF"}), 400

        # Step 2: Chunk text
        chunks = chunk_text(pages)

        # Step 3: Delete existing collection if exists, create new
        try:
            chroma_client.delete_collection(system_state["collection_name"])
        except Exception:
            pass

        collection = chroma_client.create_collection(
            name=system_state["collection_name"],
            metadata={"hnsw:space": "cosine"}
        )

        # Step 4: Generate embeddings and store in ChromaDB (batch API)
        batch_size = 20  # Keep small to avoid rate limits on free tier
        total_stored = 0
        failed_embeddings = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            ids = [c["id"] for c in batch]
            documents = [c["text"] for c in batch]
            metadatas = [c["metadata"] for c in batch]

            # Generate embeddings for entire batch in one API call
            batch_embeddings = generate_batch_embeddings(documents)

            if batch_embeddings and len(batch_embeddings) == len(documents):
                embeddings = batch_embeddings
            else:
                # Fallback: try individually with delays
                print(f"  Batch failed, falling back to individual embeddings...")
                embeddings = []
                for doc in documents:
                    emb = generate_embedding(doc)
                    if emb:
                        embeddings.append(emb)
                    else:
                        embeddings.append([0.0] * 3072)
                        failed_embeddings += 1
                    time.sleep(1)  # 1s delay between individual calls

            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            total_stored += len(batch)
            print(f"  ✅ Stored batch {i // batch_size + 1}: {total_stored}/{len(chunks)} chunks")
            time.sleep(2)  # 2s delay between batches to respect rate limits

        # Update system state
        system_state["pdf_loaded"] = True
        system_state["pdf_name"] = filename
        system_state["chunk_count"] = total_stored

        msg = f"Successfully processed {filename}: {len(pages)} pages → {total_stored} chunks"
        if failed_embeddings > 0:
            msg += f" ({failed_embeddings} chunks had embedding failures)"

        return jsonify({
            "success": True,
            "filename": filename,
            "pages_extracted": len(pages),
            "chunks_created": total_stored,
            "message": msg
        })

    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


@app.route("/api/chunks")
def get_chunks():
    """Return all stored text chunks with metadata."""
    try:
        collection = chroma_client.get_collection(system_state["collection_name"])
        results = collection.get(include=["documents", "metadatas"])

        chunks = []
        for i in range(len(results["ids"])):
            chunks.append({
                "id": results["ids"][i],
                "text": results["documents"][i],
                "metadata": results["metadatas"][i]
            })

        return jsonify({
            "chunks": chunks,
            "total": len(chunks)
        })
    except Exception as e:
        return jsonify({"chunks": [], "total": 0, "error": str(e)})


@app.route("/api/embeddings/<chunk_id>")
def get_embedding(chunk_id):
    """Return the vector embedding for a specific chunk."""
    try:
        collection = chroma_client.get_collection(system_state["collection_name"])
        results = collection.get(
            ids=[chunk_id],
            include=["embeddings", "documents", "metadatas"]
        )

        if not results["ids"]:
            return jsonify({"error": "Chunk not found"}), 404

        raw_embedding = results["embeddings"][0]
        embedding = raw_embedding.tolist() if hasattr(raw_embedding, "tolist") else list(raw_embedding)
        
        return jsonify({
            "chunk_id": chunk_id,
            "text": results["documents"][0],
            "metadata": results["metadatas"][0],
            "embedding": embedding,
            "embedding_dimension": len(embedding),
            "embedding_preview": embedding[:20]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/query", methods=["POST"])
def query_rag():
    """Accept a question, retrieve relevant chunks, generate answer."""
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "No question provided"}), 400

    question = data["question"]

    try:
        collection = chroma_client.get_collection(system_state["collection_name"])

        # Generate query embedding
        query_emb = generate_query_embedding(question)
        if not query_emb:
            return jsonify({"error": "Failed to generate query embedding. Check your API key."}), 500

        # Semantic search - retrieve top 5 relevant chunks
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=5,
            include=["documents", "metadatas", "distances"]
        )

        # Build context chunks
        context_chunks = []
        sources = []
        for i in range(len(results["ids"][0])):
            chunk = {
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            }
            context_chunks.append(chunk)
            sources.append({
                "chunk_id": chunk["id"],
                "page": chunk["metadata"].get("page_number", "N/A"),
                "relevance_score": round(1 - chunk["distance"], 4),
                "preview": chunk["text"][:150] + "..."
            })

        # Generate response
        answer = generate_response(question, context_chunks)

        return jsonify({
            "question": question,
            "answer": answer,
            "sources": sources,
            "chunks_retrieved": len(context_chunks)
        })

    except Exception as e:
        return jsonify({"error": f"Query failed: {str(e)}"}), 500


# ═══════════════════════════════════════════════════════════════════════
#  RUN SERVER
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  📚 RAG System - Stock Market & Investment Analysis")
    print("=" * 60)
    if key_manager.has_keys:
        print(f"  ✅ {len(key_manager.keys)} Gemini API Key(s) configured")
    else:
        print("  ⚠️  No Gemini API Keys set - update .env file")
    print(f"  📂 ChromaDB: {CHROMA_DB_DIR}")
    print(f"  📤 Uploads: {UPLOAD_FOLDER}")
    print("  🌐 Server: http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
