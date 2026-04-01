/* ═══════════════════════════════════════════════════════════════
   RAG System — Frontend JavaScript
   Stock Market & Investment Analysis
   ═══════════════════════════════════════════════════════════════ */

const API_BASE = '';

// ─── DOM Elements ──────────────────────────────────────────────
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadProgress = document.getElementById('upload-progress');
const progressLabel = document.getElementById('progress-label');
const progressPercent = document.getElementById('progress-percent');
const progressFill = document.getElementById('progress-fill');
const progressDetail = document.getElementById('progress-detail');
const uploadResult = document.getElementById('upload-result');
const statusIndicator = document.getElementById('status-indicator');
const statusText = document.getElementById('status-text');
const chunksInfo = document.getElementById('chunks-info');
const chunksList = document.getElementById('chunks-list');
const chunkSelect = document.getElementById('chunk-select');
const btnLoadEmbedding = document.getElementById('btn-load-embedding');
const embeddingDisplay = document.getElementById('embedding-display');
const queryInput = document.getElementById('query-input');
const btnQuery = document.getElementById('btn-query');
const responseArea = document.getElementById('response-area');
const responseLoading = document.getElementById('response-loading');
const responseContent = document.getElementById('response-content');
const responseText = document.getElementById('response-text');
const responseSources = document.getElementById('response-sources');


// ─── Status Management ─────────────────────────────────────────
function setStatus(type, text) {
    statusIndicator.className = `status-badge status-${type}`;
    statusText.textContent = text;
}

async function checkStatus() {
    try {
        const res = await fetch(`${API_BASE}/api/status`);
        const data = await res.json();
        if (data.pdf_loaded) {
            setStatus('ready', `Ready — ${data.chunk_count} chunks`);
            loadChunks();
        } else if (!data.api_configured) {
            setStatus('error', 'API Key Missing');
        }
    } catch (e) {
        console.error('Status check failed:', e);
    }
}


// ═══════════════════════════════════════════════════════════════
//  UPLOAD FUNCTIONALITY
// ═══════════════════════════════════════════════════════════════

// Drag & drop handlers
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) {
        handleFileUpload(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFileUpload(e.target.files[0]);
    }
});

function showProgress(label, percent, detail = '') {
    uploadProgress.classList.remove('hidden');
    uploadResult.classList.add('hidden');
    progressLabel.textContent = label;
    progressPercent.textContent = `${percent}%`;
    progressFill.style.width = `${percent}%`;
    progressDetail.textContent = detail;
}

async function handleFileUpload(file) {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
        showResult('error', '❌ Only PDF files are accepted.');
        return;
    }

    setStatus('processing', 'Processing PDF...');
    showProgress('Uploading PDF...', 10, `File: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`);

    const formData = new FormData();
    formData.append('file', file);

    try {
        showProgress('Extracting text & generating embeddings...', 30, 'This may take a few minutes depending on PDF size...');

        const res = await fetch(`${API_BASE}/api/upload`, {
            method: 'POST',
            body: formData
        });

        showProgress('Finalizing...', 80, 'Storing in ChromaDB...');

        const data = await res.json();

        if (data.success) {
            showProgress('Complete!', 100, data.message);
            setTimeout(() => {
                uploadProgress.classList.add('hidden');
                showResult('success', `✅ ${data.message}`);
                setStatus('ready', `Ready — ${data.chunks_created} chunks`);
                loadChunks();
            }, 800);
        } else {
            showResult('error', `❌ ${data.error}`);
            setStatus('error', 'Upload Failed');
        }
    } catch (err) {
        showResult('error', `❌ Upload failed: ${err.message}`);
        setStatus('error', 'Error');
    }
}

function showResult(type, message) {
    uploadResult.classList.remove('hidden', 'success', 'error');
    uploadResult.classList.add(type);
    uploadResult.textContent = message;
}


// ═══════════════════════════════════════════════════════════════
//  CHUNKS & EMBEDDINGS VIEWER
// ═══════════════════════════════════════════════════════════════

// Tab switching
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById(`panel-${tab.dataset.tab}`).classList.add('active');
    });
});

async function loadChunks() {
    try {
        const res = await fetch(`${API_BASE}/api/chunks`);
        const data = await res.json();

        if (data.chunks && data.chunks.length > 0) {
            chunksInfo.innerHTML = `<span>Showing <strong>${data.total}</strong> text chunks stored in ChromaDB</span>`;

            // Render chunk cards
            chunksList.innerHTML = data.chunks.map(chunk => `
                <div class="chunk-card" id="chunk-card-${chunk.id}">
                    <div class="chunk-header">
                        <span class="chunk-id">${chunk.id}</span>
                        <span class="chunk-meta">Page ${chunk.metadata.page_number || 'N/A'} • ${chunk.text.length} chars</span>
                    </div>
                    <div class="chunk-text">${escapeHtml(chunk.text)}</div>
                </div>
            `).join('');

            // Populate embedding selector
            chunkSelect.innerHTML = '<option value="">— Select a chunk to view embedding —</option>';
            data.chunks.forEach(chunk => {
                const opt = document.createElement('option');
                opt.value = chunk.id;
                opt.textContent = `${chunk.id} — Page ${chunk.metadata.page_number || 'N/A'} — "${chunk.text.substring(0, 60)}..."`;
                chunkSelect.appendChild(opt);
            });
        }
    } catch (e) {
        chunksInfo.innerHTML = '<span>Failed to load chunks.</span>';
    }
}

// Embedding loading
chunkSelect.addEventListener('change', () => {
    btnLoadEmbedding.disabled = !chunkSelect.value;
});

btnLoadEmbedding.addEventListener('click', async () => {
    const chunkId = chunkSelect.value;
    if (!chunkId) return;

    btnLoadEmbedding.disabled = true;
    btnLoadEmbedding.textContent = 'Loading...';

    try {
        const res = await fetch(`${API_BASE}/api/embeddings/${chunkId}`);
        const data = await res.json();

        if (data.error) {
            embeddingDisplay.innerHTML = `<p style="color: var(--accent-error)">Error: ${data.error}</p>`;
        } else {
            embeddingDisplay.innerHTML = `
                <div class="embedding-meta">
                    <p><strong>Chunk ID:</strong> ${data.chunk_id}</p>
                    <p><strong>Embedding Dimension:</strong> ${data.embedding_dimension}</p>
                    <p><strong>Source Page:</strong> ${data.metadata.page_number || 'N/A'}</p>
                    <p><strong>Chunk Text:</strong> "${escapeHtml(data.text.substring(0, 200))}${data.text.length > 200 ? '...' : ''}"</p>
                </div>
                <p style="font-size: 0.78rem; color: var(--text-muted); margin-bottom: 8px;">
                    Full ${data.embedding_dimension}-dimensional vector:
                </p>
                <div class="embedding-vector">
[${data.embedding.map((v, i) => {
    const formatted = v.toFixed(8);
    return (i > 0 && i % 5 === 0 ? '\n' : '') + formatted;
}).join(', ')}]
                </div>
            `;
        }

        embeddingDisplay.classList.remove('hidden');
    } catch (e) {
        embeddingDisplay.innerHTML = `<p style="color: var(--accent-error)">Failed to load embedding: ${e.message}</p>`;
        embeddingDisplay.classList.remove('hidden');
    }

    btnLoadEmbedding.disabled = false;
    btnLoadEmbedding.textContent = 'Load Embedding';
});


// ═══════════════════════════════════════════════════════════════
//  QUERY / Q&A
// ═══════════════════════════════════════════════════════════════

// Quick question chips
document.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', () => {
        queryInput.value = chip.dataset.q;
        queryInput.focus();
    });
});

// Query submission
btnQuery.addEventListener('click', submitQuery);
queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') submitQuery();
});

async function submitQuery() {
    const question = queryInput.value.trim();
    if (!question) return;

    // Show loading state
    responseArea.classList.remove('hidden');
    responseLoading.classList.remove('hidden');
    responseContent.classList.add('hidden');
    btnQuery.disabled = true;
    setStatus('processing', 'Querying...');

    try {
        const res = await fetch(`${API_BASE}/api/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });

        const data = await res.json();

        if (data.error) {
            responseText.textContent = `Error: ${data.error}`;
            responseSources.innerHTML = '';
        } else {
            // Render answer
            responseText.textContent = data.answer;

            // Render sources
            if (data.sources && data.sources.length > 0) {
                responseSources.innerHTML = `
                    <h4>📚 Retrieved Sources (${data.chunks_retrieved} chunks)</h4>
                    ${data.sources.map(s => `
                        <div class="source-item">
                            <span class="source-badge">Page ${s.page} • ${(s.relevance_score * 100).toFixed(1)}%</span>
                            <span class="source-preview">${escapeHtml(s.preview)}</span>
                        </div>
                    `).join('')}
                `;
            }
        }

        responseLoading.classList.add('hidden');
        responseContent.classList.remove('hidden');
        setStatus('ready', 'Ready');
    } catch (err) {
        responseLoading.classList.add('hidden');
        responseContent.classList.remove('hidden');
        responseText.textContent = `Query failed: ${err.message}`;
        responseSources.innerHTML = '';
        setStatus('error', 'Query Failed');
    }

    btnQuery.disabled = false;
}


// ─── Utility ───────────────────────────────────────────────────
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}


// ─── Initialize ────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    checkStatus();
});
