# Simple Local RAG (Docker Compose)

A minimal RAG pipeline with:
- Frontend: simple chat UI (`http://localhost:8080`)
- Backend: FastAPI + LangChain (`http://localhost:8000`)
- Vector DB: PostgreSQL with pgvector
- LLMs/Embeddings: Docker Desktop Model Runner (OpenAI-compatible endpoint)

## How It Works

1. On backend startup, files in `HOST_DOCS_PATH` are loaded.
2. Files are split into chunks using recursive character splitting.
3. Chunks are embedded and written into pgvector.
4. During chat, the user query is rewritten for retrieval quality.
5. Top relevant chunks are fetched and used as context for final answer generation.

Supported ingestion file types include text/code files, `.pdf`, and `.docx` (Word).
Legacy `.doc` files are not supported directly; convert them to `.docx` or `.pdf` first.

## 1) Configure

1. Copy `.env.example` to `.env`.
2. Set `HOST_DOCS_PATH` to your local folder containing documents.
3. Set `MODEL_RUNNER_BASE_URL`, `EMBEDDING_MODEL`, `QUERY_MODEL`, `ANSWER_MODEL` to values supported by your local model runner.

## 2) Start

```bash
docker compose up --build
```

The backend ingests documents from `HOST_DOCS_PATH` on startup.

## 3) Use

- Open `http://localhost:8080`
- Ask questions in the chat UI
- The backend:
  - reads + chunks local docs
  - generates embeddings
  - stores vectors in pgvector
  - expands your query using a query LLM
  - retrieves relevant chunks
  - answers with an answer LLM using retrieved context

## API quick checks

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/ingest
```

## Notes

- `RESET_COLLECTION_ON_START=true` re-creates the collection each backend restart.
- To keep previously ingested vectors, set `RESET_COLLECTION_ON_START=false`.
- If `HOST_DOCS_PATH` is a Windows path, prefer forward slashes (for example `C:/Users/name/docs`).

## File Guide

- `backend/app/main.py`: API routes and complete RAG pipeline logic.
- `docker-compose.yml`: db/backend/frontend orchestration.
- `frontend/app.js`: chat UI behavior and backend API calls.
