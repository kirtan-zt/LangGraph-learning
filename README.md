# Content Research Agent (LangGraph + FastAPI)

Users upload PDFs, text files, or paste content, then ask questions like **“What are the main findings?”** or **“Compare these reports.”** The system retrieves relevant passages from your documents, routes the request through a **LangGraph** workflow (nodes + edges + state), and returns **answer text + citations** plus an optional **downloadable Markdown report**.

This repository contains a **FastAPI** backend that:
- Stores uploaded documents and chat/message metadata in **PostgreSQL**
- Builds an offline-capable **Chroma** vector index on disk (after initial embedding model download)
- Uses a configurable chat model via LangChain’s `init_chat_model` (default: **Groq**)

---

## Features

- **Document ingestion**: Upload `.pdf`, `.txt`, or raw text
- **Document chunking + indexing**: Chunking via `RecursiveCharacterTextSplitter`, embedding via `HuggingFaceEmbeddings`, persistence in **Chroma**
- **Question answering (RAG)**: Answers grounded in retrieved chunks with **file + page citations**
- **Document summarization**: Bullet summaries with page citations
- **Multi-document comparison**: Side-by-side comparison in a Markdown table
- **Data extraction**: Pulls out numbers/metrics/tables and formats as Markdown tables
- **Insight generation**: Analytical recommendations based on the provided documents
- **Report export**: Download the latest AI response as `report.md` for a chat

---

## Prerequisites

### System requirements

- **Python**: 3.10+ recommended
- **PostgreSQL**: required (async SQLAlchemy/asyncpg)

### API keys / accounts

- **Groq API key** (default): set `GROQ_API_KEY` if you keep `LLM_PROVIDER=groq`
  - If you switch to a local provider (example: `ollama`), you may not need an API key, but you must run the provider locally.

### Notes on “offline after setup”

- **Vector search can work offline** once the embedding model is downloaded/cached and your Chroma index exists on disk.
- **LLM inference may still require network** if you use hosted models (e.g., Groq). To run fully offline, configure a local model provider supported by `langchain.chat_models.init_chat_model`.

---

## Installation

1. **Clone and enter the repo**

```bash
git clone https://github.com/kirtan-zt/LangGraph-learning.git
```

2. **Create and activate a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r app/requirements.txt
```

4. **Create your `.env`**

Create a file named `.env` in the repository root (or wherever you run from). The app loads environment variables via `pydantic-settings` + `python-dotenv`.

Example:

```bash
# App
ENV_STATE=development

# Postgres
DB_HOST=localhost
DB_PORT=5432
DB_NAME=langgraph
DB_USER=postgres
DB_PASSWORD=postgres

# Security
SECRET_KEY=change-me

# LLM (default: Groq)
GROQ_API_KEY=your_groq_key
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile

# Vector store
# IMPORTANT: this should be a directory path you can write to
VECTOR_DB_PATH=/Users/ztlab93/Desktop/work/LangGraph/vectorstore
CHUNK_SIZE=1200
CHUNK_OVERLAP=250
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

5. **Run the API**

```bash
uvicorn app.main:app --reload
```

FastAPI will create database tables on startup (`init_db()` calls `Base.metadata.create_all`).

---

## Usage / API Documentation

### Quick mental model

1. **Ingest documents** via `POST /documents/upload` (PDF/TXT/raw text)
2. **Create a chat** tied to one or more `document_id`s via `POST /chats/`
3. **Ask questions** via `POST /chats/{chat_id}/messages`
4. Optionally **download a report** via `GET /chats/{chat_id}/report`

### API endpoints

#### 1) Upload a document

**Endpoint**: `POST /documents/upload`  
**Content-Type**: `multipart/form-data`

Inputs:
- `name` (form field, required): friendly document name
- Either:
  - `file` (multipart file): `.pdf` or `.txt`
  - `text` (form field): raw text content

Response:
- `{ "document_id": "<uuid>" }`

Example (PDF):

```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -F "name=Q4 Earnings Report" \
  -F "file=@/path/to/report.pdf"
```

Example (raw text):

```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -F "name=Meeting Notes" \
  -F "text=Decisions: ... Action items: ..."
```

#### 2) List documents

**Endpoint**: `GET /documents/`  
**Response**: array of `{ id, name, upload_date }`

```bash
curl "http://localhost:8000/documents/"
```

#### 3) Create a chat for one or more documents

**Endpoint**: `POST /chats/`  
**Body**: JSON

Input payload:
- `document_ids`: list of document UUIDs (must be valid; at least one required)

Response:
- `{ id, name, documents: [...] }`

```bash
curl -X POST "http://localhost:8000/chats/" \
  -H "Content-Type: application/json" \
  -d '{
    "document_ids": ["<DOCUMENT_UUID_1>", "<DOCUMENT_UUID_2>"]
  }'
```

#### 4) Ask a question (runs retrieval + LangGraph)

**Endpoint**: `POST /chats/{chat_id}/messages`  
**Body**: JSON

Input payload:
- `content`: the user question/request (e.g., “summarize…”, “compare…”, “extract…”)

Response:
- `answer`: model output (usually the key findings)
- `references`: list of `"file_name – Page N"` strings
- `confidence`: heuristic score in \([0, 1]\)

```bash
curl -X POST "http://localhost:8000/chats/<CHAT_UUID>/messages" \
  -H "Content-Type: application/json" \
  -d '{ "content": "Compare the main findings across these reports." }'
```

#### 5) Download the latest report (Markdown)

**Endpoint**: `GET /chats/{chat_id}/report`  
**Response**: `text/markdown` (download as `report.md`)

```bash
curl -L "http://localhost:8000/chats/<CHAT_UUID>/report" -o report.md
```

#### 6) Logs (message history across chats)

**Endpoint**: `GET /logs/?skip=0&limit=100`  
**Response**: array of messages (human + AI)

```bash
curl "http://localhost:8000/logs/?skip=0&limit=50"
```

---

### How the LangGraph workflow works 

#### State (`ResearchState`)

The graph state is a typed Pydantic model that travels through nodes:
- `question`: the user’s request
- `task_type`: router decision (`summarize | qa | compare | extract | insights`)
- `documents`: retrieved `langchain_core.documents.Document` chunks
- `answer`: primary textual output
- `sources`: citation strings (`file_name – Page N`)
- `confidence`: heuristic confidence score
- `report_md`: a Markdown report representation (used by `/chats/{chat_id}/report`)

#### Nodes

- **router**: prompts the LLM to return exactly one task label and sets `state.task_type`
- **summarize**: bullet summary with citations
- **qa**: answers using ONLY retrieved context + page citations
- **compare**: produces a Markdown comparison table across documents
- **extract**: extracts metrics/tables/numbers into Markdown tables
- **insights**: generates recommendations/insights grounded in retrieved docs

#### Edges

- Entry point is `router`
- A **conditional edge** selects the next node based on `state.task_type`
- Each specialized node connects to `END`

In effect:

```text
router ──(task_type=summarize)──> summarize ──> END
   ├──(qa)───────────────> qa ───────────────> END
   ├──(compare)──────────> compare ──────────> END
   ├──(extract)──────────> extract ──────────> END
   └──(insights)─────────> insights ─────────> END
```

#### Typical request flow

When you call `POST /chats/{chat_id}/messages`:
1. The service classifies the request (`task_type`)
2. It retrieves relevant chunks from Chroma for the selected `document_ids`
3. It constructs a `ResearchState(question, documents, task_type)`
4. It runs `research_graph.ainvoke(state)`
5. It stores the AI message, citations, confidence, and `report_md` in the database

---