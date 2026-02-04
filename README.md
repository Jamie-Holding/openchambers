# OpenChambers

An AI-powered research assistant for UK parliamentary debates. Query Hansard records using natural language, explore MP voting patterns, and surface evidence from parliamentary proceedings with semantic search.

## Overview

OpenChambers combines semantic search with a conversational LLM interface to make UK parliamentary records accessible and queryable. The system ingests Hansard debate transcripts, generates vector embeddings for semantic retrieval, and provides an agentic chatbot that can search debates, look up MPs and analyze voting records.

**What's inside:**

- **Semantic retrieval pipeline**: Transforms raw parliamentary XML into chunked, embedded documents with context-aware preprocessing (question/answer detection, speaker attribution and hierarchical topic tracking)
- **Agentic RAG architecture**: LangGraph-based agent that decides when to search, which filters to apply and how to synthesize evidence from multiple sources
- **Production patterns**: Streaming responses, conversation persistence, resumable batch processing and efficient vector indexing with pgvector HNSW

## What you can ask

- *"Please compare Labour and Tory views on immigration"*
  - Follow up: *"What about other parties?"*
- *"What has David Davis been talking about recently?"*
- *"What has Tim Farron said about environmental issues?"*
  - Follow up: *"What is his voting record on this - does it line up with his statements?*"
- *"What has been said about NHS waiting times in the past month?"*

The agent provides a summary to answer the user's question based on retrieved hansard utterances in addition to citing sources with dates, speakers and their party at the time. It can also cross-reference voting records with stated positions.

![Demo](demo_images/example_search_immigration.png)

![Demo](demo_images/example_search_environmental_issues.png)

![Demo](demo_images/example_search_voting_record.png)


## Key features

| Feature | Description |
|---------|-------------|
| **Chatbot agent** | LangGraph react agent inteprets nuanced user questions, automatically applies filters, runs targeted searches and summarises results
| **Semantic search** | Vector similarity search over parliamentary utterances using sentence-transformers embeddings and pgvector HNSW indexing |
| **Structured filtering** | Automatically applies filters by party, speaker, date range and resolves MP name ambiguity with user help |
| **Voting record analysis** | Query MP voting patterns across policy areas |
| **Context preservation** | Tracks debate hierarchy (oral heading → department → topic) and links answers to their triggering questions/statements |
| **Streaming chat** | Real-time token streaming via Server-Sent Events for responsive UX |
| **Conversation memory** | PostgreSQL-backed conversation persistence across sessions |
| **Resumable pipelines** | Checkpoint-based batch processing for large-scale data ingestion |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Frontend (Next.js)                        │
│              Chat UI with SSE streaming                     │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP/SSE
┌─────────────────────────▼───────────────────────────────────┐
│                   FastAPI Backend                           │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              LangGraph Agent                           │ │
│  │  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐  │ │
│  │  │  fetch   │  │ list_people  │  │ get_voting_record│  │ │
│  │  │ (search) │  │   (lookup)   │  │     (search)     │  │ │
│  │  └────┬─────┘  └──────┬───────┘  └────────┬─────────┘  │ │
│  └───────┼───────────────┼───────────────────┼────────────┘ │
└──────────┼───────────────┼───────────────────┼──────────────┘
           │               │                   │
┌──────────▼───────────────▼───────────────────▼──────────────┐
│                    PostgreSQL + pgvector                    │
│  ┌────────────┐ ┌──────────┐ ┌────────┐ ┌────────────────┐  │
│  │ utterances │ │embeddings│ │ people │ │ voting_records │  │
│  │  + chunks  │ │  (HNSW)  │ │        │ │                │  │
│  └────────────┘ └──────────┘ └────────┘ └────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Data flow:**
1. XML debates → Parse & extract utterances with context
2. Utterances → Summarize context (if long) → Format with metadata → Chunk with overlap
3. Chunks → Generate embeddings (sentence-transformers) → Store in pgvector
4. Query → Embed → HNSW similarity search → Filter → Return to agent → Synthesize response

## Repository structure

```
openchambers/
├── backend/
│   ├── src/
│   │   ├── api/              # FastAPI application and endpoints
│   │   ├── chatbot/          # LangGraph agent, tools, and prompts
│   │   └── data/
│   │       ├── database/     # SQLAlchemy models and repositories
│   │       ├── loaders/      # XML and metadata parsers
│   │       ├── pipelines/    # Batch ingestion orchestration
│   │       └── transformers/ # Summarization, formatting, chunking
│   ├── scripts/              # CLI scripts for data ingestion
│   └── data/                 # Raw data directory
│       └── hansard/
│           ├── debates/      # TheyWorkForYou XML files
│           └── metadata/     # People, votes, policies
├── frontend/                 # Next.js chat interface
└── README.md
```

## Quickstart

### Prerequisites

- Python 3.10+
- PostgreSQL 13+ with [pgvector](https://github.com/pgvector/pgvector) extension
- Node.js 18+ (for frontend)
- OpenAI API key

### 1. Set up the database

**Install pgvector extension** (if not already installed):

```bash
# Ubuntu/Debian (adjust version number to match your PostgreSQL)
sudo apt install postgresql-18-pgvector

# macOS with Homebrew
brew install pgvector
```

For other platforms, see the [pgvector installation guide](https://github.com/pgvector/pgvector#installation).

**Create the database and user:**

```bash
sudo -u postgres psql -c "CREATE USER hansard_user WITH PASSWORD 'your_password';"
sudo -u postgres psql -c "CREATE DATABASE hansard OWNER hansard_user;"
sudo -u postgres psql -d hansard -c "CREATE EXTENSION vector;"
```

The application will automatically create tables and indexes on first run.

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `backend/.env` with your database credentials and OpenAI key:
```
DATABASE_URL=postgresql+psycopg://hansard_user:your_password@localhost:5432/hansard
OPENAI_API_KEY=sk-...
EMBEDDING_MODEL_NAME=sentence-transformers/multi-qa-MiniLM-L6-cos-v1
AGENT_MODEL_NAME=gpt-4o-mini
```

### 3. Install dependencies

```bash
# Backend
cd backend
python3 -m venv venv
pip install -r requirements.txt
cd ..

# Frontend
cd frontend
npm install
cd ..
```

### 4. Download data

**Hansard debates** (example for January 2026):
```bash
rsync -az --progress --exclude '.svn' --exclude 'tmp/' \
  data.theyworkforyou.com::parldata/scrapedxml/debates/debates2026-01* \
  backend/data/hansard/debates/
```

**Metadata files:**
```bash
mkdir -p backend/data/hansard/metadata

curl -L -o backend/data/hansard/metadata/people.json \
  https://raw.githubusercontent.com/mysociety/parlparse/master/members/people.json

curl -L -o backend/data/hansard/metadata/divisions.parquet \
  https://votes.theyworkforyou.com/static/data/divisions.parquet

curl -L -o backend/data/hansard/metadata/votes.parquet \
  https://votes.theyworkforyou.com/static/data/votes.parquet

curl -L -o backend/data/hansard/metadata/policy_calc_to_load.parquet \
  https://votes.theyworkforyou.com/static/data/policy_calc_to_load.parquet

curl -L -o backend/data/hansard/metadata/policies.json \
  https://votes.theyworkforyou.com/policies.json
```

### 5. Ingest data

Edit the start date in *backend/scripts/add_debates_to_db.py* and run from the project root:

```bash
source backend/venv/bin/activate

# Process debates and generate embeddings (creates tables on first run)
python3 -m backend.scripts.add_debates_to_db

# Load metadata (people, votes, policies)
python3 -m backend.scripts.add_metadata_to_db
```

Checkpointing is in place which avoids duplicate records when re-running the scripts.

### 6. Run the application

```bash
# Terminal 1: Backend (from project root)
source backend/venv/bin/activate
uvicorn backend.src.api.app:app --reload

# Terminal 2: Frontend
cd frontend
npm run dev
```

Visit `http://localhost:3000` to start querying.

## Tech stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | Next.js 16, React 19, TypeScript, Tailwind CSS |
| **API** | FastAPI, Server-Sent Events, Pydantic |
| **Agent** | LangGraph, LangChain, OpenAI gpt-4o-mini |
| **Embeddings** | sentence-transformers (multi-qa-MiniLM-L6-cos-v1) |
| **Database** | PostgreSQL, pgvector (HNSW), SQLAlchemy |
| **Data processing** | Pandas, PyArrow, lxml |

## Data sources

This project uses open data from:

- **[TheyWorkForYou](https://www.theyworkforyou.com/)** — Hansard debate transcripts in XML format
- **[mySociety parlparse](https://github.com/mysociety/parlparse)** — MP biographical data
- **[TheyWorkForYou Votes](https://votes.theyworkforyou.com/)** — Voting records

All data is made available under open licenses by these organizations. This project is not affiliated with TheyWorkForYou or mySociety.

## License

MIT

## Contact

- GitHub: [@Jamie-Holding](https://github.com/Jamie-Holding)
- LinkedIn: [Jamie Holding](https://www.linkedin.com/in/jamie-holding/)
