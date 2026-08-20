# Weather Agent MLOps

A production-oriented LangGraph agent that answers weather and air-quality questions,
combining live weather data with a RAG pipeline over historical air-quality records.
Built for deployment on GCP Cloud Run with a full CI/CD and evaluation layer.

## Architecture

The system is split into two independently deployable units:

- **Ingestion job** (Cloud Run Job, scheduled) — fetches raw data and indexes it into Qdrant.
- **Agent service** (Cloud Run Service, always-on) — serves the LangGraph agent over an HTTP API.

Both share common config/secret-loading and tool code.

```
Client → server.py (FastAPI) → guardrails → LangGraph agent → tools (weather API, vector search) → response
                                                                 ↑
                                              Qdrant ← index_qdrant.py ← fetch_raw_data.py (ingestion job)
```

## Stack

- **Orchestration:** LangGraph (`src/agent/`)
- **LLM:** Gemini via Vertex AI
- **Vector store / RAG:** Qdrant, over historical air-quality data
- **Live data:** OpenWeatherMap API (`src/tools/weather_api.py`)
- **Deployment:** GCP Cloud Run — agent as a Service, ingestion as a Job
- **CI/CD:** GitHub Actions

## Project structure

```
├── .github/workflows/       # CI/CD: ingestion job, agent service, PR checks
├── config/                  # Per-environment config (dev/prod)
├── src/
│   ├── common/               # Shared config + secret loading
│   ├── data/                 # Ingestion: fetch raw data, index into Qdrant
│   ├── tools/                 # Agent tools: weather API, vector search
│   ├── agent/                 # LangGraph state, nodes, graph
│   └── api/                   # HTTP layer, guardrails
├── eval/                     # Golden queries, eval runner, LLM-as-judge
├── tests/                    # Unit tests (no live LLM calls)
├── deploy/                   # Dockerfiles + Cloud Build config
├── main.py                   # Local CLI entrypoint
├── server.py                 # FastAPI production entrypoint
└── Makefile                  # make test / make eval / make deploy-agent
```

## Conventions

- Guardrails (input/output checks) live in `src/api/guardrails.py`, **not** inside LangGraph nodes.
- Secrets are loaded via `src/common/config.py` — never hardcoded or committed.
- New files should follow the folder structure documented here.

## Getting started

```bash
# install dependencies
pip install -r requirements.txt -r requirements-dev.txt

# run the agent locally via CLI
python main.py

# run the API server locally
python server.py

# run unit tests
make test

# run the agent eval suite
make eval
```

## Deployment

- `deploy-ingestion-job.yaml` builds and deploys the ingestion pipeline as a Cloud Run Job on changes under `src/data/`.
- `deploy-agent-service.yaml` builds and deploys the agent API as a Cloud Run Service on changes under `src/agent/`, `src/api/`, `src/tools/`.
- `ci-test.yaml` runs lint, unit tests, and eval on every pull request.

## Current focus

- Migrating orchestration from plain LangChain to LangGraph.
- Adding the production MLOps layer: Docker, Cloud Run, Secret Manager, CI/CD.
