# Weather Agent MLOps

## Stack
- LangGraph agent (state.py, nodes.py, graph.py under src/agent/)
- Gemini via Vertex AI for the LLM calls
- Qdrant for RAG (historical air quality data)
- OpenWeatherMap API for live weather (src/tools/weather_api.py)
- Deploying to GCP Cloud Run (agent = Service, ingestion = Job)

## Conventions
- Guardrails (input/output checks) live in src/api/guardrails.py, NOT inside the LangGraph nodes
- Secrets loaded via src/common/config.py, never hardcoded or committed
- Follow the folder structure in README.md when adding new files

## Current focus
- Migrating orchestration from plain LangChain to LangGraph
- Adding production MLOps layer (Docker, Cloud Run, Secret Manager, CI/CD)