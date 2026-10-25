.PHONY: test eval frontend-install frontend-test frontend-build

# Backend unit tests (no live LLM calls)
test:
	python -m pytest tests/ -v

# Agent eval suite: calls the real Gemini API and Qdrant, needs .env
eval:
	python -m eval.run_eval

frontend-install:
	cd frontend && npm ci

frontend-test:
	cd frontend && npm test

frontend-build:
	cd frontend && npm run build
