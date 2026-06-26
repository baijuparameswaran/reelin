# reel — common tasks. Run `make help` for the list.
PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help setup setup-image run demo models update update-all install-cron clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Create venv and install dependencies
	python3 -m venv .venv
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -r requirements.txt
	@echo "setup complete"

setup-models: ## Pull all preferred Ollama models (qwen3:4b, qwen3:8b, gemma3:12b)
	@echo "Pulling preferred models from config/models.yaml …"
	@$(PY) -m reel.manifest | while read m; do echo "  pull $$m"; ollama pull "$$m"; done
	@echo "done — run 'make models' to verify"

setup-image: ## Install optional deps for casting image rendering (diffusers/torch)
	$(PIP) install -r requirements-image.txt
	@echo "image rendering deps installed (image.backend: diffusers)"

demo: ## Run the bundled sample story (1 scene — all shots — per-agent profiles) [RESUME=1] [PROFILE=fast]
	$(PY) -m reel.cli samples/sample_story.txt --max-scenes 1 $(if $(PROFILE),--profile $(PROFILE),) $(if $(RESUME),--resume,)

run: ## Run on your own file:  make run SRC=path/to/story.txt [SCENES=1] [RESUME=1]
	$(PY) -m reel.cli $(SRC) --max-scenes $(or $(SCENES),1) $(if $(RESUME),--resume,)

models: ## Show local model / profile status
	$(PY) -m reel.cli --list-models

update: ## Pull/refresh the agents' models + smoke test (the cadence job)
	scripts/update-models.sh

update-all: ## Same as update, but also pull fallback models
	scripts/update-models.sh --all

install-cron: ## Install weekly + monthly model-update cron jobs
	@bash scripts/install-cron.sh

secrets: ## Manage encrypted API keys (set / get / delete / status)
	$(PY) -m reel.secrets $(or $(CMD),status)

clean: ## Remove generated output
	rm -rf output /tmp/reel_smoke
