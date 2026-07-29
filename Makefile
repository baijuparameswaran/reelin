# reel — common tasks. Run `make help` for the list.
PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help setup hardware-config setup-models setup-image run demo models update update-all install-cron test clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## Create venv, install deps, and pull the agents' Ollama models [SKIP_MODELS=1]
	python3 -m venv .venv
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -r requirements.txt
	@echo "python env ready — activate with: source .venv/bin/activate"
	@$(MAKE) --no-print-directory hardware-config
	@$(if $(SKIP_MODELS),echo "skipping model pull (SKIP_MODELS set)",$(MAKE) --no-print-directory setup-models)
	@echo "setup complete"

# Runs BEFORE the pull, so the models this host downloads are the ones it can
# actually run. A no-op on the machine config/models.yaml is tuned for.
hardware-config: ## Match each profile's model to this host's GPU/RAM (writes config/models.local.yaml)
	@$(PY) -m reel.hardware_config --apply

# Delegates to the same script `make update` uses (daemon start, Ollama version
# gate, hardware-fit filtering, per-model failure tolerance) rather than keeping
# a second, weaker pull loop here — just without the smoke test, since setup
# runs before there's anything to smoke.  Ollama itself needs sudo to install,
# so a missing binary prints the one-liner and skips instead of failing setup.
setup-models: ## Pull the Ollama models the agents need (from config/models.yaml)
	@if command -v ollama >/dev/null 2>&1; then \
	  scripts/update-models.sh --no-test; \
	  echo "models ready — run 'make models' to verify"; \
	else \
	  echo "ollama not installed — skipping model pull."; \
	  echo "  install: curl -fsSL https://ollama.com/install.sh | sh"; \
	  echo "  (the snap package can't reach the GPU — use the installer above)"; \
	  echo "  then: make setup-models"; \
	fi

setup-image: ## Install optional deps for casting image rendering (diffusers/torch)
	$(PIP) install -r requirements-image.txt
	@echo "image rendering deps installed (image.backend: diffusers)"

demo: test ## Run the bundled sample story (1 scene by default — all shots — per-agent profiles) [SCENES=1|all] [RESUME=1] [PROFILE=fast] [NORENDER=1] [TIMEOUT=N]
	$(PY) -m reel.cli samples/sample_story.txt --max-scenes $(or $(SCENES),1) $(if $(PROFILE),--profile $(PROFILE),) $(if $(RESUME),--resume,) $(if $(NORENDER),--no-render,) $(if $(TIMEOUT),--gate-timeout $(TIMEOUT),)

run: test ## Run on your own file:  make run SRC=path/to/story.txt [SCENES=1|all] [RESUME=1] [NORENDER=1] [TIMEOUT=N]
	$(PY) -m reel.cli $(SRC) --max-scenes $(or $(SCENES),1) $(if $(RESUME),--resume,) $(if $(NORENDER),--no-render,) $(if $(TIMEOUT),--gate-timeout $(TIMEOUT),)

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

test: ## Run the offline prompt-rule test suite (no LLM/API calls, <1s)
	$(PY) -m unittest discover -s tests -v

clean: ## Remove generated output
	rm -rf output /tmp/reel_smoke
