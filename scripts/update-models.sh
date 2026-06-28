#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# update-models.sh — keep the reel agents' local models current.
#
# What it does:
#   1. Ensures the Ollama daemon is reachable (starts it if not).
#   2. Checks the Ollama version against the minimum the models need, and prints
#      upgrade instructions if it's too old (Qwen3 needs a newer Ollama).
#   3. Pulls every model the agents depend on (from config/models.yaml).
#   4. Runs a tiny smoke test so a model update can't silently break the agents.
#   5. Appends a dated entry to scripts/model-updates.log.
#
# Usage:
#   scripts/update-models.sh            # pull preferred models + smoke test
#   scripts/update-models.sh --all      # also pull fallback models
#   scripts/update-models.sh --no-test  # skip the smoke test
#
# Cadence: wire to cron (see `make install-cron`) for weekly/monthly runs, or
# run on demand any time.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv/bin/python"
[ -x "$PY" ] || PY="python3"
LOG="${ROOT}/scripts/model-updates.log"
HOST="${OLLAMA_HOST:-http://localhost:11434}"

INCLUDE_FALLBACKS=0
RUN_TEST=1
GPU_FILTER=1
for arg in "$@"; do
  case "$arg" in
    --all) INCLUDE_FALLBACKS=1 ;;
    --no-test) RUN_TEST=0 ;;
    --no-gpu-filter) GPU_FILTER=0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') | $*" | tee -a "$LOG"; }

# 1. Ensure daemon is up.
if ! curl -fsS "${HOST}/api/version" >/dev/null 2>&1; then
  echo "Ollama not reachable; starting daemon…"
  nohup ollama serve >/tmp/ollama.log 2>&1 &
  for _ in $(seq 1 15); do
    curl -fsS "${HOST}/api/version" >/dev/null 2>&1 && break
    sleep 1
  done
fi
CUR_VER="$(curl -fsS "${HOST}/api/version" | sed -E 's/.*"version":"([^"]+)".*/\1/')"
MIN_VER="$("$PY" -m reel.manifest --min-ollama 2>/dev/null || echo 0.0.0)"
HW="$("$PY" -m reel.manifest --hardware 2>/dev/null || echo "unknown hardware")"
log "Hardware: $HW"

# 2. Version gate (does not abort — pulls of already-supported models still work).
lowest="$(printf '%s\n%s\n' "$CUR_VER" "$MIN_VER" | sort -V | head -1)"
if [ "$CUR_VER" != "$MIN_VER" ] && [ "$lowest" = "$CUR_VER" ]; then
  log "WARNING: Ollama $CUR_VER < required $MIN_VER. Newer models (e.g. Qwen3) may refuse to pull."
  cat <<'EOF'
  To upgrade Ollama, run this in YOUR terminal (needs your sudo password):
      curl -fsSL https://ollama.com/install.sh | sh
  Then re-run: scripts/update-models.sh
EOF
fi

# 3. Pull models.
MANIFEST_ARGS=()
[ "$INCLUDE_FALLBACKS" = 1 ] && MANIFEST_ARGS+=(--include-fallbacks)
[ "$GPU_FILTER"        = 1 ] && MANIFEST_ARGS+=(--runnable-only)
if [ "$GPU_FILTER" = 1 ]; then
  log "Ollama $CUR_VER — pulling models that fit hardware (--no-gpu-filter to pull all; fallbacks=$INCLUDE_FALLBACKS)"
else
  log "Ollama $CUR_VER — pulling all manifest models, no hardware filter (fallbacks=$INCLUDE_FALLBACKS)"
fi
mapfile -t MODELS < <("$PY" -m reel.manifest "${MANIFEST_ARGS[@]}")
for m in "${MODELS[@]}"; do
  echo "── pull $m"
  if ollama pull "$m"; then
    log "pulled OK: $m"
  else
    log "pull FAILED (skipped): $m"
  fi
done

# 4. Smoke test — confirms the agents still run end-to-end after updates.
# Headless run: the human-in-the-loop gates auto-approve on EOF (no TTY here),
# so the pipeline passes straight through without blocking.
if [ "$RUN_TEST" = 1 ]; then
  echo "── smoke test"
  if "$PY" -m reel.cli samples/sample_story.txt --out /tmp/reel_smoke --max-scenes 1 >/tmp/reel_smoke.log 2>&1; then
    log "smoke test PASSED"
  else
    log "smoke test FAILED — see /tmp/reel_smoke.log"
    tail -20 /tmp/reel_smoke.log || true
  fi
fi

log "current local models:"
ollama list | tee -a "$LOG"
log "update run complete"
