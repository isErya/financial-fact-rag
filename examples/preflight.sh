#!/usr/bin/env sh
# Pre-demo check, from the repo root: sh examples/preflight.sh
#
# Prints, in order: the images compose built, whether the index volume
# matches the corpus (the indexer prints "index up to date" and exits),
# whether the web port is free or already serving, /health with its
# llm_ready flag, and, when the API backend is configured, whether the key
# is accepted by the provider's models list (a GET; no completion is
# requested). Ends with GO or NO-GO.
set -u
cd "$(dirname "$0")/.."
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi
PORT="${WEB_PORT:-8804}"
BACKEND="${LLM_MODEL_BACKEND:-fake}"
ok=1

say() { printf '%s\n' "$*"; }
fail() { say "  NO: $*"; ok=0; }

say "images:"
docker image ls --filter "reference=*eliza*" --format '  {{.Repository}}:{{.Tag}}  {{.Size}}' | sed '/^$/d' || fail "docker not reachable"

say "index volume:"
if docker compose run --rm --no-deps indexer 2>/dev/null | grep -q "index up to date"; then
  say "  index up to date"
else
  fail "index missing or stale; run: docker compose up indexer"
fi

say "port $PORT:"
if curl -s --max-time 3 "localhost:$PORT/health" >/dev/null 2>&1; then
  say "  serving"
else
  say "  free (stack not up); run: docker compose up -d"
fi

say "health:"
health="$(curl -s --max-time 3 "localhost:$PORT/health" 2>/dev/null || true)"
if [ -n "$health" ]; then
  say "  $health"
  case "$health" in
    *'"llm_ready": true'*|*'"llm_ready":true'*) say "  llm_ready: true" ;;
    *) fail "llm_ready is false for backend $BACKEND" ;;
  esac
else
  fail "no /health answer on port $PORT"
fi

say "key liveness:"
if [ "$BACKEND" = "anthropic" ]; then
  if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    fail "ANTHROPIC_API_KEY is empty"
  else
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 https://api.anthropic.com/v1/models \
      -H "x-api-key: $ANTHROPIC_API_KEY" -H "anthropic-version: 2023-06-01")"
    if [ "$code" = "200" ]; then
      say "  models list answered 200"
    else
      fail "models list answered HTTP $code"
    fi
  fi
else
  say "  skipped (backend is $BACKEND; no key needed)"
fi

if [ "$ok" = 1 ]; then say "GO"; else say "NO-GO"; exit 1; fi
