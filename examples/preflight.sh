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
# The fingerprint in the volume against one computed from the corpus now,
# read through the web image on CPU. Starting the indexer here would ask
# for a GPU, which a demo laptop without the container toolkit cannot
# give, and Docker then waits rather than failing.
project="$(basename "$(pwd)")"
check="$(docker run --rm -v "${project}_index_data:/index:ro" -v "$(pwd)/data:/app/data:ro" \
  --entrypoint python -e INDEX_DIR=/index "${project}-web" -c '
import json, os, sys
sys.path.insert(0, "/app/service"); os.chdir("/app")
import config, index
try:
    rec = json.load(open("/index/fingerprint.json"))
except FileNotFoundError:
    print("missing"); sys.exit(0)
now = index.fingerprint(config.CORPUS_ZIP, config.DENSE == "1", None)
dense = os.path.exists("/index/dense.npy")
print("match" if rec.get("sha256") == now and (dense or config.DENSE != "1") else "stale")
' 2>/dev/null || echo "unreadable")"
case "$check" in
  match) say "  index up to date (fingerprint matches the corpus and the dense setting)" ;;
  missing) fail "index missing; on a GPU host run: docker compose up indexer" ;;
  stale) fail "index stale for this corpus or dense setting; on a GPU host run: docker compose up indexer" ;;
  *) fail "could not read the index volume through the web image (is it built?)" ;;
esac

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
