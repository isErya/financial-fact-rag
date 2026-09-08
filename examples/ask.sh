#!/usr/bin/env sh
# One full request against the running stack, from the repo root:
#   sh examples/ask.sh
# Set dry_run to true in examples/request.json to stop after retrieval.
set -eu
cd "$(dirname "$0")/.."
curl -s "localhost:${WEB_PORT:-8804}/ask" \
  -H 'content-type: application/json' \
  -d @examples/request.json | python3 -m json.tool
