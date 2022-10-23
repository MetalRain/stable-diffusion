#!/bin/bash -e
# Start worker prosess for processing all requests
#
# Args: waits
# Example:
# ./scripts/start-worker.sh "1,3,5"
BASE_DIR="$(dirname "$(dirname "$(realpath $0)")")"
source "$BASE_DIR/scripts/config.sh"

python "$BASE_DIR/worker.py" \
    --waits "$WORKER_WAITS"