#!/usr/bin/env bash
# Run the sequential collector under task-spooler and report its exit status.
set -uo pipefail

topic=$1
shift
status=0
.venv/bin/python -u collect_ten_runs.py --ntfy-topic "$topic" "$@" || status=$?

if [ "$status" -eq 0 ]; then
    message='Ten-run WRN gradient collection completed successfully.'
else
    message="Ten-run WRN gradient collection stopped with exit status $status. Inspect the tsp job output and stored shards."
fi
curl -fsS --max-time 10 --retry 3 -H 'Title: WRN collection status' \
    --data-binary "$message" "https://ntfy.sh/$topic" >/dev/null || true
exit "$status"
