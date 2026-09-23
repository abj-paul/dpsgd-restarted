#!/usr/bin/env bash
# Run the paired clipping-advantage collector under task-spooler.
set -uo pipefail

topic=$1
shift
status=0
.venv/bin/python -u counterfactual_advantage.py --ntfy-topic "$topic" "$@" || status=$?

if [ "$status" -eq 0 ]; then
    message='Counterfactual clipping-advantage collection completed successfully.'
    tags='white_check_mark,bar_chart'
else
    message="Counterfactual clipping-advantage collection stopped with exit status $status. Inspect the tsp output; the SQLite snapshot is resumable."
    tags='x,warning'
fi
curl -fsS --max-time 15 --retry 3 \
    -H 'Title: Counterfactual advantage status' -H "Tags: $tags" \
    --data-binary "$message" "https://ntfy.sh/$topic" >/dev/null || true
exit "$status"
