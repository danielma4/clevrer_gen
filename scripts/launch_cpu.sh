#!/usr/bin/env bash
# Distribute generation across CPU cores on a non-GPU node (e.g. sapphire
# partition): N shards, each capped to a fixed thread count via
# render.threads, so N * THREADS_PER_SHARD <= total cores.
#
# Usage:
#   scripts/launch_cpu.sh --scenario configs/scenarios/two_balls_nocollide.yaml --num_videos 250
#   SHARDS=14 THREADS=8 scripts/launch_cpu.sh --scenario ... --num_videos 250
set -euo pipefail

PY=${PY:-/net/holy-isilon/ifs/rc_labs/ydu_lab/dma/miniconda3/envs/clevrergen/bin/python}
HERE="$(cd "$(dirname "$0")/.." && pwd)"

TOTAL_CORES=$(nproc --all)
THREADS=${THREADS:-8}
SHARDS=${SHARDS:-$((TOTAL_CORES / THREADS))}
[ "$SHARDS" -gt 0 ] || { echo "THREADS ($THREADS) > available cores ($TOTAL_CORES)"; exit 1; }
echo "Node has $TOTAL_CORES cores; launching $SHARDS shards x $THREADS threads = $((SHARDS * THREADS)) cores."

LOCKFILE="$HERE/.gen.lock"
exec 200>"$LOCKFILE"
if ! flock -n 200; then
  echo "ERROR: another gen run is already active (lock: $LOCKFILE)." >&2
  echo "  Check for existing processes: pgrep -af generate_dataset.py" >&2
  exit 1
fi

mkdir -p "$HERE/log"
pids=()

cleanup() {
  echo ""
  echo "Caught signal, killing all shards..."
  kill -9 "${pids[@]}" 2>/dev/null
  exit 1
}
trap cleanup INT TERM

for ((i=0; i<SHARDS; i++)); do
  "$PY" "$HERE/generate_dataset.py" --num_shards "$SHARDS" --shard_id "$i" "$@" \
    --set render.device=CPU render.threads="$THREADS" \
    > "$HERE/log/cpu_shard_${i}.log" 2>&1 &
  pids+=($!)
  echo "  shard $i -> $THREADS threads (log: log/cpu_shard_${i}.log)"
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "All CPU shards finished (fail=$fail)."
exit $fail
