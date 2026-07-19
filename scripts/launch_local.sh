#!/usr/bin/env bash
# Distribute generation across every visible MIG slice on this node: one process
# per slice, each pinned via CUDA_VISIBLE_DEVICES. The total video count
# (output.num_videos) is split round-robin across the slices.
#
# Usage (args after the script are passed through to generate_dataset.py):
#   scripts/launch_local.sh --scenario configs/scenarios/two_collide.yaml --num_videos 800
#   (-> output/two_collide/ ; use --name to choose the run dir)
set -euo pipefail

PY=${PY:-/net/holy-isilon/ifs/rc_labs/ydu_lab/dma/miniconda3/envs/clevrergen/bin/python}
HERE="$(cd "$(dirname "$0")/.." && pwd)"

LOCKFILE="$HERE/.gen.lock"
exec 200>"$LOCKFILE"
if ! flock -n 200; then
  echo "ERROR: another gen run is already active (lock: $LOCKFILE)." >&2
  echo "  Check for existing processes: pgrep -af generate_dataset.py" >&2
  exit 1
fi

mapfile -t MIGS < <(nvidia-smi -L | grep -oP '(?<=UUID: )MIG-[^)]+')
N=${#MIGS[@]}
[ "$N" -gt 0 ] || { echo "No MIG devices found (nvidia-smi -L)"; exit 1; }
echo "Found $N MIG slices; launching $N shards."

mkdir -p "$HERE/log"
pids=()

cleanup() {
  echo ""
  echo "Caught signal, killing all shards..."
  kill -9 "${pids[@]}" 2>/dev/null
  exit 1
}
trap cleanup INT TERM

for i in "${!MIGS[@]}"; do
  CUDA_VISIBLE_DEVICES="${MIGS[$i]}" \
    "$PY" "$HERE/generate_dataset.py" --num_shards "$N" --shard_id "$i" "$@" \
    > "$HERE/log/shard_${i}.log" 2>&1 &
  pids+=($!)
  echo "  shard $i -> ${MIGS[$i]} (log: log/shard_${i}.log)"
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "All shards finished (fail=$fail)."
exit $fail
