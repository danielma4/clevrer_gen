#!/usr/bin/env bash
# Distribute generation across N processes on ONE full (non-MIG) GPU.
#
# launch_local.sh needs MIG slices and exits without them; a full B200 renders
# 480x320 at 16 samples using ~1.6 GB and a fraction of the SMs, so the real
# limit is CPU-side (bpy scene updates, compositing, x264 mux). N defaults to the
# core count, and each shard is pinned to one render thread so N shards do not
# oversubscribe N cores.
#
# Writes to $OUTBASE/<name>, defaulting to the dir the vwm dataset configs read
# (`_clevrer_root` in configurations/dataset/clevrer_ood_base.yaml).
#
# Usage:
#   scripts/launch_gpu_shards.sh --scenario configs/scenarios/tiny_nocollide.yaml \
#       --num_videos 300 --name tiny_nocollide
#   SHARDS=4 OUTBASE=/tmp/foo scripts/launch_gpu_shards.sh ...
set -euo pipefail

PY=${PY:-/work/hu/alpha_hu/akiruga_hu/sw/conda/envs/clevrergen/bin/python}
HERE="$(cd "$(dirname "$0")/.." && pwd)"
SHARDS=${SHARDS:-$(nproc)}
OUTBASE=${OUTBASE:-/scratch/akiruga_hu/datasets/clevrer_gen_new}

LOCKFILE="$HERE/.gen.lock"
exec 200>"$LOCKFILE"
if ! flock -n 200; then
  echo "ERROR: another gen run is already active (lock: $LOCKFILE)." >&2
  echo "  Check for existing processes: pgrep -af generate_dataset.py" >&2
  exit 1
fi

echo "Launching $SHARDS shards on one GPU -> $OUTBASE"
mkdir -p "$HERE/log"
pids=()

cleanup() {
  echo ""
  echo "Caught signal, killing all shards..."
  kill -9 "${pids[@]}" 2>/dev/null
  exit 1
}
trap cleanup INT TERM

for ((i = 0; i < SHARDS; i++)); do
  PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 \
    "$PY" -s "$HERE/generate_dataset.py" \
      --num_shards "$SHARDS" --shard_id "$i" --output_base "$OUTBASE" \
      --set render.threads=1 "$@" \
    > "$HERE/log/shard_${i}.log" 2>&1 &
  pids+=($!)
  echo "  shard $i (log: log/shard_${i}.log)"
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "All shards finished (fail=$fail)."
exit $fail
