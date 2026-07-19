#!/usr/bin/env bash
# SLURM array job: one task per shard, one GPU per task. SLURM sets
# CUDA_VISIBLE_DEVICES per task, so no manual MIG pinning is needed.
# The array size IS the shard count, so keep --array and your intended
# parallelism in sync (here: 8 shards).
#
# Usage (args after the script pass through to generate_dataset.py):
#   sbatch scripts/launch_slurm.sbatch --scenario configs/scenarios/two_collide.yaml --num_videos 800
#   (-> output/two_collide/ ; use --name to choose the run dir)
#
#SBATCH --job-name=clevrergen
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=clevrergen_%A_%a.log
# NOTE: set --partition / --account for your cluster as needed.
set -euo pipefail

PY=${PY:-/net/holy-isilon/ifs/rc_labs/ydu_lab/dma/miniconda3/envs/clevrergen/bin/python}
HERE="$(cd "$(dirname "$0")/.." && pwd)"

"$PY" "$HERE/generate_dataset.py" \
    --num_shards "$SLURM_ARRAY_TASK_COUNT" \
    --shard_id "$SLURM_ARRAY_TASK_ID" "$@"
