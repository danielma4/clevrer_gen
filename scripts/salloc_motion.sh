#!/usr/bin/env bash
# Interactive allocation for the motion-label pipeline (CPU only, no GPU needed:
# nothing here renders or re-simulates). Paste the salloc, then run the steps.
#
#   bash scripts/salloc_motion.sh        # prints the commands
#   eval "$(bash scripts/salloc_motion.sh --salloc)"   # or just copy it

SALLOC='salloc --partition=test --nodes=1 --ntasks=1 --cpus-per-task=112 \
  --mem=64G --time=04:00:00 --job-name=clevrer_motion'

if [[ "${1:-}" == "--salloc" ]]; then
  echo "$SALLOC"
  exit 0
fi

cat <<EOF
# 1) allocate
$SALLOC

# 2) inside the allocation
source /net/holy-isilon/ifs/rc_labs/ydu_lab/dma/miniconda3/etc/profile.d/conda.sh
conda activate clevrergen
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/dma/clevrer-gen

# 3) backfill camera + numeric geometry into the annotations (~minutes, I/O bound)
python scripts/backfill_annotations.py output/*_nocollide

# 4) build the motion tensors (~8 CPU-hours of work, ~10 min on 112 workers)
python scripts/build_motion.py output/*_nocollide --workers 100

# The four OOD subsets deliberately have no splits: they are eval-only.

# 5) validate
python scripts/validate_motion.py output/*_nocollide --clips 5
EOF
