#!/usr/bin/env bash
#SBATCH --job-name=clevrer_gen_all_500
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=log/gen_all_%j.log

source /net/holy-isilon/ifs/rc_labs/ydu_lab/dma/miniconda3/etc/profile.d/conda.sh
conda activate clevrergen

cd /net/holy-isilon/ifs/rc_labs/ydu_lab/dma/clevrer-gen
SHARDS=14 THREADS=8 bash scripts/gen_all_cpu.sh 5000
