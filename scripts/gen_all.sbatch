#!/usr/bin/env bash
#SBATCH --job-name=clevrer_gen_all
#SBATCH --partition=gpu_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --output=gen_all_%j.log

source /net/holy-isilon/ifs/rc_labs/ydu_lab/dma/miniconda3/etc/profile.d/conda.sh
conda activate clevrergen

cd /net/holy-isilon/ifs/rc_labs/ydu_lab/dma/clevrer-gen
bash scripts/gen_all.sh 1000
