#!/bin/bash
#SBATCH --job-name=kaggle-submit
#SBATCH --account=cs_gy_9223-2026sp
#SBATCH --partition=n2c48m24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# CPU-only Kaggle submission — no GPU needed for CSV upload.
# Usage: sbatch submit_kaggle_cpu.sbatch [exp-name]
EXP=${1:-}
EXP_ARG=""
[ -n "$EXP" ] && EXP_ARG="--exp-name ${EXP}"

mkdir -p ~/cell-segmentation-assignment/logs

singularity exec \
  --overlay ~/cell-segmentation-assignment/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh && cd ~/cell-segmentation-assignment && \
    python -u submit_best_to_kaggle.py ${EXP_ARG}" &
wait $!
