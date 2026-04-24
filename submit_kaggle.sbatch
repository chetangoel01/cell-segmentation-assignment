#!/bin/bash
#SBATCH --job-name=kaggle-submit
#SBATCH --account=cs_gy_9223-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --comment="gpu_mps=yes"
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Usage: sbatch submit_kaggle.sbatch [exp-name]
# If exp-name given, submits that specific model. Otherwise picks the best.
EXP=${1:-}
EXP_ARG=""
[ -n "$EXP" ] && EXP_ARG="--exp-name ${EXP}"

mkdir -p ~/cell-segmentation-assignment/logs

singularity exec --nv \
  --overlay ~/cell-segmentation-assignment/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh && cd ~/cell-segmentation-assignment && \
    python -u submit_best_to_kaggle.py ${EXP_ARG}" &
wait $!
