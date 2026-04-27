#!/bin/bash
#SBATCH --job-name=best-ckpt
#SBATCH --account=cs_gy_9223-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:30:00
#SBATCH --comment="gpu_mps=yes"
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Usage: sbatch eval_best_checkpoint.sbatch <exp-name> [spot-sigmas] [--zstats]
# e.g.:  sbatch eval_best_checkpoint.sbatch cyto2
#        sbatch eval_best_checkpoint.sbatch multiscale 4,8,16
#        sbatch eval_best_checkpoint.sbatch cyto2_zstats 8 --zstats
EXP=${1:-cyto2}
SIGMAS=${2:-8}
EXTRA=${3:-}

mkdir -p ~/cell-segmentation-assignment/logs

singularity exec --nv \
  --overlay ~/cell-segmentation-assignment/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh && cd ~/cell-segmentation-assignment && \
    python -u eval_best_checkpoint.py --exp-name ${EXP} --spot-sigmas ${SIGMAS} ${EXTRA}" &
wait $!
