#!/bin/bash
#SBATCH --job-name=stitch-thresh
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

# Usage: sbatch infer_stitch_threshold.sbatch <exp-name> <stitch-threshold>
EXP=${1:-nuclei_cosine}
ST=${2:-0.1}
SIGMAS=${3:-8}

PARAMS_JSON=""
if [ -f ~/cell-segmentation-assignment/best_params_${EXP}.json ]; then
    PARAMS_JSON="--params-json best_params_${EXP}.json"
fi

mkdir -p ~/cell-segmentation-assignment/logs

singularity exec --nv \
  --overlay ~/cell-segmentation-assignment/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh && cd ~/cell-segmentation-assignment && \
    python -u infer.py \
      --exp-name ${EXP} \
      --spot-sigmas ${SIGMAS} \
      --stitch-threshold ${ST} \
      ${PARAMS_JSON}" &
wait $!
