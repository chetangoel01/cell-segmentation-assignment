#!/bin/bash
#SBATCH --job-name=infer-unet
#SBATCH --account=cs_gy_9223-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:15:00
#SBATCH --comment="gpu_mps=yes"
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Usage: sbatch infer_unet.sbatch [exp-name] [fg-thresh] [marker-thresh]
EXP=${1:-unet}
FG=${2:-0.5}
MARKER=${3:-0.7}

mkdir -p ~/cell-segmentation-assignment/logs

singularity exec --nv \
  --overlay ~/cell-segmentation-assignment/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh && cd ~/cell-segmentation-assignment && \
    python -u infer_unet.py --exp-name ${EXP} --fg-thresh ${FG} --marker-thresh ${MARKER}" &
wait $!
