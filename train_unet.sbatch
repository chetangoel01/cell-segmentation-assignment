#!/bin/bash
#SBATCH --job-name=train-unet
#SBATCH --account=cs_gy_9223-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --comment="gpu_mps=yes"
#SBATCH --requeue
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Usage: sbatch train_unet.sbatch [exp-name] [epochs]
EXP=${1:-unet}
EPOCHS=${2:-80}

mkdir -p ~/cell-segmentation-assignment/logs

singularity exec --nv \
  --overlay ~/cell-segmentation-assignment/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh && cd ~/cell-segmentation-assignment && \
    python -u train_unet.py --exp-name ${EXP} --epochs ${EPOCHS}" &
wait $!
