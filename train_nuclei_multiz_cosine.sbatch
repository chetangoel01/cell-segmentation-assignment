#!/bin/bash
#SBATCH --job-name=nuclei-mz-cos
#SBATCH --account=cs_gy_9223-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --comment="gpu_mps=yes"
#SBATCH --requeue
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

mkdir -p ~/cell-segmentation-assignment/logs

singularity exec --nv \
  --overlay ~/cell-segmentation-assignment/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh && cd ~/cell-segmentation-assignment && \
    python -u train.py \
      --base-model nuclei \
      --exp-name nuclei_multiz_cosine \
      --augment \
      --multi-z \
      --lr-schedule cosine \
      --lr-min 1e-7" &
wait $!
