#!/bin/bash
#SBATCH --job-name=cellpose-finetune
#SBATCH --account=cs_gy_9223-2026sp
#SBATCH --partition=g2-standard-12
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --requeue

mkdir -p ~/cell-segmentation-assignment/logs

singularity exec --nv \
  --overlay ~/cell-segmentation-assignment/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh && cd ~/cell-segmentation-assignment && python train.py"
