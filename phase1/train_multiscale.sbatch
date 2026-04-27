#!/bin/bash
#SBATCH --job-name=train-multiscale
#SBATCH --account=cs_gy_9223-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --comment="gpu_mps=yes"
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --requeue
#SBATCH --signal=B:SIGUSR1@120

# Exp 006: cyto2 base, 5-channel input (polyt_max, dapi_max, density_sigma4,
#          density_sigma8, density_sigma16), 300 epochs.
# Multi-scale spot density gives the model spatial mRNA context at fine
# (sub-cellular), medium, and coarse (cell-body) granularities.

mkdir -p ~/cell-segmentation-assignment/logs

singularity exec --nv \
  --overlay ~/cell-segmentation-assignment/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh && cd ~/cell-segmentation-assignment && \
    python -u train.py --base-model cyto2 \
                       --exp-name multiscale \
                       --spot-sigmas 4,8,16" &
wait $!
