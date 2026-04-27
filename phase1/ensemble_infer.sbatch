#!/bin/bash
#SBATCH --job-name=ensemble-infer
#SBATCH --account=cs_gy_9223-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --comment="gpu_mps=yes"
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Ensembles all four trained models: cyto2, cyto3, nuclei, multiscale.
# Run after all four training jobs + eval_best_checkpoint runs are done.

mkdir -p ~/cell-segmentation-assignment/logs

singularity exec --nv \
  --overlay ~/cell-segmentation-assignment/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh && cd ~/cell-segmentation-assignment && \
    python -u ensemble_infer.py \
      --exp-names cyto2 cyto3 nuclei multiscale \
      --spot-sigmas-map cyto2:8 cyto3:8 nuclei:8 multiscale:4,8,16" &
wait $!
