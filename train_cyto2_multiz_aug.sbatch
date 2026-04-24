#!/bin/bash
#SBATCH --job-name=train-cyto2-mza
#SBATCH --account=cs_gy_9223-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --comment="gpu_mps=yes"
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --requeue
#SBATCH --signal=B:SIGUSR1@120

# Combined: multi-z (210 samples) + 8x augmentation = up to 1680 effective samples.
# This is the maximum-data training run.  Expect significantly better generalisation
# and a smaller val->test gap.  Longer wall time (24h) to account for the data volume.

mkdir -p ~/cell-segmentation-assignment/logs

singularity exec --nv \
  --overlay ~/cell-segmentation-assignment/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh && cd ~/cell-segmentation-assignment && \
    python -u train.py --base-model cyto2 --exp-name cyto2_multiz_aug \
      --multi-z --augment" &
wait $!
