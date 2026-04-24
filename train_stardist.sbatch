#!/bin/bash
#SBATCH --job-name=train-stardist
#SBATCH --account=cs_gy_9223-2026sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --gres=gpu:a100:2
#SBATCH --time=08:00:00
#SBATCH --comment="gpu_mps=yes"
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --requeue
#SBATCH --signal=B:SIGUSR1@120

# StarDist 2D_versatile_fluo fine-tuned on MERFISH DAPI max-projection
# Polygon-based segmentation — fundamentally different from Cellpose flow fields
# Usage: sbatch train_stardist.sbatch [exp-name] [epochs] [extra-flags...]
#   sbatch train_stardist.sbatch stardist 200             # fresh train, 200 epochs
#   sbatch train_stardist.sbatch stardist 100 --resume    # resume + 100 more epochs
EXP=${1:-stardist}
EPOCHS=${2:-200}
shift 2 2>/dev/null
EXTRA="$@"

mkdir -p ~/cell-segmentation-assignment/logs

singularity exec --nv \
  --overlay ~/cell-segmentation-assignment/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh && cd ~/cell-segmentation-assignment && \
    python -u train_stardist.py --channel dapi --exp-name ${EXP} --epochs ${EPOCHS} ${EXTRA}" &
wait $!
