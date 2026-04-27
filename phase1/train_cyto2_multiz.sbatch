#!/bin/bash
#SBATCH --job-name=train-cyto2-multiz
#SBATCH --account=cs_gy_9223-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --comment="gpu_mps=yes"
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --requeue
#SBATCH --signal=B:SIGUSR1@120

# --multi-z: 6 training samples per FOV (max-proj w/ union-z GT + 5 per-z-plane)
# = 210 total samples vs 35 previously.  Also implies --use-union-z for the
# max-projection sample so the GT footprint matches what the model sees.
# Time estimate: ~3x longer than single-sample training (210 vs 35 samples).

mkdir -p ~/cell-segmentation-assignment/logs

singularity exec --nv \
  --overlay ~/cell-segmentation-assignment/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh && cd ~/cell-segmentation-assignment && \
    python -u train.py --base-model cyto2 --exp-name cyto2_multiz --multi-z" &
wait $!
