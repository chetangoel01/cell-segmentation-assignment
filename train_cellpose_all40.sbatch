#!/bin/bash
#SBATCH --job-name=cellpose-all40
#SBATCH --account=cs_gy_9223-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --comment="gpu_mps=yes"
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --requeue
#SBATCH --signal=B:SIGUSR1@120

# Cellpose trained on ALL 40 FOVs (no held-out val) on preemptable partition.
# Config = nuclei + cosine schedule, the best Cellpose variant on Kaggle so far (0.7588).
# 300 epochs total, chunked checkpoints every 5 epochs → preempt-safe.
EXP=${1:-nuclei_cosine_all40}
EPOCHS=${2:-300}
shift 2 2>/dev/null
EXTRA="$@"

mkdir -p ~/cell-segmentation-assignment/logs

singularity exec --nv \
  --overlay ~/cell-segmentation-assignment/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh && cd ~/cell-segmentation-assignment && \
    python -u train.py --base-model nuclei --lr-schedule cosine --all-fovs \
      --exp-name ${EXP} --epochs ${EPOCHS} ${EXTRA}" &
wait $!
