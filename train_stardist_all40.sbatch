#!/bin/bash
#SBATCH --job-name=stardist-all40
#SBATCH --account=cs_gy_9223-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --comment="gpu_mps=yes"
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --requeue
#SBATCH --signal=B:SIGUSR1@120

# StarDist trained on ALL 40 FOVs (no held-out val) on preemptable partition.
# 28 epochs was the sweet spot at 35 FOVs → Kaggle 0.7627. With 40 FOVs we aim
# for the same: ~30 epochs before we hit the diminishing-returns / overfit wall.
# If preempted, --resume continues on next submission.
EXP=${1:-stardist_all40}
EPOCHS=${2:-30}
shift 2 2>/dev/null
EXTRA="$@"

mkdir -p ~/cell-segmentation-assignment/logs

# Auto-resume if weights exist (so requeue picks up mid-chain)
RESUME_FLAG=""
if [ -f ~/cell-segmentation-assignment/models/${EXP}/weights_best.h5 ]; then
  RESUME_FLAG="--resume"
  echo "Found existing weights, resuming for ${EPOCHS} more epochs"
fi

singularity exec --nv \
  --overlay ~/cell-segmentation-assignment/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh && cd ~/cell-segmentation-assignment && \
    python -u train_stardist.py --channel dapi --all-fovs --exp-name ${EXP} --epochs ${EPOCHS} ${RESUME_FLAG} ${EXTRA}" &
wait $!
