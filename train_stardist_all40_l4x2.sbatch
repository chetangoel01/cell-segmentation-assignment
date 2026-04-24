#!/bin/bash
#SBATCH --job-name=stardist-all40-l4x2
#SBATCH --account=cs_gy_9223-2026sp
#SBATCH --partition=g2-standard-24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=90G
#SBATCH --gres=gpu:l4:2
#SBATCH --time=08:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --requeue
#SBATCH --signal=B:SIGUSR1@120

# g2-standard-24 — empty queue, no kill cron. EXCLUSIVE node so we get
# whole 2× L4 box; train_stardist.py only uses 1 GPU. ~2× A100 wallclock.
EXP=${1:-stardist_all40}
EPOCHS=${2:-30}
shift 2 2>/dev/null
EXTRA="$@"

mkdir -p ~/cell-segmentation-assignment/logs

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
