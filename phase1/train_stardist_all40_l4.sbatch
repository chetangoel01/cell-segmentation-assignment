#!/bin/bash
#SBATCH --job-name=stardist-all40-l4
#SBATCH --account=cs_gy_9223-2026sp
#SBATCH --partition=g2-standard-12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --gres=gpu:l4:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --requeue
#SBATCH --signal=B:SIGUSR1@120

# L4 variant — A100 partition keeps getting scancel'd by a cluster cron.
# L4 is ~30-40% A100 speed, so budget ~2x walltime per epoch.
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
