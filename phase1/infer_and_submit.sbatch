#!/bin/bash
#SBATCH --job-name=infer
#SBATCH --account=cs_gy_9223-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --comment="gpu_mps=yes"
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Usage: sbatch infer_and_submit.sbatch <exp-name> [spot-sigmas]
# Runs inference with best thresholds from best_params_<exp>.json.
# Does NOT auto-submit to Kaggle — check val ARI first, submit manually.
EXP=${1:-cyto2}
SIGMAS=${2:-8}

mkdir -p ~/cell-segmentation-assignment/logs

PARAMS_JSON=""
if [ -f ~/cell-segmentation-assignment/best_params_${EXP}.json ]; then
    PARAMS_JSON="--params-json best_params_${EXP}.json"
fi

singularity exec --nv \
  --overlay ~/cell-segmentation-assignment/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh && cd ~/cell-segmentation-assignment && \
    python -u infer.py --exp-name ${EXP} --spot-sigmas ${SIGMAS} ${PARAMS_JSON}" &
wait $!
