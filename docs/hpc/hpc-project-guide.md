# Running cell-segmentation-assignment on NYU HPC

This guide covers the end-to-end workflow for running this project (cell segmentation with
Cellpose, spot assignment, ARI evaluation) on the NYU Torch cluster.

---

## Quick-start decision

| Scenario | Recommended path |
|---|---|
| Interactive EDA / notebooks | **Cloud Bursting OOD** → Jupyter on an L4 GPU |
| SLURM batch segmentation run | Torch login node → `sbatch` |
| Data sync | `rsync` via `torch-dtn` |

Cloud Bursting OOD (the browser-based path) is the fastest way to start because there is no
queue wait and Jupyter is available directly. Use it for notebooks and one-off runs. Switch to
SLURM batch scripts only when you want fully unattended runs.

---

## 1. Local SSH setup (already done)

Your `~/.ssh/config` has the correct aliases:

```
ssh torch        # login node (SLURM submission, monitoring)
ssh torch-dtn    # data transfer node (rsync, scp)
```

**Note:** SSH keys are not supported on Torch. You will be prompted for your NYU password + Duo
MFA each time. If you hit a "REMOTE HOST IDENTIFICATION HAS CHANGED" error despite the config,
run: `ssh-keygen -R login.torch.hpc.nyu.edu`

---

## 2. HPC project layout (recommended)

```
/home/cg4652/
  cell-seg/               ← git-tracked code lives here (safe from purge)

/scratch/cg4652/
  cell-seg-data/          ← .dax files and competition data (large, fast I/O)
  cell-seg-overlay/       ← Singularity overlay with conda env
```

Keep code in `/home` (not purged). Keep data and the conda overlay in `/scratch` (5 TB, purged
after 60 days of inactivity — re-access files periodically or archive completed work).

---

## 3. Sync code to HPC

From your local machine:

```bash
rsync -avz \
  --exclude '__pycache__' \
  --exclude '.git' \
  --exclude 'data/' \
  --exclude '.venv' \
  /Users/chetangoel/Desktop/Repositories/cell-segmentation-assignment/ \
  cg4652@dtn.torch.hpc.nyu.edu:/home/cg4652/cell-seg/
```

Sync large data files separately to scratch (not home):

```bash
rsync -avz \
  /Users/chetangoel/Desktop/Repositories/cell-segmentation-assignment/data/ \
  cg4652@dtn.torch.hpc.nyu.edu:/scratch/cg4652/cell-seg-data/
```

---

## 4. Set up the Python environment on HPC

This project uses `cellpose`, `shapely`, `scikit-image`, and friends. The recommended way on
Torch is a Singularity overlay with a Miniforge conda env (avoids `/home` inode exhaustion from
Conda).

### One-time setup (run on a login node or interactive compute node)

```bash
# Create overlay directory in scratch
mkdir -p /scratch/cg4652/cell-seg-overlay
cd /scratch/cg4652/cell-seg-overlay

# Copy and decompress a 15 GB overlay image
cp /share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz .
gunzip overlay-15GB-500K.ext3.gz

# Launch the container in read/write mode
singularity exec --fakeroot \
  --overlay overlay-15GB-500K.ext3:rw \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash
```

Inside the Singularity shell:

```bash
# Install Miniforge into the overlay (not into /home)
wget --no-check-certificate \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p /ext3/miniforge3

# Create the env activation script
cat > /ext3/env.sh << 'EOF'
#!/bin/bash
unset -f which
source /ext3/miniforge3/etc/profile.d/conda.sh
export PATH=/ext3/miniforge3/bin:$PATH
export PYTHONPATH=/ext3/miniforge3/bin:$PATH
EOF

# Activate and install dependencies
source /ext3/env.sh
conda update -n base conda -y

# PyTorch (needed by cellpose)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Project dependencies
pip install \
  "cellpose>=2.4.2" \
  "shapely>=2.0" \
  "anndata>=0.10" \
  "scikit-image>=0.21" \
  "matplotlib>=3.7" \
  "numpy>=1.24" \
  "pandas>=2.0" \
  tifffile \
  "scikit-learn>=1.3"

exit  # leave Singularity
```

**Warning:** If your `~/.bashrc` has a `>>> conda initialize >>>` block, comment it out — it
will interfere with Singularity environments. Log out and back in after editing.

---

## 5. Option A: Interactive Jupyter via Cloud Bursting OOD (recommended for notebooks)

1. Connect to NYU VPN.
2. Go to: https://ood-burst-001.hpc.nyu.edu/
3. Log in with your NetID (`cg4652`), account: `cs_gy_9223-2026sp`.
4. Navigate to **Interactive Apps → Jupyter Notebook**.
5. Fill in resources:
   - **Machine type:** `g2-standard-12` (1 L4 GPU) — sufficient for Cellpose
   - **Time:** 2–4 hours
   - **Singularity image:** `/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif`
   - **Overlay:** `/scratch/cg4652/cell-seg-overlay/overlay-15GB-500K.ext3`
6. Launch and wait for the session to start (~1–2 min).
7. Open a terminal in JupyterLab and run:

```bash
source /ext3/env.sh
cd /home/cg4652/cell-seg
jupyter lab
```

Your notebooks (`01_eda.ipynb`, `02_baseline.ipynb`) will open with full GPU access and all
dependencies available.

---

## 6. Option B: SLURM batch job (for unattended runs)

Create a SLURM script at `/home/cg4652/cell-seg/hpc_run.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=cell-seg
#SBATCH --account=torch_pr_62_tandon_priority
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/home/cg4652/cell-seg/logs/slurm_%j.out
#SBATCH --error=/home/cg4652/cell-seg/logs/slurm_%j.err

mkdir -p /home/cg4652/cell-seg/logs

singularity exec --nv \
  --overlay /scratch/cg4652/cell-seg-overlay/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
    source /ext3/env.sh
    cd /home/cg4652/cell-seg
    export PYTHONPATH=/home/cg4652/cell-seg:\$PYTHONPATH
    python -m pytest tests/ -v --ignore=tests/test_io.py -m 'not integration'
  "
```

Submit it:

```bash
ssh torch
cd /home/cg4652/cell-seg
sbatch hpc_run.sbatch
squeue -u $USER   # monitor
```

Replace the `pytest` command with whatever entrypoint you want to run (notebook conversion via
`nbconvert`, a training script, etc.).

**Check your SLURM account name first:**

```bash
my_slurm_accounts
```

Use that value for `--account`.

---

## 7. Running tests on HPC

The test suite uses `pytest`. Integration tests that require competition data are marked with
`-m integration` and need the data mounted.

```bash
# Unit tests only (no data needed)
singularity exec --nv \
  --overlay /scratch/cg4652/cell-seg-overlay/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
    source /ext3/env.sh
    cd /home/cg4652/cell-seg
    export PYTHONPATH=.
    python -m pytest tests/ -v -m 'not integration'
  "

# Integration tests (needs data in /scratch/cg4652/cell-seg-data/)
singularity exec --nv \
  --overlay /scratch/cg4652/cell-seg-overlay/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
    source /ext3/env.sh
    cd /home/cg4652/cell-seg
    export PYTHONPATH=.
    DATA_DIR=/scratch/cg4652/cell-seg-data python -m pytest tests/ -v -m integration
  "
```

---

## 8. Key paths summary

| Local | HPC |
|---|---|
| `~/Desktop/Repositories/cell-segmentation-assignment/` | `/home/cg4652/cell-seg/` |
| `data/` (`.dax` files, competition) | `/scratch/cg4652/cell-seg-data/` |
| `.venv/` (local Python env) | `/scratch/cg4652/cell-seg-overlay/overlay-15GB-500K.ext3` |

---

## 9. Monitoring and troubleshooting

```bash
squeue -u $USER                          # running/queued jobs
sacct -u $USER --starttime=today         # job history
scancel JOBID                            # cancel a job
myquota                                  # check disk/inode usage
```

| Problem | Fix |
|---|---|
| `REMOTE HOST IDENTIFICATION HAS CHANGED` | `ssh-keygen -R login.torch.hpc.nyu.edu` |
| `ModuleNotFoundError` in job | Check `source /ext3/env.sh` is in the sbatch script |
| Job finishes instantly | `cat logs/slurm_<jobid>.err` to see the error |
| `QOSGrpGRES` stuck in queue | GPU quota hit; wait or add `--partition=h200_tandon` |
| Inode quota full on `/home` | Don't install conda envs in `/home`; use the overlay in `/scratch` |
| Low GPU utilization warning | Cellpose uses GPU by default; verify `torch.cuda.is_available()` returns True |
