# The Complete Guide to NYU HPC (Torch Cluster)

Everything you need to know to go from zero to productively running jobs on NYU's Torch HPC cluster — accounts, storage, SLURM, containers, ML workflows, and best practices.

---

## 1. What is Torch?

Torch is NYU's primary HPC cluster. It has 518 Intel Xeon Platinum 8592+ (64-core) CPUs, 29 NVIDIA H200 GPUs, and 68 NVIDIA L40S GPUs, all connected via Infiniband NDR400 interconnect. It was benchmarked at 10.79 PF/s on LINPACK (ranked #133 on Top500) and ranked #40 on the Green500 list for energy efficiency, thanks to its liquid cooling system.

Many GPUs on Torch are owned by stakeholder groups (Tandon, Courant, CDS, CILVR, etc.) who get priority access. Non-stakeholders can still use these resources via preemptible jobs.

---

## 2. Getting an Account

### Who is eligible

NYU HPC resources are available to full-time NYU faculty and to all NYU staff and students **with sponsorship from a full-time NYU faculty member**. NYU Med School faculty, staff, and students also need a faculty sponsor from an eligible department.

### Requesting an account

1. Connect to NYU VPN.
2. Go to [NYU Identity Management](https://identity.it.nyu.edu/) and log in with your NetID.
3. Navigate to **Manage Access → Request HPC Account**.
4. Fill out the form — you'll need to specify your faculty sponsor.
5. Your sponsor will receive an email to approve/deny the request.
6. Once approved, the account is typically created within a day.

Students, alumni, and external collaborators all need a faculty sponsor. External collaborators must first obtain an NYU affiliate NetID via the [Affiliate Management Form](https://start.nyu.edu/).

### Renewing your account

All non-faculty accounts expire after 12 months. To renew:

1. Connect to NYU VPN.
2. Go to [NYU Identity Management](https://identity.it.nyu.edu/).
3. Navigate to **Update/Renew HPC Account**.
4. Fill out the renewal form and submit.
5. Your sponsor approves — account is renewed, usually within a day.

If your renewal stalls, check with your sponsor first, then email hpc@nyu.edu.

### The HPC Projects Portal (getting a SLURM account)

Having an HPC account gives you access to log in to Torch, but to actually **run jobs**, you need an active allocation within the [HPC Project Management Portal](https://projects.hpc.nyu.edu) (accessible on VPN).

The workflow is:

1. **PI creates a project** — title, description, associated school.
2. **PI requests an allocation** — choose the resource (e.g., `torch` for general access, or school-specific stakeholder resources like `tandon_advanced`, `courant`, `cds`, etc.).
3. **School approver approves** the allocation → status changes to "Active."
4. The allocation has a **SLURM account name** (visible in the portal) that you'll pass via `--account` in all job submissions.

You can check which SLURM accounts you have access to by running:

```
my_slurm_accounts
```

### Stakeholder resources by school

- **Tandon**: `tandon_advanced` (A100, H100, H200), `tandon_priority` (A100, H100)
- **Courant**: `courant` (L40S, H200), `lpinto` (L40S), `bpeher` (H200), `cilvr` (A100)
- **CDS**: `cds` (RTX8000, A100, H200), `mren` (L40S)
- **Arts & Science**: `chemistry` (A100)

---

## 3. Connecting to Torch

### SSH Configuration

Add this to your `~/.ssh/config` (Mac/Linux) or create it via PowerShell on Windows:

```
Host dtn.torch.hpc.nyu.edu
  User <NetID>
  StrictHostKeyChecking no
  ServerAliveInterval 60
  ForwardAgent yes
  UserKnownHostsFile /dev/null
  LogLevel ERROR

Host torch login.torch.hpc.nyu.edu
  Hostname login.torch.hpc.nyu.edu
  User <NetID>
  StrictHostKeyChecking no
  ServerAliveInterval 60
  ForwardAgent yes
  UserKnownHostsFile /dev/null
  LogLevel ERROR
```

Then connect:

```
ssh torch
```

**Note:** SSH keys are not supported on Torch due to security restrictions. You'll authenticate via NYU's device login (Microsoft MFA) flow.

The cluster has multiple login nodes (`log-1`, `log-2`, `log-3`) — the `StrictHostKeyChecking no` directive prevents the "REMOTE HOST IDENTIFICATION HAS CHANGED" warning that arises from this.

### Windows

Use PowerShell to create `$HOME\.ssh\config` with the same contents. Alternatively, use MobaXterm.

### Open OnDemand (Web GUI)

If you prefer a browser-based interface, go to [https://ood.torch.hpc.nyu.edu](https://ood.torch.hpc.nyu.edu) (requires VPN). OOD provides:

- File management (upload/download)
- Shell access without a local SSH client
- Job management and monitoring
- Interactive apps: Jupyter, RStudio, MATLAB, full Linux desktop — no port forwarding needed

If OOD won't load, try a private browser window (it's often a cache issue). Clear the site data for `ood.torch.hpc.nyu.edu` in your browser settings.

---

## 4. Storage and Data Management

**Critical:** High-risk data (PII, ePHI, CUI) must NOT be stored on HPC. Use NYU's Secure Research Data Environments (SRDE) instead.

### Filesystem overview

| Filesystem | Path | Quota | Backed up? | Purged? | Best for |
|---|---|---|---|---|---|
| **Home** | `/home/$USER` | 50 GB / 30K inodes | Yes (daily) | No | Code, configs, scripts |
| **Scratch** | `/scratch/$USER` | 5 TB / 5M inodes | No | Yes (60 days no access) | Active job data, datasets |
| **Archive** | `/archive/$USER` | 2 TB / 20K inodes | Yes | No | Long-term archival of completed work |
| **RPS** | `/rw/<sharename>` | Paid (see below) | Yes | No | Shared project data |
| **Work (public)** | `/scratch/work/public` | Read-only | — | — | Public datasets (COCO, ImageNet, etc.) |

### Key rules and gotchas

- **Scratch purge policy:** Files not accessed for 60+ days are deleted. It is a policy violation to use scripts to artificially update access times — doing so will get your account locked.
- **Inode limits are real:** The 30K inode limit on `/home` is the single most common issue. Conda environments easily eat through this. Never install conda environments in `$HOME`.
- **Check your quotas** with the `myquota` command. To see where your inodes are going:
  ```
  cd $HOME
  du --inodes -h --max-depth=1
  ```
- **Don't change permissions** on your home directory to share files. Use Scratch or RPS instead.
- **Store code in git**, not in Scratch (which gets purged).

### Research Project Space (RPS)

RPS provides backed-up, non-purged shared storage built on the same VAST parallel filesystem as Scratch. It costs money:

- 1 TB of storage: $100/year
- 200K inodes: $100/year

Minimum initial request is 1 TB + 200K inodes ($200/year). Only PIs can request RPS — email hpc@nyu.edu with size, inode count, and finance contact.

### Data transfers

**Globus** is the recommended tool for large transfers. The Torch endpoint is `nyu#torch`. Collections available: Torch home, scratch, archive.

For smaller transfers, use `rsync` or `scp` via the Data Transfer Nodes (DTNs):

```
rsync -av myfile.txt <NetID>@dtn.torch.hpc.nyu.edu:/scratch/<NetID>/
```

Never do large data transfers on login nodes — use DTNs (`dtn.torch.hpc.nyu.edu`).

**rclone** is available for cloud storage (Google Drive, S3, etc.). Load it with `module load rclone/1.68.2` and configure via `rclone config`. Use DTNs for rclone transfers too.

**OOD** can be used for small uploads/downloads only.

### Sharing data with collaborators

Use **NFSv4 ACLs** (not `chmod 777`):

```bash
# Give collaborator read access to a file
nfs4_setfacl -a "A::<NetID>:R" filename

# Give collaborator full access to a directory (recursive, inherited)
nfs4_setfacl -a "A::collaborator:RX" /scratch/you
nfs4_setfacl -R -a "A:df:collaborator:RWX" /scratch/you/shared_dir

# View current ACLs
nfs4_getfacl filename
```

For team-wide access, request an IPA Linux Group from hpc@nyu.edu and manage membership via `ipa group-add-member`.

### Handling large numbers of small files

Many datasets (e.g., ImageNet with 14M images) create filesystem bottlenecks. Options, from best to worst:

1. **SquashFS + Singularity** — pack files into a `.sqf` image and mount as a read-only overlay. This is the recommended approach for datasets.
2. **HDF5** — container file with fast random access, supports parallel I/O.
3. **LMDB** — memory-mapped key-value store, good for deep learning dataloaders.
4. **SQLite** — great for structured data.
5. **`$SLURM_TMPDIR`** — untar to job-local disk each time (slower start, but fine for smaller datasets).
6. **RAM disk** — request via `#SBATCH --comment="ram_disk=1GB"` for ultra-fast I/O, but eats into memory.

---

## 5. Software and Environment Management

### Environment Modules (Lmod)

Torch uses Lmod for managing software. Key commands:

```bash
module avail              # List available modules
module load <module>      # Load a module
module unload <module>    # Unload a module
module purge              # Unload everything
module list               # Show loaded modules
module spider <name>      # Search for a module
```

Bioinformatics tools (samtools, vcftools, etc.) are available as modules wrapping containerized environments.

### The Golden Rule: Use Apptainer (Singularity) Containers

The HPC team **strongly recommends** setting up all computational environments via Apptainer containers with overlay files. This avoids inode quota issues, keeps your environment portable, and provides reproducibility.

### Setting up a Conda environment in Singularity (the right way)

This is the standard workflow on Torch. Every ML/AI project should use this approach.

**Step 1: Create a working directory**

```bash
mkdir /scratch/<NetID>/my-project
cd /scratch/<NetID>/my-project
```

**Step 2: Copy and decompress an overlay image**

```bash
cp -rp /share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz .
gunzip overlay-15GB-500K.ext3.gz
```

Browse `/share/apps/overlay-fs-ext3` for different size options (15GB/500K is good for most conda environments).

**Step 3: Choose a Singularity base image**

```bash
ls /share/apps/images/
```

Common choice: `cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif`

**Step 4: Launch the container in read/write mode**

```bash
singularity exec --fakeroot \
  --overlay overlay-15GB-500K.ext3:rw \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash
```

You should see a `Singularity>` prompt.

**Step 5: Install Miniforge**

```bash
wget --no-check-certificate \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p /ext3/miniforge3
```

**Step 6: Create the environment wrapper script `/ext3/env.sh`**

```bash
#!/bin/bash
unset -f which
source /ext3/miniforge3/etc/profile.d/conda.sh
export PATH=/ext3/miniforge3/bin:$PATH
export PYTHONPATH=/ext3/miniforge3/bin:$PATH
```

**Step 7: Activate and install packages**

```bash
source /ext3/env.sh
conda config --remove channels defaults   # use conda-forge only
conda update -n base conda -y
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install jupyterlab matplotlib pandas scikit-learn -y
```

**Step 8: Exit and use in jobs**

```bash
exit   # leave Singularity
```

In your SLURM scripts, launch with the overlay in read-only mode (`:ro`):

```bash
singularity exec --overlay /scratch/<NetID>/my-project/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; python my_script.py"
```

### Pre-installation warning

If your prompt shows `(base)` when you log in, you have conda initialized in your `~/.bashrc`. **Comment out or remove** the entire `>>> conda initialize >>>` block from `~/.bashrc`, then log out and back in. This block will interfere with Singularity-based environments.

### Alternative: Virtual environments (without containers)

If you don't need containers, you can use Python venvs on Scratch:

```bash
module load python/intel/3.8.6
mkdir /scratch/$USER/my_project && cd /scratch/$USER/my_project
python -m venv venv
source venv/bin/activate
pip install <packages>
pip freeze > requirements.txt
```

In SLURM scripts:

```bash
module purge; source venv/bin/activate; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK; python my_script.py
```

Remember: Scratch is purged after 60 days of inactivity.

---

## 6. Submitting Jobs with SLURM

### Core concepts

- **Login nodes** are for editing files, submitting jobs, and light tasks. Never run compute-heavy work on them.
- **Compute nodes** are where your jobs actually run, managed by the SLURM scheduler.
- A **job** is a set of commands you want to run on compute nodes.
- **Batch job submission** is submitting a script for the scheduler to run when resources are available.

### Essential commands

| Command | Purpose |
|---|---|
| `sbatch script.sh` | Submit a batch job |
| `srun --pty /bin/bash` | Start an interactive session |
| `squeue -u $USER` | Check your job status |
| `scancel <jobid>` | Cancel a job |
| `sacct -u $USER -l -j <jobid>` | Detailed stats on a past job |
| `sinfo --Format=Partition,GRES,CPUs,Features:26,NodeList` | View node/partition info |
| `my_slurm_accounts` | List your SLURM accounts |

### Writing a batch script

```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --account=<your_slurm_account>
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

module purge

singularity exec --nv \
  --overlay /scratch/<NetID>/my-project/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; python train.py"
```

Key flags:

- `--account` — **required on Torch**. Get yours from `my_slurm_accounts`.
- `--gres=gpu:1` — request GPUs. Max 24 GPUs per user for jobs under 48 hours.
- `--nv` flag on `singularity exec` enables GPU passthrough.
- `--time` — wall time limit. Be accurate — overestimating hurts scheduling.
- `--mem` — memory per node. Ask for ~20-30% more than you expect.

### Partitions

Do **not** manually specify partitions on Torch. The scheduler automatically dispatches jobs to all accessible GPU partitions that match your resource request.

The exception is preemptible jobs (see below).

### Preemptible jobs

You can run jobs on stakeholder GPUs you don't own via preemption. The job may be cancelled if the stakeholder group needs their GPUs back (after a 30-minute grace period).

```bash
#SBATCH --comment="preemption=yes;requeue=true"
```

To use **only** preemption partitions (might get more resources):

```bash
#SBATCH --comment="preemption=yes;preemption_partitions_only=yes;requeue=true"
```

Cancelled preemptible jobs are automatically re-queued with `requeue=true`. **Implement checkpointing** in your training code to take advantage of this.

Preemption priority order: Stakeholder jobs > GPU jobs > CPU-only jobs on GPU nodes.

### Advanced SBATCH options

**GPU MPS** (share a GPU between multiple processes):

```bash
#SBATCH --comment="gpu_mps=yes"
```

**RAM disk** (mount RAM as a fast disk):

```bash
#SBATCH --comment="ram_disk=1GB"
```

These can be combined with preemption:

```bash
#SBATCH --comment="preemption=yes;preemption_partitions_only=yes;requeue=true;gpu_mps=yes;ram_disk=1GB"
```

### Interactive sessions

For debugging and testing:

```bash
srun --cpus-per-task=2 --mem=10GB --time=04:00:00 --gres=gpu:1 --pty /bin/bash
```

### Monitoring past jobs

```bash
sacct -u $USER -l -j <jobid> | less -S
```

Compare requested vs. actual resource usage and adjust future submissions accordingly. A good rule: request 20-30% more time and memory than expected, but not wildly more.

### Low GPU utilization warning

Jobs with low GPU utilization **will be automatically canceled** on Torch. Make sure your code actually uses the GPUs you request. Profile your code before scaling up.

---

## 7. Machine Learning and AI on Torch

### Single-GPU training with PyTorch

The recommended workflow:

1. Create a working directory in `/scratch/<NetID>/`.
2. Set up a Singularity+Conda overlay (as described in Section 5).
3. Install PyTorch: `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`
4. Write your training script.
5. **Profile first** — install `line_profiler` (`pip install line_profiler`) and use `kernprof` to find bottlenecks.
6. Submit with a SLURM script requesting 1 GPU.

Always optimize for single-GPU performance before moving to multi-GPU. More GPUs = longer queue times.

### Multi-GPU training with DDP (Distributed Data Parallel)

Two reasons to go multi-GPU: execution time is too long, or the model doesn't fit on one GPU.

**Do not use `DataParallel`** — always use `DistributedDataParallel` (DDP).

DDP uses the Single-Program Multiple Data (SPMD) paradigm: the model is copied to each GPU, input data is split evenly, gradients are computed independently then averaged across GPUs, and weights are updated identically.

Key code changes from single-GPU to DDP:

1. **Initialize the process group:**
   ```python
   dist.init_process_group("nccl", rank=rank, world_size=world_size)
   ```
   Use `nccl` (not `gloo`) — `gloo` falls back to TCP via CPU.

2. **Wrap the model:**
   ```python
   model = Net().to(local_rank)
   ddp_model = DDP(model, device_ids=[local_rank])
   optimizer = optim.Adadelta(ddp_model.parameters(), lr=args.lr)
   ```

3. **Use DistributedSampler:**
   ```python
   train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
   train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,
                             num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
                             pin_memory=True)
   ```

DDP SLURM script example (multi-node):

```bash
#!/bin/bash
#SBATCH --job-name=ddp_training
#SBATCH --account=<account>
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=64GB
#SBATCH --time=08:00:00

export MASTER_PORT=$(shuf -i 10000-65535 -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)

srun singularity exec --nv \
  --overlay /scratch/<NetID>/my-env/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; python train_ddp.py"
```

Do a **scaling analysis** to find the optimal number of GPUs — measure throughput at 1, 2, 4, 8 GPUs and look for diminishing returns.

### TensorFlow on Torch

The setup is similar to PyTorch: create a Singularity+Conda overlay and install TensorFlow:

```bash
pip install tensorflow
```

For multi-GPU TensorFlow, use `tf.distribute.MirroredStrategy` (single-node) or `tf.distribute.MultiWorkerMirroredStrategy` (multi-node).

### LLM Inference

Two pathways:

**Hugging Face Transformers (basic inference):**
- Good for feature extraction, embeddings, small-scale batch processing.
- Set up a Singularity overlay, install `transformers` and `torch`.
- Use `AutoModel` to load weights and run forward passes.

**vLLM (high-performance serving):**
- Recommended for production-level throughput and low-latency inference.
- Uses PagedAttention for efficient GPU memory management.
- Drop-in replacement for OpenAI API.

vLLM setup on Torch:

```bash
# Pull the image
apptainer pull docker://vllm/vllm-openai:latest

# Redirect caches to scratch
export HF_HOME=/scratch/$USER/hf_cache
export VLLM_CACHE_ROOT=/scratch/$USER/vllm_cache
```

**Online serving** (OpenAI-compatible API):

```bash
# Terminal 1: start server
apptainer exec --nv vllm-openai_latest.sif vllm serve "Qwen/Qwen2.5-0.5B-Instruct"

# Terminal 2: query it
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-0.5B-Instruct", "messages": [{"role": "user", "content": "Hello!"}]}'
```

**Offline inference** (batch processing):

```python
from vllm import LLM
llm = LLM(model="facebook/opt-125m")
```

For batch-only offline inference without an HTTP endpoint, **SGLang** is a simpler alternative.

### Fine-tuning LLMs

The Torch docs provide a full example fine-tuning Gemma-3-4B-PT with LoRA on the `openassistant-guanaco` dataset. Key steps:

1. Set up a Singularity container with an overlay.
2. Install PyTorch, transformers, peft, trl, bitsandbytes.
3. Load the base model with 4-bit quantization (QLoRA).
4. Apply LoRA adapters.
5. Train using the `SFTTrainer` from the `trl` library.
6. Compare base model vs. fine-tuned vs. official instruction-tuned variant.

Complete scripts: [github.com/NYU-RTS/rts-docs-examples/tree/main/hpc/llm_fine_tuning](https://github.com/NYU-RTS/rts-docs-examples/tree/main/hpc/llm_fine_tuning)

---

## 8. Jupyter Notebooks via Open OnDemand

To run Jupyter in OOD with your custom Singularity+Conda environment:

1. Set up the overlay and install JupyterLab as described in Section 5.
2. Go to [OOD](https://ood.torch.hpc.nyu.edu) → Interactive Apps → Jupyter Notebook.
3. Fill in resources (CPUs, memory, GPUs, time).
4. In the **Singularity Image** and **Overlay** fields, point to your `.sif` and `.ext3` files.
5. Launch and connect when the session starts.

Troubleshooting: check logs in OOD (click the Session ID link) or at `/home/$USER/ondemand/data/sys/dashboard/batch_connect/sys/` on the terminal.

Other interactive apps available: RStudio, MATLAB, full Linux Desktop, JBrowse genome browser, IGV.

---

## 9. Datasets on Torch

The HPC team provides several public datasets at `/projects/work/public/ml-datasets/`. Many are packaged as `.sqf` (SquashFS) files for use with Singularity overlays:

- **COCO**: `/projects/work/public/ml-datasets/coco/coco-{2014,2015,2017}.sqf`
- **ImageNet/ILSVRC**: available under the ml-datasets directory
- Various others — browse the directory or check the Datasets page in the docs

To use a `.sqf` dataset:

```bash
singularity exec \
  --overlay /scratch/<NetID>/my-env/overlay-15GB-500K.ext3:ro \
  --overlay /projects/work/public/ml-datasets/coco/coco-2017.sqf:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash
```

The dataset files will appear at the root of the container (e.g., `/coco`).

---

## 10. Best Practices and Responsible Usage

### Be kind to login nodes

Login nodes are shared by all users. Use them only for:
- Editing files
- Submitting and monitoring jobs
- Light compilation (`make`, `tar`)

Never run `python my_training.py` or any compute-intensive task on a login node. Use `srun --pty /bin/bash` to get an interactive compute session.

### Test before scaling

Submit a small test job first. Check that it works, profile it, then scale up. A mistake in a 1000-core job wastes massive resources and budget.

### Request resources accurately

Use `sacct` to review past job performance:

```bash
sacct -u $USER -l -j <jobid> | less -S
```

Compare requested vs. actual CPU time, memory, and wall time. Adjust your next submission to request ~20-30% more than actual usage. Wildly overestimating time/memory means longer queue waits.

### Protect your data

- Use `myquota` regularly to stay within limits.
- Never rely on Scratch for important data — it's not backed up and gets purged.
- Store code in git repos and `/home`.
- Archive completed projects in `/archive` or RPS.
- Don't use `chmod 777` — use NFSv4 ACLs for sharing.

### Data transfers

Always use Data Transfer Nodes (`dtn.torch.hpc.nyu.edu`) for moving data, not login nodes. Use Globus for large transfers.

### Bundling files for transfer

Use `tar` to bundle many small files before transferring:

```bash
tar -czf my_data.tar.gz my_data_directory/
```

This is faster and more reliable than transferring thousands of individual files.

---

## 11. Quick Reference

### Key hostnames

| Host | Purpose |
|---|---|
| `login.torch.hpc.nyu.edu` (alias: `torch`) | Login nodes |
| `dtn.torch.hpc.nyu.edu` | Data transfer node |
| `ood.torch.hpc.nyu.edu` | Open OnDemand web interface |
| `projects.hpc.nyu.edu` | Project/allocation management portal |

### Key directories

| Path | Purpose |
|---|---|
| `/home/$USER` | Permanent code/config storage (50 GB, 30K inodes) |
| `/scratch/$USER` | Temporary job data (5 TB, 5M inodes, purged after 60 days) |
| `/archive/$USER` | Long-term archival (2 TB, 20K inodes) |
| `/share/apps/images/` | Singularity/Apptainer base images |
| `/share/apps/overlay-fs-ext3/` | Overlay file templates |
| `/projects/work/public/ml-datasets/` | Public datasets |
| `/scratch/work/public/examples/` | Example scripts |

### Key commands

| Command | What it does |
|---|---|
| `myquota` | Check disk/inode usage across filesystems |
| `my_slurm_accounts` | List available SLURM accounts |
| `sbatch script.sh` | Submit a batch job |
| `srun --pty /bin/bash` | Interactive compute session |
| `squeue -u $USER` | Check job queue |
| `scancel <jobid>` | Cancel a job |
| `sacct -u $USER` | Job history/statistics |
| `module avail` | List available software modules |
| `module load <name>` | Load a software module |

### Support

Email **hpc@nyu.edu** for any questions or issues. The HPC team is responsive and helpful.

---

*Source: NYU HPC Documentation at services.rt.nyu.edu/docs/hpc/*