Torch is a GPU-heavy NYU cluster with multiple CPU and GPU node types, including H200, L40S, H100, and A100 nodes, plus separate login and data-transfer nodes. ([NYU Research Technology Services][1])

## 1. The core mental model

On Torch, getting access is a three-step chain: first you get an **HPC account**, then a **project** is created in the HPC projects portal, then that project gets an **allocation**, and that allocation is what gives you a **SLURM account** you can actually charge jobs to. An active HPC account alone is not enough to run jobs. Sponsored accounts are generally created for 12 months and non-faculty users renew yearly. NYU faculty can sponsor students, staff, visitors, and outside collaborators, but external collaborators need affiliate status first. ([NYU Research Technology Services][2])

The practical consequence is simple: if `ssh` works but `sbatch` fails, the first thing to check is whether your allocation is active and whether you are using the right `--account` value. In the portal, projects and allocations live at `projects.hpc.nyu.edu`, access requires NYU VPN, and the `slurm_account_name` attached to an active allocation is the value you use with `srun` and `sbatch`. ([NYU Research Technology Services][3])

## 2. How you actually connect

Torch is primarily a Linux cluster accessed over SSH. The docs recommend configuring SSH aliases for `login.torch.hpc.nyu.edu` and `dtn.torch.hpc.nyu.edu`. NYU also notes that SSH keys are **not supported on Torch** because of security restrictions, so expect NetID/password/MFA-based access rather than the usual personal key workflow. ([NYU Research Technology Services][4])

A good default is to think of the cluster as three entry points. Use the **login node** for light command-line work, file navigation, editing, job submission, and monitoring. Use a **compute node** for anything resource-intensive. Use the **data-transfer node** for moving large datasets in and out. For browser-based access, Open OnDemand is available at `ood.torch.hpc.nyu.edu` on-campus or through NYU VPN. It gives you file management, a shell, job monitoring, Jupyter, RStudio, and even a full desktop without local SSH, X11, or port-forwarding setup. ([NYU Research Technology Services][4])

## 3. Storage: what goes where

The single most important storage rule is that Torch is for **Moderate Risk data only**. High-risk data such as PII, ePHI, or CUI should not be stored in the HPC environment; NYU directs those workloads to SRDE instead. ([NYU Research Technology Services][5])

The next rule is to treat storage tiers differently. The docs consistently frame `/home` as the place to keep your important small-footprint materials such as source code, scripts, and job files; `/scratch` as fast working storage; and RPS as shared paid project storage. The transfer docs note that `/home`, `/scratch`, `/archive`, and RPS are all available through the data-transfer node. ([NYU Research Technology Services][6])

For shared lab or project work, **RPS** is the premium option. It is mounted on compute nodes, backed up, and not subject to an old-file purging policy, which makes it much better than scratch for long-lived shared data and code. It is also separate from your usual filesystem quotas, but it is limited in number, has an annual cost, and requests must be made by a PI via the HPC team. ([NYU Research Technology Services][7])

For routine hygiene, watch quotas and inodes. NYU explicitly calls out inode exhaustion in home directories as a common problem, especially when people install large Conda environments there. The `myquota` command is the first diagnostic command you should learn. ([NYU Research Technology Services][8])

For large transfers, use **Globus** unless you have a strong reason not to. NYU describes it as the recommended option for large-volume transfers, and the Torch DTNs are optimized for moving data between cluster filesystems and endpoints outside the cluster. ([NYU Research Technology Services][6])

For sharing files with collaborators on Torch, prefer **NFSv4 ACLs** over broad `chmod 777` permissions. The docs explicitly discourage `777` because it can lead to accidental or malicious data loss. ([NYU Research Technology Services][9])

For datasets made of millions of small files, do not just dump and expand everything onto scratch and hope for the best. NYU’s guidance is to consider archives, container-style formats, or database-like formats such as SQLite, HDF5, or LMDB, and in some cases unpack into job-local temporary space like `SLURM_TMPDIR`. The point is to reduce metadata pressure and improve read performance. ([NYU Research Technology Services][10])

## 4. Jobs and SLURM: how work actually runs

Torch uses SLURM, and every real HPC workflow reduces to this: request resources, run on compute nodes, and charge the job to the right account. NYU says all job submissions must include `--account`, and you can list the SLURM accounts you have access to with `my_slurm_accounts`. ([NYU Research Technology Services][11])

A useful command set to memorize is: `srun` for interactive runs, `sbatch` for batch scripts, `salloc` for interactive allocations, `sinfo` for node/partition info, `squeue` for queued/running jobs, `sacct` for historical accounting, and `scancel` to stop work. That is the core operational surface area of Torch. ([NYU Research Technology Services][12])

Torch-specific scheduling guidance matters. The docs say not to manually specify partitions except for preemption-related cases; the scheduler should usually choose the right default. They also call out a user GPU quota of 24 GPUs for jobs under 48 hours, and note that public users using stakeholder resources may be preempted when stakeholders submit new jobs. ([NYU Research Technology Services][11])

The deeper performance lesson is to estimate resources from evidence rather than guessing. NYU’s intro HPC material emphasizes using past job statistics to refine future `cpus`, memory, and wall time requests. On a shared cluster, accurate requests help both your turnaround time and everyone else’s. ([NYU Research Technology Services][13])

## 5. Software environments: the NYU-preferred way

Torch uses **Lmod modules** for centrally managed software. At minimum, you should know `module load`, `module unload`, `module purge`, and `module show`. Modules are how the cluster exposes multiple versions of compilers, interpreters, and libraries without polluting the base system. ([NYU Research Technology Services][14])

For Python and Conda, NYU’s recommendation is stronger than on many clusters: they explicitly encourage creating your computational environments inside **Apptainer overlay files** rather than relying on bare Conda environments scattered across home or scratch. They also warn against mixing multiple Conda-style environment patterns in a way that causes package contamination. ([NYU Research Technology Services][15])

If you do use plain Python virtual environments or plain Conda, keep them **per-project** and avoid bloating `/home`. The docs emphasize reproducibility by keeping an environment in the project directory itself, and they repeatedly remind users that files in `/scratch` are subject to purging after 60 days of inactivity. ([NYU Research Technology Services][15])

## 6. Containers: the platform Torch wants you to use

The container story on Torch is straightforward: **Apptainer is the supported container runtime**. NYU explicitly says they support Apptainer rather than Docker or Kubernetes directly on the cluster. The reason is mostly operational and security-oriented: Apptainer does not require root access, fits HPC better, and is the supported route for custom environments. ([NYU Research Technology Services][16])

The standard workflow is: pull a container image, attach an overlay if you need a writable environment, then launch it on a compute node. The docs also warn against running containers on login nodes because those processes may be terminated under login-node resource limits. ([NYU Research Technology Services][16])

In practice, this means you should think of “environment setup” and “compute execution” as distinct steps. Build or update your overlay/container environment, then submit or launch jobs that use that environment. That is the most reproducible path on Torch. ([NYU Research Technology Services][16])

## 7. Open OnDemand: the easiest way to get started fast

OOD is the best entry point for users who do not want to live in terminal-only workflows immediately. It lets you manage files, open a shell, monitor jobs, and launch interactive apps such as Jupyter and RStudio through the browser, all without local SSH/X11 setup. ([NYU Research Technology Services][17])

This is especially useful for interactive data science, quick file inspection, and cases where a full Linux desktop is more convenient than terminal forwarding. It is also the cleanest path for many users who would otherwise try to force GUI apps over SSH. ([NYU Research Technology Services][17])

## 8. ML and AI on Torch

Torch’s docs include a real ML/AI section rather than leaving users to guess. The platform guidance is to optimize for the **single-GPU case first**, because larger allocations mean longer queue times and wasted resources if your code is not already efficient. NYU’s PyTorch guide explicitly says to tune a single-GPU workflow before moving to multi-GPU training. ([NYU Research Technology Services][18])

A subtle but important operational detail is that **compute nodes do not have internet access** for ordinary download workflows, so download datasets and assets in advance or structure your setup accordingly. The PyTorch tutorial calls this out directly for MNIST. ([NYU Research Technology Services][18])

For LLM work, NYU documents two main paths. The first is a **basic Hugging Face transformers** workflow: create a project directory in scratch, use `srun --pty` to move to a compute node, attach a writable overlay, install Miniforge and required packages, and then run the model through an `sbatch` script. The second is **vLLM**, which NYU recommends for higher-throughput inference and OpenAI-compatible serving. ([NYU Research Technology Services][19])

Their vLLM guidance is especially practical: put Hugging Face and vLLM caches in scratch, not home; keep code and `.slurm` files in home; and remember scratch is not backed up and is purged after 60 days of inactivity. NYU also reports that on Torch, vLLM outperformed llama-cpp in their tested Qwen cases and exposes an OpenAI-compatible HTTP API by default on `localhost:8000`. ([NYU Research Technology Services][20])

## 9. The mistakes most likely to bite you

The highest-probability mistakes on Torch are: assuming an HPC account alone lets you run jobs; storing the wrong class of data on the cluster; installing giant environments in home until inode quota breaks things; doing large transfers with ad hoc tools instead of Globus; using `chmod 777` instead of ACLs; running heavy work on login nodes; and unpacking massive small-file datasets directly onto shared filesystems without thinking about metadata overhead. ([NYU Research Technology Services][2])

## 10. A good default workflow for you

A strong “do this every time” pattern on Torch is: get your HPC account approved; make sure your project and allocation are active; connect over SSH or OOD; create a project directory; keep code and job scripts in home or another durable location; keep large transient working data in scratch; build your environment in an Apptainer overlay; launch test runs interactively with `srun`; then move to reproducible `sbatch` scripts once the workflow is stable. Use Globus for big transfers, `myquota` to watch storage health, and OOD when you want browser-native interactive work. ([NYU Research Technology Services][21])

## 11. Where to get unstuck

NYU’s support page points users first to the Intro to Shell and Intro to HPC tutorials, then to trainings and workshops, and finally to direct help at `hpc@nyu.edu` for both simple and advanced cases. ([NYU Research Technology Services][22])

[1]: https://services.rt.nyu.edu/docs/hpc/getting_started/intro/ "Start here! | Connecting researchers to computational resources."
[2]: https://services.rt.nyu.edu/docs/hpc/getting_started/getting_and_renewing_an_account/ "Getting and Renewing an Account | Connecting researchers to computational resources."
[3]: https://services.rt.nyu.edu/docs/hpc/getting_started/hpc_project_management_portal/ "HPC Project Management Portal | Connecting researchers to computational resources."
[4]: https://services.rt.nyu.edu/docs/hpc/connecting_to_hpc/connecting_to_hpc/ "Connecting to the HPC Cluster | Connecting researchers to computational resources."
[5]: https://services.rt.nyu.edu/docs/hpc/storage/intro_and_data_management/ "HPC Storage | Connecting researchers to computational resources."
[6]: https://services.rt.nyu.edu/docs/hpc/storage/data_transfers/ "Data Transfers | Connecting researchers to computational resources."
[7]: https://services.rt.nyu.edu/docs/hpc/storage/research_project_space/ "Research Project Space (RPS) | Connecting researchers to computational resources."
[8]: https://services.rt.nyu.edu/docs/hpc/storage/best_practices/ "Best Practices on HPC Storage | Connecting researchers to computational resources."
[9]: https://services.rt.nyu.edu/docs/hpc/storage/sharing_data_on_hpc/ "Sharing Data on HPC | Connecting researchers to computational resources."
[10]: https://services.rt.nyu.edu/docs/hpc/storage/large_number_of_small_files/ "Large Number of Small Files | Connecting researchers to computational resources."
[11]: https://services.rt.nyu.edu/docs/hpc/submitting_jobs/slurm_submitting_jobs/ "Submitting Jobs on Torch | Connecting researchers to computational resources."
[12]: https://services.rt.nyu.edu/docs/hpc/submitting_jobs/slurm_main_commands/ "Slurm: Command reference | Connecting researchers to computational resources."
[13]: https://services.rt.nyu.edu/docs/hpc/tutorial_intro_hpc/using_resources_effectively/ "Using resources effectively | Connecting researchers to computational resources."
[14]: https://services.rt.nyu.edu/docs/hpc/tools_and_software/modules/ "Modules | Connecting researchers to computational resources."
[15]: https://services.rt.nyu.edu/docs/hpc/tools_and_software/python_packages_with_virtual_environments/ "Python Packages with Virtual Environments | Connecting researchers to computational resources."
[16]: https://services.rt.nyu.edu/docs/hpc/containers/intro/ "Custom Applications with Containers | Connecting researchers to computational resources."
[17]: https://services.rt.nyu.edu/docs/hpc/ood/ood_intro/ "Introduction to Open OnDemand (OOD) | Connecting researchers to computational resources."
[18]: https://services.rt.nyu.edu/docs/hpc/ml_ai_hpc/pytorch_intro/ "Single-GPU Training with PyTorch | Connecting researchers to computational resources."
[19]: https://services.rt.nyu.edu/docs/hpc/ml_ai_hpc/LLM%20inference/run_hf_model/ "Basic LLM Inference with Hugging Face transformers | Connecting researchers to computational resources."
[20]: https://services.rt.nyu.edu/docs/hpc/ml_ai_hpc/LLM%20inference/vLLM/ "High-performance LLM inference with vLLM | Connecting researchers to computational resources."
[21]: https://services.rt.nyu.edu/docs/hpc/getting_started/requesting_an_allocation/ "Managing allocations for your project | Connecting researchers to computational resources."
[22]: https://services.rt.nyu.edu/docs/hpc/support/support/ "Support | Connecting researchers to computational resources."
