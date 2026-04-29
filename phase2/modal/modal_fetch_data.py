"""Pull phase-2 classification data into a NEW Modal volume `cell-seg-phase2`.

Why a new volume: keeps phase 2 storage isolated from phase 1's `cell-seg-data`
so we can wipe/rebuild without disturbing the segmentation training set.

Why Modal: BIL throttles residential IPs to ~35 KB/s for files >100 MB; Modal
egresses through cloud IPs and isn't subject to the same per-IP bucket. AWS S3
(allen-brain-cell-atlas) is unauthenticated and fast everywhere.

Targets (~6.6 GB total) that enable the Spatial-ID classifier path:
  - AWS Zhuang-ABCA-4-log2.h5ad      (107 MB)  reference cell-by-gene matrix
  - AWS cell_metadata_with_cluster_annotation.csv (51 MB)  4-level Allen labels
  - AWS gene.csv                     (85 KB)   gene table
  - BIL counts_mouse4_sagittal.h5ad  (1.0 GB)  has obs.fov for FOV-aware splits
  - BIL cell_boundaries_*.csv        (3.6 GB)  polygons → centroids → kNN graph
  - BIL tiled_positions.txt          (40 KB)   tile (x,y) µm
  - BIL codebook_32bit_v2.csv        (93 KB)   gene barcode reference
  - BIL spots_*.csv                  (1.8 GB, optional) raw decoded spots

Run from REPO ROOT:

    # 1. Sanity check: AWS + BIL reachable from Modal egress?
    modal run phase2/modal/modal_fetch_data.py::ping

    # 2. Pull the small AWS files (~158 MB).
    modal run phase2/modal/modal_fetch_data.py::run --args "fetch-aws"

    # 3. Pull the BIL support bundle (~6.5 GB; counts h5ad + boundaries + spots
    #    + positions + codebook).
    modal run phase2/modal/modal_fetch_data.py::run --args "fetch-support"

    # OR: convenience — both AWS + support in one shot.
    modal run phase2/modal/modal_fetch_data.py::all

    # 4. Inspect what landed in the volume.
    modal run phase2/modal/modal_fetch_data.py::ls
"""
from __future__ import annotations

import shlex
import modal

app = modal.App("phase2-fetch-data")

# NEW volume — created on first use under profile chetangoel2011.
data_vol = modal.Volume.from_name("cell-seg-phase2", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    # numpy: required by phase2/src/io.py (top-level import). The fetch paths don't
    # use it, but io.py is the canonical source of `data_root()` / `fov_dir()`.
    .pip_install("numpy")
    .add_local_file("phase2/scripts/fetch_bil.py", "/root/repo/phase2/scripts/fetch_bil.py")
    .add_local_file("phase2/src/io.py", "/root/repo/phase2/src/io.py")
    .add_local_file("phase2/src/__init__.py", "/root/repo/phase2/src/__init__.py")
)

VOLUMES = {"/root/data": data_vol}
ENV = {"MERFISH_DATA_ROOT": "/root/data"}


@app.function(image=image, timeout=300, env=ENV)
def _ping_remote():
    """HEAD a few AWS + BIL URLs to confirm reachability + measure latency."""
    import time
    import urllib.request

    print("=== egress IP ===")
    try:
        with urllib.request.urlopen("https://api.ipify.org", timeout=10) as r:
            print(r.read().decode())
    except Exception as e:
        print(f"(unknown: {e})")

    test_urls = [
        ("AWS gene.csv (85 KB)",
         "https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/"
         "metadata/Zhuang-ABCA-4/20241115/gene.csv",
         None),
        ("AWS log2.h5ad (107 MB)",
         "https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/"
         "expression_matrices/Zhuang-ABCA-4/20230830/Zhuang-ABCA-4-log2.h5ad",
         None),
        ("BIL codebook (93 KB)",
         "https://download.brainimagelibrary.org/29/3c/293cc39ceea87f6d/"
         "additional_files/codebook_32bit_v2.csv",
         None),
        ("BIL counts h5ad (1.0 GB)",
         "https://download.brainimagelibrary.org/29/3c/293cc39ceea87f6d/"
         "processed_data/counts_updated/counts_mouse4_sagittal.h5ad",
         None),
        ("BIL cell_boundaries (3.6 GB)",
         "https://download.brainimagelibrary.org/29/3c/293cc39ceea87f6d/"
         "processed_data/cell_boundaries_updated/mouse4_sagittal/"
         "220912_wb3_sa2_2_5z18R_merfish5.csv",
         None),
    ]

    print("\n=== HEAD probes ===")
    for label, url, expected in test_urls:
        print(f"\n{label}")
        print(f"  {url}")
        t0 = time.time()
        try:
            req = urllib.request.Request(url, method="HEAD")
            req.add_header("User-Agent", "modal-phase2-fetch/1.0")
            with urllib.request.urlopen(req, timeout=30) as r:
                cl = r.headers.get("Content-Length")
                dt = time.time() - t0
                tag = ""
                if expected is not None:
                    tag = " [MATCH]" if cl == str(expected) else f" [size {cl} != expected {expected}]"
                print(f"  → {r.status} {r.reason}  Content-Length={cl}  ({dt*1000:.0f}ms){tag}")
        except Exception as e:
            dt = time.time() - t0
            print(f"  → FAIL after {dt*1000:.0f}ms: {type(e).__name__}: {e}")


@app.function(image=image, timeout=21600, volumes=VOLUMES, env=ENV)
def _run_remote(args: str):
    """Subprocess into phase2/scripts/fetch_bil.py inside Modal."""
    import os
    import subprocess
    import sys

    data_vol.reload()
    cmd = [sys.executable, "/root/repo/phase2/scripts/fetch_bil.py", *shlex.split(args)]
    print(f">>> {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    env = {**os.environ, "MERFISH_DATA_ROOT": "/root/data"}
    result = subprocess.run(cmd, env=env)
    data_vol.commit()
    if result.returncode != 0:
        raise SystemExit(f"fetch_bil.py exited with code {result.returncode}")


@app.function(image=image, timeout=300, volumes=VOLUMES, env=ENV)
def _ls_remote():
    """List the volume contents with sizes."""
    import subprocess
    data_vol.reload()
    print("=== /root/data ===")
    subprocess.run(["ls", "-laRh", "/root/data"], check=False)


@app.local_entrypoint()
def ping():
    """Confirm Modal egress can reach AWS + BIL."""
    _ping_remote.remote()


@app.local_entrypoint()
def run(args: str):
    """Forward an arbitrary fetch_bil.py subcommand to Modal.

    Example:
      modal run phase2/modal/modal_fetch_data.py::run --args "fetch-aws"
    """
    _run_remote.remote(args)


@app.local_entrypoint()
def all():
    """Pull both AWS files and BIL support bundle in one shot."""
    print(">>> Step 1/2: fetching AWS files (~158 MB)")
    _run_remote.remote("fetch-aws")
    print(">>> Step 2/2: fetching BIL support bundle (~6.5 GB)")
    _run_remote.remote("fetch-support")


@app.local_entrypoint()
def ls():
    """List the volume contents."""
    _ls_remote.remote()
