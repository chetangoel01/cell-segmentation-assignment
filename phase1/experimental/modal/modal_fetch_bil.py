"""Run BIL fetches from Modal's cloud egress when residential/VPN can't reach BIL.

Why: BIL's download host (download.brainimagelibrary.org → bil-data.ddns.psc.edu →
192.231.243.62) is TCP-unreachable from many residential ISPs and commercial VPN exits.
Modal containers egress through cloud IP space that BIL accepts.

Targets the existing `cell-seg-data` volume so fetched extras land alongside the phase 1
training data (under /root/data/external/competition_extras/).

Run from the REPO ROOT (so add_local_dir captures both phase1/ and phase2/):

    # 1. Sanity check: does BIL respond from Modal?
    modal run phase1/experimental/modal/modal_fetch_bil.py::ping

    # 2. Discover phase 1 test FOV lab tile indices (so we can exclude them from extras).
    modal run phase1/experimental/modal/modal_fetch_bil.py::run \\
        --args "verify FOV_A --split test"
    # repeat for FOV_B, FOV_C, FOV_D

    # 3. Pull cell_boundaries (segmentation labels — 3.6 GB) + positions (~6.4 GB total).
    modal run phase1/experimental/modal/modal_fetch_bil.py::run --args "fetch-support"

    # 4. Pull 60 extra training tiles (~133 GB rounds-only).
    #    Replace XXX,YYY,ZZZ,WWW with the lab tile indices discovered in step 2.
    modal run phase1/experimental/modal/modal_fetch_bil.py::run \\
        --args "fetch-source --exclude-tiles '001-040,XXX,YYY,ZZZ,WWW' \\
                --tiles 041-100 --rounds-only --workers 8"
"""
from __future__ import annotations

import shlex
import modal

app = modal.App("bil-fetch")

data_vol = modal.Volume.from_name("cell-seg-data")

image = (
    modal.Image.debian_slim(python_version="3.11")
    # Only ship the 3 files fetch_bil.py needs — keeps upload tiny so a flaky
    # connection doesn't break the context push.
    .add_local_file("phase2/scripts/fetch_bil.py", "/root/repo/phase2/scripts/fetch_bil.py")
    .add_local_file("phase2/src/io.py", "/root/repo/phase2/src/io.py")
    .add_local_file("phase2/src/__init__.py", "/root/repo/phase2/src/__init__.py")
)

VOLUMES = {"/root/data": data_vol}
ENV = {"MERFISH_DATA_ROOT": "/root/data"}


@app.function(image=image, timeout=300, env=ENV)
def _ping_remote():
    """HEAD a few BIL URLs to confirm reachability + measure latency."""
    import time
    import urllib.request

    print("=== egress IP ===")
    try:
        with urllib.request.urlopen("https://api.ipify.org", timeout=10) as r:
            print(r.read().decode())
    except Exception as e:
        print(f"(unknown: {e})")

    test_urls = [
        # Tiny: codebook (~93 KB) — sanity that the BIL deposit path resolves.
        ("codebook (93 KB)",
         "https://download.brainimagelibrary.org/29/3c/293cc39ceea87f6d/"
         "additional_files/codebook_32bit_v2.csv",
         None),
        # Mid: one round dax file from FOV_001 (~136 MB) — HEAD only, no body.
        ("FOV_001 round dax (136 MB)",
         "https://download.brainimagelibrary.org/29/3c/293cc39ceea87f6d/"
         "sagittal_2/220912_wb3_sa2_2_5z18R_merfish5/data/"
         "Epi-750s5-635s5-545s1_001_00.dax",
         142_606_336),
        # Mid: tile 041 (one of the candidate extras) — confirms the extras range exists.
        ("tile 041 round dax (136 MB)",
         "https://download.brainimagelibrary.org/29/3c/293cc39ceea87f6d/"
         "sagittal_2/220912_wb3_sa2_2_5z18R_merfish5/data/"
         "Epi-750s5-635s5-545s1_041_00.dax",
         142_606_336),
    ]

    print("\n=== BIL HEAD probes ===")
    for label, url, expected in test_urls:
        print(f"\n{label}")
        print(f"  {url}")
        t0 = time.time()
        try:
            req = urllib.request.Request(url, method="HEAD")
            req.add_header("User-Agent", "modal-bil-fetch/1.0")
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


@app.local_entrypoint()
def ping():
    """Confirm Modal egress can reach BIL."""
    _ping_remote.remote()


@app.local_entrypoint()
def run(args: str):
    """Forward an arbitrary fetch_bil.py subcommand to Modal.

    Example:
      modal run phase1/experimental/modal/modal_fetch_bil.py::run \\
          --args "verify FOV_A --split test"
    """
    _run_remote.remote(args)
