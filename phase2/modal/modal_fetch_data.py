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


@app.function(image=image, timeout=900, volumes=VOLUMES, env=ENV)
def _verify_remote(fovs: list[str], split: str, skip_bil: bool) -> bool:
    """Stat volume FOVs + (optionally) HEAD BIL for each filename. All on Modal.

    Two-tier check, runs entirely on the Modal container — no laptop network IO:
      1. Canonical fingerprint (always): filename → kind → expected byte size.
      2. BIL HEAD (if reachable): content-length match against competition source.

    Returns True on overall pass, False on any mismatch / unknown / missing.
    Modal forwards stdout to the caller, so the local entrypoint just prints.
    """
    import re
    import urllib.request
    from pathlib import Path
    from urllib.error import HTTPError, URLError

    BIL_BASE = "https://download.brainimagelibrary.org/29/3c/293cc39ceea87f6d"
    COMPETITION_SOURCE = "sagittal_2/220912_wb3_sa2_2_5z18R_merfish5"
    UA = "phase2-fetch_bil/1.0 (chetangoel2011@gmail.com)"
    CANONICAL = {"round": 142_606_336, "fiducial": 41_943_040, "preimage": 226_492_416}

    def classify(name: str) -> str | None:
        if "473s5-408s5" in name:
            return "preimage"
        if name.startswith("Epi-750s1-635s1-545s1_"):
            return "fiducial"
        if name.startswith("Epi-750s5-635s5-545s1_"):
            return "round"
        return None

    def head_size(url: str, timeout: int = 20) -> int | None:
        req = urllib.request.Request(url, method="HEAD")
        req.add_header("User-Agent", UA)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                cl = r.headers.get("Content-Length")
                return int(cl) if cl else None
        except HTTPError as e:
            if e.code == 404:
                return None
            raise

    data_vol.reload()
    print(f"egress IP: ", end="", flush=True)
    try:
        with urllib.request.urlopen("https://api.ipify.org", timeout=10) as r:
            print(r.read().decode())
    except Exception as e:
        print(f"(unknown: {e})")

    bil_ok = False
    if not skip_bil:
        print(">>> probing BIL reachability …")
        try:
            sz = head_size(f"{BIL_BASE}/additional_files/codebook_32bit_v2.csv", timeout=15)
            bil_ok = (sz == 92_715)
        except (URLError, TimeoutError, OSError) as e:
            print(f"    BIL HEAD failed: {type(e).__name__}: {e}")
        print(f"    BIL reachable: {bil_ok}")
        if not bil_ok:
            print("    (host bil-data.ddns.psc.edu / 192.231.243.62 is a PSC SPOF — "
                  "blocks TCP from anywhere when down; falling back to canonical-only)")
    else:
        print(">>> --skip-bil: skipping BIL HEAD probes")

    overall_ok = True
    tile_re = re.compile(r"_(\d{3,4})[._]")
    for fov in fovs:
        folder = Path("/root/data") / split / fov
        print(f"\n========== {fov} ==========")
        if not folder.is_dir():
            print(f"  MISSING — directory not found at {folder}")
            overall_ok = False
            continue
        files = sorted(folder.glob("*.dax"))
        if not files:
            print(f"  MISSING — no .dax files in {folder}")
            overall_ok = False
            continue

        tiles = {tile_re.search(f.name).group(1) for f in files if tile_re.search(f.name)}
        tile = next(iter(tiles)) if len(tiles) == 1 else "??"
        print(f"discovered lab tile index: {tile}  (course FOV: {fov})\n")

        m_match = m_mis = m_miss = m_unknown = 0
        for f in files:
            name = f.name
            local_size = f.stat().st_size
            kind = classify(name)
            expected = CANONICAL.get(kind) if kind else None

            if expected is None:
                print(f"  UNKNOWN  {name}  (filename does not match merfish5 pattern)")
                m_unknown += 1
                continue
            if local_size != expected:
                print(f"  BAD-SIZE {name}  modal={local_size:,}  canonical {kind}={expected:,}")
                m_mis += 1
                continue

            if bil_ok:
                url = f"{BIL_BASE}/{COMPETITION_SOURCE}/data/{name}"
                try:
                    remote = head_size(url, timeout=20)
                except (URLError, TimeoutError, OSError) as e:
                    print(f"  CANON+?  {name}  {local_size:,} B  (BIL HEAD failed mid-run: {type(e).__name__})")
                    m_match += 1
                    continue
                if remote is None:
                    print(f"  CANON+404 {name}  remote not in competition source")
                    m_miss += 1
                elif remote == local_size:
                    print(f"  MATCH    {name}  {local_size:,} B  (canon + BIL)")
                    m_match += 1
                else:
                    print(f"  MISMATCH {name}  modal={local_size:,}  bil={remote:,}")
                    m_mis += 1
            else:
                print(f"  CANON-OK {name}  {local_size:,} B  ({kind})")
                m_match += 1

        if bil_ok:
            tag = f"{m_match} match, {m_mis} mismatch, {m_miss} missing-on-bil, {m_unknown} unknown"
        else:
            tag = f"{m_match} canonical-OK, {m_mis} bad-size, {m_unknown} unknown"
        print(f"\nsummary: {tag} of {len(files)} files")
        print(f"source:  {BIL_BASE}/{COMPETITION_SOURCE}/data/")
        print(f"tile:    {tile}")

        ok = (m_mis == 0 and m_miss == 0 and m_unknown == 0 and m_match == len(files))
        m = re.match(r"FOV_(\d+)$", fov)
        if m and tile != "??" and int(tile) != int(m.group(1)):
            print(f"  WARNING: tile id {tile} != FOV suffix {m.group(1)}")
            ok = False
        if not ok:
            overall_ok = False

    mode = "BIL+canonical" if bil_ok else "canonical-only"
    print(f"\n========== overall ==========")
    print(f"  {'PASS' if overall_ok else 'FAIL'}  ({len(fovs)} FOV(s), mode={mode})")
    return overall_ok


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


@app.local_entrypoint()
def verify(fovs: str, split: str = "train", skip_bil: bool = False):
    """Verify volume FOVs match the BIL competition source — entirely on Modal.

    All network IO (file stat + BIL HEAD probes) runs in the Modal container.
    Modal forwards stdout to your terminal, so you see the same report as if
    it ran locally — but no laptop bandwidth is used.

    Two-tier check inside the container:
      1. Canonical fingerprint (always runs): filename → kind → expected byte
         size for the merfish5 protocol (round / fiducial / preimage).
      2. BIL HEAD vs the competition source. Skipped when --skip-bil or when
         BIL is unreachable (the host is a single PSC SPOF that drops TCP
         from anywhere when down — see memory `bil_download_spof.md`).

    Example:
      modal run phase2/modal/modal_fetch_data.py::verify --fovs FOV_102,FOV_103,FOV_104
      modal run phase2/modal/modal_fetch_data.py::verify --fovs FOV_E --split test
      modal run phase2/modal/modal_fetch_data.py::verify --fovs FOV_102 --skip-bil
    """
    fov_list = [f.strip() for f in fovs.split(",") if f.strip()]
    if not fov_list:
        raise SystemExit("--fovs must be a non-empty comma-separated list")
    ok = _verify_remote.remote(fov_list, split, skip_bil)
    if not ok:
        raise SystemExit(1)
