"""Pull matched-distribution training data from BIL deposit 10.35077/act-bag.

Source mapping verified 2026-04-28 against the deposit's experiment_metadata.csv.
Competition data lives at sagittal_2/220912_wb3_sa2_2_5z18R_merfish5/ (mouse4).

Subcommands:
  verify       — confirm BIL has matching files for a local FOV (content-length match).
  list-pool    — print the safe-list of session URLs (excludes competition source).
  inspect      — list .dax files in a session URL with sizes.
  fetch        — download .dax files from a session to MERFISH_DATA_ROOT.
  fetch-counts — pull cell-by-gene .h5ad matrices for the classifier path.

Usage:
  python scripts/fetch_bil.py verify FOV_101
  python scripts/fetch_bil.py list-pool
  python scripts/fetch_bil.py inspect sa1_sample1
  python scripts/fetch_bil.py fetch sa1_sample1 --rounds-only --workers 4
  python scripts/fetch_bil.py fetch-counts mouse4
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError

# Local imports — reuse io.py conventions.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io import _parse_fov_id, data_root, fov_dir  # noqa: E402

BIL_BASE = "https://download.brainimagelibrary.org/29/3c/293cc39ceea87f6d"
COMPETITION_SOURCE = "sagittal_2/220912_wb3_sa2_2_5z18R_merfish5"

# AWS S3 mirror — same dataset, ~500x faster than BIL on residential. Verified 2026-04-28.
AWS_BASE = "https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com"
AWS_DATASET = "Zhuang-ABCA-4"  # mouse4_sagittal — competition source = section .001
AWS_EXPR_VERSION = "20230830"
AWS_META_VERSION = "20241115"

# Verified 2026-04-28 from experiment_metadata.csv.
# Each entry: sample_id -> (relative_path, animal, note).
SAFE_POOL: dict[str, tuple[str, str, str]] = {
    # Same animal as competition (mouse4_sagittal). Tightest distribution match.
    "sa2_sample3": ("sagittal_2/220908_wb3_sa2_12_5z18R_merfish5", "mouse4", "same-animal"),
    # Different animal (mouse3_sagittal), same protocol/codebook_v2/merfish5 panel.
    "sa1_sample1":  ("sagittal_1/220609_wb3_sa1_1_5z18R_merfish5", "mouse3", "cross-animal"),
    "sa1_sample2":  ("sagittal_1/220613_wb3_sa1_2_5z18R_merfish5", "mouse3", "cross-animal"),
    "sa1_sample5":  ("sagittal_1/220627_wb3_sa1_5_5z18R_merfish5", "mouse3", "cross-animal"),
    "sa1_sample7":  ("sagittal_1/220710_wb3_sa1_B_7_5z18R_merfish5", "mouse3", "cross-animal"),
    "sa1_sample9":  ("sagittal_1/220717_wb3_sa1_B_9_5z18R_merfish5", "mouse3", "cross-animal"),
    "sa1_sample12": ("sagittal_1/220706_wb3_sa1_B_12_21_5z18R_merfish5", "mouse3", "cross-animal"),
    "sa1_sample13": ("sagittal_1/220713_wb3_sa1_B_13_20_5z18R_merfish5", "mouse3", "cross-animal"),
}

USER_AGENT = "phase2-fetch_bil/1.0 (chetangoel2011@gmail.com)"


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _request(url: str, range_header: str | None = None, method: str = "GET", timeout: int = 60):
    req = urllib.request.Request(url, method=method)
    req.add_header("User-Agent", USER_AGENT)
    if range_header:
        req.add_header("Range", range_header)
    return urllib.request.urlopen(req, timeout=timeout)


def head_size(url: str) -> int | None:
    """Return Content-Length via HEAD, or None on 404/error."""
    try:
        with _request(url, method="HEAD") as r:
            cl = r.headers.get("Content-Length")
            return int(cl) if cl else None
    except HTTPError as e:
        if e.code == 404:
            return None
        raise


# nginx autoindex line format (verified 2026-04-28):
#   <a href="NAME">NAME</a>{spaces}13-Sep-2022 10:06{spaces}41943040
# Trailing token is byte size (or "-" for directories).
_AUTOINDEX_LINE = re.compile(
    r'<a href="(?P<href>[^"?][^"]*)">[^<]*</a>\s+\S+\s+\S+\s+(?P<size>\d+|-)\s*$',
    re.MULTILINE,
)


def list_directory(url: str, timeout: int = 600) -> list[str]:
    return [name for name, _ in list_directory_sized(url, timeout=timeout)]


def list_directory_sized(url: str, timeout: int = 600) -> list[tuple[str, int | None]]:
    """Return (name, size_bytes_or_None). size is None for directories.

    Tolerates truncated chunked responses on huge autoindex pages (~15 MB).
    """
    if not url.endswith("/"):
        url = url + "/"
    import http.client
    with _request(url, timeout=timeout) as r:
        try:
            raw = r.read()
        except http.client.IncompleteRead as e:
            raw = e.partial
    html = raw.decode("utf-8", errors="replace")

    out: list[tuple[str, int | None]] = []
    seen: set[str] = set()
    for m in _AUTOINDEX_LINE.finditer(html):
        href = m.group("href")
        if href == "../" or href in seen:
            continue
        seen.add(href)
        size_s = m.group("size")
        size = None if size_s == "-" else int(size_s)
        out.append((href, size))

    if not out:
        # Fallback: names only, no sizes.
        for href in re.findall(r'<a href="([^"?][^"]*)">', html):
            if href != "../" and href not in seen:
                seen.add(href)
                out.append((href, None))
    return out


# ---------- verify ----------

def expected_url(fov: str, dax_filename: str, source: str = COMPETITION_SOURCE) -> str:
    return f"{BIL_BASE}/{source}/data/{dax_filename}"


_TILE_FROM_FILENAME = re.compile(r"_(\d{3,4})[._]")


def discover_tile_from_files(folder: Path) -> str | None:
    """Find the lab tile index by inspecting .dax filenames inside a local FOV dir.

    Course-distributed FOV dirs (incl. lettered test FOVs like FOV_E) keep the lab's
    numeric tile index inside each .dax filename, even when the dir name is a letter.
    """
    tiles: set[str] = set()
    for f in folder.glob("*.dax"):
        m = _TILE_FROM_FILENAME.search(f.name)
        if m:
            tiles.add(m.group(1))
    if len(tiles) == 1:
        return tiles.pop()
    if len(tiles) > 1:
        # Pick the most common (in case of stray helpers).
        from collections import Counter
        return Counter(t for f in folder.glob("*.dax") for t in _TILE_FROM_FILENAME.findall(f.name)).most_common(1)[0][0]
    return None


def cmd_verify(args: argparse.Namespace) -> int:
    folder = fov_dir(args.fov, split=args.split)
    if not folder.is_dir():
        log(f"local FOV folder not found: {folder}")
        return 2

    local_files = sorted(folder.glob("*.dax"))
    if not local_files:
        log(f"no .dax files in {folder}")
        return 2

    tile = discover_tile_from_files(folder)
    if not tile:
        log(f"could not extract tile index from filenames in {folder}")
        return 2

    print(f"discovered lab tile index: {tile}  (course FOV: {args.fov})")
    print()

    matches = mismatches = missing = 0
    for f in local_files:
        local_size = f.stat().st_size
        url = f"{BIL_BASE}/{COMPETITION_SOURCE}/data/{f.name}"
        remote_size = head_size(url)
        if remote_size is None:
            print(f"  MISSING  {f.name}  remote 404")
            missing += 1
        elif remote_size == local_size:
            print(f"  MATCH    {f.name}  {local_size:,} B")
            matches += 1
        else:
            print(f"  MISMATCH {f.name}  local={local_size:,}  remote={remote_size:,}")
            mismatches += 1

    print()
    print(f"summary: {matches} match, {mismatches} mismatch, {missing} missing of {len(local_files)} files")
    print(f"source:  {BIL_BASE}/{COMPETITION_SOURCE}/data/")
    print(f"tile:    {tile}")
    return 0 if (matches and not mismatches and not missing) else 1


# ---------- list-pool ----------

def cmd_list_pool(args: argparse.Namespace) -> int:
    print(f"Competition source (EXCLUDE — leakage):")
    print(f"  {BIL_BASE}/{COMPETITION_SOURCE}")
    print()
    print("Safe-list of matched-distribution sessions (verified 2026-04-28):")
    for sample, (path, animal, note) in SAFE_POOL.items():
        print(f"  [{animal:6} {note:13}] {sample:13} -> {BIL_BASE}/{path}")
    return 0


# ---------- inspect ----------

@dataclass
class FileEntry:
    name: str
    size: int


def session_url(sample: str) -> str:
    if sample not in SAFE_POOL:
        raise ValueError(f"unknown sample id: {sample}. Known: {list(SAFE_POOL)}")
    return f"{BIL_BASE}/{SAFE_POOL[sample][0]}"


def cmd_inspect(args: argparse.Namespace) -> int:
    base = session_url(args.sample) + "/data/"
    log(f"listing {base}")
    sized = list_directory_sized(base)
    entries: list[FileEntry] = [
        FileEntry(name=n, size=sz) for n, sz in sized if n.endswith(".dax") and sz is not None
    ]
    log(f"parsed {len(entries)} .dax entries from autoindex")

    entries.sort(key=lambda e: e.name)
    total = sum(e.size for e in entries)

    # Bucket by size to characterize: round files (~136 MB) vs preimage (~216 MB).
    by_size: dict[int, int] = {}
    for e in entries:
        by_size[e.size] = by_size.get(e.size, 0) + 1

    print(f"# session: {args.sample}  url: {base}")
    print(f"# {len(entries)} files, total {total / 1e9:.2f} GB")
    print(f"# size buckets: {dict(sorted(by_size.items()))}")
    if args.long:
        for e in entries:
            print(f"  {e.size:>14,} B  {e.name}")
    else:
        # Per-tile summary. Tile padding varies per session (3 or 4 digits).
        tile_re = re.compile(r"_(\d{3,4})[._]")
        per_tile: dict[str, list[FileEntry]] = {}
        for e in entries:
            m = tile_re.search(e.name)
            tile = m.group(1).lstrip("0") or "0" if m else "??"
            per_tile.setdefault(tile, []).append(e)
        for tile in sorted(per_tile, key=lambda t: int(t) if t.isdigit() else -1):
            files = per_tile[tile]
            sz = sum(f.size for f in files)
            print(f"  tile {tile:>4}: {len(files):>3} files, {sz / 1e9:.2f} GB")
    return 0


# ---------- fetch ----------

def _download_one(url: str, dest: Path, expected_size: int | None) -> tuple[Path, str]:
    """Download url -> dest. Skips if dest already has expected_size bytes."""
    if expected_size is not None and dest.exists() and dest.stat().st_size == expected_size:
        return dest, "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with _request(url) as r, open(tmp, "wb") as out:
            while True:
                chunk = r.read(1 << 20)  # 1 MiB
                if not chunk:
                    break
                out.write(chunk)
        if expected_size is not None and tmp.stat().st_size != expected_size:
            return dest, f"FAIL size {tmp.stat().st_size} != {expected_size}"
        tmp.rename(dest)
        return dest, "ok"
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass


def cmd_fetch(args: argparse.Namespace) -> int:
    base = session_url(args.sample) + "/data/"
    out_root = Path(args.out) if args.out else (data_root() / "external" / args.sample / "data")
    out_root.mkdir(parents=True, exist_ok=True)

    log(f"fetching from {base}")
    log(f"writing to    {out_root}")
    sized = list_directory_sized(base)
    sizes: dict[str, int | None] = {n: sz for n, sz in sized if n.endswith(".dax")}
    dax = sorted(sizes.keys())

    if args.tile_prefix:
        # Match the tile id with either 3- or 4-digit padding (sessions vary).
        tile_pat = re.compile(rf"_0*({args.tile_prefix})[._]")
        dax = [n for n in dax if tile_pat.search(n)]
    if args.rounds_only:
        dax = [n for n in dax if "473s5-408s5" not in n]
    if args.preimage_only:
        dax = [n for n in dax if "473s5-408s5" in n]
    if args.limit:
        dax = dax[: args.limit]

    total_bytes = sum(sizes[n] for n in dax if sizes[n])
    log(f"total to fetch: {total_bytes / 1e9:.2f} GB across {len(dax)} files")
    if args.dry_run:
        for n in dax:
            print(f"  would fetch {sizes[n]:>14,} B  {base + n}  ->  {out_root / n}")
        return 0

    t0 = time.time()
    n_ok = n_skip = n_fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_download_one, base + n, out_root / n, sizes[n]): n for n in dax}
        for i, fut in enumerate(as_completed(futs), 1):
            n = futs[fut]
            try:
                _, status = fut.result()
            except Exception as e:
                status = f"FAIL {e}"
            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_fail += 1
            log(f"  [{i:>4}/{len(dax)}] {status:<6} {n}")

    dt = time.time() - t0
    log(f"done in {dt:.0f}s — ok={n_ok} skip={n_skip} fail={n_fail}")
    return 0 if n_fail == 0 else 1


# ---------- fetch-source ----------

# Course's training FOVs are tiles 101..160 in the source session.
COMPETITION_TRAIN_TILES = set(f"{i:03d}" for i in range(101, 161))


def parse_tile_spec(spec: str) -> set[str]:
    """Parse a comma-separated tile spec like '050,200,300-310' into 3-digit zero-padded tile ids."""
    out: set[str] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            for i in range(int(a), int(b) + 1):
                out.add(f"{i:03d}")
        else:
            out.add(f"{int(part):03d}")
    return out


def cmd_fetch_source(args: argparse.Namespace) -> int:
    """Fetch extras from the competition source session — same animal/section/panel/protocol.

    Requires --exclude-tiles to list the test FOV tile indices (run `verify` on each test
    FOV on HPC to discover them). Train tiles 101..160 are excluded automatically.
    """
    base = f"{BIL_BASE}/{COMPETITION_SOURCE}/data/"
    out_root = Path(args.out) if args.out else (data_root() / "external" / "competition_extras" / "data")
    out_root.mkdir(parents=True, exist_ok=True)

    exclude = COMPETITION_TRAIN_TILES | parse_tile_spec(args.exclude_tiles or "")
    log(f"excluding {len(exclude)} tiles "
        f"({len(COMPETITION_TRAIN_TILES)} train + {len(exclude) - len(COMPETITION_TRAIN_TILES)} test)")

    log(f"listing {base}")
    sized = list_directory_sized(base)
    sizes: dict[str, int | None] = {n: sz for n, sz in sized if n.endswith(".dax")}
    dax_all = sorted(sizes.keys())

    # Group by tile id; drop tiles in exclude set.
    by_tile: dict[str, list[str]] = {}
    for n in dax_all:
        m = _TILE_FROM_FILENAME.search(n)
        if not m:
            continue
        tile = m.group(1).zfill(3)
        if tile in exclude:
            continue
        by_tile.setdefault(tile, []).append(n)

    candidate_tiles = sorted(by_tile.keys(), key=int)
    log(f"available extras: {len(candidate_tiles)} tiles ({len(dax_all)} total files in session)")

    # Pick tiles by strategy.
    if args.tiles:
        wanted = parse_tile_spec(args.tiles)
        picked = [t for t in candidate_tiles if t in wanted]
        log(f"explicit --tiles selected: {len(picked)} tiles")
    elif args.adjacent:
        # Tiles closest in numeric distance to the train range (proxy for spatial proximity).
        train_lo, train_hi = 101, 160
        scored = sorted(
            candidate_tiles,
            key=lambda t: min(abs(int(t) - train_lo), abs(int(t) - train_hi)),
        )
        picked = scored[: args.limit or 60]
        log(f"adjacent strategy: picked {len(picked)} tiles closest to train range 101-160")
    else:
        picked = candidate_tiles[: args.limit or 60]
        log(f"first-N strategy: picked {len(picked)} tiles starting at tile {picked[0] if picked else '-'}")

    # Build file list.
    files: list[str] = []
    for t in picked:
        for n in by_tile[t]:
            if args.rounds_only and "473s5-408s5" in n:
                continue
            if args.preimage_only and "473s5-408s5" not in n:
                continue
            files.append(n)

    total = sum(sizes[n] for n in files if sizes[n])
    log(f"selected {len(files)} files across {len(picked)} tiles, total {total / 1e9:.2f} GB")

    if args.dry_run:
        for t in picked:
            n_files = sum(1 for n in by_tile[t]
                          if not (args.rounds_only and "473s5-408s5" in n)
                          and not (args.preimage_only and "473s5-408s5" not in n))
            tile_total = sum(sizes[n] for n in by_tile[t]
                             if sizes[n]
                             and not (args.rounds_only and "473s5-408s5" in n)
                             and not (args.preimage_only and "473s5-408s5" not in n))
            print(f"  tile {t}: {n_files:>2} files, {tile_total / 1e9:.2f} GB")
        print()
        print(f"# {len(files)} files, {total / 1e9:.2f} GB total")
        print(f"# excluded train tiles: {sorted(COMPETITION_TRAIN_TILES)}")
        if args.exclude_tiles:
            print(f"# excluded test tiles: {sorted(parse_tile_spec(args.exclude_tiles))}")
        return 0

    if not args.exclude_tiles:
        log("REFUSING to fetch without --exclude-tiles. Run `verify FOV_E ... FOV_N` on HPC")
        log("first to discover the test tile indices, or pass --exclude-tiles '' to override.")
        return 2

    t0 = time.time()
    n_ok = n_skip = n_fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_download_one, base + n, out_root / n, sizes[n]): n for n in files}
        for i, fut in enumerate(as_completed(futs), 1):
            n = futs[fut]
            try:
                _, status = fut.result()
            except Exception as e:
                status = f"FAIL {e}"
            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_fail += 1
            log(f"  [{i:>4}/{len(files)}] {status:<6} {n}")

    dt = time.time() - t0
    log(f"done in {dt:.0f}s — ok={n_ok} skip={n_skip} fail={n_fail}")
    return 0 if n_fail == 0 else 1


# ---------- fetch-support ----------

def cmd_fetch_support(args: argparse.Namespace) -> int:
    """Pull the support files (positions, spots, cell boundaries, counts) for the source session.

    These mirror what the course distributes as `reference/fov_metadata.csv`,
    `spots_train.csv`, `cell_boundaries_train.csv`, `counts_train.h5ad`, and the labels
    embedded in `counts_*.h5ad.obs`.
    """
    out_root = Path(args.out) if args.out else (data_root() / "external" / "competition_support")
    out_root.mkdir(parents=True, exist_ok=True)

    session = COMPETITION_SOURCE
    session_name = session.split("/")[-1]  # 220912_wb3_sa2_2_5z18R_merfish5

    targets = [
        # (relative URL path, local filename, label, mandatory)
        (f"{session}/settings/tiled_positions.txt",
         "tiled_positions.txt", "FOV positions (per-tile x,y in µm)", True),
        (f"processed_data/decoded_spots/mouse4_sagittal/spots_{session_name}.csv",
         f"spots_{session_name}.csv", "decoded mRNA spots (sorted by fov)", True),
        (f"processed_data/cell_boundaries_updated/mouse4_sagittal/{session_name}.csv",
         f"cell_boundaries_{session_name}.csv", "cell polygon boundaries (cell_id-keyed, no fov column)", True),
        (f"processed_data/counts_updated/counts_mouse4_sagittal.h5ad",
         "counts_mouse4_sagittal.h5ad", "cell-by-gene matrix + obs metadata (incl. fov, taxonomy labels)", True),
        (f"additional_files/codebook_32bit_v2.csv",
         "codebook_32bit_v2.csv", "gene barcode table (codebook v2 — competition uses this)", True),
        (f"additional_files/dataorganization/mouse4_sagittal/dataorganization_{session_name}.csv",
         f"dataorganization_{session_name}.csv", "per-round bit/channel/z manifest", False),
    ]

    if args.with_raw_counts:
        targets.append((
            f"processed_data/counts_updated/raw_counts_mouse4_sagittal.h5ad",
            "raw_counts_mouse4_sagittal.h5ad", "raw (un-normalized) cell-by-gene matrix", True,
        ))

    log(f"writing to {out_root}")
    log(f"session: {session_name}")

    sizes = {}
    for path, _, label, _ in targets:
        sz = head_size(f"{BIL_BASE}/{path}")
        sizes[path] = sz
        log(f"  {label:60s} {sz / 1e9:>7.2f} GB" if sz else f"  {label:60s} MISSING")

    total = sum(s or 0 for s in sizes.values())
    log(f"total: {total / 1e9:.2f} GB")

    if args.dry_run:
        for path, name, _, _ in targets:
            print(f"  would fetch {sizes[path] or 0:>14,} B  {BIL_BASE}/{path}  ->  {out_root / name}")
        return 0

    n_ok = n_skip = n_fail = 0
    for path, name, label, mandatory in targets:
        if sizes[path] is None:
            log(f"  SKIP   {name} (404)")
            if mandatory:
                n_fail += 1
            continue
        _, status = _download_one(f"{BIL_BASE}/{path}", out_root / name, sizes[path])
        log(f"  {status:<6} {name}")
        if status == "ok":
            n_ok += 1
        elif status == "skip":
            n_skip += 1
        else:
            n_fail += 1

    log(f"done — ok={n_ok} skip={n_skip} fail={n_fail}")
    return 0 if n_fail == 0 else 1


# ---------- fetch-aws ----------

def cmd_fetch_aws(args: argparse.Namespace) -> int:
    """Pull Zhuang-ABCA-4 files from AWS S3 (Allen Brain Cell Atlas).

    Same dataset as BIL but ~500x faster on residential (BIL throttles to ~35 KB/s,
    AWS delivers ~18 MB/s). Use this for the classifier path.

    AWS section .001 = competition source (220912_wb3_sa2_2). cell_label is the join
    key with BIL cell_boundaries.
    """
    out_root = Path(args.out) if args.out else (data_root() / "external" / "aws")
    out_root.mkdir(parents=True, exist_ok=True)

    targets = [
        # (relative URL, local filename, label, mandatory)
        (f"metadata/{AWS_DATASET}/{AWS_META_VERSION}/views/cell_metadata_with_cluster_annotation.csv",
         "cell_metadata_with_cluster_annotation.csv",
         "cell labels (class/subclass/supertype/cluster + confidence)", True),
        (f"metadata/{AWS_DATASET}/{AWS_META_VERSION}/cell_metadata.csv",
         "cell_metadata.csv",
         "cell metadata (CCF coords, abc_sample_id)", False),
        (f"metadata/{AWS_DATASET}/{AWS_META_VERSION}/gene.csv",
         "gene.csv", "gene table", True),
        (f"expression_matrices/{AWS_DATASET}/{AWS_EXPR_VERSION}/{AWS_DATASET}-log2.h5ad",
         f"{AWS_DATASET}-log2.h5ad",
         "log2-normalized cell-by-gene matrix", not args.raw_only),
        (f"expression_matrices/{AWS_DATASET}/{AWS_EXPR_VERSION}/{AWS_DATASET}-raw.h5ad",
         f"{AWS_DATASET}-raw.h5ad",
         "raw counts matrix", args.with_raw or args.raw_only),
    ]

    # Filter targets the user disabled
    targets = [t for t in targets if t[3]]

    log(f"writing to {out_root}")
    log(f"AWS dataset: {AWS_DATASET}  (competition source = section .001)")

    sizes = {}
    for path, _, label, _ in targets:
        url = f"{AWS_BASE}/{path}"
        sz = head_size(url)
        sizes[path] = sz
        if sz:
            log(f"  {label:55s} {sz / 1e6:>8.1f} MB")
        else:
            log(f"  {label:55s} MISSING")

    total = sum(s or 0 for s in sizes.values())
    log(f"total: {total / 1e6:.1f} MB")

    if args.dry_run:
        for path, name, _, _ in targets:
            print(f"  would fetch {sizes[path] or 0:>14,} B  {AWS_BASE}/{path}  ->  {out_root / name}")
        return 0

    n_ok = n_skip = n_fail = 0
    for path, name, label, _ in targets:
        if sizes[path] is None:
            log(f"  SKIP   {name} (missing on AWS)")
            n_fail += 1
            continue
        url = f"{AWS_BASE}/{path}"
        _, status = _download_one(url, out_root / name, sizes[path])
        log(f"  {status:<6} {name}  ({sizes[path] / 1e6:.1f} MB)")
        if status == "ok":
            n_ok += 1
        elif status == "skip":
            n_skip += 1
        else:
            n_fail += 1

    log(f"done — ok={n_ok} skip={n_skip} fail={n_fail}")
    return 0 if n_fail == 0 else 1


# ---------- fetch-counts ----------

ANIMAL_TO_H5AD = {
    "mouse3": ["counts_mouse3_sagittal.h5ad", "raw_counts_mouse3_sagittal.h5ad"],
    "mouse4": ["counts_mouse4_sagittal.h5ad", "raw_counts_mouse4_sagittal.h5ad"],
    "all": [
        "counts_mouse1_coronal.h5ad", "raw_counts_mouse1_coronal.h5ad",
        "counts_mouse2_coronal.h5ad", "raw_counts_mouse2_coronal.h5ad",
        "counts_mouse3_sagittal.h5ad", "raw_counts_mouse3_sagittal.h5ad",
        "counts_mouse4_sagittal.h5ad", "raw_counts_mouse4_sagittal.h5ad",
    ],
}


def cmd_fetch_counts(args: argparse.Namespace) -> int:
    if args.animal not in ANIMAL_TO_H5AD:
        log(f"unknown animal: {args.animal}. Choose from {list(ANIMAL_TO_H5AD)}")
        return 2
    out_root = Path(args.out) if args.out else (data_root() / "external" / "counts")
    out_root.mkdir(parents=True, exist_ok=True)
    base = f"{BIL_BASE}/processed_data/"

    files = ANIMAL_TO_H5AD[args.animal]
    log(f"fetching {len(files)} count files to {out_root}")
    sizes = {f: head_size(base + f) for f in files}
    total = sum(s for s in sizes.values() if s)
    log(f"total: {total / 1e9:.2f} GB")
    if args.dry_run:
        for f in files:
            print(f"  would fetch {sizes[f]:>14,} B  {base + f}")
        return 0
    for f in files:
        if sizes[f] is None:
            log(f"  MISSING {f}")
            continue
        _, status = _download_one(base + f, out_root / f, sizes[f])
        log(f"  {status:<6} {f}")
    return 0


# ---------- main ----------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser("verify", help="check BIL has matching files for a local FOV")
    pv.add_argument("fov", help="FOV id, e.g. FOV_101")
    pv.add_argument("--split", default="train", choices=["train", "test"])
    pv.set_defaults(fn=cmd_verify)

    pl = sub.add_parser("list-pool", help="print the safe-list of session URLs")
    pl.set_defaults(fn=cmd_list_pool)

    pi = sub.add_parser("inspect", help="list .dax files in a session")
    pi.add_argument("sample", help="sample id, e.g. sa1_sample1")
    pi.add_argument("--workers", type=int, default=8)
    pi.add_argument("--long", action="store_true", help="per-file listing instead of per-tile summary")
    pi.set_defaults(fn=cmd_inspect)

    pf = sub.add_parser("fetch", help="download .dax from a session")
    pf.add_argument("sample", help="sample id from list-pool")
    pf.add_argument("--out", help="output dir (default: $MERFISH_DATA_ROOT/external/<sample>/data/)")
    pf.add_argument("--tile-prefix", help="regex for the tile id (e.g. '0[0-2][0-9]' for tiles 0-29)")
    pf.add_argument("--rounds-only", action="store_true", help="skip the multichannel preimage")
    pf.add_argument("--preimage-only", action="store_true", help="only fetch the multichannel preimage")
    pf.add_argument("--limit", type=int, help="cap number of files (for testing)")
    pf.add_argument("--workers", type=int, default=4)
    pf.add_argument("--dry-run", action="store_true")
    pf.set_defaults(fn=cmd_fetch)

    ps = sub.add_parser("fetch-source", help="pull extras from competition source session (same animal/section)")
    ps.add_argument("--exclude-tiles", default=None,
                    help="comma-sep list/range of test FOV lab tile indices, e.g. '050,200,300-310'. "
                         "Train tiles 101-160 are auto-excluded. Discover via `verify FOV_E --split test` on HPC.")
    ps.add_argument("--tiles", help="explicit tile spec to fetch (overrides strategy)")
    ps.add_argument("--adjacent", action="store_true", help="pick tiles closest in index to train range 101-160")
    ps.add_argument("--limit", type=int, default=60, help="max number of tiles (default 60)")
    ps.add_argument("--rounds-only", action="store_true")
    ps.add_argument("--preimage-only", action="store_true")
    ps.add_argument("--out", help="output dir (default: $MERFISH_DATA_ROOT/external/competition_extras/data/)")
    ps.add_argument("--workers", type=int, default=4)
    ps.add_argument("--dry-run", action="store_true")
    ps.set_defaults(fn=cmd_fetch_source)

    pss = sub.add_parser("fetch-support",
                         help="pull positions/spots/boundaries/counts/codebook for the source session")
    pss.add_argument("--with-raw-counts", action="store_true", help="also fetch raw_counts h5ad")
    pss.add_argument("--out", help="output dir (default: $MERFISH_DATA_ROOT/external/competition_support/)")
    pss.add_argument("--dry-run", action="store_true")
    pss.set_defaults(fn=cmd_fetch_support)

    pa = sub.add_parser("fetch-aws", help="pull Zhuang-ABCA-4 files from AWS S3 (~500x faster than BIL)")
    pa.add_argument("--with-raw", action="store_true", help="also fetch raw counts h5ad (in addition to log2)")
    pa.add_argument("--raw-only", action="store_true", help="fetch only raw counts h5ad (skip log2)")
    pa.add_argument("--out", help="output dir (default: $MERFISH_DATA_ROOT/external/aws/)")
    pa.add_argument("--dry-run", action="store_true")
    pa.set_defaults(fn=cmd_fetch_aws)

    pc = sub.add_parser("fetch-counts", help="pull cell-by-gene .h5ad matrices")
    pc.add_argument("animal", help="mouse3 | mouse4 | all")
    pc.add_argument("--out", help="output dir (default: $MERFISH_DATA_ROOT/external/counts/)")
    pc.add_argument("--dry-run", action="store_true")
    pc.set_defaults(fn=cmd_fetch_counts)

    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
