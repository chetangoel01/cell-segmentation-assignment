"""run_experiment.py — the autoresearch harness. Not edited by the agent.

Reads experiment.py (the agent's current config), runs the full pipeline
(segmentation + classification + spot-level prediction) on a small fast-val
set or the full 10-FOV val, computes mean ARI per the Kaggle metric, and
appends the result to phase2/autoresearch/results/.

Usage:
  .venv/bin/python phase2/autoresearch/run_experiment.py            # fast (3 FOV) val
  .venv/bin/python phase2/autoresearch/run_experiment.py --full-val # full 10 FOV val

The script writes:
  phase2/autoresearch/results/<timestamp>.json   (machine-readable)
  appends one line to phase2/autoresearch/results.md  (human-readable log)

If the new run improves the local best by > 0.001, it's marked '*'. The
agent is expected to read results.md to decide what to try next.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUTORESEARCH = Path(__file__).resolve().parent
RESULTS_DIR = AUTORESEARCH / "results"
RESULTS_LOG = AUTORESEARCH / "results.md"

FAST_VAL_FOVS = ["FOV_151", "FOV_154", "FOV_159"]  # mix of densities
FULL_VAL_FOVS = [f"FOV_{i:03d}" for i in range(151, 161)]


def _load_experiment():
    spec = importlib.util.spec_from_file_location(
        "experiment", AUTORESEARCH / "experiment.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read_best_so_far() -> tuple[float, str | None]:
    """Best mean ARI in results.md so far, or (0.0, None) if none yet."""
    if not RESULTS_LOG.exists():
        return 0.0, None
    best, best_id = 0.0, None
    for line in RESULTS_LOG.read_text().splitlines():
        if "ARI=" not in line:
            continue
        try:
            ari_str = line.split("ARI=")[1].split()[0]
            ari = float(ari_str)
            iid = line.split()[1] if len(line.split()) > 1 else None
            if ari > best:
                best, best_id = ari, iid
        except (ValueError, IndexError):
            continue
    return best, best_id


def _config_summary(exp) -> str:
    return (f"seg={Path(exp.SEG_CHECKPOINT).name if exp.SEG_CHECKPOINT else 'cpsam'} "
            f"cp={exp.CELLPROB_THRESHOLD} fl={exp.FLOW_THRESHOLD} "
            f"d={exp.CELLPOSE_DIAMETER} nn={exp.NN_RADIUS} "
            f"3ch={exp.INCLUDE_SPOT_DENSITY} cls={Path(exp.CLASSIFIER_DIR).name}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--full-val", action="store_true",
                   help="Use all 10 val FOVs (151-160) instead of the fast 3-FOV set.")
    p.add_argument("--note", default="",
                   help="Extra note appended to the results.md entry.")
    args = p.parse_args()

    sys.path.insert(0, str(ROOT))
    from phase2.scripts.validate_local import main as validate_main  # noqa: E402

    exp = _load_experiment()
    iter_id = time.strftime("%Y%m%d-%H%M%S")
    val_fovs = FULL_VAL_FOVS if args.full_val else FAST_VAL_FOVS
    print(f"[{iter_id}] config: {_config_summary(exp)}")
    print(f"  hypothesis: {exp.HYPOTHESIS.strip()[:200]}")
    print(f"  val FOVs ({'full' if args.full_val else 'fast'}): {val_fovs}")

    out_json = RESULTS_DIR / f"{iter_id}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    argv = [
        "--models-dir",       str(exp.CLASSIFIER_DIR),
        "--val-fovs",         ",".join(val_fovs),
        "--cellpose-diameter", str(exp.CELLPOSE_DIAMETER),
        "--cellprob-threshold", str(exp.CELLPROB_THRESHOLD),
        "--flow-threshold",   str(exp.FLOW_THRESHOLD),
        "--nn-radius",        str(exp.NN_RADIUS),
        "--out",              str(out_json),
    ]
    if exp.SEG_CHECKPOINT:
        argv.extend(["--seg-checkpoint", exp.SEG_CHECKPOINT])
    if exp.INCLUDE_SPOT_DENSITY:
        argv.append("--include-spot-density")
        argv.extend(["--spot-density-sigma", str(exp.SPOT_DENSITY_SIGMA)])

    t0 = time.time()
    rv = validate_main(argv)
    wall = time.time() - t0
    if rv:
        print(f"[fatal] validate_local failed (rv={rv})")
        return int(rv)

    # Read result + decide if it improved
    result = json.loads(out_json.read_text())
    ari = result["mean_ari"]
    best_before, best_id = _read_best_so_far()
    improved = ari > best_before + 0.001
    marker = "*" if improved else " "
    val_tag = "FULL" if args.full_val else "fast"

    line = (f"- [{marker}] {iter_id}  ARI={ari:.4f}  ({val_tag}, {wall:.0f}s) "
            f"{_config_summary(exp)}")
    if args.note:
        line += f"  ({args.note})"
    print()
    print(line)
    if improved:
        print(f"  ⬆ new best (was {best_before:.4f} from {best_id})")
    else:
        print(f"  (best so far: {best_before:.4f} from {best_id})")

    # Append to log
    if not RESULTS_LOG.exists():
        RESULTS_LOG.write_text("# Autoresearch results log\n\n"
                                "Each line: `[*] <id> ARI=N (FAST|FULL, Ns) <config>`\n"
                                "`*` marks improvements over the prior best.\n\n")
    with RESULTS_LOG.open("a") as f:
        f.write(line + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
