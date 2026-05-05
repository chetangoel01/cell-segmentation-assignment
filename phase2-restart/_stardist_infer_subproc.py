"""Run StarDist inference on a list of phase-1 FOVs. Called from pilot.py via /tmp/stardist_venv.

Outputs <FOV>.npy (int32 2048x2048 labeled mask) per FOV in --out-dir.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

# Inject phase1/src for io.load_fov_images
HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "phase1"))

from src.io import load_fov_images  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fovs", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--norm-low", type=float, default=1.0)
    p.add_argument("--norm-high", type=float, default=99.8)
    args = p.parse_args()

    from stardist.models import StarDist2D  # noqa
    from csbdeep.utils import normalize  # noqa

    print(f"[stardist-infer] loading model {args.model_name} from {args.model_dir}")
    model = StarDist2D(None, name=args.model_name, basedir=args.model_dir)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fovs = [s.strip() for s in args.fovs.split(",") if s.strip()]
    data_root = Path(args.data_root)

    split = args.split
    for fov in fovs:
        if split == "test":
            fov_dir = data_root / "test" / fov
        else:
            fov_dir = data_root / "train" / fov
        if not fov_dir.exists():
            print(f"[stardist-infer] missing {fov_dir}, skipping")
            continue
        dapi, _polyt = load_fov_images(str(fov_dir))
        img = np.max(dapi, axis=0).astype(np.float32)
        img = normalize(img, args.norm_low, args.norm_high, axis=(0, 1))
        labels, _ = model.predict_instances(img)
        labels = labels.astype(np.int32)
        out_path = out_dir / f"{fov}.npy"
        np.save(out_path, labels)
        print(f"[stardist-infer] {fov}: {labels.max()} cells -> {out_path}")


if __name__ == "__main__":
    main()
