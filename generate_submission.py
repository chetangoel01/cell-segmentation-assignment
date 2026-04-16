"""
Generate a Kaggle submission CSV from your segmentation masks.

The competition evaluates spot-to-cell clustering using the Adjusted Rand Index (ARI).
Your submission is just: for each spot in test_spots.csv, which cluster (cell) does it
belong to?

You provide a segmentation mask (2048 x 2048, 0 = background, >0 = cell ID) for each of
the 4 test FOVs. This script looks up each spot in the mask using the pre-computed
image_row / image_col columns from test_spots.csv.

-----------------------------------------------------------------------------
Example usage in a notebook (recommended):
-----------------------------------------------------------------------------

    import pandas as pd
    from generate_submission import build_submission

    # Run your segmentation pipeline on each test FOV.
    masks = {}
    for fov in ['FOV_A', 'FOV_B', 'FOV_C', 'FOV_D']:
        dapi = load_dapi(f'test/{fov}/...')   # your loading code
        masks[fov] = my_segmentation(dapi)     # (2048, 2048), 0 = bg, >0 = cell

    test_spots = pd.read_csv('test_spots.csv')
    submission = build_submission(masks, test_spots)
    submission.to_csv('submission.csv', index=False)

-----------------------------------------------------------------------------
Example usage from the command line:
-----------------------------------------------------------------------------

    python generate_submission.py \
        --mask_A FOV_A_mask.npy \
        --mask_B FOV_B_mask.npy \
        --mask_C FOV_C_mask.npy \
        --mask_D FOV_D_mask.npy \
        --test_spots test_spots.csv \
        --output submission.csv
"""

import argparse
import numpy as np
import pandas as pd


def build_submission(masks: dict, test_spots: pd.DataFrame) -> pd.DataFrame:
    """
    Build a Kaggle submission DataFrame from segmentation masks and test spots.

    Parameters
    ----------
    masks : dict
        Dict mapping FOV name to a 2D segmentation mask.
        e.g. {'FOV_A': np.ndarray(2048, 2048), 'FOV_B': ..., ...}
        Each mask has 0 = background, >0 = cell ID.
    test_spots : pd.DataFrame
        The test_spots.csv DataFrame. Must have columns:
        spot_id, fov, image_row, image_col

    Returns
    -------
    pd.DataFrame
        Submission DataFrame with columns: spot_id, fov, cluster_id
        (same row order as test_spots)
    """
    required = ['spot_id', 'fov', 'image_row', 'image_col']
    missing = [c for c in required if c not in test_spots.columns]
    if missing:
        raise ValueError(f"test_spots DataFrame missing columns: {missing}")

    parts = []
    for fov, mask in masks.items():
        fov_spots = test_spots[test_spots['fov'] == fov].copy()
        if fov_spots.empty:
            print(f"  WARNING: no spots for {fov}, skipping")
            continue

        if mask.shape != (2048, 2048):
            raise ValueError(
                f"Mask for {fov} must be shape (2048, 2048), got {mask.shape}"
            )

        rows = fov_spots['image_row'].values
        cols = fov_spots['image_col'].values
        valid = (rows >= 0) & (rows < 2048) & (cols >= 0) & (cols < 2048)

        cluster_ids = np.full(len(fov_spots), 'background', dtype=object)
        mask_vals = np.zeros(len(fov_spots), dtype=int)
        mask_vals[valid] = mask[rows[valid], cols[valid]]
        for i in range(len(fov_spots)):
            if mask_vals[i] > 0:
                # Prefix with FOV name so cluster IDs are unique across FOVs
                cluster_ids[i] = f'{fov}_cell_{mask_vals[i]}'

        n_assigned = (cluster_ids != 'background').sum()
        print(
            f"  {fov}: {len(fov_spots):,} spots, {int(mask.max())} cells in mask, "
            f"{n_assigned:,} spots assigned ({100 * n_assigned / len(fov_spots):.1f}%)"
        )

        parts.append(pd.DataFrame({
            'spot_id': fov_spots['spot_id'].values,
            'fov': fov,
            'cluster_id': cluster_ids,
        }))

    combined = pd.concat(parts, ignore_index=True)

    # Preserve the original row order from test_spots.csv
    submission = (
        test_spots[['spot_id', 'fov']]
        .merge(combined[['spot_id', 'cluster_id']], on='spot_id', how='left')
    )
    submission['cluster_id'] = submission['cluster_id'].fillna('background')

    return submission[['spot_id', 'fov', 'cluster_id']]


def main():
    parser = argparse.ArgumentParser(
        description='Generate Kaggle submission CSV from segmentation masks'
    )
    parser.add_argument('--mask_A', required=True, help='Path to FOV_A .npy mask (2048x2048 int)')
    parser.add_argument('--mask_B', required=True, help='Path to FOV_B .npy mask (2048x2048 int)')
    parser.add_argument('--mask_C', required=True, help='Path to FOV_C .npy mask (2048x2048 int)')
    parser.add_argument('--mask_D', required=True, help='Path to FOV_D .npy mask (2048x2048 int)')
    parser.add_argument('--test_spots', default='test_spots.csv', help='Path to test_spots.csv')
    parser.add_argument('--output', default='submission.csv', help='Output submission CSV path')
    args = parser.parse_args()

    print("Loading test_spots.csv...")
    test_spots = pd.read_csv(args.test_spots)
    print(f"  {len(test_spots):,} spots across {test_spots['fov'].nunique()} FOVs")

    print("\nLoading masks...")
    masks = {
        'FOV_A': np.load(args.mask_A),
        'FOV_B': np.load(args.mask_B),
        'FOV_C': np.load(args.mask_C),
        'FOV_D': np.load(args.mask_D),
    }
    for fov, mask in masks.items():
        print(f"  {fov}: shape={mask.shape}, dtype={mask.dtype}, {int(mask.max())} cells")

    print("\nBuilding submission...")
    submission = build_submission(masks, test_spots)

    submission.to_csv(args.output, index=False)
    print(f"\nSubmission written to {args.output}")
    print(f"  Rows: {len(submission):,}")
    print(f"  Columns: {submission.columns.tolist()}")
    print(f"  Unique clusters: {submission['cluster_id'].nunique()}")
    print(f"  Background spots: {(submission['cluster_id'] == 'background').sum():,}")
    print("\nUpload this file to Kaggle for scoring.")


if __name__ == '__main__':
    main()
