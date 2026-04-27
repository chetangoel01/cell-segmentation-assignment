from __future__ import annotations

from pathlib import Path
import numpy as np

# Frame indices within the 27-frame Epi file (from dataorganization.csv)
DAPI_FRAMES = [6, 11, 16, 21, 26]  # z0..z4, 405 nm
POLYT_FRAMES = [5, 10, 15, 20, 25]  # z0..z4, 488 nm


def load_dax(path: str, n_pixels: int = 2048) -> np.ndarray:
    """Load a raw .dax file as a (n_frames, n_pixels, n_pixels) uint16 array."""
    raw = np.fromfile(path, dtype=np.uint16)
    n_frames = raw.size // (n_pixels * n_pixels)
    if n_frames * (n_pixels * n_pixels) != raw.size:
        raise ValueError(
            f"DAX size {raw.size} not divisible by {n_pixels}x{n_pixels}; "
            f"got remainder {raw.size % (n_pixels * n_pixels)}"
        )
    return raw.reshape(n_frames, n_pixels, n_pixels)


def get_dapi_stack(raw: np.ndarray) -> np.ndarray:
    """Extract 5 DAPI z-plane images from the Epi raw array. Returns (5, H, W)."""
    return raw[DAPI_FRAMES]


def get_polyt_stack(raw: np.ndarray) -> np.ndarray:
    """Extract 5 polyT z-plane images from the Epi raw array. Returns (5, H, W)."""
    return raw[POLYT_FRAMES]


def _parse_fov_id(fov: str) -> str:
    """Return the FOV suffix for file glob (e.g. 'FOV_001' -> '001', 'FOV_A' -> 'A')."""
    fov = fov.strip()
    if fov.upper().startswith("FOV_"):
        fov = fov.split("_", 1)[1]
    # Only zero-pad if purely numeric (training FOVs); leave alpha IDs (FOV_A) as-is
    return fov.zfill(3) if fov.isdigit() else fov


def _pick_epi_file(candidates: list[Path]) -> Path:
    """Pick the 'best' Epi dax among candidates (usually the large 27-frame file)."""
    if not candidates:
        raise FileNotFoundError("No candidate Epi files provided.")
    # Prefer larger file (typically more frames); tie-break by name.
    return max(candidates, key=lambda p: (p.stat().st_size, -len(p.name), p.name))


def load_fov_images(
    fov_dir: str,
    *,
    fov: str | None = None,
    n_pixels: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Load DAPI and polyT stacks for a single FOV.

    Args:
        fov_dir: Either a directory containing the Epi `.dax` file(s), or a direct
            path to an Epi `.dax` file.
        fov: Optional FOV identifier (e.g. "FOV_001"). Needed when `fov_dir` points
            at a flat folder containing multiple FOVs' files.
    Returns:
        dapi_stack: (5, H, W) uint16
        polyt_stack: (5, H, W) uint16
    """
    fov_path = Path(fov_dir)

    if fov_path.is_file() and fov_path.suffix.lower() == ".dax":
        epi_file = fov_path
    else:
        # Layout A (competition-style): .../train/FOV_001/ contains a single Epi file
        # Layout B (local sample): ./data contains many Epi files, keyed by _001 etc.
        candidates: list[Path] = []

        # Most common competition file naming.
        candidates.extend(fov_path.glob("Epi-750s5-635s5-545s1-473s5-408s5_*.dax"))

        # If nothing found, try resolving by FOV id in a flat directory.
        if not candidates:
            if fov is None:
                fov = fov_path.name if fov_path.name.upper().startswith("FOV_") else None
            if fov is not None:
                fov_id = _parse_fov_id(fov)
                candidates.extend(
                    fov_path.glob(f"Epi-750s5-635s5-545s1-473s5-408s5_{fov_id}.dax")
                )
                candidates.extend(
                    fov_path.glob(f"Epi-750s5-635s5-545s1-473s5-408s5_{fov_id}_*.dax")
                )

        if not candidates:
            # Fall back: any .dax containing "Epi" (useful for alternative naming).
            candidates.extend([p for p in fov_path.glob("*.dax") if "Epi" in p.name])

        if not candidates:
            # Provide a more actionable error than the previous version.
            existing = sorted([p.name for p in fov_path.glob("*.dax")])[:20]
            hint = f" Found .dax files: {existing}" if existing else " No .dax files present."
            raise FileNotFoundError(
                f"No Epi file found in {fov_path}. "
                f"Searched common Epi patterns.{hint}"
            )

        epi_file = _pick_epi_file(sorted(set(candidates)))

    raw = load_dax(str(epi_file), n_pixels=n_pixels)
    return get_dapi_stack(raw), get_polyt_stack(raw)

