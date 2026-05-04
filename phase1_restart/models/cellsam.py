"""CellSAM adapter (zero-shot path; fine-tune defers to Modal).

CellSAM expects (H, W, 3) with channel order (blank, nuclear, membrane). We map:
  channel 0 = zeros
  channel 1 = DAPI max-projection (nuclear)
  channel 2 = polyT max-projection (membrane proxy)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

from phase1_restart.pilot.adapter import SegAdapter


class CellSAMAdapter(SegAdapter):
    name = "cellsam"
    expects_channels = ["DAPI", "polyT"]
    runtime = "mps"

    def __init__(self):
        self._model = None
        self._device = "mps" if torch.backends.mps.is_available() else "cpu"

    def load_pretrained(self) -> None:
        from cellSAM import get_model
        self._model = get_model()  # default: cellsam_general

    def _to_hwc3(self, image: np.ndarray) -> np.ndarray:
        """(C, H, W) with channels [DAPI, polyT] → (H, W, 3) with [blank, DAPI, polyT]."""
        assert image.shape[0] == 2, f"expected 2 input channels (DAPI, polyT), got {image.shape}"
        H, W = image.shape[1], image.shape[2]
        out = np.zeros((H, W, 3), dtype=np.float32)
        out[..., 1] = image[0]  # DAPI → nuclear
        out[..., 2] = image[1]  # polyT → membrane
        return out

    def predict(self, image: np.ndarray) -> np.ndarray:
        if self._model is None:
            self.load_pretrained()
        from cellSAM import cellsam_pipeline
        img = self._to_hwc3(image)
        # Whole 2048x2048: use_wsi=True with tiling. block_size 400 + overlap 56 by default.
        mask = cellsam_pipeline(img, use_wsi=True)
        return np.asarray(mask).astype(np.int32)

    def fine_tune(self, train_fovs, val_fovs, output_dir: Path, n_epochs: int, **hp) -> Path:
        raise NotImplementedError(
            "Fine-tune is provided by phase1_restart.modal_app::fine_tune (Task 12)"
        )

    def load_checkpoint(self, path: Path) -> None:
        from cellSAM import get_model
        if self._model is None:
            self._model = get_model()
        state = torch.load(str(path), map_location=self._device, weights_only=False)
        # Best-effort load; fine-tune harness writes either full state_dict or {'state_dict': ...}.
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self._model.load_state_dict(state, strict=False)
