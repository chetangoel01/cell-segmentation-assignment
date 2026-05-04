from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np


class SegAdapter(ABC):
    """Abstract segmentation adapter. One concrete subclass per foundation model.

    Subclasses MUST set class attrs: name (str), expects_channels (list[str]), runtime (str).
    """

    name: str
    expects_channels: list[str]
    runtime: str  # "mps" | "modal"

    @abstractmethod
    def load_pretrained(self) -> None:
        """Download / instantiate the pretrained model weights."""

    @abstractmethod
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict instance mask.

        image: (C, H, W) float32 in [0, 1], channel order matches expects_channels.
        returns: (H, W) int32, 0 = background, 1..N = cell IDs.
        """

    @abstractmethod
    def fine_tune(
        self,
        train_fovs: list[str],
        val_fovs: list[str],
        output_dir: Path,
        n_epochs: int,
        **hparams,
    ) -> Path:
        """Fine-tune from current weights. Returns path to best-val-ARI checkpoint."""

    @abstractmethod
    def load_checkpoint(self, path: Path) -> None:
        """Load a fine-tuned checkpoint from disk."""

    def predict_tiled(
        self, image: np.ndarray, tile: int = 512, overlap: int = 64
    ) -> np.ndarray:
        """OOM-safe tiled prediction. Default impl: tile, predict, stitch with mask-id offset."""
        _C, H, W = image.shape
        out = np.zeros((H, W), dtype=np.int32)
        next_id = 1
        step = tile - overlap
        for y0 in range(0, H, step):
            for x0 in range(0, W, step):
                y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
                patch = image[:, y0:y1, x0:x1]
                if patch.shape[1] < 64 or patch.shape[2] < 64:
                    continue
                m = self.predict(patch)
                m_relabeled = np.where(m > 0, m + next_id - 1, 0).astype(np.int32)
                empty = out[y0:y1, x0:x1] == 0
                out[y0:y1, x0:x1] = np.where(empty, m_relabeled, out[y0:y1, x0:x1])
                if m.max() > 0:
                    next_id += int(m.max())
        return out
