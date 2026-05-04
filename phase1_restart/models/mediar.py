"""MEDIAR adapter (zero-shot path; fine-tune deferred to Modal).

MEDIAR (Lee et al., NeurIPS 2022 Cell Segmentation Challenge winner) is a SegFormer +
MA-Net architecture trained on multi-modality microscopy (CellPose + LiveCell +
DataScienceBowl + OmniPose). Pretrained on 7,241 fluorescence + brightfield images.

Weights source: HuggingFace Space "ghlee94/MEDIAR/resolve/main/main_model.pt" (anonymous,
no auth required, ~463 MB). Downloaded once into phase1_restart/external/MEDIAR/weights/.

The pickled model has a few version-rot issues vs current libs:
  - timm renamed timm.models.layers -> timm.layers
  - segmentation_models_pytorch renamed PAB/MFAB -> PABBlock/MFABBlock
This adapter installs aliases before torch.load so the pickle resolves.

Inference uses the HF Space's predict.py functions (sliding_window_inference, post_process)
which are bundled into phase1_restart/external/mediar_inference.py.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import torch

from phase1_restart.pilot.adapter import SegAdapter

REPO_ROOT = Path(__file__).resolve().parents[2]
MEDIAR_DIR = REPO_ROOT / "phase1_restart" / "external"
DEFAULT_WEIGHTS = MEDIAR_DIR / "MEDIAR" / "weights" / "main_model.pt"


def _install_compat_shims() -> None:
    """Patch known import-path / class-rename mismatches in pickled MEDIAR weights."""
    import timm.layers.drop  # noqa: F401
    if "timm.models.layers" not in sys.modules:
        sys.modules["timm.models.layers"] = sys.modules["timm.layers"]
    if "timm.models.layers.drop" not in sys.modules:
        sys.modules["timm.models.layers.drop"] = sys.modules["timm.layers.drop"]
    from segmentation_models_pytorch.decoders.manet import decoder as _md
    if not hasattr(_md, "PAB"):
        _md.PAB = _md.PABBlock
    if not hasattr(_md, "MFAB"):
        _md.MFAB = _md.MFABBlock


def _import_mediar_inference():
    """Import phase1_restart/external/mediar_inference.py and alias to __main__ so
    torch.load can resolve SegformerGH / DeepSegmantationHead / etc."""
    if str(MEDIAR_DIR) not in sys.path:
        sys.path.insert(0, str(MEDIAR_DIR))
    import mediar_inference as mi  # type: ignore
    sys.modules["__main__"] = mi
    return mi


class MediarAdapter(SegAdapter):
    name = "mediar"
    expects_channels = ["polyT", "DAPI"]  # MEDIAR was trained on (cyto, nuclear, blank)
    runtime = "mps"

    def __init__(self):
        self._model = None
        self._mi = None  # mediar_inference module
        self._device = "mps" if torch.backends.mps.is_available() else "cpu"

    def load_pretrained(self, weights_path: Path | None = None) -> None:
        _install_compat_shims()
        self._mi = _import_mediar_inference()
        path = weights_path or DEFAULT_WEIGHTS
        if not path.exists():
            raise FileNotFoundError(
                f"MEDIAR weights not found at {path}. "
                f"Download via: curl -L -o {path} "
                f"https://huggingface.co/spaces/ghlee94/MEDIAR/resolve/main/main_model.pt"
            )
        self._model = torch.load(str(path), map_location=self._device, weights_only=False)
        self._model.eval()
        self._model.to(self._device)
        # Current SMP's check_input_shape calls encoder.output_stride, which is a
        # property that reads self._output_stride and self._depth. The pickled mit_b5
        # encoder is missing both. Set the underlying instance attrs.
        enc = self._model.encoder
        if not hasattr(enc, "_output_stride"):
            enc._output_stride = 32  # SegFormer mit_b5 is 32x downsample at deepest stage
        if not hasattr(enc, "_depth"):
            enc._depth = 5  # mit_b5 has 5 stages; min(32, 2**5) = 32
        # Also, MAnetDecoder may check encoder.output_stride at decode time
        if not hasattr(self._model, "_output_stride"):
            self._model._output_stride = 32

    def _to_mediar_input(self, image: np.ndarray) -> torch.Tensor:
        """(C, H, W) with channels=[polyT, DAPI] -> (1, 3, H, W) MEDIAR-style.

        MEDIAR's pred_transforms expects (H, W, 3) with cytoplasm-like channels.
        We replicate the closest analogue: place polyT (cyto-like) and DAPI (nuclear)
        with a third channel as the polyT/DAPI mean (a synthetic "brightfield" proxy).
        Then min-max normalize, transpose to (1, 3, H, W).
        """
        assert image.shape[0] == 2, f"expected (polyT, DAPI), got shape {image.shape}"
        polyT = image[0]
        dapi = image[1]
        third = ((polyT + dapi) / 2.0).astype(np.float32)
        hwc = np.stack([polyT, dapi, third], axis=-1).astype(np.float32)
        # mimic MEDIAR's _normalize: nonzero-percentile rescale then min-max
        mn, mx = hwc.min(), hwc.max()
        if mx > mn:
            hwc = (hwc - mn) / (mx - mn)
        chw = np.moveaxis(hwc, -1, 0)
        return torch.from_numpy(chw).unsqueeze(0).float()

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> np.ndarray:
        if self._model is None:
            self.load_pretrained()
        mi = self._mi
        x = self._to_mediar_input(image).to(self._device)

        outputs = mi.sliding_window_inference(
            x,
            roi_size=512,
            sw_batch_size=2,
            predictor=self._model,
            padding_mode="reflect",
            mode="gaussian",
            overlap=0.5,
            device="cpu",
        )
        outputs = outputs.cpu().squeeze()
        # post_process expects (3, H, W) numpy: gradflow_x, gradflow_y, cellprob.
        pred_mask = mi.post_process(outputs.numpy(), self._device)
        return np.asarray(pred_mask).astype(np.int32)

    def fine_tune(self, train_fovs, val_fovs, output_dir: Path, n_epochs: int, **hp) -> Path:
        raise NotImplementedError(
            "Fine-tune is provided by phase1_restart.modal_app::fine_tune (Task 12)"
        )

    def load_checkpoint(self, path: Path) -> None:
        if self._model is None:
            self.load_pretrained()
        state = torch.load(str(path), map_location=self._device, weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # If state is a full model object, replace; if it's a state_dict, load_state_dict.
        if hasattr(state, "state_dict"):
            self._model = state
            self._model.eval().to(self._device)
        else:
            self._model.load_state_dict(state, strict=False)
