import numpy as np
import pytest
from phase1_restart.pilot.adapter import SegAdapter


def test_seg_adapter_is_abstract():
    with pytest.raises(TypeError):
        SegAdapter()  # type: ignore[abstract]


def test_concrete_subclass_must_implement_predict():
    class Incomplete(SegAdapter):
        name = "x"
        expects_channels = ["DAPI"]
        runtime = "mps"

    with pytest.raises(TypeError):
        Incomplete()


def test_complete_subclass_instantiates():
    class Complete(SegAdapter):
        name = "x"
        expects_channels = ["DAPI"]
        runtime = "mps"

        def load_pretrained(self):
            pass

        def predict(self, image):
            return np.zeros(image.shape[1:], dtype=np.int32)

        def fine_tune(self, train_fovs, val_fovs, output_dir, n_epochs, **hp):
            from pathlib import Path
            return Path("/tmp/fake.pt")

        def load_checkpoint(self, path):
            pass

    a = Complete()
    out = a.predict(np.zeros((1, 8, 8), dtype=np.float32))
    assert out.shape == (8, 8)
    assert out.dtype == np.int32
