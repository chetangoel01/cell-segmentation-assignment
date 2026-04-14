import os
import numpy as np
import pytest
from src.io import load_dax, get_dapi_stack, get_polyt_stack


DATA_ROOT = "/scratch/pl2820/competition"


def _epi_path():
    return f"{DATA_ROOT}/train/FOV_001/Epi-750s5-635s5-545s1-473s5-408s5_001.dax"


@pytest.mark.integration
def test_load_dax_shape():
    """Epi file has 27 frames, each 2048x2048."""
    path = _epi_path()
    if not os.path.exists(path):
        pytest.skip(f"Integration data not found at {path}")
    raw = load_dax(path, n_pixels=2048)
    assert raw.shape == (27, 2048, 2048)
    assert raw.dtype == np.uint16


@pytest.mark.integration
def test_get_dapi_stack_shape():
    path = _epi_path()
    if not os.path.exists(path):
        pytest.skip(f"Integration data not found at {path}")
    raw = load_dax(path, n_pixels=2048)
    dapi = get_dapi_stack(raw)
    assert dapi.shape == (5, 2048, 2048)  # 5 z-planes


@pytest.mark.integration
def test_get_polyt_stack_shape():
    path = _epi_path()
    if not os.path.exists(path):
        pytest.skip(f"Integration data not found at {path}")
    raw = load_dax(path, n_pixels=2048)
    polyt = get_polyt_stack(raw)
    assert polyt.shape == (5, 2048, 2048)

