import numpy as np
import pytest


@pytest.fixture
def dummy_image() -> np.ndarray:
    """Return a zero-filled image of size 256x256x3"""
    return np.zeros((256, 256, 3), dtype=np.uint8)
