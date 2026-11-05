import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

NAV_ROOT = Path(__file__).resolve().parents[1] / "pipelines" / "nav"
sys.path.insert(0, str(NAV_ROOT))
spec = importlib.util.spec_from_file_location("omniecho_nav_geometry", NAV_ROOT / "geometry.py")
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)


@pytest.mark.parametrize("source,expected", [
    ([0, 0, 1], [1, 0, 0]),
    ([-1, 0, 0], [0, 1, 0]),
    ([0, -1, 0], [0, 0, 1]),
    ([0, 0, -1], [-1, 0, 0]),
])
def test_camera_axes_map_to_forward_left_up(source, expected):
    direction, distance = geometry.source_dir_local(np.eye(4), np.array(source))
    np.testing.assert_allclose(direction, expected)
    assert distance == 1.0


def test_foa_channel_order_and_attenuation():
    mono = np.ones(16)
    near = geometry.encode_static_foa(mono, np.array([0, 1, 0]), 1.0)
    far = geometry.encode_static_foa(mono, np.array([0, 1, 0]), 2.0)
    assert near.shape == (4, 16)
    np.testing.assert_allclose(near[0], 0.3)
    np.testing.assert_allclose(near[1], 0.3)
    np.testing.assert_allclose(near[2:], 0)
    np.testing.assert_allclose(far, near / 2)
