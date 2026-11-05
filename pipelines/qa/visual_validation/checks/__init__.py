"""Validation checks for the visual validation pipeline.

This sub-package contains four orthogonal checks that together decide
whether a rendered video is consistent with its declarative ``scene.json``
metadata:

* :mod:`.visibility` – visibility window (when did the object appear /
  disappear?).
* :mod:`.depth` – relative depth ranking versus declared distances.
* :mod:`.spatial` – 2D pixel position versus the projected metadata
  azimuth / elevation.
* :mod:`.camera` – global camera yaw motion estimated from optical flow.

Each module exposes a single ``check_*`` entry point and a result
dataclass.  All entry points are re-exported here for convenience.
"""

from .camera import CameraResult, check_camera_motion
from .depth import DepthResult, check_depth_consistency
from .quantity import QuantityCheckResult, QuantityResult, check_quantity
from .spatial import (
    ObjectSpatialMetrics,
    SpatialResult,
    check_spatial_alignment,
)
from .visibility import VisibilityResult, check_visibility

__all__ = [
    "CameraResult",
    "DepthResult",
    "ObjectSpatialMetrics",
    "QuantityCheckResult",
    "QuantityResult",
    "SpatialResult",
    "VisibilityResult",
    "check_camera_motion",
    "check_depth_consistency",
    "check_quantity",
    "check_spatial_alignment",
    "check_visibility",
]
