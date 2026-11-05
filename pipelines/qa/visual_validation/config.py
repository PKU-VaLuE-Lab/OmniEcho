"""Configuration objects for the visual validation pipeline.

Centralises model paths, video / camera parameters, inference parameters and
validation thresholds so that every downstream module can consume a single,
strongly-typed configuration instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


# Default external-asset locations.  These are the canonical on-disk paths the
# user has prepared; ``ValidationConfig.default()`` instantiates the dataclass
# with these so that callers do not need to know about them.
_DEFAULT_DEPTH_MODEL_PATH = Path(os.environ.get("OMNIECHO_DEPTH_MODEL", "/path/to/depth-model.pth"))

_DEFAULT_SAM3_MODEL_PATH = Path(os.environ.get("OMNIECHO_SEGMENTATION_MODEL", "/path/to/segmentation-model.pt"))


@dataclass
class ValidationConfig:
    """Top level configuration for visual validation runs.

    Attributes are grouped into four blocks:

    * Model paths / encoder selection.
    * Video / camera intrinsics describing the *generated* video.
    * Inference-time parameters controlling sampling and device placement.
    * Validation thresholds used by the check modules to decide PASS/FAIL.
    """

    # ------------------------------------------------------------------ models
    depth_model_path: Path = field(default_factory=lambda: _DEFAULT_DEPTH_MODEL_PATH)
    """OmniEcho processing component."""

    depth_encoder: str = "vitl"
    """One of ``vits``/``vitb``/``vitl`` – must match the loaded weights."""

    sam3_model_path: Path = field(default_factory=lambda: _DEFAULT_SAM3_MODEL_PATH)
    """Filesystem path to the SAM3 weights (.pt)."""

    # ------------------------------------------------------------------- video
    fov_horizontal_deg: float = 90.0
    """Horizontal field-of-view of the rendered camera, in degrees."""

    frame_width: int = 1280
    """Width of generated video frames, in pixels."""

    frame_height: int = 720
    """Height of generated video frames, in pixels."""

    video_fps: int = 24
    """Frame-rate of the source video."""

    # --------------------------------------------------------------- inference
    sample_fps: float = 4.0
    """How many frames per second to actually run the heavy models on."""

    sam3_confidence: float = 0.25
    """Minimum confidence/score for a SAM3 detection to be kept."""

    device: str = "cuda"
    """Torch device string, e.g. ``cuda``, ``cuda:0`` or ``cpu``."""

    # -------------------------------------------------------------- thresholds
    azimuth_tolerance_deg: float = 15.0
    """Maximum allowed azimuth error between projected and detected centre."""

    elevation_tolerance_deg: float = 10.0
    """Maximum allowed elevation error between projected and detected centre."""

    depth_correlation_threshold: float = 0.6
    """Minimum Pearson correlation between expected and observed depth ranks."""

    visibility_frame_tolerance: int = 10
    """Maximum allowed enter/exit-frame mismatch (in frames)."""


    use_video_tracking: bool = True
    """OmniEcho processing component."""

    video_tracker_conf: float = 0.50
    """OmniEcho processing component."""

    min_masklet_length: int = 3
    """OmniEcho processing component."""

    min_track_coverage: float = 0.15
    """OmniEcho processing component."""

    # ----------------------------------------------------------------- helpers
    def __post_init__(self) -> None:
        """Coerce path-like fields to :class:`pathlib.Path`."""

        self.depth_model_path = Path(self.depth_model_path)
        self.sam3_model_path = Path(self.sam3_model_path)
        if self.depth_encoder not in {"vits", "vitb", "vitl"}:
            raise ValueError(
                f"depth_encoder must be one of vits/vitb/vitl, got {self.depth_encoder!r}"
            )

    @classmethod
    def default(cls) -> "ValidationConfig":
        """Return a configuration populated with project-wide defaults."""

        return cls()
