"""OmniEcho processing component."""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch


# ``visual_validation/models/`` so we go up two levels to reach

THIRD_PARTY_DIR = Path(os.environ.get("OMNIECHO_DEPTH_SOURCE", "/path/to/Depth-Anything-V2"))


class DepthEstimator:
    """OmniEcho processing component."""

    MODEL_CONFIGS = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
    }

    def __init__(
        self,
        model_path: Union[str, Path],
        encoder: str = "vitl",
        device: str = "cuda",
    ) -> None:
        """OmniEcho processing component."""

        if encoder not in self.MODEL_CONFIGS:
            raise ValueError(
                f"unknown encoder {encoder!r}; expected one of {list(self.MODEL_CONFIGS)}"
            )

        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"depth model weights not found: {model_path}")

        # Inject the bundled source tree so ``from depth_anything_v2.dpt
        # import DepthAnythingV2`` resolves.  We append (rather than prepend)
        # to avoid clobbering an externally installed copy when present.
        third_party = str(THIRD_PARTY_DIR)
        if third_party not in sys.path:
            sys.path.append(third_party)

        from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore  # noqa: E402

        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.encoder = encoder

        config = self.MODEL_CONFIGS[encoder]
        model = DepthAnythingV2(**config)
        state_dict = torch.load(str(model_path), map_location="cpu")
        model.load_state_dict(state_dict)
        self.model = model.to(device).eval()

    # ------------------------------------------------------------------ inference
    @torch.inference_mode()
    def estimate_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """OmniEcho processing component."""

        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must be an HxWx3 BGR image")
        depth = self.model.infer_image(frame_bgr)
        return np.asarray(depth, dtype=np.float32)

    def estimate_video(
        self,
        video_path: Union[str, Path],
        sample_fps: float = 4.0,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Sample ``video_path`` at ``sample_fps`` and run depth inference.

        Returns
        -------
        ``(depth_maps, frame_indices)`` where ``depth_maps[i]`` is the depth
        map for the *original* video frame ``frame_indices[i]``.  The first
        and last frames are always sampled even when ``sample_fps`` does not
        divide evenly.
        """

        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"failed to open video: {video_path}")

        try:
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if src_fps <= 0 or total <= 0:
                raise RuntimeError(
                    f"invalid metadata for video {video_path}: fps={src_fps}, frames={total}"
                )

            step = max(1, int(round(src_fps / float(sample_fps))))
            target_indices: List[int] = list(range(0, total, step))
            if total - 1 not in target_indices:
                target_indices.append(total - 1)
            target_set = set(target_indices)

            depth_maps: List[np.ndarray] = []
            kept_indices: List[int] = []

            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx in target_set:
                    depth_maps.append(self.estimate_frame(frame))
                    kept_indices.append(idx)
                idx += 1

            return depth_maps, kept_indices
        finally:
            cap.release()
