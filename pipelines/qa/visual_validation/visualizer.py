"""OmniEcho processing component."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .config import ValidationConfig
from .models.segmentor import FrameDetection
from .projection import (
    azimuth_elevation_to_pixel,
    interpolate_camera_yaw,
    interpolate_trajectory,
)


logger = logging.getLogger(__name__)


# Distinct BGR colours assigned cyclically to objects.  Bright, saturated
# tones – they remain legible on top of the jet depth heat map.
OBJECT_COLORS: List[tuple[int, int, int]] = [
    (0, 255, 0),      # green
    (255, 0, 0),      # blue (BGR)
    (0, 0, 255),      # red
    (255, 255, 0),    # cyan
    (255, 0, 255),    # magenta
    (0, 255, 255),    # yellow
]


def _normalize_to_uint8(depth_map: np.ndarray) -> np.ndarray:
    """Min-max normalise ``depth_map`` to ``uint8`` for colormap application."""

    arr = np.asarray(depth_map, dtype=np.float32)
    if arr.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo = float(finite.min())
    hi = float(finite.max())
    if hi - lo < 1e-8:
        return np.zeros(arr.shape, dtype=np.uint8)
    norm = (arr - lo) / (hi - lo)
    norm = np.clip(norm, 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def _depth_to_color(depth_map: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Return a BGR colour heat map sized to ``target_shape`` (H, W)."""

    u8 = _normalize_to_uint8(depth_map)
    if u8.shape[:2] != target_shape:
        u8 = cv2.resize(
            u8, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR
        )
    return cv2.applyColorMap(u8, cv2.COLORMAP_JET)


def _nearest_index(values: List[int], target: int) -> Optional[int]:
    """Return the index in ``values`` whose entry is closest to ``target``.

    ``None`` is returned when ``values`` is empty.
    """

    if not values:
        return None
    arr = np.asarray(values, dtype=np.int64)
    return int(np.argmin(np.abs(arr - int(target))))


def _build_detection_lookup(
    detections: Dict[str, List[FrameDetection]],
) -> Dict[str, Dict[int, FrameDetection]]:
    """Pre-index detections by ``label -> {frame_idx: FrameDetection}``."""

    out: Dict[str, Dict[int, FrameDetection]] = {}
    for label, dets in (detections or {}).items():
        out[label] = {int(d.frame_idx): d for d in dets}
    return out


class Visualizer:
    """Annotation helper for the visual validation pipeline."""

    def __init__(self, config: ValidationConfig) -> None:
        self.config = config
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_thickness = 1

    # ---------------------------------------------------------------- helpers
    def _color_for(self, idx: int) -> tuple[int, int, int]:
        return OBJECT_COLORS[idx % len(OBJECT_COLORS)]

    def _resolve_label(self, obj: dict) -> str:
        """Return the SAM3 prompt label used as the detection key.

        The pipeline uses the object's ``type`` (with underscores replaced by
        spaces) as the SAM3 prompt by default; ``label`` is kept as a fallback
        for objects that do not carry a ``type`` field.
        """

        t = obj.get("type")
        if t:
            return str(t).replace("_", " ")
        return str(obj.get("label", obj.get("id", "")))

    def _draw_object_overlay(
        self,
        canvas: np.ndarray,
        obj: dict,
        color: tuple[int, int, int],
        detection: Optional[FrameDetection],
        expected_xy: Optional[tuple[float, float]],
    ) -> None:
        """Overlay a single object's annotations onto ``canvas``."""

        h, w = canvas.shape[:2]
        label_text = str(obj.get("label", obj.get("id", "")))

        # Draw mask (if provided) as a filled translucent layer.
        if detection is not None and detection.mask is not None:
            mask = detection.mask
            if mask.shape[:2] != (h, w):
                mh = min(mask.shape[0], h)
                mw = min(mask.shape[1], w)
                fitted = np.zeros((h, w), dtype=mask.dtype)
                fitted[:mh, :mw] = mask[:mh, :mw]
                mask = fitted
            mask_bool = mask.astype(bool)
            if mask_bool.any():
                tinted = canvas.copy()
                tinted[mask_bool] = (
                    0.5 * tinted[mask_bool].astype(np.float32)
                    + 0.5 * np.array(color, dtype=np.float32)
                ).astype(np.uint8)
                canvas[:] = tinted

        # Draw bounding box and observed centre.
        observed_xy: Optional[tuple[float, float]] = None
        if detection is not None:
            x1, y1, x2, y2 = detection.bbox
            x1 = max(0, min(int(x1), w - 1))
            y1 = max(0, min(int(y1), h - 1))
            x2 = max(0, min(int(x2), w - 1))
            y2 = max(0, min(int(y2), h - 1))
            if x2 > x1 and y2 > y1:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                cx, cy = detection.center
                observed_xy = (float(cx), float(cy))
                cv2.circle(canvas, (int(cx), int(cy)), 5, color, -1)
                cv2.putText(
                    canvas,
                    f"{label_text} {detection.score:.2f}",
                    (x1, max(0, y1 - 6)),
                    self.font,
                    self.font_scale,
                    color,
                    self.font_thickness,
                    cv2.LINE_AA,
                )

        # Expected position marker (×) plus name.
        if expected_xy is not None:
            ex, ey = expected_xy
            if 0 <= ex < w and 0 <= ey < h:
                ex_i, ey_i = int(ex), int(ey)
                size = 8
                cv2.line(
                    canvas,
                    (ex_i - size, ey_i - size),
                    (ex_i + size, ey_i + size),
                    color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.line(
                    canvas,
                    (ex_i - size, ey_i + size),
                    (ex_i + size, ey_i - size),
                    color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    label_text,
                    (ex_i + size + 2, ey_i - size),
                    self.font,
                    self.font_scale,
                    color,
                    self.font_thickness,
                    cv2.LINE_AA,
                )

        # Connector line between expected and observed positions.
        if expected_xy is not None and observed_xy is not None:
            ex, ey = expected_xy
            ox, oy = observed_xy
            if 0 <= ex < w and 0 <= ey < h:
                cv2.line(
                    canvas,
                    (int(ex), int(ey)),
                    (int(ox), int(oy)),
                    color,
                    1,
                    cv2.LINE_AA,
                )

    # ---------------------------------------------------------------- public
    def annotate_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        depth_map: Optional[np.ndarray],
        detections: Dict[str, Optional[FrameDetection]],
        objects: List[dict],
        camera_trajectory: List[dict],
        fps: int = 24,
    ) -> np.ndarray:
        """OmniEcho processing component."""

        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # Depth heat-map overlay (alpha=0.3).
        if depth_map is not None and depth_map.size > 0:
            heat = _depth_to_color(depth_map, (h, w))
            canvas = cv2.addWeighted(heat, 0.3, canvas, 0.7, 0.0)

        # Per-object overlays.
        t_sec = float(frame_idx) / float(max(fps, 1))
        yaw_traj = camera_trajectory if camera_trajectory else [{"t": 0.0, "yaw": 0.0}]
        cam_yaw = float(interpolate_camera_yaw(yaw_traj, t_sec))

        for i, obj in enumerate(objects):
            color = self._color_for(i)
            obj_id = str(obj.get("id", ""))
            traj = obj.get("trajectory") or []
            expected_xy: Optional[tuple[float, float]] = None
            if traj:
                interp = interpolate_trajectory(traj, t_sec)
                expected_xy = azimuth_elevation_to_pixel(
                    float(interp["azimuth"]),
                    float(interp["elevation"]),
                    cam_yaw,
                    self.config.fov_horizontal_deg,
                    self.config.frame_width,
                    self.config.frame_height,
                )

            self._draw_object_overlay(
                canvas,
                obj,
                color,
                detections.get(obj_id),
                expected_xy,
            )

        # Header: frame index + time.
        header = f"frame {frame_idx} | t={t_sec:.2f}s | yaw={cam_yaw:+.1f}deg"
        cv2.rectangle(canvas, (0, 0), (max(220, len(header) * 9), 24), (0, 0, 0), -1)
        cv2.putText(
            canvas,
            header,
            (6, 17),
            self.font,
            self.font_scale,
            (255, 255, 255),
            self.font_thickness,
            cv2.LINE_AA,
        )
        return canvas

    # ------------------------------------------------------------------ video
    def _detections_for_frame(
        self,
        objects: List[dict],
        det_lookup: Dict[str, Dict[int, FrameDetection]],
        sampled_frames: List[int],
        frame_idx: int,
    ) -> Dict[str, Optional[FrameDetection]]:
        """Resolve the detection (if any) for every object at ``frame_idx``."""

        nearest_pos = _nearest_index(sampled_frames, frame_idx)
        nearest_frame = (
            sampled_frames[nearest_pos] if nearest_pos is not None else None
        )

        out: Dict[str, Optional[FrameDetection]] = {}
        for obj in objects:
            obj_id = str(obj.get("id", ""))
            label = self._resolve_label(obj)
            label_dets = det_lookup.get(label, {})
            chosen: Optional[FrameDetection] = label_dets.get(int(frame_idx))
            if chosen is None and nearest_frame is not None:
                chosen = label_dets.get(int(nearest_frame))
            out[obj_id] = chosen
        return out

    def generate_annotated_video(
        self,
        video_path: Path,
        output_path: Path,
        depth_maps: List[np.ndarray],
        depth_frame_indices: List[int],
        detections: Dict[str, List[FrameDetection]],
        objects: List[dict],
        camera_trajectory: List[dict],
        fps: int = 24,
    ) -> None:
        """Write a fully annotated MP4 to ``output_path``."""

        video_path = Path(video_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not video_path.is_file():
            raise FileNotFoundError(f"source video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"failed to open video: {video_path}")

        try:
            src_fps = cap.get(cv2.CAP_PROP_FPS) or float(fps)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self.config.frame_width)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.config.frame_height)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(output_path), fourcc, float(src_fps), (width, height)
            )
            if not writer.isOpened():  # pragma: no cover - codec/permissions issue
                raise RuntimeError(
                    f"failed to open video writer: {output_path} "
                    f"(codec=mp4v, fps={src_fps}, size={width}x{height})"
                )

            det_lookup = _build_detection_lookup(detections)
            sampled_sorted = sorted(int(i) for i in depth_frame_indices) if depth_frame_indices else []

            try:
                idx = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    depth_map: Optional[np.ndarray] = None
                    if sampled_sorted:
                        nearest_pos = _nearest_index(sampled_sorted, idx)
                        if nearest_pos is not None:
                            # Map back to the depth_maps list in original order.
                            target_frame = sampled_sorted[nearest_pos]
                            if target_frame in depth_frame_indices:
                                original_idx = depth_frame_indices.index(target_frame)
                                if original_idx < len(depth_maps):
                                    depth_map = depth_maps[original_idx]

                    frame_dets = self._detections_for_frame(
                        objects, det_lookup, sampled_sorted, idx
                    )
                    annotated = self.annotate_frame(
                        frame=frame,
                        frame_idx=idx,
                        depth_map=depth_map,
                        detections=frame_dets,
                        objects=objects,
                        camera_trajectory=camera_trajectory,
                        fps=int(round(src_fps)),
                    )
                    writer.write(annotated)
                    idx += 1

                logger.info(
                    "annotated video: %d frames written to %s", idx, output_path
                )
            finally:
                writer.release()
        finally:
            cap.release()

    # ------------------------------------------------------------ keyframes
    def save_keyframes(
        self,
        video_path: Path,
        output_dir: Path,
        depth_maps: List[np.ndarray],
        depth_frame_indices: List[int],
        detections: Dict[str, List[FrameDetection]],
        objects: List[dict],
        camera_trajectory: List[dict],
        fps: int = 24,
        n_keyframes: int = 6,
    ) -> List[Path]:
        """Save ``n_keyframes`` evenly spaced annotated frames as PNGs."""

        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"failed to open video: {video_path}")

        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            src_fps = cap.get(cv2.CAP_PROP_FPS) or float(fps)
            if total <= 0:
                logger.warning("video %s reports zero frames; skipping keyframes", video_path)
                return []

            n = max(1, int(n_keyframes))
            if n >= total:
                target_indices = list(range(total))
            else:
                target_indices = [
                    int(round(i * (total - 1) / (n - 1))) if n > 1 else 0
                    for i in range(n)
                ]
            target_set = set(target_indices)

            det_lookup = _build_detection_lookup(detections)
            sampled_sorted = sorted(int(i) for i in depth_frame_indices)

            saved: List[Path] = []
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx in target_set:
                    depth_map: Optional[np.ndarray] = None
                    if sampled_sorted:
                        nearest_pos = _nearest_index(sampled_sorted, idx)
                        if nearest_pos is not None:
                            original_idx = depth_frame_indices.index(
                                sampled_sorted[nearest_pos]
                            )
                            depth_map = depth_maps[original_idx]

                    frame_dets = self._detections_for_frame(
                        objects, det_lookup, sampled_sorted, idx
                    )
                    annotated = self.annotate_frame(
                        frame=frame,
                        frame_idx=idx,
                        depth_map=depth_map,
                        detections=frame_dets,
                        objects=objects,
                        camera_trajectory=camera_trajectory,
                        fps=int(round(src_fps)),
                    )
                    out_path = output_dir / f"frame_{idx:05d}.png"
                    cv2.imwrite(str(out_path), annotated)
                    saved.append(out_path)
                idx += 1
            return saved
        finally:
            cap.release()

    # ------------------------------------------------------------- depth map
    def save_depth_maps(
        self,
        depth_maps: List[np.ndarray],
        depth_frame_indices: List[int],
        output_dir: Path,
    ) -> List[Path]:
        """Persist depth maps as colourised PNG images."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved: List[Path] = []
        for depth_map, fidx in zip(depth_maps, depth_frame_indices):
            if depth_map is None or depth_map.size == 0:
                continue
            u8 = _normalize_to_uint8(depth_map)
            color = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
            out_path = output_dir / f"depth_{int(fidx):05d}.png"
            cv2.imwrite(str(out_path), color)
            saved.append(out_path)
        return saved


__all__ = ["Visualizer", "OBJECT_COLORS"]
