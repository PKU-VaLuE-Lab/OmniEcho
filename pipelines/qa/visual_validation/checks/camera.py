"""OmniEcho processing component."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover - cv2 is a hard dependency
    cv2 = None  # type: ignore[assignment]
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None

from ..projection import interpolate_camera_yaw


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CameraResult:
    """Outcome of the camera-motion check."""

    metadata_has_rotation: bool
    motion_detected: bool
    yaw_correlation: float
    max_yaw_error_deg: float
    estimated_total_yaw: float
    metadata_total_yaw: float
    status: str
    estimated_yaw_curve: list[dict] = field(default_factory=list)
    metadata_yaw_curve: list[dict] = field(default_factory=list)
    segment_correlation: float = float("nan")
    max_segment_error_deg: float = float("nan")
    segments: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Return Pearson r between two equal-length arrays.

    Returns ``nan`` when either input is constant or shorter than 2 –
    matches NumPy's ``corrcoef`` behaviour but avoids a noisy warning.
    """

    if a.size != b.size or a.size < 2:
        return float("nan")
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _segment_based_comparison(
    frame_records: list[dict],
    fps: int,
    segment_duration: float = 1.0,
) -> tuple[float, float, list[dict]]:
    """Compare per-segment recovered yaw deltas against metadata.

    ``frame_records`` is a list of ``{"frame", "t", "delta_yaw", "meta_yaw"}``
    entries produced by :func:`check_camera_motion`'s main loop.  We split
    them into time-buckets of ``segment_duration`` seconds, sum the optical
    flow ``delta_yaw`` inside each bucket, and compare against the metadata
    yaw delta over the same time window (``meta_yaw`` at the bucket end
    minus ``meta_yaw`` at the bucket start).  This avoids the unbounded
    drift of the global cumulative integration.

    Returns ``(correlation, max_segment_error_deg, segments)``.
    """

    if not frame_records or fps <= 0:
        return float("nan"), float("nan"), []

    # Bucket records by floor(t / segment_duration).
    buckets: dict[int, list[dict]] = {}
    for rec in frame_records:
        t = float(rec.get("t", 0.0))
        key = int(t // float(segment_duration))
        buckets.setdefault(key, []).append(rec)

    segments: list[dict] = []
    for key in sorted(buckets):
        items = buckets[key]
        if not items:
            continue
        est_delta = float(
            sum(float(it.get("delta_yaw", 0.0)) for it in items)
        )
        meta_yaws = [float(it.get("meta_yaw", 0.0)) for it in items]
        # Metadata delta over the segment: end - start.  When only a
        # single sample falls into the bucket the delta is 0.
        meta_delta = meta_yaws[-1] - meta_yaws[0]
        segments.append(
            {
                "segment_idx": int(key),
                "t_start": float(key * segment_duration),
                "t_end": float((key + 1) * segment_duration),
                "n_samples": len(items),
                "estimated_delta_deg": est_delta,
                "metadata_delta_deg": meta_delta,
                "abs_error_deg": float(abs(est_delta - meta_delta)),
            }
        )

    if not segments:
        return float("nan"), float("nan"), []

    est = np.asarray([s["estimated_delta_deg"] for s in segments], dtype=float)
    meta = np.asarray([s["metadata_delta_deg"] for s in segments], dtype=float)
    correlation = _pearson(est, meta)
    max_err = float(np.max(np.abs(est - meta))) if est.size else float("nan")
    return correlation, max_err, segments


def _classify_camera(
    metadata_has_rotation: bool,
    motion_detected: bool,
    correlation: float,
    max_error_deg: float,
) -> str:
    """Decide PASS/WARN/FAIL given the diagnostic numbers."""

    # No rotation expected and none detected → trivially PASS.
    if not metadata_has_rotation and not motion_detected:
        return "PASS"
    # Metadata claims a static camera but we recovered noticeable motion.
    if not metadata_has_rotation and motion_detected:
        return "WARN" if max_error_deg < 10.0 else "FAIL"
    # Metadata claims rotation but we did not recover any.
    if metadata_has_rotation and not motion_detected:
        return "FAIL"
    # Both sides agree there is motion – use correlation + error.
    if np.isnan(correlation):
        return "FAIL"
    if correlation >= 0.8 and max_error_deg <= 10.0:
        return "PASS"
    if correlation >= 0.5 and max_error_deg <= 25.0:
        return "WARN"
    return "FAIL"


def _central_crop_gray(frame: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Return a downscaled grayscale central crop of ``frame``.

    The central 60% of width and height is kept – this avoids the strong
    radial distortion at the FOV edges and reduces the influence of objects
    that traverse the frame near the borders.
    """

    h, w = frame.shape[:2]
    cx0 = int(w * 0.2)
    cx1 = int(w * 0.8)
    cy0 = int(h * 0.2)
    cy1 = int(h * 0.8)
    crop = frame[cy0:cy1, cx0:cx1]
    if scale != 1.0:
        new_w = max(1, int(crop.shape[1] * scale))
        new_h = max(1, int(crop.shape[0] * scale))
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return crop


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_camera_motion(
    video_path: str,
    camera_trajectory: list[dict],
    fps: int = 24,
    fov_h_deg: float = 90.0,
    frame_width: int = 1280,
    sample_stride: int = 2,
    motion_threshold_deg: float = 1.0,
) -> CameraResult:
    """Validate that the rendered camera motion matches the metadata.

    Parameters
    ----------
    video_path:
        Path to the rendered video.
    camera_trajectory:
        ``listener.orientation.trajectory`` – list of ``{"t", "yaw"}``.
    fps:
        Frame rate of the rendered video.
    fov_h_deg:
        Horizontal field of view (degrees).
    frame_width:
        Width of the rendered frames (pixels) – used by the pixels→degrees
        conversion.
    sample_stride:
        Compute optical flow between every ``sample_stride``-th frame.
        Defaults to ``2`` (every other frame) which is plenty for the
        smooth pans the renderer emits.
    motion_threshold_deg:
        Total recovered |yaw| above which we say "motion detected".

    Returns
    -------
    A populated :class:`CameraResult`.
    """

    if cv2 is None:  # pragma: no cover
        raise RuntimeError(
            f"cv2 is required for camera motion estimation: {_CV2_IMPORT_ERROR}"
        )
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps!r}")
    if frame_width <= 0:
        raise ValueError(f"frame_width must be positive, got {frame_width!r}")
    sample_stride = max(1, int(sample_stride))

    video_path_obj = Path(video_path)
    if not video_path_obj.is_file():
        raise FileNotFoundError(f"video not found: {video_path_obj}")

    yaw_traj = (
        camera_trajectory if camera_trajectory else [{"t": 0.0, "yaw": 0.0}]
    )
    metadata_yaw_values = [float(kf.get("yaw", 0.0)) for kf in yaw_traj]
    metadata_total_yaw = float(
        max(metadata_yaw_values) - min(metadata_yaw_values)
    ) if metadata_yaw_values else 0.0
    metadata_has_rotation = metadata_total_yaw > motion_threshold_deg

    cap = cv2.VideoCapture(str(video_path_obj))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path_obj}")

    notes: list[str] = []
    estimated_curve: list[dict] = []
    metadata_curve: list[dict] = []
    frame_records: list[dict] = []

    try:
        prev_gray: Optional[np.ndarray] = None
        prev_idx = 0
        cumulative_yaw = 0.0
        idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % sample_stride != 0:
                idx += 1
                continue

            gray = _central_crop_gray(frame)

            if prev_gray is None:
                # Anchor at yaw 0 at the first sampled frame.
                t_sec = float(idx) / float(fps)
                meta_yaw = float(interpolate_camera_yaw(yaw_traj, t_sec))
                estimated_curve.append(
                    {"frame": idx, "t": t_sec, "yaw": 0.0}
                )
                metadata_curve.append(
                    {"frame": idx, "t": t_sec, "yaw": meta_yaw}
                )
                prev_gray = gray
                prev_idx = idx
                idx += 1
                continue

            if gray.shape != prev_gray.shape:
                # Defensive – frame size changed mid-clip.  Skip the pair.
                prev_gray = gray
                prev_idx = idx
                idx += 1
                continue

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=21,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            # Median horizontal flow on the cropped frame – needs to be
            # rescaled to the original frame width for the angle conversion.
            median_flow_x_crop = float(np.median(flow[..., 0]))
            crop_w = prev_gray.shape[1]
            # The crop covered 60% of the original width and was further
            # rescaled by 0.5 inside ``_central_crop_gray``; the conversion
            # below maps a pixel of motion in the crop to an angle in the
            # original FOV.
            scale_factor = (frame_width * 0.6 * 0.5) / crop_w  # ≈1.0
            median_flow_full_x = median_flow_x_crop * scale_factor
            # The OmniEcho renderer pans such that yaw increasing makes
            # the rendered scene drift to the right (positive ``flow_x``).
            # Therefore the recovered Δyaw shares the sign of ``flow_x`` –
            # no extra inversion is needed (this matched against the
            # metadata yaw of omniecho_qa_001 which goes 0→55 over the clip).
            delta_yaw = median_flow_full_x * fov_h_deg / frame_width
            cumulative_yaw += delta_yaw

            t_sec = float(idx) / float(fps)
            meta_yaw = float(interpolate_camera_yaw(yaw_traj, t_sec))
            estimated_curve.append(
                {"frame": idx, "t": t_sec, "yaw": float(cumulative_yaw)}
            )
            metadata_curve.append(
                {"frame": idx, "t": t_sec, "yaw": meta_yaw}
            )
            frame_records.append(
                {
                    "frame": int(idx),
                    "t": float(t_sec),
                    "delta_yaw": float(delta_yaw),
                    "meta_yaw": float(meta_yaw),
                }
            )

            prev_gray = gray
            prev_idx = idx
            idx += 1
    finally:
        cap.release()

    if not estimated_curve:
        notes.append("no frames decoded; cannot estimate camera motion")
        return CameraResult(
            metadata_has_rotation=metadata_has_rotation,
            motion_detected=False,
            yaw_correlation=float("nan"),
            max_yaw_error_deg=float("nan"),
            estimated_total_yaw=0.0,
            metadata_total_yaw=metadata_total_yaw,
            status="FAIL",
            estimated_yaw_curve=[],
            metadata_yaw_curve=[],
            notes=notes,
        )

    est_yaws = np.asarray([e["yaw"] for e in estimated_curve], dtype=float)
    meta_yaws = np.asarray([e["yaw"] for e in metadata_curve], dtype=float)
    estimated_total_yaw = float(est_yaws.max() - est_yaws.min())
    motion_detected = estimated_total_yaw > motion_threshold_deg

    correlation = _pearson(est_yaws, meta_yaws)

    # Re-anchor the metadata curve to the same starting yaw (0) before
    # measuring per-frame errors – we only care about *relative* motion.
    meta_yaws_centered = meta_yaws - meta_yaws[0]
    max_yaw_error = float(np.max(np.abs(est_yaws - meta_yaws_centered)))

    # Segment-based comparison – less prone to integration drift than the
    # global cumulative metric above.  We compare the *delta* yaw within
    # each 1-second window; max_segment_error then bounds the worst-case
    # mismatch independently of how long the clip is.
    segment_correlation, max_segment_error, segments = _segment_based_comparison(
        frame_records, fps=fps, segment_duration=1.0,
    )

    # Use the *better* of the cumulative and the segment-based metrics for
    # the verdict.  The segment-based pair is far less prone to drift on
    # long clips, but the cumulative pair stays useful when the metadata
    # trajectory is too sparse to bucket reliably.
    seg_ok = (
        not np.isnan(segment_correlation) and not np.isnan(max_segment_error)
    )
    if seg_ok:
        eff_correlation = max(
            segment_correlation,
            correlation if not np.isnan(correlation) else float("-inf"),
        )
        eff_max_error = min(max_yaw_error, max_segment_error)
    else:
        eff_correlation = correlation
        eff_max_error = max_yaw_error

    status = _classify_camera(
        metadata_has_rotation,
        motion_detected,
        eff_correlation,
        eff_max_error,
    )

    return CameraResult(
        metadata_has_rotation=metadata_has_rotation,
        motion_detected=motion_detected,
        yaw_correlation=correlation,
        max_yaw_error_deg=max_yaw_error,
        estimated_total_yaw=estimated_total_yaw,
        metadata_total_yaw=metadata_total_yaw,
        status=status,
        estimated_yaw_curve=estimated_curve,
        metadata_yaw_curve=metadata_curve,
        segment_correlation=float(segment_correlation),
        max_segment_error_deg=float(max_segment_error),
        segments=segments,
        notes=notes,
    )


__all__ = ["CameraResult", "check_camera_motion"]
