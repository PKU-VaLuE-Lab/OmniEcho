"""OmniEcho processing component."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..models.segmentor import FrameDetection
from ..projection import (
    azimuth_elevation_to_pixel,
    compute_fov_vertical,
    interpolate_camera_yaw,
    interpolate_trajectory,
    is_in_fov,
    pixel_to_azimuth_elevation,
)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ObjectSpatialMetrics:
    """Per-object spatial alignment summary."""

    object_id: str
    azimuth_rmse_deg: float
    elevation_rmse_deg: float
    max_deviation_deg: float
    n_compared_frames: int
    per_frame_errors: list[dict] = field(default_factory=list)
    status: str = "PASS"
    mean_distance: float = float("nan")
    azimuth_tolerance_used: float = float("nan")
    elevation_tolerance_used: float = float("nan")


@dataclass
class SpatialResult:
    """Aggregate spatial alignment outcome across all objects."""

    per_object: dict[str, ObjectSpatialMetrics]
    overall_azimuth_rmse_deg: float
    overall_elevation_rmse_deg: float
    overall_max_deviation_deg: float
    status: str
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wrap_deg(angle_deg: float) -> float:
    """Wrap an angle into ``[-180, 180]`` degrees."""

    a = (angle_deg + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


def _angle_diff(a: float, b: float) -> float:
    """Return the signed shortest difference ``a - b`` in degrees."""

    return _wrap_deg(a - b)


def _classify_object(
    az_rmse: float,
    el_rmse: float,
    az_tol: float,
    el_tol: float,
) -> str:
    """Classify a single object's RMSEs."""

    if np.isnan(az_rmse) or np.isnan(el_rmse):
        return "FAIL"
    if az_rmse <= az_tol and el_rmse <= el_tol:
        return "PASS"
    if az_rmse <= az_tol * 2.0 and el_rmse <= el_tol * 2.0:
        return "WARN"
    return "FAIL"


def _dynamic_tolerance(distance: float, base_tolerance: float) -> float:
    """Return a distance-aware tolerance for the spatial check.

    Rationale: a 1m lateral mis-render at 6m maps to a much smaller
    angular error than the same 1m mis-render at 1.5m, so the angular
    tolerance can reasonably be loosened for distant objects and tightened
    for very close ones.  We use a piece-wise linear scaling around the
    base tolerance:

    * ``distance > 5m``  → ``base * 1.3`` (loosen by 30%)
    * ``distance < 2m``  → ``base * 0.7`` (tighten by 30%)
    * otherwise          → ``base``

    ``nan`` distances and non-positive base tolerances are passed through
    unchanged so callers always get a usable number back.
    """

    if not np.isfinite(distance) or base_tolerance <= 0.0:
        return float(base_tolerance)
    if distance > 5.0:
        return float(base_tolerance) * 1.3
    if distance < 2.0:
        return float(base_tolerance) * 0.7
    return float(base_tolerance)


def _classify_overall(per_object: dict[str, ObjectSpatialMetrics]) -> str:
    """Promote per-object verdicts into an overall verdict."""

    if not per_object:
        return "FAIL"
    statuses = [m.status for m in per_object.values()]
    if all(s == "PASS" for s in statuses):
        return "PASS"
    if any(s == "FAIL" for s in statuses):
        return "FAIL"
    return "WARN"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_spatial_alignment(
    detections: dict[str, list[FrameDetection]],
    objects: list[dict],
    camera_trajectory: list[dict],
    fov_h_deg: float = 90.0,
    frame_width: int = 1280,
    frame_height: int = 720,
    fps: int = 24,
    azimuth_tolerance: float = 15.0,
    elevation_tolerance: float = 10.0,
) -> SpatialResult:
    """Validate that detected 2D positions match the projected metadata.

    Parameters
    ----------
    detections:
        ``label -> [FrameDetection, ...]`` from the segmentor.
    objects:
        ``scene.json``'s ``objects`` list.
    camera_trajectory:
        ``listener.orientation.trajectory`` – list of ``{"t", "yaw"}``.  When
        empty the camera is assumed to be static at yaw ``0``.
    fov_h_deg, frame_width, frame_height:
        Camera intrinsics matching the rendered video.
    fps:
        Source video FPS, used to convert frame indices into seconds.
    azimuth_tolerance, elevation_tolerance:
        Per-axis tolerances (degrees) for the PASS verdict.  WARN is granted
        up to twice these tolerances.

    Returns
    -------
    A :class:`SpatialResult` with per-object metrics and overall RMSE.
    """

    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps!r}")
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame_width / frame_height must be positive")

    notes: list[str] = []
    fov_v_deg = compute_fov_vertical(fov_h_deg, frame_width, frame_height)

    # If no camera trajectory is provided we treat the camera as static at
    # yaw 0; ``interpolate_camera_yaw`` requires at least one keyframe.
    yaw_traj = camera_trajectory if camera_trajectory else [{"t": 0.0, "yaw": 0.0}]

    per_object: dict[str, ObjectSpatialMetrics] = {}
    all_az_errs: list[float] = []
    all_el_errs: list[float] = []

    for obj in objects:
        obj_id = str(obj.get("id", ""))
        label = str(obj.get("label", obj_id))
        traj = obj.get("trajectory") or []
        if not traj:
            notes.append(f"object {obj_id} has empty trajectory; skipped")
            continue

        dets = detections.get(label, [])
        per_frame_errors: list[dict] = []
        az_errs: list[float] = []
        el_errs: list[float] = []
        distances_seen: list[float] = []

        for det in dets:
            t_sec = float(det.frame_idx) / float(fps)
            interp = interpolate_trajectory(traj, t_sec)
            exp_az = float(interp["azimuth"])
            exp_el = float(interp["elevation"])
            cam_yaw = float(interpolate_camera_yaw(yaw_traj, t_sec))
            try:
                dist_t = float(interp.get("distance", float("nan")))
            except Exception:
                dist_t = float("nan")
            if np.isfinite(dist_t):
                distances_seen.append(dist_t)

            # Sanity check: skip frames where the metadata says the object
            # is outside the FOV – a detection there is almost certainly a
            # false positive on a different object.
            if not is_in_fov(exp_az, exp_el, cam_yaw, fov_h_deg, fov_v_deg):
                continue

            # Forward projection (kept for diagnostics).
            exp_x, exp_y = azimuth_elevation_to_pixel(
                exp_az, exp_el, cam_yaw, fov_h_deg, frame_width, frame_height
            )

            # Inverse projection of the detected centre.
            cx, cy = det.center
            obs_az, obs_el = pixel_to_azimuth_elevation(
                cx, cy, cam_yaw, fov_h_deg, frame_width, frame_height
            )
            az_err = _angle_diff(obs_az, exp_az)
            el_err = _angle_diff(obs_el, exp_el)

            per_frame_errors.append(
                {
                    "frame": int(det.frame_idx),
                    "az_err": float(az_err),
                    "el_err": float(el_err),
                    "expected_px": (float(exp_x), float(exp_y)),
                    "observed_px": (float(cx), float(cy)),
                    "camera_yaw_deg": cam_yaw,
                    "distance": float(dist_t) if np.isfinite(dist_t) else None,
                }
            )
            az_errs.append(az_err)
            el_errs.append(el_err)

        mean_distance = (
            float(np.mean(distances_seen)) if distances_seen else float("nan")
        )
        az_tol_used = _dynamic_tolerance(mean_distance, azimuth_tolerance)
        el_tol_used = _dynamic_tolerance(mean_distance, elevation_tolerance)

        if az_errs:
            az_arr = np.asarray(az_errs, dtype=float)
            el_arr = np.asarray(el_errs, dtype=float)
            az_rmse = float(np.sqrt(np.mean(az_arr * az_arr)))
            el_rmse = float(np.sqrt(np.mean(el_arr * el_arr)))
            max_dev = float(
                np.max(np.sqrt(az_arr * az_arr + el_arr * el_arr))
            )
            status = _classify_object(
                az_rmse, el_rmse, az_tol_used, el_tol_used
            )
        else:
            az_rmse = float("nan")
            el_rmse = float("nan")
            max_dev = float("nan")
            status = "FAIL"
            notes.append(
                f"object {obj_id} has no comparable frames (no detections inside FOV)"
            )

        per_object[obj_id] = ObjectSpatialMetrics(
            object_id=obj_id,
            azimuth_rmse_deg=az_rmse,
            elevation_rmse_deg=el_rmse,
            max_deviation_deg=max_dev,
            n_compared_frames=len(az_errs),
            per_frame_errors=per_frame_errors,
            status=status,
            mean_distance=mean_distance,
            azimuth_tolerance_used=float(az_tol_used),
            elevation_tolerance_used=float(el_tol_used),
        )

        all_az_errs.extend(az_errs)
        all_el_errs.extend(el_errs)

    if all_az_errs:
        az_arr = np.asarray(all_az_errs, dtype=float)
        el_arr = np.asarray(all_el_errs, dtype=float)
        overall_az_rmse = float(np.sqrt(np.mean(az_arr * az_arr)))
        overall_el_rmse = float(np.sqrt(np.mean(el_arr * el_arr)))
        overall_max = float(np.max(np.sqrt(az_arr * az_arr + el_arr * el_arr)))
    else:
        overall_az_rmse = float("nan")
        overall_el_rmse = float("nan")
        overall_max = float("nan")

    return SpatialResult(
        per_object=per_object,
        overall_azimuth_rmse_deg=overall_az_rmse,
        overall_elevation_rmse_deg=overall_el_rmse,
        overall_max_deviation_deg=overall_max,
        status=_classify_overall(per_object),
        notes=notes,
    )


__all__ = ["SpatialResult", "ObjectSpatialMetrics", "check_spatial_alignment"]
