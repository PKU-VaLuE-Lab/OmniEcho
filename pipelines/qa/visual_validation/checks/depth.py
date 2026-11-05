"""OmniEcho processing component."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

try:  # SciPy is part of the project's hard dependencies but guard anyway.
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover - exercised only when scipy is missing
    spearmanr = None  # type: ignore[assignment]

from ..models.segmentor import FrameDetection
from ..projection import interpolate_trajectory


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class DepthResult:
    """Aggregate depth-consistency outcome.

    Attributes
    ----------
    mean_rank_correlation:
        Average Spearman ρ across all frames where at least two objects were
        detected and ranks could be computed.  ``nan`` when no such frame
        exists.
    min_rank_correlation:
        Worst (lowest) per-frame Spearman ρ, useful to flag isolated bad
        frames even when the average looks healthy.
    per_object_temporal:
        ``object_id -> ρ`` between depth time series and metadata distance.
    outlier_frames:
        Frame indices whose per-frame ρ is below
        ``correlation_threshold * 0.5`` (i.e. severely inverted).
    per_frame_correlations:
        Diagnostic list of ``{"frame": int, "rho": float, "n_objects": int}``
        – kept for the report layer to draw a curve.
    status:
        ``"PASS"`` / ``"WARN"`` / ``"FAIL"`` based on
        :func:`_classify_depth`.
    """

    mean_rank_correlation: float
    min_rank_correlation: float
    per_object_temporal: dict[str, float]
    outlier_frames: list[int]
    status: str
    per_frame_correlations: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spearman(a: Iterable[float], b: Iterable[float]) -> float:
    """Return Spearman ρ between two equal-length sequences.

    Handles the edge cases SciPy is finicky about (constant inputs, length
    < 2) by returning ``nan`` so callers can filter them out.
    """

    arr_a = np.asarray(list(a), dtype=float)
    arr_b = np.asarray(list(b), dtype=float)
    if arr_a.shape != arr_b.shape or arr_a.size < 2:
        return float("nan")
    if np.std(arr_a) == 0 or np.std(arr_b) == 0:
        return float("nan")
    if spearmanr is None:  # pragma: no cover
        # Manual rank-based correlation fallback.
        ra = arr_a.argsort().argsort().astype(float)
        rb = arr_b.argsort().argsort().astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])
    rho, _ = spearmanr(arr_a, arr_b)
    return float(rho) if rho is not None and not np.isnan(rho) else float("nan")


def _depth_in_region(
    depth_map: np.ndarray,
    detection: FrameDetection,
) -> Optional[float]:
    """Return the median depth inside ``detection``'s mask (or bbox).

    Falls back to the bounding box when no mask is available.  Returns
    ``None`` when the region has no valid pixels (out of frame, empty mask).
    """

    if depth_map is None or depth_map.size == 0:
        return None
    h, w = depth_map.shape[:2]

    if detection.mask is not None and detection.mask.size > 0:
        mask = detection.mask
        # Guard against shape mismatch by cropping/padding lazily.
        if mask.shape[:2] != (h, w):
            mh, mw = mask.shape[:2]
            mh = min(mh, h)
            mw = min(mw, w)
            mask_crop = np.zeros((h, w), dtype=mask.dtype)
            mask_crop[:mh, :mw] = mask[:mh, :mw]
            mask = mask_crop
        values = depth_map[mask.astype(bool)]
        if values.size > 0:
            return float(np.median(values))

    x1, y1, x2, y2 = detection.bbox
    x1 = max(0, min(int(x1), w - 1))
    x2 = max(0, min(int(x2), w))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(0, min(int(y2), h))
    if x2 <= x1 or y2 <= y1:
        return None
    region = depth_map[y1:y2, x1:x2]
    if region.size == 0:
        return None
    return float(np.median(region))


def _detection_at_frame(
    dets: list[FrameDetection],
    frame_idx: int,
) -> Optional[FrameDetection]:
    """Return the detection in ``dets`` whose ``frame_idx`` matches.

    SAM3 sampling and depth sampling use the same frame index list, so an
    exact equality lookup is appropriate here.
    """

    for d in dets:
        if int(d.frame_idx) == int(frame_idx):
            return d
    return None


def _classify_depth(
    mean_rho: float,
    min_rho: float,
    threshold: float,
) -> str:
    """Map the (mean, min) pair to PASS/WARN/FAIL."""

    if np.isnan(mean_rho):
        return "FAIL"
    if mean_rho >= threshold and min_rho >= threshold * 0.5:
        return "PASS"
    if mean_rho >= threshold * 0.5:
        return "WARN"
    return "FAIL"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_depth_consistency(
    depth_maps: list[np.ndarray],
    frame_indices: list[int],
    detections: dict[str, list[FrameDetection]],
    objects: list[dict],
    fps: int = 24,
    correlation_threshold: float = 0.6,
) -> DepthResult:
    """Validate that depth ranking matches the metadata distance ranking.

    Parameters
    ----------
    depth_maps:
        List of ``H x W`` depth maps, one per sampled frame.  Larger values
        must mean "further away" (the convention used by
        :class:`visual_validation.models.depth_estimator.DepthEstimator`).
    frame_indices:
        Frame index in the original video for each entry of ``depth_maps``.
        Must have the same length as ``depth_maps``.
    detections:
        ``label -> [FrameDetection, ...]`` from the segmentor.
    objects:
        ``scene.json``'s ``objects`` list.  Each must define
        ``id`` / ``label`` / ``trajectory``.
    fps:
        Source video FPS – used to convert frame indices into seconds for
        trajectory interpolation.
    correlation_threshold:
        Minimum Spearman ρ that counts as a good match.

    Returns
    -------
    A populated :class:`DepthResult`.
    """

    if len(depth_maps) != len(frame_indices):
        raise ValueError(
            f"depth_maps and frame_indices length mismatch: "
            f"{len(depth_maps)} vs {len(frame_indices)}"
        )
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps!r}")

    notes: list[str] = []
    obj_by_id = {str(o.get("id", "")): o for o in objects}
    label_to_id = {str(o.get("label", "")): str(o.get("id", "")) for o in objects}

    # ----- Per-frame cross-object Spearman ρ -----
    per_frame: list[dict] = []
    valid_rhos: list[float] = []
    # Track depth/distance time series per object for the temporal pass.
    obj_depth_series: dict[str, list[tuple[int, float, float]]] = {
        str(o.get("id", "")): [] for o in objects
    }

    for depth_map, fidx in zip(depth_maps, frame_indices):
        depths: list[float] = []
        distances: list[float] = []
        per_frame_objs: list[str] = []
        t_sec = float(fidx) / float(fps)

        for obj in objects:
            obj_id = str(obj.get("id", ""))
            label = str(obj.get("label", obj_id))
            traj = obj.get("trajectory") or []
            if not traj:
                continue
            det = _detection_at_frame(detections.get(label, []), fidx)
            if det is None:
                continue
            d_med = _depth_in_region(depth_map, det)
            if d_med is None:
                continue
            interp = interpolate_trajectory(traj, t_sec)
            distance = float(interp.get("distance", float("nan")))
            if np.isnan(distance):
                continue
            depths.append(d_med)
            distances.append(distance)
            per_frame_objs.append(obj_id)
            obj_depth_series[obj_id].append((int(fidx), d_med, distance))

        if len(depths) >= 2:
            rho = _spearman(depths, distances)
            per_frame.append(
                {"frame": int(fidx), "rho": float(rho), "n_objects": len(depths)}
            )
            if not np.isnan(rho):
                valid_rhos.append(rho)
        else:
            per_frame.append(
                {"frame": int(fidx), "rho": float("nan"), "n_objects": len(depths)}
            )

    if valid_rhos:
        mean_rho = float(np.mean(valid_rhos))
        min_rho = float(np.min(valid_rhos))
    else:
        notes.append(
            "no frame had >=2 detected objects; cross-object correlation undefined"
        )
        mean_rho = float("nan")
        min_rho = float("nan")

    # ----- Per-object temporal correlation -----
    per_object_temporal: dict[str, float] = {}
    for obj_id, series in obj_depth_series.items():
        if len(series) < 2:
            per_object_temporal[obj_id] = float("nan")
            continue
        depths = [s[1] for s in series]
        distances = [s[2] for s in series]
        per_object_temporal[obj_id] = _spearman(depths, distances)

    # ----- Outlier frames -----
    outlier_threshold = correlation_threshold * 0.5
    outlier_frames = [
        int(entry["frame"])
        for entry in per_frame
        if not np.isnan(entry["rho"]) and entry["rho"] < outlier_threshold
    ]

    status = _classify_depth(mean_rho, min_rho, correlation_threshold)

    # ``label_to_id`` is intentionally kept – downstream code may want it but
    # we expose it via notes to avoid widening the public dataclass.
    if not label_to_id:
        notes.append("no objects with labels in scene.json")

    return DepthResult(
        mean_rank_correlation=mean_rho,
        min_rank_correlation=min_rho,
        per_object_temporal=per_object_temporal,
        outlier_frames=outlier_frames,
        status=status,
        per_frame_correlations=per_frame,
        notes=notes,
    )


__all__ = ["DepthResult", "check_depth_consistency"]
