"""OmniEcho processing component."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..models.segmentor import FrameDetection


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class VisibilityResult:
    """Per-object visibility check outcome.

    Attributes
    ----------
    object_id:
        Stable identifier from ``scene.json`` (``objects[i].id``).
    object_label:
        Human-readable label used as the SAM3 text prompt.
    expected_enter_frame, expected_exit_frame:
        Visibility window from metadata, converted to frame indices.
    detected_first_frame, detected_last_frame:
        Inclusive bounds of the frames in which SAM3 actually detected the
        object; ``None`` when the object was never detected.
    detected_frame_count:
        Number of distinct frames the object was detected in.
    temporal_iou:
        ``intersection / union`` of the two ``[enter, exit]`` frame intervals.
    status:
        One of ``"PASS"`` / ``"WARN"`` / ``"FAIL"``.
    """

    object_id: str
    object_label: str
    expected_enter_frame: int
    expected_exit_frame: int
    detected_first_frame: Optional[int]
    detected_last_frame: Optional[int]
    detected_frame_count: int
    temporal_iou: float
    status: str
    is_visual: bool = True
    skipped: bool = False
    detection_label: Optional[str] = None
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _interval_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Return the IoU of two integer ``[start, end]`` intervals (inclusive).

    Empty / inverted intervals (``end < start``) are treated as having zero
    length but still contribute to the union when at least one endpoint is
    valid.  Returns ``0.0`` when both intervals are empty.
    """

    a0, a1 = a
    b0, b1 = b
    inter_start = max(a0, b0)
    inter_end = min(a1, b1)
    inter = max(0, inter_end - inter_start + 1)

    union_start = min(a0, b0)
    union_end = max(a1, b1)
    union = max(0, union_end - union_start + 1)

    if union == 0:
        return 0.0
    return inter / union


def _detected_range(
    dets: list[FrameDetection],
) -> tuple[Optional[int], Optional[int], int]:
    """Return ``(first_frame, last_frame, distinct_count)`` for ``dets``."""

    if not dets:
        return None, None, 0
    frames = sorted({int(d.frame_idx) for d in dets})
    return frames[0], frames[-1], len(frames)


def _classify(iou: float, *, pass_threshold: float = 0.7, warn_threshold: float = 0.4) -> str:
    """Map a temporal IoU value to a PASS/WARN/FAIL label."""

    if iou > pass_threshold:
        return "PASS"
    if iou > warn_threshold:
        return "WARN"
    return "FAIL"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_visibility(
    detections: dict[str, list[FrameDetection]],
    objects: list[dict],
    fps: int = 24,
    total_frames: int = 144,
    frame_tolerance: int = 10,
    visual_flags: Optional[dict[str, bool]] = None,
    detection_labels: Optional[dict[str, str]] = None,
) -> list[VisibilityResult]:
    """Validate the visibility window of every object.

    Parameters
    ----------
    detections:
        Mapping ``label -> [FrameDetection, ...]`` produced by
        :class:`~visual_validation.models.segmentor.VideoSegmentor`.
    objects:
        ``scene.json``'s ``objects`` array.  Each entry must carry an ``id``
        and a ``label``; ``visibility_window`` is optional – objects without
        one are assumed to be visible for the whole clip.
    fps:
        Frame rate of the rendered video.  Used to translate the
        ``visibility_window`` seconds into frame indices.
    total_frames:
        Total frame count of the rendered video.  Used as the upper bound
        when ``video_exit_t`` is missing.
    frame_tolerance:
        Slack (in frames) added on both sides of the expected window before
        computing the IoU – this absorbs sub-second tolerance between SAM3
        sampling and the metadata.

    Returns
    -------
    A list of :class:`VisibilityResult` in the same order as ``objects``.
    """

    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps!r}")
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames!r}")

    visual_flags = visual_flags or {}
    detection_labels = detection_labels or {}

    results: list[VisibilityResult] = []
    last_frame = total_frames - 1

    for obj in objects:
        obj_id = str(obj.get("id", ""))
        label = str(obj.get("label", obj_id))
        notes: list[str] = []
        is_visual = bool(visual_flags.get(obj_id, True))
        det_label = detection_labels.get(obj_id)

        if not is_visual:
            # Non-visual audio source (e.g. footsteps, ambient water).
            # Skip detection-based comparison entirely; record a SKIPPED
            # entry so the report still surfaces the object.
            results.append(
                VisibilityResult(
                    object_id=obj_id,
                    object_label=label,
                    expected_enter_frame=0,
                    expected_exit_frame=last_frame,
                    detected_first_frame=None,
                    detected_last_frame=None,
                    detected_frame_count=0,
                    temporal_iou=0.0,
                    status="SKIPPED",
                    is_visual=False,
                    skipped=True,
                    detection_label=det_label,
                    notes=["non-visual object: skipped detection check"],
                )
            )
            continue

        window = obj.get("visibility_window") or {}
        enter_t = float(window.get("video_enter_t", 0.0))
        exit_t = float(
            window.get("video_exit_t", float(total_frames) / float(fps))
        )

        expected_enter = max(0, int(round(enter_t * fps)))
        expected_exit = min(last_frame, int(round(exit_t * fps)))
        if expected_exit < expected_enter:
            # Degenerate / invalid window – record but keep going.
            notes.append(
                f"invalid metadata window enter={enter_t}s exit={exit_t}s"
            )
            expected_exit = expected_enter

        # Only consider detections that fall inside the rendered video.
        # Look up under the segmentor label first (English prompt) and
        
        det_lookup_keys = [k for k in (det_label, label, obj_id, obj.get("type")) if k]
        raw_dets: list[FrameDetection] = []
        for k in det_lookup_keys:
            cand = detections.get(str(k))
            if cand:
                raw_dets = cand
                break
        dets = [
            d for d in raw_dets if 0 <= int(d.frame_idx) < total_frames
        ]
        first_det, last_det, det_count = _detected_range(dets)

        if first_det is None or last_det is None:
            results.append(
                VisibilityResult(
                    object_id=obj_id,
                    object_label=label,
                    expected_enter_frame=expected_enter,
                    expected_exit_frame=expected_exit,
                    detected_first_frame=None,
                    detected_last_frame=None,
                    detected_frame_count=0,
                    temporal_iou=0.0,
                    status="FAIL",
                    is_visual=True,
                    skipped=False,
                    detection_label=det_label,
                    notes=notes + ["no detections for this label"],
                )
            )
            continue

        # Apply tolerance symmetrically around the expected window.
        tol = max(0, int(frame_tolerance))
        expanded = (
            max(0, expected_enter - tol),
            min(last_frame, expected_exit + tol),
        )
        observed = (first_det, last_det)
        iou = _interval_iou(expanded, observed)

        results.append(
            VisibilityResult(
                object_id=obj_id,
                object_label=label,
                expected_enter_frame=expected_enter,
                expected_exit_frame=expected_exit,
                detected_first_frame=first_det,
                detected_last_frame=last_det,
                detected_frame_count=det_count,
                temporal_iou=float(iou),
                status=_classify(iou),
                is_visual=True,
                skipped=False,
                detection_label=det_label,
                notes=notes,
            )
        )

    return results


__all__ = ["VisibilityResult", "check_visibility"]
