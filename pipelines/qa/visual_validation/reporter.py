"""OmniEcho processing component."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .checks.camera import CameraResult
from .checks.depth import DepthResult
from .checks.quantity import QuantityCheckResult
from .checks.spatial import SpatialResult
from .checks.visibility import VisibilityResult


logger = logging.getLogger(__name__)


# Worst-of-three ordering used to compute the overall verdict – any FAIL
# pulls the report down, otherwise WARN, otherwise PASS.  ERROR (a check
# that crashed) is treated as FAIL but reported separately.
_STATUS_RANK = {"PASS": 0, "WARN": 1, "ERROR": 2, "FAIL": 3}


@dataclass
class ValidationReport:
    """Top-level validation report serialised as JSON.

    Attributes
    ----------
    sample_id:
        Stable identifier of the validated sample (typically the directory
        name, e.g. ``omniecho_qa_001``).
    timestamp:
        ISO-8601 timestamp in UTC marking when the report was generated.
    overall_status:
        ``PASS`` / ``WARN`` / ``FAIL``.  See :func:`_combine_status`.
        The verdict is computed *only* over visual objects – non-visual
        audio sources (footsteps, ambient water, etc.) are recorded with
        ``status="SKIPPED"`` but do not affect the overall outcome.
    visibility, depth, spatial, camera:
        Serialised forms of the corresponding check results.  Any of these
        may be ``None`` (or carry ``status == "ERROR"``) when the
        underlying check failed to run.
    summary:
        Concise dictionary of headline numbers used by dashboards.
    recommendations:
        Auto-generated, human-readable suggestions based on the diagnostic
        numbers in :attr:`summary`.
    """

    sample_id: str
    timestamp: str
    overall_status: str
    visibility: Optional[List[Dict[str, Any]]] = None
    depth: Optional[Dict[str, Any]] = None
    spatial: Optional[Dict[str, Any]] = None
    camera: Optional[Dict[str, Any]] = None
    quantity: Optional[Dict[str, Any]] = None
    summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _sanitise(value: Any) -> Any:
    """Recursively convert ``value`` into something JSON can encode.

    * ``nan`` / ``inf`` floats become ``None`` (so the JSON stays valid).
    * ``Path`` instances become POSIX-style strings.
    * Tuples are flattened to lists.
    * Dataclasses are reduced to their dict form.
    """

    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if is_dataclass(value):
        return _sanitise(asdict(value))
    if isinstance(value, dict):
        return {str(k): _sanitise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitise(v) for v in value]
    # Fallback for numpy scalars, etc.
    try:
        if hasattr(value, "tolist"):
            return _sanitise(value.tolist())
    except Exception:  # pragma: no cover
        pass
    if hasattr(value, "item"):
        try:
            return _sanitise(value.item())
        except Exception:  # pragma: no cover
            pass
    return str(value)


def _combine_status(statuses: List[str]) -> str:
    """Return the worst (highest-ranked) status from ``statuses``.

    ``ERROR`` is preserved as-is in individual sections but collapses to
    ``FAIL`` for the overall verdict so consumers only need to handle the
    canonical PASS / WARN / FAIL trio.
    """

    if not statuses:
        return "FAIL"

    worst_rank = -1
    worst_label = "PASS"
    for s in statuses:
        rank = _STATUS_RANK.get(str(s), _STATUS_RANK["FAIL"])
        if rank > worst_rank:
            worst_rank = rank
            worst_label = str(s)

    if worst_label == "ERROR":
        return "FAIL"
    return worst_label


def _serialise_visibility(results: Optional[List[VisibilityResult]]) -> Optional[List[Dict[str, Any]]]:
    if results is None:
        return None
    return [_sanitise(r) for r in results]


def _serialise_depth(result: Optional[DepthResult]) -> Optional[Dict[str, Any]]:
    if result is None:
        return None
    return _sanitise(result)


def _serialise_spatial(result: Optional[SpatialResult]) -> Optional[Dict[str, Any]]:
    if result is None:
        return None
    payload = _sanitise(result)
    return payload


def _serialise_camera(result: Optional[CameraResult]) -> Optional[Dict[str, Any]]:
    if result is None:
        return None
    return _sanitise(result)


def _serialise_quantity(result: Optional[QuantityCheckResult]) -> Optional[Dict[str, Any]]:
    """Serialise the quantity check result into a report-friendly structure."""
    if result is None:
        return None
    payload: Dict[str, Any] = {
        "overall_status": result.overall_status,
        "n_pass": result.n_pass,
        "n_mismatch": result.n_mismatch,
        "n_undetected": result.n_undetected,
        "n_skipped": result.n_skipped,
        "objects": {},
    }
    for obj_id, qr in result.per_object.items():
        payload["objects"][obj_id] = {
            "object_type": qr.object_type,
            "expected": qr.expected_count,
            "detected_max": qr.detected_count,
            "median_count": _sanitise(qr.median_count),
            "status": qr.status,
            "details": qr.details,
        }
    if result.notes:
        payload["notes"] = result.notes
    return payload


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _summarise(
    visibility_results: Optional[List[VisibilityResult]],
    depth_result: Optional[DepthResult],
    spatial_result: Optional[SpatialResult],
    camera_result: Optional[CameraResult],
    quantity_result: Optional[QuantityCheckResult] = None,
) -> Dict[str, Any]:
    """Build the headline ``summary`` block for dashboards / quick triage."""

    summary: Dict[str, Any] = {}

    # ----- Visibility
    if visibility_results is not None:
        n = len(visibility_results)
        visual = [r for r in visibility_results if getattr(r, "is_visual", True)]
        non_visual = [r for r in visibility_results if not getattr(r, "is_visual", True)]
        n_visual = len(visual)
        n_non_visual = len(non_visual)
        n_pass = sum(1 for r in visual if r.status == "PASS")
        n_warn = sum(1 for r in visual if r.status == "WARN")
        n_fail = sum(1 for r in visual if r.status == "FAIL")
        n_skipped = sum(1 for r in visibility_results if r.status == "SKIPPED")
        n_detected = sum(
            1 for r in visual if (r.detected_frame_count or 0) > 0
        )
        ious = [float(r.temporal_iou) for r in visual]
        mean_iou = float(sum(ious) / len(ious)) if ious else None
        summary["visibility"] = {
            "total_objects": n,
            "visual_objects": n_visual,
            "non_visual_objects": n_non_visual,
            "detected_objects": n_detected,
            "n_objects": n_visual,  # back-compat: visual-only count
            "n_pass": n_pass,
            "n_warn": n_warn,
            "n_fail": n_fail,
            "n_skipped": n_skipped,
            "mean_temporal_iou": _sanitise(mean_iou),
        }
    else:
        summary["visibility"] = None

    # ----- Depth
    if depth_result is not None:
        summary["depth"] = {
            "status": depth_result.status,
            "mean_rank_correlation": _sanitise(depth_result.mean_rank_correlation),
            "min_rank_correlation": _sanitise(depth_result.min_rank_correlation),
            "n_outlier_frames": len(depth_result.outlier_frames),
        }
    else:
        summary["depth"] = None

    # ----- Spatial
    if spatial_result is not None:
        summary["spatial"] = {
            "status": spatial_result.status,
            "overall_azimuth_rmse_deg": _sanitise(spatial_result.overall_azimuth_rmse_deg),
            "overall_elevation_rmse_deg": _sanitise(spatial_result.overall_elevation_rmse_deg),
            "overall_max_deviation_deg": _sanitise(spatial_result.overall_max_deviation_deg),
            "n_objects": len(spatial_result.per_object),
        }
    else:
        summary["spatial"] = None

    # ----- Camera
    if camera_result is not None:
        summary["camera"] = {
            "status": camera_result.status,
            "metadata_has_rotation": camera_result.metadata_has_rotation,
            "motion_detected": camera_result.motion_detected,
            "yaw_correlation": _sanitise(camera_result.yaw_correlation),
            "max_yaw_error_deg": _sanitise(camera_result.max_yaw_error_deg),
            "segment_correlation": _sanitise(
                getattr(camera_result, "segment_correlation", float("nan"))
            ),
            "max_segment_error_deg": _sanitise(
                getattr(camera_result, "max_segment_error_deg", float("nan"))
            ),
            "n_segments": len(getattr(camera_result, "segments", []) or []),
        }
    else:
        summary["camera"] = None

    # ----- Quantity
    if quantity_result is not None:
        summary["quantity"] = {
            "overall_status": quantity_result.overall_status,
            "n_pass": quantity_result.n_pass,
            "n_mismatch": quantity_result.n_mismatch,
            "n_undetected": quantity_result.n_undetected,
            "n_skipped": quantity_result.n_skipped,
        }
    else:
        summary["quantity"] = None

    return summary


def _build_recommendations(
    visibility_results: Optional[List[VisibilityResult]],
    depth_result: Optional[DepthResult],
    spatial_result: Optional[SpatialResult],
    camera_result: Optional[CameraResult],
    errors: Optional[Dict[str, str]],
    quantity_result: Optional[QuantityCheckResult] = None,
) -> List[str]:
    """Generate human-readable improvement suggestions from check outcomes.

    The list is intentionally short and actionable – it is meant to feed a
    triage dashboard, not to replace a deep-dive investigation.
    """

    recs: List[str] = []
    errors = errors or {}

    # Visibility-driven hints (visual-only).
    if visibility_results:
        visual = [r for r in visibility_results if getattr(r, "is_visual", True)]
        if visual:
            n_undetected = sum(
                1 for r in visual if (r.detected_frame_count or 0) == 0
            )
            if n_undetected > 0:
                recs.append(
                    f"{n_undetected} visual object(s) had zero detections – "
                    "consider tightening the SAM3 prompt / mapping "
                    "(visual_validation.object_classifier.LABEL_MAPPING) or "
                    "lowering sam3_confidence."
                )
            low_iou = [
                r for r in visual
                if (r.detected_frame_count or 0) > 0 and r.temporal_iou < 0.4
            ]
            if low_iou:
                recs.append(
                    f"{len(low_iou)} object(s) have low temporal IoU; their "
                    "visibility window in scene.json may be misaligned with "
                    "the rendered frames."
                )

    # Camera-driven hints – prefer the segment-based metric.
    if camera_result is not None:
        seg_err = getattr(camera_result, "max_segment_error_deg", float("nan"))
        cum_err = camera_result.max_yaw_error_deg
        if (
            isinstance(cum_err, float)
            and math.isfinite(cum_err)
            and cum_err > 25.0
            and isinstance(seg_err, float)
            and math.isfinite(seg_err)
            and seg_err < cum_err * 0.5
        ):
            recs.append(
                "Cumulative yaw error dwarfs the segment-based error – the "
                "global integration is drifting; trust the per-segment "
                "metric for the camera verdict."
            )
        if camera_result.metadata_has_rotation and not camera_result.motion_detected:
            recs.append(
                "Metadata expects camera rotation but optical flow detected "
                "none \u2013 verify the renderer actually applies the listener "
                "yaw trajectory."
            )

    # Depth-driven hints.
    if depth_result is not None:
        mean_rho = depth_result.mean_rank_correlation
        if isinstance(mean_rho, float) and not math.isnan(mean_rho) and mean_rho < 0.0:
            recs.append(
                "Depth ranking is anti-correlated with metadata distance \u2013 "
                "check whether the depth model's far/near sign convention "
                "matches the renderer's distance field."
            )
        if depth_result.outlier_frames and len(depth_result.outlier_frames) > 5:
            recs.append(
                f"{len(depth_result.outlier_frames)} frames have severely "
                "inverted depth ranking; inspect the annotated depth maps."
            )

    # Spatial-driven hints.
    if spatial_result is not None and spatial_result.per_object:
        bad = [
            m for m in spatial_result.per_object.values()
            if m.status == "FAIL" and m.n_compared_frames > 0
        ]
        if bad:
            recs.append(
                f"{len(bad)} object(s) failed spatial alignment despite being "
                "detected; verify scene.json trajectories or detector "
                "label mapping."
            )

    if errors:
        for key in sorted(errors):
            recs.append(f"Pipeline subsystem '{key}' raised an error \u2013 see summary.errors.")

    # Quantity-driven hints.
    if quantity_result is not None:
        if quantity_result.n_mismatch > 0:
            mismatch_objs = [
                f"{qr.object_id}({qr.object_type}: expected {qr.expected_count}, got {qr.detected_count})"
                for qr in quantity_result.per_object.values()
                if qr.status == "MISMATCH"
            ]
            recs.append(
                f"Object-count mismatch: {', '.join(mismatch_objs)}. "
                "Spatial/depth results may be unreliable."
            )

    if not recs:
        recs.append("All checks within tolerance \u2013 no action recommended.")
    return recs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_report(
    sample_id: str,
    visibility_results: Optional[List[VisibilityResult]],
    depth_result: Optional[DepthResult],
    spatial_result: Optional[SpatialResult],
    camera_result: Optional[CameraResult],
    quantity_result: Optional[QuantityCheckResult] = None,
    errors: Optional[Dict[str, str]] = None,
) -> ValidationReport:
    """Aggregate per-check results into a :class:`ValidationReport`.

    Parameters
    ----------
    sample_id:
        Identifier used in the output document.
    visibility_results, depth_result, spatial_result, camera_result:
        Outcome of each of the four checks.  Pass ``None`` when a check did
        not run (the reporter will mark the corresponding section as
        ``ERROR`` in the overall status calculation when an entry is
        present in ``errors``).
    quantity_result:
        Outcome of the quantity check.  When ``MISMATCH`` is reported it
        contributes to the overall FAIL verdict.
    errors:
        Optional ``{check_name: error_message}`` describing per-check
        failures.  Reflected in ``overall_status`` and surfaced in the
        ``summary`` block.
    """

    errors = errors or {}

    # Collect per-section statuses for the overall verdict.  We deliberately
    # *exclude* SKIPPED visibility entries (non-visual audio sources) so
    # they cannot drag the verdict down.
    statuses: List[str] = []
    if visibility_results is not None:
        statuses.extend(
            r.status for r in visibility_results
            if getattr(r, "is_visual", True) and r.status != "SKIPPED"
        )
    elif "visibility" in errors:
        statuses.append("ERROR")

    if depth_result is not None:
        statuses.append(depth_result.status)
    elif "depth" in errors:
        statuses.append("ERROR")

    if spatial_result is not None:
        statuses.append(spatial_result.status)
    elif "spatial" in errors:
        statuses.append("ERROR")

    if camera_result is not None:
        statuses.append(camera_result.status)
    elif "camera" in errors:
        statuses.append("ERROR")

    if quantity_result is not None:
        # MISMATCH counts as FAIL for overall verdict
        if quantity_result.overall_status == "MISMATCH":
            statuses.append("FAIL")
        elif quantity_result.overall_status == "UNDETECTED":
            pass  # already covered by visibility FAIL
    elif "quantity" in errors:
        statuses.append("ERROR")

    overall = _combine_status(statuses) if statuses else "FAIL"

    summary = _summarise(
        visibility_results, depth_result, spatial_result, camera_result,
        quantity_result=quantity_result,
    )
    if errors:
        summary["errors"] = dict(errors)

    recommendations = _build_recommendations(
        visibility_results,
        depth_result,
        spatial_result,
        camera_result,
        errors,
        quantity_result=quantity_result,
    )

    return ValidationReport(
        sample_id=str(sample_id),
        timestamp=datetime.now(timezone.utc).isoformat(),
        overall_status=overall,
        visibility=_serialise_visibility(visibility_results),
        depth=_serialise_depth(depth_result),
        spatial=_serialise_spatial(spatial_result),
        camera=_serialise_camera(camera_result),
        quantity=_serialise_quantity(quantity_result),
        summary=summary,
        recommendations=recommendations,
    )


def save_report(report: ValidationReport, output_path: Path) -> Path:
    """Serialise ``report`` to ``output_path`` as pretty-printed JSON."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _sanitise(report)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    logger.info("validation report written to %s", output_path)
    return output_path


__all__ = ["ValidationReport", "generate_report", "save_report"]
