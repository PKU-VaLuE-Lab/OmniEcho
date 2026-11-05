"""OmniEcho processing component."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..object_classifier import is_visual_object, label_for_segmentor


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class QuantityResult:
    """OmniEcho processing component."""

    object_id: str
    object_type: str
    expected_count: int  
    detected_count: int  
    status: str  # PASS / MISMATCH / UNDETECTED / SKIPPED
    frame_counts: list[int] = field(default_factory=list)  
    median_count: float = 0.0
    max_count: int = 0
    details: str = ""


@dataclass
class QuantityCheckResult:
    """OmniEcho processing component."""

    per_object: Dict[str, QuantityResult]
    overall_status: str  # PASS / MISMATCH / UNDETECTED
    n_pass: int = 0
    n_mismatch: int = 0
    n_undetected: int = 0
    n_skipped: int = 0
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_quantity(
    counts_per_frame: Dict[str, List[int]],
    objects: List[Dict[str, Any]],
    visual_flags: Optional[Dict[str, bool]] = None,
    expected_counts: Optional[Dict[str, int]] = None,
) -> QuantityCheckResult:
    """OmniEcho processing component."""

    visual_flags = visual_flags or {}
    expected_counts = expected_counts or {}
    results: Dict[str, QuantityResult] = {}
    notes: List[str] = []

    for obj in objects:
        obj_id = str(obj.get("id", ""))
        obj_type = str(obj.get("type", ""))
        is_visual = visual_flags.get(obj_id, is_visual_object(obj))
        expected = expected_counts.get(obj_id, 1)


        if not is_visual:
            results[obj_id] = QuantityResult(
                object_id=obj_id,
                object_type=obj_type,
                expected_count=expected,
                detected_count=0,
                status="SKIPPED",
                details=f"Nonvisual object; skipping quantity check",
            )
            continue


        label = label_for_segmentor(obj)


        frame_counts = counts_per_frame.get(label, [])

        if not frame_counts:
            results[obj_id] = QuantityResult(
                object_id=obj_id,
                object_type=obj_type,
                expected_count=expected,
                detected_count=0,
                status="UNDETECTED",
                details=f"No {label} detected",
            )
            continue

        max_count = int(max(frame_counts)) if frame_counts else 0
        median_count = float(np.median(frame_counts)) if frame_counts else 0.0


        if max_count == 0:
            status = "UNDETECTED"
            details = f"No {label} detected"
        elif max_count == expected:
            status = "PASS"
            details = f"Count matches: expected {expected}, detected {max_count}"
        elif max_count > expected:
            status = "MISMATCH"
            details = (
                f"Too many objects: expected {expected}, maximum detected {max_count}"
                f" (median {median_count:.1f})"
            )
        else:
            
            status = "MISMATCH"
            details = f"Too few objects: expected {expected}, maximum detected {max_count}"

        results[obj_id] = QuantityResult(
            object_id=obj_id,
            object_type=obj_type,
            expected_count=expected,
            detected_count=max_count,
            status=status,
            frame_counts=frame_counts,
            median_count=median_count,
            max_count=max_count,
            details=details,
        )


    n_pass = sum(1 for r in results.values() if r.status == "PASS")
    n_mismatch = sum(1 for r in results.values() if r.status == "MISMATCH")
    n_undetected = sum(1 for r in results.values() if r.status == "UNDETECTED")
    n_skipped = sum(1 for r in results.values() if r.status == "SKIPPED")


    if n_mismatch > 0:
        overall_status = "MISMATCH"
    elif n_undetected > 0:
        overall_status = "UNDETECTED"
    else:
        overall_status = "PASS"

    return QuantityCheckResult(
        per_object=results,
        overall_status=overall_status,
        n_pass=n_pass,
        n_mismatch=n_mismatch,
        n_undetected=n_undetected,
        n_skipped=n_skipped,
        notes=notes,
    )


__all__ = ["QuantityResult", "QuantityCheckResult", "check_quantity"]
