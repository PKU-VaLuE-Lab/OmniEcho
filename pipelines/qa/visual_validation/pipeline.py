"""OmniEcho processing component."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from .checks import (
    check_camera_motion,
    check_depth_consistency,
    check_quantity,
    check_spatial_alignment,
    check_visibility,
)
from .checks.quantity import QuantityCheckResult
from .config import ValidationConfig
from .models.depth_estimator import DepthEstimator
from .models.segmentor import VideoSegmentor
from .object_classifier import is_visual_object, label_for_segmentor as _label_for_segmentor
from .reporter import ValidationReport, generate_report, save_report

from .visualizer import Visualizer


logger = logging.getLogger(__name__)


# Sub-directory names under each sample's ``validation/`` folder.
_VALIDATION_DIRNAME = "validation"
_FRAMES_SUBDIR = "frames"
_DEPTH_SUBDIR = "depth_maps"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def label_for_segmentor(obj: Dict[str, Any]) -> str:
    """Backwards-compatible alias for :func:`object_classifier.label_for_segmentor`.

    Kept as a top-level export of this module so historical callers
    (and tests) that import it from ``visual_validation.pipeline`` keep
    working.  All real logic now lives in
    :mod:`visual_validation.object_classifier`.
    """

    return _label_for_segmentor(obj)


def _read_video_metadata(video_path: Path) -> tuple[int, int, int, float]:
    """Return ``(width, height, total_frames, fps)`` for ``video_path``."""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        return width, height, total, fps
    finally:
        cap.release()


def _discover_video(sample_dir: Path) -> Optional[Path]:
    """Locate the rendered video inside ``sample_dir``.

    Falls back to the first ``.mp4`` we can find – tolerates the slightly
    different naming used across OmniEcho versions.
    """

    candidates = [
        sample_dir / "video.mp4",
        sample_dir / "video_with_audio.mp4",
    ]
    for c in candidates:
        if c.is_file():
            return c
    for c in sorted(sample_dir.glob("*.mp4")):
        return c
    return None


def _is_sample_dir(p: Path) -> bool:
    """Heuristic test for OmniEcho sample directories."""

    if not p.is_dir():
        return False
    if not (p / "scene.json").is_file():
        return False
    if _discover_video(p) is None:
        return False
    return True


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class VisualValidationPipeline:
    """End-to-end visual validation orchestrator."""

    def __init__(self, config: Optional[ValidationConfig] = None) -> None:
        self.config = config or ValidationConfig.default()
        self._depth_model: Optional[DepthEstimator] = None
        self._segmentor: Optional[VideoSegmentor] = None
        self._visualizer: Optional[Visualizer] = None

    # ------------------------------------------------------- lazy components
    @property
    def depth_model(self) -> DepthEstimator:
        """OmniEcho processing component."""

        if self._depth_model is None:
            logger.info(
                "loading depth model %s on %s",
                self.config.depth_model_path,
                self.config.device,
            )
            self._depth_model = DepthEstimator(
                model_path=self.config.depth_model_path,
                encoder=self.config.depth_encoder,
                device=self.config.device,
            )
        return self._depth_model

    @property
    def segmentor(self) -> VideoSegmentor:
        """Load the SAM3 wrapper on first access."""

        if self._segmentor is None:
            logger.info(
                "loading SAM3 segmentor %s on %s",
                self.config.sam3_model_path,
                self.config.device,
            )
            self._segmentor = VideoSegmentor(
                model_path=self.config.sam3_model_path,
                device=self.config.device,
                conf=self.config.sam3_confidence,
                keep_masks=True,
            )
        return self._segmentor

    @property
    def visualizer(self) -> Visualizer:
        """Lazily build the :class:`Visualizer`."""

        if self._visualizer is None:
            self._visualizer = Visualizer(self.config)
        return self._visualizer

    # ------------------------------------------------------------ scene I/O
    @staticmethod
    def _load_scene(sample_dir: Path) -> Dict[str, Any]:
        scene_path = sample_dir / "scene.json"
        with scene_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    @staticmethod
    def _load_metadata(sample_dir: Path) -> Optional[Dict[str, Any]]:
        meta_path = sample_dir / "metadata.json"
        if not meta_path.is_file():
            return None
        try:
            with meta_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:  # pragma: no cover - corrupt metadata
            logger.warning("failed to load %s: %s", meta_path, exc)
            return None

    # ------------------------------------------------------ single sample
    def validate_sample(
        self,
        sample_dir: Path,
        generate_video: bool = True,
        output_dir: Optional[Path] = None,
    ) -> ValidationReport:
        """Run the full validation pipeline on a single sample directory.

        Parameters
        ----------
        sample_dir:
            Directory containing ``scene.json`` and ``video.mp4``.
        generate_video:
            When ``True`` (default) the annotated MP4 and keyframe PNGs are
            written under ``{sample_dir}/validation/``.
        output_dir:
            Override the location of the validation artefacts.  Defaults to
            ``{sample_dir}/validation``.

        Returns
        -------
        The :class:`ValidationReport` produced; the JSON file is written
        to ``<output_dir>/report.json``.
        """

        sample_dir = Path(sample_dir).resolve()
        if not sample_dir.is_dir():
            raise FileNotFoundError(f"sample directory not found: {sample_dir}")

        scene = self._load_scene(sample_dir)
        sample_id = str(scene.get("scene_id") or sample_dir.name)
        objects = list(scene.get("objects") or [])
        camera_trajectory = list(
            (scene.get("listener") or {}).get("orientation", {}).get("trajectory") or []
        )

        # ----- Classify objects: visual vs non-visual.  Non-visual audio
        # sources (footsteps, ambient water, wind, ...) are excluded from
        # segmentation so the detector is not asked to find something that
        # isn't there; they will still be reported with status="SKIPPED".
        visual_flags: Dict[str, bool] = {
            str(obj.get("id", "")): bool(is_visual_object(obj)) for obj in objects
        }
        detection_labels: Dict[str, str] = {
            str(obj.get("id", "")): label_for_segmentor(obj) for obj in objects
        }
        visual_objects = [
            obj for obj in objects
            if visual_flags.get(str(obj.get("id", "")), True)
        ]
        non_visual_objects = [
            obj for obj in objects
            if not visual_flags.get(str(obj.get("id", "")), True)
        ]

        video_path = _discover_video(sample_dir)
        if video_path is None:
            raise FileNotFoundError(f"no video.mp4 in {sample_dir}")
        width, height, total_frames, src_fps = _read_video_metadata(video_path)
        if width > 0:
            self.config.frame_width = width
        if height > 0:
            self.config.frame_height = height
        if src_fps > 0:
            self.config.video_fps = int(round(src_fps))

        out_root = Path(output_dir) if output_dir else sample_dir / _VALIDATION_DIRNAME
        out_root.mkdir(parents=True, exist_ok=True)

        labels: List[str] = [
            label_for_segmentor(obj) for obj in visual_objects
        ]
        # Build a stable label-per-object mapping so detections can be
        # looked up by the same key regardless of duplicates.
        unique_labels = list(dict.fromkeys(labels))
        logger.info(
            "validating %s: %d objects (%d visual, %d non-visual skipped), "
            "video=%s (%dx%d @ %sfps, %d frames)",
            sample_id,
            len(objects),
            len(visual_objects),
            len(non_visual_objects),
            video_path.name,
            width,
            height,
            self.config.video_fps,
            total_frames,
        )
        if non_visual_objects:
            logger.info(
                "non-visual objects skipped: %s",
                ", ".join(
                    f"{o.get('id')}({o.get('type')})" for o in non_visual_objects
                ),
            )

        errors: Dict[str, str] = {}
        depth_maps: List[Any] = []
        depth_indices: List[int] = []
        detections: Dict[str, list] = {label: [] for label in unique_labels}

        # --------------------------------------------------- segmentation
        try:
            detections = self.segmentor.segment_objects(
                video_path=video_path,
                object_labels=unique_labels,
                sample_fps=self.config.sample_fps,
            )
            # Ensure every requested label has an entry.
            for label in unique_labels:
                detections.setdefault(label, [])
            # Surface a soft segmentor warning (e.g. backend unavailable)
            # so the report explains why visibility/depth/spatial are empty.
            seg_warning = getattr(self.segmentor, "last_error", None)
            if seg_warning:
                errors["segmentor"] = seg_warning
        except Exception as exc:
            logger.exception("SAM3 segmentation failed for %s", sample_id)
            errors["segmentor"] = f"{type(exc).__name__}: {exc}"
            detections = {label: [] for label in unique_labels}

        # The check modules look up detections under ``obj['label']`` (the
        # display name) while the segmentor produces them under
        # the English ``label_for_segmentor(obj)`` prompt.  Mirror each list
        # under every alias the downstream code might use so that all of
        # visibility / depth / spatial can find them.  Non-visual objects
        # have no segmentor label, so they get an empty list under their
        # aliases for graceful downstream handling.
        for obj in objects:
            if not visual_flags.get(str(obj.get("id", "")), True):
                # Ensure non-visual objects have empty entries under their
                # aliases so downstream lookups never raise KeyError.
                for alias in (
                    obj.get("label"),
                    obj.get("id"),
                    obj.get("type"),
                ):
                    if alias:
                        detections.setdefault(str(alias), [])
                continue
            seg_label = label_for_segmentor(obj)
            dets = detections.get(seg_label, [])
            for alias in (
                obj.get("label"),
                obj.get("id"),
                obj.get("type"),
            ):
                if not alias:
                    continue
                alias_str = str(alias)
                if alias_str == seg_label:
                    continue
                # Do not overwrite a non-empty list that may have been
                # produced under the alias key directly.
                existing = detections.get(alias_str)
                if not existing:
                    detections[alias_str] = dets

        # ============================================================
        # Step 1: QUANTITY CHECK (runs first)
        # ============================================================
        quantity_result: Optional[QuantityCheckResult] = None
        try:
            counts_per_frame = self.segmentor.count_objects_per_frame(
                video_path=video_path,
                object_labels=unique_labels,
                sample_fps=self.config.sample_fps,
            )
            quantity_result = check_quantity(
                counts_per_frame=counts_per_frame,
                objects=objects,
                visual_flags=visual_flags,
            )
            logger.info(
                "quantity check: %s (pass=%d, mismatch=%d, undetected=%d, skipped=%d)",
                quantity_result.overall_status,
                quantity_result.n_pass,
                quantity_result.n_mismatch,
                quantity_result.n_undetected,
                quantity_result.n_skipped,
            )
        except Exception as exc:
            logger.exception("quantity check failed for %s", sample_id)
            errors["quantity"] = f"{type(exc).__name__}: {exc}"

        # Build per-object quantity status lookup for downstream flow control.
        _quantity_status: Dict[str, str] = {}
        if quantity_result is not None:
            for obj_id, qr in quantity_result.per_object.items():
                _quantity_status[obj_id] = qr.status

        # ============================================================
        # Step 2: VISIBILITY CHECK
        # ============================================================
        # If quantity reports UNDETECTED for an object, visibility will
        # naturally fail too – we still run it for completeness.
        visibility_results = None
        try:
            visibility_results = check_visibility(
                detections=detections,
                objects=objects,
                fps=self.config.video_fps,
                total_frames=max(total_frames, 1),
                frame_tolerance=self.config.visibility_frame_tolerance,
                visual_flags=visual_flags,
                detection_labels=detection_labels,
            )
        except Exception as exc:
            logger.exception("visibility check failed for %s", sample_id)
            errors["visibility"] = f"{type(exc).__name__}: {exc}"

        # ============================================================
        # Step 3: CAMERA CHECK (independent of object detection)
        # ============================================================
        camera_result = None
        try:
            camera_result = check_camera_motion(
                video_path=str(video_path),
                camera_trajectory=camera_trajectory,
                fps=self.config.video_fps,
                fov_h_deg=self.config.fov_horizontal_deg,
                frame_width=self.config.frame_width,
            )
        except Exception as exc:
            logger.exception("camera motion check failed for %s", sample_id)
            errors["camera"] = f"{type(exc).__name__}: {exc}"

        # ============================================================
        # Step 4: SPATIAL CHECK
        # ============================================================
        # If quantity reports MISMATCH, spatial results are unreliable.
        spatial_result = None
        spatial_unreliable = any(
            s == "MISMATCH" for s in _quantity_status.values()
        )
        try:
            spatial_result = check_spatial_alignment(
                detections=detections,
                objects=visual_objects,
                camera_trajectory=camera_trajectory,
                fov_h_deg=self.config.fov_horizontal_deg,
                frame_width=self.config.frame_width,
                frame_height=self.config.frame_height,
                fps=self.config.video_fps,
                azimuth_tolerance=self.config.azimuth_tolerance_deg,
                elevation_tolerance=self.config.elevation_tolerance_deg,
            )
            if spatial_unreliable and spatial_result is not None:
                spatial_result.notes.append(
                    "UNRELIABLE: quantity mismatch detected – "
                    "spatial results may be inaccurate due to ambiguous detections"
                )
        except Exception as exc:
            logger.exception("spatial alignment check failed for %s", sample_id)
            errors["spatial"] = f"{type(exc).__name__}: {exc}"

        # ============================================================
        # Step 5: DEPTH CHECK
        # ============================================================
        # Depth inference runs first (shared by depth check).
        try:
            depth_maps, depth_indices = self.depth_model.estimate_video(
                video_path=video_path,
                sample_fps=self.config.sample_fps,
            )
        except Exception as exc:
            logger.exception("depth estimation failed for %s", sample_id)
            errors["depth_model"] = f"{type(exc).__name__}: {exc}"
            depth_maps, depth_indices = [], []

        depth_result = None
        if depth_maps and depth_indices:
            try:
                depth_result = check_depth_consistency(
                    depth_maps=depth_maps,
                    frame_indices=depth_indices,
                    detections=detections,
                    objects=visual_objects,
                    fps=self.config.video_fps,
                    correlation_threshold=self.config.depth_correlation_threshold,
                )
                if spatial_unreliable and depth_result is not None:
                    depth_result.notes.append(
                        "UNRELIABLE: quantity mismatch detected – "
                        "depth results may be inaccurate due to ambiguous detections"
                    )
            except Exception as exc:
                logger.exception("depth consistency check failed for %s", sample_id)
                errors["depth"] = f"{type(exc).__name__}: {exc}"
        elif "depth_model" not in errors:
            errors["depth"] = "no depth maps produced"

        # ------------------------------------------------------ visualisation
        # Map detections from "label" key to "object_id" key for the visualiser.
        detections_by_label_for_viz = detections

        if generate_video:
            try:
                annotated_path = out_root / "annotated_video.mp4"
                self.visualizer.generate_annotated_video(
                    video_path=video_path,
                    output_path=annotated_path,
                    depth_maps=depth_maps,
                    depth_frame_indices=depth_indices,
                    detections=detections_by_label_for_viz,
                    objects=objects,
                    camera_trajectory=camera_trajectory,
                    fps=self.config.video_fps,
                )
            except Exception as exc:
                logger.exception("annotated video generation failed for %s", sample_id)
                errors["annotated_video"] = f"{type(exc).__name__}: {exc}"

        try:
            self.visualizer.save_keyframes(
                video_path=video_path,
                output_dir=out_root / _FRAMES_SUBDIR,
                depth_maps=depth_maps,
                depth_frame_indices=depth_indices,
                detections=detections_by_label_for_viz,
                objects=objects,
                camera_trajectory=camera_trajectory,
                fps=self.config.video_fps,
                n_keyframes=6,
            )
        except Exception as exc:
            logger.exception("keyframe export failed for %s", sample_id)
            errors["keyframes"] = f"{type(exc).__name__}: {exc}"

        if depth_maps:
            try:
                self.visualizer.save_depth_maps(
                    depth_maps=depth_maps,
                    depth_frame_indices=depth_indices,
                    output_dir=out_root / _DEPTH_SUBDIR,
                )
            except Exception as exc:
                logger.exception("depth map export failed for %s", sample_id)
                errors["depth_maps_export"] = f"{type(exc).__name__}: {exc}"

        # ------------------------------------------------------------- report
        report = generate_report(
            sample_id=sample_id,
            visibility_results=visibility_results,
            depth_result=depth_result,
            spatial_result=spatial_result,
            camera_result=camera_result,
            quantity_result=quantity_result,
            errors=errors,
        )
        save_report(report, out_root / "report.json")

        logger.info(
            "validation finished for %s: overall=%s",
            sample_id,
            report.overall_status,
        )
        return report

    # --------------------------------------------------------------- batch
    def validate_batch(
        self,
        output_dir: Path,
        generate_video: bool = True,
    ) -> List[ValidationReport]:
        """Validate every sample directory under ``output_dir``."""

        output_dir = Path(output_dir).resolve()
        if not output_dir.is_dir():
            raise FileNotFoundError(f"batch directory not found: {output_dir}")

        sample_dirs = sorted(p for p in output_dir.iterdir() if _is_sample_dir(p))
        if not sample_dirs:
            logger.warning("no sample directories found under %s", output_dir)
            return []

        try:
            from tqdm import tqdm  # type: ignore

            iterator = tqdm(sample_dirs, desc="validating", unit="sample")
        except Exception:  # pragma: no cover - tqdm always installed in our env
            iterator = sample_dirs

        reports: List[ValidationReport] = []
        for sample_dir in iterator:
            try:
                report = self.validate_sample(
                    sample_dir=sample_dir,
                    generate_video=generate_video,
                )
                reports.append(report)
            except Exception as exc:
                logger.exception("validation crashed for %s", sample_dir)
                # Record a synthetic failed report so the roll-up surfaces
                # the directory rather than silently dropping it.
                reports.append(
                    generate_report(
                        sample_id=sample_dir.name,
                        visibility_results=None,
                        depth_result=None,
                        spatial_result=None,
                        camera_result=None,
                        quantity_result=None,
                        errors={"pipeline": f"{type(exc).__name__}: {exc}"},
                    )
                )

        roll_up_path = output_dir / "batch_validation_report.json"
        roll_up = {
            "n_samples": len(reports),
            "n_pass": sum(1 for r in reports if r.overall_status == "PASS"),
            "n_warn": sum(1 for r in reports if r.overall_status == "WARN"),
            "n_fail": sum(1 for r in reports if r.overall_status == "FAIL"),
            "samples": [
                {
                    "sample_id": r.sample_id,
                    "overall_status": r.overall_status,
                    "summary": r.summary,
                }
                for r in reports
            ],
        }
        with roll_up_path.open("w", encoding="utf-8") as fh:
            json.dump(roll_up, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        logger.info(
            "batch validation finished: %d samples (%d pass / %d warn / %d fail)",
            roll_up["n_samples"],
            roll_up["n_pass"],
            roll_up["n_warn"],
            roll_up["n_fail"],
        )
        return reports


__all__ = ["VisualValidationPipeline", "label_for_segmentor"]
