"""OmniEcho processing component."""

from __future__ import annotations

import copy
import json
import logging
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace as _dc_replace
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from omniecho_audio.schema import (
    AnnotatedAudioObject,
    CameraMode,
    OffScreenStrategy,
    AnnotatedScene,
    AnnotatedTrajectoryPoint,
    VisibilityState,
)

from .config import ValidationConfig
from .models.segmentor import FrameDetection, VideoSegmentor
from .object_classifier import is_visual_object, label_for_segmentor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


@dataclass
class VisibilityInfo:
    """OmniEcho processing component."""

    state: VisibilityState
    visual_enter: float = -1.0    
    visual_exit: float = -1.0     
    first_seen_frame: int = -1
    last_seen_frame: int = -1
    n_frames_detected: int = 0
    n_frames_total: int = 0
    
    presumed_visible: bool = False


@dataclass
class DriftReport:
    """OmniEcho processing component."""

    is_static: bool
    mean_drift_px: float
    max_drift_px: float
    n_frames_sampled: int
    threshold_px: float
    compensation: Optional[list[float]] = None  


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


def _wrap_deg(angle_deg: float) -> float:
    """OmniEcho processing component."""

    a = (angle_deg + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


def _inverse_project(
    cx: float,
    cy: float,
    width: int,
    height: int,
    fov_h_deg: float,
) -> tuple[float, float]:
    """OmniEcho processing component."""

    half_w = width / 2.0
    half_h = height / 2.0
    half_fov_h = math.radians(fov_h_deg) / 2.0
    tan_half_h = math.tan(half_fov_h)
    f = half_w / tan_half_h
    
    tan_half_v = tan_half_h * (height / width)
    f_v = half_h / tan_half_v

    azimuth = math.degrees(math.atan2(half_w - cx, f))
    elevation = math.degrees(math.atan2(half_h - cy, f_v))
    return _wrap_deg(azimuth), float(np.clip(elevation, -89.999, 89.999))


def _smooth_trajectory(
    values: np.ndarray, window: int = 11, poly: int = 3
) -> np.ndarray:
    """OmniEcho processing component."""

    n = len(values)
    if n < poly + 2:
        return values.astype(np.float64)
    w = min(window, n)
    if w % 2 == 0:
        w -= 1
    if w < poly + 2:
        return values.astype(np.float64)
    try:
        return savgol_filter(values, w, poly)
    except Exception:
        return values.astype(np.float64)


def _resample_trajectory(
    times: list[float],
    azimuths: list[float],
    elevations: list[float],
    distances: list[float],
    confidences: list[float],
    state: VisibilityState,
    duration: float,
    interval: float = 0.25,
) -> list[AnnotatedTrajectoryPoint]:
    """OmniEcho processing component."""

    if not times:
        return []
    times_arr = np.asarray(times, dtype=np.float64)
    az_arr = np.asarray(azimuths, dtype=np.float64)
    el_arr = np.asarray(elevations, dtype=np.float64)
    d_arr = np.asarray(distances, dtype=np.float64)
    c_arr = np.asarray(confidences, dtype=np.float64)

    if len(times) == 1:
        
        sample_times = np.arange(0.0, duration + interval / 2, interval)
        return [
            AnnotatedTrajectoryPoint(
                time=round(float(t), 2),
                azimuth=round(float(az_arr[0]), 2),
                elevation=round(float(el_arr[0]), 2),
                distance=round(float(d_arr[0]), 3),
                gain_db=0.0,
                visibility_state=state,
                confidence=float(c_arr[0]),
            )
            for t in sample_times
        ]

    az_interp = interp1d(times_arr, az_arr, kind="linear", fill_value="extrapolate")
    el_interp = interp1d(times_arr, el_arr, kind="linear", fill_value="extrapolate")
    d_interp = interp1d(times_arr, d_arr, kind="linear", fill_value="extrapolate")
    c_interp = interp1d(times_arr, c_arr, kind="linear", fill_value="extrapolate")

    t0 = max(0.0, float(times_arr[0]))
    t1 = max(t0 + interval, min(duration, float(times_arr[-1])))
    sample_times = np.arange(t0, t1 + interval / 2, interval)

    points: list[AnnotatedTrajectoryPoint] = []
    for t in sample_times:
        points.append(
            AnnotatedTrajectoryPoint(
                time=round(float(t), 2),
                azimuth=round(float(_wrap_deg(float(az_interp(t)))), 2),
                elevation=round(float(np.clip(float(el_interp(t)), -89.0, 89.0)), 2),
                distance=round(float(max(0.1, float(d_interp(t)))), 3),
                gain_db=0.0,
                visibility_state=state,
                confidence=float(np.clip(float(c_interp(t)), 0.0, 1.0)),
            )
        )
    return points


def _extrapolate_trajectory(
    last_points: list[AnnotatedTrajectoryPoint],
    duration: float,
    sample_interval: float = 0.25,
    fov_h_deg: float = 90.0,
) -> list[AnnotatedTrajectoryPoint]:
    """OmniEcho processing component."""

    if not last_points:
        return []

    tail = last_points[-3:] if len(last_points) >= 2 else last_points[-1:]
    last = tail[-1]

    if len(tail) >= 2 and tail[-1].time > tail[0].time:
        dt = tail[-1].time - tail[0].time
        d_az = _wrap_deg(tail[-1].azimuth - tail[0].azimuth) / dt
        d_el = (tail[-1].elevation - tail[0].elevation) / dt
        d_dist = (tail[-1].distance - tail[0].distance) / dt
    else:
        d_az = 0.0
        d_el = 0.0
        d_dist = 0.0


    d_az = float(np.clip(d_az, -90.0, 90.0))
    d_el = float(np.clip(d_el, -45.0, 45.0))

    az_max = 135.0
    extrapolated: list[AnnotatedTrajectoryPoint] = []
    t = last.time + sample_interval
    conf = 0.8
    while t <= duration + sample_interval / 2:
        delta = t - last.time
        new_az = _wrap_deg(last.azimuth + d_az * delta)
        
        if new_az > az_max:
            new_az = az_max
        elif new_az < -az_max:
            new_az = -az_max
        new_el = float(np.clip(last.elevation + d_el * delta, -45.0, 45.0))
        new_dist = float(max(0.3, last.distance + d_dist * delta))

        extrapolated.append(
            AnnotatedTrajectoryPoint(
                time=round(float(t), 2),
                azimuth=round(float(new_az), 2),
                elevation=round(float(new_el), 2),
                distance=round(float(new_dist), 3),
                gain_db=0.0,
                visibility_state=VisibilityState.OFF_SCREEN_DYNAMIC,
                confidence=round(float(max(0.2, conf)), 2),
            )
        )
        conf -= 0.1
        t += sample_interval
    return extrapolated


def _trajectory_point_to_dict(p: AnnotatedTrajectoryPoint) -> dict:
    """OmniEcho processing component."""

    return {
        "t": float(p.time),
        "azimuth": float(p.azimuth),
        "elevation": float(p.elevation),
        "distance": float(p.distance),
        "gain_db": float(p.gain_db),
        "visibility_state": p.visibility_state.value,
        "confidence": float(p.confidence),
    }


def _audio_object_to_dict(obj: AnnotatedAudioObject) -> dict:
    """OmniEcho processing component."""

    return {
        "id": obj.object_id,
        "label": obj.label,
        "trajectory": [_trajectory_point_to_dict(p) for p in obj.trajectory],
        "acoustic_start": float(obj.acoustic_start),
        "acoustic_end": float(obj.acoustic_end),
        "visual_enter": float(obj.visual_enter),
        "visual_exit": float(obj.visual_exit),
        "off_screen_strategy": obj.off_screen_strategy.value,
        "spread_deg": float(obj.spread_deg),
        "gain_db": float(obj.gain_db),
    }


# ---------------------------------------------------------------------------
# MetadataCorrector
# ---------------------------------------------------------------------------


class MetadataCorrector:
    """OmniEcho processing component."""

    STATIC_DRIFT_THRESHOLD_PX = 2.0    
    OFFSCREEN_ALWAYS_CONFIDENCE = 0.5  
    AZIMUTH_PHYSICAL_BOUND = 135.0     

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig.default()
        self._segmentor: Optional[VideoSegmentor] = None
        self._depth_estimator = None  # type: Any
        self.last_report: dict = {}

    # ------------------------------------------------------------ lazy loaders
    @property
    def segmentor(self) -> VideoSegmentor:
        if self._segmentor is None:
            self._segmentor = VideoSegmentor(
                model_path=self.config.sam3_model_path,
                device=self.config.device,
                conf=self.config.sam3_confidence,
                keep_masks=False,
            )
        return self._segmentor

    @property
    def depth_estimator(self):
        if self._depth_estimator is None:
            from .models.depth_estimator import DepthEstimator

            self._depth_estimator = DepthEstimator(
                model_path=self.config.depth_model_path,
                encoder=self.config.depth_encoder,
                device=self.config.device,
            )
        return self._depth_estimator

    # ===================================================== main entry point
    def correct_sample(
        self,
        video_path: Path | str,
        scene_path: Path | str,
        output_dir: Path | str,
    ) -> dict:
        """OmniEcho processing component."""

        video_path = Path(video_path)
        scene_path = Path(scene_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not video_path.is_file():
            raise FileNotFoundError(f"video not found: {video_path}")
        if not scene_path.is_file():
            raise FileNotFoundError(f"scene.json not found: {scene_path}")

        scene = self._load_scene(scene_path)
        scene_id = scene.get("scene_id", scene_path.stem)
        duration = float(scene.get("duration_sec", scene.get("duration", 6.0)))
        logger.info("[OmniEcho] correct_sample: scene_id=%s duration=%.2fs", scene_id, duration)


        try:
            logger.info("[Stage A] verify static camera ...")
            drift_report = self._verify_static_camera(video_path)
            logger.info(
                "[Stage A] done: is_static=%s mean=%.2fpx max=%.2fpx",
                drift_report.is_static,
                drift_report.mean_drift_px,
                drift_report.max_drift_px,
            )
        except Exception as exc:
            logger.warning("[Stage A] failed: %s", exc, exc_info=True)
            drift_report = DriftReport(
                is_static=True,
                mean_drift_px=0.0,
                max_drift_px=0.0,
                n_frames_sampled=0,
                threshold_px=self.STATIC_DRIFT_THRESHOLD_PX,
            )


        try:
            logger.info("[Stage B] SAM3 detection ...")
            detections, label_map, video_meta = self._detect_objects_sam3(
                video_path, scene
            )
            logger.info(
                "[Stage B] done: %d labels, %d total detections",
                len(label_map),
                sum(len(v) for v in detections.values()),
            )
        except Exception as exc:
            logger.warning("[Stage B] failed: %s", exc, exc_info=True)
            detections = {}
            label_map = {}
            video_meta = self._probe_video(video_path)


        try:
            logger.info("[Stage C] visibility windows ...")
            visibility_windows = self._compute_visibility_windows(
                detections, label_map, video_meta, duration, scene
            )
        except Exception as exc:
            logger.warning("[Stage C] failed: %s", exc, exc_info=True)
            visibility_windows = {}


        depth_maps_list: list[np.ndarray] = []
        depth_frame_indices: list[int] = []
        try:
            logger.info("[Stage D] depth estimation ...")
            depth_estimates, depth_maps_list, depth_frame_indices = (
                self._estimate_depth(
                    video_path, detections, label_map, video_meta
                )
            )
            logger.info(
                "[Stage D] done: %d objects with depth, %d depth maps cached",
                len(depth_estimates),
                len(depth_maps_list),
            )
        except Exception as exc:
            logger.warning("[Stage D] failed: %s", exc, exc_info=True)
            depth_estimates = {}
            depth_maps_list = []
            depth_frame_indices = []


        try:
            logger.info("[Stage E] correct visible trajectories ...")
            visible_trajs = self._correct_visible_trajectories(
                detections,
                label_map,
                visibility_windows,
                depth_estimates,
                scene,
                video_meta,
                duration,
            )
            logger.info("[Stage E] done: %d objects corrected", len(visible_trajs))
        except Exception as exc:
            logger.warning("[Stage E] failed: %s", exc, exc_info=True)
            visible_trajs = {}


        try:
            azimuth_diagnostics = self._check_azimuth_sign_consistency(
                visible_trajs, scene
            )
            if azimuth_diagnostics.get("azimuth_sign_flip_count", 0) > 0:
                logger.warning(
                    "[Diagnostics] azimuth sign flip detected: count=%d objects=%s",
                    azimuth_diagnostics["azimuth_sign_flip_count"],
                    [f["object_id"] for f in azimuth_diagnostics["flipped_objects"]],
                )
        except Exception as exc:
            logger.warning("[Diagnostics] azimuth sign check failed: %s", exc)
            azimuth_diagnostics = {
                "azimuth_sign_flip_count": 0,
                "flipped_objects": [],
                "warning": False,
            }


        try:
            logger.info("[Stage F] infer offscreen trajectories ...")
            final_objects = self._infer_offscreen_trajectories(
                visible_trajs, visibility_windows, scene, duration
            )
            logger.info("[Stage F] done: %d total objects", len(final_objects))
        except Exception as exc:
            logger.warning("[Stage F] failed: %s", exc, exc_info=True)
            final_objects = {}


        try:
            logger.info("[Stage G] quantity check & split ...")
            final_obj_list, split_actions = self._check_quantity_and_split(
                final_objects, detections, label_map, scene, video_meta, duration
            )
            logger.info("[Stage G] done: %d final objects after split", len(final_obj_list))
        except Exception as exc:
            logger.warning("[Stage G] failed: %s", exc, exc_info=True)
            final_obj_list = list(final_objects.values())
            split_actions = []


        try:
            final_obj_list = [
                _dc_replace(
                    o,
                    trajectory=tuple(
                        self._extend_trajectory_to_duration(list(o.trajectory), duration)
                    ),
                )
                for o in final_obj_list
            ]
        except Exception as exc:
            logger.warning("[trajectory extension] failed: %s", exc, exc_info=True)


        


        direction_warnings: list[dict] = []
        try:
            for obj in final_obj_list:
                strategy = getattr(obj, "off_screen_strategy", None)
                if strategy == OffScreenStrategy.TRUST_METADATA:
                    warn = self._validate_trusted_direction(
                        obj, output_dir, video_meta
                    )
                    if warn is not None:
                        direction_warnings.append(warn)
        except Exception as exc:
            logger.warning(
                "[direction_validate] batch validation failed: %s", exc
            )


        presumed_visible_warnings = [
            {
                "object_id": oid,
                "action": "UNDETECTED_PRESUMED_VISIBLE",
                "visual_enter": info.visual_enter,
                "visual_exit": info.visual_exit,
            }
            for oid, info in visibility_windows.items()
            if getattr(info, "presumed_visible", False)
        ]


        try:
            logger.info("[Stage H] build & save output ...")
            corrected_scene_dict = self._build_corrected_scene(
                final_obj_list, scene, drift_report, duration
            )
            self._save_output(
                output_dir,
                corrected_scene_dict,
                drift_report,
                visibility_windows,
                split_actions,
                final_obj_list,
                scene,
                diagnostics={
                    "azimuth_sign": azimuth_diagnostics,
                    "presumed_visible": presumed_visible_warnings,
                },
            )
        except Exception as exc:
            logger.warning("[Stage H] failed: %s", exc, exc_info=True)


        tracking_video_path = output_dir / "tracking_video.mp4"
        try:
            logger.info("[Stage I] generating tracking video → %s", tracking_video_path)
            from .visualizer import Visualizer

            viz = Visualizer(self.config)


            stage_i_objects = scene.get("objects", []) if isinstance(scene, dict) else []

            viz.generate_annotated_video(
                video_path=video_path,
                output_path=tracking_video_path,
                depth_maps=depth_maps_list if depth_maps_list else [],
                depth_frame_indices=depth_frame_indices if depth_frame_indices else [],
                detections=detections if detections else {},
                objects=stage_i_objects,
                camera_trajectory=[],  
            )


            if tracking_video_path.is_file():
                size_mb = tracking_video_path.stat().st_size / (1024 * 1024)
                if size_mb < 0.001:
                    logger.warning(
                        "[Stage I] tracking video file is nearly empty (%.4f MB), "
                        "ffmpeg encoding may have failed: %s",
                        size_mb, tracking_video_path,
                    )
                else:
                    logger.info(
                        "[Stage I] tracking video saved: %s (%.1f MB)",
                        tracking_video_path, size_mb,
                    )
            else:
                logger.warning(
                    "[Stage I] tracking video file was NOT created: %s",
                    tracking_video_path,
                )
        except Exception as exc:
            logger.warning(
                "[Stage I] tracking video generation failed: %s",
                exc,
                exc_info=True,
            )

        report = self._generate_report(
            scene_id,
            scene,
            drift_report,
            visibility_windows,
            final_obj_list,
            split_actions,
            diagnostics={
                "azimuth_sign": azimuth_diagnostics,
                "presumed_visible": presumed_visible_warnings,
            },
        )
        self.last_report = report
        return report

    # ===================================================== Stage A
    def _verify_static_camera(self, video_path: Path) -> DriftReport:
        """OmniEcho processing component."""

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return DriftReport(
                is_static=True,
                mean_drift_px=0.0,
                max_drift_px=0.0,
                n_frames_sampled=0,
                threshold_px=self.STATIC_DRIFT_THRESHOLD_PX,
            )

        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total < 2:
                return DriftReport(
                    is_static=True,
                    mean_drift_px=0.0,
                    max_drift_px=0.0,
                    n_frames_sampled=total,
                    threshold_px=self.STATIC_DRIFT_THRESHOLD_PX,
                )
            target = 60
            step = max(1, total // target)

            grays: list[np.ndarray] = []
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % step == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    if gray.shape[1] > 640:
                        scale = 640.0 / gray.shape[1]
                        gray = cv2.resize(
                            gray, (640, int(gray.shape[0] * scale))
                        )
                    grays.append(gray)
                idx += 1
                if len(grays) >= target:
                    break
        finally:
            cap.release()

        if len(grays) < 2:
            return DriftReport(
                is_static=True,
                mean_drift_px=0.0,
                max_drift_px=0.0,
                n_frames_sampled=len(grays),
                threshold_px=self.STATIC_DRIFT_THRESHOLD_PX,
            )

        drifts: list[float] = []
        for i in range(1, len(grays)):
            flow = cv2.calcOpticalFlowFarneback(
                grays[i - 1], grays[i], None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mx = float(np.median(flow[..., 0]))
            my = float(np.median(flow[..., 1]))
            drifts.append(math.hypot(mx, my))

        mean_drift = float(np.mean(drifts))
        max_drift = float(np.max(drifts))
        is_static = mean_drift <= self.STATIC_DRIFT_THRESHOLD_PX
        if not is_static:
            logger.warning(
                "Camera not strictly static: mean_drift=%.2fpx/frame > %.1fpx; "
                "continuing with the static-camera assumption used by OmniEcho",
                mean_drift,
                self.STATIC_DRIFT_THRESHOLD_PX,
            )
        return DriftReport(
            is_static=is_static,
            mean_drift_px=mean_drift,
            max_drift_px=max_drift,
            n_frames_sampled=len(grays),
            threshold_px=self.STATIC_DRIFT_THRESHOLD_PX,
            compensation=None,
        )

    # ===================================================== Stage B
    def _detect_objects_sam3(
        self, video_path: Path, scene: dict
    ) -> tuple[dict[str, list[FrameDetection]], dict[str, str], dict]:
        """OmniEcho processing component."""

        scene_objects = scene.get("objects", [])
        visual_objects = [obj for obj in scene_objects if is_visual_object(obj)]
        label_map: dict[str, str] = {}
        for obj in visual_objects:
            label_map[obj["id"]] = label_for_segmentor(obj)
        unique_labels = list(dict.fromkeys(label_map.values()))

        video_meta = self._probe_video(video_path)
        video_meta["use_track_id"] = False

        if not unique_labels:
            return {}, label_map, video_meta

        detections: dict[str, list[FrameDetection]] = {}
        if getattr(self.config, "use_video_tracking", True):
            try:
                detections = self.segmentor.segment_objects_video(
                    str(video_path),
                    unique_labels,
                    video_tracker_conf=getattr(
                        self.config, "video_tracker_conf", 0.50
                    ),
                    min_masklet_length=getattr(
                        self.config, "min_masklet_length", 3
                    ),
                )
                has_track_ids = any(
                    d.track_id is not None
                    for dets in detections.values()
                    for d in dets
                )
                if has_track_ids:
                    video_meta["use_track_id"] = True
                    logger.info("Stage B: video-level tracking succeeded")
                else:
                    logger.info("Stage B: video-level tracking returned no track_id; falling back to frame-by-frame tracking")
                    detections = {}
            except Exception as exc:
                logger.warning("Stage B: video-level tracking failed: %s", exc)
                detections = {}

        if not detections:
            detections = self.segmentor.segment_objects(
                str(video_path), unique_labels, sample_fps=self.config.sample_fps
            )
            video_meta["use_track_id"] = False

        for label in unique_labels:
            n = len(detections.get(label, []))
            logger.info("  [%s] %d detections", label, n)
        return detections, label_map, video_meta

    def _probe_video(self, video_path: Path) -> dict:
        """OmniEcho processing component."""

        cap = cv2.VideoCapture(str(video_path))
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or self.config.video_fps)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self.config.frame_width)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.config.frame_height)
        finally:
            cap.release()
        if fps <= 0:
            fps = float(self.config.video_fps)
        return {
            "fps": fps,
            "total_frames": total,
            "width": width,
            "height": height,
        }

    # ===================================================== Stage C
    def _compute_visibility_windows(
        self,
        detections: dict[str, list[FrameDetection]],
        label_map: dict[str, str],
        video_meta: dict,
        duration: float,
        scene: Optional[dict] = None,
    ) -> dict[str, VisibilityInfo]:
        """OmniEcho processing component."""

        fps = float(video_meta.get("fps", self.config.video_fps))
        info_map: dict[str, VisibilityInfo] = {}

        scene_objs_by_id: dict[str, dict] = {}
        if scene is not None:
            scene_objs_by_id = {
                obj["id"]: obj for obj in scene.get("objects", []) if "id" in obj
            }


        all_frames: set[int] = set()
        for dets in detections.values():
            for d in dets:
                all_frames.add(int(d.frame_idx))
        n_total_frames = len(all_frames) if all_frames else int(round(duration * fps))
        if n_total_frames <= 0:
            n_total_frames = 1

        for object_id, label in label_map.items():
            obj_dets = detections.get(label, [])
            if not obj_dets:
                
                orig_obj = scene_objs_by_id.get(object_id, {})
                orig_vis = orig_obj.get("visibility_window") or {}
                try:
                    orig_enter = float(orig_vis.get("video_enter_t", -1.0))
                except (TypeError, ValueError):
                    orig_enter = -1.0
                try:
                    orig_exit = float(orig_vis.get("video_exit_t", duration))
                except (TypeError, ValueError):
                    orig_exit = duration
                if orig_enter >= 0:
                    logger.warning(
                        "[%s] Director marks the source visible (enter=%.2fs, exit=%.2fs), but SAM3 found no detection; "
                        "presuming visibility (UNDETECTED_PRESUMED_VISIBLE, confidence=0.3)",
                        object_id,
                        orig_enter,
                        orig_exit,
                    )
                    info_map[object_id] = VisibilityInfo(
                        state=VisibilityState.ON_SCREEN,
                        visual_enter=round(max(0.0, orig_enter), 2),
                        visual_exit=round(min(duration, orig_exit), 2),
                        n_frames_total=n_total_frames,
                        presumed_visible=True,
                    )
                else:
                    info_map[object_id] = VisibilityInfo(
                        state=VisibilityState.OFF_SCREEN_ALWAYS,
                        visual_enter=-1.0,
                        visual_exit=-1.0,
                        n_frames_total=n_total_frames,
                    )
                continue

            frames = sorted({int(d.frame_idx) for d in obj_dets})
            first_f = frames[0]
            last_f = frames[-1]
            visual_enter = round(first_f / fps, 2)
            visual_exit = round(last_f / fps, 2)
            n_detected = len(frames)
            coverage = n_detected / max(1, n_total_frames)

            state: VisibilityState
            if coverage >= 0.8:
                state = VisibilityState.ON_SCREEN
            elif visual_exit < duration * 0.7 and visual_enter <= duration * 0.3:
                state = VisibilityState.OFF_SCREEN_DYNAMIC
            elif visual_enter > duration * 0.3:
                state = VisibilityState.ENTERING
            else:
                state = VisibilityState.ON_SCREEN

            info_map[object_id] = VisibilityInfo(
                state=state,
                visual_enter=visual_enter,
                visual_exit=visual_exit,
                first_seen_frame=first_f,
                last_seen_frame=last_f,
                n_frames_detected=n_detected,
                n_frames_total=n_total_frames,
            )
            logger.info(
                "  [%s] state=%s enter=%.2fs exit=%.2fs coverage=%.1f%%",
                object_id,
                state.value,
                visual_enter,
                visual_exit,
                coverage * 100,
            )
        return info_map

    # ===================================================== Stage D
    def _estimate_depth(
        self,
        video_path: Path,
        detections: dict[str, list[FrameDetection]],
        label_map: dict[str, str],
        video_meta: dict,
    ) -> tuple[
        dict[str, list[tuple[float, float]]],
        list[np.ndarray],
        list[int],
    ]:
        """OmniEcho processing component."""

        if not detections or not label_map:
            return {}, [], []


        frame_to_count: dict[int, int] = defaultdict(int)
        for dets in detections.values():
            for d in dets:
                frame_to_count[int(d.frame_idx)] += 1
        if not frame_to_count:
            return {}, [], []

        key_frames = sorted(
            frame_to_count.keys(), key=lambda f: -frame_to_count[f]
        )[:3]
        key_frames = sorted(key_frames)

        try:
            depth_est = self.depth_estimator
        except Exception as exc:
            logger.warning("DepthEstimator init failed: %s", exc)
            return {}, [], []

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {}, [], []
        try:
            fps = float(video_meta.get("fps", self.config.video_fps))
            depth_maps: dict[int, np.ndarray] = {}
            kf_set = set(key_frames)
            idx = 0
            max_kf = max(key_frames)
            while True:
                ok, frame = cap.read()
                if not ok or idx > max_kf:
                    break
                if idx in kf_set:
                    try:
                        depth_maps[idx] = depth_est.estimate_frame(frame)
                    except Exception as exc:
                        logger.debug("depth est at frame %d failed: %s", idx, exc)
                idx += 1
        finally:
            cap.release()


        ordered_indices = sorted(depth_maps.keys())
        depth_maps_list = [depth_maps[i] for i in ordered_indices]
        depth_frame_indices = list(ordered_indices)

        if not depth_maps:
            return {}, depth_maps_list, depth_frame_indices


        depth_per_obj: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for object_id, label in label_map.items():
            for d in detections.get(label, []):
                fidx = int(d.frame_idx)
                if fidx not in depth_maps:
                    continue
                cx, cy = d.center
                dm = depth_maps[fidx]
                h, w = dm.shape[:2]
                ix = int(np.clip(cx, 0, w - 1))
                iy = int(np.clip(cy, 0, h - 1))
                rel_depth = float(dm[iy, ix])
                t = fidx / fps
                depth_per_obj[object_id].append((round(t, 3), rel_depth))
        return dict(depth_per_obj), depth_maps_list, depth_frame_indices

    # ===================================================== Stage E
    def _correct_visible_trajectories(
        self,
        detections: dict[str, list[FrameDetection]],
        label_map: dict[str, str],
        visibility_windows: dict[str, VisibilityInfo],
        depth_estimates: dict[str, list[tuple[float, float]]],
        scene: dict,
        video_meta: dict,
        duration: float,
    ) -> dict[str, list[AnnotatedTrajectoryPoint]]:
        """OmniEcho processing component."""

        fps = float(video_meta.get("fps", self.config.video_fps))
        width = int(video_meta.get("width", self.config.frame_width))
        height = int(video_meta.get("height", self.config.frame_height))
        fov_h = float(self.config.fov_horizontal_deg)

        scene_objs_by_id = {
            obj["id"]: obj for obj in scene.get("objects", [])
        }
        result: dict[str, list[AnnotatedTrajectoryPoint]] = {}

        for object_id, label in label_map.items():
            info = visibility_windows.get(object_id)
            if info is None or info.state == VisibilityState.OFF_SCREEN_ALWAYS:
                continue

            obj_dets = detections.get(label, [])
            if not obj_dets:
                continue


            frames_to_dets: dict[int, list[FrameDetection]] = defaultdict(list)
            for d in obj_dets:
                frames_to_dets[int(d.frame_idx)].append(d)

            times: list[float] = []
            azs: list[float] = []
            els: list[float] = []
            confs: list[float] = []
            prev_center: Optional[tuple[float, float]] = None
            for fidx in sorted(frames_to_dets.keys()):
                t = fidx / fps
                if t > duration + 0.1:
                    continue
                dets_here = frames_to_dets[fidx]
                if len(dets_here) > 1 and prev_center is not None:
                    distances = [
                        math.hypot(d.center[0] - prev_center[0], d.center[1] - prev_center[1])
                        for d in dets_here
                    ]
                    det = dets_here[int(np.argmin(distances))]
                elif len(dets_here) > 1:
                    det = max(dets_here, key=lambda d: float(d.score))
                else:
                    det = dets_here[0]
                cx, cy = det.center
                prev_center = (float(cx), float(cy))
                az, el = _inverse_project(cx, cy, width, height, fov_h)
                times.append(t)
                azs.append(az)
                els.append(el)
                confs.append(float(det.score))

            if not times:
                continue


            az_arr = _smooth_trajectory(np.asarray(azs), window=11, poly=3)
            el_arr = _smooth_trajectory(np.asarray(els), window=11, poly=3)


            orig_obj = scene_objs_by_id.get(object_id, {})
            orig_dists = [
                float(p.get("distance", 3.0))
                for p in orig_obj.get("trajectory", [])
            ]
            base_dist = float(np.mean(orig_dists)) if orig_dists else 3.0
            depth_pts = depth_estimates.get(object_id, [])
            if depth_pts:
                depths = np.asarray([d for _, d in depth_pts])
                dmin = float(depths.min())
                dmax = float(depths.max())
                
                if dmax > dmin:
                    norm = (depths - dmin) / (dmax - dmin)
                    scaled = base_dist * (0.7 + 0.6 * (1.0 - norm))
                else:
                    scaled = np.full_like(depths, base_dist)
                d_times = np.asarray([t for t, _ in depth_pts])
                if len(d_times) >= 2:
                    d_interp = interp1d(
                        d_times, scaled, kind="linear", fill_value="extrapolate"
                    )
                    distances = [float(max(0.3, float(d_interp(t)))) for t in times]
                else:
                    distances = [float(scaled[0])] * len(times)
            else:
                distances = [base_dist] * len(times)


            try:
                distances = self._smooth_distance(
                    distances, az_arr.tolist(), el_arr.tolist()
                )
            except Exception as exc:
                logger.debug("[%s] distance smoothing failed: %s", object_id, exc)


            points = _resample_trajectory(
                times,
                az_arr.tolist(),
                el_arr.tolist(),
                distances,
                confs,
                state=info.state if info.state != VisibilityState.OFF_SCREEN_ALWAYS
                else VisibilityState.ON_SCREEN,
                duration=info.visual_exit if info.visual_exit > 0 else duration,
                interval=0.25,
            )
            result[object_id] = points
        return result

    # ===================================================== Stage F
    def _infer_offscreen_trajectories(
        self,
        corrected: dict[str, list[AnnotatedTrajectoryPoint]],
        visibility_windows: dict[str, VisibilityInfo],
        scene: dict,
        duration: float,
    ) -> dict[str, AnnotatedAudioObject]:
        """OmniEcho processing component."""

        fov_h = float(self.config.fov_horizontal_deg)
        half_fov = fov_h / 2.0
        scene_objs_by_id = {obj["id"]: obj for obj in scene.get("objects", [])}
        out: dict[str, AnnotatedAudioObject] = {}


        all_ids = list(scene_objs_by_id.keys())

        for object_id in all_ids:
            orig_obj = scene_objs_by_id[object_id]
            label = str(orig_obj.get("type") or orig_obj.get("label") or object_id)
            acoustic_start = float(orig_obj.get("start_time", 0.0))
            acoustic_end = float(
                orig_obj.get("end_time", duration)
                if orig_obj.get("end_time") is not None
                else duration
            )
            spread_deg = float(orig_obj.get("spread_deg", 10.0))
            gain_db_val = float(orig_obj.get("gain_db", 0.0))

            info = visibility_windows.get(object_id)
            visible_pts = corrected.get(object_id, [])

            if info is None or info.state == VisibilityState.OFF_SCREEN_ALWAYS:
                
                points = self._build_trust_metadata_points(
                    orig_obj, duration, half_fov
                )
                visual_enter = -1.0
                visual_exit = -1.0
                strategy = OffScreenStrategy.TRUST_METADATA
            elif info.state in (VisibilityState.ON_SCREEN, VisibilityState.ENTERING):
                
                if visible_pts:
                    points = list(visible_pts)
                    strategy = OffScreenStrategy.STATIC_POSITION
                elif getattr(info, "presumed_visible", False):
                    
                    points = self._build_presumed_visible_points(orig_obj, duration)
                    strategy = OffScreenStrategy.STATIC_POSITION
                else:
                    points = self._build_trust_metadata_points(
                        orig_obj, duration, half_fov
                    )
                    strategy = OffScreenStrategy.TRUST_METADATA
                visual_enter = info.visual_enter
                visual_exit = info.visual_exit
            elif info.state == VisibilityState.OFF_SCREEN_DYNAMIC:
                if not visible_pts:
                    points = self._build_trust_metadata_points(
                        orig_obj, duration, half_fov
                    )
                    strategy = OffScreenStrategy.TRUST_METADATA
                else:
                    extra = _extrapolate_trajectory(
                        visible_pts, duration, sample_interval=0.25, fov_h_deg=fov_h
                    )
                    points = list(visible_pts) + extra
                    strategy = OffScreenStrategy.EXTRAPOLATE
                visual_enter = info.visual_enter
                visual_exit = info.visual_exit
            else:  
                points = list(visible_pts) if visible_pts else self._build_trust_metadata_points(
                    orig_obj, duration, half_fov
                )
                strategy = OffScreenStrategy.STATIC_POSITION
                visual_enter = info.visual_enter
                visual_exit = info.visual_exit

            if not points:
                points = self._build_trust_metadata_points(
                    orig_obj, duration, half_fov
                )
                strategy = OffScreenStrategy.TRUST_METADATA
                visual_enter = -1.0
                visual_exit = -1.0

            out[object_id] = AnnotatedAudioObject(
                object_id=object_id,
                label=label,
                trajectory=tuple(points),
                acoustic_start=acoustic_start,
                acoustic_end=acoustic_end,
                visual_enter=float(visual_enter),
                visual_exit=float(visual_exit),
                off_screen_strategy=strategy,
                spread_deg=spread_deg,
                gain_db=gain_db_val,
            )
        return out

    def _build_trust_metadata_points(
        self, orig_obj: dict, duration: float, half_fov: float
    ) -> list[AnnotatedTrajectoryPoint]:
        """OmniEcho processing component."""

        traj = orig_obj.get("trajectory") or []
        if not traj:
            return [
                AnnotatedTrajectoryPoint(
                    time=0.0,
                    azimuth=float(half_fov + 5.0),
                    elevation=0.0,
                    distance=3.0,
                    gain_db=0.0,
                    visibility_state=VisibilityState.OFF_SCREEN_ALWAYS,
                    confidence=self.OFFSCREEN_ALWAYS_CONFIDENCE,
                )
            ]

        times = [float(p.get("t", 0.0)) for p in traj]
        azs = [float(p.get("azimuth", 0.0)) for p in traj]
        els = [float(p.get("elevation", 0.0)) for p in traj]
        dists = [float(p.get("distance", 3.0)) for p in traj]
        confs = [self.OFFSCREEN_ALWAYS_CONFIDENCE] * len(traj)


        bound = half_fov + 5.0
        adj_azs: list[float] = []
        for az in azs:
            if abs(az) <= half_fov:
                az = math.copysign(bound, az if az != 0 else 1.0)
            adj_azs.append(_wrap_deg(az))

        return _resample_trajectory(
            times,
            adj_azs,
            els,
            dists,
            confs,
            state=VisibilityState.OFF_SCREEN_ALWAYS,
            duration=duration,
            interval=0.25,
        )

    def _build_presumed_visible_points(
        self, orig_obj: dict, duration: float
    ) -> list[AnnotatedTrajectoryPoint]:
        """OmniEcho processing component."""

        traj = orig_obj.get("trajectory") or []
        if not traj:
            return [
                AnnotatedTrajectoryPoint(
                    time=0.0,
                    azimuth=0.0,
                    elevation=0.0,
                    distance=3.0,
                    gain_db=0.0,
                    visibility_state=VisibilityState.ON_SCREEN,
                    confidence=0.3,
                )
            ]
        times = [float(p.get("t", 0.0)) for p in traj]
        azs = [float(p.get("azimuth", 0.0)) for p in traj]
        els = [float(p.get("elevation", 0.0)) for p in traj]
        dists = [float(p.get("distance", 3.0)) for p in traj]
        confs = [0.3] * len(traj)
        return _resample_trajectory(
            times,
            azs,
            els,
            dists,
            confs,
            state=VisibilityState.ON_SCREEN,
            duration=duration,
            interval=0.25,
        )


    def _smooth_distance(
        self,
        distances: list[float],
        azimuths: list[float],
        elevations: list[float],
    ) -> list[float]:
        """OmniEcho processing component."""

        if not distances:
            return distances
        d_arr = np.asarray(distances, dtype=np.float64)
        if len(d_arr) < 3:
            return distances
        az_arr = np.asarray(azimuths, dtype=np.float64) if azimuths else None
        el_arr = np.asarray(elevations, dtype=np.float64) if elevations else None
        if (
            az_arr is not None
            and el_arr is not None
            and len(az_arr) == len(d_arr)
            and len(el_arr) == len(d_arr)
            and float(np.std(az_arr)) < 2.0
            and float(np.std(el_arr)) < 2.0
        ):
            median_dist = float(np.median(d_arr))
            logger.info(
                "  distance smoothing: static object → median=%.3fm (n=%d)",
                median_dist,
                len(d_arr),
            )
            return [median_dist] * len(d_arr)
        try:
            from scipy.ndimage import median_filter

            window = min(5, len(d_arr) if len(d_arr) % 2 == 1 else len(d_arr) - 1)
            window = max(3, window)
            smoothed = median_filter(d_arr, size=window, mode="nearest")
            return [float(x) for x in smoothed]
        except Exception as exc:
            logger.debug("median_filter failed: %s", exc)
            return distances

    def _check_azimuth_sign_consistency(
        self,
        corrected_objects: dict[str, list[AnnotatedTrajectoryPoint]],
        original_scene: dict,
    ) -> dict:
        """OmniEcho processing component."""

        sign_flips: list[dict] = []
        orig_objs = {
            o["id"]: o for o in original_scene.get("objects", []) if "id" in o
        }
        for obj_id, traj_points in corrected_objects.items():
            if not traj_points:
                continue
            try:
                corrected_az_mean = float(
                    np.mean([float(p.azimuth) for p in traj_points])
                )
            except Exception:
                continue
            orig_obj = orig_objs.get(obj_id)
            if not orig_obj:
                continue
            orig_traj = orig_obj.get("trajectory") or []
            if not orig_traj:
                continue
            try:
                orig_az_mean = float(
                    np.mean([float(t.get("azimuth", 0.0)) for t in orig_traj])
                )
            except Exception:
                continue
            if abs(orig_az_mean) > 10.0 and abs(corrected_az_mean) > 10.0:
                if (orig_az_mean > 0) != (corrected_az_mean > 0):
                    sign_flips.append(
                        {
                            "object_id": obj_id,
                            "original_az_mean": round(orig_az_mean, 1),
                            "corrected_az_mean": round(corrected_az_mean, 1),
                        }
                    )
        return {
            "azimuth_sign_flip_count": len(sign_flips),
            "flipped_objects": sign_flips,
            "warning": len(sign_flips) >= 3,
        }

    def _validate_trusted_direction(
        self,
        obj: AnnotatedAudioObject,
        sample_dir: Path,
        video_meta: dict,
    ) -> Optional[dict]:
        """OmniEcho processing component."""

        obj_id = getattr(obj, "object_id", "") or ""
        if not obj_id:
            return None
        sub_video_path = sample_dir / "sub_videos" / f"{obj_id}.mp4"
        if not sub_video_path.is_file():
            logger.debug(
                "[direction_validate] sub-video not found for %s, skip", obj_id
            )
            return None

        try:
            cap = cv2.VideoCapture(str(sub_video_path))
            if not cap.isOpened():
                return None
            try:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if total_frames < 2 or width < 10:
                    return None


                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret1, frame1 = cap.read()
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 2))
                ret2, frame2 = cap.read()
            finally:
                cap.release()

            if not ret1 or not ret2 or frame1 is None or frame2 is None:
                return None

            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY).astype(np.float32)
            diff = np.abs(gray2 - gray1)

            mid_x = max(1, width // 2)
            left_energy = float(np.sum(diff[:, :mid_x]))
            right_energy = float(np.sum(diff[:, mid_x:]))
            total_energy = left_energy + right_energy
            if total_energy < 1e-3:
                
                return None
            left_ratio = left_energy / total_energy

            traj = list(getattr(obj, "trajectory", ()) or ())
            if not traj:
                return None
            mid_idx = len(traj) // 2
            director_az = float(getattr(traj[mid_idx], "azimuth", 0.0))


            
            video_suggests_left = left_ratio > 0.6
            video_suggests_right = left_ratio < 0.4

            direction_conflict = False
            suggested_sign = ""
            if video_suggests_left and director_az < -20:
                direction_conflict = True
                suggested_sign = "positive (left)"
            elif video_suggests_right and director_az > 20:
                direction_conflict = True
                suggested_sign = "negative (right)"

            if direction_conflict:
                logger.warning(
                    "[direction_validate] %s: sub-video motion contradicts Director "
                    "azimuth (%.1f degrees). Image motion favors %s (left_ratio=%.2f); "
                    "suggested azimuth sign: %s. Retaining the original value and flagging a warning.",
                    obj_id,
                    director_az,
                    "left" if video_suggests_left else "right",
                    left_ratio,
                    suggested_sign,
                )
                return {
                    "object_id": obj_id,
                    "conflict": True,
                    "left_ratio": round(left_ratio, 3),
                    "director_azimuth": round(director_az, 1),
                    "video_suggests": (
                        "left" if video_suggests_left else "right"
                    ),
                    "suggested_sign": suggested_sign,
                }
            logger.info(
                "[direction_validate] %s: direction check passed (az=%.1f degrees, "
                "left_ratio=%.2f)",
                obj_id,
                director_az,
                left_ratio,
            )
            return None
        except Exception as exc:
            logger.debug(
                "[direction_validate] %s: nonblocking validation error: %s", obj_id, exc
            )
            return None

    def _extend_trajectory_to_duration(
        self,
        trajectory: list[AnnotatedTrajectoryPoint],
        duration: float,
    ) -> list[AnnotatedTrajectoryPoint]:
        """OmniEcho processing component."""

        if not trajectory:
            return trajectory
        last_point = trajectory[-1]
        if float(last_point.time) < float(duration) - 0.01:
            extended = list(trajectory)
            extended.append(
                AnnotatedTrajectoryPoint(
                    time=round(float(duration), 2),
                    azimuth=last_point.azimuth,
                    elevation=last_point.elevation,
                    distance=last_point.distance,
                    gain_db=last_point.gain_db,
                    visibility_state=last_point.visibility_state,
                    confidence=round(
                        float(np.clip(last_point.confidence * 0.9, 0.0, 1.0)), 2
                    ),
                )
            )
            return extended
        return trajectory

    # ===================================================== Stage G
    def _check_quantity_and_split(
        self,
        final_objects: dict[str, AnnotatedAudioObject],
        detections: dict[str, list[FrameDetection]],
        label_map: dict[str, str],
        scene: dict,
        video_meta: dict,
        duration: float,
    ) -> tuple[list[AnnotatedAudioObject], list[dict]]:
        """OmniEcho processing component."""


        use_track = bool(video_meta.get("use_track_id", False))
        min_coverage = float(getattr(self.config, "min_track_coverage", 0.15))

        label_to_n_instances: dict[str, int] = {}
        label_to_track_groups: dict[str, dict[int, list[FrameDetection]]] = {}
        for label, dets in detections.items():
            if not dets:
                label_to_n_instances[label] = 0
                continue
            if use_track:
                track_frames: dict[int, set[int]] = defaultdict(set)
                track_dets: dict[int, list[FrameDetection]] = defaultdict(list)
                for d in dets:
                    if d.track_id is None:
                        continue
                    track_frames[d.track_id].add(int(d.frame_idx))
                    track_dets[d.track_id].append(d)
                if not track_frames:
                    label_to_n_instances[label] = 1
                    continue
                all_frames = {int(d.frame_idx) for d in dets}
                total = max(1, max(all_frames) - min(all_frames) + 1)
                min_required = max(3, int(total * min_coverage))
                valid = {
                    tid: track_dets[tid]
                    for tid, frames in track_frames.items()
                    if len(frames) >= min_required
                }
                label_to_track_groups[label] = valid
                if not valid:
                    label_to_n_instances[label] = 1
                else:
                    
                    frame_to_active: dict[int, set[int]] = defaultdict(set)
                    for tid, dlist in valid.items():
                        for d in dlist:
                            frame_to_active[int(d.frame_idx)].add(tid)
                    counts = sorted(len(s) for s in frame_to_active.values())
                    label_to_n_instances[label] = max(1, int(np.median(counts)))
            else:
                label_to_n_instances[label] = 1  


        label_to_metadata_count: dict[str, list[str]] = defaultdict(list)
        for object_id, label in label_map.items():
            label_to_metadata_count[label].append(object_id)

        final_list: list[AnnotatedAudioObject] = []
        split_actions: list[dict] = []


        processed: set[str] = set()

        for label, object_ids in label_to_metadata_count.items():
            n_detected = label_to_n_instances.get(label, 0)
            n_meta = len(object_ids)
            if n_detected > n_meta and n_detected >= 2:
                
                base_id = object_ids[0]
                base_obj = final_objects.get(base_id)
                if base_obj is None:
                    continue
                track_groups = label_to_track_groups.get(label, {})
                if track_groups:
                    sub_ids = self._split_object_by_tracks(
                        base_obj,
                        track_groups,
                        n_detected,
                        video_meta,
                        duration,
                    )
                    if sub_ids:
                        final_list.extend(sub_ids)
                        split_actions.append(
                            {
                                "object_id": base_id,
                                "action": "SPLIT",
                                "n_meta": n_meta,
                                "n_detected": int(n_detected),
                                "sub_ids": [o.object_id for o in sub_ids],
                            }
                        )
                        processed.add(base_id)
                        
                        for oid in object_ids[1:]:
                            if oid in final_objects:
                                final_list.append(final_objects[oid])
                                processed.add(oid)
                        continue
                
                for oid in object_ids:
                    if oid in final_objects and oid not in processed:
                        final_list.append(final_objects[oid])
                        processed.add(oid)
            elif n_detected < n_meta:
                
                for i, oid in enumerate(object_ids):
                    obj = final_objects.get(oid)
                    if obj is None:
                        continue
                    if i >= n_detected:
                        obj = self._mark_object_offscreen(obj)
                        split_actions.append(
                            {
                                "object_id": oid,
                                "action": "MISSING",
                                "n_meta": n_meta,
                                "n_detected": int(n_detected),
                            }
                        )
                    final_list.append(obj)
                    processed.add(oid)
            else:
                for oid in object_ids:
                    if oid in final_objects and oid not in processed:
                        final_list.append(final_objects[oid])
                        processed.add(oid)


        for oid, obj in final_objects.items():
            if oid not in processed:
                final_list.append(obj)
        return final_list, split_actions

    def _split_object_by_tracks(
        self,
        base_obj: AnnotatedAudioObject,
        track_groups: dict[int, list[FrameDetection]],
        n_target: int,
        video_meta: dict,
        duration: float,
    ) -> list[AnnotatedAudioObject]:
        """OmniEcho processing component."""

        fps = float(video_meta.get("fps", self.config.video_fps))
        width = int(video_meta.get("width", self.config.frame_width))
        height = int(video_meta.get("height", self.config.frame_height))
        fov_h = float(self.config.fov_horizontal_deg)
        half_fov = fov_h / 2.0

        sorted_tracks = sorted(
            track_groups.items(), key=lambda kv: -len(kv[1])
        )[:n_target]
        if len(sorted_tracks) < 2:
            return []

        gain_offset = round(10.0 * math.log10(len(sorted_tracks)), 2)
        sub_objects: list[AnnotatedAudioObject] = []
        for i, (track_id, dets) in enumerate(sorted_tracks):
            dets_sorted = sorted(dets, key=lambda d: int(d.frame_idx))
            times: list[float] = []
            azs: list[float] = []
            els: list[float] = []
            confs: list[float] = []
            for d in dets_sorted:
                t = int(d.frame_idx) / fps
                if t > duration + 0.1:
                    continue
                az, el = _inverse_project(d.center[0], d.center[1], width, height, fov_h)
                times.append(t)
                azs.append(az)
                els.append(el)
                confs.append(float(d.score))
            if not times:
                continue
            az_arr = _smooth_trajectory(np.asarray(azs), window=11, poly=3)
            el_arr = _smooth_trajectory(np.asarray(els), window=11, poly=3)
            distances = [
                float(np.mean([p.distance for p in base_obj.trajectory]))
            ] * len(times)
            visible_exit = times[-1]
            points_visible = _resample_trajectory(
                times,
                az_arr.tolist(),
                el_arr.tolist(),
                distances,
                confs,
                state=VisibilityState.ON_SCREEN,
                duration=visible_exit,
                interval=0.25,
            )
            
            if points_visible and visible_exit < duration - 0.5:
                extra = _extrapolate_trajectory(
                    points_visible, duration, sample_interval=0.25, fov_h_deg=fov_h
                )
                points = points_visible + extra
                strategy = OffScreenStrategy.EXTRAPOLATE
            else:
                points = points_visible
                strategy = OffScreenStrategy.STATIC_POSITION


            adj_points = tuple(
                AnnotatedTrajectoryPoint(
                    time=p.time,
                    azimuth=p.azimuth,
                    elevation=p.elevation,
                    distance=p.distance,
                    gain_db=round(float(p.gain_db - gain_offset), 2),
                    visibility_state=p.visibility_state,
                    confidence=p.confidence,
                )
                for p in points
            )
            sub_objects.append(
                AnnotatedAudioObject(
                    object_id=f"{base_obj.object_id}_{i + 1}",
                    label=f"{base_obj.label}#{i + 1}",
                    trajectory=adj_points,
                    acoustic_start=base_obj.acoustic_start,
                    acoustic_end=base_obj.acoustic_end,
                    visual_enter=times[0],
                    visual_exit=visible_exit,
                    off_screen_strategy=strategy,
                    spread_deg=base_obj.spread_deg,
                    gain_db=base_obj.gain_db,
                )
            )
        return sub_objects if len(sub_objects) >= 2 else []

    def _mark_object_offscreen(self, obj: AnnotatedAudioObject) -> AnnotatedAudioObject:
        """OmniEcho processing component."""

        new_traj = tuple(
            AnnotatedTrajectoryPoint(
                time=p.time,
                azimuth=p.azimuth,
                elevation=p.elevation,
                distance=p.distance,
                gain_db=p.gain_db,
                visibility_state=VisibilityState.OFF_SCREEN_ALWAYS,
                confidence=self.OFFSCREEN_ALWAYS_CONFIDENCE,
            )
            for p in obj.trajectory
        )
        return AnnotatedAudioObject(
            object_id=obj.object_id,
            label=obj.label,
            trajectory=new_traj,
            acoustic_start=obj.acoustic_start,
            acoustic_end=obj.acoustic_end,
            visual_enter=-1.0,
            visual_exit=-1.0,
            off_screen_strategy=OffScreenStrategy.TRUST_METADATA,
            spread_deg=obj.spread_deg,
            gain_db=obj.gain_db,
        )

    # ===================================================== Stage H
    def _build_corrected_scene(
        self,
        objects: list[AnnotatedAudioObject],
        original_scene: dict,
        drift_report: DriftReport,
        duration: float,
    ) -> dict:
        """OmniEcho processing component."""

        scene = copy.deepcopy(original_scene)
        scene["scene_id"] = original_scene.get("scene_id", "omniecho_scene")
        scene["camera_mode"] = CameraMode.STATIC.value
        scene["fov_horizontal_deg"] = float(self.config.fov_horizontal_deg)
        scene["frame_width"] = int(self.config.frame_width)
        scene["frame_height"] = int(self.config.frame_height)
        scene["duration_sec"] = float(duration)
        scene["annotation_metadata"] = {
            "pipeline": "OmniEchoBench-QA",
            "drift": {
                "is_static": drift_report.is_static,
                "mean_drift_px": drift_report.mean_drift_px,
                "max_drift_px": drift_report.max_drift_px,
            },
        }

        scene["objects"] = [_audio_object_to_dict(o) for o in objects]


        listener = scene.get("listener", {})
        listener["orientation"] = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
        scene["listener"] = listener
        return scene

    def _save_output(
        self,
        output_dir: Path,
        corrected_scene: dict,
        drift_report: DriftReport,
        visibility_windows: dict[str, VisibilityInfo],
        split_actions: list[dict],
        objects: list[AnnotatedAudioObject],
        original_scene: dict,
        diagnostics: Optional[dict] = None,
    ) -> None:
        """OmniEcho processing component."""

        output_dir.mkdir(parents=True, exist_ok=True)
        scene_out = output_dir / "corrected_scene.json"
        scene_out.write_text(json.dumps(corrected_scene, indent=2, ensure_ascii=False))


        per_object_actions: list[dict] = []
        action_lookup = {a.get("object_id"): a for a in split_actions}


        orig_ids = {o.get("id") for o in original_scene.get("objects", [])}


        final_ids = {o.object_id for o in objects}

        for obj in objects:
            base_id = obj.object_id.rsplit("_", 1)[0] if "_" in obj.object_id else obj.object_id
            base_action = action_lookup.get(base_id) or action_lookup.get(obj.object_id)
            info = visibility_windows.get(base_id) or visibility_windows.get(obj.object_id)
            action_label = "CORRECTED"
            if base_action and base_action.get("action") == "SPLIT":
                action_label = "SPLIT"
            elif base_action and base_action.get("action") == "MISSING":
                action_label = "MISSING"
            elif obj.off_screen_strategy == OffScreenStrategy.TRUST_METADATA:
                action_label = "TRUSTED"
            elif obj.off_screen_strategy == OffScreenStrategy.EXTRAPOLATE:
                action_label = "EXTRAPOLATED"
            per_object_actions.append(
                {
                    "object_id": obj.object_id,
                    "label": obj.label,
                    "action": action_label,
                    "visibility_state": info.state.value if info else None,
                    "visual_enter": float(obj.visual_enter),
                    "visual_exit": float(obj.visual_exit),
                    "n_traj_points": len(obj.trajectory),
                    "off_screen_strategy": obj.off_screen_strategy.value,
                }
            )


        split_base_ids = {a.get("object_id") for a in split_actions if a.get("action") == "SPLIT"}
        for oid in orig_ids - final_ids - split_base_ids:
            per_object_actions.append(
                {"object_id": oid, "action": "MISSING_AT_OUTPUT"}
            )

        report = {
            "drift": {
                "is_static": drift_report.is_static,
                "mean_drift_px": drift_report.mean_drift_px,
                "max_drift_px": drift_report.max_drift_px,
                "n_frames_sampled": drift_report.n_frames_sampled,
                "threshold_px": drift_report.threshold_px,
            },
            "objects": per_object_actions,
            "summary": {
                "n_objects_input": len(orig_ids),
                "n_objects_output": len(objects),
                "n_split": sum(1 for a in split_actions if a.get("action") == "SPLIT"),
                "n_missing": sum(1 for a in split_actions if a.get("action") == "MISSING"),
            },
            "diagnostics": diagnostics or {},
        }
        (output_dir / "correction_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False)
        )
        logger.info(
            "[Stage H] corrected_scene.json + correction_report.json -> %s",
            output_dir,
        )

    # ===================================================== utilities
    def _load_scene(self, scene_path: Path) -> dict:
        return json.loads(scene_path.read_text())

    def _generate_report(
        self,
        scene_id: str,
        original_scene: dict,
        drift_report: DriftReport,
        visibility_windows: dict[str, VisibilityInfo],
        objects: list[AnnotatedAudioObject],
        split_actions: list[dict],
        diagnostics: Optional[dict] = None,
    ) -> dict:
        """OmniEcho processing component."""

        return {
            "scene_id": scene_id,
            "drift": asdict(drift_report),
            "visibility": {
                oid: {
                    "state": info.state.value,
                    "visual_enter": info.visual_enter,
                    "visual_exit": info.visual_exit,
                    "n_frames_detected": info.n_frames_detected,
                    "n_frames_total": info.n_frames_total,
                    "presumed_visible": getattr(info, "presumed_visible", False),
                }
                for oid, info in visibility_windows.items()
            },
            "objects": [
                {
                    "object_id": o.object_id,
                    "label": o.label,
                    "off_screen_strategy": o.off_screen_strategy.value,
                    "visual_enter": o.visual_enter,
                    "visual_exit": o.visual_exit,
                    "n_traj_points": len(o.trajectory),
                }
                for o in objects
            ],
            "split_actions": split_actions,
            "summary": {
                "n_objects_input": len(original_scene.get("objects", [])),
                "n_objects_output": len(objects),
                "n_split": sum(1 for a in split_actions if a.get("action") == "SPLIT"),
                "n_missing": sum(1 for a in split_actions if a.get("action") == "MISSING"),
            },
            "diagnostics": diagnostics or {},
        }


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

MetadataCorrector = MetadataCorrector


__all__ = [
    "MetadataCorrector",
    "MetadataCorrector",
    "VisibilityInfo",
    "DriftReport",
    "_inverse_project",
    "_smooth_trajectory",
    "_resample_trajectory",
    "_extrapolate_trajectory",
]
