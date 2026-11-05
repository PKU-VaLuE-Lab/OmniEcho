"""Video segmentation / detection wrapper for the visual validation pipeline.

This module implements :class:`VideoSegmentor`, a thin abstraction that
returns per-frame :class:`FrameDetection` records keyed by object label.

Backend
-------
SAM3 (Ultralytics ``SAM3SemanticPredictor`` / ``SAM3VideoSemanticPredictor``)
is the **only** detection backend.  The Ultralytics 8.4.61 distribution ships
native SAM3 support with a text-prompt classification head, which provides
high-confidence open-vocabulary segmentation on compound labels such as
``"blacksmith hammering"``.  Requires the SAM3 ``.pt`` weights and a working
``clip`` install.  We monkey-patch ``clip.simple_tokenizer.SimpleTokenizer``
to be callable so that the SAM3 text encoder can drive it directly.
"""

from __future__ import annotations

import logging
import sys
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


logger = logging.getLogger(__name__)


# Resolve the bundled-weights directory relative to the ``visual_validation``
# package so that callers do not need to know the full filesystem path.
_THIRD_PARTY_DIR = Path(__file__).resolve().parent.parent / "third_party"

# Default location of the SAM3 weights bundled with the upstream Facebook
# release.  Used when the caller did not supply a usable ``model_path``.
_DEFAULT_SAM3_WEIGHTS = Path(os.environ.get("OMNIECHO_SEGMENTATION_MODEL", "/path/to/segmentation-model.pt"))
_LOCAL_SAM3_WEIGHTS = _THIRD_PARTY_DIR / "sam3.pt"


# ----------------------------------------------------------------------------
# Compatibility shims for the openai/CLIP package required by SAM3.  The PyPI
# ``clip-anytorch`` build imports ``packaging`` via the deprecated
# ``pkg_resources.packaging`` shortcut which was removed in modern setuptools.
# We restore that attribute so that ``import clip`` succeeds.  In addition,
# the SAM3 text encoder calls ``self.tokenizer(text, ...)`` but
# ``clip.simple_tokenizer.SimpleTokenizer`` is *not* callable out of the box
# – we add a thin ``__call__`` proxy that delegates to ``clip.tokenize``.
# ----------------------------------------------------------------------------


def _patch_clip_pkg_resources() -> None:
    """Make ``import clip`` succeed under modern ``setuptools``.

    Idempotent.  Safe to call even when ``clip`` is not installed – we only
    register the compatibility attributes; nothing imports clip eagerly.
    """

    try:
        import packaging  # noqa: F401
        import packaging.version  # noqa: F401
    except Exception:  # pragma: no cover - packaging is a transitive of pip
        return
    sys.modules.setdefault("pkg_resources.packaging", sys.modules["packaging"])
    pkg_pkg = sys.modules["pkg_resources.packaging"]
    if not hasattr(pkg_pkg, "version"):
        pkg_pkg.version = sys.modules["packaging"].version  # type: ignore[attr-defined]


def _patch_clip_tokenizer() -> None:
    """Make ``clip.simple_tokenizer.SimpleTokenizer`` callable.

    The SAM3 text encoder (``ultralytics.models.sam.sam3.text_encoder_ve``)
    invokes ``self.tokenizer(text, context_length=...)`` but the upstream
    ``clip-anytorch`` ``SimpleTokenizer`` only exposes ``encode``/``decode``
    and is not callable.  We attach a ``__call__`` proxy that delegates to
    the module-level ``clip.tokenize`` helper, which performs full
    BPE+SOT/EOT tokenisation and returns an ``[N, context_length]`` long
    tensor – exactly what the SAM3 encoder expects.
    """

    _patch_clip_pkg_resources()
    try:
        import clip  # type: ignore
        from clip.simple_tokenizer import SimpleTokenizer  # type: ignore
    except Exception:  # pragma: no cover - clip is optional
        return

    if getattr(SimpleTokenizer, "_qoder_call_patched", False):
        return

    def _tokenizer_call(self, texts, context_length: int = 77, truncate: bool = True):
        return clip.tokenize(texts, context_length=context_length, truncate=truncate)

    SimpleTokenizer.__call__ = _tokenizer_call  # type: ignore[assignment]
    SimpleTokenizer._qoder_call_patched = True  # type: ignore[attr-defined]


# ----------------------------------------------------------------------------
# FrameDetection
# ----------------------------------------------------------------------------


@dataclass
class FrameDetection:
    """A single object detection in a single frame.

    Attributes
    ----------
    frame_idx:
        Zero-based index of the frame in the *original* video.
    mask:
        Optional ``H x W`` binary mask.  ``None`` when only the bounding box
        was retained.  The SAM3 backend keeps the mask when
        ``keep_masks=True``.
    bbox:
        ``(x1, y1, x2, y2)`` integer pixel bounding box.
    score:
        Detection confidence in ``[0, 1]``.
    """

    frame_idx: int
    mask: Optional[np.ndarray]
    bbox: Tuple[int, int, int, int]
    score: float
    track_id: Optional[int] = None  

    @property
    def center(self) -> Tuple[float, float]:
        """Return the bounding box centre as ``(cx, cy)`` floats."""

        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> int:
        """Return the bounding box area in pixels."""

        x1, y1, x2, y2 = self.bbox
        return max(0, int(x2 - x1)) * max(0, int(y2 - y1))


# ----------------------------------------------------------------------------
# VideoSegmentor
# ----------------------------------------------------------------------------


class VideoSegmentor:
    """Segment named objects across the frames of a video using SAM3.

    Parameters
    ----------
    model_path:
        Filesystem path to the SAM3 weights (``.pt``).  When the supplied
        path does not exist we fall back to ``visual_validation/third_party/
        sam3.pt`` and finally to the well-known internal default location.
    device:
        Torch device string (``cuda`` / ``cuda:0`` / ``cpu``).
    conf:
        Minimum detection score (probability) to keep.
    keep_masks:
        Whether to retain the per-detection segmentation masks.  Honoured
        by the SAM3 backend (which natively produces masks).
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "cuda",
        conf: float = 0.25,
        keep_masks: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.conf = float(conf)
        self.keep_masks = bool(keep_masks)
        self.last_error: Optional[str] = None

        self._backend: Optional[str] = None
        self._model = None
        # SAM3 caches the encoded text classes on the model itself; we
        # additionally remember the most recent label list so we can avoid
        # re-encoding when callers reuse the same set across frames.
        self._sam3_classes: Optional[List[str]] = None
        self._init_backend()

    # ------------------------------------------------------------------ backend
    def _resolve_sam3_weights(self) -> Optional[Path]:
        """Return the first plausible SAM3 weights file, or ``None``."""

        candidates: List[Path] = []
        if self.model_path and str(self.model_path).strip():
            candidates.append(self.model_path)
        candidates.append(_LOCAL_SAM3_WEIGHTS)
        candidates.append(_DEFAULT_SAM3_WEIGHTS)
        for p in candidates:
            try:
                if p.is_file() and p.suffix == ".pt":
                    return p
            except Exception:
                continue
        return None

    def _init_backend(self) -> None:
        """Initialise the SAM3 semantic predictor backend."""

        sam3_path = self._resolve_sam3_weights()
        if sam3_path is None:
            self._backend = None
            self._model = None
            logger.warning(
                "VideoSegmentor: SAM3 weights not found; visibility/depth/"
                "spatial checks will report empty detections."
            )
            return

        try:
            _patch_clip_tokenizer()
            from ultralytics.models.sam import (  # type: ignore
                SAM3SemanticPredictor,
            )

            overrides = dict(
                conf=self.conf,
                iou=0.6,  
                task="segment",
                mode="predict",
                model=str(sam3_path),
                half=True,
                save=False,
                verbose=False,
            )
            # ``device`` is consumed by Ultralytics overrides as well; only
            # forward it when callers explicitly asked for a non-empty
            # value to keep the predictor's auto-selection behaviour.
            if self.device:
                overrides["device"] = self.device
            self._model = SAM3SemanticPredictor(overrides=overrides)
            self._backend = "sam3_semantic"
            logger.info(
                "VideoSegmentor: using SAM3 semantic predictor (%s) on %s",
                sam3_path.name,
                self.device,
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning(
                "SAM3 semantic predictor load failed (%s); detections will be empty",
                exc,
            )
            self._backend = None
            self._model = None

    # --------------------------------------------------------------- public API
    def segment_objects(
        self,
        video_path: Union[str, Path],
        object_labels: List[str],
        sample_fps: float = 4.0,
    ) -> Dict[str, List[FrameDetection]]:
        """Detect ``object_labels`` across ``video_path`` using SAM3.

        Parameters
        ----------
        video_path:
            Path to the video file.
        object_labels:
            List of free-form text labels (e.g. ``["bicycle bell", "pigeon"]``)
            describing the objects of interest.
        sample_fps:
            How many frames per second to feed to the detector.

        Returns
        -------
        ``{label: [FrameDetection, ...]}``.  Each list is sorted by frame
        index.  Labels with zero detections still appear in the dictionary
        with an empty list to make downstream iteration uniform.
        """

        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"video not found: {video_path}")
        if not object_labels:
            return {}

        frames, frame_indices = self._sample_frames(video_path, sample_fps)
        if not frames:
            self.last_error = "no frames could be decoded from the video"
            return {label: [] for label in object_labels}

        if self._backend == "sam3_semantic" and self._model is not None:
            try:
                results = self._segment_sam3(
                    frames, frame_indices, object_labels
                )
                self.last_error = None
                return results
            except Exception as exc:
                self.last_error = f"sam3_semantic: {type(exc).__name__}: {exc}"
                logger.warning("SAM3 inference failed: %s", self.last_error)

        # No backend available.
        self.last_error = (
            "SAM3 detection backend unavailable; see segmentor logs for details"
        )
        logger.warning(self.last_error)
        return {label: [] for label in object_labels}

    # ------------------------------------------------------------ frame sampling
    @staticmethod
    def _sample_frames(
        video_path: Path,
        sample_fps: float,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Decode ``video_path`` and return ``(frames, frame_indices)``."""

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
            wanted = list(range(0, total, step))
            if total - 1 not in wanted:
                wanted.append(total - 1)
            wanted_set = set(wanted)

            frames: List[np.ndarray] = []
            kept: List[int] = []
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx in wanted_set:
                    frames.append(frame)
                    kept.append(idx)
                idx += 1
            return frames, kept
        finally:
            cap.release()

    # ------------------------------------------------------------ SAM3 path
    def _sam3_predict(
        self,
        frame: np.ndarray,
        labels: List[str],
        conf: float,
    ):
        """Run SAM3 on a single frame, returning ``(xyxy, conf, cls, masks)``.

        ``cls`` indexes into ``labels``.  ``masks`` is a per-detection
        ``H x W`` boolean array (``None`` when ``keep_masks`` is false).
        """

        # Set the per-call confidence threshold; ``args.conf`` is consulted
        # inside ``postprocess`` for NMS filtering.
        try:
            self._model.args.conf = float(conf)  # type: ignore[union-attr]
        except Exception:
            pass

        self._model.set_image(frame)  # type: ignore[union-attr]
        if self._sam3_classes != labels:
            self._sam3_classes = list(labels)
        preds = self._model(text=list(labels))  # type: ignore[misc]
        if not preds:
            return None
        r = preds[0]
        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                None,
            )
        try:
            xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            confs = boxes.conf.detach().cpu().numpy().astype(np.float32)
            clses = boxes.cls.detach().cpu().numpy().astype(np.int64)
        except Exception as exc:
            logger.debug("SAM3 box extraction failed: %s", exc)
            return None

        masks_arr: Optional[np.ndarray] = None
        if self.keep_masks:
            masks = getattr(r, "masks", None)
            if masks is not None and getattr(masks, "data", None) is not None:
                try:
                    masks_arr = (
                        masks.data.detach().cpu().numpy().astype(bool)
                    )
                except Exception:
                    masks_arr = None
        return xyxy, confs, clses, masks_arr

    def _segment_sam3(
        self,
        frames: List[np.ndarray],
        frame_indices: List[int],
        object_labels: List[str],
    ) -> Dict[str, List[FrameDetection]]:
        """Run SAM3SemanticPredictor frame-by-frame against the requested labels."""

        results: Dict[str, List[FrameDetection]] = {
            label: [] for label in object_labels
        }
        n_labels = len(object_labels)

        for fidx, frame in zip(frame_indices, frames):
            try:
                out = self._sam3_predict(frame, object_labels, self.conf)
            except Exception as exc:
                logger.debug("SAM3 predict failed at frame %d: %s", fidx, exc)
                continue
            if out is None:
                continue
            xyxy, confs, clses, masks_arr = out
            for i in range(xyxy.shape[0]):
                cls_idx = int(clses[i])
                if cls_idx < 0 or cls_idx >= n_labels:
                    continue
                score = float(confs[i])
                if score < self.conf:
                    continue
                label = object_labels[cls_idx]
                x1, y1, x2, y2 = (int(round(float(v))) for v in xyxy[i])
                mask: Optional[np.ndarray] = None
                if masks_arr is not None and i < masks_arr.shape[0]:
                    mask = masks_arr[i]
                results[label].append(
                    FrameDetection(
                        frame_idx=int(fidx),
                        mask=mask,
                        bbox=(x1, y1, x2, y2),
                        score=score,
                    )
                )
        return results

    # --------------------------------------------------------------- public API
    def segment_objects_video(
        self,
        video_path: Union[str, Path],
        object_labels: List[str],
        video_tracker_conf: float = 0.50,
        min_masklet_length: int = 3,
    ) -> Dict[str, List[FrameDetection]]:
        """Run video-level SAM3 tracking and return detections with ``track_id``.

        Uses :class:`ultralytics.models.sam.SAM3VideoSemanticPredictor` which
        natively tracks masklets across frames and assigns a stable
        ``track_id`` per object instance.  This dramatically improves
        Quantity-style checks compared with the per-frame predictor since
        the same physical object cannot be double-counted across frames.

        Parameters
        ----------
        video_path:
            Path to the video file.
        object_labels:
            List of free-form text labels.
        video_tracker_conf:
            Detection confidence threshold forwarded to the predictor's
            ``conf`` override (default ``0.50``).
        min_masklet_length:
            Minimum number of consecutive detections required to confirm a
            masklet (default ``3``).  Forwarded to
            ``masklet_confirmation_consecutive_det_thresh``.

        Returns
        -------
        ``{label: [FrameDetection, ...]}`` with the same shape as
        :meth:`segment_objects`, but every detection carries a stable
        ``track_id`` field.

        On any failure (missing weights, predictor exception, unsupported
        backend) we transparently fall back to :meth:`segment_objects`.
        """

        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"video not found: {video_path}")
        if not object_labels:
            return {}

        # Pre-condition: only the SAM3 backend supports video-level tracking.
        sam3_path = self._resolve_sam3_weights()
        if self._backend != "sam3_semantic" or sam3_path is None:
            logger.info(
                "segment_objects_video: backend=%s sam3_weights=%s -> falling "
                "back to per-frame segment_objects()",
                self._backend,
                sam3_path,
            )
            return self.segment_objects(video_path, object_labels)

        try:
            _patch_clip_pkg_resources()
            _patch_clip_tokenizer()
            from ultralytics.models.sam import (  # type: ignore
                SAM3VideoSemanticPredictor,
            )

            overrides = dict(
                conf=float(video_tracker_conf),
                task="segment",
                mode="predict",
                model=str(sam3_path),
                half=True,
                save=False,
                verbose=False,
            )
            if self.device:
                overrides["device"] = self.device

            video_predictor = SAM3VideoSemanticPredictor(
                overrides=overrides,
                masklet_confirmation_enable=True,
                masklet_confirmation_consecutive_det_thresh=int(min_masklet_length),
            )
            logger.info(
                "segment_objects_video: SAM3VideoSemanticPredictor initialised "
                "(conf=%.2f, min_masklet_length=%d, weights=%s)",
                float(video_tracker_conf),
                int(min_masklet_length),
                sam3_path.name,
            )

            results: Dict[str, List[FrameDetection]] = {
                label: [] for label in object_labels
            }
            n_labels = len(object_labels)
            unique_track_ids: set = set()
            frame_count = 0
            next_temp_id = -1  # negative ids reserved for synthetic placeholders

            stream = video_predictor(
                source=str(video_path),
                text=list(object_labels),
                stream=True,
            )
            for frame_number, r in enumerate(stream):
                frame_count += 1
                boxes = getattr(r, "boxes", None)
                if boxes is None or len(boxes) == 0:
                    continue
                try:
                    xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                    confs = boxes.conf.detach().cpu().numpy().astype(np.float32)
                    clses = boxes.cls.detach().cpu().numpy().astype(np.int64)
                except Exception as exc:
                    logger.debug(
                        "video predictor box extraction failed at frame %d: %s",
                        frame_number,
                        exc,
                    )
                    continue

                ids_arr: Optional[np.ndarray] = None
                raw_ids = getattr(boxes, "id", None)
                if raw_ids is not None:
                    try:
                        ids_arr = raw_ids.detach().cpu().numpy().astype(np.int64)
                    except Exception:
                        ids_arr = None

                masks_arr: Optional[np.ndarray] = None
                if self.keep_masks:
                    masks = getattr(r, "masks", None)
                    if masks is not None and getattr(masks, "data", None) is not None:
                        try:
                            masks_arr = (
                                masks.data.detach().cpu().numpy().astype(bool)
                            )
                        except Exception:
                            masks_arr = None

                for i in range(xyxy.shape[0]):
                    cls_idx = int(clses[i])
                    if cls_idx < 0 or cls_idx >= n_labels:
                        continue
                    score = float(confs[i])
                    label = object_labels[cls_idx]
                    x1, y1, x2, y2 = (int(round(float(v))) for v in xyxy[i])

                    if ids_arr is not None and i < ids_arr.shape[0]:
                        track_id = int(ids_arr[i])
                    else:
                        track_id = next_temp_id
                        next_temp_id -= 1
                    unique_track_ids.add(track_id)

                    mask: Optional[np.ndarray] = None
                    if masks_arr is not None and i < masks_arr.shape[0]:
                        mask = masks_arr[i]

                    results[label].append(
                        FrameDetection(
                            frame_idx=int(frame_number),
                            mask=mask,
                            bbox=(x1, y1, x2, y2),
                            score=score,
                            track_id=track_id,
                        )
                    )

            logger.info(
                "segment_objects_video: processed %d frames, found %d unique "
                "track_id(s) across %d label(s)",
                frame_count,
                len(unique_track_ids),
                n_labels,
            )
            self.last_error = None
            return results
        except Exception as exc:
            self.last_error = (
                f"sam3_video_semantic: {type(exc).__name__}: {exc}"
            )
            logger.warning(
                "SAM3VideoSemanticPredictor failed (%s); falling back to "
                "per-frame segment_objects()",
                self.last_error,
            )
            return self.segment_objects(video_path, object_labels)

    # --------------------------------------------------------------- public API
    def count_objects_per_frame(
        self,
        video_path: Union[str, Path],
        object_labels: List[str],
        sample_fps: float = 4.0,
    ) -> Dict[str, List[int]]:
        """Return per-frame detection counts for each label.

        Unlike :meth:`segment_objects` which keeps only the highest-scoring
        detection per label per frame, this method counts **all** detections
        above the confidence threshold for each label in each sampled frame.

        Parameters
        ----------
        video_path:
            Path to the video file.
        object_labels:
            List of free-form text labels.
        sample_fps:
            How many frames per second to feed to the detector.

        Returns
        -------
        ``{label: [count_frame0, count_frame1, ...]}``.  Each list has the
        same length (number of sampled frames).  A label with zero detections
        in all frames still appears with a list of zeros.
        """

        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"video not found: {video_path}")
        if not object_labels:
            return {}

        frames, frame_indices = self._sample_frames(video_path, sample_fps)
        if not frames:
            return {label: [] for label in object_labels}

        if self._backend == "sam3_semantic" and self._model is not None:
            try:
                return self._count_sam3(frames, frame_indices, object_labels)
            except Exception as exc:
                logger.warning("SAM3 count failed: %s", exc)

        # Fallback: no counts available.
        return {label: [0] * len(frames) for label in object_labels}

    # ---------------------------------------------------------- counting helpers
    def _count_sam3(
        self,
        frames: List[np.ndarray],
        frame_indices: List[int],
        object_labels: List[str],
    ) -> Dict[str, List[int]]:
        """Count all detections per label per frame using SAM3."""

        n_labels = len(object_labels)
        # Use a more permissive threshold for counting so that lower-confidence
        # additional instances are still tallied for the Quantity check.
        count_conf = max(min(self.conf, 0.25), 0.15)  
        counts: Dict[str, List[int]] = {label: [] for label in object_labels}

        # Disable mask materialisation while counting – we only need the
        # boxes and class indices, and the mask copy is the dominant cost.
        prev_keep = self.keep_masks
        self.keep_masks = False
        try:
            for fidx, frame in zip(frame_indices, frames):
                frame_counts: Dict[str, int] = {label: 0 for label in object_labels}
                try:
                    out = self._sam3_predict(frame, object_labels, count_conf)
                except Exception:
                    out = None
                if out is not None:
                    _, confs, clses, _ = out
                    for i in range(clses.shape[0]):
                        cls_idx = int(clses[i])
                        if cls_idx < 0 or cls_idx >= n_labels:
                            continue
                        if float(confs[i]) < count_conf:
                            continue
                        frame_counts[object_labels[cls_idx]] += 1
                for label in object_labels:
                    counts[label].append(frame_counts[label])
        finally:
            self.keep_masks = prev_keep
        return counts
