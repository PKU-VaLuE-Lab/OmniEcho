"""cv2 ``VideoCapture`` / ``VideoWriter`` compatibility shims.

The OpenCV build available in this environment was compiled *without*
FFmpeg/GStreamer support, so the bundled ``cv2.VideoCapture`` cannot decode
the H.264 MP4 files produced by the OmniEcho pipeline.  Rather than
rewriting every call-site we install thin drop-in replacements:

* :class:`_DecordCapture` mimics the subset of the ``cv2.VideoCapture`` API
  used by the visual-validation modules.  It is backed by ``decord`` (which
  ships with its own libavcodec) so it can decode the H.264 MP4 files
  produced by the OmniEcho pipeline.
* :class:`_FFmpegWriter` mimics ``cv2.VideoWriter`` and pipes BGR frames to
  the system ``ffmpeg`` binary, producing an H.264 MP4.

The :func:`install` helper transparently swaps these shims onto the ``cv2``
module *only* when OpenCV was built without FFmpeg support, so environments
that already have working video I/O are not affected.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decord-backed VideoCapture
# ---------------------------------------------------------------------------


class _DecordCapture:
    """Drop-in shim implementing the bits of :class:`cv2.VideoCapture` we use."""

    def __init__(self, filename: str, *_, **__) -> None:
        self._filename = str(filename)
        self._reader = None
        self._idx = 0
        self._fps = 0.0
        self._total = 0
        self._width = 0
        self._height = 0
        self._opened = False
        self._grabbed: Optional[np.ndarray] = None
        try:
            import decord  # type: ignore

            decord.bridge.set_bridge("native")
            self._reader = decord.VideoReader(self._filename)
            self._total = len(self._reader)
            try:
                self._fps = float(self._reader.get_avg_fps())
            except Exception:
                self._fps = 0.0
            if self._total > 0:
                shape = self._reader[0].shape  # (H, W, 3)
                self._height = int(shape[0])
                self._width = int(shape[1])
            self._opened = True
        except Exception as exc:
            logger.debug("decord failed to open %s: %s", self._filename, exc)
            self._reader = None
            self._opened = False

    # cv2.VideoCapture API surface ----------------------------------------
    def isOpened(self) -> bool:  # noqa: N802 - mirror cv2 API
        return bool(self._opened)

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return float(self._fps)
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self._total)
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self._idx)
        return 0.0

    def set(self, prop: int, value: float) -> bool:
        if prop == cv2.CAP_PROP_POS_FRAMES:
            new_idx = int(value)
            if 0 <= new_idx <= self._total:
                self._idx = new_idx
                return True
        return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.grab():
            return False, None
        return self.retrieve()

    def grab(self) -> bool:
        """Advance to the next frame and cache it (no separate decode step).

        Mirrors :meth:`cv2.VideoCapture.grab`.  Decord does not expose an
        advance-without-decode primitive, so we eagerly decode the next frame
        into ``self._grabbed`` and let :meth:`retrieve` return it.
        """

        if not self._opened or self._reader is None:
            self._grabbed = None
            return False
        if self._idx >= self._total:
            self._grabbed = None
            return False
        try:
            arr = self._reader[self._idx]
        except Exception as exc:
            logger.debug("decord grab failure at %d: %s", self._idx, exc)
            self._grabbed = None
            return False
        self._idx += 1
        # decord returns RGB; cv2 callers expect BGR.
        rgb = arr.asnumpy() if hasattr(arr, "asnumpy") else np.asarray(arr)
        self._grabbed = np.ascontiguousarray(rgb[..., ::-1])
        return True

    def retrieve(self, *_args, **_kwargs) -> Tuple[bool, Optional[np.ndarray]]:
        """Return the frame previously cached by :meth:`grab`."""

        if self._grabbed is None:
            return False, None
        return True, self._grabbed

    def release(self) -> None:
        self._reader = None
        self._opened = False
        self._grabbed = None


# ---------------------------------------------------------------------------
# FFmpeg-pipe backed VideoWriter
# ---------------------------------------------------------------------------


class _FFmpegWriter:
    """Stream BGR frames to a subprocess ``ffmpeg`` to encode an H.264 MP4."""

    def __init__(
        self,
        filename: str,
        fourcc: int,
        fps: float,
        frame_size: Tuple[int, int],
        is_color: bool = True,
    ) -> None:
        self._filename = str(filename)
        self._fps = max(float(fps), 1.0)
        self._size = (int(frame_size[0]), int(frame_size[1]))
        self._proc: Optional[subprocess.Popen] = None
        self._opened = False
        self._ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
        if shutil.which(self._ffmpeg) is None and self._ffmpeg == "ffmpeg":
            return
        try:
            cmd = [
                self._ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{self._size[0]}x{self._size[1]}",
                "-r",
                f"{self._fps:.6f}",
                "-i",
                "-",
                "-an",
                # ``-vcodec libx264`` would be ideal but the bundled ffmpeg in
                # this environment is built with ``--disable-libx264``; fall
                # back to ``libopenh264`` which is always present.
                "-vcodec",
                "libopenh264",
                "-pix_fmt",
                "yuv420p",
                "-b:v",
                "3000k",
                self._filename,
            ]
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            self._opened = self._proc.stdin is not None
        except Exception as exc:
            logger.debug("failed to start ffmpeg writer: %s", exc)
            self._proc = None
            self._opened = False

    # cv2.VideoWriter API surface -----------------------------------------
    def isOpened(self) -> bool:  # noqa: N802 - mirror cv2 API
        return bool(self._opened)

    def write(self, frame: np.ndarray) -> None:
        if not self._opened or self._proc is None or self._proc.stdin is None:
            return
        if frame is None:
            return
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        if frame.shape[1] != self._size[0] or frame.shape[0] != self._size[1]:
            frame = cv2.resize(frame, self._size)
        try:
            self._proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, ValueError) as exc:
            logger.warning("ffmpeg writer pipe broke: %s", exc)
            self._opened = False

    def release(self) -> None:
        if self._proc is None:
            return
        stderr_bytes = b""
        try:
            if self._proc.stdin is not None:
                try:
                    self._proc.stdin.close()
                except Exception:
                    pass
            try:
                stderr_bytes = self._proc.stderr.read() if self._proc.stderr else b""
            except Exception:
                stderr_bytes = b""
            self._proc.wait(timeout=120)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        rc = self._proc.returncode if self._proc is not None else None
        if rc not in (0, None):
            msg = stderr_bytes.decode("utf-8", errors="replace").strip()
            logger.warning("ffmpeg exited with code %s: %s", rc, msg[-500:])
        self._proc = None
        self._opened = False


def _fourcc_stub(*_args, **_kwargs) -> int:
    """``cv2.VideoWriter_fourcc`` replacement – the value is ignored by ffmpeg."""

    return 0


# ---------------------------------------------------------------------------
# Installer
# ---------------------------------------------------------------------------


_INSTALLED = False


def install() -> None:
    """Patch ``cv2`` with the shims when its FFmpeg backend is unavailable.

    Idempotent.  When OpenCV already has FFmpeg support the function returns
    without modifying anything.
    """

    global _INSTALLED
    if _INSTALLED:
        return

    build_info = ""
    try:
        build_info = cv2.getBuildInformation()
    except Exception:
        build_info = ""

    needs_shim = "FFMPEG:                      YES" not in build_info
    if not needs_shim:
        # Quick smoke test – occasionally the plugin is reported as YES but
        # cannot actually decode a file.  We still skip the shim in that
        # case because the shim is only needed when basic decoding fails.
        _INSTALLED = True
        return

    if shutil.which("ffmpeg") is None:
        logger.warning(
            "OpenCV lacks FFmpeg support and the 'ffmpeg' binary is not on PATH; "
            "video encoding will be disabled."
        )

    cv2.VideoCapture = _DecordCapture  # type: ignore[assignment]
    cv2.VideoWriter = _FFmpegWriter  # type: ignore[assignment]
    cv2.VideoWriter_fourcc = _fourcc_stub  # type: ignore[assignment]
    _INSTALLED = True
    logger.info(
        "installed decord/ffmpeg shims for cv2.VideoCapture and cv2.VideoWriter"
    )


__all__ = ["install"]
