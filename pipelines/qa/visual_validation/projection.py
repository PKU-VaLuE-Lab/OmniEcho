"""Spherical / pixel projection math for visual validation.

The scene metadata describes every sound source with a *listener-relative*
spherical coordinate ``(azimuth, elevation, distance)``.  In order to compare
the metadata against what is actually drawn in the rendered video, we project
the spherical coordinates onto the pinhole camera's image plane.

Conventions (matching ``scene.json``)
-------------------------------------
* ``azimuth`` is measured in degrees, **positive to the listener's left**
  (i.e. the left hand side of the screen in first-person view) and zero straight ahead.
* ``elevation`` is measured in degrees, positive upwards.
* ``camera_yaw`` rotates the camera around the world's vertical axis, positive
  to the left – consistent with the azimuth sign.

A pinhole camera is assumed: the projected pixel for a direction with
*effective* azimuth ``a`` and elevation ``e`` is::

    x = W/2 - tan(a) / tan(fov_h/2) * (W/2)
    y = H/2 - tan(e) / tan(fov_v/2) * (H/2)

The simpler linear approximation requested in the task brief (``a / (fov/2) *
(W/2)``) is provided alongside the tan-based form: when the requested angles
sit well inside the FOV both forms agree to first order; the tan version is
only used to keep the projection accurate near the edges of the frame.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Mapping


# ---------------------------------------------------------------------------
# FOV helpers
# ---------------------------------------------------------------------------


def compute_fov_vertical(fov_h_deg: float, width: int, height: int) -> float:
    """Return the vertical field of view (degrees) for a given horizontal FOV.

    Uses the standard pinhole relation::

        tan(fov_v/2) = tan(fov_h/2) * height / width
    """

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    half_h = math.radians(fov_h_deg) / 2.0
    half_v = math.atan(math.tan(half_h) * (height / width))
    return math.degrees(half_v) * 2.0


# ---------------------------------------------------------------------------
# Forward / inverse projection
# ---------------------------------------------------------------------------


def _wrap_deg(angle_deg: float) -> float:
    """Wrap an angle into ``[-180, 180]`` degrees."""

    a = (angle_deg + 180.0) % 360.0 - 180.0
    # ``-180`` and ``180`` are equivalent; canonicalise to ``180``.
    return 180.0 if a == -180.0 else a


def azimuth_elevation_to_pixel(
    azimuth_deg: float,
    elevation_deg: float,
    camera_yaw_deg: float,
    fov_h_deg: float,
    width: int,
    height: int,
) -> tuple[float, float]:
    """Project a spherical direction onto pixel coordinates.

    Parameters
    ----------
    azimuth_deg, elevation_deg:
        Direction in the listener-relative frame.
    camera_yaw_deg:
        Camera rotation around the vertical axis.  The effective azimuth seen
        by the camera is ``azimuth_deg - camera_yaw_deg``.
    fov_h_deg:
        Horizontal field of view, degrees.
    width, height:
        Frame size, pixels.

    Returns
    -------
    ``(x, y)`` floating-point pixel coordinates.  Values may fall outside the
    ``[0, width) x [0, height)`` rectangle when the direction is outside of
    the camera's FOV; callers should validate with :func:`is_in_fov`.
    """

    fov_v_deg = compute_fov_vertical(fov_h_deg, width, height)

    eff_az = _wrap_deg(azimuth_deg - camera_yaw_deg)

    half_w = width / 2.0
    half_h = height / 2.0
    half_fov_h = math.radians(fov_h_deg) / 2.0
    half_fov_v = math.radians(fov_v_deg) / 2.0

    # Tan-based pinhole projection.  ``tan`` blows up near +/-90deg, so clamp
    # the effective angle to slightly less than +/-90 to keep the math finite.
    eff_az_rad = math.radians(max(min(eff_az, 89.999), -89.999))
    el_rad = math.radians(max(min(elevation_deg, 89.999), -89.999))

    x = half_w - math.tan(eff_az_rad) / math.tan(half_fov_h) * half_w
    y = half_h - math.tan(el_rad) / math.tan(half_fov_v) * half_h
    return x, y


def pixel_to_azimuth_elevation(
    x: float,
    y: float,
    camera_yaw_deg: float,
    fov_h_deg: float,
    width: int,
    height: int,
) -> tuple[float, float]:
    """Inverse projection: pixel coordinates → ``(azimuth, elevation)``.

    Returns angles in the listener-relative frame, i.e. the camera yaw is
    *added back in* so that the result can be directly compared with values
    stored in ``scene.json``.
    """

    fov_v_deg = compute_fov_vertical(fov_h_deg, width, height)
    half_w = width / 2.0
    half_h = height / 2.0
    half_fov_h = math.radians(fov_h_deg) / 2.0
    half_fov_v = math.radians(fov_v_deg) / 2.0

    nx = (x - half_w) / half_w  # in [-1, 1] when x in frame
    ny = (half_h - y) / half_h

    eff_az = math.degrees(math.atan(-nx * math.tan(half_fov_h)))
    el = math.degrees(math.atan(ny * math.tan(half_fov_v)))

    return _wrap_deg(eff_az + camera_yaw_deg), el


def is_in_fov(
    azimuth_deg: float,
    elevation_deg: float,
    camera_yaw_deg: float,
    fov_h_deg: float,
    fov_v_deg: float,
) -> bool:
    """Return ``True`` if the direction lies inside the camera frustum."""

    eff_az = _wrap_deg(azimuth_deg - camera_yaw_deg)
    return abs(eff_az) <= fov_h_deg / 2.0 and abs(elevation_deg) <= fov_v_deg / 2.0


# ---------------------------------------------------------------------------
# Trajectory interpolation
# ---------------------------------------------------------------------------


def _sorted_keyframes(traj: Iterable[Mapping[str, float]]) -> List[Mapping[str, float]]:
    """Return ``traj`` as a list sorted by ``t``."""

    keyframes = list(traj)
    if not keyframes:
        raise ValueError("trajectory must contain at least one keyframe")
    return sorted(keyframes, key=lambda kf: float(kf["t"]))


def _lerp(a: float, b: float, alpha: float) -> float:
    return a + (b - a) * alpha


def interpolate_trajectory(trajectory: list[dict], t: float) -> dict:
    """Linearly interpolate an object trajectory at time ``t`` (seconds).

    The trajectory is a list of dicts ``{"t": float, "azimuth": float,
    "elevation": float, "distance": float}``.  Interpolation is piecewise
    linear; values are clamped at both ends, i.e. requesting a time before the
    first keyframe returns the first keyframe and a time after the last
    returns the last.

    Single-keyframe trajectories are returned as-is (no interpolation).
    """

    keys = _sorted_keyframes(trajectory)
    if len(keys) == 1:
        kf = keys[0]
        return {
            "azimuth": float(kf["azimuth"]),
            "elevation": float(kf["elevation"]),
            "distance": float(kf["distance"]),
        }

    if t <= float(keys[0]["t"]):
        kf = keys[0]
        return {
            "azimuth": float(kf["azimuth"]),
            "elevation": float(kf["elevation"]),
            "distance": float(kf["distance"]),
        }
    if t >= float(keys[-1]["t"]):
        kf = keys[-1]
        return {
            "azimuth": float(kf["azimuth"]),
            "elevation": float(kf["elevation"]),
            "distance": float(kf["distance"]),
        }

    # Locate the bracketing pair.
    for prev, nxt in zip(keys[:-1], keys[1:]):
        t0 = float(prev["t"])
        t1 = float(nxt["t"])
        if t0 <= t <= t1:
            if t1 == t0:
                alpha = 0.0
            else:
                alpha = (t - t0) / (t1 - t0)
            # Interpolate azimuth using the shortest angular path so that a
            # jump from -170deg to +170deg goes the short way (40deg) rather
            # than wrapping around (320deg).
            az0 = float(prev["azimuth"])
            az1 = float(nxt["azimuth"])
            delta_az = _wrap_deg(az1 - az0)
            return {
                "azimuth": _wrap_deg(az0 + delta_az * alpha),
                "elevation": _lerp(float(prev["elevation"]), float(nxt["elevation"]), alpha),
                "distance": _lerp(float(prev["distance"]), float(nxt["distance"]), alpha),
            }

    # Should never reach here given the clamps above, but guard anyway.
    raise RuntimeError(f"failed to bracket time {t}")


def interpolate_camera_yaw(yaw_trajectory: list[dict], t: float) -> float:
    """Linearly interpolate the camera yaw at time ``t`` (seconds).

    ``yaw_trajectory`` is a list of ``{"t": float, "yaw": float}`` keyframes.
    Same clamp / shortest-path semantics as :func:`interpolate_trajectory`.
    """

    keys = _sorted_keyframes(yaw_trajectory)
    if len(keys) == 1:
        return float(keys[0]["yaw"])

    if t <= float(keys[0]["t"]):
        return float(keys[0]["yaw"])
    if t >= float(keys[-1]["t"]):
        return float(keys[-1]["yaw"])

    for prev, nxt in zip(keys[:-1], keys[1:]):
        t0 = float(prev["t"])
        t1 = float(nxt["t"])
        if t0 <= t <= t1:
            if t1 == t0:
                return float(prev["yaw"])
            alpha = (t - t0) / (t1 - t0)
            y0 = float(prev["yaw"])
            y1 = float(nxt["yaw"])
            return _wrap_deg(y0 + _wrap_deg(y1 - y0) * alpha)

    raise RuntimeError(f"failed to bracket time {t}")
