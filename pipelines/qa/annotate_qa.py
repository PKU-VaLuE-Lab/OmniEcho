#!/usr/bin/env python3
"""OmniEcho processing component."""

from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

logger = logging.getLogger("aligned_av_qa")


SOUND_VOCAB: List[Tuple[str, str, str]] = [
    ("a rattling sound", "rattling cardboard box", "rattling"),
    ("a dragging sound", "dragging", "dragging"),
    ("a whirring motor sound", "motor", "whirring motor"),
    ("a device alert tone", "device alert tone", "device alert"),
    ("a notification tone", "notification tone", "notification"),
    ("a whistling sound", "whistling", "whistling"),
    ("a rustling plastic sound", "rustling plastic", "rustling plastic"),
    ("a rolling sound", "rolling", "rolling"),
    ("an alarm ringing sound", "alarm ringing", "alarm ringing"),
    ("a metallic clinking sound", "metallic clinking", "metallic clinking"),
    ("a singing voice", "singing", "singing"),
    ("a buzzing vibration sound", "buzzing vibration", "buzzing vibration"),
]


_PHRASE_OVERRIDES = {
    "footsteps": "footsteps",
    "cat_footsteps": "cat footsteps",
    "seagull": "a seagull call",
    "seagulls": "seagull calls",
    "water_drip": "water dripping",
    "water_drop": "water dripping",
    "water_drops": "water dripping",
    "clock_ticking": "a ticking clock",
    "page_turning": "pages turning",
    "pages_turning": "pages turning",
    "singing": "a singing voice",
    "typing": "typing on a keyboard",
    "keyboard_typing": "typing on a keyboard",
    "coughing": "a cough",
    "knocking": "knocking",
    "door_knocking": "knocking on a door",
    "phone_ringing": "a phone ringing",
    "dog_barking": "a dog barking",
    "birds_chirping": "birds chirping",
}


def _label_to_phrase(label: str) -> str:
    """OmniEcho processing component."""
    key = str(label).lower().strip()
    if key in _PHRASE_OVERRIDES:
        return _PHRASE_OVERRIDES[key]
    words = key.replace("_", " ").replace("-", " ").strip()
    if not words:
        return "a sound"
    art = "an" if words[:1] in "aeiou" else "a"
    return f"{art} {words} sound"


DIR8_ORDER = [
    "directly ahead", "ahead and to the left", "directly to the left",
    "behind and to the left", "directly behind", "behind and to the right",
    "directly to the right", "ahead and to the right",
]
DIR8_OPTIONS_STR = " / ".join(DIR8_ORDER)


_AMBIENT_KEYWORDS = frozenset([
    "wind", "rain", "ambient", "ambience", "background", "hvac", "noise",
    "crowd", "traffic", "water", "birds", "insects", "thunder", "ocean",
    "river", "stream", "breeze", "nature", "weather", "storm",
])


DIST_MIN, DIST_MAX = 0.2, 2.5


DIST_MAX_OFF = 4.5

OFFSCREEN_MIN_ABS_AZ = 50.0

MOTION_CHANGE_DEG = 45.0


TURN_CLASSIFY_DEG = 67.5

TURN_MIN_SEG = 0.35


MOVE_MIN_DIST = 0.8
WALK_MIN_DIST = 1.6


TURN_DEG_LO, TURN_DEG_HI = 90.0, 135.0


MIN_OFFSCREEN = 2

ABS_MAX_SOURCES = 6


TURN_SAMPLE_FRAC = 0.5


VIS_YES_BIAS = 0.57        
VIS_OVER_YES_BIAS = 0.62   


ENTER_BEHIND_MIN, ENTER_BEHIND_MAX = 120.0, 150.0

QA_PER_MIN, QA_PER_MAX = 15, 20


def wrap_deg(a: float) -> float:
    """OmniEcho processing component."""
    return (float(a) + 180.0) % 360.0 - 180.0


def smoothstep(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3.0 - 2.0 * u)


def dir8_from_az(az_deg: float) -> str:
    """OmniEcho processing component."""
    a = wrap_deg(az_deg)
    aa = abs(a)
    if aa < 22.5:
        return "directly ahead"
    if aa > 157.5:
        return "directly behind"
    left = a > 0  
    if aa < 67.5:
        return "ahead and to the left" if left else "ahead and to the right"
    if aa < 112.5:
        return "directly to the left" if left else "directly to the right"
    return "behind and to the left" if left else "behind and to the right"


def answer_bucket(answer: str) -> str:
    """OmniEcho processing component."""
    a = answer.strip().lower()
    if a in ("yes", "no"):
        return a
    if "ahead" in a:
        return "front"
    if "behind" in a:
        return "back"
    if "to the left" in a:
        return "left"
    if "to the right" in a:
        return "right"
    return "front"


def turn_and_degrees(signed_deg: float) -> Tuple[str, int]:
    """OmniEcho processing component."""
    turn = "left" if signed_deg > 0 else "right"
    return turn, int(round(abs(signed_deg)))


def polar_to_xy(az_deg: float, dist: float) -> Tuple[float, float]:
    a = math.radians(az_deg)
    return dist * math.cos(a), dist * math.sin(a)


def xy_to_polar(x: float, y: float) -> Tuple[float, float]:
    return wrap_deg(math.degrees(math.atan2(y, x))), math.hypot(x, y)


def ray_line_t(az_deg: float, w0: Tuple[float, float], w1: Tuple[float, float]) -> Optional[float]:
    """OmniEcho processing component."""
    ux, uy = math.cos(math.radians(az_deg)), math.sin(math.radians(az_deg))
    vx, vy = w1[0] - w0[0], w1[1] - w0[1]
    cross_uv = ux * vy - uy * vx
    if abs(cross_uv) < 1e-9:
        return None
    t = (w0[0] * vy - w0[1] * vx) / cross_uv
    return t if t > 1e-6 else None


def _seg_points(p0: Tuple[float, float], p1: Tuple[float, float], n: int) -> List[Tuple[float, float]]:
    return [(p0[0] + (p1[0] - p0[0]) * (i / (n - 1)),
             p0[1] + (p1[1] - p0[1]) * (i / (n - 1))) for i in range(n)]


def _robust_max_turn(xs: List[float], ys: List[float],
                     min_seg: float = TURN_MIN_SEG) -> float:
    """OmniEcho processing component."""
    n = len(xs)
    if n < 3:
        return 0.0
    best = 0.0
    for i in range(1, n - 1):
        v1x, v1y = xs[i] - xs[0], ys[i] - ys[0]
        v2x, v2y = xs[-1] - xs[i], ys[-1] - ys[i]
        if math.hypot(v1x, v1y) < min_seg or math.hypot(v2x, v2y) < min_seg:
            continue
        ang = abs(wrap_deg(math.degrees(math.atan2(v2y, v2x))
                           - math.degrees(math.atan2(v1y, v1x))))
        best = max(best, ang)
    return best


class SourceTrack:
    """OmniEcho processing component."""

    def __init__(self, obj: dict, is_offscreen: bool):
        self.obj = obj
        self.is_offscreen = is_offscreen
        self.label = str(obj.get("label", obj.get("id", "source")))
        pts = obj["trajectory"]
        self.t = np.array([float(p["t"]) for p in pts], dtype=float)
        self.az = np.array([float(p.get("azimuth", p.get("az", 0.0))) for p in pts], dtype=float)
        self.el = np.array([float(p.get("elevation", p.get("el", 0.0))) for p in pts], dtype=float)
        self.di = np.array([float(p.get("distance", p.get("dist", 1.0))) for p in pts], dtype=float)
        self.vis_states = [str(p.get("visibility_state", "")) for p in pts]
        self._az_unwrap = np.unwrap(np.deg2rad(self.az))
        self.start_time = float(obj.get("start_time", 0.0))


        self.audible = stem_is_audible(str(obj.get("wav_path", "")))

    def world_az(self, t: float) -> float:
        tt = t - self.start_time
        return wrap_deg(math.degrees(float(np.interp(tt, self.t, self._az_unwrap,
                                                      left=self._az_unwrap[0], right=self._az_unwrap[-1]))))

    def elevation(self, t: float) -> float:
        tt = t - self.start_time
        return float(np.interp(tt, self.t, self.el, left=self.el[0], right=self.el[-1]))

    def distance(self, t: float) -> float:
        tt = t - self.start_time
        return float(np.interp(tt, self.t, self.di, left=self.di[0], right=self.di[-1]))

    def visibility_state_at(self, t: float) -> str:
        """OmniEcho processing component."""
        tt = t - self.start_time
        idx = int(np.argmin(np.abs(self.t - tt)))
        return self.vis_states[idx] if idx < len(self.vis_states) else ""


    def _world_xy(self) -> Tuple[np.ndarray, np.ndarray]:
        rad = np.deg2rad(self.az)
        return self.di * np.cos(rad), self.di * np.sin(rad)

    def world_path_len(self) -> float:
        """OmniEcho processing component."""
        xs, ys = self._world_xy()
        if len(xs) < 2:
            return 0.0
        return float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))

    def max_turn_deg(self) -> float:
        """OmniEcho processing component."""
        xs, ys = self._world_xy()
        return _robust_max_turn(list(map(float, xs)), list(map(float, ys)))

    def motion_class(self) -> str:
        """OmniEcho processing component."""
        if self.world_path_len() < MOVE_MIN_DIST:
            return "static"
        return "turn" if self.max_turn_deg() > TURN_CLASSIFY_DEG else "line"


def is_ambient_label(label: str) -> bool:
    low = label.lower()
    return any(k in low for k in _AMBIENT_KEYWORDS)


class Camera:
    """OmniEcho processing component."""

    def __init__(self, duration: float, yaw_end_deg: float, fov_h_deg: float,
                 frame_w: int, frame_h: int, frac_w: float = 1.0,
                 yaw_start_deg: float = 0.0):
        self.duration = duration
        self.yaw_start = yaw_start_deg
        self.yaw_end = yaw_end_deg
        self.fov_h = fov_h_deg
        self.W = frame_w
        self.H = frame_h
        self.frac_w = frac_w
        self.f = (frame_w / 2.0) / math.tan(math.radians(fov_h_deg / 2.0))
        
        self.rotating = abs(yaw_end_deg - yaw_start_deg) > 1e-6
        self.cropped = frac_w < 1.0 - 1e-6

    def yaw(self, t: float) -> float:
        if not self.rotating:
            return self.yaw_start          
        u = t / self.duration if self.duration > 0 else 0.0
        return self.yaw_start + (self.yaw_end - self.yaw_start) * smoothstep(u)

    def half_hfov(self) -> float:
        """OmniEcho processing component."""
        return math.degrees(math.atan((self.frac_w * self.W / 2.0) / self.f))

    def half_vfov(self) -> float:
        return math.degrees(math.atan((self.frac_w * self.H / 2.0) / self.f))

    def offscreen_min_abs_az(self) -> float:
        """OmniEcho processing component."""
        return max(OFFSCREEN_MIN_ABS_AZ,
                   self.half_hfov() + max(abs(self.yaw_start), abs(self.yaw_end)) + 6.0)

    def body_az(self, world_az: float, t: float) -> float:
        return wrap_deg(world_az - self.yaw(t))

    def visible(self, world_az: float, el: float, t: float) -> bool:
        """OmniEcho processing component."""
        return abs(self.body_az(world_az, t)) < self.half_hfov() and abs(el) < self.half_vfov()

    def crop_center_x(self, t: float) -> float:
        """OmniEcho processing component."""
        return self.W / 2.0 - self.f * math.tan(math.radians(self.yaw(t)))


def choose_yaw_end(rng: random.Random, cam_f: float, frame_w: int, frac_w: float) -> float:
    """OmniEcho processing component."""
    max_shift = frame_w * (1.0 - frac_w) / 2.0            
    max_yaw = math.degrees(math.atan(max_shift / cam_f))   
    lo, hi = min(12.0, 0.6 * max_yaw), max(12.0, 0.95 * max_yaw)
    mag = rng.uniform(lo, hi)
    return mag if rng.random() < 0.5 else -mag


def build_stem_pool(input_root: Path) -> List[Path]:
    """OmniEcho processing component."""
    seen = set()
    pool: List[Path] = []
    for pat in ("stems/*.wav", "*/stems/*.wav"):
        for wav in input_root.glob(pat):
            r = wav.resolve()
            if r not in seen:
                seen.add(r)
                pool.append(r)
    return sorted(pool)


def _scan_scene_pairs(sc: Path) -> List[Tuple[str, str]]:
    """OmniEcho processing component."""
    out: List[Tuple[str, str]] = []
    try:
        data = json.loads(sc.read_text())
    except Exception:
        return out
    parent = str(sc.parent)
    for o in data.get("objects", []):
        label = str(o.get("label", "")).strip()
        if not label or is_ambient_label(label):
            continue
        wp = str(o.get("wav_path", ""))
        if not wp:
            continue
        p = wp if os.path.isabs(wp) else os.path.normpath(os.path.join(parent, wp))
        out.append((label, p))
    return out


def build_sound_library(input_root: Path, workers: int = 32) -> List[Tuple[str, str]]:
    """OmniEcho processing component."""
    
    scenes: List[Path] = []
    seen_sc = set()
    for pat in ("corrected_scene.json", "*/corrected_scene.json",
                "scene.json", "*/scene.json"):
        for sc in input_root.glob(pat):
            s = str(sc)
            if s not in seen_sc:
                seen_sc.add(s)
                scenes.append(sc)


    pairs_all: List[Tuple[str, str]] = []
    if scenes:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for res in ex.map(_scan_scene_pairs, scenes):
                pairs_all.extend(res)


    keys = list(dict.fromkeys(pairs_all))          
    uniq_paths = list(dict.fromkeys(p for _, p in keys))


    audible: Dict[str, bool] = {}
    if uniq_paths:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for p, ok in zip(uniq_paths, ex.map(stem_is_audible, uniq_paths)):
                audible[p] = ok

    pairs = [(lab, p) for (lab, p) in keys if audible.get(p, False)]
    n_silent = len(uniq_paths) - sum(1 for p in uniq_paths if audible.get(p, False))
    logger.info("Sound library: scanned %d scenes, %d unique stems; filtered %d near-silent/unreadable stems, retained %d audible sources (%d pairs)",
                len(scenes), len(uniq_paths), n_silent,
                sum(1 for p in uniq_paths if audible.get(p, False)), len(pairs))
    return sorted(pairs)


_STEM_LEVEL_CACHE: Dict[str, Optional[Tuple[float, float]]] = {}
_STEM_RMS_FLOOR_DB = -50.0    
_STEM_PEAK_FLOOR_DB = -40.0   
_OFFSCREEN_TARGET_RMS_DB = -20.0  


def stem_level_dbfs(wav_path: str) -> Optional[Tuple[float, float]]:
    """OmniEcho processing component."""
    key = str(wav_path)
    if key in _STEM_LEVEL_CACHE:
        return _STEM_LEVEL_CACHE[key]
    lvl: Optional[Tuple[float, float]] = None
    try:
        import soundfile as sf
        data, _sr = sf.read(key, dtype="float32", always_2d=True)
        if data.size:
            mono = data.mean(axis=1).astype(np.float64)
            peak = float(np.max(np.abs(mono)))
            rms = float(np.sqrt(np.mean(mono ** 2)))
            eps = 1e-9
            lvl = (20.0 * math.log10(rms + eps), 20.0 * math.log10(peak + eps))
    except Exception:
        lvl = None
    _STEM_LEVEL_CACHE[key] = lvl
    return lvl


def stem_is_audible(wav_path: str) -> bool:
    """OmniEcho processing component."""
    lvl = stem_level_dbfs(wav_path)
    if lvl is None:
        return False
    rms_db, peak_db = lvl
    return rms_db >= _STEM_RMS_FLOOR_DB and peak_db >= _STEM_PEAK_FLOOR_DB


def _offscreen_region_az(rng: random.Random, region: str, lo: float) -> float:
    """OmniEcho processing component."""
    if region == "back":
        return math.copysign(rng.uniform(max(lo, 118.0), 178.0), 1.0 if rng.random() < 0.5 else -1.0)
    if region == "left":
        return rng.uniform(lo + 3.0, 168.0)
    if region == "right":
        return -rng.uniform(lo + 3.0, 168.0)
    # 'any'
    a = rng.uniform(lo + 3.0, 178.0)
    return a if rng.random() < 0.5 else -a


def _offscreen_trajectory(rng: random.Random, duration: float, kind: str,
                          min_abs_az: float) -> List[dict]:
    """OmniEcho processing component."""
    lo = min_abs_az + 3.0
    el = rng.uniform(-10.0, 10.0)
    n = 6
    ts = [round(duration * i / (n - 1), 3) for i in range(n)]

    def rand_pt(region: str) -> Tuple[float, float]:
        return polar_to_xy(_offscreen_region_az(rng, region, lo), rng.uniform(0.4, 2.4))

    def all_offscreen(pts: List[Tuple[float, float]]) -> bool:
        for (x, y) in pts:
            az, d = xy_to_polar(x, y)
            
            if d < 0.25 or d > DIST_MAX_OFF or abs(az) < lo:
                return False
        return True

    def path_len(pts: List[Tuple[float, float]]) -> float:
        return sum(math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
                   for i in range(1, len(pts)))


    line_regions = [("left", "left"), ("right", "right"), ("back", "back"),
                    ("left", "back"), ("right", "back"), ("back", "left"), ("back", "right")]

    pts_xy: Optional[List[Tuple[float, float]]] = None
    for _ in range(240):
        if kind == "static":
            p = rand_pt(rng.choice(["left", "right", "back"]))
            cand = [(p[0] + rng.uniform(-0.05, 0.05), p[1] + rng.uniform(-0.05, 0.05)) for _ in ts]
        elif kind == "polyline":


            
            
            p0 = rand_pt(rng.choices(["back", "left", "right"], weights=[3, 1, 1])[0])
            step = rng.uniform(0.34, 0.5)   
            heading = rng.uniform(0.0, 360.0)
            turn = rng.uniform(TURN_DEG_LO, TURN_DEG_HI) * (1.0 if rng.random() < 0.5 else -1.0)
            h = n // 2
            cand = [p0]; cur = p0; hd = heading
            for i in range(1, n):
                if i == h:
                    hd += turn
                rad = math.radians(hd)
                cur = (cur[0] + step * math.cos(rad), cur[1] + step * math.sin(rad))
                cand.append(cur)
        else:  
            p0 = rand_pt(rng.choice(["left", "right", "back"]))
            step = rng.uniform(0.35, 0.6)
            rad = math.radians(rng.uniform(0.0, 360.0))
            dx, dy = step * math.cos(rad), step * math.sin(rad)
            cand = [(p0[0] + dx * i, p0[1] + dy * i) for i in range(n)]
        
        need_walk = kind in ("line", "polyline")
        if len(cand) == n and all_offscreen(cand) and (not need_walk or path_len(cand) >= WALK_MIN_DIST):
            pts_xy = cand
            break
    if pts_xy is None:                       
        p = rand_pt(rng.choice(["left", "right", "back"]))
        pts_xy = [p] * n

    out = []
    for i, (x, y) in enumerate(pts_xy):
        az, d = xy_to_polar(x, y)
        out.append({"t": ts[i], "azimuth": round(az, 1),
                    "elevation": round(el + rng.uniform(-2.0, 2.0), 1),
                    "distance": round(min(DIST_MAX_OFF, max(DIST_MIN, d)), 2),
                    "gain_db": 0.0, "visibility_state": "off_screen_always", "confidence": 0.5})
    return out


def _rot2d(v: Tuple[float, float], deg: float) -> Tuple[float, float]:
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    return (v[0] * c - v[1] * s, v[0] * s + v[1] * c)


def _world_seg_points(cam: Camera, anchor: Tuple[float, float],
                      dir_unit: Tuple[float, float], length: float,
                      times: List[float], t_u0: float, t_u1: float,
                      el0: float, el1: float, vis_state: str,
                      margin: float = 3.0) -> List[dict]:
    """OmniEcho processing component."""
    if not times:
        return []
    span = (t_u1 - t_u0) if abs(t_u1 - t_u0) > 1e-6 else 1e-6
    half = cam.half_hfov() + margin
    pts: List[dict] = []
    ok = True
    for tt in times:
        u = (tt - t_u0) / span
        x = anchor[0] + dir_unit[0] * length * u
        y = anchor[1] + dir_unit[1] * length * u
        az, d = xy_to_polar(x, y)


        d = min(DIST_MAX_OFF, max(DIST_MIN, d))
        el = el0 + (el1 - el0) * u
        if abs(cam.body_az(az, tt)) < half:
            ok = False
        pts.append({"t": round(tt, 3), "azimuth": round(az, 1),
                    "elevation": round(el, 1), "distance": round(d, 2),
                    "gain_db": 0.0, "visibility_state": vis_state, "confidence": 0.5})
    return pts if ok else []


def reshape_original_object(obj: dict, cam: Camera, rng: random.Random,
                            want_turn: bool = True) -> Optional[dict]:
    """OmniEcho processing component."""
    pts = obj.get("trajectory") or []
    if len(pts) < 2:
        return None
    times = [float(p["t"]) for p in pts]
    azs = [float(p.get("azimuth", p.get("az", 0.0))) for p in pts]
    els = [float(p.get("elevation", p.get("el", 0.0))) for p in pts]
    dis = [float(p.get("distance", p.get("dist", 1.0))) for p in pts]
    N = len(pts)

    on = [cam.visible(azs[i], els[i], times[i]) for i in range(N)]
    if not any(on):
        return None                                   
    k0 = on.index(True)
    k1 = N - 1 - on[::-1].index(True)                 


    d0 = rng.uniform(0.6, 2.2)
    d1 = rng.uniform(0.6, 2.2)
    seg_az = azs[k0:k1 + 1]
    mono = (all(seg_az[i + 1] >= seg_az[i] - 1e-6 for i in range(len(seg_az) - 1)) or
            all(seg_az[i + 1] <= seg_az[i] + 1e-6 for i in range(len(seg_az) - 1)))
    w0 = polar_to_xy(azs[k0], d0)
    w1 = polar_to_xy(azs[k1], d1)
    used_line = False
    new_dist = list(dis)
    if k1 > k0 and mono:
        ok = True
        cand = {}
        for i in range(k0, k1 + 1):
            t = ray_line_t(azs[i], w0, w1)
            if t is None or not (DIST_MIN <= t <= DIST_MAX + 0.3):
                ok = False
                break
            cand[i] = min(DIST_MAX, max(DIST_MIN, t))
        if ok:
            for i in range(k0, k1 + 1):
                new_dist[i] = cand[i]
            used_line = True
    if not used_line:                                 
        span = max(1, k1 - k0)
        for i in range(k0, k1 + 1):
            new_dist[i] = d0 + (d1 - d0) * ((i - k0) / span)

    head: List[dict] = []
    tail: List[dict] = []
    entered = exited = turned = False


    
    w0 = polar_to_xy(azs[k0], new_dist[k0])
    w1 = polar_to_xy(azs[k1], new_dist[k1])
    Vx, Vy = w1[0] - w0[0], w1[1] - w0[1]
    Vlen = math.hypot(Vx, Vy)
    seg_dt = max(1e-3, times[k1] - times[k0])
    crosses = (k0 > 0) or (k1 < N - 1)
    proj_total = (Vlen / seg_dt) * max(1e-3, times[-1] - times[0])
    is_mover = crosses or (Vlen >= 0.15 and proj_total >= MOVE_MIN_DIST)

    if not is_mover:
        
        vals = [new_dist[i] for i in range(k0, k1 + 1)]
        d_const = round(min(DIST_MAX, max(DIST_MIN, sum(vals) / len(vals))), 2)
        obj["trajectory"] = [
            {"t": round(times[i], 3), "azimuth": round(azs[i], 1),
             "elevation": round(els[i], 1), "distance": d_const,
             "gain_db": float(pts[i].get("gain_db", 0.0)),
             "visibility_state": "on_screen", "confidence": 1.0}
            for i in range(k0, k1 + 1)]
        obj["_reshaped"] = True
        obj["_entered_from_behind"] = False
        obj["_exits_screen"] = False
        obj["_motion_kind"] = "static"
        return {"entered": False, "exited": False, "line": False}


    
    if Vlen >= 0.15:
        Vn = (Vx / Vlen, Vy / Vlen)
        speed = Vlen / seg_dt                                
    else:                                                    
        p_a = polar_to_xy(azs[k0], 1.0)
        p_b = polar_to_xy(azs[k1], 1.0)
        dvx, dvy = p_b[0] - p_a[0], p_b[1] - p_a[1]
        dl = math.hypot(dvx, dvy) or 1.0
        Vn = (dvx / dl, dvy / dl)
        speed = max(0.3, MOVE_MIN_DIST / max(1e-3, times[-1] - times[0]))


    
    do_turn = want_turn and (rng.random() < 0.5)
    turn_side = rng.choice(["head", "tail"]) if do_turn else None
    turn_deg = rng.uniform(TURN_DEG_LO, TURN_DEG_HI) * (1.0 if rng.random() < 0.5 else -1.0)

    def _len_for(dt: float) -> float:
        
        return float(min(3.0, max(0.6, speed * dt)))


    if k0 > 0:
        head_times = [times[i] for i in range(0, k0)]
        L = _len_for(times[k0] - times[0])
        el_far = rng.uniform(-6.0, 6.0)
        d_head = _rot2d(Vn, turn_deg) if (do_turn and turn_side == "head") else Vn
        anchor = (w0[0] - d_head[0] * L, w0[1] - d_head[1] * L)   
        seg = _world_seg_points(cam, anchor, d_head, L, head_times,
                                times[0], times[k0], el_far, els[k0], "entering")
        if not seg and d_head is not Vn:                         
            anchor = (w0[0] - Vn[0] * L, w0[1] - Vn[1] * L)
            seg = _world_seg_points(cam, anchor, Vn, L, head_times,
                                    times[0], times[k0], el_far, els[k0], "entering")
        head = seg
        entered = bool(head)
        turned = turned or (do_turn and turn_side == "head" and bool(head))


    if k1 < N - 1:
        tail_times = [times[i] for i in range(k1 + 1, N)]
        L = _len_for(times[-1] - times[k1])
        el_far = rng.uniform(-6.0, 6.0)
        d_tail = _rot2d(Vn, turn_deg) if (do_turn and turn_side == "tail") else Vn
        seg = _world_seg_points(cam, w1, d_tail, L, tail_times,
                                times[k1], times[-1], els[k1], el_far, "exiting")
        if not seg and d_tail is not Vn:                         
            seg = _world_seg_points(cam, w1, Vn, L, tail_times,
                                    times[k1], times[-1], els[k1], el_far, "exiting")
        tail = seg
        exited = bool(tail)
        turned = turned or (do_turn and turn_side == "tail" and bool(tail))


    mid: List[dict] = []
    for i in range(k0, k1 + 1):
        mid.append({"t": round(times[i], 3), "azimuth": round(azs[i], 1),
                    "elevation": round(els[i], 1),
                    "distance": round(min(DIST_MAX, max(DIST_MIN, new_dist[i])), 2),
                    "gain_db": float(pts[i].get("gain_db", 0.0)),
                    "visibility_state": "on_screen", "confidence": 1.0})
    obj["trajectory"] = head + mid + tail

    obj["_reshaped"] = True
    obj["_entered_from_behind"] = entered
    obj["_exits_screen"] = exited
    
    obj["_motion_kind"] = ("polyline" if turned else ("line" if used_line else "seg"))
    if entered:
        obj["_entry_time"] = round(times[k0], 3)
        obj["_entry_side"] = "left" if cam.body_az(azs[k0 - 1], times[k0 - 1]) > 0 else "right"
    if exited:
        obj["_exit_time"] = round(times[k1], 3)
    return {"entered": entered, "exited": exited, "line": used_line}


def augment_scene(scene_raw: dict, sample_dir: Path, out_dir: Path, cam: Camera,
                  lib_pairs: List[Tuple[str, str]], sources_target: int, max_sources: int,
                  rng: random.Random, want_turn: bool = True
                  ) -> Tuple[dict, List[SourceTrack], List[SourceTrack]]:
    """OmniEcho processing component."""
    scene = json.loads(json.dumps(scene_raw))  # deep copy
    duration = float(scene.get("duration_sec", scene.get("duration", 6.0)))


    orig_objs = list(scene.get("objects", []))


    
    _n_before = len(orig_objs)
    orig_objs = [o for o in orig_objs if float(o.get("start_time", 0.0)) < duration]
    if len(orig_objs) < _n_before:
        logger.info("[augment] Dropped %d objects starting after duration %.2fs", _n_before - len(orig_objs), duration)
    for obj in orig_objs:
        wp = str(obj.get("wav_path", ""))
        obj["wav_path"] = str((sample_dir / wp).resolve()) if wp else wp
    ambient_objs: List[dict] = []
    nonamb_objs: List[dict] = []
    for o in orig_objs:
        (ambient_objs if is_ambient_label(str(o.get("label", o.get("id", "")))) else nonamb_objs).append(o)


    stats = {"entered": 0, "exited": 0, "line": 0}
    for obj in nonamb_objs:
        r = reshape_original_object(obj, cam, rng, want_turn=want_turn)
        if r:
            stats["entered"] += int(r["entered"])
            stats["exited"] += int(r["exited"])
            stats["line"] += int(r["line"])


    


    
    
    visible_origs = [o for o in nonamb_objs if o.get("_reshaped")]      
    invisible_origs = [o for o in nonamb_objs if not o.get("_reshaped")]  
    n_vis = len(visible_origs)


    
    
    tgt = max(1, int(sources_target))
    slots = max(0, tgt - n_vis)
    n_amb_keep = 0
    n_off = 0
    if slots >= 1:
        n_off = 1
        slots -= 1
        if slots >= 1 and ambient_objs:
            n_amb_keep = 1
            slots -= 1
        n_off += slots
    n_amb = min(len(ambient_objs), n_amb_keep)
    ambient_objs = ambient_objs[:n_amb]


    
    n_off = max(n_off, MIN_OFFSCREEN)
    n_off = max(0, min(n_off, ABS_MAX_SOURCES - (n_amb + n_vis)))

    original_tracks: List[SourceTrack] = [
        SourceTrack(o, is_offscreen=False) for o in (ambient_objs + visible_origs)]
    scene["objects"] = list(ambient_objs + visible_origs)   


    


    out_stems = out_dir / "stems"
    out_stems.mkdir(parents=True, exist_ok=True)
    used_labels = {str(o.get("label", "")).lower() for o in scene["objects"]}
    offscreen_tracks: List[SourceTrack] = []
    
    reuse = [o for o in invisible_origs if stem_is_audible(str(o.get("wav_path", "")))]
    n_reused = 0
    n_from_lib = 0


    
    line_static = (["line"] * 7 + ["static"] * 1)      
    off_kinds = [rng.choice(line_static) for _ in range(n_off)]
    if want_turn and n_off > 0:
        off_kinds[0] = "polyline"
    for i in range(n_off):
        traj = _offscreen_trajectory(rng, duration, off_kinds[i],
                                     cam.offscreen_min_abs_az())
        if reuse:                          
            obj = reuse.pop(0)
            obj["trajectory"] = traj
            obj["visual_enter"] = -1.0
            obj["visual_exit"] = -1.0
            obj["off_screen_strategy"] = "trust_metadata"
            obj["_reshaped"] = False
            obj["_behind_source"] = True
            used_labels.add(str(obj.get("label", "")).lower())
            n_reused += 1
        else:                              
            if not lib_pairs:
                continue
            lab, stem_src = None, None
            for _try in range(12):         
                cl, cp = lib_pairs[rng.randrange(len(lib_pairs))]
                if cl.lower() not in used_labels:
                    lab, stem_src = cl, Path(cp)
                    break
            if lab is None:                
                lab, stem_src = cl, Path(cp)
            if not stem_src.exists() or not stem_is_audible(str(stem_src)):
                continue                    
            used_labels.add(lab.lower())
            dst = out_stems / f"offscreen_{i:02d}{stem_src.suffix}"
            try:
                shutil.copyfile(stem_src, dst)
            except Exception:
                dst = stem_src  
            
            lvl = stem_level_dbfs(str(stem_src))
            gain = (max(-3.0, min(12.0, _OFFSCREEN_TARGET_RMS_DB - lvl[0]))
                    if lvl is not None else 0.0)
            obj = {
                "id": f"offscreen_{i:02d}",
                "label": lab,               
                "trajectory": traj,
                "acoustic_start": 0.0, "acoustic_end": duration,
                "visual_enter": -1.0, "visual_exit": -1.0,
                "off_screen_strategy": "trust_metadata",
                "spread_deg": 10.0, "gain_db": round(float(gain), 2),
                "start_time": 0.0, "end_time": duration,
                "onset_t": 0.0, "offset_t": duration,
                "wav_path": str(Path(dst).resolve()),
                "_behind_source": True,
            }
            n_from_lib += 1
        scene["objects"].append(obj)
        offscreen_tracks.append(SourceTrack(obj, is_offscreen=True))


    scene.setdefault("listener", {"position": [0, 0, 0], "orientation": {}})
    orient = scene["listener"].setdefault("orientation", {})
    orient.setdefault("yaw", 0.0)
    orient.setdefault("pitch", 0.0)
    orient.setdefault("roll", 0.0)
    if cam.rotating:
        n = 9
        orient["trajectory"] = [
            {"t": round(duration * k / (n - 1), 3),
             "yaw": round(cam.yaw(duration * k / (n - 1)), 3)}
            for k in range(n)
        ]
        scene["camera_mode"] = "rotate"
    else:


        orient["yaw"] = round(cam.yaw(0.0), 3)
        scene["camera_mode"] = "crop" if cam.cropped else "static"

    scene["_n_entered_from_behind"] = stats["entered"]
    scene["_source_stats"] = {
        "n_ambient": n_amb,
        "n_visible_orig_kept": len(visible_origs),      
        "n_offscreen": len(offscreen_tracks),           
        "n_behind_reused": n_reused,                    
        "n_invisible_orig_dropped": max(0, len(invisible_origs) - n_reused),
        "n_total": len(scene["objects"]),
        "n_entered": stats["entered"],
        "n_exited": stats["exited"],
        "n_line": stats["line"],
    }
    return scene, offscreen_tracks, original_tracks


def render_scene_foa(scene_path: Path, out_wav: Path) -> Tuple[int, int]:
    """OmniEcho processing component."""
    from omniecho_audio.render import render_foa, write_multichannel
    from omniecho_audio.schema import load_scene
    scene = load_scene(str(scene_path))
    foa = render_foa(scene)
    write_multichannel(out_wav, foa, scene.environment.sample_rate)
    return int(foa.shape[0]), int(scene.environment.sample_rate)


def foa_to_binaural(foa_wav: Path, out_wav: Path) -> bool:
    """OmniEcho processing component."""
    from omniecho_audio.render import render_foa_to_binaural
    render_foa_to_binaural(str(foa_wav), str(out_wav), hrtf_path=None)
    return out_wav.exists()


def pan_crop_video(src_video: Path, dst_video: Path, cam: Camera, frac_w: float) -> bool:
    """OmniEcho processing component."""
    import cv2
    cap = cv2.VideoCapture(str(src_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {src_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or cam.W
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or cam.H
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cw = int(round(frac_w * W))
    ch = int(round(cw * H / W))            
    ch = min(ch, H)
    y0 = (H - ch) // 2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dst_video), fourcc, fps, (W, H))
    idx = 0
    ok = True
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = idx / fps
        
        cx = cam.crop_center_x(t) * (W / cam.W)
        x0 = int(round(cx - cw / 2.0))
        x0 = max(0, min(W - cw, x0))
        crop = frame[y0:y0 + ch, x0:x0 + cw]
        crop = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)
        writer.write(crop)
        idx += 1
        if nframes and idx >= nframes:
            break
    cap.release()
    writer.release()
    return ok and idx > 0


def probe_video_duration(video_path: Path) -> Optional[float]:
    """OmniEcho processing component."""
    if shutil.which("ffprobe") is None:
        return None
    try:
        ret = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(video_path)],
            capture_output=True, text=True)
        if ret.returncode == 0 and ret.stdout.strip():
            return float(ret.stdout.strip())
    except Exception:
        pass
    return None


def mux_audio_into_video(video_path: Path, audio_path: Path, out_path: Path,
                         codec: str = "aac", bitrate: str = "512k") -> None:
    """OmniEcho processing component."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg not found")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-i", str(video_path), "-i", str(audio_path),
           "-map", "0:v:0", "-map", "1:a:0",
           "-c:v", "copy", "-c:a", codec, "-b:a", bitrate,
           "-shortest", str(out_path)]
    ret = subprocess.run(cmd, capture_output=True, text=True)
    if ret.returncode != 0:
        raise RuntimeError(f"FFmpeg muxing failed: {ret.stderr.strip()}")


def _az_to_plot(az_deg: float, r: float) -> Tuple[float, float]:
    a = math.radians(az_deg)
    return (-r * math.sin(a), r * math.cos(a))


def render_topdown_video(all_tracks: List[SourceTrack], cam: Camera, duration: float,
                         out_path: Path, audio_wav: Optional[Path], sid: str,
                         fps: int = 12) -> bool:
    """OmniEcho processing component."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge, Circle
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import cv2

    R = 3.0                                     
    cmap = plt.get_cmap("tab10")
    tracks = list(all_tracks)
    colors = [cmap(i % 10) for i in range(len(tracks))]
    phrases = [_phrase_of(tr)[0].replace("a ", "").replace("an ", "") for tr in tracks]

    n_frames = max(1, int(round(duration * fps)))
    fig = plt.figure(figsize=(6.4, 6.4), dpi=100)
    canvas = FigureCanvasAgg(fig)
    frame_size = None
    tmp_silent = out_path.with_name(out_path.stem + "_silent.mp4")
    writer = None

    for fi in range(n_frames):
        t = min(duration, fi / fps)
        fig.clf()
        ax = fig.add_subplot(111)
        ax.set_xlim(-R, R); ax.set_ylim(-R, R)
        ax.set_aspect("equal"); ax.axis("off")


        for rr in (1.0, 2.0, 3.0):
            ax.add_patch(Circle((0, 0), rr, fill=False, ec="0.85", lw=0.8, zorder=0))
            ax.text(0.03, rr, f"{rr:.0f}m", color="0.6", fontsize=6, va="bottom")
        ax.text(0, R * 0.98, "front", ha="center", va="top", color="0.5", fontsize=8)
        ax.text(0, -R * 0.98, "back", ha="center", va="bottom", color="0.5", fontsize=8)
        ax.text(-R * 0.98, 0, "left", ha="left", va="center", color="0.5", fontsize=8)
        ax.text(R * 0.98, 0, "right", ha="right", va="center", color="0.5", fontsize=8)


        yaw = cam.yaw(t)
        half = cam.half_hfov()


        th_center = 90.0 + yaw
        ax.add_patch(Wedge((0, 0), R, th_center - half, th_center + half,
                           facecolor="#4c78a8", alpha=0.12, ec="#4c78a8", lw=0.8, zorder=1))
        
        ex, ey = _az_to_plot(yaw, R * 0.5)
        ax.annotate("", xy=(ex, ey), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="#4c78a8", lw=1.4), zorder=2)


        ax.plot(0, 0, marker="^", ms=11, color="#222", zorder=5)
        ax.text(0, -0.18, "camera", ha="center", va="top", color="#222", fontsize=7)


        for tr, col, ph in zip(tracks, colors, phrases):
            
            ts_tail = np.linspace(max(0.0, t - 1.5), t, 10)
            tail = [_az_to_plot(tr.world_az(float(tt)),
                                min(R, max(0.2, tr.distance(float(tt))))) for tt in ts_tail]
            if len(tail) >= 2:
                ax.plot([p[0] for p in tail], [p[1] for p in tail],
                        color=col, alpha=0.35, lw=1.2, zorder=3)
            r = min(R, max(0.2, tr.distance(t)))
            px, py = _az_to_plot(tr.world_az(t), r)
            visible = (not tr.is_offscreen) and cam.visible(tr.world_az(t), tr.elevation(t), t)
            if visible:
                ax.plot(px, py, "o", ms=11, color=col, zorder=6)
            else:
                ax.plot(px, py, "o", ms=11, mfc="none", mec=col, mew=1.8, zorder=6)
            tag = ph + (" (enters)" if tr.obj.get("_entered_from_behind") else "")
            ax.text(px, py + 0.16, tag, ha="center", va="bottom",
                    color=col, fontsize=6.5, zorder=7)

        ax.set_title(f"{sid}   t={t:4.1f}s   yaw={yaw:+5.1f}°   "
                     f"(● in-view  ○ off-screen)", fontsize=9)
        canvas.draw()
        img = np.asarray(canvas.buffer_rgba())[..., :3]     # RGB (H, W, 3)
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if writer is None:
            frame_size = (frame.shape[1], frame.shape[0])
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(tmp_silent), fourcc, float(fps), frame_size)
        writer.write(frame)

    if writer is not None:
        writer.release()
    plt.close(fig)


    if audio_wav is not None and audio_wav.exists() and shutil.which("ffmpeg") is not None:
        try:
            mux_audio_into_video(tmp_silent, audio_wav, out_path)
            tmp_silent.unlink(missing_ok=True)
        except Exception:
            tmp_silent.replace(out_path)          
    else:
        tmp_silent.replace(out_path)
    return out_path.exists()


def _phrase_of(track: SourceTrack) -> Tuple[str, str]:
    """OmniEcho processing component."""
    return _label_to_phrase(track.label), ""


def _world_xy(track: SourceTrack, t: float) -> Tuple[float, float]:
    az = math.radians(track.world_az(t))
    d = track.distance(t)
    return d * math.cos(az), d * math.sin(az)     # X=front, Y=left


def _annotations(track: SourceTrack, cam: Camera, t: float) -> dict:
    waz = track.world_az(t)
    yaw = cam.yaw(t)
    baz = wrap_deg(waz - yaw)
    wx, wy = _world_xy(track, t)
    phrase_en, sound_zh = _phrase_of(track)
    return {
        "object_id": str(track.obj.get("id", "")),          
        "object_label_raw": str(track.label),                
        "object_label_en": phrase_en,
        "object_label_zh": sound_zh,
        "sound_type_zh": sound_zh,
        "sound_phrase_en": phrase_en,
        "world_x": round(wx, 2), "world_y": round(wy, 2),
        "world_azimuth_deg": round(waz, 1),
        "camera_yaw_theta_deg": round(yaw, 1),
        "body_azimuth_deg": round(baz, 1),
        "distance_m": round(track.distance(t), 2),
        "visible_in_main": bool(cam.visible(waz, track.elevation(t), t)),
    }


def _track_visible_at(track: SourceTrack, cam: Camera, t: float) -> bool:
    """OmniEcho processing component."""
    if track.is_offscreen:
        return False
    return cam.visible(track.world_az(t), track.elevation(t), t)


def _fmt_time(t: float) -> str:
    return f"{t:.1f}"


def make_qa(subtype: str, track: SourceTrack, cam: Camera, duration: float,
            t: float, a: float, b: float) -> Optional[dict]:
    """OmniEcho processing component."""
    phrase, _ = _phrase_of(track)

    if subtype == "visibility_at_time":
        vis = _track_visible_at(track, cam, t)
        return dict(category="visibility", subtype="at_time", coordinate_frame="camera",
                    time=round(t, 1), interval=None,
                    base_question=f"At t={_fmt_time(t)} s, is the source of {phrase} visible in the main camera view?",
                    answer="yes" if vis else "no", gt={"visible": bool(vis)})

    if subtype == "visibility_over_interval":
        ts = np.linspace(a, b, 11)
        vis_all = all(_track_visible_at(track, cam, float(tt)) for tt in ts)
        return dict(category="visibility", subtype="over_interval", coordinate_frame="camera",
                    time=None, interval=[round(a, 1), round(b, 1)],
                    base_question=(f"Is the source of {phrase} visible in the main camera continuously "
                                   f"throughout the interval t=[{_fmt_time(a)}, {_fmt_time(b)}] s?"),
                    answer="yes" if vis_all else "no", gt={"visible_all": bool(vis_all)})

    if subtype == "direction_approx_body":
        baz = cam.body_az(track.world_az(t), t)
        return dict(category="direction", subtype="approx_body", coordinate_frame="camera-body",
                    time=round(t, 1), interval=None,
                    base_question=(f"At t={_fmt_time(t)} s, in the CAMERA-BODY frame (relative to where the "
                                   f"camera is currently pointing), in which approximate direction is the source "
                                   f"of {phrase}? Answer with one of: {DIR8_OPTIONS_STR}."),
                    answer=dir8_from_az(baz), gt={"body_azimuth_deg": round(baz, 1)})

    if subtype == "direction_degrees_body":
        baz = cam.body_az(track.world_az(t), t)
        turn, deg = turn_and_degrees(baz)
        return dict(category="direction", subtype="degrees_body", coordinate_frame="camera-body",
                    time=round(t, 1), interval=None,
                    base_question=(f"At t={_fmt_time(t)} s, by how many degrees and in which direction must the "
                                   f"camera rotate (left or right, about the vertical axis) so that the source of "
                                   f"{phrase} becomes centred in its view?"),
                    answer=f"{deg} degrees to the {turn}", gt={"turn": turn, "degrees": round(abs(baz), 1)})

    if subtype == "motion_world_heading":
        wx0, wy0 = _world_xy(track, a)
        wx1, wy1 = _world_xy(track, b)
        heading = math.degrees(math.atan2(wy1 - wy0, wx1 - wx0))
        return dict(category="motion", subtype="world_heading", coordinate_frame="world",
                    time=None, interval=[round(a, 1), round(b, 1)],
                    base_question=(f"In the WORLD frame (fixed to the camera's initial heading; X = front, "
                                   f"Y = left), what is the overall direction of travel of the source of {phrase} "
                                   f"from t={_fmt_time(a)} to t={_fmt_time(b)} s? Answer with one of: {DIR8_OPTIONS_STR}."),
                    answer=dir8_from_az(heading), gt={"heading_deg": round(heading, 1)})

    if subtype == "motion_change":


        ts = np.linspace(a, b, 33)
        pts = [_world_xy(track, float(tt)) for tt in ts]
        max_turn = _robust_max_turn([p[0] for p in pts], [p[1] for p in pts])
        changed = max_turn > TURN_CLASSIFY_DEG
        return dict(category="motion", subtype="change", coordinate_frame="world",
                    time=None, interval=[round(a, 1), round(b, 1)],
                    base_question=(f"In the WORLD frame, does the direction of motion of the source of {phrase} "
                                   f"change significantly (by more than ~{int(MOTION_CHANGE_DEG)} deg) at any point "
                                   f"during its trajectory?"),
                    answer="yes" if changed else "no",
                    gt={"changed": bool(changed), "max_turn_deg": round(max_turn, 1)})

    if subtype == "localization3d_body":
        baz = cam.body_az(track.world_az(t), t)
        dist = min(DIST_MAX, max(DIST_MIN, track.distance(t)))
        d1 = round(dist, 1)
        return dict(category="localization3d", subtype="body", coordinate_frame="camera-body",
                    time=round(t, 1), interval=None,
                    base_question=(f"At t={_fmt_time(t)} s, give the position of the source of {phrase} in the "
                                   f"CAMERA-BODY frame (relative to where the camera is currently pointing) as an "
                                   f"approximate direction (one of the eight sectors) plus a distance in metres, "
                                   f"in the form '<direction>, about <D> m'."),
                    answer=f"{dir8_from_az(baz)}, about {d1} m",
                    gt={"body_azimuth_deg": round(baz, 1), "distance_m": d1})

    if subtype == "camera_rotation_degrees":
        dtheta = cam.yaw(b) - cam.yaw(a)
        turn, deg = turn_and_degrees(dtheta)
        return dict(category="camera_rotation", subtype="degrees", coordinate_frame="camera",
                    time=None, interval=[round(a, 1), round(b, 1)],
                    base_question=(f"Between t={_fmt_time(a)} and t={_fmt_time(b)} s, by how many degrees and in "
                                   f"which direction (left or right) did the camera rotate about its vertical axis?"),
                    answer=f"{deg} degrees to the {turn}",
                    gt={"turn": turn, "degrees": round(abs(dtheta), 1)})

    if subtype == "camera_rotation_consistency":
        dtheta = cam.yaw(b) - cam.yaw(a)
        
        ts = np.linspace(a, b, 13)
        azs = np.unwrap(np.deg2rad([track.world_az(float(tt)) for tt in ts]))
        src_daz = math.degrees(azs[-1] - azs[0])
        consistent = (dtheta == 0 and abs(src_daz) < 5.0) or (np.sign(dtheta) == np.sign(src_daz) and dtheta != 0)
        return dict(category="camera_rotation", subtype="consistency", coordinate_frame="mixed",
                    time=None, interval=[round(a, 1), round(b, 1)],
                    base_question=(f"Between t={_fmt_time(a)} and t={_fmt_time(b)} s, does the camera turn in the "
                                   f"same overall direction (left or right) that the sounding source travels around "
                                   f"the camera?"),
                    answer="yes" if consistent else "no",
                    gt={"consistent": bool(consistent), "camera_dtheta_deg": round(dtheta, 1),
                        "source_cumulative_daz_deg": round(src_daz, 1)})
    return None


def _floor1(x: float) -> float:
    """OmniEcho processing component."""
    return math.floor(x * 10.0) / 10.0


def _pick_time(rng: random.Random, duration: float) -> float:
    lo, hi = 0.5, max(0.5, _floor1(duration) - 0.5)
    return round(rng.uniform(lo, hi) * 2) / 2.0     


def _pick_interval(rng: random.Random, duration: float) -> Tuple[float, float]:
    end = _floor1(duration)
    if rng.random() < 0.6:
        return 0.0, end                              
    a = round(rng.uniform(0.0, duration * 0.4), 1)
    b = round(rng.uniform(duration * 0.6, duration), 1)
    return a, min(b, end)


def build_sample_qas(sid: str, cam: Camera, duration: float,
                     original_tracks: List[SourceTrack], offscreen_tracks: List[SourceTrack],
                     merged_av_rel: str, rng: random.Random,
                     want_turn: bool = True) -> List[dict]:
    """OmniEcho processing component."""


    
    non_ambient_orig = [t for t in original_tracks if not is_ambient_label(t.label) and t.audible]
    offscreen_tracks = [t for t in offscreen_tracks if t.audible]
    visible_src = [t for t in non_ambient_orig if any(
        _track_visible_at(t, cam, tt) for tt in np.linspace(0.3, duration - 0.3, 8))]

    def ahead_time(track: SourceTrack) -> Optional[float]:
        """OmniEcho processing component."""
        cands = [float(tt) for tt in np.linspace(0.4, duration - 0.4, 12)
                 if abs(cam.body_az(track.world_az(tt), tt)) < 40.0 and _track_visible_at(track, cam, float(tt))]
        return rng.choice(cands) if cands else None

    def visible_time(track: SourceTrack) -> Optional[float]:
        cands = [float(tt) for tt in np.linspace(0.4, duration - 0.4, 12)
                 if _track_visible_at(track, cam, float(tt))]
        return rng.choice(cands) if cands else None

    def visible_interval(track: SourceTrack) -> Optional[Tuple[float, float]]:
        """OmniEcho processing component."""
        grid = np.linspace(0.2, duration - 0.2, 16)
        vis = [_track_visible_at(track, cam, float(tt)) for tt in grid]
        best = None
        i = 0
        while i < len(grid):
            if vis[i]:
                j = i
                while j + 1 < len(grid) and vis[j + 1]:
                    j += 1
                if j > i and (best is None or (grid[j] - grid[i]) > (best[1] - best[0])):
                    best = (round(float(grid[i]), 1), round(float(grid[j]), 1))
                i = j + 1
            else:
                i += 1
        return best


    front_src = [t for t in visible_src if ahead_time(t) is not None]
    
    dir_src = offscreen_tracks or non_ambient_orig


    moving_src = ([t for t in (list(offscreen_tracks) + list(non_ambient_orig))
                   if t.motion_class() != "static"] or offscreen_tracks or dir_src)

    def pick(lst: List[SourceTrack]) -> Optional[SourceTrack]:
        return rng.choice(lst) if lst else None

    plan: List[Tuple[str, SourceTrack, dict]] = []

    def add(subtype: str, track: Optional[SourceTrack], anchors: dict):
        if track is not None:
            plan.append((subtype, track, anchors))


    


    
    
    all_dir = list(offscreen_tracks) + list(non_ambient_orig)      
    t_end = _floor1(duration)                                       
    grid = [round(float(x), 1) for x in np.linspace(0.4, max(0.6, t_end - 0.4), 13)]
    _hd = round(duration / 2.0, 1)
    ivs = [(0.0, t_end), (0.0, _hd), (_hd, t_end)]
    dir_tally: dict = {}
    head_tally: dict = {}
    turn_tally: dict = {}
    yn_tally: dict = {}
    obj_tally: dict = {}      

    def _bump_obj(tr: SourceTrack):
        obj_tally[tr.label] = obj_tally.get(tr.label, 0) + 1

    def _least(cands: list, tally: dict):
        """OmniEcho processing component."""
        if not cands:
            return None
        best = min(cands, key=lambda c: (tally.get(c[0], 0),
                                         obj_tally.get(c[1].label, 0), rng.random()))
        tally[best[0]] = tally.get(best[0], 0) + 1
        _bump_obj(best[1])
        return best

    def pick_balanced(lst: List[SourceTrack]) -> Optional[SourceTrack]:
        """OmniEcho processing component."""
        if not lst:
            return None
        ch = min(lst, key=lambda tr: (obj_tally.get(tr.label, 0), rng.random()))
        _bump_obj(ch)
        return ch

    def _body_time_ok(tr: SourceTrack, t: float) -> bool:
        
        return tr.is_offscreen or _track_visible_at(tr, cam, t)

    def add_balanced_body(subtype: str, n: int):
        for _ in range(n):
            cands = [(dir8_from_az(cam.body_az(tr.world_az(t), t)), tr, {"time": t})
                     for tr in all_dir for t in grid if _body_time_ok(tr, t)]
            ch = _least(cands, dir_tally)
            if ch:
                add(subtype, ch[1], ch[2])

    def add_balanced_heading(n: int):
        cands = []
        for tr in all_dir:
            if tr.motion_class() == "static":
                continue
            for (a, b) in ivs:
                x0, y0 = _world_xy(tr, a); x1, y1 = _world_xy(tr, b)
                if math.hypot(x1 - x0, y1 - y0) < 0.2:
                    continue
                sec = dir8_from_az(math.degrees(math.atan2(y1 - y0, x1 - x0)))
                cands.append((sec, tr, {"interval": (a, b)}))
        for _ in range(n):
            ch = _least(list(cands), head_tally)
            if ch:
                add("motion_world_heading", ch[1], ch[2])

    def add_balanced_degrees(n: int):
        for _ in range(n):
            cands = [(("left" if cam.body_az(tr.world_az(t), t) > 0 else "right"), tr, {"time": t})
                     for tr in all_dir for t in grid
                     if abs(cam.body_az(tr.world_az(t), t)) > 5.0 and _body_time_ok(tr, t)]
            ch = _least(cands, turn_tally)
            if ch:
                add("direction_degrees_body", ch[1], ch[2])

    def add_visibility(subtype: str):
        """OmniEcho processing component."""
        want_yes = rng.random() < (VIS_OVER_YES_BIAS if subtype == "visibility_over_interval" else VIS_YES_BIAS)
        if subtype == "visibility_over_interval":
            if want_yes and visible_src:
                vtr = pick_balanced(visible_src); vi = visible_interval(vtr)
                if vi:
                    add(subtype, vtr, dict(interval=vi)); yn_tally["yes"] = yn_tally.get("yes", 0) + 1
                    return
            add(subtype, pick_balanced(dir_src), dict(interval=_pick_interval(rng, duration)))
            yn_tally["no"] = yn_tally.get("no", 0) + 1
        else:                                            # visibility_at_time
            if want_yes and visible_src:
                atr = pick_balanced(visible_src); vt = visible_time(atr)
                if vt is not None:
                    add(subtype, atr, dict(time=vt)); yn_tally["yes"] = yn_tally.get("yes", 0) + 1
                    return
            add(subtype, pick_balanced(dir_src), dict(time=_pick_time(rng, duration)))
            yn_tally["no"] = yn_tally.get("no", 0) + 1

    def add_motion_change(n: int):
        """OmniEcho processing component."""
        turn_srcs = [t for t in moving_src if t.motion_class() == "turn"]
        line_srcs = [t for t in moving_src if t.motion_class() == "line"]
        use_turn = want_turn and bool(turn_srcs)
        mc_src = turn_srcs if use_turn else (line_srcs or moving_src)
        for _ in range(n):
            if mc_src:
                add("motion_change", pick_balanced(mc_src), dict(interval=(0.0, t_end)))


    add_visibility("visibility_over_interval")
    
    add_visibility("visibility_at_time")
    add_visibility("visibility_at_time")
    add_balanced_body("direction_approx_body", 4)        
    add_balanced_body("localization3d_body", 3)
    add_balanced_degrees(3)
    add_balanced_heading(3)                              
    add_motion_change(1)                                 


    if cam.rotating:
        add("camera_rotation_degrees", pick_balanced(dir_src), dict(interval=_pick_interval(rng, duration)))
        
        a_cr, b_cr = 0.0, t_end
        dtheta_cr = cam.yaw(b_cr) - cam.yaw(a_cr)
        def _cons_of(tr: SourceTrack) -> bool:
            ts = np.linspace(a_cr, b_cr, 13)
            azs = np.unwrap(np.deg2rad([tr.world_az(float(tt)) for tt in ts]))
            sdaz = math.degrees(azs[-1] - azs[0])
            return (dtheta_cr == 0 and abs(sdaz) < 5.0) or (np.sign(dtheta_cr) == np.sign(sdaz) and dtheta_cr != 0)
        want_cons = rng.random() < 0.5
        cons_cands = [t for t in moving_src if _cons_of(t) == want_cons] or moving_src
        add("camera_rotation_consistency", pick_balanced(cons_cands), dict(interval=(a_cr, b_cr)))


    
    _bump = 0
    while len(plan) < QA_PER_MIN and _bump < 40:
        before = len(plan)
        add_balanced_body("direction_approx_body" if _bump % 2 == 0 else "localization3d_body", 1)
        _bump += 1
        if len(plan) == before:                          
            break
    if len(plan) > QA_PER_MAX:
        del plan[QA_PER_MAX:]


    out: List[dict] = []
    for i, (subtype, track, anchors) in enumerate(plan):
        t = anchors.get("time", 0.0)
        a, b = anchors.get("interval", (0.0, t_end))
        core = make_qa(subtype, track, cam, duration, float(t), float(a), float(b))
        if core is None:
            continue
        anchor_t = core["time"] if core["time"] is not None else float(b)
        ann = _annotations(track, cam, anchor_t)
        rec = {
            "sample_id": sid,
            "qa_id": f"{sid}_q{i:02d}",
            "category": core["category"], "subtype": core["subtype"],
            "coordinate_frame": core["coordinate_frame"],
            "time": core["time"], "interval": core["interval"],
            "question": core["base_question"],         
            "base_question": core["base_question"],
            "answer": core["answer"], "answer_bucket": answer_bucket(core["answer"]),
            "gt": core["gt"], "annotations": ann,
            "media": {"merged_av": merged_av_rel, "duration_s": round(duration, 3)},
        }
        out.append(rec)
    return out


def to_message_record(rec: dict) -> dict:
    """OmniEcho processing component."""
    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "video", "video": rec["media"]["merged_av"],
                 "use_audio": True, "use_multichannel": 4},
                {"type": "text", "text": rec["question"] + "\nAnswer with a single word or short phrase."},
            ]},
            {"role": "assistant", "content": rec["answer"]},
        ]
    }


def _is_sample_dir(d: Path) -> bool:
    if not d.is_dir() or not (d / "video.mp4").exists():
        return False
    return (d / "corrected_scene.json").exists() or (d / "scene.json").exists()


def process_sample(task: dict) -> dict:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s [%(processName)s] %(message)s")
    sample_dir = Path(task["sample_dir"])
    sid = sample_dir.name
    out_dir = Path(task["out_root"]) / sid
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(task["seed"])
    try:
        scene_path = (sample_dir / "corrected_scene.json")
        if not scene_path.exists():
            scene_path = sample_dir / "scene.json"
        scene_raw = json.loads(scene_path.read_text())
        scene_dur = float(scene_raw.get("duration_sec", scene_raw.get("duration", 6.0)))


        


        src_video = sample_dir / "video.mp4"
        vid_dur = probe_video_duration(src_video)
        duration = round(min(scene_dur, vid_dur), 3) if vid_dur else scene_dur
        scene_raw["duration_sec"] = duration
        fov = float(scene_raw.get("fov_horizontal_deg", 90.0))
        W = int(scene_raw.get("frame_width", 1280))
        H = int(scene_raw.get("frame_height", 720))


        f_px = (W / 2.0) / math.tan(math.radians(fov / 2.0))
        cam_mode = task.get("cam_mode", "rotate" if task.get("rotating") else "full")
        cropped = cam_mode in ("rotate", "crop")
        frac_w = task["frac_w"] if cropped else 1.0
        if cam_mode == "rotate":
            yaw_start, yaw_end = 0.0, choose_yaw_end(rng, f_px, W, frac_w)
        elif cam_mode == "crop":
            
            yaw_start = yaw_end = choose_yaw_end(rng, f_px, W, frac_w)
        else:                                   # full
            yaw_start = yaw_end = 0.0
        cam = Camera(duration, yaw_end, fov, W, H, frac_w=frac_w, yaw_start_deg=yaw_start)
        rotating = cam.rotating


        
        want_turn = rng.random() < TURN_SAMPLE_FRAC


        scene_aug, off_tracks, orig_tracks = augment_scene(
            scene_raw, sample_dir, out_dir, cam, task["lib_pairs"],
            task["sources_target"], task["max_sources"], rng, want_turn=want_turn)
        aug_scene_path = out_dir / "augmented_scene.json"
        aug_scene_path.write_text(json.dumps(scene_aug, ensure_ascii=False, indent=2))

        merged_dir = Path(task["out_root"]) / "merged_av"
        merged_av = merged_dir / f"{sid}.mp4"


        
        
        if task.get("qa_only"):
            if not merged_av.exists():        
                return {"sample": sid, "ok": False, "rotating": rotating, "n_qa": 0, "qas": [],
                        "error": "qa_only: merged_av missing (original rendering failed)"}
            merged_av_str = str(merged_av.resolve())
            qas = build_sample_qas(sid, cam, duration, orig_tracks, off_tracks, merged_av_str, rng,
                                   want_turn=want_turn)
            _ss = scene_aug.get("_source_stats", {})
            return {"sample": sid, "ok": True, "rotating": rotating, "cam_mode": cam_mode,
                    "yaw_end": round(yaw_end, 1),
                    "yaw_fixed": round(yaw_start, 1) if cam_mode == "crop" else 0.0,
                    "n_qa": len(qas), "qas": qas,
                    "n_entered": int(scene_aug.get("_n_entered_from_behind", 0)),
                    "n_sources": _ss.get("n_total", 0), "src_stats": _ss,
                    "topdown": None, "error": None}


        foa_wav = out_dir / "audio_foa_aligned.wav"
        nch, sr = render_scene_foa(aug_scene_path, foa_wav)
        
        binaural_wav = out_dir / "audio_binaural_aligned.wav"
        try:
            foa_to_binaural(foa_wav, binaural_wav)
        except Exception as e:
            logger.warning("[%s] FOA-to-binaural conversion failed; using FOA for the top-down video: %s", sid, e)
            binaural_wav = foa_wav


        merged_dir = Path(task["out_root"]) / "merged_av"
        merged_av = merged_dir / f"{sid}.mp4"
        if cropped:
            tmp_video = out_dir / "video_crop.mp4"
            pan_crop_video(src_video, tmp_video, cam, frac_w)
            mux_audio_into_video(tmp_video, foa_wav, merged_av)
        else:
            mux_audio_into_video(src_video, foa_wav, merged_av)

        merged_av_str = str(merged_av.resolve())
        qas = build_sample_qas(sid, cam, duration, orig_tracks, off_tracks, merged_av_str, rng,
                               want_turn=want_turn)


        topdown_str = None
        if task.get("topdown", True):
            topdown_path = out_dir / f"topdown_{sid}.mp4"
            try:
                render_topdown_video(orig_tracks + off_tracks, cam, duration,
                                     topdown_path, binaural_wav, sid,
                                     fps=int(task.get("topdown_fps", 12)))
                topdown_str = str(topdown_path.resolve())
            except Exception as e:
                logger.warning("[%s] Top-down video generation failed: %s", sid, e)

        n_entered = int(scene_aug.get("_n_entered_from_behind", 0))
        src_stats = scene_aug.get("_source_stats", {})
        return {"sample": sid, "ok": True, "rotating": rotating, "cam_mode": cam_mode,
                "yaw_end": round(yaw_end, 1), "yaw_fixed": round(yaw_start, 1) if cam_mode == "crop" else 0.0,
                "n_qa": len(qas), "qas": qas,
                "n_entered": n_entered, "n_sources": src_stats.get("n_total", 0),
                "src_stats": src_stats, "topdown": topdown_str, "error": None}
    except Exception as e:
        return {"sample": sid, "ok": False, "rotating": task.get("rotating"),
                "n_qa": 0, "qas": [], "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()}


def discover_samples(input_path: Path) -> List[Path]:
    if _is_sample_dir(input_path):
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    return sorted(d for d in input_path.iterdir() if _is_sample_dir(d))


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(processName)s] %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="Batch directory containing sample subdirectories, or one sample directory")
    ap.add_argument("--out", type=Path, required=True, help="Output root directory")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--rotating-ratio", type=float, default=7.0 / 15.0,
                    help="Fraction of rotating samples (default about 0.47, matching the real 8-static:7-rotating ratio)")
    ap.add_argument("--sources-target", type=int, default=0,
                    help="Total sources per sample, including ambient, original, and off-screen sources. "
                         "Default 0 samples a weighted count from 2..5; a positive integer fixes the count for all samples")
    ap.add_argument("--max-sources", type=int, default=4,
                    help="Hard cap on sources per sample, including background (default: 4)")
    ap.add_argument("--frac-w", type=float, default=0.6, help="Crop width fraction, shared by rotating and fixed crops")
    ap.add_argument("--crop-static", dest="crop_static", action="store_true", default=True,
                    help="Apply a fixed crop to nonrotating samples as well, keeping its window and offset constant; "
                         "this places some objects off-screen throughout for greater diversity (enabled by default)")
    ap.add_argument("--no-crop-static", dest="crop_static", action="store_false",
                    help="Use the original full frame for nonrotating samples, without cropping")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stem-root", type=Path, default=None,
                    help="Root for scanning the stem pool (default: --input); used to borrow stems/*.wav across samples")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--qa-only", dest="qa_only", action="store_true", default=False,
                    help="Reuse rendered media and recompute QA with current balancing settings, rewriting both JSONL files without rerendering")
    ap.add_argument("--topdown", dest="topdown", action="store_true", default=True,
                    help="Generate top-down diagnostic videos (enabled by default)")
    ap.add_argument("--no-topdown", dest="topdown", action="store_false",
                    help="Disable top-down diagnostic videos")
    ap.add_argument("--topdown-fps", type=int, default=12, help="Top-down video frame rate")
    args = ap.parse_args()

    samples = discover_samples(args.input)
    if not samples:
        logger.error("No valid samples found; each must contain video.mp4 and scene.json.")
        sys.exit(1)

    stem_root = args.stem_root or (args.input if args.input.is_dir() and not _is_sample_dir(args.input)
                                   else args.input.parent)


    lib_pairs = build_sound_library(stem_root)
    logger.info("Found %d samples; global stem library: %d records (root=%s), %d unique labels",
                len(samples), len(lib_pairs), stem_root,
                len({l for l, _ in lib_pairs}))
    if not lib_pairs:
        logger.warning("Stem library is empty; off-screen sources will be skipped, limiting directional coverage.")

    args.out.mkdir(parents=True, exist_ok=True)
    
    try:
        (args.out / "sound_library.json").write_text(
            json.dumps([{"label": l, "stem": s} for l, s in lib_pairs],
                       ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to write sound_library.json: %s", e)

    rng = random.Random(args.seed)
    
    order = list(range(len(samples)))
    rng.shuffle(order)
    n_rot = int(round(args.rotating_ratio * len(samples)))
    rot_set = set(order[:n_rot])


    _src_choices, _src_weights = [2, 3, 4], [0.35, 0.40, 0.25]

    tasks = []
    for i, d in enumerate(samples):
        if args.skip_existing and (args.out / "merged_av" / f"{d.name}.mp4").exists():
            logger.info("Skipping existing sample: %s", d.name)
            continue
        if i in rot_set:
            cam_mode = "rotate"
        else:
            cam_mode = "crop" if args.crop_static else "full"
        src_tgt = (args.sources_target if args.sources_target > 0
                   else rng.choices(_src_choices, weights=_src_weights)[0])
        tasks.append({
            "sample_dir": str(d), "out_root": str(args.out),
            "rotating": i in rot_set, "cam_mode": cam_mode,
            "sources_target": src_tgt, "max_sources": args.max_sources,
            "frac_w": args.frac_w, "lib_pairs": lib_pairs,
            "seed": args.seed + 1000 + i,
            "topdown": args.topdown, "topdown_fps": args.topdown_fps,
            "qa_only": args.qa_only,
        })
    _sc = collections.Counter(t["sources_target"] for t in tasks)
    logger.info("Source-count distribution: %s", dict(sorted(_sc.items())))
    _cnt = collections.Counter(t["cam_mode"] for t in tasks)
    logger.info("Pending samples: %d (rotate=%d, crop=%d, full=%d, workers=%d)",
                len(tasks), _cnt.get("rotate", 0), _cnt.get("crop", 0),
                _cnt.get("full", 0), args.workers)

    results: List[dict] = []
    if args.workers <= 1:
        for i, task in enumerate(tasks, 1):
            res = process_sample(task)
            results.append(res)
            _log_res(res, i, len(tasks))
    else:
        ctx = get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
            fut2n = {ex.submit(process_sample, t): Path(t["sample_dir"]).name for t in tasks}
            done = 0
            for fut in as_completed(fut2n):
                done += 1
                try:
                    res = fut.result()
                except Exception as e:
                    res = {"sample": fut2n[fut], "ok": False, "n_qa": 0, "qas": [],
                           "error": f"fatal {type(e).__name__}: {e}"}
                results.append(res)
                _log_res(res, done, len(tasks))


    pairs_path = args.out / "aligned_qa_pairs.jsonl"
    msgs_path = args.out / "aligned_qa_messages.jsonl"
    n_qa = 0
    with pairs_path.open("w", encoding="utf-8") as fp, msgs_path.open("w", encoding="utf-8") as fm:
        for res in sorted(results, key=lambda r: r["sample"]):
            for rec in res.get("qas", []):
                fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fm.write(json.dumps(to_message_record(rec), ensure_ascii=False) + "\n")
                n_qa += 1

    report = {
        "input": str(args.input), "out": str(args.out),
        "n_samples": len(samples), "n_processed": len(results),
        "n_ok": sum(1 for r in results if r.get("ok")),
        "n_rotating": sum(1 for r in results if r.get("rotating")),
        "n_entered_from_behind": sum(r.get("n_entered", 0) for r in results),
        "n_exited": sum(r.get("src_stats", {}).get("n_exited", 0) for r in results),
        "n_topdown": sum(1 for r in results if r.get("topdown")),
        "sources_per_sample_avg": round(
            sum(r.get("n_sources", 0) for r in results if r.get("ok"))
            / max(1, sum(1 for r in results if r.get("ok"))), 2),
        "sources_per_sample": {r["sample"]: r.get("n_sources", 0)
                               for r in results if r.get("ok")},
        "n_qa": n_qa,
        "pairs_jsonl": str(pairs_path), "messages_jsonl": str(msgs_path),
        "failures": [{"sample": r["sample"], "error": r.get("error")}
                     for r in results if not r.get("ok")],
    }
    (args.out / "aligned_qa_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("=" * 60)
    logger.info("Done: %d/%d samples succeeded, %d QA pairs total", report["n_ok"], len(results), n_qa)
    logger.info("  Rich metadata -> %s", pairs_path)
    logger.info("  messages -> %s", msgs_path)
    if report["failures"]:
        logger.warning("  %d samples failed; see aligned_qa_report.json", len(report["failures"]))


def _log_res(res: dict, i: int, total: int):
    if res.get("ok"):
        mode = res.get("cam_mode", "rotate" if res.get("rotating") else "full")
        yaw = res.get("yaw_end") if mode == "rotate" else res.get("yaw_fixed", 0.0)
        logger.info("(%d/%d) [%s] OK mode=%s yaw=%s n_qa=%d src=%s entered=%s exited=%s topdown=%s",
                    i, total, res["sample"], mode, yaw,
                    res.get("n_qa"), res.get("n_sources"), res.get("n_entered"),
                    res.get("src_stats", {}).get("n_exited"),
                    "y" if res.get("topdown") else "-")
    else:
        logger.error("(%d/%d) [%s] FAIL: %s", i, total, res["sample"], res.get("error"))


if __name__ == "__main__":
    main()
