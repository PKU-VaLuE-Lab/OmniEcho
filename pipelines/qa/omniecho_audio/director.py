"""OmniEcho processing component."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from omniecho_audio.api_client import (
    call_llm_with_system,
    APIPermanentError,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Output container
# ---------------------------------------------------------------------------

@dataclass
class DirectorOutput:
    """OmniEcho processing component."""

    scene: dict                       
    video_prompt: str                 
    sub_prompts: list[str]            
    raw_response: str = field(default="", repr=False)  


# ---------------------------------------------------------------------------
# 2. LLM endpoint configuration (Gemini via annotation service proxy)
# ---------------------------------------------------------------------------


class _LLMPermanentError(RuntimeError):
    """OmniEcho processing component."""


# ---------------------------------------------------------------------------
# 3. System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a Spatial Audio-Video Scene Director. Design a creative spatial-audio
scene and write a matching video-generation script. Write all descriptions,
labels, prompts, and explanations in English.

===========================================================================
Input
===========================================================================
Design a scene containing:
- N_FOREGROUND foreground sound sources, such as a dripping tap, a cat, or
  footsteps. Set role="foreground".
- One diffuse ambient/background source, such as rain, wind, or traffic.
  Set role="ambient".

Every object must have a role field with value "foreground" or "ambient".
Foreground sources have snr_db > 0. The ambient source has snr_db = 0 and is
the reference. Exactly one object must have role="ambient".
The user specifies the source count and scene duration; otherwise design freely.

===========================================================================
Coordinates and spatial layout
===========================================================================
- Use a right-handed coordinate system with the listener facing +y.
- azimuth is in [-180, 180] degrees: 0=front, +90=left, -90=right.
  Positive azimuth is the listener's left; negative azimuth is their right.
- elevation is in [-90, 90] degrees; positive is upward.
- distance is in [0.5, 15] meters.
- The listener stays at the origin: position=[0, 0, 0].
- Each source trajectory has at least two time points: start and end.
- Speed between any adjacent trajectory points must be <10 m/s.
- Initial azimuths of different sources should be at least 30 degrees apart.
- All distances must be >=0.5 m; the absolute minimum is 0.3 m.

===========================================================================
Camera and listener orientation
===========================================================================
- The camera is attached to the listener at listener.position.
- Horizontal camera FOV is approximately 90 degrees (half-FOV: 45 degrees).
- listener.orientation.yaw is the camera's azimuth in the same coordinates.
- Time-varying yaw is represented by a trajectory, for example:
  [{"t": 0, "yaw": 0}, {"t": 3, "yaw": 30}, {"t": 6, "yaw": 30}].
- Visibility follows:
  camera_relative_azimuth(t) = object_azimuth(t) - listener_yaw(t).
  Normalize this to [-180, 180]. The source is visible when its magnitude
  is <45 degrees.
- Each object must include:
  "visibility_window": {"video_enter_t": <float>, "video_exit_t": <float>}.
  This visible interval must agree with the geometry above.
- Always visible: video_enter_t=0.0, video_exit_t=duration.
- Never visible: video_enter_t=video_exit_t=-1.0.
- A source may be audible throughout, but appears on screen only within its
  visibility_window.

===========================================================================
Visibility diversity (recommended, not mandatory)
===========================================================================
Avoid repeatedly making every source either always visible [0.0, duration]
or always off-screen [-1.0, -1.0]. Include dynamic visibility often enough to
exercise OFF_SCREEN_DYNAMIC handling.
- In multisource scenes, preferably give at least one source an intermediate
  visibility window:
  * Visible first, then exits while continuing to sound:
    video_enter_t=0.0, video_exit_t=3.5.
  * Initially off-screen, then enters:
    video_enter_t=2.0, video_exit_t=6.0.
  * Crosses the frame briefly:
    video_enter_t=1.0, video_exit_t=3.0.
- Trajectories must describe the actual entry/exit direction:
  * Exiting right at t=3.5 s: move from an in-frame azimuth such as -20
    toward -45, then beyond the edge to -60 or -80 degrees.
  * Entering left at t=2.0 s: move from +60 toward +45, then to +20 degrees.
- When the camera also rotates, the boundary-crossing times of azimuth-yaw
  must match video_enter_t and video_exit_t.
- Sources continue sounding outside their visible windows. Their trajectories
  must cover the full [onset_t, offset_t] interval, continue in the exit
  direction off-screen, and remain spatially continuous.
- Not every source must be dynamic; avoid repeatedly using only the extremes.

===========================================================================
Physical consistency of camera rotation
===========================================================================
Use the variant_6_physics_correct object-anchor convention. The original
experiment reported 100% direction accuracy for this template. Inconsistent
anchors can make the video model reverse movement or show almost no movement.

Direction mapping:
- Yaw increasing from 0 to a positive value (e.g. +55): camera turns left.
- Yaw decreasing from 0 to a negative value (e.g. -55): camera turns right.
- Constant yaw: the camera remains completely fixed; do not describe rotation
  or translation.

Required object-anchor behavior:
- Camera turns left: the centered object slides out to the right; a new object
  on the left enters from the left edge and reaches the center.
- Camera turns right: the centered object slides out to the left; a new object
  on the right enters from the right edge and reaches the center.

Never combine:
- A left camera turn with an object entering the center from the right.
- A left camera turn with the central object exiting to the left.
- A right camera turn with an object entering the center from the left.
- A right camera turn with the central object exiting to the right.

===========================================================================
Scene styles
===========================================================================
Alternate among these styles as permitted by the user's camera constraint:
1. Fixed camera: constant yaw, with sources moving into or out of view.
2. Rotating camera: gradually changing yaw makes sources enter or leave view.
   Follow the physical-consistency rules above.
3. Entirely in-frame: every source stays within |azimuth|<45 degrees, suitable
   for close scenes with multiple nearby sources.

===========================================================================
Output format
===========================================================================
Return only one JSON object, without markdown fences or explanatory text:
{
  "scene": {
    "scene_id": "experiment_001",
    "duration": 6.0,
    "environment": {
      "type": "indoor",
      "room": {"dimensions": [5, 6, 3], "rt60": 0.5},
      "description": "<one-sentence English scene description>"
    },
    "listener": {
      "position": [0, 0, 0],
      "orientation": {
        "yaw": 0, "pitch": 0, "roll": 0,
        "trajectory": [{"t": 0, "yaw": 0}, {"t": 6, "yaw": 0}]
      }
    },
    "objects": [
      {
        "id": "source_001",
        "type": "<English source type>",
        "label": "<English label>",
        "role": "foreground",
        "loudness_lufs": -18,
        "snr_db": 15,
        "onset_t": 0.0,
        "offset_t": 6.0,
        "trajectory": [
          {"t": 0, "azimuth": -60, "elevation": 0, "distance": 3.0},
          {"t": 6, "azimuth": -20, "elevation": 0, "distance": 2.5}
        ],
        "visibility_window": {"video_enter_t": 2.5, "video_exit_t": 6.0},
        "wav_path": ""
      }
    ]
  },
  "sub_prompts": [
    "<close-up prompt for foreground source 1>",
    "<close-up prompt for foreground source 2>",
    "<close-up prompt for foreground source 3>",
    "<ambient source video prompt>"
  ],
  "video_prompt": "<video prompt in the format below>"
}
environment.type may be "indoor" or "outdoor"; room.dimensions is [width,
length, height], and room.rt60 is typically 0.3-0.8.
The example illustrates the schema; include every requested source and one
ambient source, with matching objects/sub_prompts lengths.

===========================================================================
Loudness
===========================================================================
- loudness_lufs is target integrated loudness:
  * Nearby foreground (<2 m): -14 to -18 LUFS.
  * Mid-distance foreground (2-5 m): -20 to -26 LUFS.
  * Distant background/ambient (>5 m): -30 to -40 LUFS.
  * Doubling distance reduces level by approximately 6 dB.
- snr_db is relative to the quietest source, which is the 0 dB reference:
  * Main foreground: +12 to +20 dB.
  * Secondary foreground: +6 to +12 dB.
  * Ambient background: 0 dB.
- Every object must include loudness_lufs, snr_db, and role.
- role="ambient" iff snr_db=0; role="foreground" iff snr_db>0.
- Do not output a gain field.

===========================================================================
Active sound intervals
===========================================================================
Every source must specify:
- onset_t: sound onset in seconds, >=0 and <duration.
- offset_t: sound offset in seconds, >onset_t and <=duration.
At least one foreground source starts at onset_t=0, so the scene does not
start silently. Foreground events should typically last 2-5 seconds.
Intervals may overlap and need not be synchronized.

Ambient/continuous sounds must cover the entire scene:
onset_t=0.0, offset_t=duration. Examples include water flow, rain, wind,
HVAC, traffic background, and insects. These continuous sounds must not
start or stop abruptly. Only foreground events such as knocking, footsteps,
or page turning may use onset_t>0 or offset_t<duration.

===========================================================================
Per-source video prompts
===========================================================================
Write one sub_prompt per source to generate a close-up containing only that
source's sound. Preserve the environment's acoustics while excluding every
other noise source.

Foreground template:
"In a quiet {environment}, a close-up of {sounding object} performing
{sound-producing action}. Strict audio requirement: {source} is the only
sound-producing object. There is no wind, HVAC hum, crowd, mechanical
background noise, electrical noise, or other additional noise source.
Preserve the natural acoustics of the {environment}: reverberation, echoes,
and a sense of space. The listener should hear {source} alone in a quiet
{environment}, sensing its size and materials without any other sound.
720p, realistic style, no music, narration, subtitles, or watermark."

Each sub-video lasts offset_t-onset_t. The source sounds from the beginning
to the end of the sub-video. Do not put onset times, duration numbers, or
specific second counts in sub_prompts.

Stationary sources:
- For a clock, HVAC unit, tap, speaker, or other nontranslating object,
  explicitly state: "The object stays stationary without any displacement.
  The camera is completely fixed."
- Moving sources such as walkers or birds may have a described trajectory.
- Treat azimuth variation <5 degrees as stationary for this prompt rule.
- A stationary source can contain moving parts, such as a clock pendulum.

Describe the visible cause of sound precisely:
- Instead of "a person sits down", describe a seated person rocking slightly,
  causing wooden chair joints to creak.
- Instead of "turning a book", describe a hand slowly turning a page whose
  paper bends and springs back with a faint sound.
- Instead of "footsteps", describe a person in leather shoes walking across
  a wooden floor, with each step clearly audible.
- Friction: show two surfaces rubbing.
- Impact: show the moment of contact.
- Flowing water: show water moving.
- Animal vocalization: show the mouth, throat, or other sound-producing action.

Framing:
- Center the sounding object and let it occupy at least 20% of the frame.
- Only the sounding object moves; everything else remains still, except
  when the sound source itself is ambient.
- Exclude unrelated moving people, birds, curtains, and other noise sources.
- End every sub_prompt with:
  "No other moving or sounding object may appear besides {sounding object}.
  Do not show unrelated walking, door opening, or window closing."

Action examples to refine:
- Creaking chair: a person adjusts their posture in an old wooden chair.
- Page turning: a hand slowly turns one page right-to-left in an open book.
- Clock: a pendulum swings regularly while gears tick.
- Footsteps: legs and feet visibly walk in {shoes} on {floor material}.
- Dripping tap: a droplet forms, elongates, and falls into the sink.
- Typing: fingers rise and fall quickly on a keyboard.
- Birdsong: a bird on a branch opens its beak with visible throat vibration.
- Cat: a seated cat faces the camera and opens its mouth to meow.
- Wind chime: hanging chimes gently sway and collide in a breeze.
- Knocking: close-up of knuckles striking a wooden door.

Ambient template:
"A wide view of {environment}, with {environmental sound-producing activity}.
Strict audio requirement: only {ambient sound}, without unrelated artificial
noise such as HVAC or machinery. Fully preserve the environment's natural
acoustic space. 720p, realistic style, no music, narration, subtitles, or
watermark." Do not include foreground sources.

===========================================================================
Absolute agreement between sub-prompt positions and audio directions
===========================================================================
- Positive azimuth: describe the source on the left, moving from left toward
  center or from center toward left as appropriate.
- Negative azimuth: describe the source on the right, moving from right toward
  center or from center toward right as appropriate.
- Explicitly describe a moving source's entry side and motion direction.
- azimuth >+15: source on the left side or arriving from the left.
- azimuth in [-15,+15]: source at the center.
- azimuth <-15: source on the right side or arriving from the right.
- Diffuse/textural sources such as streams, wind, and rain still need explicit
  image positions, e.g. "the stream flows through the lower left" or
  "rain drips from the roof at the upper right".
- Corresponding azimuth and elevation must match those descriptions.
  Never describe a stream on the left while assigning negative azimuth.
- A walker moving left-to-center should change azimuth from +60 to +10.
  A walker moving right-to-center should change it from -60 to -10.
- sub_prompts order must match scene.objects exactly.

===========================================================================
Main video prompt format (Variant C)
===========================================================================
Use 720p, 16:9, and 6 seconds unless the user specifies a different duration.
Use realistic style. No music, narration, subtitles, watermarks, or transitions.
Show each source only within its visibility_window.

Part 1: realistic natural-language scene description.
Describe visible objects and an action timeline including entry/exit times.
Example style: "A quiet kitchen. At 2.0 seconds, a gray cat gently enters
from the right edge..."

Part 2: camera description.
State first-person viewpoint, 720p, 16:9, and the scene duration. Distinguish a
fixed camera position from horizontal rotation.
- Constant yaw: "First-person view, camera completely stationary, with no
  horizontal rotation." Do not use head-turn, pan, or translation language.
- Changing yaw: use this physically consistent template:
  "First-person view, camera position completely fixed, rotating only
  horizontally and slowly {left|right} by {N} degrees. As the camera turns
  {left|right}, the initially centered {object} slides toward the
  {right|left} edge and exits. Meanwhile, {another object} on the
  {left|right} enters from that edge and moves toward the center."
- N is the absolute maximum yaw change in listener.orientation.trajectory.
- Positive yaw change means left; negative yaw change means right.
- Object labels must refer to actual visible scene.objects.
- Match all anchors to the physical camera-rotation rules above.
- Additional stationary references such as lamps, posts, boats, or streetlights
  can provide useful visual anchors.

Part 3: leave a blank line and append:
---
METADATA:
listener.position = [x,y,z]
listener.yaw_trajectory = [(t=0, yaw=X), ...]
source_001 [static|moving] type: trajectory=[...], visibility=[enter_t, exit_t]
source_002 [static|moving] type: trajectory=[...], visibility=[enter_t, exit_t]
...
camera FOV approximately 90 degrees

Include a motion_type annotation per source:
static means no displacement, with azimuth variation <5 degrees;
moving means translational movement.
Examples:
source_001 [static] clock: trajectory=[...], visibility=[0.0, 6.0]
source_002 [moving] footsteps: trajectory=[...], visibility=[4.5, 6.0]

===========================================================================
Precise object-position descriptions
===========================================================================
To align video with spatial metadata (target azimuth error <5 degrees),
describe each visible object's position with grid region, angles, and times.
- Left third: azimuth [+15,+45], from the left edge to the left-third line.
- Center third: azimuth [-15,+15].
- Right third: azimuth [-45,-15], from the right-third line to the right edge.
- Off-screen: |azimuth-listener_yaw|>45 degrees.
- Upper: elevation [+15,+45], such as sky, roofs, and high flying objects.
- Middle: elevation [-5,+15], near eye level or table height.
- Lower: elevation [-45,-5], such as the ground, feet, or low animals.

Time precision:
- Do not say "around the third second", "about the fourth second", or
  "a few seconds later".
- Say "at 3.6 seconds" or "from 2.0 to 4.5 seconds".
- Use 0.1-second precision.

Describe at least three moments for every moving object:
"Seagull:
- t=0.0s: [left third, upper], az=+30, el=+25, d=4.0m, circling slowly.
- t=3.0s: [center, upper], az=+5, el=+30, d=3.5m, gliding right.
- t=6.0s: [right third, upper], az=-25, el=+22, d=4.2m, moving away."

Visibility:
- Entering: "At X.X seconds, enters from the [direction] edge."
- Exiting: "At X.X seconds, exits through the [direction] edge."
- Always visible: "Visible throughout the full 6.0 seconds."
- When several objects are visible together, describe their relative layout,
  e.g. a seagull in the upper right third and a pedestrian in the lower left
  third, separated horizontally by about half the frame.

===========================================================================
Scene diversity
===========================================================================
Every scene needs a distinct environment and source combination. Vary:
- Indoor/outdoor.
- Urban/natural/domestic/industrial or workplace.
- Day/night/dusk.
- Quiet/noisy.
Alternate animals, human activities, machinery, natural phenomena, and
musical instruments as appropriate. Avoid consecutive similar themes,
such as repeated seaside or boardwalk scenes.

===========================================================================
Final requirements
===========================================================================
- Return one JSON object only, without fences or explanation.
- Use English for all generated text.
- No music, narration, subtitles, or watermarks in video_prompt.
- visibility_window must agree with azimuth/yaw geometry.
- duration must exactly match the requested scene_duration (default 6.0).
- sub_prompts and objects must have equal lengths.
- For rotation, explicitly fix the camera position, specify the horizontal
  turn in degrees, and use physically consistent object anchors.
- Describe visible positions with grid, angle, and time; replace vague "left"
  with a precise description such as "left third, az=+30 degrees".
"""


# ---------------------------------------------------------------------------
# 4. User prompt builder
# ---------------------------------------------------------------------------

def _build_user_prompt(
    n_foreground: int,
    include_ambient: bool,
    scene_duration: float,
    allow_camera_rotation: bool,
    previous_scenes: list[str] | None = None,
) -> str:
    lines: list[str] = []
    n_total = n_foreground + (1 if include_ambient else 0)
    lines.append(f"Design a spatial-audio scene lasting {scene_duration:.1f} seconds.")
    lines.append(f"- Include {n_foreground} foreground sound sources")
    if include_ambient:
        lines.append("- Include one ambient/background sound source")
    lines.append("")
    lines.append("Constraints:")
    lines.append(f"- scene_duration = {scene_duration:.1f} seconds")
    if allow_camera_rotation:
        lines.append("- listener.orientation.yaw may change over time; camera rotation is encouraged")
    else:
        lines.append("- listener.orientation.yaw must remain constant (fixed camera)")
    lines.append("- Initial source azimuths must be at least 30 degrees apart")
    lines.append("- Include visibility_window for every object")
    lines.append("- Include sub_prompts in one-to-one correspondence with objects")
    if previous_scenes:
        lines.append("")
        lines.append("These scenes were generated previously. Choose a different environment and source combination:")
        for s in previous_scenes:
            lines.append(f"  - {s}")
    lines.append("")
    lines.append("Return only a JSON object matching the system prompt's schema, with all text in English.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. JSON extraction
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """OmniEcho processing component."""
    if not text:
        raise ValueError("Empty LLM response")


    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass


    fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", stripped, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass


    first = stripped.find("{")
    last = stripped.rfind("}")
    if first != -1 and last > first:
        candidate = stripped[first : last + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON object: {e}\nCandidate head: {candidate[:300]}"
            ) from e

    raise ValueError(f"No JSON object found in response. Head: {stripped[:300]}")


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

def _normalize_az(deg: float) -> float:
    """OmniEcho processing component."""
    a = (deg + 180.0) % 360.0 - 180.0
    return a


def _interp_yaw(yaw_traj: list[dict], t: float) -> float:
    """OmniEcho processing component."""
    if not yaw_traj:
        return 0.0
    pts = sorted(yaw_traj, key=lambda p: p.get("t", 0.0))
    if t <= pts[0].get("t", 0.0):
        return float(pts[0].get("yaw", 0.0))
    if t >= pts[-1].get("t", 0.0):
        return float(pts[-1].get("yaw", 0.0))
    for i in range(len(pts) - 1):
        t1 = float(pts[i].get("t", 0.0))
        t2 = float(pts[i + 1].get("t", 0.0))
        if t1 <= t <= t2:
            if t2 - t1 < 1e-9:
                return float(pts[i].get("yaw", 0.0))
            r = (t - t1) / (t2 - t1)
            y1 = float(pts[i].get("yaw", 0.0))
            y2 = float(pts[i + 1].get("yaw", 0.0))
            return y1 + r * (y2 - y1)
    return float(pts[-1].get("yaw", 0.0))


def _interp_az(traj: list[dict], t: float) -> float:
    if not traj:
        return 0.0
    pts = sorted(traj, key=lambda p: p.get("t", 0.0))
    if t <= pts[0].get("t", 0.0):
        return float(pts[0].get("azimuth", 0.0))
    if t >= pts[-1].get("t", 0.0):
        return float(pts[-1].get("azimuth", 0.0))
    for i in range(len(pts) - 1):
        t1 = float(pts[i].get("t", 0.0))
        t2 = float(pts[i + 1].get("t", 0.0))
        if t1 <= t <= t2:
            if t2 - t1 < 1e-9:
                return float(pts[i].get("azimuth", 0.0))
            r = (t - t1) / (t2 - t1)
            a1 = float(pts[i].get("azimuth", 0.0))
            a2 = float(pts[i + 1].get("azimuth", 0.0))
            return a1 + r * (a2 - a1)
    return float(pts[-1].get("azimuth", 0.0))


def validate_scene(scene: dict, scene_duration: float = 6.0) -> tuple[list[str], list[str]]:
    """OmniEcho processing component."""
    errors: list[str] = []
    warnings: list[str] = []

    # duration
    duration = scene.get("duration", scene.get("duration_sec"))
    if duration is None:
        errors.append("scene.duration missing")
    else:
        try:
            duration = float(duration)
        except (TypeError, ValueError):
            errors.append(f"scene.duration not numeric: {duration!r}")
            duration = scene_duration
        if abs(duration - scene_duration) > 1e-3:
            errors.append(
                f"scene.duration ({duration}) != expected ({scene_duration})"
            )

    duration = float(duration) if duration is not None else scene_duration

    # listener
    listener = scene.get("listener")
    if not isinstance(listener, dict):
        errors.append("scene.listener missing or not dict")
        listener = {}
    orientation = listener.get("orientation", {}) if isinstance(listener, dict) else {}
    yaw_traj = orientation.get("trajectory") if isinstance(orientation, dict) else None
    if yaw_traj is None:
        
        single_yaw = float(orientation.get("yaw", 0.0)) if isinstance(orientation, dict) else 0.0
        yaw_traj = [{"t": 0.0, "yaw": single_yaw}, {"t": duration, "yaw": single_yaw}]

    # objects
    objects = scene.get("objects")
    if not isinstance(objects, list) or not objects:
        errors.append("scene.objects missing or empty")
        return errors, warnings

    n_ambient = 0
    for i, obj in enumerate(objects):
        ctx = f"objects[{i}]"


        role = obj.get("role")
        if role is None:
            errors.append(f"{ctx}: role field missing (must be 'foreground' or 'ambient')")
        elif role not in ("foreground", "ambient"):
            errors.append(
                f"{ctx}: role={role!r} invalid (must be 'foreground' or 'ambient')"
            )
        elif role == "ambient":
            n_ambient += 1


        lufs = obj.get("loudness_lufs")
        if lufs is not None:
            try:
                lufs_val = float(lufs)
                if lufs_val < -50 or lufs_val > 0:
                    errors.append(f"{ctx}: loudness_lufs={lufs_val} out of [-50, 0]")
            except (TypeError, ValueError):
                errors.append(f"{ctx}: loudness_lufs not numeric: {lufs!r}")

        snr = obj.get("snr_db")
        if snr is not None:
            try:
                snr_val = float(snr)
                if snr_val < 0:
                    errors.append(f"{ctx}: snr_db={snr_val} must be >= 0")
            except (TypeError, ValueError):
                errors.append(f"{ctx}: snr_db not numeric: {snr!r}")

        traj = obj.get("trajectory")
        if not isinstance(traj, list) or len(traj) < 2:
            errors.append(f"{ctx}: trajectory must have >= 2 points")
            continue


        for j, p in enumerate(traj):
            t = p.get("t")
            if t is None:
                errors.append(f"{ctx}.trajectory[{j}]: missing t")
                continue
            if t < -1e-3 or t > duration + 1e-3:
                errors.append(f"{ctx}.trajectory[{j}]: t={t} out of [0,{duration}]")

            az = float(p.get("azimuth", 0.0))
            if az < -180.0 - 1e-3 or az > 180.0 + 1e-3:
                errors.append(f"{ctx}.trajectory[{j}]: azimuth {az} out of [-180,180]")

            el = float(p.get("elevation", 0.0))
            if el < -90.0 - 1e-3 or el > 90.0 + 1e-3:
                errors.append(f"{ctx}.trajectory[{j}]: elevation {el} out of [-90,90]")

            dist = float(p.get("distance", 0.0))
            if dist < 0.3:
                errors.append(
                    f"{ctx}.trajectory[{j}]: distance {dist:.2f}m < 0.3m minimum"
                )


        for j in range(len(traj) - 1):
            p1, p2 = traj[j], traj[j + 1]
            dt = float(p2.get("t", 0.0)) - float(p1.get("t", 0.0))
            if dt <= 0:
                errors.append(f"{ctx}.trajectory[{j}->{j+1}]: time not increasing")
                continue
            az1 = math.radians(float(p1.get("azimuth", 0.0)))
            el1 = math.radians(float(p1.get("elevation", 0.0)))
            d1 = float(p1.get("distance", 1.0))
            az2 = math.radians(float(p2.get("azimuth", 0.0)))
            el2 = math.radians(float(p2.get("elevation", 0.0)))
            d2 = float(p2.get("distance", 1.0))
            x1 = d1 * math.cos(el1) * math.sin(az1)
            y1 = d1 * math.cos(el1) * math.cos(az1)
            z1 = d1 * math.sin(el1)
            x2 = d2 * math.cos(el2) * math.sin(az2)
            y2 = d2 * math.cos(el2) * math.cos(az2)
            z2 = d2 * math.sin(el2)
            disp = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
            speed = disp / dt
            if speed > 10.0:
                errors.append(
                    f"{ctx}.trajectory[{j}->{j+1}]: speed {speed:.2f} m/s > 10 m/s"
                )


        vis = obj.get("visibility_window")
        if not isinstance(vis, dict):
            errors.append(f"{ctx}: visibility_window missing")
            continue
        v_in = vis.get("video_enter_t")
        v_out = vis.get("video_exit_t")
        if v_in is None or v_out is None:
            errors.append(f"{ctx}: visibility_window missing video_enter_t/exit_t")
            continue
        try:
            v_in = float(v_in)
            v_out = float(v_out)
        except (TypeError, ValueError):
            errors.append(f"{ctx}: visibility_window not numeric")
            continue


        if v_in >= 0 and v_out > v_in:
            samples = 11
            ok_count = 0
            for k in range(samples):
                tt = v_in + (v_out - v_in) * (k / max(samples - 1, 1))
                az_t = _interp_az(traj, tt)
                yaw_t = _interp_yaw(yaw_traj, tt)
                rel = _normalize_az(az_t - yaw_t)
                if abs(rel) <= 47.0:  
                    ok_count += 1
            if ok_count < int(samples * 0.6):
                errors.append(
                    f"{ctx}: visibility_window [{v_in},{v_out}] inconsistent with FOV "
                    f"({ok_count}/{samples} samples in-frame)"
                )


    
    has_any_role = any(
        isinstance(o, dict) and o.get("role") is not None for o in objects
    )
    if has_any_role and n_ambient != 1:
        errors.append(
            f"objects must contain exactly 1 ambient role (found {n_ambient})"
        )


    env = scene.get("environment")
    if not isinstance(env, dict):
        errors.append("scene.environment missing")

    return errors, warnings


# ---------------------------------------------------------------------------
# 6b. Spatial trajectory markup (appended to video_prompt)
# ---------------------------------------------------------------------------

def _build_spatial_trajectory_markup(scene: dict) -> str:
    """OmniEcho processing component."""
    lines: list[str] = []
    lines.append("")
    lines.append("---")
    lines.append("SPATIAL LAYOUT REFERENCE:")
    duration = scene.get("duration", scene.get("duration_sec", 6.0))
    lines.append(f"Duration: {float(duration):.1f}s | FOV: 90° horizontal")

    # Listener/camera trajectory
    listener = scene.get("listener", {})
    orientation = listener.get("orientation", {})
    yaw_traj = orientation.get("trajectory", [])
    if yaw_traj:
        yaw_str = ", ".join(
            f"t={float(p.get('t', 0)):.1f}s→yaw={float(p.get('yaw', 0)):.0f}°"
            for p in yaw_traj
        )
        lines.append(f"Camera: {yaw_str}")

    lines.append("")

    # Object trajectories
    for obj in scene.get("objects", []):
        obj_type = obj.get("type", "?")
        label = obj.get("label", obj_type)
        traj = obj.get("trajectory", [])
        vis = obj.get("visibility_window", {})

        lines.append(f"[{label}] ({obj_type}):")

        # Visibility
        enter_t = vis.get("video_enter_t")
        exit_t = vis.get("video_exit_t")
        if enter_t is not None and exit_t is not None:
            enter_t_f = float(enter_t)
            exit_t_f = float(exit_t)
            if enter_t_f >= 0 and exit_t_f > enter_t_f:
                lines.append(f"  Visible: {enter_t_f:.1f}s ~ {exit_t_f:.1f}s")
            elif enter_t_f < 0:
                lines.append("  Visible: NEVER (offscreen entire duration)")

        # Trajectory points with grid position
        for pt in traj:
            t = float(pt.get("t", 0))
            az = float(pt.get("azimuth", 0))
            el = float(pt.get("elevation", 0))
            d = float(pt.get("distance", 0))

            # Compute grid position (relative to camera at this time)
            # For markup, use absolute azimuth (camera-relative computed by model)
            if abs(az) > 45:
                h_grid = "OFFSCREEN"
            elif az > 15:
                h_grid = "LEFT-1/3"
            elif az >= -15:
                h_grid = "CENTER"
            else:
                h_grid = "RIGHT-1/3"

            if el > 15:
                v_grid = "UPPER"
            elif el >= -5:
                v_grid = "MIDDLE"
            else:
                v_grid = "LOWER"

            grid = f"{h_grid},{v_grid}" if h_grid != "OFFSCREEN" else "OFFSCREEN"
            lines.append(
                f"  t={t:.1f}s: [{grid}] az={az:+.0f}° el={el:+.0f}° d={d:.1f}m"
            )

        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6c. Direction consistency validation & auto-fix
# ---------------------------------------------------------------------------
#


#


_DIR_SWAP_PAIRS: list[tuple[str, str]] = [
    ("left third", "right third"),
    ("left-third", "right-third"),
    ("LEFT-1/3", "RIGHT-1/3"),
    ("frame left", "frame right"),
    ("from the left", "from the right"),
    ("toward the left", "toward the right"),
    ("left side", "right side"),
    ("Left", "Right"),
    ("LEFT", "RIGHT"),
    ("left", "right"),
]
_LEFT_KEYWORDS: tuple[str, ...] = tuple(p[0] for p in _DIR_SWAP_PAIRS)
_RIGHT_KEYWORDS: tuple[str, ...] = tuple(p[1] for p in _DIR_SWAP_PAIRS)


def _has_left_word(text: str) -> bool:
    return any(re.search(r"(?<!\w)" + re.escape(kw) + r"(?!\w)", text, re.I) for kw in _LEFT_KEYWORDS)


def _has_right_word(text: str) -> bool:
    return any(re.search(r"(?<!\w)" + re.escape(kw) + r"(?!\w)", text, re.I) for kw in _RIGHT_KEYWORDS)


def _swap_left_right(text: str) -> str:
    """OmniEcho processing component."""
    if not text:
        return text
    out = text
    placeholders: list[tuple[str, str]] = []
    for i, (l, r) in enumerate(_DIR_SWAP_PAIRS):
        ph_l = f"\x00LPH{i}\x00"
        ph_r = f"\x00RPH{i}\x00"
        out = re.sub(r"(?<!\w)" + re.escape(l) + r"(?!\w)", lambda _: ph_l, out)
        out = re.sub(r"(?<!\w)" + re.escape(r) + r"(?!\w)", lambda _: ph_r, out)
        placeholders.append((ph_l, r))
        placeholders.append((ph_r, l))
    for ph, repl in placeholders:
        out = out.replace(ph, repl)
    return out


def _representative_az(
    traj: list[dict], vis_window: dict | None = None
) -> tuple[float, float, float]:
    """OmniEcho processing component."""
    if not traj:
        return 0.0, 0.0, 0.0
    pts = sorted(traj, key=lambda p: float(p.get("t", 0.0)))
    azs = [float(p.get("azimuth", 0.0)) for p in pts]
    start_az = azs[0]
    span = max(azs) - min(azs)

    mean_az = sum(azs) / len(azs)
    if isinstance(vis_window, dict):
        v_in = vis_window.get("video_enter_t")
        v_out = vis_window.get("video_exit_t")
        try:
            v_in_f = float(v_in) if v_in is not None else -1.0
            v_out_f = float(v_out) if v_out is not None else -1.0
        except (TypeError, ValueError):
            v_in_f = v_out_f = -1.0
        if v_in_f >= 0 and v_out_f > v_in_f:
            samples = 11
            ss = [
                _interp_az(pts, v_in_f + (v_out_f - v_in_f) * (k / (samples - 1)))
                for k in range(samples)
            ]
            mean_az = sum(ss) / len(ss)
    return mean_az, start_az, span


def _expected_direction_label(
    traj: list[dict], vis_window: dict | None = None
) -> tuple[str, float]:
    """OmniEcho processing component."""
    mean_az, start_az, span = _representative_az(traj, vis_window)
    rep_az = start_az if span > 30.0 else mean_az
    if rep_az > 15.0:
        return "left", rep_az
    if rep_az < -15.0:
        return "right", rep_az
    return "center", rep_az


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?\n])(?=\s|$)")


def _simple_swap_direction_consistency(
    scene_dict: dict,
    output_dir: str | os.PathLike | None = None,
) -> tuple[dict, list[dict]]:
    """OmniEcho processing component."""
    fixes: list[dict] = []
    objects = scene_dict.get("objects")
    if not isinstance(objects, list) or not objects:
        return scene_dict, fixes

    sub_prompts = scene_dict.get("sub_prompts")
    if not isinstance(sub_prompts, list):
        sub_prompts = []

    video_prompt = scene_dict.get("video_prompt") or ""
    if not isinstance(video_prompt, str):
        video_prompt = str(video_prompt)

    for i, obj in enumerate(objects):
        if not isinstance(obj, dict):
            continue
        traj = obj.get("trajectory")
        if not isinstance(traj, list) or len(traj) < 1:
            continue
        vis = obj.get("visibility_window")
        vis_dict = vis if isinstance(vis, dict) else None
        expected_dir, rep_az = _expected_direction_label(traj, vis_dict)
        if expected_dir == "center":
            continue

        source_id = obj.get("id") or f"source_{i + 1:03d}"
        source_label = obj.get("label") or obj.get("type") or source_id
        source_type = obj.get("type") or ""
        
        keywords: list[str] = []
        for k in (source_label, source_type):
            if isinstance(k, str) and k.strip() and k not in keywords:
                keywords.append(k.strip())

        wrong_is_right = expected_dir == "left"


        if i < len(sub_prompts) and isinstance(sub_prompts[i], str):
            sp = sub_prompts[i]
            has_left = _has_left_word(sp)
            has_right = _has_right_word(sp)
            contradiction = (
                (wrong_is_right and has_right and not has_left)
                or ((not wrong_is_right) and has_left and not has_right)
            )
            if contradiction:
                fixed_sp = _swap_left_right(sp)
                if fixed_sp != sp:
                    fixes.append(
                        {
                            "source": source_id,
                            "source_name": source_label,
                            "field": f"sub_prompts[{i}]",
                            "azimuth_mean": round(rep_az, 1),
                            "expected_direction": expected_dir,
                            "original_text": sp,
                            "fixed_text": fixed_sp,
                        }
                    )
                    sub_prompts[i] = fixed_sp


        if video_prompt and keywords:
            sentences = _SENT_SPLIT_RE.split(video_prompt)
            changed_any = False
            new_sentences: list[str] = []
            for sent in sentences:
                if not sent or not sent.strip():
                    new_sentences.append(sent)
                    continue
                if not any(kw and kw in sent for kw in keywords):
                    new_sentences.append(sent)
                    continue
                has_left = _has_left_word(sent)
                has_right = _has_right_word(sent)
                contradiction = (
                    (wrong_is_right and has_right and not has_left)
                    or ((not wrong_is_right) and has_left and not has_right)
                )
                if not contradiction:
                    new_sentences.append(sent)
                    continue
                fixed_sent = _swap_left_right(sent)
                if fixed_sent == sent:
                    new_sentences.append(sent)
                    continue
                fixes.append(
                    {
                        "source": source_id,
                        "source_name": source_label,
                        "field": "video_prompt",
                        "azimuth_mean": round(rep_az, 1),
                        "expected_direction": expected_dir,
                        "original_text": sent,
                        "fixed_text": fixed_sent,
                    }
                )
                new_sentences.append(fixed_sent)
                changed_any = True
            if changed_any:
                video_prompt = "".join(new_sentences)

    scene_dict["sub_prompts"] = sub_prompts
    scene_dict["video_prompt"] = video_prompt

    if fixes and output_dir is not None:
        try:
            from pathlib import Path

            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            report_path = out_path / "direction_fix_report.json"
            report_path.write_text(
                json.dumps(
                    {"fixes": fixes, "total_fixes": len(fixes)},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            logger.info(
                "Direction fix report saved: %s (%d fixes)",
                report_path, len(fixes),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Failed to write direction_fix_report.json to %s: %s",
                output_dir, e,
            )

    return scene_dict, fixes


# ---------------------------------------------------------------------------
# 6d. LLM-based spatial consistency review (replaces simple swap)
# ---------------------------------------------------------------------------
#


_LEFT_PATTERNS: list[str] = [
    "left third", "left-third", "left-1/3", "frame left", "from the left", "toward the left",
    "left side", "left", "leftward", "left half",
]
_RIGHT_PATTERNS: list[str] = [
    "right third", "right-third", "right-1/3", "frame right", "from the right", "toward the right",
    "right side", "right", "rightward", "right half",
]
_UP_PATTERNS: list[str] = [
    "above", "high up", "overhead", "sky", "roof", "upper",
    "high altitude", "top", "aloft", "airborne", "flying",
]
_DOWN_PATTERNS: list[str] = [
    "below", "ground", "underfoot", "bottom", "lower",
    "low down", "underneath", "floor", "crouching", "prone", "crawling",
]
_FAR_PATTERNS: list[str] = [
    "far", "distant", "far away", "far end", "horizon",
]
_NEAR_PATTERNS: list[str] = [
    "near", "nearby", "close", "beside", "next to", "close by",
    "adjacent", "within reach",
]


def _detect_text_direction(text: str) -> dict[str, str | None]:
    """OmniEcho processing component."""
    result: dict[str, str | None] = {
        "horizontal": None, "vertical": None, "distance": None,
    }
    has_left = any(re.search(r"(?<!\w)" + re.escape(p) + r"(?!\w)", text, re.I) for p in _LEFT_PATTERNS)
    has_right = any(re.search(r"(?<!\w)" + re.escape(p) + r"(?!\w)", text, re.I) for p in _RIGHT_PATTERNS)
    
    if has_left and not has_right:
        result["horizontal"] = "left"
    elif has_right and not has_left:
        result["horizontal"] = "right"

    has_up = any(re.search(r"(?<!\w)" + re.escape(p) + r"(?!\w)", text, re.I) for p in _UP_PATTERNS)
    has_down = any(re.search(r"(?<!\w)" + re.escape(p) + r"(?!\w)", text, re.I) for p in _DOWN_PATTERNS)
    if has_up and not has_down:
        result["vertical"] = "up"
    elif has_down and not has_up:
        result["vertical"] = "down"

    has_far = any(re.search(r"(?<!\w)" + re.escape(p) + r"(?!\w)", text, re.I) for p in _FAR_PATTERNS)
    has_near = any(re.search(r"(?<!\w)" + re.escape(p) + r"(?!\w)", text, re.I) for p in _NEAR_PATTERNS)
    if has_far and not has_near:
        result["distance"] = "far"
    elif has_near and not has_far:
        result["distance"] = "near"

    return result


def _detect_contradictions(scene_dict: dict) -> list[dict]:
    """OmniEcho processing component."""
    contradictions: list[dict] = []
    objects = scene_dict.get("objects")
    if not isinstance(objects, list):
        return contradictions

    sub_prompts = scene_dict.get("sub_prompts", [])
    video_prompt = scene_dict.get("video_prompt", "")


    _metadata_sep = "\n---\nMETADATA:"
    if _metadata_sep in video_prompt:
        video_prompt = video_prompt[:video_prompt.index(_metadata_sep)]
    else:
        _layout_sep = "\n---\nSPATIAL LAYOUT REFERENCE"
        if _layout_sep in video_prompt:
            video_prompt = video_prompt[:video_prompt.index(_layout_sep)]

    for i, obj in enumerate(objects):
        if not isinstance(obj, dict):
            continue
        traj = obj.get("trajectory")
        if not isinstance(traj, list) or len(traj) < 1:
            continue


        azs = [float(p.get("azimuth", 0.0)) for p in traj]
        els = [float(p.get("elevation", 0.0)) for p in traj]
        dists = [float(p.get("distance", 1.0)) for p in traj]
        mean_az = sum(azs) / len(azs)
        mean_el = sum(els) / len(els)
        mean_dist = sum(dists) / len(dists)

        source_id = obj.get("id", f"source_{i+1:03d}")
        source_label = obj.get("label", obj.get("type", source_id))


        texts_to_check: list[tuple[str, str]] = []  # (field_name, text)
        if i < len(sub_prompts) and isinstance(sub_prompts[i], str) and sub_prompts[i].strip():
            texts_to_check.append((f"sub_prompts[{i}]", sub_prompts[i]))


        keywords: list[str] = []
        for k in (source_label, obj.get("type", "")):
            if isinstance(k, str) and k.strip():
                full = k.strip()
                if full not in keywords:
                    keywords.append(full)
                
                if "_" in full:
                    for part in full.split("_"):
                        if len(part) >= 2 and part not in keywords:
                            keywords.append(part)
                
                for word in re.findall(r"[A-Za-z]+", full):
                    if len(word) >= 3 and word not in keywords:
                        keywords.append(word)


        if video_prompt and keywords:
            paragraphs = video_prompt.split("\n\n")
            for para in paragraphs:
                para_stripped = para.strip()
                if para_stripped and any(kw in para_stripped for kw in keywords):
                    texts_to_check.append(("video_prompt", para_stripped))


        for field_name, text in texts_to_check:
            dirs = _detect_text_direction(text)


            if abs(mean_az) > 15.0 and dirs["horizontal"]:
                value_side = "left" if mean_az > 0 else "right"
                if dirs["horizontal"] != value_side:
                    contradictions.append({
                        "source_id": source_id,
                        "source_label": source_label,
                        "field": field_name,
                        "dimension": "horizontal (azimuth)",
                        "text_says": f"{dirs['horizontal']} side",
                        "value_says": f"{value_side} side (azimuth={mean_az:.1f} degrees)",
                        "text_excerpt": text[:200],
                    })


            if abs(mean_el) > 10.0 and dirs["vertical"]:
                value_vert = "up" if mean_el > 0 else "down"
                if dirs["vertical"] != value_vert:
                    contradictions.append({
                        "source_id": source_id,
                        "source_label": source_label,
                        "field": field_name,
                        "dimension": "vertical (elevation)",
                        "text_says": f"{dirs['vertical']}",
                        "value_says": f"{value_vert} (elevation={mean_el:.1f} degrees)",
                        "text_excerpt": text[:200],
                    })


            if dirs["distance"]:
                if dirs["distance"] == "near" and mean_dist > 5.0:
                    contradictions.append({
                        "source_id": source_id,
                        "source_label": source_label,
                        "field": field_name,
                        "dimension": "distance",
                        "text_says": "near (expected d<1.5m)",
                        "value_says": f"far (distance={mean_dist:.1f}m, >5.0m)",
                        "text_excerpt": text[:200],
                    })
                elif dirs["distance"] == "far" and mean_dist < 1.5:
                    contradictions.append({
                        "source_id": source_id,
                        "source_label": source_label,
                        "field": field_name,
                        "dimension": "distance",
                        "text_says": "far (expected d>5.0m)",
                        "value_says": f"near (distance={mean_dist:.1f}m, <1.5m)",
                        "text_excerpt": text[:200],
                    })

    return contradictions


_SPATIAL_REVIEW_SYSTEM_PROMPT = """\
You review spatial-audio scenes and correct contradictions between text descriptions and spatial coordinates.

## Coordinate conventions (mandatory)
- azimuth: positive=left, negative=right, zero=front. A clear direction requires |az|>15 degrees.
- elevation: positive=up/high, negative=down/ground, zero=eye level. A clear vertical direction requires |el|>10 degrees.
- distance is in meters: <1.5=near, >5=far, 1.5-5=mid-distance.

## Detection rules
- Horizontal contradiction: text says left but azimuth<-15 degrees (right).
- Vertical contradiction: text says up/high but elevation<-10 degrees (down).
- Distance contradiction: text says near but distance>5 m (far).

## Output format
Return the complete corrected JSON, preserving the input scene schema. Write all text in English, without markdown fences or explanations.
"""


def _call_llm_for_fix(
    scene_dict: dict,
    contradictions: list[dict],
    extra_context: str = "",
) -> dict | None:
    """OmniEcho processing component."""
    contradictions_desc = "\n".join(
        f"- [{c['source_label']}] ({c['source_id']}) in {c['field']}: "
        f"text says '{c['text_says']}', but coordinates indicate '{c['value_says']}'"
        f"\n  Relevant text: {c['text_excerpt'][:150]}"
        for c in contradictions
    )


    scene_for_llm = json.loads(json.dumps(scene_dict))
    for obj in scene_for_llm.get("objects", []):
        obj.pop("_audio_stats", None)
        obj.pop("wav_path", None)
        obj.pop("gain_db", None)
        obj.pop("start_time", None)
        obj.pop("end_time", None)

    user_prompt = f"""\
{extra_context}

## Detected contradictions
{contradictions_desc}

## Original scene data
{json.dumps(scene_for_llm, ensure_ascii=False, indent=2)}

## Task
Correct the contradictions so text and coordinates agree. You may:
1. Edit directional descriptions in video_prompt / sub_prompts.
2. Edit trajectory azimuth / elevation / distance.
Choose the correction that makes the entire scene most plausible.

Mandatory coordinate conventions:
- azimuth: positive=left, negative=right, zero=front. A clear direction requires |az|>15 degrees.
- elevation: positive=up/high, negative=down/ground, zero=eye level. A clear vertical direction requires |el|>10 degrees.
- distance is in meters: <1.5=near, >5=far, 1.5-5=mid-distance.

Return the complete corrected JSON in the same schema, with English text and no markdown fences."""

    try:
        raw_response = _call_llm(
            _SPATIAL_REVIEW_SYSTEM_PROMPT,
            user_prompt,
            timeout=180.0,
            max_http_retries=3,
        )
    except Exception as e:
        logger.warning("LLM spatial review call failed: %s", e)
        return None

    try:
        fixed_data = _extract_json(raw_response)
    except (ValueError, json.JSONDecodeError) as e:
        logger.warning("LLM spatial review JSON parsing failed: %s", e)
        return None


    required_keys = {"objects", "video_prompt", "sub_prompts"}
    if not required_keys.issubset(set(fixed_data.keys())):
        if "scene" in fixed_data:
            inner_scene = fixed_data["scene"]
            fixed_data = {
                "objects": inner_scene.get("objects", scene_dict.get("objects", [])),
                "video_prompt": fixed_data.get("video_prompt", scene_dict.get("video_prompt", "")),
                "sub_prompts": fixed_data.get("sub_prompts", scene_dict.get("sub_prompts", [])),
            }
            for k, v in inner_scene.items():
                if k not in fixed_data:
                    fixed_data[k] = v
        else:
            logger.warning(
                "LLM spatial review response is missing required field %s",
                required_keys - set(fixed_data.keys()),
            )
            return None


    original_objects = scene_dict.get("objects", [])
    fixed_objects = fixed_data.get("objects", [])
    for idx, fobj in enumerate(fixed_objects):
        if idx < len(original_objects):
            orig = original_objects[idx]
            for internal_key in ("_audio_stats", "wav_path", "gain_db", "start_time", "end_time"):
                if internal_key in orig and internal_key not in fobj:
                    fobj[internal_key] = orig[internal_key]


    result = json.loads(json.dumps(scene_dict))  # deep copy
    result["objects"] = fixed_objects
    result["video_prompt"] = fixed_data.get("video_prompt", scene_dict.get("video_prompt", ""))
    result["sub_prompts"] = fixed_data.get("sub_prompts", scene_dict.get("sub_prompts", []))
    return result


def _llm_review_spatial_consistency(
    scene_dict: dict,
    output_dir: str | os.PathLike | None = None,
    max_rounds: int = 3,
) -> tuple[dict, dict]:
    """OmniEcho processing component."""
    report: dict = {"rounds": [], "final_status": "unknown", "total_rounds": 0}
    current_scene = json.loads(json.dumps(scene_dict))  # deep copy

    for round_num in range(1, max_rounds + 1):
        contradictions = _detect_contradictions(current_scene)

        if not contradictions:
            report["rounds"].append({
                "round": round_num,
                "contradictions": [],
                "status": "passed",
            })
            report["final_status"] = "passed"
            report["total_rounds"] = round_num
            logger.info(
                "LLM spatial review: round %d passed with no contradictions.", round_num,
            )
            break

        logger.info(
            "LLM spatial review: round %d found %d contradictions; submitting for LLM review...",
            round_num, len(contradictions),
        )


        if round_num > 1:
            extra_context = (
                f"This is correction round {round_num}. The previous correction left the following "
                f"contradictions. Review them carefully and resolve them fully without repeating the same errors."
            )
        else:
            extra_context = ""

        fixed = _call_llm_for_fix(current_scene, contradictions, extra_context)

        if fixed is None:
            
            report["rounds"].append({
                "round": round_num,
                "contradictions": contradictions,
                "status": "llm_failed",
            })
            current_scene, _ = _simple_swap_direction_consistency(current_scene)
            report["final_status"] = "fallback_to_simple_swap"
            report["total_rounds"] = round_num
            logger.warning(
                "LLM spatial review: round %d call failed; falling back to simple direction swapping.",
                round_num,
            )
            break

        report["rounds"].append({
            "round": round_num,
            "contradictions": contradictions,
            "status": "llm_fixed",
        })
        current_scene = fixed
    else:
        
        report["final_status"] = "max_rounds_exceeded"
        report["total_rounds"] = max_rounds
        
        current_scene, _ = _simple_swap_direction_consistency(current_scene)
        logger.warning(
            "LLM spatial review: contradictions remain after %d rounds; falling back to simple direction swapping.",
            max_rounds,
        )


    if output_dir is not None:
        try:
            from pathlib import Path
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            report_path = out_path / "spatial_review_report.json"
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("Spatial review report saved: %s", report_path)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to save spatial_review_report.json: %s", e)

    return current_scene, report


# ---------------------------------------------------------------------------
# 7. LLM call
# ---------------------------------------------------------------------------


def _call_llm(
    system_prompt: str,
    user_prompt: str,
    timeout: float = 300.0,
    max_http_retries: int = 5,
) -> str:
    """OmniEcho processing component."""
    try:
        return call_llm_with_system(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=65536,
            timeout=timeout,
            max_retries=max_http_retries,
        )
    except APIPermanentError as e:
        raise _LLMPermanentError(str(e)) from e


# ---------------------------------------------------------------------------
# 8. Public API
# ---------------------------------------------------------------------------

def generate_scene_and_prompts(
    n_foreground: int = 3,
    include_ambient: bool = True,
    scene_duration: float = 6.0,
    allow_camera_rotation: bool = True,
    max_retries: int = 5,
    previous_scenes: list[str] | None = None,
    output_dir: str | os.PathLike | None = None,
) -> DirectorOutput:
    """OmniEcho processing component."""
    n_total = n_foreground + (1 if include_ambient else 0)
    if n_foreground < 1:
        raise ValueError("n_foreground must be >= 1")

    user_prompt = _build_user_prompt(
        n_foreground,
        include_ambient,
        scene_duration,
        allow_camera_rotation,
        previous_scenes,
    )

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            raw = _call_llm(_SYSTEM_PROMPT, user_prompt)
            parsed = _extract_json(raw)

            scene = parsed.get("scene")
            if scene is None:
                raise ValueError(
                    f"LLM output missing 'scene'. Keys: {list(parsed.keys())}"
                )
            if not isinstance(scene, dict):
                raise ValueError("LLM output: scene is not an object")


            video_prompt = None
            for key in ("video_prompt", "prompt", "prompt_c"):
                if key in parsed:
                    video_prompt = str(parsed[key]).strip()
                    break
            if not video_prompt:
                raise ValueError(
                    f"LLM output missing 'video_prompt'. "
                    f"Available keys: {list(parsed.keys())}"
                )


            sub_prompts = parsed.get("sub_prompts")
            if sub_prompts is None:
                raise ValueError(
                    f"LLM output missing 'sub_prompts'. "
                    f"Available keys: {list(parsed.keys())}"
                )
            if not isinstance(sub_prompts, list):
                raise ValueError("LLM output: sub_prompts is not an array")


            n_objects = len(scene.get("objects", []))
            if len(sub_prompts) != n_objects:
                raise ValueError(
                    f"sub_prompts length ({len(sub_prompts)}) != objects length ({n_objects}); "
                    f"a one-to-one correspondence is required"
                )


            _scene_dur_default = scene.get(
                "duration", scene.get("duration_sec", scene_duration)
            )
            try:
                _scene_dur_default = float(_scene_dur_default)
            except (TypeError, ValueError):
                _scene_dur_default = float(scene_duration)
            for obj in scene.get("objects", []):
                if "onset_t" not in obj:
                    obj["onset_t"] = 0.0
                if "offset_t" not in obj:
                    obj["offset_t"] = _scene_dur_default


            for idx, sp in enumerate(sub_prompts):
                if not isinstance(sp, str) or not sp.strip():
                    raise ValueError(
                        f"sub_prompts[{idx}] is empty or not a string: {sp!r}"
                    )
            sub_prompts = [sp.strip() for sp in sub_prompts]

            errors, warnings = validate_scene(scene, scene_duration=scene_duration)
            if warnings:
                logger.warning(
                    "Scene validation warnings (non-blocking):\n%s",
                    "\n".join(f"  - {w}" for w in warnings),
                )
            if errors:
                raise ValueError(
                    "Scene validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
                )


            
            _wrapper = {
                "objects": scene.get("objects", []),
                "sub_prompts": sub_prompts,
                "video_prompt": video_prompt,
            }
            _wrapper, _review_report = _llm_review_spatial_consistency(
                _wrapper, output_dir=output_dir,
            )
            
            scene["objects"] = _wrapper.get("objects", scene.get("objects", []))
            sub_prompts = _wrapper["sub_prompts"]
            video_prompt = _wrapper["video_prompt"]
            if _review_report.get("total_rounds", 0) > 0:
                _has_contradictions = any(
                    r.get("contradictions") for r in _review_report.get("rounds", [])
                )
                if _has_contradictions:
                    logger.info(
                        "Director: LLM spatial-consistency review done "
                        "(status=%s, rounds=%d)",
                        _review_report.get("final_status", "unknown"),
                        _review_report.get("total_rounds", 0),
                    )


            spatial_markup = _build_spatial_trajectory_markup(scene)
            if spatial_markup.strip():
                video_prompt = video_prompt.rstrip() + "\n" + spatial_markup

            logger.info(
                "Director: scene generated successfully on attempt %d/%d (objects=%d)",
                attempt + 1, max_retries, n_objects,
            )
            return DirectorOutput(
                scene=scene,
                video_prompt=video_prompt,
                sub_prompts=sub_prompts,
                raw_response=raw,
            )

        except _LLMPermanentError:


            logger.error(
                "Director: permanent LLM failure; aborting immediately without fallback."
            )
            raise
        except Exception as e:  
            last_err = e
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)  # 2, 4, 8, 16, 32
                logger.warning(
                    "Director attempt %d/%d failed: %s — retrying in %ds",
                    attempt + 1, max_retries, e, wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "Director failed after %d attempts: %s", max_retries, e,
                )

    raise RuntimeError(
        f"Director failed after {max_retries} attempts. Last error: {last_err}"
    )


# ---------------------------------------------------------------------------
# 9. CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    out = generate_scene_and_prompts(
        n_foreground=3,
        include_ambient=True,
        scene_duration=6.0,
        allow_camera_rotation=True,
    )
    print(json.dumps(out.scene, indent=2, ensure_ascii=False))
    print("\n--- video_prompt ---\n", out.video_prompt)
    print("\n--- sub_prompts ---")
    for i, sp in enumerate(out.sub_prompts):
        print(f"  [{i}] {sp}")
