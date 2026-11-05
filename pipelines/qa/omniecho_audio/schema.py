from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
import json


@dataclass(frozen=True)
class TrajectoryPoint:
    time: float
    azimuth: float
    elevation: float
    distance: float
    gain_db: float


@dataclass(frozen=True)
class AudioObject:
    object_id: str
    wav_path: Path
    trajectory: tuple[TrajectoryPoint, ...]
    start_time: float
    end_time: float | None
    gain_db: float
    spread_deg: float


@dataclass(frozen=True)
class Environment:
    sample_rate: int
    duration: float
    normalization: str
    ambisonics_order: int
    coordinate_system: str
    room: dict[str, Any]
    listener: dict[str, Any]


@dataclass(frozen=True)
class Scene:
    scene_id: str
    environment: Environment
    objects: tuple[AudioObject, ...]


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required field {context}.{key}")
    return mapping[key]


def _load_trajectory(points: list[dict[str, Any]], context: str) -> tuple[TrajectoryPoint, ...]:
    if not points:
        raise ValueError(f"{context}.trajectory must contain at least one point")
    loaded: list[TrajectoryPoint] = []
    previous_time = -1.0
    for index, point in enumerate(points):
        point_context = f"{context}.trajectory[{index}]"
        time = float(_require(point, "t", point_context))
        if time < 0:
            raise ValueError(f"{point_context}.t must be non-negative")
        if time < previous_time:
            raise ValueError(f"{context}.trajectory times must be sorted")
        previous_time = time
        distance = float(point.get("distance", point.get("dist", 1.0)))
        if distance <= 0:
            raise ValueError(f"{point_context}.distance must be positive")
        loaded.append(
            TrajectoryPoint(
                time=time,
                azimuth=float(_require(point, "azimuth", point_context) if "azimuth" in point else _require(point, "az", point_context)),
                elevation=float(point.get("elevation", point.get("el", 0.0))),
                distance=distance,
                gain_db=float(point.get("gain_db", 0.0)),
            )
        )
    return tuple(loaded)


def load_scene(path: str | Path) -> Scene:
    scene_path = Path(path)
    data = json.loads(scene_path.read_text())
    env_data = _require(data, "environment", "scene")
    audio_data = env_data.get("audio", {})
    environment = Environment(
        sample_rate=int(audio_data.get("sample_rate", 48000)),
        duration=float(_require(data, "duration_sec", "scene")),
        normalization=str(audio_data.get("normalization", "N3D_ACN")),
        ambisonics_order=int(audio_data.get("ambisonics_order", 1)),
        coordinate_system=str(data.get("coordinate_system", "front-left-up_degrees")),
        room=dict(env_data.get("room", {})),
        listener=dict(data.get("listener", {})),
    )
    if environment.duration <= 0:
        raise ValueError("scene.duration_sec must be positive")
    if environment.sample_rate <= 0:
        raise ValueError("environment.audio.sample_rate must be positive")
    if environment.normalization != "N3D_ACN":
        raise ValueError("Only N3D_ACN Ambisonics is supported in this implementation")
    if environment.ambisonics_order != 1:
        raise ValueError("This pipeline is FOA-only: environment.audio.ambisonics_order must be 1")

    objects = []
    for index, obj in enumerate(_require(data, "objects", "scene")):
        context = f"scene.objects[{index}]"
        objects.append(
            AudioObject(
                object_id=str(_require(obj, "id", context)),
                wav_path=(scene_path.parent / str(_require(obj, "wav_path", context))).resolve(),
                trajectory=_load_trajectory(list(_require(obj, "trajectory", context)), context),
                start_time=float(obj.get("start_time", 0.0)),
                end_time=float(obj["end_time"]) if obj.get("end_time") is not None else None,
                gain_db=float(obj.get("gain_db", 0.0)),
                spread_deg=float(obj.get("spread_deg", 0.0)),
            )
        )
    if not objects:
        raise ValueError("scene.objects must contain at least one object")
    return Scene(str(data.get("scene_id", scene_path.stem)), environment, tuple(objects))


class CameraMode(str, Enum):
    STATIC = "static"          
    PAN = "pan"                
    ROTATE = "rotate"          


class VisibilityState(str, Enum):
    ON_SCREEN = "on_screen"                    
    EXITING = "exiting"                        
    OFF_SCREEN_DYNAMIC = "off_screen_dynamic"  
    OFF_SCREEN_ALWAYS = "off_screen_always"    
    ENTERING = "entering"                      
    OCCLUDED = "occluded"                      


@dataclass(frozen=True)
class AnnotatedTrajectoryPoint:
    """OmniEcho processing component."""
    time: float
    azimuth: float        
    elevation: float      
    distance: float       
    gain_db: float        
    visibility_state: VisibilityState  
    confidence: float = 1.0            


class OffScreenStrategy(str, Enum):
    EXTRAPOLATE = "extrapolate"          
    STATIC_POSITION = "static_position"  
    TRUST_METADATA = "trust_metadata"    


@dataclass(frozen=True)
class AnnotatedAudioObject:
    """OmniEcho processing component."""
    object_id: str
    label: str
    trajectory: tuple[AnnotatedTrajectoryPoint, ...]
    acoustic_start: float    
    acoustic_end: float
    visual_enter: float      
    visual_exit: float       
    off_screen_strategy: OffScreenStrategy  
    spread_deg: float = 10.0  
    gain_db: float = 0.0  


@dataclass(frozen=True)
class AnnotatedScene:
    """OmniEcho processing component."""
    camera_mode: CameraMode
    objects: tuple[AnnotatedAudioObject, ...]
    duration: float            
    sample_rate: int = 48000
    fov_horizontal_deg: float = 90.0
