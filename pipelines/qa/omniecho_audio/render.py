from __future__ import annotations

from pathlib import Path
import copy
import json
import math
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import soundfile as sf
from scipy import signal
import scipy.special as _scipy_special

# Compatibility shim: scipy>=1.17 removed `sph_harm` in favour of `sph_harm_y`.

# importing spaudiopy. Old API: sph_harm(m, n, theta=azimuth, phi=polar).
# New API: sph_harm_y(n, m, theta=polar, phi=azimuth).
if not hasattr(_scipy_special, "sph_harm"):
    _sph_harm_y = _scipy_special.sph_harm_y

    def _sph_harm_compat(m, n, theta, phi):
        return _sph_harm_y(n, m, phi, theta)

    _scipy_special.sph_harm = _sph_harm_compat

from spaudiopy import decoder, io as spa_io, sph

from .adm import build_adm_xml
from .schema import AudioObject, Scene, load_scene


def db_to_linear(db: float | np.ndarray) -> float | np.ndarray:
    return np.power(10.0, np.asarray(db, dtype=np.float64) / 20.0)


def az_el_to_azi_zen(azimuth_deg: np.ndarray, elevation_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.deg2rad(azimuth_deg), np.deg2rad(90.0 - elevation_deg)


def az_el_dist_to_xyz(azimuth_deg: float, elevation_deg: float, distance: float, origin: np.ndarray) -> np.ndarray:
    azimuth = np.deg2rad(azimuth_deg)
    elevation = np.deg2rad(elevation_deg)
    return origin + distance * np.array([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ], dtype=np.float64)


def az_el_dist_arrays_to_xyz(
    azimuth_deg: np.ndarray,
    elevation_deg: np.ndarray,
    distance: np.ndarray,
    origin: np.ndarray,
) -> np.ndarray:
    azimuth = np.deg2rad(azimuth_deg)
    elevation = np.deg2rad(elevation_deg)
    xyz = np.column_stack([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ])
    return origin[np.newaxis, :] + distance[:, np.newaxis] * xyz


def wrap_degrees(angle: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(angle) + 180.0) % 360.0 - 180.0


def xyz_to_az_el_dist(position: np.ndarray, origin: np.ndarray) -> tuple[float, float, float]:
    vector = np.asarray(position, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    distance = float(np.linalg.norm(vector))
    if distance <= 0:
        return 0.0, 0.0, 1e-6
    azimuth = float(np.rad2deg(np.arctan2(vector[1], vector[0])))
    elevation = float(np.rad2deg(np.arcsin(np.clip(vector[2] / distance, -1.0, 1.0))))
    return azimuth, elevation, distance


def xyz_arrays_to_az_el_dist(positions: np.ndarray, origin: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors = np.asarray(positions, dtype=np.float64) - origin[np.newaxis, :]
    distances = np.linalg.norm(vectors, axis=1)
    safe_distances = np.maximum(distances, 1e-6)
    azimuth = np.rad2deg(np.arctan2(vectors[:, 1], vectors[:, 0]))
    elevation = np.rad2deg(np.arcsin(np.clip(vectors[:, 2] / safe_distances, -1.0, 1.0)))
    return azimuth, elevation, safe_distances


def read_mono_audio(path: Path, sample_rate: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Object audio file does not exist: {path}")
    audio, file_sample_rate = sf.read(path, always_2d=True)
    audio = audio.astype(np.float64)
    mono = np.mean(audio, axis=1)
    if file_sample_rate != sample_rate:
        gcd = math.gcd(int(sample_rate), int(file_sample_rate))
        mono = signal.resample_poly(mono, sample_rate // gcd, file_sample_rate // gcd)
    return mono


def fit_to_scene(audio: np.ndarray, source: AudioObject, total_samples: int, sample_rate: int) -> np.ndarray:
    start = int(round(source.start_time * sample_rate))
    if start >= total_samples:
        raise ValueError(f"Object {source.object_id} starts outside scene duration")
    if source.end_time is None:
        available = total_samples - start
    else:
        end = int(round(source.end_time * sample_rate))
        if end <= start:
            raise ValueError(f"Object {source.object_id} end_time must be greater than start_time")
        available = min(total_samples - start, end - start)
    fitted = np.zeros(total_samples, dtype=np.float64)
    copy_count = min(available, len(audio))
    fitted[start : start + copy_count] = audio[:copy_count]
    return fitted


def interpolate_object_parameters(source: AudioObject, total_samples: int, sample_rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    times = np.arange(total_samples, dtype=np.float64) / sample_rate
    point_times = np.array([point.time for point in source.trajectory], dtype=np.float64) + source.start_time
    azimuths = np.unwrap(np.deg2rad([point.azimuth for point in source.trajectory]))
    elevations = np.array([point.elevation for point in source.trajectory], dtype=np.float64)
    distances = np.array([point.distance for point in source.trajectory], dtype=np.float64)
    gains_db = np.array([point.gain_db for point in source.trajectory], dtype=np.float64)
    azimuth_interp = np.rad2deg(np.interp(times, point_times, azimuths, left=azimuths[0], right=azimuths[-1]))
    elevation_interp = np.interp(times, point_times, elevations, left=elevations[0], right=elevations[-1])
    distance_interp = np.interp(times, point_times, distances, left=distances[0], right=distances[-1])
    gain_interp = np.interp(times, point_times, gains_db, left=gains_db[0], right=gains_db[-1])
    return azimuth_interp, elevation_interp, distance_interp, gain_interp


def interpolate_listener_yaw(scene: Scene, total_samples: int, sample_rate: int) -> np.ndarray:
    listener = scene.environment.listener
    orientation = listener.get("orientation", {}) if isinstance(listener, dict) else {}
    trajectory = orientation.get("trajectory") if isinstance(orientation, dict) else None
    times = np.arange(total_samples, dtype=np.float64) / sample_rate
    if trajectory:
        point_times = np.array([float(point.get("t", point.get("time", 0.0))) for point in trajectory], dtype=np.float64)
        yaws = np.unwrap(np.deg2rad([float(point.get("yaw", 0.0)) for point in trajectory]))
        return np.rad2deg(np.interp(times, point_times, yaws, left=yaws[0], right=yaws[-1]))
    yaw = float(orientation.get("yaw", listener.get("yaw", 0.0)) if isinstance(orientation, dict) else 0.0)
    return np.full(total_samples, yaw, dtype=np.float64)


def world_to_camera_relative_azimuth(world_azimuth: np.ndarray, listener_yaw: np.ndarray) -> np.ndarray:
    return np.asarray(wrap_degrees(world_azimuth - listener_yaw), dtype=np.float64)


def encode_signal_to_ambisonics(
    signal_in: np.ndarray,
    azimuth: np.ndarray,
    elevation: np.ndarray,
    distance: np.ndarray,
    gain: float | np.ndarray,
    order: int,
) -> np.ndarray:
    distance = np.maximum(distance, 1e-6)
    signal_with_gain = signal_in * np.asarray(gain, dtype=np.float64) / distance
    azi, zen = az_el_to_azi_zen(azimuth, elevation)
    sh_matrix = sph.sh_matrix(order, azi, zen, sh_type="real")
    return (sh_matrix.T * signal_with_gain[np.newaxis, :]).astype(np.float64)


def encode_object_to_ambisonics(source_signal: np.ndarray, source: AudioObject, order: int, sample_rate: int, listener_yaw: np.ndarray | None = None) -> np.ndarray:
    total_samples = len(source_signal)
    azimuth, elevation, distance, point_gain_db = interpolate_object_parameters(source, total_samples, sample_rate)
    if listener_yaw is not None:
        azimuth = world_to_camera_relative_azimuth(azimuth, listener_yaw)
    gain = db_to_linear(source.gain_db) * db_to_linear(point_gain_db) / np.maximum(distance, 1e-6)
    azi, zen = az_el_to_azi_zen(azimuth, elevation)
    sh_matrix = sph.sh_matrix(order, azi, zen, sh_type="real")
    return (sh_matrix.T * (source_signal * gain)[np.newaxis, :]).astype(np.float64)


def get_listener_position(scene: Scene) -> np.ndarray:
    position = scene.environment.listener.get("position", [0.0, 0.0, 0.0])
    return np.array(position, dtype=np.float64)


def render_room_mono(source_signal: np.ndarray, source: AudioObject, scene: Scene) -> np.ndarray:
    import pyroomacoustics as pra

    room_config = scene.environment.room
    if str(room_config.get("type", "direct")).lower() in {"direct", "anechoic", "anechoic_or_external_renderer"}:
        return source_signal
    dimensions = np.array(room_config.get("dimensions", [6.0, 5.0, 3.0]), dtype=np.float64)
    if dimensions.shape != (3,) or np.any(dimensions <= 0):
        raise ValueError("environment.room.dimensions must contain three positive values")
    listener = get_listener_position(scene)
    if np.allclose(listener, 0.0):
        listener = dimensions / 2.0
    if np.any(listener <= 0) or np.any(listener >= dimensions):
        raise ValueError("listener.position must be inside room dimensions for room rendering")
    first_point = source.trajectory[0]
    source_position = az_el_dist_to_xyz(first_point.azimuth, first_point.elevation, first_point.distance, listener)
    source_position = np.minimum(np.maximum(source_position, 0.05), dimensions - 0.05)
    rt60 = float(room_config.get("rt60", 0.3))
    max_order = int(room_config.get("max_order", 3))
    if rt60 > 0:
        absorption, sabine_order = pra.inverse_sabine(rt60, dimensions)
        max_order = max(max_order, int(sabine_order))
    else:
        absorption = 1.0
        max_order = 0
    room = pra.ShoeBox(dimensions, fs=scene.environment.sample_rate, materials=pra.Material(absorption), max_order=max_order)
    room.add_source(source_position, signal=source_signal)
    room.add_microphone_array(pra.MicrophoneArray(listener.reshape(3, 1), fs=scene.environment.sample_rate))
    room.simulate()
    rendered = np.asarray(room.mic_array.signals[0], dtype=np.float64)
    if len(rendered) < len(source_signal):
        rendered = np.pad(rendered, (0, len(source_signal) - len(rendered)))
    return rendered[: len(source_signal)]


def room_parameters(scene: Scene) -> tuple[np.ndarray, np.ndarray, float, int]:
    room_config = scene.environment.room
    dimensions = np.array(room_config.get("dimensions", [6.0, 5.0, 3.0]), dtype=np.float64)
    if dimensions.shape != (3,) or np.any(dimensions <= 0):
        raise ValueError("environment.room.dimensions must contain three positive values")
    listener = get_listener_position(scene)
    if np.allclose(listener, 0.0):
        listener = dimensions / 2.0
    if np.any(listener <= 0) or np.any(listener >= dimensions):
        raise ValueError("listener.position must be inside room dimensions for room rendering")
    rt60 = float(room_config.get("rt60", 0.3))
    max_order = int(room_config.get("max_order", 1))
    return dimensions, listener, rt60, max_order


def reflection_absorption(scene: Scene, dimensions: np.ndarray, rt60: float) -> float:
    import pyroomacoustics as pra

    if rt60 <= 0:
        return 1.0
    try:
        absorption, _ = pra.inverse_sabine(rt60, dimensions)
    except ValueError:
        volume = float(np.prod(dimensions))
        surface = float(2.0 * (dimensions[0] * dimensions[1] + dimensions[0] * dimensions[2] + dimensions[1] * dimensions[2]))
        absorption = 0.161 * volume / max(surface * rt60, 1e-6)
    return float(np.clip(absorption, 0.0, 1.0))


def delayed_scaled(signal_in: np.ndarray, delay_samples: int, scale: float) -> np.ndarray:
    output = np.zeros_like(signal_in)
    if delay_samples >= len(signal_in):
        return output
    output[delay_samples:] = signal_in[: len(signal_in) - delay_samples] * scale
    return output


def variable_delay(signal_in: np.ndarray, delay_samples: np.ndarray, scale: float | np.ndarray) -> np.ndarray:
    indices = np.arange(len(signal_in), dtype=np.int64) - np.rint(delay_samples).astype(np.int64)
    valid = (indices >= 0) & (indices < len(signal_in))
    output = np.zeros_like(signal_in)
    output[valid] = signal_in[indices[valid]] * np.asarray(scale, dtype=np.float64)[valid]
    return output


def first_order_image_sources(source_position: np.ndarray, dimensions: np.ndarray) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    for axis in range(3):
        lower = source_position.copy()
        lower[axis] = -source_position[axis]
        images.append(lower)
        upper = source_position.copy()
        upper[axis] = 2.0 * dimensions[axis] - source_position[axis]
        images.append(upper)
    return images


def first_order_image_source_paths(source_positions: np.ndarray, dimensions: np.ndarray) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    for axis in range(3):
        lower = source_positions.copy()
        lower[:, axis] = -source_positions[:, axis]
        images.append(lower)
        upper = source_positions.copy()
        upper[:, axis] = 2.0 * dimensions[axis] - source_positions[:, axis]
        images.append(upper)
    return images


def render_room_early_foa(source_signal: np.ndarray, source: AudioObject, scene: Scene) -> np.ndarray:
    dimensions, listener, rt60, _ = room_parameters(scene)
    order = scene.environment.ambisonics_order
    sample_rate = scene.environment.sample_rate
    total_samples = len(source_signal)
    listener_yaw = interpolate_listener_yaw(scene, total_samples, sample_rate)
    azimuth, elevation, distance, point_gain_db = interpolate_object_parameters(source, total_samples, sample_rate)
    source_positions = az_el_dist_arrays_to_xyz(azimuth, elevation, distance, listener)
    source_positions = np.minimum(np.maximum(source_positions, 0.05), dimensions[np.newaxis, :] - 0.05)
    direct_azimuth, direct_elevation, direct_distance = xyz_arrays_to_az_el_dist(source_positions, listener)
    direct_azimuth = world_to_camera_relative_azimuth(direct_azimuth, listener_yaw)
    source_gain = db_to_linear(source.gain_db) * db_to_linear(point_gain_db)
    direct = encode_signal_to_ambisonics(
        source_signal,
        direct_azimuth,
        direct_elevation,
        direct_distance,
        source_gain,
        order,
    )
    absorption = reflection_absorption(scene, dimensions, rt60)
    reflection_gain = math.sqrt(max(0.0, 1.0 - absorption))
    speed_of_sound = float(scene.environment.room.get("speed_of_sound", 343.0))
    output = direct
    for image_positions in first_order_image_source_paths(source_positions, dimensions):
        image_azimuth, image_elevation, image_distance = xyz_arrays_to_az_el_dist(image_positions, listener)
        image_azimuth = world_to_camera_relative_azimuth(image_azimuth, listener_yaw)
        delay = np.maximum(0.0, (image_distance - direct_distance) / speed_of_sound)
        delayed = variable_delay(source_signal, delay * sample_rate, np.full(total_samples, reflection_gain))
        output += encode_signal_to_ambisonics(
            delayed,
            image_azimuth,
            image_elevation,
            image_distance,
            source_gain,
            order,
        )
    return output


def render_foa(scene: Scene) -> np.ndarray:
    sample_rate = scene.environment.sample_rate
    total_samples = int(round(scene.environment.duration * sample_rate))
    channel_count = (scene.environment.ambisonics_order + 1) ** 2
    output = np.zeros((channel_count, total_samples), dtype=np.float64)
    room_type = str(scene.environment.room.get("type", "direct")).lower()
    for source in scene.objects:
        mono = read_mono_audio(source.wav_path, sample_rate)
        fitted = fit_to_scene(mono, source, total_samples, sample_rate)
        listener_yaw = interpolate_listener_yaw(scene, total_samples, sample_rate)
        if room_type in {"foa_early", "shoebox_foa", "spatial_shoebox"}:
            output += render_room_early_foa(fitted, source, scene)
        elif room_type in {"direct", "anechoic", "anechoic_or_external_renderer"}:
            output += encode_object_to_ambisonics(fitted, source, scene.environment.ambisonics_order, sample_rate, listener_yaw=listener_yaw)
        else:
            fitted = render_room_mono(fitted, source, scene)
            output += encode_object_to_ambisonics(fitted, source, scene.environment.ambisonics_order, sample_rate, listener_yaw=listener_yaw)
    return output


def peak_normalize(audio: np.ndarray, peak: float = 0.98) -> np.ndarray:
    current_peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if current_peak <= 0:
        return audio
    if current_peak <= peak:
        return audio
    return audio * (peak / current_peak)


def write_multichannel(path: str | Path, channels_first: np.ndarray, sample_rate: int) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(target, peak_normalize(channels_first.T), sample_rate, subtype="FLOAT")
    return target


def export_object_package(scene_path: str | Path, output_path: str | Path) -> Path:
    scene = load_scene(scene_path)
    scene_path = Path(scene_path)
    scene_data = json.loads(scene_path.read_text())
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "SpatialObjectPackage",
        "version": 1,
        "scene_file": "scene.json",
        "audio_objects": [],
        "coordinate_system": scene.environment.coordinate_system,
        "ambisonics_normalization": scene.environment.normalization,
    }
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as package:
        packaged_scene = copy.deepcopy(scene_data)
        used_names: set[str] = set()
        for index, source in enumerate(scene.objects):
            name = f"audio/{source.object_id}_{source.wav_path.name}"
            while name in used_names:
                name = f"audio/{source.object_id}_{len(used_names)}_{source.wav_path.name}"
            used_names.add(name)
            package.write(source.wav_path, name)
            manifest["audio_objects"].append({"id": source.object_id, "path": name})
            packaged_scene["objects"][index]["wav_path"] = name
        package.writestr("scene.json", json.dumps(packaged_scene, indent=2, ensure_ascii=False))
        adm_xml = ET.tostring(build_adm_xml(scene).getroot(), encoding="utf-8", xml_declaration=True)
        package.writestr("adm.xml", adm_xml)
        manifest["adm_sidecar"] = "adm.xml"
        package.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    return output_path


def render_scene_to_foa(scene_path: str | Path, output_path: str | Path) -> Path:
    scene = load_scene(scene_path)
    foa = render_foa(scene)
    return write_multichannel(output_path, foa, scene.environment.sample_rate)


def load_hrirs(sample_rate: int, hrtf_path: str | Path | None = None):
    if hrtf_path is not None and str(hrtf_path).lower() == "dummy":
        return spa_io.load_hrirs(sample_rate, filename="dummy")
    return spa_io.load_hrirs(sample_rate, filename=str(hrtf_path) if hrtf_path else None)


def render_foa_to_binaural(foa_path: str | Path, output_path: str | Path, hrtf_path: str | Path | None = None, sample_rate: int | None = None) -> Path:
    foa, file_sample_rate = sf.read(foa_path, always_2d=True)
    foa = foa.T.astype(np.float64)
    if foa.shape[0] != 4:
        raise ValueError(f"Binaural renderer expects first-order Ambisonics with 4 channels, got {foa.shape[0]}")
    render_sample_rate = sample_rate or int(file_sample_rate)
    if render_sample_rate != file_sample_rate:
        raise ValueError("Resample FOA before binaural rendering or omit --sample-rate")
    hrirs = load_hrirs(render_sample_rate, hrtf_path)
    decoder_ir = decoder.magls_bin(hrirs, N_sph=1)
    binaural = decoder.sh2bin(foa, decoder_ir)
    return write_multichannel(output_path, binaural, render_sample_rate)


def render_scene_to_binaural(scene_path: str | Path, output_path: str | Path, hrtf_path: str | Path | None = None) -> Path:
    scene = load_scene(scene_path)
    foa = render_foa(scene)
    hrirs = load_hrirs(scene.environment.sample_rate, hrtf_path)
    decoder_ir = decoder.magls_bin(hrirs, N_sph=scene.environment.ambisonics_order)
    binaural = decoder.sh2bin(foa, decoder_ir)
    return write_multichannel(output_path, binaural, scene.environment.sample_rate)


def unpack_object_package(package_path: str | Path, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_root = output_dir.resolve()
    with zipfile.ZipFile(package_path) as package:
        for member in package.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe package member path: {member.filename}")
            target = (output_root / member_path).resolve()
            if output_root not in target.parents and target != output_root:
                raise ValueError(f"Unsafe package member path: {member.filename}")
        package.extractall(output_root)
    return output_dir
