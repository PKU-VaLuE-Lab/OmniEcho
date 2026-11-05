from __future__ import annotations

from pathlib import Path
import json
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import soundfile as sf

from omniecho_audio.render import export_object_package, render_foa_to_binaural, render_scene_to_binaural, render_scene_to_foa, unpack_object_package
from omniecho_audio.adm import write_adm_xml
from omniecho_audio.production_audit import prompt_has_required_contract, write_audit_report


def make_scene(tmp_path: Path) -> Path:
    sample_rate = 16000
    t = np.arange(sample_rate // 2) / sample_rate
    sf.write(tmp_path / "tone.wav", 0.1 * np.sin(2 * np.pi * 440 * t), sample_rate)
    scene = {
        "scene_id": "test_scene",
        "duration_sec": 0.5,
        "environment": {"audio": {"sample_rate": sample_rate, "normalization": "N3D_ACN", "ambisonics_order": 1}},
        "objects": [
            {
                "id": "tone",
                "wav_path": "tone.wav",
                "trajectory": [{"t": 0.0, "azimuth": 0.0, "elevation": 0.0, "distance": 1.0}],
            }
        ],
    }
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(scene))
    return path


def test_package_and_foa(tmp_path: Path) -> None:
    scene = make_scene(tmp_path)
    package = export_object_package(scene, tmp_path / "scene.spatpkg")
    assert package.exists()
    with zipfile.ZipFile(package) as archive:
        names = set(archive.namelist())
    assert {"manifest.json", "scene.json", "audio/tone_tone.wav", "adm.xml"}.issubset(names)
    with zipfile.ZipFile(package) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["adm_sidecar"] == "adm.xml"

    foa = render_scene_to_foa(scene, tmp_path / "scene_foa.wav")
    data, sample_rate = sf.read(foa, always_2d=True)
    assert sample_rate == 16000
    assert data.shape == (8000, 4)
    assert np.max(np.abs(data)) > 0


def test_package_roundtrip_scene_renders_after_unpack(tmp_path: Path) -> None:
    scene = make_scene(tmp_path)
    package = export_object_package(scene, tmp_path / "scene.spatpkg")
    unpacked = unpack_object_package(package, tmp_path / "unpacked")
    roundtrip_foa = render_scene_to_foa(unpacked / "scene.json", tmp_path / "roundtrip_foa.wav")
    data, sample_rate = sf.read(roundtrip_foa, always_2d=True)
    assert sample_rate == 16000
    assert data.shape == (8000, 4)
    assert np.max(np.abs(data)) > 0


def test_unpack_rejects_zip_slip(tmp_path: Path) -> None:
    malicious = tmp_path / "bad.spatpkg"
    with zipfile.ZipFile(malicious, "w") as archive:
        archive.writestr("../escape.txt", "bad")
    try:
        unpack_object_package(malicious, tmp_path / "out")
    except ValueError as exc:
        assert "Unsafe package member path" in str(exc)
    else:
        raise AssertionError("unsafe zip member was not rejected")


def test_pipeline_is_foa_only(tmp_path: Path) -> None:
    scene = json.loads(make_scene(tmp_path).read_text())
    scene["environment"]["audio"]["ambisonics_order"] = 2
    path = tmp_path / "hoa_scene.json"
    path.write_text(json.dumps(scene))
    try:
        render_scene_to_foa(path, tmp_path / "hoa.wav")
    except ValueError as exc:
        assert "FOA-only" in str(exc)
    else:
        raise AssertionError("ambisonics_order=2 should be rejected")


def test_azimuth_channel_sign_convention(tmp_path: Path) -> None:
    sample_rate = 16000
    impulse = np.zeros(128)
    impulse[0] = 1.0
    sf.write(tmp_path / "impulse.wav", impulse, sample_rate)

    def render_for_azimuth(azimuth: float) -> np.ndarray:
        scene = {
            "scene_id": f"az_{azimuth}",
            "duration_sec": 128 / sample_rate,
            "environment": {"audio": {"sample_rate": sample_rate, "normalization": "N3D_ACN", "ambisonics_order": 1}},
            "objects": [{"id": "impulse", "wav_path": "impulse.wav", "trajectory": [{"t": 0.0, "azimuth": azimuth, "elevation": 0.0, "distance": 1.0}]}],
        }
        path = tmp_path / f"az_{azimuth}.json"
        path.write_text(json.dumps(scene))
        out = render_scene_to_foa(path, tmp_path / f"az_{azimuth}.wav")
        data, _ = sf.read(out, always_2d=True)
        return data[0]

    left = render_for_azimuth(90.0)
    right = render_for_azimuth(-90.0)
    assert left[1] > 0
    assert right[1] < 0


def test_listener_yaw_rotates_camera_relative_audio(tmp_path: Path) -> None:
    sample_rate = 16000
    impulse = np.zeros(128)
    impulse[0] = 1.0
    sf.write(tmp_path / "impulse.wav", impulse, sample_rate)
    scene = {
        "scene_id": "yaw_relative",
        "duration_sec": 128 / sample_rate,
        "environment": {"audio": {"sample_rate": sample_rate, "normalization": "N3D_ACN", "ambisonics_order": 1}},
        "listener": {"position": [0.0, 0.0, 0.0], "orientation": {"yaw": 90.0}},
        "objects": [{"id": "impulse", "wav_path": "impulse.wav", "trajectory": [{"t": 0.0, "azimuth": 0.0, "elevation": 0.0, "distance": 1.0}]}],
    }
    path = tmp_path / "yaw_relative.json"
    path.write_text(json.dumps(scene))
    out = render_scene_to_foa(path, tmp_path / "yaw_relative.wav")
    data, _ = sf.read(out, always_2d=True)
    assert data[0, 1] < 0


def test_listener_yaw_trajectory_changes_foa(tmp_path: Path) -> None:
    sample_rate = 16000
    tone = 0.1 * np.sin(2 * np.pi * 440 * np.arange(sample_rate) / sample_rate)
    sf.write(tmp_path / "tone.wav", tone, sample_rate)
    base = {
        "scene_id": "yaw_dynamic",
        "duration_sec": 1.0,
        "environment": {"audio": {"sample_rate": sample_rate, "normalization": "N3D_ACN", "ambisonics_order": 1}},
        "objects": [{"id": "tone", "wav_path": "tone.wav", "trajectory": [{"t": 0.0, "azimuth": 0.0, "elevation": 0.0, "distance": 1.0}, {"t": 1.0, "azimuth": 0.0, "elevation": 0.0, "distance": 1.0}]}],
    }
    static_scene = dict(base, listener={"orientation": {"yaw": 0.0}})
    dynamic_scene = dict(base, listener={"orientation": {"trajectory": [{"t": 0.0, "yaw": 0.0}, {"t": 1.0, "yaw": 90.0}]}})
    static_path = tmp_path / "static_yaw.json"
    dynamic_path = tmp_path / "dynamic_yaw.json"
    static_path.write_text(json.dumps(static_scene))
    dynamic_path.write_text(json.dumps(dynamic_scene))
    static_out = render_scene_to_foa(static_path, tmp_path / "static_yaw.wav")
    dynamic_out = render_scene_to_foa(dynamic_path, tmp_path / "dynamic_yaw.wav")
    static_data, _ = sf.read(static_out, always_2d=True)
    dynamic_data, _ = sf.read(dynamic_out, always_2d=True)
    assert np.max(np.abs(static_data[:, 1] - dynamic_data[:, 1])) > 1e-4


def test_adm_xml_export(tmp_path: Path) -> None:
    scene = make_scene(tmp_path)
    adm_path = write_adm_xml(scene, tmp_path / "scene_adm.xml")
    root = ET.parse(adm_path).getroot()
    assert root.tag.endswith("ebuCoreMain")
    xml = adm_path.read_text()
    assert "audioProgramme" in xml
    assert "audioObject" in xml
    assert "trajectoryMetadata" in xml


def test_binaural_paths(tmp_path: Path) -> None:
    scene = make_scene(tmp_path)
    direct = render_scene_to_binaural(scene, tmp_path / "direct_binaural.wav", hrtf_path="dummy")
    direct_data, sample_rate = sf.read(direct, always_2d=True)
    assert sample_rate == 16000
    assert direct_data.shape[1] == 2

    foa = render_scene_to_foa(scene, tmp_path / "scene_foa.wav")
    from_foa = render_foa_to_binaural(foa, tmp_path / "from_foa_binaural.wav", hrtf_path="dummy")
    from_foa_data, _ = sf.read(from_foa, always_2d=True)
    assert from_foa_data.shape[1] == 2


def test_room_parameters_change_foa(tmp_path: Path) -> None:
    scene = make_scene(tmp_path)
    direct = render_scene_to_foa(scene, tmp_path / "direct_foa.wav")
    direct_data, _ = sf.read(direct, always_2d=True)

    room_scene = json.loads(scene.read_text())
    room_scene["environment"]["room"] = {"type": "shoebox", "dimensions": [4.0, 5.0, 3.0], "rt60": 0.2, "max_order": 2}
    room_scene["listener"] = {"position": [2.0, 2.5, 1.5]}
    room_path = tmp_path / "room_scene.json"
    room_path.write_text(json.dumps(room_scene))
    room = render_scene_to_foa(room_path, tmp_path / "room_foa.wav")
    room_data, _ = sf.read(room, always_2d=True)

    assert room_data.shape == direct_data.shape
    assert np.max(np.abs(room_data - direct_data)) > 1e-5


def test_spatial_early_reflections_change_directional_channels(tmp_path: Path) -> None:
    scene = make_scene(tmp_path)
    direct = render_scene_to_foa(scene, tmp_path / "direct_foa.wav")
    direct_data, _ = sf.read(direct, always_2d=True)

    reflection_scene = json.loads(scene.read_text())
    reflection_scene["environment"]["room"] = {"type": "foa_early", "dimensions": [4.0, 5.0, 3.0], "rt60": 0.35}
    reflection_scene["listener"] = {"position": [2.0, 2.5, 1.5]}
    reflection_path = tmp_path / "reflection_scene.json"
    reflection_path.write_text(json.dumps(reflection_scene))
    reflected = render_scene_to_foa(reflection_path, tmp_path / "reflected_foa.wav")
    reflected_data, _ = sf.read(reflected, always_2d=True)

    assert reflected_data.shape == direct_data.shape
    directional_delta = np.max(np.abs(reflected_data[:, 1:] - direct_data[:, 1:]))
    assert directional_delta > 1e-5


def test_spatial_early_reflections_follow_dynamic_trajectory(tmp_path: Path) -> None:
    sample_rate = 16000
    t = np.arange(sample_rate) / sample_rate
    sf.write(tmp_path / "tone.wav", 0.1 * np.sin(2 * np.pi * 330 * t), sample_rate)
    base_scene = {
        "scene_id": "dynamic_reflection_test",
        "duration_sec": 1.0,
        "environment": {"audio": {"sample_rate": sample_rate, "normalization": "N3D_ACN", "ambisonics_order": 1}, "room": {"type": "foa_early", "dimensions": [5.0, 4.0, 3.0], "rt60": 0.3}},
        "listener": {"position": [2.5, 2.0, 1.5]},
        "objects": [{"id": "tone", "wav_path": "tone.wav", "trajectory": [{"t": 0.0, "azimuth": -45.0, "elevation": 0.0, "distance": 1.0}]}],
    }
    static_path = tmp_path / "static.json"
    static_path.write_text(json.dumps(base_scene))
    dynamic_scene = json.loads(json.dumps(base_scene))
    dynamic_scene["objects"][0]["trajectory"] = [
        {"t": 0.0, "azimuth": -45.0, "elevation": 0.0, "distance": 1.0},
        {"t": 1.0, "azimuth": 45.0, "elevation": 0.0, "distance": 2.0},
    ]
    dynamic_path = tmp_path / "dynamic.json"
    dynamic_path.write_text(json.dumps(dynamic_scene))

    static_foa = render_scene_to_foa(static_path, tmp_path / "static_foa.wav")
    dynamic_foa = render_scene_to_foa(dynamic_path, tmp_path / "dynamic_foa.wav")
    static_data, _ = sf.read(static_foa, always_2d=True)
    dynamic_data, _ = sf.read(dynamic_foa, always_2d=True)

    assert dynamic_data.shape == static_data.shape
    second_half_delta = np.max(np.abs(dynamic_data[sample_rate // 2 :, 1:] - static_data[sample_rate // 2 :, 1:]))
    assert second_half_delta > 1e-5








def test_prompt_contract_requires_audio_and_negative_constraints() -> None:
    prompt = "Realistic meeting room, 16:9, 720p, 6 seconds. Spatial layout: A is 1.5 meters to the front left. 0-2 seconds: speech; 2-4 seconds: a pen taps the table; 4-6 seconds: a response. Audio design: pen taps must be clear. No background music, subtitles, or watermarks."
    contract = prompt_has_required_contract(prompt)
    assert all(contract.values())
    weak = prompt_has_required_contract("A realistic meeting room with two people talking.")
    assert weak["timeline"] is False
    assert weak["audio_design"] is False
    assert weak["negative_constraints"] is False
