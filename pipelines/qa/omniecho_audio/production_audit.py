from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re
import subprocess
from typing import Any

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class AuditCheck:
    name: str
    passed: bool
    severity: str
    value: Any = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "severity": self.severity,
            "value": self.value,
            "message": self.message,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audio_info(path: Path) -> dict[str, Any]:
    data, sample_rate = sf.read(path, always_2d=True)
    return {
        "sample_rate": int(sample_rate),
        "channels": int(data.shape[1]),
        "frames": int(data.shape[0]),
        "duration_sec": float(data.shape[0] / sample_rate),
        "rms": float(np.sqrt(np.mean(data**2))),
        "peak": float(np.max(np.abs(data))),
        "sha256": sha256_file(path),
    }


def video_streams(path: Path) -> list[dict[str, Any]]:
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=index,codec_type,codec_name,width,height,sample_rate,channels",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    return list(json.loads(output).get("streams", []))


def add(checks: list[AuditCheck], name: str, passed: bool, severity: str = "error", value: Any = None, message: str = "") -> None:
    checks.append(AuditCheck(name, bool(passed), severity, value, message))


def prompt_has_required_contract(prompt: str) -> dict[str, bool]:
    text = (prompt or "").lower()
    return {
        "timeline": bool(re.search(r"\b\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\s*(?:s\b|seconds?\b)", text)),
        "spatial_layout": bool(re.search(r"\b(?:spatial layout|left|right|front|above|distance|meters?|metres?)\b", text)),
        "audio_design": bool(re.search(r"\b(?:sound design|audio design|audio)\b", text)),
        "negative_constraints": bool(re.search(r"\b(?:no|without|forbid(?:den)?|prohibit(?:ed)?|do not)\b", text)),
        "no_music": bool(re.search(r"\b(?:no|without|forbid(?:den)?|prohibit(?:ed)?)\s+(?:background\s+)?music\b", text)),
    }


def transient_score(path: Path) -> float:
    data, sample_rate = sf.read(path, always_2d=True)
    mono = data.mean(axis=1)
    if len(mono) < sample_rate // 10:
        return 0.0
    global_rms = float(np.sqrt(np.mean(mono**2))) + 1e-9
    peak_to_rms = float(np.max(np.abs(mono)) / global_rms)
    envelope = np.abs(mono)
    onset = np.diff(envelope, prepend=envelope[0])
    onset_score = float(np.percentile(np.maximum(onset, 0.0), 99.9) / global_rms)
    return max(peak_to_rms, onset_score)


def pulse_score(path: Path, sample_rate: int = 48000) -> dict[str, Any]:
    data, file_sample_rate = sf.read(path, always_2d=True)
    mono = data.mean(axis=1)
    block = max(1, int(0.25 * file_sample_rate))
    rms = [float(np.sqrt(np.mean(mono[index : index + block] ** 2))) for index in range(0, len(mono), block) if len(mono[index : index + block])]
    if not rms:
        return {"active_blocks": 0, "silent_blocks": 0, "alternating": False, "block_rms": []}
    threshold = max(np.max(rms) * 0.25, 1e-5)
    active = [value > threshold for value in rms]
    return {
        "active_blocks": int(sum(active)),
        "silent_blocks": int(len(active) - sum(active)),
        "alternating": any(active[index] and not active[index + 1] for index in range(len(active) - 1)),
        "block_rms": rms,
        "sample_rate": sample_rate,
    }


def audit_sample(sample_dir: str | Path) -> dict[str, Any]:
    sample = Path(sample_dir)
    checks: list[AuditCheck] = []
    gt = sample / "ground_truth"
    ref = sample / "realistic_reference"
    debug = sample / "debug"
    metadata_path = sample / "metadata.json"
    scene_path = gt / "scene.json"

    required = [
        scene_path,
        gt / "audio_foa.wav",
        gt / "audio_binaural.wav",
        gt / "frame_labels.json",
        gt / "qa.json",
        gt / "object_package.spatpkg",
        ref / "real_video.mp4",
        ref / "reference_stereo_audio.wav",
        debug / "verification.json",
        metadata_path,
        sample / "agent_trace.json",
        sample / "README.md",
    ]
    for path in required:
        add(checks, f"exists:{path.relative_to(sample)}", path.exists(), value=str(path.relative_to(sample)))

    if not scene_path.exists() or not metadata_path.exists():
        return {"sample": str(sample), "passed": False, "checks": [check.to_dict() for check in checks]}

    scene = json.loads(scene_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    duration = float(scene.get("duration_sec", 0.0))
    objects = list(scene.get("objects", []))
    add(checks, "object_count", len(objects) > 0, value=len(objects))

    for rel, channels in [("audio_foa.wav", 4), ("audio_binaural.wav", 2)]:
        path = gt / rel
        if path.exists():
            info = audio_info(path)
            add(checks, f"audio_shape:{rel}", info["sample_rate"] == 48000 and info["channels"] == channels, value=info)
            add(checks, f"audio_non_silent:{rel}", info["rms"] > 1e-7, value=info["rms"])
            add(checks, f"duration_match:{rel}", abs(info["duration_sec"] - duration) < 0.02 or rel == "audio_binaural.wav", value=info["duration_sec"])

    ref_audio = ref / "reference_stereo_audio.wav"
    if ref_audio.exists():
        info = audio_info(ref_audio)
        add(checks, "reference_audio_is_stereo", info["sample_rate"] == 48000 and info["channels"] == 2, value=info)

    contract = metadata.get("truth_audio_contract", {})
    add(checks, "truth_contract_foa", contract.get("training_truth") == "ground_truth/audio_foa.wav", value=contract)
    add(checks, "reference_not_truth", "reference_stereo_audio.wav" in json.dumps(contract, ensure_ascii=False) and bool(re.search(r"\bnot\b", json.dumps(contract, ensure_ascii=False), re.I)), value=contract)
    camera_contract = metadata.get("camera_audio_contract", {})
    orientation = scene.get("listener", {}).get("orientation", {})
    add(checks, "camera_audio_contract", camera_contract.get("enabled") is True and "yaw" in json.dumps(camera_contract), value=camera_contract)
    add(checks, "listener_orientation_trajectory", bool(orientation.get("trajectory")), value=orientation)

    semantic_stems = metadata.get("semantic_stems", {})
    for obj in objects:
        object_id = str(obj.get("id"))
        stem_rel = str(obj.get("wav_path", ""))
        stem_path = gt / stem_rel
        add(checks, f"stem_exists:{object_id}", stem_path.exists(), value=stem_rel)
        if stem_path.exists():
            info = audio_info(stem_path)
            add(checks, f"stem_shape:{object_id}", info["sample_rate"] == 48000 and info["channels"] == 1, value=info)
            add(checks, f"stem_non_silent:{object_id}", info["rms"] > 1e-6, value=info["rms"])
            provenance = semantic_stems.get(object_id, {})
            add(checks, f"stem_provenance:{object_id}", bool(provenance.get("source_type") and provenance.get("path") and provenance.get("sha256")), value=provenance)
            if provenance.get("sha256"):
                add(checks, f"stem_sha:{object_id}", provenance.get("sha256") == info["sha256"], value={"metadata": provenance.get("sha256"), "actual": info["sha256"]})
            if "tap" in object_id:
                score = transient_score(stem_path)
                add(checks, f"transient_audibility:{object_id}", score > 3.0, value=score, message="short events should have clear attacks")
            if "alarm" in object_id:
                score = pulse_score(stem_path)
                add(checks, f"pulse_alarm:{object_id}", score["active_blocks"] > 1 and score["silent_blocks"] > 1 and score["alternating"], value=score)
            if "bird" in object_id:
                provenance_text = json.dumps(provenance, ensure_ascii=False).lower()
                add(checks, f"bird_not_toy_music:{object_id}", "toy" not in provenance_text and "song.mp3" not in provenance_text, value=provenance)

    labels_path = gt / "frame_labels.json"
    if labels_path.exists():
        labels = json.loads(labels_path.read_text())
        add(checks, "frame_label_count", len(labels) >= 2, value=len(labels))
        if labels:
            source_counts = {len(frame.get("sources", [])) for frame in labels}
            add(checks, "frame_label_source_count", source_counts == {len(objects)}, value=sorted(source_counts))
            add(checks, "frame_label_has_listener_yaw", all("listener_yaw" in frame for frame in labels), value=labels[0].get("listener_yaw"))
            add(
                checks,
                "frame_label_camera_relative_azimuth",
                all("camera_relative_azimuth" in source and "azimuth_world" in source for frame in labels for source in frame.get("sources", [])),
                value=labels[0].get("sources", [])[:2],
            )

    qa_path = gt / "qa.json"
    if qa_path.exists():
        qa = json.loads(qa_path.read_text())
        add(checks, "qa_count", len(qa) >= len(objects), value=len(qa))
        add(checks, "qa_has_evidence", all("evidence" in item for item in qa), value=len(qa))

    prompt = metadata.get("realistic_assets", {}).get("generation_prompt_improved") or metadata.get("realistic_assets", {}).get("generation_prompt", "")
    prompt_contract = prompt_has_required_contract(prompt)
    for key, passed in prompt_contract.items():
        add(checks, f"prompt_contract:{key}", passed, value=prompt[:500])

    readme = (sample / "README.md").read_text() if (sample / "README.md").exists() else ""
    add(checks, "readme_warns_reference_not_truth", "reference_stereo_audio.wav" in readme and "ground truth" in readme.lower(), value=None)

    verification_path = debug / "verification.json"
    if verification_path.exists():
        verification = json.loads(verification_path.read_text())
        add(checks, "debug_verifier_passed", verification.get("passed") is True, value=verification.get("passed"))

    trace_path = sample / "agent_trace.json"
    if trace_path.exists():
        trace = json.loads(trace_path.read_text())
        add(checks, "agent_trace_sequence", trace.get("agent_sequence") == ["scene_perception_agent", "director_agent", "audio_synthesis_agent", "video_generation_agent", "qa_generation_agent", "verifier_agent"], value=trace.get("agent_sequence"))
        add(checks, "agent_trace_camera_contract", trace.get("camera_audio_contract", {}).get("enabled") is True, value=trace.get("camera_audio_contract"))
        agents = trace.get("agents", {})
        for agent_name in ["scene_perception_agent", "director_agent", "audio_synthesis_agent", "video_generation_agent", "qa_generation_agent", "verifier_agent"]:
            add(checks, f"agent_trace_has:{agent_name}", agent_name in agents, value=list(agents))

    passed = all(check.passed or check.severity != "error" for check in checks)
    return {"sample": str(sample), "passed": passed, "checks": [check.to_dict() for check in checks]}


def audit_delivery(delivery_root: str | Path) -> dict[str, Any]:
    root = Path(delivery_root)
    checks: list[AuditCheck] = []
    expected = ["binaural_audio.wav", "detailed_visual_description.md", "video.mp4"]
    sample_dirs = sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("sf_")) if root.exists() else []
    add(checks, "delivery_root_exists", root.exists(), value=str(root))
    add(checks, "delivery_sample_count", len(sample_dirs) >= 1, value=len(sample_dirs))
    for sample in sample_dirs:
        files = sorted(path.name for path in sample.iterdir() if path.is_file())
        add(checks, f"delivery_three_files:{sample.name}", files == expected, value=files)
        video = sample / "video.mp4"
        if video.exists():
            streams = video_streams(video)
            add(checks, f"delivery_video_silent:{sample.name}", not any(stream.get("codec_type") == "audio" for stream in streams), value=streams)
        for wav in ["binaural_audio.wav"]:
            path = sample / wav
            if path.exists():
                info = audio_info(path)
                add(checks, f"delivery_audio_shape:{sample.name}:{wav}", info["sample_rate"] == 48000 and info["channels"] == 2, value=info)
        desc = sample / "detailed_visual_description.md"
        if desc.exists():
            text = desc.read_text()
            for token in ["audio design", "not spatial ground truth", "seedance", "spatial ground-truth object trajectories"]:
                add(checks, f"delivery_desc:{sample.name}:{token}", token in text.lower(), value=None)
    passed = all(check.passed or check.severity != "error" for check in checks)
    return {"delivery_root": str(root), "passed": passed, "checks": [check.to_dict() for check in checks]}


def audit_dataset(dataset_root: str | Path, delivery_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(dataset_root)
    samples = sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("sf_")) if root.exists() else []
    sample_reports = [audit_sample(sample) for sample in samples]
    delivery_report = audit_delivery(delivery_root) if delivery_root is not None else None
    passed = bool(samples) and all(report["passed"] for report in sample_reports) and (delivery_report is None or delivery_report["passed"])
    return {
        "dataset_root": str(root),
        "sample_count": len(samples),
        "passed": passed,
        "samples": sample_reports,
        "delivery": delivery_report,
    }


def write_audit_report(dataset_root: str | Path, output: str | Path, delivery_root: str | Path | None = None) -> Path:
    report = audit_dataset(dataset_root, delivery_root=delivery_root)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        failing = []
        for sample in report["samples"]:
            failing.extend(f"{sample['sample']}::{check['name']}" for check in sample["checks"] if not check["passed"] and check["severity"] == "error")
        if report["delivery"]:
            failing.extend(f"delivery::{check['name']}" for check in report["delivery"]["checks"] if not check["passed"] and check["severity"] == "error")
        raise RuntimeError(f"Production audit failed; report={output_path}; failing={failing[:20]}")
    return output_path
