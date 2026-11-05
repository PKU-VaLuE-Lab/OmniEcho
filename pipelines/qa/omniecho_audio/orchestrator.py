"""OmniEcho processing component."""

from __future__ import annotations

import json
import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .director import generate_scene_and_prompts, DirectorOutput
from .schema import CameraMode
from .video_prompt import (
    submit_video_job,
    poll_video_result,
    download_video,
    generate_video,
    SubVideoTaskFailed,
)
from .audio_extract import (
    extract_audio_from_video, normalize_stem,
    measure_lufs, normalize_to_target_lufs,
    apply_noise_gate, detect_and_trim_onset, apply_fade_envelope,
)
from .render import render_scene_to_foa, render_foa_to_binaural

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metadata correction helper
# ---------------------------------------------------------------------------

def run_metadata_correction(
    sample_dir: Path,
    corrected_scene_path: Path,
    camera_mode: CameraMode = CameraMode.STATIC,
) -> Path:
    """OmniEcho processing component."""
    
    if camera_mode != CameraMode.STATIC:
        msg = (
            f"OmniEcho metadata correction currently supports only camera_mode=STATIC; "
            f"received {camera_mode.value!r}. PAN/ROTATE are not implemented; legacy yaw estimation is not used."
        )
        logger.error(msg)
        raise NotImplementedError(msg)


    from visual_validation.config import ValidationConfig
    from visual_validation.metadata_correction import MetadataCorrector

    video_path = sample_dir / "video.mp4"
    scene_path = sample_dir / "scene.json"
    if not video_path.is_file():
        raise FileNotFoundError(f"video.mp4 not found: {video_path}")
    if not scene_path.is_file():
        raise FileNotFoundError(f"scene.json not found: {scene_path}")

    config = ValidationConfig.default()
    corrector = MetadataCorrector(config)

    logger.info(
        "[metadata_correction] Starting OmniEcho STATIC pipeline: video=%s scene=%s output=%s",
        video_path.name, scene_path.name, sample_dir,
    )


    report = corrector.correct_sample(
        video_path=video_path,
        scene_path=scene_path,
        output_dir=sample_dir,
    )
    logger.info(
        "[metadata_correction] OmniEcho complete: scene_id=%s, n_objects=%d",
        report.get("scene_id", "?"),
        len(report.get("per_object", []) or report.get("visibility", {})),
    )


    default_corrected = sample_dir / "corrected_scene.json"
    if default_corrected.resolve() != corrected_scene_path.resolve():
        
        if default_corrected.is_file():
            corrected_scene_path.write_text(
                default_corrected.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

    if not corrected_scene_path.is_file():
        raise RuntimeError(
            f"MetadataCorrector did not produce {corrected_scene_path}"
        )


    
    try:
        corrected_scene = json.loads(corrected_scene_path.read_text(encoding="utf-8"))
        original_scene = json.loads(scene_path.read_text(encoding="utf-8"))
        orig_by_id = {obj.get("id"): obj for obj in original_scene.get("objects", [])}
        for corrected_obj in corrected_scene.get("objects", []):
            obj_id = corrected_obj.get("id", "")
            parent_id = (
                "_".join(obj_id.rsplit("_", 1)[:-1])
                if obj_id not in orig_by_id
                else obj_id
            )
            orig_obj = orig_by_id.get(obj_id) or orig_by_id.get(parent_id) or {}
            for fld in ("start_time", "end_time", "onset_t", "offset_t", "wav_path", "gain_db"):
                if fld in orig_obj and fld not in corrected_obj:
                    corrected_obj[fld] = orig_obj[fld]
        corrected_scene_path.write_text(
            json.dumps(corrected_scene, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as ex:  
        logger.warning("Failed to populate missing corrected_scene.json fields: %s", ex)

    logger.info("corrected_scene.json → %s", corrected_scene_path)
    return corrected_scene_path


def _ensure_role_field(scene_dict: dict) -> dict:
    """OmniEcho processing component."""
    objects = scene_dict.get("objects")
    if not isinstance(objects, list):
        return scene_dict
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        role = obj.get("role")
        if role not in ("foreground", "ambient"):
            snr = obj.get("snr_db")
            try:
                snr_val = float(snr) if snr is not None else None
            except (TypeError, ValueError):
                snr_val = None
            inferred = "ambient" if snr_val is not None and abs(snr_val) < 1e-6 else "foreground"
            
            if "label" in obj:
                new_obj = {}
                inserted = False
                for k, v in obj.items():
                    new_obj[k] = v
                    if not inserted and k == "label":
                        new_obj["role"] = inferred
                        inserted = True
                if not inserted:
                    new_obj["role"] = inferred
                obj.clear()
                obj.update(new_obj)
            else:
                obj["role"] = inferred
    return scene_dict


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

def lufs_to_gain_db(loudness_lufs: float, reference_lufs: float = -23.0) -> float:
    """OmniEcho processing component."""
    return loudness_lufs - reference_lufs


def convert_lufs_to_gain_db(scene_dict: dict, reference_lufs: float = -23.0) -> dict:
    """OmniEcho processing component."""
    objects = scene_dict.get("objects", [])
    for obj in objects:
        lufs = obj.get("loudness_lufs")
        if lufs is not None:
            obj["gain_db"] = lufs_to_gain_db(float(lufs), reference_lufs)
        
        obj.pop("gain", None)
    return scene_dict


def normalize_scene_for_render(scene_dict: dict) -> dict:
    """OmniEcho processing component."""
    # 1. duration → duration_sec
    if "duration_sec" not in scene_dict and "duration" in scene_dict:
        scene_dict["duration_sec"] = scene_dict["duration"]


    env = scene_dict.get("environment", {})
    if "audio" not in env:
        env["audio"] = {
            "sample_rate": 48000,
            "ambisonics_order": 1,
            "normalization": "N3D_ACN",
        }
    scene_dict["environment"] = env


    room = env.get("room", {})
    if "type" not in room:
        
        dims = room.get("dimensions", [])
        has_valid_dims = (
            isinstance(dims, list) and len(dims) == 3
            and all(isinstance(d, (int, float)) and d > 0 for d in dims)
        )
        has_rt60 = isinstance(room.get("rt60"), (int, float)) and room["rt60"] > 0
        
        is_outdoor = str(env.get("type", "")).lower() == "outdoor"
        if has_valid_dims and has_rt60 and not is_outdoor:
            room["type"] = "foa_early"
        else:
            room["type"] = "direct"
    env["room"] = room

    return scene_dict


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """OmniEcho processing component."""
    output_root: Path = field(default_factory=lambda: Path("outputs/OmniEchoBench-QA"))
    n_foreground: int = 3
    include_ambient: bool = True
    
    allow_camera_rotation: bool = False
    
    camera_mode: CameraMode = CameraMode.STATIC
    scene_duration: float = 6.0
    skip_video: bool = False
    max_stage_retries: int = 3
    
    enable_offscreen_agent: bool = False

    def __post_init__(self) -> None:
        
        if isinstance(self.camera_mode, str):
            self.camera_mode = CameraMode(self.camera_mode)
        
        if self.camera_mode == CameraMode.STATIC and self.allow_camera_rotation:
            warnings.warn(
                "camera_mode=STATIC conflicts with allow_camera_rotation=True; "
                "forcing allow_camera_rotation=False to follow the static-camera contract.",
                stacklevel=2,
            )
            self.allow_camera_rotation = False


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

def _log_stage(stage_name: str, status: str, detail: str = "", elapsed: float = 0.0):
    """OmniEcho processing component."""
    emoji = "✓" if status == "success" else "✗" if status == "failed" else "…"
    logger.info(
        "[%s] %s %s (%.1fs)%s",
        stage_name, emoji, status, elapsed,
        f" | {detail}" if detail else "",
    )


def _retry_stage(stage_name: str, func, max_retries: int = 3):
    """OmniEcho processing component."""
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        try:
            result = func()
            elapsed = time.time() - t0
            _log_stage(stage_name, "success",
                       f"attempt {attempt}/{max_retries}", elapsed)
            return result
        except Exception as e:
            elapsed = time.time() - t0
            last_error = e
            _log_stage(stage_name, "failed",
                       f"attempt {attempt}/{max_retries}: {e}", elapsed)
            if attempt < max_retries:
                backoff = 2 ** attempt  # 2s, 4s, 8s
                logger.info("[%s] Retrying after %ds...", stage_name, backoff)
                time.sleep(backoff)

    raise RuntimeError(
        f"Stage [{stage_name}] failed after {max_retries} attempts. "
        f"Last error: {last_error}"
    )


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

def generate_single_sample(
    config: PipelineConfig,
    sample_id: str,
    previous_scenes: list[str] | None = None,
) -> Path:
    """OmniEcho processing component."""
    t_start = time.time()
    retries = config.max_stage_retries

    logger.info("=" * 60)
    logger.info(
        "Generating sample: %s (pipeline camera_mode=%s)",
        sample_id, config.camera_mode.value,
    )
    logger.info("=" * 60)


    sample_dir = config.output_root / sample_id
    stems_dir = sample_dir / "stems"
    sub_videos_dir = sample_dir / "sub_videos"
    sub_prompts_dir = sample_dir / "sub_prompts"
    stems_dir.mkdir(parents=True, exist_ok=True)
    sub_videos_dir.mkdir(parents=True, exist_ok=True)
    sub_prompts_dir.mkdir(parents=True, exist_ok=True)


    director_output: DirectorOutput = _retry_stage(
        "director",
        lambda: generate_scene_and_prompts(
            n_foreground=config.n_foreground,
            include_ambient=config.include_ambient,
            scene_duration=config.scene_duration,
            allow_camera_rotation=config.allow_camera_rotation,
            previous_scenes=previous_scenes,
            output_dir=str(sample_dir),
        ),
        max_retries=retries,
    )
    scene_dict = director_output.scene
    sub_prompts = director_output.sub_prompts
    video_prompt_text = director_output.video_prompt
    objects = scene_dict.get("objects", [])

    logger.info("Director complete: %d objects, %d sub_prompts",
                len(objects), len(sub_prompts))


    for i, (obj, prompt) in enumerate(zip(objects, sub_prompts)):
        obj_id = obj.get("id", f"source_{i+1:03d}")
        prompt_file = sub_prompts_dir / f"{obj_id}.txt"
        prompt_file.write_text(prompt, encoding="utf-8")


    video_prompt_file = sample_dir / "video_prompt.txt"
    video_prompt_file.write_text(video_prompt_text, encoding="utf-8")


    
    
    def _duration_decision():
        from .duration_agent import DurationAgent

        agent = DurationAgent()
        try:
            env = scene_dict.get("environment", {}) or {}
            scene_desc = (
                env.get("description")
                or scene_dict.get("description")
                or video_prompt_text
                or ""
            )

            try:
                result = agent.decide_durations(
                    scene_description=scene_desc,
                    objects=objects,
                    scene_duration=config.scene_duration,
                )
            except Exception as ex:
                logger.warning(
                    "DurationAgent LLM decision failed; retaining original Director timing: %s", ex,
                )
                return {"decisions": [], "raw_response": "", "error": str(ex)}


            decisions_map = {
                d["object_id"]: d for d in result.get("decisions", [])
            }
            for obj in objects:
                obj_id = obj.get("id", "")
                if obj_id in decisions_map:
                    d = decisions_map[obj_id]
                    old_onset = obj.get("onset_t", 0.0)
                    old_offset = obj.get("offset_t", config.scene_duration)
                    obj["onset_t"] = d["onset_t"]
                    obj["offset_t"] = d["offset_t"]
                    logger.info(
                        "  %s timing adjustment: [%.1f, %.1f] -> [%.1f, %.1f] (%s)",
                        obj_id, old_onset, old_offset,
                        d["onset_t"], d["offset_t"],
                        d.get("reason", "")[:60],
                    )


            (sample_dir / "duration_decision.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return result
        finally:
            agent.close()

    _retry_stage("duration_decision", _duration_decision, max_retries=retries)


    


    


    max_submit_retries = max(1, retries)

    def _submit_one(prompt: str, sub_duration_int: int) -> str:
        return submit_video_job(prompt, duration=sub_duration_int)

    def _submit_poll_download_all():
        """OmniEcho processing component."""
        import concurrent.futures


        tasks_info: list[dict] = []
        for i, prompt in enumerate(sub_prompts):
            obj = objects[i] if i < len(objects) else {}
            obj_id = obj.get("id", f"source_{i+1:03d}")
            onset_t = float(obj.get("onset_t", 0.0))
            offset_t = float(obj.get("offset_t", config.scene_duration))
            sub_duration_int = max(2, min(10, int(round(offset_t - onset_t))))
            video_path = sub_videos_dir / f"{obj_id}.mp4"
            tasks_info.append({
                "index": i,
                "prompt": prompt,
                "obj_id": obj_id,
                "sub_duration_int": sub_duration_int,
                "video_path": video_path,
            })

        def _process_one_source(info: dict) -> Path:
            """OmniEcho processing component."""
            idx = info["index"]
            prompt = info["prompt"]
            obj_id = info["obj_id"]
            sub_duration_int = info["sub_duration_int"]
            video_path = info["video_path"]

            video_url: Optional[str] = None
            last_failed: Optional[SubVideoTaskFailed] = None

            for sub_attempt in range(1, max_submit_retries + 1):
                task_id = _submit_one(prompt, sub_duration_int)
                logger.info(
                    "Sub-video %d/%d submitted (sub_attempt=%d/%d), task_id=%s (%s, duration=%ds)",
                    idx + 1, len(sub_prompts), sub_attempt, max_submit_retries,
                    task_id, obj_id, sub_duration_int,
                )
                try:
                    logger.info(
                        "Polling sub-video %d/%d (task_id=%s)...",
                        idx + 1, len(sub_prompts), task_id,
                    )
                    video_url = poll_video_result(task_id, timeout=600)
                    break
                except SubVideoTaskFailed as fe:
                    last_failed = fe
                    logger.warning(
                        "Detected FAILED task (source=%s task_id=%s code=%s message=%s); "
                        "resubmitting, sub_attempt=%d/%d...",
                        obj_id, fe.task_id, fe.code, fe.message,
                        sub_attempt, max_submit_retries,
                    )
                    if sub_attempt >= max_submit_retries:
                        raise RuntimeError(
                            f"Sub-video {obj_id} still FAILED after {max_submit_retries} resubmissions: "
                            f"last_task_id={fe.task_id} code={fe.code} message={fe.message}"
                        ) from fe
                    continue

            if video_url is None:
                raise RuntimeError(
                    f"Sub-video {obj_id} returned no video_url (last_failed={last_failed})"
                )

            download_video(video_url, video_path)
            logger.info(
                "Sub-video %d/%d downloaded: %s", idx + 1, len(sub_prompts), video_path,
            )
            return video_path


        paths: list[Path] = [None] * len(tasks_info)  # type: ignore
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks_info)) as executor:
            future_map = {
                executor.submit(_process_one_source, info): info["index"]
                for info in tasks_info
            }
            for future in concurrent.futures.as_completed(future_map):
                idx = future_map[future]
                paths[idx] = future.result()  

        return paths

    sub_video_paths: list[Path] = _retry_stage(
        "submit_poll_download_sub_videos",
        _submit_poll_download_all,
        max_retries=retries,
    )


    def _extract_stems():
        paths = []
        for i, (video_path, obj) in enumerate(zip(sub_video_paths, objects)):
            obj_id = obj.get("id", f"source_{i+1:03d}")
            stem_path = stems_dir / f"{obj_id}.wav"
            extract_audio_from_video(video_path, stem_path)
            paths.append(stem_path)
            logger.info("Audio extraction %d/%d: %s -> %s", i + 1, len(sub_video_paths), video_path.name, stem_path.name)
        return paths

    stems_paths: list[Path] = _retry_stage("extract_audio", _extract_stems, max_retries=retries)


    AMBIENT_KEYWORDS = {"rain", "wind", "hvac", "water", "stream", "traffic",
                        "breeze", "ambient", "hum", "drone", "cicada", "cricket",
                        "river", "ocean", "waves", "thunder", "rain", "wind", "water flow",
                        "ambient", "background noise", "air conditioning"}


    def _trim_stems_to_duration():
        from scipy.io import wavfile as _wf
        import numpy as _np
        for i, stem_path in enumerate(stems_paths):
            obj = objects[i] if i < len(objects) else {}
            onset_t = float(obj.get("onset_t", 0.0))
            offset_t = float(obj.get("offset_t", config.scene_duration))
            target_dur = offset_t - onset_t
            
            sr, data = _wf.read(stem_path)
            actual_dur = len(data) / sr
            if actual_dur > target_dur + 0.1:  
                target_samples = int(target_dur * sr)
                data_trimmed = data[:target_samples]
                _wf.write(stem_path, sr, data_trimmed)
                logger.info("  Trimmed stem %s: %.2fs -> %.2fs",
                           stem_path.name, actual_dur, target_dur)
            elif actual_dur < target_dur - 0.1:  
                target_samples = int(target_dur * sr)
                
                label_lower = obj.get("label", "").lower()
                obj_type = obj.get("type", obj.get("sound_type", "")).lower()
                is_ambient = any(kw in label_lower or kw in obj_type for kw in AMBIENT_KEYWORDS)

                if is_ambient:
                    
                    orig_dtype = data.dtype
                    crossfade_samples = int(0.05 * sr)  # 50ms
                    
                    looped = data.astype(_np.float64)
                    src = data.astype(_np.float64)
                    while len(looped) < target_samples:
                        overlap = min(crossfade_samples, len(looped), len(src))
                        if overlap > 0:
                            fade_out = _np.linspace(1.0, 0.0, overlap)
                            fade_in = _np.linspace(0.0, 1.0, overlap)
                            if looped.ndim == 1:
                                looped[-overlap:] = looped[-overlap:] * fade_out + src[:overlap] * fade_in
                                looped = _np.concatenate([looped, src[overlap:]])
                            else:
                                
                                looped[-overlap:] = (looped[-overlap:] * fade_out[:, None]
                                                     + src[:overlap] * fade_in[:, None])
                                looped = _np.concatenate([looped, src[overlap:]], axis=0)
                        else:
                            looped = _np.concatenate([looped, src] if looped.ndim == 1
                                                    else [looped, src], axis=0)
                    
                    looped = looped[:target_samples]
                    
                    if _np.issubdtype(orig_dtype, _np.integer):
                        info = _np.iinfo(orig_dtype)
                        looped = _np.clip(looped, info.min, info.max).astype(orig_dtype)
                    else:
                        looped = looped.astype(orig_dtype)
                    _wf.write(stem_path, sr, looped)
                    logger.warning("  Stem %s too short (%.2fs < %.2fs); looped to the required duration",
                                   stem_path.name, actual_dur, target_dur)
                else:
                    
                    pad_samples = target_samples - len(data)
                    if data.ndim == 1:
                        padded = _np.concatenate([data, _np.zeros(pad_samples, dtype=data.dtype)])
                    else:
                        padded = _np.concatenate([data, _np.zeros((pad_samples, data.shape[1]),
                                                                   dtype=data.dtype)], axis=0)
                    _wf.write(stem_path, sr, padded)
                    logger.warning("  Stem %s too short (%.2fs < %.2fs); padded with zeros",
                                   stem_path.name, actual_dur, target_dur)

    _retry_stage("trim_stems_to_duration", _trim_stems_to_duration, max_retries=retries)


    def _volume_statistics():
        """OmniEcho processing component."""
        import numpy as _np
        from scipy.io import wavfile as _wf

        stats: list[dict] = []
        for i, stem_path in enumerate(stems_paths):
            obj = objects[i] if i < len(objects) else {}
            obj_id = obj.get("id", f"source_{i+1:03d}")

            sr, data = _wf.read(stem_path)
            
            if data.dtype == _np.int16:
                audio_float = data.astype(_np.float64) / 32768.0
            elif data.dtype == _np.int32:
                audio_float = data.astype(_np.float64) / 2147483648.0
            else:
                audio_float = data.astype(_np.float64)


            if audio_float.ndim > 1:
                audio_float = audio_float[:, 0]


            duration_sec = len(audio_float) / sr
            rms = float(_np.sqrt(_np.mean(audio_float ** 2)))
            rms_db = float(20 * _np.log10(rms + 1e-10))
            peak = float(_np.max(_np.abs(audio_float)))
            peak_dbfs = float(20 * _np.log10(peak + 1e-10))


            approx_lufs = rms_db - 0.691  


            from numpy.fft import rfft, rfftfreq
            fft_mag = _np.abs(rfft(audio_float))
            freqs = rfftfreq(len(audio_float), 1.0/sr)

            low_mask = freqs < 300
            mid_mask = (freqs >= 300) & (freqs < 3000)
            high_mask = freqs >= 3000

            total_energy = float(_np.sum(fft_mag ** 2)) + 1e-10
            low_ratio = float(_np.sum(fft_mag[low_mask] ** 2) / total_energy)
            mid_ratio = float(_np.sum(fft_mag[mid_mask] ** 2) / total_energy)
            high_ratio = float(_np.sum(fft_mag[high_mask] ** 2) / total_energy)

            stat = {
                "object_id": obj_id,
                "label": obj.get("label", ""),
                "type": obj.get("type", ""),
                "duration_sec": round(duration_sec, 2),
                "rms_db": round(rms_db, 1),
                "peak_dbfs": round(peak_dbfs, 1),
                "approx_lufs": round(approx_lufs, 1),
                "spectral_balance": {
                    "low_pct": round(low_ratio * 100, 1),
                    "mid_pct": round(mid_ratio * 100, 1),
                    "high_pct": round(high_ratio * 100, 1),
                },
            }
            stats.append(stat)

            logger.info(
                "  %s (%s): RMS=%.1fdB peak=%.1fdBFS spectrum=[low %.0f%%/mid %.0f%%/high %.0f%%] dur=%.1fs",
                obj_id, obj.get("label", ""), rms_db, peak_dbfs,
                low_ratio*100, mid_ratio*100, high_ratio*100, duration_sec,
            )


        stats_report = {
            "stems_count": len(stats),
            "scene_duration": config.scene_duration,
            "statistics": stats,
        }
        stats_path = sample_dir / "volume_stats.json"
        stats_path.write_text(
            json.dumps(stats_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Loudness statistics saved: %s", stats_path)
        return stats_report

    volume_stats = _retry_stage("volume_statistics", _volume_statistics, max_retries=retries)


    def _enhanced_audio_postprocess():
        for i, stem_path in enumerate(stems_paths):
            obj = objects[i]
            target_lufs = float(obj.get("loudness_lufs", -23.0))

            logger.info("Stage 5: enhanced postprocessing for stem %d/%d: %s (target LUFS=%.1f)",
                       i + 1, len(stems_paths), stem_path.name, target_lufs)


            apply_noise_gate(stem_path)


            trimmed_sec = detect_and_trim_onset(stem_path, max_silence_sec=2.0)
            if trimmed_sec > 0:
                logger.info("  Trimmed %.2f seconds of leading silence", trimmed_sec)


            correction = normalize_to_target_lufs(stem_path, target_lufs=target_lufs)
            logger.info("  LUFS correction: %+.1f dB", correction)


            apply_fade_envelope(stem_path, fade_in_ms=30.0, fade_out_ms=80.0)

    _retry_stage("enhanced_audio_postprocess", _enhanced_audio_postprocess, max_retries=retries)


    

    for obj in objects:
        label_lower = obj.get("label", "").lower()
        obj_type = obj.get("type", obj.get("sound_type", "")).lower()
        is_ambient = any(kw in label_lower or kw in obj_type for kw in AMBIENT_KEYWORDS)
        if is_ambient:
            onset = obj.get("onset_t", 0.0)
            offset = obj.get("offset_t", config.scene_duration)
            if onset > 0.0 or offset < config.scene_duration - 0.1:
                logger.warning(
                    "Ambient source %s (label=%s) has an invalid interval [%.1f, %.1f]; "
                    "forcing [0.0, %.1f]",
                    obj.get("id"), obj.get("label"), onset, offset, config.scene_duration,
                )
                obj["onset_t"] = 0.0
                obj["offset_t"] = config.scene_duration


    
    
    def _volume_decision():
        from .volume_agent import VolumeAgent
        from .audio_extract import measure_lufs as _measure_lufs


        measured: dict[str, float] = {}
        for i, stem_path in enumerate(stems_paths):
            obj = objects[i] if i < len(objects) else {}
            obj_id = obj.get("id", f"source_{i+1:03d}")
            try:
                lufs_val = _measure_lufs(stem_path)
            except Exception as ex:
                logger.warning("  %s LUFS measurement failed: %s", obj_id, ex)
                continue
            measured[obj_id] = lufs_val
            logger.info("  %s measured LUFS: %.1f", obj_id, lufs_val)


        if volume_stats and volume_stats.get("statistics"):
            stats_by_id = {s["object_id"]: s for s in volume_stats["statistics"]}
            for obj in objects:
                obj_id = obj.get("id", "")
                if obj_id in stats_by_id:
                    st = stats_by_id[obj_id]
                    obj["_audio_stats"] = {
                        "rms_db": st["rms_db"],
                        "peak_dbfs": st["peak_dbfs"],
                        "spectral_balance": st["spectral_balance"],
                    }


        env = scene_dict.get("environment", {}) or {}
        scene_desc = (
            env.get("description")
            or scene_dict.get("description")
            or video_prompt_text
            or ""
        )


        agent = VolumeAgent()
        try:
            try:
                result = agent.decide_volumes(
                    scene_description=scene_desc,
                    objects=objects,
                    measured_lufs=measured,
                    scene_duration=config.scene_duration,
                )
            except Exception as ex:
                logger.warning(
                    "VolumeAgent LLM decision failed; retaining original Director loudness_lufs: %s", ex,
                )
                return {"decisions": [], "raw_response": "", "error": str(ex)}


            decisions_map = {
                d["object_id"]: d for d in result.get("decisions", [])
            }
            for obj in objects:
                obj_id = obj.get("id", "")
                if obj_id in decisions_map:
                    old_lufs = obj.get("loudness_lufs", -23.0)
                    new_lufs = decisions_map[obj_id]["target_lufs"]
                    obj["loudness_lufs"] = new_lufs
                    logger.info(
                        "  %s LUFS: %.1f → %.1f (%s)",
                        obj_id, old_lufs, new_lufs,
                        decisions_map[obj_id].get("reason", ""),
                    )


            (sample_dir / "volume_decision.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )


            if result.get("decisions"):
                decisions_sorted = sorted(result["decisions"], key=lambda d: d["target_lufs"], reverse=True)
                logger.info("  +-- Loudness decision summary -------------------")
                for d in decisions_sorted:
                    obj_label = next((o.get("label", "") for o in objects if o.get("id") == d["object_id"]), "")
                    logger.info("  │ %s (%s): %.1f LUFS | %s",
                               d["object_id"], obj_label, d["target_lufs"], d.get("reason", "")[:50])
                loudest = decisions_sorted[0]["target_lufs"]
                quietest = decisions_sorted[-1]["target_lufs"]
                logger.info("  | Dynamic range: %.1f dB (loudest %.1f -> quietest %.1f)",
                           loudest - quietest, loudest, quietest)
                logger.info("  └────────────────────────────────────────")

            return result
        finally:
            agent.close()

    _retry_stage("volume_decision", _volume_decision, max_retries=retries)


    for i, obj in enumerate(objects):
        if i < len(stems_paths):
            rel_path = str(stems_paths[i].relative_to(sample_dir))
            obj["wav_path"] = rel_path
        
        obj["start_time"] = obj.get("onset_t", 0.0)
        obj["end_time"] = obj.get("offset_t", None)


    _ensure_role_field(scene_dict)


    convert_lufs_to_gain_db(scene_dict)


    normalize_scene_for_render(scene_dict)

    scene_json_path = sample_dir / "scene.json"
    scene_json_path.write_text(
        json.dumps(scene_dict, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _log_stage("write_scene", "success", str(scene_json_path))


    foa_path = sample_dir / "audio_foa.wav"

    def _render_foa():
        return render_scene_to_foa(scene_json_path, foa_path)

    _retry_stage("render_foa", _render_foa, max_retries=retries)


    binaural_path = sample_dir / "audio_binaural.wav"

    def _render_binaural():
        return render_foa_to_binaural(foa_path, binaural_path)

    _retry_stage("render_binaural", _render_binaural, max_retries=retries)


    video_path = sample_dir / "video.mp4"
    if not config.skip_video:
        def _gen_video():
            return generate_video(video_prompt_text, video_path, max_retries=retries)

        _retry_stage("video_generation", _gen_video, max_retries=retries)
    else:
        _log_stage("video_generation", "skipped", "skip_video=True")


    
    corrected_scene_path = sample_dir / "corrected_scene.json"
    if video_path.is_file():
        def _correct_metadata():
            return run_metadata_correction(
                sample_dir,
                corrected_scene_path,
                camera_mode=config.camera_mode,
            )

        _retry_stage("metadata_correction", _correct_metadata, max_retries=retries)


        def _render_foa_corrected():
            return render_scene_to_foa(corrected_scene_path, foa_path)

        _retry_stage("render_foa_corrected", _render_foa_corrected, max_retries=retries)


        def _render_binaural_corrected():
            return render_foa_to_binaural(foa_path, binaural_path)

        _retry_stage("render_binaural_corrected", _render_binaural_corrected, max_retries=retries)


        if config.enable_offscreen_agent:
            def _off_screen_review():
                from .off_screen_agent import OffScreenAgent

                agent = OffScreenAgent()
                try:
                    report = agent.review_offscreen_sources(
                        corrected_scene_path=corrected_scene_path,
                        original_scene_path=scene_json_path,
                        video_description=video_prompt_text,
                    )
                    if report.total_offscreen_objects > 0:
                        agent.apply_adjustments(
                            corrected_scene_path,
                            report,
                            output_path=corrected_scene_path,
                        )
                    
                    (sample_dir / "off_screen_agent_report.json").write_text(
                        json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    return report
                finally:
                    agent.close()

            _retry_stage("off_screen_review", _off_screen_review, max_retries=retries)
        else:
            _log_stage("off_screen_review", "skipped", "enable_offscreen_agent=False")
    else:
        _log_stage("metadata_correction", "skipped", "video.mp4 not found")

    # ─── Stage 13: metadata.json ─────────────────────────────────────
    
    metadata = _build_metadata(
        sample_id=sample_id,
        objects=objects,
        scene_duration=config.scene_duration,
        allow_camera_rotation=config.allow_camera_rotation,
        camera_mode=config.camera_mode,
        n_foreground=config.n_foreground,
        include_ambient=config.include_ambient,
    )
    (sample_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    total_duration = time.time() - t_start
    logger.info("=" * 60)
    logger.info("Sample %s generated (%.1fs)", sample_id, total_duration)
    logger.info("=" * 60)

    return sample_dir


def _build_metadata(
    sample_id: str,
    objects: list[dict],
    scene_duration: float,
    allow_camera_rotation: bool,
    camera_mode: CameraMode,
    n_foreground: int,
    include_ambient: bool,
) -> dict:
    """OmniEcho processing component."""
    return {
        "sample_id": sample_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "OmniEchoBench-QA",
        "director_model": os.environ.get(
            "OPENAI_MODEL", "OmniEcho"
        ),
        "video_model": os.environ.get(
            "VIDEO_MODEL", "OmniEcho"
        ),
        "scene_duration": scene_duration,
        "camera_mode": camera_mode.value,
        "camera_rotation": allow_camera_rotation,
        "prompt_strategy": "geometry_aligned",
        "n_foreground": n_foreground,
        "include_ambient": include_ambient,
        "objects": [
            {
                "id": obj.get("id", f"source_{i+1:03d}"),
                "type": obj.get("type", obj.get("sound_type", "")),
                "sub_video": f"sub_videos/{obj.get('id', f'source_{i+1:03d}')}.mp4",
            }
            for i, obj in enumerate(objects)
        ],
    }


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

def run_batch(
    config: PipelineConfig,
    n_samples: int = 10,
) -> Path:
    """OmniEcho processing component."""
    logger.info("=" * 60)
    logger.info(
        "Batch generation (camera_mode=%s): n=%d",
        config.camera_mode.value, n_samples,
    )
    logger.info("Output directory: %s", config.output_root)
    logger.info("=" * 60)

    config.output_root.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    results: list[dict] = []
    previous_scenes: list[str] = []

    is_static = config.camera_mode == CameraMode.STATIC
    sample_prefix = "omniecho_qa"

    for i in range(1, n_samples + 1):
        sample_id = f"{sample_prefix}_{i:03d}"
        logger.info("─" * 40)
        logger.info("Batch progress: %d/%d (%s)", i, n_samples, sample_id)
        logger.info("─" * 40)

        try:
            sample_dir = generate_single_sample(
                config=config,
                sample_id=sample_id,
                previous_scenes=list(previous_scenes),
            )
            results.append({
                "sample_id": sample_id,
                "status": "success",
                "output_dir": str(sample_dir),
            })


            scene_path = sample_dir / "scene.json"
            if scene_path.exists():
                try:
                    scene_data = json.loads(scene_path.read_text(encoding="utf-8"))
                    desc = scene_data.get("environment", {}).get("description", "")
                    if desc:
                        previous_scenes.append(desc)
                except Exception as ex:
                    logger.warning("Failed to read scene.json for %s: %s", sample_id, ex)
        except Exception as e:
            logger.error("Sample %s failed: %s", sample_id, e)
            results.append({
                "sample_id": sample_id,
                "status": "failed",
                "error": str(e),
            })


    total_duration = time.time() - t_start
    index = {
        "pipeline": "OmniEchoBench-QA",
        "camera_mode": config.camera_mode.value,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_duration_sec": total_duration,
        "n_samples": n_samples,
        "prompt_strategy": "geometry_aligned",
        "samples": results,
        "summary": {
            "success": sum(1 for r in results if r["status"] == "success"),
            "failed": sum(1 for r in results if r["status"] == "failed"),
        },
    }
    index_path = config.output_root / "index.json"
    index_path.write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info("=" * 60)
    logger.info(
        "Batch complete: succeeded=%d, failed=%d, elapsed=%.1fs",
        index["summary"]["success"],
        index["summary"]["failed"],
        total_duration,
    )
    logger.info("index.json → %s", index_path)
    logger.info("=" * 60)

    return config.output_root
