"""OmniEcho processing component."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np
from scipy.io import wavfile

logger = logging.getLogger(__name__)


def extract_audio_from_video(video_path: Path, output_wav: Path) -> Path:
    """OmniEcho processing component."""
    video_path = Path(video_path)
    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    
    if not video_path.exists():
        raise RuntimeError(f"Video file does not exist: {video_path}")
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "48000",
        "-ac", "1",
        str(output_wav),
    ]
    
    logger.info("Extracting audio: %s -> %s", video_path.name, output_wav.name)
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
    )
    
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg audio extraction failed (code={result.returncode}): {result.stderr[-500:]}"
        )
    
    if not output_wav.exists() or output_wav.stat().st_size == 0:
        raise RuntimeError(f"FFmpeg output is empty or missing: {output_wav}")
    
    logger.info("Audio extracted: %s (%.1f KB)", output_wav.name, output_wav.stat().st_size / 1024)
    return output_wav


def normalize_stem(wav_path: Path, target_peak_db: float = -3.0) -> Path:
    """OmniEcho processing component."""
    wav_path = Path(wav_path)
    sr, data = wavfile.read(wav_path)


    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.float32:
        audio = data
    else:
        audio = data.astype(np.float32)


    if audio.ndim > 1:
        audio = audio[:, 0]
    
    # Peak normalize
    peak = np.max(np.abs(audio))
    if peak > 1e-8:
        target_linear = 10 ** (target_peak_db / 20.0)
        audio = audio * (target_linear / peak)


    audio_int16 = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
    wavfile.write(wav_path, 48000, audio_int16)
    
    logger.info("Normalization complete: %s (peak -> %.1f dBFS)", wav_path.name, target_peak_db)
    return wav_path


def measure_lufs(wav_path: Path) -> float:
    """OmniEcho processing component."""
    import pyloudnorm as pyln

    wav_path = Path(wav_path)
    sr, data = wavfile.read(wav_path)


    if data.dtype == np.int16:
        audio = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float64) / 2147483648.0
    else:
        audio = data.astype(np.float64)

    if audio.ndim > 1:
        audio = audio[:, 0]

    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio)


    if loudness == float('-inf') or np.isnan(loudness):
        return -70.0

    return float(loudness)


def normalize_to_target_lufs(wav_path: Path, target_lufs: float = -23.0) -> float:
    """OmniEcho processing component."""
    import pyloudnorm as pyln

    wav_path = Path(wav_path)
    sr, data = wavfile.read(wav_path)

    if data.dtype == np.int16:
        audio = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float64) / 2147483648.0
    else:
        audio = data.astype(np.float64)

    if audio.ndim > 1:
        audio = audio[:, 0]


    min_samples = int(0.4 * sr)
    if len(audio) < min_samples:
        logger.warning("Audio too short (%.2fs < 0.4s); skipping LUFS normalization: %s",
                      len(audio) / sr, wav_path.name)
        return 0.0

    meter = pyln.Meter(sr)
    current_lufs = meter.integrated_loudness(audio)

    if current_lufs == float('-inf') or np.isnan(current_lufs):
        logger.warning("Silent file; skipping LUFS normalization: %s", wav_path.name)
        return 0.0

    correction_db = target_lufs - current_lufs


    if correction_db > 35.0:
        logger.warning("  %s gain exceeds limit (+%.1fdB > +35dB); clamping to +35dB",
                      wav_path.name, correction_db)
        correction_db = 35.0
    elif correction_db > 30.0:
        logger.warning("  %s needs high gain (+%.1fdB); the original stem may be low quality",
                      wav_path.name, correction_db)


    gain_linear = 10 ** (correction_db / 20.0)
    audio_corrected = audio * gain_linear


    ceiling_linear = 10 ** (-1.0 / 20.0)  # ≈ 0.891
    peak = np.max(np.abs(audio_corrected))
    if peak > ceiling_linear:
        
        logger.info("  LUFS normalization triggered the hard limiter: %s (peak=%.3f > ceiling=%.3f; clipping peaks)",
                   wav_path.name, peak, ceiling_linear)
        np.clip(audio_corrected, -ceiling_linear, ceiling_linear, out=audio_corrected)


    audio_int16 = np.clip(audio_corrected * 32768.0, -32768, 32767).astype(np.int16)
    wavfile.write(wav_path, sr, audio_int16)

    logger.info("LUFS normalization: %s (%.1f -> %.1f LUFS, correction %+.1f dB)",
               wav_path.name, current_lufs, target_lufs, correction_db)
    return correction_db


def apply_noise_gate(wav_path: Path, threshold_db: float = -50.0) -> Path:
    """OmniEcho processing component."""
    import noisereduce as nr

    wav_path = Path(wav_path)
    sr, data = wavfile.read(wav_path)

    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    else:
        audio = data.astype(np.float32)

    if audio.ndim > 1:
        audio = audio[:, 0]


    
    audio_denoised = nr.reduce_noise(
        y=audio,
        sr=sr,
        stationary=True,
        prop_decrease=0.6,
        n_fft=2048,
        hop_length=512,
    )


    audio_int16 = np.clip(audio_denoised * 32768.0, -32768, 32767).astype(np.int16)
    wavfile.write(wav_path, sr, audio_int16)

    logger.info("Noise gate complete: %s", wav_path.name)
    return wav_path


def detect_and_trim_onset(wav_path: Path, max_silence_sec: float = 1.0,
                          threshold_db: float = -40.0) -> float:
    """OmniEcho processing component."""
    wav_path = Path(wav_path)
    sr, data = wavfile.read(wav_path)

    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    else:
        audio = data.astype(np.float32)

    if audio.ndim > 1:
        audio = audio[:, 0]


    frame_size = int(0.01 * sr)  
    threshold_linear = 10 ** (threshold_db / 20.0)

    onset_sample = 0
    for i in range(0, len(audio) - frame_size, frame_size):
        frame = audio[i:i + frame_size]
        rms = np.sqrt(np.mean(frame ** 2))
        if rms > threshold_linear:
            onset_sample = max(0, i - int(0.05 * sr))  
            break

    onset_sec = onset_sample / sr

    if onset_sec > max_silence_sec:
        
        trim_start = onset_sample
        audio_trimmed = audio[trim_start:]
        audio_int16 = np.clip(audio_trimmed * 32768.0, -32768, 32767).astype(np.int16)
        wavfile.write(wav_path, sr, audio_int16)
        trimmed_sec = trim_start / sr
        logger.info("Onset trimming: %s (removed %.2f seconds of leading silence)", wav_path.name, trimmed_sec)
        return trimmed_sec

    logger.info("Onset detection: %s (%.2f seconds of leading silence; no trimming needed)", wav_path.name, onset_sec)
    return 0.0


def apply_fade_envelope(wav_path: Path, fade_in_ms: float = 30.0,
                        fade_out_ms: float = 80.0) -> Path:
    """OmniEcho processing component."""
    wav_path = Path(wav_path)
    sr, data = wavfile.read(wav_path)

    if data.dtype == np.int16:
        audio = data.astype(np.float64) / 32768.0
    else:
        audio = data.astype(np.float64)

    if audio.ndim > 1:
        audio = audio[:, 0]


    fade_in_samples = int(fade_in_ms * sr / 1000.0)
    if fade_in_samples > 0 and fade_in_samples < len(audio):
        fade_in = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, fade_in_samples)))
        audio[:fade_in_samples] *= fade_in


    fade_out_samples = int(fade_out_ms * sr / 1000.0)
    if fade_out_samples > 0 and fade_out_samples < len(audio):
        fade_out = 0.5 * (1 + np.cos(np.pi * np.linspace(0, 1, fade_out_samples)))
        audio[-fade_out_samples:] *= fade_out


    audio_int16 = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
    wavfile.write(wav_path, sr, audio_int16)

    logger.info("Fade envelope: %s (in=%.0fms, out=%.0fms)", wav_path.name, fade_in_ms, fade_out_ms)
    return wav_path
