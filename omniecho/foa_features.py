# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Extract raw N3D/ACN FOA features expected by the OmniEcho spatial encoder."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np


_ACN = {"W": 0, "Y": 1, "Z": 2, "X": 3}  # first-order ACN channel order
TARGET_SR = 16000
N_FFT = 400
HOP = 160
N_MELS = 128
_EPS = 1e-8

_MEL_FB = None  # lazy (n_mels, n_freq)

# Video containers whose muxed (>=4)-channel audio track carries FOA. Decoded via
# PyAV. merged_av FOA is rendered N3D (spaudiopy real SH), matching the wav path,
# so NO SN3D->N3D correction is applied here.
_VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}


def _load_foa_from_video(path: str) -> Optional[np.ndarray]:
    """Decode a video's muxed (>=4)-channel FOA track -> (4, num_samples) float64 @16k,
    ACN order [W, Y, Z, X], N3D (raw, no normalization change). Returns None on failure
    or if the stream has <4 channels. Never raises."""
    try:
        import logging

        import av

        av.logging.set_level(av.logging.PANIC)
        logging.getLogger("libav").setLevel(logging.CRITICAL)
        container = av.open(path)
        try:
            stream = container.streams.audio[0]
            n_ch = stream.channels or 0
            if n_ch < 4:
                return None
            # Planar float, keep the source channel layout (NO downmix), resample to 16k.
            resampler = av.AudioResampler(format="fltp", layout=stream.layout, rate=TARGET_SR)
            chans = [[] for _ in range(n_ch)]
            for packet in container.demux(audio=0):
                try:
                    frames = packet.decode()
                except av.error.InvalidDataError:
                    continue
                for frame in frames:
                    for resampled in resampler.resample(frame):
                        arr = resampled.to_ndarray().astype(np.float32)  # planar -> (ch, samples)
                        if arr.ndim == 1:
                            arr = arr.reshape(1, -1)
                        for c in range(min(n_ch, arr.shape[0])):
                            chans[c].append(arr[c])
        finally:
            container.close()
    except Exception:
        return None
    if any(len(c) == 0 for c in chans[:4]):
        return None
    audio = np.stack([np.concatenate(chans[c]) for c in range(4)], axis=0)  # (4, samples), ACN, N3D
    return np.ascontiguousarray(audio, dtype=np.float64)


def _mel_filterbank() -> np.ndarray:
    global _MEL_FB
    if _MEL_FB is None:
        import librosa

        _MEL_FB = librosa.filters.mel(sr=TARGET_SR, n_fft=N_FFT, n_mels=N_MELS).astype(np.float64)
    return _MEL_FB


def _stft(x: np.ndarray) -> np.ndarray:
    import librosa

    return librosa.stft(x, n_fft=N_FFT, hop_length=HOP, center=True)


def foa_num_channels(path: str) -> int:
    """Cheap channel-count probe (no full decode). wav/flac via soundfile.info,
    video containers via a lightweight PyAV probe. Returns 0 on failure."""
    try:
        if isinstance(path, str) and os.path.splitext(path)[1].lower() in _VIDEO_EXTS:
            import av

            container = av.open(path)
            try:
                if not container.streams.audio:
                    return 0
                return int(container.streams.audio[0].channels or 0)
            finally:
                container.close()
        import soundfile as sf

        return int(sf.info(path).channels)
    except Exception:
        return 0


def load_foa(path: str) -> Optional[np.ndarray]:
    """Read a >=4-channel FOA source -> (4, num_samples) float64 @16k, ACN order (assumed N3D).

    wav/flac go through soundfile; video containers (mp4 etc., whose muxed FOA track
    soundfile cannot decode) go through PyAV. Both branches return N3D raw (no
    SN3D->N3D correction), consistent with the encoder's pretraining features.
    """
    if isinstance(path, str) and os.path.splitext(path)[1].lower() in _VIDEO_EXTS:
        return _load_foa_from_video(path)
    try:
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32", always_2d=True)  # (samples, ch)
    except Exception:
        return None
    audio = data.T
    if audio.shape[0] < 4:
        return None
    audio = audio[:4]
    if sr != TARGET_SR:
        import librosa

        audio = librosa.resample(np.ascontiguousarray(audio), orig_sr=sr, target_sr=TARGET_SR, res_type="soxr_hq")
    return np.ascontiguousarray(audio, dtype=np.float64)


def compute_features(foa: np.ndarray) -> np.ndarray:
    """(4, num_samples) N3D FOA -> (5, n_mels, T) float32 fused feature map.

    T is the native STFT frame count (~100 Hz); the model side pads to the batch max.
    """
    W = _stft(foa[_ACN["W"]])
    X = _stft(foa[_ACN["X"]])
    Y = _stft(foa[_ACN["Y"]])
    Z = _stft(foa[_ACN["Z"]])

    # Active-intensity vector I = Re(conj(W) * [X, Y, Z]); energy density.
    Ix = np.real(np.conj(W) * X)
    Iy = np.real(np.conj(W) * Y)
    Iz = np.real(np.conj(W) * Z)
    energy = 0.5 * (np.abs(W) ** 2 + np.abs(X) ** 2 + np.abs(Y) ** 2 + np.abs(Z) ** 2)

    mel_fb = _mel_filterbank()  # (n_mels, n_freq)
    Ix_m = mel_fb @ Ix
    Iy_m = mel_fb @ Iy
    Iz_m = mel_fb @ Iz
    E_m = mel_fb @ energy
    powW_m = mel_fb @ (np.abs(W) ** 2)  # (n_mels, T) omni power for the log-mel channel

    I_norm = np.sqrt(Ix_m ** 2 + Iy_m ** 2 + Iz_m ** 2) + _EPS
    ix = Ix_m / I_norm
    iy = Iy_m / I_norm
    iz = Iz_m / I_norm
    diffuseness = np.clip(1.0 - (I_norm / (E_m + _EPS)), 0.0, 1.0)
    logmel_w = np.log(powW_m + 1e-6)  # (n_mels, T)

    feat = np.stack([logmel_w, ix, iy, iz, diffuseness], axis=0)  # (5, n_mels, T)
    return feat.astype(np.float32)


def foa_path_to_encoder_features(path: str) -> Optional[np.ndarray]:
    """FOA wav / muxed-FOA video path -> (5, 128, T) float32, or None if not a
    >=4-channel FOA source (or on any decode/compute error)."""
    foa = load_foa(path)
    if foa is None:
        return None
    try:
        return compute_features(foa)
    except Exception:
        return None
