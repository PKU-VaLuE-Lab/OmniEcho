#!/usr/bin/env python3

#   python 03b_make_spatial_voice_vln_noise.py \
#       --in_jsonl  <OmniEchoBench-Nav>/spatial_voice_vln_full.jsonl \
#       --clean_dir <OmniEchoBench-Nav>/spatial_audio \
#       --out_audio_dir <OmniEchoBench-Nav>/spatial_audio_noise \
#       --out_jsonl <OmniEchoBench-Nav>/spatial_voice_vln_full_noise.jsonl
"""OmniEcho processing component."""
from __future__ import annotations

import argparse
import json
import os
from settings import DATA_ROOT as TRAJECTORY_ROOT, NAV_ROOT as OUTPUT_ROOT, MODEL_ROOT, API_BASE_URL as SERVICE_URL, API_MODEL
import sys
import zlib

import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geometry as G  

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
NAV_ROOT = OUTPUT_ROOT
DEFAULT_IN_JSONL = os.path.join(NAV_ROOT, "spatial_voice_vln_full.jsonl")
DEFAULT_CLEAN_DIR = os.path.join(NAV_ROOT, "spatial_audio")
DEFAULT_OUT_AUDIO = os.path.join(NAV_ROOT, "spatial_audio_noise")
DEFAULT_OUT_JSONL = os.path.join(NAV_ROOT, "spatial_voice_vln_full_noise.jsonl")

EPS = 1e-12


# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def pink_noise(n, rng):
    """OmniEcho processing component."""
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.arange(spec.size, dtype=np.float64)
    f[0] = 1.0                       
    spec = spec / np.sqrt(f)
    y = np.fft.irfft(spec, n=n)
    return y / (np.std(y) + EPS)


def white_noise(n, rng):
    """OmniEcho processing component."""
    y = rng.standard_normal(n)
    return y / (np.std(y) + EPS)


NOISE_FUNCS = {"pink": pink_noise, "white": white_noise}


# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def azel_to_unit(az_deg, el_deg):
    """OmniEcho processing component."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    ce = np.cos(el)
    return np.array([ce * np.cos(az), ce * np.sin(az), np.sin(el)], dtype=np.float64)


def sample_noise_directions(k, sig_unit, rng, min_sep_deg, el_range_deg, max_tries=200):
    """OmniEcho processing component."""
    cos_thr = np.cos(np.radians(min_sep_deg))
    out = []
    for _ in range(k):
        for _ in range(max_tries):
            az = float(rng.uniform(-180.0, 180.0))
            el = float(rng.uniform(-el_range_deg, el_range_deg))
            u = azel_to_unit(az, el)
            if float(np.dot(u, sig_unit)) <= cos_thr:      
                out.append((az, el, u))
                break
        else:
            
            u = -sig_unit
            az = float(np.degrees(np.arctan2(u[1], u[0])))
            el = float(np.degrees(np.arcsin(np.clip(u[2], -1.0, 1.0))))
            out.append((az, el, u))
    return out


def encode_plane_wave(mono, unit):
    """OmniEcho processing component."""
    return np.stack([mono, mono * unit[1], mono * unit[2], mono * unit[0]], axis=0)


# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def load_clean_foa(path):
    """OmniEcho processing component."""
    sr, data = wavfile.read(path)
    data = np.asarray(data)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float64) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float64)
    if data.ndim == 1:                 
        data = data[:, None]
    return sr, data.T                  # [n, ch] -> [ch, n]


def add_directional_noise(clean_foa, rng, snr_db, noise_fn, k, min_sep_deg, el_range_deg,
                          sig_unit, noise_floor_rms=1e-3):
    """OmniEcho processing component."""
    n = clean_foa.shape[1]
    dirs = sample_noise_directions(k, sig_unit, rng, min_sep_deg, el_range_deg)

    noise_foa = np.zeros_like(clean_foa)
    for (_, _, u) in dirs:
        mono = noise_fn(n, rng)
        noise_foa += encode_plane_wave(mono, u)


    p_sig = float(np.mean(clean_foa[0] ** 2))
    p_noise = float(np.mean(noise_foa[0] ** 2))
    if p_noise < EPS:
        gain = 0.0
        eff_snr = float("inf")
    elif p_sig < EPS:
        
        gain = noise_floor_rms / np.sqrt(p_noise)
        eff_snr = None
    else:
        target_pn = p_sig / (10.0 ** (snr_db / 10.0))
        gain = float(np.sqrt(target_pn / p_noise))
        eff_snr = float(snr_db)

    noisy = clean_foa + gain * noise_foa
    sources = [{"azimuth_deg": round(az, 3), "elevation_deg": round(el, 3)}
               for (az, el, _) in dirs]
    return noisy, sources, gain, eff_snr


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_jsonl", default=DEFAULT_IN_JSONL, help="Stage 3 spatial_voice_vln_full.jsonl")
    ap.add_argument("--clean_dir", default=DEFAULT_CLEAN_DIR, help="Clean FOA WAV directory (spatial_audio)")
    ap.add_argument("--out_audio_dir", default=DEFAULT_OUT_AUDIO)
    ap.add_argument("--out_jsonl", default=DEFAULT_OUT_JSONL)
    ap.add_argument("--scene", default="all", help='Process a specific scene_id, or "all"')
    ap.add_argument("--max_samples", type=int, default=None, help="Maximum samples for debugging")
    ap.add_argument("--dry_run", action="store_true", help="Compute and write JSONL without writing WAV files")
    ap.add_argument("--snr_db_set", default="20,15,10,5,0",
                    help="Randomly select one SNR value in dB per sample")
    ap.add_argument("--min_noise_sources", type=int, default=1)
    ap.add_argument("--max_noise_sources", type=int, default=3, help="Maximum directional noise sources per sample")
    ap.add_argument("--min_sep_deg", type=float, default=30.0, help="Minimum angular separation between noise and signal directions")
    ap.add_argument("--el_range_deg", type=float, default=45.0, help="Noise elevation range in degrees (+/- value)")
    ap.add_argument("--noise_type", default="pink", choices=list(NOISE_FUNCS.keys()))
    ap.add_argument("--seed", type=int, default=0, help="Global seed base, combined with each filename to derive its sample seed")
    ap.add_argument("--clip", action="store_true",
                    help="Clip noisy FOA to [-1,1]; disabled by default to preserve linear amplitude and distance cues")
    args = ap.parse_args()

    snr_set = [float(x) for x in args.snr_db_set.split(",") if x.strip() != ""]
    assert snr_set, "--snr_db_set must not be empty"
    noise_fn = NOISE_FUNCS[args.noise_type]

    if not args.dry_run:
        os.makedirs(args.out_audio_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)), exist_ok=True)


    done = set()
    n_written = 0
    if os.path.exists(args.out_jsonl):
        with open(args.out_jsonl) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add(os.path.basename(r["foa_audio_path"]))
                    n_written += 1
                except Exception:
                    continue
        print(f"[resume] Found {n_written} records; skipping them")

    fout = open(args.out_jsonl, "a")
    n_skip_scene = n_skip_noclean = n_err = 0
    stop_all = False


    total = sum(1 for _ in open(args.in_jsonl))
    pbar = tqdm(total=total, desc="samples", unit="smp", dynamic_ncols=True)
    with open(args.in_jsonl) as fin:
        for line in fin:
            pbar.update(1)
            if stop_all:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            if args.scene != "all" and rec.get("scene_id") != args.scene:
                n_skip_scene += 1
                continue

            foa_name = os.path.basename(rec["foa_audio_path"])
            if foa_name in done:
                continue

            clean_path = os.path.join(args.clean_dir, foa_name)
            if not os.path.exists(clean_path):
                n_skip_noclean += 1
                continue

            try:
                
                seed = zlib.crc32(foa_name.encode("utf-8")) ^ (args.seed & 0xFFFFFFFF)
                rng = np.random.default_rng(seed)

                snr_db = float(rng.choice(snr_set))
                k = int(rng.integers(args.min_noise_sources, args.max_noise_sources + 1))
                sig_unit = azel_to_unit(rec.get("azimuth_deg", 0.0), rec.get("elevation_deg", 0.0))

                sr, clean_foa = load_clean_foa(clean_path)
                noisy, sources, gain, eff_snr = add_directional_noise(
                    clean_foa, rng, snr_db, noise_fn, k,
                    args.min_sep_deg, args.el_range_deg, sig_unit)

                if args.clip:
                    noisy = np.clip(noisy, -1.0, 1.0)

                out_path = os.path.join(args.out_audio_dir, foa_name)
                if not args.dry_run:
                    G.write_foa_wav(out_path, noisy, sr)

                out_rec = dict(rec)
                out_rec["foa_audio_path"] = out_path
                out_rec["clean_foa_audio_path"] = clean_path
                out_rec["noise_type"] = args.noise_type
                out_rec["noise_snr_db"] = eff_snr
                out_rec["noise_num_sources"] = k
                out_rec["noise_sources"] = sources
                out_rec["noise_gain"] = round(float(gain), 6)
                out_rec["noise_min_sep_deg"] = args.min_sep_deg
                out_rec["noise_seed"] = int(seed)
                out_rec["noise_clipped"] = bool(args.clip)

                fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                n_written += 1
                if args.max_samples and n_written >= args.max_samples:
                    stop_all = True
            except Exception as e:
                n_err += 1
                tqdm.write(f"[err] {foa_name}: {type(e).__name__}: {e}")

            if n_written % 200 == 0:
                fout.flush()
            pbar.set_postfix(written=n_written, err=n_err)

    pbar.close()
    fout.close()

    print(f"\n[done] Samples written: {n_written}{' (dry_run, no WAV files written)' if args.dry_run else ''}")
    print(f"[done] Skipped other scenes: {n_skip_scene}  missing clean WAV: {n_skip_noclean}  errors: {n_err}")
    print(f"[done] jsonl -> {args.out_jsonl}")
    if not args.dry_run:
        print(f"[done] Noisy FOA WAV files -> {args.out_audio_dir}")


if __name__ == "__main__":
    main()
