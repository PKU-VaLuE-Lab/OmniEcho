#!/usr/bin/env python3


#   python 02_batch_tts.py --stage all

"""OmniEcho processing component."""
from __future__ import annotations

import argparse
import json
import os
from settings import DATA_ROOT as TRAJECTORY_ROOT, NAV_ROOT as OUTPUT_ROOT, MODEL_ROOT, API_BASE_URL as SERVICE_URL, API_MODEL
import re
import sys
import time
import traceback


_PT = MODEL_ROOT
QWEN_TTS_DIR = os.environ.get("OMNIECHO_TTS_MODEL", os.path.join(MODEL_ROOT, "tts"))
STABLE_AUDIO_DIR = os.environ.get("OMNIECHO_SOUND_MODEL", os.path.join(MODEL_ROOT, "sound"))
STABLE_AUDIO_SRC = os.environ.get("OMNIECHO_SOUND_SOURCE", "/path/to/stable-audio-source")

NAV_ROOT = OUTPUT_ROOT
DEFAULT_IN = os.path.join(NAV_ROOT, "audio_scripts.jsonl")
DEFAULT_OUT = os.path.join(NAV_ROOT, "audio_scripts_with_audio.jsonl")
DEFAULT_VOICE_DIR = os.path.join(NAV_ROOT, "OmniEchoBench-Nav_voice")


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# =============================================================================

# =============================================================================

def wav_name(rec):
    """OmniEchoBench-Nav_<scene>_ep<ep>_<setting>.wav."""
    setting = rec.get("setting", "125cm_0deg")
    return f"{rec['dataset']}_{rec['scene_id']}_ep{int(rec['episode_id'])}_{setting}.wav"


def out_wav_path(voice_dir, rec):
    return os.path.join(voice_dir, wav_name(rec))


def rel_audio_path(voice_dir, rec):
    """OmniEcho processing component."""
    return os.path.relpath(out_wav_path(voice_dir, rec), NAV_ROOT)


def load_scripts(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def wav_ok(path):
    return os.path.exists(path) and os.path.getsize(path) > 1024


def stable_seed(rec, base=0):
    """OmniEcho processing component."""
    key = f"{rec['dataset']}|{rec['scene_id']}|{rec['episode_id']}|{rec['category']}|{base}"
    h = 0
    for ch in key:
        h = (h * 131 + ord(ch)) & 0x7FFFFFFF
    return h


# =============================================================================

# =============================================================================


_FEMALE = ["Serena", "Vivian", "Sohee", "Ono_Anna"]
_MALE = ["Ryan", "Aiden", "Uncle_Fu"]


def pick_speaker(voice_style: str, idx: int) -> str:
    """OmniEcho processing component."""
    s = (voice_style or "").lower()
    female = bool(re.search(r"\b(female|woman|women|girl|lady|she|her|mother|mom|grandma|"
                            r"granny|aunt|daughter|sister|feminine)\b", s))
    male = bool(re.search(r"\b(male|man|men|boy|guy|he|his|father|dad|grandpa|grandfather|"
                          r"uncle|brother|son|masculine)\b", s))
    elderly = bool(re.search(r"\b(elderly|old|older|aged|senior|grandma|granny|grandpa|"
                             r"grandfather|seasoned|mature)\b", s))
    emotional = bool(re.search(r"\b(cry|crying|sob|tremb|anxious|pleading|desperate|"
                               r"emotional|scared|frightened|panic|worried)\b", s))
    if female and not male:
        if emotional:
            return "Sohee"
        return _FEMALE[idx % 2]              
    if male and not female:
        if elderly:
            return "Uncle_Fu"
        return _MALE[idx % 2]                
    
    pool = ["Ryan", "Serena", "Aiden", "Vivian"]
    return pool[idx % len(pool)]


def run_tts(records, voice_dir, overwrite, batch_size, max_new_tokens, model_dir):
    import torch
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel

    todo = []
    for i, rec in enumerate(records):
        p = out_wav_path(voice_dir, rec)
        if not overwrite and wav_ok(p):
            continue
        todo.append((rec, p, pick_speaker(rec.get("voice_style", ""), i)))
    print(f"[tts] Pending synthesis: {len(todo)} / {len(records)} records (cat1)")
    if not todo:
        return {}

    print(f"[tts] Loading Qwen3-TTS: {model_dir}")
    model = Qwen3TTSModel.from_pretrained(
        model_dir, device_map="cuda:0", dtype=torch.bfloat16,
        attn_implementation="sdpa",       
    )
    try:
        
        speakers = {s.lower() for s in model.get_supported_speakers()}
        print(f"[tts] Available speakers: {sorted(speakers)}")
    except Exception:  # noqa: BLE001
        speakers = None

    done = {}
    t0 = time.time()
    for b in range(0, len(todo), batch_size):
        chunk = todo[b:b + batch_size]
        texts = [(r.get("sound_content") or "").strip() or "Hello." for r, _, _ in chunk]
        spks = [s if (speakers is None or s.lower() in speakers) else "Ryan"
                for _, _, s in chunk]
        insts = [(r.get("voice_style") or "") for r, _, _ in chunk]
        try:
            wavs, sr = model.generate_custom_voice(
                text=texts, language=["English"] * len(texts),
                speaker=spks, instruct=insts, max_new_tokens=max_new_tokens,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [tts][batch {b}] Batch failed; retrying individually: {type(e).__name__}: {e}")
            wavs, sr = [], None
            for (r, _, _), t, sp, ins in zip(chunk, texts, spks, insts):
                try:
                    w, sr = model.generate_custom_voice(
                        text=t, language="English", speaker=sp, instruct=ins,
                        max_new_tokens=max_new_tokens)
                    wavs.append(w[0])
                except Exception as e2:  # noqa: BLE001
                    print(f"    [tts][fail] {r['scene_id']}/ep{r['episode_id']}: {e2}")
                    wavs.append(None)
        for (rec, path, spk), wav in zip(chunk, wavs):
            if wav is None:
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            sf.write(path, wav, sr)
            done[path] = {"audio_sr": int(sr), "audio_model": "qwen3-tts-customvoice",
                          "audio_speaker": spk, "audio_seconds": round(len(wav) / float(sr), 3)}
        n = min(b + batch_size, len(todo))
        print(f"[tts] {n}/{len(todo)}  ({(time.time()-t0):.0f}s)")

    del model
    try:
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return done


# =============================================================================

# =============================================================================

def load_stable_audio(stable_dir, stable_src, device="cuda", model_half=True):
    if stable_src not in sys.path:
        sys.path.insert(0, stable_src)
    from stable_audio_3 import StableAudioModel
    from stable_audio_3.loading_utils import load_diffusion_cond

    with open(os.path.join(stable_dir, "model_config.json")) as f:
        cfg = json.load(f)
    
    t5_local = os.path.join(stable_dir, "t5gemma-b-b-ul2")
    for c in cfg["model"]["conditioning"]["configs"]:
        if c.get("type") == "t5gemma":
            c.setdefault("config", {})
            c["config"]["model_path"] = t5_local
            c["config"].pop("repo_id", None)
            c["config"].pop("subfolder", None)
    ckpt = os.path.join(stable_dir, "model.safetensors")
    print(f"[sfx] Loading Stable Audio 3: {ckpt}\n[sfx] t5gemma: {t5_local}")
    model = load_diffusion_cond(cfg, ckpt, device=device, model_half=model_half)
    model.use_lora = False
    model.lora_names = []
    return StableAudioModel(model, cfg, device, model_half)


def run_sfx(records, voice_dir, overwrite, duration, steps, cfg_scale, stable_dir, stable_src):
    import torch
    import soundfile as sf

    todo = []
    for rec in records:
        p = out_wav_path(voice_dir, rec)
        if not overwrite and wav_ok(p):
            continue
        todo.append((rec, p))
    print(f"[sfx] Pending synthesis: {len(todo)} / {len(records)} records (cat2)")
    if not todo:
        return {}

    sam = load_stable_audio(stable_dir, stable_src)
    sr = int(sam.model.sample_rate)
    done = {}
    t0 = time.time()
    for k, (rec, path) in enumerate(todo, 1):
        
        prompt = (rec.get("sound_content") or "").strip()
        vs = (rec.get("voice_style") or "").strip()
        if vs and vs.lower() not in prompt.lower():
            prompt = f"{prompt} {vs}"
        try:
            with torch.no_grad():
                audio = sam.generate(prompt=prompt, duration=float(duration),
                                     steps=steps, cfg_scale=cfg_scale,
                                     seed=stable_seed(rec))
            wav = audio[0].to(torch.float32).cpu().clamp_(-1, 1).numpy()  # [C, N]
            os.makedirs(os.path.dirname(path), exist_ok=True)
            sf.write(path, wav.T, sr)                                     
            done[path] = {"audio_sr": sr, "audio_model": "stable-audio-3-medium",
                          "audio_speaker": "", "audio_seconds": round(wav.shape[1] / float(sr), 3)}
        except Exception as e:  # noqa: BLE001
            print(f"  [sfx][fail] {rec['scene_id']}/ep{rec['episode_id']}: {e}")
            traceback.print_exc()
        if k % 20 == 0 or k == len(todo):
            print(f"[sfx] {k}/{len(todo)}  ({(time.time()-t0):.0f}s)")

    del sam
    try:
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return done


# =============================================================================

# =============================================================================

def write_manifest(records, voice_dir, meta_by_path, manifest_path):
    """OmniEcho processing component."""
    n = 0
    with open(manifest_path, "w", encoding="utf-8") as f:
        for rec in records:
            path = out_wav_path(voice_dir, rec)
            if not wav_ok(path):
                continue
            r = dict(rec)
            r["audio_path"] = rel_audio_path(voice_dir, rec)
            extra = meta_by_path.get(path)
            if extra:
                r.update(extra)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"[manifest] Wrote {n} rows -> {manifest_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    
    ap.add_argument("--in", dest="in_path", default=DEFAULT_IN, help="Stage 1 audio_scripts.jsonl")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Manifest JSONL with populated audio_path fields")
    ap.add_argument("--voice_dir", default=DEFAULT_VOICE_DIR, help="Output WAV directory")
    ap.add_argument("--stage", default="all", choices=["all", "tts", "sfx"],
                    help="all: both categories; tts: speech only (cat1); sfx: environmental sounds only (cat2)")
    ap.add_argument("--overwrite", action="store_true", help="Regenerate existing WAV files")
    ap.add_argument("--max", type=int, default=None, help="Maximum records per category for debugging")


    ap.add_argument("--shard", type=int, default=0, help="Shard index in [0, num_shards)")
    ap.add_argument("--num-shards", dest="num_shards", type=int, default=1, help="Total shards for multi-GPU processing")
    # Qwen-TTS
    ap.add_argument("--model", default=QWEN_TTS_DIR, help="Qwen3-TTS model directory")
    ap.add_argument("--tts_batch", type=int, default=8)
    ap.add_argument("--tts_max_new_tokens", type=int, default=2048)
    # Stable Audio (cat2)
    ap.add_argument("--stable_audio_dir", default=STABLE_AUDIO_DIR, help="Stable Audio checkpoint and configuration directory")
    ap.add_argument("--stable_audio_src", default=STABLE_AUDIO_SRC, help="stable_audio_3 source directory")
    ap.add_argument("--sfx_duration", type=float, default=10.0, help="Environmental sound duration in seconds")
    ap.add_argument("--sfx_steps", type=int, default=8)
    ap.add_argument("--sfx_cfg", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.voice_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    rows = load_scripts(args.in_path)
    cat1 = [r for r in rows if int(r.get("category", 0)) == 1]
    cat2 = [r for r in rows if int(r.get("category", 0)) == 2]
    if args.max:
        cat1, cat2 = cat1[: args.max], cat2[: args.max]
    sharded = args.num_shards > 1
    if sharded:
        assert 0 <= args.shard < args.num_shards, "--shard must be in [0, num_shards)"
        cat1 = [r for i, r in enumerate(cat1) if i % args.num_shards == args.shard]
        cat2 = [r for i, r in enumerate(cat2) if i % args.num_shards == args.shard]
        print(f"[plan] Shard {args.shard}/{args.num_shards}")
    print(f"[plan] Script rows: {len(rows)} (cat1={len(cat1)}, cat2={len(cat2)}), stage={args.stage}")
    print(f"[plan] wav -> {args.voice_dir}")

    meta = {}
    if args.stage in ("all", "tts") and cat1:
        meta.update(run_tts(cat1, args.voice_dir, args.overwrite,
                            args.tts_batch, args.tts_max_new_tokens, args.model))
    if args.stage in ("all", "sfx") and cat2:
        meta.update(run_sfx(cat2, args.voice_dir, args.overwrite,
                            args.sfx_duration, args.sfx_steps, args.sfx_cfg,
                            args.stable_audio_dir, args.stable_audio_src))


    
    used = cat1 + cat2 if args.stage == "all" else (cat1 if args.stage == "tts" else cat2)
    scope = used if args.max else rows
    write_manifest(scope, args.voice_dir, meta, args.out)
    print("[done]")


if __name__ == "__main__":
    main()
