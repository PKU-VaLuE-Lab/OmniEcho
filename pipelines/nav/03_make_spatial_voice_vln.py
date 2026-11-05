#!/usr/bin/env python3

#   python 03_make_spatial_voice_vln.py \
#       --scene all --episodes all --out_jsonl <OmniEchoBench-Nav>/spatial_voice_vln_full.jsonl
"""OmniEcho processing component."""
from __future__ import annotations

import argparse
import json
import os
from settings import DATA_ROOT as TRAJECTORY_ROOT, NAV_ROOT as OUTPUT_ROOT, MODEL_ROOT, API_BASE_URL as SERVICE_URL, API_MODEL
import sys

import numpy as np
from tqdm import tqdm


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geometry as G  # noqa: E402

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
DATA_ROOT = TRAJECTORY_ROOT
DATASET = "OmniEchoBench-Nav"
NAV_ROOT = OUTPUT_ROOT
AUDIO_SCRIPTS = os.path.join(NAV_ROOT, "audio_scripts_with_audio.jsonl")
DEFAULT_OUT_AUDIO = os.path.join(NAV_ROOT, "OmniEchoBench-Nav_spatial_voice")
DEFAULT_OUT_JSONL = os.path.join(NAV_ROOT, "spatial_voice_vln_full.jsonl")


HEIGHT, PITCH_1, PITCH_2 = G.HEIGHT, G.PITCH_1, G.PITCH_2
NUM_HISTORY = G.NUM_HISTORY

IMAGE_TOKEN = "<image>"
AUDIO_TOKEN = "<audio>"

NAV_PROMPT_TEMPLATE = (
    "You are an autonomous navigation robot that locates and moves toward a sound source. "
    "You perceive the environment through your recent egocentric RGB observations and a 4-channel "
    "first-order ambisonics (FOA, ACN/SN3D) spatial audio clip captured at your current position; "
    "the spatial audio encodes the direction and distance of the sound source relative to you.\n"
    "These are your historical observations: {history}\n"
    "This is your current observation: {current}\n"
    "This is the spatial audio recorded at your current position: {audio}\n"
    "Task (what the sound is asking you to do): {instruction}\n"
    "Using the spatial audio together with the visual observations, plan where to go next to reach the "
    "sound source and complete the task. Output ONLY the future trajectory as a JSON list of waypoints "
    "relative to your current pose, ordered from the next step onward, each waypoint formatted as "
    "[x, y, yaw] where x is forward in meters, y is left in meters, and yaw is the heading in radians. "
    "Do not output any explanation, coordinates in the image, or intermediate steps."
)


def build_conversation(history_paths, text_instruction, future_traj):
    """OmniEcho processing component."""
    history_imgs = " ".join([IMAGE_TOKEN] * len(history_paths)) if history_paths else "(none)"
    human_value = NAV_PROMPT_TEMPLATE.format(
        history=history_imgs,
        current=IMAGE_TOKEN,
        audio=AUDIO_TOKEN,
        instruction=text_instruction.strip().rstrip("."),
    )
    gpt_value = json.dumps([[round(float(a), 4) for a in wp] for wp in future_traj],
                           ensure_ascii=False)
    return [
        {"from": "human", "value": human_value},
        {"from": "gpt", "value": gpt_value},
    ]


def frame_rel_path(scene, setting, ep_id, frame_id):
    """OmniEcho processing component."""
    return (f"{DATASET}/{scene}/videos/chunk-{ep_id // 1000:03d}/"
            f"observation.images.rgb.{setting}/episode_{ep_id:06d}_{frame_id}.jpg")


def resolve_audio_path(audio_path):
    """OmniEcho processing component."""
    if os.path.isabs(audio_path):
        return audio_path
    return os.path.join(NAV_ROOT, audio_path)


def parse_episodes(spec, available):
    if spec == "all":
        return available
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [e for e in out if e in available]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="all", help='Scene identifier, or "all" to process every scene')
    ap.add_argument("--episodes", default="all", help='Episode selection, e.g. "0-2", "0,3,5", or "all"')
    ap.add_argument("--max_samples", type=int, default=None, help="Maximum total samples for debugging")
    ap.add_argument("--data_root", default=DATA_ROOT, help="Data root containing OmniEchoBench-Nav/<scene>/...")
    ap.add_argument("--audio_scripts", default=AUDIO_SCRIPTS, help="Stage 2 *_with_audio.jsonl")
    ap.add_argument("--out_audio_dir", default=DEFAULT_OUT_AUDIO)
    ap.add_argument("--out_jsonl", default=DEFAULT_OUT_JSONL)
    ap.add_argument("--dry_run", action="store_true", help="Compute and write JSONL with audio paths, without writing WAV files")
    ap.add_argument("--pred_steps", type=int, default=8,
                    help="Keep only the nearest N trajectory steps (default: 8, matching R2R)")
    args = ap.parse_args()

    data_root = args.data_root
    setting = f"{HEIGHT}cm_{PITCH_1}deg"


    audio_idx = {}
    audio_scenes = set()
    with open(args.audio_scripts) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            audio_idx[(r["scene_id"], r["episode_id"])] = r
            audio_scenes.add(r["scene_id"])


    ds_root = os.path.join(data_root, DATASET)
    if args.scene == "all":
        scenes = sorted(s for s in os.listdir(ds_root)
                        if s in audio_scenes
                        and os.path.isdir(os.path.join(ds_root, s, "data", "chunk-000")))
    else:
        scenes = [args.scene]
    print(f"[info] Scenes to process: {len(scenes)}")

    if not args.dry_run:
        os.makedirs(args.out_audio_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)), exist_ok=True)


    tasks = []
    for scene in scenes:
        scene_dir = os.path.join(data_root, DATASET, scene)
        chunk0 = os.path.join(scene_dir, "data", "chunk-000")
        if not os.path.isdir(chunk0):
            print(f"[skip] Scene {scene}: no parquet data")
            continue
        available = sorted(int(fn.split("_")[1].split(".")[0])
                           for fn in os.listdir(chunk0) if fn.endswith(".parquet"))
        for ep_id in parse_episodes(args.episodes, available):
            tasks.append((scene, scene_dir, ep_id))
    print(f"[info] Total episodes to process: {len(tasks)}")

    mono_cache = {}
    n_written = 0
    n_skip_noaudio = 0
    n_ep_done = 0
    n_ep_err = 0
    stop_all = False


    done = set()
    if os.path.exists(args.out_jsonl):
        with open(args.out_jsonl) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["scene_id"], r["episode_id"]))
                except Exception:
                    continue
        n_written = sum(1 for _ in open(args.out_jsonl))
        print(f"[resume] Found {n_written} samples covering {len(done)} episodes; skipping these episodes")
    fout = open(args.out_jsonl, "a")

    pbar = tqdm(tasks, desc="episodes", unit="ep", dynamic_ncols=True)
    for (scene, scene_dir, ep_id) in pbar:
        if stop_all:
            break
        key = (scene, ep_id)
        if key in done:
            continue
        arec = audio_idx.get(key)
        if arec is None or "audio_path" not in arec or "text_instruction" not in arec:
            n_skip_noaudio += 1
            continue
        audio_path = resolve_audio_path(arec["audio_path"])
        text_instruction = arec["text_instruction"]

        try:
            actions, c2w_0, c2w_2, pixel_goals = G.load_episode(scene_dir, ep_id)
            source_world = c2w_0[-1, :3, 3].copy()   

            samples = G.enumerate_pixel_goal_samples(actions, pixel_goals)
            if not samples:
                n_ep_done += 1
                continue

            if audio_path not in mono_cache:
                mono_cache[audio_path] = G.load_mono(audio_path)
            sr, mono = mono_cache[audio_path]

            for (start, goal_len, pg_coords) in samples:
                
                if start != 0:
                    history_id = np.unique(np.linspace(0, start - 1, NUM_HISTORY, dtype=np.int32)).tolist()
                else:
                    history_id = []
                history_paths = [frame_rel_path(scene, setting, ep_id, hid) for hid in history_id]
                current_path = frame_rel_path(scene, setting, ep_id, start)


                pose_slice = c2w_2[start:start + goal_len + 1]
                future_rel = G.get_trajectory_relative_to_frame(pose_slice, camera_deg=PITCH_2)


                unit, dist = G.source_dir_local(c2w_0[start], source_world)
                az = float(np.degrees(np.arctan2(unit[1], unit[0])))   
                el = float(np.degrees(np.arcsin(np.clip(unit[2], -1.0, 1.0))))

                foa_name = f"OmniEchoBench-Nav_{scene}_ep{ep_id}_f{start}_{setting}.wav"
                foa_path = os.path.join(args.out_audio_dir, foa_name)
                if not args.dry_run:
                    foa = G.encode_static_foa(mono, unit, dist)
                    G.write_foa_wav(foa_path, foa, sr)
                n_written += 1


                output_traj = np.round(future_rel[1:1 + args.pred_steps], 5)

                conversations = build_conversation(history_paths, text_instruction, output_traj)

                rec = {
                    "dataset": DATASET,
                    "scene_id": scene,
                    "episode_id": ep_id,
                    "setting": setting,
                    "sample_start_frame": start,
                    "goal_len": int(goal_len),
                    "num_history": NUM_HISTORY,
                    "conversations": conversations,
                    "history_frame_paths": history_paths,
                    "current_frame_path": current_path,
                    "foa_audio_path": foa_path,
                    
                    "text_instruction": text_instruction,
                    "sound_content": arec.get("sound_content", ""),
                    "voice_style": arec.get("voice_style", ""),
                    "category": arec.get("category"),
                    "source_mono_audio_path": audio_path,
                    "audio_sample_rate": int(sr),
                    "audio_channels": 4,
                    "audio_format": "FOA_ACN_SN3D",
                    "source_world": source_world.tolist(),
                    "listener_world": c2w_0[start, :3, 3].tolist(),
                    "azimuth_deg": az,
                    "elevation_deg": el,
                    "distance_m": float(dist),
                    "pixel_goal": pg_coords,
                    "future_trajectory": output_traj.tolist(),
                    "pred_steps": int(output_traj.shape[0]),
                }

                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if args.max_samples and n_written >= args.max_samples:
                    stop_all = True
                    break
            n_ep_done += 1
        except Exception as e:
            n_ep_err += 1
            tqdm.write(f"[err] scene {scene} ep{ep_id}: {type(e).__name__}: {e}")

        fout.flush()
        pbar.set_postfix(samples=n_written, scene=scene, err=n_ep_err)

    pbar.close()
    fout.close()

    print(f"\n[done] Scenes: {len(scenes)}  processed episodes: {n_ep_done}"
          f"  failed episodes: {n_ep_err}")
    print(f"[done] Samples (= FOA WAV files): {n_written}"
          f"{' (dry_run, no WAV files written)' if args.dry_run else ''}")
    print(f"[done] Episodes skipped for missing audio: {n_skip_noaudio}")
    print(f"[done] jsonl -> {args.out_jsonl}")
    if not args.dry_run:
        print(f"[done] FOA wav -> {args.out_audio_dir}")


if __name__ == "__main__":
    main()
