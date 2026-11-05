#!/usr/bin/env python3

#   python 01_generate_audio_scripts.py \

"""OmniEcho processing component."""
from __future__ import annotations

import argparse
import base64
import json
import os
from settings import DATA_ROOT as TRAJECTORY_ROOT, NAV_ROOT as OUTPUT_ROOT, MODEL_ROOT, API_BASE_URL as SERVICE_URL, API_MODEL
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

API_BASE_URL = SERVICE_URL


DEFAULT_TRAJ_ROOT = os.path.join(TRAJECTORY_ROOT, "OmniEchoBench-Nav")
DEFAULT_OUT = os.path.join(OUTPUT_ROOT, "audio_scripts.jsonl")


# =============================================================================
# Prompt
# =============================================================================

SYSTEM_PROMPT = """\
You are a scenario designer for an indoor "sound-source navigation" task. A robot \
starts somewhere in a house and must reach a specific target object/location ONLY by \
following a sound emitted from that target (plus a short text instruction). You are \
shown the LAST few first-person camera frames of a navigation episode; these frames \
show the destination area (the target). Your job: invent a plausible sound event that \
would emit from this destination, so the robot can localize it acoustically.

Design two categories of scripts; pick ONE per episode (respect the requested category \
when given, and keep the two categories roughly balanced across the dataset):
  Category 1 - "spoken utterance that calls the robot over": the sound itself is a short \
    spoken English utterance from a person at the destination. This is NOT limited to \
    emergencies or calls for help - ANY natural reason a person would summon the robot is \
    valid, and you should keep them DIVERSE across episodes, e.g.: asking for help \
    ("Robot, I fell down, please come and find me."), requesting an item be fetched or \
    carried ("Hey robot, can you bring this over here?"), inviting it to come look at \
    something ("Robot, come here, I want to show you something."), assigning a task, a \
    casual/curious call, a playful summon, an impatient shout, etc. Vary the speaker, \
    wording, tone and intent widely. The text_instruction can briefly restate the intent \
    (e.g. "Go to the person who is calling you.").
    For Category 1 you MUST also fill "voice_style": one English sentence describing the \
    SPEAKER'S VOICE for downstream TTS - gender, approximate age, timbre, emotion/tone, \
    speaking pace, and optionally accent. Make these rich and varied across episodes, e.g. \
    "A young man's bright, energetic voice, cheerful and fast-paced, mild American accent." \
    or "An elderly woman's soft, trembling voice, anxious and pleading, speaking slowly."
  Category 2 - "ambient sound source + text command": the sound is a common non-speech \
    sound believable in this room (knocking, running water / flushing, dragging furniture, \
    a pet - dog barking / cat meowing, human chatter, music, TV, kettle whistle, phone \
    ringing, etc.). The text_instruction is a short English command telling the robot what \
    to home in on, e.g. "Find where the knocking sound is coming from." For Category 2, set \
    "voice_style" to a short English description of the sound's acoustic character \
    (e.g. "Sharp, bright metallic clattering.") or "" if not applicable.

Use the visible scene ONLY to make the sound realistic (bathroom -> water/flush; kitchen -> \
clattering dishes / kettle; living room -> TV / music; bedroom -> alarm clock / phone).

HARD CONSTRAINTS:
- NEVER reveal the destination's location, room name, or nameable furniture in either \
  sound_content or text_instruction. The robot must find it by sound, not by being told \
  where it is. (Do NOT say things like "in the bathroom" / "near the toilet".)
- sound_content, text_instruction and voice_style must be in ENGLISH.
- Keep each field to one short natural sentence.

Return ONLY a JSON object, no markdown fences, with exactly these keys:
{
  "category": 1 or 2,
  "sound_content": "<the audio content: category 1 = the spoken line; category 2 = a short description of the ambient sound>",
  "text_instruction": "<short English instruction given to the robot>",
  "voice_style": "<category 1 = the speaker's voice (gender/age/timbre/emotion/pace/accent); category 2 = the sound's acoustic character or empty>",
  "scene_reasoning": "<one short sentence: why this sound fits what you see>"
}
"""


def build_user_content(instruction: str, preferred_category: int, images_b64: list):
    """OmniEcho processing component."""
    hint = (instruction or "").replace("<INSTRUCTION_SEP>", " / ").strip()
    text = (
        f"Requested category for this episode: {preferred_category} "
        f"(you may switch only if the scene makes the other category far more natural).\n"
        f"Context - the robot's original navigation goal (for YOUR understanding of what "
        f"the destination is; DO NOT leak it into the output): "
        f"\"{hint or 'unknown'}\".\n"
        f"Below are the last {len(images_b64)} first-person frames at the destination. "
        f"Design the sound-source script now and return only the JSON object."
    )
    content = [{"type": "text", "text": text}]
    for b64 in images_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    return content


# =============================================================================

# =============================================================================

def iter_scenes(traj_root):
    """OmniEcho processing component."""
    root_base = os.path.basename(traj_root.rstrip("/"))
    for name in sorted(os.listdir(traj_root)):
        child = os.path.join(traj_root, name)
        if not os.path.isdir(child):
            continue
        
        if os.path.isfile(os.path.join(child, "meta", "episodes.jsonl")):
            yield root_base, name, child
            continue
        
        for scene in sorted(os.listdir(child)):
            sdir = os.path.join(child, scene)
            if os.path.isfile(os.path.join(sdir, "meta", "episodes.jsonl")):
                yield name, scene, sdir


def load_episodes(scene_dir):
    """OmniEcho processing component."""
    out = []
    with open(os.path.join(scene_dir, "meta", "episodes.jsonl")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            instr = (e.get("tasks") or [""])[0]
            out.append((int(e["episode_index"]), int(e["length"]), instr))
    return out


_SETTING_CACHE = {}
_SETTING_LOCK = threading.Lock()


def resolve_setting(scene_dir, requested):
    """OmniEcho processing component."""
    with _SETTING_LOCK:
        if scene_dir in _SETTING_CACHE:
            return _SETTING_CACHE[scene_dir]
    vid_chunk = os.path.join(scene_dir, "videos", "chunk-000")
    prefix = "observation.images.rgb."
    resolved = requested
    if requested != "auto" and os.path.isdir(os.path.join(vid_chunk, prefix + requested)):
        resolved = requested
    else:
        avail = []
        if os.path.isdir(vid_chunk):
            avail = [d[len(prefix):] for d in sorted(os.listdir(vid_chunk))
                     if d.startswith(prefix)]
        pick = next((s for s in avail if s.endswith("_0deg")), avail[0] if avail else requested)
        if requested != "auto" and requested not in avail:
            print(f"    [setting] {os.path.basename(scene_dir)}: '{requested}' not found; using '{pick}'")
        resolved = pick
    with _SETTING_LOCK:
        _SETTING_CACHE[scene_dir] = resolved
    return resolved


def last_frame_paths(scene_dir, episode_id, length, setting, num_frames):
    """OmniEcho processing component."""
    chunk = f"chunk-{episode_id // 1000:03d}"
    rgb_dir = os.path.join(scene_dir, "videos", chunk,
                           f"observation.images.rgb.{setting}")
    start = max(0, length - num_frames)
    paths = []
    for i in range(start, length):
        fp = os.path.join(rgb_dir, f"episode_{episode_id:06d}_{i}.jpg")
        if os.path.exists(fp):
            paths.append(fp)
    return paths


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# =============================================================================

# =============================================================================

def parse_json_reply(text):
    """OmniEcho processing component."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.endswith("```"):
            t = t[: -3]
        if t.lstrip().startswith("json"):
            t = t.lstrip()[4:]
    l, r = t.find("{"), t.rfind("}")
    if l != -1 and r != -1 and r > l:
        t = t[l:r + 1]
    return json.loads(t)


def request_script(client, model, instruction, preferred_category, images_b64,
                   retries=3, timeout=60):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_content(instruction, preferred_category, images_b64)},
    ]
    last_err = None
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                timeout=timeout,
            )
            reply = completion.choices[0].message.content
            data = parse_json_reply(reply)
            cat = int(data.get("category", preferred_category))
            return {
                "category": cat if cat in (1, 2) else preferred_category,
                "sound_content": str(data.get("sound_content", "")).strip(),
                "text_instruction": str(data.get("text_instruction", "")).strip(),
                "voice_style": str(data.get("voice_style", "")).strip(),
                "scene_reasoning": str(data.get("scene_reasoning", "")).strip(),
            }
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = 2 * (attempt + 1)
            print(f"    [retry {attempt + 1}/{retries}] {type(e).__name__}: {e} -> wait {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"API call failed after {retries} attempts: {last_err}")


# =============================================================================

# =============================================================================

def load_done_keys(out_path):
    """OmniEcho processing component."""
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    done.add((r["scene_id"], int(r["episode_id"])))
                except Exception:  # noqa: BLE001
                    continue
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj_root", default=DEFAULT_TRAJ_ROOT,
                    help="OmniEchoBench-Nav data root: the dataset itself or a directory containing <dataset>/<scene>/")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output JSONL path (append mode with resume support)")
    ap.add_argument("--setting", default="125cm_0deg",
                    help='Forward camera setting (RGB directory suffix); "auto" selects *_0deg per scene')
    ap.add_argument("--num_frames", type=int, default=5, help="Number of final frames to use from each episode")
    ap.add_argument("--model", default=API_MODEL, help="Annotation service model identifier; must support image input")
    ap.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""),
                    help="Annotation service API key (defaults to OPENAI_API_KEY)")
    ap.add_argument("--workers", type=int, default=32, help="Number of concurrent threads")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=0.0, help="Delay in seconds after each call to reduce rate-limit errors")
    ap.add_argument("--max_scenes", type=int, default=None, help="Maximum scenes to process for debugging")
    ap.add_argument("--max_episodes_per_scene", type=int, default=None,
                    help="Maximum episodes per scene for debugging")
    ap.add_argument("--overwrite", action="store_true", help="Clear the existing output and regenerate")
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("Provide an API key via --api_key or OPENAI_API_KEY")

    client = OpenAI(base_url=API_BASE_URL, api_key=args.api_key)

    if args.overwrite and os.path.exists(args.out):
        os.remove(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    done = load_done_keys(args.out)
    if done:
        print(f"[resume] Skipping {len(done)} completed records.")


    worklist = []
    n_scene = 0
    for dataset, scene, scene_dir in iter_scenes(args.traj_root):
        if args.max_scenes is not None and n_scene >= args.max_scenes:
            break
        n_scene += 1
        episodes = load_episodes(scene_dir)
        for k, (ep_id, length, instr) in enumerate(episodes):
            if args.max_episodes_per_scene is not None and k >= args.max_episodes_per_scene:
                break
            if (scene, ep_id) in done:
                continue
            worklist.append((dataset, scene, scene_dir, ep_id, length, instr))
    
    for idx in range(len(worklist)):
        worklist[idx] = worklist[idx] + (1 if idx % 2 == 0 else 2,)
    print(f"[plan] Scenes: {n_scene}, pending episodes: {len(worklist)}, threads: {args.workers}")
    if not worklist:
        print("No episodes need processing.")
        return


    lock = threading.Lock()
    stats = {"written": 0, "failed": 0, "cat": {1: 0, 2: 0}}

    def worker(item):
        dataset, scene, scene_dir, ep_id, length, instr, preferred = item
        setting = resolve_setting(scene_dir, args.setting)
        paths = last_frame_paths(scene_dir, ep_id, length, setting, args.num_frames)
        if not paths:
            return ("skip", scene, ep_id, f"No frames found for {setting}")
        try:
            images_b64 = [encode_image(p) for p in paths]
            script = request_script(client, args.model, instr, preferred, images_b64,
                                    retries=args.retries, timeout=args.timeout)
        except Exception as e:  # noqa: BLE001
            return ("fail", scene, ep_id, str(e))
        record = {
            "dataset": dataset,   # "OmniEchoBench-Nav"
            "scene_id": scene,
            "episode_id": ep_id,
            "setting": setting,
            "num_frames_used": len(paths),
            "frame_paths": [os.path.relpath(p, args.traj_root) for p in paths],
            "original_instruction": instr.strip(),  
            "category": script["category"],
            "sound_content": script["sound_content"],
            "text_instruction": script["text_instruction"],
            "voice_style": script["voice_style"],
            "scene_reasoning": script["scene_reasoning"],
        }
        return ("ok", scene, ep_id, record)

    total = len(worklist)
    with open(args.out, "a", encoding="utf-8") as fout, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(worker, it) for it in worklist]
        for done_i, fut in enumerate(as_completed(futures), 1):
            status, scene, ep_id, payload = fut.result()
            with lock:
                if status == "ok":
                    fout.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    fout.flush()
                    stats["written"] += 1
                    stats["cat"][payload["category"]] = stats["cat"].get(payload["category"], 0) + 1
                    vs = payload.get("voice_style", "")
                    print(f"[{done_i}/{total}] ok {scene}/ep{ep_id} "
                          f"cat{payload['category']}: {payload['sound_content'][:55]}"
                          + (f" | voice: {vs[:40]}" if vs else ""))
                elif status == "skip":
                    print(f"[{done_i}/{total}] skip {scene}/ep{ep_id}: {payload}")
                else:
                    stats["failed"] += 1
                    print(f"[{done_i}/{total}] FAIL {scene}/ep{ep_id}: {payload}")
            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"\nDone: scenes={n_scene}, written={stats['written']}, failed={stats['failed']} "
          f"(cat1={stats['cat'].get(1, 0)}, cat2={stats['cat'].get(2, 0)}) -> {args.out}")


if __name__ == "__main__":
    main()
