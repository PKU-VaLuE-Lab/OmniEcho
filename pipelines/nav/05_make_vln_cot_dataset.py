#!/usr/bin/env python3

#   OPENAI_API_KEY=sk-xxx python 05_make_vln_cot_dataset.py \


"""OmniEcho processing component."""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
from settings import DATA_ROOT as TRAJECTORY_ROOT, NAV_ROOT as OUTPUT_ROOT, MODEL_ROOT, API_BASE_URL as SERVICE_URL, API_MODEL
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geometry as G  # noqa: E402

# =============================================================================

# =============================================================================
DATA_ROOT = TRAJECTORY_ROOT
DATASET = "OmniEchoBench-Nav"
NAV_ROOT = OUTPUT_ROOT
DEFAULT_IN = os.path.join(NAV_ROOT, "spatial_voice_vln_omni_sft.jsonl")

API_BASE_URL = SERVICE_URL


_AUDIO_RE = re.compile(r"OmniEchoBench-Nav_(?P<scene>.+)_ep(?P<ep>\d+)_f(?P<frame>\d+)_(?P<setting>\d+cm_\d+deg)\.wav$")

_TASK_RE = re.compile(r"Task \(what the sound is asking you to do\):\s*(.*?)\s*\nUsing", re.S)


DIST_NEAR = 1.5
DIST_FAR = 4.0


TRANS_EPS = 0.05   
YAW_EPS = 0.10     


SEG_TURN_MIN_DEG = 15.0
SEG_TRANS_MIN_M = 0.10
SEG_FWD_CONE_DEG = 45.0

DEFAULT_MODEL = API_MODEL


# =============================================================================

# =============================================================================
def _extract_json_object(text: str):
    """OmniEcho processing component."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.endswith("```"):
            t = t[:-3]
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
    
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    
    start = t.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        chunk = t[start:i + 1]
                        try:
                            obj = json.loads(chunk)
                            if isinstance(obj, dict):
                                return obj
                        except Exception:
                            break
        start = t.find("{", start + 1)
    return None


# =============================================================================

# =============================================================================
def parse_audio_name(audio_path: str):
    """OmniEcho processing component."""
    m = _AUDIO_RE.search(os.path.basename(audio_path))
    if not m:
        return None
    return m.group("scene"), int(m.group("ep")), int(m.group("frame")), m.group("setting")


_geo_cache: dict = {}
_geo_lock = threading.Lock()


def get_source_dir(scene: str, ep: int, frame: int, data_root: str):
    """OmniEcho processing component."""
    key = (scene, ep)
    with _geo_lock:
        cached = _geo_cache.get(key)
    if cached is None:
        scene_dir = os.path.join(data_root, DATASET, scene)
        _actions, c2w_0, _c2w_2, _pg = G.load_episode(scene_dir, ep)
        source_world = c2w_0[-1, :3, 3].copy()
        cached = (c2w_0, source_world)
        with _geo_lock:
            _geo_cache[key] = cached
    c2w_0, source_world = cached
    unit, dist = G.source_dir_local(c2w_0[frame], source_world)
    az = float(np.degrees(np.arctan2(unit[1], unit[0])))          
    el = float(np.degrees(np.arcsin(np.clip(unit[2], -1.0, 1.0))))  
    return az, el, float(dist)


def classify_direction8(az_deg: float) -> str:
    """OmniEcho processing component."""
    a = ((az_deg + 180.0) % 360.0) - 180.0
    aa = abs(a)
    if aa <= 22.5:
        return "front"
    if aa >= 157.5:
        return "behind"
    side = "left" if a > 0 else "right"
    if aa <= 67.5:
        return f"front-{side}"
    if aa <= 112.5:
        return side
    return f"behind-{side}"


def distance_bucket(d: float) -> str:
    if d < DIST_NEAR:
        return "near"
    if d > DIST_FAR:
        return "far"
    return "medium range"


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def motion_label(heading_deg: float) -> str:
    """OmniEcho processing component."""
    a = ((heading_deg + 180.0) % 360.0) - 180.0
    aa = abs(a)
    if aa <= 22.5:
        return "go straight forward"
    side = "left" if a > 0 else "right"
    if aa <= 67.5:
        return f"go forward-{side}"
    if aa <= 112.5:
        return f"turn {side} and move"
    return "back up"


def segment_meta_action(bx, by, dyaw):
    """OmniEcho processing component."""
    trans = math.hypot(bx, by)
    moving = trans >= SEG_TRANS_MIN_M
    turning = abs(dyaw) >= math.radians(SEG_TURN_MIN_DEG)

    if moving and turning:
        return "move forward-left" if dyaw > 0 else "move forward-right"
    if moving:
        ang = math.degrees(math.atan2(by, bx))
        aa = abs(ang)
        if aa <= SEG_FWD_CONE_DEG:
            return "move forward"
        if aa >= 180.0 - SEG_FWD_CONE_DEG:
            return "move backward"
        return "move left" if by > 0 else "move right"
    if turning:
        return "turn left" if dyaw > 0 else "turn right"
    return "stay in place"


def derive_meta_actions(traj, n_segments=2):
    """OmniEcho processing component."""
    pts = [[0.0, 0.0, 0.0]] + [list(map(float, wp)) for wp in traj]

    steps = []
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        dyaw = _wrap(pts[i][2] - pts[i - 1][2])
        trans = math.hypot(dx, dy)
        if trans < TRANS_EPS:
            if dyaw > YAW_EPS:
                steps.append("turn left in place")
            elif dyaw < -YAW_EPS:
                steps.append("turn right in place")
            else:
                steps.append("hold position")
        else:
            yaw0 = pts[i - 1][2]
            bx = dx * math.cos(yaw0) + dy * math.sin(yaw0)
            by = -dx * math.sin(yaw0) + dy * math.cos(yaw0)
            steps.append(motion_label(math.degrees(math.atan2(by, bx))))

    compressed = []
    for s in steps:
        if not compressed or compressed[-1] != s:
            compressed.append(s)

    n = len(traj)
    two_actions = []
    if n > 0:
        bounds = [round(k * n / n_segments) for k in range(n_segments + 1)]
        for k in range(n_segments):
            i0, i1 = bounds[k], bounds[k + 1]
            if i1 <= i0:
                continue
            p0, p1 = pts[i0], pts[i1]
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            dyaw = _wrap(p1[2] - p0[2])
            yaw0 = p0[2]
            bx = dx * math.cos(yaw0) + dy * math.sin(yaw0)
            by = -dx * math.sin(yaw0) + dy * math.cos(yaw0)
            two_actions.append(segment_meta_action(bx, by, dyaw))
    if not two_actions:
        two_actions = ["stay in place"]

    net_x, net_y, net_yaw = pts[-1][0], pts[-1][1], pts[-1][2]
    net_heading = math.degrees(math.atan2(net_y, net_x)) if math.hypot(net_x, net_y) > 1e-6 else 0.0

    return {
        "steps": steps,
        "sequence": compressed,
        "two_actions": two_actions,
        "net_forward_m": round(net_x, 3),
        "net_left_m": round(net_y, 3),
        "net_yaw_deg": round(math.degrees(net_yaw), 1),
        "net_heading_deg": round(net_heading, 1),
    }


# =============================================================================

# =============================================================================
def parse_record(rec):
    """OmniEcho processing component."""
    msgs = rec.get("messages")
    if not msgs or len(msgs) < 2:
        return None
    user = msgs[0]
    content = user.get("content")
    if not isinstance(content, list):
        return None
    images = [c["image"] for c in content if c.get("type") == "image" and c.get("image")]
    audios = [c["audio"] for c in content if c.get("type") == "audio" and c.get("audio")]
    texts = [c["text"] for c in content if c.get("type") == "text" and c.get("text")]
    if not images or not audios or not texts:
        return None

    prompt_text = texts[0]
    mt = _TASK_RE.search(prompt_text)
    instruction = mt.group(1).strip() if mt else "(no explicit instruction)"

    assistant = msgs[-1].get("content", "")
    traj = None
    if isinstance(assistant, str):
        try:
            traj = json.loads(assistant)
        except json.JSONDecodeError:
            traj = None
    if not isinstance(traj, list):
        return None

    return {
        "history_images": images[:-1],   
        "current_image": images[-1],
        "audio": audios[0],
        "instruction": instruction,
        "trajectory": traj,
        "trajectory_str": assistant if isinstance(assistant, str) else json.dumps(traj),
    }


# =============================================================================

# =============================================================================
_OUTPUT_INSTR_RE = re.compile(r"Output ONLY the future trajectory.*", re.S)

_WAYPOINT_FMT = (
    "the future trajectory as a JSON list of waypoints relative to your current pose, ordered from "
    "the next step onward, each waypoint formatted as [x, y, yaw] where x is forward in meters, "
    "y is left in meters, and yaw is the heading in radians"
)


def _new_output_instruction(think_tags: bool) -> str:
    reason = ("First, reason step by step about what you see in your observations, the direction and "
              "distance of the sound source, and how to move toward it while avoiding obstacles")
    if think_tags:
        reason += ", and wrap this reasoning in <think> and </think> tags"
    return (
        f"Then act in three parts, in order:\n"
        f"1. {reason}.\n"
        f"2. Next, on a new line, output your two main meta-actions — one for the first half of "
        f"your path and one for the second half — in the form \"Meta actions: <first> then <second>\".\n"
        f"3. Finally, on a new line, output {_WAYPOINT_FMT}."
    )


def rewrite_user_prompt(user_msg, think_tags: bool):
    """OmniEcho processing component."""
    content = user_msg.get("content")
    if not isinstance(content, list):
        return user_msg
    new_instr = _new_output_instruction(think_tags)
    new_content = []
    replaced = False
    for c in content:
        if (not replaced and c.get("type") == "text" and isinstance(c.get("text"), str)
                and "Output ONLY the future trajectory" in c["text"]):
            c = dict(c)
            c["text"] = _OUTPUT_INSTR_RE.sub(lambda _m: new_instr, c["text"])
            replaced = True
        new_content.append(c)
    new_msg = dict(user_msg)
    new_msg["content"] = new_content
    return new_msg


# =============================================================================

# =============================================================================
def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


COT_SYSTEM = (
    "You are writing first-person chain-of-thought (CoT) reasoning traces AS a sound-source "
    "navigation robot. You (the robot) hear a spatial-audio cue and must move toward the sound source "
    "while avoiding obstacles. You are given YOUR OWN recent egocentric RGB frames (oldest to newest, "
    "the last one is your current view), the task the sound is asking you to do, the GEOMETRICALLY "
    "COMPUTED direction and distance of the sound source relative to you, the ground-truth meta-action "
    "sequence, and the ground-truth future trajectory. Write a single coherent, physically plausible "
    "reasoning trace, STRICTLY IN THE FIRST PERSON (use 'I', 'me', 'my', 'ahead of me'; NEVER say "
    "'the robot' or describe yourself from the outside). The computed direction/distance and the "
    "meta-action sequence are the SOURCE OF TRUTH: never contradict them. Describe the visual scene "
    "ONLY from what you actually see in the frames, as if you are standing there looking."
)

COT_TASK_TMPL = """Write a first-person chain-of-thought reasoning trace for my next navigation step.

=== Task the sound is asking me to do ===
{instruction}

=== Computed sound-source cue (ground truth — do NOT contradict) ===
- Direction of the sound source relative to me: {direction}  (azimuth {az:+.0f}deg, where + is to my left, 0 is straight ahead)
- Distance: {dist:.1f} m ({dist_bucket})
- Elevation: {el:+.0f} deg (roughly at my ear level unless large)

=== Ground-truth motion (derived from the future trajectory — do NOT contradict) ===
- My two main meta-actions (first half of my path, then second half): {meta_seq}
- Net motion over this horizon: {net_fwd:+.2f} m forward, {net_left:+.2f} m to my left; my heading changes by {net_yaw:+.0f} deg.
- Overall travel direction of my path: {path_dir}
- {deviation_note}

=== What to write ===
Produce ONE first-person reasoning trace (a few sentences, natural English, no bullet headers, no markdown) that flows through these stages IN ORDER:
1. SCENE: briefly describe what I see in front of me / around me in my current view (e.g. a wall, a doorway, a hallway, a living room, furniture, an open space). Use my past frames for context if helpful.
2. INSTRUCTION: restate, in one clause, what the sound is asking me to do.
3. SOUND DIRECTION: state where I hear the sound source ({direction}, {dist_bucket}) and tie it to what I see (e.g. "I hear it coming from my right, likely from beyond the doorway on my right").
4. PLAN: give a short high-level plan that reconciles what I see (obstacles/openings) with where I hear the sound. If my path direction differs from the straight line to the source, explain it as me going around an obstacle or leaving my current room; if the source is far, consider that it may be in another room.
5. DECISION: conclude the reasoning in ONE short first-person sentence that commits to my two sequential moves — what I do over the first half of my path, then over the second half — consistent with: {meta_seq}. Phrase it naturally (e.g. "so I will first move forward-right and then go straight forward toward it"); do NOT write a "Meta actions:" label, do NOT enumerate step-by-step sub-moves, and do NOT output any coordinates. A separate canonical meta-action line and the trajectory will be added after your text.

Write everything in the FIRST PERSON. Do NOT refer to "the robot" in the third person. Do NOT output the numeric trajectory or any coordinates. Stop at the meta-action level.

Output STRICT JSON and nothing else, exactly:
{{"cot": "<the first-person reasoning trace as a single string>"}}
"""


def build_task_prompt(pr, geo, meta):
    az, el, dist = geo
    direction = classify_direction8(az)
    path_dir = classify_direction8(meta["net_heading_deg"])
    dev = abs(_wrap(math.radians(meta["net_heading_deg"] - az)))
    dev_deg = math.degrees(dev)
    if dev_deg > 60:
        deviation_note = (f"My path's travel direction ({path_dir}) clearly differs from the straight "
                          f"line to the source ({direction}) by ~{dev_deg:.0f} deg, so I am NOT heading "
                          f"straight at the sound — I am going around an obstacle or repositioning first.")
    else:
        deviation_note = (f"My path's travel direction ({path_dir}) is roughly consistent with heading "
                          f"toward the source ({direction}).")

    return COT_TASK_TMPL.format(
        instruction=pr["instruction"],
        direction=direction,
        az=az, el=el, dist=dist, dist_bucket=distance_bucket(dist),
        meta_seq=" then ".join(meta["two_actions"]),
        net_fwd=meta["net_forward_m"], net_left=meta["net_left_m"], net_yaw=meta["net_yaw_deg"],
        path_dir=path_dir, deviation_note=deviation_note,
    )


def call_vlm(client, model, pr, task_prompt, max_history_imgs, timeout, retries):
    """OmniEcho processing component."""
    hist = pr["history_images"][-max_history_imgs:] if max_history_imgs > 0 else []
    img_paths = hist + [pr["current_image"]]
    content = []
    n_hist = len(hist)
    for j, ip in enumerate(img_paths):
        if not os.path.exists(ip):
            continue
        tag = "current view" if j == n_hist else f"past frame {j + 1}/{n_hist}"
        content.append({"type": "text", "text": f"[{tag}]"})
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encode_image(ip)}"}})
    content.append({"type": "text", "text": task_prompt})

    messages = [{"role": "system", "content": COT_SYSTEM},
                {"role": "user", "content": content}]

    last_raw = ""
    for _ in range(max(1, retries)):
        try:
            completion = client.chat.completions.create(
                model=model, messages=messages, timeout=timeout)
            last_raw = completion.choices[0].message.content or ""
        except Exception as e:
            last_raw = f"(api error: {e})"
            print(e)
            continue
        obj = _extract_json_object(last_raw)
        cot = str(obj.get("cot", "")).strip() if isinstance(obj, dict) else ""
        if cot:
            return cot
        if last_raw and not last_raw.startswith("(api error"):
            cleaned = last_raw.strip()
            if len(cleaned) > 20:
                return cleaned
    return None


def fallback_cot(pr, geo, meta):
    """OmniEcho processing component."""
    az, el, dist = geo
    direction = classify_direction8(az)
    seq = " then ".join(meta["two_actions"])
    return (f"Looking at my current view, I see the space ahead of me. "
            f"The sound is asking me to: {pr['instruction']}. "
            f"I hear the source to my {direction} at {distance_bucket(dist)} "
            f"(about {dist:.1f} m away). To reach it, I will {seq}.")


# =============================================================================

# =============================================================================
def process_one(rec, data_root, client, model, args):
    pr = parse_record(rec)
    if pr is None:
        return None, "bad_record"
    ai = parse_audio_name(pr["audio"])
    if ai is None:
        return None, "bad_audio_name"
    scene, ep, frame, _setting = ai

    try:
        az, el, dist = get_source_dir(scene, ep, frame, data_root)
    except Exception as e:
        return None, f"geo_err:{type(e).__name__}"
    geo = (az, el, dist)
    meta = derive_meta_actions(pr["trajectory"])

    task_prompt = build_task_prompt(pr, geo, meta)
    cot = call_vlm(client, model, pr, task_prompt,
                   args.max_history_imgs, args.timeout, args.retries)
    if not cot:
        if args.allow_fallback:
            cot = fallback_cot(pr, geo, meta)
        else:
            return None, "vlm_failed"


    meta_line = "Meta actions: " + " then ".join(meta["two_actions"])
    cot_block = f"<think>\n{cot}\n</think>" if args.think_tags else cot
    new_assistant = f"{cot_block}\n\n{meta_line}\n\n{pr['trajectory_str']}"

    out = dict(rec)   
    new_user = rewrite_user_prompt(rec["messages"][0], args.think_tags)
    out["messages"] = [new_user, {"role": "assistant", "content": new_assistant}]
    return out, "ok"


# =============================================================================

# =============================================================================
def sample_key(rec):
    """OmniEcho processing component."""
    try:
        for c in rec["messages"][0]["content"]:
            if c.get("type") == "audio" and c.get("audio"):
                return os.path.basename(c["audio"])
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_jsonl", default=DEFAULT_IN, help="Input Omni SFT JSONL from stage 4")
    ap.add_argument("--out_jsonl", default=None, help="Output JSONL (default: input stem + _cot.jsonl)")
    ap.add_argument("--data_root", default=DATA_ROOT, help="Data root containing OmniEchoBench-Nav/<scene>/data/...")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="VLM identifier; must support image input")
    ap.add_argument("--api_key", default=None, help="Annotation service API key (defaults to OPENAI_API_KEY)")
    ap.add_argument("--base_url", default=API_BASE_URL, help="OpenAI-compatible endpoint")
    
    ap.add_argument("--local", action="store_true",
                    help="Use a local model, ignoring the annotation service key; balance samples across ports round-robin")
    ap.add_argument("--local-model", default="OmniEcho", help="Local model identifier (used with --local)")
    ap.add_argument("--local-host", default="127.0.0.1", help="Local service host")
    ap.add_argument("--local-ports", default="8200,8201",
                    help="Comma-separated local service ports; distribute samples round-robin")
    ap.add_argument("--max-history-imgs", type=int, default=0, help="Maximum history frames sent to the API (0: current frame only)")
    ap.add_argument("--workers", type=int, default=8, help="Concurrent threads for API requests")
    ap.add_argument("--timeout", type=int, default=120, help="Timeout per API call in seconds")
    ap.add_argument("--retries", type=int, default=3, help="API retry count per sample")
    ap.add_argument("--max-samples", type=int, default=None, help="Process only the first N records for debugging")
    ap.add_argument("--shard", type=int, default=0, help="Shard index used with --num-shards")
    ap.add_argument("--num-shards", type=int, default=1, help="Total shards for multiprocess or multimachine processing")
    ap.add_argument("--think-tags", action="store_true", help="Wrap CoT in <think>...</think> (disabled by default)")
    ap.add_argument("--allow-fallback", action="store_true", help="Use a deterministic template on API failure instead of skipping")
    ap.add_argument("--no-resume", action="store_true", help="Disable resume and overwrite the output")
    args = ap.parse_args()

    out_path = args.out_jsonl or (os.path.splitext(args.in_jsonl)[0] + "_cot.jsonl")

    try:
        from openai import OpenAI
    except Exception as e:
        print(f"[fatal] Cannot import openai: {e}", file=sys.stderr)
        sys.exit(1)

    if args.local:
        ports = [p.strip() for p in args.local_ports.split(",") if p.strip()]
        if not ports:
            print("[fatal] --local requires at least one --local-ports value.", file=sys.stderr)
            sys.exit(1)
        model = args.local_model
        clients = [OpenAI(base_url=f"http://{args.local_host}:{p}/v1",
                          api_key=(args.api_key or "EMPTY")) for p in ports]
        print(f"[info] Local model '{model}' @ " +
              ", ".join(f"http://{args.local_host}:{p}/v1" for p in ports))
    else:
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("[fatal] No API key set (--api_key / OPENAI_API_KEY).", file=sys.stderr)
            sys.exit(1)
        model = args.model
        clients = [OpenAI(base_url=args.base_url, api_key=api_key)]


    with open(args.in_jsonl) as f:
        all_lines = [ln for ln in f if ln.strip()]
    records = []
    for idx, ln in enumerate(all_lines):
        if idx % args.num_shards != args.shard:
            continue
        try:
            records.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    if args.max_samples:
        records = records[:args.max_samples]
    print(f"[info] Input rows: {len(all_lines)}; shard {args.shard}/{args.num_shards}: {len(records)} records")


    done = set()
    write_mode = "a"
    if args.no_resume:
        write_mode = "w"
    elif os.path.exists(out_path):
        with open(out_path) as f:
            for ln in f:
                try:
                    k = sample_key(json.loads(ln))
                    if k:
                        done.add(k)
                except Exception:
                    continue
        print(f"[resume] Output exists; skipping {len(done)} completed records")

    todo = [r for r in records if sample_key(r) not in done]
    print(f"[info] Pending records: {len(todo)} -> {out_path}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fout = open(out_path, write_mode)
    write_lock = threading.Lock()
    stats = {"ok": 0, "skip": 0}
    reasons = {}

    def worker(rec, client):
        return process_one(rec, args.data_root, client, model, args)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(worker, r, clients[i % len(clients)]): r
                   for i, r in enumerate(todo)}
        pbar = tqdm(as_completed(futures), total=len(futures), desc="cot", unit="s", dynamic_ncols=True)
        for fut in pbar:
            try:
                out, reason = fut.result()
            except Exception as e:
                out, reason = None, f"exc:{type(e).__name__}"
            if out is not None:
                with write_lock:
                    fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                    fout.flush()
                stats["ok"] += 1
            else:
                stats["skip"] += 1
                reasons[reason] = reasons.get(reason, 0) + 1
            pbar.set_postfix(ok=stats["ok"], skip=stats["skip"])
        pbar.close()
    fout.close()

    print(f"\n[done] Succeeded: {stats['ok']}, skipped: {stats['skip']}")
    if reasons:
        print(f"[done] Reasons for skipping: {reasons}")
    print(f"[done] Output -> {out_path}")


if __name__ == "__main__":
    main()
