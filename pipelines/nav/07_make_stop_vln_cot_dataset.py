#!/usr/bin/env python3

#   OPENAI_API_KEY=sk-xxx python 07_make_stop_vln_cot_dataset.py \


"""OmniEcho processing component."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from settings import DATA_ROOT as TRAJECTORY_ROOT, NAV_ROOT as OUTPUT_ROOT, MODEL_ROOT, API_BASE_URL as SERVICE_URL, API_MODEL
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

# =============================================================================

# =============================================================================
DEFAULT_INPUT = os.path.join(OUTPUT_ROOT, "spatial_voice_vln_omni_sft_cot.jsonl")
DEFAULT_OUT_DIR = OUTPUT_ROOT
DEFAULT_OUT_NAME = "vln_stop_within1m_cot.jsonl"

API_BASE_URL = SERVICE_URL
DEFAULT_MODEL = API_MODEL


N_WAYPOINTS = 8                   
STOP_TOKEN = "<stop>"


DEFAULT_MAX_DIST = 1.0

DEFAULT_MAX_KEEP = 3
DEFAULT_KEEP_MARGIN = 0.3         


TRANS_EPS = 0.05
YAW_EPS = 0.10
SEG_TURN_MIN_DEG = 15.0
SEG_TRANS_MIN_M = 0.10
SEG_FWD_CONE_DEG = 45.0


_DIST_RES = [re.compile(p) for p in (
    r"distance of ([\d.]+)\s*meters?",
    r"([\d.]+)\s*meters?\s+away",
    r"within ([\d.]+)\s*meters?",
    r"at ([\d.]+)\s*meters?",
    r"([\d.]+)\s*m\s+away",
)]


def parse_source_distance(assistant: str):
    """OmniEcho processing component."""
    for rgx in _DIST_RES:
        m = rgx.search(assistant)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None

_TASK_RE = re.compile(r"Task \(what the sound is asking you to do\):\s*(.*?)\s*\nUsing", re.S)

_OUT_INSTR_RE = re.compile(r"Using the spatial audio together with the visual observations.*", re.S)


# =============================================================================

# =============================================================================
def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


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


def motion_label(heading_deg: float) -> str:
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


def first_half_action(kept):
    """OmniEcho processing component."""
    if not kept:
        return "stay in place"
    pts = [[0.0, 0.0, 0.0]] + [list(map(float, wp)) for wp in kept]
    p0, p1 = pts[0], pts[-1]
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    dyaw = _wrap(p1[2] - p0[2])
    yaw0 = p0[2]
    bx = dx * math.cos(yaw0) + dy * math.sin(yaw0)
    by = -dx * math.sin(yaw0) + dy * math.cos(yaw0)
    return segment_meta_action(bx, by, dyaw)


def net_heading_deg(traj):
    """OmniEcho processing component."""
    if not traj:
        return 0.0
    nx, ny = float(traj[-1][0]), float(traj[-1][1])
    if math.hypot(nx, ny) <= 1e-6:
        return 0.0
    return math.degrees(math.atan2(ny, nx))


def cumulative_lengths(traj):
    """OmniEcho processing component."""
    pts = [(0.0, 0.0)] + [(float(w[0]), float(w[1])) for w in traj]
    out, acc = [], 0.0
    for i in range(1, len(pts)):
        acc += math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        out.append(acc)
    return out  


# =============================================================================

# =============================================================================
def _extract_json_object(text: str):
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
def parse_record(rec):
    msgs = rec.get("messages")
    if not msgs or len(msgs) < 2:
        return None
    content = msgs[0].get("content")
    if not isinstance(content, list):
        return None
    images = [c["image"] for c in content if c.get("type") == "image" and c.get("image")]
    audios = [c["audio"] for c in content if c.get("type") == "audio" and c.get("audio")]
    texts = [c["text"] for c in content if c.get("type") == "text" and c.get("text")]
    if not images or not texts:
        return None

    task_text = texts[-1]
    mt = _TASK_RE.search(task_text)
    instruction = mt.group(1).strip() if mt else "reach the sound source"

    assistant = msgs[-1].get("content", "")
    if not isinstance(assistant, str):
        return None
    bi = assistant.rfind("[[")
    if bi < 0:
        return None
    try:
        traj = json.loads(assistant[bi:])
    except json.JSONDecodeError:
        return None
    if not isinstance(traj, list) or not traj or not all(len(wp) == 3 for wp in traj):
        return None

    dist = parse_source_distance(assistant)

    return {
        "history_images": images[:-1],
        "current_image": images[-1],
        "audio": audios[-1] if audios else None,
        "instruction": instruction,
        "trajectory": traj,
        "source_distance": dist,
    }


# =============================================================================

# =============================================================================
def build_stop_plan(pr, args):
    """OmniEcho processing component."""
    D = pr["source_distance"]
    if D is None or D > args.max_dist:
        return None
    traj = pr["trajectory"]


    cum = cumulative_lengths(traj)
    keep_upper = D - args.keep_margin
    max_keep = min(args.max_keep, len(traj))
    K = 0
    for i in range(max_keep):
        if cum[i] <= keep_upper:
            K = i + 1
        else:
            break

    if K == 0:
        
        return {
            "stop_type": "direct",
            "kept": [],
            "traj_elems": [STOP_TOKEN] * N_WAYPOINTS,
            "meta_actions": ["stay in place", "stay in place"],
            "arrive_remaining": round(D, 3),
        }

    kept = traj[:K]
    remaining = D - cum[K - 1]
    elems = [json.dumps([round(float(v), 5) for v in wp]) for wp in kept] + \
            [STOP_TOKEN] * (N_WAYPOINTS - K)
    return {
        "stop_type": "move",
        "kept": kept,
        "traj_elems": elems,
        "meta_actions": [first_half_action(kept), "stay in place"],
        "arrive_remaining": round(remaining, 3),
    }


def traj_string(traj_elems):
    return "[" + ", ".join(traj_elems) + "]"


# =============================================================================

# =============================================================================
_WAYPOINT_FMT = (
    "the future trajectory as a JSON-style list of EXACTLY 8 waypoints relative to your current "
    "pose, ordered from the next step onward. Each waypoint is either [x, y, yaw] (x forward in "
    "meters, y left in meters, yaw heading in radians) or the token <stop>. Once you have reached "
    "the sound source (within about 1 meter), output <stop> for every remaining waypoint; if you "
    "are already at the source, output <stop> for all 8"
)


def _new_output_instruction(think_tags: bool) -> str:
    reason = ("First, reason step by step about what you see in your observations, the direction and "
              "distance of the sound source, and whether you have reached it")
    if think_tags:
        reason += ", and wrap this reasoning in <think> and </think> tags"
    return (
        "Using the spatial audio together with the visual observations, plan where to go next to "
        "reach the sound source and complete the task. Then act in three parts, in order:\n"
        f"1. {reason}.\n"
        "2. Next, on a new line, output your two main meta-actions — one for the first half of your "
        "path and one for the second half — in the form \"Meta actions: <first> then <second>\".\n"
        f"3. Finally, on a new line, output {_WAYPOINT_FMT}."
    )


def rewrite_user_prompt(user_msg, think_tags: bool):
    content = user_msg.get("content")
    if not isinstance(content, list):
        return user_msg
    new_instr = _new_output_instruction(think_tags)
    new_content = []
    replaced = False
    for c in content:
        if (not replaced and c.get("type") == "text" and isinstance(c.get("text"), str)
                and "Using the spatial audio together with the visual observations" in c["text"]):
            c = dict(c)
            c["text"] = _OUT_INSTR_RE.sub(lambda _m: new_instr, c["text"])
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
    "You are writing first-person chain-of-thought (CoT) reasoning traces AS an autonomous navigation "
    "robot that locates a SOUND SOURCE using your egocentric RGB frames and 4-channel FOA spatial "
    "audio. You are given YOUR OWN recent frames (oldest to newest, the last is your current view), "
    "the task the sound is asking you to do, the ground-truth direction and distance of the sound "
    "source, and the ground-truth decision: you have essentially ARRIVED — the source is within about "
    "a meter — so you STOP (either immediately, or after one or two short steps). Write a single "
    "coherent, physically plausible reasoning trace, STRICTLY IN THE FIRST PERSON (use 'I', 'me', "
    "'my'; NEVER say 'the robot' or describe yourself from the outside). The decision to stop and the "
    "source direction/distance are the SOURCE OF TRUTH: never contradict them, never plan to keep "
    "traveling far. Describe the visual scene ONLY from what you actually see in the frames. It is a "
    "sound-source task, so you SHOULD refer to the sound / where it comes from."
)

COT_TASK_TMPL = """Write a first-person chain-of-thought reasoning trace for my next navigation step.

=== What the sound is asking me to do ===
{instruction}

=== Ground-truth situation (do NOT contradict) ===
- The sound source is to my {src_dir}, at a distance of {dist:.1f} meters.
- I have essentially arrived: the source is (or will be, after {n_steps}) within about 1 meter of me.
- My two main meta-actions (first half of my path, then second half): {meta_seq}
- Decision: {decision}

=== What to write ===
Produce ONE first-person reasoning trace (a few sentences, natural English, no bullet headers, no markdown) that flows through these stages IN ORDER:
1. SCENE: briefly describe what I see right in front of me / around me in my current view (a wall, a doorway, furniture, a person, an open area, etc.). Use my past frames for context if helpful.
2. LOCALIZE: state, from the spatial audio, where the sound is coming from ({src_dir}) and that it is now very close — only about {dist:.1f} meters away.
3. ARRIVAL: conclude that I have (essentially) reached the sound source, so there is no need to keep traveling.
4. DECISION: end in ONE short first-person sentence that commits to {decision}, consistent with: {meta_seq}. Phrase it naturally (e.g. "so I will stop here" or "so I will take a couple of small steps forward and then stop"). Do NOT write a "Meta actions:" label, do NOT enumerate step-by-step sub-moves, and do NOT output any coordinates or the <stop> token. A separate canonical meta-action line and the trajectory will be added after your text.

Write everything in the FIRST PERSON. Do NOT refer to "the robot" in the third person. Do NOT output any numeric trajectory, coordinates, or the <stop> token. Stop at the meta-action level.

Output STRICT JSON and nothing else, exactly:
{{"cot": "<the first-person reasoning trace as a single string>"}}
"""


def build_task_prompt(pr, plan, src_dir):
    D = pr["source_distance"]
    if plan["stop_type"] == "direct":
        n_steps = "no more movement"
        decision = "stopping right here because I have already reached the sound source"
    else:
        k = len(plan["kept"])
        step_word = "one short step" if k == 1 else f"{k} short steps"
        n_steps = step_word
        decision = f"taking {step_word} forward to close the last bit of distance and then stopping"
    return COT_TASK_TMPL.format(
        instruction=pr["instruction"],
        src_dir=src_dir,
        dist=D,
        n_steps=n_steps,
        meta_seq=" then ".join(plan["meta_actions"]),
        decision=decision,
    )


def call_vlm(client, model, pr, task_prompt, max_history_imgs, timeout, retries):
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
            completion = client.chat.completions.create(model=model, messages=messages, timeout=timeout)
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


def fallback_cot(pr, plan, src_dir):
    D = pr["source_distance"]
    if plan["stop_type"] == "direct":
        tail = ("The sound is essentially right where I am standing, so I have reached the source and "
                "there is no need to move any further. So I will stop here.")
    else:
        k = len(plan["kept"])
        step_word = "one short step" if k == 1 else f"{k} short steps"
        tail = (f"I am almost on top of the source, so I only need {step_word} forward to close the "
                f"last bit of distance before I stop. So I will take {step_word} forward and then stop.")
    instr = pr["instruction"]
    instr = instr[:1].lower() + instr[1:] if instr else "reach the sound source"
    return (f"Looking at my current view, I can see the space right around me. My task is to {instr}. "
            f"From the spatial audio, the sound is coming from my {src_dir} at only about {D:.1f} meters, "
            f"which means I am very close to it. {tail}")


# =============================================================================

# =============================================================================
def process_one(rec, client, model, args):
    pr = parse_record(rec)
    if pr is None:
        return None, "bad_record"
    plan = build_stop_plan(pr, args)
    if plan is None:
        return None, "not_a_stop_candidate"

    src_dir = classify_direction8(net_heading_deg(pr["trajectory"]))

    if client is not None:
        task_prompt = build_task_prompt(pr, plan, src_dir)
        cot = call_vlm(client, model, pr, task_prompt, args.max_history_imgs, args.timeout, args.retries)
    else:
        cot = None
    if not cot:
        if args.allow_fallback or client is None:
            cot = fallback_cot(pr, plan, src_dir)
        else:
            return None, "vlm_failed"

    meta_line = "Meta actions: " + " then ".join(plan["meta_actions"])
    traj_str = traj_string(plan["traj_elems"])
    cot_block = f"<think>\n{cot}\n</think>" if args.think_tags else cot
    new_assistant = f"{cot_block}\n\n{meta_line}\n\n{traj_str}"

    out = dict(rec)
    out["messages"] = [rewrite_user_prompt(rec["messages"][0], args.think_tags),
                       {"role": "assistant", "content": new_assistant}]
    out["stop_type"] = plan["stop_type"]
    out["source_distance_m"] = pr["source_distance"]
    out["source_dir"] = src_dir
    out["kept_waypoints"] = len(plan["kept"])
    out["arrive_remaining_m"] = plan["arrive_remaining"]
    out["has_cot"] = True
    return out, "ok"


# =============================================================================

# =============================================================================
def sample_uid(rec):
    """OmniEcho processing component."""
    try:
        content = rec["messages"][0]["content"]
        imgs = [c["image"] for c in content if c.get("type") == "image" and c.get("image")]
        auds = [c["audio"] for c in content if c.get("type") == "audio" and c.get("audio")]
        cur = imgs[-1] if imgs else ""
        aud = auds[-1] if auds else ""
    except Exception:
        cur, aud = "", ""
    return cur + "||" + aud


def hash01(uid: str) -> float:
    """OmniEcho processing component."""
    h = int(hashlib.md5(uid.encode("utf-8")).hexdigest(), 16)
    return (h % 10_000_000) / 10_000_000.0


# =============================================================================

# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=DEFAULT_INPUT, help="Source sound-guided navigation CoT JSONL")
    ap.add_argument("--out_jsonl", default=None,
                    help=f"Output JSONL (default: {DEFAULT_OUT_DIR}/{DEFAULT_OUT_NAME}; adds .shardN when num-shards > 1)")
    ap.add_argument("--target", "-n", type=int, default=0,
                    help="Target sample count; <=0 selects all candidates (about 2703 satisfy the original max-dist filter)")
    
    ap.add_argument("--max-dist", type=float, default=DEFAULT_MAX_DIST,
                    help="Maximum true source-to-planning-start distance (default: 1.0 m); discard candidates beyond it")
    ap.add_argument("--max-keep", type=int, default=DEFAULT_MAX_KEEP,
                    help="Maximum coordinate waypoints to retain (default: 3, leaving at least 5 stop positions)")
    ap.add_argument("--keep-margin", type=float, default=DEFAULT_KEEP_MARGIN, help="Margin to the sound source when retaining waypoints")
    # VLM
    ap.add_argument("--model", default=DEFAULT_MODEL, help="VLM identifier; must support image input")
    ap.add_argument("--api_key", default=None, help="Annotation service API key (defaults to OPENAI_API_KEY)")
    ap.add_argument("--base_url", default=API_BASE_URL, help="OpenAI-compatible endpoint")
    ap.add_argument("--local", action="store_true", help="Use a local VLM with round-robin port selection")
    ap.add_argument("--local-model", default="OmniEcho")
    ap.add_argument("--local-host", default="127.0.0.1")
    ap.add_argument("--local-ports", default="8200,8201")
    ap.add_argument("--max-history-imgs", type=int, default=0, help="Maximum history frames sent to the API (0: current frame only)")
    ap.add_argument("--workers", type=int, default=8, help="Number of concurrent threads")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--max-samples", type=int, default=None, help="Keep only the first N candidates for debugging")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--think-tags", action="store_true", help="Wrap CoT in <think>...</think> (disabled by default)")
    ap.add_argument("--allow-fallback", action="store_true", help="Use deterministic templates on API failure or when no key is available")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    out_dir = DEFAULT_OUT_DIR
    out_path = args.out_jsonl or os.path.join(out_dir, DEFAULT_OUT_NAME)
    if args.num_shards > 1 and not args.out_jsonl:
        base, ext = os.path.splitext(out_path)
        out_path = f"{base}.shard{args.shard}{ext}"


    clients = [None]
    model = args.model
    try:
        from openai import OpenAI
    except Exception as e:
        OpenAI = None
        print(f"[warn] Cannot import openai ({e}); using template CoT.")

    if OpenAI is not None:
        if args.local:
            ports = [p.strip() for p in args.local_ports.split(",") if p.strip()]
            model = args.local_model
            clients = [OpenAI(base_url=f"http://{args.local_host}:{p}/v1",
                              api_key=(args.api_key or "EMPTY")) for p in ports]
            print(f"[info] Local VLM '{model}' @ " + ", ".join(f":{p}" for p in ports))
        else:
            api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
            if api_key:
                clients = [OpenAI(base_url=args.base_url, api_key=api_key)]
                print(f"[info] annotation service VLM '{model}' @ {args.base_url}")
            else:
                print("[warn] No API key and no --local; using template CoT (set --allow-fallback to select this explicitly).")
    if clients == [None] and not args.allow_fallback:
        print("[warn] No VLM available; forcing template CoT (equivalent to --allow-fallback).")


    if not os.path.exists(args.input):
        print(f"[fatal] Input does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    candidates = []          # (hash, uid, rec, stop_type)
    n_all = n_direct = n_move = 0
    with open(args.input) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            n_all += 1
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                continue
            pr = parse_record(rec)
            if pr is None or pr["source_distance"] is None:
                continue
            plan = build_stop_plan(pr, args)
            if plan is None:
                continue
            if plan["stop_type"] == "direct":
                n_direct += 1
            else:
                n_move += 1
            uid = sample_uid(rec)
            candidates.append((hash01(uid), uid, rec, plan["stop_type"]))
    candidates.sort(key=lambda x: x[0])
    print(f"[info] Source rows: {n_all}; stop candidates: {len(candidates)} (direct={n_direct}, move={n_move})")


    limit = args.target if args.target and args.target > 0 else len(candidates)
    picked, n_pick_direct = [], 0
    for h, uid, rec, st in candidates:
        if len(picked) >= limit:
            break
        if st == "direct":
            n_pick_direct += 1
        picked.append((uid, rec))
    print(f"[info] Selected {len(picked)} records (direct={n_pick_direct}, move={len(picked) - n_pick_direct}) -> {out_path}")


    picked = [(uid, r) for i, (uid, r) in enumerate(picked) if i % args.num_shards == args.shard]
    if args.max_samples:
        picked = picked[:args.max_samples]


    done = set()
    write_mode = "a"
    if args.no_resume:
        write_mode = "w"
    elif os.path.exists(out_path):
        with open(out_path) as f:
            for ln in f:
                try:
                    done.add(sample_uid(json.loads(ln)))
                except Exception:
                    continue
        print(f"[resume] Output exists; skipping {len(done)} completed records")

    todo = [(uid, r) for (uid, r) in picked if uid not in done]
    print(f"[info] Shard {args.shard}/{args.num_shards}: {len(todo)} pending records")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fout = open(out_path, write_mode)
    write_lock = threading.Lock()
    stats = {"ok": 0, "skip": 0}
    reasons = {}

    def worker(rec, client):
        return process_one(rec, client, model, args)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(worker, r, clients[i % len(clients)]): uid
                   for i, (uid, r) in enumerate(todo)}
        pbar = tqdm(as_completed(futures), total=len(futures), desc="stop-cot", unit="s", dynamic_ncols=True)
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
    if args.num_shards > 1:
        base = os.path.splitext(out_path)[0].rsplit(".shard", 1)[0]
        print(f"[note] After all shards finish, concatenate them, for example:\n"
              f"       cat {base}.shard*.jsonl > {os.path.join(out_dir, DEFAULT_OUT_NAME)}")


if __name__ == "__main__":
    main()
