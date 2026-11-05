#!/usr/bin/env python3


#
"""OmniEcho processing component."""
from __future__ import annotations

import argparse
import json
import os
from settings import DATA_ROOT as TRAJECTORY_ROOT, NAV_ROOT as OUTPUT_ROOT, MODEL_ROOT, API_BASE_URL as SERVICE_URL, API_MODEL
import re

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
OMNIECHO_NAV_DIR = OUTPUT_ROOT
DEFAULT_IN = os.path.join(OMNIECHO_NAV_DIR, "spatial_voice_vln_omni_sft_cot.jsonl")
DEFAULT_OUT = os.path.join(OMNIECHO_NAV_DIR, "vln_text_instruction_metaaction_from_cot.jsonl")

DATA_ROOT = os.path.join(TRAJECTORY_ROOT, "OmniEchoBench-Nav")

INSTRUCTION_SEP = "<INSTRUCTION_SEP>"
IMAGE_TOKEN = "<image>"
META_MARKER = "Meta actions:"


_SCENE_RE = re.compile(r"/images/([^/]+)/episode_")
_EP_RE = re.compile(r"/episode_(\d+)_\d+\.jpg$")

# =============================================================================

# =============================================================================
NAV_PROMPT_TEMPLATE = (
    "You are an autonomous navigation robot that follows natural-language navigation "
    "instructions. You perceive the environment through your recent egocentric RGB observations.\n"
    "These are your historical observations: {history}\n"
    "This is your current observation: {current}\n"
    "Navigation instruction: {instruction}\n"
    "Using the instruction together with your visual observations, plan where to go next. "
    "Act in two parts, in order:\n"
    "1. First, output your two main meta-actions — one for the first half of your path and one "
    "for the second half — in the form \"Meta actions: <first> then <second>\".\n"
    "2. Then, on a new line, output the future trajectory as a JSON list of waypoints relative to "
    "your current pose, ordered from the next step onward, each waypoint formatted as [x, y, yaw] "
    "where x is forward in meters, y is left in meters, and yaw is the heading in radians.\n"
    "Do not output any explanation, coordinates in the image, or intermediate steps."
)


def clean_text(prompt: str) -> str:
    t = prompt.replace(IMAGE_TOKEN, "")
    lines = [" ".join(line.split()) for line in t.splitlines()]
    return "\n".join(lines).strip()


def build_user_text(n_history: int, instruction: str) -> str:
    history_imgs = " ".join([IMAGE_TOKEN] * n_history) if n_history > 0 else "(none)"
    raw = NAV_PROMPT_TEMPLATE.format(
        history=history_imgs,
        current=IMAGE_TOKEN,
        instruction=instruction.strip().rstrip("."),
    )
    return clean_text(raw)


# =============================================================================

# =============================================================================
def load_instructions(scene):
    """OmniEcho processing component."""
    path = os.path.join(DATA_ROOT, scene, "meta", "episodes.jsonl")
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ep = int(r["episode_index"])
            tasks = r.get("tasks", [])
            raw = " ".join(tasks) if isinstance(tasks, list) else str(tasks)
            instrs = [s.strip() for s in raw.split(INSTRUCTION_SEP) if s.strip()]
            out[ep] = instrs
    return out


# =============================================================================

# =============================================================================
def extract_meta_and_traj(assistant: str):
    idx = assistant.find(META_MARKER)
    if idx < 0:
        return None, None
    block = assistant[idx:].strip()
    meta_line, _, _traj = block.partition("\n")
    meta_line = meta_line.strip()
    two_actions = None
    m = meta_line[len(META_MARKER):].strip() if meta_line.startswith(META_MARKER) else ""
    if m:
        two_actions = [s.strip() for s in m.split(" then ")]
    return block, two_actions


def parse_scene_ep(image_paths):
    scene = ep = None
    for p in image_paths:
        if scene is None:
            ms = _SCENE_RE.search(p)
            if ms:
                scene = ms.group(1)
        if ep is None:
            me = _EP_RE.search(p)
            if me:
                ep = int(me.group(1))
        if scene is not None and ep is not None:
            break
    return scene, ep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_jsonl", default=DEFAULT_IN)
    ap.add_argument("--out_jsonl", default=DEFAULT_OUT)
    ap.add_argument("--max", type=int, default=None, help="Maximum records to process for debugging")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)), exist_ok=True)

    instr_cache = {}
    n_in = n_out = 0
    n_no_scene = n_no_instr = n_no_meta = 0

    fout = open(args.out_jsonl, "w")
    with open(args.in_jsonl) as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            if args.max and n_in > args.max:
                n_in -= 1
                break
            try:
                rec = json.loads(line)
            except Exception:
                continue
            msgs = rec.get("messages", [])
            user = next((m for m in msgs if m.get("role") == "user"), None)
            asst = next((m for m in msgs if m.get("role") == "assistant"), None)
            if user is None or asst is None:
                continue

            content = user["content"] if isinstance(user["content"], list) else []
            
            image_paths = [it.get("image") for it in content if it.get("type") == "image"]
            if not image_paths:
                n_no_scene += 1
                continue

            scene, ep = parse_scene_ep(image_paths)
            if scene is None or ep is None:
                n_no_scene += 1
                continue

            if scene not in instr_cache:
                instr_cache[scene] = load_instructions(scene)
            instrs = instr_cache[scene].get(ep, [])
            if not instrs:
                n_no_instr += 1
                continue
            instruction = instrs[0]

            asst_str = asst["content"] if isinstance(asst["content"], str) else ""
            meta_block, two_actions = extract_meta_and_traj(asst_str)
            if meta_block is None:
                n_no_meta += 1
                continue

            n_history = max(0, len(image_paths) - 1)   
            new_content = [{"type": "image", "image": p} for p in image_paths]
            new_content.append({"type": "text", "text": build_user_text(n_history, instruction)})

            out_rec = {
                "messages": [
                    {"role": "user", "content": new_content},
                    {"role": "assistant", "content": meta_block},
                ],
                
                "dataset": "OmniEchoBench-Nav",
                "scene_id": scene,
                "episode_id": ep,
                "setting": "125cm_0deg",
                "instruction": instruction,
                "meta_actions": two_actions,
                "num_history": n_history,
                "source": "spatial_voice_vln_omni_sft_cot.jsonl",
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            n_out += 1
    fout.close()

    print(f"[done] Read: {n_in}  written: {n_out}")
    print(f"[skip] Missing image or unparseable scene/episode: {n_no_scene}  missing original instruction: {n_no_instr}  missing Meta actions: {n_no_meta}")
    print(f"[done] jsonl -> {args.out_jsonl}")


if __name__ == "__main__":
    main()
