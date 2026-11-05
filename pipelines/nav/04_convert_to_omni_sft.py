#!/usr/bin/env python3

#   python 04_convert_to_omni_sft.py
"""OmniEcho processing component."""
from __future__ import annotations

import argparse
import json
import os
from settings import DATA_ROOT as TRAJECTORY_ROOT, NAV_ROOT as OUTPUT_ROOT, MODEL_ROOT, API_BASE_URL as SERVICE_URL, API_MODEL

from tqdm import tqdm

NAV_ROOT = OUTPUT_ROOT

IMG_ROOT = TRAJECTORY_ROOT
IN_JSONL = os.path.join(NAV_ROOT, "spatial_voice_vln_full.jsonl")
OUT_JSONL = os.path.join(NAV_ROOT, "spatial_voice_vln_omni_sft.jsonl")


def clean_text(prompt: str) -> str:
    """OmniEcho processing component."""
    t = prompt.replace("<image>", "").replace("<audio>", "")
    lines = [" ".join(line.split()) for line in t.splitlines()]
    return "\n".join(lines).strip()


def build_record(r, img_root):
    content = []
    for rel in r.get("history_frame_paths", []):
        content.append({"type": "image", "image": os.path.join(img_root, rel)})
    content.append({"type": "image", "image": os.path.join(img_root, r["current_frame_path"])})
    content.append({"type": "audio", "audio": r["foa_audio_path"]})
    content.append({"type": "text", "text": clean_text(r["conversations"][0]["value"])})

    answer = json.dumps(r["future_trajectory"], ensure_ascii=False)
    return {"messages": [
        {"role": "user", "content": content},
        {"role": "assistant", "content": answer},
    ]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_jsonl", default=IN_JSONL)
    ap.add_argument("--out_jsonl", default=OUT_JSONL)
    ap.add_argument("--img_root", default=IMG_ROOT)
    ap.add_argument("--check_exist", action="store_true",
                    help="Check image/audio file existence for each sample (slow; intended for spot checks)")
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out_jsonl)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    total = sum(1 for _ in open(args.in_jsonl))

    n = 0
    n_img = 0
    n_missing = 0
    with open(args.in_jsonl) as fin, open(args.out_jsonl, "w") as fout:
        for line in tqdm(fin, total=total, desc="convert", unit="rec"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rec = build_record(r, args.img_root)
            if args.check_exist:
                for it in rec["messages"][0]["content"]:
                    p = it.get("image") or it.get("audio")
                    if p and not os.path.exists(p):
                        n_missing += 1
            n_img += sum(1 for it in rec["messages"][0]["content"] if it["type"] == "image")
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print(f"[done] Records: {n}")
    print(f"[done] Image content items: {n_img} (average {n_img / max(n,1):.1f}/sample)")
    if args.check_exist:
        print(f"[done] Missing files: {n_missing}")
    print(f"[done] Output -> {args.out_jsonl}")


if __name__ == "__main__":
    main()
