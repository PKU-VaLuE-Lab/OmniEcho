#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Merge per-shard results from N-GPU parallel spatial-audio (audio) navigation -> global SR/SPL/OS/NE/TL.

Each shard's worker (OMNIECHO_NAV_SPLIT={SPLIT}_g{i}) writes the global habitat_vlnce_stuck metrics to
  {out_root}/{SPLIT}_g{i}/audio/result.json
fields: success / spl / oracle_success / distance_to_goal / trajectory_length / length.
Since each field is the mean over that shard's episodes, a length-weighted average yields the
**exact** overall mean.

Usage:
  python scripts/eval/spatial_audio/merge_audio_parallel.py --split case3_smoke --n 2
  python scripts/eval/spatial_audio/merge_audio_parallel.py --split case3_vln --n 8 \
      --out_root logs/habitat_omniecho_nav_all
"""
import argparse
import json
import os


def load_result(path):
    if not os.path.exists(path):
        return None
    last = None
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                try:
                    last = json.loads(ln)
                except Exception:
                    pass
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='case3_smoke')
    ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--out_root', default='logs/habitat_omniecho_nav_all')
    ap.add_argument('--model', default='audio',
                    help='model subdirectory name written by the worker (audio / audio_nostop)')
    args = ap.parse_args()
    if args.n < 1:
        ap.error("n must be positive")

    keys = ['success', 'spl', 'oracle_success', 'distance_to_goal', 'trajectory_length']
    tot_n = 0
    acc = {k: 0.0 for k in keys}
    per_shard = []
    for i in range(args.n):
        rp = os.path.join(args.out_root, f'{args.split}_g{i}', args.model, 'result.json')
        r = load_result(rp)
        if r is None:
            raise SystemExit(f"Missing result for shard {i}: {rp}; refusing partial aggregate.")
        nn = int(r.get('length', 0) or 0)
        if nn <= 0 or any(r.get(key) is None for key in keys):
            raise SystemExit(f'Incomplete metrics for shard {i}; refusing aggregate.')
        per_shard.append((i, r))
        tot_n += nn
        for k in keys:
            v = r.get(k)
            if v is not None:
                acc[k] += float(v) * nn

    print(f'\naudio parallel merge -- split={args.split}  N={args.n}  (out_root: {args.out_root})')
    print(f"{'shard':8s} {'N':>4s} {'SR%':>7s} {'SPL%':>7s} {'OS%':>7s} {'NE(m)':>7s} {'TL(m)':>7s}")
    print('-' * 52)
    for i, r in per_shard:
        if r is None:
            print(f'g{i:<7d}  -- no result --')
            continue
        nn = int(r.get('length', 0) or 0)
        print(f"g{i:<7d} {nn:4d} {r['success']*100:7.2f} {r['spl']*100:7.2f} "
              f"{r['oracle_success']*100:7.2f} {r['distance_to_goal']:7.2f} "
              f"{r['trajectory_length']:7.2f}")
    print('-' * 52)
    if tot_n > 0:
        g = {k: acc[k] / tot_n for k in keys}
        print(f"{'ALL':8s} {tot_n:4d} {g['success']*100:7.2f} {g['spl']*100:7.2f} "
              f"{g['oracle_success']*100:7.2f} {g['distance_to_goal']:7.2f} "
              f"{g['trajectory_length']:7.2f}")
        out_dir = os.path.join(args.out_root, args.split, args.model)
        os.makedirs(out_dir, exist_ok=True)
        merged = {'success': g['success'], 'spl': g['spl'],
                  'oracle_success': g['oracle_success'],
                  'distance_to_goal': g['distance_to_goal'],
                  'trajectory_length': g['trajectory_length'], 'length': tot_n}
        with open(os.path.join(out_dir, 'result.json'), 'w') as f:
            f.write(json.dumps(merged) + '\n')
        print(f"\n[merge] merged result written to {os.path.join(out_dir, 'result.json')} "
              f"(split={args.split})")
    else:
        print('(no valid shard results)')
    print()


if __name__ == '__main__':
    main()
