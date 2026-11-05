#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Split any OmniEchoBench-Nav split evenly by **scene** into N independently runnable sub-splits, for
**true multi-GPU parallelism** in spatial-audio (audio) navigation.

Background: habitat_env sharding is per-scene (per_scene_eps[rank::world_size]) -- on the
smoke set (1 episode/scene) this degenerates (rank0 gets everything, rank>=1 gets 0), so
torchrun multi-GPU effectively runs on a single GPU. So instead of torchrun sharding, this
distributes whole scenes round-robin across N shards; each shard is run by an **independent
single-process worker** (each bound to 1 GPU with its own backend/proxy), achieving true
N-GPU parallelism.

Sharding by **scene** (rather than by episode index) means all episodes of the same scene
land in the same shard, so each scene GLB is loaded only once on a single GPU, avoiding
repeated cross-shard loading.

Writes to the episodes/ directory:
  {SPLIT}_g{i}.json.gz                 (habitat episodes, i=0..N-1)
  {SPLIT}_g{i}_audio_sidecar.jsonl     (aligned FOA sidecar)

Usage:
  python scripts/eval/spatial_audio/split_nway.py --split case3_smoke --n 2
  python scripts/eval/spatial_audio/split_nway.py --split case3_vln   --n 8
"""
import argparse
import gzip
import json
import os
import sys
from collections import OrderedDict, defaultdict

# Repo root derived from this file's location (scripts/eval/spatial_audio/ -> up 4 levels), so a relocated copy stays self-consistent.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _REPO)
from internnav.benchmark_paths import episodes_directory

EP_DIR = str(episodes_directory())


def scene_key(scene_id):
    # "omniecho_nav/scene_003/scene_003.glb" -> "scene_003"
    for part in scene_id.replace('.glb', '').split('/'):
        if part.startswith('scene_'):
            return part
    return scene_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='case3_smoke',
                    help='source split name (case3_smoke / case3_vln)')
    ap.add_argument('--n', type=int, required=True, help='number of shards = number of GPUs (2 or 8)')
    ap.add_argument('--ep_dir', default=EP_DIR)
    args = ap.parse_args()
    n = args.n
    assert n >= 1, 'n must be >=1'

    src = os.path.join(args.ep_dir, f'{args.split}.json.gz')
    d = json.load(gzip.open(src, 'rt'))
    eps = d['episodes']
    vocab = d.get('instruction_vocab')

    # Read the audio sidecar, indexed by (scene_key, episode_id)
    sc_path = os.path.join(args.ep_dir, f'{args.split}_audio_sidecar.jsonl')
    side = {}
    if os.path.exists(sc_path):
        with open(sc_path) as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                r = json.loads(ln)
                side[(scene_key(str(r['scene_id'])), int(r['episode_id']))] = ln

    # Group by scene (preserving order), then distribute whole scenes round-robin across N shards
    by_scene = OrderedDict()
    for e in eps:
        by_scene.setdefault(scene_key(e['scene_id']), []).append(e)

    if n > len(by_scene):
        ap.error(f"Requested {n} shards for only {len(by_scene)} scenes.")
    missing = [(scene_key(episode['scene_id']), int(episode['episode_id']))
               for episode in eps
               if (scene_key(episode['scene_id']), int(episode['episode_id'])) not in side]
    if missing:
        ap.error(f"Missing audio sidecar entries for {len(missing)} episodes.")

    shards = defaultdict(list)
    for j, (_scene, scene_eps) in enumerate(by_scene.items()):
        shards[j % n].extend(scene_eps)

    for i in range(n):
        sub = shards.get(i, [])
        out_split = f'{args.split}_g{i}'
        # episodes json.gz
        outp = os.path.join(args.ep_dir, f'{out_split}.json.gz')
        obj = {'episodes': sub}
        if vocab is not None:
            obj['instruction_vocab'] = vocab
        with gzip.open(outp, 'wt') as f:
            json.dump(obj, f)
        # audio sidecar
        outs = os.path.join(args.ep_dir, f'{out_split}_audio_sidecar.jsonl')
        n_side = 0
        with open(outs, 'w') as f:
            for e in sub:
                k = (scene_key(e['scene_id']), int(e['episode_id']))
                if k in side:
                    f.write(side[k] + '\n')
                    n_side += 1
        scenes = sorted({scene_key(e['scene_id']) for e in sub})
        print(f'[g{i}] {len(sub)} episodes over {len(scenes)} scenes, '
              f'{n_side} sidecar lines -> {out_split}')
        print(f'      scenes: {scenes}')


if __name__ == '__main__':
    main()
