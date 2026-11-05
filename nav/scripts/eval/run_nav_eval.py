#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate OmniEcho sound-guided navigation in an InternNav/Habitat environment."""
import argparse
import importlib
import importlib.util
import os
import sys

sys.path.append('.')
sys.path.append('./third_party/diffusion-policy')
sys.path.append('./internnav/model/encoder')

from internnav.benchmark_paths import DEFAULT_SPLIT, episodes_directory, scenes_directory

# Each model: (config relative path, list of modules to import to trigger registration)
MODELS = {'audio': ('scripts/eval/configs/habitat_audio_cfg.py', ['internnav.agent.spatial_audio_agent']), 'audio_nostop': ('scripts/eval/configs/habitat_audio_nostop_cfg.py', ['internnav.agent.spatial_audio_agent'])}


def patch_habitat_config(local_rank, split):
    """Replace get_habitat_config referenced in evaluator modules with a wrapped version
       that, after loading, sets sim gpu_device_id by rank and overrides dataset.split."""
    import habitat

    def make_wrapper(orig):
        def wrapper(config_path):
            cfg = orig(config_path)
            with habitat.config.read_write(cfg):
                try:
                    cfg.habitat.simulator.habitat_sim_v0.gpu_device_id = int(local_rank)
                except Exception as e:
                    print(f"[run_omniecho_nav] warn: set gpu_device_id failed: {e}", flush=True)
                if split:
                    cfg.habitat.dataset.split = split
                cfg.habitat.dataset.data_path = str(episodes_directory() / '{split}.json.gz')
                cfg.habitat.dataset.scenes_dir = str(scenes_directory())
            return cfg
        return wrapper

    for mod_name in ('internnav.evaluator.habitat_vlnce_evaluator',
                     'internnav.habitat_extensions.vln.habitat_vln_evaluator'):
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        if hasattr(mod, 'get_habitat_config') and not getattr(mod, '_omniecho_nav_patched', False):
            mod.get_habitat_config = make_wrapper(mod.get_habitat_config)
            mod._omniecho_nav_patched = True
            print(f"[run_omniecho_nav] patched get_habitat_config in {mod_name} "
                  f"(gpu={local_rank}, split={split})", flush=True)


def load_eval_cfg(config_path):
    spec = importlib.util.spec_from_file_location("omniecho_nav_eval_cfg_mod", config_path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["omniecho_nav_eval_cfg_mod"] = m
    spec.loader.exec_module(m)
    return m.eval_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS.keys()))
    args = ap.parse_args()

    split = os.environ.get('OMNIECHO_NAV_SPLIT', DEFAULT_SPLIT)
    out_root = os.environ.get('OMNIECHO_NAV_OUT_ROOT', 'logs/habitat_omniecho_nav_all')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    model = args.model

    cfg_path, reg_modules = MODELS[model]

    # 1) Override habitat_config (rank->gpu, split). Must import evaluator modules first so the symbol exists.
    patch_habitat_config(local_rank, split)

    # 2) Import agent registration modules (evaluator registration is triggered by the import at the top of each cfg)
    for mn in reg_modules:
        importlib.import_module(mn)

    # 3) Load cfg, override output directory (per split/model to avoid progress.json cross-contamination)
    eval_cfg = load_eval_cfg(cfg_path)
    out_path = os.path.join(out_root, split, model)
    eval_cfg.eval_settings['output_path'] = out_path
    # the audio agent's scratch/pred paths also follow the output directory
    if model in ('audio', 'audio_nostop'):
        ms = eval_cfg.agent.model_settings
        ms['scratch_dir'] = os.path.join(out_path, 'scratch')
        ms['pred_jsonl'] = os.path.join(out_path, 'pred_audio.jsonl')
    os.makedirs(out_path, exist_ok=True)

    from internnav.evaluator import Evaluator
    print(f"[run_omniecho_nav] model={model} split={split} out={out_path} "
          f"rank={os.environ.get('RANK','0')}/{os.environ.get('WORLD_SIZE','1')} "
          f"local_rank={local_rank}", flush=True)
    evaluator = Evaluator.init(eval_cfg)
    evaluator.eval()


if __name__ == '__main__':
    main()
