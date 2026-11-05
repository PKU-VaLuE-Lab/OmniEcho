import os

from internnav.benchmark_paths import DEFAULT_SPLIT, episodes_directory
from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg, TaskCfg
import internnav.evaluator.habitat_vlnce_stuck_evaluator  # noqa: F401  (registers habitat_vlnce_stuck)

# =============================================================================
# OmniEchoBench-Nav spatial-audio navigation -- **pure-policy STOP variant** (oracle_stop_dist=0, disabling
# the "STOP once known within <=1m of the source" oracle shortcut). Tests whether, without the
# oracle shortcut, the model can decide it has arrived near the source and STOP **on its own**
# (via the spatial_audio_agent stationary / empty-trajectory STOP path).
# All other settings are **identical** to habitat_audio_cfg.py (same env / sidecar / evaluator /
# stuck detection / 500-step cap); the only difference is oracle_stop_dist: 1.0 -> 0.0.
# =============================================================================
EP_DIR = str(episodes_directory())
SPLIT = os.environ.get('OMNIECHO_NAV_SPLIT', DEFAULT_SPLIT)
AUDIO_JSONL = os.environ.get('OMNIECHO_NAV_AUDIO_SIDECAR',
                             os.path.join(EP_DIR, f'{SPLIT}_audio_sidecar.jsonl'))
SERVER_URL = os.environ.get('OMNIECHO_NAV_SERVER_URL', 'http://127.0.0.1:8100')
OUTPUT_PATH = os.environ.get('OMNIECHO_NAV_AUDIO_OUT', './logs/habitat_omniecho_nav_all/audio_nostop')

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='OmniEcho',
        ckpt_path='',  # weights live on the model server side
        model_settings={
            'server_url': SERVER_URL,
            'audio_jsonl': AUDIO_JSONL,
            'scratch_dir': os.path.join(OUTPUT_PATH, 'scratch'),
            'worker_id': 0,
            'num_workers': 1,
            'pred_jsonl': os.path.join(OUTPUT_PATH, 'pred_audio.jsonl'),
            'num_history': 8,
            'num_waypoints': 8,
            'exec_waypoints': 8,
            'max_new_tokens': 512,
            'stop_disp': 0.3,
            'max_replans': 50,
            'oracle_stop_dist': 0.0,   # disable the oracle shortcut: rely solely on the model to STOP
            'max_audio_sec': 10.0,
            'request_timeout': 600,
        },
    ),
    env=EnvCfg(
        env_type='habitat',
        env_settings={'config_path': 'scripts/eval/configs/vln_text.yaml'},
    ),
    task=TaskCfg(task_name='OmniEcho_eval', task_settings={}),
    eval_type='habitat_vlnce_stuck',
    eval_settings={
        'output_path': OUTPUT_PATH,
        'save_video': False,
        'epoch': 0,
        'max_steps_per_episode': 500,
        'stuck_window': 16,
        'stuck_eps': 0.25,
        'max_episodes': 0,
        'port': '2333',
        'dist_url': 'env://',
    },
)
