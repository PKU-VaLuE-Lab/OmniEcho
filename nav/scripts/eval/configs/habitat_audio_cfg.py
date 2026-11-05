import os

from internnav.benchmark_paths import DEFAULT_SPLIT, episodes_directory
from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg, TaskCfg
import internnav.evaluator.habitat_vlnce_stuck_evaluator  # noqa: F401  (registers habitat_vlnce_stuck)

# =============================================================================
# OmniEchoBench-Nav **spatial-audio-guided navigation** evaluation (OmniEcho, via the /infer server).
# Runs the **exact same unified case set** (same start/goal) as the 5 text VLN models; the only
# difference is that localization uses real 4ch FOA plus the sidecar's task description,
# rather than a route instruction. It selects the recording
# at the nearest recording point at each step.
#   * env = vln_text.yaml -- same episodes as the VLN models (split injected via OMNIECHO_NAV_SPLIT);
#   * audio_jsonl = published audio sidecar (same episode_id as the corresponding split);
#   * eval_type = habitat_vlnce_stuck -- TL + stuck-stop when net 16-step displacement < 0.25m;
#   * Multi-GPU: torchrun shards episodes by rank at the framework level, so the agent's internal worker sharding is disabled (num_workers=1).
# =============================================================================
EP_DIR = str(episodes_directory())
SPLIT = os.environ.get('OMNIECHO_NAV_SPLIT', DEFAULT_SPLIT)
# Audio sidecar shares the split name (case3_vln_audio_sidecar.jsonl / case3_smoke_audio_sidecar.jsonl)
AUDIO_JSONL = os.environ.get('OMNIECHO_NAV_AUDIO_SIDECAR',
                             os.path.join(EP_DIR, f'{SPLIT}_audio_sidecar.jsonl'))
SERVER_URL = os.environ.get('OMNIECHO_NAV_SERVER_URL', 'http://127.0.0.1:8100')
OUTPUT_PATH = os.environ.get('OMNIECHO_NAV_AUDIO_OUT', './logs/habitat_omniecho_nav_all/audio')

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='OmniEcho',
        ckpt_path='',  # weights live on the model server side
        model_settings={
            'server_url': SERVER_URL,
            'audio_jsonl': AUDIO_JSONL,
            'scratch_dir': os.path.join(OUTPUT_PATH, 'scratch'),
            'worker_id': 0,
            'num_workers': 1,          # torchrun already shards by rank; disable the agent's internal sharding
            'pred_jsonl': os.path.join(OUTPUT_PATH, 'pred_audio.jsonl'),
            'num_history': 8,
            'num_waypoints': 8,
            'exec_waypoints': 8,
            'max_new_tokens': 512,
            'stop_disp': 0.3,
            'max_replans': 50,   # upper bound on inference calls; whichever stop condition is reached first applies
            'oracle_stop_dist': 1.0,   # STOP once known within <=1m of the source (1m success criterion)
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
        'max_steps_per_episode': 500,   # hard cap on steps per episode
        'stuck_window': 16,            # net displacement < 0.25m over 16 simulator actions -> stuck STOP
        'stuck_eps': 0.25,
        'max_episodes': 0,
        'port': '2333',
        'dist_url': 'env://',
    },
)
