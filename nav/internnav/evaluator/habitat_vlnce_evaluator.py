"""Habitat evaluation loop for OmniEchoBench-Nav."""
import argparse
import os

import numpy as np
import torch

import internnav.habitat_extensions.vln.measures  # noqa: F401  (habitat registry side effects)
from internnav.configs.evaluator import EvalCfg
from internnav.evaluator import DistributedEvaluator, Evaluator

try:
    import habitat
    from habitat.config.default import get_agent_config
    from habitat.config.default_structured_configs import (
        CollisionsMeasurementConfig,
        FogOfWarConfig,
        TopDownMapMeasurementConfig,
    )
    from habitat_baselines.config.default import get_config as get_habitat_config
except Exception as e:  # pragma: no cover
    print("Habitat Error:", e)


@Evaluator.register('habitat_vlnce')
class HabitatVLNCEEvaluator(DistributedEvaluator):
    def __init__(self, cfg: EvalCfg):
        args = argparse.Namespace(**cfg.eval_settings)
        self.args = args
        self.save_video = getattr(args, 'save_video', False)
        self.epoch = getattr(args, 'epoch', 0)
        self.max_steps_per_episode = getattr(args, 'max_steps_per_episode', 500)
        self.max_episodes = getattr(args, 'max_episodes', 0)  # 0 = all
        self.output_path = args.output_path

        self.config_path = cfg.env.env_settings['config_path']
        self.config = get_habitat_config(self.config_path)
        self.agent_config = get_agent_config(self.config.habitat.simulator)
        self.sim_sensors_config = self.config.habitat.simulator.agents.main_agent.sim_sensors

        with habitat.config.read_write(self.config):
            self.config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_padding=3,
                        map_resolution=1024,
                        draw_source=True,
                        draw_border=True,
                        draw_shortest_path=True,
                        draw_view_points=True,
                        draw_goal_positions=True,
                        draw_goal_aabbs=True,
                        fog_of_war=FogOfWarConfig(draw=True, visibility_dist=5.0, fov=90),
                    ),
                    "collisions": CollisionsMeasurementConfig(),
                }
            )
        cfg.env.env_settings['habitat_config'] = self.config
        cfg.env.env_settings['output_path'] = self.output_path

        super().__init__(cfg)

    def eval_action(self):
        import json
        import traceback

        sucs, spls, oss, nes, tls = [], [], [], [], []
        env = self.env
        n_done = 0


        
        
        _only = os.environ.get('OMNIECHO_ONLY_EPISODES', '').strip()
        only_eps = {int(x) for x in _only.replace(',', ' ').split()} if _only else None
        if only_eps is not None:
            print(f"[habitat_vlnce] OMNIECHO_ONLY_EPISODES enabled; running only episode_id in {sorted(only_eps)}", flush=True)

        def _agent_pos():
            
            return np.asarray(env._env.sim.get_agent_state().position, dtype=np.float64)


        
        os.makedirs(self.output_path, exist_ok=True)
        jsonl_path = os.path.join(self.output_path, f"episodes_rank{self.rank}.jsonl")
        jsonl_f = open(jsonl_path, "w")

        def _record(scene, ep_id, steps, m, tl, error=None):
            sucs.append(m["success"])
            spls.append(m["spl"])
            oss.append(m["oracle_success"])
            nes.append(m["distance_to_goal"])
            tls.append(tl)
            rec = {
                "rank": int(self.rank), "n": len(sucs), "scene": scene, "episode_id": ep_id,
                "steps": steps, "success": float(m["success"]), "spl": float(m["spl"]),
                "oracle_success": float(m["oracle_success"]),
                "distance_to_goal": float(m["distance_to_goal"]),
                "trajectory_length": float(tl),
            }
            if error is not None:
                rec["error"] = str(error)
            jsonl_f.write(json.dumps(rec) + "\n")
            jsonl_f.flush()
            os.fsync(jsonl_f.fileno())  

        while env.is_running:
            if self.max_episodes and n_done >= self.max_episodes:
                break
            obs = env.reset()
            if not env.is_running or obs is None:
                break

            episode = env.get_current_episode()
            scene = episode.scene_id.split('/')[-1]
            ep_id = episode.episode_id
            if only_eps is not None and int(ep_id) not in only_eps:
                continue  
            try:
                self.agent.reset(episode, env)
                done = False
                step_id = 0
                tl = 0.0                      # trajectory length in meters
                prev_pos = _agent_pos()
                while not done and step_id <= self.max_steps_per_episode:
                    action = self.agent.act(obs, env, info=None)
                    obs, reward, done, info = env.step(action)
                    cur_pos = _agent_pos()
                    tl += float(np.linalg.norm(cur_pos - prev_pos))  
                    prev_pos = cur_pos
                    step_id += 1

                m = env.get_metrics()
                _record(scene, ep_id, step_id, m, tl)
                n_done += 1
                print(f"[habitat_vlnce] ep {n_done} (scene={scene} "
                      f"id={ep_id}) steps={step_id} "
                      f"succ={m['success']:.0f} spl={m['spl']:.3f} "
                      f"ne={m['distance_to_goal']:.2f} tl={tl:.2f}", flush=True)
            except Exception as e:


                
                traceback.print_exc()
                m = {"success": 0.0, "spl": 0.0, "oracle_success": 0.0,
                     "distance_to_goal": float('nan')}
                _record(scene, ep_id, -1, m, float('nan'), error=e)
                n_done += 1
                print(f"[habitat_vlnce] Episode {n_done} (scene={scene} id={ep_id}) failed and was skipped: {e}", flush=True)

        jsonl_f.close()
        env.close()
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        to_t = lambda x: torch.tensor(x, dtype=torch.float32, device=dev)  # noqa: E731
        return {"sucs": to_t(sucs), "spls": to_t(spls), "oss": to_t(oss),
                "nes": to_t(nes), "tls": to_t(tls)}

    def calc_metrics(self, global_metrics: dict) -> dict:
        sucs, spls, oss, nes, tls = (global_metrics[k] for k in ("sucs", "spls", "oss", "nes", "tls"))
        n = len(sucs)
        return {
            
            "success": float(sucs.mean().item()) if n else 0.0,          # SR
            "spl": float(spls.mean().item()) if n else 0.0,              # SPL
            "oracle_success": float(oss.mean().item()) if n else 0.0,    # OS
            
            "distance_to_goal": float(torch.nanmean(nes).item()) if n else 0.0,      # NE
            "trajectory_length": float(torch.nanmean(tls).item()) if n else 0.0,     # TL
        }
