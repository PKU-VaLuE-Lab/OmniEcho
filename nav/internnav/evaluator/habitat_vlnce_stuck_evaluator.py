"""VLN-CE baseline evaluator with **stuck detection** and no practical step cap.

Subclasses ``HabitatVLNCEEvaluator`` (registered as ``habitat_vlnce``) so nothing
existing is touched. Used for the OmniEchoBench-Nav basic VLN test where — matching VLN-CE
stop conditions — an episode ends only when:

  * the policy autonomously emits STOP, **or**
  * the agent makes no net progress for ``stuck_window`` steps (< ``stuck_eps`` m net
    displacement) — a forced STOP is then issued so Success/SPL are scored at that pose,
  * (backstop) ``max_steps_per_episode`` / the sim's ``max_episode_steps`` is reached.

Net-displacement-over-a-window (first-vs-last of the window) flags turning-in-place,
wall-bumping (collisions freeze GPS), and back-and-forth oscillation alike.

Registered as ``habitat_vlnce_stuck``. Pair with the ``vlnce_habitat`` agent.
"""
import numpy as np
import torch

from internnav.configs.evaluator import EvalCfg
from internnav.evaluator import Evaluator
# Import the base module for its registration side effect, then fetch the class from the
# registry: ``@Evaluator.register`` returns None, so the module-level name is None and we
# must subclass via ``Evaluator.evaluators`` instead of importing the symbol directly.
import internnav.evaluator.habitat_vlnce_evaluator  # noqa: F401  (registers 'habitat_vlnce')

_HabitatVLNCEEvaluator = Evaluator.evaluators['habitat_vlnce']


@Evaluator.register('habitat_vlnce_stuck')
class HabitatVLNCEStuckEvaluator(_HabitatVLNCEEvaluator):
    def __init__(self, cfg: EvalCfg):
        es = cfg.eval_settings
        self.stuck_window = int(es.get('stuck_window', 4))
        self.stuck_eps = float(es.get('stuck_eps', 0.25))
        super().__init__(cfg)

    @staticmethod
    def _xy(obs, env):
        """Agent planar position for progress tracking. Prefer the gps lab-sensor
        (configured in the yaml); fall back to the sim agent state."""
        try:
            g = obs.get('gps') if isinstance(obs, dict) else None
            if g is not None:
                g = np.asarray(g, dtype=np.float64).reshape(-1)
                return g[:2].copy()
        except Exception:
            pass
        try:
            p = env.sim.get_agent_state().position
            return np.array([float(p[0]), float(p[2])], dtype=np.float64)
        except Exception:
            return None

    def eval_action(self):
        sucs, spls, oss, nes, tls = [], [], [], [], []
        env = self.env
        n_done = 0
        W, EPS = self.stuck_window, self.stuck_eps

        while env.is_running:
            if self.max_episodes and n_done >= self.max_episodes:
                break
            obs = env.reset()
            if not env.is_running or obs is None:
                break

            episode = env.get_current_episode()
            self.agent.reset(episode, env)

            done = False
            step_id = 0
            stopped_reason = 'policy'
            hist = []
            tl = 0.0                     # trajectory length (m): sum of consecutive step displacements
            p0 = self._xy(obs, env)
            if p0 is not None:
                hist.append(p0)

            while not done and step_id <= self.max_steps_per_episode:
                action = self.agent.act(obs, env, info=None)
                # stuck: no net progress across the last W positions -> force STOP
                if action != 'stop' and len(hist) > W:
                    net = float(np.linalg.norm(hist[-1] - hist[-1 - W]))
                    if net < EPS:
                        action = 'stop'
                        stopped_reason = 'stuck'
                obs, reward, done, info = env.step(action)
                step_id += 1
                p = self._xy(obs, env)
                if p is not None:
                    if hist:
                        tl += float(np.linalg.norm(p - hist[-1]))
                    hist.append(p)

            m = env.get_metrics()
            sucs.append(m["success"])
            spls.append(m["spl"])
            oss.append(m["oracle_success"])
            nes.append(m["distance_to_goal"])
            tls.append(tl)
            n_done += 1
            print(f"[habitat_vlnce_stuck] ep {n_done} (scene={episode.scene_id.split('/')[-1]} "
                  f"id={episode.episode_id}) steps={step_id} stop={stopped_reason} "
                  f"succ={m['success']:.0f} spl={m['spl']:.3f} ne={m['distance_to_goal']:.2f} "
                  f"tl={tl:.2f}",
                  flush=True)

        env.close()
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        to_t = lambda x: torch.tensor(x, dtype=torch.float32, device=dev)  # noqa: E731
        return {"sucs": to_t(sucs), "spls": to_t(spls), "oss": to_t(oss),
                "nes": to_t(nes), "tls": to_t(tls)}

    def calc_metrics(self, global_metrics: dict) -> dict:
        # extend the base metrics with TL (mean trajectory length in meters)
        res = super().calc_metrics(global_metrics)
        tls = global_metrics.get("tls")
        if tls is not None and len(tls) > 0:
            res["trajectory_length"] = float(tls.mean().item())
        return res
