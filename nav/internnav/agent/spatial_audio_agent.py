"""Habitat spatial-audio navigation agent for real-recorded spatial audio (OmniEcho).

The key difference from ``spatial_audio_omni``: instead of synthesizing FOA from
mono audio + a geometric direction, this agent uses real-recorded 4-channel FOA
recordings:

  * At each decision step, take the robot's real-time world position and select the
    real FOA recording from the nearest recording point (rec_pos);
  * The recording's reference orientation is real +x (forward) / real +y (left),
    world-aligned; the robot's heading is not necessarily along real +x, so the
    horizontal directional components of the FOA must be rotated into the robot's
    local frame (forward = local x, left = local y).
    Correct transform: first map the real-frame directional components into the
    glb (habitat) frame with A^-1, then project onto the robot's glb ego basis
    (forward fwd_glb / left left_glb):
        [Xg; Zg] = A^-1 @ [X_real; Y_real]
        X' = [Xg,Zg] . fwd_glb        (forward)
        Y' = [Xg,Zg] . left_glb       (left)          (W, Z unchanged)
    i.e. M = B @ A^-1 (rows of B = habitat ego basis fwd_glb/left_glb),
    [X';Y'] = M @ [X_real;Y_real].
    Habitat is a mirror world of real (det(A)<0), so the sound field must be
    brought into the mirror frame via A^-1 to keep left/right consistent with the
    (mirrored) visual observation; projecting real FOA directly onto the real-frame
    robot basis (A@basis) does not undo the mirroring and flips left/right at wide
    angles. (Verified against ground truth: this M exactly reproduces the habitat
    ego azimuth at any heading.)
  * No distance re-calibration (use the loudness/reverberation of the real
    recording directly);
  * The prompt / three-part output contract matches ``spatial_audio_omni`` (same
    model); the task instruction (speech / sound categories) comes from the
    sidecar's ``task_instruction``, and the prompt contains no target-position
    description.

Coordinate chain (inlined in the converter-produced sidecar): habitat world == GLB
(Y-up identity); real(x,y) = A @ [glb_x, glb_z] + t. Direction transforms use the
linear part A (ignoring translation); after normalization only rotation/mirroring
remains.

Registered as ``OmniEcho``; reuses the ``habitat_vlnce`` evaluator.
"""
import json
import math
import os

import cv2
import numpy as np
import requests
import torch  # noqa: F401  (as with the other agents, ensures the cuda context / dependencies are ready)

from habitat_sim.utils.common import quat_rotate_vector

from internnav.agent.base import Agent
from internnav.benchmark_paths import resolve_recording_path

ACT_STOP, ACT_FWD, ACT_LEFT, ACT_RIGHT = "stop", "move_forward", "turn_left", "turn_right"

FORWARD_STEP = 0.25
TURN_RAD = np.radians(15)

# Prompt text fragments, aligned verbatim with the training data.
# The training samples split the prompt into interleaved content parts: text(intro)

# text("spatial audio ...") -> audio -> text(Task ...).
# So this agent does not concatenate one text block; it builds content_parts in the
# same order for the /infer server to render.


NAV_INTRO = (
    "You are an autonomous navigation robot that locates and moves toward a "
    "sound source. You perceive the environment through your recent egocentric "
    "RGB observations and a 4-channel first-order ambisonics (FOA, ACN/SN3D) "
    "spatial audio clip captured at your current position; the spatial audio "
    "encodes the direction and distance of the sound source relative to you.\n"
    "These are your historical observations:"
)
NAV_CUR_OBS = "This is your current observation:"
NAV_AUDIO_LINE = "This is the spatial audio recorded at your current position:"
# {n} = number of output trajectory waypoints (typically 8 in training), aligned
# verbatim with the "exactly N waypoints" phrasing in training.
NAV_TASK = (
    "Task (what the sound is asking you to do): {instruction}\n"
    "Using the spatial audio together with the visual observations, plan where "
    "to go next to reach the sound source and complete the task. Then act in "
    "three parts, in order:\n"
    "1. First, reason step by step about what you see in your observations, the "
    "direction and distance of the sound source, and how to move toward it "
    "while avoiding obstacles.\n"
    "2. Next, on a new line, output your two main meta-actions — one for the "
    "first half of your path and one for the second half — in the form "
    "\"Meta actions: <first> then <second>\".\n"
    "3. Finally, on a new line, output the future trajectory as a JSON list of "
    "exactly {n} waypoints relative to your current pose, ordered from the next "
    "step onward, each waypoint formatted as [x, y, yaw] where x is forward in "
    "meters, y is left in meters, and yaw is the heading in radians."
)
# Stuck mitigation (output prefill): if the previous step was a turn-in-place, the
# robot has already turned to face the sound source, so this step prefills the
# assistant output with this prefix, letting the model continue its reasoning under
# the premise that "the sound source is in front" and move forward.
NAV_OUTPUT_PREFIX_FRONT = "I hear the sound coming from in front of me."


def _build_content(history_images, current_image, foa_path, instruction, num_waypoints):
    """Build interleaved content_parts aligned verbatim with the training data.
    Returns (content_parts, question_text). question_text is only for writing to /
    inspecting pred_jsonl."""
    intro = NAV_INTRO if history_images else NAV_INTRO + " (none)"
    parts = [{"type": "text", "value": intro}]
    for p in history_images:
        parts.append({"type": "image", "value": p})
    parts.append({"type": "text", "value": NAV_CUR_OBS})
    parts.append({"type": "image", "value": current_image})
    parts.append({"type": "text", "value": NAV_AUDIO_LINE})
    parts.append({"type": "audio", "value": foa_path})
    parts.append({"type": "text",
                  "value": NAV_TASK.format(instruction=instruction, n=num_waypoints)})
    question = "\n".join(p["value"] for p in parts if p["type"] == "text")
    return parts, question


@Agent.register('OmniEcho')
class SpatialAudioOmniEchoAgent(Agent):
    def __init__(self, agent_config):
        super().__init__(agent_config)
        ms = dict(agent_config.model_settings)
        self.server_url = ms.get('server_url', 'http://127.0.0.1:8100').rstrip('/')
        self.jsonl_path = ms['audio_jsonl']            # omniecho_nav_agent_sidecar.jsonl
        self.scratch_dir = ms.get('scratch_dir', './logs/habitat_omniecho_nav_spatial_audio/scratch')
        self.num_history = int(ms.get('num_history', 8))
        # Number of trajectory waypoints the model outputs; written into the
        # "exactly N waypoints" prompt (aligned with training, typically 8).
        self.num_waypoints = int(ms.get('num_waypoints', 8))
        self.exec_waypoints = int(ms.get('exec_waypoints', 8))
        self.max_new_tokens = int(ms.get('max_new_tokens', 512))
        self.stop_disp = float(ms.get('stop_disp', 0.3))
        self.max_replans = int(ms.get('max_replans', 10))   # hard cap on decision steps per episode; STOP when reached even if the goal is not attained
        self.oracle_stop_dist = float(ms.get('oracle_stop_dist', 0.0))  # 0=disabled
        self.max_audio_sec = float(ms.get('max_audio_sec', 10.0))       # FOA clip length in seconds
        self.request_timeout = float(ms.get('request_timeout', 600))
        self.clean_scratch = bool(ms.get('clean_scratch', False))
        # Two-GPU data parallelism: this worker only handles episodes where
        # ep_counter % num_workers == worker_id; other episodes are marked done in
        # reset() -> STOP at step 1 (no model request, negligible overhead).
        self.worker_id = int(ms.get('worker_id', 0))
        self.num_workers = int(ms.get('num_workers', 1))
        # Write one record per step (per model call) in the reference format for later
        # inspection. A separate file per worker avoids write contention.
        self.pred_jsonl = ms.get('pred_jsonl', '')
        os.makedirs(self.scratch_dir, exist_ok=True)
        if self.pred_jsonl:
            os.makedirs(os.path.dirname(self.pred_jsonl), exist_ok=True)

        self.meta = self._load_jsonl(self.jsonl_path)
        self._foa_cache = {}
        self._pred_index = 0     # pred_jsonl record index (incremented within this worker)
        self._ep_counter = -1    # episode counter (this worker's iteration order, used for worker sharding)
        self._wait_server_ready()
        self._reset_state()

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _load_jsonl(path):
        meta = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                meta[(str(r['scene_id']), int(r['episode_id']))] = r
        return meta

    def _wait_server_ready(self, retries=120, interval=10):
        import time
        for i in range(retries):
            try:
                r = requests.get(self.server_url + '/health', timeout=10)
                if r.ok:
                    print(f"[OmniEcho] server ready: {r.json()}", flush=True)
                    return
            except Exception:
                pass
            if i == 0:
                print(f"[OmniEcho] waiting for model service {self.server_url} to be ready ...", flush=True)
            time.sleep(interval)
        raise RuntimeError(f"model service not ready: {self.server_url}")

    def _load_foa(self, audio_path):
        """Read a real 4-channel FOA recording -> (sr, float64 [N,4]); channel order [W,Y,Z,X] (ACN/SN3D)."""
        if audio_path not in self._foa_cache:
            from scipy.io import wavfile
            sr, x = wavfile.read(audio_path)
            x = np.asarray(x)
            if x.ndim == 1:
                x = x[:, None]
            if np.issubdtype(x.dtype, np.integer):
                x = x.astype(np.float64) / np.iinfo(x.dtype).max
            else:
                x = x.astype(np.float64)
            self._foa_cache[audio_path] = (int(sr), x)
        return self._foa_cache[audio_path]

    # ------------------------------------------------------------------ state
    def _reset_state(self):
        self._action_queue = []
        self._history_frames = []
        self._replan_id = 0
        self._done = False
        self._ep_tag = 'ep'
        self._turn_only_streak = 0   # count of consecutive turn-in-place replans; used to force forward motion to avoid getting stuck
        self._prev_turn_only = False  # whether the previous replan was a turn-in-place (decides whether to add the turn hint this step)

    def reset(self, episode=None, env=None):
        prev_tag = getattr(self, '_ep_tag', None)
        self._reset_state()
        if self.clean_scratch and prev_tag:
            import glob
            for f in glob.glob(os.path.join(self.scratch_dir, f"{prev_tag}_*")):
                try:
                    os.remove(f)
                except OSError:
                    pass
        scene_short = self._scene_short(episode.scene_id)
        ep_id = int(episode.episode_id)
        self._ep_tag = f"{scene_short}_ep{ep_id}"
        # Worker sharding: skip episodes not assigned to this worker (STOP at step 1, no model request).
        self._ep_counter += 1
        if self.num_workers > 1 and (self._ep_counter % self.num_workers) != self.worker_id:
            self._done = True
            self._recs, self._source_glb, self._task_instr = [], None, ""
            print(f"[OmniEcho] {self._ep_tag}: not this worker's "
                  f"(id={self.worker_id}/{self.num_workers}) task, skipping", flush=True)
            return
        rec = self.meta.get((scene_short, ep_id))
        if rec is None:
            print(f"[OmniEcho] warning: sidecar missing ({scene_short},{ep_id}), STOP directly", flush=True)
            self._done = True
            self._recs, self._source_glb, self._task_instr = [], None, ""
            return
        cs = rec['coordinate_system']['forward']
        self._A = np.array(cs['A'], dtype=np.float64)          # 2x2, real = A@[glb_x,glb_z]+t
        self._t = np.array(cs['t'], dtype=np.float64)
        self._recs = rec['recordings']                          # [{glb:[x,y,z], audio_path, real}]
        self._rec_glb = np.array([r['glb'] for r in self._recs], dtype=np.float64)  # [K,3]
        self._source_glb = np.asarray(episode.goals[0].position, dtype=np.float64)
        self._task_instr = (rec.get('task_instruction', '') or '').strip().rstrip('.')

    @staticmethod
    def _scene_short(scene_id):
        base = scene_id.split('/')[-1]
        return base[:-4] if base.endswith('.glb') else scene_id.split('/')[-2]

    # ------------------------------------------------------------ pose / heading
    @staticmethod
    def _unit2(v):
        n = float(np.hypot(v[0], v[1]))
        return v / n if n > 1e-9 else np.array([1.0, 0.0])

    def _pose(self, env):
        """Return (glb horizontal position [x,z], robot forward fwd_glb unit vector,
        left left_glb unit vector, heading angle psi in radians (log only), horizontal
        distance to the sound source in m). fwd/left are 2D unit vectors in the
        glb (habitat) frame, used for the FOA rotation (M = B @ A^-1)."""
        st = env._env.sim.get_agent_state()
        pos = np.asarray(st.position, dtype=np.float64)        # habitat world == GLB
        fwd = np.asarray(quat_rotate_vector(st.rotation, np.array([0.0, 0.0, -1.0])),
                         dtype=np.float64)
        left = np.asarray(quat_rotate_vector(st.rotation, np.array([-1.0, 0.0, 0.0])),
                          dtype=np.float64)
        glb_xz = np.array([pos[0], pos[2]], dtype=np.float64)
        fwd_glb = self._unit2(np.array([fwd[0], fwd[2]], dtype=np.float64))
        left_glb = self._unit2(np.array([left[0], left[2]], dtype=np.float64))
        # psi is log-only: the robot forward heading angle in the real frame
        f_real = self._unit2(self._A @ fwd_glb)
        psi = math.atan2(f_real[1], f_real[0])
        dsrc = float(np.hypot(pos[0] - self._source_glb[0], pos[2] - self._source_glb[2]))
        return glb_xz, fwd_glb, left_glb, psi, dsrc

    def _nearest_rec(self, glb_xz):
        d = np.hypot(self._rec_glb[:, 0] - glb_xz[0], self._rec_glb[:, 2] - glb_xz[1])
        i = int(np.argmin(d))
        return i, float(d[i])

    def _render_foa(self, rec_idx, fwd_glb, left_glb, out_path):
        """Take the real 4-channel FOA from the nearest recording point, rotate its
        horizontal directional components into the robot's habitat ego frame, then
        truncate and write a wav.

        M = B @ A^-1, rows of B = [fwd_glb; left_glb] (glb ego basis), A^-1 = the
        real->glb linear part. [X';Y'] = M @ [X_real; Y_real]. Bringing the sound
        field into the mirror (glb) frame via A^-1 keeps left/right consistent with
        the mirrored visual observation.
        """
        from scipy.io import wavfile
        audio_path = resolve_recording_path(self._recs[rec_idx]['audio_path'])
        sr, foa = self._load_foa(audio_path)                   # [N,4] = [W,Y,Z,X]
        n = foa.shape[0]
        if self.max_audio_sec > 0:
            n = min(n, int(self.max_audio_sec * sr))
        foa = foa[:n].copy()
        A_inv = np.linalg.inv(self._A)                          # real -> glb (linear)
        B = np.stack([fwd_glb, left_glb], axis=0)               # 2x2, rows = glb ego basis
        M = B @ A_inv                                           # [X';Y'] = M @ [X_real;Y_real]
        X = foa[:, 3].copy()   # real +x component (X_real)
        Y = foa[:, 1].copy()   # real +y component (Y_real)
        foa[:, 3] = M[0, 0] * X + M[0, 1] * Y   # X' = robot forward component
        foa[:, 1] = M[1, 0] * X + M[1, 1] * Y   # Y' = robot left component
        # W (0), Z (2) unchanged
        wavfile.write(out_path, sr, foa.astype(np.float32))
        return out_path

    def _save_frame(self, obs, tag):
        rgb = np.asarray(obs['rgb'])
        path = os.path.join(self.scratch_dir, f"{self._ep_tag}_{tag}.jpg")
        cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return path

    # ------------------------------------------------------ trajectory -> discrete actions
    @staticmethod
    def _traj_to_actions(traj):
        actions = []
        px, py, pyaw = 0.0, 0.0, 0.0
        for wp in traj:
            x, y, yaw = float(wp[0]), float(wp[1]), float(wp[2])
            dpos = np.hypot(x - px, y - py)
            dyaw = (yaw - pyaw + np.pi) % (2 * np.pi) - np.pi
            if dpos > 0.5 * FORWARD_STEP:
                actions += [ACT_FWD] * max(1, int(round(dpos / FORWARD_STEP)))
            elif abs(dyaw) > 0.5 * TURN_RAD:
                turn = ACT_LEFT if dyaw > 0 else ACT_RIGHT
                actions += [turn] * max(1, int(round(abs(dyaw) / TURN_RAD)))
            px, py, pyaw = x, y, yaw
        return actions

    # ------------------------------------------------------------------ act
    def act(self, obs, env=None, info=None):
        if self._done:
            return ACT_STOP
        if self._action_queue:
            return self._action_queue.pop(0)

        # Hard cap on decision steps: after max_replans replans without STOP -> stop here (counted as failure if the goal is not reached).
        if self._replan_id >= self.max_replans:
            _, _, _, _, dsrc = self._pose(env)
            print(f"[OmniEcho] {self._ep_tag}: reached decision-step cap {self.max_replans} -> STOP "
                  f"(dist_to_src={dsrc:.2f})", flush=True)
            self._done = True
            return ACT_STOP

        glb_xz, fwd_glb, left_glb, psi, dsrc = self._pose(env)
        if self.oracle_stop_dist > 0 and dsrc <= self.oracle_stop_dist:
            print(f"[OmniEcho] {self._ep_tag} r{self._replan_id}: reached sound source STOP "
                  f"(dist_to_src={dsrc:.2f} <= {self.oracle_stop_dist})", flush=True)
            self._done = True
            return ACT_STOP

        rec_idx, rec_dist = self._nearest_rec(glb_xz)
        cur_frame = self._save_frame(obs, f"r{self._replan_id}")
        foa_path = self._render_foa(
            rec_idx, fwd_glb, left_glb,
            os.path.join(self.scratch_dir, f"{self._ep_tag}_r{self._replan_id}_foa.wav"))

        hist = self._history_frames
        if len(hist) > self.num_history:
            idx = np.unique(np.linspace(0, len(hist) - 1, self.num_history, dtype=int))
            hist = [hist[i] for i in idx]
        images = hist + [cur_frame]  # log only

        # Interleaved content_parts aligned verbatim with the training data (incl. "exactly N waypoints").
        content_parts, question = _build_content(
            hist, cur_frame, foa_path, self._task_instr, self.num_waypoints)
        # If the previous step was a turn-in-place -> prefill the assistant output with
        # the "sound source is in front" prefix, letting the model continue reasoning
        # and move forward (breaking the spinning loop); otherwise -> no prefill
        # (strictly the training format).
        assistant_prefix = NAV_OUTPUT_PREFIX_FRONT if self._prev_turn_only else None
        traj, raw = self._call_server(content_parts, assistant_prefix)
        self._log_pred(images, foa_path, question, raw)
        self._history_frames.append(cur_frame)
        self._replan_id += 1

        if not traj:
            print(f"[OmniEcho] {self._ep_tag} r{self._replan_id-1}: empty trajectory -> STOP "
                  f"(dist_to_src={dsrc:.2f}) raw={raw[:200]!r}", flush=True)
            self._done = True
            return ACT_STOP

        disp = max(np.hypot(wp[0], wp[1]) for wp in traj)
        # Max magnitude of the target heading relative to the current heading (for a
        # turn-in-place trajectory disp is ~0 but yaw is significant).
        max_turn = max(abs(float(wp[2])) for wp in traj)
        actions = self._traj_to_actions(traj[: self.exec_waypoints])
        # Arrival check: only STOP when both displacement and turn are very small (the
        # model really wants to stay put) or there are no executable actions at all.
        # A turn-in-place (disp<stop_disp but significant max_turn) must execute the
        # turn to face the sound source and must not be misjudged as arrival --
        # otherwise a bad starting heading would STOP immediately at step 1.
        stationary = disp < self.stop_disp and max_turn < TURN_RAD
        if not actions or stationary:
            print(f"[OmniEcho] {self._ep_tag} r{self._replan_id-1}: arrival check STOP "
                  f"(disp={disp:.2f} max_turn={math.degrees(max_turn):.0f}deg "
                  f"dist_to_src={dsrc:.2f} nactions={len(actions)})", flush=True)
            self._done = True
            return ACT_STOP

        # Stuck avoidance: if this replan is a turn-in-place (no forward action, only
        # turns), accumulate the consecutive count; two consecutive turn-in-place
        # replans => assume spinning in place, force one forward step after the turn
        # to break the stall, then replan and observe.
        turn_only = ACT_FWD not in actions
        self._prev_turn_only = turn_only     # used next step to decide whether to add the turn hint
        if turn_only:
            self._turn_only_streak += 1
        else:
            self._turn_only_streak = 0
        if self._turn_only_streak >= 2:
            actions = actions + [ACT_FWD]
            print(f"[OmniEcho] {self._ep_tag} r{self._replan_id-1}: {self._turn_only_streak} "
                  f"consecutive turn-in-place replans -> force one forward step after turning", flush=True)
            self._turn_only_streak = 0

        print(f"[OmniEcho] {self._ep_tag} r{self._replan_id-1}: dist_to_src={dsrc:.2f} "
              f"nearest_rec={self._recs[rec_idx]['id']}({rec_dist:.2f}m) psi={math.degrees(psi):.0f}deg "
              f"disp={disp:.2f} max_turn={math.degrees(max_turn):.0f}deg "
              f"waypoints={len(traj)} actions={len(actions)}", flush=True)
        self._action_queue = actions
        return self._action_queue.pop(0)

    def _log_pred(self, images, foa_path, prompt, raw):
        """Write one inference record in the reference prediction format for later
        inspection. Extra ep_tag/replan/worker fields are added for locating records
        without affecting the reference fields."""
        if not self.pred_jsonl:
            return
        rec = {
            "index": self._pred_index,
            "images": list(images),
            "media": images[-1] if images else "",
            "audio": foa_path,
            "n_images": len(images),
            "question": prompt,
            "prediction": raw,
            "has_spatial": 1,
            "ep_tag": self._ep_tag,
            "replan": self._replan_id,
            "worker": self.worker_id,
        }
        try:
            with open(self.pred_jsonl, 'a') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError as e:
            print(f"[OmniEcho] failed to write pred_jsonl: {e}", flush=True)
        self._pred_index += 1

    def _call_server(self, content_parts, assistant_prefix=None):
        try:
            payload = {"content_parts": content_parts,
                       "max_new_tokens": self.max_new_tokens}
            if assistant_prefix:
                payload["assistant_prefix"] = assistant_prefix
            r = requests.post(
                self.server_url + '/infer',
                json=payload,
                timeout=self.request_timeout,
            )
            r.raise_for_status()
            d = r.json()
            return d.get("trajectory", []), (d.get("raw", "") or "")
        except Exception as e:
            print(f"[OmniEcho] model service request failed: {e}", flush=True)
            return [], ""
