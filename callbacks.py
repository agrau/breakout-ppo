"""Evolution-tracking callback: saves a named checkpoint the first time the agent
enters each skill tier (Beginner → Learning → Skilled → Expert)."""
import json
import os
import time

import ale_py
import cv2
import gymnasium
import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

import config

gymnasium.register_envs(ale_py)

# (name, lower_bound, color_code, description)
SKILL_TIERS = [
    ("beginner",   0, "\033[91m", "Random paddle motion, barely hits the ball"),
    ("learning",   5, "\033[93m", "Clears bottom rows, loses lives quickly"),
    ("skilled",   30, "\033[96m", "Clears most of the wall, keeps the ball alive"),
    ("expert",   100, "\033[92m", "Digs tunnels and clears the wall consistently"),
]


def _tier_for_reward(reward: float):
    """Return the highest tier (name, color, description) the reward qualifies for."""
    current = SKILL_TIERS[0]
    for tier in SKILL_TIERS:
        if reward >= tier[1]:
            current = tier
    return current


class PruningCheckpointCallback(CheckpointCallback):
    """CheckpointCallback that keeps only the `keep_last` most-trained checkpoints,
    deleting older ones so periodic saves don't fill the disk."""

    def __init__(self, *args, keep_last=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.keep_last = keep_last

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.keep_last and self.n_calls % self.save_freq == 0:
            self._prune()
        return result

    def _prune(self):
        suffix = "_steps.zip"
        prefix = f"{self.name_prefix}_"
        ckpts = [f for f in os.listdir(self.save_path)
                 if f.startswith(prefix) and f.endswith(suffix)]
        if len(ckpts) <= self.keep_last:
            return
        ckpts.sort(key=lambda n: int(n[len(prefix):-len(suffix)]))
        for old in ckpts[:-self.keep_last]:
            try:
                os.remove(os.path.join(self.save_path, old))
            except OSError:
                pass


class EvolutionCallback(EvalCallback):
    """Saves a named checkpoint the first time the agent enters each skill tier."""

    def __init__(self, eval_env, models_dir, videos_dir, evolution_log_path,
                 record_video=True, n_record_episodes=2, stop_at_tier=None, **kwargs):
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(videos_dir, exist_ok=True)
        super().__init__(
            eval_env,
            best_model_save_path=models_dir,
            log_path=os.path.join(models_dir, "eval_log"),
            **kwargs,
        )
        self.models_dir = models_dir
        self.videos_dir = videos_dir
        self.evolution_log_path = evolution_log_path
        self.record_video = record_video
        self.n_record_episodes = n_record_episodes
        self.stop_at_tier = stop_at_tier
        self.evolution_history = []
        self._reached_tiers: set = set()
        self._last_eval_count = 0
        self._should_stop = False
        self._load_existing_log()

    def _load_existing_log(self):
        """Restore tier state from a prior run so resume doesn't re-save tiers."""
        if not os.path.exists(self.evolution_log_path):
            return
        try:
            with open(self.evolution_log_path) as f:
                prior = json.load(f)
            self.evolution_history = prior
            self._reached_tiers = {e["tier"] for e in prior}
            if self._reached_tiers:
                print(f"[Resume] Already-reached tiers: {sorted(self._reached_tiers)}")
        except (json.JSONDecodeError, KeyError):
            pass

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.evaluations_results and len(self.evaluations_results) > self._last_eval_count:
            self._last_eval_count = len(self.evaluations_results)
            mean_reward = float(np.mean(self.evaluations_results[-1]))
            std_reward = float(np.std(self.evaluations_results[-1]))
            self._maybe_save_tier(mean_reward, std_reward)
        if self._should_stop:
            print(f"\n[Auto-stop] Target tier '{self.stop_at_tier}' reached — ending training.")
            return False
        return result

    def _maybe_save_tier(self, mean_reward: float, std_reward: float):
        tier_name, tier_lower, color, description = _tier_for_reward(mean_reward)
        if tier_name in self._reached_tiers:
            return
        self._reached_tiers.add(tier_name)

        name = f"tier_{tier_name}_t{self.num_timesteps}_r{int(round(mean_reward)):+d}"
        model_path = os.path.join(self.models_dir, name)
        self.model.save(model_path)

        video_path = None
        if self.record_video:
            video_path = os.path.join(self.videos_dir, f"{name}.mp4")
            self._record_video(video_path)

        entry = {
            "tier": tier_name,
            "description": description,
            "timesteps": int(self.num_timesteps),
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "model_path": model_path + ".zip",
            "video_path": video_path,
            "wall_time": time.time(),
        }
        self.evolution_history.append(entry)
        self._save_log()

        if self.stop_at_tier and tier_name == self.stop_at_tier:
            self._should_stop = True

        print(
            f"\n{color}[Evolution → {tier_name.upper()}] "
            f"{mean_reward:+.2f} ± {std_reward:.2f} @ {self.num_timesteps:,} steps  "
            f"— {description}\033[0m"
        )
        if video_path:
            print(f"  Video → {video_path}")

    def _record_video(self, video_path: str):
        try:
            vec_env = make_atari_env(
                config.ENV_ID, n_envs=1, seed=0,
                env_kwargs={"render_mode": "rgb_array"},
            )
            vec_env = VecFrameStack(vec_env, n_stack=config.N_STACK)
            vec_env = VecTransposeImage(vec_env)
            frames = []
            obs = vec_env.reset()

            for _ in range(self.n_record_episodes):
                done = False
                while not done:
                    frame = vec_env.venv.envs[0].render()
                    if frame is not None:
                        frames.append(frame)
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, dones, _ = vec_env.step(action)
                    done = bool(dones[0])
                obs = vec_env.reset()

            vec_env.close()

            if frames:
                h, w = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(video_path, fourcc, config.VIDEO_FPS, (w, h))
                for frame in frames:
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                writer.release()
        except Exception as exc:
            print(f"  [Warning] Video recording failed: {exc}")

    def _save_log(self):
        with open(self.evolution_log_path, "w") as f:
            json.dump(self.evolution_history, f, indent=2)
