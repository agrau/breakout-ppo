"""Evaluate a saved model with RAW full-game scoring (no reward clipping, full
5-life games) — the same metric the RL Zoo benchmark (~398) reports.

Uses wrappers.make_fullgame_env, which fires after each life loss so the agent
(trained with auto-fire-per-life) can actually play all 5 lives. Run ONLY when
the trainer is stopped (two PPO processes exhaust this box's RAM)."""
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from wrappers import make_fullgame_env

path = sys.argv[1] if len(sys.argv) > 1 else "models/best_model.zip"
n_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 10

env = make_fullgame_env(clip_reward=False)
model = PPO.load(path, env=env, device="cuda")
print(f"Model: {path}  num_timesteps={model.num_timesteps:,}")
rewards, lengths = evaluate_policy(model, env, n_eval_episodes=n_ep,
                                   deterministic=True, return_episode_rewards=True)
rewards = np.array(rewards)
print(f"RAW full-game score over {n_ep} games: mean={rewards.mean():.1f} +/- {rewards.std():.1f}")
print(f"  min={rewards.min():.0f}  max={rewards.max():.0f}")
print(f"  scores={[int(r) for r in rewards]}")
print(f"benchmark (RL Zoo PPO 10M steps) = 398 +/- 16")
env.close()
