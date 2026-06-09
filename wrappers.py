"""Inference-time helpers for playing FULL Breakout games.

The policy trains with SB3's EpisodicLifeEnv, which auto-fires a new ball at the
start of every life. In a real continuous game (terminal_on_life_loss=False)
nothing auto-fires after a life is lost, and the agent never learned to press
FIRE itself, so it idles after the first death. FireAfterLifeLoss injects a FIRE
action whenever a life is lost (but the game isn't over) so the ball relaunches,
letting the trained agent play all 5 lives.
"""
import ale_py
import gymnasium as gym
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecFrameStack, VecTransposeImage,
)

import config

gym.register_envs(ale_py)

FIRE = 1  # Breakout action index for FIRE / serve ball


class FireAfterLifeLoss(gym.Wrapper):
    """Press FIRE to relaunch the ball after each lost life in a full game."""

    def __init__(self, env):
        super().__init__(env)
        self._lives = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._lives = info.get("lives", self.env.unwrapped.ale.lives())
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        lives = info.get("lives", self._lives)
        # A life was lost but the game continues -> serve a new ball.
        if 0 < lives < self._lives and not (terminated or truncated):
            obs, r2, terminated, truncated, info = self.env.step(FIRE)
            reward += r2
            lives = info.get("lives", lives)
        self._lives = lives
        return obs, reward, terminated, truncated, info


def make_fullgame_env(clip_reward=False):
    """Single full-game (5-life) Breakout env with standard Atari preprocessing
    + frame stack the policy expects, plus FireAfterLifeLoss so the agent can
    relaunch the ball after each death. clip_reward=False -> raw game score."""
    def thunk():
        e = gym.make(config.ENV_ID)
        e = AtariWrapper(e, terminal_on_life_loss=False, clip_reward=clip_reward)
        e = FireAfterLifeLoss(e)
        return e
    venv = DummyVecEnv([thunk])
    venv = VecFrameStack(venv, n_stack=config.N_STACK)
    return VecTransposeImage(venv)


# --- Full-game TRAINING env (terminal_on_life_loss=False -> the agent optimizes
# the true 5-life return and learns to sustain across lives; FireAfterLifeLoss
# auto-serves so it doesn't waste steps idle). Module-level factory so it is
# picklable for SubprocVecEnv. clip_reward=True for training stability. ---
def _make_atari_fullgame():
    e = gym.make(config.ENV_ID)
    e = AtariWrapper(e, terminal_on_life_loss=False, clip_reward=True)
    e = FireAfterLifeLoss(e)
    return e


def make_fullgame_train(n_envs, seed, subproc=False):
    # NOTE: SubprocVecEnv deadlocks with FireAfterLifeLoss (a worker hangs the
    # pipe), so force DummyVecEnv here — reliable, fine for a warm-start fine-tune.
    venv = make_vec_env(_make_atari_fullgame, n_envs=n_envs, seed=seed, vec_env_cls=DummyVecEnv)
    return VecFrameStack(venv, n_stack=config.N_STACK)


def make_fullgame_eval(seed):
    venv = make_vec_env(_make_atari_fullgame, n_envs=1, seed=seed)
    venv = VecFrameStack(venv, n_stack=config.N_STACK)
    return VecTransposeImage(venv)
