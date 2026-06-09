"""Throughput benchmark: DummyVecEnv vs SubprocVecEnv for the training setup.
Measures env-steps/sec for a predict->step rollout loop, matching train.py."""
import time

import ale_py
import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecFrameStack, VecTransposeImage,
)

import config

gymnasium.register_envs(ale_py)


def make_env(vec_cls):
    e = make_atari_env(config.ENV_ID, n_envs=config.N_ENVS, seed=0, vec_env_cls=vec_cls)
    e = VecFrameStack(e, n_stack=config.N_STACK)
    return VecTransposeImage(e)


def bench(vec_cls, name, warmup=10, iters=150):
    env = make_env(vec_cls)
    model = PPO("CnnPolicy", env, device="cuda", verbose=0)
    obs = env.reset()
    for _ in range(warmup):
        a, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = env.step(a)
    t0 = time.perf_counter()
    for _ in range(iters):
        a, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = env.step(a)
    dt = time.perf_counter() - t0
    env.close()
    steps = iters * config.N_ENVS
    print(f"{name:>16s}: {steps/dt:7.1f} env-steps/s   ({dt:.1f}s for {steps} steps)")
    return steps / dt


def main():
    print(f"N_ENVS={config.N_ENVS}  (live training is running -> Subproc result is a lower bound)\n")
    d = bench(DummyVecEnv, "DummyVecEnv")
    s = bench(SubprocVecEnv, "SubprocVecEnv")
    print(f"\nSpeedup (Subproc / Dummy): {s/d:.2f}x")


if __name__ == "__main__":
    main()
