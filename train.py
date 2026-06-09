"""Train a PPO agent to play Breakout.

Usage:
    python3 train.py
    python3 train.py --timesteps 5000000
    python3 train.py --resume models/best_model.zip
"""
import argparse
import os

import ale_py
import gymnasium
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.logger import configure as configure_logger
from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecFrameStack, VecTransposeImage,
)

import config
from callbacks import EvolutionCallback, PruningCheckpointCallback

gymnasium.register_envs(ale_py)


def linear_schedule(initial: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial
    return func


def latest_checkpoint(checkpoint_dir: str):
    """Return the most-trained checkpoint (highest step count) or None."""
    if not os.path.isdir(checkpoint_dir):
        return None
    ckpts = [f for f in os.listdir(checkpoint_dir)
             if f.startswith("ckpt_") and f.endswith("_steps.zip")]
    if not ckpts:
        return None
    def steps(name):
        return int(name[len("ckpt_"):-len("_steps.zip")])
    return os.path.join(checkpoint_dir, max(ckpts, key=steps))


def make_envs(n_envs: int, seed: int):
    if getattr(config, "FULL_GAME", False):
        # Train on continuous 5-life games (learns to sustain across lives).
        import wrappers
        return wrappers.make_fullgame_train(
            n_envs, seed, subproc=getattr(config, "USE_SUBPROC_ENV", False))
    vec_cls = SubprocVecEnv if getattr(config, "USE_SUBPROC_ENV", False) else DummyVecEnv
    env = make_atari_env(config.ENV_ID, n_envs=n_envs, seed=seed, vec_env_cls=vec_cls)
    return VecFrameStack(env, n_stack=config.N_STACK)


def make_eval_env(seed: int):
    if getattr(config, "FULL_GAME", False):
        import wrappers
        return wrappers.make_fullgame_eval(seed + 1000)
    env = make_atari_env(config.ENV_ID, n_envs=1, seed=seed + 1000)
    env = VecFrameStack(env, n_stack=config.N_STACK)
    return VecTransposeImage(env)


def build_model(env, device: str):
    ppo_kwargs = dict(config.PPO_CONFIG)
    # Anneal clip_range 0.1 -> 0 over training (RL-Zoo recipe uses lin_0.1).
    ppo_kwargs["clip_range"] = linear_schedule(ppo_kwargs["clip_range"])
    return PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=linear_schedule(config.INITIAL_LR),
        verbose=1,
        device=device,
        **ppo_kwargs,
    )


def attach_csv_logger(model):
    os.makedirs(config.TRAINING_LOG_DIR, exist_ok=True)
    logger = configure_logger(config.TRAINING_LOG_DIR, ["stdout", "csv"])
    model.set_logger(logger)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=config.TOTAL_TIMESTEPS)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to model to resume from, or 'latest' for the newest checkpoint")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    os.makedirs(config.VIDEOS_DIR, exist_ok=True)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_env = make_envs(config.N_ENVS, config.SEED)
    eval_env = make_eval_env(config.SEED)

    resume_path = args.resume
    if resume_path == "latest":
        resume_path = latest_checkpoint(config.CHECKPOINT_DIR)
        if resume_path is None:
            raise SystemExit(f"No checkpoints found in {config.CHECKPOINT_DIR}")

    if resume_path:
        print(f"Resuming from: {resume_path}")
        model = PPO.load(resume_path, env=train_env, device=device)
        model.learning_rate = linear_schedule(config.INITIAL_LR)
        model.ent_coef = config.PPO_CONFIG["ent_coef"]   # apply current config on resume
        # Anneal clip_range toward 0 over the absolute budget (exploit phase: tighter
        # trust region as it converges). PPO calls self.clip_range(progress) each update.
        model.clip_range = linear_schedule(config.PPO_CONFIG["clip_range"])
        print(f"  Applied ent_coef={model.ent_coef}, clip_range now annealing from "
              f"{config.PPO_CONFIG['clip_range']} toward 0")
    else:
        model = build_model(train_env, device)

    attach_csv_logger(model)

    evolution_callback = EvolutionCallback(
        eval_env=eval_env,
        models_dir=config.MODELS_DIR,
        videos_dir=config.VIDEOS_DIR,
        evolution_log_path=config.EVOLUTION_LOG,
        record_video=config.RECORD_VIDEO,
        n_record_episodes=config.N_RECORD_EPISODES,
        stop_at_tier=getattr(config, "STOP_AT_TIER", None),
        eval_freq=max(config.EVAL_FREQ // config.N_ENVS, 1),
        n_eval_episodes=config.N_EVAL_EPISODES,
        deterministic=True,
        verbose=1,
    )

    # Periodic full-state checkpoints so a crash/reboot only loses work since the
    # last save, not since the last eval improvement.
    checkpoint_callback = PruningCheckpointCallback(
        save_freq=max(config.CHECKPOINT_FREQ // config.N_ENVS, 1),
        save_path=config.CHECKPOINT_DIR,
        name_prefix="ckpt",
        keep_last=getattr(config, "KEEP_LAST_CHECKPOINTS", None),
        verbose=1,
    )

    callback = CallbackList([checkpoint_callback, evolution_callback])

    # On resume, run only the *remaining* steps toward the absolute budget so the
    # LR schedule's horizon stays fixed at args.timesteps. SB3 does
    # `_total_timesteps = total_timesteps + num_timesteps` when reset_num_timesteps
    # is False, so passing the full budget here would keep pushing the decay
    # horizon out on every resume (leaving the LR stuck too high).
    if resume_path:
        learn_steps = max(args.timesteps - model.num_timesteps, 1)
        print(f"  Remaining steps to budget {args.timesteps:,}: {learn_steps:,}")
    else:
        learn_steps = args.timesteps

    print(f"\nTraining PPO on {config.ENV_ID} toward {args.timesteps:,} total steps")
    print(f"Checkpoints → {config.MODELS_DIR}")
    print(f"Videos      → {config.VIDEOS_DIR}")
    print(f"Evolution   → {config.EVOLUTION_LOG}\n")

    model.learn(
        total_timesteps=learn_steps,
        callback=callback,
        reset_num_timesteps=not bool(resume_path),
        tb_log_name="PPO_Breakout",
        progress_bar=True,
    )

    final_path = os.path.join(config.MODELS_DIR, "final_model")
    model.save(final_path)
    print(f"\nTraining complete. Final model saved to {final_path}.zip")

    train_env.close()
    eval_env.close()

    print("\nGenerating plots...")
    import visualize
    visualize.plot_evolution(show=False)
    visualize.plot_training_curves(show=False)


if __name__ == "__main__":
    main()
