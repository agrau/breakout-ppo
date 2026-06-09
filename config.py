"""Central configuration for the Breakout PPO training system."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
EVOLUTION_LOG = os.path.join(MODELS_DIR, "evolution_log.json")
TRAINING_LOG_DIR = os.path.join(LOGS_DIR, "run")
TRAINING_CSV = os.path.join(TRAINING_LOG_DIR, "progress.csv")

ENV_ID = "BreakoutNoFrameskip-v4"   # was ALE/Breakout-v5: that has built-in frameskip=4 (→ effective 16 with AtariWrapper's skip=4) + 0.25 sticky actions, which capped both runs at ~30. NoFrameskip-v4 = clean frameskip=4, no sticky (matches the ~398 RL-Zoo benchmark).
N_ENVS = 4       # 8→4 to cut memory (env workers + rollout buffer); box has 4 cores + zram OOM pressure
USE_SUBPROC_ENV = True   # True = SubprocVecEnv (envs stepped in parallel processes; ~1.7x faster here, GPU was idle/CPU-bound)
N_STACK = 4      # stacked frames per observation
FULL_GAME = True  # train on continuous 5-life games (terminal_on_life_loss=False + fire-after-life) so the agent learns full-game play, not isolated lives
SEED = 42

# PPO hyperparameters tuned for Atari Breakout (same family as Pong)
PPO_CONFIG = {
    "n_steps": 128,
    "batch_size": 256,   # back to 256 with N_ENVS 8→4 (buffer 128*4=512 → 2 minibatches)
    "n_epochs": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.1,
    "clip_range_vf": None,
    "normalize_advantage": True,
    "vf_coef": 0.5,
    "ent_coef": 0.01,   # 0.04→0.01: exploit phase — let policy sharpen into tunneling after the retry plateaued at ~40
    "max_grad_norm": 0.5,
}

INITIAL_LR = 2.5e-4   # linearly decays to 0

# Breakout has sparser reward signal than Pong — needs more steps to play well
TOTAL_TIMESTEPS = 8_000_000   # warm-start full-game fine-tune from the 2M expert per-life model (~6M of full-game training; DummyVecEnv is slower)
CHECKPOINT_DIR = os.path.join(MODELS_DIR, "checkpoints")
CHECKPOINT_FREQ = 500_000   # save a full resumable checkpoint every N training steps (across all envs)
KEEP_LAST_CHECKPOINTS = 5   # prune older periodic checkpoints, keeping only this many newest
EVAL_FREQ = 100_000    # evaluate every N training steps (across all envs)
N_EVAL_EPISODES = 10   # episodes per evaluation
RECORD_VIDEO = False   # disabled as a memory precaution; render later with play.py
N_RECORD_EPISODES = 2  # gameplay episodes to record per checkpoint
VIDEO_FPS = 15   # Atari runs 60Hz w/ frameskip=4 → 15 FPS = real-time playback

# Auto-stop training when the agent enters this tier (set to None to disable)
# Disabled for full-game mode: episode reward is now per-GAME (5 lives), so the
# per-life tier thresholds (5/30/100) no longer apply — train the full budget.
STOP_AT_TIER = None
