"""Watch saved model versions play Breakout.

Usage:
    python3 play.py                            # show evolution menu
    python3 play.py --tier expert              # watch the 'expert' tier
    python3 play.py --all                      # record every tier in sequence
    python3 play.py --model path/model.zip     # play any saved model
    python3 play.py --compare                  # side-by-side first vs best (saves video)
"""
import argparse
import json
import os

import ale_py
import cv2
import gymnasium
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

import config

gymnasium.register_envs(ale_py)


def load_history():
    if not os.path.exists(config.EVOLUTION_LOG):
        print("No evolution log found. Run train.py first.")
        return []
    with open(config.EVOLUTION_LOG) as f:
        return json.load(f)


def load_model_and_env(model_path: str):
    env = make_atari_env(config.ENV_ID, n_envs=1, seed=0,
                          env_kwargs={"render_mode": "rgb_array"})
    env = VecFrameStack(env, n_stack=config.N_STACK)
    env = VecTransposeImage(env)
    model = PPO.load(model_path, env=env)
    return model, env


def collect_frames(model, vec_env, n_episodes: int = 1, max_steps: int = 10000,
                   target_frames: int = None):
    """Record gameplay frames. If target_frames is set, keep starting new games
    back-to-back until at least that many frames are captured (target seconds x
    VIDEO_FPS) — so short games (e.g. a weak tier that dies fast) are
    concatenated into a longer clip. Otherwise play exactly n_episodes."""
    frames = []
    obs = vec_env.reset()
    episode = 0
    while True:
        done = False
        steps = 0
        while not done and steps < max_steps:
            frame = vec_env.venv.envs[0].render()
            if frame is not None:
                frames.append(frame)
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = vec_env.step(action)
            done = bool(dones[0])
            steps += 1
            if target_frames is not None and len(frames) >= target_frames:
                obs = vec_env.reset()
                return frames
        obs = vec_env.reset()
        episode += 1
        if target_frames is None and episode >= n_episodes:
            return frames


def save_video(frames, path, fps=config.VIDEO_FPS):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"Saved → {path}")


def play_tier(entry, n_episodes=1, target_frames=None):
    path = entry["model_path"]
    if not os.path.exists(path):
        print(f"Model file not found: {path}")
        return
    span = (f"~{target_frames / config.VIDEO_FPS:.0f}s of games"
            if target_frames else f"{n_episodes} game(s)")
    print(f"\nTier {entry['tier'].upper()}  "
          f"steps={entry['timesteps']:,}  reward={entry['mean_reward']:+.2f}  [{span}]")
    print(f"  {entry['description']}")
    model, vec_env = load_model_and_env(path)
    frames = collect_frames(model, vec_env, n_episodes=n_episodes, target_frames=target_frames)
    vec_env.close()
    out = os.path.join(config.VIDEOS_DIR, f"play_tier_{entry['tier']}.mp4")
    save_video(frames, out)
    return frames


def menu(history):
    print("\n=== Saved Tiers ===")
    for e in history:
        print(f"  [{e['tier']:>9s}] "
              f"{e['timesteps'] / 1e6:5.1f}M steps  "
              f"reward={e['mean_reward']:+6.2f}  "
              f"{'[video]' if e.get('video_path') else ''}")
    print()
    choice = input("Tier name to watch (or 'all', 'q' to quit): ").strip().lower()
    if choice == "q":
        return
    if choice == "all":
        for e in history:
            play_tier(e)
        return
    entry = next((e for e in history if e["tier"] == choice), None)
    if entry:
        play_tier(entry, n_episodes=2)
    else:
        print("Tier not found.")


def _resolve_baseline(history, ref):
    """Resolve --baseline (a tier name or a model file path) to a tier-like entry.
    Defaults to the first (weakest) tier when ref is None."""
    if ref is None:
        return history[0]
    entry = next((e for e in history if e["tier"] == ref.lower()), None)
    if entry:
        return entry
    if os.path.exists(ref):
        name = os.path.splitext(os.path.basename(ref))[0]
        return {"tier": name, "mean_reward": float("nan"), "model_path": ref}
    print(f"Baseline {ref!r} not a known tier or existing file; using first tier.")
    return history[0]


def compare_with_best(history, baseline=None, episodes=1, max_steps=10000):
    """Record a split-screen video: a baseline model (left) vs the best tier
    (right). The shorter run is frozen on its last frame (marked GAME OVER) so
    the full, longer episode — e.g. the expert digging its tunnel — plays out
    instead of the clip being cut to the weaker model's quick death."""
    best = max(history, key=lambda e: e["mean_reward"])
    base = _resolve_baseline(history, baseline)

    def _r(x):
        return f"{x:+.1f}" if x == x else "?"  # x != x -> NaN (custom model path)

    print(f"Recording {base['tier']} (r={_r(base['mean_reward'])}) "
          f"vs best {best['tier']} (r={_r(best['mean_reward'])})")

    model_a, env_a = load_model_and_env(base["model_path"])
    model_b, env_b = load_model_and_env(best["model_path"])
    frames_a = collect_frames(model_a, env_a, n_episodes=episodes, max_steps=max_steps)
    frames_b = collect_frames(model_b, env_b, n_episodes=episodes, max_steps=max_steps)
    env_a.close()
    env_b.close()

    la, lb = len(frames_a), len(frames_b)
    n = max(la, lb)
    if n == 0:
        print("No frames captured.")
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    combined = []
    for i in range(n):
        fa = frames_a[i] if i < la else frames_a[-1]  # freeze on last frame
        fb = frames_b[i] if i < lb else frames_b[-1]
        divider = np.ones((fa.shape[0], 4, 3), dtype=np.uint8) * 200
        row = np.concatenate([fa, divider, fb], axis=1)
        cv2.putText(row, f"{base['tier'].upper()} | {_r(base['mean_reward'])}",
                    (10, 20), font, 0.5, (255, 255, 100), 1)
        cv2.putText(row, f"{best['tier'].upper()} | {_r(best['mean_reward'])}",
                    (fa.shape[1] + 14, 20), font, 0.5, (100, 255, 100), 1)
        if i >= la:
            cv2.putText(row, "GAME OVER", (10, fa.shape[0] - 8), font, 0.5, (255, 80, 80), 1)
        if i >= lb:
            cv2.putText(row, "GAME OVER", (fa.shape[1] + 14, fa.shape[0] - 8), font, 0.5, (255, 80, 80), 1)
        combined.append(row)

    out = os.path.join(config.VIDEOS_DIR,
                       f"comparison_{base['tier']}_vs_{best['tier']}.mp4")
    save_video(combined, out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", type=str, default=None,
                        help="Tier name to watch (beginner/learning/skilled/expert)")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--baseline", type=str, default=None,
                        help="For --compare: tier name (e.g. learning/skilled) or a "
                             "model path to put on the left vs the best tier. "
                             "Default: the weakest (first) tier.")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=None,
                        help="Episodes (lives) to record. Default 2, or 1 for --compare.")
    parser.add_argument("--seconds", type=float, default=None,
                        help="Target video length in seconds: keep playing new games "
                             "back-to-back until reached (overrides --episodes). "
                             "Great for weak tiers that die fast.")
    args = parser.parse_args()

    os.makedirs(config.VIDEOS_DIR, exist_ok=True)
    history = load_history()

    # --seconds takes precedence over --episodes (target clip length in frames).
    target_frames = round(args.seconds * config.VIDEO_FPS) if args.seconds else None

    if args.model:
        env = make_atari_env(config.ENV_ID, n_envs=1, seed=0,
                              env_kwargs={"render_mode": "rgb_array"})
        env = VecFrameStack(env, n_stack=config.N_STACK)
        env = VecTransposeImage(env)
        model = PPO.load(args.model, env=env)
        frames = collect_frames(model, env, n_episodes=args.episodes or 2,
                                target_frames=target_frames)
        env.close()
        save_video(frames, os.path.join(config.VIDEOS_DIR, "custom_play.mp4"))
        return

    if not history:
        return

    if args.compare:
        compare_with_best(history, baseline=args.baseline, episodes=args.episodes or 1)
        return

    if args.all:
        for e in history:
            play_tier(e, n_episodes=args.episodes or 2, target_frames=target_frames)
        return

    if args.tier is not None:
        entry = next((e for e in history if e["tier"] == args.tier.lower()), None)
        if entry:
            play_tier(entry, n_episodes=args.episodes or 2, target_frames=target_frames)
        else:
            print(f"Tier {args.tier!r} not found in evolution log.")
        return

    menu(history)


if __name__ == "__main__":
    main()
