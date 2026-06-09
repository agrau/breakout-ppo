"""Generate evolution and training-curves plots.

Usage:
    python3 visualize.py                 # both plots
    python3 visualize.py --curves-only   # training curves only
    python3 visualize.py --evolution-only
    python3 visualize.py --show          # also open in window
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import config


_SKILL_BANDS = [
    (  0,   5, "#ffcccc", "Beginner (barely hits)"),
    (  5,  30, "#fff2cc", "Learning (clears bottom rows)"),
    ( 30, 100, "#ccf2ff", "Skilled (clears most of wall)"),
    (100, 432, "#ccffcc", "Expert (digs tunnels)"),
]
_REWARD_MIN = 0
_REWARD_MAX = 432   # one full wall clear ≈ 432 points


def load_log(log_path: str) -> list:
    if not os.path.exists(log_path):
        print(f"No evolution log found at {log_path}")
        return []
    with open(log_path) as f:
        return json.load(f)


def plot_evolution(log_path: str = config.EVOLUTION_LOG,
                   output_path: str = None,
                   show: bool = False):
    history = load_log(log_path)
    if not history:
        print("Nothing to plot yet — run train.py first.")
        return

    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    if output_path is None:
        output_path = os.path.join(config.PLOTS_DIR, "evolution.png")

    timesteps = [e["timesteps"] for e in history]
    rewards = [e["mean_reward"] for e in history]
    stds = [e["std_reward"] for e in history]
    tiers = [e["tier"] for e in history]

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("#1a1a2e")

    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1],
                          hspace=0.35, wspace=0.3,
                          left=0.07, right=0.97, top=0.90, bottom=0.08)

    ax_main = fig.add_subplot(gs[0, 0])
    ax_table = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, 0])

    _style_ax(ax_main)
    _style_ax(ax_table)
    _style_ax(ax_bar)

    # — skill background bands —
    for lo, hi, color, _ in _SKILL_BANDS:
        ax_main.axhspan(lo, hi, alpha=0.18, color=color, zorder=0)

    # — reward curve with shaded std —
    t = np.array(timesteps)
    r = np.array(rewards)
    s = np.array(stds)
    ax_main.fill_between(t, r - s, r + s, alpha=0.25, color="#4fc3f7")
    ax_main.plot(t, r, color="#4fc3f7", linewidth=2.5, zorder=3)

    # — milestone markers —
    cmap = plt.cm.RdYlGn
    r_norm = (r - _REWARD_MIN) / float(_REWARD_MAX - _REWARD_MIN)
    for i, (x, y, c) in enumerate(zip(t, r, r_norm)):
        ax_main.scatter(x, y, color=cmap(c), s=120, zorder=5,
                        edgecolors="white", linewidths=0.8)
        ax_main.annotate(
            tiers[i].upper(),
            xy=(x, y), xytext=(6, 8), textcoords="offset points",
            fontsize=8, color="white", alpha=0.95, weight="bold",
        )

    ax_main.set_xlim(0, max(t) * 1.05)
    ax_main.set_ylim(_REWARD_MIN - 5, _REWARD_MAX + 10)
    ax_main.set_xlabel("Training Steps", color="white", fontsize=11)
    ax_main.set_ylabel("Mean Reward", color="white", fontsize=11)
    ax_main.set_title("PPO Breakout — Learning Curve", color="white", fontsize=13, pad=10)
    ax_main.axhline(0, color="white", linestyle="--", alpha=0.3, linewidth=0.8)

    # legend patches
    patches = [mpatches.Patch(color=c, alpha=0.6, label=lbl)
               for _, _, c, lbl in _SKILL_BANDS]
    ax_main.legend(handles=patches, loc="upper left",
                   facecolor="#2d2d4e", edgecolor="#555", labelcolor="white",
                   fontsize=8)

    # — tier milestone table —
    ax_table.axis("off")
    ax_table.set_title("Skill Tiers Reached", color="white", fontsize=11, pad=6)
    col_labels = ["Tier", "Steps", "Reward"]
    rows = []
    for e in history:
        rows.append([
            e["tier"].capitalize(),
            f"{e['timesteps'] / 1e6:.1f}M",
            f"{e['mean_reward']:+.1f}",
        ])
    if rows:
        tbl = ax_table.table(
            cellText=rows,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        for (row, col), cell in tbl.get_celld().items():
            cell.set_facecolor("#2d2d4e" if row > 0 else "#3a3a5c")
            cell.set_text_props(color="white")
            cell.set_edgecolor("#555")
            if row > 0:
                val = float(rows[row - 1][2])
                norm = (val - _REWARD_MIN) / float(_REWARD_MAX - _REWARD_MIN)
                cell.set_facecolor(
                    plt.cm.RdYlGn(norm, alpha=0.5) if col == 2 else "#2d2d4e"
                )

    # — bar chart of tier rewards —
    bar_colors = [cmap((v - _REWARD_MIN) / float(_REWARD_MAX - _REWARD_MIN)) for v in rewards]
    ax_bar.bar(range(len(rewards)), rewards, color=bar_colors, edgecolor="white",
               linewidth=0.5, alpha=0.85)
    ax_bar.set_ylim(_REWARD_MIN, _REWARD_MAX)
    ax_bar.set_xlabel("Tier", color="white", fontsize=9)
    ax_bar.set_ylabel("Reward", color="white", fontsize=9)
    ax_bar.set_title("Reward at Tier Entry", color="white", fontsize=10, pad=6)
    ax_bar.set_xticks(range(len(rewards)))
    ax_bar.set_xticklabels([t.capitalize() for t in tiers], fontsize=8, color="white")
    ax_bar.axhline(0, color="white", linestyle="--", alpha=0.3, linewidth=0.8)

    fig.suptitle(
        f"Breakout PPO Evolution  |  {len(history)} tiers reached  |  "
        f"Best: {max(rewards):+.2f}",
        color="white", fontsize=14, y=0.96,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Plot saved → {output_path}")

    if show:
        matplotlib.use("TkAgg")
        plt.show()

    plt.close(fig)


def _style_ax(ax):
    ax.set_facecolor("#2d2d4e")
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")


# ── Training curves ───────────────────────────────────────────────────────────

def _load_csv(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    data: dict = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                data.setdefault(k, [])
                try:
                    data[k].append(float(v))
                except (ValueError, TypeError):
                    data[k].append(None)
    return data


def _smooth(values, window: int = 20):
    """Simple moving average, returns same length with NaN padding."""
    out = np.full(len(values), np.nan)
    for i in range(len(values)):
        chunk = [v for v in values[max(0, i - window + 1):i + 1] if v is not None]
        if chunk:
            out[i] = np.mean(chunk)
    return out


def _clean_series(data: dict, key: str):
    """Return (updates, values) with None rows dropped."""
    vals = data.get(key, [])
    iters = data.get("time/iterations", list(range(len(vals))))
    pairs = [(u, v) for u, v in zip(iters, vals) if v is not None]
    if not pairs:
        return np.array([]), np.array([])
    u, v = zip(*pairs)
    return np.array(u), np.array(v)


_PANELS = [
    ("rollout/ep_rew_mean",           "Mean Episode Reward",  "#4fc3f7", False),
    ("train/policy_gradient_loss",    "Policy Loss",          "#ff8a65", False),
    ("train/entropy_loss",            "Policy Entropy",       "#a5d6a7", False),
]


def plot_training_curves(csv_path: str = None,
                         output_path: str = None,
                         show: bool = False):
    if csv_path is None:
        csv_path = config.TRAINING_CSV

    data = _load_csv(csv_path)
    if not data:
        print(f"No training CSV found at {csv_path} — run train.py first.")
        return

    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    if output_path is None:
        output_path = os.path.join(config.PLOTS_DIR, "training_curves.png")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("Breakout PPO — Training Comparison", color="white", fontsize=14, y=0.98)

    for ax, (key, title, color, smoothed) in zip(axes.flat, _PANELS):
        _style_ax(ax)
        updates, vals = _clean_series(data, key)

        if len(vals) == 0:
            ax.set_title(title + " (no data)", color="white", fontsize=9)
            ax.set_xlabel("Update", color="white", fontsize=8)
            continue

        if smoothed:
            ax.plot(updates, vals, color=color, alpha=0.2, linewidth=0.8)
            ax.plot(updates, _smooth(list(vals), window=30),
                    color=color, linewidth=1.8, label="smoothed")
        else:
            ax.plot(updates, vals, color=color, linewidth=1.0, alpha=0.85)

        ax.set_title(title, color="white", fontsize=10, pad=6)
        ax.set_xlabel("Update", color="white", fontsize=8)

    def _safe_max(key):
        vals = [v for v in data.get(key, []) if v is not None]
        return int(max(vals)) if vals else 0

    n_updates = _safe_max("time/iterations")
    n_steps = _safe_max("time/total_timesteps")
    fig.suptitle(
        f"Breakout PPO — Training Comparison  |  "
        f"{n_updates:,} updates  |  {n_steps / 1e6:.2f}M steps",
        color="white", fontsize=13, y=0.995,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Plot saved → {output_path}")

    if show:
        matplotlib.use("TkAgg")
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=config.EVOLUTION_LOG)
    parser.add_argument("--csv", default=config.TRAINING_CSV)
    parser.add_argument("--output", default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--curves-only", action="store_true")
    parser.add_argument("--evolution-only", action="store_true")
    args = parser.parse_args()

    if not args.curves_only:
        plot_evolution(args.log, args.output, args.show)
    if not args.evolution_only:
        plot_training_curves(args.csv, args.output, args.show)
