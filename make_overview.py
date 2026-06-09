"""Generate a single training-overview image combining all main charts.

Reads:
  models/evolution_log.json       (tier milestones)
  logs/run/progress.csv           (training metrics)
Writes:
  plots/training_overview.png     (everything in one image)
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import config


SKILL_BANDS = [
    (  0,   5, "#ffcccc", "Beginner (barely hits)"),
    (  5,  30, "#fff2cc", "Learning (clears bottom rows)"),
    ( 30, 100, "#ccf2ff", "Skilled (clears most of wall)"),
    (100, 432, "#ccffcc", "Expert (digs tunnels)"),
]
REWARD_MIN = 0
REWARD_MAX = 432


def _style(ax):
    ax.set_facecolor("#2d2d4e")
    ax.tick_params(colors="white", labelsize=8)
    for s in ax.spines.values():
        s.set_edgecolor("#555")


def _load_csv(path):
    data = {}
    if not os.path.exists(path):
        return data
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


def _clean(data, key):
    vals = data.get(key, [])
    iters = data.get("time/iterations", list(range(len(vals))))
    pairs = [(u, v) for u, v in zip(iters, vals) if v is not None]
    if not pairs:
        return np.array([]), np.array([])
    u, v = zip(*pairs)
    return np.array(u), np.array(v)


def _smooth(vals, window=30):
    out = np.full(len(vals), np.nan)
    for i in range(len(vals)):
        chunk = [v for v in vals[max(0, i - window + 1):i + 1] if not np.isnan(v)]
        if chunk:
            out[i] = np.mean(chunk)
    return out


def main():
    history = []
    if os.path.exists(config.EVOLUTION_LOG):
        with open(config.EVOLUTION_LOG) as f:
            history = json.load(f)
    data = _load_csv(config.TRAINING_CSV)

    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(config.PLOTS_DIR, "training_overview.png")

    # Figure: 3 rows. Row 0: evolution. Row 1-2: 6-panel curves.
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor("#1a1a2e")

    gs = fig.add_gridspec(
        4, 3,
        height_ratios=[2, 0.6, 1.2, 1.2],
        hspace=0.5, wspace=0.25,
        left=0.05, right=0.97, top=0.94, bottom=0.05,
    )

    # ─── Evolution main chart (top, full width) ───────────────────────────
    ax_evo = fig.add_subplot(gs[0, :])
    _style(ax_evo)

    # background skill bands
    for lo, hi, color, _ in SKILL_BANDS:
        ax_evo.axhspan(lo, hi, alpha=0.18, color=color, zorder=0)

    # rolling reward curve from CSV
    if data:
        # convert iterations to total_timesteps for the x axis
        x_iters, x_steps = _clean(data, "time/total_timesteps")
        _, rewards = _clean(data, "rollout/ep_rew_mean")
        if len(x_steps) and len(rewards):
            n = min(len(x_steps), len(rewards))
            x = x_steps[:n]
            r = rewards[:n]
            ax_evo.plot(x, r, color="#4fc3f7", alpha=0.35, linewidth=0.8,
                        label="Rolling reward")
            ax_evo.plot(x, _smooth(list(r), 30), color="#4fc3f7",
                        linewidth=2.0, label="Smoothed (30 updates)")

    # tier milestone markers
    cmap = plt.cm.RdYlGn
    if history:
        ts = [e["timesteps"] for e in history]
        rs = [e["mean_reward"] for e in history]
        tiers = [e["tier"] for e in history]
        for x, y, t in zip(ts, rs, tiers):
            norm = (y - REWARD_MIN) / float(REWARD_MAX - REWARD_MIN)
            ax_evo.scatter(x, y, color=cmap(norm), s=180, zorder=5,
                           edgecolors="white", linewidths=1.2)
            ax_evo.annotate(
                f"{t.upper()}\n{int(round(y)):+d}",
                xy=(x, y), xytext=(8, 10), textcoords="offset points",
                fontsize=9, color="white", weight="bold", zorder=6,
            )

    ax_evo.set_xlabel("Training Steps", color="white", fontsize=11)
    ax_evo.set_ylabel("Mean Reward", color="white", fontsize=11)
    ax_evo.set_title("Breakout PPO — Reward Evolution",
                     color="white", fontsize=14, pad=10)
    ax_evo.set_ylim(REWARD_MIN - 5, REWARD_MAX + 10)
    ax_evo.axhline(0, color="white", linestyle="--", alpha=0.3, linewidth=0.6)

    patches = [mpatches.Patch(color=c, alpha=0.6, label=lbl)
               for _, _, c, lbl in SKILL_BANDS]
    ax_evo.legend(handles=patches, loc="upper left",
                  facecolor="#2d2d4e", edgecolor="#555",
                  labelcolor="white", fontsize=9)

    # ─── Tier milestone summary (between rows) ────────────────────────────
    ax_tbl = fig.add_subplot(gs[1, :])
    _style(ax_tbl)
    ax_tbl.axis("off")
    ax_tbl.set_title("Skill Tier Milestones", color="white", fontsize=12, pad=4)

    if history:
        col_labels = ["Tier", "Step", "Eval Reward", "Description"]
        descriptions = {
            "beginner": "Random/lucky paddle play",
            "learning": "Clears bottom rows",
            "skilled":  "Clears most of the wall",
            "expert":   "Tunnel strategy",
        }
        rows = [
            [
                e["tier"].capitalize(),
                f"{e['timesteps'] / 1e6:.2f}M",
                f"{e['mean_reward']:+.2f}",
                descriptions.get(e["tier"], "—"),
            ]
            for e in history
        ]
        tbl = ax_tbl.table(
            cellText=rows, colLabels=col_labels,
            loc="center", cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.0, 1.6)
        for (row, col), cell in tbl.get_celld().items():
            cell.set_facecolor("#2d2d4e" if row > 0 else "#3a3a5c")
            cell.set_text_props(color="white")
            cell.set_edgecolor("#555")
            if row > 0 and col == 2:
                val = float(rows[row - 1][2])
                norm = (val - REWARD_MIN) / float(REWARD_MAX - REWARD_MIN)
                cell.set_facecolor(plt.cm.RdYlGn(norm, alpha=0.5))

    # ─── Six-panel training-curve grid (rows 2-3) ─────────────────────────
    panels = [
        ("rollout/ep_rew_mean",        "Mean Episode Reward",  "#4fc3f7", False),
        ("train/policy_gradient_loss", "Policy Loss",          "#ff8a65", False),
        ("train/entropy_loss",         "Policy Entropy",       "#a5d6a7", False),
        ("rollout/ep_rew_mean",        "Reward (smoothed)",    "#4fc3f7", True),
        ("train/value_loss",           "Value Loss",           "#ce93d8", False),
        ("train/clip_fraction",        "Clip Fraction",        "#fff176", False),
    ]
    for idx, (key, title, color, smoothed) in enumerate(panels):
        r, c = divmod(idx, 3)
        ax = fig.add_subplot(gs[2 + r, c])
        _style(ax)
        u, v = _clean(data, key)
        if len(v) == 0:
            ax.set_title(title + " (no data)", color="white", fontsize=10)
            continue
        if smoothed:
            ax.plot(u, v, color=color, alpha=0.2, linewidth=0.7)
            ax.plot(u, _smooth(list(v), 30), color=color, linewidth=1.6)
        else:
            ax.plot(u, v, color=color, linewidth=0.9, alpha=0.85)
        ax.set_title(title, color="white", fontsize=10, pad=4)
        ax.set_xlabel("Update", color="white", fontsize=8)

    # ─── Header summary ───────────────────────────────────────────────────
    n_updates = int(max((x for x in data.get("time/iterations", []) if x is not None), default=0))
    n_steps = int(max((x for x in data.get("time/total_timesteps", []) if x is not None), default=0))
    last_reward = next(
        (v for v in reversed(data.get("rollout/ep_rew_mean", [])) if v is not None),
        None,
    )
    best = max((e["mean_reward"] for e in history), default=0)
    fig.suptitle(
        f"Breakout PPO — Training Overview   |   "
        f"{n_steps / 1e6:.2f}M steps  ·  {n_updates:,} updates  ·  "
        f"current reward {last_reward:.1f}  ·  best eval {best:+.1f}",
        color="white", fontsize=15, y=0.985,
    )

    plt.savefig(out_path, dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
