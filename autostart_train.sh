#!/usr/bin/env bash
# Auto-resume launcher for the Breakout PPO training systemd service.
# Resumes from the newest checkpoint; exits cleanly (no restart) once the
# run has produced a final_model, so reboots don't relaunch a finished job.
set -u
cd /home/ag/toshiba/work/Breakout || exit 1

if [ -f models/final_model.zip ]; then
    echo "[autostart] final_model.zip exists — training already complete, nothing to do." >> training.log
    exit 0
fi

# Startup delay: at boot the GPU/CUDA driver isn't ready the instant systemd
# fires this service, which caused on-failure restarts. Wait until CUDA is
# available (polls up to ~120s); returns immediately on manual restarts since
# the GPU is already up. Proceeds anyway after the timeout so on-failure can catch real issues.
for i in $(seq 1 24); do
    if /usr/bin/python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        [ "$i" -gt 1 ] && echo "[autostart] CUDA ready after ~$(( (i-1)*5 ))s wait" >> training.log
        break
    fi
    echo "[autostart] waiting for CUDA (attempt $i)..." >> training.log
    sleep 5
done

# One-shot resume override: if models/.resume_override exists, resume from the
# path it contains (then consume it) — used to start a run from a specific model
# (e.g. best_model). Normal crash/reboot recovery still uses the latest checkpoint.
OVERRIDE_FILE=models/.resume_override
if [ -f "$OVERRIDE_FILE" ]; then
    RESUME="$(cat "$OVERRIDE_FILE")"
    rm -f "$OVERRIDE_FILE"
    echo "[autostart] one-shot resume override: $RESUME" >> training.log
# Otherwise resume from the most-trained checkpoint if one exists, else best_model, else fresh.
elif ls models/checkpoints/ckpt_*_steps.zip >/dev/null 2>&1; then
    RESUME="latest"
elif [ -f models/best_model.zip ]; then
    RESUME="models/best_model.zip"
else
    RESUME=""
fi

echo "[autostart] $(date -Is) launching train.py (resume=${RESUME:-none})" >> training.log
if [ -n "$RESUME" ]; then
    exec /usr/bin/python3 -u train.py --resume "$RESUME" >> training.log 2>&1
else
    exec /usr/bin/python3 -u train.py >> training.log 2>&1
fi
