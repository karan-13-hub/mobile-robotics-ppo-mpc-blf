"""Focused sweep on the 20-ep trajectory-tracking eval.

Starting point (user's current best, 20 eps seed 0):
    tube=0.12, stride=2, horizon=7, rho=1e3, w_v=5e4, alpha=0.03
    pos_RMSE=0.0796, vel_RMSE=0.0895, peak=0.1089, fallback=1.0%

Weak spots: ep 13 (pos=0.117, fb=12%) and ep 14 (pos=0.100, fb=8%) -- the
fast/aggressive trajectory segments where the 7-step horizon + 12 cm tube
is too tight, so MPC infeasibles trigger PPO-fallback storms.

Hypothesis for what can help the tails without hurting the good episodes:
  * longer horizon     (lookahead into the curve)
  * stride=1           (every-step re-plan on the aggressive bits)
  * slightly looser tube (room for the physical peak error)

Fixed at the user's winner: rho=1e3, w_v=5e4, alpha=0.03, smooth=1e-2.

Two-phase:
  1. Screen each config on 5 episodes.
  2. Top 4 by screening pos_RMSE re-run on 20 episodes.

Targets: acceptable => crash=0, reach=100, fallback <= 2%.
Objective: minimise 20-ep pos_RMSE, tiebreak vel_RMSE.
"""

from __future__ import annotations

import itertools
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)

LOG_DIR = REPO / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_PATH = LOG_DIR / "tune_trajectory.log"
SUM_PATH = LOG_DIR / "tune_trajectory.summary"
DEC_PATH = LOG_DIR / "tune_trajectory.decision"

SCREEN_EPISODES = 5
FINAL_EPISODES = 20
MAX_FALLBACK = 2.0  # percent

# PPO baseline (20 eps, seed 0, measured).
PPO_POS_RMSE = 0.1443
PPO_VEL_RMSE = 0.1599
PPO_EE_RMSE  = 0.2243

# Current best (user), 20 eps seed 0.
BASELINE_POS = 0.0796
BASELINE_VEL = 0.0895

# --- Grid -------------------------------------------------------------- #
HORIZONS = [7, 10, 14]
TUBES    = [0.12, 0.15, 0.18]
STRIDES  = [1, 2]

# Fixed from previous sweeps.
RHO   = 1e3
W_V   = 5e4
ALPHA = 0.03


def run_cfg(*, horizon, tube, stride, episodes, save_video: Path | None = None):
    args = [
        sys.executable, "eval_ppo_mpc.py",
        "--episodes", str(episodes), "--seed", "0",
        "--mpc-horizon", str(horizon),
        "--tube", f"{tube}", "--mpc-stride", str(stride),
        "--slack-penalty", f"{RHO}",
        "--velocity-penalty", f"{W_V}",
        "--barrier-velocity-weight", f"{ALPHA}",
    ]
    if save_video is not None:
        args += ["--save-video", str(save_video), "--video-episode", "0"]
    p = subprocess.run(args, capture_output=True, text=True)
    return (p.stdout or "") + (p.stderr or "")


def parse(out: str) -> dict:
    def grab(pat):
        m = re.search(pat, out)
        return float(m.group(1)) if m else None
    return {
        "pos_rmse": grab(r"Position tracking RMSE\s*\[m\]\s*:\s*([\d.]+)"),
        "vel_rmse": grab(r"Velocity tracking RMSE\s*\[m/s\]\s*:\s*([\d.]+)"),
        "ee_rmse":  grab(r"End-effector tracking RMSE \[m\]\s*:\s*([\d.]+)"),
        "peak_e":   grab(r"Peak tracking error\s*\[m\]\s*:\s*([\d.]+)"),
        "tube_viol": grab(r"Tube violation rate.*?:\s*([\d.]+)%"),
        "fallback": grab(r"MPC fallback rate\s*:\s*([\d.]+)%") or 0.0,
        "crash":    grab(r"Crash rate\s*:\s*([\d.]+)%") or 0.0,
        "reach":    grab(r"Goal reach rate.*?:\s*([\d.]+)%") or 0.0,
    }


def fmt(tag: str, r: dict) -> str:
    def f(x):
        return "?     " if x is None else f"{x:.4f}"
    return (f"{tag:<55}  pos={f(r['pos_rmse'])} vel={f(r['vel_rmse'])} "
            f"ee={f(r['ee_rmse'])} peak={f(r['peak_e'])} "
            f"fb={r['fallback']:.1f}% viol={r['tube_viol']:.1f}% "
            f"crash={r['crash']:.0f}% reach={r['reach']:.0f}%")


def acceptable(r: dict) -> bool:
    return (r["crash"] == 0.0
            and r["pos_rmse"] is not None
            and r["vel_rmse"] is not None
            and r["fallback"] <= MAX_FALLBACK)


def main():
    t0 = time.time()
    LOG_PATH.write_text("")
    SUM_PATH.write_text("")
    DEC_PATH.write_text("")

    SUM_PATH.write_text(
        f"# ppo     pos={PPO_POS_RMSE} vel={PPO_VEL_RMSE} ee={PPO_EE_RMSE}\n"
        f"# current pos={BASELINE_POS} vel={BASELINE_VEL}\n"
        f"# fixed   rho={RHO} w_v={W_V} alpha={ALPHA}\n"
        f"# max fallback = {MAX_FALLBACK}%   target: min 20-ep pos_RMSE\n\n"
    )

    configs = list(itertools.product(HORIZONS, TUBES, STRIDES))
    print(f"[plan] {len(configs)} configs screen @ {SCREEN_EPISODES} eps",
          flush=True)

    survivors = []
    for (horizon, tube, stride) in configs:
        tag = f"N={horizon} tube={tube:g} stride={stride}"
        print(f"[screen] {tag}", flush=True)
        out = run_cfg(horizon=horizon, tube=tube, stride=stride,
                      episodes=SCREEN_EPISODES)
        with LOG_PATH.open("a") as f:
            f.write(f"\n\n========== SCREEN {tag} ==========\n{out}")
        r = parse(out)
        line = fmt(f"SCREEN {tag}", r)
        print("       ", line, flush=True)
        with SUM_PATH.open("a") as f:
            f.write(line + "\n")
        if acceptable(r):
            survivors.append((r["pos_rmse"], horizon, tube, stride))

    survivors.sort(key=lambda x: x[0])
    top = survivors[:4]
    print(f"\n[plan] {len(top)} survivors -> final {FINAL_EPISODES} eps",
          flush=True)
    with SUM_PATH.open("a") as f:
        f.write(f"\n# {len(top)} survivors -> {FINAL_EPISODES}-ep final\n")

    best = None
    for (_, horizon, tube, stride) in top:
        tag = f"N={horizon} tube={tube:g} stride={stride}"
        print(f"[final ] {tag}", flush=True)
        out = run_cfg(horizon=horizon, tube=tube, stride=stride,
                      episodes=FINAL_EPISODES)
        with LOG_PATH.open("a") as f:
            f.write(f"\n\n========== FINAL {tag} ==========\n{out}")
        r = parse(out)
        line = fmt(f"FINAL  {tag}", r)
        print("        ", line, flush=True)
        with SUM_PATH.open("a") as f:
            f.write(line + "\n")
        if acceptable(r) and r["pos_rmse"] is not None:
            key = (r["pos_rmse"], r["vel_rmse"] or 1e9)
            if best is None or key < (best[0]["pos_rmse"],
                                      best[0]["vel_rmse"] or 1e9):
                best = (r, horizon, tube, stride)

    if best is None:
        print("[done] no acceptable config; keeping user baseline")
        DEC_PATH.write_text("no_improvement=1\n")
        return

    r, horizon, tube, stride = best
    improved = (r["pos_rmse"] < BASELINE_POS - 1e-4)
    DEC_PATH.write_text(
        f"horizon={horizon}\ntube={tube:g}\nstride={stride}\n"
        f"rho={RHO:g}\nw_v={W_V:g}\nalpha={ALPHA:g}\n"
        f"final_pos_rmse={r['pos_rmse']}\n"
        f"final_vel_rmse={r['vel_rmse']}\n"
        f"final_ee_rmse={r['ee_rmse']}\n"
        f"final_peak_e={r['peak_e']}\n"
        f"final_fallback={r['fallback']}\n"
        f"final_tube_viol={r['tube_viol']}\n"
        f"baseline_pos_rmse={BASELINE_POS}\n"
        f"baseline_vel_rmse={BASELINE_VEL}\n"
        f"ppo_pos_rmse={PPO_POS_RMSE}\n"
        f"ppo_vel_rmse={PPO_VEL_RMSE}\n"
        f"improved_over_user_best={improved}\n"
        f"elapsed_s={time.time() - t0:.1f}\n"
    )
    print(f"[done] winner: N={horizon} tube={tube} stride={stride} -> "
          f"pos={r['pos_rmse']:.4f} vel={r['vel_rmse']:.4f} "
          f"(user={BASELINE_POS:.4f}/{BASELINE_VEL:.4f}) improved={improved}",
          flush=True)


if __name__ == "__main__":
    main()
