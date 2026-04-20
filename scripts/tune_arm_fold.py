"""Tune the MPC-BLF filter on the arm-fold disturbance.

Unlike the wind case (three gusts over 2 s, then external force = 0),
arm-fold is a *persistent* COM-shift disturbance: the arm stays folded
throughout the stabilize phase, so the filter has to hold the drone at
a shifted equilibrium rather than transiently reject a pulse.

Target: minimise stabilize-phase velocity RMSE, subject to
  - not crashed,
  - fallback < 1 %,
  - stabilize pos RMSE <= PPO baseline.

Grid is centred on the wind-high winner but widens around the axes
most likely to matter for a sustained disturbance (rho and alpha).
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
LOG_PATH = LOG_DIR / "tune_arm_fold.log"
SUM_PATH = LOG_DIR / "tune_arm_fold.summary"
DEC_PATH = LOG_DIR / "tune_arm_fold.decision"

BASE_ARGS = [
    "--mode", "arm-fold", "--seed", "0", "--mpc-blf",
    "--tube", "0.12", "--mpc-stride", "2",
]

RHOS = [3e2, 1e3, 3e3, 1e4]
ALPHAS = [0.01, 0.03, 0.1]
WVS = [1e4, 5e4, 2e5]


def run_once(rho: float, alpha: float, w_v: float,
             save_video: Path | None = None) -> str:
    args = [sys.executable, "eval_ppo_disturbance.py", *BASE_ARGS,
            "--slack-penalty", f"{rho}",
            "--barrier-velocity-weight", f"{alpha}",
            "--velocity-penalty", f"{w_v}"]
    if save_video is not None:
        args.extend(["--save-video", str(save_video)])
    p = subprocess.run(args, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    tag = f"rho={rho:g} alpha={alpha:g} w_v={w_v:g}"
    with LOG_PATH.open("a") as f:
        f.write(f"\n\n========== {tag} ==========\n")
        f.write(out)
    return out


def parse(out: str) -> dict:
    def grab(pat: str) -> float | None:
        m = re.search(pat, out, re.S)
        return float(m.group(1)) if m else None

    return {
        "pos_stab_rmse": grab(r"Phase 3 stabil\..*?pos\s+mean=[\d.]+\s+max=[\d.]+\s+RMSE=([\d.]+)"),
        "pos_stab_max":  grab(r"Phase 3 stabil\..*?pos\s+mean=[\d.]+\s+max=([\d.]+)"),
        "pos_disturb_rmse": grab(r"Phase 2 disturb.*?pos\s+mean=[\d.]+\s+max=[\d.]+\s+RMSE=([\d.]+)"),
        "vel_stab_rmse": grab(r"Phase 3 stabil\..*?vel\s+mean=[\d.]+\s+max=[\d.]+\s+RMSE=([\d.]+)"),
        "vel_stab_max":  grab(r"Phase 3 stabil\..*?vel\s+mean=[\d.]+\s+max=([\d.]+)"),
        "vel_disturb_rmse": grab(r"Phase 2 disturb.*?vel\s+mean=[\d.]+\s+max=[\d.]+\s+RMSE=([\d.]+)"),
        "crashed": "Crashed    : True" in out,
        "fallback": grab(r"fallback=([\d.]+)%") or 0.0,
    }


def fmt(tag: str, r: dict) -> str:
    def f(x):
        return "?    " if x is None else f"{x:.4f}"
    return (f"{tag:<40}  pos_stab={f(r['pos_stab_rmse'])}/{f(r['pos_stab_max'])} "
            f"vel_stab={f(r['vel_stab_rmse'])}/{f(r['vel_stab_max'])} "
            f"pos_dist={f(r['pos_disturb_rmse'])} vel_dist={f(r['vel_disturb_rmse'])} "
            f"crashed={r['crashed']} fallback={r['fallback']:.1f}%")


def main() -> None:
    t_start = time.time()
    LOG_PATH.write_text("")
    SUM_PATH.write_text("")
    DEC_PATH.write_text("")

    # Arm-fold PPO baseline (just measured).
    PPO_POS_STAB_RMSE = 0.0571
    PPO_VEL_STAB_RMSE = 0.2357
    pos_budget = PPO_POS_STAB_RMSE  # don't regress position at all

    with SUM_PATH.open("a") as f:
        f.write(f"# ppo_pos_stab_rmse={PPO_POS_STAB_RMSE}\n")
        f.write(f"# ppo_vel_stab_rmse={PPO_VEL_STAB_RMSE}\n")
        f.write(f"# pos_budget={pos_budget}\n")
        f.write(f"# target: minimise vel_stab_rmse\n\n")

    configs = list(itertools.product(RHOS, ALPHAS, WVS))
    print(f"[plan] {len(configs)} configs", flush=True)

    best = None
    best_any = None

    for rho, alpha, w_v in configs:
        print(f"[run] rho={rho:g} alpha={alpha:g} w_v={w_v:g}", flush=True)
        out = run_once(rho, alpha, w_v)
        r = parse(out)
        line = fmt(f"rho={rho:g} alpha={alpha:g} w_v={w_v:g}", r)
        print("     ", line, flush=True)
        with SUM_PATH.open("a") as f:
            f.write(line + "\n")

        ok = (r["vel_stab_rmse"] is not None
              and not r["crashed"]
              and r["fallback"] < 1.0)
        if not ok:
            continue
        vel = r["vel_stab_rmse"]
        if best_any is None or vel < best_any[0]:
            best_any = (vel, rho, alpha, w_v, r)
        if r["pos_stab_rmse"] is not None and r["pos_stab_rmse"] <= pos_budget:
            if best is None or vel < best[0]:
                best = (vel, rho, alpha, w_v, r)

    if best is not None:
        _, rho, alpha, w_v, r = best
        tag = "WINNER (within pos budget)"
    elif best_any is not None:
        _, rho, alpha, w_v, r = best_any
        tag = "WINNER (pos budget exceeded)"
    else:
        rho, alpha, w_v = 1e3, 0.03, 5e4
        r = None
        tag = "FALLBACK"

    print(f"\n[{tag}] rho={rho:g} alpha={alpha:g} w_v={w_v:g}", flush=True)

    video = REPO / "videos/disturbance/arm_fold_mpc_blf.mp4"
    print(f"[video] saving {video}", flush=True)
    out = run_once(rho, alpha, w_v, save_video=video)
    r_final = parse(out)
    with SUM_PATH.open("a") as f:
        f.write(fmt(f"FINAL rho={rho:g} alpha={alpha:g} w_v={w_v:g}",
                    r_final) + "\n")

    DEC_PATH.write_text(
        f"final_rho={rho:g}\n"
        f"final_alpha={alpha:g}\n"
        f"final_w_v={w_v:g}\n"
        f"final_vel_stab_rmse={r_final['vel_stab_rmse']}\n"
        f"final_vel_stab_max={r_final['vel_stab_max']}\n"
        f"final_pos_stab_rmse={r_final['pos_stab_rmse']}\n"
        f"final_pos_stab_max={r_final['pos_stab_max']}\n"
        f"final_vel_disturb_rmse={r_final['vel_disturb_rmse']}\n"
        f"final_pos_disturb_rmse={r_final['pos_disturb_rmse']}\n"
        f"ppo_vel_stab_rmse={PPO_VEL_STAB_RMSE}\n"
        f"ppo_pos_stab_rmse={PPO_POS_STAB_RMSE}\n"
        f"vel_gain_vs_ppo={PPO_VEL_STAB_RMSE - r_final['vel_stab_rmse']}\n"
        f"fallback_pct={r_final['fallback']}\n"
        f"tag={tag}\n"
        f"elapsed_s={time.time() - t_start:.1f}\n"
    )
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
