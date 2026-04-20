"""Tune wind-high MPC-BLF to reduce Phase 3 peak velocity.

Starting point (current disturbance winner, 14 s episode, 10 s stabilize):
    rho=1e4, w_v=5e4, alpha=0.1, tube=0.12, stride=2, horizon=7
    Phase 3 pos RMSE=0.047 max=0.075
    Phase 3 vel RMSE=0.067 max=0.295   <-- optimise this
    Settled 0.87 s, fallback 0.6%

The peak velocity happens at the moment the last gust ends and the drone
snaps back toward the hover setpoint. To cut it we want the filter to
  - allow less position drift during the gust (tighter V),
  - actively penalise absolute velocity along the recovery.

Knobs varied:
  * alpha    (velocity-aware BLF weight)  -- directly inflates V with ||e_v||
  * w_v      (stage-wise velocity penalty) -- cost on ||v_k - v_ref_k||^2
  * rho      (BLF slack penalty)           -- tighter descent enforcement
  * stride   (1 means re-plan every step; snappier recovery)

Horizon fixed at 7 (longer means slower solve; we'd rather re-plan every step).
Tube fixed at 0.12 (looser tubes only added peak velocity in earlier runs).

Target: minimise Phase 3 vel max, subject to
  - not crashed, reach (implicit in disturbance eval),
  - fallback <= 2 %,
  - Phase 3 pos max <= 0.09 m  (no worse than current 0.075 by much),
  - Phase 3 vel RMSE <= 0.09 m/s (don't trade RMSE for peak).

Then the winner is re-run with --save-video.
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
LOG_PATH = LOG_DIR / "tune_wind_high.log"
SUM_PATH = LOG_DIR / "tune_wind_high.summary"
DEC_PATH = LOG_DIR / "tune_wind_high.decision"

# Baseline (14 s, 10 s stabilize, seed 0).
BASELINE = dict(rho=1e4, w_v=5e4, alpha=0.1, stride=2,
                vel_max=0.2950, vel_rmse=0.0671,
                pos_max=0.0746, pos_rmse=0.0471)

# --- Grid -------------------------------------------------------------- #
ALPHAS  = [0.1, 0.3, 0.6]
W_VS    = [5e4, 2e5, 5e5]
RHOS    = [1e4, 3e4]
STRIDES = [1, 2]

HORIZON = 7
TUBE    = 0.12

# Constraints.
MAX_FALLBACK = 2.0
MAX_POS_MAX = 0.090
MAX_VEL_RMSE = 0.090


def run_cfg(*, rho, w_v, alpha, stride, save_video: Path | None = None):
    args = [
        sys.executable, "eval_ppo_disturbance.py",
        "--mode", "wind-high", "--seed", "0",
        "--stabilize-time", "10.0", "--mpc-blf",
        "--tube", f"{TUBE}", "--mpc-stride", str(stride),
        "--slack-penalty", f"{rho}",
        "--velocity-penalty", f"{w_v}",
        "--barrier-velocity-weight", f"{alpha}",
    ]
    if save_video is not None:
        args += ["--save-video", str(save_video)]
    p = subprocess.run(args, capture_output=True, text=True)
    return (p.stdout or "") + (p.stderr or "")


def parse(out: str) -> dict:
    # Extract Phase 3 stabilize pos line then vel line.
    m = re.search(
        r"Phase 3 stabil\..*?pos\s+mean=([\d.]+)\s+max=([\d.]+)\s+RMSE=([\d.]+).*?"
        r"vel\s+mean=([\d.]+)\s+max=([\d.]+)\s+RMSE=([\d.]+)",
        out, re.S,
    )
    if m:
        pos_mean, pos_max, pos_rmse, vel_mean, vel_max, vel_rmse = [
            float(x) for x in m.groups()
        ]
    else:
        pos_mean = pos_max = pos_rmse = vel_mean = vel_max = vel_rmse = None

    crashed = "Crashed    : True" in out
    fb = re.search(r"fallback=([\d.]+)%", out)
    fb_pct = float(fb.group(1)) if fb else 0.0

    settle = re.search(r"Settling.*?:\s*([\d.]+)\s*s after", out)
    settle_s = float(settle.group(1)) if settle else None

    return dict(
        pos_max=pos_max, pos_rmse=pos_rmse,
        vel_max=vel_max, vel_rmse=vel_rmse,
        crashed=crashed, fallback=fb_pct, settle_s=settle_s,
    )


def fmt(tag: str, r: dict) -> str:
    def f(x):
        return "?     " if x is None else f"{x:.4f}"
    s = (f"{tag:<55}  p3 pos={f(r['pos_rmse'])}/{f(r['pos_max'])} "
         f"vel={f(r['vel_rmse'])}/{f(r['vel_max'])} "
         f"fb={r['fallback']:.1f}% crash={r['crashed']}")
    if r["settle_s"] is not None:
        s += f" settle={r['settle_s']:.2f}s"
    return s


def acceptable(r: dict) -> bool:
    return (not r["crashed"]
            and r["vel_max"] is not None
            and r["pos_max"] is not None
            and r["vel_rmse"] is not None
            and r["fallback"] <= MAX_FALLBACK
            and r["pos_max"] <= MAX_POS_MAX
            and r["vel_rmse"] <= MAX_VEL_RMSE)


def main():
    t0 = time.time()
    LOG_PATH.write_text("")
    SUM_PATH.write_text("")
    DEC_PATH.write_text("")

    SUM_PATH.write_text(
        f"# baseline {BASELINE}\n"
        f"# fixed horizon={HORIZON} tube={TUBE}\n"
        f"# constraints: fb<={MAX_FALLBACK}% pos_max<={MAX_POS_MAX} "
        f"vel_rmse<={MAX_VEL_RMSE}\n"
        f"# objective: minimise Phase 3 vel_max\n\n"
    )

    configs = list(itertools.product(RHOS, W_VS, ALPHAS, STRIDES))
    print(f"[plan] {len(configs)} configs", flush=True)

    best = None
    for (rho, w_v, alpha, stride) in configs:
        tag = f"rho={rho:g} w_v={w_v:g} alpha={alpha:g} stride={stride}"
        print(f"[run] {tag}", flush=True)
        out = run_cfg(rho=rho, w_v=w_v, alpha=alpha, stride=stride)
        with LOG_PATH.open("a") as f:
            f.write(f"\n\n========== {tag} ==========\n{out}")
        r = parse(out)
        line = fmt(tag, r)
        print("     ", line, flush=True)
        with SUM_PATH.open("a") as f:
            f.write(line + "\n")
        if not acceptable(r):
            continue
        if best is None or r["vel_max"] < best[0]["vel_max"]:
            best = (r, rho, w_v, alpha, stride)

    if best is None:
        print("[done] no acceptable config; keeping baseline")
        DEC_PATH.write_text("no_acceptable_config=1\n")
        return

    r, rho, w_v, alpha, stride = best
    tag = f"rho={rho:g} w_v={w_v:g} alpha={alpha:g} stride={stride}"
    print(f"\n[winner] {tag}  vel_max={r['vel_max']:.4f} "
          f"(baseline {BASELINE['vel_max']:.4f})", flush=True)

    video = REPO / "videos/disturbance/wind_high_mpc_blf.mp4"
    print(f"[video] saving {video}", flush=True)
    out = run_cfg(rho=rho, w_v=w_v, alpha=alpha, stride=stride, save_video=video)
    r_final = parse(out)
    with SUM_PATH.open("a") as f:
        f.write(fmt(f"FINAL {tag}", r_final) + "\n")

    DEC_PATH.write_text(
        f"rho={rho:g}\nw_v={w_v:g}\nalpha={alpha:g}\nstride={stride}\n"
        f"horizon={HORIZON}\ntube={TUBE}\n"
        f"final_pos_rmse={r_final['pos_rmse']}\n"
        f"final_pos_max={r_final['pos_max']}\n"
        f"final_vel_rmse={r_final['vel_rmse']}\n"
        f"final_vel_max={r_final['vel_max']}\n"
        f"final_fallback={r_final['fallback']}\n"
        f"final_settle_s={r_final['settle_s']}\n"
        f"baseline_vel_max={BASELINE['vel_max']}\n"
        f"vel_max_reduction_pct="
        f"{100.0 * (BASELINE['vel_max'] - r_final['vel_max']) / BASELINE['vel_max']:.2f}\n"
        f"elapsed_s={time.time() - t0:.1f}\n"
    )
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
