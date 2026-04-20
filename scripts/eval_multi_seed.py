"""Run eval_ppo.py and eval_ppo_mpc.py across multiple seeds and save the
per-episode metrics so we can report mean +/- std.

For each seed in --seeds (default 0 20 40 60 80) we launch 20 deterministic
episodes, parse the `  ep NN | ... | pos_RMSE=... vel_RMSE=... ... tube_viol=...
peak_e=... solve=a/b ms[ fb=c%]` lines from stdout, and dump them to JSON.

Output: logs/multi_seed/summary.json  with structure
  {
    "mpc_config": {...},
    "runs": {
        "ppo": {
            "per_seed": [{"seed": 0, "episodes": [...], "mean": {...}}, ...],
            "overall_mean":     {"pos_rmse": .., ...},   # mean over 5 seed means
            "overall_std_seeds": {"pos_rmse": .., ...},  # std over 5 seed means
            "overall_std_eps":   {"pos_rmse": .., ...}   # std over 100 episodes
        },
        "mpc": {...}
    }
  }

Raw stdout of every run is also kept under logs/multi_seed/*.txt for audit.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from statistics import fmean, pstdev

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "logs", "multi_seed")

# Matches both eval_ppo.py and eval_ppo_mpc.py per-episode lines.  MPC variant
# has an optional trailing "fb=NN%" field.
EP_RE = re.compile(
    r"ep\s+(?P<ep>\d+)\s+\|.*?"
    r"pos_RMSE=(?P<pos>[-\d\.]+)\s+"
    r"vel_RMSE=(?P<vel>[-\d\.]+)\s+"
    r"ee_RMSE=(?P<ee>[-\d\.]+)\s*\|.*?"
    r"tube_viol=(?P<viol>[-\d\.]+)%\s+"
    r"peak_e=(?P<peak>[-\d\.]+)\s+"
    r"solve=(?P<med>[-\d\.]+)/(?P<max>[-\d\.]+)ms"
    r"(?:\s+fb=(?P<fb>[-\d\.]+)%)?",
    re.DOTALL,
)


def parse_episodes(text):
    rows = []
    for m in EP_RE.finditer(text):
        rows.append({
            "ep": int(m.group("ep")),
            "pos_rmse": float(m.group("pos")),
            "vel_rmse": float(m.group("vel")),
            "ee_rmse": float(m.group("ee")),
            "tube_viol_pct": float(m.group("viol")),
            "peak_err": float(m.group("peak")),
            "solve_median_ms": float(m.group("med")),
            "solve_max_ms": float(m.group("max")),
            "fallback_pct": float(m.group("fb")) if m.group("fb") is not None else 0.0,
        })
    return rows


METRIC_KEYS = [
    "pos_rmse", "vel_rmse", "ee_rmse",
    "tube_viol_pct", "peak_err",
    "solve_median_ms", "solve_max_ms", "fallback_pct",
]


def mean_dict(rows, keys=METRIC_KEYS):
    return {k: fmean(r[k] for r in rows) for k in keys}


def std_dict(rows, keys=METRIC_KEYS):
    if len(rows) < 2:
        return {k: 0.0 for k in keys}
    return {k: pstdev(r[k] for r in rows) for k in keys}


def run_one(cmd, log_path):
    t0 = time.time()
    print(f">>> {' '.join(cmd)}", flush=True)
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    with open(log_path) as f:
        text = f.read()
    if proc.returncode != 0:
        sys.stderr.write(f"[ERROR] {cmd} exited with {proc.returncode}\n")
        sys.stderr.write(text[-2000:])
        raise SystemExit(1)
    eps = parse_episodes(text)
    print(f"    parsed {len(eps)} episodes in {dt:.1f}s", flush=True)
    return eps


def build_run_block(per_seed_episodes):
    per_seed = []
    all_eps = []
    seed_means = []
    for seed, eps in per_seed_episodes:
        sm = mean_dict(eps)
        per_seed.append({"seed": seed, "n_episodes": len(eps),
                         "mean": sm, "episodes": eps})
        seed_means.append(sm)
        all_eps.extend(eps)

    overall_mean = {
        k: fmean(m[k] for m in seed_means) for k in METRIC_KEYS
    }
    overall_std_seeds = {
        k: pstdev(m[k] for m in seed_means) if len(seed_means) > 1 else 0.0
        for k in METRIC_KEYS
    }
    overall_std_eps = std_dict(all_eps)
    return {
        "per_seed": per_seed,
        "overall_mean": overall_mean,
        "overall_std_seeds": overall_std_seeds,
        "overall_std_eps": overall_std_eps,
        "n_seeds": len(seed_means),
        "n_episodes_total": len(all_eps),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[0, 20, 40, 60, 80])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--model", default="logs/continue_7M/best_model.zip")
    parser.add_argument("--tube", type=float, default=0.12)
    parser.add_argument("--mpc-horizon", type=int, default=7)
    parser.add_argument("--mpc-stride", type=int, default=1)
    parser.add_argument("--slack-penalty", type=float, default=1e3)
    parser.add_argument("--velocity-penalty", type=float, default=5e4)
    parser.add_argument("--barrier-velocity-weight", type=float, default=0.03)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    ppo_runs = []
    mpc_runs = []
    for s in args.seeds:
        ppo_cmd = [
            args.python, "eval_ppo.py",
            "--model", args.model,
            "--episodes", str(args.episodes),
            "--seed", str(s),
            "--tube", str(args.tube),
        ]
        mpc_cmd = [
            args.python, "eval_ppo_mpc.py",
            "--model", args.model,
            "--episodes", str(args.episodes),
            "--seed", str(s),
            "--mpc-horizon", str(args.mpc_horizon),
            "--tube", str(args.tube),
            "--mpc-stride", str(args.mpc_stride),
            "--slack-penalty", str(args.slack_penalty),
            "--velocity-penalty", str(args.velocity_penalty),
            "--barrier-velocity-weight", str(args.barrier_velocity_weight),
        ]
        ppo_runs.append((s, run_one(ppo_cmd,
                                    os.path.join(OUT_DIR, f"ppo_seed{s}.txt"))))
        mpc_runs.append((s, run_one(mpc_cmd,
                                    os.path.join(OUT_DIR, f"mpc_seed{s}.txt"))))

    summary = {
        "model": args.model,
        "episodes_per_seed": args.episodes,
        "seeds": args.seeds,
        "mpc_config": {
            "horizon": args.mpc_horizon,
            "tube": args.tube,
            "stride": args.mpc_stride,
            "slack_penalty": args.slack_penalty,
            "velocity_penalty": args.velocity_penalty,
            "barrier_velocity_weight": args.barrier_velocity_weight,
        },
        "runs": {
            "ppo": build_run_block(ppo_runs),
            "mpc": build_run_block(mpc_runs),
        },
    }

    out_path = os.path.join(OUT_DIR, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== summary ===")
    for label in ("ppo", "mpc"):
        m = summary["runs"][label]["overall_mean"]
        sS = summary["runs"][label]["overall_std_seeds"]
        sE = summary["runs"][label]["overall_std_eps"]
        print(f"[{label}] n_seeds={summary['runs'][label]['n_seeds']} "
              f"n_eps={summary['runs'][label]['n_episodes_total']}")
        for k in ("pos_rmse", "vel_rmse", "peak_err",
                  "tube_viol_pct", "solve_median_ms", "solve_max_ms",
                  "fallback_pct"):
            print(f"    {k:>16s}: {m[k]:.4f}   "
                  f"std_seeds={sS[k]:.4f}   std_eps={sE[k]:.4f}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
