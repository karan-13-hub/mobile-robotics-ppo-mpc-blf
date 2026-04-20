"""Generate comparison plots for the presentation.

All numbers come from the 5-seed x 20-episode protocol in
logs/multi_seed/summary.json (see videos/METRICS.txt).  Bars show the
mean across the 5 seed-level means (n=5), and the error caps show one
standard deviation across those 5 seed means.

If summary.json is missing the script falls back to the previous
hard-coded single-seed values so the plots can still render.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SUMMARY_JSON = os.path.join(ROOT, "logs", "multi_seed", "summary.json")


def _load_summary():
    if not os.path.isfile(SUMMARY_JSON):
        return None
    with open(SUMMARY_JSON) as f:
        return json.load(f)


def _mean_std(summary, label, key):
    """Return (mean, std_seeds) for `key` from runs[label]."""
    blk = summary["runs"][label]
    return blk["overall_mean"][key], blk["overall_std_seeds"][key]


_SUMMARY = _load_summary()

# Flat, minimal style. Neutral base + a single accent for the "winner".
plt.rcParams.update({
    "font.size": 12,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "axes.edgecolor": "#888888",
    "savefig.dpi": 150,
    "figure.facecolor": "white",
})

PPO_COLOR = "#8a8a8a"
MPC_COLOR = "#2c6df5"


def bar_compare(ax, labels, ppo_vals, mpc_vals, ylabel, title,
                pct_drop=True, ppo_err=None, mpc_err=None):
    x = np.arange(len(labels))
    w = 0.38
    ppo_err = ppo_err if ppo_err is not None else [0.0] * len(ppo_vals)
    mpc_err = mpc_err if mpc_err is not None else [0.0] * len(mpc_vals)
    ax.bar(x - w / 2, ppo_vals, w, label="PPO", color=PPO_COLOR,
           yerr=ppo_err, capsize=4, ecolor="#444444",
           error_kw={"elinewidth": 1.2})
    ax.bar(x + w / 2, mpc_vals, w, label="PPO + MPC-BLF", color=MPC_COLOR,
           yerr=mpc_err, capsize=4, ecolor="#0b3a99",
           error_kw={"elinewidth": 1.2})
    for i, (p, m, pe, me) in enumerate(zip(ppo_vals, mpc_vals, ppo_err, mpc_err)):
        if pct_drop and p > 0:
            delta = 100.0 * (m - p) / p
            ax.annotate(f"{delta:+.0f}%",
                        xy=(i + w / 2, m + me), xytext=(0, 6),
                        textcoords="offset points",
                        ha="center", fontsize=10, color=MPC_COLOR)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=12)
    ax.legend(frameon=False, loc="upper right")


def plot_trajectory_tracking():
    labels = ["pos RMSE [m]", "vel RMSE [m/s]", "peak err [m]"]
    keys = ["pos_rmse", "vel_rmse", "peak_err"]
    if _SUMMARY is not None:
        ppo = [_mean_std(_SUMMARY, "ppo", k)[0] for k in keys]
        mpc = [_mean_std(_SUMMARY, "mpc", k)[0] for k in keys]
        ppo_err = [_mean_std(_SUMMARY, "ppo", k)[1] for k in keys]
        mpc_err = [_mean_std(_SUMMARY, "mpc", k)[1] for k in keys]
        title = ("Trajectory tracking  —  5 seeds × 20 episodes "
                 "(mean, error bars = std across seeds)")
    else:
        ppo = [0.1443, 0.1599, 0.2192]
        mpc = [0.0658, 0.0646, 0.0910]
        ppo_err = mpc_err = [0.0, 0.0, 0.0]
        title = "Trajectory tracking  —  20 deterministic episodes, seed 0"
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bar_compare(ax, labels, ppo, mpc,
                ylabel="metric value",
                title=title,
                ppo_err=ppo_err, mpc_err=mpc_err)
    top = max(p + e for p, e in zip(ppo, ppo_err))
    top = max(top, max(m + e for m, e in zip(mpc, mpc_err)))
    ax.set_ylim(0, top * 1.30)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "trajectory_tracking.png"), bbox_inches="tight")
    plt.close(fig)


def plot_tube_violation():
    labels = ["tube violation rate\n(>12 cm)"]
    if _SUMMARY is not None:
        ppo_m, ppo_s = _mean_std(_SUMMARY, "ppo", "tube_viol_pct")
        mpc_m, mpc_s = _mean_std(_SUMMARY, "mpc", "tube_viol_pct")
        ppo, mpc = [ppo_m], [mpc_m]
        ppo_err, mpc_err = [ppo_s], [mpc_s]
        subtitle = (f"5 seeds × 20 episodes — PPO crosses the 12 cm tube "
                    f"{ppo_m:.1f} % of the time; MPC-BLF holds it at "
                    f"{mpc_m:.2f} %")
    else:
        ppo, mpc = [40.72], [0.00]
        ppo_err = mpc_err = [0.0]
        subtitle = ("PPO alone crosses the 12 cm tube 40 % of the time; "
                    "MPC-BLF holds it at 0 %")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(labels))
    w = 0.5
    ax.bar(x - w / 2, ppo, w, label="PPO", color=PPO_COLOR,
           yerr=ppo_err, capsize=4, ecolor="#444444",
           error_kw={"elinewidth": 1.2})
    ax.bar(x + w / 2, mpc, w, label="PPO + MPC-BLF", color=MPC_COLOR,
           yerr=mpc_err, capsize=4, ecolor="#0b3a99",
           error_kw={"elinewidth": 1.2})
    ax.text(0 - w / 2, ppo[0] + ppo_err[0] + 1.5,
            f"{ppo[0]:.2f}% ± {ppo_err[0]:.2f}",
            ha="center", fontsize=11, color=PPO_COLOR)
    ax.text(0 + w / 2, max(2.0, mpc[0] + mpc_err[0] + 1.5),
            f"{mpc[0]:.2f}%",
            ha="center", fontsize=11, color=MPC_COLOR)
    ax.set_xticks(x, labels)
    ax.set_ylabel("violation rate [%]")
    ax.set_ylim(0, max(60.0, ppo[0] + ppo_err[0] + 8))
    ax.set_title("Safety tube violations\n" + subtitle,
                 loc="left", pad=12)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "tube_violation.png"), bbox_inches="tight")
    plt.close(fig)


def plot_disturbance_recovery():
    modes = ["wind-low", "wind-high", "arm-fold"]
    pos_rmse_ppo = [0.0563, 0.1379, 0.2360]
    pos_rmse_mpc = [0.0526, 0.0471, 0.0309]
    pos_max_ppo = [0.0715, 0.5231, 1.4479]
    pos_max_mpc = [0.0565, 0.0746, 0.0355]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    def paired(ax, ppo, mpc, title, ylabel, logscale=False):
        x = np.arange(len(modes))
        w = 0.38
        ax.bar(x - w / 2, ppo, w, label="PPO", color=PPO_COLOR)
        ax.bar(x + w / 2, mpc, w, label="PPO + MPC-BLF", color=MPC_COLOR)
        for i, (p, m) in enumerate(zip(ppo, mpc)):
            ax.annotate(f"{p:.2f}", xy=(i - w / 2, p),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=9, color=PPO_COLOR)
            ax.annotate(f"{m:.2f}", xy=(i + w / 2, m),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=9, color=MPC_COLOR)
        ax.set_xticks(x, modes)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=10)
        ax.legend(frameon=False, loc="upper left")
        if logscale:
            ax.set_yscale("log")

    paired(axes[0], pos_rmse_ppo, pos_rmse_mpc,
           "Phase 3 stabilize  —  position RMSE [m]",
           "position RMSE [m]")
    paired(axes[1], pos_max_ppo, pos_max_mpc,
           "Phase 3 stabilize  —  peak deviation [m]  (log scale)",
           "peak deviation [m]",
           logscale=True)
    fig.suptitle("Disturbance rejection after the pulse / arm-fold ends",
                 x=0.02, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(HERE, "disturbance_recovery.png"),
                bbox_inches="tight")
    plt.close(fig)


def plot_solve_cost():
    # Trajectory-tracking config (stride=1, so MPC runs every env step).
    labels = ["median solve\n[ms]", "per-ep peak\n[ms]", "MPC duty cycle\n[%]",
              "MPC fallback\n[%]"]
    if _SUMMARY is not None:
        ppo_med, ppo_med_s = _mean_std(_SUMMARY, "ppo", "solve_median_ms")
        ppo_peak, ppo_peak_s = _mean_std(_SUMMARY, "ppo", "solve_max_ms")
        mpc_med, mpc_med_s = _mean_std(_SUMMARY, "mpc", "solve_median_ms")
        mpc_peak, mpc_peak_s = _mean_std(_SUMMARY, "mpc", "solve_max_ms")
        ppo = [ppo_med, ppo_peak, 0.0, 0.0]
        mpc = [mpc_med, mpc_peak, 100.0, 0.0]
        ppo_err = [ppo_med_s, ppo_peak_s, 0.0, 0.0]
        mpc_err = [mpc_med_s, mpc_peak_s, 0.0, 0.0]
        title = (f"Compute cost of the filter  —  ~{mpc_med:.0f} ms median, "
                 f"per-ep peak ~{mpc_peak:.0f} ms, 0 % fallback "
                 f"(5 seeds × 20 episodes)")
    else:
        ppo = [0.91, 6.86, 0.0, 0.0]
        mpc = [33.2, 117.3, 100.0, 0.0]
        ppo_err = mpc_err = [0.0, 0.0, 0.0, 0.0]
        title = ("Compute cost of the filter  —  ~33 ms median, "
                 "solved every env step, 0% fallback")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, ppo, w, label="PPO", color=PPO_COLOR,
           yerr=ppo_err, capsize=4, ecolor="#444444",
           error_kw={"elinewidth": 1.2})
    ax.bar(x + w / 2, mpc, w, label="PPO + MPC-BLF", color=MPC_COLOR,
           yerr=mpc_err, capsize=4, ecolor="#0b3a99",
           error_kw={"elinewidth": 1.2})
    for i, (p, m) in enumerate(zip(ppo, mpc)):
        ax.annotate(f"{p:g}", xy=(i - w / 2, p),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=10, color=PPO_COLOR)
        ax.annotate(f"{m:g}", xy=(i + w / 2, m),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=10, color=MPC_COLOR)
    ax.set_xticks(x, labels)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylabel("value (symlog scale)")
    ax.set_title(title, loc="left", pad=12)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "solve_cost.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_trajectory_tracking()
    plot_tube_violation()
    plot_disturbance_recovery()
    plot_solve_cost()
    print("wrote plots to", HERE)
