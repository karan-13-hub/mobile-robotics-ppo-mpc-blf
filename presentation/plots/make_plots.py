"""Generate comparison plots for the presentation from METRICS.txt figures.

All numbers are hard-coded from videos/METRICS.txt so the plots are
reproducible even without re-running eval. Any change to the metrics
only needs an update here.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

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


def bar_compare(ax, labels, ppo_vals, mpc_vals, ylabel, title, pct_drop=True):
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, ppo_vals, w, label="PPO", color=PPO_COLOR)
    ax.bar(x + w / 2, mpc_vals, w, label="PPO + MPC-BLF", color=MPC_COLOR)
    for i, (p, m) in enumerate(zip(ppo_vals, mpc_vals)):
        if pct_drop and p > 0:
            delta = 100.0 * (m - p) / p
            ax.annotate(f"{delta:+.0f}%",
                        xy=(i + w / 2, m), xytext=(0, 4),
                        textcoords="offset points",
                        ha="center", fontsize=10, color=MPC_COLOR)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=12)
    ax.legend(frameon=False, loc="upper right")


def plot_trajectory_tracking():
    labels = ["pos RMSE [m]", "vel RMSE [m/s]", "peak err [m]", "EE RMSE [m]"]
    ppo = [0.1443, 0.1599, 0.2192, 0.2243]
    mpc = [0.0658, 0.0646, 0.0910, 0.1868]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bar_compare(ax, labels, ppo, mpc,
                ylabel="metric value",
                title="Trajectory tracking  —  20 deterministic episodes, seed 0")
    ax.set_ylim(0, max(max(ppo), max(mpc)) * 1.25)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "trajectory_tracking.png"), bbox_inches="tight")
    plt.close(fig)


def plot_tube_violation():
    labels = ["tube violation rate\n(>12 cm)"]
    ppo = [40.72]
    mpc = [0.00]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(labels))
    w = 0.5
    ax.bar(x - w / 2, ppo, w, label="PPO", color=PPO_COLOR)
    ax.bar(x + w / 2, mpc, w, label="PPO + MPC-BLF", color=MPC_COLOR)
    ax.text(0 - w / 2, ppo[0] + 1.5, f"{ppo[0]:.2f} %",
            ha="center", fontsize=12, color=PPO_COLOR)
    ax.text(0 + w / 2, 1.5, f"{mpc[0]:.2f} %",
            ha="center", fontsize=12, color=MPC_COLOR)
    ax.set_xticks(x, labels)
    ax.set_ylabel("violation rate [%]")
    ax.set_ylim(0, 50)
    ax.set_title("Safety tube violations  —  PPO alone crosses the 12 cm tube\n"
                 "40% of the time; MPC-BLF holds it at 0%",
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
    labels = ["median solve\n[ms]", "peak solve\n[ms]", "MPC duty cycle\n[%]",
              "MPC fallback\n[%]"]
    ppo = [0.91, 6.86, 0, 0]
    mpc = [33.2, 117.3, 100.0, 0.0]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, ppo, w, label="PPO", color=PPO_COLOR)
    ax.bar(x + w / 2, mpc, w, label="PPO + MPC-BLF", color=MPC_COLOR)
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
    ax.set_title("Compute cost of the filter  —  ~33 ms median, "
                 "solved every env step, 0% fallback",
                 loc="left", pad=12)
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
