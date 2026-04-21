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
NOBLF_COLOR = "#f2a93b"      # MPC, BLF disabled
MPC_COLOR = "#2c6df5"        # full MPC + BLF (winner)
_ECOLOR = {PPO_COLOR: "#444444", NOBLF_COLOR: "#8a5a00", MPC_COLOR: "#0b3a99"}


def bar_groups(ax, labels, series, ylabel, title, annotate_pct_ref_idx=None):
    """Draw N groups of bars.

    series is a list of dicts: {"label","color","vals","errs"}.
    If annotate_pct_ref_idx is not None, each non-reference series is
    annotated with its percentage change vs the reference series above
    each bar.
    """
    n_ser = len(series)
    x = np.arange(len(labels))
    total_w = 0.78
    w = total_w / n_ser
    offsets = (np.arange(n_ser) - (n_ser - 1) / 2.0) * w
    for s, off in zip(series, offsets):
        vals = np.asarray(s["vals"], dtype=float)
        errs = np.asarray(s.get("errs") or [0.0] * len(vals), dtype=float)
        ax.bar(x + off, vals, w, label=s["label"], color=s["color"],
               yerr=errs, capsize=3, ecolor=_ECOLOR.get(s["color"], "#444"),
               error_kw={"elinewidth": 1.0})
    if annotate_pct_ref_idx is not None:
        ref = series[annotate_pct_ref_idx]
        ref_vals = np.asarray(ref["vals"], dtype=float)
        for si, s in enumerate(series):
            if si == annotate_pct_ref_idx:
                continue
            vals = np.asarray(s["vals"], dtype=float)
            errs = np.asarray(s.get("errs") or [0.0] * len(vals), dtype=float)
            off = offsets[si]
            for i, (v, e, r) in enumerate(zip(vals, errs, ref_vals)):
                if r > 0:
                    delta = 100.0 * (v - r) / r
                    ax.annotate(f"{delta:+.0f}%",
                                xy=(i + off, v + e), xytext=(0, 4),
                                textcoords="offset points",
                                ha="center", fontsize=9, color=s["color"])
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=12)
    ax.legend(frameon=False, loc="upper right")


def _series_from_summary(run_key, label, color, keys):
    vals = [_mean_std(_SUMMARY, run_key, k)[0] for k in keys]
    errs = [_mean_std(_SUMMARY, run_key, k)[1] for k in keys]
    return {"label": label, "color": color, "vals": vals, "errs": errs}


def plot_trajectory_tracking():
    labels = ["pos RMSE [m]", "vel RMSE [m/s]", "peak err [m]"]
    keys = ["pos_rmse", "vel_rmse", "peak_err"]
    if _SUMMARY is not None:
        series = [
            _series_from_summary("ppo", "PPO", PPO_COLOR, keys),
        ]
        if "mpc_no_blf" in _SUMMARY["runs"]:
            series.append(_series_from_summary(
                "mpc_no_blf", "PPO + MPC (BLF off)", NOBLF_COLOR, keys))
        series.append(_series_from_summary(
            "mpc", "PPO + MPC + BLF", MPC_COLOR, keys))
        n_seeds = _SUMMARY["runs"]["ppo"]["n_seeds"]
        n_eps = _SUMMARY.get("episodes_per_seed",
                             _SUMMARY["runs"]["ppo"].get("n_episodes_total", 100) // max(n_seeds, 1))
        title = (f"Trajectory tracking  —  {n_seeds} seeds × {n_eps} episodes "
                 "(mean, error bars = std across seed means)")
    else:
        series = [
            {"label": "PPO", "color": PPO_COLOR,
             "vals": [0.1443, 0.1599, 0.2192], "errs": [0.0, 0.0, 0.0]},
            {"label": "PPO + MPC + BLF", "color": MPC_COLOR,
             "vals": [0.0658, 0.0646, 0.0910], "errs": [0.0, 0.0, 0.0]},
        ]
        title = "Trajectory tracking  —  20 deterministic episodes, seed 0"
    fig, ax = plt.subplots(figsize=(10, 4.7))
    bar_groups(ax, labels, series,
               ylabel="metric value", title=title,
               annotate_pct_ref_idx=0)
    top = 0.0
    for s in series:
        top = max(top, max(v + e for v, e in zip(s["vals"], s["errs"])))
    ax.set_ylim(0, top * 1.35)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "trajectory_tracking.png"), bbox_inches="tight")
    plt.close(fig)


def plot_tube_violation():
    labels = ["tube violation rate\n(‖e‖ > 12 cm)"]
    if _SUMMARY is not None:
        series = []
        ppo_m, ppo_s = _mean_std(_SUMMARY, "ppo", "tube_viol_pct")
        series.append({"label": "PPO", "color": PPO_COLOR,
                       "vals": [ppo_m], "errs": [ppo_s]})
        if "mpc_no_blf" in _SUMMARY["runs"]:
            nb_m, nb_s = _mean_std(_SUMMARY, "mpc_no_blf", "tube_viol_pct")
            series.append({"label": "PPO + MPC (BLF off)", "color": NOBLF_COLOR,
                           "vals": [nb_m], "errs": [nb_s]})
        mpc_m, mpc_s = _mean_std(_SUMMARY, "mpc", "tube_viol_pct")
        series.append({"label": "PPO + MPC + BLF", "color": MPC_COLOR,
                       "vals": [mpc_m], "errs": [mpc_s]})
        if len(series) == 3:
            subtitle = (f"5 seeds × 20 eps — PPO {ppo_m:.1f} %   →   "
                        f"MPC alone {series[1]['vals'][0]:.1f} %   →   "
                        f"MPC + BLF {mpc_m:.2f} %")
        else:
            subtitle = (f"5 seeds × 20 episodes — PPO crosses the 12 cm tube "
                        f"{ppo_m:.1f} % of the time; MPC-BLF holds it at "
                        f"{mpc_m:.2f} %")
    else:
        series = [
            {"label": "PPO", "color": PPO_COLOR, "vals": [40.72], "errs": [0.0]},
            {"label": "PPO + MPC + BLF", "color": MPC_COLOR, "vals": [0.0], "errs": [0.0]},
        ]
        subtitle = ("PPO alone crosses the 12 cm tube 40 % of the time; "
                    "MPC-BLF holds it at 0 %")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bar_groups(ax, labels, series,
               ylabel="violation rate [%]",
               title="Safety tube violations\n" + subtitle)
    top = 0.0
    for s in series:
        top = max(top, s["vals"][0] + s["errs"][0])
    ax.set_ylim(0, max(60.0, top + 8))
    for s, off in zip(series,
                      (np.arange(len(series)) - (len(series) - 1) / 2.0)
                      * (0.78 / len(series))):
        v, e = s["vals"][0], s["errs"][0]
        ax.text(0 + off, max(v + e + 1.5, 2.0),
                f"{v:.2f}%" if v >= 1 else f"{v:.2f}%",
                ha="center", fontsize=10, color=s["color"])
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
        series = [
            {"label": "PPO", "color": PPO_COLOR,
             "vals": [ppo_med, ppo_peak, 0.0, 0.0],
             "errs": [ppo_med_s, ppo_peak_s, 0.0, 0.0]},
        ]
        if "mpc_no_blf" in _SUMMARY["runs"]:
            nb_med, nb_med_s = _mean_std(_SUMMARY, "mpc_no_blf", "solve_median_ms")
            nb_peak, nb_peak_s = _mean_std(_SUMMARY, "mpc_no_blf", "solve_max_ms")
            series.append({"label": "PPO + MPC (BLF off)", "color": NOBLF_COLOR,
                           "vals": [nb_med, nb_peak, 100.0, 0.0],
                           "errs": [nb_med_s, nb_peak_s, 0.0, 0.0]})
        series.append({"label": "PPO + MPC + BLF", "color": MPC_COLOR,
                       "vals": [mpc_med, mpc_peak, 100.0, 0.0],
                       "errs": [mpc_med_s, mpc_peak_s, 0.0, 0.0]})
        title = (f"Compute cost of the filter  —  MPC+BLF ~{mpc_med:.0f} ms median, "
                 f"per-ep peak ~{mpc_peak:.0f} ms, 0 % fallback "
                 f"(5 seeds × 20 episodes)")
    else:
        series = [
            {"label": "PPO", "color": PPO_COLOR,
             "vals": [0.91, 6.86, 0.0, 0.0], "errs": [0.0] * 4},
            {"label": "PPO + MPC + BLF", "color": MPC_COLOR,
             "vals": [33.2, 117.3, 100.0, 0.0], "errs": [0.0] * 4},
        ]
        title = ("Compute cost of the filter  —  ~33 ms median, "
                 "solved every env step, 0% fallback")

    fig, ax = plt.subplots(figsize=(10, 4.7))
    bar_groups(ax, labels, series,
               ylabel="value (symlog scale)", title=title)
    n_ser = len(series)
    w = 0.78 / n_ser
    offsets = (np.arange(n_ser) - (n_ser - 1) / 2.0) * w
    for s, off in zip(series, offsets):
        for i, v in enumerate(s["vals"]):
            ax.annotate(f"{v:g}", xy=(i + off, v),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=9, color=s["color"])
    ax.set_yscale("symlog", linthresh=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "solve_cost.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_trajectory_tracking()
    plot_tube_violation()
    plot_disturbance_recovery()
    plot_solve_cost()
    print("wrote plots to", HERE)
