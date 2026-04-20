"""Background PPO training pipeline that mirrors the original 7 M-step run.

Protocol (same hyperparameters as the original train_ppo.py +
train_ppo_continue.py):

  Phase 1: PPO from scratch, 5 M env steps, lr = 3e-4, 16 parallel envs.
  Phase 2: warm-start from the Phase 1 best checkpoint, another 2 M env
           steps at the fine-tuning lr = 1e-4.

All artefacts go under ``logs/retrain_7M/`` so the committed
``logs/continue_7M/best_model.zip`` reference checkpoint is NEVER touched.

After training finishes, parses the tensorboard event files and writes
training plots to ``presentation/plots/training_*.png``.

Intended usage (background):

    nohup python scripts/retrain_from_scratch.py \\
        > logs/retrain_7M/pipeline.log 2>&1 &

Expected wall-clock: several hours on a single RTX 6000 Ada.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Allow importing the project modules regardless of where we launch from.
REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import EvalCallback  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402

from env.aerial_manipulator import AerialManipulatorEnv  # noqa: E402


ROOT_DIR  = REPO / "logs" / "retrain_7M"
TB_DIR    = ROOT_DIR / "tb"
PHASE1_DIR = ROOT_DIR / "phase1"  # 0 -> 5 M
PHASE2_DIR = ROOT_DIR / "phase2"  # 5 M -> 7 M
PLOT_DIR  = REPO / "presentation" / "plots"
XML_PATH  = "models/skydio_arm.xml"


def _make_envs(n_envs: int):
    env = make_vec_env(
        lambda: AerialManipulatorEnv(model_path=XML_PATH, render_mode=None),
        n_envs=n_envs,
    )
    eval_env = make_vec_env(
        lambda: AerialManipulatorEnv(model_path=XML_PATH, render_mode=None),
        n_envs=1,
    )
    return env, eval_env


def phase1_from_scratch(
    *, total_steps: int, n_envs: int, device: str, tb_log_name: str,
) -> Path:
    """5 M-step from-scratch PPO. Mirrors train_ppo.py."""
    print(f"[phase1] training {total_steps:,} steps from scratch")
    PHASE1_DIR.mkdir(parents=True, exist_ok=True)

    env, eval_env = _make_envs(n_envs)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(PHASE1_DIR),
        log_path=str(PHASE1_DIR),
        eval_freq=5000,
        deterministic=True,
        render=False,
    )
    model = PPO(
        "MlpPolicy", env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=str(TB_DIR),
        device=device,
    )
    model.learn(total_timesteps=total_steps, callback=eval_cb,
                tb_log_name=tb_log_name)
    final_path = PHASE1_DIR / "final_model.zip"
    model.save(final_path)
    print(f"[phase1] saved {final_path}")

    best = PHASE1_DIR / "best_model.zip"
    if not best.exists():
        # Fallback: EvalCallback didn't record a best (shouldn't happen
        # in practice, but guard anyway so phase2 has something to load).
        model.save(best)
    return best


def phase2_continue(
    *, init_ckpt: Path, total_steps: int, n_envs: int, device: str,
    lr: float, tb_log_name: str,
) -> Path:
    """2 M-step fine-tune from the phase-1 best checkpoint."""
    print(f"[phase2] continuing {total_steps:,} steps from {init_ckpt}")
    PHASE2_DIR.mkdir(parents=True, exist_ok=True)

    env, eval_env = _make_envs(n_envs)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(PHASE2_DIR),
        log_path=str(PHASE2_DIR),
        eval_freq=5000,
        deterministic=True,
        render=False,
    )
    model = PPO.load(
        str(init_ckpt), env=env, device=device,
        tensorboard_log=str(TB_DIR),
    )
    model.learning_rate = lr
    model._setup_lr_schedule()
    print(f"[phase2] loaded model at num_timesteps={model.num_timesteps:,}")
    model.learn(
        total_timesteps=total_steps, callback=eval_cb,
        reset_num_timesteps=False, tb_log_name=tb_log_name,
    )
    final_path = PHASE2_DIR / "final_model.zip"
    model.save(final_path)
    print(f"[phase2] saved {final_path}")
    return final_path


# ---------------------------------------------------------------------- #
# Tensorboard -> plots                                                   #
# ---------------------------------------------------------------------- #
def _load_tb_scalars(tb_root: Path) -> dict[str, dict[str, np.ndarray]]:
    """Return {run_name: {tag: (steps, values)}} from tensorboard events."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError as exc:
        raise RuntimeError(
            "tensorboard package is required to parse event files. "
            "pip install tensorboard"
        ) from exc

    runs: dict[str, dict[str, np.ndarray]] = {}
    for sub in sorted(p for p in tb_root.glob("*") if p.is_dir()):
        ea = EventAccumulator(str(sub), size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        data: dict[str, np.ndarray] = {}
        for tag in tags:
            events = ea.Scalars(tag)
            steps = np.asarray([e.step for e in events], dtype=np.int64)
            vals  = np.asarray([e.value for e in events], dtype=np.float64)
            data[tag] = np.stack([steps, vals], axis=0)
        if data:
            runs[sub.name] = data
    return runs


def _concat_across_runs(runs: dict[str, dict[str, np.ndarray]],
                        tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate (steps, values) for `tag` across runs ordered by step."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for _, tags in runs.items():
        arr = tags.get(tag)
        if arr is None or arr.shape[1] == 0:
            continue
        xs.append(arr[0])
        ys.append(arr[1])
    if not xs:
        return np.empty(0), np.empty(0)
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    order = np.argsort(x, kind="stable")
    return x[order], y[order]


def _smooth_ema(y: np.ndarray, alpha: float = 0.02) -> np.ndarray:
    """Tensorboard-style EMA smoothing."""
    if len(y) == 0:
        return y
    out = np.empty_like(y, dtype=np.float64)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1.0 - alpha) * out[i - 1]
    return out


def _plot_single(step, val, title, ylabel, out_path, *, smooth=True,
                 phase1_end=None):
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 12, "font.family": "DejaVu Sans",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25,
        "figure.facecolor": "white", "savefig.dpi": 150,
    })
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(step, val, color="#b5c5f0", lw=1.0, label="raw")
    if smooth and len(val) > 5:
        ax.plot(step, _smooth_ema(val, 0.02),
                color="#2c6df5", lw=1.8, label="EMA (α=0.02)")
    if phase1_end is not None and step.size and step.max() > phase1_end:
        ax.axvline(phase1_end, color="#888", ls="--", lw=1.0)
        ax.annotate(
            "phase1 → phase2", xy=(phase1_end, ax.get_ylim()[1]),
            xytext=(5, -5), textcoords="offset points",
            fontsize=10, color="#666", va="top",
        )
    ax.set_xlabel("env steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=12)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def make_plots(phase1_steps: int):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[plots] reading tensorboard events from {TB_DIR}")
    runs = _load_tb_scalars(TB_DIR)
    if not runs:
        print("[plots] no tb runs found; skipping plots")
        return

    def _plot(tag: str, title: str, ylabel: str, stem: str, *,
              smooth: bool = True):
        x, y = _concat_across_runs(runs, tag)
        if x.size == 0:
            print(f"[plots] tag {tag!r} not found in any run; skipped")
            return
        out = PLOT_DIR / f"training_{stem}.png"
        _plot_single(x, y, title, ylabel, out,
                     smooth=smooth, phase1_end=phase1_steps)

    _plot("rollout/ep_rew_mean",
          "Mean episode reward during PPO training",
          "mean episode reward", "reward")
    _plot("train/value_loss",
          "PPO value-function loss",
          "value loss", "value_loss")
    _plot("rollout/ep_len_mean",
          "Mean episode length",
          "mean episode length", "ep_len")
    _plot("train/policy_gradient_loss",
          "PPO policy gradient loss",
          "policy gradient loss", "policy_loss",
          smooth=True)
    _plot("train/explained_variance",
          "Explained variance of the value function",
          "explained variance", "explained_variance",
          smooth=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-steps", type=int, default=5_000_000)
    parser.add_argument("--phase2-steps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--phase2-lr", type=float, default=1e-4)
    parser.add_argument("--plots-only", action="store_true",
                        help="Skip training, just regenerate plots from "
                             "existing tb events under logs/retrain_7M/tb.")
    args = parser.parse_args()

    ROOT_DIR.mkdir(parents=True, exist_ok=True)
    TB_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    if not args.plots_only:
        best_p1 = phase1_from_scratch(
            total_steps=args.phase1_steps,
            n_envs=args.n_envs,
            device=args.device,
            tb_log_name="phase1_from_scratch",
        )
        print(f"[wall] phase1 took {(time.time() - t0) / 60:.1f} min")

        t1 = time.time()
        phase2_continue(
            init_ckpt=best_p1,
            total_steps=args.phase2_steps,
            n_envs=args.n_envs,
            device=args.device,
            lr=args.phase2_lr,
            tb_log_name="phase2_continue_7M",
        )
        print(f"[wall] phase2 took {(time.time() - t1) / 60:.1f} min")

    print("[plots] generating training plots")
    make_plots(phase1_steps=args.phase1_steps)
    print(f"[done] total wall time {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
