"""
Continue PPO training from an existing checkpoint for an additional N steps.

Usage:
    python train_ppo_continue.py
    python train_ppo_continue.py --init logs/best_model_5M.zip --steps 2000000 --tag 7M

Key design choices vs. train_ppo.py:
  - Loads weights from `--init` instead of starting from scratch.
  - Keeps the same env + vec layout so observation/action shapes match.
  - Uses a SEPARATE best-model save path (logs/best_model_<tag>.zip) so the
    pre-existing best is never overwritten by continuation.
  - Lower learning rate (1e-4 default) since the policy is already converged
    and we want to fine-tune, not re-explore.
  - Continues tensorboard / num_timesteps counters across runs.
"""

import argparse
import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.aerial_manipulator import AerialManipulatorEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", default="logs/best_model.zip",
                        help="Checkpoint to warm-start from")
    parser.add_argument("--steps", type=int, default=2_000_000,
                        help="Additional environment steps to train")
    parser.add_argument("--tag", default="7M",
                        help="Suffix for the new best-model save path")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Fine-tuning learning rate (lower than from-scratch)")
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--xml", default="models/skydio_arm.xml")
    parser.add_argument("--device", default="cuda:6")
    args = parser.parse_args()

    if not os.path.exists(args.init):
        raise FileNotFoundError(f"Checkpoint not found: {args.init}")

    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print(f"Continuing PPO training")
    print(f"  init checkpoint : {args.init}")
    print(f"  additional steps: {args.steps:,}")
    print(f"  save tag        : {args.tag}")
    print(f"  learning rate   : {args.lr}")

    env = make_vec_env(
        lambda: AerialManipulatorEnv(model_path=args.xml, render_mode=None),
        n_envs=args.n_envs,
    )
    eval_env = make_vec_env(
        lambda: AerialManipulatorEnv(model_path=args.xml, render_mode=None),
        n_envs=1,
    )

    save_dir = f"./logs/continue_{args.tag}/"
    os.makedirs(save_dir, exist_ok=True)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=5000,
        deterministic=True,
        render=False,
    )

    model = PPO.load(
        args.init,
        env=env,
        device=args.device,
        tensorboard_log="./logs/ppo_aerial_manipulator_tensorboard/",
    )
    model.learning_rate = args.lr
    model._setup_lr_schedule()

    print(f"Loaded model. Current num_timesteps={model.num_timesteps:,}")
    print(f"Training for {args.steps:,} more steps "
          f"(target total {model.num_timesteps + args.steps:,})...")

    model.learn(
        total_timesteps=args.steps,
        callback=eval_callback,
        reset_num_timesteps=False,
        tb_log_name=f"continue_{args.tag}",
    )

    final_path = f"models/ppo_aerial_manipulator_{args.tag}"
    model.save(final_path)
    print(f"Saved final model to {final_path}.zip")
    print(f"Best model (per eval) at {save_dir}best_model.zip")


if __name__ == "__main__":
    main()
