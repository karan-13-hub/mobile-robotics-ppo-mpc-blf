"""
Eval the PPO baseline with an MPC + BLF safety filter interposed.

Mirrors `eval_ppo.py` so headline metrics (crash, reach, RMSE, final_d) are
directly comparable to the unfiltered baseline. Adds two metrics that capture
the BLF objective explicitly:

  - Tube violation rate          fraction of in-trajectory steps with ||e|| > delta
  - Peak tracking error          max ||e|| per episode, averaged across episodes

Per-step diagnostics include MPC V(e_0), V(e_N), solve time, and fallback usage
so we can tell when the filter was actively pushing PPO around vs. idling.

Usage:
    python eval_ppo_mpc.py                                          # tuned defaults
    python eval_ppo_mpc.py --no-mpc                                 # PPO-only ablation
    python eval_ppo_mpc.py --tube 0.10                              # tighter BLF tube
    python eval_ppo_mpc.py --mpc-stride 1                           # solve every step
    python eval_ppo_mpc.py --save-video out.mp4 --video-episode 3
"""

import os
import sys

if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"

import argparse
import numpy as np
import mujoco
from stable_baselines3 import PPO

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.aerial_manipulator import AerialManipulatorEnv
from controllers import MPCBLFSafetyFilter
from controllers.mpc_blf_filter import mj_state_to_x12

# Reuse rendering, plotting and saving helpers from the baseline eval.
from eval_ppo import (
    _draw_trajectory_overlays,
    save_video,
    plot_episode,
)


def _sample_ref_horizon(env, t_now, N, dt):
    """Sample (N+1, 3) base reference positions at t_now + j*dt for j=0..N."""
    return np.array(
        [env._eval_trajectory(t_now + j * dt)[0] for j in range(N + 1)],
        dtype=np.float64,
    )


def _sample_ref_vel_horizon(env, t_now, N, dt):
    """Sample (N+1, 3) base reference velocities at t_now + j*dt."""
    return np.array(
        [env._eval_trajectory(t_now + j * dt)[1] for j in range(N + 1)],
        dtype=np.float64,
    )


def rollout_episode(
    env,
    model,
    *,
    mpc_filter=None,
    mpc_stride=1,
    seed=None,
    renderer=None,
    camera="track",
    frame_stride=1,
):
    """Run one deterministic episode with optional MPC safety filtering."""
    obs, info = env.reset(seed=seed)
    if mpc_filter is not None:
        mpc_filter.reset()

    T_traj = info["T_traj"]
    goal_pos = info["goal_pos"]
    dt = env.model.opt.timestep
    N = mpc_filter.N if mpc_filter is not None else 0

    log = {
        "t": [],
        "actual_pos": [],
        "actual_vel": [],
        "actual_ee": [],
        "ref_pos": [],
        "ref_vel": [],
        "pos_err": [],
        "vel_err": [],
        "ee_err": [],
        "goal_dist": [],
        "ppo_action": [],
        "safe_action": [],
        "mpc_V0": [],
        "mpc_VN": [],
        "mpc_sigma": [],
        "mpc_solve_ms": [],
        "mpc_fallback": [],
        "mpc_active": [],
        "mpc_u_change": [],
    }
    frames = [] if renderer is not None else None
    crashed = False
    done = False
    step_idx = 0
    last_safe_action = None  # for ZOH between MPC solves when stride > 1

    ref_path = None
    if renderer is not None:
        n_samples = max(20, int(np.ceil(T_traj / 0.05)))
        ts = np.linspace(0.0, T_traj, n_samples)
        ref_path = np.array([env._eval_trajectory(t)[0] for t in ts])

    if renderer is not None:
        renderer.update_scene(env.data, camera=camera)
        _draw_trajectory_overlays(
            renderer, ref_path, [env.data.qpos[0:3].copy()],
            goal_pos, env.base_target_pos,
        )
        frames.append(renderer.render().copy())

    while not done:
        ppo_action, _ = model.predict(obs, deterministic=True)

        if mpc_filter is not None:
            # MPC sees current state (before env.step has advanced).
            t_now = env.current_step * dt
            ref_traj = _sample_ref_horizon(env, t_now, N, dt)
            ref_vel_traj = _sample_ref_vel_horizon(env, t_now, N, dt)
            x_meas = mj_state_to_x12(env.data)

            do_solve = (step_idx % mpc_stride == 0) or (last_safe_action is None)
            if do_solve:
                safe_action, mpc_info = mpc_filter.filter(
                    x_meas, ppo_action, ref_traj, ref_vel_traj
                )
                last_safe_action = safe_action.copy()
            else:
                # Between MPC solves, pass PPO through directly. We deliberately
                # do NOT zero-order-hold the previous rotor command: the inner
                # attitude loop is fast (~5 Hz settling for tilt) so locking
                # rotor cmds for >10 ms quickly drives the drone unstable.
                safe_action = np.asarray(ppo_action, dtype=np.float32)
                mpc_info = {
                    "fallback": False, "mpc_active": False,
                    "V0": 0.0, "VN": 0.0, "sigma": 0.0,
                    "solve_time_ms": 0.0, "u_changed_norm": 0.0,
                }
        else:
            safe_action = np.asarray(ppo_action, dtype=np.float32)
            mpc_info = {
                "fallback": True, "mpc_active": False,
                "V0": 0.0, "VN": 0.0, "sigma": 0.0,
                "solve_time_ms": 0.0, "u_changed_norm": 0.0,
            }

        obs, _, terminated, truncated, info = env.step(safe_action)
        step_idx += 1

        t = env.current_step * dt
        log["t"].append(t)
        log["actual_pos"].append(env.data.qpos[0:3].copy())
        log["actual_vel"].append(env.data.qvel[0:3].copy())
        log["actual_ee"].append(info["end_effector_pos"])
        log["ref_pos"].append(info["ref_pos"])
        log["ref_vel"].append(info["ref_vel"])
        log["pos_err"].append(info["pos_error"])
        log["vel_err"].append(info["vel_error"])
        log["ee_err"].append(info["ee_error"])
        log["goal_dist"].append(info["goal_dist"])
        log["ppo_action"].append(np.asarray(ppo_action, dtype=np.float32).copy())
        log["safe_action"].append(np.asarray(safe_action, dtype=np.float32).copy())
        log["mpc_V0"].append(float(mpc_info.get("V0", 0.0)))
        log["mpc_VN"].append(float(mpc_info.get("VN", 0.0)))
        log["mpc_sigma"].append(float(mpc_info.get("sigma", 0.0)))
        log["mpc_solve_ms"].append(float(mpc_info.get("solve_time_ms", 0.0)))
        log["mpc_fallback"].append(bool(mpc_info.get("fallback", False)))
        log["mpc_active"].append(bool(mpc_info.get("mpc_active", False)))
        log["mpc_u_change"].append(float(mpc_info.get("u_changed_norm", 0.0)))

        if renderer is not None and (step_idx % frame_stride == 0):
            renderer.update_scene(env.data, camera=camera)
            _draw_trajectory_overlays(
                renderer, ref_path, log["actual_pos"],
                goal_pos, info["ref_pos"],
            )
            frames.append(renderer.render().copy())

        if terminated:
            crashed = True
        done = terminated or truncated

    for k, v in log.items():
        log[k] = np.asarray(v)

    return log, {
        "crashed": crashed,
        "goal_pos": goal_pos,
        "T_traj": T_traj,
        "dt": dt,
        "frames": frames,
    }


def episode_metrics(log, meta, *, goal_tol=0.15, tube=0.05):
    """Episode-level reduction (extends the baseline with tube metrics)."""
    T_traj = meta["T_traj"]
    in_traj = log["t"] <= T_traj
    if in_traj.sum() == 0:
        in_traj = np.ones_like(log["t"], dtype=bool)

    pos_rmse = float(np.sqrt(np.mean(log["pos_err"][in_traj] ** 2)))
    vel_rmse = float(np.sqrt(np.mean(log["vel_err"][in_traj] ** 2)))
    ee_rmse = float(np.sqrt(np.mean(log["ee_err"][in_traj] ** 2)))
    final_goal_dist = float(log["goal_dist"][-1])
    reached_goal = (not meta["crashed"]) and (final_goal_dist < goal_tol)

    e_norm = log["pos_err"][in_traj]
    tube_violation_rate = float(np.mean(e_norm > tube))
    peak_err = float(np.max(log["pos_err"]))

    nonzero_solve = log["mpc_solve_ms"][log["mpc_solve_ms"] > 0]
    median_solve_ms = float(np.median(nonzero_solve)) if nonzero_solve.size else 0.0
    max_solve_ms = float(np.max(log["mpc_solve_ms"])) if log["mpc_solve_ms"].size else 0.0
    fallback_rate = float(np.mean(log["mpc_fallback"]))
    active_rate = float(np.mean(log["mpc_active"])) if log["mpc_active"].size else 0.0

    return {
        "pos_rmse": pos_rmse,
        "vel_rmse": vel_rmse,
        "ee_rmse": ee_rmse,
        "final_goal_dist": final_goal_dist,
        "reached_goal": reached_goal,
        "crashed": meta["crashed"],
        "T_traj": T_traj,
        "ep_len_s": float(log["t"][-1]),
        "tube_violation_rate": tube_violation_rate,
        "peak_err": peak_err,
        "median_solve_ms": median_solve_ms,
        "max_solve_ms": max_solve_ms,
        "fallback_rate": fallback_rate,
        "active_rate": active_rate,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="logs/continue_7M/best_model.zip")
    parser.add_argument("--xml", default="models/skydio_arm.xml")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--goal-tol", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--no-blf", action="store_true",
                        help="Disable the BLF safety constraint inside the "
                             "MPC NLP, leaving a pure tracking optimiser. "
                             "Useful as a 'MPC without BLF' ablation.")
    parser.add_argument("--no-mpc", action="store_true",
                        help="Bypass the MPC filter; reproduces eval_ppo.py.")
    parser.add_argument("--mpc-horizon", type=int, default=7)
    parser.add_argument("--tube", type=float, default=0.15,
                        help="BLF tube radius delta (metres, default 15 cm). "
                             "V(e) = ||e||^2 / (delta^2 - ||e||^2). Finite V "
                             "<=> ||e|| < delta.")
    parser.add_argument("--beta-start", type=float, default=0.95,
                        help="BLF descent factor at k=0 (loose; near-term "
                             "errors are largely unavoidable so beta_start ~ 1 "
                             "is permissive).")
    parser.add_argument("--beta-end", type=float, default=0.15,
                        help="BLF descent factor at k=N-1 (strict, <1 forces "
                             "V to contract at the horizon end).")
    parser.add_argument("--slack-penalty", type=float, default=3e3,
                        help="Quadratic penalty rho on the scalar BLF slack "
                             "sigma (rho * sigma^2). sigma softens the "
                             "V-descent constraint against motor saturation; "
                             "larger rho => tighter descent but higher "
                             "infeasibility risk.")
    parser.add_argument("--smooth-penalty", type=float, default=1e-2)
    parser.add_argument("--velocity-penalty", type=float, default=0.0,
                        help="Stage-wise velocity penalty w_v: adds "
                             "(w_v/N) * sum_k ||v_k||^2 to the cost. Acts "
                             "as a derivative-style damping term.")
    parser.add_argument("--barrier-velocity-weight", type=float, default=0.0,
                        help="alpha for velocity-aware BLF: V uses "
                             "z = ||e_p||^2 + alpha * ||e_v||^2 instead of "
                             "pure position error (0 => position-only BLF).")
    parser.add_argument("--mpc-stride", type=int, default=3,
                        help="Solve MPC every Nth step; PPO passes through "
                             "directly on the intervening steps (no ZOH).")
    parser.add_argument("--mpc-solver", default="fatrop",
                        choices=["ipopt", "sqpmethod", "fatrop"],
                        help="NLP backend for the MPC (default: fatrop).")
    parser.add_argument("--mpc-verbose", action="store_true")

    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-episode", type=int, default=0)
    parser.add_argument("--save-plot", default=None)
    parser.add_argument("--save-video", default=None)
    parser.add_argument("--video-episode", type=int, default=None)
    parser.add_argument("--video-camera", default="track")
    parser.add_argument("--video-width", type=int, default=960)
    parser.add_argument("--video-height", type=int, default=540)
    parser.add_argument("--video-stride", type=int, default=2)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found at {args.model}.")
        return

    if args.no_mpc:
        label = "PPO baseline (no MPC)"
    elif args.no_blf:
        label = (f"PPO + MPC (BLF disabled, tube={args.tube*100:.1f} cm, "
                 f"N={args.mpc_horizon})")
    else:
        label = (f"PPO + MPC-BLF (tube={args.tube*100:.1f} cm, "
                 f"N={args.mpc_horizon})")
    print(f"Evaluating: {label}")

    model = PPO.load(args.model)
    env = AerialManipulatorEnv(model_path=args.xml, render_mode=None)

    mpc_filter = None
    if not args.no_mpc:
        mpc_filter = MPCBLFSafetyFilter(
            env.model,
            horizon=args.mpc_horizon,
            tube=args.tube,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            slack_penalty=args.slack_penalty,
            smooth_penalty=args.smooth_penalty,
            velocity_penalty=args.velocity_penalty,
            barrier_velocity_weight=args.barrier_velocity_weight,
            solver=args.mpc_solver,
            verbose=args.mpc_verbose,
            enable_blf=not args.no_blf,
        )

    rng_seed = args.seed
    metrics = []
    plot_log, plot_meta = None, None

    video_episode = args.video_episode if args.video_episode is not None else args.plot_episode
    renderer = None
    if args.save_video:
        renderer = mujoco.Renderer(env.model, height=args.video_height,
                                   width=args.video_width, max_geom=20000)

    print(f"Running {args.episodes} deterministic episodes...")
    for ep in range(args.episodes):
        record_this = (renderer is not None) and (ep == video_episode)
        log, meta = rollout_episode(
            env, model,
            mpc_filter=mpc_filter,
            mpc_stride=args.mpc_stride,
            seed=rng_seed + ep,
            renderer=renderer if record_this else None,
            camera=args.video_camera,
            frame_stride=args.video_stride,
        )
        m = episode_metrics(log, meta, goal_tol=args.goal_tol, tube=args.tube)
        metrics.append(m)

        if ep == args.plot_episode:
            plot_log, plot_meta = log, meta

        if record_this and meta.get("frames"):
            dt = env.model.opt.timestep
            fps = max(1, int(round(1.0 / (dt * args.video_stride))))
            save_video(meta["frames"], args.save_video, fps=fps)

        status = "CRASH" if m["crashed"] else ("REACH" if m["reached_goal"] else "miss ")
        mpc_part = ""
        if mpc_filter is not None:
            mpc_part = (f" | tube_viol={100*m['tube_violation_rate']:.1f}% "
                        f"peak_e={m['peak_err']:.3f} "
                        f"solve={m['median_solve_ms']:.1f}/{m['max_solve_ms']:.1f}ms"
                        f" act={100*m['active_rate']:.0f}%"
                        f" fb={100*m['fallback_rate']:.0f}%")
        print(
            f"  ep {ep:02d} | T={m['T_traj']:.2f}s len={m['ep_len_s']:.2f}s"
            f" | pos_RMSE={m['pos_rmse']:.3f} vel_RMSE={m['vel_rmse']:.3f}"
            f" ee_RMSE={m['ee_rmse']:.3f} | final_d={m['final_goal_dist']:.3f}"
            f" | {status}{mpc_part}"
        )

    n = len(metrics)
    crash_rate = 100.0 * sum(m["crashed"] for m in metrics) / n
    reach_rate = 100.0 * sum(m["reached_goal"] for m in metrics) / n
    mean_pos = np.mean([m["pos_rmse"] for m in metrics])
    mean_vel = np.mean([m["vel_rmse"] for m in metrics])
    mean_ee = np.mean([m["ee_rmse"] for m in metrics])
    mean_final = np.mean([m["final_goal_dist"] for m in metrics])
    mean_tube_viol = 100.0 * np.mean([m["tube_violation_rate"] for m in metrics])
    mean_peak = np.mean([m["peak_err"] for m in metrics])

    print(f"\n--- {label} ---")
    print(f"Episodes                            : {n}")
    print(f"Crash rate                          : {crash_rate:.2f}%")
    print(f"Goal reach rate (tol={args.goal_tol:.2f} m)   : {reach_rate:.2f}%")
    print(f"Position tracking RMSE  [m]         : {mean_pos:.4f}")
    print(f"Velocity tracking RMSE  [m/s]       : {mean_vel:.4f}")
    print(f"End-effector tracking RMSE [m]      : {mean_ee:.4f}")
    print(f"Final settling error    [m]         : {mean_final:.4f}")
    print(f"Tube violation rate (||e|| > {args.tube*100:.1f} cm) : {mean_tube_viol:.2f}%")
    print(f"Peak tracking error     [m]         : {mean_peak:.4f}")

    if mpc_filter is not None:
        all_solve = np.concatenate([np.atleast_1d(np.asarray([m["median_solve_ms"]])) for m in metrics])
        all_max = np.concatenate([np.atleast_1d(np.asarray([m["max_solve_ms"]])) for m in metrics])
        all_fb = np.concatenate([np.atleast_1d(np.asarray([m["fallback_rate"]])) for m in metrics])
        all_active = np.concatenate([np.atleast_1d(np.asarray([m["active_rate"]])) for m in metrics])
        print(f"MPC median solve time   [ms]        : {np.median(all_solve):.2f}")
        print(f"MPC peak  solve time    [ms]        : {np.max(all_max):.2f}")
        print(f"MPC active rate (NLP actually solved): {100.0 * np.mean(all_active):.2f}%")
        print(f"MPC fallback rate                   : {100.0 * np.mean(all_fb):.2f}%")

    if args.plot and plot_log is not None:
        plot_episode(plot_log, plot_meta, save_path=args.save_plot)


if __name__ == "__main__":
    main()
