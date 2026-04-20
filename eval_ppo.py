"""
Phase-2 PPO baseline evaluation against the min-snap reference trajectory.

Reports, over N episodes, the metrics that matter for the downstream
MPC-CBF / flow-matching stages:

  - Crash rate                       (terminations from OOB / ground impact)
  - Goal reach rate                  (||p - goal|| < goal_tol at end)
  - Position tracking RMSE           (||p - p_ref||  during the trajectory)
  - Velocity tracking RMSE           (||v - v_ref||  during the trajectory)
  - End-effector tracking RMSE       (||ee - ee_ref|| during the trajectory)
  - Final settling error             (||p - goal||  at the end of the episode)

Optionally plots the reference vs. realized 3D path of the first episode.

Usage:
    python eval_ppo.py
    python eval_ppo.py --episodes 50 --plot
    python eval_ppo.py --plot --plot-episode 3 --save-plot eval.png
    python eval_ppo.py --save-video out.mp4 --video-episode 19
"""

import os
import sys

# Headless rendering: if no X display is available, force MuJoCo to use EGL
# for off-screen rendering. Must run BEFORE `import mujoco`.
if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"

import argparse
import numpy as np
import mujoco
from stable_baselines3 import PPO

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.aerial_manipulator import AerialManipulatorEnv


_IDENT_MAT = np.eye(3, dtype=np.float64).flatten()


def _scene_add_capsule(scene, p1, p2, rgba, width=0.008):
    """Append a capsule connecting p1 -> p2 to a MjvScene's geom buffer."""
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=np.zeros(3),
        pos=np.zeros(3),
        mat=_IDENT_MAT,
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        g,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        width,
        np.asarray(p1, dtype=np.float64),
        np.asarray(p2, dtype=np.float64),
    )
    scene.ngeom += 1


def _scene_add_sphere(scene, pos, rgba, radius=0.05):
    """Append a sphere marker to a MjvScene's geom buffer."""
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, radius, radius], dtype=np.float64),
        pos=np.asarray(pos, dtype=np.float64),
        mat=_IDENT_MAT,
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _draw_trajectory_overlays(renderer, ref_path, actual_path, goal_pos,
                              current_ref):
    """Draw the planned reference, the realized path, goal, and current setpoint."""
    scene = renderer.scene

    # Planned reference trajectory: translucent blue
    for i in range(len(ref_path) - 1):
        _scene_add_capsule(scene, ref_path[i], ref_path[i + 1],
                           rgba=[0.20, 0.50, 1.00, 0.55], width=0.010)

    # Actual path traced so far: solid red, slightly thicker
    if len(actual_path) >= 2:
        ap = np.asarray(actual_path)
        for i in range(len(ap) - 1):
            _scene_add_capsule(scene, ap[i], ap[i + 1],
                               rgba=[1.00, 0.20, 0.15, 0.95], width=0.012)

    # Goal marker: gold sphere
    _scene_add_sphere(scene, goal_pos,
                      rgba=[1.00, 0.84, 0.00, 1.00], radius=0.06)

    # Current reference setpoint (where the policy is being asked to be):
    # small cyan sphere — useful to see lag visually.
    _scene_add_sphere(scene, current_ref,
                      rgba=[0.20, 1.00, 1.00, 0.95], radius=0.035)


def rollout_episode(env, model, seed=None, renderer=None, camera="track",
                    frame_stride=1):
    """
    Run one deterministic episode, returning per-step logs.

    If `renderer` (mujoco.Renderer) is provided, also captures one RGB frame
    every `frame_stride` sim steps and returns them as a list of HxWx3 uint8
    arrays in `meta["frames"]`. Each frame overlays:
      - the full planned reference trajectory (blue)
      - the realized aerial-manipulator path so far (red)
      - the goal (gold sphere)
      - the current reference setpoint (cyan sphere)
    """
    obs, info = env.reset(seed=seed)
    T_traj = info["T_traj"]
    goal_pos = info["goal_pos"]
    dt = env.model.opt.timestep

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
    }
    frames = [] if renderer is not None else None
    crashed = False
    done = False
    step_idx = 0

    # Pre-sample the planned reference path once (it's fixed for the episode).
    ref_path = None
    if renderer is not None:
        n_samples = max(20, int(np.ceil(T_traj / 0.05)))  # ~20 Hz sampling
        ts = np.linspace(0.0, T_traj, n_samples)
        ref_path = np.array([env._eval_trajectory(t)[0] for t in ts])

    # Capture a frame of the initial state with overlays.
    if renderer is not None:
        renderer.update_scene(env.data, camera=camera)
        _draw_trajectory_overlays(
            renderer, ref_path, [env.data.qpos[0:3].copy()],
            goal_pos, env.base_target_pos,
        )
        frames.append(renderer.render().copy())

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
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


def save_video(frames, path, fps):
    """Write a list of HxWx3 uint8 arrays to an mp4 (or gif) at the given fps."""
    import imageio.v2 as imageio
    ext = os.path.splitext(path)[1].lower()
    if ext == ".gif":
        imageio.mimsave(path, frames, duration=1.0 / fps)
    else:
        # Default to mp4 via ffmpeg.
        imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
    print(f"Saved video to {path} ({len(frames)} frames at {fps} fps)")


def episode_metrics(log, meta, goal_tol=0.15):
    """Reduce a per-step log to scalar metrics for one episode."""
    T_traj = meta["T_traj"]
    # Restrict tracking metrics to the actual trajectory window; the post-T_traj
    # phase is just hover regulation against a held setpoint.
    in_traj = log["t"] <= T_traj
    if in_traj.sum() == 0:
        in_traj = np.ones_like(log["t"], dtype=bool)

    pos_rmse = float(np.sqrt(np.mean(log["pos_err"][in_traj] ** 2)))
    vel_rmse = float(np.sqrt(np.mean(log["vel_err"][in_traj] ** 2)))
    ee_rmse = float(np.sqrt(np.mean(log["ee_err"][in_traj] ** 2)))

    final_goal_dist = float(log["goal_dist"][-1])
    reached_goal = (not meta["crashed"]) and (final_goal_dist < goal_tol)

    return {
        "pos_rmse": pos_rmse,
        "vel_rmse": vel_rmse,
        "ee_rmse": ee_rmse,
        "final_goal_dist": final_goal_dist,
        "reached_goal": reached_goal,
        "crashed": meta["crashed"],
        "T_traj": T_traj,
        "ep_len_s": float(log["t"][-1]),
    }


def plot_episode(log, meta, save_path=None):
    """3D path + per-axis tracking + error curves for a single episode."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        print("matplotlib not installed; skipping plot. `pip install matplotlib`")
        return

    fig = plt.figure(figsize=(14, 8))

    # 3D path
    ax3d = fig.add_subplot(2, 3, (1, 4), projection="3d")
    ref = log["ref_pos"]
    act = log["actual_pos"]
    ax3d.plot(ref[:, 0], ref[:, 1], ref[:, 2], "b--", lw=2, label="reference")
    ax3d.plot(act[:, 0], act[:, 1], act[:, 2], "r-", lw=1.5, label="actual")
    ax3d.scatter(*act[0], c="g", s=60, label="start")
    ax3d.scatter(*meta["goal_pos"], c="k", marker="*", s=120, label="goal")
    ax3d.set_xlabel("x [m]"); ax3d.set_ylabel("y [m]"); ax3d.set_zlabel("z [m]")
    ax3d.set_title("Reference vs Actual Path")
    ax3d.legend()

    # Per-axis position tracking
    ax_xyz = fig.add_subplot(2, 3, 2)
    for i, name in enumerate("xyz"):
        ax_xyz.plot(log["t"], ref[:, i], "--", label=f"{name}_ref")
        ax_xyz.plot(log["t"], act[:, i], "-", alpha=0.8, label=f"{name}_act")
    ax_xyz.axvline(meta["T_traj"], color="gray", ls=":", label="T_traj")
    ax_xyz.set_xlabel("t [s]"); ax_xyz.set_ylabel("position [m]")
    ax_xyz.set_title("Per-axis position tracking")
    ax_xyz.legend(fontsize=7, ncol=2)
    ax_xyz.grid(alpha=0.3)

    # Per-axis velocity tracking
    ax_v = fig.add_subplot(2, 3, 3)
    for i, name in enumerate("xyz"):
        ax_v.plot(log["t"], log["ref_vel"][:, i], "--", label=f"v{name}_ref")
        ax_v.plot(log["t"], log["actual_vel"][:, i], "-", alpha=0.8, label=f"v{name}_act")
    ax_v.axvline(meta["T_traj"], color="gray", ls=":")
    ax_v.set_xlabel("t [s]"); ax_v.set_ylabel("velocity [m/s]")
    ax_v.set_title("Per-axis velocity tracking")
    ax_v.legend(fontsize=7, ncol=2)
    ax_v.grid(alpha=0.3)

    # Tracking error norms over time
    ax_e = fig.add_subplot(2, 3, 5)
    ax_e.plot(log["t"], log["pos_err"], label="‖p − p_ref‖")
    ax_e.plot(log["t"], log["vel_err"], label="‖v − v_ref‖")
    ax_e.plot(log["t"], log["ee_err"], label="‖ee − ee_ref‖")
    ax_e.axvline(meta["T_traj"], color="gray", ls=":")
    ax_e.set_xlabel("t [s]"); ax_e.set_ylabel("error")
    ax_e.set_title("Tracking errors")
    ax_e.legend(); ax_e.grid(alpha=0.3)

    # Distance to goal
    ax_g = fig.add_subplot(2, 3, 6)
    ax_g.plot(log["t"], log["goal_dist"], "k-")
    ax_g.axvline(meta["T_traj"], color="gray", ls=":")
    ax_g.set_xlabel("t [s]"); ax_g.set_ylabel("‖p − goal‖ [m]")
    ax_g.set_title("Distance to goal")
    ax_g.grid(alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="logs/continue_7M/best_model.zip")
    parser.add_argument("--xml", default="models/skydio_arm.xml")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--goal-tol", type=float, default=0.15,
                        help="‖p − goal‖ tolerance to count as reached")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", action="store_true",
                        help="Plot a single episode after evaluation")
    parser.add_argument("--plot-episode", type=int, default=0,
                        help="Index of the episode to plot (0-based)")
    parser.add_argument("--save-plot", default=None,
                        help="If set, save the plot to this path instead of showing")
    parser.add_argument("--save-video", default=None,
                        help="If set, save an mp4 (or gif) of one episode to this path")
    parser.add_argument("--video-episode", type=int, default=None,
                        help="Index of the episode to record (defaults to --plot-episode)")
    parser.add_argument("--video-camera", default="track",
                        help="MuJoCo camera name to use for rendering")
    parser.add_argument("--video-width", type=int, default=960)
    parser.add_argument("--video-height", type=int, default=540)
    parser.add_argument("--video-stride", type=int, default=2,
                        help="Render every Nth sim step (2 → 50 fps at dt=0.01)")
    args = parser.parse_args()

    print("Evaluating Phase 2: PPO baseline on min-snap reference tracking")

    if not os.path.exists(args.model):
        print(f"Model not found at {args.model}. Train it first.")
        return

    model = PPO.load(args.model)
    env = AerialManipulatorEnv(model_path=args.xml, render_mode=None)

    rng_seed = args.seed
    metrics = []
    plot_log, plot_meta = None, None

    # Set up an off-screen renderer only if we're recording.
    video_episode = args.video_episode if args.video_episode is not None else args.plot_episode
    renderer = None
    if args.save_video:
        # max_geom budget: scene geoms + planned reference (~100 capsules) +
        # full realized path (up to ~1000 capsules) + a few markers.
        renderer = mujoco.Renderer(env.model, height=args.video_height,
                                   width=args.video_width, max_geom=20000)

    print(f"Running {args.episodes} deterministic episodes...")
    for ep in range(args.episodes):
        record_this = (renderer is not None) and (ep == video_episode)
        log, meta = rollout_episode(
            env, model,
            seed=rng_seed + ep,
            renderer=renderer if record_this else None,
            camera=args.video_camera,
            frame_stride=args.video_stride,
        )
        m = episode_metrics(log, meta, goal_tol=args.goal_tol)
        metrics.append(m)

        if ep == args.plot_episode:
            plot_log, plot_meta = log, meta

        if record_this and meta.get("frames"):
            dt = env.model.opt.timestep
            fps = max(1, int(round(1.0 / (dt * args.video_stride))))
            save_video(meta["frames"], args.save_video, fps=fps)

        status = "CRASH" if m["crashed"] else ("REACH" if m["reached_goal"] else "miss ")
        print(
            f"  ep {ep:02d} | T={m['T_traj']:.2f}s len={m['ep_len_s']:.2f}s "
            f"| pos_RMSE={m['pos_rmse']:.3f} vel_RMSE={m['vel_rmse']:.3f} "
            f"ee_RMSE={m['ee_rmse']:.3f} | final_d={m['final_goal_dist']:.3f} "
            f"| {status}"
        )

    n = len(metrics)
    crash_rate = 100.0 * sum(m["crashed"] for m in metrics) / n
    reach_rate = 100.0 * sum(m["reached_goal"] for m in metrics) / n
    mean_pos = np.mean([m["pos_rmse"] for m in metrics])
    mean_vel = np.mean([m["vel_rmse"] for m in metrics])
    mean_ee = np.mean([m["ee_rmse"] for m in metrics])
    mean_final = np.mean([m["final_goal_dist"] for m in metrics])

    print("\n--- Baseline PPO (trajectory-tracking) Metrics ---")
    print(f"Episodes                         : {n}")
    print(f"Crash rate                       : {crash_rate:.2f}%")
    print(f"Goal reach rate (tol={args.goal_tol:.2f} m): {reach_rate:.2f}%")
    print(f"Position tracking RMSE  [m]      : {mean_pos:.4f}")
    print(f"Velocity tracking RMSE  [m/s]    : {mean_vel:.4f}")
    print(f"End-effector tracking RMSE [m]   : {mean_ee:.4f}")
    print(f"Final settling error    [m]      : {mean_final:.4f}")

    if args.plot and plot_log is not None:
        plot_episode(plot_log, plot_meta, save_path=args.save_plot)


if __name__ == "__main__":
    main()
