"""
Disturbance-rejection evaluation for the PPO policy, with and without the
MPC-BLF safety filter.

Each episode has three clearly-labelled phases (labels drawn on the
rendered video via Pillow):

  Phase 1 -- Hover          t in [0, 2) s          reference held, no
                                                   disturbance.
  Phase 2 -- Disturbance    t in [2, 4) s          wind or arm fold.
  Phase 3 -- Stabilize      t in [4, 9) s          disturbance removed,
                                                   controller must
                                                   drive the state back
                                                   to the hover setpoint.

The disturb-phase metric is how much the base deviates from the held
hover setpoint while the disturbance is active; the stabilize-phase
metric is how quickly the controller re-settles once it's gone.

Disturbance modes:

  - wind-low    gusty world-frame force on the drone base body, peak
                magnitude ~0.8 N (about 1/7 of the total AM weight).
                Three raised-cosine gusts over the disturbance window,
                pointing along +world-y so they blow across the track
                camera's field of view (left-right in frame).
  - wind-high   same direction and gust pattern but ~2.5 N peak
                (challenging for PPO alone).
  - arm-fold    the arm is driven toward a 90-degree bent-elbow pose
                (joint2 = -pi/2, joint3 = +pi/2 so the upper arm is
                horizontal-forward and the forearm hangs vertically
                down) via a PD controller on the joint-torque
                actuators. The rotors (PPO, optionally filtered) must
                re-stabilize the base against the shifted COM. No
                wind. 0.2 s PD ramp-in avoids a torque jerk at the
                mode switch. In the stabilize phase the PD keeps
                holding the pose.

The hover reference is held constant for the full 4-second episode, so
the min-snap trajectory is degenerate (start == goal, zero velocity).
The MPC-BLF filter sees an (N+1, 3) reference that is constantly the
hover setpoint, and must reject whatever the disturbance injects.

Video overlay (when --save-video is set):

  - Gold sphere    : the hover setpoint
  - Red trail      : realized base path
  - Translucent red halo overhead : "disturbance active" indicator
  - Caption band (top) : phase title + current time + per-phase sub-line
                         (wind magnitude for wind modes, pose/description
                         for arm-fold).

Usage:
  python eval_ppo_disturbance.py --mode wind-low
  python eval_ppo_disturbance.py --mode wind-high --mpc-blf \
         --save-video videos/disturbance/wind_high_mpc.mp4
  python eval_ppo_disturbance.py --mode arm-fold --mpc-blf
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

from eval_ppo import (
    _scene_add_capsule,
    _scene_add_sphere,
    save_video,
)


_IDENT_MAT = np.eye(3, dtype=np.float64).flatten()


# ------------------------------------------------------------------ #
# Caption overlay                                                    #
# ------------------------------------------------------------------ #
_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)
_FONT_CANDIDATES_SMALL = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)


def _load_font(candidates, size):
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        from PIL import ImageFont as _IF
        return _IF.load_default()
    except Exception:
        return None


def _annotate_frame(frame, phase_title, phase_sub, t, color=(255, 255, 255)):
    """Overlay a 2-line caption onto an RGB frame using Pillow."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return frame
    img = Image.fromarray(frame).convert("RGBA")
    W, H = img.size
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    pad = 14
    box_h = 74
    draw.rectangle([(0, 0), (W, box_h)], fill=(0, 0, 0, 170))

    font_big = _load_font(_FONT_CANDIDATES, 28)
    font_sm  = _load_font(_FONT_CANDIDATES_SMALL, 20)
    title = f"{phase_title}    t = {t:5.2f} s"
    if font_big is not None:
        draw.text((pad, 8), title, fill=color + (255,), font=font_big)
    else:
        draw.text((pad, 8), title, fill=color + (255,))
    if phase_sub:
        if font_sm is not None:
            draw.text((pad, 42), phase_sub, fill=(230, 220, 170, 255),
                      font=font_sm)
        else:
            draw.text((pad, 42), phase_sub, fill=(230, 220, 170, 255))

    composed = Image.alpha_composite(img, overlay).convert("RGB")
    return np.asarray(composed)


def _find_drone_base_body_id(model):
    """Return the body id that carries the drone's free joint."""
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return int(model.jnt_bodyid[j])
    raise RuntimeError("No free joint found in model; cannot identify drone base.")


# ------------------------------------------------------------------ #
# Episode setup                                                      #
# ------------------------------------------------------------------ #
def setup_hover_episode(env, seed, hover_pos, total_time):
    """Reset the env, teleport to hover_pos, and build a constant reference."""
    obs, info = env.reset(seed=seed)
    env.data.qpos[0:3] = hover_pos
    env.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])  # identity quat
    env.data.qvel[0:6] = 0.0
    mujoco.mj_forward(env.model, env.data)

    # Force the env trajectory to be degenerate (start == goal, v=0):
    # min-snap with identical boundary conditions gives p(t) = p0 for all t.
    env._generate_trajectory(hover_pos, np.zeros(3), hover_pos)
    env.goal_pos = hover_pos.copy()

    dt = env.model.opt.timestep
    env.max_steps = int(np.ceil(total_time / dt))

    pos0, vel0 = env._eval_trajectory(0.0)
    env.base_target_pos = pos0
    env.base_target_vel = vel0
    env.ee_target_pos = env.base_target_pos + env.EE_OFFSET

    return env._get_obs(), info


# ------------------------------------------------------------------ #
# Disturbance schedules                                              #
# ------------------------------------------------------------------ #
# The `track` camera in skydio_arm.xml has xyaxes = "0 -1 0  1 0 2", so:
#   image-right  = -world_y     -> lateral axis in frame is world_y
#   image-up     ~  world_x + 2*world_z (mostly +z)
#   into-screen  ~  +world_x
# Any purely +/- world_y force is therefore perfectly horizontal in the
# rendered frame; using pure +y makes the wind arrow traverse the frame.
_WIND_DIRECTION = np.array([0.0, 1.0, 0.0], dtype=np.float64)

# Three raised-cosine gusts over a 2 s disturbance window. Each tuple is
# (center_s, width_s, peak_amplitude_in_[0,1]). The envelope returns to
# zero between gusts so the viewer sees discrete puffs, not a smear.
_GUSTS = (
    (0.35, 0.45, 0.65),
    (1.00, 0.55, 1.00),
    (1.65, 0.40, 0.80),
)


def _gust_envelope(t_rel: float) -> float:
    """Multi-bump gust envelope in [0, 1]; zero outside the gusts."""
    env = 0.0
    for c, w, a in _GUSTS:
        d = abs(t_rel - c)
        if d < 0.5 * w:
            env += a * 0.5 * (1.0 + np.cos(2.0 * np.pi * d / w))
    return float(env)


# Target arm pose for a clean 90-degree bent elbow [rad]
# (joint1, joint2, joint3). With joint1=0, joint2=-pi/2 rotates the
# upper arm (link2) from hanging -z to horizontal +x; joint3=+pi/2
# then rotates link3 by +pi/2 in link2's frame, so link3 ends up
# pointing -z in world (back to vertical-down). Net: upper arm
# horizontal-forward, forearm vertical-down, elbow at exactly 90 deg.
# Targets sit 0.07 rad inside the +/- pi/2 joint limits so the PD has
# margin to settle without pushing the limit.
_ARM_TARGET = np.array([0.0, -1.50, 1.50], dtype=np.float64)
_ARM_KP = np.array([4.0, 6.0, 5.0], dtype=np.float64)
_ARM_KD = np.array([0.4, 0.6, 0.5], dtype=np.float64)


def _arm_pd(env, t_rel: float) -> np.ndarray:
    """PD controller driving the arm toward the bent-elbow target pose.

    Output is in the normalized [-1, 1] action space (the env maps it to
    the [-1, 1] ctrlrange of each torque_jointN actuator, which then
    scales by gear = 0.2 N*m per unit).

    A 0.2 s ramp on the command magnitude prevents a torque jerk at the
    instant the disturbance phase begins.
    """
    qpos_arm = np.asarray(env.data.qpos[7:10], dtype=np.float64)
    qvel_arm = np.asarray(env.data.qvel[6:9], dtype=np.float64)
    ramp = float(np.clip(t_rel / 0.2, 0.0, 1.0))
    e = _ARM_TARGET - qpos_arm
    de = -qvel_arm
    u = ramp * (_ARM_KP * e + _ARM_KD * de)
    return np.clip(u, -1.0, 1.0)


def disturbance_at(env, mode, t_rel, force_N):
    """
    Compute the active disturbance at time t_rel into the disturbance phase.

    Returns (kind, value):
      kind="wind"  : value is a 3-vector world-frame force to apply to the
                     drone base via data.xfrc_applied[:, 0:3].
      kind="arm"   : value is a 3-vector override for action[4:7] (arm cmds,
                     normalized [-1, 1]), produced by the bent-elbow PD.
      kind="none"  : value is None.
    """
    if mode in ("wind-low", "wind-high"):
        gust = _gust_envelope(t_rel)
        return "wind", gust * force_N * _WIND_DIRECTION
    if mode == "arm-fold":
        return "arm", _arm_pd(env, t_rel)
    return "none", None


# ------------------------------------------------------------------ #
# Rollout                                                            #
# ------------------------------------------------------------------ #
def _phase_of(t, hover_duration, disturb_duration):
    """Return ('hover'|'disturb'|'stabilize', title_str)."""
    if t < hover_duration:
        return "hover", "Phase 1  -  Hover"
    if t < hover_duration + disturb_duration:
        return "disturb", "Phase 2  -  Disturbance"
    return "stabilize", "Phase 3  -  Stabilize"


def rollout(
    env, model,
    *,
    mode,
    force_N,
    hover_pos,
    hover_duration=2.0,
    disturb_duration=2.0,
    stabilize_duration=5.0,
    mpc_filter=None,
    mpc_stride=3,
    seed=0,
    renderer=None,
    camera="track",
    frame_stride=2,
):
    dt = env.model.opt.timestep
    total_time = hover_duration + disturb_duration + stabilize_duration
    N_horizon = mpc_filter.N if mpc_filter is not None else 0

    obs, _ = setup_hover_episode(env, seed, hover_pos, total_time)
    if mpc_filter is not None:
        mpc_filter.reset()

    drone_body_id = _find_drone_base_body_id(env.model)

    log = {
        "t": [], "actual_pos": [], "pos_err": [],
        "base_vel": [], "vel_norm": [],
        "phase": [], "wind_force": [], "arm_cmd": [],
        "mpc_fallback": [], "mpc_solve_ms": [],
    }
    frames = []
    crashed = False
    step_idx = 0
    last_safe_action = None
    done = False

    # Disturbance label shown in Phase 2 (depends on mode).
    if mode in ("wind-low", "wind-high"):
        disturb_sub = f"Wind gusts, peak {force_N:.2f} N along +y"
    elif mode == "arm-fold":
        disturb_sub = "Arm folding to 90-degree bent-elbow pose"
    else:
        disturb_sub = ""
    stabilize_sub = {
        "wind-low":  "Wind removed; controller settling",
        "wind-high": "Wind removed; controller settling",
        "arm-fold":  "Arm held; controller rejecting shifted COM",
    }.get(mode, "")

    def _build_scene(wind_vec, disturbing):
        renderer.update_scene(env.data, camera=camera)
        scene = renderer.scene
        _scene_add_sphere(scene, hover_pos,
                          rgba=[1.00, 0.84, 0.00, 1.00], radius=0.06)
        if len(log["actual_pos"]) >= 2:
            ap = np.asarray(log["actual_pos"])
            for i in range(len(ap) - 1):
                _scene_add_capsule(scene, ap[i], ap[i + 1],
                                   rgba=[1.00, 0.20, 0.15, 0.95], width=0.010)
        if disturbing:
            _scene_add_sphere(
                scene,
                env.data.qpos[0:3] + np.array([0.0, 0.0, 0.42]),
                rgba=[1.00, 0.20, 0.10, 0.30], radius=0.07,
            )

    def _record_frame(t_now, wind_vec, phase_kind, phase_title):
        if renderer is None:
            return
        _build_scene(wind_vec, phase_kind == "disturb")
        raw = renderer.render().copy()
        if phase_kind == "hover":
            sub = "Reference held; no disturbance"
        elif phase_kind == "disturb":
            if wind_vec is not None and np.linalg.norm(wind_vec) > 1e-3:
                sub = (f"Wind: {np.linalg.norm(wind_vec):.2f} N  "
                       f"(+y, horizontal in frame)")
            else:
                sub = disturb_sub
        else:  # stabilize
            sub = stabilize_sub
        frames.append(_annotate_frame(raw, phase_title, sub, t_now))

    # Initial frame (t = 0, hover phase).
    _record_frame(0.0, None, "hover", "Phase 1  -  Hover")

    while not done:
        t = env.current_step * dt
        t_rel_disturb = t - hover_duration
        phase_kind, phase_title = _phase_of(t, hover_duration, disturb_duration)
        disturbing = (phase_kind == "disturb")

        # Clear any external force each step before deciding current-step force.
        env.data.xfrc_applied[drone_body_id, 0:6] = 0.0

        # Disturbance value for this step:
        #   - wind modes : force is non-zero only during Phase 2.
        #   - arm-fold   : the PD that holds the bent-elbow pose runs in both
        #                  Phase 2 and Phase 3 (ramp-in starts at Phase 2 so
        #                  the arm finishes folding in ~0.1 s and is then held
        #                  through Phase 3 as the "shifted COM" disturbance).
        kind, value = ("none", None)
        if disturbing:
            kind, value = disturbance_at(env, mode, t_rel_disturb, force_N)
        elif phase_kind == "stabilize" and mode == "arm-fold":
            # Hold the bent-elbow pose through stabilize (PD fully ramped).
            kind, value = "arm", _arm_pd(env, t_rel_disturb)

        ppo_action, _ = model.predict(obs, deterministic=True)
        safe_action = np.asarray(ppo_action, dtype=np.float32).copy()

        mpc_solve_ms = 0.0
        mpc_fallback = False
        if mpc_filter is not None:
            ref_traj = np.tile(hover_pos, (N_horizon + 1, 1)).astype(np.float64)
            x_meas = mj_state_to_x12(env.data)
            do_solve = (step_idx % mpc_stride == 0) or (last_safe_action is None)
            if do_solve:
                filtered, mpc_info = mpc_filter.filter(x_meas, ppo_action, ref_traj)
                safe_action = filtered
                last_safe_action = filtered.copy()
                mpc_solve_ms = float(mpc_info.get("solve_time_ms", 0.0))
                mpc_fallback = bool(mpc_info.get("fallback", False))

        wind_vec_world = None
        if kind == "wind":
            env.data.xfrc_applied[drone_body_id, 0:3] = value
            wind_vec_world = value.copy()
        elif kind == "arm":
            safe_action[4:7] = np.asarray(value, dtype=np.float32)

        obs, _, terminated, truncated, info = env.step(safe_action)
        step_idx += 1

        base_pos = env.data.qpos[0:3].copy()
        base_vel = env.data.qvel[0:3].copy()
        t_next = t + dt
        log["t"].append(t_next)
        log["actual_pos"].append(base_pos)
        log["pos_err"].append(float(np.linalg.norm(base_pos - hover_pos)))
        log["base_vel"].append(base_vel)
        log["vel_norm"].append(float(np.linalg.norm(base_vel)))
        log["phase"].append(phase_kind)
        log["wind_force"].append(
            wind_vec_world.copy() if wind_vec_world is not None else np.zeros(3)
        )
        log["arm_cmd"].append(
            np.asarray(value, dtype=np.float64).copy()
            if kind == "arm" else np.zeros(3)
        )
        log["mpc_solve_ms"].append(mpc_solve_ms)
        log["mpc_fallback"].append(mpc_fallback)

        if renderer is not None and (step_idx % frame_stride == 0):
            phase_kind_post, phase_title_post = _phase_of(
                t_next, hover_duration, disturb_duration
            )
            _record_frame(t_next, wind_vec_world, phase_kind_post,
                          phase_title_post)

        if terminated:
            crashed = True
        done = terminated or truncated

    env.data.xfrc_applied[drone_body_id, 0:6] = 0.0

    out = {
        "t": np.asarray(log["t"]),
        "actual_pos": np.asarray(log["actual_pos"]),
        "pos_err": np.asarray(log["pos_err"]),
        "base_vel": np.asarray(log["base_vel"]),
        "vel_norm": np.asarray(log["vel_norm"]),
        "phase": np.asarray(log["phase"]),
        "wind_force": np.asarray(log["wind_force"]),
        "arm_cmd": np.asarray(log["arm_cmd"]),
        "mpc_solve_ms": np.asarray(log["mpc_solve_ms"]),
        "mpc_fallback": np.asarray(log["mpc_fallback"], dtype=bool),
        "crashed": crashed,
        "frames": frames,
    }
    return out


# ------------------------------------------------------------------ #
# Main                                                               #
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="logs/continue_7M/best_model.zip")
    parser.add_argument("--xml", default="models/skydio_arm.xml")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--mode", default="wind-low",
                        choices=["wind-low", "wind-high", "arm-fold"])
    parser.add_argument("--force-low",  type=float, default=0.8,
                        help="Wind force [N] used in mode=wind-low.")
    parser.add_argument("--force-high", type=float, default=2.5,
                        help="Wind force [N] used in mode=wind-high.")

    parser.add_argument("--hover-x", type=float, default=0.0)
    parser.add_argument("--hover-y", type=float, default=0.0)
    parser.add_argument("--hover-z", type=float, default=1.2)
    parser.add_argument("--hover-time", type=float, default=2.0)
    parser.add_argument("--disturb-time", type=float, default=2.0)
    parser.add_argument("--stabilize-time", type=float, default=5.0)

    parser.add_argument("--mpc-blf", action="store_true",
                        help="Run with the MPC-BLF filter in front of PPO.")
    parser.add_argument("--no-blf", action="store_true",
                        help="When --mpc-blf is set, disable the barrier-"
                             "Lyapunov safety constraint (MPC becomes a pure "
                             "tracking optimiser). Used for the BLF ablation.")
    parser.add_argument("--mpc-horizon", type=int, default=7)
    parser.add_argument("--tube", type=float, default=0.15)
    parser.add_argument("--mpc-stride", type=int, default=3)
    parser.add_argument("--velocity-penalty", type=float, default=10.0,
                        help="Weight on the stage-wise planned velocity "
                             "(D-term): adds (w_v/N) * sum_{k=1..N} "
                             "||v_k||^2. 0.0 disables it.")
    parser.add_argument("--smooth-penalty", type=float, default=1e-2,
                        help="Weight on consecutive-stage rotor-command "
                             "differences: adds lambda_s * sum "
                             "||U[k] - U[k-1]||^2 over k=1..N-1.")
    parser.add_argument("--slack-penalty", type=float, default=3e3,
                        help="Weight rho on the scalar BLF slack: adds "
                             "rho * sigma^2 so the barrier contraction "
                             "V_{k+1} <= beta_k V_k + sigma is soft. "
                             "Lower -> filter tolerates more slack.")
    parser.add_argument("--barrier-velocity-weight", type=float, default=0.0,
                        help="alpha [s^2]: puts velocity INSIDE the "
                             "barrier via z = ||e_p||^2 + alpha*||e_v||^2 "
                             "and V(z) = z/(delta^2 - z). 0.0 = legacy "
                             "position-only tube.")

    parser.add_argument("--save-video", default=None)
    parser.add_argument("--video-camera", default="track")
    parser.add_argument("--video-width", type=int, default=960)
    parser.add_argument("--video-height", type=int, default=540)
    parser.add_argument("--video-stride", type=int, default=2)

    args = parser.parse_args()

    if args.mode == "wind-high":
        force_N = args.force_high
    elif args.mode == "wind-low":
        force_N = args.force_low
    else:
        force_N = 0.0  # arm-fold: no external force

    if not os.path.exists(args.model):
        print(f"Model not found at {args.model}.")
        return

    if args.mpc_blf and args.no_blf:
        label_filter = "PPO + MPC (BLF off)"
    elif args.mpc_blf:
        label_filter = "PPO + MPC-BLF"
    else:
        label_filter = "PPO only"
    label_force = (f"{force_N:.2f} N peak along {_WIND_DIRECTION.tolist()}, "
                   f"3 gusts over {args.disturb_time:.1f}s"
                   if args.mode.startswith("wind")
                   else f"bent-elbow PD hold (target={_ARM_TARGET.tolist()} rad)")
    print(f"Disturbance eval: mode={args.mode} ({label_force}) "
          f"controller={label_filter} seed={args.seed}")

    model = PPO.load(args.model)
    env = AerialManipulatorEnv(model_path=args.xml, render_mode=None)

    mpc_filter = None
    if args.mpc_blf:
        mpc_filter = MPCBLFSafetyFilter(
            env.model,
            horizon=args.mpc_horizon,
            tube=args.tube,
            beta_start=0.95, beta_end=0.15,
            slack_penalty=args.slack_penalty, smooth_penalty=args.smooth_penalty,
            velocity_penalty=args.velocity_penalty,
            barrier_velocity_weight=args.barrier_velocity_weight,
            enable_blf=not args.no_blf,
            solver="fatrop", verbose=False,
        )

    renderer = None
    if args.save_video:
        renderer = mujoco.Renderer(env.model, height=args.video_height,
                                   width=args.video_width, max_geom=20000)

    # Warm up the PyTorch/CUDA path so timings are meaningful.
    warm_obs, _ = env.reset(seed=args.seed)
    for _ in range(3):
        model.predict(warm_obs, deterministic=True)

    hover_pos = np.array([args.hover_x, args.hover_y, args.hover_z],
                         dtype=np.float64)

    log = rollout(
        env, model,
        mode=args.mode,
        force_N=force_N,
        hover_pos=hover_pos,
        hover_duration=args.hover_time,
        disturb_duration=args.disturb_time,
        stabilize_duration=args.stabilize_time,
        mpc_filter=mpc_filter,
        mpc_stride=args.mpc_stride,
        seed=args.seed,
        renderer=renderer,
        camera=args.video_camera,
        frame_stride=args.video_stride,
    )

    if args.save_video and log["frames"]:
        dt = env.model.opt.timestep
        fps = max(1, int(round(1.0 / (dt * args.video_stride))))
        save_video(log["frames"], args.save_video, fps=fps)

    t = log["t"]
    e = log["pos_err"]
    v = log["vel_norm"]
    phase = log["phase"]
    hover_mask = (phase == "hover") & (t > 0.05)
    disturb_mask = (phase == "disturb")
    stabilize_mask = (phase == "stabilize")

    def _stats(signal, mask):
        if mask.sum() == 0:
            return 0.0, 0.0, 0.0
        return (float(np.mean(signal[mask])),
                float(np.max(signal[mask])),
                float(np.sqrt(np.mean(signal[mask] ** 2))))

    h_mean, h_max, h_rmse = _stats(e, hover_mask)
    d_mean, d_max, d_rmse = _stats(e, disturb_mask)
    s_mean, s_max, s_rmse = _stats(e, stabilize_mask)

    vh_mean, vh_max, vh_rmse = _stats(v, hover_mask)
    vd_mean, vd_max, vd_rmse = _stats(v, disturb_mask)
    vs_mean, vs_max, vs_rmse = _stats(v, stabilize_mask)

    # Settling-time proxy: first time after the disturbance phase ends at which
    # the error falls below 5 cm and stays below for >= 0.5 s. If it never
    # settles, reports the full stabilize duration.
    settling_time = float("nan")
    settle_tol = 0.05
    settle_hold = 0.5
    if stabilize_mask.any():
        s_idx = np.where(stabilize_mask)[0]
        t_stabilize = t[s_idx]
        e_stabilize = e[s_idx]
        t_disturb_end = args.hover_time + args.disturb_time
        dt = env.model.opt.timestep
        hold_steps = int(round(settle_hold / dt))
        settled_flag = e_stabilize < settle_tol
        for j in range(len(e_stabilize)):
            if settled_flag[j] and settled_flag[j:j + hold_steps].all():
                settling_time = float(t_stabilize[j] - t_disturb_end)
                break

    print("\n--- Disturbance-rejection metrics ---")
    print(f"Mode       : {args.mode}")
    print(f"Filter     : {label_filter}")
    print(f"Crashed    : {log['crashed']}")
    print(f"Phase 1 hover   [t in 0.0-{args.hover_time:.1f}s]"
          f"        pos  mean={h_mean:.4f}  max={h_max:.4f}  RMSE={h_rmse:.4f}  [m]")
    print(f"                                         vel  "
          f"mean={vh_mean:.4f}  max={vh_max:.4f}  RMSE={vh_rmse:.4f}  [m/s]")
    print(f"Phase 2 disturb [t in {args.hover_time:.1f}-{args.hover_time + args.disturb_time:.1f}s]"
          f"        pos  mean={d_mean:.4f}  max={d_max:.4f}  RMSE={d_rmse:.4f}  [m]")
    print(f"                                         vel  "
          f"mean={vd_mean:.4f}  max={vd_max:.4f}  RMSE={vd_rmse:.4f}  [m/s]")
    print(f"Phase 3 stabil. [t in {args.hover_time + args.disturb_time:.1f}-"
          f"{args.hover_time + args.disturb_time + args.stabilize_time:.1f}s]  "
          f"pos  mean={s_mean:.4f}  max={s_max:.4f}  RMSE={s_rmse:.4f}  [m]")
    print(f"                                         vel  "
          f"mean={vs_mean:.4f}  max={vs_max:.4f}  RMSE={vs_rmse:.4f}  [m/s]")
    if not np.isnan(settling_time):
        print(f"Settling (<{100*settle_tol:.0f} cm held {settle_hold:.1f}s): "
              f"{settling_time:.2f} s after disturbance ends")
    else:
        print(f"Settling (<{100*settle_tol:.0f} cm held {settle_hold:.1f}s): "
              f"not settled within {args.stabilize_time:.1f} s")
    if mpc_filter is not None:
        solve = log["mpc_solve_ms"]
        nz = solve[solve > 0]
        print(f"MPC solve  : median={np.median(nz):.1f} ms "
              f"peak={np.max(solve):.1f} ms  "
              f"fallback={100.0 * np.mean(log['mpc_fallback']):.1f}%")


if __name__ == "__main__":
    main()
