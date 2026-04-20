import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco


class AerialManipulatorEnv(gym.Env):
    """
    Gymnasium Environment for a Skydio X2 quadrotor with a 3-DoF arm payload.

    Tracking is performed against a smooth minimum-snap (7th order) reference
    trajectory generated from the current state to a randomly sampled goal at
    every reset. The policy observes the *next-step* desired pose/velocity, so
    the learned PPO controller behaves as a reference tracker that can be
    seamlessly composed with a downstream MPC-CBF safety filter or replaced
    with a flow-matching trajectory generator producing the same reference.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    # Fixed offset from base reference to end-effector reference (body frame
    # along x and below). Kept consistent with the previous static-target
    # convention so reward scales remain comparable.
    EE_OFFSET = np.array([0.2, 0.0, -0.2])

    def __init__(
        self,
        model_path="models/skydio_arm.xml",
        render_mode=None,
        wind_magnitude=0.0,
        moving_target=False,
        traj_avg_speed=0.5,   # m/s, used to size trajectory duration
        traj_min_duration=2.0,
        traj_max_duration=8.0,
        hover_time=2.0,       # extra seconds after T_traj for terminal regulation
        max_episode_steps=1000,  # hard upper bound (safety net)
    ):
        super().__init__()

        self.render_mode = render_mode
        self.wind_magnitude = wind_magnitude
        self.moving_target = moving_target
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # 4 rotors + 3 arm joint torques
        self.num_actions = self.model.nu
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32
        )

        self.nq = self.model.nq
        self.nv = self.model.nv

        # Observation layout:
        #   qpos                        (nq)
        #   qvel                        (nv)
        #   rel_base_pos  = des - cur   (3)
        #   rel_base_vel  = des - cur   (3)
        #   rel_ee_pos    = des - cur   (3)
        # = nq + nv + 9
        self.obs_dim = self.nq + self.nv + 9
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # Cache whether the EE site exists (avoids per-step name scans).
        self._site_names = [self.model.site(i).name for i in range(self.model.nsite)]
        self._has_ee_site = "end_effector" in self._site_names
        if self._has_ee_site:
            self._ee_site_id = self.model.site("end_effector").id

        # Cache the "hover" keyframe id (drone in mid-air, hover thrust pre-loaded
        # into ctrl). Without this we'd reset to the body's default z = 0.1 which
        # is right on the OOB floor and crashes ~half the time after init noise.
        self._hover_key_id = -1
        for i in range(self.model.nkey):
            if self.model.key(i).name == "hover":
                self._hover_key_id = i
                break

        self.viewer = None

        # Trajectory bookkeeping (filled in reset).
        self.traj_avg_speed = traj_avg_speed
        self.traj_min_duration = traj_min_duration
        self.traj_max_duration = traj_max_duration
        self.hover_time = hover_time
        self.max_episode_steps = max_episode_steps
        self.traj_coeffs = np.zeros((3, 8))   # 7th order poly per axis
        self.T_traj = 1.0
        self.goal_pos = np.zeros(3)

        # Current reference (desired pos/vel at the *next* sim step).
        self.base_target_pos = np.array([0.0, 0.0, 1.0])
        self.base_target_vel = np.zeros(3)
        self.ee_target_pos = self.base_target_pos + self.EE_OFFSET

        # max_steps is set per-episode in reset() based on T_traj + hover_time,
        # capped by max_episode_steps.
        self.max_steps = self.max_episode_steps
        self.current_step = 0

    # ------------------------------------------------------------------ #
    # Minimum-snap trajectory utilities                                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _min_snap_1d(p0, v0, pf, vf, T):
        """
        7th order polynomial p(t) = sum_{k=0..7} a_k t^k that minimizes the
        integral of snap^2, with boundary conditions:
            p(0)=p0, p'(0)=v0, p''(0)=0, p'''(0)=0
            p(T)=pf, p'(T)=vf, p''(T)=0, p'''(T)=0
        Returns coefficients [a0..a7].
        """
        a0 = p0
        a1 = v0
        a2 = 0.0
        a3 = 0.0

        # Solve for [a4, a5, a6, a7] from terminal BCs.
        A = np.array([
            [T**4,    T**5,    T**6,     T**7],
            [4*T**3,  5*T**4,  6*T**5,   7*T**6],
            [12*T**2, 20*T**3, 30*T**4,  42*T**5],
            [24*T,    60*T**2, 120*T**3, 210*T**4],
        ])
        b = np.array([
            pf - a0 - a1 * T,
            vf - a1,
            0.0,
            0.0,
        ])
        a4, a5, a6, a7 = np.linalg.solve(A, b)
        return np.array([a0, a1, a2, a3, a4, a5, a6, a7])

    def _generate_trajectory(self, start_pos, start_vel, goal_pos):
        """Build a per-axis 7th order min-snap polynomial to the goal."""
        dist = np.linalg.norm(goal_pos - start_pos)
        T = float(np.clip(
            dist / max(self.traj_avg_speed, 1e-3),
            self.traj_min_duration,
            self.traj_max_duration,
        ))
        coeffs = np.zeros((3, 8))
        for i in range(3):
            coeffs[i] = self._min_snap_1d(
                p0=start_pos[i], v0=start_vel[i],
                pf=goal_pos[i],  vf=0.0,
                T=T,
            )
        self.traj_coeffs = coeffs
        self.T_traj = T
        self.goal_pos = goal_pos.copy()

    def _eval_trajectory(self, t):
        """Evaluate (pos, vel) of the trajectory at time t. Holds at goal after T."""
        t = float(np.clip(t, 0.0, self.T_traj))
        # Powers of t up to t^7
        tp = np.array([t**k for k in range(8)])
        # Derivative coefficients: d/dt (t^k) = k * t^(k-1)
        dtp = np.array([0.0] + [k * t**(k - 1) for k in range(1, 8)])
        pos = self.traj_coeffs @ tp
        vel = self.traj_coeffs @ dtp
        return pos, vel

    # ------------------------------------------------------------------ #
    # Gym API                                                             #
    # ------------------------------------------------------------------ #
    def _get_obs(self):
        base_pos = self.data.qpos[0:3]
        base_vel = self.data.qvel[0:3]

        rel_base_pos = self.base_target_pos - base_pos
        rel_base_vel = self.base_target_vel - base_vel

        if self._has_ee_site:
            ee_pos = self.data.site_xpos[self._ee_site_id].copy()
            rel_ee_pos = self.ee_target_pos - ee_pos
        else:
            rel_ee_pos = np.zeros(3)

        return np.concatenate([
            self.data.qpos.copy(),
            self.data.qvel.copy(),
            rel_base_pos,
            rel_base_vel,
            rel_ee_pos,
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset to the "hover" keyframe (z=0.3, hover thrust in ctrl) when
        # available. This avoids initializing right at the OOB floor (z<0.1)
        # which would terminate the episode on step 1 after init noise.
        if self._hover_key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self._hover_key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0

        # Small initial-state perturbation, leaving the quaternion identity.
        qpos_noise = self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nq)
        qvel_noise = self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nv)
        qpos_noise[3:7] = 0.0
        self.data.qpos[:] = self.data.qpos + qpos_noise
        quat = self.data.qpos[3:7]
        self.data.qpos[3:7] = quat / np.linalg.norm(quat)
        self.data.qvel[:] = self.data.qvel + qvel_noise

        if self.wind_magnitude > 0.0:
            self.model.opt.wind[:] = self.np_random.uniform(
                -self.wind_magnitude, self.wind_magnitude, size=3
            )

        mujoco.mj_forward(self.model, self.data)

        # Sample a random goal in the workspace and build a min-snap trajectory
        # from the current pose/velocity to it.
        start_pos = self.data.qpos[0:3].copy()
        start_vel = self.data.qvel[0:3].copy()
        goal = np.array([
            self.np_random.uniform(-1.5, 1.5),
            self.np_random.uniform(-1.5, 1.5),
            self.np_random.uniform(0.6, 1.8),
        ])
        self._generate_trajectory(start_pos, start_vel, goal)

        # Episode horizon = trajectory duration + hover budget, capped.
        dt = self.model.opt.timestep
        self.max_steps = int(min(
            self.max_episode_steps,
            np.ceil((self.T_traj + self.hover_time) / dt),
        ))

        # Initialize reference at t=0 (next step will advance it).
        pos0, vel0 = self._eval_trajectory(0.0)
        self.base_target_pos = pos0
        self.base_target_vel = vel0
        self.ee_target_pos = self.base_target_pos + self.EE_OFFSET

        return self._get_obs(), {
            "goal_pos": self.goal_pos.copy(),
            "T_traj": self.T_traj,
            "max_steps": self.max_steps,
        }

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Map normalized action [-1, 1] to actuator control range.
        for i in range(self.model.nu):
            act_range = self.model.actuator_ctrlrange[i]
            if self.model.actuator_ctrllimited[i]:
                self.data.ctrl[i] = act_range[0] + (action[i] + 1.0) * 0.5 * (
                    act_range[1] - act_range[0]
                )
            else:
                self.data.ctrl[i] = action[i]

        mujoco.mj_step(self.model, self.data)
        self.current_step += 1

        # Reference for the *next* control tick: the policy must drive the
        # system toward where the trajectory wants it to be at t + dt.
        dt = self.model.opt.timestep
        t_next = (self.current_step + 1) * dt
        des_pos, des_vel = self._eval_trajectory(t_next)
        self.base_target_pos = des_pos
        self.base_target_vel = des_vel
        self.ee_target_pos = self.base_target_pos + self.EE_OFFSET

        # Optional moving-target perturbation on top of the planned reference.
        if self.moving_target:
            t = self.current_step * dt
            wobble = np.array([0.05 * np.cos(t), 0.05 * np.sin(t), 0.0])
            self.base_target_pos = self.base_target_pos + wobble
            self.ee_target_pos = self.base_target_pos + self.EE_OFFSET

        # ------------- Reward: trajectory tracking ------------------- #
        base_pos = self.data.qpos[0:3]
        base_vel = self.data.qvel[0:3]

        pos_error = np.linalg.norm(base_pos - self.base_target_pos)
        vel_error = np.linalg.norm(base_vel - self.base_target_vel)

        if self._has_ee_site:
            ee_pos = self.data.site_xpos[self._ee_site_id].copy()
            ee_error = np.linalg.norm(ee_pos - self.ee_target_pos)
        else:
            ee_pos = np.zeros(3)
            ee_error = 0.0

        # Bounded exponential shaping (scale-invariant, well-behaved for PPO).
        pos_reward = np.exp(-2.0 * pos_error ** 2)
        vel_reward = np.exp(-1.0 * vel_error ** 2)
        ee_reward = np.exp(-3.0 * ee_error ** 2)

        ctrl_effort = float(np.sum(np.square(action)))

        reward = (
            2.0 * pos_reward
            + 1.0 * vel_reward
            + 3.0 * ee_reward
            - 0.01 * ctrl_effort
            + 0.1                        # small survival bonus
        )

        # Bonus for being SETTLED at the goal during the hover phase: rewards
        # both small position error AND small velocity simultaneously. Peaks at
        # +5.0 when (||p − goal||, ||v||) → (0, 0); decays sharply otherwise.
        # This avoids the bang-bang failure mode where the policy dives through
        # the goal at high speed to clip the bonus.
        goal_dist = np.linalg.norm(base_pos - self.goal_pos)
        base_speed_sq = float(np.dot(base_vel, base_vel))
        if t_next >= self.T_traj:
            reward += 5.0 * np.exp(-50.0 * goal_dist ** 2 - 2.0 * base_speed_sq)

        # Termination conditions (crash or out-of-bounds).
        # Floor at z=0.05 (was 0.1) gives ~0.25 m of margin to the hover
        # keyframe (z=0.3) so init noise + early-step transients can never
        # accidentally trigger termination before the policy gets to act.
        z = base_pos[2]
        terminated = bool(
            z < 0.05 or z > 3.0
            or abs(base_pos[0]) > 3.0
            or abs(base_pos[1]) > 3.0
        )
        truncated = bool(self.current_step >= self.max_steps)

        if terminated:
            reward -= 50.0

        info = {
            "pos_error": pos_error,
            "vel_error": vel_error,
            "ee_error": ee_error,
            "goal_dist": goal_dist,
            "ref_pos": self.base_target_pos.copy(),
            "ref_vel": self.base_target_vel.copy(),
            "end_effector_pos": ee_pos,
            "terminated": terminated,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                from mujoco import viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
