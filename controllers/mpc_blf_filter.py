"""
Eval-time MPC + BLF safety filter for the trained PPO controller.

At every control step the filter solves a receding-horizon NLP that
plans the next N rotor commands under the full nonlinear 12-state
quadrotor dynamics. The safety constraint is a softened discrete
Lyapunov descent on the barrier function

    V(e) = ||e||^2 / (delta^2 - ||e||^2),

imposed at every stage k = 0, 1, ..., N-1:

    V(e_{k+1}) <= beta_k * V(e_k) + sigma,   sigma >= 0,

with beta_k a geometrically-decaying schedule from beta_start (loose,
near-term errors are ~unavoidable) to beta_end (strict, force descent
at the horizon end). sigma is a *scalar* non-negative slack shared
across all stages: it absorbs exactly the V-descent infeasibility the
motor envelope imposes on aggressive trajectories, so the NLP almost
never aborts. A large quadratic penalty rho * sigma^2 in the cost
drives sigma to zero whenever the barrier contraction is physically
achievable. Finite V <=> strict membership in the delta-tube; an
outer-tube hard fence ||e_k|| <= outer_tube keeps V well-defined.
Only infeasibilities in the *hard* constraints (motor limits,
dynamics, outer tube) cause a fallback to PPO's raw rotor command.

Dynamics model: full nonlinear 12-state quadrotor (R(eul) thrust rotation,
complete roll/pitch/yaw gravity compensation, full Euler kinematics, and
Coriolis/gyroscopic torques). Implemented in CasADi; the NLP can be solved
by IPOPT, fatrop, or SQPmethod.

State (12):  x = [pos(3), vel(3), eul_zyx(3), omega_body(3)]
Input  (4):  u = [c1, c2, c3, c4]   raw rotor commands in [u_lo, u_hi] N
Reference:   p_ref(t) sampled from the env's min-snap polynomial.

The filter only modifies the 4 rotor commands; the 3 arm-joint commands
pass through unchanged. PPO weights are frozen.
"""

from __future__ import annotations

import time
from typing import Tuple

import casadi as ca
import mujoco
import numpy as np

GRAVITY = 9.81


# ====================================================================== #
# Helpers                                                                 #
# ====================================================================== #
def _quat_to_euler_zyx(q: np.ndarray) -> np.ndarray:
    """ZYX Euler angles [roll, pitch, yaw] from quaternion [w, x, y, z].

    No small-angle assumption; valid up to gimbal lock at pitch = +-pi/2.
    Sign-stable across the quaternion double cover.
    """
    if q[0] < 0.0:
        q = -np.asarray(q, dtype=np.float64)
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=np.float64)


def mj_state_to_x12(data: mujoco.MjData) -> np.ndarray:
    """Pack a MuJoCo data snapshot into the 12-d MPC state vector.

    qpos = [x, y, z, qw, qx, qy, qz, j1, j2, j3]
    qvel = [vx, vy, vz, wx, wy, wz, dj1, dj2, dj3]   (omega is body-frame)

    The 12-d state is [pos, vel, eul_zyx, omega_body] where eul_zyx is the
    full ZYX Euler triple (roll, pitch, yaw), not a small-angle approximation.
    """
    pos = np.asarray(data.qpos[0:3], dtype=np.float64).copy()
    vel = np.asarray(data.qvel[0:3], dtype=np.float64).copy()
    eul = _quat_to_euler_zyx(np.asarray(data.qpos[3:7], dtype=np.float64))
    omega = np.asarray(data.qvel[3:6], dtype=np.float64).copy()
    return np.concatenate([pos, vel, eul, omega])


# ====================================================================== #
# Filter                                                                  #
# ====================================================================== #
class MPCBLFSafetyFilter:
    """BLF MPC safety filter applied to PPO rotor commands.

    At every control step the filter solves a receding-horizon NLP that
    plans the next N rotor commands under the full nonlinear quadrotor
    dynamics. The BLF constraint is a softened discrete Lyapunov
    descent on the barrier function
    V(e) = ||e||^2 / (delta^2 - ||e||^2), imposed at every stage
    k = 0, 1, ..., N-1:

        V(e_{k+1}) <= beta_k * V(e_k) + sigma,   sigma >= 0,

    with beta_k scheduled geometrically from beta_start (loose, near-
    term errors are ~unavoidable) to beta_end (strict, force descent
    toward the reference at the horizon end) and sigma a *scalar*
    non-negative slack shared across all stages, heavily penalized
    (rho * sigma^2) so it is only nonzero when the motor envelope
    physically rules out the strict contraction. Finite V iff the error
    is strictly inside the tube, so when sigma stays small the planned
    trajectory is forward-invariant in the tube. Infeasibility of the
    *hard* constraints (motor limits, dynamics, outer tube) falls back
    to PPO's raw rotor command.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        *,
        horizon: int = 10,
        tube: float = 0.05,
        beta_start: float = 2.0,
        beta_end: float = 0.5,
        slack_penalty: float = 1e3,
        smooth_penalty: float = 1e-3,
        velocity_penalty: float = 0.0,
        barrier_velocity_weight: float = 0.0,
        outer_tube: float = 0.5,
        v_floor: float = 1e-4,
        solver: str = "ipopt",
        verbose: bool = False,
    ):
        self.model = mj_model
        self.N = int(horizon)
        self.delta = float(tube)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.rho = float(slack_penalty)
        self.lambda_smooth = float(smooth_penalty)
        # Stage-wise velocity damping: adds `w_v * sum_{k=1..N} ||v_k||^2`
        # to the cost -- conceptually the D-term of a PID, applied to every
        # planned future state rather than just the terminal one. The
        # position-only BLF is indifferent between "stop at the setpoint"
        # and "coast through it", which is what was exciting the tail
        # oscillation on tight-tube stabilize phases; penalising velocity
        # at every stage breaks that indifference over the whole horizon.
        # Normalised by N so the weight scale is independent of horizon.
        self.w_v = float(velocity_penalty)
        # Velocity-aware BLF: extends the barrier from a pure position
        # tube ||e_p||^2 < delta^2 to a (p, v) tube
        #     z(e_p, e_v) = ||e_p||^2 + alpha * ||e_v||^2 < delta^2,
        # so V(z) = z / (delta^2 - z) blows up when *either* the position
        # error OR a weighted velocity error approaches the tube edge.
        # Units of alpha: s^2 (lookahead-time squared). alpha = 0 -> legacy
        # position-only BLF.
        self.alpha_blf = float(barrier_velocity_weight)
        self.outer_tube = float(outer_tube)
        self.v_floor = float(v_floor)
        self.solver_name = str(solver).lower()
        self.verbose = bool(verbose)

        self.dt = float(mj_model.opt.timestep)

        # beta_k schedule (geometric), one per BLF-descent constraint.
        self.betas = self._build_beta_schedule(self.N, self.beta_start,
                                               self.beta_end)

        # Inertial parameters from the model (one-time).
        self.mass, self.J_diag = self._extract_inertial_params()

        # Rotor allocation matrix M (4x4): [F_z, tau_x, tau_y, tau_z] = M @ u
        self.M = self._extract_allocation_matrix()
        # Per-rotor command range (matches env action mapping).
        self.u_lo, self.u_hi = self._extract_actuator_range()

        if self.verbose:
            print(f"[MPC-BLF] solver = {self.solver_name}")
            print(f"[MPC-BLF] mass = {self.mass:.4f} kg, "
                  f"J_diag = {self.J_diag.tolist()}")
            print(f"[MPC-BLF] u in [{self.u_lo}, {self.u_hi}] N per rotor")
            print(f"[MPC-BLF] N={self.N}, delta={self.delta}, dt={self.dt}")
            print(f"[MPC-BLF] BLF: V(e) = ||e||^2 / (delta^2 - ||e||^2), "
                  f"soft descent V_{{k+1}} <= beta_k * V_k + sigma "
                  f"(sigma >= 0, scalar)")
            print(f"[MPC-BLF] beta schedule (k=0..N-1) = "
                  f"{[round(float(b), 3) for b in self.betas]}")
            print(f"[MPC-BLF] slack_penalty (rho) = {self.rho} "
                  f"on scalar sigma absorbing V-descent infeasibility")
            print(f"[MPC-BLF] outer_tube = {self.outer_tube} m (hard fence)")
            print(f"[MPC-BLF] M (alloc, rows = [F_z, tau_x, tau_y, tau_z]):\n{self.M}")

        self._build_nlp()
        self.reset()

    # ------------------------------------------------------------------ #
    # BLF beta schedule                                                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_beta_schedule(N: int, beta_start: float,
                             beta_end: float) -> np.ndarray:
        """Geometric schedule beta_0 .. beta_{N-1}.

        beta_k ties the BLF descent constraint V_{k+1} <= beta_k * V_k.
        beta_0 = beta_start (loose -- near-term error is largely
        unavoidable), beta_{N-1} = beta_end (strict -- force descent at
        the horizon end). Any finite beta_k > 0 preserves the barrier
        guarantee; only the magnitude controls how aggressively V is
        driven down.
        """
        if N < 1:
            raise ValueError("horizon must be >= 1")
        if beta_start <= 0.0 or beta_end <= 0.0:
            raise ValueError("beta_start and beta_end must be positive")
        if N == 1:
            return np.array([beta_end])
        exps = np.linspace(0.0, 1.0, N)  # k=0 -> 0, k=N-1 -> 1
        return beta_start * (beta_end / beta_start) ** exps

    # ------------------------------------------------------------------ #
    # Model extraction                                                    #
    # ------------------------------------------------------------------ #
    def _extract_inertial_params(self) -> Tuple[float, np.ndarray]:
        """Total drone+arm mass and diagonal inertia about the COM.

        Aggregated via parallel-axis theorem from per-body inertias evaluated
        at the hover keyframe (so xipos / ximat are well-defined). Off-diagonal
        terms are dropped because the X-config drone is symmetric about both
        body axes; the residual coupling is below 1% of the diagonal.
        """
        data = mujoco.MjData(self.model)
        hover_id = -1
        for i in range(self.model.nkey):
            if self.model.key(i).name == "hover":
                hover_id = i
                break
        if hover_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, data, hover_id)
        else:
            mujoco.mj_resetData(self.model, data)
        mujoco.mj_forward(self.model, data)

        total_mass = 0.0
        total_com = np.zeros(3)
        for b in range(1, self.model.nbody):
            m = float(self.model.body_mass[b])
            if m <= 0.0:
                continue
            total_mass += m
            total_com += m * np.asarray(data.xipos[b])
        if total_mass <= 0.0:
            raise RuntimeError("Model has zero subtree mass; cannot build MPC.")
        total_com /= total_mass

        J = np.zeros((3, 3))
        for b in range(1, self.model.nbody):
            m = float(self.model.body_mass[b])
            if m <= 0.0:
                continue
            I_b_diag = np.asarray(self.model.body_inertia[b], dtype=np.float64)
            R_b = np.asarray(data.ximat[b], dtype=np.float64).reshape(3, 3)
            I_b_world = R_b @ np.diag(I_b_diag) @ R_b.T
            r = np.asarray(data.xipos[b], dtype=np.float64) - total_com
            J += I_b_world + m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))

        return total_mass, np.diag(J).copy()

    def _extract_allocation_matrix(self) -> np.ndarray:
        """Build M with [F_z, tau_x, tau_y, tau_z]^T = M @ [c1..c4]^T.

        Each motor's `gear` is its body-frame wrench per unit ctrl.
        Rotor sites for this model have lateral COM offset of ~0 (symmetric),
        so the small body-frame z-shift between body origin and COM does not
        change the lateral torques. We use raw site positions.
        """
        M = np.zeros((4, 4))
        for i in range(4):
            trntype = self.model.actuator_trntype[i]
            if trntype != mujoco.mjtTrn.mjTRN_SITE:
                raise RuntimeError(
                    f"Actuator {i} must be a site-transmission motor "
                    f"(got trntype={trntype}); MPC builder needs rotor sites."
                )
            site_id = int(self.model.actuator_trnid[i, 0])
            r_site = np.asarray(self.model.site_pos[site_id], dtype=np.float64)
            gear = np.asarray(self.model.actuator_gear[i], dtype=np.float64)
            f_z = gear[2]
            tau_z_react = gear[5]
            M[0, i] = f_z
            M[1, i] = r_site[1] * f_z
            M[2, i] = -r_site[0] * f_z
            M[3, i] = tau_z_react
        return M

    def _extract_actuator_range(self) -> Tuple[float, float]:
        lo = float(self.model.actuator_ctrlrange[0, 0])
        hi = float(self.model.actuator_ctrlrange[0, 1])
        for i in range(1, 4):
            if (float(self.model.actuator_ctrlrange[i, 0]) != lo
                    or float(self.model.actuator_ctrlrange[i, 1]) != hi):
                raise RuntimeError("MPC assumes all 4 rotors share the same ctrl range.")
        return lo, hi

    # ------------------------------------------------------------------ #
    # Continuous + discrete dynamics (CasADi)                             #
    # ------------------------------------------------------------------ #
    def _continuous_dynamics(self, x, u):
        """Full nonlinear quadrotor dynamics (12-state, 4-input).

        State : [px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r]
                positions and velocities in world frame; (phi, theta, psi)
                are ZYX Euler angles (roll, pitch, yaw); (p, q, r) is the
                body-frame angular velocity.

        Translational accel uses the FULL rotation matrix R(eul) applied to
        the body thrust [0, 0, F_z], so vertical thrust loss when tilted
        (cos(theta)*cos(phi)) is captured exactly. Rotational dynamics include
        the Coriolis/gyroscopic term omega x (J*omega). Euler kinematics map
        body rates (p, q, r) to Euler rates correctly (no small-angle).
        """
        v = x[3:6]
        phi = x[6]
        theta = x[7]
        psi = x[8]
        p = x[9]
        q = x[10]
        r = x[11]

        wrench = ca.mtimes(self.M, u)
        F_z = wrench[0]
        tau = wrench[1:4]

        sphi, cphi = ca.sin(phi), ca.cos(phi)
        sth,  cth  = ca.sin(theta), ca.cos(theta)
        spsi, cpsi = ca.sin(psi), ca.cos(psi)

        # Body z-axis expressed in world (third column of R_zyx(phi, theta, psi))
        # = [cpsi*sth*cphi + spsi*sphi, spsi*sth*cphi - cpsi*sphi, cth*cphi]
        thrust_per_m = F_z / self.mass
        a_x = thrust_per_m * (cpsi * sth * cphi + spsi * sphi)
        a_y = thrust_per_m * (spsi * sth * cphi - cpsi * sphi)
        a_z = thrust_per_m * (cth * cphi) - GRAVITY
        a = ca.vertcat(a_x, a_y, a_z)

        # Body-rate -> Euler-rate transformation (ZYX). cos(theta) stays well
        # away from zero for any non-acrobatic flight; clamp magnitude to 1e-3
        # so the NLP cannot accidentally request a gimbal-lock state.
        cth_safe = ca.if_else(ca.fabs(cth) > 1e-3, cth, 1e-3 * ca.sign(cth + 1e-12))
        tan_th = sth / cth_safe
        phi_dot   = p + sphi * tan_th * q + cphi * tan_th * r
        theta_dot = cphi * q - sphi * r
        psi_dot   = (sphi / cth_safe) * q + (cphi / cth_safe) * r
        eul_dot = ca.vertcat(phi_dot, theta_dot, psi_dot)

        # Rotational dynamics with Coriolis: J * omega_dot = tau - omega x J*omega.
        Jx, Jy, Jz = self.J_diag[0], self.J_diag[1], self.J_diag[2]
        p_dot = (tau[0] - (Jz - Jy) * q * r) / Jx
        q_dot = (tau[1] - (Jx - Jz) * p * r) / Jy
        r_dot = (tau[2] - (Jy - Jx) * p * q) / Jz
        omega_dot = ca.vertcat(p_dot, q_dot, r_dot)

        return ca.vertcat(v, a, eul_dot, omega_dot)

    def _rk4(self, x, u, dt):
        k1 = self._continuous_dynamics(x, u)
        k2 = self._continuous_dynamics(x + dt / 2.0 * k1, u)
        k3 = self._continuous_dynamics(x + dt / 2.0 * k2, u)
        k4 = self._continuous_dynamics(x + dt * k3, u)
        return x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # ------------------------------------------------------------------ #
    # Solver configuration                                                #
    # ------------------------------------------------------------------ #
    def _configure_solver(self, opti: ca.Opti) -> None:
        """Attach the selected NLP solver to `opti`.

        Supported:
          - 'ipopt'      robust general-purpose NLP (baseline)
          - 'sqpmethod'  CasADi SQP + qpoases QP subsolver (fast, small NLPs)
          - 'fatrop'     structure-exploiting interior point for OCPs
        """
        name = self.solver_name
        if name == "ipopt":
            opti.solver("ipopt", {
                "print_time": False,
                "ipopt.print_level": 0,
                "ipopt.max_iter": 20,
                "ipopt.warm_start_init_point": "yes",
                "ipopt.tol": 1e-3,
                "ipopt.acceptable_tol": 1e-2,
                "ipopt.acceptable_iter": 3,
                "ipopt.sb": "yes",
            })
        elif name == "sqpmethod":
            opti.solver("sqpmethod", {
                "print_time": False,
                "print_header": False,
                "print_iteration": False,
                "print_status": False,
                "max_iter": 15,
                "tol_du": 1e-3,
                "tol_pr": 1e-3,
                "qpsol": "qpoases",
                "qpsol_options": {
                    "printLevel": "none",
                    "sparse": True,
                    "error_on_fail": False,
                },
            })
        elif name == "fatrop":
            opti.solver("fatrop", {
                "print_time": False,
                "expand": True,
                "fatrop": {
                    "print_level": 0,
                    "max_iter": 30,
                    "tol": 1e-3,
                    "acceptable_tol": 1e-2,
                    "warm_start_init_point": True,
                },
            })
        else:
            raise ValueError(
                f"Unknown solver '{self.solver_name}'. "
                f"Use one of: 'ipopt', 'sqpmethod', 'fatrop'."
            )

    # ------------------------------------------------------------------ #
    # NLP construction                                                    #
    # ------------------------------------------------------------------ #
    def _build_nlp(self):
        N = self.N
        nx, nu = 12, 4

        # Standalone RK4 function reused for numerical open-loop prediction
        # (predictive gating). Building it once here avoids re-tracing the
        # graph at every filter() call.
        x_sym = ca.MX.sym("x_rk4", nx)
        u_sym = ca.MX.sym("u_rk4", nu)
        self._rk4_fn = ca.Function(
            "rk4_fn", [x_sym, u_sym], [self._rk4(x_sym, u_sym, self.dt)]
        )

        opti = ca.Opti()

        X = opti.variable(nx, N + 1)
        U = opti.variable(nu, N)
        # Scalar non-negative slack shared across all BLF descent stages.
        # It absorbs the infeasibility the motor envelope forces on the
        # strict barrier contraction; a heavy quadratic penalty keeps it
        # at zero whenever the contraction is physically achievable.
        sigma = opti.variable()

        x0_p = opti.parameter(nx)
        u_ppo_p = opti.parameter(nu)
        ref_p = opti.parameter(3, N + 1)
        # Reference velocity at each horizon stage. Zero for hover/regulation;
        # the env's min-snap trajectory provides non-zero v_ref. Feeding the
        # filter the actual v_ref is what makes BLF/damping work on moving
        # targets: "error" must be on velocity error, not absolute velocity.
        ref_v_p = opti.parameter(3, N + 1)

        opti.subject_to(X[:, 0] == x0_p)
        for k in range(N):
            opti.subject_to(X[:, k + 1] == self._rk4(X[:, k], U[:, k], self.dt))

        opti.subject_to(opti.bounded(self.u_lo, U, self.u_hi))

        # --- Softened BLF descent with decaying beta schedule ------------
        # Barrier Lyapunov function V(e) = ||e||^2 / (delta^2 - ||e||^2).
        # Finite V <=> strict membership in the delta-tube. The denominator
        # is clamped by v_floor >= 0 for CasADi numerical stability near
        # ||e|| = delta; at feasibility the clamp is inactive.
        delta2 = self.delta ** 2
        outer2 = self.outer_tube ** 2
        v_floor = self.v_floor
        alpha = self.alpha_blf

        def z_expr(e_p, e_v):
            """z = ||e_p||^2 + alpha * ||e_v||^2 (position + weighted vel)."""
            return ca.dot(e_p, e_p) + alpha * ca.dot(e_v, e_v)

        def V_expr(e_p, e_v):
            z = z_expr(e_p, e_v)
            return z / ca.fmax(delta2 - z, v_floor)

        # Outer-tube hard fence at every stage on the same combined norm,
        # so V stays well-defined everywhere the NLP can wander. Velocity
        # error is v - v_ref so a moving trajectory is not penalised for
        # simply having non-zero velocity.
        for k in range(N + 1):
            e_p_k = X[0:3, k] - ref_p[:, k]
            e_v_k = X[3:6, k] - ref_v_p[:, k]
            opti.subject_to(z_expr(e_p_k, e_v_k) <= outer2)

        # Softened BLF descent V_{k+1} <= beta_k * V_k + sigma,
        # sigma >= 0, k = 0 .. N-1.
        opti.subject_to(sigma >= 0.0)
        e_p0 = X[0:3, 0] - ref_p[:, 0]
        e_v0 = X[3:6, 0] - ref_v_p[:, 0]
        V_prev = V_expr(e_p0, e_v0)
        for k in range(N):
            e_p_kp1 = X[0:3, k + 1] - ref_p[:, k + 1]
            e_v_kp1 = X[3:6, k + 1] - ref_v_p[:, k + 1]
            V_next = V_expr(e_p_kp1, e_v_kp1)
            opti.subject_to(V_next <= float(self.betas[k]) * V_prev + sigma)
            V_prev = V_next

        cost = ca.sumsqr(U[:, 0] - u_ppo_p)
        for k in range(1, N):
            cost += self.lambda_smooth * ca.sumsqr(U[:, k] - U[:, k - 1])
        cost += self.rho * sigma * sigma
        # Stage-wise velocity damping (optional). D-term-style: penalises
        # the velocity *tracking error* ||v_k - v_ref_k||^2 at every planned
        # stage k = 1 .. N, normalised by N so the weight scale does not
        # change with horizon length. Using tracking error (not absolute
        # velocity) is what lets this term behave well on moving trajectories.
        if self.w_v > 0.0:
            vel_cost = 0.0
            for k in range(1, N + 1):
                vel_cost = vel_cost + ca.sumsqr(X[3:6, k] - ref_v_p[:, k])
            cost += (self.w_v / N) * vel_cost
        opti.minimize(cost)

        self._configure_solver(opti)

        self._opti = opti
        self._X = X
        self._U = U
        self._sigma = sigma
        self._x0_p = x0_p
        self._u_ppo_p = u_ppo_p
        self._ref_p = ref_p
        self._ref_v_p = ref_v_p

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Drop warm-start state. Call between episodes."""
        self._warm_X: np.ndarray | None = None
        self._warm_U: np.ndarray | None = None
        self._warm_sigma: float = 0.0

    def filter(
        self,
        x_meas: np.ndarray,
        a_ppo_full: np.ndarray,
        ref_pos_traj: np.ndarray,
        ref_vel_traj: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, dict]:
        """Filter PPO's action through the hard-BLF MPC.

        At every control step:
          1. If ||e_0|| >= outer_tube, V(e_0) is undefined / ill-posed.
             Hand control back to PPO (cannot make a safety claim from
             outside the outer fence).
          2. Otherwise, solve the MPC NLP with the hard BLF descent
             constraints V(e_{k+1}) <= beta_k * V(e_k) at every stage,
             minimising ||U[0] - u_ppo||^2 + smoothness.
          3. Apply U[0]. On solver failure (infeasible / numerical), fall
             back to PPO's raw action.

        Args:
            x_meas:        12-d state pulled from MuJoCo via mj_state_to_x12.
            a_ppo_full:    7-d PPO action in [-1, 1] (rotors + arm).
            ref_pos_traj:  (N+1, 3) min-snap reference positions sampled at dt.
        """
        a_out = np.asarray(a_ppo_full, dtype=np.float32).copy()
        info = {
            "fallback": False,
            "fallback_reason": "",
            "mpc_active": False,
            "solve_time_ms": 0.0,
            "V0": 0.0,
            "VN": 0.0,
            "sigma": 0.0,
            "terminal_err": 0.0,
            "u_changed_norm": 0.0,
        }

        if ref_pos_traj.shape != (self.N + 1, 3):
            raise ValueError(
                f"ref_pos_traj must have shape ({self.N + 1}, 3); "
                f"got {ref_pos_traj.shape}"
            )
        if ref_vel_traj is None:
            ref_vel_traj = np.zeros_like(ref_pos_traj)
        else:
            ref_vel_traj = np.asarray(ref_vel_traj, dtype=np.float64)
            if ref_vel_traj.shape != (self.N + 1, 3):
                raise ValueError(
                    f"ref_vel_traj must have shape ({self.N + 1}, 3); "
                    f"got {ref_vel_traj.shape}"
                )

        u_ppo_rotor = self._normalized_to_motor(a_ppo_full[:4])

        # Outer-tube safety fence on the combined (p, v) norm
        # z = ||e_p||^2 + alpha * ||e_v||^2. V is only a valid barrier
        # where z < delta^2; the outer_tube is a much wider safety limit
        # that keeps the NLP well-posed. If we are already past that,
        # no safety claim is possible -- hand back to PPO rather than
        # feed the solver a nonsense start.
        e0 = x_meas[0:3] - ref_pos_traj[0]
        ev0 = x_meas[3:6] - ref_vel_traj[0]
        z0 = float(np.dot(e0, e0) + self.alpha_blf * np.dot(ev0, ev0))
        outer2 = self.outer_tube ** 2
        if z0 >= outer2:
            info["fallback"] = True
            info["fallback_reason"] = "outside_outer_tube"
            self.reset()
            return a_out, info

        # V(e_0) for diagnostics.
        denom0 = max(self.delta ** 2 - z0, self.v_floor)
        info["V0"] = float(z0 / denom0)

        # ---- MPC solve (always active) ---------------------------------- #
        info["mpc_active"] = True

        self._opti.set_value(self._x0_p, x_meas)
        self._opti.set_value(self._u_ppo_p, u_ppo_rotor)
        self._opti.set_value(self._ref_p, ref_pos_traj.T)
        self._opti.set_value(self._ref_v_p, ref_vel_traj.T)

        if self._warm_X is None:
            X_init = np.tile(x_meas[:, None], (1, self.N + 1))
            U_init = np.tile(u_ppo_rotor[:, None], (1, self.N))
            sigma_init = 0.0
        else:
            X_init = np.concatenate([self._warm_X[:, 1:], self._warm_X[:, -1:]], axis=1)
            U_init = np.concatenate([self._warm_U[:, 1:], u_ppo_rotor[:, None]], axis=1)
            sigma_init = float(self._warm_sigma)
        self._opti.set_initial(self._X, X_init)
        self._opti.set_initial(self._U, U_init)
        self._opti.set_initial(self._sigma, sigma_init)

        t0 = time.perf_counter()
        try:
            sol = self._opti.solve()
            X_sol = np.asarray(sol.value(self._X))
            U_sol = np.asarray(sol.value(self._U))
            sigma_sol = float(sol.value(self._sigma))
        except RuntimeError as exc:
            info["fallback"] = True
            info["fallback_reason"] = f"solver_failure: {exc.__class__.__name__}"
            info["solve_time_ms"] = (time.perf_counter() - t0) * 1000.0
            self.reset()
            return a_out, info
        info["solve_time_ms"] = (time.perf_counter() - t0) * 1000.0

        self._warm_X = X_sol
        self._warm_U = U_sol
        self._warm_sigma = sigma_sol

        u0 = np.clip(U_sol[:, 0], self.u_lo, self.u_hi)
        a_out[0:4] = self._motor_to_normalized(u0).astype(np.float32)

        e_N = X_sol[0:3, -1] - ref_pos_traj[-1]
        ev_N = X_sol[3:6, -1] - ref_vel_traj[-1]
        eN_sq = float(np.dot(e_N, e_N))
        zN = float(eN_sq + self.alpha_blf * np.dot(ev_N, ev_N))
        denomN = max(self.delta ** 2 - zN, self.v_floor)
        info["VN"] = float(zN / denomN)
        info["sigma"] = float(max(sigma_sol, 0.0))
        info["terminal_err"] = float(np.sqrt(eN_sq))
        info["u_changed_norm"] = float(np.linalg.norm(u0 - u_ppo_rotor))
        return a_out, info

    # ------------------------------------------------------------------ #
    # Action <-> motor conversion (matches env.step exactly)              #
    # ------------------------------------------------------------------ #
    def _normalized_to_motor(self, a: np.ndarray) -> np.ndarray:
        a = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)
        return self.u_lo + (a + 1.0) * 0.5 * (self.u_hi - self.u_lo)

    def _motor_to_normalized(self, c: np.ndarray) -> np.ndarray:
        c = np.asarray(c, dtype=np.float64)
        return -1.0 + 2.0 * (c - self.u_lo) / max(self.u_hi - self.u_lo, 1e-9)
