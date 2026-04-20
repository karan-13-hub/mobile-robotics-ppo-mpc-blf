# PPO + MPC-BLF safety filter for an aerial manipulator

Trajectory-tracking stack for a Skydio X2 quadrotor carrying a 3-DoF arm in
MuJoCo. A PPO policy generates rotor commands at every control step, and a
receding-horizon MPC with a **Barrier Lyapunov Function** (BLF) constraint
filters those commands so the base position error stays inside a user-chosen
safety tube.

```
 reference            PPO policy              MPC-BLF filter          MuJoCo
  min-snap   ─▶  (learned, reactive) ─▶  (model-based, safe)  ─▶   simulation
  (p*, v*)           (u_ppo)                (u_safe)               (12-state)
```

## Results (20 seeds, deterministic eval, default flags)

| metric                                | PPO only   | PPO + MPC-BLF |
| :------------------------------------ | ---------: | ------------: |
| Crash rate                            |    0.00 %  |      0.00 %   |
| Goal reach rate (15 cm tol.)          |  100.00 %  |    100.00 %   |
| Position tracking RMSE  [m]           |    0.1443  |    **0.0861** |
| Velocity tracking RMSE  [m/s]         |    0.1599  |      0.1742   |
| End-effector tracking RMSE [m]        |    0.2243  |      0.1989   |
| Final settling error [m]              |    0.0613  |      0.0756   |
| Tube violation rate (‖e‖ > 15 cm)     |   40.72 %  |    **0.94 %** |
| Peak tracking error [m]               |    0.2192  |    **0.1323** |
| Median solve time [ms]                |    0.91    |     39.91     |
| Peak   solve time [ms]                |    6.86    |    119.25     |
| MPC active rate (NLP actually solved) |      –     |     33.39 %   |
| MPC fallback rate                     |      –     |      0.62 %   |

The filter cuts tube violations by **~43×** (40.72 % → 0.94 %) and peak tracking
error by **~40 %** (0.22 m → 0.13 m), while also tightening the mean position
RMSE by ~40 %, all with a sub-1 % fallback rate to the unfiltered PPO command.
The added cost is a ~40 ms median solve once every 3 env steps (≈33 % duty
cycle); between solves the raw PPO command is applied directly.

<sub>* The baseline PPO peak of ~113 ms is a **one-off warm-up** on the very
first `model.predict` call (PyTorch lazy module init, CUDA kernel JIT /
allocator setup); steady-state inference is ~0.9 ms, which is why the median
is ~120× smaller than the peak. The MPC peak is instead a *typical* worst-case
NLP solve.</sub>

## Repository layout

```
controllers/
  mpc_blf_filter.py      # CasADi-based MPC with softened BLF descent
env/
  aerial_manipulator.py  # Gymnasium env: quadrotor + 3-DoF arm, min-snap ref
models/
  skydio_arm.xml         # MJCF for the base + arm combo
logs/continue_7M/
  best_model.zip         # reference PPO checkpoint used by eval_ppo_mpc.py
train_ppo.py             # from-scratch PPO training (5 M steps)
train_ppo_continue.py    # warm-started fine-tuning (2 M extra steps)
eval_ppo.py              # PPO-only evaluation (baseline)
eval_ppo_mpc.py          # PPO + MPC-BLF evaluation (this project)
```

## Install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Dependencies: MuJoCo 3+, Stable-Baselines3 2+, CasADi 3.6+ (for the MPC NLP),
imageio/matplotlib (for videos and plots). A recent NumPy is pinned below 2.2
for SB3 compatibility.

## Quick start

Evaluate the shipped 7 M-step checkpoint with the tuned MPC-BLF filter:

```bash
python eval_ppo_mpc.py
```

Baseline ablation (no MPC, pure PPO):

```bash
python eval_ppo_mpc.py --no-mpc
```

Record a video of a specific episode:

```bash
python eval_ppo_mpc.py --save-video videos/demo.mp4 --video-episode 3
```

Train PPO from scratch, then fine-tune:

```bash
python train_ppo.py
python train_ppo_continue.py --init logs/best_model.zip --steps 2000000 --tag 7M
```

## MPC-BLF formulation

At each solve the filter picks the next `N` rotor commands `U_0, …, U_{N-1}`
that roll out the full nonlinear 12-state quadrotor dynamics
`x = [p, v, φ, ω]` via RK4, and minimises deviation from PPO's proposal
`u_ppo` subject to:

- **Motor limits** (hard): box bounds on each rotor thrust.
- **Outer tube** (hard): `‖p_k − p*_k‖ ≤ outer_tube` for all `k` so the BLF
  is well-defined everywhere along the horizon.
- **Softened BLF descent** (this is the safety constraint):
  ```
  V(e)       = ‖e‖² / (δ² − ‖e‖²),          e_k = p_k − p*_k
  V(e_{k+1}) ≤ β_k · V(e_k) + σ,            σ ≥ 0,   k = 0, …, N−1
  ```
  with `β_k` scheduled **geometrically** from `β_start = 0.95` (loose,
  near-term errors are near-unavoidable) down to `β_end = 0.15` (strict,
  force contraction at the horizon end), and a single scalar slack `σ`
  heavily penalised (`ρ · σ², ρ = 3 × 10³`) in the cost. `σ` is only used
  when the motor envelope physically rules out strict contraction — in
  practice it is ~0 at steady state and briefly nonzero during aggressive
  accelerations. A finite `V` corresponds to an error strictly inside the
  tube, so whenever `σ ≈ 0` the planned trajectory is forward-invariant
  inside the tube.

If the NLP is infeasible (motor saturation + outer-tube collision), the
filter falls back to PPO's raw command.

### Key flags (tuned defaults in parentheses)

```
--tube             δ — BLF tube radius, metres         (0.15)
--beta-start       β_0, loose near-term                 (0.95)
--beta-end         β_{N-1}, strict terminal             (0.15)
--slack-penalty    ρ for ρ·σ² in cost                   (3e3)
--mpc-horizon      N discrete steps @ env dt            (7)
--mpc-stride       solve once per k env steps           (3)
--mpc-solver       ipopt / sqpmethod / fatrop          (fatrop)
--smooth-penalty   λ for ‖ΔU_k‖² smoothing term         (1e-2)
```

`mpc-stride=3` with Fatrop gives ~39 ms median solve time and a ~33 % duty
cycle; on the intervening steps PPO's raw rotor command is applied directly
(we deliberately do *not* zero-order-hold the last solved command — the
inner attitude loop destabilises under ≥ 10 ms input holds).

## License

MIT — see [`LICENSE`](LICENSE).
