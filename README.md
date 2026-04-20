# PPO + MPC-BLF safety filter for an aerial manipulator

Trajectory tracking **and** disturbance rejection for a Skydio X2 quadrotor
carrying a 3-DoF arm in MuJoCo. A PPO policy generates rotor commands at
every control step; a receding-horizon MPC with a **Barrier Lyapunov
Function** (BLF) constraint filters those commands so the base stays inside
a user-chosen safety tube and so the velocity is damped at every planned
stage.

```
 reference            PPO policy              MPC-BLF filter          MuJoCo
  min-snap   ─▶  (learned, reactive) ─▶  (model-based, safe)  ─▶   simulation
  (p*, v*)           (u_ppo)                (u_safe)               (12-state)
```

The filter

1. rolls the full 12-state quadrotor dynamics `x = [p, v, φ, ω]` out with RK4
   for `N` steps,
2. keeps the state inside a combined position–velocity tube
   `‖e_p‖² + α‖e_v‖² < δ²`, where `e_p = p − p*`, `e_v = v − v*`,
3. forces that barrier to contract geometrically along the horizon,
   `V_{k+1} ≤ β_k V_k + σ` with a penalised scalar slack `σ`,
4. adds a stage-wise velocity-tracking penalty `(w_v/N) Σ ‖v_k − v_k*‖²`
   to break the position-only BLF's indifference between *stop at setpoint*
   and *coast through it* (the origin of the tail oscillation seen in
   early experiments),
5. and minimises deviation from PPO's proposal subject to hard motor
   limits.

## Results

All numbers come from `videos/METRICS.txt`, reproducible with the commands
at the bottom of this README.

### Trajectory tracking (min-snap reference, 20 seeds, seed 0)

| metric                                | PPO only   | PPO + MPC-BLF | delta        |
| :------------------------------------ | ---------: | ------------: | :----------- |
| Crash rate                            |    0.00 %  |      0.00 %   |              |
| Goal reach rate (15 cm tol.)          |  100.00 %  |    100.00 %   |              |
| Position tracking RMSE  [m]           |    0.1443  |    **0.0658** | **−54.4 %**  |
| Velocity tracking RMSE  [m/s]         |    0.1599  |    **0.0646** | **−59.6 %**  |
| Peak tracking error [m]               |    0.2192  |    **0.0910** | **−58.5 %**  |
| Tube violation rate (‖e‖ > 12 cm)     |   40.72 %  |    **0.00 %** | **−100 %**   |
| Median solve time [ms]                |    0.91    |     33.2      |              |
| Peak   solve time [ms]                |    6.86    |    117.3      |              |
| MPC active rate (NLP actually solved) |      –     |    100.00 %   |              |
| MPC fallback rate                     |      –     |      0.00 %   |              |

The filter cuts every tracking-error metric roughly in half and eliminates
tube violations entirely, at the cost of an extra ~33 ms median NLP solve
every control step (stride=1 on trajectory tracking).

### Disturbance rejection (2 s hover + 2 s disturb + 10 s stabilize, seed 0)

Settling time = first moment position error stays below 5 cm for ≥ 0.5 s
after the disturbance ends.

| scenario                                  | metric (Phase 3 — stabilize) | PPO only        | PPO + MPC-BLF |
| :---------------------------------------- | :--------------------------- | --------------: | ------------: |
| **wind-low** (0.8 N peak, 3 gusts)        | pos RMSE / max [m]           | 0.056 / 0.072   | 0.053 / 0.057 |
|                                           | vel RMSE / max [m/s]         | 0.057 / 0.164   | 0.020 / 0.101 |
|                                           | settled within 10 s          | no              | **yes (8.97 s)** |
|                                           | crashed                      | no              | no            |
| **wind-high** (2.5 N peak, 3 gusts)       | pos RMSE / max [m]           | 0.138 / 0.523   | 0.047 / 0.075 |
|                                           | vel RMSE / max [m/s]         | 0.629 / 2.141   | 0.067 / 0.295 |
|                                           | settled within 10 s          | no              | **yes (0.87 s)** |
|                                           | crashed                      | no              | no            |
| **arm-fold** (90° bent-elbow PD hold)     | pos RMSE / max [m]           | 0.236 / 1.448   | 0.031 / 0.036 |
|                                           | vel RMSE / max [m/s]         | 1.204 / 6.061   | 0.024 / 0.055 |
|                                           | settled within 10 s          | –               | **yes (0.01 s)** |
|                                           | crashed                      | **YES**         | no            |

Same filter, same code path, two different hyper-parameter sets: one tuned
for a moving min-snap reference (trajectory table), one tuned for hover
plus disturbance (this table).

### Videos

`videos/` contains paired PPO and MPC-BLF recordings for each scenario
above. Every disturbance video is 14 s long and has a caption band
labelling each phase (Hover / Disturbance / Stabilize) with the current
time. `videos/METRICS.txt` has the per-video numbers next to the file
names.

```
videos/
  ppo/ep00_seed{0,1,2}.mp4             # trajectory, PPO only
  ppo_mpc_blf/ep00_seed{0,1,2}.mp4     # trajectory, PPO + MPC-BLF
  disturbance/{wind_low,wind_high,arm_fold}_ppo.mp4
  disturbance/{wind_low,wind_high,arm_fold}_mpc_blf.mp4
  METRICS.txt
```

## Best-found MPC-BLF parameters

Picked by the sweeps in [`scripts/`](scripts/README.md). See
`logs/tune_*.decision` for the final numbers.

### Trajectory tracking (moving reference)

```
--mpc-horizon 7               # N
--tube 0.12                   # δ  [m]
--mpc-stride 1                # solve every env step
--slack-penalty 1e3           # ρ
--velocity-penalty 5e4        # w_v (stage-wise velocity tracking error)
--barrier-velocity-weight 0.03  # α  [s²] (velocity in the BLF)
--beta-start 0.95
--beta-end   0.15
--smooth-penalty 1e-2
--mpc-solver fatrop
```

### Disturbance rejection (hover with external load)

```
--mpc-horizon 7
--tube 0.12
--mpc-stride 2                # every 2nd step; stride=1 over-fights gusts
--slack-penalty 1e4           # heavier slack penalty for persistent loads
--velocity-penalty 5e4
--barrier-velocity-weight 0.1
--beta-start 0.95
--beta-end   0.15
--smooth-penalty 1e-2
--mpc-solver fatrop
```

Reference velocity is fed into the filter (`ref_v_p`), so both the BLF and
the stage-wise damping term operate on velocity *error* `v − v*`. This is
what lets one filter handle a moving reference (`v* ≠ 0`) and a hover
setpoint (`v* = 0`) with the same formulation; in disturbance mode
`ref_vel_traj` defaults to zeros.

## Repository layout

```
controllers/
  mpc_blf_filter.py       CasADi MPC with softened velocity-aware BLF descent
env/
  aerial_manipulator.py   Gymnasium env: quadrotor + 3-DoF arm, min-snap reference
models/
  skydio_arm.xml          MJCF for the base + arm combo
logs/continue_7M/
  best_model.zip          Reference PPO checkpoint used by the eval scripts
train_ppo.py              PPO training from scratch
train_ppo_continue.py     Warm-started fine-tuning
eval_ppo.py               PPO-only evaluation (baseline)
eval_ppo_mpc.py           PPO + MPC-BLF evaluation on the min-snap trajectory
eval_ppo_disturbance.py   PPO +/- MPC-BLF on wind / arm-fold disturbances
scripts/                  Sweep orchestrators that picked the defaults above
videos/
  METRICS.txt             Per-video metrics for every committed recording
  ppo/, ppo_mpc_blf/      Trajectory-tracking recordings
  disturbance/            Disturbance-rejection recordings
```

## Install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Runtime: MuJoCo 3+, Stable-Baselines3 2+, CasADi 3.6+ (NLP solver) with the
Fatrop backend, imageio, Pillow (caption overlay on disturbance videos).

## Quick start

Reproduce the trajectory-tracking numbers in the table above:

```bash
python eval_ppo.py     --episodes 20 --seed 0
python eval_ppo_mpc.py --episodes 20 --seed 0 \
    --mpc-horizon 7 --tube 0.12 --mpc-stride 1 \
    --slack-penalty 1e3 --velocity-penalty 5e4 --barrier-velocity-weight 0.03
```

Reproduce the three disturbance recordings:

```bash
for m in wind-low wind-high arm-fold; do
  python eval_ppo_disturbance.py --mode $m --seed 0 --stabilize-time 10.0 \
      --save-video videos/disturbance/${m//-/_}_ppo.mp4
  python eval_ppo_disturbance.py --mode $m --seed 0 --stabilize-time 10.0 --mpc-blf \
      --tube 0.12 --mpc-stride 2 \
      --slack-penalty 1e4 --velocity-penalty 5e4 --barrier-velocity-weight 0.1 \
      --save-video videos/disturbance/${m//-/_}_mpc_blf.mp4
done
```

Re-run the parameter sweeps:

```bash
python scripts/tune_trajectory.py   # horizon / tube / stride on min-snap
python scripts/tune_arm_fold.py     # rho / alpha / w_v on arm-fold disturbance
python scripts/tune_wind_high.py    # peak velocity reduction on wind-high
```

Train PPO from scratch and fine-tune:

```bash
python train_ppo.py
python train_ppo_continue.py --init logs/best_model.zip --steps 2000000 --tag 7M
```

## MPC-BLF formulation (one page)

At each solve the filter picks `U_0, …, U_{N−1}` that minimise deviation
from PPO's proposal `u_ppo` subject to:

- **Motor limits** (hard): box bounds on each rotor thrust.
- **Outer tube** (hard): `‖e_p,k‖² + α‖e_v,k‖² ≤ outer²` for every `k`, so
  the BLF is well-defined everywhere along the horizon.
- **Softened BLF descent**:

  ```
  z(e_p, e_v) = ‖e_p‖² + α ‖e_v‖²
  V(z)        = z / (δ² − z)                        (infinite at the tube edge)
  V_{k+1}     ≤ β_k · V_k + σ,   σ ≥ 0,   k = 0, …, N−1
  ```

  `β_k` is scheduled **geometrically** from `β_start = 0.95` (loose
  near-term; transient errors are unavoidable) down to `β_end = 0.15`
  (strict contraction at the horizon end). The scalar slack `σ` is
  heavily penalised (`ρ σ²`, ρ = 10³–10⁴). A finite `V` means the state is
  strictly inside the tube; when `σ ≈ 0` the planned horizon is
  forward-invariant there.

- **Cost**:

  ```
  J = ‖U_0 − u_ppo‖²  +  λ_smooth Σ ‖U_k − U_{k-1}‖²  +  ρ σ²
      + (w_v / N) Σ_{k=1..N} ‖v_k − v_k*‖²
  ```

  The last term is the stage-wise velocity-tracking penalty. The
  position-only BLF is indifferent between "stop at the setpoint" and
  "coast through it" — penalising *velocity tracking error* at every
  planned stage breaks that indifference and is what eliminates the tail
  oscillation that was present in early experiments.

If the NLP is infeasible (motor saturation + outer-tube collision) the
filter falls back to PPO's raw command. In the 20-episode trajectory eval
this fallback never triggers; on the disturbance tests it fires on
~0.6 % of steps during the gust peak.

### Flag reference

```
--tube                 δ — BLF tube radius, metres              (0.12)
--beta-start           β_0, loose near-term                      (0.95)
--beta-end             β_{N-1}, strict terminal                  (0.15)
--slack-penalty        ρ in ρ σ²                                 (1e3 / 1e4)
--velocity-penalty     w_v, stage-wise velocity error damping    (5e4)
--barrier-velocity-weight  α [s²], velocity in the BLF           (0.03 / 0.1)
--smooth-penalty       λ_smooth for ‖ΔU_k‖² smoothing            (1e-2)
--mpc-horizon          N discrete steps @ env dt                 (7)
--mpc-stride           solve once per k env steps                (1 traj / 2 dist)
--mpc-solver           ipopt / sqpmethod / fatrop                (fatrop)
```

## License

MIT — see [`LICENSE`](LICENSE).
