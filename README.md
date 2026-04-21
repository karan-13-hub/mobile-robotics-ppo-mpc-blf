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

### Trajectory tracking (min-snap reference, 5 seeds × 20 episodes = 100 episodes)

Seeds = `[0, 20, 40, 60, 80]`.  Each entry is `mean ± std` where `std` is taken
across the five seed-level means (n=5).  Per-episode pooled stds (n=100) are
larger but qualitatively unchanged; full breakdown is in
`logs/multi_seed/summary.json`.

| metric                                | PPO only            | PPO + MPC (BLF **off**) | PPO + MPC + BLF         | Δ (PPO → BLF) |
| :------------------------------------ | ------------------: | ----------------------: | ----------------------: | :------------ |
| Crash rate                            |       0.00 %        |        0.00 %           |        0.00 %           |               |
| Goal reach rate (15 cm tol.)          |     100.00 %        |      100.00 %           |      100.00 %           |               |
| Position tracking RMSE  [m]           | 0.1413  ± 0.0038    |   0.0808  ± 0.0041      | **0.0643  ± 0.0008**    | **−54.5 %**   |
| Velocity tracking RMSE  [m/s]         | 0.1719  ± 0.0157    |   0.0744  ± 0.0026      | **0.0665  ± 0.0032**    | **−61.3 %**   |
| Peak tracking error [m]               | 0.2146  ± 0.0057    |   0.1182  ± 0.0069      | **0.0884  ± 0.0013**    | **−58.8 %**   |
| Tube violation rate (‖e‖ > 12 cm)     |  53.53 % ± 1.45 %   |  14.45 % ± 5.21 %       |   **0.00 % ± 0.00 %**   | **−100 %**    |
| Median solve time [ms]                |  1.22   ± 0.07      |  28.16   ± 0.34         |  33.39    ± 0.28        |               |
| Peak  solve time [ms] (per-ep mean)   |  2.58   ± 0.35      |  40.75   ± 0.52         |  58.22    ± 1.87        |               |
| Peak  solve time [ms] (abs. across 100 eps) | 10.44         | 111.70                  | 109.90                  |               |
| MPC active rate (NLP actually solved) |      –              |     100.00 %            |     100.00 %            |               |
| MPC fallback rate                     |      –              |       0.00 %            |       0.00 %            |               |

**BLF ablation.** The middle column runs the same MPC NLP every control
step but with the barrier-Lyapunov safety constraint disabled
(`eval_ppo_mpc.py --no-blf`), so MPC acts as a pure tracking
optimiser. That alone already cuts position RMSE 0.1413 → 0.0808 (−43 %)
and tube violations 53.5 % → 14.5 %. Re-enabling the BLF descent
constraint drops tracking by another 20 % **and** drives tube violations
all the way to 0 % across all 100 episodes — the filter does useful work
beyond just "MPC is a better controller than PPO". Run-to-run variance
(across seeds) is also an order of magnitude smaller for the full filter:
position RMSE seed-std ≈ 0.8 mm with BLF vs 4 mm without. The cost is an
extra ~5 ms median NLP (33 ms vs 28 ms) per control step at stride = 1.

### Disturbance rejection (2 s hover + 2 s disturb + 10 s stabilize, seed 0)

Settling time = first moment position error stays below 5 cm for ≥ 0.5 s
after the disturbance ends.

| scenario                                  | metric (Phase 3 — stabilize) | PPO only        | PPO + MPC (BLF off) | PPO + MPC-BLF |
| :---------------------------------------- | :--------------------------- | --------------: | ------------------: | ------------: |
| **wind-low** (0.8 N peak, 3 gusts)        | pos RMSE / max [m]           | 0.056 / 0.072   | 0.058 / 0.063       | 0.053 / 0.057 |
|                                           | vel RMSE / max [m/s]         | 0.057 / 0.164   | 0.024 / 0.087       | 0.020 / 0.101 |
|                                           | settled within 10 s          | no              | no                  | **yes (8.97 s)** |
|                                           | crashed                      | no              | no                  | no            |
| **wind-high** (2.5 N peak, 3 gusts)       | pos RMSE / max [m]           | 0.138 / 0.523   | 0.059 / 0.078       | 0.047 / 0.075 |
|                                           | vel RMSE / max [m/s]         | 0.629 / 2.141   | 0.101 / 0.279       | 0.067 / 0.295 |
|                                           | settled within 10 s          | no              | no                  | **yes (0.87 s)** |
|                                           | crashed                      | no              | no                  | no            |
| **arm-fold** (90° bent-elbow PD hold)     | pos RMSE / max [m]           | 0.236 / 1.448   | 0.032 / 0.037       | 0.031 / 0.036 |
|                                           | vel RMSE / max [m/s]         | 1.204 / 6.061   | 0.029 / 0.083       | 0.024 / 0.055 |
|                                           | settled within 10 s          | –               | **yes (0.01 s)**    | **yes (0.01 s)** |
|                                           | crashed                      | **YES**         | no                  | no            |

Same filter, same code path, two different hyper-parameter sets: one tuned
for a moving min-snap reference (trajectory table), one tuned for hover
plus disturbance (this table).

**Disturbance BLF ablation.**  MPC-without-BLF already recovers most of
the gain over raw PPO — it prevents the arm-fold crash outright and cuts
wind-high Phase-3 pos RMSE 0.138 → 0.059 m — because most of the work
under small disturbances is done by the tracking optimiser, not the
safety constraint.  The BLF earns its keep on the *settling* criterion:
both wind modes plateau at ≈ 6 cm error without it (just outside the
5 cm settling band) and only snap back inside once the barrier descent
term `V_{k+1} ≤ β·V_k` is re-enabled, giving the only checkmarks in the
"settled within 10 s" column for wind-low and wind-high.

### Videos

`videos/` contains paired PPO and MPC-BLF recordings for each scenario
above. Every disturbance video is 14 s long and has a caption band
labelling each phase (Hover / Disturbance / Stabilize) with the current
time. `videos/METRICS.txt` has the per-video numbers next to the file
names.

```
videos/
  ppo/ep00_seed{0,1,2}.mp4             # trajectory, PPO only
  ppo_mpc_noblf/ep00_seed{0,1,2}.mp4   # trajectory, PPO + MPC (BLF off)
  ppo_mpc_blf/ep00_seed{0,1,2}.mp4     # trajectory, PPO + MPC-BLF
  disturbance/{wind_low,wind_high,arm_fold}_ppo.mp4
  disturbance/{wind_low,wind_high,arm_fold}_mpc_noblf.mp4   # BLF ablation
  disturbance/{wind_low,wind_high,arm_fold}_mpc_blf.mp4
  METRICS.txt
```

### PPO training curves (≈ 6.74 M env steps)

Fully-from-scratch training run under the same protocol as the baseline
checkpoint (`logs/retrain_7M/`): 5 M env-step phase 1 at lr 3e-4, then a
2 M env-step phase 2 fine-tune at lr 1e-4 that warm-starts from phase 1's
`best_model.zip` (saved by `EvalCallback` at step 4.74 M, not the final
5 M checkpoint).  Cumulative env steps therefore top out at ≈ 6.74 M, not
a clean 7 M.  The dashed vertical line in every plot marks the true
phase-2 warm-start at 4.74 M.

The curves below come from `scripts/retrain_from_scratch.py --plots-only`,
which parses the tensorboard event files under `logs/retrain_7M/tb/`.

| metric                         | plot                                                    |
| :----------------------------- | :------------------------------------------------------ |
| Mean episode reward            | `presentation/plots/training_reward.png`                |
| Mean episode length            | `presentation/plots/training_ep_len.png`                |
| Value-function loss            | `presentation/plots/training_value_loss.png`            |
| Policy gradient loss           | `presentation/plots/training_policy_loss.png`           |
| Explained variance of value fn | `presentation/plots/training_explained_variance.png`    |

```bash
# Reproduce the training run end-to-end (~1 hour on RTX 6000 Ada)
nohup python scripts/retrain_from_scratch.py \
    > logs/retrain_7M/pipeline.log 2>&1 &

# Or just (re)render the plots from existing tensorboard events
python scripts/retrain_from_scratch.py --plots-only
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

Reproduce the trajectory-tracking numbers in the table above (5 seeds × 20
episodes, writes `logs/multi_seed/summary.json`):

```bash
# PPO vs PPO + MPC-BLF (both configurations, 5 seeds x 20 episodes each)
python scripts/eval_multi_seed.py   # seeds [0, 20, 40, 60, 80] by default

# BLF ablation: same MPC, safety constraint disabled
python scripts/eval_multi_seed.py --skip-ppo --skip-mpc --out-suffix _only_no_blf
```

A single-seed run (matches one column of the multi-seed protocol):

```bash
python eval_ppo.py     --episodes 20 --seed 0 --tube 0.12
python eval_ppo_mpc.py --episodes 20 --seed 0 \
    --mpc-horizon 7 --tube 0.12 --mpc-stride 1 \
    --slack-penalty 1e3 --velocity-penalty 5e4 --barrier-velocity-weight 0.03
# same call + --no-blf reproduces the middle column of the ablation table
python eval_ppo_mpc.py --episodes 20 --seed 0 --no-blf \
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
  # BLF ablation: same MPC NLP, outer fence + descent constraint disabled
  python eval_ppo_disturbance.py --mode $m --seed 0 --stabilize-time 10.0 --mpc-blf --no-blf \
      --tube 0.12 --mpc-stride 2 \
      --slack-penalty 1e4 --velocity-penalty 5e4 --barrier-velocity-weight 0.1 \
      --save-video videos/disturbance/${m//-/_}_mpc_noblf.mp4
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
