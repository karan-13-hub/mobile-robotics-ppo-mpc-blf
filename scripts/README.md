# Tuning orchestrators

These scripts run coarse parameter sweeps on top of `eval_ppo_mpc.py` and
`eval_ppo_disturbance.py`, parse the stdout of each run, and pick a
constrained-optimal configuration. They wrote the numbers baked into the
defaults in the evaluation scripts and in `videos/METRICS.txt`.

All three write three artefacts into `logs/` (git-ignored):

| file                       | meaning                                           |
| :------------------------- | :------------------------------------------------ |
| `<name>.log`               | full stdout/stderr of every sub-run               |
| `<name>.summary`           | one line per config with the parsed metrics       |
| `<name>.decision`          | final winner knobs and the before/after delta     |

## `tune_trajectory.py`

Focused sweep of the MPC horizon, tube radius, and solver stride on the
min-snap trajectory-tracking task (5-episode screen, then 20-episode
validation of the top 3). Slack/velocity/BLF velocity weights are fixed
at values already found by the disturbance sweeps.

**Selected:** `horizon=7, tube=0.12 m, stride=1, slack=1e3, w_v=5e4, alpha=0.03`.
Beats the PPO baseline by ~55 % on position RMSE with 0 % tube violations.

## `tune_arm_fold.py`

3-D grid over `(slack_penalty, barrier_velocity_weight, velocity_penalty)`
on the arm-fold disturbance (PD-held bent-elbow pose, persistent COM
shift through the whole stabilize phase).

**Selected:** `rho=1e4, alpha=0.1, w_v=5e4, stride=2`. PPO alone crashes
this test; the filter settles in < 0.1 s with 0 % fallback. These same
parameters also improved the wind-low / wind-high disturbance runs, so
they are used for the full disturbance-rejection suite.

## `tune_wind_high.py`

Targeted sweep that tries to reduce the Phase 3 peak velocity on the
wind-high test under the constraints `fallback <= 2 %`, `pos_max <= 9 cm`,
`vel_rmse <= 0.09 m/s`. Explores `alpha`, `w_v`, `rho`, and stride; horizon
and tube are fixed. Confirms that the arm-fold config is still the
constrained optimum and that `stride=1` is unsuitable for the gusty
wind-high scenario (it crashes every time because it overrides PPO's
high-frequency damping).

## Usage

```bash
source venv/bin/activate
python scripts/tune_trajectory.py
python scripts/tune_arm_fold.py
python scripts/tune_wind_high.py
```

Each sweep takes ~30 min to 1 h on a laptop-class CPU.
