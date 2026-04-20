# Presentation

Self-contained [reveal.js](https://revealjs.com) slide deck for the final
project, using the videos from `../videos/` and comparison plots generated
from `videos/METRICS.txt`.

## View

Videos in Chromium-based browsers only autoplay when served over HTTP, so run
a tiny static server from the repo root:

```bash
cd ..                                    # at repo root
python -m http.server 8000
# then open http://localhost:8000/presentation/
```

Navigation: `→ / ←` next / previous · `f` fullscreen · `s` speaker notes ·
`Esc` overview.

## Contents

```
presentation/
  index.html                     # reveal.js deck (CDN, no build step)
  plots/
    make_plots.py                # regenerates the PNGs from METRICS.txt
    trajectory_tracking.png      # RMSE / peak-err bar chart, 20-ep agg
    tube_violation.png           # 40.7% → 0% tube violations
    disturbance_recovery.png     # Phase-3 stabilize RMSE & max, log-y
    solve_cost.png               # compute cost of the filter
```

Regenerate plots whenever `videos/METRICS.txt` changes:

```bash
python presentation/plots/make_plots.py
```

## Videos referenced

All paths are relative to `presentation/index.html`:

| slide                      | video                                           |
| -------------------------- | ----------------------------------------------- |
| Problem (Slide 2)          | `../videos/ppo_7M_demo.mp4`                     |
| Trajectory side-by-side    | `../videos/ppo/ep00_seed0.mp4`, `../videos/ppo_mpc_blf/ep00_seed0.mp4` |
| Arm-fold disturbance       | `../videos/disturbance/arm_fold_ppo.mp4`, `../videos/disturbance/arm_fold_mpc_blf.mp4` |
| Wind-high disturbance      | `../videos/disturbance/wind_high_ppo.mp4`, `../videos/disturbance/wind_high_mpc_blf.mp4` |

If you move the `videos/` folder the deck will break; update the `src=` paths
in `index.html` accordingly.
