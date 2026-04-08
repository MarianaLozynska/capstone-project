# Model Card — BBO Optimisation Approach

## Overview

- **Name**: GP-Guided Hybrid Sequential Optimiser
- **Type**: Bayesian optimisation with manual overrides, informed by Gaussian Process sensitivity diagnostics
- **Version**: 3.0 (evolved through three phases over 10 weeks)

## Intended Use

- **Suitable for**: Expensive black-box function optimisation with extreme budget scarcity (1-10 queries per round); continuous inputs in bounded domains; problems where automated BO fails on narrow spikes or ultra-sensitive dimensions.
- **Avoid for**: Real-time optimisation; safety-critical systems; problems requiring convergence guarantees; categorical or discrete inputs; high-noise settings without replicate observations.

## Details — Strategy Evolution

**Phase 1 (Weeks 1-4): Automated BO**
Standard GP + EI/UCB acquisition functions. Sensitivity analysis introduced in Week 3 to identify per-dimension length scales. Thompson Sampling tested for unreliable GP fits (F3, F6). Key discovery: F5 optimum at corner [1,1,1,1]; F1's narrow spike (ls=0.008) defeats automated methods.

**Phase 2 (Weeks 5-7): Manual Override**
Shifted to GP-informed manual nudges after automation failed on 5/8 functions. Core technique: single-dimension perturbations sized at fractions of the kernel length scale. Introduced failed-strategy tracking (Week 7) and peer intelligence integration. Locked hyper-sensitive dimensions (F6 dims 4&5). Key success: F1 achieved 6 consecutive improvements through alternating-dimension +0.005 nudges.

**Phase 3 (Weeks 8-10): Experimental**
Broke from micro-nudge playbook after 8-week strategy audit. Fresh dimension exploration (F3 dim3, F8 dim3) produced F8's biggest gain since Week 4. Trust-region EI tested on F4 (catastrophic failure: predicted 1.086, observed -0.043 — EI permanently banned for F4). Week 10 introduced doubled step sizes (F1), ultra-sensitive dimension nudges (F6 dim4), multi-dim manual moves (F4), and unexplored region probes (F2, F5).

**Utility modules**: Custom Python library (`utils/`) providing `bayesian_optimization.py` (GP fitting, EI/UCB/TS acquisition), `sensitivity.py` (per-dimension gradient analysis), and `data_utils.py` (cumulative data management).

## Performance

| Function | Dims | Best Score | Best Week | Trajectory |
|----------|------|-----------|-----------|------------|
| F1 | 2D | 0.899 | W8 | Steady improvement W3-W8 (6 consecutive bests) |
| F2 | 2D | 0.614 | W2 | Stuck — 8 weeks without improvement |
| F3 | 3D | -0.006 | W4 | Plateau — marginal variation since W4 |
| F4 | 4D | 0.724 | W7 | Volatile — two EI catastrophes (W6, W9) |
| F5 | 4D | 8662 | W4 | Corner optimum found early, locked in |
| F6 | 5D | -0.521 | W6 | Stuck — noisy function limits progress |
| F7 | 6D | 1.870 | W9 | Steady improvement (4 consecutive bests W6-W9) |
| F8 | 8D | 9.775 | W9 | Fresh dimension breakthrough in W9 |

**Metrics**: Best observed output per function (maximisation). Weekly hit rate (% of functions improving). Cumulative running best tracked across all weeks.

**Overall hit rate**: ~51% of weekly queries improved over prior best (typical for 1-query BO).

## Assumptions and Limitations

**Assumptions**:
1. Functions are deterministic — violated by F6 (same input, different outputs in W2 vs W7).
2. GP kernel length scales reflect true sensitivity — uncertain with 19 points in 3-8D spaces.
3. Dimensional independence — all nudges tested one dim at a time (no interaction probing until W9-W10).
4. Stationarity — function landscapes do not change over time.

**Limitations**:
1. **Cannot distinguish noise from signal** with 1 query per week. Every inference ("this direction improved") assumes the observation is the true function value.
2. **Path dependency** — later queries depend on early results. If early exploration missed a promising region, the entire subsequent strategy drifts away from it (F2's 8-week stagnation in dim1 > 0.67).
3. **EI failure on non-smooth functions** — automated acquisition functions failed catastrophically on F1 (W2: 3.1e-39) and F4 (W9: -0.043) where the GP surface was unreliable.
4. **Dimensionality curse** — F8's 48 points in 8D gives ~1.6 points per dimension. The GP extrapolates in most of the space.

**Failure modes**: GP overconfidence on sparse data (F4 W9); multi-dim perturbation on narrow spikes (F1 W9 combo); exhaustive same-dimension nudges with diminishing returns (F6 dim2 for 4 weeks).

## Ethical Considerations

**Transparency**: All decisions are documented in weekly Jupyter notebooks with inline code comments explaining strategy, what each move avoids, and the evidence base. Cumulative .npz data files and weekly reflections provide a complete audit trail. A reviewer can re-run any notebook to verify GP predictions against actual outcomes.

**Reproducibility**: The approach is reproducible given the initial seed data, the utility module code, and the weekly strategy documentation. One gap: peer intelligence (F2 bimodality, F8 peer scores) came from discussion board posts and cannot be independently verified without access to those posts.

**Real-world adaptation**: The core lesson — that automated BO requires human diagnostic oversight, especially under budget scarcity — transfers directly to production ML. Blindly trusting acquisition functions without checking GP reliability led to our worst results (F4 W9). The hybrid approach of "automate where GP is reliable, manually override where it isn't" is a practical pattern for expensive real-world optimisation.
