# Datasheet — BBO Capstone Project Dataset

## Motivation

This dataset was created to support sequential black-box optimisation of eight unknown functions as part of the Imperial College ML/AI Capstone Project. The task is to maximise each function's output using one query per function per week over 10 weeks, with no access to gradients, closed-form expressions, or function structure. The dataset records all query-response pairs to enable surrogate modelling, strategy evaluation, and reproducibility analysis.

## Composition

- **Size**: 80 weekly queries (10 weeks x 8 functions) plus 10 initial seed points per function = ~160 total observations.
- **Format**: NumPy compressed archives (.npz) containing input arrays and output arrays per function. Weekly Jupyter notebooks (.ipynb) contain analysis code and strategy documentation.
- **Structure**:
  - Function 1: 2D inputs, scalar output (best: 0.899)
  - Function 2: 2D inputs, scalar output (best: 0.614)
  - Function 3: 3D inputs, scalar output (best: -0.006)
  - Function 4: 4D inputs, scalar output (best: 0.724)
  - Function 5: 4D inputs, scalar output (best: 8662)
  - Function 6: 5D inputs, scalar output (best: -0.521)
  - Function 7: 6D inputs, scalar output (best: 1.870)
  - Function 8: 8D inputs, scalar output (best: 9.775)
- **Input range**: All coordinates in [0, 1], specified to six decimal places.
- **Gaps**: F2 has a 47% gap in dim1 (0.20-0.68 unsampled). F8 has ~1.6 points per dimension — severe undersampling in 8D. F6 exhibits confirmed noise (identical inputs gave different outputs in W2 vs W7).

## Collection Process

- **Timeframe**: 10 weeks (January-April 2026), one query per function per week.
- **Strategy evolution**:
  - Weeks 1-3: Broad exploration using GP + EI/UCB acquisition functions with sensitivity analysis.
  - Weeks 4-5: Shift to manual nudges guided by GP kernel length scales after automated methods failed on narrow spikes (F1) and ultra-sensitive dimensions (F6).
  - Weeks 6-7: Lock-and-nudge strategy — freeze hyper-sensitive dimensions, single-dim micro-nudges on active dimensions. Peer intelligence integration (F2 bimodality, F8 performance gap).
  - Weeks 8-9: Fresh dimension exploration — switching to untested dimensions (F3 dim3, F8 dim3) based on 8-week strategy audit. Trust-region EI for F4 (failed catastrophically).
  - Week 10: Experimental strategies — doubled steps (F1), ultra-sensitive dim nudges (F6), multi-dim manual moves (F4), unexplored region probes (F2, F5).
- **Generation method**: Hybrid — GP-informed manual selection for most functions, automated EI/UCB for smooth landscapes (F4), manual exploration for stuck functions (F2, F5).
- **No human participants**: All data is function evaluations from a computational black-box oracle.

## Preprocessing and Uses

- **Transformations**: None applied to raw query-response pairs. GP surrogate models use Matern 2.5 kernel with ARD (automatic relevance determination) for per-dimension length scale estimation. No output warping or Box-Cox transforms were applied.
- **Intended uses**: Benchmarking BO strategies under extreme budget scarcity; studying exploration-exploitation trade-offs with 1-query-per-week constraints; analysing GP reliability on sparse high-dimensional data.
- **Inappropriate uses**: Training supervised models (insufficient samples); generalising to functions outside [0,1] bounds; assuming noise-free observations (F6 violates this).

## Distribution and Maintenance

- **Availability**: Public GitHub repository.
- **Format**: .npz (NumPy), .ipynb (Jupyter), .md (documentation).
- **Terms**: Educational use. Course-specific black-box functions are not redistributable.
- **Maintenance**: Single researcher. Dataset is static post-course — no further queries will be added after Week 10.
