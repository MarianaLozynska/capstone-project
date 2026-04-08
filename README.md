# Bayesian Optimization Capstone Project

> GP-Guided Hybrid Sequential Optimiser for Black-Box Functions

**[Model Card](BBO_Model_Card.md)** | **[BBO Datasheet](BBO_Datasheet.md)**

---

## Overview

This project tackles the **Black-Box Optimization (BBO) challenge**: maximising eight unknown functions (2D to 8D) with no access to equations, derivatives, or internal structure. One query per function per week over 10 weeks, using Gaussian Process surrogate models and acquisition functions to balance exploration and exploitation.

## Results

| Function | Dims | Best Score | Best Week | Trend |
|----------|------|-----------|-----------|-------|
| 1 | 2D | 0.899 | W8 | 6 consecutive improvements via alternating-dim nudges |
| 2 | 2D | 0.614 | W2 | Exploring for second peak (peer found 0.829) |
| 3 | 3D | -0.006 | W4 | Plateau — dim1 bracketed, dim3 in progress |
| 4 | 4D | 0.724 | W7 | Volatile — EI failed twice, manual nudges work |
| 5 | 4D | 8662 | W4 | Corner optimum [1,1,1,1] locked in |
| 6 | 5D | -0.521 | W6 | Noisy function — trying ultra-sensitive dims |
| 7 | 6D | 1.870 | W9 | 4 consecutive bests via dim2 decrease |
| 8 | 8D | 9.775 | W9 | Fresh dim3 breakthrough — closing peer gap |

## Technical Approach

- **Surrogate model**: Gaussian Process with Matern 2.5 kernel and automatic relevance determination (ARD)
- **Acquisition functions**: Expected Improvement (EI), Upper Confidence Bound (UCB), Thompson Sampling
- **Diagnostics**: Per-dimension sensitivity analysis via GP kernel length scales
- **Strategy**: Hybrid — automated BO where GP is reliable, manual GP-informed nudges where it is not

### Strategy Evolution

| Phase | Weeks | Approach |
|-------|-------|----------|
| Automated BO | 1-4 | PCA exploration, then GP + EI/UCB/TS with sensitivity-driven tuning |
| Manual Override | 5-7 | Single-dim nudges guided by length scales, locked hyper-sensitive dims, peer intelligence |
| Experimental | 8-10 | Fresh dimension exploration, trust-region EI, bold step sizes, ultra-sensitive probes |

## Project Structure

```
├── BBO_Model_Card.md         # Model card
├── BBO_Datasheet.md          # Dataset documentation
├── utils/                    # GP fitting, sensitivity analysis, data management
├── initial_data/             # Seed data for all 8 functions
└── week 1/ ... week 10/      # Weekly notebooks, data (.npz), and reflections
```

## Getting Started

```bash
pip install -r requirements.txt
cd "week 10"
jupyter notebook week10_function_analysis.ipynb
```

Each notebook loads the previous week's data, adds results, and generates recommendations.

## Tools

Python 3.12 | scikit-learn | SciPy | NumPy | Matplotlib
