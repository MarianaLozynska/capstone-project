# Bayesian Optimization Capstone Project

> GP-Guided Hybrid Sequential Optimiser for Black-Box Functions

**[Model Card](BBO_Model_Card.md)** | **[Datasheet](BBO_Datasheet.md)** | **[Presentation](BBO_Presentation.md)** | **[Final Reflection](Final_Project_Reflection.md)**

---

## In Plain Language

Imagine you have eight invisible hills and you have to find the highest point on each one. You cannot see the hills, you cannot ask for hints, and you can only take one step a week for thirteen weeks. Each step gives you a single number — the height where you landed. That is the puzzle this project solves. I built a system that learns the shape of each hill from where I have already stepped, suggests where to step next, and remembers every wrong turn so I do not repeat it. Five of the eight hills now show clear improvement from where I started, and one was solved completely.

---

## Overview

This project tackles the **Black-Box Optimization (BBO) challenge**: maximising eight unknown functions (2D to 8D) with no access to equations, derivatives, or internal structure. One query per function per week over 13 weeks, using Gaussian Process surrogate models and acquisition functions to balance exploration and exploitation.

## Results

![Score progression across 13 weeks](visuals/score_progression.png)

| Function | Dims | Initial | Best Score | Best Week | Notes |
|----------|------|---------|-----------|-----------|-------|
| 1 | 2D | 0.098 | **0.965** | W10 | Narrow spike; alternating-dim +0.005 nudges plus one +0.010 |
| 2 | 2D | 0.557 | **0.614** | W2 | Single peak found early, no second peak found later |
| 3 | 3D | -0.059 | **-0.0046** | W10 | Plateau near optimum; broken by dim3 increase |
| 4 | 4D | -4.42 | **0.724** | W7 | Volatile — EI failed twice; manual nudges held |
| 5 | 4D | 1232 | **8662** | W4 | Corner optimum [1,1,1,1]; symmetry confirmed |
| 6 | 5D | -0.59 | **-0.513** | W12 | Noisy function; broken by dim5 nudge after 5-week plateau |
| 7 | 6D | 1.36 | **1.890** | W13 | **8 consecutive bests** via dim2 decrease |
| 8 | 8D | 9.59 | **9.799** | W12 | 4 consecutive wins via dim3 increase (fresh dimension) |

## Technical Approach

- **Surrogate model**: Gaussian Process with Matern 2.5 kernel and automatic relevance determination (ARD)
- **Acquisition functions**: Expected Improvement (EI), Upper Confidence Bound (UCB), Thompson Sampling
- **Diagnostics**: Per-dimension sensitivity analysis via GP kernel length scales
- **Strategy**: Hybrid — automated BO where GP is reliable, manual GP-informed nudges where it is not
- **Failure tracking**: All previous submissions verified against each new proposal (~100 by Week 13)

### Strategy Evolution

| Phase | Weeks | Approach |
|-------|-------|----------|
| Automated BO | 1-4 | PCA exploration, then GP + EI/UCB/Thompson Sampling with sensitivity-driven tuning |
| Manual Override | 5-7 | Single-dim nudges guided by length scales, locked hyper-sensitive dims, peer intelligence |
| Experimental | 8-10 | Fresh dimension exploration, trust-region EI, bold step sizes, ultra-sensitive probes |
| Defensive Refinement | 11-12 | Halved step sizes after overshoots, peer-informed exploration |
| Bold Final Round | 13 | No resubmits, untested moves only, max-across-weeks scoring exploited |

## Project Structure

```
├── README.md                       # This file
├── BBO_Model_Card.md               # Model card
├── BBO_Datasheet.md                # Dataset documentation
├── BBO_Presentation.md             # Capstone presentation
├── Final_Project_Reflection.md     # End-of-project reflection
├── Successful_Strategies_Reflection.md  # Strategy analysis
├── requirements.txt
├── utils/                          # GP fitting, sensitivity analysis, data management
├── initial_data/                   # Seed data for all 8 functions
├── visuals/                        # Score progression chart
└── week 1/ ... week 13/            # Weekly notebooks, data (.npz), and reflections
```

Each weekly folder contains:
- `weekN_function_analysis.ipynb` — analysis notebook with strategy and recommendations
- `weekN_clean_data.npz` — cumulative dataset for the next week
- `WeekN_Reflection.md` — strategy rationale and lessons learned (where applicable)

## Getting Started

```bash
pip install -r requirements.txt
cd "week 13"
jupyter notebook week13_function_analysis.ipynb
```

Each notebook loads the previous week's data, adds results, and generates next-week recommendations.

## Tools

Python 3.12 | scikit-learn | SciPy | NumPy | Matplotlib
