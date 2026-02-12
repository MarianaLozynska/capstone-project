# Week 3 Reflection

## 1. How has your query strategy changed?

Week 1 used manual PCA projections to pick points visually — this lost information in higher dimensions. Week 2 switched to Bayesian Optimization with GP in native dimensionality, using EI and UCB with fixed parameters per function (6/8 improved).

Week 3 added **sensitivity analysis**: I vary each dimension independently from the best point and measure the GP gradient magnitude to identify which dimensions matter. I then **tune xi/kappa per function** based on the gradient pattern — for example, xi=0.1 for Function 1 (lost peak, needs exploration) vs xi=0.001 for Function 5 (steep dims already maxed, needs exploitation). I rely on model predictions but verify them with sensitivity analysis before committing.

## 2. How do you balance exploration vs exploitation?

I use sensitivity gradients to diagnose each function and set the balance accordingly:

- **Explore** (xi=0.05-0.1): Functions 1 and 3 — flat gradients + poor results mean they're stuck or lost
- **Moderate** (xi=0.01): Functions 2 and 4 — improving, keep balanced search near breakthrough points
- **Exploit** (xi=0.001): Functions 5 and 6 — Function 5's dims 3&4 have steep gradients (45.6, 98.9) and are at max, so I lock them to [0.9, 1.0] and only explore flat dims
- **Fine-tune** (UCB, kappa=0.5): Functions 7 and 8 — all flat + high values, near convergence

Sensitivity analysis also acts as a safety check: without constrained bounds, BO recommended moving Function 5's dim 4 from 1.0 to 0.42 (predicted 25 vs best 1688) and Function 4 far from its breakthrough (predicted -8.97). Constraining bounds prevented these destructive moves.

## 3. How would SVMs change your approach?

A **soft-margin SVM** could classify regions as "high" vs "low" performing, helping prune unpromising areas. A **kernel SVM** (RBF) would capture non-linear boundaries — like Function 5's radial pattern around dims 3-4.

However, SVMs don't provide uncertainty estimates like GPs do. My GP gives both prediction and confidence at every point, which EI needs to balance exploration and exploitation. SVMs could complement the GP — use SVM to identify promising regions, then run BO within them — but wouldn't replace it.

## 4. What limitations become apparent as data grows?

**Convergence warnings** appear for Functions 1 and 8. Function 1's length_scale hits the lower bound — the Matern kernel can't model its narrow peak, confirmed by both sensitivity gradients being exactly 0.0000. Function 8's dims 5 and 7 hit the upper bound, suggesting these dimensions are irrelevant.

The **anisotropic Matern kernel** learns per-dimension importance through length_scale, but with limited data (12-42 points) the estimates are noisy. GP also scales **O(n³)**, so fitting will slow as data grows over more weeks.

## 5. How does this prepare you to think like a data scientist?

This project teaches the core workflow: **observe, model, diagnose, adjust**. The one-query-per-week constraint made every decision count — like real scenarios where experiments cost time or money.

The most valuable lesson was adding **diagnostic tools** to verify model outputs. Sensitivity analysis caught bad BO recommendations for Functions 4 and 5 before I acted on them. This mirrors real practice — you don't blindly trust model outputs, you validate with additional analysis. The project also taught adaptive strategies: different functions needed different exploration levels, just like different models or datasets need different optimization approaches in production.
