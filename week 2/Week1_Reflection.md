# Week 1 Reflection & Week 2 Strategy

## Week 1 Results

Used **manual visual analysis** with PCA projections: 3 improved, 3 worse, 2 unchanged

**What went well:**
- Function 1: Found first signal through exploration
- Functions 5-6: Strong improvements via exploitation

**What went wrong:**
- Functions 2-4 moved away from optimum - PCA projections lost information in higher dimensions
- No uncertainty quantification
- Manual selection lacked precision

---

## Week 2 Strategy: Bayesian Optimization

### Why Bayesian Optimization?

Week 2 requires BO, which addresses Week 1's limitations:
1. Works in native dimensionality (3D-8D) without information loss
2. Quantifies uncertainty σ(x) at every point
3. Optimizes mathematically via acquisition functions

### How Decisions Are Made

**Gaussian Process**: Provides μ(x) prediction and σ(x) uncertainty

**Acquisition Functions**:
- **EI (Expected Improvement)**: xi = 0.001 (exploit), xi = 0.01 (moderate explore)
- **UCB (Upper Confidence Bound)**: kappa = 0.5 (gentle exploration)

### Function-Specific Strategies

| Functions | Strategy | Reasoning |
|-----------|----------|-----------|
| 2-4 | EI, xi=0.01 | Course correction - Week 1 failed, GP works in native space |
| 5-6 | EI, xi=0.001 | Continue climbing successful gradient |
| 7-8 | UCB, kappa=0.5 | Fine-tune near plateau |
| 1 | EI, xi=0.001 | Refine discovered peak |

### Decision Process

1. Fit GP to all data (initial + Week 1)
2. Calculate acquisition function for candidate points
3. Optimize using L-BFGS-B
4. Select point with highest value

### Expected Outcomes

**Functions 2-4**: GP should find better regions PCA missed
**Functions 5-6**: Continued climbing
**Functions 7-8**: Marginal gains or confirm convergence
**Function 1**: Better peak localization

---

## Key Lessons

1. Visual analysis useful for understanding, but mathematical optimization needed for precision
2. PCA projections mislead - native dimensionality avoids information loss
3. Uncertainty quantification crucial - GP's σ(x) distinguishes unexplored from unpromising
4. Adaptive strategies essential - different functions need different xi/kappa parameters
