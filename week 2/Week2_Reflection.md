# Week 2 Reflection & Week 3 Strategy

## Week 2 Results

Used **Bayesian Optimization with Gaussian Processes**: 6 improved, 2 worse

**What went well:**
- Function 4: Breakthrough! (-4.03 → 0.352) - GP found optimal region in 4D space
- Functions 2, 5, 6, 7, 8: Steady improvements via adaptive EI/UCB strategies
- Native dimensionality eliminated PCA information loss from Week 1

**What went wrong:**
- Function 1: Lost narrow peak (0.098 → ~0) - exploitation too aggressive
- Function 3: Stuck on plateau (-0.07 → -0.03) - exploration insufficient

---

## Week 3 Strategy: Sensitivity Analysis

### Why Sensitivity Analysis?

Week 2's mixed results revealed a critical gap: **we don't know which dimensions matter**. Bayesian Optimization optimizes acquisition functions mathematically, but without understanding feature importance, we risk:
- Over-exploring unimportant dimensions
- Under-exploring critical ones
- Misinterpreting plateau vs convergence

Sensitivity analysis addresses this by:
1. Varying each dimension independently from best point
2. Measuring GP gradient magnitude (∇μ(x)) to quantify importance
3. Adjusting exploration parameters based on gradient patterns

### How Decisions Are Made

**Sensitivity Analysis Process:**
1. Fit GP to all data (initial + Week 1 + Week 2)
2. Use best point as baseline
3. For each dimension d: vary xd ∈ [0,1] while holding others constant
4. Calculate max gradient: `max(|∇μ(x)|)`
5. Interpret:
   - **Steep gradient (>1)**: Important feature, explore carefully
   - **Flat gradient (<0.5)**: Either converged or stuck - needs diagnosis

**Strategy Adjustment Rules:**

| Gradient Pattern | Interpretation | Strategy |
|------------------|----------------|----------|
| All flat + worsening | Lost peak | High exploration (xi=0.1) |
| All flat + plateau | Stuck | Increased exploration (xi=0.05) |
| Balanced + improving | Working | Keep strategy (xi=0.01) |
| Steep + at bounds | Critical dims maxed | Exploitation (xi=0.001) |
| All flat + high value | Near optimum | Fine-tune (UCB, kappa=0.5) |

### Function-Specific Week 3 Strategies

| Function | Week 2 Result | Gradient Pattern | Week 3 Strategy | Reasoning |
|----------|---------------|------------------|-----------------|-----------|
| 1 | Lost peak (→0) | Flat (0.008, 0.011) | EI, xi=0.1 | High exploration - relocate lost narrow peak |
| 2 | Improved (+0.6) | Flat but rising | EI, xi=0.01 | Moderate exploration - working |
| 3 | Stuck (-0.03) | All flat (~0.0) | EI, xi=0.05 | Increased exploration - escape plateau |
| 4 | Breakthrough (+4.4) | Balanced | EI, xi=0.01 | Keep strategy - breakthrough working |
| 5 | Strong (+1688) | Dims 3&4 steep (45.6, 98.9) | EI, xi=0.001 | Exploit - critical dims maxed |
| 6 | Improved (-0.52) | All flat | EI, xi=0.001 | Exploit - near plateau |
| 7 | Improved (+1.78) | All flat | UCB, kappa=0.5 | Fine-tune - near convergence |
| 8 | Improved (+9.65) | All flat | UCB, kappa=0.5 | Fine-tune - near convergence |

### Decision Process

1. **Run sensitivity analysis** on all 8 functions
2. **Identify gradient patterns**:
   - Functions 1&3: Flat + poor performance → increase exploration
   - Function 4: Balanced + breakthrough → maintain
   - Function 5: Steep dims 3&4 → exploit (already at max)
   - Functions 6-8: Flat + high values → fine-tune
3. **Adjust xi/kappa** based on diagnosis
4. **Generate recommendations** using adjusted acquisition functions
5. **Validate** via GP predictions with uncertainty

### Expected Outcomes

**Functions 1&3**: High exploration should relocate peaks or escape plateaus
**Function 4**: Continued improvement with balanced strategy
**Function 5**: Optimize dims 1&2 while keeping 3&4 maxed
**Functions 6-8**: Marginal gains or confirm convergence

---

## Key Lessons

1. **BO success depends on right exploration level** - Week 2's fixed strategies worked for some functions but failed for others
2. **Sensitivity analysis enables adaptive strategies** - gradient patterns reveal whether flat = stuck or converged
3. **Feature importance guides parameter tuning** - steep gradients need careful exploration; flat gradients need diagnosis
4. **Data-driven adjustments outperform fixed rules** - analyzing actual GP behavior beats one-size-fits-all parameters

---

## Week 3 Recommendations

Based on sensitivity analysis and adjusted strategies:

```
Function 1: 0.074136-0.522573
Function 2: 0.944327-0.539616
Function 3: 0.868014-0.000000-0.000000
Function 4: 0.394731-0.386792-0.430560-0.420751
Function 5: 0.892638-0.105745-0.847216-0.970375
Function 6: 0.997121-0.195367-0.734602-0.561939-0.155898
Function 7: 0.000000-0.239509-0.991084-0.103951-0.353088-0.883735
Function 8: 0.234779-0.082798-0.162727-0.159577-1.000000-0.376083-0.261560-0.962760
```
