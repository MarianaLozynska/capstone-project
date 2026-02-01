# Week 1 Reflection & Week 2 Strategy

## Week 1 Approach

In Week 1, I used **manual analysis** to select next points:
- Analyzed scatter plots and PCA visualizations
- Identified best-known points from initial data
- Made intuitive decisions about exploration vs exploitation
- Selected points based on visual patterns

## Week 1 Results Summary

| Function | Strategy | Result | Analysis |
|----------|----------|--------|----------|
| 1 | Exploration | ✅ **Improved** (0 → 0.098) | Successfully found first non-zero signal |
| 2 | Exploitation | ❌ Worse (0.611 → 0.557) | Sampled away from true peak |
| 3 | Exploitation | ❌ Worse (-0.035 → -0.059) | Point was too far from optimum |
| 4 | Exploitation | ❌ Worse (-4.02 → -4.42) | Moved in wrong direction |
| 5 | Exploitation | ✅ **Improved** (1089 → 1232) | Excellent - climbing the peak |
| 6 | Exploitation | ✅ **Improved** (-0.714 → -0.592) | Good local refinement |
| 7 | Exploitation | ≈ Same (1.365 → 1.365) | Very close to peak, marginal difference |
| 8 | Exploitation | ≈ Same (9.598 → 9.586) | Near optimal, slight variation |

**Overall**: 3 improved, 3 worse, 2 unchanged

## What Went Well

1. ✅ **Strategic thinking was sound**
   - Correctly identified when to explore (Function 1)
   - Correctly identified when to exploit (Functions 2-8)

2. ✅ **Landscape understanding**
   - Recognized Function 1 had sparse, narrow peaks
   - Identified Function 5 as unimodal with dominant peak
   - Understood Functions 7-8 required fine-tuning

3. ✅ **Successful functions**
   - Function 1: Found meaningful signal where none existed
   - Functions 5-6: Made strong improvements through exploitation

## What Went Wrong

1. ❌ **Manual point selection was imprecise**
   - Functions 2, 3, 4 all moved away from optima
   - Visual intuition can be misleading, especially with PCA projections
   - No quantitative measure to guide decisions

2. ❌ **No uncertainty quantification**
   - Couldn't assess which regions were truly unexplored
   - No confidence estimates for predictions

3. ❌ **Inefficient sampling**
   - Points chosen without formal optimization
   - No balance between predicted value and uncertainty

## Week 2 Improvements: Bayesian Optimization

To address Week 1 limitations, Week 2 uses **Bayesian Optimization**:

### Key Differences

| Aspect | Week 1 (Manual) | Week 2 (Bayesian Optimization) |
|--------|----------------|-------------------------------|
| **Method** | Visual analysis, intuition | Gaussian Process + Acquisition Functions |
| **Uncertainty** | None | GP provides uncertainty estimates |
| **Optimization** | Manual selection | Quantitative optimization of acquisition function |
| **Exploration/Exploitation** | Qualitative judgment | Controlled via xi/kappa parameters |
| **Prediction** | Visual extrapolation | Statistical model with confidence intervals |

### How Bayesian Optimization Works

1. **Gaussian Process (GP)**:
   - Fits a probabilistic model to observed data
   - Predicts both mean (expected value) and uncertainty at unsampled points
   - Updates beliefs as more data is collected

2. **Acquisition Functions**:
   - **Expected Improvement (EI)**: Samples where improvement over current best is likely
   - **Upper Confidence Bound (UCB)**: Optimistically samples high-uncertainty regions
   - Balances exploitation (sampling near known good points) with exploration (sampling uncertain regions)

3. **Adaptive Strategy**:
   - **Function 1**: Low exploration (xi=0.001) - exploit the discovery
   - **Functions 2-4**: Moderate exploration (xi=0.01) - correct course back to better regions
   - **Functions 5-6**: Low exploration (xi=0.001) - continue climbing
   - **Functions 7-8**: UCB with low kappa (0.5) - fine refinement near peak

## Expected Week 2 Outcomes

Based on GP predictions:
- **Functions 2-4**: GP should guide back toward better regions identified in initial data
- **Functions 5-6**: Continued upward progress (GP predicts 1449 for Function 5!)
- **Functions 7-8**: Fine-tuning to squeeze marginal gains
- **Function 1**: Continued exploitation around [0.4, 0.4] region

## Key Lessons

1. **Manual analysis is valuable for understanding**, but formal optimization is needed for precise sampling
2. **Visual intuition can mislead**, especially in high dimensions or with PCA projections
3. **Uncertainty quantification is crucial** - knowing where we're uncertain guides better exploration
4. **Bayesian Optimization balances exploration and exploitation** through principled, quantitative methods
