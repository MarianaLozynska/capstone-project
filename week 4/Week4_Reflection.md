# Week 4 Reflection

## 1. Which inputs acted like support vectors?

Several points sit at boundaries between high and low-performing regions. Function 1's [0.408, 0.405] → 0.325 is surrounded by near-zero outputs (1e-79, 1e-46) just ±0.15 away — this single point defines the "good" region like a support vector. Function 5's dims 3&4 dropping below ~0.9 marks a hard boundary where output collapses from 7599 to ~25. Function 4's breakthrough region borders points at -32 nearby, with dim 3 being the steepest boundary direction (gradient 1.03).

Recognising these boundary points directly defines search bounds — they tell the acquisition function where the viable region ends, guiding tight constraints in Week 4.

## 2. Surrogate model and gradient exploration

I chose a **Gaussian Process** over a neural network because: (1) with 13–43 points per function, an NN would overfit; (2) GPs provide built-in uncertainty σ(x) that acquisition functions require; (3) the Matern kernel gives analytical gradients equivalent to backpropagation.

I explored output changes via **sensitivity analysis** — computing max|∇μ(x)| per dimension. Function 5's dims 3&4 showed gradients of 45.6 and 98.9, confirming they're critical and catching a bad BO recommendation to move dim 4 from 1.0 to 0.42. Function 4's dim 3 gradient (1.03) identified the most sensitive direction. Function 1's zero gradients diagnosed kernel failure. Function 2's dim2 gradient ≈ 0 revealed only dim1 matters for optimisation.

A key insight this week: the choice of acquisition function should be matched to **GP reliability**. EI and UCB are deterministic — given the same GP, they always converge to the same point. This works well when the GP is accurate (F1, F2, F4, F5), but fails when the GP is unreliable. For F3 (0/3 weeks improved) and F6 (W3 regressed), I switched to **Thompson Sampling (TS)**, which draws a random function from the GP posterior and optimises that instead. TS doesn't trust the GP mean surface — where the GP is uncertain, drawn functions vary wildly, producing natural exploration. For F3, I also increased the GP's alpha to 0.1 (regularisation) because the standard alpha=1e-6 was overfitting noise, producing a misleading surrogate surface.

## 3. Classification framing ('good' vs 'bad')

**Logistic regression** fits a linear boundary — works for F5 ("dims 3&4 high → good") but fails for F1's narrow spike. **SVM with RBF kernel** captures non-linear boundaries — for F1 it would learn a tight circle around the spike, with support vectors being nearest "bad" points. **Neural networks** could learn complex boundaries but would overfit at our data sizes.

The key trade-off: a conservative boundary (exploitation) is safe but misses unexplored areas; a liberal boundary (exploration) risks wasting queries. For BBO, queries near the decision boundary are most informative. My bound-constrained BO does this implicitly — sensitivity analysis defines the viable region, BO searches within it.

## 4. Most appropriate model type

The **GP** felt most appropriate. **Interpretability**: per-dimension length scales directly reveal feature importance; convergence warnings diagnosed F1's narrow peak and F8's irrelevant dims. **Uncertainty**: native σ(x) is essential for acquisition functions — SVM and NN lack this. **Flexibility trade-off**: the Matern kernel fails for F1's sharp spike, but with 13–43 points, NN overfitting risk outweighs flexibility gains.

I balanced this by using GP as the primary model with **acquisition strategies matched to GP reliability**: EI for functions where the GP guides well (F1 tight ±0.03, F2 dim1-only, F4 ±0.05, F5 push to corner), UCB for fine-tuning near-optimal functions (F7, F8 ±0.08), and Thompson Sampling for functions where the GP has failed to guide improvements (F3 with alpha=0.1, F6 ±0.10). The GP remains the surrogate in all cases — only the decision-making layer on top changed.

## 5. Steepest gradients and experiment prioritisation

GP sensitivity analysis at best points:

| Function | Steepest Dims | Flattest Dims |
|----------|--------------|---------------|
| F1 (2D) | Both ≈ 0 (kernel failure) | Both |
| F4 (4D) | Dim 3 (1.03) | Dims 1,2,4 |
| F5 (4D) | Dim 4 (98.9), Dim 3 (45.6) | Dims 1&2 |
| F6–F8 | All flat (max 0.065–0.088) | All |

Steep dims get locked (F5 dims 3&4 at 1.0). Flat dims get wider exploration (F5 dims 1&2). F2's dim2 is flat (gradient≈0), so only dim1 is constrained. All-flat + high value = fine-tune with UCB (F7, F8). GP failure (F3: 0/3 improved, F6: W3 regressed) → switch to Thompson Sampling which doesn't rely on the GP mean surface being accurate.

## 6. Classification boundary and backpropagation

I did not train an NN classifier. For 2D functions, an MLP could learn effective boundaries — e.g., a circular "good" region around F1's spike. Backpropagation would compute ∂P(good)/∂x, with gradients largest at the decision boundary, identifying the most informative query locations. For 5–8D functions, an NN would likely memorise rather than generalise with 20–43 points.

My GP sensitivity already provides equivalent information: ∇μ(x) indicates improvement directions, σ(x) identifies model uncertainty analogous to boundary regions.

## 7. Neural network vs simpler models

**Linear regression** fails for most functions — F5's exponential scaling and F1's spike are fundamentally non-linear. **The GP** captures non-linearity via the Matern kernel and guided improvements in 6/8 functions. Its length scales are directly interpretable as feature importance. **A neural network** would offer more flexibility for sharp transitions but at significant cost: overfitting with limited data, no native uncertainty for acquisition functions, and additional hyperparameter tuning.

The GP's moderate flexibility was the right trade-off — sufficient for effective optimisation, with interpretability and built-in uncertainty that an NN cannot match at this data scale.
