# Week 12 Reflection — Strategy Evolution & Variance Analysis

## How has your optimisation strategy evolved since your first few rounds of queries? Which elements now feel more structured or systematic?

The biggest change is that I no longer treat the eight functions uniformly. In the first three weeks, I ran the same Bayesian Optimisation setup on all of them, varying only the exploration parameters. By now each function has its own documented playbook, its own list of proven and failed strategies, and its own decision rule for when to exploit versus explore.

What feels most systematic is the diagnostic routine I run every week before choosing any query. I fit a Gaussian Process per function, extract the per-dimension length scales, and compare them against the step size I am considering. Any step larger than one length scale on a hyper-sensitive dimension is automatically rejected. Any direction that has already failed is cross-referenced against my 96-submission history log before being tried again. This stops me from repeating mistakes I've already made, which was a real problem in the first five weeks.

The other systematic element is what I think of as the "confirm, continue, question" rule. If last week's move produced a new best, I continue the same direction with the same step. If it regressed slightly, I halve the step. If it regressed badly, I reverse direction. Applied consistently, this has produced F7's six consecutive improvements and F8's three consecutive dim3 breakthroughs.

## If you think of your current data set as a 'high-dimensional' space, which variables or behaviours seem to drive the largest variation in your results?

Analogous to PCA identifying the highest-variance directions, the two variables that explain most of the variation in my results are: **the GP kernel length scale per dimension**, and **the step size relative to that length scale**. Together, these two quantities have predicted success or failure better than any other diagnostic.

When length scales are small (under 0.01, as on F1 and F6's dims 4 and 5), the function changes rapidly with tiny input movements. Any step larger than one length scale usually produces a catastrophic regression. When length scales are large (over 10, as on F8's dims 6 and 8), the dimension is effectively irrelevant — moving it in any direction produces near-zero change.

The third most influential variable is whether I am using single-dimension or multi-dimensional moves. My data shows multi-dim moves regressed in three out of four attempts (F1 Week 9 combo, F4 Week 9 EI, F4 Week 10 two-dim), while single-dim moves have a much higher success rate. This is effectively another "principal component" of my results.

## How do you decide which aspects of your strategy to keep exploring versus which to reduce or simplify?

My rule parallels PCA's variance threshold: I keep what produces meaningful variance in outcomes, and drop what does not.

Strategies I've kept: GP sensitivity analysis, failed-strategy tracking, single-dimension nudges, and alternating-dimension patterns on narrow-spike functions. These have produced nearly all of my new bests.

Strategies I've dropped: Expected Improvement on non-smooth functions (banned after two F4 catastrophes), Thompson Sampling on sparse data (underperformed deterministic methods), uniform per-function parameters (replaced with per-function tuning), and multi-dim moves on narrow spikes.

The simplification has been deliberate. Early weeks had an elaborate acquisition function selection logic that varied xi and kappa across functions. By Week 7 I realised most of that complexity did not change outcomes — the real decision was whether to use automation at all, and whether the step size respected the length scale. Everything else was noise. Reducing the strategy to these two core decisions has made the weekly process faster and the results more predictable.

## How might this round of optimisation influence your next and final round of query submission?

Week 12 is the second-to-last round, so Week 13 becomes my last chance to produce new bests. This changes how I balance exploration and exploitation.

For functions that are clearly still improving (F1, F7, F8), I will continue exploitation with the same proven steps — no point changing a winning strategy on the last attempt.

For F3, which has been showing small consistent gains, I will continue the halved-step approach.

For the stuck functions (F2, F4, F5, F6), Week 13 is a decision point. I will either commit to the last remaining unexplored option (for F2, a different region; for F6, the last sensitive dimension), or accept that the current best is near-optimal and resubmit the anchor point.

The broader shift is that Week 13 has to be defensive. I cannot afford a W11-style overshoot on a critical function. Tiny safe steps on productive directions, and accepting "no improvement" on truly stuck functions, is the right trade-off when there are no further rounds to recover from mistakes.

## Reflect briefly on how insights from PCA might apply to how you interpret your BBO results

PCA's core idea — that most of the variance in a high-dimensional dataset can be captured by a small number of principal directions — applies directly to how I now think about these functions.

For most of my functions, the GP has already revealed that only a few dimensions actually matter. F2's length scales show dim2 is essentially irrelevant (ls=1.015). F8's dim6 and dim8 are effectively locked with ls > 10,000. F6's dims 4 and 5 are hyper-sensitive while dims 1 and 3 are flat. In each case, the true "principal components" of the function's behaviour are a subset of its input dimensions, and my strategy should focus queries along those directions exclusively.

The analogy to redundancy removal is also useful. F5 turned out to be symmetric between (dim1, dim2) and (dim3, dim4) pairs — meaning dim1 and dim2 encode similar information, as do dim3 and dim4. This is analogous to two highly correlated features in PCA that get collapsed into a single principal component. Recognising this symmetry let me stop wasting queries exploring redundant regions.

More broadly, treating my query history as data in a high-dimensional space helps me avoid overfitting to individual observations. F6's noise (identical inputs producing different outputs in W2 and W7) is like low-variance noise in PCA that should be filtered rather than modelled. The interpretive shift is to stop treating every query as signal and start asking whether the apparent pattern is actually one of the top few principal directions or just dimension-specific noise.
