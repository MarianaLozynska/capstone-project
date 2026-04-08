# Week 9 Reflection — Scaling Laws, Emergent Behaviours & Strategic Trade-offs

## How do scaling laws influence your current query choices? Do you see diminishing returns or steady improvements?

Scaling laws manifest differently across the eight functions. Function 1 shows the clearest power-law pattern: early gains were massive (W5: +0.196, W7: +0.173) but W8's dim1 nudge yielded only +0.021 — a 9x reduction. However, by combining both dimensions simultaneously in Week 9, I aim to exploit the additive effect of two individually productive directions rather than accepting single-dimension diminishing returns. This is a direct application of scaling insight: when one axis saturates, combine axes rather than pushing harder on either alone.

Function 8 follows a similar diminishing trajectory — improvements of 0.104 (W1-W2) down to 0.002 (W7-W8). However, a peer achieving 9.96 versus our 9.76 reveals that we are in a local plateau, not at the global ceiling. This informed my decision to switch from dim4 micro-nudges (converged) to a bolder dim3 exploration (+0.020). Scaling laws told me *where* returns were diminishing; peer intelligence told me the ceiling was higher than assumed.

Functions 2 and 3 show genuine stagnation — F2's best has been 0.614 since Week 2 despite 7 subsequent queries, and F3 has been within 0.002 of its best for 5 weeks. These represent the "flat region" of the scaling curve where additional data within the explored region provides negligible marginal value. My response: redirect queries to completely unexplored regions (F2's low-dim1 quadrant) or untested dimensions (F3's dim3).

## Where might emergent behaviours alter your expectations, and how are you preparing for them?

Emergent behaviour appeared most clearly in Function 6, where resubmitting the identical W2 input in W7 produced a dramatically different output (-0.521 versus -0.606). This revealed stochastic noise invisible from deterministic GP modelling — an emergent property only observable through repeated sampling. My preparation: I now treat F6 results with wider confidence intervals and choose precision nudges (dim2+0.001) rather than bold moves on ultra-sensitive dimensions.

Function 2's bimodality is another emergent feature — a peer discovered a second peak at 0.829 that our GP never predicted because all training data clustered around dim1=0.7. Charlie's analysis of EI boundary-hugging explains why automated acquisition functions failed to find it: the GP's uncertainty surface is highest at domain boundaries, causing the optimizer to propose corner points rather than exploring the mid-space where the second peak likely resides. My preparation: I override the optimizer with a manually chosen exploration point [0.20, 0.50], deliberately targeting the centre of the unexplored low-dim1 region.

Function 4's EI regression in Week 6 (0.710 to 0.466) was an emergent failure of automation — four consecutive improvements created false confidence in the optimizer. I now use trust-region bounds (plus or minus 0.03) to constrain the EI search space, treating the optimizer as a local search tool rather than a global oracle.

## What trade-offs between cost, robustness and performance are shaping your strategy now?

**Cost allocation**: With limited queries remaining, I split the portfolio into three tiers. High-confidence exploitation (F1, F7: proven strategies, expected new bests), moderate-risk fresh exploration (F3, F4, F6, F8: new dimensions or approaches with GP-supported predictions), and strategic information gathering (F2, F5: unlikely to set new bests but providing landscape knowledge for potential future use).

**Robustness versus performance**: For F7, I chose robustness — reverting to the proven dim2 decrease rather than experimenting with dim5. Three consecutive bests represent the most reliable improvement trajectory in the project; risking it on an untested dimension prioritises novelty over results. Conversely, for F8, I chose performance risk — the 0.2 gap to a peer's score justifies a bolder dim3 move (+0.020 instead of +0.010) despite higher uncertainty, because conservative moves cannot close that gap.

**Computational cost versus decision quality**: Charlie's observation about ensemble optimizers averaging to "uninformative midpoints" in F8's multi-region landscape resonates with my experience. Rather than running multiple acquisition functions and averaging, I chose a single EI with tight trust-region bounds for F4 — simpler, faster, and avoiding the averaging trap.

## How do you balance predictable optimisation with the risk of sudden but uneven emergent capabilities?

My portfolio strategy explicitly separates these concerns. Functions 1 and 7 receive predictable optimisation — proven directions, small steps, high confidence. These are the "base return" of the portfolio. Functions 2, 3, and 8 receive exploratory allocations targeting potential emergent gains — untested dimensions, unexplored regions, and peer-informed bold moves. The asymmetry is deliberate: predictable functions get conservative treatment to protect accumulated gains, while stagnant functions get aggressive treatment because their downside is already priced in (the best score is locked regardless of this week's result).

The key lesson from 9 weeks is that emergent capabilities in black-box functions are not random — they are revealed by systematic exploration of untested dimensions and regions. F1's massive W7 gain (+0.173) emerged not from a lucky guess but from the disciplined alternating-dimension strategy that happened to cross a steep gradient. Similarly, F4's breakthrough from -4.416 to 0.724 over weeks resulted from systematic narrowing of the search space. The balance is not between prediction and luck, but between exploiting known structure and investing in revealing unknown structure — a distinction that maps directly to the exploration-exploitation trade-off governing all Bayesian optimisation.
