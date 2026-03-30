# Week 7 Reflection — Hyperparameter Tuning in BBO Strategy

## Which hyperparameters did you choose to tune, and why did you prioritise them?

I tuned three categories. First, **GP kernel length scales** (ARD Matern 2.5) — these reveal which dimensions matter and drive all downstream decisions. Function 6's dims 4 and 5 have length scales of 0.001 and 0.0003, so tiny movements cause large output changes. Function 4's length scales (~1.5–1.7) indicate a smooth landscape safe for broader moves.

Second, **acquisition function parameters**: xi for EI and kappa for UCB. With one query per week, overly exploratory settings waste rounds. For Week 7, I increased F2's xi to 0.5 after a classmate revealed a second peak at 0.829 — far above our 0.614 — confirming four weeks of micro-nudges had been exploiting the wrong peak.

Third, **search bounds per function** — radii around the best point that prevent the optimizer from proposing points outside the GP's reliable range.

## How has hyperparameter tuning changed your query strategy compared to earlier rounds?

In Weeks 2–3, every function got uniform treatment. By Week 5, sensitivity analysis split functions into two groups: GP-reliable (optimizer-driven) and GP-unreliable (manual nudges). Week 6 confirmed this — three new bests came from manual strategies, while F4's EI regressed from 0.710 to 0.466.

Week 7 adds two new elements: **peer results analysis** (F2's second peak discovery) and **failed strategy elimination** — systematically tracking what failed per function and ensuring it is not repeated. For example, F2's four consecutive micro-nudge failures triggered a switch to wide exploration. For F5, recognising that resubmitting [1,1,1,1] wastes a query led me to explore an untested corner [0,0,1,1] instead — the best score is already locked in.

## Which tuning method(s) did you apply, and what trade-offs did you notice?

I used **Bayesian optimisation** (GP + EI) for F2 with high exploration, and **manual adjustment informed by GP diagnostics** for the remaining seven functions. The manual approach reads length scales and sensitivity rankings, then makes single-dimension nudges sized at fractions of the relevant length scale.

Manual tuning captures domain knowledge that optimizers cannot encode — like "F6 dims 4 and 5 must be locked" or "F4 dim3 should keep decreasing." But it does not scale and requires reviewing diagnostics for each function weekly. The key trade-off: F4's automated EI produced four consecutive improvements before failing, showing automation works on smooth landscapes but breaks when GP uncertainty estimates are miscalibrated.

## As your data set grows to 16 points, what limitations become clearer?

**Sparse coverage in high dimensions**: F8's 8 dimensions with 16 points means the GP extrapolates aggressively — results have plateaued around 9.74. **Irrelevant features**: F8's dims 6 and 8 have length scales above 10,000 but the GP still fits them. **Diminishing returns**: F6's dim2 nudges improved by only 0.0003 in Week 6. **GP overconfidence**: F1's short length scales (0.012, 0.008) mean the function changes faster than the GP can model with 16 samples.

## How might you apply hyperparameter tuning techniques to larger data sets?

With more data: **LML-based kernel selection** (fit RBF, Matern 1.5, Matern 2.5 per function, pick best), **ensemble acquisition** (run EI/UCB/PI and take the centroid — would have prevented F4's regression), and **output warping** for functions like F5 with outputs in the thousands. For complex models, frameworks like Optuna or BoTorch automate this — but the logic is identical to my capstone workflow: fit a surrogate, identify which parameters matter, allocate queries where expected improvement is highest.

## How does tuning in this black-box set-up prepare you for professional ML/AI practice?

This project enforces **extreme budget scarcity** — one query per week, no validation set, no re-run. That forced habits transferable to production ML: diagnosing model reliability before trusting suggestions, recognising when automation fails and manual intervention is justified, and knowing when to stop tuning. F5's optimum was found in Week 4. Weeks 5–6 wasted queries on micro-deviations. For Week 7, I redirected that query toward exploring an untested corner — gaining information at zero cost to our best result. In production, that is reallocating compute from a converged model toward exploring alternatives.
