# BBO Capstone Project — Final Reflection

## Initial codebase

I built my codebase from scratch using `scikit-learn` as the foundation, rather than starting from a public BBO library like BoTorch or HEBO. This was a deliberate choice. I wanted to understand exactly what each component was doing — the kernel, the acquisition functions, the optimisation routine — and the easiest way to learn that was to write the wrappers myself.

The code is organised into a small custom utility package:

- `utils/bayesian_optimization.py` — GP fitting with Matern 2.5 kernel and ARD, plus Expected Improvement, Upper Confidence Bound and Thompson Sampling acquisition functions
- `utils/sensitivity.py` — per-dimension gradient analysis using the GP surrogate
- `utils/data_utils.py` — cumulative data management with `.npz` persistence

This setup was simple enough to reason about and rich enough to do real Bayesian Optimisation. The repo is public at [github.com/MarianaLozynska/capstone-project](https://github.com/MarianaLozynska/capstone-project) and includes weekly notebooks, reflections, a Datasheet, a Model Card and a final Presentation.

## Code modification

The codebase itself stayed fairly stable after Week 4 — the bigger changes were in how I used it.

**Weeks 1-3** were the "automation phase". Week 1 was PCA visual exploration to choose initial query points; from Week 2 I switched to GP + EI/UCB. Week 3 added the sensitivity analysis module, which extracted per-dimension length scales from the fitted GP. That diagnostic became the most important tool in the project.

**Week 4** added Thompson Sampling for functions where the GP appeared unreliable. TS produced F3's first meaningful improvement that week (-0.0056), which remained F3's best until Week 10. That was the last real algorithmic addition.

**Weeks 5-7** were where my approach changed most. After F1 crashed to 3.1e-39 in Week 2 and F4's EI catastrophically regressed in Week 6 (0.71 → 0.47), I stopped trusting automated acquisition functions on non-smooth functions. I started making manual decisions — single-dimension nudges sized to a fraction of the kernel length scale. This single change produced F1's run of consecutive improvements from Week 5 through Week 8 (0.611 → 0.899), and is responsible for most of my best scores.

Week 7 added a per-function failed-strategy log, which became my equivalent of an experience replay buffer. Before every weekly query I would cross-reference my proposed move against the log of every previous failure to make sure I was not repeating a known mistake.

**Weeks 8-10** were experimental. After diminishing returns set in around Week 8, I started doubling step sizes where gains were still positive (F1's dim2+0.010 in Week 10 produced a +0.066 gain — significant in the late-stage micro-nudge regime) and exploring previously untested dimensions on stuck functions (F8's dim3 breakthrough in Week 9 added +0.012 after weeks of dim4 stagnation). Both moves opened up improvements that conservative approaches had missed.

**Weeks 11-13** were defensive but with one bold late call. After Week 11's overshoot lesson on F1, I halved step sizes on most functions. But for the final week I went back to bolder steps after recognising the scoring system: max-across-weeks means a missed query does not hurt your maximum, so bolder moves are strictly better. That decision paid off on F7 — going back to dim2-0.005 (instead of halving to -0.002) gave +0.007, nearly double the typical late-stage gain of +0.004.

The single change with the most impact was the shift to manual GP-informed decisions in Week 5. Without that, F1 would still be at around 0.4 instead of 0.965, and my F4 would have continued regressing every time the EI got confident.

## Final result and what I would do differently

My final maximums across thirteen weeks:

| Function | Initial | Final | Best week |
|----------|---------|-------|-----------|
| F1 | 0.098 | **0.965** | W10 |
| F2 | 0.557 | **0.614** | W2 |
| F3 | -0.059 | **-0.005** | W10 |
| F4 | -4.42 | **0.724** | W7 |
| F5 | 1232 | **8662** | W4 |
| F6 | -0.59 | **-0.513** | W12 |
| F7 | 1.36 | **1.890** | W13 |
| F8 | 9.59 | **9.799** | W12 |

Five of the eight functions improved meaningfully. F2 was stuck at its Week 2 best for the remaining eleven weeks. F3 plateaued near -0.005 from Week 4 onward. F6 plateaued at -0.521 from Week 6 through Week 11 before breaking through in Week 12.

The final weeks were uneven. Week 12 produced three new bests (F6, F7, F8) — the F6 result broke a five-week plateau (Week 6 best had not been beaten since) via a tiny nudge on dimension 5, the only sensitive dimension I had never tested. Week 13 was bolder by design and produced one new best (F7's 8th consecutive new best at +0.007 — nearly double the typical late-stage gain of +0.004). Several Week 13 queries missed but did not damage my maximums under the scoring rules.

If I had a fresh start, I would change three things. First, I would shift to manual GP-informed decisions earlier — by Week 4 rather than Week 5 — once the F1 crash showed me the GP could not handle narrow spikes. Second, I would explore more aggressively on stuck functions in the middle weeks rather than continuing micro-nudges; F2's eleven-week stagnation was largely my fault for keeping queries near the known peak when the GP was clearly telling me there was no further structure to find there. Third, I would test the "fresh dimension" trick earlier on F6 — that Week 12 breakthrough on dimension 5 was the only sensitive dimension I had never touched, and it was sitting there waiting to be tried for weeks.

## Trade-offs and decisions

The biggest trade-off was between trusting the surrogate model and trusting my own diagnostic reasoning. The Gaussian Process is mathematically elegant and gives uncertainty estimates, but on functions like F1 (a narrow spike) and F4 (smooth but with sparse data), it produced confidently wrong predictions. The hardest decisions were the ones where I had to ignore an EI suggestion that looked reasonable in favour of a manual choice that felt riskier. After F4's two EI catastrophes (Week 6: 0.71 → 0.47, and Week 9: 0.72 → -0.04), I banned automation entirely on that function — a decision that initially felt unscientific but turned out to be correct.

The exploration-exploitation balance shifted across the project. Weeks 1-4 were exploration-heavy because the surrogate knew almost nothing. Weeks 5-10 were mostly exploitation, with F1, F7 and F8 in clear winning trajectories. Weeks 11-13 were defensive on the protected functions but bold on those still showing slope.

The other significant trade-off was per-function attention. With one query per function per week, I had to ration my analysis time. Functions with clear progress (F1, F7) got the most thinking; stuck functions like F2 and F6 got proportionally less because I had run out of ideas. In hindsight that was the wrong allocation — F6's late breakthrough showed that stuck functions sometimes need more careful thought, not less.

## Learning and application

The most important lesson I take from this project is that automated optimisation tools work brilliantly when their assumptions hold and fail silently when they do not. A Gaussian Process with a well-chosen kernel will outperform a human on a smooth, well-sampled function. But it will produce confident nonsense on a narrow spike or a noisy function, and the only signal that something is wrong is the next observation. The hybrid pattern I ended up with — automate where the diagnostics support it, override manually where they do not — is exactly what production ML engineers do for hyperparameter tuning, AutoML pipelines and any expensive-evaluation problem. I expect to use this pattern in future ML work whenever evaluations are costly and the function class is unknown.

A second lesson is the value of writing things down. My weekly reflections and per-function failure logs were the difference between learning from past mistakes and repeating them. By the final weeks I had a 96+ submission history I could cross-reference instantly. Without that documentation I would have lost track of which directions had failed, and several of my Week 13 decisions would have been blind guesses rather than informed ones.

What surprised me most was how differently the eight functions behaved despite all being treated the same way at the start. F1 was a narrow spike where steps under one length scale always worked; F5 was a corner-optimum function where any deviation lost points; F6 was stochastic and produced different outputs for identical inputs. Treating them all with one strategy was always going to fail. The realisation that each function has its own personality — its own dimensionality, its own length scales, its own noise profile — shifted my entire approach by Week 5.

I was also surprised by how much my peers' insights helped, even though we could not share specific coordinates. When a classmate reported their F2 had a second peak at value 0.829, I could not use their location, but their existence claim shifted my prior and prompted exploration I would not have otherwise tried. This mirrors how research communities actually work — knowing that something is possible is itself useful information, separate from how to do it. My own peer insights came back too late in some cases (F2 was probably unfindable for me by Week 11), but the pattern of treating peers as a noisy oracle was instructive.

In real-world ML, the same constraints apply: every experiment is expensive, every choice is irreversible, and the difference between competence and luck is whether you can articulate why you chose what you chose, recognise when your tools are failing, and update your priors when new information arrives. I am leaving this project with a clearer sense of what those skills feel like in practice — and a documentation habit I intend to keep.
