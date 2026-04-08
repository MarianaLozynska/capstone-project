# Week 10 Reflection

## What reasoning guided your submission? How did patterns from previous rounds influence your decisions?

Week 10 shifts from conservative micro-nudges to experimental strategies, driven by diminishing returns and peer intelligence revealing performance gaps.

**F1** (dim2+0.010): Six weeks of +0.005 steps showed shrinking gains. Doubled the step to test the spike's limit. W9's combo failure taught me multi-dim moves overshoot, but a larger single-dim step is a different experiment. **F2** ([0.45, 0.50]): All 9 prior queries had dim1 > 0.67 or dim1=0.20 (dead). The 0.20-0.68 gap is completely unsampled. Peer confirmed bimodality. **F3** (dim3+0.004): dim1 bracketed, dim2 failed (W6). W9's dim3 decrease failed — reversed direction. **F4** (dim1 and dim4 both -0.005): EI catastrophically failed twice (W6, W9). Banned automation. Two fresh untested dims on a smooth landscape. **F5** ([0.5,0.5,0.5,0.5]): All prior queries targeted boundaries. Centre is maximally unexplored — could reveal hidden structure. **F6** (dim4+0.0002): dim2 exhausted after 4 weeks. dim4 has ls=0.001 — ultra-sensitive but untapped. Single controlled nudge, unlike W5's multi-dim disaster. **F7** (dim2-0.003): Four consecutive bests. Accelerated the step while staying above the 0.271 failure zone. **F8** (dim3+0.030): W9's dim3 breakthrough (+0.012) was the biggest gain since W4. Bolder step to close the 0.18 gap to a peer's 9.96.

## How transparent is your decision-making process?

Fully reproducible from three artefacts: weekly Jupyter notebooks with GP fits, sensitivity analyses, and coded recommendations with inline rationale; cumulative .npz data files; and these reflections. Every recommendation includes what it avoids and the evidence base. A reviewer could re-run any notebook to verify GP predictions against outcomes. One gap: peer intelligence (F2 bimodality, F8 peer score) came from discussion posts and cannot be independently verified.

## What assumptions are you making?

**Key assumption: functions are deterministic.** My "direction failed, reverse it" logic requires same input to produce same output. F6 violates this — identical W2/W7 inputs gave -0.521 and -0.606 respectively. This means my dim2 bracketing may be unreliable. A secondary assumption is that GP length scales accurately reflect sensitivity with only 19 data points per function — these estimates have high uncertainty and could shift with new data in unexplored regions.

## Where do you see gaps or biases?

**F2 sampling bias**: 9 of 10 queries cluster in dim1=0.67-0.81. The 0.20-0.68 gap (47% of the input space) is unsampled along the only relevant dimension. **F8 dimensionality**: 48 points in 8D gives ~1.6 points per dimension — the GP extrapolates aggressively. **Temporal bias**: early weeks explored broadly, later weeks micro-nudged near bests. The GP's global picture rests on early, less-informed queries while recent data only refines local knowledge.

## What is one significant limitation?

**The single-query budget makes it impossible to distinguish noise from signal.** When F6 returns -0.567 for dim2=0.277 versus -0.521 for 0.276, I cannot determine whether the difference is a real gradient or noise. Every inference chain — "this direction improved, therefore continue" — assumes each observation faithfully represents the true function. The F7 "4 consecutive bests" could be genuine improvement or lucky draws from a flat, noisy function. Without replicate observations, my strategy adapts to patterns that may be partially illusory. This is the fundamental constraint of black-box optimisation under extreme budget scarcity.
