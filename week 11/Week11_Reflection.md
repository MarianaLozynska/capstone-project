# Week 11 Reflection — Patterns, Clusters & Strategy Refinement

## How have patterns in your past queries influenced your latest choices?

Week 10 produced 4/8 new bests — the best hit rate since Week 6 — and every win came from a specific pattern that now drives Week 11.

**Double step pattern (F1)**: Alternating-dim +0.005 nudges produced 6 consecutive bests but with shrinking gains. W10's doubled step (dim2+0.010) gave +0.066 — 3x larger than any +0.005 gain. This pattern now applies to dim1+0.010 in W11, alternating the dimension while keeping the larger stride.

**Confirmed direction pattern (F3, F8)**: F3's dim3+0.004 broke a 5-week plateau. F8's dim3+0.030 gave a second consecutive gain. Both continue the same direction and step — when a direction works, push it until it stops.

**Failure avoidance pattern (F4)**: W6's dim2=0.348 scored 0.466 (bad). W10's multi-dim move also failed. For W11 I nudge dim2+0.005 to 0.372 — explicitly moving away from the known failure zone. All 80 previous submissions were checked against W11 proposals.

**Exhaustion pattern (F6)**: Four weeks of dim2 tweaks around 0.276 gave negligible variation. W10's dim4+0.0002 went wrong direction. W11 reverses to dim4-0.0002.

## Have you identified any clusters or recurring regions?

**F1 spike cluster**: All bests cluster tightly around [0.41-0.43, 0.41-0.43]. GP length scales (0.012, 0.008) confirm the function is a narrow spike — the cluster radius is approximately 0.02 in each dimension. The optimum is inside this cluster.

**F7 descent trajectory**: Bests form a line along dim2 decrease: 0.322→0.317→0.314→0.312→0.309→0.306, with all other dims locked. This is a 1D trajectory through 6D space — a "success filament" rather than a cluster.

**F5 corner dominance**: The highest-performing region is the [1,1,1,1] corner at 8662. Surrounding queries form a clear "performance cliff" — any deviation drops sharply (8290 at [1,1,0.98,1], 4441 at [1,0,1,1], 32 at [0.5,0.5,0.5,0.5]).

**F2 has no cluster**: 10 queries spanning dim1=0.20 to 0.81 show a single peak near dim1=0.70 (best 0.614) and nothing elsewhere. If a second peak exists, it has eluded our search entirely.

## Which strategies have proven less effective?

**Automated EI on non-smooth functions**: EI catastrophically failed on F4 twice (W6: 0.466, W9: -0.043). The GP predicted 1.086 for W9's point but reality was -0.043. Lesson: GP overconfidence on sparse data produces confidently wrong recommendations. EI is permanently banned for F4.

**Multi-dim moves on spiky functions**: F1's combo move (W9: both dims +0.005) slightly regressed. F4's two-dim move (W10: dim1 and dim4 both -0.005) regressed. Lesson: even on smooth landscapes (F4), multi-dim moves introduce interaction effects that single-dim nudges avoid.

**Wide exploration on F2**: dim1=0.20 (-0.001), dim1=0.45 (0.195) — both dead. W11 tests dim1=0.55, the last unexplored gap. If this also fails, the function is likely unimodal around dim1=0.70 despite peer claims of bimodality.

**Micro-nudges past exhaustion (F6)**: Four weeks of dim2 adjustments (0.271/0.276/0.277/0.281) all within 0.01 of each other. Diminishing returns when the optimum has been bracketed to within one length scale.

## How do your refinements parallel clustering algorithms?

My approach parallels **density-based clustering** (DBSCAN): I identify high-performing regions by query density and treat isolated good results with scepticism.

F7's success trajectory works like a **centroid tracker** — each week's best becomes the new centroid, and the next query is placed at a fixed offset. This is analogous to online k-means where the centroid updates incrementally.

F6's noise problem parallels the **outlier detection** challenge in clustering: W2 and W7 submitted identical inputs but got different outputs (-0.521 vs -0.606). A clustering algorithm would flag this inconsistency; I treat it as evidence that F6 has stochastic noise and reduce confidence in local gradients.

The failure history check — verifying W11 proposals against all 80 previous submissions — parallels **duplicate detection** in data preprocessing. Just as clustering algorithms remove or merge near-duplicate points, I ensure no query wastes budget on previously explored territory.

## What trends might appear if plotted?

**F1**: A diagonal line climbing from [0.41, 0.40] toward [0.43, 0.43] — steady ascent up a narrow spike, with the W2 outlier ([0.40, 0.67]→3.1e-39) far off the ridge.

**F7**: A straight horizontal line in dim2 (decreasing) with constant values on all other dims — the clearest 1D trajectory in the dataset.

**F8**: A vertical line in dim3 (increasing from 0.020 to 0.100) — the fresh dimension that unlocked improvement after dim4 converged.

**F5**: A "star" pattern from the [1,1,1,1] corner — rays extending outward with values dropping rapidly, confirming corner dominance.

**F2**: Scattered points across dim1 with a single peak at 0.70 and flatness everywhere else — suggesting unimodality despite peer reports.

These patterns inform W11 by confirming which functions are on productive trajectories (F1, F3, F7, F8) and which require fundamentally different approaches (F2, F6).
