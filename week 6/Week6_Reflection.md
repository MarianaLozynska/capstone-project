# Week 6 Reflection

## 1. How did progressive feature extraction influence the way you refined your BBO strategy?

CNNs build understanding in layers — edges, then textures, then objects. My pipeline does the same: GP fits raw data and learns length scales, sensitivity analysis ranks which dimensions matter, then I set bounds and pick acquisition functions accordingly.

This week that mattered for Function 6. The GP learned dims 4 and 5 have length scales of 0.001 and 0.0003. Week 5 confirmed it — moving those dims even slightly dropped the output from -0.52 to -0.99. So I locked them and only nudged dim2.

## 2. What parallels do you see between LeNet/CNN breakthroughs and your incremental BBO improvements?

LeNet proved CNNs work. Later architectures refined the idea with more depth and better training. My capstone followed a similar arc — Week 2 proved BO works, then each week added a layer: sensitivity analysis, acquisition function matching, manual overrides.

Function 4 is the incremental case — four consecutive improvements with the same EI strategy, just tighter each week. Function 1 was more like a breakthrough — switching to manual nudges in Week 5 gave a 47% jump.

## 3. Did you face trade-offs between exploring widely and exploiting promising regions?

Yes. Function 6 explored too widely in Week 5 and scored -0.99, the worst result yet. Function 5 made a tiny move on dim3 (1.0 → 0.98) and lost 372 points.

My Week 6 response was to use manual overrides for 6 out of 8 functions. The optimizer kept suggesting points outside the GP's reliable range. Simpler strategies, tighter bounds, smaller moves.

## 4. Which CNN concepts helped you think differently about how your model learns from data?

Convolutions detect local patterns with a sliding filter. My GP does something similar — short length scales (F3's 0.004) mean rapid local change, long ones (F6 dim1 at 11.6) mean smooth regions with nothing to detect.

Pooling reduces dimensionality. My sensitivity analysis does the equivalent — identifying that F8's dims 6 and 8 are irrelevant and locking them, effectively reducing the search space.

## 5. How might real-world deployment challenges help you benchmark success in your BBO project?

Edge AI means hard constraints — limited compute, strict latency. My BBO has a similar constraint: one query per function per week.

Week 5 scored 2/8, but the functions where I followed the plan (F1, F4) both improved. The ones where queries diverged (F3, F6) regressed. That distinction matters more than the raw hit rate.

Function 4's four consecutive improvements with the same approach is like a CNN that performs reliably across inputs — not flashy, but deployable. For Week 6 my benchmark shifted to "did the query stay in GP-reliable territory?" rather than just "did it beat the best?"
