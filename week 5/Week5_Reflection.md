# Week 5 Reflection

## 1. How did hierarchical feature learning influence your optimisation strategy?

In neural networks, early layers learn simple patterns and deeper layers combine them into complex ones. My optimisation strategy works in a similar layered way.

First, the GP fits raw data and learns per-dimension length scales. Then sensitivity analysis reads those length scales to identify which dimensions matter. Finally, I use that to set bounds and choose acquisition functions. Each stage builds on the previous one.

This layered thinking helped me catch a problem this week. The GP for Function 2 had a dim2 length scale of 0.004, which sensitivity analysis flagged as "hyper-sensitive, worth exploring." But a deeper check showed the GP could not predict anything meaningful more than 0.01 away in that dimension — so the recommendation was noise, not signal. Without validating each stage, I would have submitted a bad query.

## 2. What parallels do you see between AlexNet-style breakthroughs and your incremental improvements?

AlexNet succeeded not from one clever trick but because several things aligned: more data (ImageNet), more compute (GPUs), and a well-suited architecture (deep CNNs with ReLU).

My capstone shows a similar pattern. Function 5's breakthrough in Week 3 (1688 → 7600) came from three things aligning: sensitivity analysis identifying dims 3 and 4 as critical, constrained bounds locking those dims near 1.0, and EI pushing the remaining dims. No single component would have produced that jump.

On the other side, Function 3 stalled for three weeks despite trying different strategies — like how many architectures failed before the right combination emerged. Breakthroughs look sudden but depend on accumulated groundwork.

## 3. Did you encounter trade-offs between depth, complexity and efficiency?

Yes, directly. The exploration-exploitation trade-off in BO mirrors the depth-complexity trade-off in neural networks.

Wider exploration can discover new regions but wastes limited queries on bad areas — like an overparameterised network that overfits. Tight exploitation is efficient near known peaks but misses better regions — like an underfitting model.

This week I hit both extremes. Function 2's original EI recommendation explored too widely — dim1 shifted 0.07 from best and the GP predicted half the current value. Function 1's EI was too tight — it converged to within 0.0006 of the previous query, essentially a repeat.

The fix was the same as in neural networks: match complexity to the problem. For F1's narrow spike, a manual nudge worked better than the optimizer. For F2, very tight bounds kept the search in territory the GP could actually predict.

## 4. Which neural network building blocks helped you think differently?

Gradients helped the most. In neural networks, gradients tell you how to adjust weights to reduce loss. My sensitivity analysis does the same thing — it computes GP gradients per dimension to tell me which dimensions to adjust and by how much.

Function 5's dims 3 and 4 had gradients of 45.6 and 98.9, telling me to push toward the boundary. Function 8's dims 6 and 8 had gradients near zero — irrelevant dimensions, like dead neurons that can be pruned.

Loss also translated well. Each week's result is feedback on whether my query moved in the right direction. When results worsen (Function 1 after Week 2), I increase exploration. When they improve (Function 4's steady climb), I tighten exploitation.

## 5. Is your approach closer to rapid prototyping or production-ready design?

Rapid prototyping, closer to PyTorch's philosophy.

My approach changes every week: Week 2 introduced BO, Week 3 added sensitivity analysis, Week 4 added Thompson Sampling, Week 5 added manual overrides and sanity checks. Each week I inspect results, diagnose failures, and adjust. That is the iterative, experiment-driven workflow PyTorch is designed for.

A production-ready approach would have a fixed pipeline that runs identically each week without human intervention. With only 5 weeks of data per function, the problem is too small and uncertain for that. However, the `utils/` package with reusable modules is a step toward structure. If this continued for 20+ weeks, I would move toward a more automated pipeline.

## 6. How might real-world deep learning use cases inform how you benchmark success?

Thinking about real-world applications — where models need to be practical, not just accurate — made me reconsider how I measure success.

I have been counting "functions improved per week" (5/8 in Week 4), but that treats all improvements equally. A better benchmark would ask: is the improvement consistent or a one-off?

Function 8 improved every single week (8.93 → 9.65 → 9.70 → 9.76) — small gains but highly reliable. Function 5 had one massive breakthrough (1688 → 7600) that depended on a specific insight. In practice, I would value Function 8's steady pattern more, because it means the strategy actually works rather than getting lucky once.

It also makes sense to benchmark against simple baselines like random search. If my GP-driven approach is not consistently beating random queries, the added complexity is not justified.
