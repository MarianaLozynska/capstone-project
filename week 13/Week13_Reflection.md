# Week 13 Reflection — Final Round

## How has your understanding of the exploration–exploitation trade-off evolved with increasing data?

In the early weeks I treated the trade-off as a single dial — set the exploration parameter, hope for the best. By Week 5 it was clear that no single setting worked across all eight functions. Some functions, like F1 and F7, rewarded relentless exploitation with tiny consistent steps. Others, like F2 and F6, were stuck for weeks despite my best exploitation efforts and only broke open when I forced exploration into completely untested regions.

What changed most dramatically was my understanding that the trade-off is asymmetric and time-dependent. Early in the project, exploration paid off because the surrogate model knew almost nothing — any new observation was high-information. By the later weeks, exploration in already-mapped regions became wasteful, while exploration of genuinely unexplored dimensions still produced breakthroughs. F6's twelve-week stagnation broke in Week 12 only because I tried the one dimension I had never touched. That was a lesson in being patient about exploration but specific about where to direct it.

The final week's insight was that under max-across-weeks scoring, the trade-off effectively collapses. Every untested point becomes a free lottery ticket — bolder moves are strictly better because the floor is the existing best. I went from cautious to deliberately bold for Week 13, accepting that some queries would miss while others might find a peak that twelve weeks of careful nudging hadn't.

## How did the nature of feedback influence your optimisation process? Relate this to RL.

Each week's output acted like a reward signal updating my Q-values for each function-strategy pair. After Week 6's EI catastrophe on F4, the Q-value for "use EI on F4" dropped sharply and never recovered — by Week 9 a second EI failure confirmed it as a permanently low-value action. By contrast, the Q-value for "decrease dim2 on F7" climbed steadily across seven consecutive successes, until I trusted it enough to commit even on the final round.

Like an RL agent, I also faced delayed and sparse rewards. With one query per function per week, my "reward" arrived seven days after the action — long enough that I had to track my reasoning carefully or lose the credit assignment. The weekly reflections became my equivalent of an experience replay buffer: a written log of what I tried, what I expected, and what actually happened. Without it I would have repeated mistakes I had already learned to avoid.

The biggest RL parallel is in handling noise. F6 produced different outputs for the same input in Weeks 2 and 7. That is exactly the variance an RL agent faces in stochastic environments, and it forced me to reduce my confidence in single observations and weight repeated patterns more heavily.

## Does your process resemble AlphaGo Zero's self-play, and was it model-free or model-based?

Both, depending on the function. The Gaussian Process surrogate is fundamentally a model-based approach — it builds an internal representation of the function's landscape and uses that to predict outcomes before committing a query. When the GP was reliable (smooth functions, dense data), I let it plan future moves like a model-based RL agent.

But when the GP failed badly (F4's EI catastrophes, F1's narrow spike), I shifted to model-free trial-and-error. Each manual nudge was effectively a Q-learning update with no internal model: I tried a small step, observed the reward, and adjusted the next step accordingly. The "confirm, continue, halve, reverse" rule I developed is essentially a simple TD-learning policy.

The self-play parallel is weaker but real. I was not playing against an opponent, but I was using my own past observations as the training data for the next decision — a kind of bootstrapped self-improvement. AlphaGo Zero learned by playing itself; I learned by querying my own surrogate and updating it with each new result. Both approaches generate their own training data through their own actions, with no external teacher.

## How could RL strategies improve real-world optimisation?

Three practical applications. First, **adaptive exploration schedules**: an RL agent that learns when to explore versus exploit based on its own uncertainty would have prevented my F2 from getting stuck for ten weeks while exploring marginally near the known peak. Second, **transfer learning across related problems**: if I had been optimising eight similar functions, an RL approach could have used what it learned on F1's narrow spike to set priors for other potentially-spiky functions. Third, **multi-armed bandit formulations** for query budget allocation: instead of treating every function equally, an RL allocation policy would have spent more queries on functions where marginal information gain is highest, accelerating convergence on the winnable functions while cutting losses on the genuinely stuck ones.

In any expensive-evaluation setting — drug discovery, AutoML, or production tuning — these techniques compress the time between a new idea and a confident decision, which is the same compression I just lived through for thirteen weeks.
