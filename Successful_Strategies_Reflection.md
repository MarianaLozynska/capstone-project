# Successful Strategies Reflection (25.2)

## Which optimisation strategies led to your strongest results, and why were they effective? How did these strategies influence your decisions as the challenge progressed?

Looking back, three things drove most of my best results.

The first was using the GP's length scales to size my steps. Once I extracted the per-dimension length scales from the surrogate, I had a number telling me how big a "safe" move was on each dimension. If I kept my step under one length scale, I usually stayed on the productive side of the function. If I went above it, I usually fell off. That single discipline saved me from a lot of bad weeks. F1 went from 0.098 to 0.965 mainly through small +0.005 nudges between Week 5 and Week 8, and one bolder +0.010 step in Week 10 that gave me the final jump to 0.965. F7 produced eight consecutive new bests just from disciplined dim2 decreases sized to its length scale. Nothing clever about it — just a constraint that kept me out of trouble.

The second was switching to an untouched dimension whenever a function plateaued. F8 stalled at 9.76 after Week 4 no matter what I tried on dim4. In Week 9 I switched to dim3, which I had never moved as a single-dim step from the best point. That gave me three consecutive new bests over three weeks. The same thing happened with F6: it had been stuck around -0.52 for twelve weeks, and the breakthrough in Week 12 came from nudging dim5, the only sensitive dimension I had not previously tried. The pattern is simple — diminishing returns on one dimension does not mean the function is done, it usually means I have run out of room on that dimension and need to look elsewhere.

The third was keeping a written log of every direction that had failed on each function. I started this in Week 7 and used it before every single submission afterwards. By the end I had close to 100 prior queries to remember, and there was no way to keep that in my head. Cross-referencing the proposed move against the log every week stopped me from repeating mistakes, which sounds obvious but turned out to be the difference between learning over time and going in circles.

These three strategies changed how I made decisions later in the project. By Week 11 I had a fairly strict policy: no multi-dimensional moves on narrow-spike functions, no automated EI on F4 after the two catastrophes, no resubmitting points I already knew were bad, and no steps larger than one length scale unless I had explicit evidence that a bigger step worked. The strategy got more rule-based as more data accumulated.

## In your view, what qualities define a 'successful' strategy – is it outcomes alone or also adaptability, reasoning or efficiency?

I do not think outcomes alone define a successful strategy. Some peers reportedly hit very high values on certain functions through what looked like luck — one bold guess that landed in the right region. That is success on the leaderboard but it is not transferable.

For me, a successful strategy needs three things. It has to be adaptable — it has to change when the data changes rather than insisting on a fixed approach. My switch from EI to manual on F4, after EI failed twice, was an example of that. Carrying on with EI a third time would have been bad strategy. It has to have reasoning that I can explain after the fact. If the only justification I have for a decision is "I had a hunch", then I am not going to be able to repeat it on a different problem. And it has to be efficient — the same result with fewer queries, or less compute, or less complexity is genuinely better than the same result with more.

By those criteria, my length-scale-sized nudges, my fresh-dimension switches and my failure log all qualify. They are adaptive (the length scale updates as data comes in), defensible (every step has a written justification), and efficient (single-dimension moves are cheap and the log stops me wasting queries on known failures).

## How could the strategies you identified be applied or adapted to professional ML/AI projects beyond the BBO capstone project?

The closest real-world parallel is hyperparameter tuning when training is expensive. Every trial should be informed by what previous trials revealed. The hybrid I ended up using — automate where the surrogate is reliable, override where it is not — is exactly the pattern that tools like Optuna and Ray Tune are designed to support. The thing I would carry forward is that automated tuning is not "set and forget" — it needs periodic diagnostic checks, the same way I checked my GP's length scales every week, to catch when the surrogate has gone wrong.

Drug discovery and materials science have the same structure: each evaluation is expensive, you do not know the function, and a wrong query is costly. Keeping moves within learned sensitivity scales, switching to fresh experimental variables when current ones plateau, and keeping a documented log of failed approaches all translate directly. A pharmaceutical company optimising a compound is doing the same kind of thing I was doing on F1, with much higher stakes.

A/B testing is similar. Most product teams have to allocate experiment traffic across competing variants, which is the same exploration-exploitation balance I had to manage weekly. "Push winning directions until gains turn negative" is essentially a multi-armed bandit policy. "Track every failed variant so you do not retest it" is just deduplication in an experiment design system.

## What successful strategies did you notice in your peers' approaches, and what made them effective? Do you see overlap with your own strategy?

A few peer strategies stood out.

Vijaya's progressive structuring made the search increasingly evidence-based as data accumulated. He used per-function surrogate selection — different model families like SVR or MLP depending on how the function behaved — along with novelty and minimum-distance constraints to stop duplicated queries, and PCA-guided local refinement. The principle of letting the strategy itself evolve with the evidence base, rather than committing to one approach from the start, is the same instinct that drove my Phase 1 / Phase 2 / Phase 3 evolution. The overlap is in the principle — start broad, narrow as the data justifies it.

Shachar's variance-based portfolio allocation was the most rigorous version of something I was doing intuitively. He explicitly categorised his functions into "learnable", "volatile" and "flat" and allocated 60/30/10 percent of cognitive effort. His Week 9 deliberate perturbation of F5 — querying off-centre to validate that the centre was actually optimal — was hypothesis-driven experimentation in its purest form. He accepted a short-term score loss to get strategic certainty. Where my approach drifted between functions opportunistically, his portfolio was actually budgeted. That formalisation would scale far better in a team setting.

Bilal's hybrid surrogate (random forest plus GP) gave him a 12 percent improvement in a single week, which was the biggest single-week jump I saw in any peer reflection. He found through logging that his GP was overfitting on small samples — the uncertainty estimates collapsed after only 5 or 6 points. His point that getting the surrogate right matters more than anything else is something I converged on slowly. His Week 5 penalty for slow-evaluating combos is an even better example: he sacrificed 1 percent of leaderboard score to cut variance in half. That kind of "production over leaderboard" trade-off is what real ML engineers do, and it is something I did not include in my own work.

Diran's governance-first workflow made the experimental process itself a central part of his strategy. Reproducibility, traceability, cumulative observation management, structured logs, helper scripts for weekly preparation and archiving. The technical optimisation was standard GP plus EI or UCB, but the engineering wrapper around it was exceptional. His framing that "successful optimisation needs both performance and governance" changed how I think about success — leaderboard scores tell us who performed, but auditability tells us why.

The overlap with my own strategy is large. Shachar's portfolio is the same instinct as my phased evolution. Vijaya's per-function surrogate selection is the same instinct as my per-function playbooks. Bilal's surrogate-first focus is the same insight that drove my length-scale diagnostics. Diran's governance is the same idea as my failure log, just better tooled. The main difference is that they had formalised and documented decisions I had been making implicitly.

## What suggestions or perspectives could strengthen your peers' strategies, and how do their reflections broaden your view of what success means in optimisation?

For Shachar, the variance-based allocation is great but it relies on function variance being observed accurately. F6 looked low-variance for ten weeks and then had a hidden breakthrough at dim5 that no within-function variance metric would have predicted. Reserving a small "audit budget" for occasionally revisiting low-variance functions with genuinely fresh moves would catch this.

For Bilal, the slow-evaluation penalty is clever but it assumes the slow combos are not the highest-performing ones. In some domains (deep learning architectures, complex chemistry simulations) the best models are systematically slower. A diagnostic on whether penalised combos cluster in a particular region would help separate "slow because complex" from "slow because pathological".

For Vijaya, formalising the explicit trigger for switching between automation and manual override would make his hybrid strategy more reproducible. The same applies to my own work — my switching rule was implicit ("when the GP looks unreliable") rather than explicit ("when GP predicted improvement exceeds 2x recent observed gains"). That is a real reproducibility gap I plan to close.

For Diran, his governance is industry-grade but the technical optimisation could be more aggressive in late stages. His reflections suggest he was disciplined about process but conservative about score, which is the inverse of my own pattern.

Reading peers' reflections broadened my view in two ways. The first is that success in optimisation is not a single metric — it is a Pareto frontier across outcomes, adaptability, reasoning, efficiency and reproducibility. Bilal optimised for production-readiness, Shachar for portfolio rationality, Vijaya for principled per-function tuning, Diran for governance. None of them optimised purely for leaderboard score, and the strongest results seem to come from those who balanced multiple dimensions.

The second is that I had been thinking about professional ML/AI work as "solve the problem with the best technique" — and reading peers reframed it as "solve the problem with a defensible process that produces good results reliably". That gap, between "best technique" and "defensible process", is the main gap I want to close in future work. The capstone has given me the technical skills to identify good techniques. What I want to build next is the engineering discipline around them — explicit triggers, reproducible workflows, version-controlled strategy logs, the kind of audit trail that lets someone else pick up where I left off.

The deeper lesson I take from 25.2 is that a successful optimisation strategy in a professional context is rarely the cleverest single algorithm — it is the most reproducible decision process supported by the strongest reasoning. My capstone produced strong individual decisions. What I want to build next is the wrapper that makes those decisions auditable and transferable.
