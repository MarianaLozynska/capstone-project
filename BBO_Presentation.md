# BBO Capstone Project Presentation

## 1. Overview of the BBO Approach

My goal in this project was to find the maximum value of eight unknown functions, ranging from 2 to 8 dimensions. I only had one query per function per week, and no access to equations, gradients, or any information about what the functions actually were. I had to figure them out entirely from the inputs I submitted and the outputs I got back.

My approach is a hybrid one. I fit a Gaussian Process model to the data I've collected so far, use its kernel length scales to understand which dimensions matter for each function, and then decide whether to let an automated acquisition function (Expected Improvement or UCB) choose the next query, or whether to make that choice manually. Most weeks, I end up using a mix — automated where the GP seems reliable, manual where experience has shown it is not.

Each week follows the same rhythm: load the data, run the diagnostics, decide per function, submit one point, wait for results, and reflect on what happened before the next round. By now, each of the eight functions has its own personality and its own playbook.

## 2. How the Strategy Has Evolved

I started out trusting automation far more than I do now. In the first few weeks I used standard Bayesian Optimisation with Expected Improvement and UCB, varying the exploration parameters based on which functions seemed to be improving. That worked well for some functions — F4 jumped from -4.4 to 0.67 in a few weeks, and F5 climbed from 1231 to 8662. But it failed catastrophically on others. F1 has a very narrow peak, and the optimiser kept proposing points that fell off it completely.

The turning point came around Week 5, when I realised the GP's length scales were telling me something the acquisition function wasn't listening to. On F6, length scales of 0.0003 meant that any automated move would likely be too big. On F1, tiny length scales meant the same thing. So I started making manual decisions based on those diagnostics — single-dimension nudges smaller than one length scale. That's when F1 started its streak of six consecutive improvements.

By Week 7, I was keeping a written log of every strategy that had failed on each function so I wouldn't repeat the same mistake twice. By Week 10, I was willing to double step sizes where gains were still strong, and to try completely fresh dimensions on functions that had plateaued. The principle I trust most now is: the GP's length scales are usually correct, but its predictions often aren't. Diagnose before you act.

## 3. Patterns, Data and Insights

Looking back across twelve weeks of queries, the most striking pattern is how differently each function behaves. F1 is a narrow spike — the best scores all cluster within 0.02 of each other in both dimensions, and any move larger than that falls off the edge. F7 is almost the opposite: a smooth descent along a single dimension, with six consecutive weeks of improvement coming from the same type of move, decrementing dim2 by 0.003 each time.

F5 turned out to be a corner-optimum function. [1,1,1,1] gives 8662, and every deviation I've tested loses points — [1,1,0.98,1] gave 8290, [0.5,0.5,0.5,0.5] gave just 32, and [1,1,0,0] gave 1617. Interestingly, [0,0,1,1] also gave 1617, which told me the function is symmetric between its first and second dimension pairs.

F6 has the most unsettling pattern. In Week 2 I submitted a point and got -0.521. In Week 7 I submitted the exact same point and got -0.606. The function is stochastic, which means any conclusion I draw from a single observation could be noise rather than signal. That changed how cautiously I interpret marginal differences.

The variable that has influenced my results most isn't the acquisition function or the kernel — it's the GP length scale per dimension. Once I started sizing my steps relative to those length scales, the hit rate improved noticeably.

## 4. Decision-Making and Iteration

I don't use a fixed ratio of exploration to exploitation. It varies per function based on what the data is telling me. On F1 and F7, where I've got clear winning directions, I lean heavily into exploitation — small steps along the known productive path. On F2, which has been stuck at 0.614 since Week 2, I've spent most of my queries exploring new regions because continuing to refine around the known peak wasn't working.

One decision I'm proud of was switching F8 from dim4 to dim3 in Week 9. I'd spent three weeks making micro-nudges on dim4 with almost no improvement. Instead of continuing, I looked at the length scales and realised dim3 was more sensitive among the active dimensions, and I'd never touched it. The switch produced three consecutive improvements (+0.012, +0.013, +0.008) — the biggest gains F8 had seen since Week 4.

One decision I regret was Week 9's F4 move. I used Expected Improvement with tight bounds, trusting the smooth landscape. The GP predicted 1.086; the actual score was -0.043. That regression was worse than the earlier Week 6 EI failure, and it cost me a query I couldn't get back. Since then I haven't let the automated optimiser touch F4.

When results don't match my expectations, my first reaction is to compare the GP's prediction for that point against the actual outcome. If they diverge significantly, I stop trusting the GP for that function for the next few rounds and go back to manual reasoning. I've also learned to treat small score differences on F6 with scepticism — they might just be noise.

## 5. Next Steps and Reflection

If I had more rounds, the first thing I'd try is changing the kernel on F2. Twelve weeks of queries have found a single peak at dim1=0.70 with value 0.614, and my wider exploration at dim1=0.20, 0.45, and 0.55 all returned lower values. It is possible the Matern 2.5 kernel is smoothing over finer structure I need to see. A Matern 1.5 or RBF kernel might reveal different behaviour. For F6, I'd want replicate queries — submitting the same point multiple times to estimate the noise variance, which would tell me whether I'm actually stuck or just noise-limited. For F1, F7, and F8, I'd reduce my step sizes gradually as gains diminish, because the Week 11 overshoot on F1 showed me that being too bold late in the game can undo previous progress.

The broader lesson I'm taking from this project is that in any expensive-evaluation setting — whether it's drug discovery, hyperparameter tuning, or real-world A/B testing — knowing when to trust your tools is as important as having good tools. Automated Bayesian Optimisation is powerful, but it fails silently on narrow spikes, noisy functions, and sparse high-dimensional spaces. The hybrid pattern I've been using here is exactly what production ML engineers do: automate the routine, but keep a human in the loop for the hard cases.

If I had to explain this project to someone non-technical, I'd say something like: "I had to find the highest point on eight invisible hills, taking only one step a week. By paying attention to how the ground sloped under each step, I gradually learned which hills were narrow peaks and which were long gentle climbs, and matched my stride to the terrain. Five of the eight hills are now mapped well enough that I keep making progress. Three remain difficult — one because the ground itself seems to shake, and two because I haven't yet found the right slope."
