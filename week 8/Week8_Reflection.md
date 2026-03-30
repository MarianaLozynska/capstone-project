# Week 8 Reflection — Prompt Engineering & Decoding Strategies in BBO

## Which prompt patterns (zero-shot, few-shot, etc.) did you use, and why? What changed when you simplified vs structured the prompt?

I used **structured few-shot prompting** throughout, providing the LLM with the full history of inputs, outputs, and strategy evaluations as context before requesting recommendations. Each function's prompt section follows a template: best point, history of what worked/failed, GP diagnostics, and a constrained ask ("propose one point within these bounds").

Early weeks used simpler zero-shot prompts — "suggest the next point for this function given these observations." The outputs were generic and often proposed points in already-explored regions. Switching to structured prompts with explicit failed-strategy lists and per-dimension constraints dramatically improved recommendation quality. For example, the structured prompt for F4 explicitly states "dim3 decrease worked twice, EI failed once — continue manual dim3-0.005" rather than leaving the model to infer this from raw data. The trade-off: structured prompts consume more tokens but produce actionable, non-redundant suggestions. Simplified prompts are faster but lose the accumulated strategic context that prevents repeating past mistakes.

## What temperature, top-k, top-p and max-tokens settings did you choose? How did they trade off coherence vs diversity? How did they affect your chosen query?

For **strategy analysis and recommendation generation**, I used low temperature (0.0-0.2) to maximise coherence and determinism. When the LLM proposes a single point to submit per function per week, I need the most likely correct answer — not creative alternatives. Low temperature ensures the model consistently applies the logic chain (history → diagnosis → recommendation) without introducing random variation.

For **exploratory brainstorming** — like identifying where F2's second peak might be — I temporarily increased temperature to 0.7 to generate diverse hypotheses about unexplored regions. This directly influenced Week 8's F2 strategy: higher-temperature generation suggested exploring low dim1 values (never tried), while low-temperature generation kept defaulting to micro-nudges near the known peak.

Top-p was set at 0.95 (near-default) for most tasks. Max-tokens was set high enough (2000+) to avoid truncation of the full 8-function analysis, but I monitored output length to ensure the model completed all functions before hitting the limit.

## Did token boundaries or unusual input strings affect the model's behaviour? When did you notice token count limits or truncation influencing the outputs?

With 17 data points across 8 functions (2-8 dimensions each), the input context grew substantially by Week 7. I checked for truncation issues in two ways: (1) verifying the model's output addressed all 8 functions (not stopping at F5 or F6), and (2) cross-checking that the model correctly referenced specific numerical values from the history rather than hallucinating approximate ones.

I noticed the model occasionally rounded values when the full history table was very long — for example, reporting F8's best as "9.76" instead of "9.7626642." This is a soft truncation effect: the model compresses numerical precision when the attention window is saturated with data. To mitigate this, I moved exact values into code cells (NumPy arrays) and used the LLM only for strategic reasoning, not numerical recall. The floating-point input strings (e.g., "0.162862") did not cause tokenisation issues since they are standard decimal representations, but I verified by checking that the model could correctly distinguish between similar inputs like dim2=0.270580 vs dim2=0.275580.

## With 17 data points, what limitations did you encounter, such as prompt overfitting, attention focusing on irrelevant context or diminishing returns from longer inputs?

**Prompt overfitting to recent results**: The model showed a recency bias — over-weighting Week 7 outcomes and under-weighting the full trajectory. For example, F8's single near-miss in W7 (9.761 vs 9.763) prompted the model to suggest a large correction, when the 7-week trend shows F8 is nearly converged and needs only micro-adjustments. I counteracted this by including the cumulative best table, forcing the model to see the full arc.

**Attention on irrelevant dimensions**: For F8 (8D), the model sometimes proposed changes to dims 6 and 8 despite their length scales being > 10,000 (effectively irrelevant). The GP diagnostics were in the prompt, but the model did not always prioritise them. Explicitly stating "dims 6 and 8 are LOCKED — length scales > 10,000" in the prompt fixed this.

**Diminishing returns from longer context**: Adding all 17 raw data points per function provided marginal value over a well-summarised history. The model cannot meaningfully process 8x17 = 136 data points in natural language — it extracts the same insights from a 3-line summary ("best at [x], trend is dim3 decreasing, EI failed once"). I shifted to providing raw data only in code cells and using concise strategic summaries in the prompt.

## Which strategies did you try to reduce hallucinations? For example, did you use tighter instructions, retrieval of prior relevant information or constrain the output format?

Three main strategies:

1. **Constrained output format**: Every recommendation must include the exact point (array), the strategy name, what it avoids, and the rationale. This forces the model to justify each recommendation against the failure history rather than generating plausible-sounding but unsupported suggestions.

2. **Retrieval of prior results into the prompt**: Rather than asking the model to "remember" past weeks, I explicitly loaded all previous inputs, outputs, and strategy evaluations into the context. This eliminated a class of hallucinations where the model would invent plausible-sounding past results (e.g., claiming F3 improved in Week 5 when it actually regressed).

3. **Cross-validation against code outputs**: The GP predictions, sensitivity analysis, and distance checks are computed in code cells — not by the LLM. The model's role is strategic reasoning; numerical claims are verified programmatically. When the model suggested F6's dim2 decrease "should improve based on the trend," the code showed the GP predicted worse performance, catching the hallucination before submission.

## In future rounds, how would you scale your prompting and decoding strategies when working with larger data sets or more complex LLMs?

With more data points (50+), I would shift to **retrieval-augmented generation (RAG)**: store all historical results in a structured database, retrieve only the top-k most relevant observations per function per query, and inject those into the prompt. This prevents context window saturation while preserving the most decision-relevant information.

For more complex models, I would use **chain-of-thought prompting with explicit verification steps**: "First, identify the best point. Then, list all strategies that failed on this function. Then, check if the proposed point is within the GP's reliable range. Finally, output the recommendation." This structured reasoning chain scales better than free-form analysis as the decision space grows.

I would also implement **self-consistency decoding** — generating multiple recommendations at different temperatures and selecting the one that appears most frequently or that the GP rates highest. This is analogous to ensemble acquisition functions (running EI, UCB, and PI and taking the centroid), which I identified in Week 7 as a future improvement.

## How did these design choices for prompts and decoding help you think like a practitioner balancing exploration, risk and computational constraints in a black-box setting with incomplete information?

The prompt engineering workflow mirrors the practitioner's decision loop: gather evidence (retrieve history), diagnose (sensitivity analysis), hypothesise (generate candidates), validate (GP sanity check), and commit (submit one point). Each stage has a computational budget — just as I allocate one query per function per week, I allocate specific prompt sections to each reasoning stage.

The key lesson is that **the prompt is a resource allocation problem**. Every token spent on raw data is a token not spent on strategic reasoning. Every degree of temperature freedom is a trade-off between exploring novel strategies and exploiting proven ones. This directly parallels the exploration-exploitation trade-off in BBO: F1's low-temperature, highly structured prompt (exploit the alternating dim nudge) vs F2's higher-temperature, loosely constrained prompt (explore a completely new region) reflect the same judgment calls I make when choosing acquisition function parameters.

Working with incomplete information — no access to the true function, no gradient, no validation set — forced me to treat the LLM's outputs with the same scepticism I apply to GP predictions: useful as a starting point, but always verified against evidence before acting. This is the core skill of production ML: knowing when to trust your tools and when to override them.
