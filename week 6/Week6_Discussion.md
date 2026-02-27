# Week 6 Discussion — Technical Foundations of BBO Strategy

## 1. What is the main technical justification for your current BBO approach?

My approach uses Gaussian Process regression as a surrogate model combined with acquisition function optimisation. The justification is straightforward: with only one query per function per week, I need a model that provides both predictions and uncertainty estimates from very small datasets (15-45 points). GPs do both natively. The uncertainty drives the acquisition functions — Expected Improvement, Upper Confidence Bound and Thompson Sampling — which balance exploring unknown regions against exploiting known good ones. Over six weeks this has produced improvements in most functions, with the strategy evolving from standard BO to sensitivity-driven adaptive approaches as I learned more about each function's landscape.

## 2. Which academic papers guided your design?

The core framework follows the Efficient Global Optimization (EGO) algorithm from Jones, Schonlau and Welch (1998), which introduced Expected Improvement with GP surrogates for expensive black-box functions. My use of Upper Confidence Bound draws on Srinivas et al. (2010), who provided theoretical bounds on GP-UCB regret. Thompson Sampling as an alternative when the GP is unreliable comes from Russo et al. (2018), which showed TS provides natural exploration in settings where deterministic acquisition functions get stuck. The Matern kernel choice (nu=2.5) follows Rasmussen and Williams (2006), which recommends it for functions assumed to be twice differentiable but without stronger smoothness assumptions.

## 3. Which libraries are central and why?

Scikit-learn provides the GP implementation with the Matern kernel, built-in hyperparameter optimisation via marginal likelihood, and per-dimension length scales that I use directly for sensitivity analysis. SciPy provides L-BFGS-B for bounded acquisition function optimisation with multiple restarts, and the normal distribution functions needed for the EI formula. NumPy handles all data operations and persistence via .npz files.

I considered GPyTorch and BoTorch as alternatives. They offer more advanced GP models and acquisition functions, but for 2-8 dimensional problems with fewer than 50 points, scikit-learn's implementation is sufficient and simpler to debug. The added complexity was not justified at this scale.

## 4. How do you plan to document these justifications in your GitHub repository?

The repository already has a structured README covering the problem setup, strategy evolution across weeks, and the utility module architecture. Each weekly notebook documents the specific strategy rationale — what the sensitivity analysis found, why I chose a particular acquisition function, and sanity checks on the recommendations. The utils/ package (bayesian_optimization.py and sensitivity.py) is documented with clear function signatures.

I plan to add a technical foundations section to the README that links the methods to the papers above, explains the GP-to-acquisition-function pipeline, and summarises the lessons learned across weeks — particularly the shift from optimizer-driven to manual strategies when GP predictions became unreliable.

## 5. What additional sources might you consult to continue refining your strategy?

I would look into multi-fidelity optimisation (Kandasamy et al., 2017) for functions where I have many low-quality observations but few high-quality ones. Heteroscedastic GPs could help with functions like F1 where the output variance changes sharply across the input space. I would also explore input warping (Snoek et al., 2014) to handle the non-stationary length scales I see in functions like F6, where some dimensions have length scales of 0.0003 and others 24.5. On the software side, Ax (from Meta) provides a higher-level Bayesian optimisation platform that automates many of the manual strategy decisions I am currently making week by week.
