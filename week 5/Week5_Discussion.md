# Week 5 Discussion: Repository Structure, Libraries & Documentation

## Repository Structure

**How have you organised your repository so far?**

My repository follows a chronological weekly structure with shared utilities. The top level contains `initial_data/` (seed data for all 8 functions), weekly folders (`week 1/` through `week 5/`), and a `utils/` package with reusable modules. Each weekly folder contains a Jupyter notebook (`weekN_function_analysis.ipynb`), a clean data snapshot (`.npz`), and a reflection document (`WeekN_Reflection.md`). The `utils/` package — refactored in Week 2 — holds GP fitting, acquisition functions (EI, UCB, Thompson Sampling), sensitivity analysis, and visualisation code, so that weekly notebooks stay focused on strategy decisions rather than reimplementing shared logic.

**What changes will you make to improve clarity, navigability and reproducibility?**

I added a `.gitignore` to exclude `.DS_Store`, `__pycache__/`, and `.ipynb_checkpoints/` artifacts that were previously tracked. I added `requirements.txt` for dependency management and `__init__.py` to make `utils/` a proper Python package. I also renamed the Week 1 notebook for consistency (`function_analysis.ipynb` → `week1_function_analysis.ipynb`), moved `Week1_Reflection.md` into its correct folder, and removed loose scratch files (`inputs.txt`, `outputs.txt`) that had been superseded by `.npz` snapshots. In future projects, I would use hyphenated folder names (`week-1/`) instead of spaces for better command-line compatibility.

## Coding Libraries and Packages

**Which libraries or frameworks are central to your approach?**

- **scikit-learn** (`GaussianProcessRegressor` with Matern kernel): the surrogate model. GPs provide both predictions (mu) and calibrated uncertainty estimates (sigma) from very few observations (13-43 points per function). The uncertainty directly drives the acquisition functions.
- **SciPy** (`minimize` with L-BFGS-B): optimises acquisition functions over the bounded [0, 1] input space. L-BFGS-B supports box constraints natively, which is essential for this problem.
- **NumPy**: all data manipulation, `.npz` persistence for weekly data snapshots, and numerical computations underlying sensitivity analysis.
- **Matplotlib**: diagnostic visualisations — GP surface plots for 2D functions, PCA projections for higher dimensions, sensitivity profiles per dimension, and predicted-vs-actual validation with R-squared scores.

**Why are these choices appropriate, and what trade-offs did you consider?**

I considered neural network surrogates (PyTorch/TensorFlow) but rejected them for two reasons: (1) with only 13-43 data points per function, neural networks would overfit severely, whereas GPs are naturally regularised through kernel priors; (2) GPs provide calibrated uncertainty out of the box, which acquisition functions require — neural networks would need additional infrastructure (MC dropout, ensembles) to approximate this. The trade-off is that GPs scale poorly to large datasets (O(n^3) matrix inversion), but with fewer than 50 points per function, this is irrelevant.

I also implemented EI, UCB, and Thompson Sampling myself rather than using a library like BoTorch or Optuna. This forced me to understand the exploration-exploitation trade-off at a deeper level, and made it easier to debug when my GP recommendations were wrong — for example, discovering that deterministic EI kept recommending the same bad regions for Function 3 (0/3 weeks improved), which led me to add Thompson Sampling as a stochastic alternative in Week 4.

## Documentation

**How do your README and other documents currently describe the project?**

My README covers the project overview, input/output specifications (format, dimensionality per function), challenge constraints (1 query/week, no derivatives), and the technical approach through Week 4. It includes a results summary table tracking all 8 functions across 4 weeks, a strategy evolution narrative, and instructions for running the code. Each week also has a dedicated reflection document explaining strategy rationale, diagnostics used, and lessons learned.

**What updates do you need to align documentation with your most recent strategy and results?**

I have already updated the README with Week 3-4 results, the full project structure diagram, and the strategy evolution section showing the progression: PCA exploration (Week 1) → standard BO (Week 2) → sensitivity-driven adaptive strategies (Week 3) → GP reliability-matched acquisition functions with Thompson Sampling (Week 4). Remaining updates include: pinning exact library versions in `requirements.txt` for full reproducibility, and adding a failure analysis section documenting what went wrong (Function 1's lost peak in Week 2, Function 3's persistent plateau) and how those failures drove methodology improvements.
