# Week 5 Discussion: Repository Structure, Libraries & Documentation

## Repository Structure

My repository follows a chronological weekly structure with shared utilities. The top level contains `initial_data/`, weekly folders (`week 1/` through `week 5/`), and a `utils/` package with reusable modules for GP fitting, acquisition functions, sensitivity analysis, and visualisation. Each weekly folder has a Jupyter notebook, a `.npz` data snapshot, and a reflection document — so notebooks stay focused on strategy decisions rather than reimplementing shared logic.

To improve clarity and reproducibility, I added a `.gitignore` (excluding `.DS_Store`, `__pycache__/`, `.ipynb_checkpoints/`), `requirements.txt` for dependency management, and `__init__.py` to make `utils/` a proper package. I also renamed the Week 1 notebook for consistency and removed loose scratch files superseded by `.npz` snapshots. In future projects I would use hyphenated folder names (`week-1/`) instead of spaces for better command-line compatibility.

## Coding Libraries and Packages

My core stack is scikit-learn for GP regression (Matern kernel), SciPy for optimising acquisition functions over the bounded input space, NumPy for data manipulation and persistence, and Matplotlib for diagnostics and visualisation.

GPs are well suited to this problem because they handle small datasets (13–43 points per function) and provide built-in uncertainty estimates that acquisition functions need. I also implemented EI, UCB, and Thompson Sampling myself, which helped me understand the exploration-exploitation trade-off more deeply and made debugging easier — for example, discovering that deterministic EI kept recommending the same bad regions for Function 3, which led me to add Thompson Sampling in Week 4.

## Documentation

My README covers the project overview, input/output specifications, challenge constraints, and the technical approach through Week 4, including a results summary table, a strategy evolution narrative (PCA → standard BO → sensitivity-driven strategies → GP reliability matching with Thompson Sampling), and instructions for running the code. Each week also has a dedicated reflection document. Remaining updates include pinning exact library versions in `requirements.txt` and adding a failure analysis section documenting how failures (Function 1's lost peak, Function 3's plateau) drove methodology improvements.
