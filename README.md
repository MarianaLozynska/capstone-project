# Bayesian Optimization Capstone Project

## Section 1: Project Overview

This capstone project tackles the **Black-Box Optimization (BBO) challenge**: optimizing eight unknown functions where we have no access to equations, derivatives, or internal structure. The only way to learn about each function is by querying input points and observing outputs — simulating real-world scenarios where function evaluations are expensive and limited.

**Goal**: Find the maximum value for each of eight synthetic functions by iteratively submitting one query point per function per week, using intelligent sampling strategies that balance exploration and exploitation.

**Real-world relevance**: BBO mirrors critical ML challenges — hyperparameter tuning, drug discovery, chemical process optimization, and A/B testing — where each evaluation costs time, money, or computational resources. Learning to optimize efficiently under uncertainty is a core skill in applied machine learning.

**Career relevance**: This project builds practical experience with Gaussian Processes, acquisition functions, and adaptive optimization — techniques directly applicable to ML engineering, data science, and research roles. The iterative decision-making process (analyze → strategize → submit → reflect) mirrors real production ML workflows.

---

## Section 2: Inputs and Outputs

### Inputs
- **Format**: Continuous values in the range [0, 1] for each dimension
- **Dimensions**: Vary per function (2D to 8D)
- **Constraints**: One query point per function per week
- **Submission format**: Hyphen-separated values, e.g., `0.074136-0.522573`

| Function | Dimensions | Domain |
|----------|-----------|--------|
| 1 | 2D | [0,1]² |
| 2 | 2D | [0,1]² |
| 3 | 3D | [0,1]³ |
| 4 | 4D | [0,1]⁴ |
| 5 | 4D | [0,1]⁴ |
| 6 | 5D | [0,1]⁵ |
| 7 | 6D | [0,1]⁶ |
| 8 | 8D | [0,1]⁸ |

### Outputs
- **Response value**: A single scalar score returned by the black-box function
- **Performance signal**: Higher values indicate better performance (maximization)
- **No gradient information**: Only the function value is returned

### Example

```
Input:  0.074136-0.522573        (2D query for Function 1)
Output: 0.09786978019774146      (scalar response)
```

---

## Section 3: Challenge Objectives

**Primary goal**: **Maximize** each function's output value.

### Constraints and Limitations
- **Limited queries**: Only 1 point per function per week — every query must be strategic
- **Unknown function structure**: No equations, no derivatives, no knowledge of smoothness or modality
- **Response delay**: Results returned after each weekly submission, preventing rapid iteration
- **Varying complexity**: Functions range from 2D (manageable) to 8D (curse of dimensionality)
- **Diverse landscapes**: Some functions have narrow peaks (Function 1), plateaus (Function 3), or steep gradients (Function 5)
- **No resets**: Each submitted point permanently consumes a query budget

---

## Section 4: Technical Approach

### Core Method: Bayesian Optimization with Gaussian Processes

The project uses **Gaussian Process (GP) regression** as a surrogate model combined with **acquisition functions** to select query points:

1. **GP Surrogate Model**: Learns the function landscape from observed data, providing predictions (μ) and uncertainty estimates (σ) at every point
2. **Acquisition Functions**:
   - **Expected Improvement (EI)**: Balances exploitation and exploration via xi parameter
   - **Upper Confidence Bound (UCB)**: Optimistically samples high-uncertainty regions via kappa parameter
3. **Optimization**: L-BFGS-B algorithm with multiple restarts to maximize acquisition functions

### Exploration vs Exploitation Balance

The xi and kappa parameters control this trade-off:
- **High exploration** (xi=0.1): Search broadly for new peaks — used when stuck or lost
- **Moderate exploration** (xi=0.01): Balanced search — default for improving functions
- **Exploitation** (xi=0.001): Focus near known best — used when near convergence
- **Fine-tuning** (UCB, kappa=0.5): Small adjustments — used for nearly converged functions

Strategy is adjusted per function per week based on performance trends and sensitivity analysis.

### Week 1: Manual Exploration with PCA

- Visualized initial data using **PCA** to reduce high-dimensional functions to 2D
- Identified promising regions through visual analysis of projected landscapes
- Selected query points based on PCA-projected patterns and cluster analysis
- **Limitation discovered**: PCA loses information in higher dimensions, leading to suboptimal recommendations for Functions 2-4

### Week 2: Bayesian Optimization

- Switched to **Bayesian Optimization** working in native dimensionality (no PCA)
- Applied function-specific strategies based on Week 1 performance:
  - Functions with poor Week 1 results → higher exploration (EI, xi=0.01)
  - Functions with good Week 1 results → exploitation (EI, xi=0.001)
- **Results**: 6 out of 8 functions improved
  - Function 4 breakthrough: -4.03 → 0.352 (GP found optimal region in 4D)
  - Function 1 regression: 0.098 → ~0 (exploitation too aggressive, lost narrow peak)

### Week 3: Sensitivity Analysis + Adjusted Strategies

- Implemented **sensitivity analysis**: vary each dimension independently from best point, measure GP gradient magnitude
- Diagnosed stuck functions vs converging functions using gradient patterns:
  - Flat gradients + worsening → lost peak (Function 1) → high exploration (xi=0.1)
  - Flat gradients + plateau → stuck (Function 3) → increased exploration (xi=0.05)
  - Steep gradients + at bounds → critical dims identified (Function 5) → exploitation
  - Flat gradients + high value → near convergence (Functions 7-8) → fine-tuning (UCB)
- **Results**: 4 out of 8 functions improved
  - Function 1 recovery: ~0 → 0.3255 (high exploration found the narrow peak again)
  - Function 5 breakthrough: 1688 → 7600 (pushing dims 3&4 to boundary)
- **What makes this unique**: Data-driven strategy adjustment — instead of fixed parameters, sensitivity analysis diagnoses each function's state and prescribes targeted exploration levels

### Week 4: GP Reliability Matching + Thompson Sampling

- Introduced **Thompson Sampling (TS)** for functions where the GP surrogate proved unreliable
- Matched acquisition function to GP reliability per function:
  - **EI** for functions with reliable GP (F1, F2, F4, F5) — exploit accurate surrogate
  - **Thompson Sampling** for unreliable GP (F3: 0/3 improved, F6: regressed in W3) — stochastic diversity escapes GP traps
  - **UCB** for fine-tuning near-converged functions (F7, F8) — conservative exploitation
- Tuned GP regularisation: `alpha=0.1` for Function 3 to prevent overfitting on noisy data
- **Results**: 5 out of 8 functions improved
  - Function 5 continued: 7600 → 8662 (pushing all dims to [1,1,1,1])
  - Function 4 climbing: 0.423 → 0.672 (steady EI exploitation)
  - Function 1 recovery confirmed: 0.326 → 0.415 (tight bounds around spike)

---

## Project Structure

```
capstone_project/
├── README.md
├── requirements.txt
├── .gitignore
├── utils/
│   ├── bayesian_optimization.py  # GP fitting, EI/UCB/Thompson Sampling
│   ├── data_utils.py             # Data loading, merging, .npz persistence
│   ├── sensitivity.py            # Per-dimension gradient diagnostics
│   └── visualization.py          # GP surfaces, PCA projections, validation plots
├── initial_data/                 # Seed data (8 functions × inputs/outputs .npy)
│   ├── function_1/ ... function_8/
├── week 1/                       # PCA-based manual exploration
├── week 2/                       # First BO iteration + Week 1-2 reflections
├── week 3/                       # Sensitivity analysis + adaptive strategies
├── week 4/                       # GP reliability matching + Thompson Sampling
└── week 5/                       # Current iteration (in progress)
```

Each week contains:
- `weekN_function_analysis.ipynb` — analysis notebook with strategy design and recommendations
- `weekN_clean_data.npz` — cumulative data snapshot for the next week to load
- `WeekN_Reflection.md` — strategy rationale and lessons learned

## Results Summary

| Function | Dims | Week 1 | Week 2 | Week 3 | Week 4 | Best | Trend |
|----------|------|--------|--------|--------|--------|------|-------|
| 1 | 2D | 0.0979 | ~0 | 0.3255 | 0.4147 | 0.4147 | Recovered (sensitivity-guided exploration) |
| 2 | 2D | 0.1495 | 0.6138 | 0.0480 | 0.6050 | 0.6138 | Stable (dim2 is flat, dim1 drives value) |
| 3 | 3D | -0.0664 | -0.0499 | -0.1750 | -0.0056 | -0.0056 | Difficult (Thompson Sampling improving) |
| 4 | 4D | -4.0313 | 0.3523 | 0.4226 | 0.6723 | 0.6723 | Steady climb |
| 5 | 4D | 1548.81 | 1688.07 | 7599.50 | 8662.48 | 8662.48 | Strong (pushing toward [1,1,1,1]) |
| 6 | 5D | -0.9229 | -0.5210 | -1.0560 | -0.5902 | -0.5210 | Volatile (TS exploring) |
| 7 | 6D | 1.7020 | 1.7845 | 1.3720 | 1.7718 | 1.7845 | Near converged |
| 8 | 8D | 8.9296 | 9.6493 | 9.6972 | 9.7627 | 9.7627 | Steady gains |

### Strategy Evolution

1. **Week 1** — PCA visual exploration: 3/8 improved. Learned PCA loses critical information in higher dimensions.
2. **Week 2** — Standard BO in native dimensions: 6/8 improved. Learned exploitation can be too aggressive (F1 lost narrow peak).
3. **Week 3** — Sensitivity-driven adaptive strategies: 4/8 improved. Learned to diagnose function state from gradient patterns before choosing strategy.
4. **Week 4** — GP reliability matching: 5/8 improved. Learned to match acquisition function type to surrogate model quality — Thompson Sampling for unreliable GPs, EI/UCB for reliable ones.

## Tools & Libraries

- **Python 3.12**
- **scikit-learn**: Gaussian Process regression (Matérn kernel) — surrogate model with calibrated uncertainty
- **SciPy**: L-BFGS-B bounded optimization for acquisition function maximization
- **NumPy**: Data manipulation and `.npz` persistence
- **Matplotlib**: GP surface plots, sensitivity profiles, validation visualizations

## Running the Code

```bash
# Install dependencies
pip install -r requirements.txt

# Navigate to desired week
cd "week 5"

# Open Jupyter notebook
jupyter notebook week5_function_analysis.ipynb

# Run all cells to generate recommendations
```

Each week's notebook loads the previous week's clean data, adds new results, and generates next recommendations.

---

*This project demonstrates practical application of Bayesian Optimization in scenarios with expensive function evaluations, mirroring real-world ML challenges in research and industry.*
