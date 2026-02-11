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
- **What makes this unique**: Data-driven strategy adjustment — instead of fixed parameters, sensitivity analysis diagnoses each function's state and prescribes targeted exploration levels

---

## Project Structure

```
capstone_project/
├── README.md
├── utils/
├── initial_data/
├── week 1/
├── week 2/
└── week 3/  
```

## Results Summary

| Function | Dims | Week 1 | Week 2 | Trend |
|----------|------|--------|--------|-------|
| 1 | 2D | 0.0979 | ~0 | Regression (lost narrow peak) |
| 2 | 2D | 0.1495 | 0.6138 | Improving |
| 3 | 3D | -0.0664 | -0.0499 | Slow improvement |
| 4 | 4D | -4.0313 | 0.3523 | Breakthrough |
| 5 | 4D | 1548.81 | 1688.07 | Improving |
| 6 | 5D | -0.9229 | -0.5210 | Improving |
| 7 | 6D | 1.7020 | 1.7845 | Improving |
| 8 | 8D | 8.9296 | 9.6493 | Improving |

## Tools & Libraries

- **Python 3.12**
- **NumPy**: Data manipulation
- **scikit-learn**: Gaussian Process regression (Matern kernel)
- **SciPy**: Optimization (L-BFGS-B for acquisition function maximization)
- **Matplotlib**: Visualization

## Running the Code

```bash
# Navigate to desired week
cd "week 3"

# Open Jupyter notebook
jupyter notebook week3_function_analysis.ipynb

# Run all cells to generate recommendations
```

Each week's notebook loads the previous week's clean data, adds new results, and generates next recommendations.

---

*This project demonstrates practical application of Bayesian Optimization in scenarios with expensive function evaluations, mirroring real-world ML challenges in research and industry.*
