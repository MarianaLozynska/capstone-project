# Bayesian Optimization Capstone Project

## Overview

This capstone project involves optimizing eight unknown black-box functions using Bayesian Optimization techniques. The challenge simulates real-world machine learning scenarios where function evaluations are expensive and limited, requiring intelligent sampling strategies that balance exploration and exploitation.

## Project Goal

Find the maximum value for each of eight synthetic functions by iteratively querying points and updating optimization strategies based on observed outputs. Each function represents a different real-world optimization challenge, from drug discovery to hyperparameter tuning.

## The Challenge

- **8 black-box functions** with varying dimensionality (2D to 8D)
- **Weekly iterations**: Submit 1 point per function each week
- **Limited evaluations**: Each query is precious - strategic sampling is essential
- **Unknown landscapes**: No access to function equations or derivatives

## Function Descriptions

### Function 1 (2D) - Radiation Source Detection
Detect likely contamination sources in a two-dimensional area, similar to a radiation field, where only proximity yields a non-zero reading. The system uses Bayesian optimization to tune detection parameters and reliably identify both strong and weak sources.

**Challenge**: Extremely narrow peaks with most of the space yielding near-zero values.

### Function 2 (2D) - Noisy Black Box Model
A mystery ML model that takes two inputs and returns a log-likelihood score. The output is noisy, and depending on the starting point, optimization may get stuck in local optima.

**Challenge**: Multiple local maxima requiring careful exploration-exploitation balance.

### Function 3 (3D) - Drug Discovery
Testing combinations of three compounds to create a new medicine. The goal is to minimize adverse reactions (framed as maximizing the negative of side effects).

**Challenge**: Three-dimensional search space with safety constraints.

### Function 4 (4D) - Warehouse Product Placement
Optimally placing products across warehouses for a business with high online sales. The function has four hyperparameters to tune, with output reflecting the difference from an expensive baseline.

**Challenge**: Dynamic system with local optima requiring robust validation.

### Function 5 (4D) - Chemical Process Yield
Optimizing a four-variable function representing the yield of a chemical process. The function is typically unimodal with a single peak.

**Challenge**: Finding the global optimum in a four-dimensional space with a single dominant peak.

### Function 6 (5D) - Recipe Optimization
Optimizing a cake recipe with five ingredients (flour, sugar, eggs, butter, milk). Each recipe is evaluated with a combined score based on flavor, consistency, calories, waste, and cost, where lower is better (maximizing towards zero).

**Challenge**: Five-dimensional space with multiple competing objectives.

### Function 7 (6D) - ML Hyperparameter Tuning
Tuning six hyperparameters of an ML model (e.g., learning rate, regularization strength, number of hidden layers). The function returns the model's performance score.

**Challenge**: Six-dimensional hyperparameter space with complex interactions.

### Function 8 (8D) - High-Dimensional Optimization
Eight-dimensional function where each parameter affects the output independently. Finding the global maximum is difficult due to the curse of dimensionality.

**Challenge**: Very high-dimensional space where identifying strong local maxima is the practical strategy.

## Methodology

### Bayesian Optimization Approach

This project uses Gaussian Process (GP) regression with acquisition functions to intelligently select the next points to sample:

1. **Gaussian Process Surrogate Model**: Learns the function landscape from observed data, providing both predictions and uncertainty estimates
2. **Acquisition Functions**:
   - **Expected Improvement (EI)**: Balances exploitation and exploration
   - **Upper Confidence Bound (UCB)**: Optimistically samples high-uncertainty regions

### Weekly Process

1. **Data Review**: Analyze all accumulated data (initial + previous weeks)
2. **Strategy Selection**: Choose exploration vs exploitation based on current knowledge
3. **Point Generation**: Use Bayesian Optimization to suggest next sampling points
4. **Submission**: Submit 8 points (one per function) via the portal
5. **Reflection**: Document strategy, reasoning, and results

## Project Structure

```
capstone_project/
├── README.md
├── initial_data/          # Initial function data (.npy files)
│   ├── function_1/
│   ├── function_2/
│   └── ...
├── week 1/               # Week 1 analysis and exploration
│   └── function_analysis.ipynb
└── week 2/               # Week 2 Bayesian Optimization
    └── week2_bayesian_optimization.ipynb
```

## Key Insights

### Week 1 Results
- **Function 1**: Found first significant signal (0.09787) after exploring sparse landscape
- **Functions 5, 6**: Achieved strong improvements through exploitation
- **Functions 2, 3, 4**: Identified need for course correction in Week 2
- **Functions 7, 8**: Converging on optimal regions with fine-tuning

### Optimization Strategies
- **Exploration** (high xi/kappa): Used for sparse or poorly understood functions
- **Exploitation** (low xi/kappa): Used when near known peaks
- **Adaptive approach**: Strategy changes week-to-week based on results

## Results Summary

The Bayesian Optimization approach successfully improved or maintained performance across all functions, demonstrating effective balance between exploration and exploitation. The GP model's uncertainty estimates guided sampling toward promising regions while avoiding redundant evaluations.

## Tools & Libraries

- **Python 3.12**
- **NumPy**: Data manipulation
- **scikit-learn**: Gaussian Process regression
- **SciPy**: Optimization (L-BFGS-B)
- **Matplotlib**: Visualization (optional)

## Running the Code

```bash
# Navigate to week directory
cd "week 2"

# Open Jupyter notebook
jupyter notebook week2_bayesian_optimization.ipynb

# Run all cells to generate Week 2 recommendations
```

## Future Work

For subsequent weeks:
1. Add new week's results to the notebook
2. Update combined data (initial + all previous weeks)
3. Adjust exploration/exploitation parameters based on convergence
4. Run Bayesian Optimization to generate next week's recommendations

---

*This project demonstrates practical application of Bayesian Optimization in scenarios with expensive function evaluations, mirroring real-world ML challenges in research and industry.*
