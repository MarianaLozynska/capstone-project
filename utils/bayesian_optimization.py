"""
Bayesian Optimization utilities for capstone project
Reusable functions for all weeks
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.optimize import minimize
from scipy.stats import norm


def expected_improvement(X, X_sample, Y_sample, gp, xi=0.01):
    """
    Expected Improvement acquisition function

    Args:
        X: Points to evaluate (n_points, n_dims)
        X_sample: Observed inputs (n_samples, n_dims)
        Y_sample: Observed outputs (n_samples,)
        gp: Fitted Gaussian Process model
        xi: Exploration parameter (higher = more exploration)

    Returns:
        ei: Expected Improvement values (n_points,)
    """
    mu, sigma = gp.predict(X, return_std=True)
    mu_sample_opt = np.max(Y_sample)
    sigma = sigma.reshape(-1, 1)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei


def upper_confidence_bound(X, X_sample, Y_sample, gp, kappa=2.0):
    """
    Upper Confidence Bound acquisition function

    Args:
        X: Points to evaluate (n_points, n_dims)
        X_sample: Observed inputs (n_samples, n_dims)
        Y_sample: Observed outputs (n_samples,)
        gp: Fitted Gaussian Process model
        kappa: Exploration parameter (higher = more exploration)

    Returns:
        ucb: UCB values (n_points,)
    """
    mu, sigma = gp.predict(X, return_std=True)
    return mu + kappa * sigma


def thompson_sampling(X_sample, Y_sample, bounds, gp=None, n_candidates=5000, alpha=1e-6):
    """
    Thompson Sampling acquisition: draw a random function from the GP posterior,
    then pick the candidate that maximises it.

    Unlike EI/UCB which always converge to the same point (deterministic optimisation
    of a fixed acquisition surface), TS draws a *different* random function each time,
    so it naturally explores diverse regions.

    Args:
        X_sample: Observed inputs (n_samples, n_dims)
        Y_sample: Observed outputs (n_samples,)
        bounds: Bounds for each dimension as numpy array (n_dims, 2)
        gp: Pre-fitted GP (if None, fits a new one)
        n_candidates: Number of random candidates to evaluate
        alpha: GP noise parameter

    Returns:
        next_point: Suggested next point (n_dims,)
        gp: Fitted GP model
    """
    dim = X_sample.shape[1]

    if gp is None:
        gp = fit_gp(X_sample, Y_sample, alpha=alpha)

    # Generate random candidates within bounds
    candidates = np.random.uniform(
        bounds[:, 0], bounds[:, 1], size=(n_candidates, dim)
    )

    # Draw one sample from the GP posterior at all candidates
    # sample_y returns shape (n_candidates, n_samples_drawn)
    sample = gp.sample_y(candidates, n_samples=1, random_state=None)

    # Pick the candidate where the drawn function is highest
    best_idx = np.argmax(sample[:, 0])
    return candidates[best_idx], gp


def fit_gp(X_sample, Y_sample, alpha=1e-6):
    """
    Fit Gaussian Process model to observed data

    Args:
        X_sample: Observed inputs (n_samples, n_dims)
        Y_sample: Observed outputs (n_samples,)

    Returns:
        gp: Fitted Gaussian Process model
    """
    dim = X_sample.shape[1]
    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(dim), nu=2.5)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        alpha=alpha,
        normalize_y=True
    )
    gp.fit(X_sample, Y_sample)
    return gp


def propose_next_point(X_sample, Y_sample, bounds, acq_func='EI', xi=0.01, kappa=2.0,
                       n_restarts=25, alpha=1e-6, n_candidates=5000):
    """
    Proposes the next sampling point using Bayesian Optimization

    Args:
        X_sample: Observed inputs (n_samples, n_dims)
        Y_sample: Observed outputs (n_samples,)
        bounds: Bounds for each dimension [(min, max), ...] as numpy array
        acq_func: 'EI', 'UCB', or 'TS' (Thompson Sampling)
        xi: Exploration parameter for EI
        kappa: Exploration parameter for UCB
        n_restarts: Number of random restarts for EI/UCB optimization
        alpha: GP noise parameter (higher = more regularisation)
        n_candidates: Number of random candidates for Thompson Sampling

    Returns:
        next_point: Suggested next point to sample (n_dims,)
        gp: Fitted Gaussian Process model
    """
    dim = X_sample.shape[1]

    # Fit Gaussian Process
    gp = fit_gp(X_sample, Y_sample, alpha=alpha)

    # Thompson Sampling: fundamentally different — draw random function, optimise it
    if acq_func == 'TS':
        return thompson_sampling(X_sample, Y_sample, bounds, gp=gp,
                                 n_candidates=n_candidates, alpha=alpha)

    # EI / UCB: deterministic optimisation of acquisition surface
    min_val = 1e10
    min_x = None

    def min_obj(X):
        X = X.reshape(-1, dim)
        if acq_func == 'EI':
            return -expected_improvement(X, X_sample, Y_sample, gp, xi).flatten()
        else:
            return -upper_confidence_bound(X, X_sample, Y_sample, gp, kappa).flatten()

    # Multiple random restarts
    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
        if res.fun < min_val:
            min_val = res.fun[0] if isinstance(res.fun, np.ndarray) else res.fun
            min_x = res.x

    return min_x, gp


def get_strategy(func_id, week_number=2):
    """
    Get exploration/exploitation strategy for a function

    Args:
        func_id: Function number (1-8)
        week_number: Current week number (for future customization)

    Returns:
        dict: Strategy parameters (acq_func, xi, kappa)
    """
    # Week 2 strategies based on Week 1 learnings
    strategies = {
        1: {'acq_func': 'EI', 'xi': 0.001, 'kappa': 1.0},   # Exploit discovery
        2: {'acq_func': 'EI', 'xi': 0.01, 'kappa': 1.5},    # Course correction
        3: {'acq_func': 'EI', 'xi': 0.01, 'kappa': 1.5},    # Course correction
        4: {'acq_func': 'EI', 'xi': 0.01, 'kappa': 1.5},    # Course correction
        5: {'acq_func': 'EI', 'xi': 0.001, 'kappa': 1.0},   # Continue climbing
        6: {'acq_func': 'EI', 'xi': 0.001, 'kappa': 1.0},   # Continue climbing
        7: {'acq_func': 'UCB', 'xi': 0.0001, 'kappa': 0.5}, # Fine refinement
        8: {'acq_func': 'UCB', 'xi': 0.0001, 'kappa': 0.5}, # Fine refinement
    }

    return strategies.get(func_id, {'acq_func': 'EI', 'xi': 0.01, 'kappa': 1.5})
