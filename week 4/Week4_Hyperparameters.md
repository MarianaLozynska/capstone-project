# Hyperparameter Reflection

## 1. What are some key hyperparameters you have used or observed in neural networks? How did changing these affect your model's convergence, stability or performance?

**Learning rate** is the most impactful hyperparameter. Too high and the loss oscillates or diverges — the model overshoots minima. Too low and convergence is extremely slow, often getting stuck in poor local minima. A common starting point is ~0.001 (Adam default), with scheduling to reduce it as training progresses.

**Number of hidden layers and neurons** controls model capacity. Too few neurons and the network underfits — it cannot represent non-linear relationships. Adding layers and neurons increases capacity but risks overfitting, especially with small datasets.

**Batch size** affects both convergence speed and generalisation. Smaller batches introduce gradient noise that helps escape local minima but make training less stable. Larger batches give smoother gradients but may converge to sharp minima that generalise poorly.

**Regularisation** (weight decay, dropout) controls overfitting. Weight decay shrinks weights toward zero, smoothing the learned function. Dropout disables random neurons during training, forcing redundancy. Both improve generalisation but can underfit if too aggressive.

**Epochs** determines training duration. Too few leads to underfitting, too many to overfitting. Early stopping monitors validation loss to find the right balance.

## 2. Categorise the hyperparameters you've worked with. Which are continuous and which are discrete? How might the type influence the tuning method?

**Continuous**: learning rate, weight decay strength, dropout rate, momentum. These take any value in a range — small changes produce smooth effects on performance.

**Discrete**: number of layers, neurons per layer, batch size, activation function (ReLU/tanh/sigmoid), optimiser choice (Adam/SGD/RMSProp). These create discontinuous jumps — adding a layer fundamentally changes the function class.

**Tuning implications**: Continuous hyperparameters suit Bayesian Optimisation — GP surrogates model smooth response surfaces well. Discrete hyperparameters are better handled by grid search, random search, or categorical BO. Mixed spaces (continuous + discrete together) require specialised methods like TPE (tree-structured Parzen estimators).

This connects to my capstone: the BBO functions have continuous inputs in [0,1], making GP-based BO well-suited. If the functions had discrete inputs, the Matern kernel's smoothness assumption would break down.

## 3. If you are using or considering a neural network as a surrogate model in your capstone, how will your understanding of hyperparameter tuning influence your next set of decisions? Could you apply your BBO approach to improve your neural network performance directly?

I chose a GP over an NN surrogate because with 13–43 data points per function, an NN would overfit, and GPs provide built-in uncertainty σ(x) that acquisition functions require without additional calibration.

However, if I were to use an NN surrogate, my BBO methodology would apply directly to tuning it. NN hyperparameter tuning is itself a black-box optimisation problem: inputs are hyperparameters (learning rate, layer size, weight decay), output is validation error, no gradient is available with respect to hyperparameters, and each evaluation (full training run) is expensive. This is exactly the setting where BO excels — I could fit a GP over the hyperparameter space and use EI to propose the next configuration to try.

My capstone experience reinforces this directly. In Week 4, I discovered that tuning xi and kappa (continuous hyperparameters of EI/UCB) had diminishing returns — the deterministic acquisition functions converge to the same point regardless of parameter values. The key decision was matching the acquisition function to GP reliability: EI/UCB for functions where the GP is accurate (F1, F2, F4, F5, F7, F8), and **Thompson Sampling** for functions where the GP has failed to guide improvements (F3: 0/3 weeks improved, F6: W3 regressed). This acquisition function choice is itself a discrete hyperparameter (EI vs UCB vs TS) — the improvement came from changing a categorical decision, not tuning a continuous knob. This is exactly the kind of discrete hyperparameter that grid/random search handles better than gradient-based methods.

I also tuned the GP's **alpha** (noise regularisation) per function: alpha=1e-6 for well-modelled functions, alpha=0.1 for F3 where the GP was overfitting. This is analogous to tuning weight decay in a neural network — too little regularisation overfits, too much smooths away real signal.
