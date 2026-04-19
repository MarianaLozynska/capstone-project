"""Generate score progression chart across 12 weeks for all 8 functions."""
import matplotlib.pyplot as plt
import numpy as np

week_results = {
    1: [0.0979, 3.1e-39, 0.3255, 0.4147, 0.6114, 0.7062, 0.8787, 0.8992, 0.8975, 0.9648, 0.7021, None],
    2: [0.5567, 0.6138, 0.0480, 0.6050, 0.5471, 0.5725, 0.5824, 0.1273, -0.0015, 0.1945, 0.2400, None],
    3: [-0.0593, -0.0499, -0.1750, -0.0056, -0.0701, -0.0079, -0.0077, -0.0072, -0.0075, -0.0046, -0.0082, None],
    4: [-4.4163, 0.3523, 0.4226, 0.6723, 0.7101, 0.4657, 0.7243, 0.7152, -0.0432, 0.6730, 0.6942, None],
    5: [1231.61, 1688.07, 7599.50, 8662.48, 8290.38, 8643.15, 1616.64, 4440.52, 4440.52, 32.0025, 1616.64, None],
    6: [-0.5920, -0.5210, -1.0560, -0.5902, -0.9899, -0.5207, -0.6062, -0.5301, -0.5672, -0.5935, -0.5707, None],
    7: [1.3646, 1.7845, 1.3720, 1.7718, 1.4720, 1.8536, 1.8617, 1.8665, 1.8696, 1.8742, 1.8787, None],
    8: [9.5863, 9.6493, 9.6972, 9.7627, 9.7274, 9.7402, 9.7614, 9.7621, 9.7747, 9.7882, 9.7963, None],
}

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Running Best Score per Function — Weeks 1-12", fontsize=16, fontweight='bold')

weeks = list(range(1, 13))

for idx, fid in enumerate(range(1, 9)):
    ax = axes[idx // 4, idx % 4]
    scores = week_results[fid]

    running_best = []
    current = float('-inf')
    for s in scores:
        if s is not None and s > current:
            current = s
        running_best.append(current if current != float('-inf') else None)

    # Plot weekly scores
    valid_weeks = [w for w, s in zip(weeks, scores) if s is not None]
    valid_scores = [s for s in scores if s is not None]
    ax.scatter(valid_weeks, valid_scores, alpha=0.4, color='steelblue', label='Weekly query', zorder=2)

    # Plot running best
    valid_running = [(w, b) for w, b in zip(weeks, running_best) if b is not None and b != float('-inf')]
    if valid_running:
        rw, rb = zip(*valid_running)
        ax.plot(rw, rb, color='darkorange', linewidth=2, label='Running best', zorder=3)

    ax.set_title(f"F{fid}", fontweight='bold')
    ax.set_xlabel("Week")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best')
    ax.set_xticks([1, 3, 5, 7, 9, 11])

plt.tight_layout()
plt.savefig("visuals/score_progression.png", dpi=120, bbox_inches='tight')
print("Saved: visuals/score_progression.png")
