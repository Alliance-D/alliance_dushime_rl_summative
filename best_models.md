# Best Models Summary — Kigali Retail Navigator

## Best Performing Models

| Algorithm  | Best Run | Mean Reward | Std  | Train Time | Key Config |
|------------|----------|-------------|------|-----------|------------|
| **REINFORCE** | Run 12 | **+76.49** | **±19.95** | 1700s | lr=1e-3, hidden=256, ent=0.02, diff=0.2 |
| DQN | Run 38 | +17.55 | ±61.92 | 2184s | lr=5e-4, batch=128, expl=0.6, diff=0.2 |
| PPO | Run 13 | -26.73 | ±183.23 | 3197s | lr=5e-3, n_steps=1024, diff=0.2 |

**Winner: REINFORCE Run 12** — highest mean reward, lowest variance, fastest convergence.

---

## Hyperparameter Explanations

### DQN Parameters

| Parameter | Range Tested | Effect |
|-----------|--------------|--------|
| **Learning Rate (lr)** | 1e-3 to 3e-4 | Controls Q-value update step size. 5e-4 optimal: too high (1e-3) = unstable, too low (3e-4) = slow. |
| **Gamma (γ)** | 0.90, 0.95, 0.99 | Discount factor for future rewards. 0.99 essential for 4-placement sequence; lower values undervalue long-term completion bonus. |
| **Batch Size** | 32, 64, 128, 256 | Experiences per gradient update. 128 optimal: reduces noise, improves Q-value stability without reducing sample diversity. |
| **Buffer Size** | 50K, 100K, 200K | Replay buffer capacity. 100K is sweet spot: retains early successful placements throughout training. |
| **Exploration (expl)** | 0.2–0.7 | Epsilon decay fraction. 0.6 optimal: epsilon decays from 1.0→0.05 over 60% of training, preventing premature convergence. |
| **Target Update (tui)** | 500, 1000 | Steps between target network sync. 1000 slightly better; prevents Q-value oscillation in this environment size. |
| **Difficulty (diff)** | 0.2–0.7 | Rival density scaling. 0.2 best: cleaner reward signal, fewer invalid placements that add noise to gradients. |

---

### REINFORCE Parameters

| Parameter | Range Tested | Effect |
|-----------|--------------|--------|
| **Learning Rate (lr)** | 0.0005–0.002 | Policy gradient step size. 1e-3 optimal: sufficient for convergence without oscillation. |
| **Gamma (γ)** | 0.90, 0.95, 0.99 | Discount factor. 0.99 critical: completion bonus worth 0.99^100 ≈ 0.37 at step 100. Lower values (0.95, 0.90) degrade catastrophically (-700 mean). |
| **Entropy Coef (ent)** | 0.005–0.02 | Exploration bonus strength. 0.02 optimal: higher entropy prevents early policy collapse into narrow distributions. Lower (0.01) leads to premature convergence (mean -639). |
| **Hidden Size** | 64, 128, 256 | Policy network width. 256 best: sufficient capacity for 56-dim observations without overfitting. |
| **Difficulty (diff)** | 0.2–0.6 | Rival density. 0.2 best: balanced competition, accessible valid cells, clean viability gradients. |

---

### PPO Parameters

| Parameter | Range Tested | Effect |
|-----------|--------------|--------|
| **Learning Rate (lr)** | 1e-4–5e-3 | Policy update step size. 5e-3 unexpectedly best: faster escape from survey-oscillation traps. |
| **Gamma (γ)** | 0.995, 0.99 | Discount factor. 0.99 standard; same reasoning as DQN/REINFORCE. |
| **N Steps** | 512, 1024, 2048 | Rollout length per update. 1024 optimal: balance between GAE advantage quality and computational cost. |
| **Batch Size** | 64, 128 | Minibatch size for gradient updates. 128 > 64 for this environment. |
| **N Epochs** | 10, 15 | Gradient passes per rollout. 15 > 10: more passes improve convergence on sparse rewards. |
| **Clip Range (clip)** | 0.10, 0.15, 0.20, 0.30 | Clipped objective constraint. 0.15 optimal: allows policy updates without instability. |
| **Entropy Coef (ent)** | 0.001–0.05 | Exploration regularization. 0.01 best; too low = determinism, too high = excessive exploration. |
| **GAE Lambda (λ)** | 0.80, 0.95, 0.96 | Advantage trace decay. 0.95–0.96 > 0.80: longer traces capture 4-stage placement dependencies. |
| **Difficulty (diff)** | 0.2–0.7 | Rival density. 0.2 best: consistent across all algorithms. |

---

## Key Insights

1. **Difficulty = 0.2 optimal across all algorithms**: Provides meaningful competition without sparse valid placements.

2. **Exploration vs exploitation balance critical**: DQN needs long exploration (0.6), REINFORCE needs entropy (0.02), PPO needs tuned clip range (0.15).

3. **Long-horizon discounting essential**: gamma = 0.99 across all algorithms because multi-step placement tasks require valuing distant completion bonuses.

4. **REINFORCE wins for this environment**: Monte Carlo returns naturally fit sparse reward structure (placements as only signal); entropy regularization prevents collapse.

5. **PPO underperforms due to evaluation protocol**: Deterministic evaluation hides learned placement policy. Would match REINFORCE with stochastic sampling.

---

## CSV Results Files

- **dqn_results.csv**: 39 runs with mean/std rewards and hyperparameters
- **reinforce_results.csv**: 13 runs with mean/std rewards and hyperparameters  
- **ppo_results.csv**: 14 runs with mean/std rewards and hyperparameters