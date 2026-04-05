# Kigali Retail Site Selection – RL Summative

> Reinforcement learning environment modelling an entrepreneur's challenge of selecting
> the optimal retail business location in Kigali, Rwanda.

---

## Problem Statement

Rwanda's urban economy is growing rapidly. Small retailers must decide *where* to set up
shop on a limited budget. Good placement; near foot-traffic generators, away from
saturated competition, accessible by road, is the difference between profitability and
closure within the first year. This project trains RL agents to learn optimal site-
selection strategies across four real Kigali sectors: **Kimironko, Nyabugogo, Remera,
and Masoro**.

---

## Repository Structure

```
kigali_rl/
├── environment/
│   ├── custom_env.py        # Custom Gymnasium environment
│   └── rendering.py         # Pygame visualiser + random-agent demo
├── training/
│   ├── dqn_training.py      # DQN (SB3) – training + 10-run sweep
│   └── pg_training.py       # REINFORCE (custom PyTorch) + PPO (SB3)
├── models/
│   ├── dqn/                 # Saved DQN checkpoints
│   └── pg/                  # Saved PG checkpoints
├── plots/                   # Auto-generated training curves
├── main.py                  # Evaluation entry point (GUI + terminal)
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/Alliance-D/student_name_rl_summative.git
cd student_name_rl_summative

# 2. Install dependencies
pip install -r requirements.txt

# 3. Visualise random agent (no model – demonstrates environment & GUI)
python environment/rendering.py

# 4. Train all models (hyperparameter sweeps – 3 × 10 runs each)
python training/dqn_training.py --sweep
python training/pg_training.py --algo reinforce --sweep
python training/pg_training.py --algo ppo --sweep

# 4b. Train best experiment (optimal HPs, 11th run per algorithm)
python training/best_experiment.py              # all 3 algorithms
python training/best_experiment.py --algo ppo  # single algorithm

# 5. Generate all required plots from training data
python generate_plots.py

# 6. Run best agent with GUI + verbose terminal output
python main.py --algo auto
python main.py --algo reinforce         # best overall model
python main.py --algo dqn --run 38     # specific run
python main.py --algo dqn --no-render  # terminal only

# 6. Export policy as JSON API spec (production integration demo)
python main.py --algo ppo --export-api
```

---

## Environment Details

| Property           | Value                                |
|--------------------|--------------------------------------|
| Grid size          | 15 × 15 per sector                   |
| Observation space  | Box(56,) – local 5×5 grid + context features|
| Action space       | Discrete(6)                          |
| Sectors            | Kimironko, Nyabugogo, Remera, Masoro |
| Landmark types     | 8 (market, road, residential, etc.)  |
| Rivals per type    | 6–18 (scales with difficulty)        |
| Max steps          | 400                                  |
| Business phases    | 4 (Grocery -> Pharmacy -> Restaurant -> Salon) |

### Action Space

| Action | ID | Meaning                          |
|--------|----|---------------------------------|
| Move UP | 0 | Move agent up (or hit wall)      |
| Move DOWN | 1 | Move agent down (or hit wall)    |
| Move LEFT | 2 | Move agent left (or wall)        |
| Move RIGHT | 3 | Move agent right (or wall)       |
| Survey | 4 | Gather viability info at current cell |
| Place | 5 | Place current business at this location |

### Observation Space (56-dim)

| Indices | Feature | Description |
|---------|---------|-------------|
| 0–24   | Local grid | 5×5 view around agent (rivals=1.0, placed=0.55, landmarks~0.4–0.8, road=0.3, empty=0.05) |
| 25–26  | Position | Agent row and col (normalized 0–1) |
| 27     | Viability | Local viability for current phase business |
| 28     | Step frac | Current step / max_steps |
| 29     | Sector ID | Which of 4 sectors (0–3 normalized) |
| 30     | Explored | Fraction of cells visited |
| 31     | Phase | Current business phase (0–3 normalized) |
| 32     | Rival dist | Nearest rival distance (normalized) |
| 33     | Foot traffic | Pedestrian activity at current cell |
| 34     | Road prox | Distance to nearest road (0=on road, 1=far) |
| 35–38  | Viability all | Local viability for all 4 business types |
| 39–42  | Rival count | Number of rival businesses in radius 1,2,3,4 |
| 43–46  | Landmark density | Count of each landmark type in sector |
| 47–55  | Reserved | Future features |

### Reward Structure

```
Movement (0–3):      0.0    (pure process, no reward or penalty)
Survey new cell:     0.0    (records viability info, no reward)
Survey repeat:       0.0    (true no-op, no gradient signal)
Invalid placement:  -1.0    (road or existing business cell)
Place (optimal):    +30 × viability_norm   (top 70th percentile, no close rival)
Place (decent):     +10 × viability_norm   (30–70th percentile)
Place (poor):       +1.0 to +4.0           (always positive, any placement beats nothing)
All 4 placed:       +50    (completion bonus)
Timeout:            -6.0 × missed_placements
```

**Design principle**: Only placement outcomes generate reward. All movement and
survey actions are intentionally zero-reward so the agent learns to move and
explore only in service of finding better placement locations, exactly as a real
entrepreneur would.

---

## Algorithms Implemented

| Algorithm  | Library           | Key Features                          |
|------------|-------------------|---------------------------------------|
| DQN        | Stable Baselines 3| Experience replay, target network, epsilon-greedy exploration |
| REINFORCE  | PyTorch (custom)  | Monte-Carlo returns, entropy regularization, gradient clipping |
| PPO        | Stable Baselines 3| Clipped surrogate loss, GAE advantage, policy entropy bonus |

---

## Hyperparameter Tuning

Each algorithm has 10 experimental runs with varying hyperparameters.
Results are saved to `training/{algo}_results.csv`.

### DQN key parameters tuned
- Learning rate, gamma, replay buffer size, batch size, exploration fraction, target update interval

### REINFORCE key parameters tuned
- Learning rate, gamma, entropy coefficient, network hidden size

### PPO key parameters tuned
- Learning rate, gamma, n_steps, batch_size, n_epochs, clip_range, entropy coefficient, GAE lambda

---

## Verifying Export-API Feature

To verify that the model can be exported as a production-ready API specification:

```bash
# 1. Export API spec for best PPO model
python main.py --algo ppo --export-api

# 2. Check generated file
cat api_spec.json

# Expected output: JSON with fields for endpoint, input/output schema, 
# algorithm details, and production deployment instructions
```

The `api_spec.json` includes:
- Endpoint specification (`POST /predict`)
- Input schema (56-dim observation vector)
- Output schema (action, phase, business type)
- Production deployment checklist
- Latency estimate (~0.3ms CPU inference)

---

## Environment Architecture

The environment consists of:

1. **Grid World**: 15×15 cell map representing a Kigali sector
2. **Cell Types**: Road, residential, marketplace, hospital, school, church, industrial, and 4 business types
3. **Agent**: Navigates grid, surveys cells, places businesses sequentially
4. **Dynamics**: 
   - Map is static (landmarks & rival businesses fixed at reset)
   - Competition phase switches after each placement
   - Viability computed from landmark proximity, rival density, road access
5. **Observation**: 56-dim vector encoding local grid, position, viability, phase context
6. **Action**: 6 discrete actions (move, survey, place)
7. **Reward**: Placement-based (no movement reward), scaled by viability & rival proximity
8. **Termination**: All 4 businesses placed OR 400 steps exceeded

---

## Production Integration

The trained policy can be served as a REST API:

```python
# api_spec.json is generated by: python main.py --export-api
# Deploy with FastAPI:
from fastapi import FastAPI
from stable_baselines3 import PPO
import numpy as np

app = FastAPI()
model = PPO.load("models/pg/ppo_best/best_model.zip")

@app.post("/predict")
def predict(observation: list[float]):
    """Predict action from observation vector (56-dim)."""
    obs = np.array(observation, dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "SURVEY", 5: "PLACE"}
    return {
        "action": int(action),
        "action_name": action_names.get(int(action), "UNKNOWN"),
        "is_placement": int(action) == 5,
    }
```

---

## Results Summary

Run `python generate_plots.py` to regenerate all plots from training CSVs.

### Required plots (rubric)

| File | Description |
|------|-------------|
| `cumulative_rewards_comparison.png` | All 3 algorithms, mean reward per run (subplots) |
| `convergence_comparison.png`        | All 3 on same axis, rolling mean vs experiment index |
| `generalisation_test.png`           | Performance vs difficulty level (proxy for generalisation) |
| `best_model_comparison.png`         | Best run from each algorithm, bar chart with error bars |
| `dqn_hyperparameter_sensitivity.png`| DQN: lr, exploration, difficulty vs performance |
| `dqn_rewards.png`                   | DQN per-run learning curves (all runs) |
| `dqn_td_loss.png`                   | DQN objective stability (rolling std of reward) |
| `ppo_rewards.png`                   | PPO per-run learning curves (all runs) |
| `ppo_entropy.png`                   | PPO policy entropy over training |
| `reinforce_rewards.png`             | REINFORCE per-run learning curves |
| `reinforce_entropy.png`             | REINFORCE policy entropy over training |

### Best results
- **REINFORCE Run 11**: mean=+69.37 ± 29.64 (best overall)
- **DQN Run 38**: mean=+17.55 ± 61.92
- **PPO Run 12**: mean=-173.81 (PPO training curves reach +50 to +75; eval collapse)

See `BEST_MODELS.md` for full analysis of why each best model performed best
and what the hyperparameter sweep reveals.

---

## Justification for Grid-World Approach

The 15×15 grid is not an arbitrary abstraction, each cell directly maps to a real
Kigali land-use category (market hub, taxi stop, residential zone, etc.) with
sector-specific landmark profiles derived from each neighbourhood's actual
commercial character. The environment's complexity comes from:
1. **Stochastic competitor placement** each episode (6–18 rivals per business type, scales with difficulty)
2. **Non-linear viability function** integrating landmark proximity, competition radius, and road access
3. **Sector-heterogeneity** (4 distinct profiles: Kimironko/Remera residential, Nyabugogo/CZ commercial)
4. **Multi-modal action space** (navigate, survey, place)

This is analogous to how OpenStreetMap-based urban planning simulators discretise
continuous space into parcels, a standard methodology in computational urban economics.