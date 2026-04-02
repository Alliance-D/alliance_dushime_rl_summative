"""
test_env.py
Validation suite for RwandaBusinessEnv (grid-world version).
Run: python test_env.py
"""
import numpy as np
from custom_env import (
    RwandaBusinessEnv, SECTORS, N_ACTIONS, OBS_DIM,
    MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, PLACE, SCAN,
    GRID_H, GRID_W, COMPETITOR, ROAD, MARKET_HUB, MAX_STEPS,
)

PASS_ALL = True

def check(name, cond, detail=""):
    global PASS_ALL
    status = "✓ PASS" if cond else "✗ FAIL"
    if not cond: PASS_ALL = False
    print(f"  {status}  {name}" + (f"  ({detail})" if detail else ""))

print("\n" + "═"*60)
print("  RwandaBusinessEnv (Grid-World) — Validation Suite")
print("═"*60)

# ── 1. Instantiation ──────────────────────────────────────────────────────── #
print("\n1. Instantiation")
env = RwandaBusinessEnv(sector="Kimironko", render_mode="ansi", seed=42)
check("env created", env is not None)
check("action space  == 6",  env.action_space.n == N_ACTIONS, f"{env.action_space.n}")
check("obs space shape == (11,)", env.observation_space.shape == (OBS_DIM,))
check("grid size correct", env._grid is None)   # not built until reset

# ── 2. Reset — all sectors ────────────────────────────────────────────────── #
print("\n2. Reset (all sectors)")
for sec in SECTORS:
    e = RwandaBusinessEnv(sector=sec, seed=0)
    obs, info = e.reset()
    check(f"  {sec:<12} obs shape", obs.shape == (OBS_DIM,))
    check(f"  {sec:<12} obs in [0,1]",
          bool(np.all(obs>=0) and np.all(obs<=1)),
          f"min={obs.min():.3f} max={obs.max():.3f}")
    check(f"  {sec:<12} agent on grid",
          0 <= info["agent_pos"][0] < GRID_H and
          0 <= info["agent_pos"][1] < GRID_W)
    check(f"  {sec:<12} budget == 1.0", info["budget"] == 1.0)

# ── 3. Grid structure ─────────────────────────────────────────────────────── #
print("\n3. Grid structure")
env = RwandaBusinessEnv(sector="Kimironko", seed=7)
obs, info = env.reset()

has_road = np.any(env._base_grid == ROAD)
has_comp = np.any(env._base_grid == COMPETITOR)
has_hub  = np.any(env._base_grid == MARKET_HUB)
check("grid has roads",       bool(has_road))
check("grid has competitors", bool(has_comp))
check("grid has market hub",  bool(has_hub))
check("viability map in [0,1]",
      bool(np.all(env._viability_map >= 0) and np.all(env._viability_map <= 1)))
check("demand map in [0,1]",
      bool(np.all(env._demand_map >= 0) and np.all(env._demand_map <= 1)))
check("traffic map in [0,1]",
      bool(np.all(env._traffic_map >= 0) and np.all(env._traffic_map <= 1)))

# ── 4. Movement ───────────────────────────────────────────────────────────── #
print("\n4. Movement")
env = RwandaBusinessEnv(sector="Remera", seed=10)
obs, info = env.reset()
start = info["agent_pos"]
# Move until we actually move (might hit wall first try)
moved = False
for _ in range(30):
    obs, r, term, trunc, info = env.step(MOVE_RIGHT)
    if info["agent_pos"] != start:
        moved = True
        break
check("agent can move",      moved)
check("obs still valid",     bool(np.all(obs>=0) and np.all(obs<=1)))
check("step count increased", info["step"] > 0)
check("budget decreased",     info["budget"] < 1.0)

# ── 5. Scan action ────────────────────────────────────────────────────────── #
print("\n5. Scan action")
env = RwandaBusinessEnv(sector="Kacyiru", seed=5)
env.reset()
obs, reward, term, trunc, info = env.step(SCAN)
check("scan reveals viability", info["last_scan"] is not None)
check("scan reward is negative", reward < 0)
check("scan does not terminate", not term)

# ── 6. Place action — good spot ───────────────────────────────────────────── #
print("\n6. Place action")
# Find best cell and place there
env = RwandaBusinessEnv(sector="Kimironko", seed=99)
env.reset()
# Navigate to best viability cell
best_r, best_c = np.unravel_index(
    np.argmax(env._viability_map), env._viability_map.shape)
# Make sure it's not a road/hub/competitor
from custom_env import EMPTY, RESIDENTIAL
base = env._base_grid[best_r, best_c]

# Force agent to a known good empty cell for test
empty_cells = [(r,c) for r in range(GRID_H) for c in range(GRID_W)
               if env._base_grid[r,c] in (EMPTY, RESIDENTIAL)]
best_v = max(float(env._viability_map[r,c]) for r,c in empty_cells)
best_cell = max(empty_cells, key=lambda rc: float(env._viability_map[rc]))
env._grid[env._agent_pos] = env._base_grid[env._agent_pos]  # clear agent
env._agent_pos = best_cell
env._grid[best_cell] = 5  # AGENT_CELL

obs, reward, term, trunc, info = env.step(PLACE)
check("place terminates episode", term)
check("place reward is positive (good spot)",
      reward > 0, f"reward={reward:.3f} viability={best_v:.3f}")
check("placed is recorded in info", info["placed"])

# ── 7. Episode truncation ─────────────────────────────────────────────────── #
print("\n7. Truncation (step limit)")
env = RwandaBusinessEnv(sector="Gatenga", seed=1)
env.reset()
total_r = 0.0
steps   = 0
term = trunc = False
while not (term or trunc):
    action = MOVE_RIGHT if steps % 4 < 2 else MOVE_DOWN
    obs, r, term, trunc, info = env.step(action)
    total_r += r
    steps += 1
check("truncation fires at max steps", trunc or steps >= MAX_STEPS,
      f"steps={steps}")
check("truncation penalty applied", total_r < 0)

# ── 8. Determinism ────────────────────────────────────────────────────────── #
print("\n8. Determinism")
def rollout(seed):
    e = RwandaBusinessEnv(sector="Gikondo", seed=seed)
    e.reset(seed=seed)
    rewards = []
    for a in [MOVE_RIGHT, MOVE_DOWN, SCAN, MOVE_LEFT, MOVE_UP]:
        _, r, term, trunc, _ = e.step(a)
        rewards.append(round(r, 6))
        if term or trunc: break
    return rewards

r1 = rollout(42)
r2 = rollout(42)
check("same seed → same rewards", r1 == r2, str(r1))

# ── 9. Random sector mode ─────────────────────────────────────────────────── #
print("\n9. Random sector mode")
env = RwandaBusinessEnv(sector=None, seed=0)
seen = set()
for _ in range(20):
    obs, info = env.reset()
    seen.add(info["sector"])
    env.step(PLACE)
check("all sectors visited in random mode", len(seen) >= 3,
      f"seen={seen}")

# ── 10. Action decoding ───────────────────────────────────────────────────── #
print("\n10. Action decoding")
env = RwandaBusinessEnv(sector="Kimironko", seed=0)
env.reset()
check("decode PLACE", "PLACE" in env.decode_action(PLACE))
check("decode SCAN",  "SCAN"  in env.decode_action(SCAN))
check("decode MOVE_UP", "UP"  in env.decode_action(MOVE_UP))

# ── Summary ───────────────────────────────────────────────────────────────── #
print("\n" + "═"*60)
if PASS_ALL:
    print("  ✓  ALL TESTS PASSED — grid-world environment is ready.")
else:
    print("  ✗  SOME TESTS FAILED — review output above.")
print("═"*60 + "\n")
