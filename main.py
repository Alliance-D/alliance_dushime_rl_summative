"""
main.py  –  Kigali Retail Navigator v3
========================================
Evaluate the best trained RL agent with GUI + terminal verbose output.

Agent places 4 businesses sequentially:
  Grocery → Pharmacy → Restaurant → Salon
Each phase switches which business type is recognised as a rival.

Usage
-----
python main.py                          # auto-detect best saved model
python main.py --algo ppo --episodes 5
python main.py --algo dqn --episodes 5
python main.py --algo reinforce --episodes 5
python main.py --algo ppo --no-render   # terminal only
python main.py --export-api             # save api_spec.json
"""

from __future__ import annotations
import os, sys, json, argparse, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

MODELS_DQN = os.path.join(ROOT, "models", "dqn")
MODELS_PG  = os.path.join(ROOT, "models", "pg")

GS = 15

ACT_LABELS = {
    0: "Move ↑",
    1: "Move ↓",
    2: "Move ←",
    3: "Move →",
    4: "Survey",
    5: "PLACE",
}

BUSINESS_NAMES = {
    0: "Grocery",
    1: "Pharmacy",
    2: "Restaurant",
    3: "Salon",
}

# Best run IDs from training sweep results
BEST_RUNS = {"dqn": 2, "ppo": 8, "reinforce": 3}

# REINFORCE hidden sizes — must match what each run saved
# From REINFORCE_SWEEP in pg_training.py:
# runs 0-7: hidden=128, run 7: hidden=64, runs 8-9: hidden=256
REINFORCE_HIDDEN = {
    0:128, 1:128, 2:128, 3:128, 4:128,
    5:128, 6:128, 7:64,  8:256, 9:256,
}

TRAINING_STATS = {
    "dqn": {
        "best_run": 2,
        "note": "lr=2e-3, γ=0.99, buffer=50K, batch=32, expl=0.40, diff=0.3",
    },
    "ppo": {
        "best_run": 8,
        "note": "lr=3e-4, gae_λ=0.80, clip=0.2, ent=0.01, n_steps=2048, diff=0.6",
    },
    "reinforce": {
        "best_run": 3,
        "note": "lr=1e-3, γ=0.95, entropy=0.01, hidden=128, diff=0.3",
    },
}


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model(algo: str):
    """Load best saved model. Returns (model, model_type_str)."""
    best_run = BEST_RUNS.get(algo, 0)

    if algo == "dqn":
        from stable_baselines3 import DQN
        candidates = [
            os.path.join(MODELS_DQN, f"dqn_run_{best_run:02d}_best", "best_model.zip"),
            os.path.join(MODELS_DQN, f"dqn_run_{best_run:02d}.zip"),
        ]
        if os.path.exists(MODELS_DQN):
            for f in sorted(os.listdir(MODELS_DQN)):
                if f.endswith(".zip"):
                    candidates.append(os.path.join(MODELS_DQN, f))
        for path in candidates:
            if os.path.exists(path):
                print(f"  Loading DQN       : {path}")
                return DQN.load(path), "sb3"

    elif algo == "ppo":
        from stable_baselines3 import PPO
        candidates = [
            os.path.join(MODELS_PG, f"ppo_run_{best_run:02d}_best", "best_model.zip"),
            os.path.join(MODELS_PG, f"ppo_run_{best_run:02d}.zip"),
        ]
        if os.path.exists(MODELS_PG):
            for f in sorted(os.listdir(MODELS_PG)):
                if f.startswith("ppo") and f.endswith(".zip"):
                    candidates.append(os.path.join(MODELS_PG, f))
        for path in candidates:
            if os.path.exists(path):
                print(f"  Loading PPO       : {path}")
                return PPO.load(path), "sb3"

    elif algo == "reinforce":
        import torch
        sys.path.insert(0, os.path.join(ROOT, "training"))
        from pg_training import REINFORCEAgent
        from environment.custom_env import KigaliRetailEnv

        _env    = KigaliRetailEnv()
        obs_dim = _env.observation_space.shape[0]
        act_dim = _env.action_space.n
        _env.close()

        # Build candidate list sorted so best_run is tried first
        candidates = []
        if os.path.exists(MODELS_PG):
            for f in sorted(os.listdir(MODELS_PG)):
                if f.startswith("reinforce") and f.endswith(".pt"):
                    try:
                        run_id = int(f.split("_run_")[1].split(".")[0])
                    except (IndexError, ValueError):
                        run_id = 0
                    candidates.append((os.path.join(MODELS_PG, f), run_id))

        candidates.sort(key=lambda x: (0 if x[1] == best_run else 1, x[1]))

        for path, run_id in candidates:
            if os.path.exists(path):
                hidden = REINFORCE_HIDDEN.get(run_id, 128)
                agent  = REINFORCEAgent(obs_dim, act_dim, hidden=hidden)
                try:
                    agent.load(path.replace(".pt", ""))
                    print(f"  Loading REINFORCE : {path} (hidden={hidden})")
                    return agent, "reinforce"
                except RuntimeError as e:
                    print(f"  [WARN] {os.path.basename(path)}: {e}")
                    continue

    raise FileNotFoundError(
        f"No saved {algo.upper()} model found in models/.\n"
        f"Run training first:\n"
        f"  python training/dqn_training.py --sweep\n"
        f"  python training/pg_training.py --algo ppo --sweep\n"
        f"  python training/pg_training.py --algo reinforce --sweep"
    )


def auto_algo() -> str:
    for algo in ("ppo", "dqn", "reinforce"):
        try:
            load_model(algo)
            return algo
        except FileNotFoundError:
            continue
    return "ppo"


def predict(model, model_type: str, obs: np.ndarray) -> int:
    if model_type == "sb3":
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    else:   # REINFORCE
        import torch
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        probs = model.policy(obs_t).squeeze(0)
        probs = torch.clamp(probs, 1e-8, 1.0)
        probs = probs / probs.sum()
        return int(torch.argmax(probs).item())


# ── Evaluation ────────────────────────────────────────────────────────────────
def run_evaluation(model, model_type: str, algo: str,
                   n_episodes: int = 5, render: bool = True) -> None:
    from environment.custom_env import KigaliRetailEnv

    renderer = None
    if render:
        try:
            from environment.rendering import KigaliRenderer
            renderer = KigaliRenderer()
        except Exception as e:
            print(f"  [WARN] GUI unavailable ({e}). Terminal-only mode.")
            render = False

    stats = TRAINING_STATS.get(algo, {})
    W = 72

    print()
    print("═" * W)
    print(f"  Kigali Retail Navigator — {algo.upper()} Agent Evaluation")
    print(f"  Mission: Place 4 businesses across a Kigali sector")
    print("═" * W)
    print(f"  Algorithm  : {algo.upper()}")
    print(f"  Best run   : Run {stats.get('best_run','?'):02d}")
    print(f"  Config     : {stats.get('note','N/A')}")
    print(f"  Policy     : {'deterministic' if model_type=='sb3' else 'greedy argmax'}")
    print("═" * W)

    env = KigaliRetailEnv(difficulty=0.5)
    all_rewards      = []
    all_placed_counts= []
    all_viabs        = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep * 17 + 3)
        sector  = info["sector"]
        n_rivals= info["n_rivals"]

        print(f"\n{'─'*W}")
        print(f"  Episode {ep+1}/{n_episodes} | Sector: {sector}")
        print(f"  Rivals per type: {n_rivals}")
        print(f"{'─'*W}")
        print(f"  {'Step':>4}  {'Phase':>10}  {'Action':>10}  "
              f"{'Reward':>8}  {'Cumul.':>8}  Info")
        print(f"  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*22}")

        done    = False
        ep_r    = 0.0
        si      = 0
        ep_viabs= []

        while not done:
            phase = min(env._phase, 3)
            bname = BUSINESS_NAMES[phase]
            action = predict(model, model_type, obs)
            obs, reward, terminated, truncated, step_info = env.step(action)
            ep_r += reward
            si   += 1
            done  = terminated or truncated

            lbl   = ACT_LABELS.get(action, str(action))
            extra = ""

            if step_info.get("optimal"):
                v = step_info.get("viability", 0)
                n = step_info.get("business", "?")
                extra = f"✓ OPTIMAL {n}  viab={v:.3f}"
                ep_viabs.append(float(v))
            elif step_info.get("decent"):
                v = step_info.get("viability", 0)
                n = step_info.get("business", "?")
                rival = "  rival nearby" if step_info.get("rival_nearby") else ""
                extra = f"~ decent {n}{rival}  viab={v:.3f}"
                ep_viabs.append(float(v))
            elif step_info.get("poor"):
                v = step_info.get("viability", 0)
                n = step_info.get("business", "?")
                extra = f"✗ poor {n}  viab={v:.3f}"
                ep_viabs.append(float(v))
            elif step_info.get("invalid_cell"):
                extra = "✗ invalid cell"
            elif step_info.get("timeout"):
                missed = step_info.get("missed_placements", "?")
                extra  = f"TIMEOUT  missed={missed}"
            elif step_info.get("wall"):
                extra = "wall"
            elif step_info.get("surveyed"):
                v = step_info.get("viability_here", 0)
                extra = f"viab here={v:.3f}"

            if step_info.get("next_business"):
                extra += f"  → next: {step_info['next_business']}"
            if step_info.get("episode_complete"):
                extra += "  ✓ ALL 4 PLACED"

            print(f"  {si:>4}  {bname:>10}  {lbl:>10}  "
                  f"{reward:>+8.3f}  {ep_r:>+8.3f}  {extra}")

            if renderer:
                renderer.draw(
                    grid=env.grid,
                    viability=env.viability,
                    sector_name=sector,
                    agent_pos=env._pos,
                    path=env._path,
                    placed_positions=env._placed_positions,
                    placed_types=env._placed_types,
                    phase=env._phase,
                    step=si,
                    visited=env._visited,
                    surveyed=env._surveyed,
                    status=f"[{bname}] {lbl}  {extra}",
                    episode=ep + 1,
                )
                time.sleep(0.30)

        # ── Episode summary ────────────────────────────────────────────────────
        n_placed  = len(env._placed_positions)
        best_v    = max(ep_viabs) if ep_viabs else 0.0
        explored  = len(env._visited)

        all_rewards.append(ep_r)
        all_placed_counts.append(n_placed)
        all_viabs.append(best_v)

        print(f"\n  Episode {ep+1} Summary")
        print(f"  {'─'*40}")
        print(f"  Total reward       : {ep_r:+.2f}")
        print(f"  Businesses placed  : {n_placed}/4")
        for i, (pos, bt) in enumerate(
                zip(env._placed_positions, env._placed_types)):
            r2, c2 = pos
            v = float(env.viability[r2, c2, bt])
            print(f"    {i+1}. {BUSINESS_NAMES[bt]:12s}  at {pos}  "
                  f"viab={v:.3f}")
        print(f"  Steps taken        : {si}/300")
        print(f"  Cells explored     : {explored}/{GS*GS} "
              f"({explored*100//(GS*GS)}%)")

        # Hold final frame for 2 seconds
        if renderer and env._placed_positions:
            for _ in range(30 * 2):
                renderer.draw(
                    grid=env.grid,
                    viability=env.viability,
                    sector_name=sector,
                    agent_pos=env._pos,
                    path=env._path,
                    placed_positions=env._placed_positions,
                    placed_types=env._placed_types,
                    phase=env._phase,
                    step=si,
                    visited=env._visited,
                    surveyed=env._surveyed,
                    status=f"Episode {ep+1} complete!  "
                           f"{n_placed}/4 placed.",
                    episode=ep + 1,
                )
                renderer.clock.tick(30)

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'═'*W}")
    print(f"  EVALUATION COMPLETE — {algo.upper()} | {n_episodes} episodes")
    print(f"{'═'*W}")
    print(f"  Mean reward        : {np.mean(all_rewards):+.2f} "
          f"± {np.std(all_rewards):.2f}")
    print(f"  Best episode       : {max(all_rewards):+.2f}")
    print(f"  Worst episode      : {min(all_rewards):+.2f}")
    print(f"  Mean placed        : {np.mean(all_placed_counts):.2f}/4")
    print(f"  Mean best viability: {np.mean(all_viabs):.3f}")
    print(f"  Best viability seen: {max(all_viabs):.3f}")
    print(f"{'═'*W}")

    env.close()

    if renderer:
        import pygame
        print("\n  Close window to exit.")
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            renderer.draw(
                grid=env.grid,
                viability=env.viability,
                sector_name="Evaluation complete",
                agent_pos=env._pos,
                path=env._path,
                placed_positions=env._placed_positions,
                placed_types=env._placed_types,
                phase=env._phase,
                step=0,
                visited=env._visited,
                surveyed=env._surveyed,
                status="Evaluation complete — close window to exit",
            )
        renderer.close()


# ── API export ─────────────────────────────────────────────────────────────────
def export_api(algo: str) -> None:
    """Export a JSON spec showing production integration pathway."""
    spec = {
        "api_version": "3.0",
        "project":     "Kigali Retail Navigator — RL Policy API",
        "algorithm":   algo.upper(),
        "training":    TRAINING_STATS.get(algo, {}),
        "description": (
            "POST /predict with a 56-dim normalised observation vector. "
            "Returns action 0-5. Call repeatedly per step. "
            "Action 5 (PLACE) means current cell is the recommended site "
            "for the current business phase."
        ),
        "endpoint": "POST /predict",
        "input": {
            "observation": {
                "type": "array", "length": 56, "dtype": "float32",
                "description": (
                    "5x5 local grid view (25) + position (2) + "
                    "viability (1) + context (28). "
                    "Built from GPS + OSM landmark scan."
                ),
            }
        },
        "output": {
            "action":        {"type": "int", "range": "0-5"},
            "action_name":   {"type": "string"},
            "is_placement":  {"type": "bool"},
            "current_phase": {"type": "int", "range": "0-3"},
            "business_type": {"type": "string"},
        },
        "phases":  BUSINESS_NAMES,
        "actions": ACT_LABELS,
        "production": [
            "1. Wrap model.predict() in FastAPI POST /predict",
            "2. Encode entrepreneur GPS + 50m OSM landmark scan → 56-dim obs",
            "3. React Native app calls API at each walking step",
            "4. When action=5: display recommended business + location on Kigali map",
            "5. Phase auto-advances: Grocery → Pharmacy → Restaurant → Salon",
            "6. Retrain quarterly as competitor landscape evolves",
        ],
        "latency":    "~0.3ms CPU inference — suitable for real-time mobile",
        "model_file": (
            f"models/{'dqn' if algo=='dqn' else 'pg'}/"
            f"{algo}_run_{BEST_RUNS.get(algo,0):02d}.zip"
        ),
    }
    path = os.path.join(ROOT, "api_spec.json")
    with open(path, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"\n  API spec exported → {path}")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Kigali Retail Navigator v3 RL agent")
    parser.add_argument("--algo",
                        choices=["auto", "dqn", "ppo", "reinforce"],
                        default="auto",
                        help="Algorithm to evaluate (default: auto-detect)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of evaluation episodes (default: 5)")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable Pygame GUI — terminal output only")
    parser.add_argument("--export-api", action="store_true",
                        help="Export policy as JSON API spec")
    args = parser.parse_args()

    algo = auto_algo() if args.algo == "auto" else args.algo
    print(f"\n  Algorithm : {algo.upper()}")

    try:
        model, model_type = load_model(algo)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    run_evaluation(
        model, model_type, algo,
        n_episodes=args.episodes,
        render=not args.no_render,
    )

    if args.export_api:
        export_api(algo)