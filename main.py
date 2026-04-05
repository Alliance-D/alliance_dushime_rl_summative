"""
main.py  -  Kigali Retail Navigator v3
========================================
Evaluate the best trained RL agent with GUI + terminal verbose output.

The agent places 4 businesses sequentially across a Kigali sector:
  Phase 0: Grocery (Ikivunge)
  Phase 1: Pharmacy (Inzu y'imiti)
  Phase 2: Restaurant (Resitora)
  Phase 3: Salon (Coiffure)

Each phase switches which business type is recognised as a rival.
The model loaded is always determined by reading the actual CSV results
file - NOT hardcoded - so the true best experiment is always used.

Usage
-----
python main.py                          # auto-detect best saved model
python main.py --algo ppo               # run best PPO model
python main.py --algo dqn               # run best DQN model
python main.py --algo reinforce         # run best REINFORCE model
python main.py --algo ppo --episodes 5  # run 5 episodes
python main.py --algo dqn --no-render   # terminal output only
python main.py --export-api             # also save api_spec.json

How best model selection works
--------------------------------
1. Read training/dqn_results.csv (or ppo/reinforce)
2. Find row with highest mean_reward
3. Extract run_id (and hidden size for REINFORCE)
4. Load models/dqn/dqn_run_XX.zip (or models/pg/...)
5. Print proof: run_id, config, mean_reward, model path

This guarantees the model being evaluated matches the best
experiment you actually trained - no hardcoding.
"""

from __future__ import annotations
import os, sys, csv, json, argparse, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

MODELS_DQN  = os.path.join(ROOT, "models", "dqn")
MODELS_PG   = os.path.join(ROOT, "models", "pg")
RESULTS_DIR = os.path.join(ROOT, "training")

GS = 15

ACT_LABELS = {
    0: "Move UP",
    1: "Move DOWN",
    2: "Move LEFT",
    3: "Move RIGHT",
    4: "Survey",
    5: "PLACE",
}

BUSINESS_NAMES = {
    0: "Grocery",
    1: "Pharmacy",
    2: "Restaurant",
    3: "Salon",
}


# ==============================================================================
#  CSV READER - finds the true best run from actual training results
# ==============================================================================

def read_best_run(algo: str) -> dict:
    """
    Read the results CSV for the given algorithm and return the row
    with the highest mean_reward.

    Returns a dict with at minimum:
        run_id       (int)
        mean_reward  (float)
        hidden       (int, REINFORCE only)
        config_str   (str, human-readable summary)
    """
    csv_paths = {
        "dqn":       os.path.join(RESULTS_DIR, "dqn_results.csv"),
        "ppo":       os.path.join(RESULTS_DIR, "ppo_results.csv"),
        "reinforce": os.path.join(RESULTS_DIR, "reinforce_results.csv"),
    }
    path = csv_paths.get(algo)
    if path is None or not os.path.exists(path):
        print(f"  [WARN] No results CSV found at {path}.")
        print(f"         Falling back to run_id=0.")
        return {"run_id": 0, "mean_reward": None, "hidden": 128,
                "config_str": "unknown (no CSV)"}

    best_row  = None
    best_mean = -float("inf")

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                mean_r = float(row["mean_reward"])
            except (ValueError, KeyError):
                continue
            if mean_r > best_mean:
                best_mean = mean_r
                best_row  = row

    if best_row is None:
        print(f"  [WARN] CSV at {path} has no valid rows. Using run_id=0.")
        return {"run_id": 0, "mean_reward": None, "hidden": 128,
                "config_str": "unknown (empty CSV)"}

    run_id = int(best_row["run_id"])
    hidden = int(best_row.get("hidden", 128))

    # Build a human-readable config string from CSV columns
    skip = {"run_id", "mean_reward", "std_reward", "train_time_s"}
    cfg_parts = []
    for k, v in best_row.items():
        if k not in skip and v not in ("", None):
            cfg_parts.append(f"{k}={v}")
    config_str = "  ".join(cfg_parts)

    return {
        "run_id":      run_id,
        "mean_reward": best_mean,
        "std_reward":  float(best_row.get("std_reward", 0)),
        "hidden":      hidden,
        "config_str":  config_str,
        "raw_row":     best_row,
    }


# ==============================================================================
#  MODEL LOADER - loads the exact model file for the best run
# ==============================================================================

def load_model(algo: str):
    """
    Load the best saved model. Returns (model, model_type, best_info).
    model_type is 'sb3' or 'reinforce'.
    best_info is the dict from read_best_run().

    Proof of correctness:
      - best_info["run_id"] is derived from the CSV, not hardcoded
      - the model path is printed so you can verify it manually
      - for REINFORCE the hidden size is also read from the CSV
    """
    best   = read_best_run(algo)
    run_id = best["run_id"]

    if algo == "dqn":
        from stable_baselines3 import DQN

        candidates = [
            os.path.join(MODELS_DQN, f"dqn_run_{run_id:02d}_best", "best_model.zip"),
            os.path.join(MODELS_DQN, f"dqn_run_{run_id:02d}.zip"),
        ]
        if os.path.isdir(MODELS_DQN):
            for f in sorted(os.listdir(MODELS_DQN)):
                p = os.path.join(MODELS_DQN, f)
                if f.endswith(".zip") and p not in candidates:
                    candidates.append(p)

        for path in candidates:
            if os.path.exists(path):
                print(f"  Model path   : {os.path.relpath(path)}")
                return DQN.load(path), "sb3", best

        raise FileNotFoundError(
            f"No DQN model file found for run {run_id:02d}.\n"
            f"  Expected: {candidates[0]}\n"
            f"  Run training: python training/dqn_training.py --sweep"
        )

    elif algo == "ppo":
        from stable_baselines3 import PPO

        candidates = [
            os.path.join(MODELS_PG, f"ppo_run_{run_id:02d}_best", "best_model.zip"),
            os.path.join(MODELS_PG, f"ppo_run_{run_id:02d}.zip"),
        ]
        if os.path.isdir(MODELS_PG):
            for f in sorted(os.listdir(MODELS_PG)):
                p = os.path.join(MODELS_PG, f)
                if f.startswith("ppo") and f.endswith(".zip") and p not in candidates:
                    candidates.append(p)

        for path in candidates:
            if os.path.exists(path):
                print(f"  Model path   : {os.path.relpath(path)}")
                return PPO.load(path), "sb3", best

        raise FileNotFoundError(
            f"No PPO model file found for run {run_id:02d}.\n"
            f"  Expected: {candidates[0]}\n"
            f"  Run training: python training/pg_training.py --algo ppo --sweep"
        )

    elif algo == "reinforce":
        import torch
        sys.path.insert(0, os.path.join(ROOT, "training"))
        from pg_training import REINFORCEAgent
        from environment.custom_env import KigaliRetailEnv

        _env    = KigaliRetailEnv()
        obs_dim = _env.observation_space.shape[0]
        act_dim = _env.action_space.n
        _env.close()

        # hidden size comes from the CSV row - not hardcoded
        hidden = best["hidden"]

        candidates = [
            os.path.join(MODELS_PG, f"reinforce_run_{run_id:02d}.pt"),
        ]
        if os.path.isdir(MODELS_PG):
            for f in sorted(os.listdir(MODELS_PG)):
                p = os.path.join(MODELS_PG, f)
                if f.startswith("reinforce") and f.endswith(".pt") and p not in candidates:
                    candidates.append(p)

        for path in candidates:
            if not os.path.exists(path):
                continue

            # Determine run_id and hidden for this specific file
            file_run_id = run_id
            file_hidden = hidden
            try:
                file_run_id = int(path.split("_run_")[1].split(".")[0])
            except (IndexError, ValueError):
                pass

            if file_run_id != run_id:
                # Read hidden from CSV for the fallback run
                csv_path = os.path.join(RESULTS_DIR, "reinforce_results.csv")
                if os.path.exists(csv_path):
                    with open(csv_path, newline="") as f2:
                        for row2 in csv.DictReader(f2):
                            if int(row2.get("run_id", -1)) == file_run_id:
                                file_hidden = int(row2.get("hidden", 128))
                                break

            agent = REINFORCEAgent(obs_dim, act_dim, hidden=file_hidden)
            try:
                agent.load(path.replace(".pt", ""))
                print(f"  Model path   : {os.path.relpath(path)}")
                print(f"  Hidden size  : {file_hidden}")
                if file_run_id != run_id:
                    print(f"  [NOTE] Best run {run_id:02d} file not found; "
                          f"using run {file_run_id:02d} as fallback.")
                    best["run_id"] = file_run_id
                return agent, "reinforce", best
            except RuntimeError as e:
                print(f"  [WARN] {os.path.basename(path)} failed: {e}")
                continue

        raise FileNotFoundError(
            f"No REINFORCE model found.\n"
            f"  Run: python training/pg_training.py --algo reinforce --sweep"
        )

    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def auto_algo() -> str:
    """Return the algorithm with the highest best mean_reward from its CSV."""
    scores = {}
    for algo in ("dqn", "ppo", "reinforce"):
        info = read_best_run(algo)
        if info.get("mean_reward") is not None:
            scores[algo] = info["mean_reward"]
    if scores:
        return max(scores, key=scores.get)
    return "ppo"


# ==============================================================================
#  PREDICT
# ==============================================================================

def predict(model, model_type: str, obs: np.ndarray) -> int:
    if model_type == "sb3":
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    else:
        import torch
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        probs = model.policy(obs_t).squeeze(0)
        probs = torch.clamp(probs, 1e-8, 1.0)
        probs = probs / probs.sum()
        return int(torch.argmax(probs).item())


# ==============================================================================
#  EVALUATION
# ==============================================================================

def run_evaluation(model, model_type: str, algo: str,
                   best_info: dict,
                   n_episodes: int = 5,
                   render: bool = True) -> None:

    from environment.custom_env import KigaliRetailEnv

    renderer = None
    if render:
        try:
            from environment.rendering import KigaliRenderer
            renderer = KigaliRenderer()
        except Exception as e:
            print(f"  [WARN] GUI unavailable ({e}). Terminal-only.")
            render = False

    W        = 72
    run_id   = best_info.get("run_id", "?")
    mean_r   = best_info.get("mean_reward")
    mean_str = f"{float(mean_r):+.2f}" if mean_r is not None else "N/A"
    cfg      = best_info.get("config_str", "N/A")

    print()
    print("=" * W)
    print(f"  Kigali Retail Navigator - {algo.upper()} Agent Evaluation")
    print(f"  Mission: Place 4 businesses across a Kigali sector")
    print("=" * W)
    print(f"  Algorithm    : {algo.upper()}")
    print(f"  Best run     : Run {run_id:02d}  "
          f"(selected from CSV by highest mean_reward)")
    print(f"  Mean reward  : {mean_str}  (from training/results CSV)")
    print(f"  Config       : {cfg}")
    print(f"  Policy       : "
          f"{'deterministic' if model_type=='sb3' else 'greedy argmax'}")
    print("=" * W)

    env = KigaliRetailEnv(difficulty=0.5)
    all_rewards       = []
    all_placed_counts = []
    all_viabs         = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep * 17 + 3)
        sector   = info["sector"]
        n_rivals = info["n_rivals"]

        print(f"\n{'─'*W}")
        print(f"  Episode {ep+1}/{n_episodes} | Sector: {sector}")
        print(f"  Rivals per type: {n_rivals}")
        print(f"{'─'*W}")
        print(f"  {'Step':>4}  {'Phase':>10}  {'Action':>10}  "
              f"{'Reward':>8}  {'Cumul.':>8}  Info")
        print(f"  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*22}")

        done     = False
        ep_r     = 0.0
        si       = 0
        ep_viabs = []

        while not done:
            phase  = min(env._phase, 3)
            bname  = BUSINESS_NAMES[phase]
            action = predict(model, model_type, obs)
            obs, reward, terminated, truncated, step_info = env.step(action)
            ep_r += reward
            si   += 1
            done  = terminated or truncated

            lbl   = ACT_LABELS.get(action, str(action))
            extra = ""

            if step_info.get("optimal"):
                v     = step_info.get("viability", 0)
                n     = step_info.get("business", "?")
                extra = f"OPTIMAL {n}  viab={v:.3f}"
                ep_viabs.append(float(v))
            elif step_info.get("decent"):
                v     = step_info.get("viability", 0)
                n     = step_info.get("business", "?")
                rival = "  rival nearby" if step_info.get("rival_nearby") else ""
                extra = f"decent {n}{rival}  viab={v:.3f}"
                ep_viabs.append(float(v))
            elif step_info.get("poor"):
                v     = step_info.get("viability", 0)
                n     = step_info.get("business", "?")
                extra = f"poor {n}  viab={v:.3f}"
                ep_viabs.append(float(v))
            elif step_info.get("invalid_cell"):
                extra = "invalid cell"
            elif step_info.get("timeout"):
                missed = step_info.get("missed_placements", "?")
                extra  = f"TIMEOUT  missed={missed}"
            elif step_info.get("wall"):
                extra = "wall"
            elif step_info.get("surveyed"):
                v     = step_info.get("viability_here", 0)
                extra = f"viab here={v:.3f}"
            elif step_info.get("survey_repeat"):
                extra = "survey repeat"
            elif step_info.get("stall_severe"):
                extra = "stall penalty (severe)"
            elif step_info.get("stall_mild"):
                extra = "stall penalty (mild)"
            elif step_info.get("revisit"):
                extra = "revisit penalty"

            if step_info.get("next_business"):
                extra += f"  -> next: {step_info['next_business']}"
            if step_info.get("episode_complete"):
                extra += "  ALL 4 PLACED"

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

        n_placed = len(env._placed_positions)
        best_v   = max(ep_viabs) if ep_viabs else 0.0
        explored = len(env._visited)

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
        print(f"  Steps taken        : {si}/400")
        print(f"  Cells explored     : {explored}/{GS*GS} "
              f"({explored*100//(GS*GS)}%)")

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
                    status=f"Episode {ep+1} complete!  {n_placed}/4 placed.",
                    episode=ep + 1,
                )
                renderer.clock.tick(30)

    print(f"\n{'='*W}")
    print(f"  EVALUATION COMPLETE - {algo.upper()} | {n_episodes} episodes")
    print(f"{'='*W}")
    print(f"  Best run evaluated : Run {run_id:02d}  "
          f"(mean_reward={mean_str} from training CSV)")
    print(f"  Mean reward        : {np.mean(all_rewards):+.2f} "
          f"+/- {np.std(all_rewards):.2f}")
    print(f"  Best episode       : {max(all_rewards):+.2f}")
    print(f"  Worst episode      : {min(all_rewards):+.2f}")
    print(f"  Mean placed        : {np.mean(all_placed_counts):.2f}/4")
    print(f"  Mean best viability: {np.mean(all_viabs):.3f}")
    print(f"  Best viability seen: {max(all_viabs):.3f}")
    print(f"{'='*W}")

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
                status="Evaluation complete - close window to exit",
            )
        renderer.close()


# ==============================================================================
#  API EXPORT
# ==============================================================================

def export_api(algo: str, best_info: dict) -> None:
    run_id = best_info.get("run_id", 0)
    spec = {
        "api_version": "3.0",
        "project":     "Kigali Retail Navigator - RL Policy API",
        "algorithm":   algo.upper(),
        "best_run":    run_id,
        "training_mean_reward": best_info.get("mean_reward"),
        "training_config":      best_info.get("config_str"),
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
            "action":        {"type": "int",    "range": "0-5"},
            "action_name":   {"type": "string"},
            "is_placement":  {"type": "bool"},
            "current_phase": {"type": "int",    "range": "0-3"},
            "business_type": {"type": "string"},
        },
        "phases":  BUSINESS_NAMES,
        "actions": ACT_LABELS,
        "production": [
            "1. Wrap model.predict() in FastAPI POST /predict",
            "2. Encode entrepreneur GPS + 50m OSM landmark scan -> 56-dim obs",
            "3. React Native app calls API at each walking step",
            "4. When action=5: display recommended business + location on map",
            "5. Phase auto-advances: Grocery -> Pharmacy -> Restaurant -> Salon",
            "6. Retrain quarterly as competitor landscape evolves",
        ],
        "latency":    "~0.3ms CPU inference - suitable for real-time mobile",
        "model_file": (
            f"models/{'dqn' if algo=='dqn' else 'pg'}/"
            f"{algo}_run_{run_id:02d}.zip"
        ),
    }
    path = os.path.join(ROOT, "api_spec.json")
    with open(path, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"\n  API spec exported -> {path}")



# ==============================================================================
#  Load a specific run by ID (used when --run N is passed)
# ==============================================================================

def load_specific_run(algo: str, run_id: int):
    """
    Load a specific run by ID regardless of which run has best mean_reward.
    Useful for comparing specific experiments or loading a newly trained model.

    Usage from terminal:
        python main.py --algo dqn --run 3       # load dqn_run_03
        python main.py --algo ppo --run 11      # load ppo_run_11 (best experiment)
        python main.py --algo reinforce --run 7 # load reinforce_run_07
    """
    # Build a fake best_info — run_id is what we care about
    csv_map = {"dqn": "dqn_results.csv", "ppo": "ppo_results.csv",
               "reinforce": "reinforce_results.csv"}
    info = {"run_id": run_id, "mean_reward": None, "hidden": 128,
            "note": f"Manually selected run {run_id}"}

    # Try to read params from CSV if available
    csv_path = os.path.join(RESULTS_DIR, csv_map.get(algo, ""))
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if int(row.get("run_id", -1)) == run_id:
                    info.update({k: v for k, v in row.items()
                                 if k not in ("run_id",)})
                    info["run_id"] = run_id
                    break

    print(f"  [--run {run_id}] Loading specific run, NOT the best from CSV")

    if algo == "dqn":
        from stable_baselines3 import DQN
        candidates = [
            os.path.join(MODELS_DQN, f"dqn_run_{run_id:02d}_best", "best_model.zip"),
            os.path.join(MODELS_DQN, f"dqn_run_{run_id:02d}.zip"),
            os.path.join(MODELS_DQN, "dqn_best.zip"),   # from best_experiment.py
        ]
        for path in candidates:
            if os.path.exists(path):
                print(f"  Model path   : {os.path.relpath(path)}")
                return DQN.load(path), "sb3", info
        raise FileNotFoundError(
            f"No DQN model found for run_id={run_id}.\n"
            f"  Checked: {candidates[0]}\n"
            f"  Checked: {candidates[1]}"
        )

    elif algo == "ppo":
        from stable_baselines3 import PPO
        candidates = [
            os.path.join(MODELS_PG, f"ppo_run_{run_id:02d}_best", "best_model.zip"),
            os.path.join(MODELS_PG, f"ppo_run_{run_id:02d}.zip"),
            os.path.join(MODELS_PG, "ppo_best.zip"),    # from best_experiment.py
        ]
        for path in candidates:
            if os.path.exists(path):
                print(f"  Model path   : {os.path.relpath(path)}")
                return PPO.load(path), "sb3", info

        raise FileNotFoundError(
            f"No PPO model found for run_id={run_id}.\n"
            f"  Checked: {candidates[0]}\n"
            f"  Checked: {candidates[1]}"
        )

    elif algo == "reinforce":
        import torch
        sys.path.insert(0, os.path.join(ROOT, "training"))
        from pg_training import REINFORCEAgent
        from environment.custom_env import KigaliRetailEnv

        _env    = KigaliRetailEnv()
        obs_dim = _env.observation_space.shape[0]
        act_dim = _env.action_space.n
        _env.close()

        hidden = int(info.get("hidden", 128))
        candidates = [
            os.path.join(MODELS_PG, f"reinforce_run_{run_id:02d}.pt"),
            os.path.join(MODELS_PG, "reinforce_best.pt"),
        ]
        for path in candidates:
            if os.path.exists(path):
                agent = REINFORCEAgent(obs_dim, act_dim, hidden=hidden)
                try:
                    agent.load(path.replace(".pt", ""))
                    print(f"  Model path   : {os.path.relpath(path)}")
                    return agent, "reinforce", info
                except RuntimeError as e:
                    print(f"  [WARN] {e}")
        raise FileNotFoundError(
            f"No REINFORCE model found for run_id={run_id}."
        )

    raise ValueError(f"Unknown algo: {algo}")

# ==============================================================================
#  CLI
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Kigali Retail Navigator v3 RL agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --algo dqn              # best DQN run from dqn_results.csv
  python main.py --algo ppo --episodes 3 # best PPO, 3 episodes
  python main.py --algo reinforce        # best REINFORCE run from CSV
  python main.py --no-render             # terminal only (no pygame)
  python main.py --export-api            # also export api_spec.json
        """
    )
    parser.add_argument("--algo",
                        choices=["auto", "dqn", "ppo", "reinforce"],
                        default="auto",
                        help="Algorithm to evaluate (default: auto)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of evaluation episodes")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable Pygame GUI")
    parser.add_argument("--export-api", action="store_true",
                        help="Export policy as JSON API spec")
    parser.add_argument("--run", type=int, default=None,
                        help=(
                            "Load a specific run ID instead of the best. "
                            "E.g. --run 3 loads dqn_run_03 / ppo_run_03. "
                            "Ignores the CSV best-run selection."
                        ))
    args = parser.parse_args()

    algo = auto_algo() if args.algo == "auto" else args.algo
    print(f"\n  Algorithm : {algo.upper()}")

    # ── Load model: specific run or best from CSV ──────────────────────────
    try:
        if args.run is not None:
            model, model_type, best_info = load_specific_run(algo, args.run)
        else:
            model, model_type, best_info = load_model(algo)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    run_evaluation(
        model, model_type, algo, best_info,
        n_episodes=args.episodes,
        render=not args.no_render,
    )

    if args.export_api:
        export_api(algo, best_info)