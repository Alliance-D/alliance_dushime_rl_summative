"""
dqn_training.py  –  Kigali Retail Navigator v3
================================================
Trains DQN on the 4-phase sequential business placement environment.

Env specs:  OBS_DIM=56, N_ACTIONS=6, MAX_STEPS=300
Episode:    Agent places 4 businesses sequentially.
            Each placement switches the rival perception.

Sweep:      10 configurations × 500K timesteps each.
            Difficulty scales with run index (curriculum).

Usage
-----
python training/dqn_training.py --sweep        # full 10-run sweep
python training/dqn_training.py               # single run (defaults)
python training/dqn_training.py --lr 5e-4 --gamma 0.99
"""

import os, sys, argparse, csv, json, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from environment.custom_env import KigaliRetailEnv

MODELS_DIR  = os.path.join(ROOT, "models", "dqn")
PLOTS_DIR   = os.path.join(ROOT, "plots")
LOG_DIR     = os.path.join(ROOT, "logs", "dqn")
RESULTS_CSV = os.path.join(ROOT, "training", "dqn_results.csv")
TOTAL_TS    = 200_000
EVAL_EPS    = 20

for d in [MODELS_DIR, PLOTS_DIR, LOG_DIR, os.path.join(ROOT,"training")]:
    os.makedirs(d, exist_ok=True)

# ── 10-run sweep ───────────────────────────────────────────────────────────────
# Key experiments:
#   - Batch size: 32 / 64 / 128 / 256  (testing gradient noise reduction)
#   - Exploration fraction: 0.30 to 0.70  (testing premature exploitation)
#   - Difficulty curriculum: 0.2 → 0.7 across runs
#
# exploration_fraction = fraction of total timesteps over which epsilon
# decays from 1.0 to exploration_final_eps (0.05).
# Higher = more exploration before exploitation kicks in.
SWEEP = [
    # Run 1 — baseline: moderate batch, moderate exploration
    {"lr":1e-3, "gamma":0.99, "buffer":50_000,  "batch":32,  "expl":0.40, "tui":1000, "diff":0.2},
    # Run 2 — large batch (128): tests gradient noise reduction
    {"lr":5e-4, "gamma":0.99, "buffer":100_000, "batch":128, "expl":0.50, "tui":1000, "diff":0.2},
    # Run 3 — very large batch (256): maximum noise reduction
    {"lr":3e-4, "gamma":0.99, "buffer":100_000, "batch":256, "expl":0.50, "tui":1000, "diff":0.3},
    # Run 4 — long exploration (0.60): avoids premature exploitation
    {"lr":1e-3, "gamma":0.99, "buffer":50_000,  "batch":64,  "expl":0.60, "tui":1000, "diff":0.3},
    # Run 5 — very long exploration (0.70) + large batch
    {"lr":5e-4, "gamma":0.99, "buffer":100_000, "batch":128, "expl":0.70, "tui":1000, "diff":0.4},
    # Run 6 — short exploration (0.20): tests premature exploitation effect
    {"lr":1e-3, "gamma":0.99, "buffer":50_000,  "batch":32,  "expl":0.20, "tui":500,  "diff":0.4},
    # Run 7 — lower gamma + large batch: myopic but stable gradients
    {"lr":1e-3, "gamma":0.95, "buffer":50_000,  "batch":128, "expl":0.50, "tui":1000, "diff":0.5},
    # Run 8 — large buffer + large batch + long exploration (best-guess combo)
    {"lr":5e-4, "gamma":0.99, "buffer":100_000, "batch":128, "expl":0.60, "tui":500,  "diff":0.5},
    # Run 9 — batch 256 + very long exploration + hard difficulty
    {"lr":3e-4, "gamma":0.99, "buffer":100_000, "batch":256, "expl":0.70, "tui":500,  "diff":0.6},
    # Run 10 — optimised: large batch, long exploration, hard difficulty
    {"lr":5e-4, "gamma":0.99, "buffer":100_000, "batch":256, "expl":0.65, "tui":500,  "diff":0.7},
]


class EpRewardCB(BaseCallback):
    """Records episode rewards and placement counts."""
    def __init__(self):
        super().__init__()
        self.ep_rewards: list = []
        self._ep_r = 0.0

    def _on_step(self) -> bool:
        r = self.locals.get("rewards", [0])[0]
        d = self.locals.get("dones",   [False])[0]
        self._ep_r += float(r)
        if d:
            self.ep_rewards.append(self._ep_r)
            self._ep_r = 0.0
        return True


def train_dqn(cfg: dict, run_id: int = 0, total_ts: int = TOTAL_TS) -> dict:
    mpath = os.path.join(MODELS_DIR, f"dqn_run_{run_id:02d}")
    lpath = os.path.join(LOG_DIR,    f"run_{run_id:02d}")
    os.makedirs(lpath, exist_ok=True)

    train_env = Monitor(KigaliRetailEnv(difficulty=cfg["diff"]), lpath)
    eval_env  = Monitor(KigaliRetailEnv(difficulty=cfg["diff"]))

    rew_cb  = EpRewardCB()
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=mpath + "_best",
        log_path=lpath,
        eval_freq=5000,
        n_eval_episodes=EVAL_EPS,
        deterministic=True,
        verbose=0,
    )

    model = DQN(
        "MlpPolicy", train_env,
        learning_rate=cfg["lr"],
        gamma=cfg["gamma"],
        buffer_size=cfg["buffer"],
        batch_size=cfg["batch"],
        exploration_fraction=cfg["expl"],
        exploration_final_eps=0.05,
        target_update_interval=cfg["tui"],
        train_freq=4,
        gradient_steps=1,
        verbose=0,
    )

    t0 = time.time()
    model.learn(total_ts, callback=[rew_cb, eval_cb])
    elapsed = time.time() - t0
    model.save(mpath)

    # Evaluation
    ev = KigaliRetailEnv(difficulty=cfg["diff"])
    rews = []
    for _ in range(EVAL_EPS):
        obs, _ = ev.reset()
        done = False; epr = 0.0
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = ev.step(int(a))
            epr += r; done = term or trunc
        rews.append(epr)
    ev.close(); eval_env.close(); train_env.close()

    mean_r = float(np.mean(rews))
    std_r  = float(np.std(rews))
    print(f"  Run {run_id:02d} | lr={cfg['lr']:.0e} γ={cfg['gamma']} "
          f"buf={cfg['buffer']} expl={cfg['expl']} diff={cfg['diff']} "
          f"→ {mean_r:.2f}±{std_r:.2f}")

    return {
        **cfg,
        "run_id":         run_id,
        "mean_reward":    round(mean_r, 2),
        "std_reward":     round(std_r,  2),
        "train_time_s":   round(elapsed, 1),
        "episode_rewards": rew_cb.ep_rewards,
        "model_path":     mpath,
    }


def run_sweep() -> list:
    print("=" * 58)
    print("DQN Hyperparameter Sweep – 10 runs × 500K timesteps")
    print("=" * 58)
    results = []
    for i, cfg in enumerate(SWEEP):
        print(f"\n[{i+1}/10] {cfg}")
        results.append(train_dqn(cfg, run_id=i))

    # Save CSV
    fields = ["run_id","lr","gamma","buffer","batch","expl","tui","diff",
              "mean_reward","std_reward","train_time_s"]
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fields})

    # Save best config
    best = max(results, key=lambda r: r["mean_reward"])
    with open(os.path.join(MODELS_DIR, "best_config.json"), "w") as f:
        json.dump({k: best[k] for k in SWEEP[0].keys()}, f, indent=2)

    # Plots
    _plot_rewards(results)
    _plot_td_stability(results)

    print(f"\nBest DQN: Run {best['run_id']:02d}  mean={best['mean_reward']:.2f}")
    return results


def _plot_rewards(results: list) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    for idx, res in enumerate(results):
        ax  = axes[idx//5][idx%5]
        rws = res.get("episode_rewards", [])
        if rws:
            ax.fill_between(range(len(rws)), rws, alpha=0.22, color="#3B82F6")
            w  = min(50, len(rws))
            rm = np.convolve(rws, np.ones(w)/w, mode="valid")
            ax.plot(range(w-1, len(rws)), rm, lw=1.8, color="#1D4ED8")
            ax.axhline(0, color="grey", lw=0.7, ls="--", alpha=0.5)
        ax.set_title(
            f"Run {idx+1}  μ={res['mean_reward']:.1f}\n"
            f"lr={res['lr']:.0e} γ={res['gamma']} diff={res['diff']}",
            fontsize=8)
        ax.set_xlabel("Episode", fontsize=7)
        ax.set_ylabel("Reward",  fontsize=7)
        ax.tick_params(labelsize=6)
    fig.suptitle("DQN — Episode Reward (blue=rolling mean, 50-ep window)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "dqn_rewards.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved {out}")


def _plot_td_stability(results: list) -> None:
    """Rolling std of episode reward as a proxy for TD-loss stability."""
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    for idx, res in enumerate(results):
        ax  = axes[idx//5][idx%5]
        rws = res.get("episode_rewards", [])
        if len(rws) > 10:
            w  = 30
            td = [np.std(rws[max(0,i-w):i+1]) for i in range(len(rws))]
            ax.plot(td, lw=1.2, color="#EF4444", alpha=0.8)
        ax.set_title(f"Run {idx+1}  lr={res['lr']:.0e}", fontsize=8)
        ax.set_xlabel("Episode", fontsize=7)
        ax.set_ylabel("Reward Std (TD proxy)", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.suptitle("DQN — Objective Stability (rolling std of episode reward)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "dqn_td_loss.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN on Kigali env v3")
    parser.add_argument("--sweep",  action="store_true")
    parser.add_argument("--lr",     type=float, default=5e-4)
    parser.add_argument("--gamma",  type=float, default=0.99)
    parser.add_argument("--buffer", type=int,   default=50_000)
    parser.add_argument("--batch",  type=int,   default=32)
    parser.add_argument("--expl",   type=float, default=0.30)
    parser.add_argument("--tui",    type=int,   default=500)
    parser.add_argument("--diff",   type=float, default=0.4)
    args = parser.parse_args()

    if args.sweep:
        run_sweep()
    else:
        cfg = {"lr":args.lr,"gamma":args.gamma,"buffer":args.buffer,
               "batch":args.batch,"expl":args.expl,"tui":args.tui,"diff":args.diff}
        result = train_dqn(cfg, run_id=99)
        print(f"\nSingle run done. mean_reward={result['mean_reward']:.2f}")