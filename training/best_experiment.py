"""
best_experiment.py  –  Kigali Retail Navigator v3
==================================================
Train a single focused experiment with any hyperparameters you choose.
Each run gets a unique ID and saves to its own model file — nothing
is ever overwritten.

Results are appended to the same CSV files as the sweep
(dqn_results.csv / ppo_results.csv / reinforce_results.csv) so
main.py automatically picks the best across ALL experiments.

Usage
-----
# Basic — train one algorithm with the built-in best config:
python training/best_experiment.py --algo dqn
python training/best_experiment.py --algo ppo
python training/best_experiment.py --algo reinforce

# Override any hyperparameter from the command line:
python training/best_experiment.py --algo dqn --lr 3e-4 --expl 0.6 --total_ts 500000
python training/best_experiment.py --algo ppo --lr 3e-4 --batch 128 --ent_coef 0.02
python training/best_experiment.py --algo reinforce --hidden 256 --n_episodes 6000

# Specify run ID manually (default: auto-increment from CSV):
python training/best_experiment.py --algo dqn --run_id 26

How model files are named
--------------------------
DQN:       models/dqn/dqn_run_NN.zip          (NN = run_id)
PPO:       models/pg/ppo_run_NN.zip
REINFORCE: models/pg/reinforce_run_NN.pt

How to evaluate a specific run
-------------------------------
python main.py --algo dqn              # auto-loads best from CSV
python main.py --algo dqn --run 26     # loads dqn_run_26 specifically
"""

from __future__ import annotations
import os, sys, argparse, csv, json, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
import torch.optim as optim

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from environment.custom_env import KigaliRetailEnv

MODELS_DQN  = os.path.join(ROOT, "models", "dqn")
MODELS_PG   = os.path.join(ROOT, "models", "pg")
PLOTS_DIR   = os.path.join(ROOT, "plots")
LOG_DIR     = os.path.join(ROOT, "logs", "best")
RESULTS_DIR = os.path.join(ROOT, "training")

CSV_DQN       = os.path.join(RESULTS_DIR, "dqn_results.csv")
CSV_PPO       = os.path.join(RESULTS_DIR, "ppo_results.csv")
CSV_REINFORCE = os.path.join(RESULTS_DIR, "reinforce_results.csv")

EVAL_EPS  = 30
EVAL_DIFF = 0.3

for d in [MODELS_DQN, MODELS_PG, PLOTS_DIR, LOG_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)


# ── Best default configs (used when no CLI overrides given) ───────────────────
# These are the recommended starting points based on sweep analysis.
# Override any value from the command line — see usage above.

DEFAULT_DQN = {
    "lr":       5e-4,
    "gamma":    0.99,
    "buffer":   100_000,
    "batch":    128,
    "expl":     0.6,
    "tui":      1000,
    "diff":     0.2,
    "total_ts": 500_000,
}

DEFAULT_PPO = {
    "lr":         5e-3,
    "gamma":      0.99,
    "n_steps":    1024,
    "batch":      128,
    "n_epochs":   15,
    "clip_range": 0.15,
    "ent_coef":   0.01,
    "gae_lambda": 0.95,
    "diff":       0.2,
    "total_ts":   400_000,
}

DEFAULT_REINFORCE = {
    "lr":           1e-3,
    "gamma":        0.99,
    "entropy_coef": 0.02,
    "hidden":       256,
    "diff":         0.2,
    "n_episodes":   4000,
}


# ── CSV helpers ───────────────────────────────────────────────────────────────

def next_run_id(csv_path: str) -> int:
    """Return the next available run_id by reading the CSV."""
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0
    ids = []
    for r in rows:
        try:
            ids.append(int(r["run_id"]))
        except (ValueError, KeyError):
            pass
    return max(ids) + 1 if ids else 0


def append_csv(csv_path: str, row: dict, fields: list):
    """Append one row to CSV, creating file with header if needed."""
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})
    print(f"  Appended to  : {os.path.relpath(csv_path)}")


# ── Callbacks ─────────────────────────────────────────────────────────────────

class EpRewardCB(BaseCallback):
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


class PPOEntropyCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.entropies: list = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        if hasattr(self.model, "logger") and self.model.logger:
            val = self.model.logger.name_to_value.get("train/entropy_loss")
            if val is not None:
                self.entropies.append(float(val))


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_sb3(model, n_eps=EVAL_EPS, diff=EVAL_DIFF):
    ev = KigaliRetailEnv(difficulty=diff)
    rews = []
    for _ in range(n_eps):
        obs, _ = ev.reset()
        done = False; epr = 0.0
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = ev.step(int(a))
            epr += r; done = term or trunc
        rews.append(epr)
    ev.close()
    return float(np.mean(rews)), float(np.std(rews))


def evaluate_reinforce(agent, n_eps=EVAL_EPS, diff=EVAL_DIFF):
    """Stochastic eval — matches training, avoids argmax collapse."""
    ev = KigaliRetailEnv(difficulty=diff)
    rews = []
    for _ in range(n_eps):
        obs, _ = ev.reset()
        done = False; epr = 0.0
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                probs = agent.policy(obs_t).squeeze(0)
            probs = torch.clamp(probs, 1e-8, 1.0) / probs.sum()
            a = int(torch.distributions.Categorical(probs).sample().item())
            obs, r, term, trunc, _ = ev.step(a)
            epr += r; done = term or trunc
        rews.append(epr)
    ev.close()
    return float(np.mean(rews)), float(np.std(rews))


# ── Plot ──────────────────────────────────────────────────────────────────────

def save_plot(rws, title, fname, mean_r):
    fig, ax = plt.subplots(figsize=(11, 4))
    if rws:
        ax.fill_between(range(len(rws)), rws, alpha=0.18, color="#3B82F6")
        w  = min(50, len(rws))
        rm = np.convolve(rws, np.ones(w)/w, mode="valid")
        ax.plot(range(w-1, len(rws)), rm, lw=2, color="#1D4ED8",
                label=f"Rolling mean (50-ep)")
        ax.axhline(0, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Episode"); ax.set_ylabel("Reward")
    ax.set_title(f"{title}  |  eval mean={mean_r:.2f}")
    ax.legend(); plt.tight_layout()
    out = os.path.join(PLOTS_DIR, fname)
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Plot saved   : {os.path.relpath(out)}")


# ══════════════════════════════════════════════════════════════════════════════
#  DQN
# ══════════════════════════════════════════════════════════════════════════════

def train_dqn(cfg: dict, run_id: int):
    mpath = os.path.join(MODELS_DQN, f"dqn_run_{run_id:02d}")
    lpath = os.path.join(LOG_DIR, f"dqn_{run_id:02d}")
    os.makedirs(lpath, exist_ok=True)

    print(f"\n{'='*58}")
    print(f"DQN  run_id={run_id}")
    print(f"  lr={cfg['lr']:.0e}  batch={cfg['batch']}  "
          f"expl={cfg['expl']}  diff={cfg['diff']}  "
          f"ts={cfg['total_ts']:,}")
    print(f"{'='*58}")

    train_env = Monitor(KigaliRetailEnv(difficulty=cfg["diff"]), lpath)
    eval_env  = Monitor(KigaliRetailEnv(difficulty=EVAL_DIFF))

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
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        target_update_interval=cfg["tui"],
        train_freq=4,
        gradient_steps=1,
        verbose=0,
    )

    t0 = time.time()
    model.learn(cfg["total_ts"], callback=[rew_cb, eval_cb])
    elapsed = time.time() - t0
    model.save(mpath)
    print(f"  Model saved  : {os.path.relpath(mpath)}.zip")

    mean_r, std_r = evaluate_sb3(model)
    print(f"  Eval result  : {mean_r:.2f} ± {std_r:.2f}  [{elapsed:.0f}s]")

    train_env.close(); eval_env.close()

    row = {**cfg, "run_id": run_id,
           "mean_reward": round(mean_r, 2), "std_reward": round(std_r, 2),
           "train_time_s": round(elapsed, 1)}

    fields = ["run_id","lr","gamma","buffer","batch","expl","tui","diff",
              "mean_reward","std_reward","train_time_s"]
    append_csv(CSV_DQN, row, fields)
    save_plot(rew_cb.ep_rewards, f"DQN run_{run_id:02d}", 
              f"dqn_run_{run_id:02d}_rewards.png", mean_r)
    return row


# ══════════════════════════════════════════════════════════════════════════════
#  PPO
# ══════════════════════════════════════════════════════════════════════════════

def train_ppo(cfg: dict, run_id: int):
    mpath = os.path.join(MODELS_PG, f"ppo_run_{run_id:02d}")
    lpath = os.path.join(LOG_DIR, f"ppo_{run_id:02d}")
    os.makedirs(lpath, exist_ok=True)

    print(f"\n{'='*58}")
    print(f"PPO  run_id={run_id}")
    print(f"  lr={cfg['lr']:.0e}  batch={cfg['batch']}  "
          f"ent={cfg['ent_coef']}  diff={cfg['diff']}  "
          f"ts={cfg['total_ts']:,}")
    print(f"{'='*58}")

    train_env = Monitor(KigaliRetailEnv(difficulty=cfg["diff"]), lpath)
    eval_env  = Monitor(KigaliRetailEnv(difficulty=EVAL_DIFF))

    rew_cb  = EpRewardCB()
    ent_cb  = PPOEntropyCallback()
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=mpath + "_best",
        log_path=lpath,
        eval_freq=5000,
        n_eval_episodes=EVAL_EPS,
        deterministic=True,
        verbose=0,
    )

    model = PPO(
        "MlpPolicy", train_env,
        learning_rate=cfg["lr"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch"],
        n_epochs=cfg["n_epochs"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_range"],
        ent_coef=cfg["ent_coef"],
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
    )

    t0 = time.time()
    model.learn(cfg["total_ts"], callback=[rew_cb, ent_cb, eval_cb])
    elapsed = time.time() - t0
    model.save(mpath)
    print(f"  Model saved  : {os.path.relpath(mpath)}.zip")

    mean_r, std_r = evaluate_sb3(model)
    print(f"  Eval result  : {mean_r:.2f} ± {std_r:.2f}  [{elapsed:.0f}s]")

    train_env.close(); eval_env.close()

    row = {**cfg, "run_id": run_id,
           "mean_reward": round(mean_r, 2), "std_reward": round(std_r, 2),
           "train_time_s": round(elapsed, 1)}

    fields = ["run_id","lr","gamma","n_steps","batch","n_epochs",
              "clip_range","ent_coef","gae_lambda","diff",
              "mean_reward","std_reward","train_time_s"]
    append_csv(CSV_PPO, row, fields)
    save_plot(rew_cb.ep_rewards, f"PPO run_{run_id:02d}",
              f"ppo_run_{run_id:02d}_rewards.png", mean_r)
    return row


# ══════════════════════════════════════════════════════════════════════════════
#  REINFORCE
# ══════════════════════════════════════════════════════════════════════════════

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


class REINFORCEAgent:
    def __init__(self, obs_dim, act_dim, lr=1e-3, gamma=0.99,
                 entropy_coef=0.01, hidden=128):
        self.gamma        = gamma
        self.entropy_coef = entropy_coef
        self.policy       = PolicyNet(obs_dim, act_dim, hidden)
        self.opt          = optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, obs):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            probs = self.policy(obs_t).squeeze(0)
        probs = torch.clamp(probs, 1e-8, 1.0) / probs.sum()
        dist  = torch.distributions.Categorical(probs)
        action = dist.sample()
        probs_g = self.policy(obs_t).squeeze(0)
        probs_g = torch.clamp(probs_g, 1e-8, 1.0) / probs_g.sum()
        dist_g  = torch.distributions.Categorical(probs_g)
        return int(action.item()), dist_g.log_prob(action), dist_g.entropy()

    def update(self, log_probs, entropies, rewards):
        G = 0.0; returns = []
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        ret_t = torch.FloatTensor(returns)
        if ret_t.std() > 1e-8:
            ret_t = (ret_t - ret_t.mean()) / (ret_t.std() + 1e-8)
        lp_t  = torch.clamp(torch.stack(log_probs), -10.0, 0.0)
        ent_t = torch.stack(entropies)
        loss  = -(lp_t * ret_t).mean() - self.entropy_coef * ent_t.mean()
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.opt.step()
        return float(ent_t.mean().item())

    def save(self, path):
        torch.save(self.policy.state_dict(), path + ".pt")

    def load(self, path):
        self.policy.load_state_dict(
            torch.load(path + ".pt", map_location="cpu"))


def train_reinforce(cfg: dict, run_id: int):
    mpath = os.path.join(MODELS_PG, f"reinforce_run_{run_id:02d}")

    print(f"\n{'='*58}")
    print(f"REINFORCE  run_id={run_id}")
    print(f"  lr={cfg['lr']:.0e}  hidden={cfg['hidden']}  "
          f"ent={cfg['entropy_coef']}  diff={cfg['diff']}  "
          f"eps={cfg['n_episodes']}")
    print(f"{'='*58}")

    env     = KigaliRetailEnv(difficulty=cfg["diff"])
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = REINFORCEAgent(obs_dim, act_dim,
                           lr=cfg["lr"], gamma=cfg["gamma"],
                           entropy_coef=cfg["entropy_coef"],
                           hidden=cfg["hidden"])

    ep_rewards = []; entropies = []
    t0 = time.time()

    for ep in range(cfg["n_episodes"]):
        obs, _ = env.reset(seed=ep)
        log_probs, ents, rewards = [], [], []
        done = False
        while not done:
            action, lp, ent = agent.select_action(obs)
            obs, r, term, trunc, _ = env.step(action)
            log_probs.append(lp); ents.append(ent); rewards.append(r)
            done = term or trunc
        ep_rewards.append(sum(rewards))
        mean_ent = agent.update(log_probs, ents, rewards)
        entropies.append(mean_ent)

        if (ep + 1) % 500 == 0:
            mr = np.mean(ep_rewards[-100:])
            print(f"  ep={ep+1:5d}  mean100={mr:.2f}  ent={mean_ent:.4f}")

    elapsed = time.time() - t0
    env.close()
    agent.save(mpath)
    print(f"  Model saved  : {os.path.relpath(mpath)}.pt")

    mean_r, std_r = evaluate_reinforce(agent)
    print(f"  Eval result  : {mean_r:.2f} ± {std_r:.2f}  [{elapsed:.0f}s]")

    row = {**cfg, "run_id": run_id,
           "mean_reward": round(mean_r, 2), "std_reward": round(std_r, 2),
           "train_time_s": round(elapsed, 1)}

    fields = ["run_id","lr","gamma","entropy_coef","hidden","diff",
              "mean_reward","std_reward","train_time_s"]
    append_csv(CSV_REINFORCE, row, fields)
    save_plot(ep_rewards, f"REINFORCE run_{run_id:02d}",
              f"reinforce_run_{run_id:02d}_rewards.png", mean_r)
    return row


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train one focused experiment — each run saved with unique ID",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default best config (auto-increments run ID):
  python training/best_experiment.py --algo dqn
  python training/best_experiment.py --algo ppo
  python training/best_experiment.py --algo reinforce

  # Override hyperparameters:
  python training/best_experiment.py --algo dqn --lr 3e-4 --expl 0.7 --total_ts 500000
  python training/best_experiment.py --algo ppo --lr 3e-4 --batch 128 --ent_coef 0.02
  python training/best_experiment.py --algo reinforce --hidden 256 --n_episodes 6000

  # Force a specific run ID (useful if you want to reuse a slot):
  python training/best_experiment.py --algo dqn --run_id 26

  # Evaluate any run:
  python main.py --algo dqn --run 26
  python main.py --algo dqn          # auto-loads best from CSV
"""
    )
    parser.add_argument("--algo", required=True,
                        choices=["dqn", "ppo", "reinforce"],
                        help="Algorithm to train")

    # Run ID — default auto-increments from CSV
    parser.add_argument("--run_id", type=int, default=None,
                        help="Run ID to use. Default: auto-increment from CSV.")

    # DQN / shared args
    parser.add_argument("--lr",       type=float, default=None)
    parser.add_argument("--gamma",    type=float, default=None)
    parser.add_argument("--buffer",   type=int,   default=None)
    parser.add_argument("--batch",    type=int,   default=None)
    parser.add_argument("--expl",     type=float, default=None,
                        help="DQN exploration fraction")
    parser.add_argument("--tui",      type=int,   default=None,
                        help="DQN target update interval")
    parser.add_argument("--diff",     type=float, default=None,
                        help="Training difficulty (0.0=easy, 1.0=hard)")
    parser.add_argument("--total_ts", type=int,   default=None,
                        help="Total timesteps (DQN/PPO)")

    # PPO args
    parser.add_argument("--n_steps",    type=int,   default=None)
    parser.add_argument("--n_epochs",   type=int,   default=None)
    parser.add_argument("--clip_range", type=float, default=None)
    parser.add_argument("--ent_coef",   type=float, default=None)
    parser.add_argument("--gae_lambda", type=float, default=None)

    # REINFORCE args
    parser.add_argument("--entropy_coef", type=float, default=None)
    parser.add_argument("--hidden",       type=int,   default=None)
    parser.add_argument("--n_episodes",   type=int,   default=None)

    args = parser.parse_args()

    # ── Build config: start from defaults, apply CLI overrides ───────────────
    if args.algo == "dqn":
        cfg = dict(DEFAULT_DQN)
        csv_path = CSV_DQN
    elif args.algo == "ppo":
        cfg = dict(DEFAULT_PPO)
        csv_path = CSV_PPO
    else:
        cfg = dict(DEFAULT_REINFORCE)
        csv_path = CSV_REINFORCE

    # Apply any CLI overrides
    override_map = {
        "lr": args.lr, "gamma": args.gamma, "buffer": args.buffer,
        "batch": args.batch, "expl": args.expl, "tui": args.tui,
        "diff": args.diff, "total_ts": args.total_ts,
        "n_steps": args.n_steps, "n_epochs": args.n_epochs,
        "clip_range": args.clip_range, "ent_coef": args.ent_coef,
        "gae_lambda": args.gae_lambda,
        "entropy_coef": args.entropy_coef, "hidden": args.hidden,
        "n_episodes": args.n_episodes,
    }
    for key, val in override_map.items():
        if val is not None and key in cfg:
            cfg[key] = val

    # ── Determine run ID ──────────────────────────────────────────────────────
    run_id = args.run_id if args.run_id is not None else next_run_id(csv_path)
    print(f"\n  Run ID: {run_id}  (next available from CSV)")

    # ── Train ─────────────────────────────────────────────────────────────────
    if args.algo == "dqn":
        result = train_dqn(cfg, run_id)
    elif args.algo == "ppo":
        result = train_ppo(cfg, run_id)
    else:
        result = train_reinforce(cfg, run_id)

    print(f"\n{'='*58}")
    print(f"  Done.  run_id={run_id}  "
          f"mean_reward={result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
    print(f"  To evaluate: python main.py --algo {args.algo} --run {run_id}")
    print(f"  For best:    python main.py --algo {args.algo}")
    print(f"{'='*58}")