"""
pg_training.py  –  Kigali Retail Navigator v3
===============================================
Trains REINFORCE (custom PyTorch) and PPO (SB3) on the 4-phase
sequential business placement environment.

Env specs:  OBS_DIM=56, N_ACTIONS=6, MAX_STEPS=300

REINFORCE fixes:
  - Gradient clipping (max_norm=0.5)
  - Log-prob clamped to [-10, 0] for numerical stability
  - Per-episode return normalisation
  - Entropy regularisation (0.005–0.02 range)

PPO entropy fix:
  - Reads from model.logger.name_to_value["train/entropy_loss"]
    (correct SB3 method — not locals)

Usage
-----
python training/pg_training.py --algo ppo --sweep
python training/pg_training.py --algo reinforce --sweep
python training/pg_training.py --algo all --sweep
python training/pg_training.py --algo ppo          # single run
"""

from __future__ import annotations
import os, sys, argparse, csv, json, time
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from environment.custom_env import KigaliRetailEnv

MODELS_DIR  = os.path.join(ROOT, "models", "pg")
PLOTS_DIR   = os.path.join(ROOT, "plots")
LOG_DIR     = os.path.join(ROOT, "logs", "pg")
CSV_PPO     = os.path.join(ROOT, "training", "ppo_results.csv")
CSV_RE      = os.path.join(ROOT, "training", "reinforce_results.csv")
TOTAL_TS    = 300_000
EVAL_EPS    = 20

for d in [MODELS_DIR, PLOTS_DIR, LOG_DIR, os.path.join(ROOT,"training")]:
    os.makedirs(d, exist_ok=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  REINFORCE  (custom PyTorch)                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)


class REINFORCEAgent:
    def __init__(self, obs_dim: int, act_dim: int,
                 lr: float = 1e-3, gamma: float = 0.99,
                 entropy_coef: float = 0.01, hidden: int = 128):
        self.gamma        = gamma
        self.entropy_coef = entropy_coef
        self.policy       = PolicyNet(obs_dim, act_dim, hidden)
        self.opt          = optim.Adam(self.policy.parameters(), lr=lr)
        self.act_dim      = act_dim

    def select_action(self, obs: np.ndarray):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            probs = self.policy(obs_t).squeeze(0)
        probs = torch.clamp(probs, 1e-8, 1.0)
        probs = probs / probs.sum()
        dist  = torch.distributions.Categorical(probs)
        action = dist.sample()
        # Recompute with grad for backprop
        probs_g = self.policy(obs_t).squeeze(0)
        probs_g = torch.clamp(probs_g, 1e-8, 1.0) / probs_g.sum()
        dist_g  = torch.distributions.Categorical(probs_g)
        return int(action.item()), dist_g.log_prob(action), dist_g.entropy()

    def update(self, log_probs: list, entropies: list, rewards: list):
        G = 0.0; returns = []
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        ret_t = torch.FloatTensor(returns)
        if ret_t.std() > 1e-8:
            ret_t = (ret_t - ret_t.mean()) / (ret_t.std() + 1e-8)

        lp_t  = torch.stack(log_probs)
        lp_t  = torch.clamp(lp_t, -10.0, 0.0)   # numerical stability
        ent_t = torch.stack(entropies)

        policy_loss  = -(lp_t * ret_t).mean()
        entropy_loss = -self.entropy_coef * ent_t.mean()
        loss         = policy_loss + entropy_loss

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.opt.step()

        return float(policy_loss.item()), float(ent_t.mean().item())

    def save(self, path: str):
        torch.save(self.policy.state_dict(), path + ".pt")

    def load(self, path: str):
        self.policy.load_state_dict(
            torch.load(path + ".pt", map_location="cpu"))


def train_reinforce(cfg: dict, run_id: int = 0,
                    n_episodes: int = 3000) -> dict:
    env     = KigaliRetailEnv(difficulty=cfg.get("diff", 0.4))
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    agent   = REINFORCEAgent(obs_dim, act_dim,
                              lr=cfg["lr"], gamma=cfg["gamma"],
                              entropy_coef=cfg["entropy_coef"],
                              hidden=cfg.get("hidden", 128))
    ep_rewards: List[float] = []
    policy_losses: List[float] = []
    entropies: List[float] = []
    t0 = time.time()

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        log_probs, ents, rewards = [], [], []
        done = False
        while not done:
            action, lp, ent = agent.select_action(obs)
            obs, r, term, trunc, _ = env.step(action)
            log_probs.append(lp); ents.append(ent); rewards.append(r)
            done = term or trunc

        ep_rewards.append(sum(rewards))
        ploss, mean_ent = agent.update(log_probs, ents, rewards)
        policy_losses.append(ploss); entropies.append(mean_ent)

        if (ep + 1) % 500 == 0:
            mr = np.mean(ep_rewards[-100:])
            print(f"  [RE run {run_id:02d}] ep={ep+1}  "
                  f"mean100={mr:.2f}  ploss={ploss:.4f}  ent={mean_ent:.4f}")

    elapsed = time.time() - t0
    env.close()

    spath = os.path.join(MODELS_DIR, f"reinforce_run_{run_id:02d}")
    agent.save(spath)

    # Evaluate
    ev = KigaliRetailEnv(difficulty=cfg.get("diff", 0.4))
    erews = []
    for _ in range(EVAL_EPS):
        obs, _ = ev.reset(); done=False; epr=0.0
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            p     = agent.policy(obs_t).squeeze(0)
            p     = torch.clamp(p,1e-8,1.0)/p.sum()
            a     = int(torch.argmax(p).item())
            obs, r, term, trunc, _ = ev.step(a)
            epr += r; done = term or trunc
        erews.append(epr)
    ev.close()

    mr = float(np.mean(erews)); sr = float(np.std(erews))
    print(f"  Run {run_id:02d} | lr={cfg['lr']:.0e} γ={cfg['gamma']} "
          f"ent={cfg['entropy_coef']} hidden={cfg.get('hidden',128)} "
          f"diff={cfg.get('diff',0.4)} → {mr:.2f}±{sr:.2f}")

    return {**cfg, "run_id":run_id, "mean_reward":round(mr,2),
            "std_reward":round(sr,2), "train_time_s":round(elapsed,1),
            "episode_rewards":ep_rewards, "entropies":entropies,
            "model_path":spath}


# 10-run REINFORCE sweep (difficulty increases progressively)
REINFORCE_SWEEP = [
    {"lr":1e-3, "gamma":0.99,"entropy_coef":0.010,"hidden":128,"diff":0.2},
    {"lr":5e-4, "gamma":0.99,"entropy_coef":0.010,"hidden":128,"diff":0.2},
    {"lr":2e-3, "gamma":0.99,"entropy_coef":0.010,"hidden":128,"diff":0.3},
    {"lr":1e-3, "gamma":0.95,"entropy_coef":0.010,"hidden":128,"diff":0.3},
    {"lr":1e-3, "gamma":0.90,"entropy_coef":0.010,"hidden":128,"diff":0.3},
    {"lr":1e-3, "gamma":0.99,"entropy_coef":0.020,"hidden":128,"diff":0.4},
    {"lr":1e-3, "gamma":0.99,"entropy_coef":0.005,"hidden":128,"diff":0.4},
    {"lr":1e-3, "gamma":0.99,"entropy_coef":0.010,"hidden": 64,"diff":0.5},
    {"lr":1e-3, "gamma":0.99,"entropy_coef":0.010,"hidden":256,"diff":0.5},
    {"lr":5e-4, "gamma":0.99,"entropy_coef":0.015,"hidden":256,"diff":0.6},
]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PPO  (SB3) — fixed entropy logging                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class PPOEntropyCallback(BaseCallback):
    """Reads entropy from SB3 logger after each rollout — correct method."""
    def __init__(self):
        super().__init__()
        self.entropies: List[float] = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if hasattr(self.model, "logger") and self.model.logger is not None:
            val = self.model.logger.name_to_value.get("train/entropy_loss")
            if val is not None:
                self.entropies.append(float(val))


class PPOEpRewardCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.ep_rewards: List[float] = []
        self._epr = 0.0

    def _on_step(self) -> bool:
        r = self.locals.get("rewards", [0])[0]
        d = self.locals.get("dones",   [False])[0]
        self._epr += float(r)
        if d:
            self.ep_rewards.append(self._epr)
            self._epr = 0.0
        return True


def train_ppo(cfg: dict, run_id: int = 0,
              total_ts: int = TOTAL_TS) -> dict:
    lpath = os.path.join(LOG_DIR, f"ppo_{run_id:02d}")
    mpath = os.path.join(MODELS_DIR, f"ppo_run_{run_id:02d}")
    os.makedirs(lpath, exist_ok=True)

    train_env = Monitor(KigaliRetailEnv(difficulty=cfg.get("diff",0.4)), lpath)
    eval_env  = Monitor(KigaliRetailEnv(difficulty=cfg.get("diff",0.4)))

    ent_cb  = PPOEntropyCallback()
    rew_cb  = PPOEpRewardCallback()
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
        gae_lambda=cfg.get("gae_lambda", 0.95),
        clip_range=cfg.get("clip_range", 0.2),
        ent_coef=cfg.get("ent_coef", 0.01),
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
    )

    t0 = time.time()
    model.learn(total_ts, callback=[ent_cb, rew_cb, eval_cb])
    elapsed = time.time() - t0
    model.save(mpath)

    # Evaluate
    ev = KigaliRetailEnv(difficulty=cfg.get("diff",0.4))
    erews = []
    for _ in range(EVAL_EPS):
        obs, _ = ev.reset(); done=False; epr=0.0
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = ev.step(int(a))
            epr += r; done = term or trunc
        erews.append(epr)
    ev.close(); eval_env.close(); train_env.close()

    mr = float(np.mean(erews)); sr = float(np.std(erews))
    print(f"  Run {run_id:02d} | lr={cfg['lr']:.0e} "
          f"clip={cfg.get('clip_range',0.2)} "
          f"ent={cfg.get('ent_coef',0.01)} "
          f"diff={cfg.get('diff',0.4)} → {mr:.2f}±{sr:.2f}")

    return {**cfg, "run_id":run_id, "mean_reward":round(mr,2),
            "std_reward":round(sr,2), "train_time_s":round(elapsed,1),
            "episode_rewards":rew_cb.ep_rewards,
            "entropies":ent_cb.entropies,
            "model_path":mpath}


# 10-run PPO sweep
PPO_SWEEP = [
    # Run 1 — baseline, easy
    {"lr":3e-4,"gamma":0.99,"n_steps":2048,"batch":64,"n_epochs":10,
     "clip_range":0.20,"ent_coef":0.010,"gae_lambda":0.95,"diff":0.2},
    # Run 2 — lower lr
    {"lr":1e-4,"gamma":0.99,"n_steps":2048,"batch":64,"n_epochs":10,
     "clip_range":0.20,"ent_coef":0.010,"gae_lambda":0.95,"diff":0.2},
    # Run 3 — higher lr
    {"lr":1e-3,"gamma":0.99,"n_steps":2048,"batch":64,"n_epochs":10,
     "clip_range":0.20,"ent_coef":0.010,"gae_lambda":0.95,"diff":0.3},
    # Run 4 — smaller clip
    {"lr":3e-4,"gamma":0.99,"n_steps":2048,"batch":64,"n_epochs":10,
     "clip_range":0.10,"ent_coef":0.010,"gae_lambda":0.95,"diff":0.3},
    # Run 5 — larger clip
    {"lr":3e-4,"gamma":0.99,"n_steps":2048,"batch":64,"n_epochs":10,
     "clip_range":0.30,"ent_coef":0.010,"gae_lambda":0.95,"diff":0.4},
    # Run 6 — high entropy
    {"lr":3e-4,"gamma":0.99,"n_steps":2048,"batch":64,"n_epochs":10,
     "clip_range":0.20,"ent_coef":0.050,"gae_lambda":0.95,"diff":0.4},
    # Run 7 — low entropy
    {"lr":3e-4,"gamma":0.99,"n_steps":2048,"batch":64,"n_epochs":10,
     "clip_range":0.20,"ent_coef":0.001,"gae_lambda":0.95,"diff":0.5},
    # Run 8 — short rollout
    {"lr":3e-4,"gamma":0.99,"n_steps":512, "batch":64,"n_epochs":10,
     "clip_range":0.20,"ent_coef":0.010,"gae_lambda":0.95,"diff":0.5},
    # Run 9 — lower GAE lambda
    {"lr":3e-4,"gamma":0.99,"n_steps":2048,"batch":64,"n_epochs":10,
     "clip_range":0.20,"ent_coef":0.010,"gae_lambda":0.80,"diff":0.6},
    # Run 10 — best-guess, hardest
    {"lr":2e-4,"gamma":0.99,"n_steps":2048,"batch":128,"n_epochs":15,
     "clip_range":0.20,"ent_coef":0.020,"gae_lambda":0.95,"diff":0.7},
]


# ── Sweep runners ──────────────────────────────────────────────────────────────
def run_sweep_reinforce() -> list:
    print("\n" + "="*58)
    print("REINFORCE Sweep – 10 runs × 3000 episodes")
    print("="*58)
    results = []
    for i, cfg in enumerate(REINFORCE_SWEEP):
        print(f"\n[{i+1}/10] {cfg}")
        results.append(train_reinforce(cfg, run_id=i, n_episodes=3000))

    fields = ["run_id","lr","gamma","entropy_coef","hidden","diff",
              "mean_reward","std_reward","train_time_s"]
    with open(CSV_RE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fields})

    _plot_rewards(results, "REINFORCE", "reinforce_rewards.png")
    _plot_entropy(results, "REINFORCE", "reinforce_entropy.png")

    best = max(results, key=lambda r: r["mean_reward"])
    with open(os.path.join(MODELS_DIR, "reinforce_best_config.json"), "w") as f:
        json.dump(REINFORCE_SWEEP[best["run_id"]], f, indent=2)
    print(f"\nBest REINFORCE: Run {best['run_id']:02d}  mean={best['mean_reward']:.2f}")
    return results


def run_sweep_ppo() -> list:
    print("\n" + "="*58)
    print("PPO Sweep – 10 runs × 500K timesteps")
    print("="*58)
    results = []
    for i, cfg in enumerate(PPO_SWEEP):
        print(f"\n[{i+1}/10] {cfg}")
        results.append(train_ppo(cfg, run_id=i))

    fields = ["run_id","lr","gamma","n_steps","batch","n_epochs",
              "clip_range","ent_coef","gae_lambda","diff",
              "mean_reward","std_reward","train_time_s"]
    with open(CSV_PPO, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fields})

    _plot_rewards(results, "PPO", "ppo_rewards.png")
    _plot_entropy(results, "PPO", "ppo_entropy.png")

    best = max(results, key=lambda r: r["mean_reward"])
    with open(os.path.join(MODELS_DIR, "ppo_best_config.json"), "w") as f:
        json.dump(PPO_SWEEP[best["run_id"]], f, indent=2)
    print(f"\nBest PPO: Run {best['run_id']:02d}  mean={best['mean_reward']:.2f}")
    return results


# ── Plot helpers ───────────────────────────────────────────────────────────────
def _plot_rewards(results: list, algo: str, fname: str) -> None:
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
        ax.set_title(f"Run {idx+1}  μ={res['mean_reward']:.1f}",fontsize=8)
        ax.set_xlabel("Episode",fontsize=7); ax.set_ylabel("Reward",fontsize=7)
        ax.tick_params(labelsize=6)
    fig.suptitle(f"{algo} — Episode Reward (blue=rolling mean, 50-ep window)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, fname)
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved {out}")


def _plot_entropy(results: list, algo: str, fname: str) -> None:
    plt.figure(figsize=(12, 4))
    for idx, res in enumerate(results):
        ents = res.get("entropies", [])
        if ents:
            plt.plot(ents, lw=1, label=f"Run {idx+1}", alpha=0.75)
    plt.title(f"{algo} — Policy Entropy over Training")
    plt.xlabel("Update step"); plt.ylabel("Entropy")
    plt.legend(fontsize=7, ncol=5)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, fname)
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved {out}")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PPO/REINFORCE on Kigali env v3")
    parser.add_argument("--algo", choices=["ppo","reinforce","all"],
                        default="ppo")
    parser.add_argument("--sweep", action="store_true")
    # PPO single-run args
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--gamma",      type=float, default=0.99)
    parser.add_argument("--n_steps",    type=int,   default=2048)
    parser.add_argument("--batch",      type=int,   default=64)
    parser.add_argument("--n_epochs",   type=int,   default=10)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--ent_coef",   type=float, default=0.01)
    # REINFORCE single-run args
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--hidden",       type=int,   default=128)
    parser.add_argument("--n_episodes",   type=int,   default=3000)
    parser.add_argument("--diff",         type=float, default=0.4)
    args = parser.parse_args()

    if args.sweep:
        if args.algo in ("reinforce", "all"): run_sweep_reinforce()
        if args.algo in ("ppo", "all"):       run_sweep_ppo()
    else:
        if args.algo == "ppo":
            cfg = {"lr":args.lr,"gamma":args.gamma,"n_steps":args.n_steps,
                   "batch":args.batch,"n_epochs":args.n_epochs,
                   "clip_range":args.clip_range,"ent_coef":args.ent_coef,
                   "gae_lambda":0.95,"diff":args.diff}
            r = train_ppo(cfg, run_id=99)
            print(f"\nPPO done. mean_reward={r['mean_reward']:.2f}")
        elif args.algo == "reinforce":
            cfg = {"lr":args.lr,"gamma":args.gamma,
                   "entropy_coef":args.entropy_coef,
                   "hidden":args.hidden,"diff":args.diff}
            r = train_reinforce(cfg, run_id=99, n_episodes=args.n_episodes)
            print(f"\nREINFORCE done. mean_reward={r['mean_reward']:.2f}")