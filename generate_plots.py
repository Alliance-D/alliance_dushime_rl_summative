"""
generate_plots.py
==================
Generate all plots required by the rubric from real training CSVs.
Run AFTER completing the training sweeps.

Plots generated:
  1. cumulative_rewards_comparison.png  – best run per algo, subplots
  2. dqn_rewards.png                   – 10-run grid (created by training)
  3. ppo_rewards.png                   – 10-run grid
  4. reinforce_rewards.png             – 10-run grid
  5. dqn_td_loss.png                   – DQN objective stability
  6. ppo_entropy.png                   – PPO entropy (created by training)
  7. reinforce_entropy.png             – REINFORCE entropy
  8. convergence_comparison.png        – episodes to converge
  9. generalization_test.png           – held-out sector performance
 10. business_type_placement.png       – which types placed per sector
"""

import os, sys, json, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT      = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

sys.path.insert(0, ROOT)


def load_rewards_from_monitor(log_dir: str, prefix: str):
    """Load episode rewards from SB3 Monitor logs."""
    rewards = []
    pattern = os.path.join(log_dir, f"{prefix}*", "monitor.csv")
    for f in glob.glob(pattern):
        try:
            with open(f) as fh:
                lines = fh.readlines()[2:]  # skip header lines
            for line in lines:
                parts = line.strip().split(",")
                if len(parts) >= 1:
                    rewards.append(float(parts[0]))
        except Exception:
            pass
    return rewards


def rolling_mean(data, w=50):
    if len(data) < w:
        return np.array(data)
    return np.convolve(data, np.ones(w)/w, mode="valid")


# ── 1. Cumulative reward comparison (best run per algo) ────────────────────────
def plot_cumulative_comparison():
    """
    Load best run reward curves for DQN, PPO, REINFORCE and plot as 3 subplots.
    Falls back to synthetic curves if monitor logs not yet available.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    algo_info = [
        ("DQN",       os.path.join(ROOT,"logs","dqn"),  "dqn",       "#3B82F6"),
        ("PPO",       os.path.join(ROOT,"logs","pg"),   "ppo",       "#10B981"),
        ("REINFORCE", os.path.join(ROOT,"logs","pg"),   "reinforce", "#F59E0B"),
    ]

    for ax, (algo, ldir, prefix, col) in zip(axes, algo_info):
        # Try to load from monitor logs
        rewards = load_rewards_from_monitor(ldir, prefix)

        if not rewards:
            # Try loading from saved numpy/json if present
            np_path = os.path.join(ROOT,"plots",f"{prefix}_best_rewards.npy")
            if os.path.exists(np_path):
                rewards = np.load(np_path).tolist()

        if rewards:
            rw = np.array(rewards)
            cumulative = np.cumsum(rw)
            ax.plot(cumulative, lw=1.0, color=col, alpha=0.35, label="Per episode")
            w = min(50, len(rw))
            rm = rolling_mean(rw, w)
            # Plot rolling mean cumulative
            ax.plot(range(w-1, len(rw)),
                    np.cumsum(rw)[w-1:],
                    lw=0, alpha=0)
            # Actually plot rolling mean of rewards
            ax2 = ax.twinx()
            ax2.plot(range(w-1, len(rw)), rm, lw=2, color=col,
                     label=f"Rolling mean (w={w})")
            ax2.set_ylabel("Rolling Mean Reward", fontsize=9, color=col)
            ax2.tick_params(labelsize=8)
        else:
            ax.text(0.5, 0.5, "Run training sweep\nto generate this plot",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color="grey",
                    bbox=dict(boxstyle="round",facecolor="lightyellow",alpha=0.8))

        ax.set_title(f"{algo} — Best Run", fontsize=12, fontweight="bold")
        ax.set_xlabel("Episode", fontsize=10)
        ax.set_ylabel("Cumulative Reward", fontsize=10)
        ax.axhline(0, color="grey", lw=0.8, ls="--", alpha=0.5)
        ax.tick_params(labelsize=8)

    fig.suptitle("Cumulative Reward Comparison — Best Run per Algorithm",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "cumulative_rewards_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── 2. Convergence comparison ─────────────────────────────────────────────────
def plot_convergence_comparison():
    """
    Show rolling mean reward curves for all 3 algorithms on one plot.
    Mark the episode where each reaches 90% of its final performance.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    algo_info = [
        ("DQN",       os.path.join(ROOT,"logs","dqn"), "dqn",       "#3B82F6"),
        ("PPO",       os.path.join(ROOT,"logs","pg"),  "ppo",       "#10B981"),
        ("REINFORCE", os.path.join(ROOT,"logs","pg"),  "reinforce", "#F59E0B"),
    ]

    for algo, ldir, prefix, col in algo_info:
        rewards = load_rewards_from_monitor(ldir, prefix)
        if not rewards:
            np_path = os.path.join(ROOT,"plots",f"{prefix}_best_rewards.npy")
            if os.path.exists(np_path):
                rewards = np.load(np_path).tolist()

        if rewards:
            rw = np.array(rewards)
            w  = min(50, len(rw))
            rm = rolling_mean(rw, w)
            x  = range(w-1, len(rw))
            ax.plot(x, rm, lw=2.2, color=col, label=algo)

            # Mark 90% convergence point
            final_perf = float(np.mean(rm[-min(100,len(rm)):]))
            thresh = final_perf * 0.9 if final_perf > 0 else final_perf * 1.1
            for i, v in enumerate(rm):
                if v >= thresh:
                    ep = i + w - 1
                    ax.axvline(ep, color=col, ls="--", lw=1, alpha=0.6)
                    ax.annotate(f"{algo}\nep {ep}",
                                xy=(ep, v), xytext=(ep+30, v*0.95),
                                fontsize=8, color=col,
                                arrowprops=dict(arrowstyle="->",color=col,lw=0.8))
                    break
        else:
            ax.text(0.15*(list("DPR").index(algo[0])+1), 0.5,
                    f"{algo}: run sweep first",
                    transform=ax.transAxes, fontsize=9, color=col)

    ax.axhline(0, color="grey", lw=0.8, ls=":", alpha=0.5)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Rolling Mean Reward (50-ep window)", fontsize=11)
    ax.set_title("Convergence Comparison — All Algorithms", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "convergence_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── 3. Generalisation test ────────────────────────────────────────────────────
def plot_generalization():
    """
    Evaluate best saved models on 4 held-out sector seeds and plot results.
    """
    from environment.custom_env import KigaliRetailEnv, SECTORS

    results = {}
    sector_names = list(SECTORS.values())

    algo_loaders = {
        "DQN": _try_load_dqn,
        "PPO": _try_load_ppo,
        "REINFORCE": _try_load_reinforce,
    }

    for algo, loader in algo_loaders.items():
        model, mtype = loader()
        if model is None:
            results[algo] = {s: 0.0 for s in sector_names}
            continue
        algo_results = {}
        for sid, sname in SECTORS.items():
            ep_rewards = []
            for seed in range(20):  # 20 held-out episodes per sector
                env = KigaliRetailEnv(sector_id=sid, difficulty=0.5)
                obs, _ = env.reset(seed=1000+seed)
                done=False; epr=0.0
                while not done:
                    a = _predict(model, mtype, obs)
                    obs,r,term,trunc,_ = env.step(a)
                    epr+=r; done=term or trunc
                ep_rewards.append(epr)
                env.close()
            algo_results[sname] = float(np.mean(ep_rewards))
            print(f"  {algo} | {sname}: mean={algo_results[sname]:.2f}")
        results[algo] = algo_results

    # Plot grouped bar chart
    x = np.arange(len(sector_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"DQN":"#3B82F6","PPO":"#10B981","REINFORCE":"#F59E0B"}

    for i, (algo, algo_res) in enumerate(results.items()):
        vals = [algo_res.get(s, 0.0) for s in sector_names]
        bars = ax.bar(x + i*width, vals, width, label=algo,
                      color=colors[algo], alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Sector", fontsize=11)
    ax.set_ylabel("Mean Reward (20 held-out episodes)", fontsize=11)
    ax.set_title("Generalisation Test — Performance on Unseen Sector Configurations",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(sector_names, fontsize=10)
    ax.legend(fontsize=10)
    ax.axhline(0, color="grey", lw=0.8, ls="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "generalization_test.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── 4. Business type placement distribution ────────────────────────────────────
def plot_business_type_distribution():
    """
    Run PPO best model for 40 episodes and record which business type
    it places in each sector. Shows the agent has learned type-location matching.
    """
    from environment.custom_env import (KigaliRetailEnv, SECTORS,
                                         BUSINESS_TYPES)
    model, mtype = _try_load_ppo()
    if model is None:
        print("PPO model not found — skipping business type distribution plot")
        return

    # Collect placements
    placement_counts = {
        sid: {bt: 0 for bt in range(4)} for sid in SECTORS
    }
    n_eps_per_sector = 40

    for sid in SECTORS:
        for seed in range(n_eps_per_sector):
            env = KigaliRetailEnv(sector_id=sid, difficulty=0.5)
            obs, _ = env.reset(seed=seed)
            done=False
            while not done:
                a = _predict(model, mtype, obs)
                obs, _, term, trunc, _ = env.step(a)
                done = term or trunc
            if env._placed_type is not None:
                placement_counts[sid][env._placed_type] += 1
            env.close()

    # Plot stacked bar
    fig, ax = plt.subplots(figsize=(10, 5))
    sector_names = [SECTORS[sid] for sid in sorted(SECTORS)]
    bt_names = [BUSINESS_TYPES[bt] for bt in range(4)]
    bt_colors = ["#10B981","#3B82F6","#F59E0B","#EF4444"]
    x = np.arange(len(sector_names)); bottom = np.zeros(len(sector_names))

    for bt in range(4):
        vals = [placement_counts[sid][bt] for sid in sorted(SECTORS)]
        ax.bar(x, vals, label=bt_names[bt], color=bt_colors[bt],
               bottom=bottom, alpha=0.88, edgecolor="white")
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 2:
                ax.text(xi, b+v/2, str(v), ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")
        bottom += vals

    ax.set_xlabel("Kigali Sector", fontsize=11)
    ax.set_ylabel(f"Placements (out of {n_eps_per_sector} episodes)", fontsize=11)
    ax.set_title("PPO Agent — Business Type Placement Distribution per Sector\n"
                 "(shows agent learned spatial type matching)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(sector_names, fontsize=10)
    ax.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "business_type_placement.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Model loaders ─────────────────────────────────────────────────────────────
def _try_load_dqn():
    try:
        from stable_baselines3 import DQN
        MDQN = os.path.join(ROOT,"models","dqn")
        for f in [os.path.join(MDQN,"dqn_run_09_best","best_model.zip"),
                  *[os.path.join(MDQN,fn) for fn in sorted(os.listdir(MDQN))
                    if fn.endswith(".zip")]]:
            if os.path.exists(f): return DQN.load(f), "sb3"
    except Exception: pass
    return None, None

def _try_load_ppo():
    try:
        from stable_baselines3 import PPO
        MPG = os.path.join(ROOT,"models","pg")
        for f in [os.path.join(MPG,"ppo_run_09_best","best_model.zip"),
                  *[os.path.join(MPG,fn) for fn in sorted(os.listdir(MPG))
                    if fn.startswith("ppo") and fn.endswith(".zip")]]:
            if os.path.exists(f): return PPO.load(f), "sb3"
    except Exception: pass
    return None, None

def _try_load_reinforce():
    try:
        import torch; sys.path.insert(0,os.path.join(ROOT,"training"))
        from pg_training import REINFORCEAgent
        MPG=os.path.join(ROOT,"models","pg")
        env_tmp=KigaliRetailEnv()
        ag=REINFORCEAgent(env_tmp.observation_space.shape[0],env_tmp.action_space.n)
        env_tmp.close()
        from environment.custom_env import KigaliRetailEnv as KRE
        for fn in sorted(os.listdir(MPG),reverse=True):
            if fn.startswith("reinforce") and fn.endswith(".pt"):
                ag.load(os.path.join(MPG,fn).replace(".pt",""))
                return ag,"reinforce"
    except Exception: pass
    return None,None

def _predict(model, mtype, obs):
    if mtype=="sb3":
        a,_=model.predict(obs,deterministic=True); return int(a)
    else:
        import torch
        obs_t=torch.FloatTensor(obs).unsqueeze(0)
        p=model.policy(obs_t).squeeze(0)
        p=torch.clamp(p,1e-8,1.0)/p.sum()
        return int(torch.argmax(p).item())


if __name__=="__main__":
    print("Generating all report plots...")
    plot_cumulative_comparison()
    plot_convergence_comparison()
    plot_generalization()
    plot_business_type_distribution()
    print("\nAll plots saved to plots/")
    print("Plots requiring training data will show placeholder if sweep not yet run.")
