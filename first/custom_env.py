"""
custom_env.py
═══════════════════════════════════════════════════════════════════════════════
Rwanda Business Viability Explorer
Grid-World Reinforcement Learning Environment  v2.0
═══════════════════════════════════════════════════════════════════════════════

MISSION CONTEXT
───────────────
A youth entrepreneur (the agent) enters a simulated Kigali sector and must
navigate a 2D grid representing the local market. The grid contains existing
competitor businesses, roads, landmarks, and zone types. The agent reads local
market signals at each cell and must find the best location to place a new
business — one with strong demand, good foot traffic, and low competition.

This directly simulates the core problem from the research proposal:
  "Aspiring entrepreneurs lack tools for assessing business viability at the
   sector level before committing capital." (Alliance Dushime, 2026)

═══════════════════════════════════════════════════════════════════════════════
GRID  (15 × 15 cells)
─────────────────────
Cell types:
  EMPTY       — open space, placeable
  ROAD        — street / main road (high foot traffic corridor)
  MARKET_HUB  — central market (very high demand, all business types)
  RESIDENTIAL — housing zone (moderate demand)
  COMPETITOR  — existing business already placed (same type)
  AGENT       — current agent position
  PLACED      — where agent committed (terminal)

Landmark types (overlaid on EMPTY/RESIDENTIAL cells):
  TAXI_STOP   — bus/moto-taxi stop  → boosts Retail, Food, Phone Repair
  HOSPITAL    — clinic or hospital  → boosts Food, Personal Services, Pharmacy
  SCHOOL      — school/university   → boosts Food, Digital, Printing/Stationery
  CHURCH      — church or mosque    → weekend foot traffic, boosts Food/Retail
  INDUSTRIAL  — factory/workshop    → boosts Light Production, Phone Repair

SECTORS  (5 — one episode = one sector)
  Kimironko — large daily market, very high foot traffic, dense commercial
  Remera    — government/commercial hub, educated demographic
  Kacyiru   — upscale residential, parliamentary zone, high income
  Gatenga   — mixed residential + light industrial, moderate density
  Gikondo   — main industrial zone, lower foot traffic, production demand

OBSERVATION SPACE  (13 features, continuous [0,1])
  [0]  competition_nearby   — competitor count within 2-cell radius
  [1]  foot_traffic         — road/landmark proximity score
  [2]  demand_score         — sector + landmark adjusted demand
  [3]  market_gap           — demand minus local supply
  [4]  infrastructure       — road access + utilities quality
  [5]  zone_type            — encoded cell type
  [6]  distance_to_hub      — normalised distance to nearest market hub
  [7]  landmark_boost       — demand boost from nearby landmarks (0=none,1=high)
  [8]  landmark_type        — encoded nearest landmark type
  [9]  agent_x              — normalised column position
  [10] agent_y              — normalised row position
  [11] steps_remaining      — fraction of max steps left
  [12] budget_remaining     — fraction of starting budget left

ACTION SPACE  (Discrete, 6)
  0 Move UP     1 Move DOWN    2 Move LEFT
  3 Move RIGHT  4 PLACE here   5 SCAN area

REWARD STRUCTURE
  Move           : -0.005  (step cost)
  Wall/invalid   : -0.05   (boundary or competitor cell)
  Scan           : -0.02   (info cost)
  PLACE          : viability mapped to [-1.0, +1.0]
                   +0.30 bonus  if viability > 0.75
                   -0.30 penalty if viability < 0.30
  No place (end) : -0.50

VIABILITY FORMULA
  viability = demand_score   × 0.30
            + foot_traffic   × 0.25
            + market_gap     × 0.25
            + landmark_boost × 0.10
            + infrastructure × 0.10
            - competition    × 0.20
  Clipped to [0, 1]

INSTALL
  pip install numpy pygame gymnasium stable-baselines3

USAGE
  env = RwandaBusinessEnv(sector="Kimironko",
                          business_type="Food & Beverage",
                          render_mode="human")
  obs, info = env.reset()
  while True:
      obs, r, done, trunc, info = env.step(env.action_space.sample())
      env.render()
      if done or trunc: break
  env.close()
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List

# ─────────────────────────────── constants ────────────────────────────────── #

GRID_H    = 15
GRID_W    = 15
MAX_STEPS = 60
INITIAL_BUDGET = 1.0
MOVE_COST = 0.005
SCAN_COST = 0.02
WALL_COST = 0.05

# ── Cell types ──────────────────────────────────────────────────────────────
EMPTY       = 0
ROAD        = 1
MARKET_HUB  = 2
RESIDENTIAL = 3
COMPETITOR  = 4
AGENT_CELL  = 5
PLACED      = 6

CELL_NAMES = {
    EMPTY: "Empty", ROAD: "Road", MARKET_HUB: "Market hub",
    RESIDENTIAL: "Residential", COMPETITOR: "Competitor",
    AGENT_CELL: "Agent", PLACED: "Placed",
}

# ── Landmark types (stored in separate overlay grid) ─────────────────────────
NO_LANDMARK = 0
TAXI_STOP   = 1
HOSPITAL    = 2
SCHOOL      = 3
CHURCH      = 4
INDUSTRIAL  = 5

LANDMARK_NAMES = {
    NO_LANDMARK: "None",    TAXI_STOP: "Taxi/Bus stop",
    HOSPITAL:    "Hospital", SCHOOL:   "School",
    CHURCH:      "Church",  INDUSTRIAL: "Industrial site",
}

# ── Actions ──────────────────────────────────────────────────────────────────
MOVE_UP    = 0
MOVE_DOWN  = 1
MOVE_LEFT  = 2
MOVE_RIGHT = 3
PLACE      = 4
SCAN       = 5
N_ACTIONS  = 6
OBS_DIM    = 13

ACTION_NAMES = {
    MOVE_UP: "Move UP",   MOVE_DOWN: "Move DOWN",
    MOVE_LEFT: "Move LEFT", MOVE_RIGHT: "Move RIGHT",
    PLACE: "PLACE business", SCAN: "SCAN area",
}

SECTORS = ["Kimironko", "Remera", "Kacyiru", "Gatenga", "Gikondo"]

BUSINESS_TYPES = [
    "Retail", "Food & Beverage", "Personal Services",
    "Tech / Phone Repair", "Light Production", "Digital / Social Commerce",
]

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTOR PROFILES
#  Each profile defines the structural layout and demographic character of
#  the sector, grounded in the research proposal data.
# ═══════════════════════════════════════════════════════════════════════════════

SECTOR_PROFILES = {
    "Kimironko": {
        "demand_base"    : 0.78,
        "demand_hotspot" : (7, 7),
        "hub_positions"  : [(7, 7), (7, 8), (8, 7)],
        "road_rows"      : [4, 9],
        "road_cols"      : [4, 10],
        "comp_density"   : 0.18,
        "infra_base"     : 0.72,
        "landmarks"      : [          # (row, col, type, radius_of_influence)
            (4, 4,  TAXI_STOP,  3),   # taxi stop at road junction
            (4, 10, TAXI_STOP,  3),   # second taxi stop
            (2, 12, SCHOOL,     4),   # school in residential zone
            (12, 2, CHURCH,     3),   # church in lower sector
        ],
        "description"    : (
            "Kigali's 2nd largest daily market. Very high foot traffic, "
            "dense commercial activity. Strong demand for food, services, "
            "retail. Two taxi stops drive corridor foot traffic."
        ),
    },
    "Remera": {
        "demand_base"    : 0.65,
        "demand_hotspot" : (5, 9),
        "hub_positions"  : [(5, 9), (5, 10)],
        "road_rows"      : [3, 8, 12],
        "road_cols"      : [5, 9],
        "comp_density"   : 0.14,
        "infra_base"     : 0.80,
        "landmarks"      : [
            (3, 5,  TAXI_STOP,  3),
            (8, 9,  TAXI_STOP,  3),
            (1, 2,  HOSPITAL,   5),   # hospital in upper residential
            (12, 12, SCHOOL,    4),
        ],
        "description"    : (
            "Government offices and commercial corridor. Educated demographic, "
            "strong demand for tech repair, professional services, digital "
            "commerce. Hospital drives food and service demand."
        ),
    },
    "Kacyiru": {
        "demand_base"    : 0.60,
        "demand_hotspot" : (4, 10),
        "hub_positions"  : [(4, 10)],
        "road_rows"      : [4, 10],
        "road_cols"      : [6, 10],
        "comp_density"   : 0.10,
        "infra_base"     : 0.90,
        "landmarks"      : [
            (4, 6,  TAXI_STOP,  3),
            (10, 10, TAXI_STOP, 3),
            (1, 13, CHURCH,     3),
            (7, 3,  SCHOOL,     4),   # school driving digital/food demand
        ],
        "description"    : (
            "Upscale residential and parliamentary zone. High-income households, "
            "strong demand for personal services, digital commerce. Low "
            "competition makes it attractive despite moderate raw demand."
        ),
    },
    "Gatenga": {
        "demand_base"    : 0.55,
        "demand_hotspot" : (8, 6),
        "hub_positions"  : [(8, 6), (8, 7)],
        "road_rows"      : [5, 10],
        "road_cols"      : [3, 8, 12],
        "comp_density"   : 0.12,
        "infra_base"     : 0.60,
        "landmarks"      : [
            (5, 3,  TAXI_STOP,   3),
            (10, 8, TAXI_STOP,   3),
            (2, 11, INDUSTRIAL,  4),  # light industrial in upper zone
            (13, 5, CHURCH,      3),
            (7, 13, SCHOOL,      3),
        ],
        "description"    : (
            "Mixed residential and light industrial. Moderate density, good "
            "demand for retail and light production. Industrial site boosts "
            "phone repair and production business types."
        ),
    },
    "Gikondo": {
        "demand_base"    : 0.50,
        "demand_hotspot" : (6, 5),
        "hub_positions"  : [(6, 5)],
        "road_rows"      : [3, 7, 12],
        "road_cols"      : [4, 9],
        "comp_density"   : 0.10,
        "infra_base"     : 0.68,
        "landmarks"      : [
            (3, 4,  TAXI_STOP,   3),
            (7, 9,  TAXI_STOP,   3),
            (5, 1,  INDUSTRIAL,  5),  # main industrial zone
            (11, 13, INDUSTRIAL, 4),  # secondary industrial
            (13, 2, CHURCH,      3),
        ],
        "description"    : (
            "Main industrial zone. Lower foot traffic but strong demand for "
            "light production, phone repair, tech services among workers. "
            "Two industrial landmarks create concentrated opportunity zones."
        ),
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
#  LANDMARK DEMAND MODIFIERS
#  How much each landmark boosts demand for each business type within its radius.
#  Values are additive boosts (0 = no effect, 1 = maximum boost).
# ═══════════════════════════════════════════════════════════════════════════════

LANDMARK_DEMAND_BOOST = {
    #                       Retail  Food&Bev  Services  Tech  Production  Digital
    TAXI_STOP:   np.array([0.25,   0.35,     0.20,    0.30,  0.05,      0.15]),
    HOSPITAL:    np.array([0.15,   0.30,     0.40,    0.10,  0.05,      0.10]),
    SCHOOL:      np.array([0.20,   0.35,     0.10,    0.20,  0.05,      0.30]),
    CHURCH:      np.array([0.20,   0.30,     0.10,    0.05,  0.05,      0.05]),
    INDUSTRIAL:  np.array([0.10,   0.20,     0.10,    0.35,  0.40,      0.10]),
    NO_LANDMARK: np.array([0.00,   0.00,     0.00,    0.00,  0.00,      0.00]),
}

# Business type index for vectorised lookups
BIZ_INDEX = {b: i for i, b in enumerate(BUSINESS_TYPES)}

# Per-sector business type fit modifier (how well each biz type fits the sector)
BIZ_SECTOR_FIT = {
    "Retail":
        {"Kimironko":1.10,"Remera":0.90,"Kacyiru":0.80,"Gatenga":0.95,"Gikondo":0.75},
    "Food & Beverage":
        {"Kimironko":1.20,"Remera":1.00,"Kacyiru":0.90,"Gatenga":0.85,"Gikondo":0.75},
    "Personal Services":
        {"Kimironko":1.05,"Remera":0.95,"Kacyiru":1.15,"Gatenga":0.90,"Gikondo":0.65},
    "Tech / Phone Repair":
        {"Kimironko":0.95,"Remera":1.20,"Kacyiru":1.00,"Gatenga":0.90,"Gikondo":1.15},
    "Light Production":
        {"Kimironko":0.70,"Remera":0.65,"Kacyiru":0.55,"Gatenga":1.10,"Gikondo":1.35},
    "Digital / Social Commerce":
        {"Kimironko":0.85,"Remera":1.10,"Kacyiru":1.20,"Gatenga":0.80,"Gikondo":0.70},
}

# ─────────────────── minimal space shims (no gymnasium needed) ────────────── #

class _Box:
    def __init__(self, n, dtype=np.float32):
        self.low   = np.zeros(n, dtype=dtype)
        self.high  = np.ones(n,  dtype=dtype)
        self.shape = (n,)
        self.dtype = dtype

    def contains(self, x):
        x = np.asarray(x)
        return (x.shape == self.shape and
                bool(np.all(x >= 0) and np.all(x <= 1)))

    def sample(self, rng=None):
        rng = rng or np.random.default_rng()
        return rng.random(self.shape).astype(self.dtype)

class _Discrete:
    def __init__(self, n):
        self.n = n

    def contains(self, x):
        return isinstance(x, (int, np.integer)) and 0 <= int(x) < self.n

    def sample(self, rng=None):
        rng = rng or np.random.default_rng()
        return int(rng.integers(0, self.n))

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════════

class RwandaBusinessEnv:
    """
    Rwanda Business Viability Explorer — Grid-World RL Environment v2.0

    Parameters
    ----------
    sector : str or None
        One of: "Kimironko","Remera","Kacyiru","Gatenga","Gikondo".
        None = random sector each episode.
    business_type : str
        The type of business the entrepreneur wants to open.
    noise_std : float
        Gaussian noise on observations (simulates imperfect market info).
    seed : int or None
        RNG seed for reproducibility.
    render_mode : str or None
        "human" = pygame window | "ansi" = terminal | None = silent.
    episode_variety : bool
        If True (Option B), competitor positions + demand noise vary each
        episode while sector layout stays fixed. Recommended for training.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 8}

    def __init__(
        self,
        sector        : Optional[str]  = None,
        business_type : str            = "Food & Beverage",
        noise_std     : float          = 0.04,
        seed          : Optional[int]  = None,
        render_mode   : Optional[str]  = None,
        episode_variety: bool          = True,
    ):
        assert sector is None or sector in SECTORS, \
            f"sector must be one of {SECTORS} or None"
        assert business_type in BIZ_INDEX, \
            f"business_type must be one of {BUSINESS_TYPES}"

        self.sector_name     = sector
        self.business_type   = business_type
        self.biz_idx         = BIZ_INDEX[business_type]
        self.noise_std       = noise_std
        self.render_mode     = render_mode
        self.episode_variety = episode_variety

        self.observation_space = _Box(OBS_DIM)
        self.action_space      = _Discrete(N_ACTIONS)
        self._rng = np.random.default_rng(seed)

        # pygame handles
        self._screen     = None
        self._clock      = None
        self._font       = None
        self._small_font = None
        self._big_font   = None

        # episode state
        self._grid           = None
        self._base_grid      = None
        self._landmark_grid  = None
        self._demand_map     = None
        self._traffic_map    = None
        self._infra_map      = None
        self._landmark_map   = None   # per-cell boost for chosen biz type
        self._viability_map  = None
        self._comp_map       = None
        self._agent_pos      = (0, 0)
        self._step_count     = 0
        self._budget         = INITIAL_BUDGET
        self._placed         = False
        self._last_reward    = 0.0
        self._last_scan      = None
        self._active_sector  = None
        self._episode_count  = 0

    # ═══════════════════════════════ RESET ════════════════════════════════════

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._active_sector = (
            SECTORS[int(self._rng.integers(0, len(SECTORS)))]
            if self.sector_name is None else self.sector_name
        )

        self._step_count   = 0
        self._budget       = INITIAL_BUDGET
        self._placed       = False
        self._last_reward  = 0.0
        self._last_scan    = None
        self._episode_count += 1

        self._build_world()
        self._agent_pos = self._random_start_cell()
        self._grid[self._agent_pos] = AGENT_CELL

        return self._get_obs(), self._get_info()

    # ═══════════════════════════════ STEP ═════════════════════════════════════

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        reward     = 0.0
        terminated = False
        truncated  = False
        r, c       = self._agent_pos

        # ── MOVE ──────────────────────────────────────────────────────────── #
        if action in (MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT):
            dr, dc = {MOVE_UP:(-1,0), MOVE_DOWN:(1,0),
                      MOVE_LEFT:(0,-1), MOVE_RIGHT:(0,1)}[action]
            nr, nc = r+dr, c+dc
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W:
                if self._grid[nr, nc] == COMPETITOR:
                    reward = -WALL_COST
                else:
                    self._grid[r, c]    = self._base_grid[r, c]
                    self._agent_pos     = (nr, nc)
                    self._grid[nr, nc]  = AGENT_CELL
                    reward = -MOVE_COST
            else:
                reward = -WALL_COST
            self._budget = max(0.0, self._budget - MOVE_COST)

        # ── SCAN ──────────────────────────────────────────────────────────── #
        elif action == SCAN:
            self._last_scan = float(self._viability_map[r, c])
            reward = -SCAN_COST
            self._budget = max(0.0, self._budget - SCAN_COST)

        # ── PLACE ─────────────────────────────────────────────────────────── #
        elif action == PLACE:
            if self._base_grid[r, c] in (COMPETITOR, MARKET_HUB, ROAD):
                reward = -WALL_COST
            else:
                v      = float(self._viability_map[r, c])
                reward = -1.0 + 2.0 * v
                if v > 0.75: reward += 0.30
                if v < 0.30: reward -= 0.30
                self._grid[r, c] = PLACED
                self._placed     = True
                terminated       = True

        self._step_count  += 1
        self._last_reward  = reward

        if not terminated:
            if self._step_count >= MAX_STEPS or self._budget <= 0:
                truncated = True
                reward   -= 0.50

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ═══════════════════════════ WORLD BUILDER ════════════════════════════════

    def _build_world(self):
        prof    = SECTOR_PROFILES[self._active_sector]
        biz_fit = BIZ_SECTOR_FIT[self.business_type][self._active_sector]

        # ── 1. Base structural grid ──────────────────────────────────────────
        base = np.full((GRID_H, GRID_W), EMPTY, dtype=np.int8)

        for row in prof["road_rows"]:  base[row, :] = ROAD
        for col in prof["road_cols"]:  base[:, col] = ROAD
        for hr, hc in prof["hub_positions"]: base[hr, hc] = MARKET_HUB

        # Residential zones — cells far from the demand hotspot
        hr_hub, hc_hub = prof["demand_hotspot"]
        for r in range(GRID_H):
            for c in range(GRID_W):
                if base[r, c] == EMPTY:
                    if abs(r-hr_hub) + abs(c-hc_hub) > 7:
                        base[r, c] = RESIDENTIAL

        # ── 2. Landmark overlay ──────────────────────────────────────────────
        lm_grid = np.zeros((GRID_H, GRID_W), dtype=np.int8)
        for lr, lc, ltype, _ in prof["landmarks"]:
            if 0 <= lr < GRID_H and 0 <= lc < GRID_W:
                lm_grid[lr, lc] = ltype

        self._landmark_grid = lm_grid

        # ── 3. Competitors (Option B: vary per episode) ──────────────────────
        placeable = [(r, c) for r in range(GRID_H) for c in range(GRID_W)
                     if base[r, c] in (EMPTY, RESIDENTIAL)]

        n_base = int(len(placeable) * prof["comp_density"])
        if self.episode_variety:
            # Vary count by ±20% and shuffle positions
            variation  = int(self._rng.integers(-max(1,n_base//5),
                                                 max(1,n_base//5)+1))
            n_comp = max(2, n_base + variation)
        else:
            n_comp = n_base

        comp_idx = self._rng.choice(len(placeable),
                                     size=min(n_comp, len(placeable)),
                                     replace=False)
        for idx in comp_idx:
            r2, c2 = placeable[idx]
            base[r2, c2] = COMPETITOR

        self._base_grid = base.copy()
        self._grid      = base.copy()

        # ── 4. Demand map ────────────────────────────────────────────────────
        demand = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        max_dist = float(GRID_H + GRID_W)
        for r in range(GRID_H):
            for c in range(GRID_W):
                dist = abs(r-hr_hub) + abs(c-hc_hub)
                demand[r, c] = prof["demand_base"] * (1.0 - dist/max_dist)

        # Apply landmark demand boosts within each landmark's radius
        for lr, lc, ltype, radius in prof["landmarks"]:
            boost_val = float(LANDMARK_DEMAND_BOOST[ltype][self.biz_idx])
            for r in range(GRID_H):
                for c in range(GRID_W):
                    d = abs(r-lr) + abs(c-lc)
                    if d <= radius:
                        # Boost decays with distance from landmark
                        decay = 1.0 - (d / (radius + 1))
                        demand[r, c] += boost_val * decay * 0.5

        # Apply sector fit modifier + episode noise
        ep_noise = (self._rng.normal(0, 0.06, demand.shape).astype(np.float32)
                    if self.episode_variety else np.zeros_like(demand))
        self._demand_map = np.clip(demand * biz_fit + ep_noise, 0.0, 1.0)

        # ── 5. Foot traffic map ──────────────────────────────────────────────
        traffic = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        for r in range(GRID_H):
            for c in range(GRID_W):
                ct = base[r, c]
                if ct == MARKET_HUB:
                    traffic[r, c] = 1.0
                elif ct == ROAD:
                    traffic[r, c] = 0.80
                else:
                    min_d = min(
                        min(abs(r-rr) for rr in prof["road_rows"]),
                        min(abs(c-rc) for rc in prof["road_cols"]),
                    )
                    traffic[r, c] = max(0.0, 0.75 - min_d * 0.12)

        # Taxi stops give an extra traffic boost within their radius
        for lr, lc, ltype, radius in prof["landmarks"]:
            if ltype == TAXI_STOP:
                for r in range(GRID_H):
                    for c in range(GRID_W):
                        d = abs(r-lr) + abs(c-lc)
                        if d <= radius:
                            decay = 1.0 - d / (radius + 1)
                            traffic[r, c] = min(1.0, traffic[r,c] + 0.20*decay)

        noise = self._rng.normal(0, 0.03, traffic.shape).astype(np.float32)
        self._traffic_map = np.clip(traffic + noise, 0.0, 1.0)

        # ── 6. Infrastructure map ────────────────────────────────────────────
        infra = np.full((GRID_H, GRID_W), prof["infra_base"], dtype=np.float32)
        for r in range(GRID_H):
            for c in range(GRID_W):
                if base[r, c] in (ROAD, MARKET_HUB):
                    infra[r, c] = min(1.0, prof["infra_base"] + 0.12)
        noise = self._rng.normal(0, 0.03, infra.shape).astype(np.float32)
        self._infra_map = np.clip(infra + noise, 0.0, 1.0)

        # ── 7. Landmark boost map (per-cell scalar for chosen biz type) ──────
        lm_boost = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        for lr, lc, ltype, radius in prof["landmarks"]:
            boost_val = float(LANDMARK_DEMAND_BOOST[ltype][self.biz_idx])
            for r in range(GRID_H):
                for c in range(GRID_W):
                    d = abs(r-lr) + abs(c-lc)
                    if d <= radius:
                        decay = 1.0 - d/(radius+1)
                        lm_boost[r, c] = min(1.0,
                                             lm_boost[r,c] + boost_val*decay)
        self._landmark_map = lm_boost

        # ── 8. Competition density map ───────────────────────────────────────
        comp_count = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        for r in range(GRID_H):
            for c in range(GRID_W):
                cnt = sum(
                    1 for dr in range(-2,3) for dc in range(-2,3)
                    if (0<=r+dr<GRID_H and 0<=c+dc<GRID_W
                        and base[r+dr,c+dc] == COMPETITOR)
                )
                comp_count[r, c] = min(cnt / 6.0, 1.0)
        self._comp_map = comp_count

        # ── 9. Viability map (reward surface) ────────────────────────────────
        market_gap = np.clip(self._demand_map - comp_count*0.5, 0.0, 1.0)
        viability  = (
            self._demand_map  * 0.30 +
            self._traffic_map * 0.25 +
            market_gap        * 0.25 +
            self._landmark_map* 0.10 +
            self._infra_map   * 0.10 -
            comp_count        * 0.20
        )
        self._viability_map = np.clip(viability, 0.0, 1.0)

    # ════════════════════════════ OBSERVATION ═════════════════════════════════

    def _get_obs(self):
        r, c  = self._agent_pos
        prof  = SECTOR_PROFILES[self._active_sector]
        base_cell = int(self._base_grid[r, c])

        # Nearest hub distance
        min_hub = min(abs(r-hr)+abs(c-hc)
                      for hr,hc in prof["hub_positions"])
        max_d = float(GRID_H + GRID_W)

        # Nearest landmark (type + distance)
        nearest_lm_type = NO_LANDMARK
        nearest_lm_dist = max_d
        for lr, lc, ltype, _ in prof["landmarks"]:
            d = abs(r-lr) + abs(c-lc)
            if d < nearest_lm_dist:
                nearest_lm_dist = d
                nearest_lm_type = ltype

        zone_enc = {ROAD:1.0, MARKET_HUB:0.9, EMPTY:0.5,
                    RESIDENTIAL:0.3, COMPETITOR:0.0,
                    AGENT_CELL:0.5, PLACED:0.5}

        obs = np.array([
            float(self._comp_map[r, c]),
            float(self._traffic_map[r, c]),
            float(self._demand_map[r, c]),
            float(np.clip(self._demand_map[r,c] - self._comp_map[r,c]*0.5,0,1)),
            float(self._infra_map[r, c]),
            float(zone_enc.get(base_cell, 0.5)),
            float(min_hub / max_d),
            float(self._landmark_map[r, c]),
            float(nearest_lm_type / 5.0),        # normalised landmark type
            float(c / (GRID_W - 1)),
            float(r / (GRID_H - 1)),
            float(1.0 - self._step_count / MAX_STEPS),
            float(self._budget),
        ], dtype=np.float32)

        noise = self._rng.normal(0, self.noise_std, obs.shape).astype(np.float32)
        return np.clip(obs + noise, 0.0, 1.0)

    # ══════════════════════════════ INFO ══════════════════════════════════════

    def _get_info(self):
        r, c = self._agent_pos
        v    = float(self._viability_map[r, c])
        bracket = ("Excellent" if v>=0.70 else "Good" if v>=0.50
                   else "Fair" if v>=0.35 else "Poor")

        # Nearest landmark description
        prof = SECTOR_PROFILES[self._active_sector]
        nearest = NO_LANDMARK
        nd = GRID_H + GRID_W
        for lr,lc,ltype,_ in prof["landmarks"]:
            d = abs(r-lr)+abs(c-lc)
            if d < nd:
                nd, nearest = d, ltype

        return {
            "sector"            : self._active_sector,
            "business_type"     : self.business_type,
            "step"              : self._step_count,
            "agent_pos"         : self._agent_pos,
            "budget"            : round(self._budget, 4),
            "cell_type"         : CELL_NAMES[int(self._base_grid[r,c])],
            "true_viability"    : round(v, 4),
            "viability_bracket" : bracket,
            "demand_here"       : round(float(self._demand_map[r,c]),3),
            "traffic_here"      : round(float(self._traffic_map[r,c]),3),
            "landmark_boost"    : round(float(self._landmark_map[r,c]),3),
            "nearest_landmark"  : LANDMARK_NAMES[nearest],
            "competition_here"  : round(float(self._comp_map[r,c]),3),
            "last_reward"       : round(self._last_reward, 4),
            "last_scan"         : self._last_scan,
            "placed"            : self._placed,
            "episode"           : self._episode_count,
        }

    # ════════════════════════════ HELPERS ═════════════════════════════════════

    def _random_start_cell(self):
        candidates = [(r,c) for r in range(GRID_H) for c in range(GRID_W)
                      if self._grid[r,c] in (EMPTY, RESIDENTIAL)]
        idx = int(self._rng.integers(0, len(candidates)))
        return candidates[idx]

    def decode_action(self, a): return ACTION_NAMES.get(a, f"Unknown({a})")
    def get_viability_map(self): return self._viability_map.copy()

    @property
    def sector_names(self):    return SECTORS
    @property
    def business_types(self):  return BUSINESS_TYPES
    @property
    def n_actions(self):       return N_ACTIONS
    @property
    def obs_dim(self):         return OBS_DIM
    @property
    def active_sector(self):   return self._active_sector

    def close(self):
        if self._screen is not None:
            try:
                import pygame; pygame.quit()
            except Exception: pass
            self._screen = None

    # ══════════════════════════ RENDER — ANSI ═════════════════════════════════

    def _render_ansi(self):
        SYMBOLS = {EMPTY:"  ", ROAD:"░░", MARKET_HUB:"HH",
                   RESIDENTIAL:"rr", COMPETITOR:"CC",
                   AGENT_CELL:"@@", PLACED:"**"}
        LM_SYM  = {TAXI_STOP:"TT", HOSPITAL:"++", SCHOOL:"SS",
                   CHURCH:"XX", INDUSTRIAL:"II"}
        r_a, c_a = self._agent_pos
        v = self._viability_map[r_a, c_a]
        sep = "─"*(GRID_W*2+2)
        print(f"\n{sep}")
        print(f" Sector: {self._active_sector:<12} | Biz: {self.business_type}")
        print(f" Step: {self._step_count}/{MAX_STEPS} | Budget: {self._budget:.2f}"
              f" | Viability here: {v:.2f}")
        print(sep)
        for r in range(GRID_H):
            row_str = "|"
            for c in range(GRID_W):
                cell = int(self._grid[r, c])
                lm   = int(self._landmark_grid[r, c])
                if cell == AGENT_CELL: sym = "@@"
                elif cell == PLACED:   sym = "**"
                elif cell == COMPETITOR: sym = "CC"
                elif lm != NO_LANDMARK: sym = LM_SYM.get(lm,"LL")
                else: sym = SYMBOLS.get(cell, "??")
                row_str += sym
            print(row_str + "|")
        print(sep)
        bar = "█"*int(v*30) + "░"*(30-int(v*30))
        print(f" Viability [{bar}] {v:.2f}")
        print(f" @@=Agent CC=Competitor HH=Hub ░░=Road")
        print(f" TT=Taxi ++ =Hospital SS=School XX=Church II=Industrial")
        print(sep)

    # ══════════════════════════ RENDER — PYGAME ═══════════════════════════════

    CELL_PX = 40
    PANEL_W = 330

    _C = {
        "bg":(18,18,24), "empty":(38,38,50), "road":(85,83,75),
        "hub":(255,195,45), "residential":(48,52,72),
        "competitor":(200,50,50), "agent":(55,155,225),
        "placed":(75,205,115), "grid_line":(28,28,38),
        "text":(235,235,235), "muted":(135,135,155),
        "good":(75,205,115), "warn":(235,175,45), "bad":(205,75,75),
        "panel":(22,22,30), "border":(52,52,72),
        "bar_bg":(42,42,58),
        # Landmark colours
        "taxi":(255,165,0), "hospital":(220,80,80),
        "school":(80,160,220), "church":(180,120,220),
        "industrial":(140,140,100),
        # Heat
        "h_green":(29,158,117), "h_amber":(186,117,23), "h_gray":(55,55,72),
    }

    _LM_COLOR = {
        TAXI_STOP:(255,165,0), HOSPITAL:(220,80,80),
        SCHOOL:(80,160,220),   CHURCH:(180,120,220),
        INDUSTRIAL:(140,140,100),
    }

    def _heat_color(self, v):
        C = self._C
        if v >= 0.65:
            t = (v-0.65)/0.35
            return tuple(int(C["h_amber"][i]+t*(C["h_green"][i]-C["h_amber"][i]))
                         for i in range(3))
        elif v >= 0.35:
            t = (v-0.35)/0.30
            return tuple(int(C["h_gray"][i]+t*(C["h_amber"][i]-C["h_gray"][i]))
                         for i in range(3))
        return C["h_gray"]

    def _init_pygame(self):
        import pygame
        if self._screen is None:
            pygame.init()
            W = GRID_W*self.CELL_PX + self.PANEL_W
            H = GRID_H*self.CELL_PX
            self._screen = pygame.display.set_mode((W, H))
            pygame.display.set_caption(
                f"Rwanda Business Env — {self._active_sector} | {self.business_type}")
            self._clock = pygame.time.Clock()
            try:
                self._font       = pygame.font.SysFont("monospace", 13)
                self._small_font = pygame.font.SysFont("monospace", 11)
                self._big_font   = pygame.font.SysFont("monospace", 15, bold=True)
            except Exception:
                f = pygame.font.Font(None, 14)
                self._font = self._small_font = self._big_font = f

    def _render_pygame(self):
        try:
            import pygame
        except ImportError:
            self._render_ansi(); return

        self._init_pygame()
        pygame.event.pump()

        C   = self._C
        px  = self.CELL_PX
        scr = self._screen
        scr.fill(C["bg"])

        # ── grid ─────────────────────────────────────────────────────────────
        for r in range(GRID_H):
            for c in range(GRID_W):
                x, y     = c*px, r*px
                cell     = int(self._grid[r, c])
                base_c   = int(self._base_grid[r, c])
                lm       = int(self._landmark_grid[r, c])
                v        = float(self._viability_map[r, c])

                # Base fill
                if base_c in (EMPTY, RESIDENTIAL):
                    fill = self._heat_color(v)
                elif base_c == ROAD:        fill = C["road"]
                elif base_c == MARKET_HUB:  fill = C["hub"]
                else:                        fill = C["empty"]

                pygame.draw.rect(scr, fill, (x+1,y+1,px-2,px-2))

                # Landmark marker (small coloured triangle in corner)
                if lm != NO_LANDMARK and cell not in (AGENT_CELL,PLACED,COMPETITOR):
                    lc_col = self._LM_COLOR.get(lm, C["muted"])
                    pts = [(x+px-2,y+2),(x+px-10,y+2),(x+px-2,y+10)]
                    pygame.draw.polygon(scr, lc_col, pts)

                # Cell overlays
                if cell == COMPETITOR:
                    pygame.draw.rect(scr, C["competitor"], (x+1,y+1,px-2,px-2))
                    t = self._font.render("C", True, (255,255,255))
                    scr.blit(t, (x+px//2-4, y+px//2-7))
                elif cell == AGENT_CELL:
                    pygame.draw.rect(scr, C["agent"], (x+1,y+1,px-2,px-2))
                    t = self._font.render("A", True, (255,255,255))
                    scr.blit(t, (x+px//2-4, y+px//2-7))
                elif cell == PLACED:
                    pygame.draw.rect(scr, C["placed"], (x+1,y+1,px-2,px-2))
                    t = self._font.render("P", True, (0,0,0))
                    scr.blit(t, (x+px//2-4, y+px//2-7))
                elif base_c == MARKET_HUB:
                    t = self._small_font.render("M", True, (0,0,0))
                    scr.blit(t, (x+px//2-4, y+px//2-7))
                elif base_c == ROAD:
                    for dx in range(x+4, x+px-4, 8):
                        pygame.draw.rect(scr,(115,110,95),(dx,y+px//2-1,5,2))

                pygame.draw.rect(scr, C["grid_line"], (x,y,px,px), 1)

        # ── panel ─────────────────────────────────────────────────────────── #
        ox = GRID_W*px
        pygame.draw.rect(scr, C["panel"], (ox,0,self.PANEL_W,GRID_H*px))
        pygame.draw.line(scr, C["border"], (ox,0),(ox,GRID_H*px), 1)

        def T(text, x, y, col=None, font=None):
            col  = col  or C["text"]
            font = font or self._font
            scr.blit(font.render(text, True, col), (ox+x, y))

        def BAR(label, val, y0, bad=False):
            bw = 195
            T(label, 10, y0, C["muted"])
            pygame.draw.rect(scr, C["bar_bg"],  (ox+10, y0+14, bw, 7))
            col = (C["bad"] if bad and val>0.4 else
                   C["warn"] if val<0.5 else C["good"])
            pygame.draw.rect(scr, col, (ox+10, y0+14, int(val*bw), 7))
            T(f"{val:.2f}", bw+14, y0+12, C["muted"], self._small_font)

        def LINE(y0):
            pygame.draw.line(scr, C["border"],(ox+6,y0),(ox+self.PANEL_W-6,y0),1)

        y = 10
        T(f"Rwanda Business Env", 10, y, font=self._big_font); y+=22
        T(f"Sector : {self._active_sector}", 10, y, C["muted"]); y+=16
        T(f"Biz    : {self.business_type}", 10, y, C["muted"]); y+=20
        LINE(y); y+=10

        T(f"Step   {self._step_count:>3}/{MAX_STEPS}", 10, y); y+=18
        T(f"Budget {self._budget:.3f}", 10, y); y+=18
        ra,ca = self._agent_pos
        T(f"Pos    ({ra:>2},{ca:>2})", 10, y); y+=20
        LINE(y); y+=10

        v_here = float(self._viability_map[ra,ca])
        vc = (C["good"] if v_here>0.65 else C["warn"] if v_here>0.35 else C["bad"])
        T("Cell snapshot", 10, y, C["muted"]); y+=18
        T(f"Viability  {v_here:.3f}", 10, y, vc); y+=18
        BAR("Demand",          self._demand_map[ra,ca],   y); y+=30
        BAR("Foot traffic",    self._traffic_map[ra,ca],  y); y+=30
        BAR("Landmark boost",  self._landmark_map[ra,ca], y); y+=30
        BAR("Competition",     self._comp_map[ra,ca],     y, bad=True); y+=30
        BAR("Infrastructure",  self._infra_map[ra,ca],    y); y+=30
        LINE(y); y+=10

        if self._last_reward != 0.0:
            rc2 = C["good"] if self._last_reward>0 else C["bad"]
            sign = "+" if self._last_reward>=0 else ""
            T(f"Last reward: {sign}{self._last_reward:.4f}", 10, y, rc2); y+=18
        if self._last_scan is not None:
            sv = self._last_scan
            sc = C["good"] if sv>0.65 else C["warn"] if sv>0.35 else C["bad"]
            T(f"Scan:  {sv:.3f}", 10, y, sc); y+=18
        LINE(y); y+=10

        # Legend
        T("Legend", 10, y, C["muted"]); y+=16
        entries = [
            (C["agent"],      "A  Agent"),
            (C["competitor"], "C  Competitor"),
            (C["hub"],        "M  Market hub"),
            (C["road"],       "░  Road"),
            (C["placed"],     "P  Placement"),
            (C["h_green"],    "   Great spot (>0.65)"),
            (C["h_amber"],    "   Okay spot (0.35-0.65)"),
            (C["h_gray"],     "   Poor spot (<0.35)"),
        ]
        lm_entries = [
            (self._LM_COLOR[TAXI_STOP],   "▶  Taxi/bus stop"),
            (self._LM_COLOR[HOSPITAL],    "▶  Hospital"),
            (self._LM_COLOR[SCHOOL],      "▶  School"),
            (self._LM_COLOR[CHURCH],      "▶  Church"),
            (self._LM_COLOR[INDUSTRIAL],  "▶  Industrial site"),
        ]
        for col, label in entries + lm_entries:
            pygame.draw.rect(scr, col, (ox+10, y, 10, 10))
            T(label, 26, y, C["muted"], self._small_font); y+=14

        LINE(y); y+=8
        T("0=Up 1=Down 2=Left 3=Right", 10, y, C["muted"], self._small_font); y+=13
        T("4=Place  5=Scan", 10, y, C["muted"], self._small_font)

        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def render(self):
        if self.render_mode == "human":  self._render_pygame()
        elif self.render_mode == "ansi": self._render_ansi()