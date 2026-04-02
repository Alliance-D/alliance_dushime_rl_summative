"""
custom_env.py  –  Kigali Retail Navigator  v3
===============================================
Mission
-------
A Rwandan entrepreneur navigates a dense Kigali sector map and
sequentially scouts locations for 4 small businesses:
  Ikivunge (Provisions), Inzu y'imiti (Pharmacy),
  Resitora (Restaurant), Salon/Coiffure (Hair Salon)

Map structure
-------------
The map is FIXED throughout the episode — all 4 business types are
placed on the map at reset and NEVER change. The map is dense: every
cell has a land-use type. No large empty patches.

What changes per phase
----------------------
The agent SWITCHES which business type it is currently scouting.
Phase 0 → scouting for Ikivunge   → Ikivunge  cells = RED rivals
Phase 1 → scouting for Pharmacy   → Pharmacy  cells = RED rivals
Phase 2 → scouting for Resitora   → Resitora  cells = RED rivals
Phase 3 → scouting for Salon      → Salon     cells = RED rivals

After each placement:
  - Agent moves to the next phase
  - The previously placed shop stays on the map (visible, neutral)
  - Only same-type existing businesses are competitors

Action Space (Discrete 6)
--------------------------
  0  Move UP       3  Move RIGHT
  1  Move DOWN     4  Survey current cell
  2  Move LEFT     5  PLACE current business here

Observation (Box 56-dim)
-------------------------
  [0:25]  5×5 local view. Rival type = 1.0, own placed = 0.9,
          neutral businesses = 0.5, landmarks scale by type, road/empty low.
  [25:27] agent position normalised
  [27]    viability at cell for current phase (0-1)
  [28]    step fraction
  [29]    sector id
  [30]    explored fraction
  [31]    current phase (0-3 normalised)
  [32]    nearest rival distance (normalised, 0=very close)
  [33]    foot traffic here
  [34]    viability for current phase (duplicate for emphasis)
  [35:39] viability all 4 phases at current cell
  [39:43] rival count by radius 1,2,3,4
  [43:47] sector landmark counts
  [47:56] padding

Reward (placement-only)
-----------------------
  Movement:              0.0   (free — no reward, no penalty)
  Wall hit:             -0.2
  Survey new cell:      +0.3
  Place top-30%, far from rival: +20 × v_norm
  Place decent (30-60%):         +8  × v_norm
  Place poor (<30%) or close rival: -10 × (1-v_norm)
  Episode complete (all 4):     +30 bonus
  Timeout:              -15 per missed placement

Termination
-----------
  All 4 placements made → terminated
  Max steps → truncated
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List, Set

# ── Constants ──────────────────────────────────────────────────────────────────
GS         = 15
MAX_STEPS  = 400
VIEW_R     = 2
VIEW_SIZE  = (2*VIEW_R+1)**2   # 25
OBS_DIM    = 56
N_ACTIONS  = 6
N_PHASES   = 4

SECTORS = {0:"Kimironko", 1:"Nyabugogo", 2:"Remera", 3:"Commercial Zone"}

# ── Cell types ─────────────────────────────────────────────────────────────────
EMPTY       = 0
ROAD        = 1
MARKET      = 2
RESIDENTIAL = 3
TAXI        = 4
HOSPITAL    = 5
SCHOOL      = 6
CHURCH      = 7
INDUSTRIAL  = 8
# The 4 Rwandan business types — always present on map
BIZ_GROCERY    = 10   # Ikivunge (Provisions / grocery shop)
BIZ_PHARMACY   = 11   # Inzu y'imiti (Pharmacy / chemist)
BIZ_RESTAURANT = 12   # Resitora (Restaurant / food stall)
BIZ_SALON      = 13   # Salon / Coiffure (Hair salon)

PHASE_TO_CELL  = {0:BIZ_GROCERY, 1:BIZ_PHARMACY, 2:BIZ_RESTAURANT, 3:BIZ_SALON}
CELL_TO_PHASE  = {v:k for k,v in PHASE_TO_CELL.items()}

BUSINESS_NAMES = {
    0: "Grocery",       # Provisions
    1: "Pharmacy",   # Pharmacy
    2: "Restaurant",       # Restaurant
    3: "Salon",          # Hair salon
}
BUSINESS_FULL  = {
    0: "Grocery (Provisions shop)",
    1: "Pharmacy (Chemist)",
    2: "Restaurant (Food stall)",
    3: "Salon (Hair salon)",
}
BUSINESS_COLORS = {
    0: (80,  210,  90),   # green
    1: (80,  150, 255),   # blue
    2: (255, 140,  40),   # orange
    3: (220,  80, 200),   # pink/purple
}

# ── Foot traffic per cell ──────────────────────────────────────────────────────
FOOT_TRAFFIC = {
    EMPTY:0.0, ROAD:0.4, MARKET:1.0, RESIDENTIAL:0.5, TAXI:0.9,
    HOSPITAL:0.7, SCHOOL:0.6, CHURCH:0.45, INDUSTRIAL:0.35,
    BIZ_GROCERY:0.6, BIZ_PHARMACY:0.5, BIZ_RESTAURANT:0.7, BIZ_SALON:0.55,
}

# ── Landmark affinity per business type ───────────────────────────────────────
PRIMARY_LANDMARKS = {
    0: {RESIDENTIAL, TAXI, MARKET},    # Ikivunge — near people & transport
    1: {HOSPITAL, RESIDENTIAL},        # Pharmacy — near hospital
    2: {MARKET, TAXI, INDUSTRIAL},     # Restaurant — busy areas
    3: {RESIDENTIAL, SCHOOL, CHURCH},  # Salon — neighbourhood services
}
SECONDARY_LANDMARKS = {
    0: {SCHOOL, CHURCH, MARKET},
    1: {SCHOOL, CHURCH},
    2: {RESIDENTIAL, SCHOOL},
    3: {MARKET, TAXI},
}

# ── Viability ring model ───────────────────────────────────────────────────────
REWARD_RINGS  = {1: 0.60, 2: 0.35, 3: 0.15}
PENALTY_RINGS = {1: 0.80, 2: 0.40}
ROAD_BONUS    = 0.35   # higher — neighbourhood streets now reach into residential

# ── Sector profiles (dense — every cell used) ─────────────────────────────────
SECTOR_PROFILES = {
    # Kimironko: residential + school dense, some market
    0: {"market":3,"taxi":3,"school":5,"church":4,"hospital":2,"industrial":1,
        "res_fill":True},
    # Nyabugogo: transport/market hub, dense commercial
    1: {"market":7,"taxi":8,"school":1,"church":1,"hospital":1,"industrial":4,
        "res_fill":False},
    # Remera: hospital zone, mixed
    2: {"market":3,"taxi":3,"school":4,"church":3,"hospital":4,"industrial":2,
        "res_fill":True},
    # Commercial Zone: commercial dense
    3: {"market":6,"taxi":5,"school":2,"church":1,"hospital":3,"industrial":5,
        "res_fill":False},
}
LANDMARK_CELL = {
    "market":MARKET,"taxi":TAXI,"school":SCHOOL,
    "church":CHURCH,"hospital":HOSPITAL,"industrial":INDUSTRIAL,
}


class KigaliRetailEnv(gym.Env):
    """
    Kigali Retail Navigator v3.
    Dense map, 4 Rwandan business types, agent switches competition phase.
    """
    metadata = {"render_modes":["human","rgb_array"],"render_fps":30}

    def __init__(self,
                 sector_id: Optional[int]=None,
                 render_mode: Optional[str]=None,
                 difficulty: float=0.5):
        """
        difficulty: controls rival density per type.
                    0.0 = 3 rivals/type, 1.0 = 12 rivals/type
        """
        super().__init__()
        self.sector_id   = sector_id
        self.render_mode = render_mode
        self.difficulty  = difficulty

        self.observation_space = spaces.Box(0.0,1.0,(OBS_DIM,),dtype=np.float32)
        self.action_space      = spaces.Discrete(N_ACTIONS)

        self.grid      = np.zeros((GS,GS),dtype=np.int32)
        self.viability = np.zeros((GS,GS,4),dtype=np.float32)

        self._sector   = 0
        self._pos      = (0,0)
        self._step     = 0
        self._phase    = 0
        self._visited:  Set[Tuple[int,int]] = set()
        self._surveyed: Set[Tuple[int,int]] = set()
        self._placed_positions: List[Tuple[int,int]] = []
        self._placed_types:     List[int] = []
        self._path:    List[Tuple[int,int]] = []
        self._n_rivals: Dict[int,int] = {}
        self._renderer = None

    # ── Reset ──────────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._sector  = (self.sector_id if self.sector_id is not None
                         else int(self.np_random.integers(0,4)))
        self._step    = 0
        self._phase   = 0
        self._visited = set()
        self._surveyed= set()
        self._placed_positions = []
        self._placed_types     = []
        self._path    = []

        self._build_grid()
        self._compute_viability()

        roads = [(r,c) for r in range(GS) for c in range(GS)
                 if self.grid[r,c]==ROAD]
        if roads:
            self._pos = roads[int(self.np_random.integers(0,len(roads)))]
        else:
            self._pos = (int(self.np_random.integers(0,GS)),
                         int(self.np_random.integers(0,GS)))
        self._visited.add(self._pos)
        self._path.append(self._pos)

        return self._obs(), {
            "sector":   SECTORS[self._sector],
            "n_rivals": dict(self._n_rivals),
            "phase":    self._phase,
            "business": BUSINESS_NAMES[self._phase],
        }

    # ── Step ───────────────────────────────────────────────────────────────────
    def step(self, action:int):
        assert self.action_space.contains(action)
        r,c = self._pos
        reward     = 0.0
        terminated = False
        truncated  = False
        info: Dict[str,Any] = {}

        # ── Move (0-3) ────────────────────────────────────────────────────────
        if action in (0,1,2,3):
            dr,dc = [(-1,0),(1,0),(0,-1),(0,1)][action]
            nr,nc = r+dr, c+dc
            if 0<=nr<GS and 0<=nc<GS:
                self._pos = (nr,nc)
                self._visited.add(self._pos)
                self._path.append(self._pos)
            else:
                reward -= 0.2
                info["wall"] = True

        # ── Survey (4) ────────────────────────────────────────────────────────
        elif action == 4:
            if self._pos not in self._surveyed:
                self._surveyed.add(self._pos)
                reward += 0.3
                info["surveyed"]       = True
                info["viability_here"] = round(float(self.viability[r,c,self._phase]),3)
                info["best_type"]      = int(np.argmax(self.viability[r,c]))

        # ── Place (5) ─────────────────────────────────────────────────────────
        elif action == 5:
            ct         = int(self.grid[r,c])
            rival_cell = PHASE_TO_CELL[self._phase]

            # Can't place on road, or on any already-placed business cell
            if ct == ROAD or ct in PHASE_TO_CELL.values():
                reward -= 2.0
                info["invalid_cell"] = True
            else:
                # Viability for current phase
                vt     = self.viability[:,:,self._phase]
                v_min  = float(vt.min())
                v_max  = float(vt.max())
                v_rng  = max(v_max-v_min, 1e-6)
                viab   = float(self.viability[r,c,self._phase])
                v_norm = (viab-v_min)/v_rng  # 0=worst, 1=best this episode

                # Rival proximity
                rival_dist = self._nearest_rival_dist(r,c,self._phase)
                close_rival = rival_dist <= 2

                if v_norm >= 0.70 and not close_rival:
                    reward += 20.0 * v_norm
                    info["optimal"] = True
                elif v_norm >= 0.40:
                    reward += 8.0 * v_norm
                    if close_rival:
                        reward -= 5.0
                        info["rival_nearby"] = True
                    info["decent"] = True
                else:
                    reward -= 10.0 * (1.0-v_norm)
                    info["poor"] = True

                info["viability"]  = round(viab,3)
                info["v_norm"]     = round(v_norm,3)
                info["business"]   = BUSINESS_NAMES[self._phase]
                info["phase"]      = self._phase

                # Mark placement on grid and record
                self.grid[r,c] = PHASE_TO_CELL[self._phase]
                self._placed_positions.append((r,c))
                self._placed_types.append(self._phase)

                # Recompute viability now that a new business is on the map
                self._compute_viability()

                # Advance phase
                self._phase += 1
                if self._phase >= N_PHASES:
                    terminated = True
                    reward    += 30.0
                    info["episode_complete"] = True
                else:
                    info["next_phase"]    = self._phase
                    info["next_business"] = BUSINESS_NAMES[self._phase]

        self._step += 1
        if self._step >= MAX_STEPS and not terminated:
            truncated = True
            missed    = N_PHASES - self._phase
            reward   -= 10.0 * missed   # -10 per missed placement
            info["timeout"]           = True
            info["missed_placements"] = missed

        if self.render_mode=="human":
            self.render()

        info.update({
            "step":    self._step,
            "sector":  SECTORS[self._sector],
            "pos":     self._pos,
            "phase":   self._phase,
            "visited": len(self._visited),
        })
        return self._obs(), reward, terminated, truncated, info

    # ── Observation ────────────────────────────────────────────────────────────
    def _obs(self) -> np.ndarray:
        r,c   = self._pos
        obs   = np.zeros(OBS_DIM, dtype=np.float32)
        phase = min(self._phase, N_PHASES-1)
        rival_cell = PHASE_TO_CELL[phase]

        # 5×5 local view
        idx = 0
        for dr in range(-VIEW_R, VIEW_R+1):
            for dc in range(-VIEW_R, VIEW_R+1):
                nr,nc = r+dr, c+dc
                if 0<=nr<GS and 0<=nc<GS:
                    ct = int(self.grid[nr,nc])
                    if ct == rival_cell:
                        obs[idx] = 1.0          # rival = max signal
                    elif ct in PHASE_TO_CELL.values():
                        obs[idx] = 0.55         # neutral business
                    elif ct == ROAD:
                        obs[idx] = 0.3
                    elif ct == EMPTY:
                        obs[idx] = 0.05
                    else:
                        obs[idx] = 0.4 + (ct/20.0)  # landmarks ~0.5-0.8
                else:
                    obs[idx] = 0.0
                idx += 1

        obs[25] = r/GS
        obs[26] = c/GS

        vt      = self.viability[:,:,phase]
        v_min   = float(vt.min())
        v_rng   = max(float(vt.max())-v_min, 1e-6)
        obs[27] = (float(self.viability[r,c,phase])-v_min)/v_rng
        obs[28] = self._step/MAX_STEPS
        obs[29] = self._sector/3.0
        obs[30] = len(self._visited)/(GS*GS)
        obs[31] = phase/3.0
        obs[32] = min(self._nearest_rival_dist(r,c,phase)/GS, 1.0)
        obs[33] = FOOT_TRAFFIC.get(int(self.grid[r,c]),0.1)
        # Road proximity: 1.0=on road, 0.75=adjacent, 0.5=2 away, 0=far
        # Helps agent learn that road-adjacent cells are better for business
        min_road_dist = GS*2
        for dr in range(-3,4):
            for dc in range(-3,4):
                nr,nc = r+dr,c+dc
                if 0<=nr<GS and 0<=nc<GS and self.grid[nr,nc]==ROAD:
                    min_road_dist = min(min_road_dist, abs(dr)+abs(dc))
        obs[34] = max(0.0, 1.0 - min_road_dist/4.0)

        for bt in range(4):
            bt_vt  = self.viability[:,:,bt]
            bt_min = float(bt_vt.min())
            bt_rng = max(float(bt_vt.max())-bt_min, 1e-6)
            obs[35+bt] = (float(self.viability[r,c,bt])-bt_min)/bt_rng

        rival_cell_code = PHASE_TO_CELL[phase]
        for ri,radius in enumerate([1,2,3,4]):
            count = sum(1 for dr in range(-radius,radius+1)
                        for dc in range(-radius,radius+1)
                        if 0<=r+dr<GS and 0<=c+dc<GS
                        and self.grid[r+dr,c+dc]==rival_cell_code)
            obs[39+ri] = min(count/10.0, 1.0)

        obs[43] = np.sum(self.grid==MARKET)  /10.0
        obs[44] = np.sum(self.grid==TAXI)    /10.0
        obs[45] = np.sum(self.grid==SCHOOL)  /10.0
        obs[46] = np.sum(self.grid==HOSPITAL)/10.0

        return np.clip(obs,0.0,1.0)

    # ── Grid building (dense — minimal empty) ─────────────────────────────────
    def _build_grid(self):
        rng     = self.np_random
        grid    = np.zeros((GS,GS),dtype=np.int32)
        profile = SECTOR_PROFILES[self._sector]

        # ── Roads: 2-layer system ────────────────────────────────────────────
        # Layer 1: 2-3 main arteries (full-width H and V roads)
        n_main = int(rng.integers(2,4))
        main_rows = sorted(rng.choice(GS, n_main, replace=False).tolist())
        main_cols = sorted(rng.choice(GS, n_main, replace=False).tolist())
        for rr in main_rows: grid[rr,:] = ROAD
        for cc in main_cols: grid[:,cc] = ROAD

        # Layer 2: 6-10 neighbourhood streets (short segments branching off mains)
        # Streets run perpendicular from main roads into residential blocks.
        # Length: 3-6 cells, so they reach into the neighbourhood.
        n_streets = int(rng.integers(6,11))
        all_road_rows = set(main_rows); all_road_cols = set(main_cols)
        for _ in range(n_streets * 4):   # extra attempts for placement
            if n_streets <= 0: break
            # Pick a random main road cell as the branch point
            if float(rng.random()) < 0.5 and main_rows:
                # Branch vertically off a horizontal main road
                base_r = int(rng.choice(main_rows))
                base_c = int(rng.integers(1, GS-1))
                length = int(rng.integers(3,7))
                direction = 1 if base_r < GS//2 else -1
                for step in range(1, length+1):
                    nr = base_r + direction*step
                    if 0<=nr<GS and grid[nr,base_c]==EMPTY:
                        grid[nr,base_c] = ROAD
                        all_road_cols.add(base_c)
                n_streets -= 1
            elif main_cols:
                # Branch horizontally off a vertical main road
                base_c = int(rng.choice(main_cols))
                base_r = int(rng.integers(1, GS-1))
                length = int(rng.integers(3,7))
                direction = 1 if base_c < GS//2 else -1
                for step in range(1, length+1):
                    nc = base_c + direction*step
                    if 0<=nc<GS and grid[base_r,nc]==EMPTY:
                        grid[base_r,nc] = ROAD
                        all_road_rows.add(base_r)
                n_streets -= 1

        # ── Residential fill (dense blobs + scatter) ──────────────────────────
        if profile["res_fill"]:
            # Large residential zones
            for _ in range(4):
                or2 = int(rng.integers(0,GS-4))
                oc  = int(rng.integers(0,GS-4))
                for dr in range(int(rng.integers(3,6))):
                    for dc in range(int(rng.integers(3,6))):
                        nr,nc = or2+dr, oc+dc
                        if 0<=nr<GS and 0<=nc<GS and grid[nr,nc]==EMPTY:
                            grid[nr,nc] = RESIDENTIAL

        # Fill remaining EMPTY cells with RESIDENTIAL to make map dense
        for rr in range(GS):
            for cc in range(GS):
                if grid[rr,cc] == EMPTY:
                    grid[rr,cc] = RESIDENTIAL

        # ── Landmark placement ────────────────────────────────────────────────
        for lname in ["market","taxi","school","church","hospital","industrial"]:
            count = profile[lname]
            cv    = LANDMARK_CELL[lname]
            placed = 0
            for _ in range(500):
                if placed >= count: break
                rr2 = int(rng.integers(0,GS))
                cc2 = int(rng.integers(0,GS))
                if grid[rr2,cc2] in (RESIDENTIAL, EMPTY):
                    grid[rr2,cc2] = cv
                    placed += 1
                    # cluster chance
                    if float(rng.random()) < 0.45 and placed < count:
                        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr,nc = rr2+dr,cc2+dc
                            if (0<=nr<GS and 0<=nc<GS
                                    and grid[nr,nc]==RESIDENTIAL
                                    and placed < count):
                                grid[nr,nc]=cv; placed+=1; break

        # ── Place rival businesses ─────────────────────────────────────────────
        # Rivals per type: difficulty 0.0 → 6/type, 1.0 → 18/type
        # These are FIXED businesses already on the map — they don't change.
        # Higher difficulty = more competitors = harder to find good spots.
        min_rivals = 6
        max_rivals = 18
        base = int(min_rivals + self.difficulty * (max_rivals - min_rivals))
        # Add per-episode randomness (±2)
        base = int(rng.integers(max(min_rivals, base-2),
                                min(max_rivals+1, base+3)))
        self._n_rivals = {}

        for btype in range(N_PHASES):
            biz_cell = PHASE_TO_CELL[btype]
            placed = 0
            attempts = 0
            while placed < base and attempts < 1500:
                attempts += 1
                rr2 = int(rng.integers(0,GS))
                cc2 = int(rng.integers(0,GS))
                # Place on residential (or empty if residential ran out)
                if grid[rr2,cc2] in (RESIDENTIAL, EMPTY):
                    grid[rr2,cc2] = biz_cell
                    placed += 1
            self._n_rivals[btype] = placed

        self.grid = grid

    # ── Viability (spatial ring model) ────────────────────────────────────────
    def _compute_viability(self):
        """
        For each phase, viability = landmark reward rings - rival penalty rings.
        Neutral businesses (other types) count as secondary demand sources.
        Only same-type businesses are rivals.
        """
        road_set = {(r,c) for r in range(GS) for c in range(GS)
                    if self.grid[r,c]==ROAD}

        # Index positions
        cell_pos: Dict[int,List] = {}
        for ct in ([MARKET,RESIDENTIAL,TAXI,HOSPITAL,SCHOOL,CHURCH,INDUSTRIAL]
                   + list(PHASE_TO_CELL.values())):
            cell_pos[ct] = [(r,c) for r in range(GS) for c in range(GS)
                            if self.grid[r,c]==ct]

        v = np.zeros((GS,GS,4),dtype=np.float32)

        for bt in range(4):
            prim       = PRIMARY_LANDMARKS[bt]
            sec        = SECONDARY_LANDMARKS[bt]
            rival_cell = PHASE_TO_CELL[bt]

            # Reward from landmarks + neutral businesses
            for ct,positions in cell_pos.items():
                if ct == rival_cell: continue
                is_prim = ct in prim
                is_sec  = ct in sec
                # Neutral businesses contribute as secondary demand
                if ct in PHASE_TO_CELL.values() and ct != rival_cell:
                    is_sec = True
                if not is_prim and not is_sec: continue
                factor = 1.0 if is_prim else 0.18

                for (lr,lc) in positions:
                    for radius,base_rw in REWARD_RINGS.items():
                        rw = base_rw * factor
                        for r in range(max(0,lr-radius),min(GS,lr+radius+1)):
                            for c in range(max(0,lc-radius),min(GS,lc+radius+1)):
                                if abs(r-lr)+abs(c-lc) <= radius:
                                    v[r,c,bt] += rw

            # Penalty from rivals
            for (cr,cc) in cell_pos.get(rival_cell,[]):
                for radius,penalty in PENALTY_RINGS.items():
                    for r in range(max(0,cr-radius),min(GS,cr+radius+1)):
                        for c in range(max(0,cc-radius),min(GS,cc+radius+1)):
                            if abs(r-cr)+abs(c-cc) <= radius:
                                v[r,c,bt] -= penalty

            # Road bonus
            for r in range(GS):
                for c in range(GS):
                    if self.grid[r,c]==ROAD: continue
                    for dr in [-1,0,1]:
                        for dc in [-1,0,1]:
                            if (r+dr,c+dc) in road_set:
                                v[r,c,bt] += ROAD_BONUS
                                break
                        else:
                            continue
                        break

        self.viability = v

    def _nearest_rival_dist(self, r:int, c:int, phase:int) -> float:
        rival_cell = PHASE_TO_CELL[phase]
        best = float(GS*2)
        for rr in range(GS):
            for cc in range(GS):
                if self.grid[rr,cc]==rival_cell:
                    best = min(best, abs(r-rr)+abs(c-cc))
        return best

    # ── Convenience properties ─────────────────────────────────────────────────
    @property
    def _placed_pos(self):
        return self._placed_positions[-1] if self._placed_positions else None

    @property
    def _placed_type(self):
        return self._placed_types[-1] if self._placed_types else None

    @property
    def current_business(self) -> str:
        return BUSINESS_NAMES[min(self._phase, N_PHASES-1)]

    # ── Render ─────────────────────────────────────────────────────────────────
    def render(self):
        if self.render_mode=="human":
            try:
                from environment.rendering import KigaliRenderer
            except ImportError:
                from rendering import KigaliRenderer
            if self._renderer is None:
                self._renderer = KigaliRenderer()
            self._renderer.draw(
                grid=self.grid,
                viability=self.viability,
                sector_name=SECTORS[self._sector],
                agent_pos=self._pos,
                path=self._path,
                placed_positions=self._placed_positions,
                placed_types=self._placed_types,
                phase=self._phase,
                step=self._step,
                visited=self._visited,
                surveyed=self._surveyed,
            )

    def close(self):
        if self._renderer:
            self._renderer.close()
            self._renderer = None


def make_env(sector_id=None, render_mode=None, difficulty=0.5):
    def _init():
        return KigaliRetailEnv(sector_id=sector_id,
                               render_mode=render_mode,
                               difficulty=difficulty)
    return _init


if __name__=="__main__":
    env = KigaliRetailEnv(difficulty=0.5)
    obs,info = env.reset(seed=42)
    print(f"Sector: {info['sector']}")
    print(f"Rivals/type: {info['n_rivals']}")
    print(f"Phase: {info['phase']} ({info['business']})")
    print(f"Obs: {obs.shape}  Actions: {env.action_space.n}")
    print(f"Viability range: {env.viability.min():.2f} to {env.viability.max():.2f}")
    import collections
    cell_counts = collections.Counter(env.grid.flatten().tolist())
    print(f"Cell counts: {dict(sorted(cell_counts.items()))}")
    env.close()
    print("OK")