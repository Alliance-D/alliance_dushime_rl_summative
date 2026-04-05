"""
rendering.py  –  Kigali Retail Navigator v3
============================================
SimCity-style dense map. All 4 Rwandan business types have real sprites.
Rival businesses (same type as current phase) glow RED.
Own placed shops glow with their business colour.
Agent shown as animated person with blue trail.
"""

from __future__ import annotations
import sys, os, time, math, random
import numpy as np
from typing import Optional, Tuple, List, Set, Dict

# ── Layout ─────────────────────────────────────────────────────────────────────
CELL  = 42
GS    = 15
GPX   = CELL * GS        # 630
PANEL = 265
WIN_W = GPX + PANEL       # 895
WIN_H = GPX + 68          # 698
TOP   = 68
FPS   = 30

# ── Palette ────────────────────────────────────────────────────────────────────
TERRAIN    = ( 88,  68,  42)
TERRAIN_D  = ( 72,  55,  34)
GRASS_F    = ( 40,  78,  35)
GRASS_D    = ( 32,  62,  28)
ROAD_A     = ( 55,  58,  65)
ROAD_L     = (185, 175,  48)
ROAD_C     = ( 72,  76,  84)
PAVEMENT   = (125, 120, 108)
PANEL_BG   = (  8,  13,  26)
ACCENT     = ( 56, 189, 248)
TW         = (230, 240, 255)
TD         = (100, 120, 155)
SCORE_G    = ( 70, 215,  95)
SCORE_Y    = (235, 195,  45)
SCORE_R    = (225,  60,  60)

# Business colours: Grocery=green, Pharmacy=blue, Restaurant=orange, Salon=pink
BIZ_COLS = {
    10: ( 80, 210,  90),   # Ikivunge (Grocery) — green
    11: ( 80, 150, 255),   # Pharmacy — blue
    12: (255, 140,  40),   # Resitora — orange
    13: (220,  80, 200),   # Salon — pink/purple
}
BIZ_NAMES = {
    10: "Grocery", 11: "Pharmacy",
    12: "Restaurant",  13: "Salon",
}
PHASE_TO_CELL = {0:10, 1:11, 2:12, 3:13}
BNAME_BY_PHASE = {0:"Grocery",1:"Pharmacy",2:"Restaurant",3:"Salon"}


# ══════════════════════════════════════════════════════════════════════════════
#  SPRITE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _is_road(grid,r,c):
    return 0<=r<GS and 0<=c<GS and grid[r,c]==1

def draw_road_tile(surf,px,py,grid,r,c):
    import pygame as pg
    N=_is_road(grid,r-1,c); S=_is_road(grid,r+1,c)
    E=_is_road(grid,r,c+1); W=_is_road(grid,r,c-1)
    connections = sum([N,S,E,W])
    # Narrow street (1-2 connections) = lighter, narrower asphalt
    # Main road (3-4 connections) = full dark asphalt
    road_col = ROAD_A if connections >= 2 else (75,78,88)
    pg.draw.rect(surf,road_col,(px,py,CELL,CELL))
    s = 4 if connections >= 3 else 6   # wider pavement on dead-end streets
    if not N: pg.draw.rect(surf,PAVEMENT,(px,py,CELL,s))
    if not S: pg.draw.rect(surf,PAVEMENT,(px,py+CELL-s,CELL,s))
    if not W: pg.draw.rect(surf,PAVEMENT,(px,py,s,CELL))
    if not E: pg.draw.rect(surf,PAVEMENT,(px+CELL-s,py,s,CELL))
    mx,my=px+CELL//2,py+CELL//2
    # Lane markings only on main roads (2+ connections)
    if connections >= 2:
        if N or S:
            for dy in range(4,CELL-4,10): pg.draw.rect(surf,ROAD_L,(mx-1,py+dy,2,5))
        if E or W:
            for dx in range(4,CELL-4,10): pg.draw.rect(surf,ROAD_L,(px+dx,my-1,5,2))
    if (N or S) and (E or W):
        pg.draw.rect(surf,(48,52,58),(mx-7,my-7,14,14))

def draw_residential(surf,px,py,shade=False):
    import pygame as pg
    c = GRASS_D if shade else GRASS_F
    pg.draw.rect(surf,c,(px,py,CELL,CELL))
    pg.draw.rect(surf,(155,195,130),(px+5,py+16,CELL-10,CELL-20))
    pg.draw.polygon(surf,(125,72,48),[(px+3,py+18),(px+CELL//2,py+5),(px+CELL-3,py+18)])
    pg.draw.rect(surf,(90,60,28),(px+CELL//2-4,py+CELL-13,8,11))
    pg.draw.rect(surf,(175,215,250),(px+7,py+21,9,7))
    pg.draw.rect(surf,(175,215,250),(px+CELL-16,py+21,9,7))

def draw_market(surf,px,py,fn):
    import pygame as pg
    pg.draw.rect(surf,GRASS_F,(px,py,CELL,CELL))
    pg.draw.rect(surf,(215,115,18),(px+2,py+9,CELL-4,CELL-13))
    pg.draw.rect(surf,(175,78,10),(px+2,py+9,CELL-4,8))
    for i in range(0,CELL-4,7):
        pg.draw.rect(surf,(235,155,38),(px+2+i,py+9,3,8))
    for i,col in enumerate([(230,50,50),(255,190,20),(50,185,50),(255,95,18)]):
        pg.draw.circle(surf,col,(px+7+i*8,py+28),4)
    if fn:
        pg.draw.rect(surf,(175,78,10),(px+4,py+2,CELL-8,9),border_radius=2)
        t=fn.render("MASOKO",True,(255,235,185)); surf.blit(t,(px+CELL//2-t.get_width()//2,py+2))

def draw_taxi(surf,px,py,fn):
    import pygame as pg
    pg.draw.rect(surf,TERRAIN,(px,py,CELL,CELL))
    pg.draw.rect(surf,(25,115,210),(px+3,py+11,CELL-6,CELL-18))
    pg.draw.rect(surf,(18,75,155),(px+2,py+8,CELL-4,6))
    pg.draw.rect(surf,(148,128,88),(px+7,py+27,CELL-14,4))
    pg.draw.rect(surf,(108,88,58),(px+8,py+31,5,7))
    pg.draw.rect(surf,(108,88,58),(px+CELL-13,py+31,5,7))
    pg.draw.rect(surf,(248,205,18),(px+8,py+36,26,11))
    pg.draw.rect(surf,(195,155,8),(px+12,py+32,18,7))
    pg.draw.circle(surf,(22,22,22),(px+13,py+47),3)
    pg.draw.circle(surf,(22,22,22),(px+31,py+47),3)
    if fn:
        pg.draw.rect(surf,(18,75,155),(px+4,py+2,CELL-8,9),border_radius=2)
        t=fn.render("TAXI",True,(250,225,28)); surf.blit(t,(px+CELL//2-t.get_width()//2,py+2))

def draw_hospital(surf,px,py,fn):
    import pygame as pg
    pg.draw.rect(surf,GRASS_F,(px,py,CELL,CELL))
    pg.draw.rect(surf,(225,228,235),(px+3,py+9,CELL-6,CELL-13))
    pg.draw.rect(surf,(198,208,228),(px+3,py+9,CELL-6,7))
    cx,cy=px+CELL//2,py+CELL//2-1
    pg.draw.rect(surf,(212,22,44),(cx-10,cy-4,20,8))
    pg.draw.rect(surf,(212,22,44),(cx-4,cy-10,8,20))
    pg.draw.rect(surf,(155,168,188),(cx-4,py+CELL-14,8,11))
    if fn:
        pg.draw.rect(surf,(198,208,228),(px+4,py+2,CELL-8,9),border_radius=2)
        t=fn.render("INDWI",True,(55,58,72)); surf.blit(t,(px+CELL//2-t.get_width()//2,py+2))

def draw_school(surf,px,py,fn):
    import pygame as pg
    pg.draw.rect(surf,GRASS_D,(px,py,CELL,CELL))
    pg.draw.rect(surf,(235,215,112),(px+3,py+11,CELL-6,CELL-15))
    pg.draw.rect(surf,(195,175,75),(px+3,py+11,CELL-6,7))
    pg.draw.rect(surf,(175,155,65),(px+CELL//2-4,py+4,8,9))
    pg.draw.circle(surf,(215,195,95),(px+CELL//2,py+5),4)
    for i in range(3): pg.draw.rect(surf,(175,215,250),(px+5+i*12,py+23,9,7))
    if fn:
        pg.draw.rect(surf,(195,175,75),(px+4,py+2,CELL-8,9),border_radius=2)
        t=fn.render("ISHURI",True,(75,55,18)); surf.blit(t,(px+CELL//2-t.get_width()//2,py+2))

def draw_church(surf,px,py,fn):
    import pygame as pg
    pg.draw.rect(surf,GRASS_D,(px,py,CELL,CELL))
    pg.draw.rect(surf,(235,228,208),(px+7,py+17,CELL-14,CELL-21))
    pg.draw.polygon(surf,(195,175,132),[(px+CELL//2,py+3),(px+CELL//2-7,py+19),(px+CELL//2+7,py+19)])
    cx=px+CELL//2
    pg.draw.line(surf,(172,45,45),(cx,py+4),(cx,py+17),2)
    pg.draw.line(surf,(172,45,45),(cx-4,py+9),(cx+4,py+9),2)
    pg.draw.rect(surf,(155,115,72),(cx-4,py+CELL-14,8,11))

def draw_industrial(surf,px,py,fn):
    import pygame as pg
    pg.draw.rect(surf,TERRAIN,(px,py,CELL,CELL))
    pg.draw.rect(surf,(95,100,112),(px+2,py+13,CELL-4,CELL-17))
    pg.draw.rect(surf,(78,82,92),(px+2,py+13,CELL-4,7))
    pg.draw.rect(surf,(72,76,84),(px+CELL-12,py+4,9,15))
    if fn:
        pg.draw.rect(surf,(78,82,92),(px+3,py+2,CELL-6,9),border_radius=2)
        t=fn.render("USINE",True,(178,182,198)); surf.blit(t,(px+CELL//2-t.get_width()//2,py+2))

def draw_business(surf,px,py,fn,cell_type,t_anim=0.0,is_rival=False,is_own=False):
    """
    Draw one of the 4 Rwandan business types.
    is_rival = True  → red warning tint + X on rival
    is_own   = True  → glow border (agent just placed this)
    """
    import pygame as pg
    col  = BIZ_COLS.get(cell_type,(150,200,150))
    wall = tuple(max(0,c-45) for c in col)
    name = BIZ_NAMES.get(cell_type,"BIZ")

    pg.draw.rect(surf,GRASS_F,(px,py,CELL,CELL))
    # Building body
    pg.draw.rect(surf,wall,(px+3,py+9,CELL-6,CELL-13))
    pg.draw.rect(surf,col, (px+3,py+9,CELL-6,7))
    # Shopfront
    pg.draw.rect(surf,(178,218,252),(px+5,py+23,CELL-10,12))
    pg.draw.line(surf,wall,(px+CELL//2,py+23),(px+CELL//2,py+35),1)
    # Door
    pg.draw.rect(surf,wall,(px+CELL//2-4,py+CELL-13,8,11))

    if is_own:
        # Pulsing OPEN sign
        pulse = 0.55+0.45*math.sin(t_anim*4)
        g = int(120+135*pulse)
        sc = tuple(max(0,int(c*pulse*0.8)) for c in col)
        pg.draw.rect(surf,sc,(px+4,py+11,CELL-8,7),border_radius=2)
        if fn:
            t=fn.render("OPEN",True,(215,252,205))
            surf.blit(t,(px+CELL//2-t.get_width()//2,py+12))
        # Glow border
        alpha=int(60+55*pulse)
        glow=pg.Surface((CELL+10,CELL+10),pg.SRCALPHA)
        pg.draw.rect(glow,(*col,alpha),(0,0,CELL+10,CELL+10),4,border_radius=6)
        surf.blit(glow,(px-5,py-5))
    else:
        # Name sign
        if fn:
            pg.draw.rect(surf,wall,(px+4,py+2,CELL-8,9),border_radius=2)
            t=fn.render(name[:6],True,col)
            surf.blit(t,(px+CELL//2-t.get_width()//2,py+2))

    if is_rival:
        # Red warning overlay + X
        s=pg.Surface((CELL,CELL),pg.SRCALPHA)
        s.fill((255,30,30,55)); surf.blit(s,(px,py))
        pg.draw.line(surf,(255,40,40),(px+4,py+9),(px+CELL-4,py+CELL-4),3)
        pg.draw.line(surf,(255,40,40),(px+CELL-4,py+9),(px+4,py+CELL-4),3)


# ── Pedestrians ────────────────────────────────────────────────────────────────
class Pedestrian:
    COLORS=[(252,195,148),(198,228,252),(252,238,158),(188,252,198),(218,188,252)]
    def __init__(self,x,y):
        self.x=float(x); self.y=float(y)
        self.vx=random.choice([-1,1])*random.uniform(0.22,0.78)
        self.vy=random.choice([-1,1])*random.uniform(0.22,0.78)
        self.col=random.choice(self.COLORS); self.life=random.randint(110,260)
    def update(self):
        self.x+=self.vx; self.y+=self.vy; self.life-=1
        if self.x<3 or self.x>GPX-3: self.vx*=-1
        if self.y<TOP+2 or self.y>TOP+GPX-2: self.vy*=-1
    def draw(self,surf):
        import pygame as pg
        pg.draw.circle(surf,self.col,(int(self.x),int(self.y)),3)
        pg.draw.circle(surf,(0,0,0),(int(self.x)+1,int(self.y)+2),2)


# ══════════════════════════════════════════════════════════════════════════════
class KigaliRenderer:
    def __init__(self,grid_size=15):
        import pygame as pg
        self.pg=pg; pg.init()
        pg.display.set_caption("Kigali Business Navigator v3")
        self.screen=pg.display.set_mode((WIN_W,WIN_H))
        self.clock=pg.time.Clock()
        self.gs=grid_size; self._t=0.0
        try:
            self.fn_lg=pg.font.SysFont("DejaVuSans",19,bold=True)
            self.fn_md=pg.font.SysFont("DejaVuSans",13,bold=True)
            self.fn_sm=pg.font.SysFont("DejaVuSans",11)
            self.fn_xs=pg.font.SysFont("DejaVuSans",10)
        except:
            self.fn_lg=pg.font.Font(None,23)
            self.fn_md=pg.font.Font(None,17)
            self.fn_sm=pg.font.Font(None,14)
            self.fn_xs=pg.font.Font(None,13)
        self._peds:List[Pedestrian]=[]
        self._ped_t=0
        self._road_pos:List[Tuple[int,int]]=[]
        self._last_grid_b=None

    def draw(self,grid,viability,sector_name,agent_pos,
             path=None,
             placed_positions=None,placed_types=None,
             phase=0,step=0,
             visited:Set=None,surveyed:Set=None,
             status="",episode=0,
             # legacy compat
             placed_pos=None,placed_type=None):
        self._t+=1.0/FPS
        pg=self.pg
        for event in pg.event.get():
            if event.type==pg.QUIT: self.close(); sys.exit()

        self.screen.fill(TERRAIN)
        self._refresh_roads(grid)
        self._draw_cells(grid,viability,agent_pos,
                         placed_positions or [],
                         placed_types or [],
                         phase,visited or set(),surveyed or set())
        self._tick_peds(); self._draw_peds()
        self._draw_path(path or [])
        self._draw_agent(agent_pos, bool(placed_positions and len(placed_positions)==4))
        self._draw_hud(sector_name,viability,agent_pos,grid,
                       placed_positions or [],placed_types or [],
                       phase,step,len(visited or []),status)
        self._draw_panel(viability,placed_positions or [],placed_types or [],
                         phase,grid)
        pg.display.flip()
        self.clock.tick(FPS)

    # ── Cells ──────────────────────────────────────────────────────────────────
    def _draw_cells(self,grid,viability,agent_pos,
                    placed_positions,placed_types,phase,visited,surveyed):
        phase_capped = min(phase, 3)
        rival_cell   = PHASE_TO_CELL[phase_capped]
        own_cells    = {pos: bt for pos,bt in zip(placed_positions,placed_types)}
        fn = self.fn_xs

        v_bt   = viability[:,:,phase_capped]
        v_min  = float(v_bt.min())
        v_rng  = max(float(v_bt.max())-v_min, 1e-6)

        for r in range(self.gs):
            for c in range(self.gs):
                ct   = int(grid[r,c])
                px2  = c*CELL
                py2  = r*CELL+TOP
                is_rival = (ct == rival_cell)
                is_own   = (r,c) in own_cells

                if   ct==1:  draw_road_tile(self.screen,px2,py2,grid,r,c)
                elif ct==2:  draw_market(self.screen,px2,py2,fn)
                elif ct==3:  draw_residential(self.screen,px2,py2,shade=(r+c)%2==0)
                elif ct==4:  draw_taxi(self.screen,px2,py2,fn)
                elif ct==5:  draw_hospital(self.screen,px2,py2,fn)
                elif ct==6:  draw_school(self.screen,px2,py2,fn)
                elif ct==7:  draw_church(self.screen,px2,py2,fn)
                elif ct==8:  draw_industrial(self.screen,px2,py2,fn)
                elif ct in BIZ_COLS:
                    draw_business(self.screen,px2,py2,fn,ct,
                                  self._t,is_rival=is_rival,is_own=is_own)
                else:
                    # Fallback: draw residential for any other cell
                    draw_residential(self.screen,px2,py2)

                # Viability shimmer (green glow proportional to viability)
                if ct not in (1,) and ct not in BIZ_COLS:
                    v_norm = (float(viability[r,c,phase_capped])-v_min)/v_rng
                    if v_norm > 0.55:
                        alpha = int((v_norm-0.55)*160)
                        s=self.pg.Surface((CELL,CELL),self.pg.SRCALPHA)
                        s.fill((50,210,50,alpha)); self.screen.blit(s,(px2,py2))

                # Surveyed cell — subtle tint
                if (r,c) in surveyed and ct not in (1,):
                    s=self.pg.Surface((CELL,CELL),self.pg.SRCALPHA)
                    s.fill((95,175,252,12)); self.screen.blit(s,(px2,py2))

                # Grid line
                self.pg.draw.rect(self.screen,(16,20,38),(px2,py2,CELL,CELL),1)

    # ── Pedestrians ────────────────────────────────────────────────────────────
    def _refresh_roads(self,grid):
        gb=grid.tobytes()
        if gb!=self._last_grid_b:
            self._last_grid_b=gb
            self._road_pos=[(c*CELL+CELL//2,r*CELL+CELL//2+TOP)
                             for r in range(self.gs) for c in range(self.gs)
                             if grid[r,c]==1]
    def _tick_peds(self):
        self._ped_t+=1
        if self._ped_t%4==0 and self._road_pos and len(self._peds)<100:
            x2,y2=random.choice(self._road_pos)
            self._peds.append(Pedestrian(x2+random.randint(-8,8),
                                          y2+random.randint(-8,8)))
        self._peds=[p for p in self._peds if p.life>0]
        for p in self._peds: p.update()
    def _draw_peds(self):
        for p in self._peds: p.draw(self.screen)

    # ── Path (no large surface per segment) ───────────────────────────────────
    def _draw_path(self,path):
        if len(path)<2: return
        pg=self.pg
        recent=path[max(0,len(path)-30):]
        for i in range(len(recent)-1):
            r1,c1=recent[i]; r2,c2=recent[i+1]
            x1=c1*CELL+CELL//2; y1=r1*CELL+CELL//2+TOP
            x2b=c2*CELL+CELL//2; y2b=r2*CELL+CELL//2+TOP
            b=int(60+(i/max(len(recent)-1,1))*160)
            pg.draw.line(self.screen,(b//3,b//2,b),(x1,y1),(x2b,y2b),2)

    # ── Agent ──────────────────────────────────────────────────────────────────
    def _draw_agent(self,agent_pos,done=False):
        if done: return
        pg=self.pg
        r,c=agent_pos
        ax=c*CELL+CELL//2; ay=r*CELL+CELL//2+TOP
        pg.draw.ellipse(self.screen,(0,0,0),(ax-7,ay+10,14,5))
        walk=math.sin(self._t*8)*3
        pg.draw.line(self.screen,(55,95,175),(ax,ay+4),(ax-5+walk,ay+14),3)
        pg.draw.line(self.screen,(55,95,175),(ax,ay+4),(ax+5-walk,ay+14),3)
        pg.draw.rect(self.screen,(75,135,215),(ax-5,ay-4,10,10),border_radius=2)
        pg.draw.circle(self.screen,(252,212,152),(ax,ay-8),7)
        pg.draw.rect(self.screen,(45,75,145),(ax+4,ay-3,5,8),border_radius=1)
        ring_r=int(13+3*math.sin(self._t*3))
        rs=pg.Surface((ring_r*2+2,ring_r*2+2),pg.SRCALPHA)
        pg.draw.circle(rs,(95,195,252,88),(ring_r+1,ring_r+1),ring_r,2)
        self.screen.blit(rs,(ax-ring_r-1,ay-ring_r-1))

    # ── HUD ────────────────────────────────────────────────────────────────────
    def _draw_hud(self,sector_name,viability,agent_pos,grid,
                  placed_positions,placed_types,phase,step,visited_count,status):
        pg=self.pg
        pg.draw.rect(self.screen,PANEL_BG,(0,0,GPX,TOP))
        pg.draw.line(self.screen,ACCENT,(0,TOP-1),(GPX,TOP-1),2)

        # Left — sector + current mission
        t1=self.fn_lg.render(f"  {sector_name}, Kigali",True,TW)
        self.screen.blit(t1,(10,5))
        phase_cap=min(phase,3)
        bname=BNAME_BY_PHASE[phase_cap]
        bcol=BIZ_COLS.get(PHASE_TO_CELL[phase_cap],(200,200,200))
        t2=self.fn_sm.render(f"Scouting: {bname}  (phase {phase_cap+1}/4)  "
                              f"step {step}/{400}",True,bcol)
        self.screen.blit(t2,(12,28))
        if status:
            t3=self.fn_xs.render(status,True,(148,168,208))
            self.screen.blit(t3,(12,45))

        # Right — viability at current cell
        r2,c2=agent_pos
        viab_here=float(viability[r2,c2,phase_cap])
        vt=viability[:,:,phase_cap]
        v_min=float(vt.min()); v_rng=max(float(vt.max())-v_min,1e-6)
        v_norm=(viab_here-v_min)/v_rng
        score=int(v_norm*100)
        scol=SCORE_G if score>=60 else(SCORE_Y if score>=35 else SCORE_R)
        verdict=("Great spot!" if score>=70 else "Decent" if score>=45 else "Avoid rivals")
        self.screen.blit(self.fn_lg.render(f"{bname}: {score}/100",True,scol),(GPX-195,5))
        self.screen.blit(self.fn_sm.render(verdict,True,scol),(GPX-195,28))

        # Placed progress dots
        dot_x=GPX-195; dot_y=45
        for i in range(4):
            cell=PHASE_TO_CELL[i]; c2=BIZ_COLS[cell]
            if i < len(placed_positions):
                pg.draw.circle(self.screen,c2,(dot_x+i*22,dot_y+7),7)
                pg.draw.circle(self.screen,TW,(dot_x+i*22,dot_y+7),7,1)
            else:
                pg.draw.circle(self.screen,(40,50,70),(dot_x+i*22,dot_y+7),7)
                pg.draw.circle(self.screen,(70,80,100),(dot_x+i*22,dot_y+7),7,1)

    # ── Side panel ─────────────────────────────────────────────────────────────
    def _draw_panel(self,viability,placed_positions,placed_types,phase,grid):
        pg=self.pg; px2=GPX
        pg.draw.rect(self.screen,PANEL_BG,(px2,0,PANEL,WIN_H))
        pg.draw.line(self.screen,(24,34,54),(px2,0),(px2,WIN_H),2)
        y=8
        self.screen.blit(self.fn_md.render("MAP KEY",True,ACCENT),(px2+10,y)); y+=20

        legend=[
            ((215,115,18), "Market",    "High foot traffic"),
            (( 25,125,215),"Taxi",        "Busiest area"),
            ((212,22,44),  "Hospital",    "Medical demand"),
            ((228,212,108),"School",      "Student demand"),
            ((230,225,205),"Church",    "Weekend crowds"),
            ((148,192,128),"Residential",   "Local customers"),
            (( 88,94,108), "Industrial", "Worker demand"),
            (( 55,58,65),  "Road",      "Accessibility"),
        ]
        for col,name,desc in legend:
            pg.draw.rect(self.screen,col,pg.Rect(px2+8,y,16,16),border_radius=2)
            pg.draw.rect(self.screen,(48,58,78),pg.Rect(px2+8,y,16,16),1,border_radius=2)
            self.screen.blit(self.fn_sm.render(name,True,TW),(px2+28,y+1))
            self.screen.blit(self.fn_xs.render(desc,True,TD),(px2+28,y+13))
            y+=29

        pg.draw.line(self.screen,(30,42,65),(px2+6,y+3),(px2+PANEL-6,y+3),1); y+=10

        # Business types
        self.screen.blit(self.fn_md.render("BUSINESSES",True,ACCENT),(px2+10,y)); y+=16
        phase_cap=min(phase,3)
        biz_legend=[
            (10,"Grocery",    "Provisions shop"),
            (11,"Pharmacy","Pharmacy/Chemist"),
            (12,"Restaurant",    "Restaurant/Stall"),
            (13,"Salon",       "Hair salon"),
        ]
        for ct,name,desc in biz_legend:
            col=BIZ_COLS[ct]; rival=(ct==PHASE_TO_CELL[phase_cap])
            pg.draw.rect(self.screen,col,pg.Rect(px2+8,y,16,16),border_radius=2)
            if rival:
                pg.draw.rect(self.screen,(255,40,40),pg.Rect(px2+8,y,16,16),2,border_radius=2)
                lbl=name+" ◄"
            else:
                pg.draw.rect(self.screen,(48,58,78),pg.Rect(px2+8,y,16,16),1,border_radius=2)
                lbl=name
            tcol=(255,150,150) if rival else TW
            self.screen.blit(self.fn_sm.render(lbl,True,tcol),(px2+28,y+1))
            self.screen.blit(self.fn_xs.render(desc,True,TD),(px2+28,y+13))
            y+=28

        pg.draw.line(self.screen,(30,42,65),(px2+6,y+3),(px2+PANEL-6,y+3),1); y+=10

        for note in ["◄ = current rival type",
                     "Green glow = good viability",
                     "Blue trail = agent path",
                     "Coloured dots = placed shops"]:
            self.screen.blit(self.fn_xs.render(note,True,TD),(px2+10,y)); y+=13

        # Best viability bar
        y+=5
        vbt=viability[:,:,phase_cap]
        vm=float(vbt.max()); vmi=float(vbt.min())
        bs=int(((vm-vmi)/max(vm-vmi,1e-6))*0)+int(min(vm,3.0)/3.0*100)
        self.screen.blit(self.fn_sm.render("Best spot:",True,TD),(px2+8,y)); y+=13
        bw=PANEL-20
        pg.draw.rect(self.screen,(25,35,55),(px2+8,y,bw,10),border_radius=4)
        fc=SCORE_G if bs>=60 else(SCORE_Y if bs>=35 else SCORE_R)
        pg.draw.rect(self.screen,fc,(px2+8,y,int(bw*bs/100),10),border_radius=4)
        bt2=self.fn_xs.render(f"{bs}/100",True,TW)
        self.screen.blit(bt2,(px2+8+bw//2-bt2.get_width()//2,y+1))

    def close(self): self.pg.quit()


# ── Standalone demo ────────────────────────────────────────────────────────────
def run_random_demo(n_episodes=3, sector_id=None, step_delay=0.70):
    """
    Random agent demo. No model — just shows the environment.
    Action space: 0-3 move, 4 survey, 5 place.
    """
    sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from environment.custom_env import KigaliRetailEnv, BUSINESS_NAMES
    except ImportError:
        from custom_env import KigaliRetailEnv, BUSINESS_NAMES

    import pygame

    ACT_MAP = {0:"Move ↑",1:"Move ↓",2:"Move ←",3:"Move →",
               4:"Survey",5:"PLACE"}

    env     = KigaliRetailEnv(sector_id=sector_id, difficulty=0.4)
    renderer= KigaliRenderer()
    info    = {"sector":"Kigali"}
    total_r = 0.0

    for ep in range(n_episodes):
        obs,info = env.reset(seed=ep*7)
        ep_r=0.0; done=False; si=0
        print(f"\n[Ep {ep+1}] Sector:{info['sector']}  "
              f"Rivals/type:{info['n_rivals']}")

        while not done:
            for event in pygame.event.get():
                if event.type==pygame.QUIT:
                    renderer.close(); env.close(); return

            phase=min(env._phase,3)
            bname=BUSINESS_NAMES[phase]

            # Random agent: mostly explore, sometimes survey, sometimes place
            roll=np.random.random()
            if roll < 0.50:   action=np.random.randint(0,4)  # move
            elif roll < 0.72: action=4                         # survey
            else:              action=5                         # place

            obs,reward,terminated,truncated,step_info=env.step(action)
            ep_r+=reward; si+=1
            status=f"[{bname}] {ACT_MAP.get(action,'?')}  r={reward:+.2f}"

            renderer.draw(
                grid=env.grid,
                viability=env.viability,
                sector_name=step_info["sector"],
                agent_pos=env._pos,
                path=env._path,
                placed_positions=env._placed_positions,
                placed_types=env._placed_types,
                phase=env._phase,
                step=step_info["step"],
                visited=env._visited,
                surveyed=env._surveyed,
                status=status,
                episode=ep+1,
            )
            print(f"  s={si:3d}  [{bname:12s}]  "
                  f"{ACT_MAP.get(action,'?'):8s}  r={reward:+.3f}")
            time.sleep(step_delay)
            done=terminated or truncated

            if done:
                for pos,bt in zip(env._placed_positions,env._placed_types):
                    r2,c2=pos; v=float(env.viability[r2,c2,bt])
                    print(f"  ✓ {BUSINESS_NAMES[bt]:14s} at {pos}  "
                          f"viab={v:.3f}")
                # Hold result frame for 2s
                for _ in range(FPS*2):
                    renderer.draw(
                        grid=env.grid,viability=env.viability,
                        sector_name=step_info["sector"],
                        agent_pos=env._pos,path=env._path,
                        placed_positions=env._placed_positions,
                        placed_types=env._placed_types,
                        phase=env._phase,step=step_info["step"],
                        visited=env._visited,surveyed=env._surveyed,
                        status=f"Episode {ep+1} done! "
                               f"{len(env._placed_positions)}/4 placed.",
                        episode=ep+1,
                    )
                    renderer.clock.tick(FPS)

        total_r+=ep_r
        print(f"  Episode reward: {ep_r:.2f}")

    print(f"\nDemo done.  Mean: {total_r/n_episodes:.2f}")
    print("Close window to exit.")
    running=True
    while running:
        for event in pygame.event.get():
            if event.type==pygame.QUIT: running=False
        renderer.draw(
            grid=env.grid,viability=env.viability,
            sector_name=info["sector"],agent_pos=env._pos,
            path=env._path,
            placed_positions=env._placed_positions,
            placed_types=env._placed_types,
            phase=env._phase,step=0,
            visited=env._visited,surveyed=env._surveyed,
            status="Demo complete — close window to exit",
        )
    renderer.close()
    env.close()


if __name__=="__main__":
    run_random_demo(n_episodes=3, step_delay=0.70)