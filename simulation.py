"""
simulation.py
=============
Core ABM engine untuk Dashboard Streamlit Burnout Mahasiswa.
Diekstrak dari burnout_abm_v3.py (Minggu 12) dengan optimasi untuk
penggunaan interaktif (caching-friendly, fast MC runner).
"""

import random
import math
import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ── Konstanta ─────────────────────────────────────────────────────
T_AT_RISK = 0.30
T_BURNOUT = 0.60
REC_MIN   = 0.04
REC_MAX   = 0.12
EXAM_PERIODS = list(range(38, 48)) + list(range(88, 98))

STATE_COLORS = {
    "Normal"  : "#43A047",
    "At-Risk" : "#FB8C00",
    "Burnout" : "#E53935",
    "Recovery": "#1E88E5",
}

# ── Dataclasses ───────────────────────────────────────────────────
@dataclass
class SimConfig:
    """Konfigurasi lengkap satu run simulasi."""
    num_agents   : int   = 80
    steps        : int   = 100
    # Intervensi
    ps_level     : float = 0.0
    sc_level     : float = 0.0
    cbt_program  : float = 0.0
    ps_reactive  : bool  = False
    # Jaringan
    network_type : str   = "small_world"   # "small_world" | "scale_free"
    network_k    : int   = 6
    network_p    : float = 0.10
    network_m    : int   = 3
    # Parameter model
    gamma        : float = 0.15
    lambda_base  : float = 0.10
    lambda_peak  : float = 0.50
    alpha        : float = 0.50
    beta         : float = 0.30
    delta        : float = 0.20
    seed         : int   = 42


@dataclass
class StepRecord:
    step          : int
    avg_burnout   : float
    std_burnout   : float
    avg_resilience: float
    n_normal      : int
    n_atrisk      : int
    n_burnout     : int
    n_recovery    : int
    stressor      : float


# ── Agent ─────────────────────────────────────────────────────────
class StudentAgent:
    __slots__ = (
        "uid","burnout","resilience","vulnerability","social_suscept",
        "peer_capacity","selfcare_habit","cbt_skill","help_seeking",
        "support_fatigue","recovery_count","burnout_duration",
        "state","_snapshot","hist_state",
    )

    def __init__(self, uid, burnout, resilience, vulnerability,
                 social_suscept, peer_capacity, selfcare_habit,
                 cbt_skill, help_seeking):
        self.uid             = uid
        self.burnout         = float(burnout)
        self.resilience      = float(resilience)
        self.vulnerability   = float(vulnerability)
        self.social_suscept  = float(social_suscept)
        self.peer_capacity   = float(peer_capacity)
        self.selfcare_habit  = float(selfcare_habit)
        self.cbt_skill       = float(cbt_skill)
        self.help_seeking    = float(help_seeking)
        self.support_fatigue = 0.0
        self.recovery_count  = 0
        self.burnout_duration= 0
        self._snapshot       = float(burnout)
        self.state           = self._calc_state()
        self.hist_state      = [self.state]

    def _calc_state(self):
        if self.burnout < T_AT_RISK:   return "Normal"
        elif self.burnout < T_BURNOUT: return "At-Risk"
        else:                          return "Burnout"

    def step(self, model):
        self._snapshot = self.burnout
        neighbors = model.neighbors[self.uid]

        # Contagion
        mean_nb = float(np.mean([model.agents[j]._snapshot for j in neighbors])) if neighbors else 0.0
        contagion = model.gamma * self.social_suscept * mean_nb

        # Peer support
        ps_received = 0.0
        if model.ps_level > 0 and neighbors:
            cnt = 0
            for j in neighbors:
                ag_j = model.agents[j]
                if ag_j.state in ("Normal", "Recovery"):
                    ff = max(0.1, 1.0 - ag_j.support_fatigue)
                    ps_received += ag_j.peer_capacity * model.ps_level * ff
                    ag_j.support_fatigue = min(1.0, ag_j.support_fatigue + 0.01)
                    cnt += 1
            ps_received /= len(neighbors)
            if model.ps_reactive:
                if not any(model.agents[j].state == "Burnout" for j in neighbors):
                    ps_received = 0.0

        sc_effect  = self.selfcare_habit * model.sc_level
        cbt_effect = self.cbt_skill      * model.cbt_program

        # Resilience growth
        self.resilience = min(1.0, self.resilience + 0.002 * (sc_effect + cbt_effect))

        # dB equation
        dB = (model.S * self.vulnerability
              - self.resilience * (model.alpha * ps_received
                                  + model.beta  * sc_effect
                                  + model.delta * cbt_effect)
              + contagion)

        if self.state == "Recovery":
            dB -= random.uniform(REC_MIN, REC_MAX)
        if self.state == "Burnout":
            self.burnout_duration += 1

        self.burnout = max(0.0, min(1.0, self.burnout + dB))

        # State transition
        intervention_active = (model.ps_level > 0 or model.sc_level > 0 or model.cbt_program > 0)
        if self.state == "Burnout":
            if intervention_active and random.random() > 0.5 - self.help_seeking * 0.3:
                self.state = "Recovery"
                self.recovery_count += 1
        elif self.state == "Recovery":
            if   self.burnout < T_AT_RISK:  self.state = "Normal"
            elif self.burnout >= T_BURNOUT: self.state = "Burnout"
        else:
            self.state = self._calc_state()

        self.hist_state.append(self.state)


# ── Model ─────────────────────────────────────────────────────────
class BurnoutModel:
    def __init__(self, cfg: SimConfig):
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.cfg     = cfg
        self.ps_level   = cfg.ps_level
        self.sc_level   = cfg.sc_level
        self.cbt_program= cfg.cbt_program
        self.ps_reactive= cfg.ps_reactive
        self.gamma      = cfg.gamma
        self.alpha      = cfg.alpha
        self.beta       = cfg.beta
        self.delta      = cfg.delta
        self.S          = 0.0
        self.current_step = 0

        # Network
        if cfg.network_type == "scale_free":
            G = nx.barabasi_albert_graph(cfg.num_agents, cfg.network_m, seed=cfg.seed)
        else:
            G = nx.watts_strogatz_graph(cfg.num_agents, cfg.network_k, cfg.network_p, seed=cfg.seed)
        self.neighbors: Dict[int, List[int]] = {n: list(G.neighbors(n)) for n in G.nodes()}
        self.G = G

        # Agents
        self.agents: Dict[int, StudentAgent] = {}
        for uid in range(cfg.num_agents):
            self.agents[uid] = StudentAgent(
                uid            = uid,
                burnout        = random.uniform(0.05, 0.40),
                resilience     = random.uniform(0.25, 0.75),
                vulnerability  = random.uniform(0.60, 1.50),
                social_suscept = random.uniform(0.10, 0.75),
                peer_capacity  = random.uniform(0.15, 0.85),
                selfcare_habit = random.uniform(0.10, 0.65),
                cbt_skill      = random.uniform(0.10, 0.80),
                help_seeking   = random.uniform(0.20, 0.90),
            )

        self.records: List[StepRecord] = [self._snap()]

    def _snap(self) -> StepRecord:
        bs = [a.burnout    for a in self.agents.values()]
        rs = [a.resilience for a in self.agents.values()]
        ss = [a.state      for a in self.agents.values()]
        return StepRecord(
            step=self.current_step,
            avg_burnout=float(np.mean(bs)),
            std_burnout=float(np.std(bs)),
            avg_resilience=float(np.mean(rs)),
            n_normal   =ss.count("Normal"),
            n_atrisk   =ss.count("At-Risk"),
            n_burnout  =ss.count("Burnout"),
            n_recovery =ss.count("Recovery"),
            stressor   =self.S,
        )

    def _calc_stressor(self):
        lam = self.cfg.lambda_base + (0.05 if self.current_step % 5 == 0 else 0.0)
        if self.current_step in EXAM_PERIODS:
            lam += self.cfg.lambda_peak
        return min(1.0, np.random.poisson(lam) / 3.0)

    def step(self):
        self.current_step += 1
        self.S = self._calc_stressor()
        order = list(self.agents.keys())
        random.shuffle(order)
        for uid in order:
            self.agents[uid].step(self)
        self.records.append(self._snap())

    def run_all(self):
        for _ in range(self.cfg.steps):
            self.step()

    # Helpers
    def get_final_burnouts(self) -> List[float]:
        return [a.burnout for a in self.agents.values()]

    def get_agent_states(self) -> List[str]:
        return [a.state for a in self.agents.values()]

    def get_recovery_times(self) -> List[int]:
        times = []
        for a in self.agents.values():
            in_b = None
            for i, s in enumerate(a.hist_state):
                if s == "Burnout" and in_b is None: in_b = i
                elif s in ("Normal", "At-Risk") and in_b is not None:
                    times.append(i - in_b); in_b = None
        return times

    def get_records_df(self):
        import pandas as pd
        return pd.DataFrame([vars(r) for r in self.records]).set_index("step")


# ── Monte Carlo Runner ────────────────────────────────────────────
def run_monte_carlo(cfg: SimConfig, n_runs: int = 50):
    """
    Jalankan n_runs iterasi MC. Return dict arrays untuk plotting.
    Tiap run menggunakan seed = run_index (bukan cfg.seed).
    """
    steps = cfg.steps
    ts_ab = np.zeros((n_runs, steps + 1))
    ts_pb = np.zeros((n_runs, steps + 1))
    ts_pn = np.zeros((n_runs, steps + 1))
    ts_ar = np.zeros((n_runs, steps + 1))
    all_fb: List[float] = []
    all_rt: List[int]   = []

    for ri in range(n_runs):
        run_cfg = SimConfig(**{**vars(cfg), "seed": ri})
        model = BurnoutModel(run_cfg)
        model.run_all()
        df = model.get_records_df()
        ts_ab[ri] = df["avg_burnout"].values
        ts_pb[ri] = df["n_burnout"].values / cfg.num_agents * 100
        ts_pn[ri] = df["n_normal"].values  / cfg.num_agents * 100
        ts_ar[ri] = df["avg_resilience"].values
        all_fb.extend(model.get_final_burnouts())
        all_rt.extend(model.get_recovery_times())

    def ci95(arr):
        m  = arr.mean(0)
        se = arr.std(0, ddof=1) / math.sqrt(n_runs)
        return m, m - 1.96 * se, m + 1.96 * se

    return {
        "n_runs" : n_runs,
        "steps"  : steps,
        "ab"     : ci95(ts_ab),   # (mean, lo, hi)
        "pb"     : ci95(ts_pb),
        "pn"     : ci95(ts_pn),
        "ar"     : ci95(ts_ar),
        "final_b": np.array(all_fb),
        "rec_t"  : all_rt,
        "ts_ab"  : ts_ab,
        "ts_pb"  : ts_pb,
    }
