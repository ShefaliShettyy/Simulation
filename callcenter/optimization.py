from __future__ import annotations

import copy
import functools
import logging
import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ortools.sat.python import cp_model

try:
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

from core_simulation import RouterScoreWeights, SimulationConfig, SimulationEngine

logger = logging.getLogger(__name__)

_COST_SCALE: int = 100
HORIZON_MINUTES: int = 720

SHIFT_WINDOWS: Tuple[Tuple[str, int, int], ...] = (
    ("M", 0,   480),
    ("D", 120, 600),
    ("E", 240, 720),
)

COVERAGE_BANDS: Tuple[Tuple[int, int, Tuple[str, ...]], ...] = (
    (  0, 120, ("M",)),
    (120, 240, ("M", "D")),
    (240, 480, ("M", "D", "E")),
    (480, 600, ("D", "E")),
    (600, 720, ("E",)),
)

PEAK_BAND_INDEX: int = 2

_SHIFT_BREAK_MIDPOINTS: Dict[str, float] = {"M": 240.0, "D": 360.0, "E": 480.0}
_BREAK_DURATION_MINUTES: float = 30.0


def _shifts_for_band(band_idx: int) -> Tuple[str, ...]:
    return COVERAGE_BANDS[band_idx][2]


def _band_duration(band_idx: int) -> int:
    start, end, _ = COVERAGE_BANDS[band_idx]
    return end - start


def _arrival_fraction_for_band(band_idx: int) -> float:
    return _band_duration(band_idx) / HORIZON_MINUTES


def _band_index_for_minute(minute: float) -> int:
    """Coverage band whose [start,end) interval contains `minute`."""
    for b, (b_start, b_end, _) in enumerate(COVERAGE_BANDS):
        if b_start <= minute < b_end:
            return b
    return len(COVERAGE_BANDS) - 1


# ===========================================================================
# Labour-constraint configuration  (the expanded constraint set)
# ===========================================================================

@dataclass
class ShrinkageConfig:
    """Multi-factor shrinkage. Planned and unplanned categories combine
    multiplicatively (independent availability losses):

        total = 1 - (1 - planned) * (1 - unplanned)

    `total()` feeds the Erlang gross-up  required = net / (1 - total)."""
    # planned
    training: float = 0.0
    coaching: float = 0.0
    meetings: float = 0.0
    admin:    float = 0.0
    # unplanned
    absence:     float = 0.0
    sickness:    float = 0.0
    no_show:     float = 0.0
    tech_outage: float = 0.0

    def planned(self) -> float:
        return min(0.9, self.training + self.coaching + self.meetings + self.admin)

    def unplanned(self) -> float:
        return min(0.9, self.absence + self.sickness + self.no_show + self.tech_outage)

    def total(self) -> float:
        return min(0.9, 1.0 - (1.0 - self.planned()) * (1.0 - self.unplanned()))

    def breakdown(self) -> Dict[str, float]:
        return {"planned": self.planned(), "unplanned": self.unplanned(), "total": self.total()}


@dataclass
class LabourConstraintConfig:
    """Call-centre labour/business constraints layered onto the base model.

    Every field is a no-op at its default, so an empty config reproduces the
    original (Erlang-floor-only) behaviour. Switch constraints on individually.
    """
    # Shrinkage gross-up (breaks, training, absence). Applied to Erlang floors:
    # required = net / (1 - shrinkage).  THE canonical WFM constraint.
    shrinkage:             float = 0.0
    # Multi-factor shrinkage. If set, its total() OVERRIDES the scalar `shrinkage`.
    shrinkage_config:      Optional["ShrinkageConfig"] = None

    # Keep an extra buffer of agents in the bands where staff take breaks so the
    # SLA floor still holds while people are away.
    break_band_indices:    Tuple[int, ...] = ()
    break_buffer_agents:   Dict[str, int]  = field(default_factory=dict)

    # Hard £ ceiling on total wage-weighted staffing spend across all shifts.
    max_total_budget:      Optional[float] = None

    # Roster stability: cap the head-count swing between adjacent shifts.
    max_shift_swing:       Optional[int]   = None

    # Cross-training credit: a fraction of a cross-trained skill's agents counts
    # toward an adjacent skill's floor (mirrors the sim's overflow routing).
    cross_skill_pairs:     Tuple[Tuple[str, str], ...] = ()
    cross_skill_fraction:  float = 0.0
    cross_skill_cap:       int   = 2

    # Raise the peak-band floor in proportion to VIP load.
    vip_uplift_scale:      float = 0.0

    # Contracted-FTE ceiling per skill across all shifts.
    max_fte_per_skill:     Dict[str, int] = field(default_factory=dict)

    # Workload fairness: cap coverage spread across skills within a band.
    max_band_spread:       Optional[int] = None

    # Over-staffing cap: cover <= ceil(floor * (1 + ratio)) per band/skill.
    max_overstaff_ratio:   Optional[float] = None

    # Absolute minimum coverage per band/skill, regardless of Erlang.
    min_coverage_floor:    Optional[int] = None

    # ------------------------------------------------------------------
    # Break-scheduling sub-model (item 4 + 5).  All no-ops unless enabled.
    # ------------------------------------------------------------------
    enable_break_optimization: bool = False
    break_slot_minutes:        int   = 60      # granularity of break slots
    breaks_per_agent:          int   = 1       # break-slots each agent must take
    max_break_fraction_per_slot: float = 0.34  # cap on simultaneous breaks / slot
    no_break_head_minutes:     float = 60.0    # min continuous work before a break
    no_break_tail_minutes:     float = 60.0    # no break this close to shift end
    lunch_window:              Optional[Tuple[int, int]] = None   # (start,end) abs minutes
    break_balance_weight:      float = 0.0     # objective: penalise break overlap / dips

    def enabled(self) -> bool:
        return any([
            self.break_band_indices and self.break_buffer_agents,
            self.max_total_budget is not None,
            self.max_shift_swing is not None,
            self.cross_skill_pairs and self.cross_skill_fraction > 0,
            self.vip_uplift_scale > 0,
            bool(self.max_fte_per_skill),
            self.max_band_spread is not None,
            self.max_overstaff_ratio is not None,
            self.min_coverage_floor is not None,
            self.enable_break_optimization,
        ])


# ===========================================================================
# OptimizationConfig
# ===========================================================================

@dataclass
class OptimizationConfig:
    sla_target:                     float           = 0.97
    sla_threshold_minutes:          float           = 1.0
    max_agents_per_skill:           int             = 20
    min_agents_per_skill:           int             = 1
    analytical_safety_margin:       Optional[float] = 0.04
    max_iterations:                 int             = 8
    convergence_tolerance:          float           = 0.015
    engine_type:                    str             = "cost"
    random_seed:                    Optional[int]   = 42
    verbose:                        bool            = True
    sla_violation_penalty_per_call: float           = 0.0
    cost_weight:                    float           = 1.0
    sla_weight:                     float           = 0.0
    pareto_sweep:                   bool            = False
    max_total_agents:               Optional[int]   = None
    max_occupancy:                  float           = 0.85
    arrival_rate_buffer:            float           = 1.10
    debug_solver:                   bool            = False
    sla_predictor:                  Optional[Callable] = None
    sim_feedback_penalty_scale:     float           = 0.0
    skill_realism_derating:         Dict[str, float] = field(default_factory=dict)
    realism_floor_agents:           Dict[str, int]   = field(default_factory=dict)
    cpsat_time_limit_seconds:       float           = 30.0
    per_shift_optimisation:         bool            = True

    # --- restructure additions -----------------------------------------
    labour:           LabourConstraintConfig = field(default_factory=LabourConstraintConfig)
    overstaff_weight: float = 0.0     # secondary objective: penalise idle over-coverage


# ===========================================================================
# ErlangC  (unchanged)
# ===========================================================================

class ErlangC:
    @staticmethod
    @functools.lru_cache(maxsize=4096)
    def erlang_c_probability(c: int, a: float) -> float:
        if c <= 0:
            return 1.0
        rho = a / c
        if rho >= 1.0:
            return 1.0
        log_a        = math.log(a) if a > 0 else float("-inf")
        log_num_term = c * log_a - math.lgamma(c + 1)
        num_term     = math.exp(log_num_term) / (1.0 - rho)
        poisson_sum  = 0.0
        log_ak_kfact = 0.0
        for k in range(c):
            if k > 0:
                log_ak_kfact += log_a - math.log(k)
            poisson_sum += math.exp(log_ak_kfact)
        denominator = poisson_sum + num_term
        return num_term / denominator if denominator > 0 else 1.0

    @staticmethod
    @functools.lru_cache(maxsize=4096)
    def sla_probability(c: int, arrival_rate_per_min: float,
                        mean_service_min: float, sla_threshold_min: float) -> float:
        if c <= 0 or mean_service_min <= 0:
            return 0.0
        mu = 1.0 / mean_service_min
        a  = arrival_rate_per_min / mu
        if a <= 0:
            return 1.0
        C   = ErlangC.erlang_c_probability(c, round(a, 6))
        rho = a / c
        if rho >= 1.0:
            return 0.0
        exponent = -(c - a) * mu * sla_threshold_min
        sla      = 1.0 - C * math.exp(exponent)
        return float(max(0.0, min(1.0, sla)))

    @staticmethod
    def min_agents_for_sla(arrival_rate_per_min: float, mean_service_min: float,
                           sla_threshold_min: float, sla_target: float,
                           max_c: int = 50,
                           sla_predictor: Optional[Callable] = None) -> int:
        mu    = 1.0 / mean_service_min if mean_service_min > 0 else 1.0
        a     = arrival_rate_per_min / mu
        c_min = max(1, math.ceil(a) + 1)
        lam_r = round(arrival_rate_per_min, 6)
        svc_r = round(mean_service_min, 6)
        thr_r = round(sla_threshold_min, 6)
        for c in range(c_min, max_c + 1):
            try:
                sla = (float(sla_predictor(c, lam_r, svc_r, thr_r))
                       if sla_predictor is not None
                       else ErlangC.sla_probability(c, lam_r, svc_r, thr_r))
            except Exception:
                sla = ErlangC.sla_probability(c, lam_r, svc_r, thr_r)
            if sla >= sla_target:
                return c
        return max_c


# ===========================================================================
# Result containers
# ===========================================================================

@dataclass
class OptimizationResult:
    agents_per_skill:    Dict[str, int]
    shift_plan:          Dict[str, Dict[str, int]]
    band_coverage:       Dict[int, Dict[str, int]]
    analytical_sla:      Dict[str, float]
    analytical_target:   float
    total_staffing_cost: float
    status:              str
    solve_time_seconds:  float = 0.0
    constraints_applied: List[str] = field(default_factory=list)   # restructure: provenance
    # break sub-model outputs (empty unless break optimization was enabled)
    break_plan:          Dict[Tuple[str, str, int], int] = field(default_factory=dict)
    break_slot_minutes:  int = 0

    def to_sim_config(self, base: SimulationConfig) -> SimulationConfig:
        cfg = copy.copy(base)
        cfg.agents_per_skill     = dict(self.agents_per_skill)
        cfg.sim_duration_minutes = float(HORIZON_MINUTES)
        return cfg

    def __repr__(self) -> str:
        lines = [f"OptimizationResult(status={self.status}, "
                 f"cost=£{self.total_staffing_cost:,.2f}, horizon={HORIZON_MINUTES}min)"]
        for skill, n in self.agents_per_skill.items():
            sla = self.analytical_sla.get(skill, 0.0)
            lines.append(f"  {skill}: {n} agents (peak)  Erlang-C SLA={sla:.1%}")
        if self.constraints_applied:
            lines.append(f"  constraints: {', '.join(self.constraints_applied)}")
        lines.append("  Shift plan:")
        for sh, plan in self.shift_plan.items():
            lines.append(f"    Shift {sh}: {plan}")
        return "\n".join(lines)


@dataclass
class EvaluationResult:
    agents_per_skill:  Dict[str, int]
    shift_plan:        Dict[str, Dict[str, int]]
    sla:               float
    abandonment_rate:  float
    avg_csat:          float
    asa:               float
    aht:               float
    total_calls:       int
    total_cost:        float = 0.0
    cost_breakdown:    Dict[str, float] = field(default_factory=dict)
    sim_time_seconds:  float = 0.0

    def meets_sla(self, target: float, tolerance: float = 0.0) -> bool:
        return self.sla >= target - tolerance

    def __repr__(self) -> str:
        return (f"EvaluationResult(SLA={self.sla:.1%}, abandon={self.abandonment_rate:.1%}, "
                f"CSAT={self.avg_csat:.2f}, cost=£{self.total_cost:,.2f}, "
                f"horizon={HORIZON_MINUTES}min)")


# ===========================================================================
# Erlang floor builder  (now applies shrinkage gross-up)
# ===========================================================================

def _make_derated_predictor(skill: str, derating: Dict[str, float]) -> Callable:
    factor = float(derating.get(skill, 1.0))
    def predictor(c: int, lam: float, svc: float, thr: float) -> float:
        return ErlangC.sla_probability(c, lam, svc, thr) * factor
    return predictor


def _build_band_erlang_mins(sim_cfg: SimulationConfig, opt_cfg: OptimizationConfig,
                            analytical_target: float, lam_buffered: float,
                            mean_svc: float) -> Dict[Tuple[int, str], int]:
    skills    = list(sim_cfg.agents_per_skill.keys())
    skill_mix = sim_cfg.skill_mix
    total_mix = sum(skill_mix.values()) or 1.0
    n         = len(skills)
    _sc       = opt_cfg.labour.shrinkage_config
    shrink    = max(0.0, min(0.9, _sc.total() if _sc is not None else opt_cfg.labour.shrinkage))
    result: Dict[Tuple[int, str], int] = {}

    for band_idx, (b_start, b_end, _) in enumerate(COVERAGE_BANDS):
        band_frac = (b_end - b_start) / HORIZON_MINUTES
        band_lam  = lam_buffered * band_frac
        for skill in skills:
            frac      = skill_mix.get(skill, 1.0 / n) / total_mix
            skill_lam = band_lam * frac

            derating = opt_cfg.skill_realism_derating or {}
            skill_pred = (_make_derated_predictor(skill, derating)
                          if skill in derating and derating[skill] != 1.0
                          else opt_cfg.sla_predictor)

            min_c = ErlangC.min_agents_for_sla(
                arrival_rate_per_min=round(skill_lam, 6),
                mean_service_min    =round(mean_svc, 6),
                sla_threshold_min   =opt_cfg.sla_threshold_minutes,
                sla_target          =analytical_target,
                max_c               =opt_cfg.max_agents_per_skill,
                sla_predictor       =skill_pred,
            )

            mu      = 1.0 / mean_svc if mean_svc > 0 else 1.0
            offered = skill_lam / mu
            max_occ = min(0.99, float(opt_cfg.max_occupancy))
            if max_occ > 0 and offered > 0:
                min_c = max(min_c, math.ceil(offered / max_occ))

            floor = int((opt_cfg.realism_floor_agents or {}).get(skill, 0))
            min_c = min_c + floor

            # WFM shrinkage gross-up: staff enough to absorb breaks/absence.
            if shrink > 0:
                min_c = math.ceil(min_c / (1.0 - shrink))

            min_c = min(min_c, opt_cfg.max_agents_per_skill)
            min_c = max(min_c, opt_cfg.min_agents_per_skill)
            result[(band_idx, skill)] = min_c
    return result


def _band_coverage_from_shift_plan(shift_plan: Dict[str, Dict[str, int]],
                                   skills: List[str]) -> Dict[int, Dict[str, int]]:
    band_coverage: Dict[int, Dict[str, int]] = {}
    for band_idx, (_, _, active_shifts) in enumerate(COVERAGE_BANDS):
        band_coverage[band_idx] = {
            skill: sum(shift_plan.get(sh, {}).get(skill, 0) for sh in active_shifts)
            for skill in skills
        }
    return band_coverage


# ===========================================================================
# Composable CP-SAT model builder  (the restructure)
# ===========================================================================

class _StaffingModel:
    """Builds the CP-SAT staffing model one concern at a time.

    Pipeline (call in order):
        add_decision_vars -> add_coverage_links -> add_base_floors
        -> add_total_agent_cap -> add_labour_constraints -> set_objective
    """

    def __init__(self, skills: List[str], shifts: List[str], opt_cfg: OptimizationConfig,
                 band_erlang_min: Dict[Tuple[int, str], int], cost_per_agent: Dict[str, float],
                 skill_mix: Dict[str, float], vip_fraction: float) -> None:
        self.skills   = skills
        self.shifts   = shifts
        self.opt      = opt_cfg
        self.floors   = band_erlang_min
        self.cost     = cost_per_agent
        self.skill_mix = skill_mix
        self.vip_frac = vip_fraction
        self.model    = cp_model.CpModel()
        self.agents:  Dict[str, Dict[str, cp_model.IntVar]] = {}
        self.cover:   Dict[int, Dict[str, cp_model.IntVar]] = {}
        self.applied: List[str] = []
        self.brk:     Dict = {}
        self.break_peak = None
        self._break_L = 0
        self._break_n_slots = 0

    # -- structural -------------------------------------------------------

    def add_decision_vars(self) -> "_StaffingModel":
        cap = self.opt.max_agents_per_skill
        self.agents = {
            sh: {sk: self.model.NewIntVar(0, cap, f"n_{sh}_{sk}") for sk in self.skills}
            for sh in self.shifts
        }
        for sh in self.shifts:
            for sk in self.skills:
                self.model.Add(self.agents[sh][sk] >= self.opt.min_agents_per_skill)
        return self

    def add_coverage_links(self) -> "_StaffingModel":
        cap = self.opt.max_agents_per_skill * len(self.shifts)
        for b, (_, _, active) in enumerate(COVERAGE_BANDS):
            self.cover[b] = {}
            for sk in self.skills:
                cv = self.model.NewIntVar(0, cap, f"cover_b{b}_{sk}")
                self.model.Add(cv == sum(self.agents[sh][sk] for sh in active))
                self.cover[b][sk] = cv
        return self

    def add_base_floors(self) -> "_StaffingModel":
        for b in range(len(COVERAGE_BANDS)):
            for sk in self.skills:
                self.model.Add(self.cover[b][sk] >= self.floors[(b, sk)])
        # explicit, redundant peak-band floor (clarity / refactor safety)
        for sk in self.skills:
            self.model.Add(self.cover[PEAK_BAND_INDEX][sk] >= self.floors[(PEAK_BAND_INDEX, sk)])
        self.applied.append("erlang_floors")
        return self

    def add_total_agent_cap(self) -> "_StaffingModel":
        if self.opt.max_total_agents is not None:
            self.model.Add(sum(self.agents[sh][sk] for sh in self.shifts for sk in self.skills)
                           <= int(self.opt.max_total_agents))
            self.applied.append("total_agent_cap")
        return self

    # -- expanded labour constraints --------------------------------------

    def add_labour_constraints(self) -> "_StaffingModel":
        lc, m, skills = self.opt.labour, self.model, self.skills
        n_bands = len(COVERAGE_BANDS)

        # A. break-coverage buffer
        if lc.break_band_indices and lc.break_buffer_agents:
            for b in lc.break_band_indices:
                if 0 <= b < n_bands:
                    for sk in skills:
                        buf = int(lc.break_buffer_agents.get(sk, 0))
                        if buf > 0:
                            m.Add(self.cover[b][sk] >= self.floors[(b, sk)] + buf)
            self.applied.append("break_coverage")

        # B. hard £ budget cap
        if lc.max_total_budget is not None:
            terms = [int(round(self.cost.get(sk, 0.0) * _COST_SCALE)) * self.agents[sh][sk]
                     for sh in self.shifts for sk in skills]
            m.Add(sum(terms) <= int(round(lc.max_total_budget * _COST_SCALE)))
            self.applied.append("budget_cap")

        # C. inter-shift smoothness
        if lc.max_shift_swing is not None and len(self.shifts) >= 2:
            for sk in skills:
                for i in range(len(self.shifts) - 1):
                    a0, a1 = self.agents[self.shifts[i]][sk], self.agents[self.shifts[i + 1]][sk]
                    m.Add(a1 - a0 <= lc.max_shift_swing)
                    m.Add(a0 - a1 <= lc.max_shift_swing)
            self.applied.append("shift_smoothness")

        # D. cross-skill capacity credit
        if lc.cross_skill_pairs and lc.cross_skill_fraction > 0:
            frac_num = int(round(lc.cross_skill_fraction * 100))
            for b in range(n_bands):
                for (frm, to) in lc.cross_skill_pairs:
                    if frm in skills and to in skills:
                        floor = self.floors[(b, to)]
                        if floor > 0:
                            credit = m.NewIntVar(0, max(1, min(lc.cross_skill_cap, floor)),
                                                 f"xcred_b{b}_{frm}_{to}")
                            m.Add(credit * 100 <= frac_num * self.cover[b][frm])
                            m.Add(self.cover[b][to] + credit >= floor)
            self.applied.append("cross_skill_credit")

        # E. VIP peak-band uplift
        if lc.vip_uplift_scale > 0:
            for sk in skills:
                floor  = self.floors[(PEAK_BAND_INDEX, sk)]
                uplift = int(math.ceil(lc.vip_uplift_scale * self.vip_frac * max(1, floor)))
                if uplift > 0:
                    m.Add(self.cover[PEAK_BAND_INDEX][sk] >= floor + uplift)
            self.applied.append("vip_uplift")

        # F. per-skill FTE cap
        if lc.max_fte_per_skill:
            for sk, capf in lc.max_fte_per_skill.items():
                terms = [self.agents[sh][sk] for sh in self.shifts if sk in self.agents[sh]]
                if terms:
                    m.Add(sum(terms) <= int(capf))
            self.applied.append("fte_cap_per_skill")

        # G. intra-band balance
        if lc.max_band_spread is not None and len(skills) >= 2:
            for b in range(n_bands):
                for i in range(len(skills)):
                    for j in range(i + 1, len(skills)):
                        ci, cj = self.cover[b][skills[i]], self.cover[b][skills[j]]
                        m.Add(ci - cj <= lc.max_band_spread)
                        m.Add(cj - ci <= lc.max_band_spread)
            self.applied.append("band_balance")

        # H. over-staffing cap (idle protection)
        if lc.max_overstaff_ratio is not None:
            for b in range(n_bands):
                for sk in skills:
                    ceil_cov = int(math.ceil(self.floors[(b, sk)] * (1.0 + lc.max_overstaff_ratio)))
                    m.Add(self.cover[b][sk] <= max(ceil_cov, self.floors[(b, sk)]))
            self.applied.append("overstaff_cap")

        # I. absolute minimum coverage
        if lc.min_coverage_floor is not None:
            for b in range(n_bands):
                for sk in skills:
                    m.Add(self.cover[b][sk] >= int(lc.min_coverage_floor))
            self.applied.append("min_coverage_floor")

        return self

    # -- break-scheduling sub-model (items 4 & 5) -------------------------

    def add_break_distribution(self) -> "_StaffingModel":
        """Aggregate, slot-based break model layered on the headcount model.

        Decision vars  brk[sh][sk][t] >= 0  = number of (shift sh, skill sk)
        agents on break during slot t (t only over slots eligible for sh).

        Constraints (math in module docstring / review notes):
          (1) demand   : sum_t brk[sh][sk][t] = breaks_per_agent * n[sh][sk]
          (2) coverage : for every slot t & skill sk,
                         sum_{sh active} n[sh][sk] - sum_{sh} brk[sh][sk][t] >= floor(band(t),sk)
          (3) concurr. : 100 * sum_sh brk[sh][sk][t] <= round(frac*100) * sum_{sh active} n[sh][sk]
          (4) balance  : break_peak >= sum_{sh,sk} brk[sh][sk][t]  (minimised in objective)
          (5) lunch    : if a lunch window is set, sum_{t in window} brk[sh][sk][t] >= n[sh][sk]
        """
        lc = self.opt.labour
        if not lc.enable_break_optimization:
            self.brk = {}
            self.break_peak = None
            return self

        m = self.model
        L = max(15, int(lc.break_slot_minutes))
        n_slots = HORIZON_MINUTES // L
        self._break_L = L
        self._break_n_slots = n_slots

        # eligibility: slot t (midpoint mt) eligible for shift sh if it sits
        # inside [start+head, end-tail] (encodes "no break near shift start/end"
        # and a minimum continuous-work lead-in).
        def eligible(sh: str, t: int) -> bool:
            mt = (t + 0.5) * L
            s_start = next(s for nm, s, _ in SHIFT_WINDOWS if nm == sh)
            s_end   = next(e for nm, _, e in SHIFT_WINDOWS if nm == sh)
            return (s_start + lc.no_break_head_minutes) <= mt <= (s_end - lc.no_break_tail_minutes)

        def active(sh: str, t: int) -> bool:
            mt = (t + 0.5) * L
            s_start = next(s for nm, s, _ in SHIFT_WINDOWS if nm == sh)
            s_end   = next(e for nm, _, e in SHIFT_WINDOWS if nm == sh)
            return s_start <= mt < s_end

        cap = self.opt.max_agents_per_skill
        self.brk: Dict[str, Dict[str, Dict[int, cp_model.IntVar]]] = {}
        for sh in self.shifts:
            self.brk[sh] = {}
            for sk in self.skills:
                self.brk[sh][sk] = {
                    t: m.NewIntVar(0, cap, f"brk_{sh}_{sk}_{t}")
                    for t in range(n_slots) if eligible(sh, t)
                }

        bpa = max(1, int(lc.breaks_per_agent))

        # (1) demand conservation
        for sh in self.shifts:
            for sk in self.skills:
                slots = self.brk[sh][sk]
                if slots:
                    m.Add(sum(slots.values()) == bpa * self.agents[sh][sk])
                else:
                    # no eligible slot -> nobody can be assigned there; force 0 demand
                    # (only happens if head+tail erase the shift); keep feasible.
                    pass

        # (2) coverage floor every slot, and (3) concurrency cap
        frac100 = int(round(max(0.0, min(1.0, lc.max_break_fraction_per_slot)) * 100))
        for t in range(n_slots):
            band = _band_index_for_minute((t + 0.5) * L)
            for sk in self.skills:
                active_shifts = [sh for sh in self.shifts if active(sh, t)]
                if not active_shifts:
                    continue
                on_break = [self.brk[sh][sk][t] for sh in active_shifts
                            if t in self.brk[sh][sk]]
                present = sum(self.agents[sh][sk] for sh in active_shifts)
                if on_break:
                    # (2) effective coverage >= Erlang floor for this slot's band
                    m.Add(present - sum(on_break) >= self.floors[(band, sk)])
                    # (3) at most `frac` of present agents on break simultaneously
                    m.Add(100 * sum(on_break) <= frac100 * present)

        # (4) balance: minimise the busiest break slot (anti-concentration).
        self.break_peak = m.NewIntVar(0, cap * len(self.shifts) * len(self.skills), "break_peak")
        for t in range(n_slots):
            total_t = [self.brk[sh][sk][t] for sh in self.shifts for sk in self.skills
                       if t in self.brk[sh][sk]]
            if total_t:
                m.Add(self.break_peak >= sum(total_t))

        # (5) lunch window: at least one break-slot per agent inside the window.
        if lc.lunch_window is not None and bpa >= 1:
            lo, hi = lc.lunch_window
            for sh in self.shifts:
                for sk in self.skills:
                    win = [v for t, v in self.brk[sh][sk].items()
                           if lo <= (t + 0.5) * L <= hi]
                    if win:
                        m.Add(sum(win) >= self.agents[sh][sk])

        self.applied.append("break_optimization")
        return self

    # -- objective --------------------------------------------------------

    def set_objective(self, obj_coeff: Dict[str, int]) -> "_StaffingModel":
        cost_term = sum(obj_coeff[sk] * self.agents[sh][sk]
                        for sh in self.shifts for sk in self.skills)
        terms = [cost_term]
        ow = self.opt.overstaff_weight
        if ow > 0:
            overstaff = sum(self.cover[b][sk] - self.floors[(b, sk)]
                            for b in range(len(COVERAGE_BANDS)) for sk in self.skills)
            terms.append(int(round(ow * _COST_SCALE)) * overstaff)
            self.applied.append("multi_objective(cost+overstaff)")
        # break balancing: penalise the busiest break slot to spread breaks out
        bw = self.opt.labour.break_balance_weight
        peak = getattr(self, "break_peak", None)
        if bw > 0 and peak is not None:
            terms.append(int(round(bw * _COST_SCALE)) * peak)
            self.applied.append("break_balance")
        self.model.Minimize(sum(terms))
        return self


# ===========================================================================
# StaffingOptimizer  (restructured: solve() is now a pipeline)
# ===========================================================================

class StaffingOptimizer:
    _DEFAULT_WAGES    = {"billing": 18.0, "technical": 22.0, "general": 16.0}
    _DEFAULT_OVERHEAD = 1.30
    _SHIFT_HOURS      = 8.0

    _CPSAT_STATUS: Dict[int, str] = {
        cp_model.OPTIMAL: "optimal", cp_model.FEASIBLE: "feasible",
        cp_model.INFEASIBLE: "infeasible", cp_model.UNKNOWN: "unknown",
        cp_model.MODEL_INVALID: "model_invalid",
    }

    def __init__(self, sim_cfg: SimulationConfig, opt_cfg: OptimizationConfig,
                 cost_cfg=None, analytical_sla_target: Optional[float] = None) -> None:
        self._sim    = sim_cfg
        self._opt    = opt_cfg
        self._cost   = cost_cfg
        self._target = (analytical_sla_target if analytical_sla_target is not None
                        else opt_cfg.sla_target)
        self.status:  str                          = "not_solved"
        self.value:   Optional[Dict[str, int]]     = None
        self._result: Optional[OptimizationResult] = None
        self._last_sim_feedback: Optional[Dict[str, float]] = None

    # -- cost helpers ------------------------------------------------------

    def _cost_params(self) -> Tuple[Dict[str, float], float]:
        try:
            return self._cost.hourly_wage_per_skill, self._cost.overhead_factor
        except AttributeError:
            return self._DEFAULT_WAGES, self._DEFAULT_OVERHEAD

    def _agent_cost(self, skill: str, wages: Dict[str, float], overhead: float) -> float:
        return wages.get(skill, self._DEFAULT_WAGES.get(skill, 18.0)) * overhead * self._SHIFT_HOURS

    def _objective_coeffs(self, skills, cost_per_agent, band_erlang_min,
                          lam_buffered, lam_per_min_base, mean_svc, skill_mix,
                          total_mix, n) -> Dict[str, int]:
        """SLA-penalty-adjusted £ cost coefficients (kept from the original)."""
        penalty = float(self._opt.sla_violation_penalty_per_call)
        adjusted = dict(cost_per_agent)
        fb_scale = 1.0
        if self._opt.sim_feedback_penalty_scale > 0.0 and self._last_sim_feedback:
            last_gap = self._last_sim_feedback.get("sla_gap", 0.0)
            abn      = self._last_sim_feedback.get("abandonment_rate", 0.0)
            if last_gap < 0:
                fb_scale += abs(last_gap) * float(self._opt.sim_feedback_penalty_scale)
            fb_scale *= 1.0 + abn * float(self._opt.sim_feedback_penalty_scale)
        if penalty > 0.0:
            calls_per_shift = lam_per_min_base * self._SHIFT_HOURS * 60
            for skill in skills:
                frac = skill_mix.get(skill, 1.0 / n) / total_mix
                skill_lam_peak = lam_buffered * _arrival_fraction_for_band(PEAK_BAND_INDEX) * frac
                sla_at_peak = ErlangC.sla_probability(
                    band_erlang_min[(PEAK_BAND_INDEX, skill)], round(skill_lam_peak, 6),
                    round(mean_svc, 6), self._opt.sla_threshold_minutes)
                gap = max(0.0, self._target - sla_at_peak)
                adjusted[skill] += penalty * fb_scale * gap * (calls_per_shift * frac)
        return {sk: int(round(adjusted[sk] * _COST_SCALE)) for sk in skills}

    # -- the pipeline ------------------------------------------------------

    def solve(self) -> OptimizationResult:
        t0       = time.perf_counter()
        skills   = list(self._sim.agents_per_skill.keys())
        n        = len(skills)
        shifts   = [name for name, _, _ in SHIFT_WINDOWS]
        wages, overhead = self._cost_params()

        lam_per_min_base = self._sim.arrival_rate_per_minute
        lam_buffered     = lam_per_min_base * max(1.0, float(self._opt.arrival_rate_buffer))
        mean_svc         = self._sim.mean_service_minutes + self._sim.acw_mean_minutes
        skill_mix        = self._sim.skill_mix
        total_mix        = sum(skill_mix.values()) or 1.0
        vip_fraction     = self._sim.customer_tier_mix.get("vip", 0.1)

        # 1. analytical floors (with shrinkage gross-up)
        band_erlang_min = _build_band_erlang_mins(
            self._sim, self._opt, self._target, lam_buffered, mean_svc)

        # 2. £ cost + SLA-penalty-adjusted objective coefficients
        cost_per_agent = {sk: self._agent_cost(sk, wages, overhead) for sk in skills}
        obj_coeff      = self._objective_coeffs(
            skills, cost_per_agent, band_erlang_min, lam_buffered,
            lam_per_min_base, mean_svc, skill_mix, total_mix, n)

        # 3. compose the model (one concern per call)
        builder = (_StaffingModel(skills, shifts, self._opt, band_erlang_min,
                                  cost_per_agent, skill_mix, vip_fraction)
                   .add_decision_vars()
                   .add_coverage_links()
                   .add_base_floors()
                   .add_total_agent_cap()
                   .add_labour_constraints()
                   .add_break_distribution()
                   .set_objective(obj_coeff))

        # 4. solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self._opt.cpsat_time_limit_seconds
        solver.parameters.num_search_workers  = 8
        if self._opt.random_seed is not None:
            solver.parameters.random_seed = int(self._opt.random_seed)
        if self._opt.debug_solver:
            solver.parameters.log_search_progress = True
        cpsat_status = solver.Solve(builder.model)
        status_str   = self._CPSAT_STATUS.get(cpsat_status, f"cpsat_{cpsat_status}")

        # 5. extract plan (or Erlang fallback)
        if cpsat_status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            shift_plan = {sh: {sk: solver.Value(builder.agents[sh][sk]) for sk in skills}
                          for sh in shifts}
            status_out = status_str
            break_plan = {}
            if builder.brk:
                for sh in builder.brk:
                    for sk in builder.brk[sh]:
                        for t, var in builder.brk[sh][sk].items():
                            val = solver.Value(var)
                            if val > 0:
                                break_plan[(sh, sk, t)] = val
            logger.info("CP-SAT %s: plan=%s obj=£%.2f wall=%.1fms", status_str, shift_plan,
                        solver.ObjectiveValue() / _COST_SCALE, solver.WallTime() * 1000)
        else:
            logger.warning("CP-SAT %s — Erlang peak-band fallback.", status_str)
            shift_plan = {sh: {sk: band_erlang_min[(PEAK_BAND_INDEX, sk)] for sk in skills}
                          for sh in shifts}
            status_out = f"{status_str}_fallback"
            break_plan = {}

        # 6. assemble result
        band_coverage    = _band_coverage_from_shift_plan(shift_plan, skills)
        agents_per_skill = dict(band_coverage[PEAK_BAND_INDEX])
        peak_frac        = _arrival_fraction_for_band(PEAK_BAND_INDEX)
        analytical_sla   = {}
        for skill in skills:
            frac      = skill_mix.get(skill, 1.0 / n) / total_mix
            skill_lam = lam_buffered * peak_frac * frac
            analytical_sla[skill] = ErlangC.sla_probability(
                agents_per_skill[skill], round(skill_lam, 6),
                round(mean_svc, 6), self._opt.sla_threshold_minutes)
        total_staffing_cost = sum(cost_per_agent[sk] * shift_plan[sh][sk]
                                  for sh in shifts for sk in skills)

        self.status  = status_out
        self.value   = agents_per_skill
        self._result = OptimizationResult(
            agents_per_skill=agents_per_skill, shift_plan=shift_plan,
            band_coverage=band_coverage, analytical_sla=analytical_sla,
            analytical_target=self._target, total_staffing_cost=total_staffing_cost,
            status=status_out, solve_time_seconds=time.perf_counter() - t0,
            constraints_applied=builder.applied,
            break_plan=break_plan,
            break_slot_minutes=builder._break_L)
        return self._result

    @property
    def result(self) -> Optional[OptimizationResult]:
        return self._result


# ===========================================================================
# SimulationEvaluator  (unchanged)
# ===========================================================================

class SimulationEvaluator:
    def __init__(self, base_sim_cfg: SimulationConfig, opt_cfg: OptimizationConfig,
                 cost_cfg=None, beh_cfg=None,
                 weights: Optional[RouterScoreWeights] = None) -> None:
        self._base = base_sim_cfg
        self._opt  = opt_cfg
        self._cost = cost_cfg
        self._beh  = beh_cfg
        self._weights = weights

    def evaluate(self, opt_result: OptimizationResult) -> EvaluationResult:
        t0     = time.perf_counter()
        cfg    = self._build_config(opt_result)
        engine = self._build_engine(cfg)
        engine.run()
        elapsed = time.perf_counter() - t0
        kpi = engine.kpi
        total_cost, cost_breakdown = 0.0, {}
        if hasattr(engine, "cost_function"):
            bk = engine.cost_function.breakdown()
            total_cost = bk.get("total", 0.0)
            cost_breakdown = {k: v for k, v in bk.items() if k != "total"}
        return EvaluationResult(
            agents_per_skill=opt_result.agents_per_skill, shift_plan=opt_result.shift_plan,
            sla=kpi.sla_percentage(), abandonment_rate=kpi.abandonment_rate(),
            avg_csat=kpi.average_csat(), asa=kpi.average_speed_of_answer(),
            aht=kpi.average_handle_time(), total_calls=kpi.total_calls(),
            total_cost=total_cost, cost_breakdown=cost_breakdown, sim_time_seconds=elapsed)

    def _build_config(self, opt_result: OptimizationResult) -> SimulationConfig:
        cfg = copy.copy(self._base)
        cfg.agents_per_skill     = dict(opt_result.agents_per_skill)
        cfg.sim_duration_minutes = float(HORIZON_MINUTES)
        cfg.break_schedule = [
            (_SHIFT_BREAK_MIDPOINTS["M"], _BREAK_DURATION_MINUTES),
            (_SHIFT_BREAK_MIDPOINTS["D"], _BREAK_DURATION_MINUTES),
            (_SHIFT_BREAK_MIDPOINTS["E"], _BREAK_DURATION_MINUTES),
        ]
        return cfg

    def _build_engine(self, cfg: SimulationConfig):
        engine_type = self._opt.engine_type
        try:
            from cost_system import CostAwareEngine, CostAwareRealisticsEngine
            from core_simulation import BehaviorConfig
            if engine_type == "realism":
                return CostAwareRealisticsEngine(cfg, self._cost, self._beh or BehaviorConfig(),
                                                 self._weights)
            if engine_type == "cost":
                return CostAwareEngine(cfg, self._cost, self._weights)
        except ImportError:
            pass
        return SimulationEngine(cfg, self._weights)


# ===========================================================================
# Optimize -> Simulate loop  (unchanged logic; reads the same result shape)
# ===========================================================================

@dataclass
class _LoopIteration:
    iteration:         int
    analytical_target: float
    opt_result:        OptimizationResult
    eval_result:       EvaluationResult
    sla_gap:           float
    converged:         bool


class OptimizeSimulateLoop:
    def __init__(self, base_sim_cfg: SimulationConfig, opt_cfg: OptimizationConfig,
                 cost_cfg=None, beh_cfg=None,
                 weights: Optional[RouterScoreWeights] = None) -> None:
        self._base = base_sim_cfg
        self._opt  = opt_cfg
        self._cost = cost_cfg
        self._beh  = beh_cfg
        self._weights = weights
        self._evaluator = SimulationEvaluator(base_sim_cfg, opt_cfg, cost_cfg, beh_cfg, weights)
        self._history: List[_LoopIteration] = []
        self.report = LoopReport(self._history, opt_cfg)
        self.pareto_points: List[Dict] = []

    def run(self) -> List[_LoopIteration]:
        opt_cfg = self._opt
        target  = opt_cfg.sla_target
        if opt_cfg.verbose:
            adaptive_tag = ("adaptive" if opt_cfg.analytical_safety_margin is None
                            else f"fixed={opt_cfg.analytical_safety_margin:.2f}")
            print("\n" + "=" * 72)
            print("  STAFFING OPTIMISATION + SIMULATION LOOP  [OR-Tools CP-SAT]")
            print(f"  Horizon  : {HORIZON_MINUTES} min  |  Shifts: M(0-480) D(120-600) E(240-720)")
            print(f"  SLA tgt  : {opt_cfg.sla_target:.1%}  |  Engine: {opt_cfg.engine_type}"
                  f"  |  max_iter: {opt_cfg.max_iterations}  |  correction: {adaptive_tag}")
            if opt_cfg.labour.enabled():
                print(f"  Labour constraints : ENABLED")
            print("=" * 72)

        _sim_feedback: Optional[Dict[str, float]] = None
        for iteration in range(1, opt_cfg.max_iterations + 1):
            if opt_cfg.verbose:
                print(f"\n  [Iter {iteration}] Analytical SLA target = {target:.3%}")
            optimizer = StaffingOptimizer(self._base, opt_cfg, self._cost, target)
            optimizer._last_sim_feedback = _sim_feedback
            opt_result = optimizer.solve()
            if opt_cfg.verbose:
                print(f"  [Iter {iteration}] CP-SAT status={opt_result.status}"
                      f"  constraints={opt_result.constraints_applied}")
                for sh, plan in opt_result.shift_plan.items():
                    sh_start = next(s for nm, s, _ in SHIFT_WINDOWS if nm == sh)
                    sh_end   = next(e for nm, _, e in SHIFT_WINDOWS if nm == sh)
                    print(f"    Shift {sh} ({sh_start}-{sh_end}m): {plan}")
                print(f"    Peak coverage: {opt_result.agents_per_skill}"
                      f"  cost=£{opt_result.total_staffing_cost:,.2f}")
            eval_result = self._evaluator.evaluate(opt_result)
            sla_gap     = eval_result.sla - opt_cfg.sla_target
            converged   = sla_gap >= -opt_cfg.convergence_tolerance
            _sim_feedback = {"sla_gap": sla_gap, "abandonment_rate": eval_result.abandonment_rate}
            if opt_cfg.verbose:
                print(f"  [Iter {iteration}] Sim SLA={eval_result.sla:.1%}  "
                      f"abandon={eval_result.abandonment_rate:.1%}  gap={sla_gap:+.1%}  "
                      f"{'CONVERGED' if converged else 'below target'}")
            self._history.append(_LoopIteration(iteration, target, opt_result, eval_result,
                                                 sla_gap, converged))
            if converged:
                if opt_cfg.verbose:
                    print(f"\n  Converged at iteration {iteration}.")
                break
            correction = (max(0.01, min(0.10, abs(sla_gap) * 1.5))
                          if opt_cfg.analytical_safety_margin is None
                          else opt_cfg.analytical_safety_margin)
            target = min(0.999, target + correction)
        else:
            if opt_cfg.verbose:
                print(f"\n  Max iterations ({opt_cfg.max_iterations}) reached without convergence.")
        if opt_cfg.pareto_sweep:
            self._run_pareto_sweep()
        return self._history

    @property
    def best_plan(self) -> Optional[Dict[str, int]]:
        e = self._best_history_entry()
        return e.opt_result.agents_per_skill if e else None

    @property
    def best_shift_plan(self) -> Optional[Dict[str, Dict[str, int]]]:
        e = self._best_history_entry()
        return e.opt_result.shift_plan if e else None

    @property
    def best_evaluation(self) -> Optional[EvaluationResult]:
        e = self._best_history_entry()
        return e.eval_result if e else None

    def _best_history_entry(self) -> Optional[_LoopIteration]:
        if not self._history:
            return None
        converged = [h for h in self._history if h.converged]
        if converged:
            return converged[-1]
        return max(self._history, key=lambda h: h.eval_result.sla)

    def _run_pareto_sweep(self) -> None:
        # A genuine cost-vs-service frontier comes from sweeping the SLA TARGET
        # (lower target -> fewer agents -> cheaper -> lower SLA). Sweeping the
        # cost/sla weights did nothing useful here, because the Erlang floor
        # already saturates analytical SLA at ~100% for any single target.
        targets = [0.70, 0.80, 0.90, 0.95, 0.99]
        self.pareto_points = []
        for tgt in targets:
            tmp_cfg = copy.copy(self._opt)
            tmp_cfg.verbose = False
            solver = StaffingOptimizer(self._base, tmp_cfg, self._cost, tgt)
            result = solver.solve()
            avg_erlang = (sum(result.analytical_sla.values()) / len(result.analytical_sla)
                          if result.analytical_sla else 0.0)
            total_agents = sum(result.shift_plan[sh][sk]
                               for sh in result.shift_plan for sk in result.shift_plan[sh])
            self.pareto_points.append({
                "target": tgt, "peak_plan": result.agents_per_skill,
                "total_agents": total_agents, "shift_plan": result.shift_plan,
                "staffing_cost": result.total_staffing_cost, "erlang_sla_avg": avg_erlang})

    def print_pareto_summary(self) -> None:
        if not self.pareto_points:
            print("[pareto] No sweep data — set pareto_sweep=True before run().")
            return
        w = 84
        print(f"\n{'=' * w}\n  COST vs SLA FRONTIER  (Erlang-C analytical, 720-min horizon)\n{'=' * w}")
        print(f"  {'SLA target':>11}  {'Peak plan':>34}  {'Agents':>7}  {'Cost £':>10}  {'SLA (anlyt)':>12}")
        print(f"  {'-' * (w - 4)}")
        for pt in self.pareto_points:
            print(f"  {pt['target']:>10.0%}  {str(pt['peak_plan']):>34}  "
                  f"{pt['total_agents']:>7}  £{pt['staffing_cost']:>8,.0f}  "
                  f"{pt['erlang_sla_avg']:>12.1%}")
        print(f"{'=' * w}\n")

    def explain_plan(self, opt_result: Optional[OptimizationResult] = None,
                     eval_result: Optional[EvaluationResult] = None) -> None:
        entry = self._best_history_entry()
        opt_result  = opt_result  or (entry.opt_result  if entry else None)
        eval_result = eval_result or (entry.eval_result if entry else None)
        if opt_result is None:
            print("[explain] No plan available — run() first.")
            return
        cfg, opt = self._base, self._opt
        lam      = cfg.arrival_rate_per_minute
        mean_svc = cfg.mean_service_minutes + cfg.acw_mean_minutes
        try:
            wages, overhead = self._cost.hourly_wage_per_skill, self._cost.overhead_factor
        except AttributeError:
            wages, overhead = StaffingOptimizer._DEFAULT_WAGES, StaffingOptimizer._DEFAULT_OVERHEAD
        w = 72
        print(f"\n{'=' * w}\n  STAFFING PLAN EXPLANATION  [OR-Tools CP-SAT, 720-min horizon]\n{'=' * w}")
        print(f"  Arrival rate : {cfg.arrival_rate_per_hour:.0f} calls/hr  ({lam:.3f}/min)")
        print(f"  Service time : {mean_svc:.2f} min (handle + ACW)")
        print(f"  SLA target   : >={opt.sla_target:.0%} within {opt.sla_threshold_minutes:.1f} min")
        if opt_result.constraints_applied:
            print(f"  Constraints  : {', '.join(opt_result.constraints_applied)}")
        if eval_result is not None:
            sla_ok = "PASS" if eval_result.sla >= opt.sla_target else "FAIL"
            print(f"\n  Simulation outcome (720-min run):")
            print(f"    SLA={eval_result.sla:.1%} [{sla_ok}]  abandon={eval_result.abandonment_rate:.1%}"
                  f"  CSAT={eval_result.avg_csat:.2f}  cost=£{eval_result.total_cost:,.2f}")
        print(f"\n  Shift plan:")
        for sh_name, sh_start, sh_end in SHIFT_WINDOWS:
            plan = opt_result.shift_plan.get(sh_name, {})
            cost = sum(plan.get(sk, 0) * wages.get(sk, 18.0) * overhead * StaffingOptimizer._SHIFT_HOURS
                       for sk in plan)
            print(f"    Shift {sh_name} ({sh_start}-{sh_end}m) break@{_SHIFT_BREAK_MIDPOINTS[sh_name]:.0f}m"
                  f"  cost=£{cost:,.2f}: {plan}")
        print(f"\n  Coverage by band:")
        skills = sorted(opt_result.agents_per_skill.keys())
        for band_idx, (b_start, b_end, active_shifts) in enumerate(COVERAGE_BANDS):
            cov = opt_result.band_coverage.get(band_idx, {})
            tag = "  <- peak" if band_idx == PEAK_BAND_INDEX else ""
            cells = "  ".join(f"{s[:3]}={cov.get(s, 0)}" for s in skills)
            print(f"    Band{band_idx} [{b_start:>4}-{b_end:>4}m] {'+'.join(active_shifts):>7}  {cells}{tag}")
        print(f"\n  Total staffing cost : £{opt_result.total_staffing_cost:,.2f}\n{'=' * w}\n")

    def plot_results(self, save_path: Optional[str] = None) -> None:
        if not _MPL_AVAILABLE:
            warnings.warn("matplotlib not installed — cannot plot.", RuntimeWarning)
            return
        if not self._history:
            print("[plot] No history — call run() first.")
            return
        iters    = [h.iteration for h in self._history]
        sim_slas = [h.eval_result.sla * 100 for h in self._history]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(iters, sim_slas, "o-", label="Simulated SLA")
        ax.axhline(self._opt.sla_target * 100, color="red", linestyle=":", label="target")
        ax.set_xlabel("Iteration"); ax.set_ylabel("SLA (%)"); ax.legend()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()


# ===========================================================================
# LoopReport  (unchanged structure)
# ===========================================================================

class LoopReport:
    def __init__(self, history: List[_LoopIteration], cfg: OptimizationConfig) -> None:
        self._history = history
        self._cfg     = cfg

    def print_report(self) -> None:
        if not self._history:
            print("[LoopReport] No iterations to report.")
            return
        w, cfg = 88, self._cfg
        bar = "=" * w
        correction_mode = ("adaptive" if cfg.analytical_safety_margin is None
                           else f"fixed={cfg.analytical_safety_margin:.3f}")
        print(f"\n{bar}")
        print(f"  OPTIMIZE -> SIMULATE LOOP  [OR-Tools CP-SAT | {HORIZON_MINUTES}-min horizon]")
        print(f"  SLA target={cfg.sla_target:.1%}  |  Engine: {cfg.engine_type}"
              f"  |  Correction: {correction_mode}")
        if cfg.labour.enabled():
            print(f"  Labour constraints     : ENABLED  (shrinkage={cfg.labour.shrinkage:.0%})")
        print(f"{bar}")
        skill_list = sorted(self._history[0].opt_result.agents_per_skill.keys())
        pk_hdr = "  ".join(f"{s[:4]:>5}" for s in skill_list)
        print(f"  {'Iter':>4}  {'ATarget':>7}  {pk_hdr}  {'SimSLA':>7}  {'Abandon':>8}  "
              f"{'CSAT':>6}  {'Cost £':>10}  {'OK?':>4}")
        print(f"  {'-' * (w - 2)}")
        for h in self._history:
            plan   = h.opt_result.agents_per_skill
            agents = "  ".join(f"{plan.get(s, 0):>5}" for s in skill_list)
            ev     = h.eval_result
            ok     = "Y" if h.converged else "N"
            cost_s = f"£{ev.total_cost:>9,.2f}" if ev.total_cost else "    n/a   "
            print(f"  {h.iteration:>4}  {h.analytical_target:>7.3%}  {agents}  {ev.sla:>7.1%}  "
                  f"{ev.abandonment_rate:>8.1%}  {ev.avg_csat:>6.2f}  {cost_s}  {ok:>4}")
        print(f"  {'-' * (w - 2)}")
        best_entry = max(self._history, key=lambda h: h.eval_result.sla)
        best_ev    = best_entry.eval_result
        converged  = any(h.converged for h in self._history)
        print(f"\n  Final status      : {'CONVERGED' if converged else 'MAX ITER REACHED'}")
        print(f"  Iterations run    : {len(self._history)}")
        print(f"  Best SLA achieved : {best_ev.sla:.1%}  (target: {cfg.sla_target:.1%})")
        print(f"  Peak coverage     : {best_ev.agents_per_skill}")
        if best_entry.opt_result.constraints_applied:
            print(f"  Constraints       : {', '.join(best_entry.opt_result.constraints_applied)}")
        print(f"\n  Best shift plan:")
        for sh_name, sh_start, sh_end in SHIFT_WINDOWS:
            plan = best_entry.opt_result.shift_plan.get(sh_name, {})
            print(f"    Shift {sh_name} ({sh_start:>4}-{sh_end:>4}m): {plan}")
        if best_ev.total_cost:
            print(f"\n  Total cost        : £{best_ev.total_cost:,.2f}")
            if best_ev.cost_breakdown:
                print(f"  Cost breakdown:")
                for component, amount in sorted(best_ev.cost_breakdown.items(), key=lambda x: -x[1]):
                    print(f"    {component:<24}  £{amount:>10,.2f}")
        print(f"\n{bar}\n")


# ===========================================================================
# COVERAGE VALIDATION + BREAK/STABILITY REPORTING  (items 6 & 7)
# ===========================================================================

@dataclass
class CoverageValidationReport:
    feasible:            bool
    band_violations:     List[Tuple[int, str, int, int]]      # (band, skill, cover, floor)
    slot_violations:     List[Tuple[int, str, int, int]]      # (slot, skill, eff_cover, floor)
    # stability (per skill): min / mean / std / max-adjacent-dip of band coverage
    stability:           Dict[str, Dict[str, float]] = field(default_factory=dict)
    # break distribution
    breaks_total:        int = 0
    break_slots_used:    int = 0
    break_peak_overlap:  int = 0
    break_mean_per_slot: float = 0.0
    break_std_per_slot:  float = 0.0
    slot_minutes:        int = 0


class CoverageValidator:
    """Validates a solved plan: minimum staffing per band AND per break-slot
    (effective coverage net of breaks), break-limit compliance, and staffing
    stability across the horizon. Pure post-solve checking — no solver state."""

    def __init__(self, result: OptimizationResult, sim_cfg: SimulationConfig,
                 opt_cfg: OptimizationConfig) -> None:
        self._r   = result
        self._sim = sim_cfg
        self._opt = opt_cfg
        lam_buffered = sim_cfg.arrival_rate_per_minute * max(1.0, float(opt_cfg.arrival_rate_buffer))
        mean_svc     = sim_cfg.mean_service_minutes + sim_cfg.acw_mean_minutes
        self._floors = _build_band_erlang_mins(sim_cfg, opt_cfg, opt_cfg.sla_target,
                                               lam_buffered, mean_svc)
        self._skills = list(sim_cfg.agents_per_skill.keys())

    def validate(self) -> CoverageValidationReport:
        r, skills = self._r, self._skills
        band_viol: List[Tuple[int, str, int, int]] = []
        for b in range(len(COVERAGE_BANDS)):
            for sk in skills:
                cov   = r.band_coverage.get(b, {}).get(sk, 0)
                floor = self._floors[(b, sk)]
                if cov < floor:
                    band_viol.append((b, sk, cov, floor))

        # per-slot effective coverage (only meaningful if a break plan exists)
        slot_viol: List[Tuple[int, str, int, int]] = []
        L = r.break_slot_minutes
        per_slot_counts: List[int] = []
        breaks_total = sum(r.break_plan.values())
        peak_overlap = 0
        slots_used = 0
        if L and r.break_plan:
            n_slots = HORIZON_MINUTES // L
            for t in range(n_slots):
                mt   = (t + 0.5) * L
                band = _band_index_for_minute(mt)
                slot_total = 0
                for sk in skills:
                    present = sum(r.shift_plan[sh][sk]
                                  for nm, s, e in SHIFT_WINDOWS
                                  for sh in [nm] if s <= mt < e)
                    on_break = sum(v for (sh, s2, tt), v in r.break_plan.items()
                                   if s2 == sk and tt == t)
                    slot_total += on_break
                    eff = present - on_break
                    if eff < self._floors[(band, sk)]:
                        slot_viol.append((t, sk, eff, self._floors[(band, sk)]))
                per_slot_counts.append(slot_total)
                if slot_total > 0:
                    slots_used += 1
                peak_overlap = max(peak_overlap, slot_total)

        # staffing stability across bands (per skill)
        stability: Dict[str, Dict[str, float]] = {}
        for sk in skills:
            covs = [r.band_coverage.get(b, {}).get(sk, 0) for b in range(len(COVERAGE_BANDS))]
            mean = sum(covs) / len(covs)
            var  = sum((c - mean) ** 2 for c in covs) / len(covs)
            dips = [max(0, covs[i] - covs[i + 1]) for i in range(len(covs) - 1)]
            stability[sk] = {
                "min": float(min(covs)), "mean": float(mean),
                "std": float(var ** 0.5), "max_dip": float(max(dips) if dips else 0),
            }

        mean_ps = (sum(per_slot_counts) / len(per_slot_counts)) if per_slot_counts else 0.0
        std_ps  = ((sum((c - mean_ps) ** 2 for c in per_slot_counts) / len(per_slot_counts)) ** 0.5
                   if per_slot_counts else 0.0)

        return CoverageValidationReport(
            feasible=not band_viol and not slot_viol,
            band_violations=band_viol, slot_violations=slot_viol,
            stability=stability, breaks_total=breaks_total,
            break_slots_used=slots_used, break_peak_overlap=peak_overlap,
            break_mean_per_slot=mean_ps, break_std_per_slot=std_ps, slot_minutes=L or 0)

    def print_report(self) -> None:
        rep = self.validate()
        w = 78
        bar = "=" * w
        print(f"\n{bar}\n  COVERAGE VALIDATION & STAFFING STABILITY\n{bar}")
        verdict = "PASS — all minimum-staffing constraints satisfied" if rep.feasible \
            else f"FAIL — {len(rep.band_violations)} band + {len(rep.slot_violations)} slot violation(s)"
        print(f"  Verdict : {verdict}")

        if rep.band_violations:
            print(f"\n  Band-coverage violations (cover < Erlang floor):")
            for b, sk, cov, fl in rep.band_violations:
                print(f"    band {b} [{COVERAGE_BANDS[b][0]}-{COVERAGE_BANDS[b][1]}m] "
                      f"{sk:<10} cover={cov} < floor={fl}")
        if rep.slot_violations:
            print(f"\n  Break-slot effective-coverage violations (first 10):")
            for t, sk, eff, fl in rep.slot_violations[:10]:
                print(f"    slot {t} ({t*rep.slot_minutes}-{(t+1)*rep.slot_minutes}m) "
                      f"{sk:<10} eff={eff} < floor={fl}")

        print(f"\n  Staffing stability (coverage across bands):")
        print(f"    {'Skill':<12}{'Min':>6}{'Mean':>8}{'Std':>8}{'MaxDip':>8}")
        for sk, s in rep.stability.items():
            print(f"    {sk:<12}{s['min']:>6.0f}{s['mean']:>8.2f}{s['std']:>8.2f}{s['max_dip']:>8.0f}")

        if rep.slot_minutes:
            print(f"\n  Break distribution ({rep.slot_minutes}-min slots):")
            print(f"    Total break-slots assigned : {rep.breaks_total}")
            print(f"    Distinct slots used        : {rep.break_slots_used}")
            print(f"    Peak simultaneous on break : {rep.break_peak_overlap}")
            print(f"    Mean / std per slot        : {rep.break_mean_per_slot:.2f} / "
                  f"{rep.break_std_per_slot:.2f}")
            overlap_ratio = (rep.break_peak_overlap / rep.break_mean_per_slot
                             if rep.break_mean_per_slot > 0 else 0.0)
            print(f"    Overlap concentration      : {overlap_ratio:.2f}x mean "
                  f"({'well spread' if overlap_ratio < 2.0 else 'clustered'})")
        else:
            print(f"\n  (break optimization not enabled — no break-slot analysis)")
        print(f"{bar}\n")


# ===========================================================================
# SURROGATE MODEL WARM-UP  (item 4a)
# ===========================================================================
#
# A full simulation is the ground truth but is expensive. A surrogate learns
# plan-features -> simulated KPI from historical sim outcomes, so the optimiser
# can SCREEN many candidate plans cheaply and reserve full simulation for final
# validation / high-uncertainty cases.

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error
    _SKL_SURROGATE = True
except ImportError:                                   # pragma: no cover
    _SKL_SURROGATE = False


@dataclass
class SurrogateReport:
    n_train:        int
    n_test:         int
    mae:            Dict[str, float]              # per target
    sim_seconds:    float                         # mean wall-time of ONE full sim
    surrogate_seconds: float                      # mean wall-time of ONE prediction
    speedup:        float
    backend:        str


class SurrogateModel:
    """Predicts simulated KPIs (SLA, abandonment, cost) from a staffing plan.

    Features: per-skill peak headcount, total headcount, arrival rate/min.
    Backed by gradient-boosted trees when sklearn is present; otherwise a
    closed-form ridge regression in numpy (so it always works)."""

    TARGETS = ("sla", "abandonment", "cost")

    def __init__(self, skills: List[str]) -> None:
        self.skills = sorted(skills)
        self._X: List[List[float]] = []
        self._Y: Dict[str, List[float]] = {t: [] for t in self.TARGETS}
        self._models: Dict[str, object] = {}
        self._ridge:  Dict[str, np.ndarray] = {}
        self.fitted = False
        self.backend = "gbt" if _SKL_SURROGATE else "ridge"

    def features(self, plan: Dict[str, int], arrival_per_min: float) -> List[float]:
        return [float(plan.get(sk, 0)) for sk in self.skills] + \
               [float(sum(plan.values())), float(arrival_per_min)]

    def observe(self, plan: Dict[str, int], arrival_per_min: float,
                kpis: Dict[str, float]) -> None:
        self._X.append(self.features(plan, arrival_per_min))
        for t in self.TARGETS:
            self._Y[t].append(float(kpis.get(t, 0.0)))

    def fit(self) -> bool:
        if len(self._X) < 5:
            return False
        X = np.asarray(self._X, dtype=float)
        for t in self.TARGETS:
            y = np.asarray(self._Y[t], dtype=float)
            if _SKL_SURROGATE:
                m = GradientBoostingRegressor(n_estimators=120, max_depth=3,
                                              learning_rate=0.08, random_state=42)
                m.fit(X, y)
                self._models[t] = m
            else:
                # ridge: w = (XᵀX + λI)⁻¹ Xᵀy  on an intercept-augmented design
                Xa = np.hstack([X, np.ones((len(X), 1))])
                lam = 1.0
                A = Xa.T @ Xa + lam * np.eye(Xa.shape[1])
                self._ridge[t] = np.linalg.solve(A, Xa.T @ y)
        self.fitted = True
        return True

    def predict(self, plan: Dict[str, int], arrival_per_min: float) -> Dict[str, float]:
        f = np.asarray([self.features(plan, arrival_per_min)], dtype=float)
        out: Dict[str, float] = {}
        for t in self.TARGETS:
            if _SKL_SURROGATE and t in self._models:
                out[t] = float(self._models[t].predict(f)[0])
            elif t in self._ridge:
                fa = np.hstack([f, np.ones((1, 1))])
                out[t] = float(fa @ self._ridge[t])
            else:
                out[t] = 0.0
        return out


class SurrogateWarmStart:
    """Builds a surrogate from historical full-sim outcomes, measures its
    accuracy on held-out plans, and reports the compute saving from screening
    candidate plans with the surrogate instead of simulating each one."""

    def __init__(self, base_sim_cfg: SimulationConfig, opt_cfg: OptimizationConfig,
                 cost_cfg=None, beh_cfg=None,
                 weights: Optional[RouterScoreWeights] = None) -> None:
        self._base = base_sim_cfg
        self._opt  = opt_cfg
        self._evaluator = SimulationEvaluator(base_sim_cfg, opt_cfg, cost_cfg, beh_cfg, weights)
        self._skills = list(base_sim_cfg.agents_per_skill.keys())
        self.model = SurrogateModel(self._skills)

    def _kpi_of(self, plan: Dict[str, int]) -> Tuple[Dict[str, float], float]:
        """Full-sim a plan; return (kpi dict, wall seconds)."""
        res = OptimizationResult(
            agents_per_skill=dict(plan),
            shift_plan={sh: dict(plan) for sh, _, _ in SHIFT_WINDOWS},
            band_coverage={}, analytical_sla={}, analytical_target=self._opt.sla_target,
            total_staffing_cost=0.0, status="surrogate_sample")
        t0  = time.perf_counter()
        ev  = self._evaluator.evaluate(res)
        dt  = time.perf_counter() - t0
        return ({"sla": ev.sla, "abandonment": ev.abandonment_rate,
                 "cost": ev.total_cost}, dt)

    def _random_plans(self, center: Dict[str, int], n: int,
                      rng: "np.random.RandomState", spread: int = 3) -> List[Dict[str, int]]:
        plans = []
        for _ in range(n):
            plans.append({sk: int(max(self._opt.min_agents_per_skill,
                                       min(self._opt.max_agents_per_skill,
                                           center.get(sk, 4) + rng.randint(-spread, spread + 1))))
                          for sk in self._skills})
        return plans

    def build_and_validate(self, center_plan: Dict[str, int],
                           n_train: int = 14, n_test: int = 6,
                           seed: int = 42) -> SurrogateReport:
        rng = np.random.RandomState(seed)
        arr = self._base.arrival_rate_per_minute

        train_plans = self._random_plans(center_plan, n_train, rng)
        sim_times: List[float] = []
        for p in train_plans:
            kpis, dt = self._kpi_of(p)
            sim_times.append(dt)
            self.model.observe(p, arr, kpis)
        self.model.fit()

        # held-out accuracy + timing
        test_plans = self._random_plans(center_plan, n_test, rng)
        preds: Dict[str, List[float]] = {t: [] for t in SurrogateModel.TARGETS}
        actuals: Dict[str, List[float]] = {t: [] for t in SurrogateModel.TARGETS}
        sur_times: List[float] = []
        for p in test_plans:
            t0 = time.perf_counter()
            pr = self.model.predict(p, arr)
            sur_times.append(time.perf_counter() - t0)
            ak, dt = self._kpi_of(p)
            sim_times.append(dt)
            for t in SurrogateModel.TARGETS:
                preds[t].append(pr[t]); actuals[t].append(ak[t])

        mae = {}
        for t in SurrogateModel.TARGETS:
            if _SKL_SURROGATE:
                mae[t] = float(mean_absolute_error(actuals[t], preds[t]))
            else:
                mae[t] = float(np.mean(np.abs(np.array(actuals[t]) - np.array(preds[t]))))
        mean_sim = float(np.mean(sim_times)) if sim_times else 0.0
        mean_sur = float(np.mean(sur_times)) if sur_times else 1e-9
        return SurrogateReport(
            n_train=n_train, n_test=n_test, mae=mae,
            sim_seconds=mean_sim, surrogate_seconds=mean_sur,
            speedup=(mean_sim / mean_sur) if mean_sur > 0 else float("inf"),
            backend=self.model.backend)

    @staticmethod
    def print_report(rep: SurrogateReport) -> None:
        w = 70
        print(f"\n{'=' * w}\n  SURROGATE MODEL WARM-UP  (optimiser acceleration)\n{'=' * w}")
        print(f"  Backend            : {rep.backend}")
        print(f"  Training samples   : {rep.n_train} full sims")
        print(f"  Held-out test      : {rep.n_test} plans")
        print(f"\n  Prediction accuracy (mean absolute error vs full sim):")
        print(f"    SLA           : {rep.mae['sla']:.3f}  ({rep.mae['sla']*100:.1f} pts)")
        print(f"    Abandonment   : {rep.mae['abandonment']:.3f}  ({rep.mae['abandonment']*100:.1f} pts)")
        print(f"    Cost          : £{rep.mae['cost']:,.2f}")
        print(f"\n  Compute:")
        print(f"    Mean full sim      : {rep.sim_seconds*1000:.1f} ms")
        print(f"    Mean surrogate pred: {rep.surrogate_seconds*1e6:.1f} us")
        print(f"    Speed-up           : {rep.speedup:,.0f}x per screened candidate")
        print(f"\n  Use: screen candidate plans with the surrogate; reserve full")
        print(f"  simulation for final validation and near-threshold cases.")
        print(f"{'=' * w}\n")
