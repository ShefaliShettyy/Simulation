"""
disruption_injector.py — "What-If" stress-testing layer for the call centre
===========================================================================

Bolts a controllable DISRUPTION onto any of the existing SimPy engines
(SimulationEngine, RealisticsAwareEngine, CostAwareEngine, MLSimulationEngine,
HierarchicalOverflowEngine) *without editing their internals*, and isolates
KPIs into pre / impact / recovery windows so degradation and recovery are
quantifiable side-by-side.

Two orthogonal shocks, each independently toggleable (and combinable):

  - Demand shock  — multiply the live arrival rate (lambda) by a factor over a
                    fixed window (marketing surge / storm spike). Implemented as
                    a time-keyed multiplier the arrival loop reads on every draw,
                    so it works whether the engine uses a static rate or the ML
                    forecaster's rate_fn.

  - Supply shock  — seize N agents of one skill into a custom state
                    DISRUPTED_UNAVAILABLE for a window (outage / flu outbreak).
                    Capacity is removed *genuinely*: a request at priority -10
                    wins a PriorityResource server unit ahead of every waiting
                    call (call priorities are 0/1/2), so the freed slot cannot be
                    re-used while the agent is out.

Design properties
-----------------
* The Stage-1 CP-SAT optimiser stays BLIND. Nothing here feeds the solver, so a
  "perfect" analytical roster can be shown failing under volatility. The injector
  is a pure simulation-time overlay.
* The LinUCB bandit degrades gracefully. Seized agents stop appearing in the
  free-agent candidate lists the router scores (they are flagged off-floor before
  any post-seizure match), and the bandit allocates arms lazily — so a sudden
  capacity drop can never raise an index/key error. If a skill's candidates drop
  to zero the host engine already handles `agent is None` / empty-candidate paths.
* Side-effect-free policy object: `DisruptionInjector` owns only schedule maths
  and a pull-ledger; it touches the engine exclusively through public/handle
  points (`env`, `agents`, `skill_resources`, `_pick_free_agent`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# Custom agent state requested by the spec (string marker so it survives on the
# stock Agent dataclass without a schema change; also used for reporting).
DISRUPTED_UNAVAILABLE = "DISRUPTED_UNAVAILABLE"
AVAILABLE             = "AVAILABLE"


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class DemandShock:
    """Scenario A — the marketing/storm spike."""
    enabled:         bool  = False
    start_minute:    float = 300.0
    end_minute:      float = 480.0
    rate_multiplier: float = 1.5          # +50%
    label:           str   = "demand_spike"


@dataclass
class SupplyShock:
    """Scenario B — the outage/flu outbreak."""
    enabled:        bool          = False
    start_minute:   float         = 300.0
    end_minute:     Optional[float] = None   # None -> until end of horizon
    skill:          str           = "billing"
    n_agents:       int           = 3
    paid_while_out: bool          = True     # are seized agents still on the wage bill?
    label:          str           = "agent_outage"


@dataclass
class DisruptionConfig:
    demand: DemandShock = field(default_factory=DemandShock)
    supply: SupplyShock = field(default_factory=SupplyShock)
    # Priority used to seize a server unit. Must beat every call priority (0/1/2)
    # and the break priority (100); -10 does both.
    seize_priority: int = -10

    def describe(self) -> str:
        bits = []
        d, s = self.demand, self.supply
        if d.enabled:
            bits.append(f"Demand x{d.rate_multiplier:.2f} [{d.start_minute:.0f}-{d.end_minute:.0f}m]")
        if s.enabled:
            end = "EOH" if s.end_minute is None else f"{s.end_minute:.0f}m"
            bits.append(f"Supply pull {s.n_agents}x {s.skill} [{s.start_minute:.0f}-{end}]")
        return "  +  ".join(bits) if bits else "none (control run)"


# ===========================================================================
# Analysis windows
# ===========================================================================

@dataclass(frozen=True)
class AnalysisWindow:
    name:  str
    start: float
    end:   float

    def contains(self, t: float) -> bool:
        return self.start <= t < self.end

    @property
    def minutes(self) -> float:
        return max(0.0, self.end - self.start)


def default_windows(horizon: float = 720.0,
                    disruption_start: float = 300.0,
                    impact_end: float = 480.0,
                    warmup: float = 60.0) -> List[AnalysisWindow]:
    """Pre (warmup->shock) | Impact (shock->impact_end) | Recovery (impact_end->horizon)."""
    return [
        AnalysisWindow("Pre-Disruption",    warmup,          disruption_start),
        AnalysisWindow("Disruption Impact", disruption_start, impact_end),
        AnalysisWindow("Recovery",          impact_end,       horizon),
    ]


# ===========================================================================
# The injector
# ===========================================================================

class DisruptionInjector:
    """Schedules demand/supply shocks on an engine's SimPy environment and keeps
    a ledger of which agents were pulled and when (for cost / reporting)."""

    def __init__(self, config: DisruptionConfig, horizon: float) -> None:
        self.cfg     = config
        self.horizon = float(horizon)
        self.engine  = None
        self._pull_log: List[Dict[str, float]] = []   # {agent_id, skill, start, end}
        self._timeline: List[Tuple[float, str]] = []   # (minute, human-readable event)

    # -- demand-side hook (read by the arrival loop, see DisruptionArrivalMixin)
    def arrival_multiplier(self, now: float) -> float:
        d = self.cfg.demand
        if d.enabled and d.start_minute <= now < d.end_minute:
            return max(0.0, d.rate_multiplier)
        return 1.0

    # -- installation ------------------------------------------------------
    def install(self, engine) -> "DisruptionInjector":
        """Attach to a (already constructed) engine and schedule its shocks.
        Call this AFTER engine construction and BEFORE engine.run()."""
        self.engine = engine
        # tag every agent so reporting/inspection always has a state to read
        for a in engine.agents:
            if not hasattr(a, "disruption_state"):
                a.disruption_state = AVAILABLE
        # let the arrival mixin find us
        engine.disruption = self
        self._schedule(engine)
        return self

    def _schedule(self, engine) -> None:
        d, s = self.cfg.demand, self.cfg.supply
        if d.enabled:
            self._timeline.append((d.start_minute, f"DEMAND  lambda x{d.rate_multiplier:.2f} ON"))
            self._timeline.append((d.end_minute,   "DEMAND  lambda restored"))
        if s.enabled:
            end = self.horizon if s.end_minute is None else float(s.end_minute)
            for _ in range(max(0, s.n_agents)):
                engine.env.process(self._seize_agent(engine, s.skill, s.start_minute, end))
            self._timeline.append((s.start_minute, f"SUPPLY  pull {s.n_agents}x {s.skill} -> {DISRUPTED_UNAVAILABLE}"))
            self._timeline.append((end,            f"SUPPLY  restore {s.skill}"))
        self._timeline.sort(key=lambda x: x[0])

    def _seize_agent(self, engine, skill: str, start: float, end: float):
        """One agent of `skill` is removed from the floor over [start, end).

        Mirrors the engine's own `_absence_process`: grab a server unit at a
        priority that beats waiting calls (so capacity genuinely drops), then
        flag a free Agent so the bandit/heuristic router stops selecting it.
        """
        yield engine.env.timeout(max(0.0, start - engine.env.now))
        resource = engine.skill_resources.get(skill)
        if resource is None:
            return
        req = resource.request(priority=self.cfg.seize_priority)
        yield req                                   # hold a server unit for the outage
        agent = engine._pick_free_agent(skill)      # may be None if none idle this instant
        if agent is not None:
            agent.on_break = True                   # the generic off-floor flag the engine honours
            agent.disruption_state = DISRUPTED_UNAVAILABLE
            self._pull_log.append({"agent_id": agent.agent_id, "skill": skill,
                                   "start": float(engine.env.now), "end": float(end)})
        yield engine.env.timeout(max(0.0, end - engine.env.now))
        resource.release(req)
        if agent is not None:
            agent.on_break = False
            agent.disruption_state = AVAILABLE

    # -- ledger helpers ----------------------------------------------------
    def disrupted_agent_minutes(self, window: AnalysisWindow) -> float:
        total = 0.0
        for p in self._pull_log:
            lo, hi = max(p["start"], window.start), min(p["end"], window.end)
            if hi > lo:
                total += (hi - lo)
        return total

    @property
    def timeline(self) -> List[Tuple[float, str]]:
        return list(self._timeline)

    @property
    def pulled_agents(self) -> List[str]:
        return [p["agent_id"] for p in self._pull_log]


# ===========================================================================
# Arrival mixin + engine factory (the demand-shock plumbing)
# ===========================================================================

class DisruptionArrivalMixin:
    """Re-implements the arrival loop to multiply the live rate by the active
    disruption multiplier. Faithful to both engine families:

      * base/realism/cost engines: rate comes from config.arrival_rate_per_minute
      * ML engine: rate comes from self._rate_fn(now) (forecaster)

    so the spike rides on top of whatever the host engine would have produced.
    """

    def _arrival_process(self):
        while True:
            inj  = getattr(self, "disruption", None)
            base = (self._rate_fn(self.env.now) if getattr(self, "_rate_fn", None)
                    else self.config.arrival_rate_per_minute)
            mult = inj.arrival_multiplier(self.env.now) if inj is not None else 1.0
            yield self.env.timeout(self._rng.expovariate(max(1e-6, base * mult)))
            self.env.process(self._handle_call(self._generate_call()))


def make_disruption_engine(engine_cls):
    """Synthesise a disruption-aware subclass of any engine class.

        StressEngine = make_disruption_engine(MLSimulationEngine)
        engine = StressEngine(cfg, registry, ...)   # same ctor as the base class
    """
    return type(f"Disruptive_{engine_cls.__name__}",
                (DisruptionArrivalMixin, engine_cls), {})


# ===========================================================================
# Windowed KPI analysis
# ===========================================================================

@dataclass
class WindowKPIs:
    name:             str
    start:            float
    end:              float
    offered:          int
    handled:          int
    abandoned:        int
    sla:              float
    abandonment_rate: float
    asa:              float
    aht:              float
    csat:             float
    staffing_cost:    float
    cost_per_call:    float


class WindowedKPIAnalyzer:
    """Post-run, pure-read analysis. Buckets handled calls and abandonments by
    the window their ARRIVAL falls in, then computes SLA/ASA/AHT/CSAT/abandonment
    and a staffing-cost proxy per window.

    Cost model: the roster is a committed cost (agents are paid for the horizon),
    so staffing_cost per window = scheduled-agent-minutes x blended wage x
    overhead. cost_per_call therefore SPIKES under disruption because throughput
    falls while the wage bill does not. Set supply.paid_while_out=False to credit
    back seized agent-minutes instead.
    """

    SLA_THRESHOLD = 1.0   # matches KPIEngine.SLA_THRESHOLD_MINUTES

    def __init__(self, engine, windows: List[AnalysisWindow],
                 injector: Optional[DisruptionInjector] = None,
                 wages: Optional[Dict[str, float]] = None, overhead: float = 1.30) -> None:
        self.engine   = engine
        self.windows  = windows
        self.injector = injector
        self.wages    = wages or {"billing": 18.0, "technical": 22.0, "general": 16.0}
        self.overhead = overhead

    def _blended_hourly(self) -> float:
        per = self.engine.config.agents_per_skill
        tot = sum(per.values()) or 1
        return sum(self.wages.get(sk, 18.0) * n for sk, n in per.items()) / tot

    def analyze(self) -> List[WindowKPIs]:
        recs = self.engine.kpi._records
        abns = self.engine.kpi._abandonments
        n_agents = len(self.engine.agents)
        blended  = self._blended_hourly()
        out: List[WindowKPIs] = []
        for w in self.windows:
            r = [x for x in recs if w.contains(x.arrival_time)]
            a = [x for x in abns if w.contains(x["arrival_time"])]
            handled, abandoned = len(r), len(a)
            offered = handled + abandoned
            if handled:
                sla  = sum(1 for x in r if x.wait_minutes <= self.SLA_THRESHOLD) / handled
                asa  = sum(x.wait_minutes for x in r) / handled
                aht  = sum(x.handle_minutes for x in r) / handled
                csat = sum(x.csat_raw for x in r) / handled
            else:
                sla = asa = aht = csat = 0.0
            abn_rate = abandoned / offered if offered else 0.0

            paid_agent_min = n_agents * w.minutes
            if self.injector is not None and not self.injector.cfg.supply.paid_while_out:
                paid_agent_min -= self.injector.disrupted_agent_minutes(w)
            cost = (paid_agent_min / 60.0) * blended * self.overhead
            cpc  = cost / handled if handled else 0.0

            out.append(WindowKPIs(w.name, w.start, w.end, offered, handled, abandoned,
                                  sla, abn_rate, asa, aht, csat, cost, cpc))
        return out


# ===========================================================================
# Reporting (matches the pipeline's banner / aligned-column house style)
# ===========================================================================

class DisruptionReport:
    def __init__(self, window_kpis: List[WindowKPIs], injector: DisruptionInjector,
                 config: DisruptionConfig, horizon: float = 720.0) -> None:
        self.k       = window_kpis
        self.inj     = injector
        self.cfg     = config
        self.horizon = horizon

    @staticmethod
    def _delta(cur: float, base: float, pct: bool = False, money: bool = False,
               count: bool = False, higher_is_better: bool = True, tag: bool = True) -> str:
        d = cur - base
        if abs(d) < (0.0005 if pct else 5e-3):
            return "  (=)"
        suffix = ""
        if tag:
            good   = (d > 0) == higher_is_better
            suffix = f" {'up' if d > 0 else 'dn'} {'OK' if good else '!!'}"
        if count:
            return f" ({int(round(d)):+d}{suffix})"
        if pct:
            return f" ({d*100:+.1f}pt{suffix})"
        if money:
            return f" ({d:+,.2f}{suffix})"
        return f" ({d:+.2f}{suffix})"

    def print_report(self) -> None:
        k = self.k
        w = 92
        bar = "=" * w
        print(f"\n{bar}")
        print(f"  DISRUPTION INJECTOR  --  WHAT-IF STRESS TEST")
        print(f"  Scenario : {self.cfg.describe()}")
        print(f"  CP-SAT roster : BLIND (no replanning)      Horizon : {self.horizon:.0f}m")
        print(f"{bar}")

        if self.inj.timeline:
            print("  Injected timeline:")
            for t, ev in self.inj.timeline:
                print(f"    t={t:>5.0f}m   {ev}")
            if self.inj.pulled_agents:
                print(f"    seized agents : {', '.join(self.inj.pulled_agents)}")
            print(f"  {'-' * (w - 4)}")

        # side-by-side: KPI rows, window columns
        names = [win.name for win in k]
        hdr = f"  {'KPI':<20}" + "".join(f"{n:>24}" for n in names)
        print(hdr)
        spans = "  " + " " * 20 + "".join(f"{f'[{win.start:.0f}-{win.end:.0f}m]':>24}" for win in k)
        print(spans)
        print(f"  {'-' * (w - 4)}")

        base = k[0]   # pre-disruption baseline for deltas

        def row(label, fmt, getter, **dkw):
            cells = f"  {label:<20}"
            for i, win in enumerate(k):
                val = getter(win)
                s = fmt(val)
                if i == 0:
                    cells += f"{s:>24}"
                else:
                    cells += f"{(s + self._delta(val, getter(base), **dkw)):>24}"
            print(cells)

        row("Calls offered", lambda v: f"{v:,}", lambda x: x.offered, count=True, tag=False)
        row("Calls handled", lambda v: f"{v:,}", lambda x: x.handled, count=True, tag=False)
        row("Abandoned",     lambda v: f"{v:,}", lambda x: x.abandoned, count=True, tag=False)
        row("SLA (<=1m)",    lambda v: f"{v:.1%}", lambda x: x.sla,
            pct=True, higher_is_better=True)
        row("Abandonment",   lambda v: f"{v:.1%}", lambda x: x.abandonment_rate,
            pct=True, higher_is_better=False)
        row("ASA (min)",     lambda v: f"{v:.2f}", lambda x: x.asa,
            higher_is_better=False)
        row("AHT (min)",     lambda v: f"{v:.2f}", lambda x: x.aht,
            higher_is_better=False)
        row("CSAT (1-5)",    lambda v: f"{v:.2f}", lambda x: x.csat,
            higher_is_better=True)
        row("Cost/handled",  lambda v: f"GBP{v:,.2f}", lambda x: x.cost_per_call,
            money=True, higher_is_better=False)
        print(f"  {'-' * (w - 4)}")

        # narrative verdict
        impact = k[1] if len(k) > 1 else None
        recov  = k[2] if len(k) > 2 else None
        if impact is not None:
            d_sla = (impact.sla - base.sla) * 100
            d_abn = (impact.abandonment_rate - base.abandonment_rate) * 100
            print(f"  Verdict:")
            print(f"    Impact   : SLA {d_sla:+.1f}pt vs pre, abandonment {d_abn:+.1f}pt, "
                  f"ASA {impact.asa - base.asa:+.2f}m")
            if recov is not None:
                sla_gap = (base.sla - recov.sla) * 100
                state = ("fully recovered" if sla_gap <= 1.0
                         else f"still {sla_gap:.1f}pt below baseline at horizon end")
                print(f"    Recovery : SLA {(recov.sla - impact.sla)*100:+.1f}pt vs impact -> {state}")
        print(f"{bar}\n")