"""
hierarchical_overflow.py  —  Hierarchical Skill-Tier Overflow Engine
====================================================================

Problem
-------
The LinUCB router is a quality gatekeeper: chasing CSAT/FCR reward it lets calls
sit in queue waiting for a strong primary-agent match. Against Weibull-distributed
patience that drives abandonment up sharply. This module adds an *operational
escape valve*: the longer a call waits, the wider the pool of agents that may
serve it, trading service quality for queue protection.

Design
------
Two cooperating pieces, integrated with (not bolted onto) the existing engine:

1. `HierarchicalRouter` — a STANDALONE, side-effect-free policy object that owns
   the threshold math, tier classification, the degradation multipliers, the
   FCR bounding box, and the per-tier KPI accumulators. It has no SimPy
   dependency, so it is unit-testable and reusable.

2. `HierarchicalOverflowEngine(MLSimulationEngine)` — overrides only the call
   lifecycle to implement wait-time-driven pool escalation on top of the
   existing per-skill `simpy.PriorityResource` pools, the LinUCB bandit, the
   affinity model, fatigue, and decision logging. Everything else (KPIs, cost,
   forecaster arrivals) is inherited unchanged.

Tier model (wait w in minutes, tau = overflow threshold)
--------------------------------------------------------
    Tier 0  w <= tau          primary specialists only      AHT x1.00  FCR x1.00
    Tier 1  tau < w <= 2*tau   + cross-trained backups       AHT x1.25  FCR x0.85
    Tier 2  w > 2*tau          + any available agent          AHT x1.50  FCR x0.70

    penalised_FCR = max(fcr_floor, base_FCR * fcr_mult)      (fcr_floor = 0.10)

The tier is a function of the *realised* wait at the moment of match, so a call
served late is penalised even if a primary agent happened to free up — long waits
degrade the experience regardless of who finally answers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from simpy.events import AnyOf

from core_simulation import Agent, Call, RouterScoreWeights, SimulationConfig, _CallRecord
from ml_system import MLRegistry, MLSimulationEngine, build_registry

try:
    import pandas as pd
    _PANDAS = True
except ImportError:                                   # pragma: no cover
    _PANDAS = False


# ===========================================================================
# Tier configuration
# ===========================================================================

@dataclass
class TierConfig:
    """All tunables for the overflow policy. Defaults match the spec."""
    tau_overflow_minutes: float = 0.75               # 45 seconds
    fcr_floor:            float = 0.10               # FCR never below 10%
    # (aht_multiplier, fcr_multiplier) per tier
    tier1_aht_mult: float = 1.25
    tier1_fcr_mult: float = 0.85
    tier2_aht_mult: float = 1.50
    tier2_fcr_mult: float = 0.70
    enabled:        bool  = True

    def multipliers(self, tier: int) -> Tuple[float, float]:
        if tier <= 0:
            return 1.0, 1.0
        if tier == 1:
            return self.tier1_aht_mult, self.tier1_fcr_mult
        return self.tier2_aht_mult, self.tier2_fcr_mult


# ===========================================================================
# Standalone HierarchicalRouter
# ===========================================================================

@dataclass
class _TierStat:
    calls:        int   = 0
    wait_sum:     float = 0.0
    aht_sum:      float = 0.0     # penalised handle minutes actually incurred
    base_aht_sum: float = 0.0     # un-penalised handle minutes (for impact)
    resolved:     int   = 0


class HierarchicalRouter:
    """Threshold logic + degradation math + per-tier KPI tracking.

    Pure policy object: no SimPy, no global state. The engine queries it for
    (a) which agent pools are eligible at the current wait, (b) the tier of a
    realised wait, (c) the penalised AHT / FCR for that tier, and reports each
    completed call back via `record`.
    """

    def __init__(self, cross_skill_map: Dict[str, Set[str]],
                 all_skills: Sequence[str],
                 config: Optional[TierConfig] = None) -> None:
        self.cfg          = config or TierConfig()
        self._cross       = {k: set(v) for k, v in cross_skill_map.items()}
        self._all_skills  = list(all_skills)
        self._stats: Dict[int, _TierStat] = {0: _TierStat(), 1: _TierStat(), 2: _TierStat()}
        self.abandoned    = 0

    # -- thresholds --------------------------------------------------------

    @property
    def tau(self) -> float:
        return self.cfg.tau_overflow_minutes

    def tier_for_wait(self, wait_minutes: float) -> int:
        """Map a realised wait to its service tier (0/1/2)."""
        if wait_minutes <= self.tau:
            return 0
        if wait_minutes <= 2.0 * self.tau:
            return 1
        return 2

    def eligible_skills(self, call_skill: str, wait_minutes: float) -> Set[str]:
        """Agent-pool skills a call may be served from given its current wait.

        Tier 0: primary skill only.
        Tier 1: primary + cross-trained backups.
        Tier 2: every skill (generalist fallback).
        """
        tier = self.tier_for_wait(wait_minutes)
        if tier == 0:
            return {call_skill}
        if tier == 1:
            return {call_skill} | self._cross.get(call_skill, set())
        return set(self._all_skills)

    # -- degradation math --------------------------------------------------

    def degrade(self, base_aht: float, base_fcr_prob: float, tier: int) -> Tuple[float, float]:
        """Apply tier penalties. AHT is inflated; FCR probability is scaled down
        and clamped to the logical floor so it can never collapse to zero."""
        aht_mult, fcr_mult = self.cfg.multipliers(tier)
        pen_aht = base_aht * aht_mult
        pen_fcr = max(self.cfg.fcr_floor, base_fcr_prob * fcr_mult)
        return pen_aht, pen_fcr

    # -- metrics -----------------------------------------------------------

    def record(self, tier: int, wait: float, penalised_aht: float,
               base_aht: float, resolved: bool) -> None:
        s = self._stats[tier]
        s.calls        += 1
        s.wait_sum     += wait
        s.aht_sum      += penalised_aht
        s.base_aht_sum += base_aht
        s.resolved     += int(resolved)

    def record_abandonment(self) -> None:
        self.abandoned += 1

    # -- reporting ---------------------------------------------------------

    def kpi_rows(self) -> List[Dict[str, float]]:
        total_served = sum(s.calls for s in self._stats.values()) or 1
        rows = []
        names = {0: "Tier 0 (primary)", 1: "Tier 1 (cross-trained)", 2: "Tier 2 (generalist)"}
        for tier in (0, 1, 2):
            s = self._stats[tier]
            c = s.calls or 1
            rows.append({
                "tier":         tier,
                "label":        names[tier],
                "calls":        s.calls,
                "share":        s.calls / total_served,
                "avg_wait_min": s.wait_sum / c,
                "avg_aht_min":  s.aht_sum / c,
                "aht_penalty":  (s.aht_sum / s.base_aht_sum - 1.0) if s.base_aht_sum else 0.0,
                "fcr":          s.resolved / c,
            })
        return rows

    def kpi_frame(self):
        """Return a pandas DataFrame of the tier KPIs (if pandas is available)."""
        if not _PANDAS:
            raise RuntimeError("pandas not installed; use kpi_rows()/print_kpi_log()")
        return pd.DataFrame(self.kpi_rows())

    def print_kpi_log(self) -> None:
        rows = self.kpi_rows()
        served = sum(r["calls"] for r in rows)
        total  = served + self.abandoned
        w = 92
        print(f"\n{'=' * w}")
        print(f"  HIERARCHICAL OVERFLOW — TIER DISTRIBUTION & SERVICE DEGRADATION")
        print(f"  tau_overflow = {self.tau:.2f} min ({self.tau*60:.0f}s)   "
              f"FCR floor = {self.cfg.fcr_floor:.0%}")
        print(f"{'=' * w}")
        print(f"  {'Tier':<26}{'Calls':>8}{'Share':>9}{'AvgWait':>10}"
              f"{'AvgAHT':>9}{'AHTpen':>9}{'FCR':>8}")
        print(f"  {'-' * (w - 4)}")
        for r in rows:
            print(f"  {r['label']:<26}{r['calls']:>8}{r['share']:>8.1%}"
                  f"{r['avg_wait_min']:>9.2f}m{r['avg_aht_min']:>8.2f}m"
                  f"{r['aht_penalty']:>8.1%}{r['fcr']:>8.1%}")
        print(f"  {'-' * (w - 4)}")
        overflow = sum(r["calls"] for r in rows if r["tier"] > 0)
        print(f"  Served={served}   Overflowed (Tier1+2)={overflow} "
              f"({overflow/served:.1%} of served)   Abandoned={self.abandoned} "
              f"({self.abandoned/total:.1%} of offered)" if served else "  no calls served")
        print(f"{'=' * w}\n")


# ===========================================================================
# Cross-training map helper
# ===========================================================================

def build_cross_skill_map(agents: Sequence[Agent]) -> Dict[str, Set[str]]:
    """For each skill, the set of *other* skills whose agents are cross-trained
    to back it up (i.e. list it as a secondary competency)."""
    cross: Dict[str, Set[str]] = {}
    for a in agents:
        for sec in a.secondary_skills:
            cross.setdefault(sec, set()).add(a.primary_skill)
    # ensure every primary skill has an entry
    for a in agents:
        cross.setdefault(a.primary_skill, set())
    return cross


# ===========================================================================
# Integrated SimPy engine
# ===========================================================================

class HierarchicalOverflowEngine(MLSimulationEngine):
    """MLSimulationEngine + wait-time pool escalation.

    Only `_handle_call` changes. A call starts queued for its primary skill;
    when its wait crosses tau it *additionally* joins the queues of cross-trained
    pools, and at 2*tau it joins all remaining pools. The first pool to free a
    server wins; the realised wait sets the tier and its AHT/FCR penalties.
    """

    def __init__(self, config: SimulationConfig, registry: MLRegistry,
                 weights: Optional[RouterScoreWeights] = None,
                 tier_cfg: Optional[TierConfig] = None,
                 use_forecaster: bool = True,
                 stochastic_patience: bool = True) -> None:
        super().__init__(config, registry, weights,
                         use_forecaster=use_forecaster,
                         stochastic_patience=stochastic_patience)
        cross = build_cross_skill_map(self.agents)
        self.hrouter = HierarchicalRouter(cross, list(self.skill_resources.keys()), tier_cfg)

    # -- escalating, abandonment-protecting call lifecycle -----------------

    def _handle_call(self, call: Call):
        cfg_enabled = self.hrouter.cfg.enabled
        start    = self.env.now
        skill    = call.skill
        tau      = self.hrouter.tau
        patience = (self._registry.abandon.sample_patience(call.customer_type, self._rng)
                    if self._stochastic_patience else self._MAX_WAIT_PATIENCE)
        q_at_arrival = len(self.skill_resources[skill].queue)

        t_abandon = start + patience
        t1, t2    = start + tau, start + 2.0 * tau
        cross_skills = sorted(self.hrouter.eligible_skills(skill, 1.5 * tau) - {skill})
        other_skills = [s for s in self.skill_resources
                        if s != skill and s not in cross_skills]

        # one outstanding request per joined pool; primary is joined immediately
        reqs: Dict[str, object] = {skill: self.skill_resources[skill].request(priority=call.priority)}
        opened_cross = opened_all = (not cfg_enabled)   # if disabled, never escalate

        granted_skill: Optional[str] = None
        while True:
            now = self.env.now
            if now >= t_abandon - 1e-9:
                break
            if cfg_enabled and not opened_cross and now >= t1 - 1e-9:
                for sk in cross_skills:
                    reqs.setdefault(sk, self.skill_resources[sk].request(priority=call.priority))
                opened_cross = True
            if cfg_enabled and not opened_all and now >= t2 - 1e-9:
                for sk in other_skills:
                    reqs.setdefault(sk, self.skill_resources[sk].request(priority=call.priority))
                opened_all = True

            upcoming = [t for t in (t1, t2, t_abandon) if t > now + 1e-9]
            next_t   = min(upcoming) if upcoming else t_abandon
            timeout_ev = self.env.timeout(max(0.0, next_t - now))
            yield AnyOf(self.env, list(reqs.values()) + [timeout_ev])

            granted = [sk for sk, rq in reqs.items() if rq.triggered]
            if granted:
                granted_skill = self._prefer(granted, skill, cross_skills)
                break

        # ---- abandoned -----------------------------------------------------
        if granted_skill is None:
            for rq in reqs.values():
                if not rq.triggered:
                    rq.cancel()
                else:
                    pass  # safety: a stray grant would be released below
            # release any request that did get granted in the same instant
            for sk, rq in reqs.items():
                if rq.triggered:
                    self.skill_resources[sk].release(rq)
            self._registry.log.log_queue_result(
                call.customer_type, call.skill, call.is_repeat,
                patience, q_at_arrival, call.arrival_time, abandoned=True)
            self.kpi.record_abandonment(call, self.env.now)
            self.hrouter.record_abandonment()
            return

        # ---- matched: release the losing requests --------------------------
        for sk, rq in reqs.items():
            if sk == granted_skill:
                continue
            if rq.triggered:
                self.skill_resources[sk].release(rq)   # we momentarily held an extra unit
            else:
                rq.cancel()
        chosen_req = reqs[granted_skill]

        service_start = self.env.now
        wait          = service_start - start
        tier          = self.hrouter.tier_for_wait(wait)
        self._registry.log.log_queue_result(
            call.customer_type, call.skill, call.is_repeat,
            wait, q_at_arrival, call.arrival_time, abandoned=False)

        # candidate agents from the winning pool, preferring genuine competence
        candidates = self._free_agents(granted_skill)
        if granted_skill != skill:
            competent = [a for a in candidates
                         if a.primary_skill == skill or skill in a.secondary_skills]
            if competent:
                candidates = competent
        if not candidates:
            self.skill_resources[granted_skill].release(chosen_req)
            return
        agent, x_policy, base_dec = self.router.select_agent(call, candidates, self.skill_resources)
        agent.busy = True
        self.router.notify_call_started(agent.agent_id)

        # base service draw, then tier AHT penalty
        base_svc = max(0.5, self._rng.gauss(self.config.mean_service_minutes,
                                            self.config.stdev_service_minutes))
        base_acw = max(0.0, self._rng.gauss(self.config.acw_mean_minutes,
                                            self.config.acw_stdev_minutes))
        base_handle = base_svc + base_acw
        base_fcr_p  = self._resolution_prob(call, agent)
        pen_handle, pen_fcr_p = self.hrouter.degrade(base_handle, base_fcr_p, tier)

        yield self.env.timeout(pen_handle)

        resource = self.skill_resources[granted_skill]
        q_end    = len(resource.queue)
        resource.release(chosen_req)
        agent.busy = False
        service_end    = self.env.now
        handle_minutes = service_end - service_start

        resolved = self._rng.random() < pen_fcr_p
        csat_raw = self._sample_csat(call.customer_type, agent.csat_bias, self._rng)

        self._registry.fatigue.accumulate(agent.agent_id, handle_minutes)
        burnout = self._registry.fatigue.burnout_risk(agent.agent_id)
        reward  = self.router.learn(agent.agent_id, x_policy, csat_raw,
                                    resolved, handle_minutes, burnout, wait=wait)
        self._registry.affinity.update(agent.agent_id, call.skill, csat_raw,
                                       handle_minutes, resolved, call.customer_type, q_end)
        self._registry.log.log_outcome(base_dec, csat_raw, resolved)
        self.router.notify_call_ended(agent.agent_id, csat_raw, handle_minutes,
                                      call.is_repeat, skill=call.skill,
                                      tier=call.customer_type, queue_depth=q_end)
        self.hrouter.record(tier, wait, handle_minutes, base_handle, resolved)

        self.kpi.record_call(_CallRecord(
            call_id=call.call_id, skill=call.skill, customer_type=call.customer_type,
            is_repeat=call.is_repeat, arrival_time=call.arrival_time,
            service_start=service_start, service_end=service_end,
            csat_raw=csat_raw, routing_reason=f"tier{tier}->{granted_skill}|r={reward:.2f}"))

    @staticmethod
    def _prefer(granted: List[str], primary: str, cross: Sequence[str]) -> str:
        """Pick which granted pool serves: primary first, then cross-trained,
        then anything else — so we always use the most-competent freed agent."""
        if primary in granted:
            return primary
        for sk in cross:
            if sk in granted:
                return sk
        return granted[0]


# ===========================================================================
# Integration example / demonstration
# ===========================================================================

def run_overflow_demo(arrival_rate_per_hour: float = 90.0,
                      sim_minutes: float = 720.0,
                      agents=None, seed: int = 42,
                      tier_cfg: Optional[TierConfig] = None) -> None:
    """Run the baseline ML engine and the overflow engine on the SAME config and
    seed, and show that the escape valve cuts abandonment, plus the tier KPI log."""
    agents = agents or {"billing": 6, "technical": 6, "general": 5}
    cfg = SimulationConfig(
        sim_duration_minutes=sim_minutes, arrival_rate_per_hour=arrival_rate_per_hour,
        agents_per_skill=dict(agents), random_seed=seed)

    print("=" * 92)
    print("  DEMO: ML gatekeeper (baseline)  vs  Hierarchical Overflow Engine")
    print(f"  arrivals={arrival_rate_per_hour}/hr  horizon={sim_minutes:.0f}m  staffing={agents}")
    print("=" * 92)

    base = MLSimulationEngine(cfg, build_registry(cfg), use_forecaster=False)
    base.run()
    print(f"\n  [Baseline ML]   SLA={base.kpi.sla_percentage():.1%}  "
          f"abandon={base.kpi.abandonment_rate():.1%}  "
          f"({base.kpi.total_abandonments()} dropped)  "
          f"CSAT={base.kpi.average_csat():.2f}  FCR={base.kpi.first_call_resolution():.1%}")

    eng = HierarchicalOverflowEngine(cfg, build_registry(cfg),
                                     tier_cfg=tier_cfg, use_forecaster=False)
    eng.run()
    print(f"  [Overflow ON ]  SLA={eng.kpi.sla_percentage():.1%}  "
          f"abandon={eng.kpi.abandonment_rate():.1%}  "
          f"({eng.kpi.total_abandonments()} dropped)  "
          f"CSAT={eng.kpi.average_csat():.2f}  FCR={eng.kpi.first_call_resolution():.1%}")

    drop = base.kpi.total_abandonments() - eng.kpi.total_abandonments()
    print(f"\n  -> overflow valve recovered {drop} call(s) from abandonment "
          f"({base.kpi.abandonment_rate():.1%} -> {eng.kpi.abandonment_rate():.1%})")
    eng.hrouter.print_kpi_log()


if __name__ == "__main__":
    run_overflow_demo()