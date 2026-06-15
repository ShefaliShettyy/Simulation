"""
core_simulation.py — staged copy of the user's module (unmodified).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional, Tuple

import simpy


@dataclass
class SimulationConfig:
    sim_duration_minutes:    float = 480.0
    arrival_rate_per_hour:   float = 120.0
    skill_mix: Dict[str, float] = field(
        default_factory=lambda: {"billing": 0.40, "technical": 0.35, "general": 0.25}
    )
    customer_tier_mix: Dict[str, float] = field(
        default_factory=lambda: {"vip": 0.10, "premium": 0.25, "standard": 0.65}
    )
    repeat_call_probability: float = 0.12
    mean_service_minutes:    float = 5.0
    stdev_service_minutes:   float = 2.0
    acw_mean_minutes:        float = 1.5
    acw_stdev_minutes:       float = 0.5
    agents_per_skill: Dict[str, int] = field(
        default_factory=lambda: {"billing": 4, "technical": 4, "general": 3}
    )
    overflow_threshold: int = 5
    break_schedule: List[Tuple[float, float]] = field(
        default_factory=lambda: [(120.0, 15.0), (300.0, 30.0), (420.0, 15.0)]
    )
    random_seed: Optional[int] = 42

    # --- steady-state / warm-up -------------------------------------------
    # KPIs, costs and utilisation collected during [0, warmup_minutes) are
    # discarded; official measurement runs over [warmup_minutes, sim_duration].
    warmup_minutes: float = 0.0
    warmup_log:     bool  = True      # set False to silence per-run warm-up logs

    # --- staggered breaks --------------------------------------------------
    # Each agent's break start is offset within this window (minutes) so breaks
    # of one skill pool do not all begin at the same instant. 0 = legacy (clustered).
    break_stagger_window: float = 0.0

    # --- multi-factor shrinkage (sim-side stochastic availability) ---------
    agent_no_show_prob:   float = 0.0     # P(agent absent for the whole shift)
    agent_outage_prob:    float = 0.0     # P(agent has a mid-shift outage)
    agent_outage_minutes: float = 20.0    # duration of a mid-shift outage

    @property
    def arrival_rate_per_minute(self) -> float:
        return self.arrival_rate_per_hour / 60.0

    @property
    def measurement_minutes(self) -> float:
        """Length of the official (post-warm-up) measurement window."""
        return max(1e-9, self.sim_duration_minutes - max(0.0, self.warmup_minutes))


@dataclass(frozen=True)
class Call:
    call_id:       str
    skill:         str
    customer_type: str
    is_repeat:     bool
    arrival_time:  float

    @property
    def priority(self) -> int:
        return {"vip": 0, "premium": 1, "standard": 2}.get(self.customer_type, 2)


@dataclass
class Agent:
    agent_id:         str
    primary_skill:    str
    secondary_skills: List[str] = field(default_factory=list)
    experience:       str = "mid"
    csat_bias:        float = field(default=0.0)
    on_break:         bool  = field(default=False, init=False)
    busy:             bool  = field(default=False, init=False)

    def __repr__(self) -> str:
        if self.on_break:
            status = "break"
        elif self.busy:
            status = "busy"
        else:
            status = "free"
        return (f"Agent({self.agent_id}, skill={self.primary_skill}, "
                f"exp={self.experience}, bias={self.csat_bias:+.2f}, {status})")


@dataclass
class AgentPerformanceStats:
    agent_id:        str
    ema_csat:        float = 0.5
    ema_handle_time: float = 5.0
    ema_resolution:  float = 0.85
    calls_completed: int   = 0
    total_csat:      float = 0.0
    total_handle:    float = 0.0
    active_calls:    int   = 0

    EMA_ALPHA:          ClassVar[float] = 0.25
    MIN_RELIABLE_CALLS: ClassVar[int]   = 5

    def record_call(self, csat_raw: float, handle_minutes: float, is_repeat: bool) -> None:
        csat_norm  = (csat_raw - 1.0) / 4.0
        resolution = 0.0 if is_repeat else 1.0
        if self.calls_completed == 0:
            self.ema_csat        = csat_norm
            self.ema_handle_time = handle_minutes
            self.ema_resolution  = resolution
        else:
            a = self.EMA_ALPHA
            self.ema_csat        = a * csat_norm      + (1 - a) * self.ema_csat
            self.ema_handle_time = a * handle_minutes + (1 - a) * self.ema_handle_time
            self.ema_resolution  = a * resolution     + (1 - a) * self.ema_resolution
        self.calls_completed += 1
        self.total_csat      += csat_raw
        self.total_handle    += handle_minutes

    def reliability_weight(self) -> float:
        if self.calls_completed >= self.MIN_RELIABLE_CALLS:
            return 1.0
        return 0.5 + 0.5 * (self.calls_completed / self.MIN_RELIABLE_CALLS)

    def avg_csat_raw(self) -> float:
        if self.calls_completed == 0:
            return 3.5
        return self.total_csat / self.calls_completed


class AgentPerformanceTracker:
    def __init__(self, agent_pool: List[Agent], config: SimulationConfig) -> None:
        default_handle = config.mean_service_minutes + config.acw_mean_minutes
        self._stats: Dict[str, AgentPerformanceStats] = {
            a.agent_id: AgentPerformanceStats(
                agent_id=a.agent_id, ema_csat=0.5,
                ema_handle_time=default_handle, ema_resolution=0.85,
            )
            for a in agent_pool
        }

    def get(self, agent_id: str) -> AgentPerformanceStats:
        return self._stats[agent_id]

    def record_call_start(self, agent_id: str) -> None:
        if agent_id in self._stats:
            self._stats[agent_id].active_calls += 1

    def record_call_end(self, agent_id: str, csat_raw: float, handle_minutes: float, is_repeat: bool) -> None:
        if agent_id not in self._stats:
            return
        st = self._stats[agent_id]
        st.active_calls = max(0, st.active_calls - 1)
        st.record_call(csat_raw, handle_minutes, is_repeat)

    def all_stats(self) -> List[AgentPerformanceStats]:
        return sorted(self._stats.values(), key=lambda s: s.agent_id)

    def print_summary(self) -> None:
        print("\n-- Agent Performance Tracker ---------------------------------------")
        print(f"  {'Agent':<20}  {'Calls':>6}  {'AvgCSAT':>8}  {'EMA Hdl':>8}  {'EMA Res':>8}  {'Active':>7}")
        print("  " + "-" * 68)
        for st in self.all_stats():
            print(
                f"  {st.agent_id:<20}  {st.calls_completed:>6}  "
                f"{st.avg_csat_raw():>8.3f}  {st.ema_handle_time:>7.2f}m  "
                f"{st.ema_resolution:>8.3f}  {st.active_calls:>7}"
            )
        print("-" * 72)


@dataclass
class RouterScoreWeights:
    performance: float = 0.35
    efficiency:  float = 0.20
    resolution:  float = 0.20
    workload:    float = 0.10
    wait_time:   float = 0.10
    vip_fit:     float = 0.05

    def as_vector(self) -> Tuple[float, ...]:
        return (self.performance, self.efficiency, self.resolution,
                self.workload, self.wait_time, self.vip_fit)

    def normalised(self) -> "RouterScoreWeights":
        total = sum(self.as_vector()) or 1.0
        return RouterScoreWeights(
            performance=self.performance / total, efficiency=self.efficiency / total,
            resolution=self.resolution / total, workload=self.workload / total,
            wait_time=self.wait_time / total, vip_fit=self.vip_fit / total,
        )


class Router:
    _MAX_HANDLE_MINUTES: ClassVar[float] = 15.0
    _MAX_ACTIVE_CALLS:   ClassVar[int]   = 5
    _MAX_WAIT_MINUTES:   ClassVar[float] = 10.0
    _VIP_EXPERIENCE_SCORES: ClassVar[Dict[str, float]] = {
        "senior": 1.0, "mid": 0.6, "junior": 0.2,
    }

    def __init__(self, agent_pool, config, weights=None) -> None:
        self.agent_pool = agent_pool
        self.config     = config
        self.weights    = (weights or RouterScoreWeights()).normalised()
        self.tracker    = AgentPerformanceTracker(agent_pool, config)
        self._agents_by_skill: Dict[str, List[Agent]] = {}
        for agent in agent_pool:
            self._agents_by_skill.setdefault(agent.primary_skill, []).append(agent)

    def select_resource(self, call, skill_resources):
        skill            = call.skill
        primary_resource = skill_resources[skill]
        if call.customer_type == "vip":
            best_agent, best_score = self._best_agent_for_skill(skill, call, skill_resources)
            if best_agent is not None:
                return primary_resource, skill, f"vip_direct->{best_agent.agent_id}(score={best_score:.3f})"
        q_depth = len(primary_resource.queue)
        if q_depth >= self.config.overflow_threshold:
            overflow_resource, overflow_skill = self._find_overflow_resource(skill, skill_resources)
            if overflow_resource is not None:
                return overflow_resource, overflow_skill, f"overflow(q={q_depth})->{overflow_skill}"
        best_agent, best_score = self._best_agent_for_skill(skill, call, skill_resources)
        if best_agent is not None:
            return primary_resource, skill, f"scored->{best_agent.agent_id}(score={best_score:.3f})"
        return primary_resource, skill, "fallback(pool)"

    def pick_agent(self, call, candidates, skill_resources):
        if not candidates:
            return None
        best_agent = None
        best_score = float("-inf")
        for agent in candidates:
            score = self._composite_score(agent, call, skill_resources)
            if score > best_score:
                best_score = score
                best_agent = agent
        return best_agent or candidates[0]

    def estimated_wait(self, resource):
        q = len(resource.queue)
        if q == 0:
            return 0.0
        avg_handle = self._avg_handle_time()
        return max(0.0, (q / max(resource.capacity, 1)) * avg_handle)

    def notify_call_started(self, agent_id):
        self.tracker.record_call_start(agent_id)

    def notify_call_ended(self, agent_id, csat_raw, handle_minutes, is_repeat,
                          skill="", tier="standard", queue_depth=0):
        self.tracker.record_call_end(agent_id, csat_raw, handle_minutes, is_repeat)

    def _best_agent_for_skill(self, skill, call, skill_resources):
        candidates = [a for a in self._agents_by_skill.get(skill, []) if not a.on_break]
        if not candidates:
            return None, 0.0
        best_agent = None
        best_score = -1.0
        for agent in candidates:
            score = self._composite_score(agent, call, skill_resources)
            if score > best_score:
                best_score = score
                best_agent = agent
        return best_agent, best_score

    def _composite_score(self, agent, call, skill_resources):
        signals     = self._compute_signals(agent, call, skill_resources)
        w           = self.weights
        raw_score   = sum(getattr(w, k) * v for k, v in signals.items())
        reliability = self.tracker.get(agent.agent_id).reliability_weight()
        return reliability * raw_score + (1.0 - reliability) * 0.5

    def _compute_signals(self, agent, call, skill_resources):
        st = self.tracker.get(agent.agent_id)
        performance_sig = float(max(0.0, min(1.0, st.ema_csat)))
        efficiency_sig  = float(max(0.0, min(1.0, 1.0 - st.ema_handle_time / self._MAX_HANDLE_MINUTES)))
        resolution_sig  = float(max(0.0, min(1.0, st.ema_resolution)))
        workload_sig    = float(max(0.0, min(1.0, 1.0 - st.active_calls / self._MAX_ACTIVE_CALLS)))
        resource        = skill_resources.get(agent.primary_skill)
        pred_wait       = self.estimated_wait(resource) if resource else self._MAX_WAIT_MINUTES
        wait_time_sig   = float(max(0.0, min(1.0, 1.0 - pred_wait / self._MAX_WAIT_MINUTES)))
        vip_fit_sig     = (
            self._VIP_EXPERIENCE_SCORES.get(agent.experience, 0.5)
            if call.customer_type == "vip" else 0.5
        )
        return {
            "performance": performance_sig, "efficiency": efficiency_sig,
            "resolution": resolution_sig, "workload": workload_sig,
            "wait_time": wait_time_sig, "vip_fit": float(vip_fit_sig),
        }

    def _find_overflow_resource(self, skill, skill_resources):
        cross_options: Dict[str, int] = {}
        for agent in self.agent_pool:
            if skill in agent.secondary_skills and not agent.on_break:
                alt_skill = agent.primary_skill
                if alt_skill != skill and alt_skill in skill_resources:
                    depth = len(skill_resources[alt_skill].queue)
                    if alt_skill not in cross_options or depth < cross_options[alt_skill]:
                        cross_options[alt_skill] = depth
        if not cross_options:
            return None, None
        best_skill = min(cross_options, key=lambda s: cross_options[s])
        return skill_resources[best_skill], best_skill

    def _avg_handle_time(self):
        handle_times = [
            st.ema_handle_time
            for agent in self.agent_pool
            for st in [self.tracker.get(agent.agent_id)]
            if st.calls_completed > 0
        ]
        if handle_times:
            return sum(handle_times) / len(handle_times)
        return self.config.mean_service_minutes + self.config.acw_mean_minutes


@dataclass
class _CallRecord:
    call_id:        str
    skill:          str
    customer_type:  str
    is_repeat:      bool
    arrival_time:   float
    service_start:  float
    service_end:    float
    csat_raw:       float
    routing_reason: str
    abandoned:      bool = False

    @property
    def wait_minutes(self) -> float:
        return max(0.0, self.service_start - self.arrival_time)

    @property
    def handle_minutes(self) -> float:
        return max(0.0, self.service_end - self.service_start)


class KPIEngine:
    SLA_THRESHOLD_MINUTES: ClassVar[float] = 1.0

    def __init__(self) -> None:
        self._records:      List[_CallRecord] = []
        self._abandonments: List[Dict]        = []

    def record_call(self, record):
        self._records.append(record)

    def reset_measurement(self) -> None:
        """Discard everything collected so far (called at end of warm-up)."""
        self._records.clear()
        self._abandonments.clear()

    def record_abandonment(self, call, abandon_time):
        self._abandonments.append({
            "call_id": call.call_id, "skill": call.skill,
            "customer_type": call.customer_type, "abandon_time": abandon_time,
            "arrival_time": call.arrival_time,
        })

    def total_calls(self):
        return len(self._records)

    def total_abandonments(self):
        return len(self._abandonments)

    def abandonment_rate(self):
        total = self.total_calls() + self.total_abandonments()
        return self.total_abandonments() / total if total else 0.0

    def average_speed_of_answer(self):
        if not self._records:
            return 0.0
        return sum(r.wait_minutes for r in self._records) / len(self._records)

    def average_handle_time(self):
        if not self._records:
            return 0.0
        return sum(r.handle_minutes for r in self._records) / len(self._records)

    def sla_percentage(self):
        if not self._records:
            return 0.0
        within = sum(1 for r in self._records if r.wait_minutes <= self.SLA_THRESHOLD_MINUTES)
        return within / len(self._records)

    def average_csat(self):
        if not self._records:
            return 0.0
        return sum(r.csat_raw for r in self._records) / len(self._records)

    def first_call_resolution(self):
        if not self._records:
            return 0.0
        return sum(1 for r in self._records if not r.is_repeat) / len(self._records)

    def kpis_by_skill(self):
        skill_map: Dict[str, List[_CallRecord]] = {}
        for r in self._records:
            skill_map.setdefault(r.skill, []).append(r)
        return {
            skill: {
                "calls": len(records),
                "asa": sum(r.wait_minutes for r in records) / len(records),
                "aht": sum(r.handle_minutes for r in records) / len(records),
                "csat": sum(r.csat_raw for r in records) / len(records),
                "sla": sum(1 for r in records if r.wait_minutes <= self.SLA_THRESHOLD_MINUTES) / len(records),
            }
            for skill, records in skill_map.items()
        }

    def report(self) -> None:
        w   = 60
        bar = "=" * w
        print(f"\n{bar}")
        print(f"  CALL CENTRE SIMULATION -- KPI REPORT")
        print(f"{bar}")
        print(f"  {'Total calls handled':<35} {self.total_calls():>10}")
        print(f"  {'Total abandonments':<35} {self.total_abandonments():>10}")
        print(f"  {'Abandonment rate':<35} {self.abandonment_rate():>9.1%}")
        print(f"  {'SLA (<=1m)':<35} {self.sla_percentage():>9.1%}")
        print(f"  {'Avg speed of answer (ASA)':<35} {self.average_speed_of_answer():>8.2f}m")
        print(f"  {'Avg handle time (AHT)':<35} {self.average_handle_time():>8.2f}m")
        print(f"  {'Avg CSAT (1-5 scale)':<35} {self.average_csat():>8.2f}")
        print(f"  {'First-call resolution (FCR)':<35} {self.first_call_resolution():>9.1%}")
        print(f"\n  {'-' * (w - 4)}")
        print(f"  Breakdown by Skill")
        print(f"  {'-' * (w - 4)}")
        print(f"  {'Skill':<14} {'Calls':>6} {'ASA':>7} {'AHT':>7} {'CSAT':>6} {'SLA':>7}")
        print(f"  {'-' * (w - 4)}")
        for skill, m in sorted(self.kpis_by_skill().items()):
            print(
                f"  {skill:<14} {m['calls']:>6.0f} "
                f"{m['asa']:>6.2f}m {m['aht']:>6.2f}m "
                f"{m['csat']:>6.2f} {m['sla']:>6.1%}"
            )
        print(f"{bar}\n")


class SimulationEngine:
    _MAX_WAIT_PATIENCE: ClassVar[float] = 10.0
    _BREAK_PRIORITY:    ClassVar[int]   = 100

    def __init__(self, config, weights=None) -> None:
        self.config = config
        self.kpi    = KPIEngine()
        seed = config.random_seed
        if seed is not None:
            random.seed(seed)
        self._rng = random.Random(seed)
        self.env = simpy.Environment()
        self.agents = self._create_agent_pool()
        self._agent_map: Dict[str, Agent] = {a.agent_id: a for a in self.agents}
        self._agents_by_skill: Dict[str, List[Agent]] = {}
        for a in self.agents:
            self._agents_by_skill.setdefault(a.primary_skill, []).append(a)
        self.skill_resources: Dict[str, simpy.PriorityResource] = {
            skill: simpy.PriorityResource(self.env, capacity=count)
            for skill, count in config.agents_per_skill.items()
        }
        self.router = Router(self.agents, config, weights)
        self._call_counter = 0
        self.shrinkage_events: List[Dict] = []

    def run(self):
        self.env.process(self._arrival_process())
        self._schedule_breaks()
        self._schedule_shrinkage()
        if self.config.warmup_minutes and self.config.warmup_minutes > 0:
            self.env.process(self._warmup_monitor())
        self.env.run(until=self.config.sim_duration_minutes)

    # -- multi-factor shrinkage (stochastic availability) ------------------

    def _schedule_shrinkage(self):
        p_ns  = float(getattr(self.config, "agent_no_show_prob", 0.0) or 0.0)
        p_out = float(getattr(self.config, "agent_outage_prob", 0.0) or 0.0)
        if p_ns <= 0 and p_out <= 0:
            return
        dur_out = float(getattr(self.config, "agent_outage_minutes", 20.0))
        horizon = self.config.sim_duration_minutes
        for agent in self.agents:
            r = self._rng.random()
            if p_ns > 0 and r < p_ns:                  # absent whole shift (no-show/sick)
                self.env.process(self._absence_process(agent.primary_skill, 0.0, horizon, "no_show"))
            elif p_out > 0 and r < p_ns + p_out:       # mid-shift outage
                start = self._rng.uniform(0.0, max(1.0, horizon - dur_out))
                self.env.process(self._absence_process(agent.primary_skill, start, dur_out, "outage"))

    def _absence_process(self, skill, start, duration, kind):
        """Remove one agent of `skill` from the floor for `duration`. Unlike a
        break (idle-only priority), an absence wins a server unit ahead of calls
        (priority -1), so it genuinely reduces serving capacity."""
        yield self.env.timeout(max(0.0, start))
        resource = self.skill_resources.get(skill)
        if resource is None:
            return
        req = resource.request(priority=-1)
        yield req
        agent = self._pick_free_agent(skill)
        if agent is None:
            resource.release(req)
            return
        agent.on_break = True
        self.shrinkage_events.append(
            {"agent": agent.agent_id, "skill": skill, "kind": kind,
             "start": float(self.env.now), "duration": float(duration)})
        yield self.env.timeout(max(0.0, duration))
        resource.release(req)
        agent.on_break = False

    def _warmup_monitor(self):
        """Let the system fill up, then drop all warm-up-period measurements so
        only steady-state behaviour is scored."""
        wu = float(self.config.warmup_minutes)
        log = getattr(self.config, "warmup_log", True)
        if log:
            print(f"  [warm-up] system filling for {wu:.0f} min (metrics suppressed) ...")
        yield self.env.timeout(wu)
        self._reset_measurement()
        if log:
            print(f"  [warm-up] complete at t={self.env.now:.0f} — KPI collection started "
                  f"(measurement window {self.config.measurement_minutes:.0f} min).")

    def _reset_measurement(self) -> None:
        """Hook: clear all measurement collected during warm-up. Subclasses
        extend this to also reset cost ledgers and realism counters."""
        self.kpi.reset_measurement()

    def _arrival_process(self):
        while True:
            inter_arrival = self._rng.expovariate(self.config.arrival_rate_per_minute)
            yield self.env.timeout(inter_arrival)
            call = self._generate_call()
            self.env.process(self._handle_call(call))

    def _handle_call(self, call):
        resource, target_skill, routing_reason = self.router.select_resource(
            call, self.skill_resources
        )
        request = resource.request(priority=call.priority)
        result  = yield request | self.env.timeout(self._MAX_WAIT_PATIENCE)
        if request not in result:
            request.cancel()
            self.kpi.record_abandonment(call, self.env.now)
            return
        service_start = self.env.now
        agent         = self._assign_agent(call, target_skill)
        agent_id      = agent.agent_id if agent else None
        if agent_id:
            self.router.notify_call_started(agent_id)
        service_time = max(0.5, self._rng.gauss(self.config.mean_service_minutes, self.config.stdev_service_minutes))
        acw_time = max(0.0, self._rng.gauss(self.config.acw_mean_minutes, self.config.acw_stdev_minutes))
        yield self.env.timeout(service_time + acw_time)
        q_depth_end = len(resource.queue)
        resource.release(request)
        if agent is not None:
            agent.busy = False
        service_end    = self.env.now
        csat_raw       = self._sample_csat(call.customer_type, agent.csat_bias if agent else 0.0, self._rng)
        handle_minutes = service_end - service_start
        if agent_id:
            self.router.notify_call_ended(
                agent_id, csat_raw, handle_minutes, call.is_repeat,
                skill=call.skill, tier=call.customer_type, queue_depth=q_depth_end,
            )
        self.kpi.record_call(_CallRecord(
            call_id=call.call_id, skill=call.skill, customer_type=call.customer_type,
            is_repeat=call.is_repeat, arrival_time=call.arrival_time,
            service_start=service_start, service_end=service_end,
            csat_raw=csat_raw, routing_reason=routing_reason,
        ))

    def _free_agents(self, skill):
        return [a for a in self._agents_by_skill.get(skill, [])
                if not a.busy and not a.on_break]

    def _assign_agent(self, call, target_skill):
        candidates = self._free_agents(target_skill)
        if not candidates:
            return None
        if target_skill != call.skill:
            preferred = [a for a in candidates if call.skill in a.secondary_skills]
            if preferred:
                candidates = preferred
        agent = self.router.pick_agent(call, candidates, self.skill_resources)
        if agent is not None:
            agent.busy = True
        return agent

    def _pick_free_agent(self, skill):
        for a in self._agents_by_skill.get(skill, []):
            if not a.busy and not a.on_break:
                return a
        return None

    def _break_process(self, skill, start, duration):
        yield self.env.timeout(max(0.0, start - self.env.now))
        resource = self.skill_resources.get(skill)
        if resource is None:
            return
        req = resource.request(priority=self._BREAK_PRIORITY)
        yield req
        agent = self._pick_free_agent(skill)
        if agent is None:
            resource.release(req)
            return
        agent.on_break = True
        yield self.env.timeout(max(0.0, duration))
        resource.release(req)
        agent.on_break = False

    def _schedule_breaks(self):
        # Staggered scheduling: within each skill pool, spread the per-agent
        # break start across `break_stagger_window` so the whole pool does not
        # leave the floor at the same instant (which would crater coverage).
        window = float(getattr(self.config, "break_stagger_window", 0.0) or 0.0)
        for skill, members in self._agents_by_skill.items():
            n = len(members)
            for idx, agent in enumerate(members):
                # even phase offset in [0, window): agent k of n -> k/n * window
                offset = (idx / n) * window if (window > 0 and n > 0) else 0.0
                for start, duration in self.config.break_schedule:
                    self.env.process(
                        self._break_process(agent.primary_skill, start + offset, duration)
                    )

    def _create_agent_pool(self):
        experience_distribution = ["junior", "junior", "mid", "mid", "senior"]
        pool: List[Agent] = []
        skills = list(self.config.agents_per_skill.keys())
        for skill, count in self.config.agents_per_skill.items():
            other_skills = [s for s in skills if s != skill]
            for i in range(count):
                exp = experience_distribution[i % len(experience_distribution)]
                secondary = ([self._rng.choice(other_skills)] if exp == "senior" and other_skills else [])
                csat_bias = self._rng.gauss(0.0, 0.5)
                pool.append(Agent(
                    agent_id=f"{skill[:3].upper()}-{i + 1:02d}",
                    primary_skill=skill, secondary_skills=secondary,
                    experience=exp, csat_bias=csat_bias,
                ))
        return pool

    def _generate_call(self):
        self._call_counter += 1
        return Call(
            call_id=f"CALL-{self._call_counter:06d}",
            skill=self._weighted_choice(self.config.skill_mix),
            customer_type=self._weighted_choice(self.config.customer_tier_mix),
            is_repeat=self._rng.random() < self.config.repeat_call_probability,
            arrival_time=self.env.now,
        )

    def _weighted_choice(self, distribution):
        keys, weights = list(distribution.keys()), list(distribution.values())
        total = sum(weights)
        r = self._rng.uniform(0, total)
        cumulative = 0.0
        for key, w in zip(keys, weights):
            cumulative += w
            if r <= cumulative:
                return key
        return keys[-1]

    @staticmethod
    def _sample_csat(customer_type, agent_bias=0.0, rng=None):
        draw = (rng or random).gauss
        base = {"vip": 4.2, "premium": 3.9, "standard": 3.6}.get(customer_type, 3.6)
        return max(1.0, min(5.0, draw(base + agent_bias, 0.8)))


# --- Human realism layer (kept verbatim, trimmed comments) -----------------

@dataclass
class BehaviorConfig:
    fatigue_rate:             float = 0.018
    fatigue_ceiling:          float = 0.85
    max_fatigue_penalty:      float = 0.15
    fatigue_csat_drag:        float = 0.10
    recovery_rate_per_min:    float = 0.012
    learning_rate:            float = 0.35
    max_learning_gain:        float = 0.15
    min_calls_to_plateau:     int   = 40
    break_start_delay_mean:   float = 2.0
    break_start_delay_stdev:  float = 3.5
    break_extension_prob:     float = 0.25
    break_extension_mean_min: float = 4.0
    early_return_prob:        float = 0.15
    early_return_frac:        float = 0.80


class FatigueModel:
    JITTER_SCALE:          float = 2.0
    RATE_VARIANCE:         float = 0.20
    RECOVERY_NONLINEARITY: float = 0.40
    _JITTER: tuple = (0.000, 0.006, 0.012, 0.018, 0.003, 0.009, 0.015,
                      0.002, 0.008, 0.014, 0.005, 0.011, 0.017, 0.001)
    _JITTER_MID: float = 0.009

    def __init__(self, cfg, slot=0):
        self._cfg = cfg
        self._level = 0.0
        jitter = self._JITTER[slot % len(self._JITTER)]
        self._ceiling = min(1.0, cfg.fatigue_ceiling + jitter * self.JITTER_SCALE)
        rate_mult = 1.0 + (jitter - self._JITTER_MID) * self.RATE_VARIANCE / self._JITTER_MID
        self._eff_rate = max(0.001, cfg.fatigue_rate * rate_mult)

    @property
    def level(self):
        return self._level

    def accumulate(self, handle_minutes):
        headroom = max(0.0, self._ceiling - self._level)
        delta = self._eff_rate * handle_minutes * headroom
        self._level = min(self._ceiling, self._level + delta)

    def recover(self, break_minutes):
        k = self.RECOVERY_NONLINEARITY
        boost = 1.0 + k * (1.0 - self._level)
        recovered = self._cfg.recovery_rate_per_min * break_minutes * boost
        self._level = max(0.0, self._level - recovered)

    def handle_time_multiplier(self):
        return 1.0 + self._level * self._cfg.max_fatigue_penalty

    def csat_drag(self):
        return self._level * self._cfg.fatigue_csat_drag


class LearningCurveModel:
    def __init__(self, cfg):
        self._cfg = cfg
        self._n_calls = 0

    @property
    def calls_completed(self):
        return self._n_calls

    def record_call(self):
        self._n_calls += 1

    def csat_gain(self):
        n = self._n_calls
        if n == 0:
            return 0.0
        plateau = self._cfg.min_calls_to_plateau / 0.9
        return float(min(self._cfg.max_learning_gain,
                         self._cfg.max_learning_gain * (n / (n + plateau))))

    def handle_time_multiplier(self):
        gain_frac = self.csat_gain() / max(1e-6, self._cfg.max_learning_gain)
        return max(0.90, 1.0 - 0.10 * gain_frac)


class BreakVariabilityModel:
    def __init__(self, cfg, rng_seed=None):
        self._cfg = cfg
        self._rng = random.Random(rng_seed)
        self._events: List[Dict] = []

    def sample_break_timing(self, agent_id, scheduled_start, scheduled_duration):
        cfg = self._cfg
        delay = max(0.0, self._rng.gauss(cfg.break_start_delay_mean, cfg.break_start_delay_stdev))
        actual_start = scheduled_start + delay
        effect = "none"
        if self._rng.random() < cfg.early_return_prob:
            actual_duration = scheduled_duration * cfg.early_return_frac
            effect = "early_return"
        elif self._rng.random() < cfg.break_extension_prob:
            extra = self._rng.expovariate(1.0 / cfg.break_extension_mean_min)
            actual_duration = scheduled_duration + extra
            effect = f"extended+{extra:.1f}m"
        else:
            actual_duration = scheduled_duration
        self._events.append({
            "agent_id": agent_id, "sched_start": scheduled_start,
            "sched_dur": scheduled_duration, "delay": delay,
            "actual_dur": actual_duration, "effect": effect,
        })
        return actual_start, actual_duration

    def print_break_log(self) -> None:
        if not self._events:
            print("  [BreakVariabilityModel] No break events recorded.")
            return
        print("\n-- Break Variability Log -------------------------------------------")
        print(f"  {'Agent':<20}  {'Sched Start':>12}  {'Delay':>7}  {'Sched Dur':>10}  {'Actual Dur':>10}  Effect")
        print("  " + "-" * 76)
        for ev in self._events:
            print(
                f"  {ev['agent_id']:<20}  {ev['sched_start']:>12.1f}  "
                f"{ev['delay']:>6.1f}m  {ev['sched_dur']:>9.1f}m  "
                f"{ev['actual_dur']:>9.1f}m  {ev['effect']}"
            )


class HumanAgentState:
    def __init__(self, agent_id, cfg, slot=0):
        self.agent_id = agent_id
        self.cfg = cfg
        self.fatigue = FatigueModel(cfg, slot=slot)
        self.learning = LearningCurveModel(cfg)
        self.total_handle_minutes = 0.0
        self.peak_fatigue_level = 0.0
        self.calls_handled = 0

    def adjust_service_time(self, base_minutes):
        return max(0.5, base_minutes * self.fatigue.handle_time_multiplier()
                   * self.learning.handle_time_multiplier())

    def adjust_csat(self, base_csat):
        norm = (base_csat - 1.0) / 4.0
        adjusted = norm - self.fatigue.csat_drag() + self.learning.csat_gain()
        return 1.0 + max(0.0, min(1.0, adjusted)) * 4.0

    def on_call_end(self, handle_minutes):
        self.fatigue.accumulate(handle_minutes)
        self.learning.record_call()
        self.total_handle_minutes += handle_minutes
        self.calls_handled += 1
        self.peak_fatigue_level = max(self.peak_fatigue_level, self.fatigue.level)

    def on_break_end(self, break_duration):
        self.fatigue.recover(break_duration)


class HumanRealisticsEngine:
    def __init__(self, agents, cfg, beh):
        self._beh = beh
        self._states: Dict[str, HumanAgentState] = {
            a.agent_id: HumanAgentState(a.agent_id, beh, slot=i)
            for i, a in enumerate(agents)
        }
        self._break_model = BreakVariabilityModel(beh)

    def get_state(self, agent_id):
        return self._states.get(agent_id)

    def adjust_service_times(self, agent_id, service_time, acw_time):
        state = self._states.get(agent_id)
        if state is None:
            return service_time, acw_time
        return state.adjust_service_time(service_time), state.adjust_service_time(acw_time)

    def adjust_csat(self, agent_id, csat_raw):
        state = self._states.get(agent_id)
        return state.adjust_csat(csat_raw) if state else csat_raw

    def on_call_end(self, agent_id, handle_minutes):
        state = self._states.get(agent_id)
        if state:
            state.on_call_end(handle_minutes)

    def on_break_end(self, agent_id, actual_duration):
        state = self._states.get(agent_id)
        if state:
            state.on_break_end(actual_duration)

    def sample_break_timing(self, agent_id, scheduled_start, scheduled_duration):
        return self._break_model.sample_break_timing(agent_id, scheduled_start, scheduled_duration)

    def fatigue_summary(self):
        return {aid: st.fatigue.level for aid, st in self._states.items()}

    def learning_summary(self):
        return {aid: st.learning.csat_gain() for aid, st in self._states.items()}

    def reset_measurement(self) -> None:
        """End-of-warm-up reset: keep the warmed fatigue *level* (steady state)
        but zero the utilisation/peak counters so they reflect only the
        measurement window."""
        for st in self._states.values():
            st.total_handle_minutes = 0.0
            st.calls_handled        = 0
            st.peak_fatigue_level   = st.fatigue.level
        self._break_model._events.clear()

    def print_report(self) -> None:
        w   = 72
        bar = "=" * w
        print(f"\n{bar}")
        print(f"  HUMAN REALISM LAYER -- AGENT STATE REPORT")
        print(f"{bar}")
        print(
            f"  {'Agent':<20}  {'Calls':>6}  {'Fatigue':>8}  "
            f"{'PeakFat':>8}  {'LrnGain':>8}  {'TotalHdl':>9}"
        )
        print(f"  {'-' * (w - 4)}")
        for state in sorted(self._states.values(), key=lambda s: s.agent_id):
            print(
                f"  {state.agent_id:<20}  {state.calls_handled:>6}  "
                f"{state.fatigue.level:>8.3f}  {state.peak_fatigue_level:>8.3f}  "
                f"{state.learning.csat_gain():>8.4f}  {state.total_handle_minutes:>8.1f}m"
            )
        print(f"{bar}")
        self._break_model.print_break_log()


class RealisticsAwareEngine(SimulationEngine):
    def __init__(self, config, behavior=None, weights=None):
        super().__init__(config, weights)
        self.realism = HumanRealisticsEngine(self.agents, config, behavior or BehaviorConfig())

    def _reset_measurement(self) -> None:
        super()._reset_measurement()
        self.realism.reset_measurement()

    def _handle_call(self, call):
        resource, target_skill, routing_reason = self.router.select_resource(call, self.skill_resources)
        request = resource.request(priority=call.priority)
        result  = yield request | self.env.timeout(self._MAX_WAIT_PATIENCE)
        if request not in result:
            request.cancel()
            self.kpi.record_abandonment(call, self.env.now)
            return
        service_start = self.env.now
        agent = self._assign_agent(call, target_skill)
        agent_id = agent.agent_id if agent else None
        if agent_id:
            self.router.notify_call_started(agent_id)
        base_svc = max(0.5, self._rng.gauss(self.config.mean_service_minutes, self.config.stdev_service_minutes))
        base_acw = max(0.0, self._rng.gauss(self.config.acw_mean_minutes, self.config.acw_stdev_minutes))
        if agent_id:
            svc_time, acw_time = self.realism.adjust_service_times(agent_id, base_svc, base_acw)
        else:
            svc_time, acw_time = base_svc, base_acw
        yield self.env.timeout(svc_time + acw_time)
        q_depth_end = len(resource.queue)
        resource.release(request)
        if agent is not None:
            agent.busy = False
        service_end = self.env.now
        base_csat = self._sample_csat(call.customer_type, agent.csat_bias if agent else 0.0, self._rng)
        csat_raw = self.realism.adjust_csat(agent_id, base_csat) if agent_id else base_csat
        handle_minutes = service_end - service_start
        if agent_id:
            self.realism.on_call_end(agent_id, handle_minutes)
            self.router.notify_call_ended(
                agent_id, csat_raw, handle_minutes, call.is_repeat,
                skill=call.skill, tier=call.customer_type, queue_depth=q_depth_end,
            )
        self.kpi.record_call(_CallRecord(
            call_id=call.call_id, skill=call.skill, customer_type=call.customer_type,
            is_repeat=call.is_repeat, arrival_time=call.arrival_time,
            service_start=service_start, service_end=service_end,
            csat_raw=csat_raw, routing_reason=routing_reason,
        ))

    def _break_process(self, skill, start, duration):
        yield self.env.timeout(max(0.0, start - self.env.now))
        resource = self.skill_resources.get(skill)
        if resource is None:
            return
        req = resource.request(priority=self._BREAK_PRIORITY)
        yield req
        agent = self._pick_free_agent(skill)
        if agent is None:
            resource.release(req)
            return
        _, actual_duration = self.realism.sample_break_timing(agent.agent_id, start, duration)
        agent.on_break = True
        yield self.env.timeout(max(0.0, actual_duration))
        resource.release(req)
        agent.on_break = False
        self.realism.on_break_end(agent.agent_id, actual_duration)


# ===========================================================================
# MONTE CARLO LAYER  (multi-seed replication + KPI statistics)
# ===========================================================================
#
# Why: a single seed is one sample path. Replicating the run over many
# independent seeds and aggregating turns each KPI into a sampling distribution
# so we can report a mean and a confidence interval rather than one number.

import statistics as _stats
import math
from typing import Callable as _Callable

try:
    from scipy.stats import t as _student_t          # for small-n CIs
    _SCIPY_T = True
except ImportError:                                   # pragma: no cover
    _SCIPY_T = False


@dataclass
class KPIStatistic:
    """Sampling distribution of one KPI across Monte Carlo runs."""
    name:    str
    n:       int
    mean:    float
    median:  float
    std:     float            # sample standard deviation (ddof=1)
    minimum: float
    maximum: float
    ci_low:  float            # 95% CI on the mean
    ci_high: float

    @property
    def half_width(self) -> float:
        return (self.ci_high - self.ci_low) / 2.0


def _summary_stat(name: str, xs: List[float], confidence: float = 0.95) -> KPIStatistic:
    n = len(xs)
    if n == 0:
        return KPIStatistic(name, 0, 0, 0, 0, 0, 0, 0, 0)
    mean = _stats.fmean(xs)
    if n == 1:
        return KPIStatistic(name, 1, mean, mean, 0.0, xs[0], xs[0], mean, mean)
    sd      = _stats.stdev(xs)                         # ddof=1
    se      = sd / math.sqrt(n)
    alpha   = 1.0 - confidence
    if _SCIPY_T:
        crit = float(_student_t.ppf(1.0 - alpha / 2.0, df=n - 1))   # t for small n
    else:
        crit = 1.959963985                              # normal approximation
    return KPIStatistic(
        name=name, n=n, mean=mean, median=_stats.median(xs), std=sd,
        minimum=min(xs), maximum=max(xs),
        ci_low=mean - crit * se, ci_high=mean + crit * se,
    )


# KPIs whose lower value is the better outcome (everything else: higher better).
_LOWER_IS_BETTER = {"asa", "aht", "abandonment_rate", "abandoned_calls",
                    "occupancy", "total_cost", "cost_per_call"}


class MonteCarloResult:
    def __init__(self, per_run: List[Dict[str, float]], rank_metric: str = "sla") -> None:
        self.per_run     = per_run
        self.rank_metric = rank_metric
        self.metrics     = sorted({k for r in per_run for k in r})
        self.samples: Dict[str, List[float]] = {
            m: [r[m] for r in per_run if m in r and r[m] is not None] for m in self.metrics
        }
        self.stats: Dict[str, KPIStatistic] = {
            m: _summary_stat(m, xs) for m, xs in self.samples.items() if xs
        }

    # -- best / worst / representative run ---------------------------------

    def _rank_key(self, run: Dict[str, float]) -> float:
        v = run.get(self.rank_metric, 0.0)
        return -v if self.rank_metric in _LOWER_IS_BETTER else v

    def best_run(self) -> Dict[str, float]:
        return max(self.per_run, key=self._rank_key)

    def worst_run(self) -> Dict[str, float]:
        return min(self.per_run, key=self._rank_key)

    def average_run(self) -> Dict[str, float]:
        """The single run whose ranking metric is closest to the mean."""
        if self.rank_metric not in self.stats:
            return self.per_run[len(self.per_run) // 2]
        target = self.stats[self.rank_metric].mean
        return min(self.per_run, key=lambda r: abs(r.get(self.rank_metric, 0.0) - target))

    # -- reporting ----------------------------------------------------------

    _PCT  = {"sla", "abandonment_rate", "fcr", "occupancy"}
    _MONEY = {"total_cost", "cost_per_call"}
    _ORDER = ["sla", "asa", "aht", "abandonment_rate", "occupancy", "csat", "fcr",
              "handled_calls", "abandoned_calls", "total_cost", "cost_per_call"]

    def _fmt(self, metric: str, v: float) -> str:
        if metric in self._PCT:
            return f"{v:.1%}"
        if metric in self._MONEY:
            return f"£{v:,.2f}"
        if metric in ("handled_calls", "abandoned_calls"):
            return f"{v:,.0f}"
        return f"{v:.3f}"

    def _ordered_metrics(self) -> List[str]:
        head = [m for m in self._ORDER if m in self.stats]
        tail = [m for m in self.stats if m not in head]
        return head + tail

    def print_report(self, confidence: float = 0.95) -> None:
        w = 96
        bar = "=" * w
        n = len(self.per_run)
        print(f"\n{bar}")
        print(f"  MONTE CARLO ANALYSIS  ({n} runs, {int(confidence*100)}% confidence intervals)")
        print(f"{bar}")
        print(f"  {'KPI':<18}{'Mean':>11}{'Median':>11}{'Std':>10}{'Min':>11}"
              f"{'Max':>11}{'95% CI':>23}")
        print(f"  {'-' * (w - 4)}")
        for m in self._ordered_metrics():
            s = self.stats[m]
            ci = f"[{self._fmt(m, s.ci_low)}, {self._fmt(m, s.ci_high)}]"
            print(f"  {m:<18}{self._fmt(m, s.mean):>11}{self._fmt(m, s.median):>11}"
                  f"{self._fmt(m, s.std):>10}{self._fmt(m, s.minimum):>11}"
                  f"{self._fmt(m, s.maximum):>11}{ci:>23}")
        print(f"  {'-' * (w - 4)}")

        rm = self.rank_metric
        best, worst, avg = self.best_run(), self.worst_run(), self.average_run()
        print(f"\n  Scenario analysis (ranked by '{rm}'):")
        for label, run in (("Best case", best), ("Average case", avg), ("Worst case", worst)):
            bits = []
            for m in ("sla", "abandonment_rate", "csat", "total_cost"):
                if m in run:
                    bits.append(f"{m}={self._fmt(m, run[m])}")
            print(f"    {label:<14} {'  '.join(bits)}")

        # risk / variability summary
        if rm in self.stats:
            s = self.stats[rm]
            cv = (s.std / s.mean) if s.mean else 0.0
            print(f"\n  Variability & risk ('{rm}'):")
            print(f"    Coefficient of variation : {cv:.1%}")
            print(f"    CI half-width            : ±{self._fmt(rm, s.half_width)}")
            print(f"    Spread (max-min)         : {self._fmt(rm, s.maximum - s.minimum)}")
            if rm == "sla":
                target_hits = sum(1 for r in self.per_run if r.get("sla", 0) >= 0.90)
                print(f"    Runs meeting 90% SLA     : {target_hits}/{n} ({target_hits/n:.0%})")
        print(f"{bar}\n")


class MonteCarloRunner:
    """Replicates a simulation across independent seeds and aggregates KPIs.

    `engine_factory(seed)` must return a *fresh* engine configured for that seed
    (so each replication is independent). Works with any engine exposing `.kpi`,
    `.agents` and `.config`; cost is included automatically when the engine has a
    `.cost_function`.
    """

    def __init__(self, engine_factory: "_Callable[[int], object]",
                 n_runs: int = 30, base_seed: int = 1000,
                 rank_metric: str = "sla", verbose: bool = True) -> None:
        self._factory = engine_factory
        self._n       = max(1, int(n_runs))
        self._seed0   = int(base_seed)
        self._rank    = rank_metric
        self._verbose = verbose

    def run(self) -> MonteCarloResult:
        if self._verbose:
            print(f"\n  [Monte Carlo] running {self._n} independent replications ...")
        per_run: List[Dict[str, float]] = []
        for i in range(self._n):
            seed   = self._seed0 + i * 7919          # spaced seeds
            engine = self._factory(seed)
            engine.run()
            per_run.append(self._extract_kpis(engine))
            if self._verbose and (i + 1) % max(1, self._n // 10) == 0:
                print(f"      run {i + 1:>3}/{self._n}  seed={seed}  "
                      f"SLA={per_run[-1].get('sla', 0):.1%}")
        return MonteCarloResult(per_run, rank_metric=self._rank)

    @staticmethod
    def _extract_kpis(engine) -> Dict[str, float]:
        kpi = engine.kpi
        cfg = engine.config
        out: Dict[str, float] = {
            "sla":              kpi.sla_percentage(),
            "asa":              kpi.average_speed_of_answer(),
            "aht":              kpi.average_handle_time(),
            "abandonment_rate": kpi.abandonment_rate(),
            "csat":             kpi.average_csat(),
            "fcr":              kpi.first_call_resolution(),
            "handled_calls":    float(kpi.total_calls()),
            "abandoned_calls":  float(kpi.total_abandonments()),
        }
        # occupancy = served handle-time / available agent-time in the window
        n_agents = max(1, len(getattr(engine, "agents", []) or []))
        window   = getattr(cfg, "measurement_minutes", cfg.sim_duration_minutes)
        handle   = sum(r.handle_minutes for r in kpi._records)
        out["occupancy"] = min(1.0, handle / (n_agents * window)) if window > 0 else 0.0
        # cost, only if the engine carries a cost function
        cf = getattr(engine, "cost_function", None)
        if cf is not None:
            try:
                out["total_cost"]   = cf.total()
                out["cost_per_call"] = cf.cost_per_handled_call(kpi.total_calls())
            except Exception:
                pass
        return out