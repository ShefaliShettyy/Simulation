"""
stress_test.py — integration layer that wires the DisruptionInjector into the
existing optimize -> simulate -> (ML / hierarchical) pipeline.

NOTHING in your engine files is modified. This module composes on top of them,
matching the project's existing style (HierarchicalRouter, _StaffingModel are
all standalone, side-effect-free objects). It exposes three entry points:

  1. stress_test_plan(...)       — run ONE disrupted sim of a fixed roster on
                                   any engine (base / cost / realism / ML /
                                   hierarchical) and print the windowed report.

  2. stress_test_optimized(...)  — the headline "what-if": run your existing
                                   Stage 1+2 loop BLIND (optimiser never sees the
                                   shock), then stress-test the converged roster.

  3. make_disruption_evaluator(...) — a drop-in SimulationEvaluator subclass for
                                   when you WANT the evaluator to run disrupted
                                   (e.g. a reactive-replanning experiment).

Heavy deps (OR-Tools, sklearn, scipy, eda, cost_system) are imported lazily
inside the functions that need them, so this module imports cleanly with only
SimPy present.
"""

from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional, Tuple

from core_simulation import SimulationEngine, SimulationConfig
from disruption_injector import (
    DisruptionConfig, DemandShock, SupplyShock,
    DisruptionInjector, WindowedKPIAnalyzer, DisruptionReport, WindowKPIs,
    make_disruption_engine, default_windows, AnalysisWindow,
)

DEFAULT_HORIZON = 720.0


# ===========================================================================
# 1. Fixed-roster stress test (works for EVERY engine in the project)
# ===========================================================================

def stress_test_plan(roster: Dict[str, int],
                     dcfg: DisruptionConfig,
                     *,
                     base_cfg: Optional[SimulationConfig] = None,
                     engine_cls=None,
                     registry_factory: Optional[Callable[[SimulationConfig], object]] = None,
                     engine_kwargs: Optional[dict] = None,
                     horizon: float = DEFAULT_HORIZON,
                     seed: int = 42,
                     windows: Optional[List[AnalysisWindow]] = None,
                     wages: Optional[Dict[str, float]] = None,
                     report: bool = True
                     ) -> Tuple[object, DisruptionInjector, List[WindowKPIs]]:
    """Run a single disrupted simulation of `roster` and return
    (engine, injector, windowed_kpis).

    engine_cls       : SimulationEngine (default) | CostAwareEngine |
                       CostAwareRealisticsEngine | MLSimulationEngine |
                       HierarchicalOverflowEngine
    registry_factory : for engines needing an MLRegistry, e.g. build_registry.
                       Called as registry_factory(cfg); the registry is passed
                       as the engine's 2nd positional arg.
    engine_kwargs    : extra ctor kwargs (e.g. {"tier_cfg": TierConfig(),
                       "use_forecaster": False}).
    """
    cfg = copy.copy(base_cfg) if base_cfg is not None else SimulationConfig()
    cfg.agents_per_skill     = dict(roster)
    cfg.sim_duration_minutes = float(horizon)
    cfg.random_seed          = seed

    EngineCls = engine_cls or SimulationEngine
    Stress    = make_disruption_engine(EngineCls)     # adds the arrival-rate hook

    args = []
    if registry_factory is not None:
        args.append(registry_factory(cfg))            # ML / hierarchical registry

    engine   = Stress(cfg, *args, **(engine_kwargs or {}))
    injector = DisruptionInjector(dcfg, horizon=horizon).install(engine)
    engine.run()

    windows = windows or default_windows(horizon=horizon)
    kpis = WindowedKPIAnalyzer(engine, windows, injector=injector, wages=wages).analyze()
    if report:
        DisruptionReport(kpis, injector, dcfg, horizon=horizon).print_report()
    return engine, injector, kpis


# ===========================================================================
# 2. Stage-1-BLIND optimise -> stress-test the converged plan (the what-if)
# ===========================================================================

def stress_test_optimized(base_sim_cfg: SimulationConfig,
                          dcfg: DisruptionConfig,
                          *,
                          opt_cfg=None,
                          cost_cfg=None,
                          beh_cfg=None,
                          weights=None,
                          horizon: Optional[float] = None,
                          report: bool = True):
    """Run your existing OptimizeSimulateLoop UNCHANGED and BLIND, then evaluate
    its converged roster under disruption on the SAME engine your pipeline uses.

    The CP-SAT optimiser never sees the shock (the loop runs with the normal
    evaluator), so this is the canonical 'perfect schedule meets real-world
    volatility' experiment. The disrupted evaluation reuses your
    SimulationEvaluator._build_engine, so cost/realism/base selection and all
    constructor wiring stay exactly as your pipeline defines them.

    Returns (loop, (engine, injector, windowed_kpis)).
    """
    from optimization import (OptimizeSimulateLoop, OptimizationConfig,
                              SimulationEvaluator, HORIZON_MINUTES)

    opt_cfg = opt_cfg or OptimizationConfig()
    horizon = float(horizon or HORIZON_MINUTES)

    # --- Stage 1+2, exactly as-is and BLIND ---
    loop = OptimizeSimulateLoop(base_sim_cfg, opt_cfg, cost_cfg, beh_cfg, weights)
    loop.run()
    entry = loop._best_history_entry()
    if entry is None:
        raise RuntimeError("optimiser produced no plan to stress-test")
    opt_result = entry.opt_result

    # --- one disrupted evaluation of the converged plan, reusing your builder ---
    captured: Dict[str, object] = {}

    class _CapturingDisruptedEvaluator(SimulationEvaluator):
        def _build_engine(self, cfg):
            engine = super()._build_engine(cfg)                       # your engine choice
            engine.__class__ = make_disruption_engine(type(engine))   # add arrival hook in place
            inj = DisruptionInjector(dcfg, cfg.sim_duration_minutes).install(engine)
            captured["engine"], captured["injector"] = engine, inj
            return engine

    ev = _CapturingDisruptedEvaluator(base_sim_cfg, opt_cfg, cost_cfg, beh_cfg, weights)
    ev.evaluate(opt_result)                                            # runs the disrupted sim

    engine, injector = captured["engine"], captured["injector"]
    windows = default_windows(horizon=horizon)
    kpis = WindowedKPIAnalyzer(engine, windows, injector=injector).analyze()
    if report:
        DisruptionReport(kpis, injector, dcfg, horizon=horizon).print_report()
    return loop, (engine, injector, kpis)


# ===========================================================================
# 3. Drop-in disrupted SimulationEvaluator (for replanning experiments)
# ===========================================================================

def make_disruption_evaluator(disruption_cfg: DisruptionConfig):
    """Return a SimulationEvaluator subclass whose simulations run disrupted.

    USE WITH CARE: if you plug this into OptimizeSimulateLoop, the optimiser's
    convergence sim becomes disrupted and Stage 1 is NO LONGER blind (it will
    over-staff to compensate). That is exactly what you want for a *reactive
    replanning* study, and exactly what you DON'T want for the blind what-if
    (use stress_test_optimized for that).

    Implementation note: it re-classes the engine the parent already built to a
    disruption-aware subclass (the mixin only adds _arrival_process, no new
    state), so it works for whatever engine type your opt_cfg selects.
    """
    from optimization import SimulationEvaluator, HORIZON_MINUTES

    class DisruptionEvaluator(SimulationEvaluator):
        _DISRUPTION = disruption_cfg
        _HORIZON    = float(HORIZON_MINUTES)

        def _build_engine(self, cfg):
            engine = super()._build_engine(cfg)                 # parent picks the engine type
            engine.__class__ = make_disruption_engine(type(engine))   # add arrival hook in place
            DisruptionInjector(self._DISRUPTION, cfg.sim_duration_minutes).install(engine)
            return engine

    return DisruptionEvaluator


# ===========================================================================
# 4. Convenience scenario builders
# ===========================================================================

def demand_spike(start=300, end=480, pct=0.5) -> DisruptionConfig:
    return DisruptionConfig(demand=DemandShock(enabled=True, start_minute=start,
                                               end_minute=end, rate_multiplier=1.0 + pct))


def agent_outage(skill="billing", n=3, start=300, end=None) -> DisruptionConfig:
    return DisruptionConfig(supply=SupplyShock(enabled=True, start_minute=start,
                                               end_minute=end, skill=skill, n_agents=n))


def compound(skill="billing", n=3, start=300, demand_end=480, pct=0.5) -> DisruptionConfig:
    return DisruptionConfig(
        demand=DemandShock(enabled=True, start_minute=start, end_minute=demand_end,
                           rate_multiplier=1.0 + pct),
        supply=SupplyShock(enabled=True, start_minute=start, end_minute=None,
                           skill=skill, n_agents=n))


# ===========================================================================
# Self-verifying demo (base engine — runs with only SimPy installed)
# ===========================================================================

if __name__ == "__main__":
    base = SimulationConfig(arrival_rate_per_hour=120,
                            agents_per_skill={"billing": 8, "technical": 8, "general": 6})

    print("\n########## 1. fixed-roster stress test (base engine) ##########")
    stress_test_plan(base.agents_per_skill, compound(), base_cfg=base)

    # Examples for your richer engines (uncomment in your full environment):
    #
    # from ml_system import MLSimulationEngine, build_registry
    # stress_test_plan(base.agents_per_skill, demand_spike(), base_cfg=base,
    #                  engine_cls=MLSimulationEngine, registry_factory=build_registry,
    #                  engine_kwargs={"use_forecaster": False})
    #
    # from hierarchical_overflow import HierarchicalOverflowEngine, TierConfig
    # from ml_system import build_registry
    # stress_test_plan(base.agents_per_skill, agent_outage(), base_cfg=base,
    #                  engine_cls=HierarchicalOverflowEngine, registry_factory=build_registry,
    #                  engine_kwargs={"tier_cfg": TierConfig(), "use_forecaster": False})
    #
    # # 2. blind optimise -> stress (needs OR-Tools etc.):
    # from optimization import OptimizationConfig
    # loop, (eng, inj, kpis) = stress_test_optimized(base, agent_outage(),
    #                                                 opt_cfg=OptimizationConfig())