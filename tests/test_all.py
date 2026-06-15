# --- packaging bootstrap (added when the project was reorganised) -----------
# Lets `python tests/test_all.py` (and pytest) import the flat `callcenter/`
# library. Test logic below is unchanged.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "callcenter"))
# ----------------------------------------------------------------------------

import traceback
from core_simulation import SimulationConfig

def banner(t): print("\n" + "#"*70 + f"\n# {t}\n" + "#"*70)
results = {}

def check(name, fn):
    try:
        fn(); results[name] = "PASS"; print(f"[PASS] {name}")
    except Exception as e:
        results[name] = f"FAIL: {type(e).__name__}: {e}"
        print(f"[FAIL] {name}\n{traceback.format_exc()}")

base = SimulationConfig(arrival_rate_per_hour=120,
                        agents_per_skill={"billing": 8, "technical": 8, "general": 6})

banner("0. imports")
def t_imports():
    import disruption_injector, stress_test, cost_system, ml_system, optimization, hierarchical_overflow, eda
check("imports", t_imports)

banner("1. base engine stress_test_plan")
def t_base():
    from stress_test import stress_test_plan, compound
    eng, inj, kpis = stress_test_plan(base.agents_per_skill, compound(), base_cfg=base, report=False)
    assert len(kpis) == 3 and kpis[1].handled > 0
check("base_stress", t_base)

banner("2. ML engine stress_test_plan")
def t_ml():
    from stress_test import stress_test_plan, demand_spike
    from ml_system import MLSimulationEngine, build_registry
    eng, inj, kpis = stress_test_plan(base.agents_per_skill, demand_spike(), base_cfg=base,
        engine_cls=MLSimulationEngine, registry_factory=build_registry,
        engine_kwargs={"use_forecaster": False}, report=False)
    assert kpis[1].offered > kpis[0].offered  # demand spike raises offered load
check("ml_stress", t_ml)

banner("3. hierarchical engine stress_test_plan")
def t_hier():
    from stress_test import stress_test_plan, agent_outage
    from hierarchical_overflow import HierarchicalOverflowEngine
    from ml_system import build_registry
    eng, inj, kpis = stress_test_plan(base.agents_per_skill, agent_outage(end=480), base_cfg=base,
        engine_cls=HierarchicalOverflowEngine, registry_factory=build_registry,
        engine_kwargs={"use_forecaster": False}, report=False)
    assert inj.pulled_agents, "no agents were seized"
check("hier_stress", t_hier)

banner("4. make_disruption_evaluator")
def t_eval():
    from stress_test import make_disruption_evaluator, agent_outage
    from optimization import OptimizationConfig, OptimizationResult, HORIZON_MINUTES, SHIFT_WINDOWS
    DisEval = make_disruption_evaluator(agent_outage(end=480))
    ev = DisEval(base, OptimizationConfig())
    plan = {"billing":8,"technical":8,"general":6}
    res = OptimizationResult(agents_per_skill=plan,
        shift_plan={sh: dict(plan) for sh,_,_ in SHIFT_WINDOWS}, band_coverage={},
        analytical_sla={}, analytical_target=0.9, total_staffing_cost=0.0, status="test")
    er = ev.evaluate(res)
    assert er.total_calls > 0
check("disruption_evaluator", t_eval)

banner("5. stress_test_optimized (FULL pipeline, blind Stage 1)")
def t_opt():
    from stress_test import stress_test_optimized, agent_outage
    from optimization import OptimizationConfig
    oc = OptimizationConfig(max_iterations=2, verbose=False, cpsat_time_limit_seconds=5.0)
    loop, (eng, inj, kpis) = stress_test_optimized(base, agent_outage(end=480),
                                                   opt_cfg=oc, report=False)
    assert loop.best_plan and len(kpis) == 3
    print("   converged roster:", loop.best_plan)
    print("   pre SLA %.1f%% -> impact SLA %.1f%% -> recovery SLA %.1f%%"
          % (kpis[0].sla*100, kpis[1].sla*100, kpis[2].sla*100))
check("stress_test_optimized", t_opt)

banner("SUMMARY")
for k,v in results.items(): print(f"  {k:<28} {v}")