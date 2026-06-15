# --- packaging bootstrap (added when the project was reorganised) -----------
# Lets `python run.py` find the flat `callcenter/` library without installing
# anything. The library modules themselves are unchanged.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "callcenter"))
# ----------------------------------------------------------------------------

import copy
import time

from core_simulation import (
    BehaviorConfig,
    SimulationConfig,
    SimulationEngine,
)
from cost_system import (
    CostAwareEngine,
    CostAwareRealisticsEngine,
    CostConfig,
    CostReport,
    ScenarioCostResult,
)
from optimization import (
    COVERAGE_BANDS,
    HORIZON_MINUTES,
    PEAK_BAND_INDEX,
    SHIFT_WINDOWS,
    _SHIFT_BREAK_MIDPOINTS,
    _BREAK_DURATION_MINUTES,
    CoverageValidator,
    ErlangC,
    LabourConstraintConfig,
    OptimizationConfig,
    OptimizationResult,
    OptimizeSimulateLoop,
    ShrinkageConfig,
    StaffingOptimizer,
    SurrogateWarmStart,
)
from core_simulation import MonteCarloRunner
from ml_system import (
    FeedbackConfig,
    FeedbackLoop,
    FeedbackLoopReport,
    MLRegistry,
    MLSimulationEngine,
    RewardConfig,
    build_registry,
)


def section(title: str, width: int = 72) -> None:
    bar = "=" * width
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


def divider(width: int = 72) -> None:
    print("-" * width)


def print_shift_plan(opt_result, wages: dict, overhead: float) -> None:
    SHIFT_HOURS = 8.0
    print(f"\n  Shift Plan  ({HORIZON_MINUTES}-min horizon: 3 overlapping 8-hr shifts)")
    for sh_name, sh_start, sh_end in SHIFT_WINDOWS:
        plan       = opt_result.shift_plan.get(sh_name, {})
        shift_cost = sum(plan.get(sk, 0) * wages.get(sk, 18.0) * overhead * SHIFT_HOURS for sk in plan)
        agents_str = "  ".join(f"{sk}: {n}" for sk, n in sorted(plan.items()))
        break_min  = _SHIFT_BREAK_MIDPOINTS[sh_name]
        print(f"    Shift {sh_name}  ({sh_start:>4}-{sh_end:>4} min)  break@{break_min:.0f}m"
              f"  [{agents_str}]  cost=£{shift_cost:,.2f}")

    skills    = sorted(opt_result.agents_per_skill.keys())
    hdr_skills = "".join(f"  {s[:8]:>9}" for s in skills)
    print(f"\n  {'Band interval':<22}  {'Shifts':>9}{hdr_skills}")
    divider(22 + 12 + 10 * len(skills))
    for band_idx, (b_start, b_end, active_shifts) in enumerate(COVERAGE_BANDS):
        cov      = opt_result.band_coverage.get(band_idx, {})
        sh_str   = "+".join(active_shifts)
        agents_s = "".join(f"  {cov.get(sk, 0):>9}" for sk in skills)
        peak_tag = "  <- PEAK" if band_idx == PEAK_BAND_INDEX else ""
        print(f"  [{b_start:>4}-{b_end:>4} min]        {sh_str:>9}{agents_s}{peak_tag}")


def build_optimised_sim_cfg(base_cfg, opt_result):
    cfg = copy.copy(base_cfg)
    cfg.agents_per_skill     = dict(opt_result.agents_per_skill)
    cfg.sim_duration_minutes = float(HORIZON_MINUTES)
    cfg.break_schedule       = [
        (_SHIFT_BREAK_MIDPOINTS["M"], _BREAK_DURATION_MINUTES),
        (_SHIFT_BREAK_MIDPOINTS["D"], _BREAK_DURATION_MINUTES),
        (_SHIFT_BREAK_MIDPOINTS["E"], _BREAK_DURATION_MINUTES),
    ]
    return cfg


SEED                   = 42
SIM_DURATION_MINUTES   = float(HORIZON_MINUTES)
ARRIVAL_RATE_PER_HOUR  = 60.0
SLA_TARGET             = 0.90
SLA_THRESHOLD_MINUTES  = 1.0
MAX_AGENTS_PER_SKILL   = 15
MIN_AGENTS_PER_SHIFT   = 1
MAX_OPT_ITERATIONS     = 8
CONVERGENCE_TOLERANCE  = 0.015
ML_EPOCHS              = 5
WARMUP_MINUTES         = 60.0          # steady-state warm-up
BREAK_STAGGER_WINDOW   = 20.0          # spread break starts within each skill pool
MONTE_CARLO_RUNS       = 20            # replications for the Monte Carlo stage

# Multi-objective routing reward (item 1) — runtime-tunable priorities.
REWARD_CFG = RewardConfig(
    enabled               = True,
    w_csat                = 0.35,
    w_resolution          = 0.25,
    w_service_level       = 0.20,   # answered within SLA (abandonment-averting)
    w_speed               = 0.10,
    w_burnout_penalty     = 0.10,
    sla_threshold_minutes = SLA_THRESHOLD_MINUTES,
)
FATIGUE_RESOLUTION_PENALTY = 1.2    # item 3a: fatigue erodes resolution

# Multi-factor shrinkage (item 2b): planned + unplanned availability losses.
SHRINKAGE_CFG = ShrinkageConfig(
    training=0.04, coaching=0.02, meetings=0.03, admin=0.02,    # planned  ~11%
    absence=0.03, sickness=0.03, no_show=0.02, tech_outage=0.01,  # unplanned ~9%
)

# Break-scheduling + operational workforce constraints for the CP-SAT model.
LABOUR_CFG = LabourConstraintConfig(
    shrinkage_config            = SHRINKAGE_CFG,
    enable_break_optimization   = True,
    break_slot_minutes          = 60,
    breaks_per_agent            = 2,        # one break + one lunch slot
    max_break_fraction_per_slot = 0.34,     # <=34% of a skill on break at once
    no_break_head_minutes       = 60.0,     # min continuous work before a break
    no_break_tail_minutes       = 60.0,     # no break within 60 min of shift end
    lunch_window                = (180, 540),
    break_balance_weight        = 0.04,     # spread breaks across slots
)

sim_cfg = SimulationConfig(
    sim_duration_minutes  = SIM_DURATION_MINUTES,
    arrival_rate_per_hour = ARRIVAL_RATE_PER_HOUR,
    random_seed           = SEED,
    warmup_minutes        = WARMUP_MINUTES,
    break_stagger_window  = BREAK_STAGGER_WINDOW,
    break_schedule        = [
        (_SHIFT_BREAK_MIDPOINTS["M"], _BREAK_DURATION_MINUTES),
        (_SHIFT_BREAK_MIDPOINTS["D"], _BREAK_DURATION_MINUTES),
        (_SHIFT_BREAK_MIDPOINTS["E"], _BREAK_DURATION_MINUTES),
    ],
)
cost_cfg = CostConfig()
beh_cfg  = BehaviorConfig()


def stage0_erlang_preview() -> None:
    section("STAGE 0 - Erlang-C Analytical Preview")
    lam   = sim_cfg.arrival_rate_per_minute
    svc   = sim_cfg.mean_service_minutes + sim_cfg.acw_mean_minutes
    total = sum(sim_cfg.skill_mix.values()) or 1.0
    print(f"  Arrival rate : {lam:.4f} calls/min  ({ARRIVAL_RATE_PER_HOUR:.0f}/hr)")
    print(f"  Mean svc+ACW : {svc:.2f} min")
    print(f"  SLA target   : {SLA_TARGET:.0%} answered within {SLA_THRESHOLD_MINUTES:.1f} min")
    print()
    print(f"  {'Skill':<14}  {'lam peak':>12}  {'Min agents':>11}  {'SLA @ min':>10}")
    divider()
    from optimization import _arrival_fraction_for_band
    for skill, frac in sim_cfg.skill_mix.items():
        skill_lam = lam * (frac / total)
        peak_lam  = skill_lam * _arrival_fraction_for_band(PEAK_BAND_INDEX)
        min_c     = ErlangC.min_agents_for_sla(peak_lam, svc, SLA_THRESHOLD_MINUTES, SLA_TARGET,
                                                max_c=MAX_AGENTS_PER_SKILL)
        sla_c     = ErlangC.sla_probability(min_c, peak_lam, svc, SLA_THRESHOLD_MINUTES)
        print(f"  {skill:<14}  {peak_lam:>12.4f}  {min_c:>11}  {sla_c:>9.1%}")


def stage1_milp_solve() -> dict:
    section("STAGE 1 - Single CP-SAT Solve (Analytical, Multi-Shift)")
    opt_cfg = OptimizationConfig(
        sla_target               = SLA_TARGET,
        sla_threshold_minutes    = SLA_THRESHOLD_MINUTES,
        max_agents_per_skill     = MAX_AGENTS_PER_SKILL,
        min_agents_per_skill     = MIN_AGENTS_PER_SHIFT,
        analytical_safety_margin = 0.0,
        max_iterations           = 1,
        engine_type              = "cost",
        random_seed              = SEED,
        verbose                  = False,
        arrival_rate_buffer      = 1.10,
        max_occupancy            = 0.85,
        labour                   = LABOUR_CFG,
    )
    optimizer = StaffingOptimizer(sim_cfg, opt_cfg, cost_cfg)
    result    = optimizer.solve()
    wages     = cost_cfg.hourly_wage_per_skill
    overhead  = cost_cfg.overhead_factor
    print(f"  Solver status  : {result.status}")
    print(f"  Solve time     : {result.solve_time_seconds*1000:.1f} ms")
    print(f"  Total cost     : £{result.total_staffing_cost:,.2f}")
    print(f"  Constraints    : {', '.join(result.constraints_applied)}")
    print_shift_plan(result, wages, overhead)
    return result.agents_per_skill


def stage2_optimize_simulate():
    section("STAGE 2 - Optimize -> Simulate Loop")
    opt_cfg = OptimizationConfig(
        sla_target                     = SLA_TARGET,
        sla_threshold_minutes          = SLA_THRESHOLD_MINUTES,
        max_agents_per_skill           = MAX_AGENTS_PER_SKILL,
        min_agents_per_skill           = MIN_AGENTS_PER_SHIFT,
        analytical_safety_margin       = None,
        max_iterations                 = MAX_OPT_ITERATIONS,
        convergence_tolerance          = CONVERGENCE_TOLERANCE,
        engine_type                    = "cost",
        random_seed                    = SEED,
        verbose                        = True,
        arrival_rate_buffer            = 1.10,
        max_occupancy                  = 0.85,
        sla_violation_penalty_per_call = 8.0,
        pareto_sweep                   = True,
        sim_feedback_penalty_scale     = 0.5,
        labour                         = LABOUR_CFG,
    )
    loop    = OptimizeSimulateLoop(sim_cfg, opt_cfg, cost_cfg)
    history = loop.run()
    loop.report.print_report()
    loop.print_pareto_summary()
    loop.explain_plan()
    best_plan  = loop.best_plan
    best_entry = loop._best_history_entry()
    opt_result = best_entry.opt_result if best_entry else None
    if opt_result is not None:
        CoverageValidator(opt_result, sim_cfg, opt_cfg).print_report()
    return best_plan, opt_result


def stage3_ml_feedback(optimised_cfg):
    section("STAGE 3 - ML Adaptive Feedback Loop")
    registry = build_registry(optimised_cfg, reward_cfg=REWARD_CFG,
                              fatigue_resolution_penalty=FATIGUE_RESOLUTION_PENALTY)
    fb_cfg = FeedbackConfig(
        n_epochs             = ML_EPOCHS,
        convergence_metric   = "sla",
        convergence_delta    = 0.002,
        convergence_patience = 2,
        retrain_csat         = True,
        retrain_fcr          = True,
        retrain_survival     = True,
        refit_forecaster     = False,
        verbose              = True,
    )
    loop    = FeedbackLoop(optimised_cfg, fb_cfg, registry)
    history = loop.run()
    registry.print_model_report()
    FeedbackLoopReport(history, fb_cfg).print_report()
    section("STAGE 3 - Agent Affinity Table (ML-learned)")
    registry.affinity.print_affinity_table()
    return registry


def stage4_full_stack_eval(optimised_cfg, registry):
    section("STAGE 4 - Full-Stack Evaluation  (Realism + Cost + ML)")
    print(f"  Peak-band staffing : {optimised_cfg.agents_per_skill}")
    engine = CostAwareRealisticsEngine(
        config   = optimised_cfg,
        cost_cfg = cost_cfg,
        behavior = beh_cfg,
    )
    engine.run()
    section("STAGE 4a - Full KPI Report")
    engine.kpi.report()
    section("STAGE 4a - Business Cost Breakdown")
    engine.cost_function.report("Optimised + Realism + Cost")
    section("STAGE 4a - Agent Performance Tracker")
    engine.router.tracker.print_summary()
    section("STAGE 4a - Human Realism Agent State")
    engine.realism.print_report()

    section("STAGE 4b - ML-Routed Engine (same optimised staffing)")
    ml_engine = MLSimulationEngine(optimised_cfg, registry)
    ml_engine.run()
    ml_engine.kpi.report()
    ml_engine.router.print_reward_report()
    registry.affinity.print_affinity_table()

    section("STAGE 4 - Side-by-Side: Realism vs ML Engine")
    print(f"  {'Metric':<30}  {'Realism Engine':>16}  {'ML Engine':>16}  {'Delta':>10}")
    divider()
    metrics = [
        ("SLA",                engine.kpi.sla_percentage(),           ml_engine.kpi.sla_percentage(),           "pct"),
        ("Abandonment rate",   engine.kpi.abandonment_rate(),         ml_engine.kpi.abandonment_rate(),         "pct"),
        ("Avg CSAT (1-5)",     engine.kpi.average_csat(),             ml_engine.kpi.average_csat(),             "f2"),
        ("ASA (min)",          engine.kpi.average_speed_of_answer(),  ml_engine.kpi.average_speed_of_answer(),  "f2"),
        ("AHT (min)",          engine.kpi.average_handle_time(),      ml_engine.kpi.average_handle_time(),      "f2"),
        ("FCR",                engine.kpi.first_call_resolution(),    ml_engine.kpi.first_call_resolution(),    "pct"),
        ("Total calls",        float(engine.kpi.total_calls()),       float(ml_engine.kpi.total_calls()),       "int"),
        ("Total abandonments", float(engine.kpi.total_abandonments()),float(ml_engine.kpi.total_abandonments()),"int"),
    ]
    for label, v_real, v_ml, fmt in metrics:
        delta = v_ml - v_real
        if fmt == "pct":
            print(f"  {label:<30}  {v_real:>16.1%}  {v_ml:>16.1%}  {delta:>+10.1%}")
        elif fmt == "f2":
            print(f"  {label:<30}  {v_real:>16.2f}  {v_ml:>16.2f}  {delta:>+10.2f}")
        else:
            print(f"  {label:<30}  {v_real:>16.0f}  {v_ml:>16.0f}  {delta:>+10.0f}")


def stage5_scenario_comparison(optimised_cfg):
    section("STAGE 5 - Scenario Comparison: Default vs Optimised Staffing")
    default_cfg = copy.copy(sim_cfg)
    print("  Running scenario A - DEFAULT staffing ...")
    eng_a = CostAwareRealisticsEngine(default_cfg, cost_cfg, beh_cfg)
    eng_a.run()
    res_a = ScenarioCostResult.from_engine("Default staffing", eng_a)
    print("  Running scenario B - OPTIMISED staffing ...")
    eng_b = CostAwareRealisticsEngine(optimised_cfg, cost_cfg, beh_cfg)
    eng_b.run()
    res_b = ScenarioCostResult.from_engine("Optimised staffing", eng_b)
    CostReport([res_a, res_b]).print_report()


def stage6_monte_carlo(optimised_cfg):
    section("STAGE 6 - Monte Carlo Risk Analysis  (multi-seed replication)")
    print(f"  Replications : {MONTE_CARLO_RUNS}  |  warm-up: {optimised_cfg.warmup_minutes:.0f} min"
          f"  |  engine: CostAwareRealisticsEngine")

    def factory(seed):
        c = copy.copy(optimised_cfg)
        c.random_seed = seed
        c.warmup_log  = False          # silence per-run warm-up logging
        return CostAwareRealisticsEngine(c, cost_cfg, beh_cfg)

    runner = MonteCarloRunner(factory, n_runs=MONTE_CARLO_RUNS, base_seed=1000,
                              rank_metric="sla", verbose=True)
    result = runner.run()
    result.print_report()
    return result


def stage7_surrogate(optimised_plan):
    section("STAGE 7 - Surrogate Model Warm-Up  (optimiser acceleration)")
    opt_cfg = OptimizationConfig(
        sla_target            = SLA_TARGET,
        sla_threshold_minutes = SLA_THRESHOLD_MINUTES,
        max_agents_per_skill  = MAX_AGENTS_PER_SKILL,
        min_agents_per_skill  = MIN_AGENTS_PER_SHIFT,
        engine_type           = "cost",
        random_seed           = SEED,
        verbose               = False,
    )
    ws  = SurrogateWarmStart(sim_cfg, opt_cfg, cost_cfg, beh_cfg)
    rep = ws.build_and_validate(center_plan=dict(optimised_plan),
                                n_train=14, n_test=6, seed=SEED)
    SurrogateWarmStart.print_report(rep)
    return rep


def main() -> None:
    t_start = time.perf_counter()
    section("CALL CENTRE STAFFING PIPELINE - FULL RUN", width=72)
    print(f"  SLA target   : {SLA_TARGET:.0%} within {SLA_THRESHOLD_MINUTES:.1f} min")
    print(f"  Arrival rate : {ARRIVAL_RATE_PER_HOUR:.0f} calls/hr")
    print(f"  Horizon      : {HORIZON_MINUTES} min   (warm-up {WARMUP_MINUTES:.0f} min)")
    print(f"  Shrinkage    : planned {SHRINKAGE_CFG.planned():.0%} + unplanned "
          f"{SHRINKAGE_CFG.unplanned():.0%} -> total {SHRINKAGE_CFG.total():.0%}")
    print(f"  Random seed  : {SEED}")

    stage0_erlang_preview()
    milp_plan = stage1_milp_solve()
    optimised_plan, best_opt_result = stage2_optimize_simulate()

    if best_opt_result is not None:
        optimised_cfg = build_optimised_sim_cfg(sim_cfg, best_opt_result)
    else:
        optimised_cfg = copy.copy(sim_cfg)
        optimised_cfg.agents_per_skill     = dict(optimised_plan)
        optimised_cfg.sim_duration_minutes = float(HORIZON_MINUTES)

    registry = stage3_ml_feedback(optimised_cfg)
    stage4_full_stack_eval(optimised_cfg, registry)
    stage5_scenario_comparison(optimised_cfg)
    stage6_monte_carlo(optimised_cfg)
    stage7_surrogate(optimised_plan)

    elapsed = time.perf_counter() - t_start
    section("PIPELINE COMPLETE", width=72)
    print(f"  Optimised peak-band plan : {optimised_plan}")
    print(f"  Total runtime            : {elapsed:.1f}s")


if __name__ == "__main__":
    main()