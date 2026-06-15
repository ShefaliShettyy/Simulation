# Workforce Management & Optimisation

A discrete-event simulation of a multi-skill contact centre, with a cost model,
an online-learning ML routing layer, a CP-SAT staffing optimiser, and a
"what-if" disruption / stress-testing layer.

The system answers three questions:

1. **How does a given roster behave?** — simulate arrivals, skill-based routing,
   queues, breaks, abandonment and first-call-resolution, then report SLA, cost
   and utilisation KPIs.
2. **What is the cost-optimal roster?** — a CP-SAT optimiser proposes staffing
   per skill and shift, which is then validated in simulation (optimise → simulate
   loop).
3. **Does that "optimal" roster survive the real world?** — inject demand spikes
   and agent outages and measure degradation and recovery in pre / impact /
   recovery windows.

---

## Directory structure

```
call-center-wfm/
├── README.md
├── requirements.txt
├── .gitignore
├── run.py                      # end-to-end demo / entry point
├── callcenter/                 # the library (importable modules)
│   ├── core_simulation.py        # SimPy engine, config, KPIs, agents, routing
│   ├── cost_system.py            # business-cost layer on top of the sim
│   ├── ml_system.py              # LinUCB router, predictors, feedback loop
│   ├── eda.py                    # data-quality gate before model training
│   ├── hierarchical_overflow.py  # wait-time-driven pool escalation engine (NOT_COMPLETED)
│   ├── optimization.py           # CP-SAT staffing optimiser 
│   ├── optimization_ortools.py   # same optimiser, OR-Tools backend (free)
│   ├── disruption_injector.py    # demand/supply shocks + windowed KPI analyser(NOT_COMPLETED)
│   └── stress_test.py            # ties disruptions into the optimise→simulate pipeline
└── tests/
    ├── conftest.py             # makes `callcenter/` importable under pytest (NOT_COMPLETED)
    └── test_all.py             # smoke tests across all engines + full pipeline(NOT_COMPLETED)
```

The library modules live flat inside `callcenter/` and import each other by
plain module name (`from core_simulation import ...`). This is intentional — it
keeps the engine code unchanged from the original. The entry points (`run.py`,
`tests/test_all.py`) add `callcenter/` to the import path automatically, so no
installation step is required.

---

## Architecture

The modules form clear layers; each builds on the one below without modifying it.

| Layer | Module | Responsibility |
|-------|--------|----------------|
| **Core** | `core_simulation.py` | Discrete-event SimPy model: arrivals, skill mix, customer tiers, queues, breaks, KPIs. Defines `SimulationConfig`, `SimulationEngine`, `RealisticsAwareEngine`, `MonteCarloRunner`. |
| **Cost** | `cost_system.py` | Wraps an engine to attach wages, overhead, abandonment/SLA penalties and churn. `CostAwareEngine`, `CostConfig`, `CostReport`. |
| **ML** | `ml_system.py` | A LinUCB contextual bandit makes routing decisions and learns online; predictors (CSAT / resolution / abandonment) feed it as features. `MLSimulationEngine`, `MLRegistry`, `FeedbackLoop`, `build_registry`. |
| **Data quality** | `eda.py` | Gates training data: leakage guard, zero-variance / collinearity checks, class balance, cross-epoch drift. `EDALayer`, `DatasetDiagnostics`. |
| **Overflow** | `hierarchical_overflow.py` | An operational escape valve: the longer a call waits, the wider the agent pool allowed to serve it. `HierarchicalOverflowEngine`, `HierarchicalRouter`, `TierConfig`. |
| **Optimisation** | `optimization.py` / `optimization_ortools.py` | CP-SAT/MIP staffing optimiser plus the optimise→simulate loop. `StaffingOptimizer`, `OptimizationConfig`, `OptimizeSimulateLoop`, `SimulationEvaluator`, `ErlangC`. |
| **Stress test** | `disruption_injector.py` + `stress_test.py` | Bolts demand/supply shocks onto *any* engine and isolates KPIs into pre / impact / recovery windows. `stress_test_plan`, `stress_test_optimized`. |



---

## Installation

Python 3.9+ is recommended.

```bash
# (optional) create a virtual environment
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

This installs the simulation, cost, ML, EDA, hierarchical and stress-test
functionality. `scipy`, `scikit-learn`, `pandas` and `matplotlib` are optional —
the code falls back to closed-form behaviour if any are missing — but they are
included so the ML and reporting layers run at full fidelity.


## Running it

**Full end-to-end demo** (needs a solver backend for the optimiser stages):

```bash
python run.py
```



---




