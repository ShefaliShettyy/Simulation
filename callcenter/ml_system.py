from __future__ import annotations

import copy
import math
import random
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.model_selection import train_test_split
    _SKL = True
except ImportError:                                   # pragma: no cover
    _SKL = False

try:
    from scipy.stats import weibull_min
    _SCIPY = True
except ImportError:                                   # pragma: no cover
    _SCIPY = False

try:
    from scipy.optimize import minimize as _scipy_minimize
    _SCIPY_OPT = True
except ImportError:                                   # pragma: no cover
    _SCIPY_OPT = False

from core_simulation import (
    Agent, Call, Router, RouterScoreWeights,
    SimulationConfig, SimulationEngine, _CallRecord,
)
from eda import DatasetDiagnostics, EDALayer

warnings.filterwarnings("ignore", message="X does not have valid feature names")


# ===========================================================================
# 0. ENCODING PRIMITIVES
# ===========================================================================

_SKILL_IX = {"billing": 0, "technical": 1, "general": 2}
_TIER_IX  = {"standard": 0, "premium": 1, "vip": 2}
_EXP_IX   = {"junior": 0, "mid": 1, "senior": 2}


def _skill_ix(s: str) -> int: return _SKILL_IX.get(s, 0)
def _tier_ix(s: str)  -> int: return _TIER_IX.get(s, 0)
def _exp_ix(s: str)   -> int: return _EXP_IX.get(s, 1)


def _complexity_bucket(handle_minutes: float) -> str:
    if handle_minutes < 4.0:
        return "simple"
    if handle_minutes < 9.0:
        return "medium"
    return "complex"


def _sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z)))))


# ===========================================================================
# 1. FEATURE SCHEMA  (the single source of truth)
# ===========================================================================

@dataclass
class DecisionContext:
    """Everything observable at the moment an agent is chosen for a call.

    Holds raw (un-normalised) values so it is debuggable and serialisable. The
    agent's latent `csat_bias` is deliberately absent — that is what the models
    must *infer* from observable rolling statistics.
    """
    tier:            str
    skill:           str
    is_repeat:       bool
    exp:             str
    skill_match:     float        # 1.0 primary skill, 0.5 overflow/secondary
    ema_csat:        float
    ema_handle:      float
    ema_resolution:  float
    calls_completed: int
    active_calls:    int
    affinity:        float
    est_wait:        float
    queue_depth:     int
    sim_time:        float
    horizon:         float
    fatigue:         float = 0.0
    burnout:         float = 0.0


class FeatureSchema:
    """Encodes a DecisionContext into a fixed, named vector.

    `encode_base` is used to train AND query the per-outcome predictors (CSAT,
    FCR). `encode_policy` appends the predictors' own outputs for the bandit, so
    the policy can learn to weight them. Because the SAME method produces the
    vector in both phases, train/serve skew is impossible by construction.
    """

    BASE_NAMES: ClassVar[Tuple[str, ...]] = (
        "tier", "skill", "is_repeat", "exp", "skill_match",
        "ema_csat", "ema_handle_norm", "ema_resolution", "reliability",
        "active_norm", "affinity", "wait_norm", "queue_norm",
        "sim_progress", "fatigue", "burnout",
    )
    PRED_NAMES: ClassVar[Tuple[str, ...]] = ("p_csat", "p_resolve", "p_abandon")

    BASE_DIM:   ClassVar[int] = len(BASE_NAMES)
    POLICY_DIM: ClassVar[int] = BASE_DIM + len(PRED_NAMES)

    _MAX_HANDLE = 15.0
    _MAX_ACTIVE = 5.0
    _MAX_WAIT   = 10.0
    _REL_CALLS  = 10.0

    def encode_base(self, ctx: DecisionContext) -> np.ndarray:
        return np.array([
            _tier_ix(ctx.tier) / 2.0,
            _skill_ix(ctx.skill) / 2.0,
            1.0 if ctx.is_repeat else 0.0,
            _exp_ix(ctx.exp) / 2.0,
            ctx.skill_match,
            float(np.clip(ctx.ema_csat, 0.0, 1.0)),
            float(np.clip(1.0 - ctx.ema_handle / self._MAX_HANDLE, 0.0, 1.0)),
            float(np.clip(ctx.ema_resolution, 0.0, 1.0)),
            float(min(1.0, ctx.calls_completed / self._REL_CALLS)),
            float(np.clip(1.0 - ctx.active_calls / self._MAX_ACTIVE, 0.0, 1.0)),
            float(np.clip(ctx.affinity, 0.0, 1.0)),
            float(np.clip(1.0 - ctx.est_wait / self._MAX_WAIT, 0.0, 1.0)),
            float(min(1.0, ctx.queue_depth / 10.0)),
            float(np.clip(ctx.sim_time / max(1.0, ctx.horizon), 0.0, 1.0)),
            float(np.clip(ctx.fatigue, 0.0, 1.0)),
            float(np.clip(ctx.burnout, 0.0, 1.0)),
        ], dtype=float)

    def encode_policy(self, ctx: DecisionContext, preds: Sequence[float]) -> np.ndarray:
        return np.concatenate([self.encode_base(ctx),
                               np.asarray(preds, dtype=float).ravel()])


# ===========================================================================
# 2. PREDICTORS  (probability of a binary outcome from base features)
# ===========================================================================

class BinaryPredictor:
    """Logistic predictor with a hand-tuned warm prior fallback.

    Uses sklearn LogisticRegression when available; otherwise a fixed-weight
    logistic so the system is fully functional with numpy alone. Exposes a
    `staged` model so the evaluator can validate a candidate fit before it is
    committed with `commit`.
    """

    def __init__(self, name: str, warm_w: np.ndarray, warm_b: float) -> None:
        self.name     = name
        self._w       = warm_w.astype(float)
        self._b       = float(warm_b)
        self._model: Optional[Any] = None
        self._staged: Optional[Any] = None
        self.n_train  = 0

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit_staged(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Fit a candidate model without committing it. Returns True on success."""
        if not _SKL or len(X) < 20 or len(np.unique(y)) < 2:
            return False
        m = LogisticRegression(max_iter=300, C=1.0)
        m.fit(X, y)
        self._staged = m
        return True

    def commit(self) -> None:
        if self._staged is not None:
            self._model = self._staged
            self._staged = None

    def predict(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float).reshape(1, -1)
        if self._model is not None:
            try:
                return float(self._model.predict_proba(x)[0, 1])
            except Exception:
                pass
        return _sigmoid(float(np.dot(x.ravel(), self._w)) + self._b)

    def predict_staged(self, X: np.ndarray) -> np.ndarray:
        if self._staged is not None:
            return self._staged.predict_proba(X)[:, 1]
        return np.array([self.predict(row) for row in X])


def _make_csat_predictor() -> BinaryPredictor:
    # Priors over BASE_NAMES order: reward higher ema_csat/affinity/exp,
    # penalise repeats and long handle.
    w = np.zeros(FeatureSchema.BASE_DIM)
    w[3], w[5], w[6], w[7], w[10] = 0.4, 1.6, 0.6, 0.5, 1.0   # exp, csat, hdl, res, affinity
    w[2] = -0.5                                               # is_repeat
    return BinaryPredictor("csat>=4", w, -0.6)


def _make_fcr_predictor() -> BinaryPredictor:
    w = np.zeros(FeatureSchema.BASE_DIM)
    w[3], w[4], w[7], w[10] = 0.6, 0.8, 1.2, 0.9             # exp, match, res, affinity
    w[2] = -0.6                                              # is_repeat
    return BinaryPredictor("resolved", w, 0.2)


class SurvivalAbandonment:
    """Per-tier Weibull patience model.

    Provides both the abandonment CDF (for the policy feature) and a patience
    *sampler* the engine uses to make abandonment a genuine, learnable outcome.
    """

    _DEFAULTS: ClassVar[Dict[str, Tuple[float, float]]] = {
        "standard": (1.4, 4.0), "premium": (1.5, 6.0), "vip": (1.6, 9.0),
    }

    def __init__(self) -> None:
        self._params = dict(self._DEFAULTS)
        self.fitted  = False

    def fit(self, waits: List[float], abandoned: List[int], tiers: List[str]) -> None:
        """Fit per-tier Weibull patience with RIGHT-CENSORING.

        Abandoned calls give an exact patience (event at wait = patience).
        Answered calls are right-censored: their true patience is only known to
        exceed the wait they experienced before being served. Using only the
        abandoned (impatient) tail biases the scale downward and, in a feedback
        loop, makes patience collapse epoch over epoch. Censoring on the answered
        calls anchors the estimate and removes that runaway.
        """
        for tier in set(tiers):
            idx = [i for i in range(len(tiers)) if tiers[i] == tier and waits[i] > 0]
            if not idx:
                continue
            t = np.array([waits[i] for i in idx], dtype=float)
            e = np.array([abandoned[i] for i in idx], dtype=float)  # 1=event, 0=censored
            if int(e.sum()) < 8:                       # too few real events to fit
                continue
            self._params[tier] = self._fit_weibull_censored(t, e)
        self.fitted = True

    @staticmethod
    def _fit_weibull_censored(t: np.ndarray, e: np.ndarray) -> Tuple[float, float]:
        ev = t[e == 1]
        mu = float(ev.mean())
        sd = float(ev.std()) + 1e-9
        k0   = float(np.clip((sd / mu + 0.01) ** -1.086, 0.6, 4.0))
        lam0 = float(max(0.5, mu / math.gamma(1.0 + 1.0 / k0)))
        if not _SCIPY_OPT:
            return float(np.clip(k0, 0.5, 5.0)), float(np.clip(lam0, 0.3, 60.0))
        logt = np.log(t)

        def nll(p: np.ndarray) -> float:
            k, lam = p
            if k <= 1e-3 or lam <= 1e-3:
                return 1e12
            z = (t / lam) ** k
            ll = (np.sum(e * (math.log(k) - math.log(lam) + (k - 1.0) * (logt - math.log(lam))))
                  - np.sum(z))                          # events contribute log f, all contribute -z (=log S)
            return float(-ll)

        try:
            res = _scipy_minimize(nll, np.array([k0, lam0]), method="Nelder-Mead",
                                  options=dict(maxiter=400, xatol=1e-3, fatol=1e-3))
            k, lam = float(res.x[0]), float(res.x[1])
        except Exception:
            k, lam = k0, lam0
        if not (math.isfinite(k) and math.isfinite(lam)):
            k, lam = k0, lam0
        return float(np.clip(k, 0.5, 5.0)), float(np.clip(lam, 0.3, 60.0))

    def abandon_prob(self, wait: float, tier: str) -> float:
        k, scale = self._params.get(tier, self._params["standard"])
        if _SCIPY:
            return float(weibull_min.cdf(max(0.0, wait), c=k, scale=scale))
        return 1.0 - math.exp(-(max(0.0, wait) / scale) ** k)

    def sample_patience(self, tier: str, rng: random.Random) -> float:
        k, scale = self._params.get(tier, self._params["standard"])
        u = max(1e-9, rng.random())
        return float(scale * (-math.log(u)) ** (1.0 / k))   # inverse-CDF Weibull


# ===========================================================================
# 3. CONTEXTUAL AFFINITY  (shrinkage EMA: context cell -> skill -> prior)
# ===========================================================================

class AffinityModel:
    ALPHA: ClassVar[float] = 0.20

    def __init__(self) -> None:
        self._ctx:   Dict[Tuple[str, str, str, str], float] = {}
        self._skill: Dict[Tuple[str, str], float]           = {}

    def update(self, agent_id: str, skill: str, csat_raw: float, handle: float,
               resolved: bool, tier: str, queue_depth: int) -> None:
        csat_norm  = (csat_raw - 1.0) / 4.0
        speed      = max(0.0, 1.0 - handle / 15.0)
        pressure   = min(0.10, queue_depth * 0.01)
        signal     = float(np.clip(0.45 * csat_norm + 0.25 * speed
                                   + 0.20 * (1.0 if resolved else 0.0) - pressure, 0.0, 1.0))
        cplx       = _complexity_bucket(handle)
        ck = (agent_id, skill, tier, cplx)
        sk = (agent_id, skill)
        self._ctx[ck]   = self.ALPHA * signal + (1 - self.ALPHA) * self._ctx.get(ck, 0.5)
        self._skill[sk] = self.ALPHA * signal + (1 - self.ALPHA) * self._skill.get(sk, 0.5)

    def get(self, agent_id: str, skill: str, tier: str, complexity: str) -> float:
        v = self._ctx.get((agent_id, skill, tier, complexity))
        if v is not None:
            return v
        return self._skill.get((agent_id, skill), 0.5)

    def print_affinity_table(self, top: int = 12) -> None:
        print("\n  Agent affinity (ML-learned, strongest cells)")
        print(f"  {'Agent':<10}{'Skill':<12}{'Tier':<10}{'Complexity':<11}{'Affinity':>9}")
        print("  " + "-" * 50)
        cells = sorted(self._ctx.items(), key=lambda kv: -kv[1])[:top]
        if not cells:
            print("    (no affinity learned yet)")
            return
        for (aid, skill, tier, cplx), v in cells:
            print(f"  {aid:<10}{skill:<12}{tier:<10}{cplx:<11}{v:>9.3f}")


# ===========================================================================
# 4. DECISION LOG  (joins decision features to realised outcomes -> real data)
# ===========================================================================

@dataclass
class _OutcomeRow:
    base:     np.ndarray
    csat_ok:  int
    resolved: int


class DecisionLog:
    """Accumulates real (features, label) rows produced during a run."""

    def __init__(self) -> None:
        self.outcomes:      List[_OutcomeRow] = []
        self.abn_features:  List[np.ndarray]  = []   # [tier, skill, is_repeat, wait, qdepth, sim_t]
        self.abn_labels:    List[int]         = []
        self.abn_waits:     List[float]       = []
        self.abn_tiers:     List[str]         = []

    def log_outcome(self, base: np.ndarray, csat_raw: float, resolved: bool) -> None:
        self.outcomes.append(_OutcomeRow(base, int(csat_raw >= 4.0), int(resolved)))

    def log_queue_result(self, tier: str, skill: str, is_repeat: bool,
                         wait: float, queue_depth: int, sim_t: float,
                         abandoned: bool) -> None:
        self.abn_features.append(np.array(
            [_tier_ix(tier) / 2.0, _skill_ix(skill) / 2.0, 1.0 if is_repeat else 0.0,
             min(1.0, wait / 10.0), min(1.0, queue_depth / 10.0), sim_t / 720.0],
            dtype=float))
        self.abn_labels.append(int(abandoned))
        self.abn_waits.append(wait)
        self.abn_tiers.append(tier)

    # --- dataset builders (all from REAL logged rows) ----------------------

    def csat_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.outcomes:
            return np.empty((0, FeatureSchema.BASE_DIM)), np.empty(0, dtype=int)
        X = np.vstack([r.base for r in self.outcomes])
        y = np.array([r.csat_ok for r in self.outcomes], dtype=int)
        return X, y

    def fcr_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.outcomes:
            return np.empty((0, FeatureSchema.BASE_DIM)), np.empty(0, dtype=int)
        X = np.vstack([r.base for r in self.outcomes])
        y = np.array([r.resolved for r in self.outcomes], dtype=int)
        return X, y

    def survival_dataset(self) -> Tuple[List[float], List[int], List[str]]:
        return list(self.abn_waits), list(self.abn_labels), list(self.abn_tiers)


# ===========================================================================
# 5. LINUCB CONTEXTUAL BANDIT  (the real policy)
# ===========================================================================

class LinUCBPolicy:
    """Disjoint LinUCB: one ridge model per arm (agent), UCB exploration.

    Selects the candidate agent maximising  theta·x + alpha·sqrt(xᵀA⁻¹x),
    and updates the chosen arm with the realised reward. A⁻¹ is maintained
    incrementally via the Sherman–Morrison formula.
    """

    def __init__(self, dim: int, alpha: float = 0.6, l2: float = 1.0) -> None:
        self.dim    = dim
        self.alpha  = alpha
        self._l2    = l2
        self._Ainv: Dict[str, np.ndarray] = {}
        self._b:    Dict[str, np.ndarray] = {}
        self.pulls: Dict[str, int]        = {}

    def _arm(self, arm: str) -> Tuple[np.ndarray, np.ndarray]:
        if arm not in self._Ainv:
            self._Ainv[arm] = np.eye(self.dim) / self._l2
            self._b[arm]    = np.zeros(self.dim)
            self.pulls[arm] = 0
        return self._Ainv[arm], self._b[arm]

    def _ucb(self, arm: str, x: np.ndarray) -> float:
        Ainv, b = self._arm(arm)
        theta   = Ainv @ b
        mean    = float(theta @ x)
        bonus   = self.alpha * math.sqrt(max(0.0, float(x @ Ainv @ x)))
        return mean + bonus

    def select(self, arms: Sequence[str], x: np.ndarray) -> str:
        return max(arms, key=lambda a: self._ucb(a, x))

    def update(self, arm: str, x: np.ndarray, reward: float) -> None:
        Ainv, b = self._arm(arm)
        Ax      = Ainv @ x
        denom   = 1.0 + float(x @ Ax)
        self._Ainv[arm] = Ainv - np.outer(Ax, Ax) / denom    # Sherman–Morrison
        self._b[arm]    = b + reward * x
        self.pulls[arm] += 1


# ===========================================================================
# 6. BANDIT ROUTER  (Router subclass that the engine drives)
# ===========================================================================

@dataclass
class RewardConfig:
    """Configurable multi-objective reward for the routing bandit.

    The realised reward for an answered call is a weighted blend of normalised
    objective components, all in [0,1] except burnout which enters as a penalty:

        r = clip( w_csat·csat_n + w_res·resolved + w_sla·served_in_sla
                  + w_speed·speed_n − w_burnout·burnout , 0, 1 )

    where csat_n=(csat−1)/4, served_in_sla = 1[wait ≤ sla_threshold],
    speed_n = max(0, 1 − handle/_MAX_HANDLE).  Weights are runtime-mutable so the
    operating point (service-level vs satisfaction vs sustainability) can be
    shifted between epochs. `customer abandonment` enters through w_sla / w_speed:
    rewarding fast in-SLA service is what reduces downstream queueing and
    abandonment (the served call itself, by definition, did not abandon).
    """
    enabled:               bool  = True
    w_csat:                float = 0.40   # customer satisfaction
    w_resolution:          float = 0.30   # first-call resolution
    w_service_level:       float = 0.15   # answered within SLA (abandonment-averting)
    w_speed:               float = 0.10   # short handle time (throughput)
    w_burnout_penalty:     float = 0.10   # sustainability penalty
    sla_threshold_minutes: float = 1.0
    _MAX_HANDLE:           float = 15.0

    def components(self, csat_raw, resolved, wait, handle, burnout) -> Dict[str, float]:
        return {
            "csat":          self.w_csat * (csat_raw - 1.0) / 4.0,
            "resolution":    self.w_resolution * (1.0 if resolved else 0.0),
            "service_level": self.w_service_level * (1.0 if wait <= self.sla_threshold_minutes else 0.0),
            "speed":         self.w_speed * max(0.0, 1.0 - handle / self._MAX_HANDLE),
            "burnout":      -self.w_burnout_penalty * burnout,
        }


class BanditRouter(Router):
    """Routing policy: predictors -> context features -> LinUCB -> decision.

    The engine calls `select_agent` to choose an agent (and obtain the policy
    feature vector + base feature vector), then `learn` once the outcome is
    known. `pick_agent` is overridden for compatibility with the base engine
    (greedy, no exploration) but the ML engine uses `select_agent`.
    """

    REWARD_W = dict(csat=0.50, resolved=0.40, speed=0.10, burnout=0.10)   # legacy fallback

    def __init__(self, agent_pool, config, weights, registry: "MLRegistry") -> None:
        super().__init__(agent_pool, config, weights)
        self.reg    = registry
        self.schema = registry.schema
        self.policy = registry.policy
        self.reward_cfg = getattr(registry, "reward_cfg", None) or RewardConfig()
        # reward decomposition accumulators (means reported per component)
        self.reward_sums: Dict[str, float] = {}
        self.reward_n:    int   = 0
        self.reward_total_sum: float = 0.0
        self.reward_trend: List[float] = []

    # -- context construction ----------------------------------------------

    def _context(self, call: Call, agent: Agent, skill_resources) -> DecisionContext:
        st       = self.tracker.get(agent.agent_id)
        resource = skill_resources.get(call.skill)
        est_wait = self.estimated_wait(resource) if resource is not None else 5.0
        qdepth   = len(resource.queue) if resource is not None else 0
        if call.skill == agent.primary_skill:
            match = 1.0
        elif call.skill in agent.secondary_skills:
            match = 0.5
        else:
            match = 0.25
        cplx     = _complexity_bucket(st.ema_handle_time)
        affinity = self.reg.affinity.get(agent.agent_id, call.skill, call.customer_type, cplx)
        fatigue  = self.reg.fatigue.get(agent.agent_id)
        burnout  = self.reg.fatigue.burnout_risk(agent.agent_id)
        return DecisionContext(
            tier=call.customer_type, skill=call.skill, is_repeat=call.is_repeat,
            exp=agent.experience, skill_match=match,
            ema_csat=st.ema_csat, ema_handle=st.ema_handle_time,
            ema_resolution=st.ema_resolution, calls_completed=st.calls_completed,
            active_calls=st.active_calls, affinity=affinity, est_wait=est_wait,
            queue_depth=qdepth, sim_time=getattr(call, "arrival_time", 0.0),
            horizon=self.config.sim_duration_minutes, fatigue=fatigue, burnout=burnout,
        )

    def _policy_features(self, ctx: DecisionContext) -> Tuple[np.ndarray, np.ndarray]:
        base   = self.schema.encode_base(ctx)
        p_csat = self.reg.csat.predict(base)
        p_res  = self.reg.fcr.predict(base)
        p_abn  = self.reg.abandon.abandon_prob(ctx.est_wait, ctx.tier)
        x      = self.schema.encode_policy(ctx, [p_csat, p_res, p_abn])
        return x, base

    # -- selection + learning ----------------------------------------------

    def select_agent(self, call: Call, candidates: List[Agent], skill_resources):
        feats = {a.agent_id: self._policy_features(self._context(call, a, skill_resources))
                 for a in candidates}
        # Each arm is scored on its OWN context vector, so argmax UCB directly
        # (LinUCBPolicy.select assumes a shared context; here contexts differ).
        best_id, best_ucb = None, -float("inf")
        for a in candidates:
            x, _ = feats[a.agent_id]
            u = self.policy._ucb(a.agent_id, x)
            if u > best_ucb:
                best_ucb, best_id = u, a.agent_id
        chosen = next(a for a in candidates if a.agent_id == best_id)
        x, base = feats[best_id]
        return chosen, x, base

    def learn(self, agent_id: str, x: np.ndarray, csat_raw: float,
              resolved: bool, handle: float, burnout: float,
              wait: float = 0.0) -> float:
        rc = self.reward_cfg
        if rc.enabled:
            parts  = rc.components(csat_raw, resolved, wait, handle, burnout)
            reward = float(np.clip(sum(parts.values()), 0.0, 1.0))
            # accumulate decomposition
            for k, v in parts.items():
                self.reward_sums[k] = self.reward_sums.get(k, 0.0) + v
        else:                                           # legacy reward
            w = self.REWARD_W
            reward = float(np.clip(
                w["csat"] * (csat_raw - 1.0) / 4.0
                + w["resolved"] * (1.0 if resolved else 0.0)
                + w["speed"] * max(0.0, 1.0 - handle / 15.0)
                - w["burnout"] * burnout, 0.0, 1.0))
        self.reward_n         += 1
        self.reward_total_sum += reward
        self.reward_trend.append(reward)
        self.policy.update(agent_id, x, reward)
        return reward

    def reward_decomposition(self) -> Dict[str, float]:
        """Mean per-component contribution to reward across all learned calls."""
        if self.reward_n == 0:
            return {}
        out = {k: v / self.reward_n for k, v in self.reward_sums.items()}
        out["TOTAL"] = self.reward_total_sum / self.reward_n
        return out

    def print_reward_report(self) -> None:
        dec = self.reward_decomposition()
        w = 56
        print(f"\n{'=' * w}\n  MULTI-OBJECTIVE REWARD DECOMPOSITION\n{'=' * w}")
        if not dec:
            print("  (no learning events recorded)")
            print('=' * w)
            return
        rc = self.reward_cfg
        wmap = {"csat": rc.w_csat, "resolution": rc.w_resolution,
                "service_level": rc.w_service_level, "speed": rc.w_speed,
                "burnout": rc.w_burnout_penalty}
        total = dec.get("TOTAL", 0.0) or 1e-9
        print(f"  Reward config: enabled={rc.enabled}  (n={self.reward_n} calls)")
        print(f"  {'Component':<16}{'Weight':>8}{'MeanContrib':>14}{'ShareOfReward':>15}")
        print(f"  {'-' * (w - 4)}")
        for k in ("csat", "resolution", "service_level", "speed", "burnout"):
            if k in dec:
                share = dec[k] / total
                print(f"  {k:<16}{wmap.get(k, 0):>8.2f}{dec[k]:>14.4f}{share:>14.1%}")
        print(f"  {'-' * (w - 4)}")
        print(f"  {'TOTAL mean reward':<16}{'':>8}{dec['TOTAL']:>14.4f}")
        # simple trend: mean of first vs last third
        tr = self.reward_trend
        if len(tr) >= 6:
            k = len(tr) // 3
            early = sum(tr[:k]) / k
            late  = sum(tr[-k:]) / k
            arrow = "up" if late > early else ("down" if late < early else "flat")
            print(f"  Trend (first third -> last third): {early:.3f} -> {late:.3f} ({arrow})")
        print('=' * w)

    def pick_agent(self, call, candidates, skill_resources):     # base-engine path
        if not candidates:
            return None
        chosen, _, _ = self.select_agent(call, candidates, skill_resources)
        return chosen


# ===========================================================================
# 7. MODEL EVALUATOR  (held-out AUC/Brier, gated commit)
# ===========================================================================

@dataclass
class EvalReport:
    name:      str
    auc:       float
    brier:     float
    n:         int
    committed: bool


class ModelEvaluator:
    """Fits a candidate on a train split, scores it on held-out data, and only
    commits the new model if it beats the incumbent (or there is no incumbent)."""

    def __init__(self, test_frac: float = 0.25, min_auc_gain: float = -0.01) -> None:
        self._test_frac = test_frac
        self._gain      = min_auc_gain

    def evaluate_and_gate(self, predictor: BinaryPredictor,
                          X: np.ndarray, y: np.ndarray) -> EvalReport:
        if not _SKL or len(X) < 40 or len(np.unique(y)) < 2:
            return EvalReport(predictor.name, float("nan"), float("nan"), len(X), False)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=self._test_frac, random_state=42, stratify=y)
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            return EvalReport(predictor.name, float("nan"), float("nan"), len(X), False)

        # Discard any candidate left staged by a previous (rejected) round, so
        # the incumbent is always measured on the COMMITTED model (or the warm
        # prior if nothing has been committed yet) — never on a rejected candidate.
        predictor._staged = None
        incumbent_auc = roc_auc_score(yte, predictor.predict_staged(Xte))
        if not predictor.fit_staged(Xtr, ytr):
            return EvalReport(predictor.name, float("nan"), float("nan"), len(X), False)
        p_cand   = predictor.predict_staged(Xte)
        cand_auc = roc_auc_score(yte, p_cand)
        brier    = brier_score_loss(yte, p_cand)
        commit   = cand_auc >= incumbent_auc + self._gain
        if commit:
            predictor.commit()
        else:
            predictor._staged = None        # reject: do not let it linger
        return EvalReport(predictor.name, float(cand_auc), float(brier), len(X), commit)


# ===========================================================================
# 8. FATIGUE TRACKER  (consistent accumulate + recover, used by policy)
# ===========================================================================

class FatigueTracker:
    def __init__(self, acc: float = 0.0015, rec: float = 0.05,
                 threshold: float = 0.75, slope: float = 10.0) -> None:
        self._acc = acc
        self._rec = rec
        self._thr = threshold
        self._slp = slope
        self._f:    Dict[str, float] = {}
        self.peak:  Dict[str, float] = {}

    def reset(self) -> None:
        self._f.clear()
        self.peak.clear()

    def get(self, agent_id: str) -> float:
        return self._f.get(agent_id, 0.0)

    def accumulate(self, agent_id: str, minutes: float) -> None:
        nf = min(1.0, self._f.get(agent_id, 0.0) + self._acc * minutes)
        self._f[agent_id]  = nf
        self.peak[agent_id] = max(self.peak.get(agent_id, 0.0), nf)

    def recover(self, agent_id: str, minutes: float) -> None:
        self._f[agent_id] = max(0.0, self._f.get(agent_id, 0.0) * math.exp(-self._rec * minutes))

    def burnout_risk(self, agent_id: str) -> float:
        return _sigmoid(self._slp * (self.get(agent_id) - self._thr))


# ===========================================================================
# 9. ARRIVAL FORECASTER  (now wired into the engine)
# ===========================================================================

class ArrivalForecaster:
    """Smooth intraday multiplier curve; produces a rate_fn the engine consumes.

    A real deployment would fit this on historical interval counts; here it uses
    a daily-shape prior that the feedback loop can refine from observed counts.
    """

    N_INTERVALS: ClassVar[int] = 24

    def __init__(self, base_rate_per_min: float) -> None:
        self.base = base_rate_per_min
        self._mult = self._default_shape()

    def _default_shape(self) -> np.ndarray:
        t = np.linspace(0, 1, self.N_INTERVALS)
        # double-humped day: mid-morning and mid-afternoon peaks
        shape = 0.7 + 0.5 * np.exp(-((t - 0.30) ** 2) / 0.02) \
                    + 0.6 * np.exp(-((t - 0.70) ** 2) / 0.03)
        return shape / shape.mean()

    def observe(self, counts_per_min: np.ndarray) -> None:
        if counts_per_min.size == self.N_INTERVALS and counts_per_min.mean() > 0:
            obs = counts_per_min / counts_per_min.mean()
            self._mult = 0.7 * self._mult + 0.3 * obs

    def rate_fn(self, horizon: float) -> Callable[[float], float]:
        step = horizon / self.N_INTERVALS

        def fn(t: float) -> float:
            idx = min(self.N_INTERVALS - 1, int(t // step))
            return max(1e-6, self.base * float(self._mult[idx]))
        return fn


# ===========================================================================
# 10. ML REGISTRY  (facade over all learnable state)
# ===========================================================================

class MLRegistry:
    def __init__(self, base_rate_per_min: float = 2.0,
                 policy_alpha: float = 0.6,
                 reward_cfg: "Optional[RewardConfig]" = None,
                 fatigue_resolution_penalty: float = 0.0) -> None:
        self.schema    = FeatureSchema()
        self.csat      = _make_csat_predictor()
        self.fcr       = _make_fcr_predictor()
        self.abandon   = SurvivalAbandonment()
        self.affinity  = AffinityModel()
        self.fatigue   = FatigueTracker()
        self.forecaster = ArrivalForecaster(base_rate_per_min)
        self.policy    = LinUCBPolicy(FeatureSchema.POLICY_DIM, alpha=policy_alpha)
        self.evaluator = ModelEvaluator()
        self.eda       = EDALayer()
        self.log       = DecisionLog()
        self.reward_cfg = reward_cfg or RewardConfig()
        # how strongly real-time fatigue erodes resolution probability (item 3a)
        self.fatigue_resolution_penalty = float(fatigue_resolution_penalty)

    def fresh_log(self) -> None:
        self.log = DecisionLog()

    def make_router(self, agent_pool, config, weights=None) -> BanditRouter:
        return BanditRouter(agent_pool, config, weights, self)

    def print_model_report(self) -> None:
        w = 66
        print("\n" + "=" * w)
        print("  ML MODEL REGISTRY -- STATE REPORT")
        print("=" * w)
        print(f"  CSAT predictor   : fitted={self.csat.is_fitted}")
        print(f"  FCR  predictor   : fitted={self.fcr.is_fitted}")
        print(f"  Survival model   : fitted={self.abandon.fitted}")
        for tier, (k, scale) in self.abandon._params.items():
            print(f"      {tier:<9} Weibull(k={k:.2f}, scale={scale:.2f})")
        if self.policy.pulls:
            total = sum(self.policy.pulls.values())
            print(f"  Bandit (LinUCB)  : {len(self.policy.pulls)} arms, {total} total pulls")
            for arm in sorted(self.policy.pulls, key=lambda a: -self.policy.pulls[a])[:5]:
                print(f"      {arm:<10} pulls={self.policy.pulls[arm]}")
        print(f"  Forecaster peak  : {self.forecaster._mult.max():.2f}x "
              f"at interval {int(self.forecaster._mult.argmax())}")
        print("=" * w)


# ===========================================================================
# 11. ML SIMULATION ENGINE
# ===========================================================================

class MLSimulationEngine(SimulationEngine):
    """SimulationEngine with: bandit routing, stochastic patience (so
    abandonment is a real outcome), a stochastic resolution outcome (so FCR is a
    real signal), forecaster-driven time-varying arrivals, and full decision
    logging for honest training data."""

    def __init__(self, config: SimulationConfig, registry: MLRegistry,
                 weights: Optional[RouterScoreWeights] = None,
                 use_forecaster: bool = True,
                 stochastic_patience: bool = True) -> None:
        self._registry           = registry
        self._use_forecaster     = use_forecaster
        self._stochastic_patience = stochastic_patience
        super().__init__(config, weights)
        self.router = registry.make_router(self.agents, config, weights)
        self._rate_fn = registry.forecaster.rate_fn(config.sim_duration_minutes) \
            if use_forecaster else None

    # -- time-varying arrivals ---------------------------------------------

    def _arrival_process(self):
        while True:
            rate = self._rate_fn(self.env.now) if self._rate_fn \
                else self.config.arrival_rate_per_minute
            yield self.env.timeout(self._rng.expovariate(max(1e-6, rate)))
            self.env.process(self._handle_call(self._generate_call()))

    # -- true resolution dynamics the FCR model tries to learn -------------

    def _resolution_prob(self, call: Call, agent: Agent) -> float:
        """Probability that this agent resolves this call on first contact."""
        match = 1.0 if call.skill == agent.primary_skill else (
            0.6 if call.skill in agent.secondary_skills else 0.4)
        z = (0.8 + 1.2 * agent.csat_bias + 0.5 * (_exp_ix(agent.experience) - 1)
             + 0.9 * (match - 0.75) - 0.5 * (1.0 if call.is_repeat else 0.0))
        # item 3a: real-time fatigue erodes resolution effectiveness
        pen = getattr(self._registry, "fatigue_resolution_penalty", 0.0)
        if pen:
            z -= pen * self._registry.fatigue.get(agent.agent_id)
        return _sigmoid(z)

    def _true_resolution(self, call: Call, agent: Agent) -> bool:
        return self._rng.random() < self._resolution_prob(call, agent)

    # -- full call lifecycle with real logging -----------------------------

    def _handle_call(self, call: Call):
        resource, target_skill, routing_reason = self.router.select_resource(
            call, self.skill_resources)
        q_at_arrival = len(resource.queue)
        patience = (self._registry.abandon.sample_patience(call.customer_type, self._rng)
                    if self._stochastic_patience else self._MAX_WAIT_PATIENCE)

        request = resource.request(priority=call.priority)
        result  = yield request | self.env.timeout(patience)

        if request not in result:                       # ---- abandoned ----
            request.cancel()
            self._registry.log.log_queue_result(
                call.customer_type, call.skill, call.is_repeat,
                patience, q_at_arrival, call.arrival_time, abandoned=True)
            self.kpi.record_abandonment(call, self.env.now)
            return

        service_start = self.env.now
        wait          = service_start - call.arrival_time
        self._registry.log.log_queue_result(
            call.customer_type, call.skill, call.is_repeat,
            wait, q_at_arrival, call.arrival_time, abandoned=False)

        candidates = self._free_agents(target_skill)
        if target_skill != call.skill:
            preferred = [a for a in candidates if call.skill in a.secondary_skills]
            if preferred:
                candidates = preferred
        if not candidates:
            resource.release(request)
            return
        agent, x_policy, base_dec = self.router.select_agent(call, candidates, self.skill_resources)
        agent.busy = True
        self.router.notify_call_started(agent.agent_id)

        svc = max(0.5, self._rng.gauss(self.config.mean_service_minutes,
                                       self.config.stdev_service_minutes))
        acw = max(0.0, self._rng.gauss(self.config.acw_mean_minutes,
                                       self.config.acw_stdev_minutes))
        yield self.env.timeout(svc + acw)

        q_end = len(resource.queue)
        resource.release(request)
        agent.busy = False
        service_end    = self.env.now
        handle_minutes = service_end - service_start

        csat_raw = self._sample_csat(call.customer_type, agent.csat_bias, self._rng)
        resolved = self._true_resolution(call, agent)

        # fatigue + learning signals
        self._registry.fatigue.accumulate(agent.agent_id, handle_minutes)
        burnout = self._registry.fatigue.burnout_risk(agent.agent_id)

        # join outcome -> bandit reward + affinity + training rows
        reward = self.router.learn(agent.agent_id, x_policy, csat_raw,
                                   resolved, handle_minutes, burnout, wait=wait)
        self._registry.affinity.update(agent.agent_id, call.skill, csat_raw,
                                        handle_minutes, resolved,
                                        call.customer_type, q_end)
        self._registry.log.log_outcome(base_dec, csat_raw, resolved)
        self.router.notify_call_ended(
            agent.agent_id, csat_raw, handle_minutes, call.is_repeat,
            skill=call.skill, tier=call.customer_type, queue_depth=q_end)

        self.kpi.record_call(_CallRecord(
            call_id=call.call_id, skill=call.skill, customer_type=call.customer_type,
            is_repeat=call.is_repeat, arrival_time=call.arrival_time,
            service_start=service_start, service_end=service_end,
            csat_raw=csat_raw, routing_reason=f"{routing_reason}|r={reward:.2f}"))

    # breaks also recover fatigue in this engine
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
        self._registry.fatigue.recover(agent.agent_id, duration)


# ===========================================================================
# 12. FEEDBACK LOOP  (simulate -> collect real data -> evaluate -> gated retrain)
# ===========================================================================

@dataclass
class FeedbackConfig:
    n_epochs:            int   = 6
    convergence_metric:  str   = "csat"          # sla|csat|abandonment|fcr|reward
    convergence_delta:   float = 0.004
    convergence_patience: int  = 2
    retrain_csat:        bool  = True
    retrain_fcr:         bool  = True
    retrain_survival:    bool  = True
    refit_forecaster:    bool  = True
    base_seed:           int   = 42
    verbose:             bool  = True


@dataclass
class EpochRecord:
    epoch:    int
    kpi:      Dict[str, float]
    evals:    List[EvalReport]
    elapsed:  float
    eda:      List[DatasetDiagnostics] = field(default_factory=list)


class FeedbackLoop:
    def __init__(self, sim_cfg: SimulationConfig, fb_cfg: FeedbackConfig,
                 registry: MLRegistry, weights: Optional[RouterScoreWeights] = None,
                 use_forecaster: bool = True, stochastic_patience: bool = True) -> None:
        self._sim = sim_cfg
        self._fb  = fb_cfg
        self._reg = registry
        self._w   = weights
        self._fc  = use_forecaster
        self._sp  = stochastic_patience
        self.history: List[EpochRecord] = []

    def run(self) -> List[EpochRecord]:
        best, stale = -float("inf"), 0
        for epoch in range(self._fb.n_epochs):
            seed = self._fb.base_seed + epoch * 17
            random.seed(seed)
            np.random.seed(seed & 0xFFFFFFFF)
            self._reg.fatigue.reset()
            self._reg.fresh_log()

            cfg = copy.copy(self._sim)
            cfg.random_seed = seed
            engine = MLSimulationEngine(cfg, self._reg, self._w,
                                        use_forecaster=self._fc,
                                        stochastic_patience=self._sp)
            t0 = time.perf_counter()
            engine.run()
            elapsed = time.perf_counter() - t0

            kpi   = self._extract_kpi(engine)
            if epoch == self._fb.n_epochs - 1:
                evals, diags = [], []
            else:
                evals, diags = self._retrain(engine)
            self.history.append(EpochRecord(epoch, kpi, evals, elapsed, diags))

            if self._fb.verbose:
                self._print_epoch(epoch, kpi, evals, elapsed, diags)

            metric = kpi.get(self._fb.convergence_metric, 0.0)
            metric = -metric if self._fb.convergence_metric == "abandonment" else metric
            if metric > best + self._fb.convergence_delta:
                best, stale = metric, 0
            else:
                stale += 1
            if epoch > 0 and stale >= self._fb.convergence_patience:
                if self._fb.verbose:
                    print(f"  [FeedbackLoop] converged after epoch {epoch + 1}.")
                break
        return self.history

    def _retrain(self, engine: MLSimulationEngine
                 ) -> Tuple[List[EvalReport], List[DatasetDiagnostics]]:
        reg, log, fb = self._reg, self._reg.log, self._fb
        reports: List[EvalReport]        = []
        diags:   List[DatasetDiagnostics] = []
        base_names = list(FeatureSchema.BASE_NAMES)

        if fb.retrain_csat:
            X, y = log.csat_dataset()
            d = reg.eda.analyze(X, y, base_names, target_name="csat>=4")
            diags.append(d)
            if d.trainable:
                reports.append(reg.evaluator.evaluate_and_gate(reg.csat, X, y))
        if fb.retrain_fcr:
            X, y = log.fcr_dataset()
            # FCR shares the base feature matrix; skip drift double-count by not
            # updating the shared reference a second time on the same rows.
            d = reg.eda.analyze(X, y, base_names, target_name="resolved",
                                update_reference=False)
            diags.append(d)
            if d.trainable:
                reports.append(reg.evaluator.evaluate_and_gate(reg.fcr, X, y))
        if fb.retrain_survival:
            waits, flags, tiers = log.survival_dataset()
            if sum(flags) >= 8:
                reg.abandon.fit(waits, flags, tiers)
        if fb.refit_forecaster:
            reg.forecaster.observe(self._interval_counts(engine))
        return reports, diags

    def _interval_counts(self, engine: MLSimulationEngine) -> np.ndarray:
        n = self._reg.forecaster.N_INTERVALS
        horizon = self._sim.sim_duration_minutes
        step = horizon / n
        counts = np.zeros(n)
        for rec in engine.kpi._records:
            counts[min(n - 1, int(rec.arrival_time // step))] += 1
        return counts / step

    @staticmethod
    def _extract_kpi(engine: MLSimulationEngine) -> Dict[str, float]:
        recs = engine.kpi._records
        return {
            "sla":         engine.kpi.sla_percentage(),
            "abandonment": engine.kpi.abandonment_rate(),
            "csat":        engine.kpi.average_csat(),
            "asa":         engine.kpi.average_speed_of_answer(),
            "aht":         engine.kpi.average_handle_time(),
            "fcr":         engine.kpi.first_call_resolution(),
            "calls":       float(engine.kpi.total_calls()),
            "reward":      _mean_reward(recs),
        }

    @staticmethod
    def _print_epoch(epoch, kpi, evals, elapsed, diags=()):
        print(f"  Epoch {epoch + 1:>2}  "
              f"SLA={kpi['sla']:.1%}  Abn={kpi['abandonment']:.1%}  "
              f"CSAT={kpi['csat']:.3f}  reward={kpi['reward']:.3f}  "
              f"calls={kpi['calls']:.0f}  [{elapsed:.1f}s]")
        for d in diags:
            drift = "" if d.drift_score is None else f" drift={d.drift_score:.3f}"
            gate  = "ok" if d.trainable else "GATED (no train)"
            print(f"        EDA {d.target_name:<10} n={d.n_rows} "
                  f"pos={d.pos_rate:.0%}{drift} -> {gate}")
            for w in d.warnings:
                print(f"           ! {w}")
        for ev in evals:
            if not math.isnan(ev.auc):
                tag = "committed" if ev.committed else "kept incumbent"
                print(f"        {ev.name:<10} AUC={ev.auc:.3f} Brier={ev.brier:.3f} "
                      f"n={ev.n} -> {tag}")


def _mean_reward(records) -> float:
    rs = []
    for r in records:
        if "|r=" in r.routing_reason:
            try:
                rs.append(float(r.routing_reason.split("|r=")[1]))
            except ValueError:
                pass
    return float(np.mean(rs)) if rs else 0.0


class FeedbackLoopReport:
    """Tabular summary of a FeedbackLoop run (per-epoch KPIs + model AUCs)."""

    def __init__(self, history: List[EpochRecord], fb_cfg: Optional[FeedbackConfig] = None) -> None:
        self._history = history
        self._fb      = fb_cfg

    def print_report(self) -> None:
        if not self._history:
            print("[FeedbackLoopReport] No epochs to report.")
            return
        w = 84
        print("\n" + "=" * w)
        print("  ML FEEDBACK LOOP -- EPOCH REPORT")
        print("=" * w)
        print(f"  {'Epoch':>5}  {'SLA':>7}  {'Abandon':>8}  {'CSAT':>6}  {'FCR':>6}  "
              f"{'Reward':>7}  {'csatAUC':>8}  {'fcrAUC':>7}  {'Calls':>6}")
        print(f"  {'-' * (w - 4)}")
        for h in self._history:
            ev = {e.name: e for e in h.evals}
            ca = ev.get("csat>=4"); fa = ev.get("resolved")
            cs = f"{ca.auc:.3f}" if ca and not math.isnan(ca.auc) else "   -  "
            fs = f"{fa.auc:.3f}" if fa and not math.isnan(fa.auc) else "  -  "
            k = h.kpi
            print(f"  {h.epoch + 1:>5}  {k['sla']:>7.1%}  {k['abandonment']:>8.1%}  "
                  f"{k['csat']:>6.2f}  {k['fcr']:>6.1%}  {k['reward']:>7.3f}  "
                  f"{cs:>8}  {fs:>7}  {k['calls']:>6.0f}")
        print(f"  {'-' * (w - 4)}")
        best = max(self._history, key=lambda h: h.kpi.get("csat", 0.0))
        print(f"  Epochs run        : {len(self._history)}")
        print(f"  Best CSAT epoch   : {best.epoch + 1}  (CSAT={best.kpi['csat']:.3f}, "
              f"SLA={best.kpi['sla']:.1%})")
        print("=" * w + "\n")


# convenience ---------------------------------------------------------------

def build_registry(sim_cfg: SimulationConfig, policy_alpha: float = 0.6,
                   reward_cfg: Optional[RewardConfig] = None,
                   fatigue_resolution_penalty: float = 0.0) -> MLRegistry:
    return MLRegistry(base_rate_per_min=sim_cfg.arrival_rate_per_minute,
                      policy_alpha=policy_alpha, reward_cfg=reward_cfg,
                      fatigue_resolution_penalty=fatigue_resolution_penalty)
