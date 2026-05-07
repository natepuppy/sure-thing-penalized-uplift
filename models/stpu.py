"""Sure-Thing Penalized Uplift (STPU) Estimators.

Problem
-------
Standard uplift models rank users by estimated CATE τ̂(X). Under confounding,
sure-thing converters (high baseline P(Y=1|T=0,X), near-zero true CATE) are
systematically over-ranked because models conflate E[Y₁|X] with the true lift
E[Y₁-Y₀|X]. This wastes ad budget and suppresses iROAS.

Two formulations
----------------

Additive STPU (original)
    s(X; β) = τ̂(X) − β · ĝ(X)

    Fails in practice because ĝ(X) ∈ (0,1) while τ̂(X) typically lives on a
    much smaller scale (|τ̂| << 1 for rare-event conversions). Even small β
    values cause ĝ to dominate the ranking, discarding persuadables alongside
    sure-things.

Multiplicative STPU (recommended)
    s(X) = τ̂(X) · (1 − ĝ(X))

    Scale-invariant: (1-ĝ) acts as a user-specific weight in (0,1) that
    shrinks the score for high-baseline users without introducing a tunable
    β. Theoretical motivation: the optimal persuadability score separates
    users with high incremental lift AND low organic conversion, exactly what
    τ̂·(1-ĝ) measures. No β tuning required.

    Conditions for effectiveness
    - Base model τ̂ must be approximately unbiased (well-calibrated estimator)
    - The DGP structure must allow ĝ to distinguish sure-things from
      persuadables (i.e. sure-things must have substantially higher ĝ)
    - Fails under heavy subgroup mixing where ĝ is noisy

Both estimators require only the control arm to estimate ĝ, so they are
applicable to real ad data without ground-truth CATE.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Allow imports from paper1_benchmark
_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT / "paper1_benchmark"))

from models.base import BaseUpliftModel, make_base_learner  # noqa: E402


class STPUEstimator(BaseUpliftModel):
    """Sure-Thing Penalized Uplift estimator.

    Wraps any base CATE estimator and subtracts β · ĝ(X) from its scores,
    where ĝ(X) = P(Y=1 | T=0, X) is estimated from the control arm.

    Parameters
    ----------
    cate_model : BaseUpliftModel
        Any fitted (or unfitted) uplift model that implements fit() and
        predict_cate().
    beta : float
        Sure-thing penalty weight. β=0 recovers the base model exactly.
        β=1 is a reasonable default; tune via cross-validation.
    baseline_learner : str
        Base learner type for ĝ(X): "lr", "rf", or "xgb".
    seed : int
        Random seed for the baseline model.
    """

    def __init__(
        self,
        cate_model: BaseUpliftModel,
        beta: float = 0.5,
        baseline_learner: str = "rf",
        seed: int = 0,
    ):
        super().__init__(learner_type=baseline_learner, seed=seed)
        self.cate_model = cate_model
        self.beta = beta
        self.baseline_learner = baseline_learner
        self._baseline_model = None

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "STPUEstimator":
        # Fit base CATE estimator on full data
        self.cate_model.fit(X, T, Y)

        # Fit baseline conversion model on control arm only
        control = T == 0
        if control.sum() < 10:
            raise ValueError(
                f"Too few control observations ({control.sum()}) to fit baseline model."
            )

        X_ctrl, Y_ctrl = X[control], Y[control]

        if self.baseline_learner == "lr":
            self._baseline_model = LogisticRegression(
                max_iter=1000, C=1.0, random_state=self.seed
            )
        elif self.baseline_learner == "rf":
            self._baseline_model = RandomForestClassifier(
                n_estimators=100, random_state=self.seed, n_jobs=-1,
                min_samples_leaf=5,
            )
        else:  # xgb
            from xgboost import XGBClassifier
            self._baseline_model = XGBClassifier(
                n_estimators=100, random_state=self.seed,
                eval_metric="logloss", verbosity=0, n_jobs=-1,
            )

        self._baseline_model.fit(X_ctrl, Y_ctrl)
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """Returns raw CATE estimate (without penalty) for metric computation."""
        return self.cate_model.predict_cate(X)

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """Returns penalized ranking score s(X) = τ̂(X) − β · ĝ(X).

        Use this for targeting decisions; use predict_cate() for PEHE
        and other ground-truth metrics.
        """
        tau_hat = self.cate_model.predict_cate(X)
        g_hat = self._baseline_model.predict_proba(X)[:, 1]
        return tau_hat - self.beta * g_hat

    @property
    def name(self) -> str:
        return (
            f"STPU[{self.cate_model.name},β={self.beta:.1f},"
            f"{self.baseline_learner}]"
        )


class MultiplicativeSTPU(STPUEstimator):
    """Scale-invariant multiplicative STPU: s(X) = τ̂(X) · (1 − ĝ(X)).

    Addresses the scale-mismatch failure of the additive form.  The weight
    (1-ĝ) lies in (0,1) and shrinks scores for high-baseline users without
    a tunable β — users with organic conversion probability close to 1 are
    downweighted to near-zero regardless of their τ̂.

    Parameters
    ----------
    cate_model : BaseUpliftModel
        Any fitted (or unfitted) uplift model.
    baseline_learner : str
        Base learner for ĝ(X): "lr", "rf", or "xgb".
    seed : int
        Random seed.
    """

    def __init__(
        self,
        cate_model,
        baseline_learner: str = "rf",
        seed: int = 0,
    ):
        super().__init__(
            cate_model=cate_model,
            beta=1.0,               # not used in multiplicative form
            baseline_learner=baseline_learner,
            seed=seed,
        )

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """Returns multiplicative ranking score s(X) = τ̂(X) · (1 − ĝ(X))."""
        tau_hat = self.cate_model.predict_cate(X)
        g_hat = self._baseline_model.predict_proba(X)[:, 1]
        return tau_hat * (1.0 - g_hat)

    @property
    def name(self) -> str:
        return f"MSTPU[{self.cate_model.name},{self.baseline_learner}]"
