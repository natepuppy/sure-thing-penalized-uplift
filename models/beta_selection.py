"""Beta selection for STPU via cross-validated wasted-spend minimization."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT / "paper1_benchmark"))

from metrics.evaluation import wasted_spend_fraction as wasted_spend  # noqa: E402


def select_beta(
    cate_model_cls,
    cate_model_kwargs: dict,
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    tau_true: np.ndarray,
    beta_grid: list[float] | None = None,
    baseline_learner: str = "rf",
    n_folds: int = 3,
    top_k: float = 0.30,
    seed: int = 0,
) -> tuple[float, dict]:
    """Select beta that minimises wasted spend via cross-validation.

    Returns the best beta and a dict of {beta: mean_wasted_spend}.

    Note: tau_true is used only for evaluation (wasted_spend), not for
    fitting — so this is valid on real data if you substitute a proxy
    metric (e.g. Qini on held-out randomized data).
    """
    if beta_grid is None:
        beta_grid = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

    from paper2_stpu.models.stpu import STPUEstimator

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = {b: [] for b in beta_grid}

    for train_idx, val_idx in kf.split(X):
        X_tr, T_tr, Y_tr = X[train_idx], T[train_idx], Y[train_idx]
        X_val = X[val_idx]
        tau_val = tau_true[val_idx]

        for beta in beta_grid:
            base = cate_model_cls(**cate_model_kwargs)
            est = STPUEstimator(
                cate_model=base,
                beta=beta,
                baseline_learner=baseline_learner,
                seed=seed,
            )
            est.fit(X_tr, T_tr, Y_tr)
            ranking_score = est.predict_score(X_val)
            ws = wasted_spend(ranking_score, tau_val, top_k=top_k)
            scores[beta].append(ws)

    mean_scores = {b: float(np.mean(v)) for b, v in scores.items()}
    best_beta = min(mean_scores, key=mean_scores.__getitem__)
    return best_beta, mean_scores
