"""Paper 2 benchmark: Additive and Multiplicative STPU vs. baselines.

Experiment grid
---------------
  DGPs    : linear, nonlinear, heterogeneous
  Models  : causal_forest/rf, rlearner/lr, tlearner/rf
  Alpha   : 0.0, 0.5, 1.0, 1.5, 2.0  (confounding strength)
  Rho     : 0.02, 0.05, 0.10, 0.20   (base conversion rate)
  Seeds   : 5
  Beta    : 0.0 (base), 0.05, 0.10, 0.2, 0.5, 1.0 (additive STPU sweep)
  + MultiplicativeSTPU (no beta, scale-invariant)

Key metrics
-----------
  wasted_spend      : fraction of top-30% with true CATE < 0.01
  iroas_efficiency  : mean true CATE of top-30% / oracle mean CATE
  qini              : ranking quality (Qini coefficient)
  pehe              : estimation error (base model τ̂ only)
  variant           : "base" | "additive" | "multiplicative"

Usage
-----
    python paper2_stpu/experiments/run_stpu_benchmark.py
    python paper2_stpu/experiments/run_stpu_benchmark.py --quick   # 1 DGP, 3 alpha
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT / "paper1_benchmark"))
sys.path.insert(0, str(_ROOT))

from data.dgp import DGP_REGISTRY                                  # noqa: E402
from models import MODEL_REGISTRY                                   # noqa: E402
from metrics.evaluation import evaluate                             # noqa: E402
from paper2_stpu.models.stpu import STPUEstimator, MultiplicativeSTPU  # noqa: E402


# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------
ALPHA_GRID    = [0.0, 0.5, 1.0, 1.5, 2.0]
RHO_GRID      = [0.02, 0.05, 0.10, 0.20]
N_SEEDS       = 5
N             = 20_000
TRAIN_FRAC    = 0.70
TOP_K         = 0.30

# Additive STPU beta sweep (β=0 serves as the base model result)
BETA_GRID = [0.0, 0.05, 0.10, 0.20, 0.50, 1.00]

# Model families to evaluate
STPU_CONFIGS = [
    ("causal_forest", "rf"),
    ("rlearner",      "lr"),
    ("tlearner",      "rf"),
]


def run_one(dgp_name, alpha, rho, seed, model_name, learner_type, beta, variant, n=N):
    dgp_cls = DGP_REGISTRY[dgp_name]
    dgp = dgp_cls(alpha=alpha, rho=rho, seed=seed)
    sample = dgp.sample(n)

    n_train = int(n * TRAIN_FRAC)
    X_tr, T_tr, Y_tr = sample.X[:n_train], sample.T[:n_train], sample.Y[:n_train]
    X_te               = sample.X[n_train:]
    tau_te             = sample.tau[n_train:]
    Y_te, T_te         = sample.Y[n_train:], sample.T[n_train:]

    model_cls = MODEL_REGISTRY[model_name]
    base = model_cls(learner_type=learner_type, seed=seed)

    if variant == "multiplicative":
        est = MultiplicativeSTPU(cate_model=base, baseline_learner=learner_type, seed=seed)
    else:
        est = STPUEstimator(cate_model=base, beta=beta,
                            baseline_learner=learner_type, seed=seed)

    est.fit(X_tr, T_tr, Y_tr)

    score   = est.predict_score(X_te)     # penalized ranking score
    tau_hat = est.predict_cate(X_te)      # raw CATE (for PEHE)

    m_budget = evaluate(score,   Y_te, T_te, tau_true=tau_te, top_k=TOP_K)
    m_pehe   = evaluate(tau_hat, Y_te, T_te, tau_true=tau_te, top_k=TOP_K)

    # Extra: mean tau in top-k and sleeping-dog fraction in top-k
    topk_n  = int(TOP_K * len(tau_te))
    idx_top = np.argsort(score)[::-1][:topk_n]
    tau_top = tau_te[idx_top]

    return {
        "dgp":              dgp_name,
        "alpha":            alpha,
        "rho":              rho,
        "seed":             seed,
        "base_model":       model_name,
        "learner":          learner_type,
        "variant":          variant,
        "beta":             beta if variant == "additive" else float("nan"),
        "pehe":             m_pehe["pehe"],
        "wasted_spend":     m_budget["wasted_spend"],
        "iroas_efficiency": m_budget["iroas_efficiency"],
        "qini":             m_budget["qini"],
        "mean_tau_topk":    float(tau_top.mean()),
        "pct_sleeping_dog": float((tau_top < 0).mean()),
        "pct_persuadable":  float((tau_top > 0).mean()),
        "status":           "ok",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Quick run: heterogeneous DGP, 3 alphas, 1 rho, 2 seeds")
    parser.add_argument("--dgp",   default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--out",   default=None)
    args = parser.parse_args()

    if args.quick:
        dgp_names   = ["linear", "nonlinear", "heterogeneous"]
        alpha_grid  = [0.0, 1.0, 2.0]
        rho_grid    = [0.05]
        n_seeds     = 3
    else:
        dgp_names   = [args.dgp]   if args.dgp   else list(DGP_REGISTRY.keys())
        alpha_grid  = ALPHA_GRID
        rho_grid    = RHO_GRID
        n_seeds     = N_SEEDS

    stpu_configs = ([(args.model, lt) for lt in ["lr","rf"]]
                    if args.model else STPU_CONFIGS)

    out_dir  = Path(args.out) if args.out else Path("paper2_stpu/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "stpu_benchmark.csv"

    # Build full job list
    jobs = []
    for dgp in dgp_names:
        for alpha in alpha_grid:
            for rho in rho_grid:
                for seed in range(n_seeds):
                    for model_name, learner_type in stpu_configs:
                        if model_name not in MODEL_REGISTRY:
                            continue
                        # Additive sweep (β=0 = base model)
                        for beta in BETA_GRID:
                            jobs.append((dgp, alpha, rho, seed,
                                         model_name, learner_type,
                                         beta, "additive"))
                        # Multiplicative (no beta)
                        jobs.append((dgp, alpha, rho, seed,
                                     model_name, learner_type,
                                     float("nan"), "multiplicative"))

    print(f"Running {len(jobs):,} configurations "
          f"({'quick' if args.quick else 'full'} mode)")
    print(f"Results → {out_path}\n")

    # Fresh file — remove stale results from prior runs
    if out_path.exists():
        out_path.unlink()

    rows = []
    for i, job in enumerate(tqdm(jobs, desc="STPU")):
        dgp, alpha, rho, seed, model_name, learner, beta, variant = job
        try:
            row = run_one(dgp, alpha, rho, seed, model_name, learner, beta, variant)
        except Exception as e:
            row = {
                "dgp": dgp, "alpha": alpha, "rho": rho, "seed": seed,
                "base_model": model_name, "learner": learner,
                "variant": variant, "beta": beta, "status": f"error: {e}",
            }
        rows.append(row)
        # Write header on first row, append thereafter
        if i == 0:
            pd.DataFrame([row]).to_csv(out_path, index=False, mode="w")
        else:
            pd.DataFrame([row]).to_csv(out_path, index=False, mode="a", header=False)

    df = pd.read_csv(out_path)

    ok = df[df["status"] == "ok"]
    print(f"\nCompleted: {len(ok):,}/{len(df):,} successful")

    # Quick summary: base vs multiplicative, heterogeneous DGP, alpha=1.0
    sub = ok[(ok.dgp == "heterogeneous") & np.isclose(ok.alpha, 1.0) & np.isclose(ok.rho, 0.05)]
    base_sub = sub[np.isclose(sub.beta, 0.0) & (sub.variant == "additive")]
    mult_sub = sub[sub.variant == "multiplicative"]

    if len(base_sub) and len(mult_sub):
        print("\n── Heterogeneous DGP, α=1.0, ρ=0.05 ──────────────────────────────")
        print(f"{'Model':25s}  {'Base WS':>8s}  {'Mult WS':>8s}  {'Base iROAS':>10s}  {'Mult iROAS':>10s}")
        for (mn, lt), grp in base_sub.groupby(["base_model", "learner"]):
            m_base = grp[["wasted_spend","iroas_efficiency"]].mean()
            m_mult = mult_sub[(mult_sub.base_model==mn)&(mult_sub.learner==lt)][["wasted_spend","iroas_efficiency"]].mean()
            if len(m_mult):
                print(f"{mn}/{lt:25s}  {m_base['wasted_spend']:.4f}    {m_mult['wasted_spend']:.4f}"
                      f"    {m_base['iroas_efficiency']:.4f}      {m_mult['iroas_efficiency']:.4f}")

    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
