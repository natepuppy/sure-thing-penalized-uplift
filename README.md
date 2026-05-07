# Sure-Thing Penalized Uplift (STPU)

**Paper:** *Can We Fix Wasted Ad Spend by Re-ranking Users? Sure-Thing Penalized Uplift: A Study of What Works and What Doesn't*
**Author:** Nathan Clark (2025)
**SSRN:** *(link to be added)*

---

## What this paper is about

When you run an ad campaign, a large chunk of your budget goes to users who would have bought anyway — they don't need the ad at all. Paper 1 in this series showed that all standard uplift models waste 86–92% of their targeting budget this way.

This paper tests a natural fix: lower the score of users who already have a high chance of buying without the ad. We call this **Sure-Thing Penalized Uplift (STPU)**. We tested two versions:

- **Additive STPU**: `s(X) = τ̂(X) − β · ĝ(X)`
- **Multiplicative STPU (MSTPU)**: `s(X) = τ̂(X) · (1 − ĝ(X))`

**The short answer: neither works consistently.**

Additive STPU always makes things worse because the penalty is 2–5× larger in scale than the CATE estimates. Multiplicative STPU fixes the scale problem but shows no consistent improvement across 567 experiments — it's essentially a coin flip.

---

## Repo structure

```
paper2_stpu/
├── models/
│   ├── stpu.py          # STPUEstimator and MultiplicativeSTPU classes
│   ├── beta_selection.py  # Cross-validated beta selection
│   └── __init__.py
├── experiments/
│   └── run_stpu_benchmark.py   # Main experiment runner
├── analysis/
│   └── figures.py       # Generate all paper figures
├── results/
│   └── stpu_benchmark.csv   # Full benchmark results (567 configs)
└── paper/
    ├── main.tex
    ├── main.pdf
    ├── refs.bib
    └── figures/
```

---

## Reproducing the results

### Install dependencies

```bash
pip install numpy pandas scikit-learn econml matplotlib seaborn
```

### Run the benchmark

```bash
# Full run (~30 min)
python paper2_stpu/experiments/run_stpu_benchmark.py

# Quick run (3 DGPs × 3 alphas × 1 rho × 3 seeds)
python paper2_stpu/experiments/run_stpu_benchmark.py --quick
```

Results are written incrementally to `paper2_stpu/results/stpu_benchmark.csv`.

### Generate figures

```bash
python paper2_stpu/analysis/figures.py
```

Figures are saved to `paper2_stpu/paper/figures/`.

### Compile the paper

```bash
cd paper2_stpu/paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

## Key findings

| Finding | Detail |
|---|---|
| Additive STPU always degrades iROAS | The baseline probability ĝ(X) is 2–5× larger in scale than τ̂(X), so the penalty overwhelms the signal |
| Multiplicative STPU: 48% win rate | Coin-flip performance across 567 experiments |
| Wasted spend never improves | Neither variant reduces the fraction of budget spent on non-incrementals |
| The fundamental limit | At 5% conversion rates, <7% of users are truly responsive on any given day. Re-ranking can't close that gap. |

---

## Models tested

| Base CATE model | STPU variants |
|---|---|
| Causal Forest | β ∈ {0.0, 0.05, 0.10, 0.20, 0.50, 1.00} (additive) |
| R-Learner (LR) | Multiplicative (scale-invariant) |
| T-Learner (RF) | |

DGPs: Linear, Nonlinear, Heterogeneous CATE
Confounding strengths (α): 0.0, 0.5, 1.0
Base conversion rates (ρ): 0.05

---

## Related papers

- **Paper 1**: [Benchmarking Uplift Models Under Realistic Confounding](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6643279)
- **Paper 3**: When Should You Stop Showing Someone an Ad? Neural SDE Fatigue Modeling *(link to be added)*
