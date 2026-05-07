"""Generate all Paper 2 figures from STPU benchmark results.

Usage:
    cd paper2_stpu
    python analysis/figures.py --results results/stpu_benchmark.csv

Figures produced:
    fig1_main_result.pdf      -- Multiplicative STPU vs base: iROAS improvement by DGP
    fig2_additive_failure.pdf -- Why additive STPU fails: scale mismatch analysis
    fig3_beta_sensitivity.pdf -- Wasted spend vs beta (additive sweep), per DGP / alpha
    fig4_iroas_by_alpha.pdf   -- iROAS efficiency vs confounding strength
    fig5_sleeping_dog.pdf     -- Sleeping-dog fraction in top-k: base vs MSTPU
    tab_main_results.tex      -- Main LaTeX results table
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_LABELS = {
    "causal_forest": "CausalForest",
    "rlearner":      "R-Learner",
    "tlearner":      "T-Learner",
}
LEARNER_LABELS = {"lr": "LR", "rf": "RF", "xgb": "XGB"}
DGP_LABELS = {
    "linear":        "Linear CATE",
    "nonlinear":     "Nonlinear CATE",
    "heterogeneous": "Heterogeneous CATE",
}
DGP_ORDER = ["linear", "nonlinear", "heterogeneous"]

PALETTE = sns.color_palette("tab10", n_colors=6)
MODEL_COLORS = {
    "causal_forest": PALETTE[0],
    "rlearner":      PALETTE[1],
    "tlearner":      PALETTE[2],
}
DGP_COLORS = {
    "linear":        PALETTE[0],
    "nonlinear":     PALETTE[1],
    "heterogeneous": PALETTE[2],
}

STYLE = {
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
}
sns.set_theme(style="whitegrid", rc=STYLE)


def _savefig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    path = out_dir / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _model_label(row) -> str:
    return f"{MODEL_LABELS.get(row['base_model'], row['base_model'])}" \
           f"[{LEARNER_LABELS.get(row['learner'], row['learner'])}]"


# ── Load ───────────────────────────────────────────────────────────────────────
def load_results(path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (full_df, base_df, mstpu_df)."""
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy()
    df["model_label"] = df.apply(_model_label, axis=1)

    base  = df[(df["variant"] == "additive") & np.isclose(df["beta"], 0.0)].copy()
    mstpu = df[df["variant"] == "multiplicative"].copy()
    return df, base, mstpu


# ── Figure 1: Main result — iROAS improvement from Multiplicative STPU ─────────
def fig_main_result(base: pd.DataFrame, mstpu: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart: iROAS efficiency gain Δ = MSTPU − base, by model and DGP."""
    keys = ["dgp", "base_model", "learner", "alpha"]
    b = base.groupby(keys)["iroas_efficiency"].mean().reset_index().rename(
        columns={"iroas_efficiency": "ie_base"})
    m = mstpu.groupby(keys)["iroas_efficiency"].mean().reset_index().rename(
        columns={"iroas_efficiency": "ie_mstpu"})
    merged = b.merge(m, on=keys)
    merged["delta"] = merged["ie_mstpu"] - merged["ie_base"]
    merged["model_label"] = merged.apply(_model_label, axis=1)

    dgps = [d for d in DGP_ORDER if d in merged["dgp"].unique()]
    fig, axes = plt.subplots(1, len(dgps), figsize=(5 * len(dgps), 4), sharey=True)
    if len(dgps) == 1:
        axes = [axes]

    for ax, dgp in zip(axes, dgps):
        sub = (merged[merged["dgp"] == dgp]
               .groupby("model_label")["delta"]
               .mean()
               .reset_index()
               .sort_values("delta", ascending=True))
        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in sub["delta"]]
        ax.barh(sub["model_label"], sub["delta"] * 100, color=colors)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(DGP_LABELS.get(dgp, dgp))
        ax.set_xlabel("Δ iROAS efficiency (pp)")
        if ax is axes[0]:
            ax.set_ylabel("Model")

    fig.suptitle(
        "iROAS Efficiency Gain: Multiplicative STPU vs. Base Model\n"
        "(positive = MSTPU improves incremental ROAS; averaged over α, ρ, seeds)",
        fontsize=11, y=1.03,
    )
    _savefig(fig, out_dir, "fig1_main_result.pdf")


# ── Figure 2: Why additive STPU fails — scale mismatch illustration ──────────
def fig_additive_failure(df: pd.DataFrame, out_dir: Path) -> None:
    """Line plot of wasted spend vs beta for each DGP, CausalForest, alpha=1.0."""
    additive = df[(df["variant"] == "additive")
                  & (df["base_model"] == "causal_forest")
                  & np.isclose(df["alpha"], 1.0)
                  & np.isclose(df["rho"], 0.05)].copy()

    if additive.empty:
        print("  Skipping fig2 (no matching rows)")
        return

    agg = (additive.groupby(["dgp", "beta"])
           [["wasted_spend", "iroas_efficiency"]]
           .mean().reset_index())

    dgps = [d for d in DGP_ORDER if d in agg["dgp"].unique()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for dgp in dgps:
        sub = agg[agg["dgp"] == dgp].sort_values("beta")
        axes[0].plot(sub["beta"], sub["wasted_spend"] * 100,
                     marker="o", label=DGP_LABELS.get(dgp, dgp),
                     color=DGP_COLORS.get(dgp))
        axes[1].plot(sub["beta"], sub["iroas_efficiency"] * 100,
                     marker="o", label=DGP_LABELS.get(dgp, dgp),
                     color=DGP_COLORS.get(dgp))

    axes[0].set_xlabel("Penalty weight β")
    axes[0].set_ylabel("Wasted spend (%)")
    axes[0].set_title("Wasted Spend vs. β")
    axes[0].legend()

    axes[1].set_xlabel("Penalty weight β")
    axes[1].set_ylabel("iROAS efficiency (%)")
    axes[1].set_title("iROAS Efficiency vs. β")
    axes[1].legend()

    fig.suptitle(
        "Additive STPU (β-sweep): CausalForest, α=1.0, ρ=0.05\n"
        "Increasing β worsens wasted spend but may improve iROAS",
        fontsize=11, y=1.03,
    )
    _savefig(fig, out_dir, "fig2_additive_failure.pdf")


# ── Figure 3: Beta sensitivity — wasted spend vs beta by model and DGP ────────
def fig_beta_sensitivity(df: pd.DataFrame, out_dir: Path) -> None:
    """Grid: rows=DGP, cols=model. Each cell = wasted_spend vs beta."""
    additive = df[df["variant"] == "additive"].copy()
    agg = (additive.groupby(["dgp", "base_model", "learner", "beta"])
           ["wasted_spend"].mean().reset_index())

    dgps   = [d for d in DGP_ORDER if d in agg["dgp"].unique()]
    models = [(mn, lt) for mn, lt in [("causal_forest","rf"),("rlearner","lr"),("tlearner","rf")]
              if mn in agg["base_model"].unique()]

    fig, axes = plt.subplots(len(dgps), len(models),
                             figsize=(4.5 * len(models), 3.5 * len(dgps)),
                             sharey="row", sharex=True)
    if len(dgps) == 1:
        axes = axes[np.newaxis, :]
    if len(models) == 1:
        axes = axes[:, np.newaxis]

    for i, dgp in enumerate(dgps):
        for j, (mn, lt) in enumerate(models):
            ax = axes[i][j]
            sub = (agg[(agg["dgp"] == dgp)
                       & (agg["base_model"] == mn)
                       & (agg["learner"] == lt)]
                   .sort_values("beta"))
            if sub.empty:
                continue
            ax.plot(sub["beta"], sub["wasted_spend"] * 100,
                    marker="o", color=MODEL_COLORS.get(mn))
            ax.axhline(sub[np.isclose(sub["beta"], 0)]["wasted_spend"].values[0] * 100
                       if len(sub[np.isclose(sub["beta"], 0)]) else sub["wasted_spend"].iloc[0] * 100,
                       color="gray", linestyle="--", linewidth=0.8, label="β=0 (base)")
            if i == 0:
                ax.set_title(f"{MODEL_LABELS.get(mn,mn)}[{LEARNER_LABELS.get(lt,lt)}]")
            if j == 0:
                ax.set_ylabel(f"{DGP_LABELS.get(dgp,dgp)}\nWasted spend (%)")
            if i == len(dgps) - 1:
                ax.set_xlabel("β")

    fig.suptitle("Additive STPU: Wasted Spend vs. β\n(dashed = base model, lower is better)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    _savefig(fig, out_dir, "fig3_beta_sensitivity.pdf")


# ── Figure 4: iROAS efficiency vs confounding (base vs MSTPU) ─────────────────
def fig_iroas_by_alpha(base: pd.DataFrame, mstpu: pd.DataFrame, out_dir: Path) -> None:
    """Line plots: iROAS efficiency vs α, one panel per DGP."""
    b = (base.groupby(["dgp", "base_model", "learner", "alpha"])
         ["iroas_efficiency"].mean().reset_index().assign(variant="base"))
    m = (mstpu.groupby(["dgp", "base_model", "learner", "alpha"])
         ["iroas_efficiency"].mean().reset_index().assign(variant="mstpu"))
    combined = pd.concat([b, m])
    combined["model_label"] = combined.apply(_model_label, axis=1)

    dgps = [d for d in DGP_ORDER if d in combined["dgp"].unique()]
    fig, axes = plt.subplots(1, len(dgps), figsize=(5 * len(dgps), 4), sharey=True)
    if len(dgps) == 1:
        axes = [axes]

    for ax, dgp in zip(axes, dgps):
        sub = combined[combined["dgp"] == dgp]
        for (mn, lt), grp in sub.groupby(["base_model", "learner"]):
            label = f"{MODEL_LABELS.get(mn,mn)}[{LEARNER_LABELS.get(lt,lt)}]"
            color = MODEL_COLORS.get(mn, PALETTE[0])
            base_g = grp[grp["variant"] == "base"].sort_values("alpha")
            mst_g  = grp[grp["variant"] == "mstpu"].sort_values("alpha")
            ax.plot(base_g["alpha"], base_g["iroas_efficiency"] * 100,
                    marker="o", color=color, linestyle="--",
                    label=f"{label} (base)", linewidth=1.5)
            ax.plot(mst_g["alpha"], mst_g["iroas_efficiency"] * 100,
                    marker="s", color=color, linestyle="-",
                    label=f"{label} (MSTPU)", linewidth=1.5)
        ax.set_title(DGP_LABELS.get(dgp, dgp))
        ax.set_xlabel("Confounding strength (α)")
        if ax is axes[0]:
            ax.set_ylabel("iROAS efficiency (%)")
        ax.legend(fontsize=7, ncol=1)

    fig.suptitle(
        "iROAS Efficiency vs. Confounding Strength\n"
        "(dashed = base model, solid = Multiplicative STPU)",
        fontsize=11, y=1.03,
    )
    _savefig(fig, out_dir, "fig4_iroas_by_alpha.pdf")


# ── Figure 5: Sleeping-dog and persuadable fraction in top-k ──────────────────
def fig_sleeping_dog(base: pd.DataFrame, mstpu: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart comparing sleeping-dog and persuadable fraction in top-30% for
    base vs MSTPU, by DGP."""
    if "pct_sleeping_dog" not in base.columns:
        print("  Skipping fig5 (metric not in results)")
        return

    dgps = [d for d in DGP_ORDER if d in base["dgp"].unique()]
    fig, axes = plt.subplots(1, len(dgps), figsize=(5 * len(dgps), 4), sharey=True)
    if len(dgps) == 1:
        axes = [axes]

    for ax, dgp in zip(axes, dgps):
        b = base[base["dgp"] == dgp].groupby("base_model")[
            ["pct_sleeping_dog", "pct_persuadable"]].mean().reset_index()
        m = mstpu[mstpu["dgp"] == dgp].groupby("base_model")[
            ["pct_sleeping_dog", "pct_persuadable"]].mean().reset_index()

        x = np.arange(len(b))
        w = 0.35
        ax.bar(x - w/2, b["pct_sleeping_dog"] * 100, w,
               label="Base (sleeping-dog%)", color="#e74c3c", alpha=0.8)
        ax.bar(x + w/2, m["pct_sleeping_dog"] * 100, w,
               label="MSTPU (sleeping-dog%)", color="#c0392b", alpha=0.8)
        ax.bar(x - w/2, b["pct_persuadable"] * 100, w, bottom=0,
               label="Base (persuadable%)", color="#2ecc71", alpha=0.8,
               hatch="///")
        ax.bar(x + w/2, m["pct_persuadable"] * 100, w, bottom=0,
               label="MSTPU (persuadable%)", color="#27ae60", alpha=0.8,
               hatch="///")
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(mn, mn) for mn in b["base_model"]],
                            rotation=15, ha="right")
        ax.set_title(DGP_LABELS.get(dgp, dgp))
        if ax is axes[0]:
            ax.set_ylabel("% of top-30% targeted users")
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle(
        "Composition of Top-30% Targeting Set\n"
        "(red = sleeping dogs, τ<0; green = persuadables, τ>0)",
        fontsize=11, y=1.03,
    )
    _savefig(fig, out_dir, "fig5_sleeping_dog.pdf")


# ── LaTeX results table ────────────────────────────────────────────────────────
def generate_main_table(base: pd.DataFrame, mstpu: pd.DataFrame, out_dir: Path) -> None:
    """Two-panel table: base vs MSTPU at α=1.0, ρ=0.05."""
    b = (base[np.isclose(base["alpha"], 1.0) & np.isclose(base["rho"], 0.05)]
         .groupby(["dgp", "base_model", "learner"])
         .agg(ws_base=("wasted_spend","mean"), ie_base=("iroas_efficiency","mean"),
              q_base=("qini","mean"), pehe_base=("pehe","mean"))
         .reset_index())
    m = (mstpu[np.isclose(mstpu["alpha"], 1.0) & np.isclose(mstpu["rho"], 0.05)]
         .groupby(["dgp", "base_model", "learner"])
         .agg(ws_mstpu=("wasted_spend","mean"), ie_mstpu=("iroas_efficiency","mean"),
              q_mstpu=("qini","mean"))
         .reset_index())
    tbl = b.merge(m, on=["dgp","base_model","learner"])

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Benchmark results: base model vs.\ Multiplicative STPU "
        r"($\alpha=1.0$, $\rho=0.05$, mean over 5 seeds). "
        r"$\downarrow$ better for wasted spend; $\uparrow$ better for iROAS.}",
        r"\label{tab:main_results}",
        r"\small",
        r"\begin{tabular}{lll rr rr rr}",
        r"\toprule",
        r"DGP & Model & Lrnr & \multicolumn{2}{c}{Wasted Spend $\downarrow$} "
        r"& \multicolumn{2}{c}{iROAS Eff.\ $\uparrow$} "
        r"& \multicolumn{2}{c}{Qini $\uparrow$} \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r"& & & Base & MSTPU & Base & MSTPU & Base & MSTPU \\",
        r"\midrule",
    ]

    for dgp in DGP_ORDER:
        sub = tbl[tbl["dgp"] == dgp]
        if sub.empty:
            continue
        lines.append(r"\multicolumn{9}{l}{\textit{" + DGP_LABELS.get(dgp,dgp) + r"}} \\")
        for _, row in sub.iterrows():
            delta_ws = row["ws_mstpu"] - row["ws_base"]
            delta_ie = row["ie_mstpu"] - row["ie_base"]
            bold_ws   = r"\mathbf{" if delta_ws < 0  else ""
            bold_ws_e = r"}" if delta_ws < 0 else ""
            bold_ie   = r"\mathbf{" if delta_ie > 0  else ""
            bold_ie_e = r"}" if delta_ie > 0 else ""
            lines.append(
                f"  & {MODEL_LABELS.get(row['base_model'],row['base_model'])}"
                f" & {LEARNER_LABELS.get(row['learner'],row['learner'])}"
                f" & {row['ws_base']:.3f}"
                f" & ${bold_ws}{row['ws_mstpu']:.3f}{bold_ws_e}$"
                f" & {row['ie_base']:.3f}"
                f" & ${bold_ie}{row['ie_mstpu']:.3f}{bold_ie_e}$"
                f" & {row['q_base']:.3f}"
                f" & {row['q_mstpu']:.3f} \\\\"
            )
        lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path = out_dir / "tab_main_results.tex"
    path.write_text("\n".join(lines))
    print(f"  Saved: {path}")


# ── Entry point ────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Paper 2 figures.")
    parser.add_argument("--results", default="results/stpu_benchmark.csv")
    parser.add_argument("--out",     default="paper/figures/")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Results not found: {results_path}")
        print("Run: python experiments/run_stpu_benchmark.py first.")
        return

    print(f"Loading {results_path} …")
    df, base, mstpu = load_results(str(results_path))
    print(f"  {len(df):,} rows  |  {len(base):,} base  |  {len(mstpu):,} mstpu")

    print("\nGenerating figures …")
    fig_main_result(base, mstpu, out_dir)
    fig_additive_failure(df, out_dir)
    fig_beta_sensitivity(df, out_dir)
    fig_iroas_by_alpha(base, mstpu, out_dir)
    fig_sleeping_dog(base, mstpu, out_dir)
    generate_main_table(base, mstpu, out_dir)

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
