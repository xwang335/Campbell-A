"""
report_r2.py — Universal OOS R² Report for Gu-Kelly-Xiu (2020) Table 1
=======================================================================

Usage in any Colab notebook:
----------------------------
    from report_r2 import report, report_yearly

    # Overall Table 1 style report:
    report(results, "glm")

    # Per-year R² table + Figure 3 style plot:
    # yearly_records is a list of dicts with keys: year, test_r2, n_active
    report_yearly(yearly_records, model_name="glm", n_total_groups=920)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ── Paper Table 1 reference values (%) ───────────────────────────────────────
PAPER_R2 = {
    "ols":    (+0.16, +0.31, +0.17),
    "ols3":   (+0.16, +0.31, +0.17),
    "ols3+h": (+0.16, +0.31, +0.17),
    "pls":    (+0.27, +0.29, +0.22),
    "pcr":    (+0.27, +0.29, +0.22),
    "enet":   (+0.11, +0.25, +0.34),
    "enet+h": (+0.11, +0.25, +0.34),
    "glm":    (+0.19, +0.14, +0.30),
    "glm+h":  (+0.19, +0.14, +0.30),
    "rf":     (+0.33, +0.63, +0.35),
    "gbrt":   (+0.34, +0.52, +0.32),
    "gbrt+h": (+0.34, +0.52, +0.32),
    "nn1":    (+0.33, +0.49, +0.38),
    "nn2":    (+0.39, +0.62, +0.46),
    "nn3":    (+0.40, +0.70, +0.45),
    "nn4":    (+0.39, +0.67, +0.47),
    "nn5":    (+0.36, +0.64, +0.42),
}

DISPLAY_NAMES = {
    "ols":    "OLS",
    "ols3":   "OLS-3+H",
    "ols3+h": "OLS-3+H",
    "pls":    "PLS",
    "pcr":    "PCR",
    "enet":   "ENet+H",
    "enet+h": "ENet+H",
    "glm":    "GLM+H",
    "glm+h":  "GLM+H",
    "rf":     "RF",
    "gbrt":   "GBRT+H",
    "gbrt+h": "GBRT+H",
    "nn1":    "NN1",
    "nn2":    "NN2",
    "nn3":    "NN3",
    "nn4":    "NN4",
    "nn5":    "NN5",
}


# ── Core R² calculation ───────────────────────────────────────────────────────

def oos_r2(y_true, y_pred):
    """R²_oos = 1 - sum((r - r_hat)^2) / sum(r^2)  (zero benchmark)"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.sum(y_true ** 2)
    if denom == 0:
        return np.nan
    return 1.0 - np.sum((y_true - y_pred) ** 2) / denom


def compute_r2(results):
    """
    Compute OOS R² for all stocks / top 1000 / bottom 1000.

    Parameters
    ----------
    results : pd.DataFrame
        Must have columns: DATE, mvel1, y_true, y_pred

    Returns
    -------
    (r2_all, r2_top, r2_bot) as raw fractions (not %)
    """
    r2_all = oos_r2(results["y_true"], results["y_pred"])

    top1000 = (
        results.sort_values(["DATE", "mvel1"], ascending=[True, False])
        .groupby("DATE", sort=False).head(1000)
    )
    r2_top = oos_r2(top1000["y_true"], top1000["y_pred"])

    bot1000 = (
        results.sort_values(["DATE", "mvel1"], ascending=[True, True])
        .groupby("DATE", sort=False).head(1000)
    )
    r2_bot = oos_r2(bot1000["y_true"], bot1000["y_pred"])

    return r2_all, r2_top, r2_bot


# ── Table 1 style overall report ─────────────────────────────────────────────

def report(results, model_name):
    """
    Compute R² and print Table 1 style comparison.

    Parameters
    ----------
    results : pd.DataFrame
        columns: DATE, permno, mvel1, y_true, y_pred
    model_name : str
        e.g. "glm", "enet", "nn1", ...
    """
    key = model_name.strip().lower()
    display = DISPLAY_NAMES.get(key, model_name)
    has_paper = key in PAPER_R2

    r2_all, r2_top, r2_bot = compute_r2(results)
    my_vals = (r2_all, r2_top, r2_bot)

    labels = [
        "All stocks (panel)",
        "Top 1,000 (largest mvel1)",
        "Bottom 1,000 (smallest mvel1)",
    ]

    if has_paper:
        paper = PAPER_R2[key]
        w = 78
        print()
        print("=" * w)
        print(f"  {'Subsample':<35} {'OOS R²':>10}  {f'Paper {display}':>15}")
        print("-" * w)
        for label, r2, p_val in zip(labels, my_vals, paper):
            print(f"  {label:<35} {r2*100:>+10.4f}%  {f'~ {p_val:+.2f}%':>15}")
        print("=" * w)
    else:
        w = 55
        print()
        print(f"  (Model '{display}' not in paper lookup — showing R² only)")
        print("=" * w)
        print(f"  {'Subsample':<35} {'OOS R²':>10}")
        print("-" * w)
        for label, r2 in zip(labels, my_vals):
            print(f"  {label:<35} {r2*100:>+10.4f}%")
        print("=" * w)

    return {"r2_all": r2_all, "r2_top": r2_top, "r2_bot": r2_bot}


# ── Per-year R² table + Figure 3 ─────────────────────────────────────────────

def report_yearly(yearly_records, model_name="glm", n_total_groups=920, save_path=None):
    """
    Print per-year OOS R² and active groups, and plot Figure 3 style chart.

    Parameters
    ----------
    yearly_records : list of dict
        Each dict must have keys: year, test_r2, n_active
    model_name : str
        For plot title, e.g. "glm"
    n_total_groups : int
        Total number of feature groups, default 920
    save_path : str or None
        If provided, saves the figure to this path
    """
    key = model_name.strip().lower()
    display = DISPLAY_NAMES.get(key, model_name)

    df = pd.DataFrame(yearly_records).sort_values("year").reset_index(drop=True)

    # ── Per-year table ────────────────────────────────────────
    w = 55
    print()
    print("=" * w)
    print(f"  {display} — Per-year OOS R² and Active Groups")
    print("-" * w)
    print(f"  {'Year':<8} {'OOS R²':>10}  {'Active Groups':>15}")
    print("-" * w)
    for _, row in df.iterrows():
        print(
            f"  {int(row['year']):<8}"
            f" {row['test_r2']*100:>+10.4f}%"
            f"  {int(row['n_active']):>4}/{n_total_groups}"
        )
    print("-" * w)
    mean_r2 = df["test_r2"].mean()
    mean_active = df["n_active"].mean()
    print(
        f"  {'Mean':<8}"
        f" {mean_r2*100:>+10.4f}%"
        f"  {mean_active:>7.1f}/{n_total_groups}"
    )
    print("=" * w)

    # ── Figure 3 style plot ───────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Top panel: OOS R²
    axes[0].plot(df["year"], df["test_r2"] * 100,
                 color="steelblue", linewidth=1.5, marker="o", markersize=4)
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("OOS R² (%)")
    axes[0].set_title(f"{display} — Per-year OOS R²")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # Bottom panel: Active groups (Figure 3 style)
    axes[1].plot(df["year"], df["n_active"],
                 color="darkorange", linewidth=1.5, marker="o", markersize=4)
    axes[1].set_ylabel("# of Active Features")
    axes[1].set_xlabel("Year")
    axes[1].set_title(f"{display} — Time-varying Model Complexity (Active Groups / {n_total_groups})")
    axes[1].set_ylim(0, n_total_groups * 1.05)
    axes[1].grid(True, linestyle="--", alpha=0.4)

    axes[1].set_xlim(df["year"].min() - 0.5, df["year"].max() + 0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")

    plt.show()

    return df