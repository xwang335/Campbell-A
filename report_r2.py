"""
report_r2.py — Universal OOS R² Report for Gu-Kelly-Xiu (2020) Table 1
=======================================================================

Usage in any Colab notebook:
----------------------------
    # results is your DataFrame with columns: DATE, permno, mvel1, y_true, y_pred
    from report_r2 import report

    report(results, "enet")    # prints ENet+H vs paper
    report(results, "ols3")    # prints OLS-3+H vs paper
    report(results, "pcr")     # prints PCR vs paper
    report(results, "pls")     # prints PLS vs paper
    report(results, "nn1")     # prints NN1 vs paper
    ...

    # Custom model not in paper:
    report(results, "my_model")  # prints R² only, no paper column
"""

import numpy as np
import pandas as pd


# ── Paper Table 1 reference values (%) ───────────────────────────────────────
# Source: Gu, Kelly & Xiu (2020), Table 1
# Keys: model name (lowercase) -> (r2_all, r2_top, r2_bot) in percent
PAPER_R2 = {
    "ols":   (+0.16, +0.31, +0.17),   # OLS (all 94 chars, no Huber)
    "ols3":  (+0.16, +0.31, +0.17),   # OLS-3+H
    "ols3+h":(+0.16, +0.31, +0.17),
    "pls":   (+0.27, +0.29, +0.22),   # PLS
    "pcr":   (+0.27, +0.29, +0.22),   # PCR
    "enet":  (+0.11, +0.25, +0.34),   # ENet+H
    "enet+h":(+0.11, +0.25, +0.34),
    "glm":   (+0.11, +0.25, +0.34),   # GLM (same as ENet in paper)
    "rf":    (+1.35, +0.28, +2.65),   # Random Forest
    "gbrt":  (+0.34, +0.26, +0.45),   # GBRT
    "nn1":   (+0.36, +0.33, +0.39),   # NN1
    "nn2":   (+0.37, +0.33, +0.40),   # NN2
    "nn3":   (+0.36, +0.33, +0.40),   # NN3
    "nn4":   (+0.36, +0.33, +0.39),   # NN4
    "nn5":   (+0.37, +0.33, +0.40),   # NN5
}

# Display name mapping (for prettier headers)
DISPLAY_NAMES = {
    "ols":   "OLS",
    "ols3":  "OLS-3+H",
    "ols3+h":"OLS-3+H",
    "pls":   "PLS",
    "pcr":   "PCR",
    "enet":  "ENet+H",
    "enet+h":"ENet+H",
    "glm":   "GLM",
    "rf":    "RF",
    "gbrt":  "GBRT",
    "nn1":   "NN1",
    "nn2":   "NN2",
    "nn3":   "NN3",
    "nn4":   "NN4",
    "nn5":   "NN5",
}


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


def report(results, model_name):
    """
    Compute R² and print Table 1 style comparison.

    Parameters
    ----------
    results : pd.DataFrame
        columns: DATE, permno, mvel1, y_true, y_pred
    model_name : str
        e.g. "ols3", "enet", "pcr", "pls", "nn1", ...
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
