"""
report_nn.py — OOS R² Report for Neural Network models (Gu-Kelly-Xiu 2020)
===========================================================================

Usage in any Colab notebook:
    from report_nn import report_nn, report_nn_yearly

    # Overall Table 1 style report:
    report_nn(results, arch="NN1")

    # Per-year table + figure:
    # yearly_records: list of dicts with keys: year, test_r2, avg_best_epoch
    report_nn_yearly(yearly_records, arch="NN1", save_path="figure_nn1.png")
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Paper Table 1 reference values (all, top1000, bot1000) in % ──────────────
PAPER_R2 = {
    'NN1': (0.33, 0.49, 0.38),
    'NN2': (0.39, 0.62, 0.46),
    'NN3': (0.40, 0.70, 0.45),
    'NN4': (0.39, 0.67, 0.47),
    'NN5': (0.36, 0.64, 0.42),
}


# ── Core R² ──────────────────────────────────────────────────────────────────

def oos_r2(y_true, y_pred):
    """R²_oos = 1 - SS_res / SS_tot  (zero benchmark, per GKX eq. 19)"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.sum(y_true ** 2)
    if denom == 0:
        return np.nan
    return 1.0 - np.sum((y_true - y_pred) ** 2) / denom


# ── Overall report (Table 1 style) ───────────────────────────────────────────

def report_nn(results, arch='NN1', save_path=None):
    """
    Print Table-1-style OOS R² for all stocks / top 1000 / bottom 1000.

    Parameters
    ----------
    results   : pd.DataFrame with columns [y_true, y_pred, DATE, mvel1]
    arch      : str, e.g. 'NN1' ... 'NN5'
    save_path : optional str — if provided, saves the text table to .txt file
    """
    arch = arch.upper()

    r2_all = oos_r2(results['y_true'], results['y_pred'])

    top1000 = (results.sort_values(['DATE', 'mvel1'], ascending=[True, False])
                      .groupby('DATE', sort=False).head(1000))
    bot1000 = (results.sort_values(['DATE', 'mvel1'], ascending=[True, True])
                      .groupby('DATE', sort=False).head(1000))

    r2_top = oos_r2(top1000['y_true'], top1000['y_pred'])
    r2_bot = oos_r2(bot1000['y_true'], bot1000['y_pred'])

    paper = PAPER_R2.get(arch, (None, None, None))

    w = 78
    lines = []
    lines.append('')
    lines.append('=' * w)
    lines.append(f'  {"Subsample":<38} {"OOS R²":>12}   {"Paper "+arch:>12}')
    lines.append('-' * w)

    def _paper_str(val):
        return f'~ +{val}%' if val is not None else ''

    lines.append(f'  {"All stocks (panel)":<38} {r2_all*100:>+11.4f}%   {_paper_str(paper[0]):>12}')
    lines.append(f'  {"Top 1,000 (largest mvel1)":<38} {r2_top*100:>+11.4f}%   {_paper_str(paper[1]):>12}')
    lines.append(f'  {"Bottom 1,000 (smallest mvel1)":<38} {r2_bot*100:>+11.4f}%   {_paper_str(paper[2]):>12}')
    lines.append('=' * w)

    text = '\n'.join(lines)

    if save_path:
        txt_path = str(save_path).rsplit('.', 1)[0] + '_summary.txt'
        with open(txt_path, 'w') as f:
            f.write(text + '\n')
        print(f'Summary saved to {txt_path}')

    print(text)


# ── Per-year report + figure ──────────────────────────────────────────────────

def report_nn_yearly(yearly_records, arch='NN1', save_path=None):
    """
    Print per-year OOS R² table and plot Figure-3-style chart.

    Parameters
    ----------
    yearly_records : list of dicts, each with keys:
                       year, test_r2, avg_best_epoch, best_l1, best_lr
    arch           : str, e.g. 'NN1'
    save_path      : optional str path for .png figure
                     (text table saved as same stem + '_yearly.txt')
    """
    arch = arch.upper()
    df = pd.DataFrame(yearly_records).sort_values('year').reset_index(drop=True)

    # ── text table ────────────────────────────────────────────────────────────
    w = 62
    lines = []
    lines.append('')
    lines.append('=' * w)
    lines.append(f'  {arch} — Per-year OOS R² and Avg Best Epoch')
    lines.append('-' * w)
    lines.append(f'  {"Year":<8} {"OOS R²":>10}  {"Avg Epoch":>12}  {"best_l1":>10}  {"best_lr":>8}')
    lines.append('-' * w)
    for _, row in df.iterrows():
        lines.append(
            f'  {int(row["year"]):<8}'
            f' {row["test_r2"]*100:>+10.4f}%'
            f'  {row["avg_best_epoch"]:>12.1f}'
            f'  {row["best_l1"]:>10.1e}'
            f'  {row["best_lr"]:>8.4f}'
        )
    lines.append('-' * w)
    mean_r2    = df['test_r2'].mean()
    mean_epoch = df['avg_best_epoch'].mean()
    lines.append(f'  {"Mean":<8} {mean_r2*100:>+10.4f}%  {mean_epoch:>12.1f}')
    lines.append('=' * w)

    text = '\n'.join(lines)

    if save_path:
        txt_path = str(save_path).rsplit('.', 1)[0] + '_yearly.txt'
        with open(txt_path, 'w') as f:
            f.write(text + '\n')
        print(f'Yearly table saved to {txt_path}')

    print(text)

    # ── figure ────────────────────────────────────────────────────────────────
    years  = df['year'].tolist()
    r2_pct = (df['test_r2'] * 100).tolist()
    epochs = df['avg_best_epoch'].tolist()

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(f'{arch} — Per-year OOS R² and Avg Best Epoch', fontsize=13)

    # top panel: OOS R²
    ax0 = axes[0]
    ax0.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax0.plot(years, r2_pct, color='steelblue', linewidth=1.0, marker='o', markersize=4)
    for x, y in zip(years, r2_pct):
        ax0.annotate(f'{y:+.2f}', xy=(x, y),
                     xytext=(0, 6), textcoords='offset points',
                     ha='center', va='bottom', fontsize=7)
    ymax = max(abs(v) for v in r2_pct) * 1.4
    ax0.set_ylim(-ymax, ymax)
    ax0.set_ylabel('OOS R² (%)')
    ax0.set_title('OOS R² per year')
    ax0.grid(axis='y', linestyle=':', alpha=0.5)

    # bottom panel: avg best epoch
    ax1 = axes[1]
    ax1.plot(years, epochs, color='steelblue', linewidth=1.0, marker='o', markersize=4)
    for x, y in zip(years, epochs):
        ax1.annotate(f'{y:.0f}', xy=(x, y),
                     xytext=(0, 6), textcoords='offset points',
                     ha='center', va='bottom', fontsize=7)
    ax1.set_ylim(0, max(epochs) * 1.25)
    ax1.set_ylabel('Avg Best Epoch')
    ax1.set_title('Avg Best Epoch per year')
    ax1.set_xlabel('Year')
    ax1.grid(axis='y', linestyle=':', alpha=0.5)

    plt.xticks(years, rotation=45, fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Figure saved to {save_path}')

    plt.show()
    plt.close()
