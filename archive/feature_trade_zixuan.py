# %% [markdown]
# ## Features Analysis
# * should be longer than 10 years

# %% [markdown]
# * sharpe ratio (annualize, volatility issue, should ~ 0.5/above 1) /correlating pnl (should be low correlation)/ correlation matrix across different strategies
# * test on over 10 years
# * correct features before moving on

# %%
import argparse
from curses import panel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from rawdata_obtain import FEATURE_REGISTRY

# %%
def norm_rank(data):
    ranks=data.rank(method='average',na_option='keep')
    n=ranks.count()
    mapped = (ranks / (n + 1)) * 2 - 1
    return mapped.fillna(0)

def assign_decile(x):
    if len(x) < 10: # enough data points to assign deciles
        return pd.Series(index=x.index, data=np.nan)
    ranks = x.rank(method='first', ascending=True)
    return pd.qcut(ranks, 10, labels=False) + 1

def cumulate_return(df, date_col, feature_col, ret_col, weighting="equal"):
    df['decile'] = (
    df.groupby(date_col)[feature_col] # factors
    .transform(assign_decile)
    )
    if weighting == "equal":
        raw_port_ret = df.groupby([date_col, 'decile'])[ret_col].mean()
    
    if weighting == "value":
        df = df.sort_values(['permno', date_col])
        df['me_lag'] = df.groupby('permno')['me'].shift(1)
        # print(df[['date','permno','me','me_lag']].head(20))
        df = df.dropna(subset=['me_lag'])

        raw_port_ret = (
        df.groupby([date_col, 'decile'])
        .apply(lambda x: np.sum(x[ret_col] * x['me_lag']) / np.sum(x['me_lag']), include_groups=False)
        ) 

    port_ret = raw_port_ret.unstack('decile')
    port_ret['LS'] = port_ret[10] - port_ret[1]
    #port_ret['LS_cumret'] = (1 + port_ret['LS']).cumprod() - 1
    port_ret['LS_ariret'] = port_ret['LS'].cumsum()
    
    return port_ret.reset_index()

def ret_plot(port_ret, feature_name, weighting):
    plt.plot(port_ret['date'], port_ret['LS_ariret'])
    plt.title(f"{weighting.capitalize()}-Weighted Long-Short based on {feature_name}")
    plt.xlabel("Date")
    plt.ylabel("Arithmetic Return")
    plt.grid(True)
    ax = plt.gca() 
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0)) # 1.0 as 100%
    plt.savefig(f"outputs/{feature_name}_{weighting}_ret.png")
    #plt.show()
    print(f"--- Return Plot Saved to outputs/{feature_name}_{weighting}_ret.png ---")

def ret_summary(port_ret):
    ls_ret = port_ret['LS'].dropna()

    T = len(ls_ret)
    mean_monthly = ls_ret.mean()
    std_monthly = ls_ret.std()

    t_stat = mean_monthly / (std_monthly / np.sqrt(T))
    sharpe = (mean_monthly / std_monthly) * np.sqrt(12)

    return mean_monthly, std_monthly, t_stat, sharpe

def predictiveR2(df, feature_name, date_col, split_date):
    # print(f"Data before dropping NA for {feature_name} - shape: {df.shape}")
    # print(df['mom1m'].isna().mean())
    # print(df['ret'].isna().mean())
    # print(df['exret'].isna().mean())
    
    df = df.dropna(subset=[feature_name, 'exret'])

    # print(f"Data after dropping NA for {feature_name} - shape: {df.shape}")

    df_train = df[df[date_col] < split_date]
    df_test = df[df[date_col] >= split_date]

    # print(f"Training set for {feature_name} - shape: {df_train.shape}")
    # print(f"Test set for {feature_name} - shape: {df_test.shape}")

    # perform OLS regression on the training set
    X_train = df_train[feature_name]
    X_train = np.column_stack([np.ones(X_train.shape[0]), X_train])  # add intercept
    # check
    # print(f"Training data for {feature_name} - X shape: {X_train.shape}, y shape: {df_train['exret'].shape}")

    y = df_train['exret'] # use excess return for R-squared calculation to align with author
    beta = np.linalg.lstsq(X_train, y, rcond=None)[0]
    # predict on the test set
    X_test = df_test[feature_name]
    X_test = np.column_stack([np.ones(X_test.shape[0]), X_test])  # add intercept
    y_test = df_test['exret']
    y_pred = X_test @ beta
    # calculate R-squared
    ss_res = (y_test - y_pred).T @ (y_test - y_pred)
    ss_tot = y_test.T @ y_test # do not demean y_test to align with author
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return r2


def run_trade_pipeline():
    # parse the feature name from command line arguments
    parser = argparse.ArgumentParser(description="features pipeline")
    
    parser.add_argument('--features', type=str, nargs='+', 
                        choices=list(FEATURE_REGISTRY.keys()),
                        help='enter the feature name: --features mom6m baspread')
    parser.add_argument('--weighting', type=str,
                    choices=['equal', 'value'],
                    default='equal',
                    help='portfolio weighting scheme')
    
    args = parser.parse_args()
    # call the function to trade the features
    temp1 = pd.read_parquet("data/raw/crsp_monthly.parquet")

    # print(temp1['exret'])  # Debugging line to check data before cleaning
    temp2 = pd.read_parquet(f"data/features/{args.features[0]}.parquet")

    temp1['date_m'] = pd.to_datetime(temp1['date']).dt.to_period('M')
    temp2['date_m'] = pd.to_datetime(temp2['date']).dt.to_period('M')
    
    panel = temp1.merge(temp2, on=['permno', 'date_m'], how='inner', suffixes=('', '_feat'))
    # print(panel['exret'])  # Debugging line to check data before cleaning
    port_ret = cumulate_return(df=panel, date_col='date', feature_col=args.features[0], ret_col='ret', weighting=args.weighting)

    ret_plot(port_ret, args.features[0], args.weighting)
    mean_monthly, std_monthly, t_stat, sharpe = ret_summary(port_ret)
    r2 = predictiveR2(panel, args.features[0],'date_m', split_date='1972-01-01')

    output_text = (
        f"Mean Monthly Return: {mean_monthly:.4f}\n"
        f"Standard Deviation of Monthly Returns: {std_monthly:.4f}\n"
        f"T-Statistic: {t_stat:.2f}\n"
        f"Annualized Sharpe Ratio: {sharpe:.2f}\n"
        f"Predictive R-squared: {r2:.5f}\n"
    )
    file_path = f"outputs/{args.features[0]}_{args.weighting}_summary.txt"
    with open(file_path, "w") as f:
        f.write(output_text)

    print(f"--- Summary Results Saved to {file_path} ---")
    print(output_text)
    
# %%
if __name__ == "__main__":
    run_trade_pipeline()
