# %% [markdown]
# ## Features Analysis
# * should be longer than 10 years

# %% [markdown]
# * sharpe ratio (annualize, volatility issue, should ~ 0.5/above 1) /correlating pnl (should be low correlation)/ correlation matrix across different strategies
# * test on over 10 years
# * correct features before moving on

# %%
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from features_construct_zixuan import FEATURE_REGISTRY

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

def cumulate_return(df, date_col, feature_col, ret_col):
    df['decile'] = (
    df.groupby(date_col)[feature_col] # factors
    .transform(assign_decile)
    )
    raw_port_ret = df.groupby([date_col, 'decile'])[ret_col].mean()
    
    port_ret = raw_port_ret.unstack('decile')
    port_ret['LS'] = port_ret[10] - port_ret[1]
    port_ret['LS_cumret'] = (1 + port_ret['LS']).cumprod() - 1
    
    return port_ret.reset_index()

def run_trade_pipeline():
    # 1. 配置命令行参数解析
    parser = argparse.ArgumentParser(description="features pipeline")
    
    # 添加特征参数，nargs='+' 表示可以接受一个或多个值
    # choices=FEATURE_REGISTRY.keys() 可以限制用户只能输入已注册的特征
    parser.add_argument('--features', type=str, nargs='+', 
                        choices=list(FEATURE_REGISTRY.keys()),
                        help='enter the feature name: --features mom6m baspread')

    args = parser.parse_args()
    # call the function to trade the features
    temp1 = pd.read_parquet("data/raw/crsp_monthly.parquet")
    temp2 = pd.read_parquet(f"data/features/{args.features[0]}.parquet")

    temp1['date_m'] = pd.to_datetime(temp1['date']).dt.to_period('M')
    temp2['date_m'] = pd.to_datetime(temp2['date']).dt.to_period('M')
    
    panel = temp1.merge(temp2, on=['permno', 'date_m'], how='inner', suffixes=('', '_feat'))

    # panel[args.features[0]] = panel.groupby('date')[args.features[0]].transform(norm_rank)
    
    port_ret = cumulate_return(df=panel, date_col='date', feature_col=args.features[0], ret_col='ret')
    plt.plot(port_ret['date'], port_ret['LS_cumret'])
    plt.title(f"Cumulative Return of Long-Short Portfolio based on {args.features[0]}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    ax = plt.gca() # 
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0)) # 1.0 as 100%
    plt.savefig(f"outputs/{args.features[0]}_cumret.png")
    plt.show()
    
# %%
if __name__ == "__main__":
    run_trade_pipeline()
