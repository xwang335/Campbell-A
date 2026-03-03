# %%
import pandas as pd
import numpy as np
import os
import wrds

# %%
FEATURE_REGISTRY = {}

def register_feature(name):
    def decorator(func):
        FEATURE_REGISTRY[name] = func
        return func
    return decorator

# %%
def pull_data_from_wrds(start, end, freq):
    # Connect to WRDS
    db = wrds.Connection()

    # Pull data from the 'msf' table for the specified date range
    if freq == 'M':
        query = f"""
        SELECT 
            b.permno,
            b.date,
            b.ret,
            b.prc,
            b.shrout,
            c.rf,
            (b.ret - c.rf) AS exret
        FROM crsp.msf b
        LEFT JOIN ff.factors_monthly c
            ON date_trunc('month', b.date) = date_trunc('month', c.date)
        WHERE b.date >= '{start}'
        AND b.date <= '{end}'
        AND b.ret IS NOT NULL
        """
    if freq == 'D':
        query = f"""
        select
            permno,
            date,
            ret,
            prc,
            vol,
            shrout,
            askhi,
            bidlo
        from crsp.dsf
        WHERE date >= '{start}'
        AND date <= '{end}'
        """
    df = db.raw_sql(query)
    db.close()
    df['me'] = df['prc'].abs() * df['shrout']
    return df

# %%
def clean_data(df):
    # Remove rows with missing values in 'ret' column
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['permno', 'date']).reset_index(drop=True)
    print(df.describe())  # Debugging line to check data after cleaning
    return df

# %%
@register_feature('mom1m')
def mom1m_feature(df):
    # Calculate momentum features
    msf = df.copy()
    msf = msf.sort_values(['permno', 'date']).reset_index(drop=True)
    msf['mom1m'] =  msf.groupby('permno')['ret'].shift(1)
    
    msf = msf.dropna(
        subset=['mom1m']
    ).reset_index(drop=True)

    keep_cols = ['permno', 'date', 'mom1m']
    msf = msf.loc[:, keep_cols]
    msf = msf.sort_values(['date', 'permno']).reset_index(drop=True) 
    return msf

@register_feature('mom6m')
def mom6m_feature(df):
    msf = df.copy()
    msf = msf.sort_values(['permno', 'date']).reset_index(drop=True)
    msf['mom6m'] = (msf.groupby('permno')['ret']
                    .apply(lambda x: (1.0+x.shift(2)).rolling(5, min_periods=5)
                    .apply(np.prod, raw=True) - 1.0).reset_index(level=0,drop=True))
    
    msf = msf.dropna(
        subset=['mom6m']
    ).reset_index(drop=True)

    keep_cols = ['permno', 'date', 'mom6m']
    msf = msf.loc[:, keep_cols]
    msf = msf.sort_values(['date', 'permno']).reset_index(drop=True) 
    return msf

@register_feature('mom12m')
def mom12m_feature(df):
    msf = df.copy()
    msf = msf.sort_values(['permno', 'date']).reset_index(drop=True)

    msf['mom12m'] = (msf.groupby('permno')['ret']
                    .apply(lambda x: (1.0+x.shift(2)).rolling(11, min_periods=11)
                    .apply(np.prod, raw=True) - 1.0).reset_index(level=0,drop=True))
    
    msf = msf.dropna(
        subset=['mom12m']
    ).reset_index(drop=True)
    
    keep_cols = ['permno', 'date', 'mom12m']
    msf = msf.loc[:, keep_cols]
    msf = msf.sort_values(['date', 'permno']).reset_index(drop=True) 
    return msf

@register_feature('baspread')
def baspread_feature(df):
    # Calculate bid-ask spread feature, not lagged yet
    dsf = df.copy()
    dsf['yr'] = dsf['date'].dt.year
    dsf['month'] = dsf['date'].dt.month
    
    baspread = (
        dsf.groupby(['permno', 'yr', 'month'])
        .apply(lambda x: ((x['askhi'] - x['bidlo']) / ((x['askhi'] + x['bidlo']) / 2)).mean())
        .reset_index(name='baspread')
    )
    baspread['date'] = pd.to_datetime(
        dict(year=baspread['yr'], month=baspread['month'], day=1)
    )
    # SHIFT FORWARD ONE MONTH (lag alignment)
    baspread['date'] = baspread['date'] + pd.offsets.MonthBegin(1)
    
    keep_cols = ['permno', 'date', 'baspread']
    baspread = baspread.loc[:, keep_cols]
    baspread = baspread.sort_values(['date', 'permno']).reset_index(drop=True)
    return baspread

# %%
start = '1956-01-01'
end = '1987-12-31'

# Check if the raw data file exists, if not, pull from WRDS and save
if not os.path.exists("data/raw/crsp_monthly.parquet"):
    df = pull_data_from_wrds(start=start, end=end, freq='M')
    df = clean_data(df)
    print(df['exret'])
    df.to_parquet("data/raw/crsp_monthly.parquet")

# Check if the momentum features file exists, if not, create it
if not os.path.exists("data/features/mom1m.parquet"):
    # call to replicate the features
    df = pull_data_from_wrds(start=start, end=end, freq='M')
    df = clean_data(df)
    msf = mom1m_feature(df)
    msf.to_parquet("data/features/mom1m.parquet")

if not os.path.exists("data/features/mom6m.parquet"):
    # call to replicate the features
    df = pull_data_from_wrds(start=start, end=end, freq='M')
    df = clean_data(df)
    msf = mom6m_feature(df)
    msf.to_parquet("data/features/mom6m.parquet")

if not os.path.exists("data/features/mom12m.parquet"):
    # call to replicate the features
    df = pull_data_from_wrds(start=start, end=end, freq='M')
    df = clean_data(df)
    msf = mom12m_feature(df)
    msf.to_parquet("data/features/mom12m.parquet")

# Check if the bid-ask spread features file exists, if not, create it
if not os.path.exists("data/features/baspread.parquet"):
    # call to replicate the features
    df = pull_data_from_wrds(start=start, end=end, freq='D')
    df = clean_data(df)
    dsf = baspread_feature(df)
    dsf.to_parquet("data/features/baspread.parquet")
