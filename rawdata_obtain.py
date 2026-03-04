# %%
import argparse

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
parser = argparse.ArgumentParser(description="features pipeline")
parser.add_argument('--start', type=str, nargs='+', 
                    help='enter the start date for raw data pull: --start 1956-01-01')
parser.add_argument('--end', type=str, nargs='+',
                    help='enter the end date for raw data pull: --end 2016-12-31')

args = parser.parse_args()
start = args.start[0]
end = args.end[0]

# Check if the raw data file exists, if not, pull from WRDS and save
if not os.path.exists("data/raw/crsp_monthly.parquet"):
    df = pull_data_from_wrds(start=start, end=end, freq='M')
    df = clean_data(df)
    print(df['exret'])
    df.to_parquet("data/raw/crsp_monthly.parquet")

