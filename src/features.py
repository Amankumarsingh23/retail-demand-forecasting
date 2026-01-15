import pandas as pd

def create_time_features(df):
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    return df

def create_lag_features(df, lags=[7, 14, 28]):
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby(["store", "item"])["sales"].shift(lag)
    return df

def create_rolling_features(df, windows=[7, 14, 28]):
    df = df.copy()
    for window in windows:
        df[f"rolling_mean_{window}"] = (
            df.groupby(["store", "item"])["sales"]
            .shift(1)
            .rolling(window)
            .mean()
        )
    return df
