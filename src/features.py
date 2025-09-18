import pandas as pd
import numpy as np

def climatology(df, value="t_mean"):
    # media climática por ciudad y mes
    base = (df.groupby(["city","month"])[value]
              .mean().rename(f"{value}_clim").reset_index())
    return base

def add_anomaly(df, clim, value="t_mean"):
    out = df.merge(clim, on=["city","month"], how="left")
    out[f"{value}_anomaly"] = out[value] - out[f"{value}_clim"]
    return out

def rolling_features(df):
    out = df.sort_values(["city","year","month"]).copy()
    out["t_mean_roll12"] = (out.groupby("city")["t_mean"]
                              .transform(lambda s: s.rolling(12, min_periods=6).mean()))
    if "precip_mm" in out:
        out["precip_roll12"] = (out.groupby("city")["precip_mm"]
                                  .transform(lambda s: s.rolling(12, min_periods=6).sum()))
    return out

def degree_days(df, base=18.0):
    # Cooling Degree Days (CDD) simplificado mensual
    if "t_mean" not in df: return df
    df["cdd"] = (df["t_mean"] - base).clip(lower=0)
    df["hdd"] = (base - df["t_mean"]).clip(lower=0)
    return df
