import pandas as pd

def summary_stats(df, value_cols):
    return (df[value_cols]
            .agg(["count","mean","median","min","max","std","quantile"])
            )

def yearly_stats(df, value_col):
    return (df.groupby("year")[value_col]
              .agg(["count","mean","median","min","max"])
              .reset_index())

def rankings(df, value_col, n=5, ascending=False):
    return (df[["country","year",value_col]]
            .sort_values([value_col], ascending=ascending)
            .groupby("year")
            .head(n))

def correlation(df, col_x, col_y):
    return df[[col_x, col_y]].corr().loc[col_x, col_y]

def rate_of_change(df, by="country", value_col="rate"):
    return (df.sort_values(["country","year"])
              .groupby(by)[value_col].pct_change())
