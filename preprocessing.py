import pandas as pd

def outlier_limits(df: pd.DataFrame, kind):
    lower_limit = df.min()
    upper_limit = df.max()
    if kind == "iqr":
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        iqr = q3 - q1
        lower_limit = q1 - 1.5*iqr
        upper_limit = q3 + 1.5*iqr
    if kind == "zscore":
        mean = df.mean()
        std = df.std()
        lower_limit = mean - 3*std
        upper_limit = mean + 3*std
    if kind == "percentile":
        lower_limit = df.quantile(0.01)
        upper_limit = df.quantile(0.99)
    return lower_limit, upper_limit


def clip_outliers(df: pd.DataFrame, columns=None, kind='iqr'):
    if columns is None:
        lb, ub = outlier_limits(df, kind)
        return pd.DataFrame(df.clip(lb, ub, axis=1))
    else:
        lb, ub = outlier_limits(df[columns], kind)
        ret = pd.DataFrame(data=df[columns].clip(lb, ub, axis=1), columns=df.columns)
        diff = df.columns.difference(columns) 
        ret[diff] = df[diff] 
        return ret


def scale(df: pd.DataFrame, columns=None):
    if columns is None:
        return df.apply(lambda feature: (feature - feature.min()) / (feature.max() - feature.min()))
    else:
        ret = pd.DataFrame(
            data=df[columns].apply(lambda feature: (feature - feature.min()) / (feature.max() - feature.min())),
            columns=df.columns
        )
        diff = df.columns.difference(columns) 
        ret[diff] = df[diff] 
        return ret
