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


class Scaler():
    def __init__(self, df: pd.DataFrame, columns=None):
        self.columns = df.columns if columns is None else columns
        self.max = df[self.columns].max()
        self.min = df[self.columns].min()
    
    def _scale_column(self, feature: pd.Series):
        min = self.min[feature.name]
        max = self.max[feature.name]
        return (feature - min) / (max - min)

    def scale(self, df: pd.DataFrame):
        scaled = pd.DataFrame(columns=df.columns)
        scaled[self.columns] = df[self.columns].apply(self._scale_column)
        diff = df.columns.difference(self.columns)
        if not diff.empty:
            scaled[diff] = df[diff]
        return scaled

def scale(df: pd.DataFrame, columns=None):
    scaled = pd.DataFrame(columns=df.columns)
    scaled[columns] = df[columns].apply(lambda feature: (feature - feature.min()) / (feature.max() - feature.min()))
    diff = df.columns.difference(columns)
    if not diff.empty:
        scaled[diff] = df[diff]
    return scaled
