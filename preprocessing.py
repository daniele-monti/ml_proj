import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from imblearn.combine import SMOTETomek


def outlier_limits(data, kind):
    lower_limits = np.min(data, axis=0)
    upper_limits = np.max(data, axis=0)
    if kind == "iqr":
        q1 = np.quantile(data, 0.25, axis=0)
        q3 = np.quantile(data, 0.75, axis=0)
        iqr = q3 - q1
        lower_limits = q1 - 1.5*iqr
        upper_limits = q3 + 1.5*iqr
    if kind == "zscore":
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        lower_limits = mean - 3*std
        upper_limits = mean + 3*std
    if kind == "percentile":
        lower_limits = np.quantile(data, 0.01, axis=0)
        upper_limits = np.quantile(data, 0.99, axis=0)
    return lower_limits, upper_limits


class IQRClipper:
    def __init__(self):
        return
    def fit(self, X):
        self.lb, self.ub = outlier_limits(X, 'iqr')
        return 
    def transform(self, X):
        return np.clip(X, a_min=self.lb, a_max=self.ub)
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class StandardClipper:
    def __init__(self):
        return
    def fit(self, X):
        self.lb, self.ub = outlier_limits(X, 'zscore')
        return 
    def transform(self, X):
        return np.clip(X, a_min=self.lb, a_max=self.ub)
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

scalers = {
    'standard': StandardScaler(),
    'minmax': MinMaxScaler(),
    'robust': RobustScaler()
}

rebalancers = {
    'smotetomek': SMOTETomek()
}

clippers ={
    'iqr': IQRClipper(),
    'z-score': StandardClipper()
}

def preprocess(train_x, train_y, test_x, test_y, scaler='standard', rebalancer='smotetomek', clipper='zscore'):
    if clipper is not None:
        train_x = clippers[clipper].fit_transform(train_x)
        test_x = clippers[clipper].transform(test_x)
    if scaler is not None:
        train_x = scalers[scaler].fit_transform(train_x)
        test_x = scalers[scaler].transform(test_x)
    if rebalancer is not None:
        train_x, train_y = rebalancers[rebalancer].fit_resample(train_x, train_y)
    return train_x, train_y, test_x, test_y
