import numpy as np
from model import Model
from scipy.special import expit as sigmoid


GOOD = 1
BAD = -1


class SVM(Model):
    def __init__(self, iterations=None, eps=0.0001, lambda_par=0.001):
        self.w = None
        self.bias = None
        self._lambda = lambda_par
        self.T = iterations

    def _gradient(self, x, y):
        w = self._lambda * self.w
        bias = 0.0
        if y * (np.dot(self.w, x) + self.bias) < 1:
            w = w - y*x
            bias = bias - y
        return (bias, w)

    def _loss(self, x, y):
        reg = self._lambda * np.dot(self.w, self.w) / 2
        hinge = max(0, 1 - y * (np.dot(self.w, x) + self.bias))
        return hinge + reg

    def fit(self, train_X, train_Y):
        samples_number = len(train_X)
        self.w = np.zeros(len(train_X[0]))
        self.bias = 0.0
        rng = np.random.default_rng()
        if self.T is None:
            self.T = 10 * samples_number
        for t in range(1, self.T+1):
            idx = rng.integers(0, samples_number)
            x_t = train_X[idx]
            y_t = train_Y[idx]
            eta = 1 / (t * self._lambda)
            bias_der, w_grad = self._gradient(x_t, y_t)
            self.w -= eta * w_grad
            self.bias -= eta * bias_der

    def predict(self, X_test):
        return np.array([np.sign(np.dot(self.w, x) + self.bias) for x in X_test])



def to_minus_one_one(x):
    return 2*x - 1

class LogReg(Model):
    def __init__(self, iterations=None, lambda_par=0.001, kernel="linear"):
        self._lambda = lambda_par
        self.T = iterations
        self.kernel = kernel

    def fit(self, X, Y):
        n_samples = len(X)
        self.w = np.zeros(len(X[0]))
        self.bias = 0.0
        rng = np.random.default_rng()
        if self.T is None:
            self.T = 10 * n_samples
        for t in range(1, self.T+1):
            idx = rng.integers(0, n_samples)
            x_t = X[idx]
            y_t = Y[idx]
            eta = 1 / (t * self._lambda)
            common = eta * y_t * sigmoid(-y_t * (np.dot(self.w, x_t) + self.bias))
            self.w += common * x_t - self.w / t
            self.bias += common

    def predict_proba(self, X_test):
        return np.array([sigmoid(np.dot(self.w, x) + self.bias) for x in X_test])

    def predict(self, X_test, threshold=0.5):
        return to_minus_one_one( (self.predict_proba(X_test) >= threshold).astype(int) )
