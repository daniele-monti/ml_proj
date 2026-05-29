import numpy as np
from model import Model
from scipy.special import expit as sigmoid


class LinearSVM(Model):
    def __init__(            
        self,
        max_iter=1000,
        lambda_=0.0001,
        tol=1e-3,
        max_iter_no_changes=5,
    ):
        self.lambda_ = lambda_
        self.tol = tol
        self.max_iter_no_changes = max_iter_no_changes
        self.max_iter = max_iter

    def _gradient(self, x, y):
        w = self.lambda_ * self.w
        bias = 0.0
        if y * (np.dot(self.w, x) + self.bias) < 1:
            w = w - y*x
            bias = bias - y
        return (bias, w)

    def __loss(self, X, y):
        reg = self.lambda_ * np.dot(self.w, self.w) / 2
        hinge = 0.0
        for x, y in zip(X, y):
            hinge += max(0, 1 - y * (np.dot(self.w, x) + self.bias))
        return (hinge / len(X)) + reg

    def fit(self, train_X, train_Y):
        n_samples = len(train_X)
        self.w = np.zeros(len(train_X[0]))
        self.bias = 0.0
        rng = np.random.default_rng()
        sample_indices = np.array(range(n_samples))
        best_loss = np.inf
        n_iter_no_changes = 0
        t = 0
        for epoch in self.max_iter:
            rng.shuffle(sample_indices)
            for idx in sample_indices:
                x_t = train_X[idx]
                y_t = train_Y[idx]
                eta = 1 / (t * self.lambda_)
                bias_der, w_grad = self._gradient(x_t, y_t)
                self.w -= eta * w_grad
                self.bias -= eta * bias_der
            epoch_loss = self.__loss(train_X, train_Y)
            if epoch_loss <= best_loss - self.tol:
                best_loss = epoch_loss
                n_iter_no_changes = 0
            else:
                n_iter_no_changes += 1
            if n_iter_no_changes > self.max_iter_no_changes:
                break
        
    def predict(self, X_test):
        return np.array([np.sign(np.dot(self.w, x) + self.bias) for x in X_test])



def to_minus_one_one(x):
    return 2*x - 1

class LinearLogReg(Model):
    def __init__(            
        self,
        max_iter=1000,
        lambda_=0.0001,
        tol=1e-3,
        max_iter_no_changes=5,
    ):
        self.lambda_ = lambda_
        self.tol = tol
        self.max_iter_no_changes = max_iter_no_changes
        self.max_iter = max_iter

    def __loss(self, X, y):
        reg = self.lambda_ * np.dot(self.w, self.w) / 2
        for x, y in zip(X, y):
            log_loss += np.log(1 + np.exp(-y * (np.dot(self.w, x) + self.bias)))
        return (log_loss / len(X)) + reg

    def fit(self, train_X, train_Y):
        n_samples = len(train_X)
        self.w = np.zeros(len(train_X[0]))
        self.bias = 0.0
        rng = np.random.default_rng()
        sample_indices = np.array(range(n_samples))
        best_loss = np.inf
        n_iter_no_changes = 0
        t = 0
        for epoch in self.max_iter:
            rng.shuffle(sample_indices)
            for idx in sample_indices:
                x_t = train_X[idx]
                y_t = train_Y[idx]
                eta = 1 / (t * self.lambda_)
                common = eta * y_t * sigmoid(-y_t * (np.dot(self.w, x_t) + self.bias))
                self.w += common * x_t - self.w / t
                self.bias += common
            epoch_loss = self.__loss(train_X, train_Y)
            if epoch_loss <= best_loss - self.tol:
                best_loss = epoch_loss
                n_iter_no_changes = 0
            else:
                n_iter_no_changes += 1
            if n_iter_no_changes > self.max_iter_no_changes:
                break

    def predict_proba(self, X_test):
        return np.array([sigmoid(np.dot(self.w, x) + self.bias) for x in X_test])

    def predict(self, X_test, threshold=0.5):
        return to_minus_one_one( (self.predict_proba(X_test) >= threshold).astype(int) )

