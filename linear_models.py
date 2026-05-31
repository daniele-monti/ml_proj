import numpy as np
from model import Model
from scipy.special import expit as sigmoid
from numba import jit, prange

@jit(parallel=True, nopython=True)
def compute_hinge_loss(w, bias, train_X, train_Y, lambda_):
    n_samples = len(train_Y)
    hinge_loss_sum = 0.0
    for i in prange(n_samples):
        prod = np.dot(w, train_X[i]) + bias
        hinge_loss_sum += max(0.0, 1.0 - train_Y[i] * prod)
    return (hinge_loss_sum / n_samples) + (0.5 * lambda_ * np.dot(w, w))


@jit(parallel=True, nopython=True)
def compute_log_loss(w, bias, train_X, train_Y, lambda_):
    n_samples = len(train_Y)
    log_loss_sum = 0.0
    for i in prange(n_samples):
        prod = np.dot(w, train_X[i]) + bias
        log_loss_sum += np.log(1.0 + np.exp(-train_Y[i] * prod))
    return (log_loss_sum / n_samples) + (0.5 * lambda_ * np.dot(w, w))


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

    def fit(self, train_X, train_Y, test_X=None, test_Y=None, file=None, granularity=500):
        n_samples, n_features = train_X.shape
        self.w = np.zeros(n_features)
        self.bias = 0.0
        best_loss = np.inf
        n_iter_no_changes = 0
        rng = np.random.default_rng()
        if file is not None:
            file.write("Iteration,Loss,Train_acc,Test_acc\n")
        for t in range(1, self.max_iter*n_samples + 2):
            idx = rng.integers(0, n_samples)
            x_t = train_X[idx]
            y_t = train_Y[idx]
            eta = 1.0 / (self.lambda_ * t)
            # compute the inner product and the margin
            prod = np.dot(self.w, x_t) + self.bias
            margin = y_t * prod
            if margin < 1:
                self.w = (1 - 1/t)*self.w + eta*y_t*x_t 
                self.bias += eta * y_t
            else:
                self.w = (1 - 1/t)*self.w
            if file is not None:
                if t % granularity == 1:
                    loss = compute_hinge_loss(self.w, self.bias, train_X, train_Y, self.lambda_)
                    train_score = self.score(train_X, train_Y)
                    test_score = self.score(test_X, test_Y)
                    file.write(f"{t},{loss},{train_score.accuracy},{test_score.accuracy}\n")
            if t % n_samples == 1:
                epoch_loss = compute_hinge_loss(self.w, self.bias, train_X, train_Y, self.lambda_)
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

    def fit(self, train_X, train_Y, test_X=None, test_Y=None, file=None, granularity=500):
        n_samples, n_features = train_X.shape
        self.w = np.zeros(n_features)
        self.bias = 0.0
        best_loss = np.inf
        n_iter_no_changes = 0
        rng = np.random.default_rng()
        if file is not None:
            file.write("Iteration,Loss,Train_acc,Test_acc\n")
        for t in range(1, self.max_iter*n_samples + 2):
            idx = rng.integers(0, n_samples)
            x_t = train_X[idx]
            y_t = train_Y[idx]
            eta = 1 / (t * self.lambda_)
            # compute the inner product and the margin
            prod = np.dot(self.w, x_t) + self.bias
            margin = np.clip(y_t * prod, -20, 50)
            error_multiplier = y_t*sigmoid(-margin)
            self.w = (1 - 1/t)*self.w + eta*error_multiplier*x_t
            self.bias += eta * error_multiplier
            if file is not None:
                if t % granularity == 1:
                    loss = compute_log_loss(self.w, self.bias, train_X, train_Y, self.lambda_)
                    train_score = self.score(train_X, train_Y)
                    test_score = self.score(test_X, test_Y)
                    file.write(f"{t},{loss},{train_score.accuracy},{test_score.accuracy}\n")
            if t % n_samples == 1:
                epoch_loss = compute_log_loss(self.w, self.bias, train_X, train_Y, self.lambda_)
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
