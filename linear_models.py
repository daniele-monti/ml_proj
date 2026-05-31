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

    def fit(self, train_X, train_Y, test_X=None, test_Y=None, file=None, granularity=500):
        n_samples, n_features = train_X.shape
        self.w = np.zeros(n_features)
        self.bias = 0.0
        smoothed_loss = 1.0
        alpha = 0.995
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
            # update loss estimate
            single_train_point_loss = 0.5*self.lambda_*np.dot(self.w, self.w) + max(0.0, 1 - margin)
            smoothed_loss = alpha*smoothed_loss + (1 - alpha)*single_train_point_loss
            if file is not None:
                if t % granularity == 1:
                    train_score = self.score(train_X, train_Y)
                    test_score = self.score(test_X, test_Y)
                    file.write(f"{t},{smoothed_loss},{train_score.accuracy},{test_score.accuracy}\n")
            if t % n_samples == 1:
                if smoothed_loss <= best_loss - self.tol:
                    best_loss = smoothed_loss
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
        smoothed_loss = np.log(2)
        alpha = 0.995
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
            # update loss estimate
            single_train_point_loss = 0.5*self.lambda_*np.dot(self.w, self.w) + np.log(1.0 + np.exp(-margin))
            smoothed_loss = alpha*smoothed_loss + (1 - alpha)*single_train_point_loss
            if file is not None:
                if t % granularity == 1:
                    train_score = self.score(train_X, train_Y)
                    test_score = self.score(test_X, test_Y)
                    file.write(f"{t},{smoothed_loss},{train_score.accuracy},{test_score.accuracy}\n")
            if t % n_samples == 1:
                if smoothed_loss <= best_loss - self.tol:
                    best_loss = smoothed_loss
                    n_iter_no_changes = 0
                else:
                    n_iter_no_changes += 1
                if n_iter_no_changes > self.max_iter_no_changes:
                    break

    def predict_proba(self, X_test):
        return np.array([sigmoid(np.dot(self.w, x) + self.bias) for x in X_test])

    def predict(self, X_test, threshold=0.5):
        return to_minus_one_one( (self.predict_proba(X_test) >= threshold).astype(int) )

