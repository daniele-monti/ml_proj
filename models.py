import numpy as np

class Model:
    def fit(self, X, Y):
        pass
    def predict(self, X_test):
        pass


class SVM(Model):
    def __init__(self, lambda_par=0.001, iterations=5000, kernel="linear"):
        self.w = None
        self._lambda = lambda_par
        self.T = iterations
        self.kernel = kernel


    def fit(self, X, Y):
        x_b = np.c_[X, np.ones(len(X))] # add bias
        self.w = np.zeros(len(x_b[0]))
        rng = np.random.default_rng()

        for t in range(1, self.T+1):
            idx = rng.integers(0, len(X))
            x_t = x_b[idx]
            y_t = Y[idx]
            hinge = 1 - y_t * np.dot(self.w, x_t)
            eta = 1 / (t * self._lambda)
            if hinge > 0:
                self.w += (eta * y_t) * x_t - self.w / t
            else:
                self.w -= self.w / t


    def predict(self, X_test):
        x_b = np.c_[X_test, np.ones(len(X_test))]
        return np.array([np.sign(np.dot(self.w, x)) for x in x_b])



class LogReg(Model):
    def __init__(self, kernel="linear"):
        self.w = None
        self.kernel = kernel


    def fit(self, X, Y):
        pass


    def predict(self, X_test):
        pass


