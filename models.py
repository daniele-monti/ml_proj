import numpy as np

class Model:
    def fit(self, X, Y):
        pass
    def predict(self, X_test):
        pass


class SVM(Model):
    def __init__(self, iterations=None, lambda_par=0.001, kernel="linear"):
        self.w = None
        self.bias = None
        self._lambda = lambda_par
        self.T = iterations
        self._kernel = kernel


    def fit(self, X, Y):
        samples_number = len(X)
        self.w = np.zeros(len(X[0]))
        self.bias = 0.
        rng = np.random.default_rng()

        if self.T is None:
            self.T = 10 * samples_number
        for t in range(1, self.T+1):
            idx = rng.integers(0, samples_number)
            x_t = X[idx]
            y_t = Y[idx]
            hinge = 1 - y_t * (np.dot(self.w, x_t) + self.bias)
            eta = 1 / (t * self._lambda)
            if hinge > 0:
                self.w += eta * y_t * x_t - self.w / t
                self.bias += eta * y_t
            else:
                self.w -= self.w / t


    def predict(self, X_test):
        return np.array([np.sign(np.dot(self.w, x) + self.bias) for x in X_test])



from scipy.special import expit as sigmoid

def to_minus_one_one(x):
    return 2*x - 1

class LogReg(Model):
    def __init__(self, iterations=None, lambda_par=0.001, kernel="linear"):
        self.w = None
        self.bias = None
        self._lambda = lambda_par
        self.T = iterations
        self.kernel = kernel


    def fit(self, X, Y):
        tot = len(X)
        self.w = np.zeros(len(X[0]))
        self.bias = 0.
        rng = np.random.default_rng()

        if self.T is None:
            self.T = 10 * tot
        for t in range(1, self.T+1):
            idx = rng.integers(0, tot)
            x_t = X[idx]
            y_t = Y[idx]
            #print(f"x: {x_t}")
            #print(f"w: {self.w}")
            eta = 1 / (t * self._lambda)
            common = eta * y_t * sigmoid(-y_t * (np.dot(self.w, x_t) + self.bias))
            self.w += common * x_t - self.w / t
            self.bias -= common


    def predict_proba(self, X_test):
        return np.array([sigmoid(np.dot(self.w, x) + self.bias) for x in X_test])


    def predict(self, X_test, threshold=0.5):
        return to_minus_one_one( (self.predict_proba(X_test) >= threshold).astype(int) )

