import numpy as np

class Model:
    def fit(self, X, Y):
        pass
    def predict(self, X_test):
        pass


class SVM(Model):
    def __init__(self, lambda_par=0.001, iterations=5000, kernel="linear"):
        self.w = None
        self.bias = None
        self._lambda = lambda_par
        self.T = iterations
        self._kernel = kernel


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



def sigmoid(z):
    #print(f"z: {z}")
    #print(np.exp(-z))
    return 1.0 / (1.0 + np.exp(-z))

def to_minus_one_one(x):
    return 2*x - 1

class LogReg(Model):
    def __init__(self, lambda_par=1000, epochs=5000, kernel="linear"):
        self.w = None
        self._lambda = lambda_par
        self.epochs = epochs
        self.kernel = kernel


    def fit(self, X, Y):
        x_b = np.c_[np.ones(len(X)), X] # add bias
        self.w = np.random.rand(len(x_b[0]))
        rng = np.random.default_rng()

        for t in range(1, self.epochs+1):
            idx = rng.integers(0, len(X))
            x_t = x_b[idx]
            y_t = Y[idx]
            #print(f"x: {x_t}")
            #print(f"w: {self.w}")
            self.w += (y_t * sigmoid(-y_t * np.dot(self.w, x_t)) * x_t) / (t * self._lambda) - self.w / t


    def predict_proba(self, X_test):
        x_b = np.c_[X_test, np.ones(len(X_test))]
        return np.array([sigmoid(np.dot(self.w, x)) for x in x_b])


    def predict(self, X_test, threshold=0.5):
        x_b = np.c_[X_test, np.ones(len(X_test))]
        return to_minus_one_one( (self.predict_proba(X_test) >= threshold).astype(int) )

