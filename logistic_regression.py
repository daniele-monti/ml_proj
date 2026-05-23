import numpy as np

class Kernel:
    def inner_prod(self, x, y):
        pass

class Linear(Kernel):
    def inner_prod(self, x, y):
        return np.dot(x, y)

class Gaussian(Kernel):
    def __init__(self, gamma):
        self.gamma = gamma
    def inner_prod(self, x, y):
        return np.exp(-self.gamma*np.dot(x-y, x-y))

class Polynomial(Kernel):
    def __init__(self, degree):
        self.degree = degree
    def inner_prod(self, x, y):
        return (np.dot(x, y) + 1) ** self.degree


class KernelFactory:
    def get_kernel(name, **params):
        if name == "linear":
            return Linear()
        if name == "rbf":
            return Gaussian(params['gamma'])
        if name == "poly":
            return Polynomial(params["degree"])


class Model:
    def fit(self, X, Y):
        pass
    def predict(self, X_test):
        pass


class SVM(Model):
    def __init__(self, iterations=None, eps=0.0001, lambda_par=0.001, kernel="linear", **kernel_params):
        self.w = None
        self.bias = None
        self._lambda = lambda_par
        self.T = iterations
        self._kernel = KernelFactory.get_kernel(kernel, **kernel_params)

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

    def _accuracy(self, X, Y):
        pred = self.predict(X)
        loss = 0
        for i in range(0, len(X)):
            if pred[i] != Y[i]:
                loss += 1
        return loss

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
        self.bias = 0.0
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







