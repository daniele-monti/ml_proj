import numpy as np
import itertools

class Kernel:
    def gram(self, X):
        n = len(X)
        gram = np.zeros((n, n))
        print("calculating gram matrix...")
        for i, j in itertools.product(range(n), repeat=2):
            inner = self.inner_prod(X[i], X[j])
            gram[i][j] = inner
        print(gram)
        return gram
    
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
    def __init__(
            self,
            iterations=None,
            lambda_par=0.0001,
            tol=1e-3,
            n_iter_no_changes=50,
            kernel="linear",
            **kernel_params
    ):
        self._lambda = lambda_par
        self.T = iterations
        self._kernel = KernelFactory.get_kernel(kernel, **kernel_params)

    def _loss(self, gram, Y):
        reg = 0.0
        for (x_i, alpha_i), (x_j, beta_j) in itertools.product(self.H.items(), repeat=2):
            reg += alpha_i*beta_j*gram[x_i][x_j]
        reg = reg / (0.5 * self._lambda * self.s**2)
        hinge = 0.0
        m = len(gram)
        for i in range(m):
            pred = 0.0
            for x_idx, beta in self.H.items():
               pred += beta * gram[i][x_idx]
            pred = self.s*pred
            hinge += max(0, 1 - Y[i]*pred) 
        return reg + (hinge / m)

    def fit(self, train_X, train_Y):
        samples_number = len(train_X)
        gram = self._kernel.gram(train_X)
        self.H = {}
        self.s = 1.0
        norm = 0.0
        rng = np.random.default_rng()
        if self.T is None:
            self.T = 10 * samples_number
        for t in range(1, self.T+1):
            idx = rng.integers(0, samples_number)
            x_t = train_X[idx]
            y_t = train_Y[idx]
            pred = 0.0
            for x_idx, beta in self.H.items():
               pred += beta * gram[x_idx][idx]
            pred = self.s*pred
            self.s = self.s * (1 - 1/t)
            print(pred)
            print(y_t)
            if pred * y_t < 1:
                norm += 2*y_t*pred / (self._lambda*t) + gram[idx][idx] * (y_t / (self._lambda*t)) ** 2
                if idx in self.H.keys():
                    self.H[idx] = self.H[idx] + (x_t*y_t) / (self._lambda * t * self.s)
                else:
                    self.H[idx] = y_t / (self._lambda * t * self.s)
                if norm > 1/self._lambda:
                    self.s =  self.s / np.sqrt(self._lambda*norm)
                    norm = 1 / self._lambda
            if t % 100 == 1:
                print(f"iteration {t} has loss {self._loss(gram, train_Y)}\n")
        num_supports = len(self.H.keys())
        support_shape = ( num_supports, len(train_X[0]) )
        self.support = np.zeros(shape=support_shape)
        self.coeff = np.zeros(shape=num_supports)
        sup_idx = 0
        for idx, beta in self.H.items():
            self.support[sup_idx] = train_X[idx]
            self.coeff[sup_idx] = beta
            sup_idx += 1
        print(num_supports)
                          
    def predict(self, X_test):
        n = len(X_test)
        preds = np.zeros(n)
        for j, (coeff, sup_vec) in itertools.product(range(n), zip(self.coeff, self.support)):
            preds[j] += coeff * self._kernel.inner_prod(sup_vec, X_test[j])    
        preds = np.sign(self.s*preds)
        return preds



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
