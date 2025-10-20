import numpy as np

class Kernel:
    def gram(self, X):
        n = len(X)
        gram = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                inner = self.inner_prod(X[i], X[j])
                gram[i][j] = inner
                gram[j][i] = inner
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
    def __init__(self, iterations=None, lambda_par=0.001, kernel="linear", **kernel_params):
        self._lambda = lambda_par
        self.T = iterations
        self._kernel = KernelFactory.get_kernel(kernel, **kernel_params)

    def _gradient(self, xt_index, y_t, gram, alpha_t):
        alpha_grad = self._lambda * (gram @ alpha_t) / len(gram)
        bias_der = 0.0
        if y_t * (np.dot(alpha_t, gram[xt_index]) + self.bias) < 1:
            alpha_grad -= y_t * gram[xt_index]
            bias_der -= y_t
        #print(f"alpha: {alpha_t}; grad: {alpha_grad}")
        return (alpha_grad, bias_der)

    def _loss(self, gram, alpha, Y):
        reg = 0.5 * self._lambda * np.dot(alpha, gram @ alpha)
        hinge = 0.0
        m = len(gram)
        for i in range(m):
            hinge += max(0, 1 - Y[i] * (np.dot(alpha, gram[i]) + self.bias)) 
        return (hinge / m) + reg

    def fit(self, train_X, train_Y):
        samples_number = len(train_X)
        alpha = np.zeros(samples_number)
        self.bias = 0.0
        gram = self._kernel.gram(train_X)
        rng = np.random.default_rng()
        if self.T is None:
            self.T = 10 * samples_number
        print(f"iteration 0 has loss {self._loss(gram, alpha, train_Y)}\n")
        for t in range(1, self.T+1):
            idx = rng.integers(0, samples_number)
            y_t = train_Y[idx]
            eta = 1 / (t * self._lambda)
            alpha_grad, bias_der = self._gradient(idx, y_t, gram, alpha)
            alpha -= eta * alpha_grad
            self.bias -= eta * bias_der
            if t % 100 == 0:
                print(f"iteration {t} has loss {self._loss(gram, alpha, train_Y)}\n")
        num_supports = sum(np.absolute(alpha) > 1e-3)
        support_shape = ( num_supports, len(train_X[0]) )
        self.support = np.zeros(shape=support_shape)
        self.coeff = np.zeros(shape=num_supports)
        sup_idx = 0
        for i in range(samples_number):
            if np.abs(alpha[i]) > 1e-3:
                self.support[sup_idx] = train_X[i]
                self.coeff[sup_idx] = alpha[i] * train_Y[i]
                sup_idx += 1
        print(num_supports)
        print(self.coeff)
                          
    def predict(self, X_test):
        preds = np.zeros(shape=len(X_test))
        for j in range(len(X_test)):
            for i in range(len(self.support)):
                preds[j] += self.coeff[i] * self._kernel.inner_prod(self.support[i], X_test[j])
            preds[j] = np.sign(preds[j] + self.bias)
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
