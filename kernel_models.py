import numpy as np
from numba import float64, int64, jit, prange
from numba.experimental import jitclass
from model import Model
from scipy.special import expit as sigmoid


@jitclass
class Linear():
    def __init__(self):
        return

    def inner_prod(self, x, y):
        return np.dot(x, y)

    def gram(self, X, Y):
        return 1.0 + (X @ Y.T)


@jitclass([('gamma', float64)])
class Gaussian():
    def __init__(self, gamma=1.0):
        self.gamma = gamma
    
    def inner_prod(self, x, y):
        return 1.0 + np.exp(-self.gamma*np.dot(x-y, x-y))

    def gram(self, X, Y):
        return gram(self, X, Y)

@jitclass([('degree', int64)])
class Polynomial():
    def __init__(self, degree=2):
        self.degree = degree
    
    def inner_prod(self, x, y):
        return 1.0 + (np.dot(x, y) + 1) ** self.degree
    
    def gram(self, X, Y):
        return gram(self, X, Y)


@jit(parallel=True)
def gram(kernel, X, Y):
    n = len(X)
    m = len(Y)
    gram = np.zeros((n, m))
    for i in prange(n):
        for j in prange(m):
            gram[i][j] = kernel.inner_prod(X[i], Y[j])
    return gram


class KernelFactory:
    def get_kernel(name, **params):
        if name == "linear":
            return Linear()
        if name == "rbf":
            if 'gamma' in params.keys():
                return Gaussian(gamma=params['gamma'])
            else:
                return Gaussian()
        if name == "poly":
            if 'degree' in params.keys():
                return Polynomial(degree=params['degree'])
            else:
                return Polynomial()



class SVM(Model):
    def __init__(
            self,
            max_iter=1000,
            lambda_=0.0001,
            tol=1e-4,
            max_iter_no_changes=5,
            kernel="linear",
            **kernel_params
    ):
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.max_iter_no_changes = max_iter_no_changes
        self.tol = tol
        self._kernel = KernelFactory.get_kernel(kernel, **kernel_params)

    def set_params(self, **params):
        super().set_params(**params)
        if 'kernel' in params.keys():
            self._kernel = KernelFactory.get_kernel(params['kernel'], **params)

    def __loss(self, gram, Y):
        hinge_loss = 0.0
        m = len(gram)
        for i in range(m):
            prod = self.s * np.dot(self.coeff, gram[i])
            hinge_loss += max(0, 1 - Y[i]*prod)
        return 0.5*self.lambda_*self.norm_sq + (hinge_loss / m)

    def fit(self, train_X, train_Y, test_X=None, test_Y=None, file=None, granularity=1000):
        self.support = train_X.copy()
        n_samples = len(train_X)
        gram = self._kernel.gram(train_X, train_X)
        self.coeff = np.zeros(n_samples)
        self.norm_sq = 0.0
        self.s = 1.0
        rng = np.random.default_rng()
        best_loss = np.inf
        n_iter_no_changes = 0
        if file is not None:
            file.write("Iteration,Loss,Train_acc,Test_acc\n")
        for t in range(1, self.max_iter*n_samples + 2):
            idx = rng.integers(0, n_samples)
            y_t = train_Y[idx]
            eta = 1.0 / (self.lambda_ * t)
            # compute the inner product
            prod = self.s * np.dot(self.coeff, gram[idx])     
            # Compute the scaling factor
            s_next = (1.0 - 1/t) * self.s
            # Underflow handling
            if abs(s_next) < 1e-9:
                self.coeff = self.s * self.coeff
                # self.w_norm_sq does not change when scaling variables reset, 
                # because true weights stay the same.
                self.s = 1.0
                s_next = 1.0
            # prepare coefficients for the update of the squared norm of w
            old_norm_coeff = (1 - 1/t)
            new_addendum_coeff = 0.0    
            # then if the hinge loss is non zero:
            #   - add the next addendum to the linear combination that represents w
            #   - update new_addendum_coeff in order to correctly compute the new squared norm of w
            if y_t * prod < 1:
                self.coeff[idx] += (y_t*eta) / s_next
                new_addendum_coeff = eta * y_t         
            # update the squared norm of w
            self.norm_sq =                                    \
                old_norm_coeff**2 * self.norm_sq +            \
                2*old_norm_coeff*new_addendum_coeff*prod +    \
                new_addendum_coeff**2 * gram[idx][idx]
            # update the scale factor
            self.s = s_next
            if file is not None:
                if t % granularity == 1:
                    loss = self.__loss(gram, train_Y)
                    train_score = self.score(train_X, train_Y)
                    test_score = self.score(test_X, test_Y)
                    file.write(f"{t},{loss},{train_score.accuracy},{test_score.accuracy}\n")
            if t % n_samples == 1:
                epoch_loss = self.__loss(gram, train_Y)
                if epoch_loss <= best_loss - self.tol:
                    best_loss = epoch_loss
                    n_iter_no_changes = 0
                else:
                    n_iter_no_changes += 1
                if n_iter_no_changes > self.max_iter_no_changes:
                    break
        # Finalize alpha vector before exiting
        self.coeff = self.s * self.coeff
        self.s = 1.0  
        return self

    def predict(self, X_test):
        gram_mat = gram(self._kernel, X_test, self.support)
        preds = (gram_mat @ self.coeff)
        preds = np.sign(preds)
        return preds



def to_original_classes(x):
    return 2*x - 1

class LogReg(Model):
    def __init__(
            self,
            max_iter=1000,
            lambda_par=0.0001,
            tol=1e-3,
            max_iter_no_changes=5,
            kernel="linear",
            **kernel_params
    ):
        self.lambda_ = lambda_par
        self.max_iter = max_iter
        self.max_iter_no_changes = max_iter_no_changes
        self.tol = tol
        self._kernel = KernelFactory.get_kernel(kernel, **kernel_params)

    def set_params(self, **params):
        super().set_params(**params)
        if 'kernel' in params.keys():
            self._kernel = KernelFactory.get_kernel(params['kernel'], **params)

    def __loss(self, gram, Y):
        m = len(gram)
        log_loss = 0.0
        for i in range(m):
            prod = self.s * np.dot(self.coeff, gram[i])
            log_loss += np.log(1 + np.exp(-Y[i]*prod))
        return 0.5*self.lambda_*self.norm_sq + (log_loss / m)

    def fit(self, train_X, train_Y, test_X=None, test_Y=None, file=None, granularity=1000):
        self.support = train_X.copy()
        n_samples = len(train_X)
        gram = self._kernel.gram(train_X, train_X)
        self.coeff = np.zeros(n_samples)
        self.norm_sq = 0.0
        self.s = 1.0
        rng = np.random.default_rng()
        best_loss = np.inf
        n_iter_no_changes = 0
        if file is not None:
            file.write("Iteration,Loss,Train_acc,Test_acc\n")
        for t in range(1, self.max_iter*n_samples + 2):
            idx = rng.integers(0, n_samples)
            y_t = train_Y[idx]
            eta = 1.0 / (self.lambda_ * t)
            # compute the inner product
            prod = self.s * np.dot(self.coeff, gram[idx])
            z = y_t*sigmoid(-y_t*prod) 
            # Compute the scaling factor
            s_next = (1.0 - 1/t) * self.s
            # Underflow handling
            if abs(s_next) < 1e-9:
                self.coeff = self.s * self.coeff
                # self.w_norm_sq does not change when scaling variables reset, 
                # because true weights stay the same.
                self.s = 1.0
                s_next = 1.0
            # add the next addendum to the linear combination that represents w
            self.coeff[idx] += (z*eta) / s_next     
            # update the squared norm of w
            old_norm_coeff = (1 - 1/t)
            new_addendum_coeff = z / (self.lambda_ * t)
            self.norm_sq =                                    \
                old_norm_coeff**2 * self.norm_sq +            \
                2*old_norm_coeff*new_addendum_coeff*prod + \
                new_addendum_coeff**2 * gram[idx][idx]
            # update the scale factor
            self.s = s_next
            if file is not None:
                if t % granularity == 1:
                    loss = self.__loss(gram, train_Y)
                    train_score = self.score(train_X, train_Y)
                    test_score = self.score(test_X, test_Y)
                    file.write(f"{t},{loss},{train_score.accuracy},{test_score.accuracy}\n")
            if t % n_samples == 1:
                epoch_loss = self.__loss(gram, train_Y)
                if epoch_loss <= best_loss - self.tol:
                    best_loss = epoch_loss
                    n_iter_no_changes = 0
                else:
                    n_iter_no_changes += 1
                if n_iter_no_changes > self.max_iter_no_changes:
                    break
        # Finalize alpha vector before exiting
        self.coeff = self.s * self.coeff
        self.s = 1.0  
        return self

    def predict_proba(self, X_test):
        gram_mat = gram(self._kernel, X_test, self.support)
        preds = (gram_mat @ self.coeff)
        preds = sigmoid(preds)
        return preds

    def predict(self, X_test, threshold=0.5):
        return to_original_classes( (self.predict_proba(X_test) >= threshold).astype(int) )
