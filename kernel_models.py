import numpy as np
import itertools
from numba import float64, uint16, jit, prange
from numba.extending import as_numba_type
from numba.experimental import jitclass
from model import Model
from scipy.special import expit as sigmoid


@jitclass
class Linear():
    def __init__(self):
        pass

    def inner_prod(self, x, y):
        return np.dot(x, y)

    def gram(self, X, Y):
        return X @ Y.T


@jitclass([('gamma', float64)])
class Gaussian():
    def __init__(self, gamma):
        self.gamma = gamma
    
    def inner_prod(self, x, y):
        return np.exp(-self.gamma*np.dot(x-y, x-y))

    def gram(self, X, Y):
        return gram(self, X, Y)

@jitclass([('degree', uint16)])
class Polynomial():
    def __init__(self, degree):
        self.degree = degree
    
    def inner_prod(self, x, y):
        return (np.dot(x, y) + 1) ** self.degree
    
    def gram(self, X, Y):
        return gram(self, X, Y)


@jit(
    parallel=True)
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
            return Gaussian(params['gamma'])
        if name == "poly":
            return Polynomial(params["degree"])



class SVM(Model):
    def __init__(
            self,
            max_iter=1000,
            lambda_par=0.0001,
            tol=1e-3,
            n_iter_no_changes=5,
            kernel="linear",
            **kernel_params
    ):
        self._lambda = lambda_par
        self.max_iter = max_iter
        self.max_iter_no_changes = n_iter_no_changes
        self.tol = tol
        self._kernel = KernelFactory.get_kernel(kernel, **kernel_params)

    def _loss(self, t, gram, Y):
        hinge_loss = 0.0
        m = len(gram)
        for i in range(m):
            prod = np.dot(self.coeff, gram[i]) / (self._lambda * t)
            hinge_loss += max(0, 1 - Y[i]*(prod + self.bias))
        return 0.5*self._lambda*self.norm + (hinge_loss / m)

    def fit(self, train_X, train_Y):
        n_samples = len(train_X)
        print("calculating gram matrix...")
        gram = self._kernel.gram(train_X, train_X)
        print(gram)
        self.coeff = np.zeros(n_samples)
        self.norm = 0.0
        self.bias = 0.0
        rng = np.random.default_rng()
        sample_indices = np.array(range(n_samples))
        best_loss = np.inf
        n_iter_no_changes = 0
        t_prev = 1
        t = 1
        for epoch in range(self.max_iter):
            rng.shuffle(sample_indices)
            for idx in sample_indices:
                y_t = train_Y[idx]
                # first compute the inner product
                prod = np.dot(self.coeff, gram[idx]) / (self._lambda * t_prev)
                # prepare coefficients for the update of the squared norm of w
                old_norm_coeff = (1 - 1/t)
                new_addendum_coeff = 0.0
                # then if the hinge loss is non zero:
                #   - update bias
                #   - add the next addendum to the linear combination that represents w
                #   - update new_addendum_coeff in order to correctly compute the new squared norm of w
                if (prod + self.bias) * y_t < 1:
                    self.bias += y_t / (self._lambda * t)
                    self.coeff[idx] += y_t
                    new_addendum_coeff = y_t / (self._lambda * t)
                # update the squared norm of w
                self.norm =                                    \
                    old_norm_coeff**2 * self.norm +            \
                    2*old_norm_coeff*new_addendum_coeff*prod + \
                    new_addendum_coeff**2 * gram[idx][idx]
                # finally shrink w cause of regularization
                t_prev = t
                t += 1
            epoch_loss = self._loss(t_prev, gram, train_Y)
            print(f"epoch {epoch} has loss {epoch_loss}\n")
            self.T = t_prev
            if epoch_loss <= best_loss - self.tol:
                best_loss = epoch_loss
                n_iter_no_changes = 0
            else:
                n_iter_no_changes += 1
            if n_iter_no_changes > self.max_iter_no_changes:
                break
        self.support = train_X.copy()
        pos = 0
        neg = 0
        with open('alpha.txt', mode='w', encoding='utf-8') as f:
            f.write("index, coeff, original_y\n")
            for coeff_idx, coeff in enumerate(self.coeff):
                f.write(f"{coeff_idx}, {coeff}, {train_Y[coeff_idx]}\n")
                if coeff > 0:
                    pos += 1
                else:
                    neg += 1
        print(f"number of positive supports: {pos}")
        print(f"number of negative supports: {neg}")
                          
    def predict(self, X_test):
        gram_mat = gram(self._kernel, X_test, self.support)
        preds = (gram_mat @ self.coeff) / (self.T * self._lambda) + self.bias
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
            n_iter_no_changes=5,
            kernel="linear",
            **kernel_params
    ):
        self._lambda = lambda_par
        self.max_iter = max_iter
        self.max_iter_no_changes = n_iter_no_changes
        self.tol = tol
        self._kernel = KernelFactory.get_kernel(kernel, **kernel_params)

    def _loss(self, t, gram, Y):
        m = len(gram)
        log_loss = 0.0
        for i in range(m):
            prod = np.dot(self.coeff, gram[i]) / (self._lambda * t)
            log_loss += np.log(1 + np.exp(-Y[i]*(prod + self.bias)))
        return 0.5*self._lambda*self.norm + (log_loss / m)

    def fit(self, train_X, train_Y):
        n_samples = len(train_X)
        print("calculating gram matrix...")
        gram = self._kernel.gram(train_X, train_X)
        print(gram)
        self.coeff = np.zeros(n_samples)
        self.norm = 0.0
        self.bias = 0.0
        rng = np.random.default_rng()
        sample_indices = np.array(range(n_samples))
        best_loss = np.inf
        n_iter_no_changes = 0
        t_prev = 1
        t = 1
        for epoch in range(self.max_iter):
            rng.shuffle(sample_indices)
            for idx in sample_indices:
                y_t = train_Y[idx]
                # first compute the inner product
                prod = np.dot(self.coeff, gram[idx]) / (self._lambda * t_prev)
                z = y_t*sigmoid(-y_t*(prod + self.bias))
                # then update the norm of w
                old_norm_coeff = (1 - 1/t)
                new_addendum_coeff = z / (self._lambda * t)
                self.norm =                                    \
                    old_norm_coeff**2 * self.norm +            \
                    2*old_norm_coeff*new_addendum_coeff*prod + \
                    new_addendum_coeff**2 * gram[idx][idx]
                # then add the next addendum to the linear combination that represents w and update the bias
                self.coeff[idx] += z
                self.bias += z / (self._lambda * t)
                # finally shrink w cause of regularization
                t_prev = t
                t += 1
            epoch_loss = self._loss(t_prev, gram, train_Y)
            print(f"epoch {epoch} has loss {epoch_loss}\n")
            self.T = t_prev
            if epoch_loss <= best_loss - self.tol:
                best_loss = epoch_loss
                n_iter_no_changes = 0
            else:
                n_iter_no_changes += 1
            if n_iter_no_changes > self.max_iter_no_changes:
                break
        self.support = train_X.copy()
        pos = 0
        neg = 0
        with open('alpha.txt', mode='w', encoding='utf-8') as f:
            f.write("index, coeff, original_y\n")
            for coeff_idx, coeff in enumerate(self.coeff):
                f.write(f"{coeff_idx}, {coeff}, {train_Y[coeff_idx]}\n")
                if coeff > 0:
                    pos += 1
                else:
                    neg += 1
        print(f"number of positive supports: {pos}")
        print(f"number of negative supports: {neg}")

    def predict_proba(self, X_test):
        gram_mat = gram(self._kernel, X_test, self.support)
        preds = (gram_mat @ self.coeff) / (self.T * self._lambda) + self.bias
        preds = sigmoid(preds)
        return preds

    def predict(self, X_test, threshold=0.5):
        return to_original_classes( (self.predict_proba(X_test) >= threshold).astype(int) )
