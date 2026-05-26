import numpy as np
import itertools
from model import Model
from scipy.special import expit as sigmoid


class Kernel:
    def gram(self, X):
        n = len(X)
        gram = np.zeros((n, n))
        print("calculating gram matrix...")
        for i, j in itertools.product(range(n), repeat=2):
            if i <= j:
                inner = self.inner_prod(X[i], X[j])
                gram[i][j] = inner
                gram[j][i] = inner
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



class SVM(Model):
    def __init__(self, lambda_par=0.0001, epsilon=0.1, kernel="linear", **kernel_params):
        self._lambda = lambda_par
        self.epsilon = epsilon
        self._kernel = KernelFactory.get_kernel(kernel, **kernel_params)

    def _loss(self, t, gram, alpha, Y):
        hinge = 0.0
        m = len(gram)
        for i in range(m):
            prod = 0.0
            for alpha_idx, coeff in alpha.items():
               prod += coeff * gram[alpha_idx][i]
            prod = prod / (self._lambda * t)
            hinge += max(0, 1 - Y[i]*(prod + self.bias))
        return 0.5*self._lambda*self.norm + (hinge / m)

    def fit(self, train_X, train_Y):
        n_samples = len(train_X)
        gram = self._kernel.gram(train_X)
        alpha = {}
        self.norm = 0.0
        #self.bias = 0.0
        rng = np.random.default_rng()
        #sample_indices = np.array(range(n_samples))
        #best_loss = np.inf
        min_iters = int(np.ceil(1 / (self._lambda * self.epsilon)))
        iters = min_iters + (n_samples - min_iters % n_samples)
        t_prev = 1
        print(iters)
        for t in range(1, iters+1):
            idx = rng.integers(0, n_samples)
            y_t = train_Y[idx]
            # first compute the inner product
            prod = 0.0
            for alpha_idx, coeff in alpha.items():
                prod += coeff * gram[idx][alpha_idx]
            prod = prod / (self._lambda * t_prev)
            # prepare coefficients for the update of the norm of w
            old_norm_coeff = (1 - 1/t)
            new_addendum_coeff = 0.0
            # then add the next addendum to the linear combination that represents w if needed
            if (prod ) * y_t < 1:
                new_addendum_coeff = y_t / (self._lambda * t)
                #self.bias += y_t / (self._lambda * t)
                if idx in alpha.keys():
                    alpha[idx] = alpha[idx] + y_t
                else:
                    alpha[idx] = y_t
            # update the norm of w
            self.norm =                                    \
                old_norm_coeff**2 * self.norm +            \
                2*old_norm_coeff*new_addendum_coeff*prod + \
                new_addendum_coeff**2 * gram[idx][idx]
            # finally shrink w cause of regularization
            t_prev = t
            if t % n_samples == 0: print(f"iteration {t}")
            #epoch_loss = self._loss(t_prev, gram, alpha, train_Y)
            #print(f"epoch {epoch} has loss {epoch_loss}\n")
        self.T = iters
        num_supports = len(alpha.keys())
        support_shape = ( num_supports, len(train_X[0]) )
        self.support = np.zeros(shape=support_shape)
        self.coeff = np.zeros(shape=num_supports)
        sup_idx = 0
        pos = 0
        neg = 0
        with open('alpha.txt', mode='w', encoding='utf-8') as f:
            f.write("index, coeff, original_y\n")
            for alpha_idx, coeff in alpha.items():
                self.support[sup_idx] = train_X[alpha_idx]
                self.coeff[sup_idx] = coeff
                f.write(f"{alpha_idx}, {coeff}, {train_Y[alpha_idx]}\n")
                if coeff > 0:
                    pos += 1
                else:
                    neg += 1
                sup_idx += 1
        print(f"number of support vectors: {num_supports}")
        print(f"number of positive supports: {pos}")
        print(f"number of negative supports: {neg}")
                          
    def predict(self, X_test):
        n = len(X_test)
        preds = np.zeros(n)
        for j, (coeff, sup_vec) in itertools.product(range(n), zip(self.coeff, self.support)):
            preds[j] += coeff * self._kernel.inner_prod(sup_vec, X_test[j])
        preds = preds / (self.T * self._lambda) #+ self.bias
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
        log_loss = 0.0
        m = len(gram)
        for i in range(m):
            pred = 0.0
            for coeff_idx, coeff in enumerate(self.coeff):
               pred += coeff * gram[coeff_idx][i]
            pred = pred / (self._lambda * t)
            log_loss += np.log(1 + np.exp(-Y[i]*(pred + self.bias)))
        return 0.5*self._lambda*self.norm + (log_loss / m)

    def fit(self, train_X, train_Y):
        n_samples = len(train_X)
        gram = self._kernel.gram(train_X)
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
                prod = 0.0
                for coeff_idx, coeff in enumerate(self.coeff):
                    prod += coeff * gram[coeff_idx][idx]
                prod = prod / (self._lambda * t_prev)
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
        n = len(X_test)
        preds = np.zeros(n)
        for j, (coeff, sup_vec) in itertools.product(range(n), zip(self.coeff, self.support)):
            preds[j] += coeff * self._kernel.inner_prod(sup_vec, X_test[j])
        preds = preds / (self.T * self._lambda) + self.bias
        preds = sigmoid(preds)
        return preds

    def predict(self, X_test, threshold=0.5):
        return to_original_classes( (self.predict_proba(X_test) >= threshold).astype(int) )
