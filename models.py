import numpy as np
import pandas as pd
import itertools

GOOD = 1
BAD = -1

class Kernel:
    def gram(self, X):
        n = len(X)
        gram = np.zeros((n, n))
        print("calculating gram matrix...")
        for i, j in itertools.product(range(n), repeat=2):
            gram[i][j] = self.inner_prod(X[i], X[j])
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


class ConfusionMatrix():
    def __init__(self, truth, prediction):
        self.true_pos = 0
        self.false_pos = 0
        self.false_neg = 0
        self.true_neg = 0
        for index in range(len(truth)):
            t = truth[index]
            p = prediction[index]
            if p == GOOD and t == GOOD:
                self.true_pos += 1
            if p == GOOD and t == BAD:
                self.false_pos += 1
            if p == BAD and t == GOOD:
                self.false_neg += 1
            if p == BAD and t == BAD:
                self.true_neg += 1

    def __str__(self):
        char_count = 31
        rows = [
            " | ".join(["         ", "pred pos", "pred neg"]),
            #"+"*char_count,
            " | ".join(["truth pos", f"{self.true_pos:>8}", f"{self.false_neg:>8}"]),
            #"+"*char_count,
            " | ".join(["truth neg", f"{self.false_pos:>8}", f"{self.true_neg:>8}"])
        ]
        return "\n".join(rows)


class Model:
    def fit(self, X, Y):
        pass
    def predict(self, X_test):
        pass
    def score(self, X_test, Y_test):
        total = len(X_test)
        prediction = self.predict(X_test)
        truth = Y_test
        support_good = len(truth[truth == GOOD])
        support_bad = len(truth[truth == BAD])

        conf_mat = ConfusionMatrix(truth, prediction)

        accuracy = (conf_mat.true_pos + conf_mat.true_neg) / total
        precision_good = np.nan
        recall_good = np.nan
        f1_score_good = np.nan
        try:
            precision_good = conf_mat.true_pos / (conf_mat.true_pos + conf_mat.false_pos)
        except ZeroDivisionError:
            pass
        try:
            recall_good = conf_mat.true_pos / (conf_mat.true_pos + conf_mat.false_neg)
        except ZeroDivisionError:
            pass
        try:
            f1_score_good = 2 * precision_good * recall_good / (precision_good +  recall_good)
        except ZeroDivisionError:
            pass

        precision_bad = np.nan
        recall_bad = np.nan
        f1_score_bad = np.nan

        try:
            precision_bad = conf_mat.true_neg / (conf_mat.true_neg + conf_mat.false_neg)
        except ZeroDivisionError:
            pass
        try:
            recall_bad = conf_mat.true_neg / (conf_mat.true_neg + conf_mat.false_pos)
        except ZeroDivisionError:
            pass
        try:
            f1_score_bad = 2 * precision_bad * recall_bad / (precision_bad +  recall_bad)
        except ZeroDivisionError:
            pass
        
        metrics = pd.DataFrame(
            {
                "precision": [precision_good, precision_bad, np.nan],
                "recall": [recall_good, recall_bad, np.nan],
                "f1_score": [f1_score_good, f1_score_bad, accuracy],
                "support": [support_good, support_bad, total]
            },
            index=[GOOD, BAD, 'accuracy']
        )
        return (conf_mat, metrics)

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

    def _loss(self, t, gram, alpha, Y):
        reg = 0.0
        for (idx_i, count_i), (idx_j, count_j) in itertools.product(alpha.items(), repeat=2):
            reg += count_i*Y[idx_i]*count_j*Y[idx_j]*gram[idx_i][idx_j]
        reg = reg / (2 * self._lambda * t**2)
        #print(f"regularization term: {reg}")
        hinge = 0.0
        m = len(gram)
        for i in range(m):
            pred = 0.0
            for alpha_idx, count in alpha.items():
               pred += count * Y[alpha_idx] * gram[i][alpha_idx]
            pred = pred / (self._lambda * t)
            hinge += max(0, 1 - Y[i]*(pred + self.bias))
        #print(f"sum of hinges: {hinge}")
        return reg + (hinge / m)

    def fit(self, train_X, train_Y):
        n_samples = len(train_X)
        gram = self._kernel.gram(train_X)
        alpha = {}
        #self.norm = 0.0
        self.bias = 0.0
        rng = np.random.default_rng()
        sample_indices = np.array(range(n_samples))
        best_loss = np.inf
        n_iter_no_changes = 0
        t_prev = 1
        first = 0
        first_idx = 0
        t = 1
        for epoch in range(self.max_iter):
            rng.shuffle(sample_indices)
            for idx in sample_indices:
                y_t = train_Y[idx]
                # first do the prediction
                pred = 0.0
                for alpha_idx, count in alpha.items():
                    pred += y_t * count * gram[idx][alpha_idx]
                pred = pred / (self._lambda * t_prev)
                # then add the next addendum to the linear combination that represents w if needed
                if (pred + self.bias) * y_t < 1:
                    if t == 1:
                        first = y_t
                        first_idx = idx
                    self.bias = self.bias + y_t / (self._lambda * t)
                    if idx in alpha.keys():
                        alpha[idx] = alpha[idx] + 1
                    else:
                        alpha[idx] = 1
                    #print(f"point {idx} misclassified for the {alpha[idx]} time" )
                # finally shrink w cause of regularization
                t_prev = t
                t += 1
            epoch_loss = self._loss(t_prev, gram, alpha, train_Y)
            print(f"epoch {epoch} has loss {epoch_loss}\n")
            self.T = t_prev
            if epoch_loss <= best_loss - self.tol:
                best_loss = epoch_loss
                n_iter_no_changes = 0
            else:
                n_iter_no_changes += 1
            if n_iter_no_changes > self.max_iter_no_changes:
                break
        num_supports = len(alpha.keys())
        support_shape = ( num_supports, len(train_X[0]) )
        self.support = np.zeros(shape=support_shape)
        self.coeff = np.zeros(shape=num_supports)
        sup_idx = 0
        pos = 0
        neg = 0
        with open('alpha.txt', mode='w', encoding='utf-8') as f:
            f.write("index, count, original_y\n")
            for alpha_idx, count in alpha.items():
                self.support[sup_idx] = train_X[alpha_idx]
                self.coeff[sup_idx] = count * train_Y[alpha_idx]
                f.write(f"{alpha_idx}, {self.coeff[sup_idx]}, {train_Y[alpha_idx]}\n")
                if self.coeff[sup_idx] > 0:
                    pos += 1
                else:
                    neg += 1
                sup_idx += 1
        print(f"first was a sample with label {first}, counted {alpha[first_idx]*train_Y[first_idx]} times")
        print(f"number of support vectors: {num_supports}")
        print(f"number of positive supports: {pos}")
        print(f"number of negative supports: {neg}")
                          
    def predict(self, X_test):
        n = len(X_test)
        preds = np.zeros(n)
        for j, (coeff, sup_vec) in itertools.product(range(n), zip(self.coeff, self.support)):
            preds[j] += coeff * self._kernel.inner_prod(sup_vec, X_test[j])
        preds = preds / (self.T * self._lambda) + self.bias
        preds = np.sign(preds)
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
