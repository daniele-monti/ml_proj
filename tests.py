
from data import wines, features, label
import metrics
from models import SVM
import numpy as np

#print(k_fold_CV(5, SVM(iterations=100000, lambda_par=0.74), wines[features].to_numpy(), wines[label].to_numpy()))
#print(k_fold_CV(5, LogReg(iterations=100000, lambda_par=1), wines[features].to_numpy(), wines[label].to_numpy()))


# 80 - 20 split
train = wines.sample(frac=0.8)
test = wines.drop(train.index)

train_x = train[features].to_numpy()
test_x = test[features].to_numpy()

train_y = train[label].to_numpy()
test_y = test[label].to_numpy()

#metrics.train(SVM(iterations=10000, lambda_par=0.01, kernel="rbf", gamma=10), train_x, train_y, test_x, test_y)


from sklearn.datasets import load_breast_cancer

def to_minus_one_one(x):
    return 2*x - 1

def invert_one_minus_one(x):
    return -x

X, Y = load_breast_cancer(return_X_y=True)
Y = to_minus_one_one(Y)
Y = invert_one_minus_one(Y)

def split(X, Y, ratio):
    total = len(Y)
    breakpoint = int((ratio * total))
    return X[:breakpoint], X[breakpoint:], Y[:breakpoint], Y[breakpoint:]

train_x, test_x, train_y, test_y = split(X, Y, 0.8)

metrics.train(SVM(iterations=1000000, lambda_par=0.001, kernel="linear"), train_x, train_y, test_x, test_y)

#print(k_fold_CV(5, SVM(iterations=1000000, lambda_par=0.0001), X, Y))
#print(k_fold_CV(5, LogReg(iterations=1000000, lambda_par=0.0045), X, Y))
