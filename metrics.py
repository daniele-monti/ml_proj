import pandas as pd
import numpy as np
from models import Model, SVM, LogReg
from data import GOOD, BAD

def z_score_scale(X: pd.DataFrame):
    return X.apply(lambda feature: (feature - feature.mean()) / feature.std())

def log_scaling(X: pd.DataFrame):
    return X.apply(lambda x: np.log(x))

def z_score_scale_single(features: np.ndarray):
    return (features - features.mean()) / features.std()

def z_score_scale_array(dataset: np.ndarray):
    scaled = np.zeros_like(dataset, dtype=np.float32)
    for i, datapoint in enumerate(dataset):
        scaled[i] = z_score_scale_single(datapoint)
    return scaled

def evaluation_metrics(truth, prediction):
    total = len(truth)
    support_good = len(truth[truth == GOOD])
    support_bad = len(truth[truth == BAD])

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for index in range(total):
        t = truth[index]
        p = prediction[index]
        if p == GOOD and t == GOOD:
            true_positives += 1
        if p == GOOD and t == BAD:
            false_positives += 1
        if p == BAD and t == BAD:
            true_negatives += 1
        if p == BAD and t == GOOD:
            false_negatives += 1

    accuracy = (true_positives + true_negatives) / total
    precision_good = np.nan
    recall_good = np.nan
    f1_score_good = np.nan
    try:
        precision_good = true_positives / (true_positives + false_positives)
    except ZeroDivisionError:
        pass
    try:
        recall_good = true_positives / (true_positives + false_negatives)
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
        precision_bad = true_negatives / (true_negatives + false_negatives)
    except ZeroDivisionError:
        pass
    try:
        recall_bad = true_negatives / (true_negatives + false_positives)
    except ZeroDivisionError:
        pass
    try:
        f1_score_bad = 2 * precision_bad * recall_bad / (precision_bad +  recall_bad)
    except ZeroDivisionError:
        pass

    return pd.DataFrame(
        {
            "precision": [precision_good, precision_bad, np.nan],
            "recall": [recall_good, recall_bad, np.nan],
            "f1_score": [f1_score_good, f1_score_bad, accuracy],
            "support": [support_good, support_bad, total]
        },
        index=[GOOD, BAD, 'accuracy']
    )



def create_folds(X, Y, k):
    n = len(X)
    fold_x = []
    fold_y = []
    for i in range(k):
        fold_x.append(X[n*i//k : n*(i+1)//k])
        fold_y.append(Y[n*i//k : n*(i+1)//k])
    return fold_x, fold_y



def k_fold_CV(k, model: Model, X, Y):
    folds_x, folds_y = create_folds(X, Y, k)

    metrics = pd.DataFrame(
        {
            "precision": [0.0, 0.0, np.nan],
            "recall": [0.0, 0.0, np.nan],
            "f1_score": [0.0, 0.0, 0.0],
            "support": [0, 0, 0]
        },
        index=[GOOD, BAD, 'accuracy']
    )

    for i in range(k):
        print(f"fold number {i}")
        train_x = np.concat(folds_x[0:i] + folds_x[i+1:k])
        train_y = np.concat(folds_y[0:i] + folds_y[i+1:k])
        train_x = z_score_scale_array(train_x)
        model.fit(train_x, train_y)

        test_x = folds_x[i]
        test_y = folds_y[i]

        test_x = z_score_scale_array(test_x)
        pred = model.predict(test_x)

        fold_metrics = evaluation_metrics(test_y, pred)
        print(fold_metrics)
        print()
        metrics = metrics.add(fold_metrics, fill_value=0.0)
        print(f"{metrics}\n\n\n")
    
    return metrics.div(k)



from data import wines, train, dev, test, features, label

#print(k_fold_CV(5, SVM(iterations=100000, lambda_par=0.74), wines[features].to_numpy(), wines[label].to_numpy()))
#print(k_fold_CV(5, LogReg(iterations=100000, lambda_par=1), wines[features].to_numpy(), wines[label].to_numpy()))


train_x = z_score_scale_array(train[features].to_numpy())
test_x = z_score_scale_array(test[features].to_numpy())

train_y = train[label].to_numpy()
test_y = test[label].to_numpy()

svm = SVM()
svm.fit(train_x, train_y)

pred = svm.predict(test_x)
print(evaluation_metrics(test_y, pred))

pred = svm.predict(train_x)
print(evaluation_metrics(train_y, pred))

#logreg = LogReg(iterations=5)
#logreg.fit(x_train, y_train)


from sklearn.datasets import load_breast_cancer

def to_minus_one_one(x):
    return 2*x - 1

X, Y = load_breast_cancer(return_X_y=True)
Y = to_minus_one_one(Y)

def split(X, Y, ratio):
    total = len(Y)
    breakpoint = np.floor(ratio * total)
    return X[:breakpoint], X[breakpoint:], Y[:breakpoint], Y[:breakpoint]

#train_x, test_x, train_y, test_y = split(X, Y, 0.8)

#svm = SVM()
#print(k_fold_CV(5, SVM(iterations=1000000, lambda_par=0.0001), X, Y))
print(k_fold_CV(5, LogReg(iterations=1000000, lambda_par=0.0045), X, Y))
