import pandas as pd
import numpy as np
from models import Model
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


def confusion_mat(truth, prediction):
    mat = np.zeros(shape=(2, 2), dtype=np.int32)
    total = len(truth)

    for index in range(total):
        t = truth[index]
        p = prediction[index]
        if p == GOOD and t == GOOD:
            mat[0][0] += 1
        if p == GOOD and t == BAD:
            mat[1][0] += 1
        if p == BAD and t == GOOD:
            mat[0][1] += 1
        if p == BAD and t == BAD:
            mat[1][1] += 1
    return mat


def evaluation_metrics(truth, prediction):
    total = len(truth)
    support_good = len(truth[truth == GOOD])
    support_bad = len(truth[truth == BAD])

    conf = confusion_mat(truth, prediction)
    true_positives = conf[0][0]
    false_negatives = conf[0][1]
    false_positives = conf[1][0]
    true_negatives = conf[1][1]

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

    return (conf, 
            pd.DataFrame(
            {
                "precision": [precision_good, precision_bad, np.nan],
                "recall": [recall_good, recall_bad, np.nan],
                "f1_score": [f1_score_good, f1_score_bad, accuracy],
                "support": [support_good, support_bad, total]
            },
            index=[GOOD, BAD, 'accuracy']
            )
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

        _, fold_metrics = evaluation_metrics(test_y, pred)
        print(fold_metrics)
        print()
        metrics = metrics.add(fold_metrics, fill_value=0.0)
        print(f"{metrics}\n\n\n")
    
    return metrics.div(k)


def train(model: Model, train_x, train_y, test_x, test_y):
    train_x = z_score_scale_array(train_x)
    test_x = z_score_scale_array(test_x)

    model.fit(train_x, train_y)

    pred = model.predict(test_x)
    conf, metrics = evaluation_metrics(test_y, pred)
    print("test set performance")
    print(conf)
    print(metrics)
    print()

    pred = model.predict(train_x)
    conf, metrics = evaluation_metrics(train_y, pred)
    print("train set performance")
    print(conf)
    print(metrics)
    print()
