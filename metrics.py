import pandas as pd
import numpy as np
from models import Model, SVM
from data import GOOD, BAD


def evaluation_metrics(truth, prediction):
    total = len(truth)
    support_good = len(truth[truth == GOOD])
    support_bad = len(truth[truth == BAD])

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for index in range(truth.size):
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
        train_x = np.concat(folds_x[0:i] + folds_x[i+1:k])
        train_y = np.concat(folds_y[0:i] + folds_y[i+1:k])
        print(f"train x -> {len(train_x)}, train y -> {len(train_y)}")
        model.fit(train_x, train_y)

        test_x = folds_x[i]
        test_y = folds_y[i]
        print(f"test x -> {len(test_x)}, test y -> {len(test_y)}")
        pred = model.predict(test_x)

        fold_metrics = evaluation_metrics(test_y, pred)
        print(fold_metrics)
        print()
        metrics = metrics.add(fold_metrics, fill_value=0.0)
        print(f"{metrics}\n\n\n")
    
    return metrics.div(k)



from data import wines, train, dev, test, features, label

print(k_fold_CV(5, SVM(iterations=1000000, lambda_par=0.74), wines[features].to_numpy(), wines[label].to_numpy()))

