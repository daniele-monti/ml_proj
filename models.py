import pandas as pd
import numpy as np
from data import GOOD, BAD


def pegasos(hyperparams: dict, training_set: pd.DataFrame, features, label):
    regularizer = 1
    rounds = 10000
    if "regularizer" in hyperparams.keys():
        regularizer = hyperparams["regularizer"]
    if "rounds" in hyperparams.keys():
        rounds = hyperparams["rounds"]

    w_t = np.zeros(shape=len(features))
    w_sum = np.zeros(shape=len(features))
    for t in range(1, rounds+1):
        z = training_set.sample(replace=True).iloc[0]
        z_x = z[features].to_numpy()
        z_y = z[label]
        hinge = 1 - z_y * np.dot(w_t, z_x)
        w_t = (1 - 1/t) * w_t
        if hinge > 0:
            w_t += z_y*z_x / (t * regularizer)
        w_sum += w_t
    return w_sum / rounds


def evaluation_metrics(truth: pd.Series, prediction: pd.Series):
    total = len(truth)
    support_good = len(truth[truth == GOOD])
    support_bad = len(truth[truth == BAD])

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for index in truth.index:
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


def predict(model, test: pd.DataFrame):
    pred = pd.Series(index=test.index)
    for index in test.index:
        x = test.loc[index].to_numpy()
        pred[index] = np.sign(np.dot(model, x))
    return pred


def create_folds(dataset, k):
    n = len(dataset)
    return [ dataset.iloc[n*i//k : n*(i+1)//k] for i in range(0, k) ]


def k_fold_CV(k, algorithm, hyperparams, dataset: pd.DataFrame, features, label):
    # shuffle dataframe first
    dataset = dataset.sample(frac=1)
    folds = create_folds(dataset, k)

    metrics = pd.DataFrame(
        {
            "precision": [0.0, 0.0, np.nan],
            "recall": [0.0, 0.0, np.nan],
            "f1_score": [0.0, 0.0, 0.0],
            "support": [0, 0, 0]
        },
        index=[GOOD, BAD, 'accuracy']
    )

    for i in range(0, k):
        train = pd.concat(folds[0:i] + folds[i+1:k])
        test = folds[i]
        model = algorithm(hyperparams, train, features, label)
        y_pred = predict(model, test[features])
        fold_metrics = evaluation_metrics(test[label], y_pred)
        #print(f"round_number {round_number}, reg_value {reg_value}:")
        print(fold_metrics)
        print()
        metrics = metrics.add(fold_metrics, fill_value=0.0)
        print(f"{metrics}\n\n\n")
    
    return metrics.div(k)



from data import wines_discrete_quality, train, dev, test, features, label

print(k_fold_CV(5, pegasos, {"rounds": 50000, "regularizer": 0.007}, wines_discrete_quality, features, label))

