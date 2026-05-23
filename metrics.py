import pandas as pd
import numpy as np
from models import Model
from data import GOOD, BAD


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
        #train_x = z_score_scale_array(train_x)
        model.fit(train_x, train_y)

        test_x = folds_x[i]
        test_y = folds_y[i]

        #test_x = z_score_scale_array(test_x)
        pred = model.predict(test_x)

        _, fold_metrics = evaluation_metrics(test_y, pred)
        print(fold_metrics)
        print()
        metrics = metrics.add(fold_metrics, fill_value=0.0)
        print(f"{metrics}\n\n\n")
    
    return metrics.div(k)

