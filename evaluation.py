from model import Model
from kernel_models import SVM, LogReg
import numpy as np
from timeit import default_timer as timer
from preprocessing import preprocess
from metrics import ScoreMetrics
import optuna


def evaluate(model: Model, train_x, train_y, test_x, test_y, display=False, track=None):
    start = timer()
    model.fit(train_x, train_y, test_x, test_y, track)
    end = timer()
    print(f"fitting took {end - start} seconds")

    test_scores = model.score(test_x, test_y)
    if display:
        print("test set performance:")
        test_scores.print_mat()
        test_scores.print_metrics()

    train_scores = model.score(train_x, train_y)
    if display:
        print("train set performance:")
        train_scores.print_mat()
        train_scores.print_metrics()

    return {
        'train': train_scores,
        'test': test_scores
    }


def create_folds(X, y, k):
    n = len(X)
    fold_x = []
    fold_y = []
    for i in range(k):
        fold_x.append(X[n*i//k : n*(i+1)//k])
        fold_y.append(y[n*i//k : n*(i+1)//k])
    return fold_x, fold_y


def k_fold_CV(X, y, k, model: Model):
    folds_x, folds_y = create_folds(X, y, k)
    train_scores = np.empty(shape=k, dtype=ScoreMetrics)
    test_scores = np.empty(shape=k, dtype=ScoreMetrics)
    for i in range(k):
        print(f"fold number {i+1}")
        train_x = np.concat(folds_x[0:i] + folds_x[i+1:k])
        train_y = np.concat(folds_y[0:i] + folds_y[i+1:k])
        test_x = folds_x[i]
        test_y = folds_y[i]
        train_x, train_y, test_x, test_y = preprocess(train_x, train_y, test_x, test_y)
        fold_scores = evaluate(model, train_x, train_y, test_x, test_y)
        train_scores[i] = fold_scores['train']
        test_scores[i] = fold_scores['test']
    return {
        'train': train_scores,
        'test': test_scores
    }


def nested_CV(X, y, model:Model, kind='linear', outer_k=5, inner_k=5, n_trials=40, metric='accuracy'):
    folds_x, folds_y = create_folds(X, y, outer_k)
    train_scores = np.empty(shape=outer_k, dtype=ScoreMetrics)
    test_scores = np.empty(shape=outer_k, dtype=ScoreMetrics)
    f = open("hyperparameters.txt", "+w", encoding="utf-8")
    if kind == 'linear':
        f.write("Fold,Lambda\n")
    elif kind == 'rbf':
        f.write("Fold,Lambda,Gamma\n")
    elif kind == 'poly':
        f.write("Fold,Lambda,Degree\n")
    for i in range(outer_k):
        print(f"OUTER FOLD NUMBER {i+1}")
        train_x = np.concat(folds_x[0:i] + folds_x[i+1:outer_k])
        train_y = np.concat(folds_y[0:i] + folds_y[i+1:outer_k])
        test_x = folds_x[i]
        test_y = folds_y[i]

        def objective(trial):
            params = {
                #'model': trial.suggest_categorical('model', ['SVM', 'LogReg']),
                'lambda_': trial.suggest_float("lambda_", 1e-12, 10, log=True),
                #'tol': trial.suggest_float("tol", 1e-6, 0.01, log=True),
                #'kernel': trial.suggest_categorical("kernel", ['linear', 'rbf', 'poly']),
            }
            if kind == 'rbf':
                params['gamma'] = trial.suggest_float('gamma', 0.00001, 100, log=True)
            if kind == 'poly':
                params['degree'] = trial.suggest_int('degree', 2, 15)
            #if params['model'] == 'SVM':
            #    model = SVM(**params)
            #elif params['model'] == 'LogReg':
            #    model = LogReg(**params)
            model.set_params(**params)
            scores = k_fold_CV(train_x, train_y, inner_k, model)['test']
            overall_score = 0.0
            for s in scores:
                overall_score += getattr(s, metric)
            return overall_score / inner_k
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
        best_params = study.best_params

        if kind == 'linear':
            f.write(f"{i},{best_params['lambda_']}\n")
        elif kind == 'rbf':
            f.write(f"{i},{best_params['lambda_']},{best_params['gamma']}\n")
        elif kind == 'poly':
            f.write(f"{i},{best_params['lambda_']},{best_params['degree']}\n")
        
        #if best_params['model'] == 'SVM':
        #    model = SVM(**best_params)
        #elif best_params['model'] == 'LogReg':
        #    model = LogReg(**best_params)
        #model.set_params(**best_params)
        model.set_params(**best_params)
        perf = open(f"fold_{i}.txt", "+w", encoding="utf-8")
        
        train_x, train_y, test_x, test_y = preprocess(train_x, train_y, test_x, test_y)
        fold_scores = evaluate(model, train_x, train_y, test_x, test_y, track=perf)
        train_score = fold_scores['train']
        test_score = fold_scores['test']
        train_scores[i] = train_score
        test_scores[i] = test_score
        #print("Fold test performance:")
        #test_score.print_mat()
        #test_score.print_metrics()
        #print("Fold train performance:")
        #train_score.print_mat()
        #train_score.print_metrics()
        #perf.close()
    train_aggregate = ScoreMetrics.aggregate(train_scores)
    test_aggregate = ScoreMetrics.aggregate(test_scores)
    print("Overall test performance:")
    test_aggregate.print_mat()
    test_aggregate.print_metrics()
    print("Overall train performance:")
    train_aggregate.print_mat()
    train_aggregate.print_metrics()
    f.close()
    return {
        'train': train_aggregate,
        'test': test_aggregate
    }
