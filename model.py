import numpy as np
import pandas as pd
from main import GOOD, BAD


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
        #char_count = 31
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
