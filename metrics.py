from data import GOOD, BAD
import numpy as np
import copy


class ScoreMetrics:
    def __init__(self, truth, prediction):
        self.total = len(truth)
        self.support_pos = len(truth[truth == GOOD])
        self.support_neg = len(truth[truth == BAD])
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
        # accuracy
        self.accuracy = (self.true_pos + self.true_neg) / self.total
        # positive case metrics
        try:
            self.precision_pos = self.true_pos / (self.true_pos + self.false_pos)
        except ZeroDivisionError:
            self.precision_pos = np.nan
        try:
            self.recall_pos = self.true_pos / (self.true_pos + self.false_neg)
        except ZeroDivisionError:
            self.recall_pos = np.nan
        try:
            self.f1_score_pos = 2 * self.precision_pos * self.recall_pos / (self.precision_pos + self.recall_pos)
        except ZeroDivisionError:
            self.f1_score_pos = 0.0
        # negative case metrics
        try:
            self.precision_neg = self.true_neg / (self.true_neg + self.false_neg)
        except ZeroDivisionError:
            self.precision_neg = np.nan
        try:
            self.recall_neg = self.true_neg / (self.true_neg + self.false_pos)
        except ZeroDivisionError:
            self.recall_neg = np.nan
        try:
            self.f1_score_neg = 2 * self.precision_neg * self.recall_neg / (self.precision_neg + self.recall_neg)
        except ZeroDivisionError:
            self.f1_score_neg = 0.0
        return

    def print_metrics(self):
        rows = [
            " ".join([ "        "  ,  "precision"                 ,  "  recall"               , "f1_score"                  , "support"]),
            " ".join([f"{GOOD: <8}", f"{self.precision_pos:>9.6f}", f"{self.recall_pos:>8.6f}", f"{self.f1_score_pos:>8.6f}", f"{self.support_pos:>7.2f}"]),
            " ".join([f"{BAD: <8}" , f"{self.precision_neg:>9.6f}", f"{self.recall_neg:>8.6f}", f"{self.f1_score_neg:>8.6f}", f"{self.support_neg:>7.2f}"]),
            " ".join([ "accuracy"  ,  "         "                 ,  "        "               , f"{self.accuracy:>8.6f}"    , f"{self.total:>7.2f}"]),
        ]
        print("\n".join(rows))

    def print_mat(self):
        #char_count = 31
        rows = [
            " | ".join(["         ", "pred pos", "pred neg"]),
            #"+"*char_count,
            " | ".join(["truth pos", f"{self.true_pos:>8}", f"{self.false_neg:>8}"]),
            #"+"*char_count,
            " | ".join(["truth neg", f"{self.false_pos:>8}", f"{self.true_neg:>8}"])
        ]
        print("\n".join(rows))

    def aggregate(scores):
        agg = copy.copy(scores[0])
        k = len(scores)
        for i in range(1, k):
            for attr_name, value in vars(agg).items():
                setattr(agg, attr_name, getattr(scores[i], attr_name)+value)
        for attr_name, value in vars(agg).items():
                setattr(agg, attr_name, value / k)
        return agg
