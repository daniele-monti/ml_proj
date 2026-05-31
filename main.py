import pandas as pd
from itertools import accumulate, chain, pairwise
from typing import Iterator, List
from preprocessing import preprocess
from evaluation import nested_CV, evaluate
from kernel_models import SVM, LogReg
from linear_models import LinearLogReg, LinearSVM


from data import GOOD, BAD, features, label


def to_binary_class(quality):
    if quality >= 6:
        return GOOD
    else:
        return BAD


def load_csv(red_path, white_path):
    red = pd.read_csv(red_path, sep=";")
    white = pd.read_csv(white_path, sep=";")
    return pd.concat([red, white], ignore_index=True)



def shuffle_split(frame: pd.DataFrame, ratios: Iterator[float]) -> List[pd.DataFrame]:
    """
    returns a shuffled split of the frame based on the ratios in the list specifed
    """
    assert sum(ratios) == 1.0
    splits = []
    shuffled = frame.sample(frac=1)
    for start, end in pairwise(int(r*len(frame)) for r in accumulate(chain([0], ratios))):
        splits.append(
            shuffled.iloc[start:end]
        )
    return splits


def single_evaluation():
    wines = load_csv("winequality-red.csv", "winequality-white.csv")
    wines[label] = wines[label].apply(to_binary_class)

    train, test = shuffle_split(wines, [0.8, 0.2])
    train_x = train[features].to_numpy()
    train_y = train[label].to_numpy()
    test_x = test[features].to_numpy()
    test_y = test[label].to_numpy()

    train_x, train_y, test_x, test_y = preprocess(train_x, train_y, test_x, test_y)

    evaluate(
        SVM(lambda_=0.0005, kernel='poly', degree=9, gamma=0.35617476, max_iter=200, max_iter_no_changes=2),
        train_x,
        train_y,
        test_x,
        test_y,
        display=True,
    )

def nested():
    wines = load_csv("winequality-red.csv", "winequality-white.csv")
    wines[label] = wines[label].apply(to_binary_class)

    wines = wines.sample(frac=1, ignore_index=True)

    X = wines[features].to_numpy()
    y = wines[label].to_numpy()

    nested_CV(X, y, SVM(kernel='linear', max_iter_no_changes=5, max_iter=200), kind='linear', inner_k=5, outer_k=5, n_trials=20)

def main():
    nested()
    #single_evaluation()


if __name__ == "__main__":
    main()
