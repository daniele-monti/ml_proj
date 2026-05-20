import pandas as pd
from itertools import accumulate, chain, pairwise
from typing import Iterator, List
import preprocessing as prep

GOOD = 1
BAD = -1

features = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol"
]

label = "quality"


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


def count(splits):
    for i, s in enumerate(splits):
        counts = s[label].value_counts().sort_index()
        norm = s[label].value_counts(normalize=True).sort_index()
        print(f"\nNumber of good (1) and bad (-1) wines in split number {i}:\n")
        for quality in counts.index:
            print(f"quality {quality} -> {counts[quality]} wines \t({round(norm[quality]*100, 2)}%)\n")


from logistic_regression import SVM
import metrics
from sklearn.linear_model import SGDClassifier

def main():
    wines = load_csv("winequality-red.csv", "winequality-white.csv")
    wines[label] = wines[label].apply(to_binary_class)

    clipped = prep.clip_outliers(wines, columns=features)

    train, test = shuffle_split(clipped, [0.8, 0.2])
    scaler = prep.Scaler(train, columns=features)
    train = scaler.scale(train)
    test = scaler.scale(test)

    metrics.train(
        SVM(iterations=1000000, lambda_par=0.0001),
        train[features].to_numpy(),
        train[label].to_numpy(),
        test[features].to_numpy(),
        test[label].to_numpy()
    )

    #print("sklearn --------------------------------\n")
    #metrics.train(
    #    SGDClassifier(max_iter=10000, tol=1e-3),
    #    train[features].to_numpy(),
    #    train[label].to_numpy(),
    #    test[features].to_numpy(),
    #    test[label].to_numpy()
    #)

if __name__ == "__main__":
    main()
