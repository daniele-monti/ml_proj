import pandas as pd
from itertools import accumulate, chain, pairwise
from typing import Iterator, List
from preprocessing import preprocess, clip_outliers
from evaluation import k_fold_CV, nested_CV, evaluate
from kernel_models import SVM, LogReg
#from linear_models import LogReg, SVM

from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split


from data import GOOD, BAD


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


def a_main():
    #data = load_breast_cancer(as_frame=True)
    #features = data['feature_names']
    #label = 'target'
    #dataset = data['frame']
    #dataset[label] = dataset[label].map(lambda x: 2*x - 1)
    #train, test = shuffle_split(dataset, [0.8, 0.2])
    
    #train_x = train[features].to_numpy()
    #train_y = train[label].to_numpy()
    #test_x = test[features].to_numpy()
    #test_y = test[label].to_numpy()
    
    X, y = make_classification(
        n_samples=20000,
        n_features=20,
        n_informative=20,
        n_redundant=0,
        random_state=123,
        flip_y=0.2,
        hypercube=False,
        weights=[0.60]
    )
    y = (lambda x: 2*x - 1)(y)
    #train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
    #train_x, train_y, test_x, test_y = preprocess(train_x, train_y, test_x, test_y)

    #evaluate(
    #    LogReg(lambda_par=0.00001, n_iter_no_changes=5, tol=0.001, max_iter=1500, kernel='rbf', degree=2, gamma=0.3),
    #    train_x,
    #    train_y,
    #    test_x,
    #    test_y
    #)
    k_fold_CV(
        X, 
        y,
        5, 
        LogReg(lambda_par=0.00001, n_iter_no_changes=5, tol=0.001, max_iter=1500, kernel='rbf', degree=2, gamma=0.3),
    )

def main():
    wines = load_csv("winequality-red.csv", "winequality-white.csv")
    wines[label] = wines[label].apply(to_binary_class)

    clipped = clip_outliers(wines, columns=features)
    X = clipped[features].to_numpy()
    y = clipped[label].to_numpy()

    nested_CV(
        X,
        y,
        SVM(),
    )
    #train, test = shuffle_split(clipped, [0.8, 0.2])
    #scaler = prep.Scaler(train, columns=features)
    #train = scaler.scale(train)
    #test = scaler.scale(test)
    #train_x = train[features].to_numpy()
    #train_y = train[label].to_numpy()
    #test_x = test[features].to_numpy()
    #test_y = test[label].to_numpy()

    #train_x, train_y, test_x, test_y = preprocess(train_x, train_y, test_x, test_y)

    #evaluate(
    #    LogReg(lambda_par=0.00003, n_iter_no_changes=5, tol=0.001, max_iter=1500, kernel='rbf', degree=4, gamma=1),
    #    train_x,
    #    train_y,
    #    test_x,
    #    test_y
    #)

if __name__ == "__main__":
    main()
