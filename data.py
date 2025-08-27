import pandas as pd

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


red = pd.read_csv("winequality-red.csv", sep=";")
white = pd.read_csv("winequality-white.csv", sep=";")

wines = pd.concat([red, white], ignore_index=True)

wines_discrete_quality = wines.copy(deep=True)
wines[label] = wines[label].apply(to_binary_class)

# 80 - 20 split for train+dev - test sets
train_dev = wines.sample(frac=0.8)
test = wines.drop(train_dev.index)

# 75 - 25 split for train - dev in order to obtain an overall split of
# 60 - 20 - 20 for train - dev - test sets
train = train_dev.sample(frac=0.75)
dev = train_dev.drop(train.index)
