import pandas as pd
import matplotlib.pyplot as plt 

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
        return 1
    else:
        return -1


red = pd.read_csv("winequality-red.csv", sep=";")
white = pd.read_csv("winequality-white.csv", sep=";")

wines = pd.concat([red, white], ignore_index=True)


with open("analysis.txt", mode="w") as f:
    f.write(f"Sample size: {len(wines)}\n\n")

    counts = wines[label].value_counts().sort_index()
    norm = wines[label].value_counts(normalize=True).sort_index()
    f.write(f"Number of wines for each quality value:\n")
    for quality in counts.index:
        f.write(f"quality {quality} -> {counts[quality]} wines \t\t\t({round(norm[quality]*100, 2)}%)\n")

    # redo the same analysis with quality transofrmed into a binary class
    wines[label] = wines[label].apply(to_binary_class)
    counts = wines[label].value_counts().sort_index()
    norm = wines[label].value_counts(normalize=True).sort_index()
    f.write(f"\nNumber of good (1) and bad (-1) wines:\n")
    for quality in counts.index:
        f.write(f"quality {quality} -> {counts[quality]} wines \t\t\t({round(norm[quality]*100, 2)}%)\n")

    desc = wines.describe().T
    desc["skew"] = wines.skew(axis=0)
    desc['kurtosis'] = wines.kurtosis(axis=0)
    desc.drop(columns='count', inplace=True, axis=1) 
    f.write("\n" + desc.to_string() + "\n")

    for col in wines.columns[:-1]:
        plt.title(col)
        wines[col].plot.hist()
        plt.savefig(f"images/{col}_hist")
        plt.close()
    

# 80 - 20 split for train+dev - test sets
train_dev = wines.sample(frac=0.8)
test = wines.drop(train_dev.index)

# 75 - 25 split for train - dev in order to obtain an overall split of
# 60 - 20 - 20 for train - dev - test sets
train = train_dev.sample(frac=0.75)
dev = train_dev.drop(train.index)
