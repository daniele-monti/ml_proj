import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data import wines_discrete_quality, wines, label


with open("analysis.txt", mode="w") as f:
    f.write(f"Sample size: {len(wines)}\n\n")

    counts = wines_discrete_quality[label].value_counts().sort_index()
    norm = wines_discrete_quality[label].value_counts(normalize=True).sort_index()
    f.write(f"Number of wines for each quality value:\n")
    for quality in counts.index:
        f.write(f"quality {quality} -> {counts[quality]} wines \t\t\t({round(norm[quality]*100, 2)}%)\n")

    # now quality is a binary class
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
        plt.savefig(f"images/hist/{col}_hist")
        plt.close()
    

    for feature_1 in wines.columns[:-1]:
        for feature_2 in wines.columns[wines.columns.get_loc(feature_1)+1:-1]:
            sns.scatterplot(x=feature_1, y=feature_2, data=wines, hue=label)
            plt.savefig(f"images/scatter/{feature_1}_{feature_2}_scatter")
            plt.close()
    
    sns.pairplot(wines, hue=label).savefig("images/pairplot")
