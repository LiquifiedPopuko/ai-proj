import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

encodingDict = dict()

def encode(frame, feature):
    ordering = pd.DataFrame()
    ordering["val"] = frame[feature].unique()
    ordering.index = ordering.val
    ordering["spmean"] = (
        frame[[feature, "SalePrice"]]
        .groupby(feature, observed=True)
        .mean()["SalePrice"]
    )
    ordering = ordering.sort_values("spmean")
    ordering["ordering"] = range(1, ordering.shape[0] + 1)
    ordering = ordering["ordering"].to_dict()
    encodingDict[feature + "_E"] = ordering

    for cat, o in ordering.items():
        frame.loc[frame[feature] == cat, feature + "_E"] = o

class SampleCreator:
    @staticmethod
    def createBlindTestSamples():
        data = pd.read_csv("./lib/data/train.csv")
        # https://www.kaggle.com/code/dgawlik/house-prices-eda
        quantitative = [f for f in data.columns if data.dtypes[f] != "object"]
        quantitative.remove("SalePrice")
        quantitative.remove("Id")
        qualitative = [f for f in data.columns if data.dtypes[f] == "object"]
        for c in qualitative:
            data[c] = data[c].astype("category")
            if data[c].isnull().any():
                data[c] = data[c].cat.add_categories(["MISSING"])
                data[c] = data[c].fillna("MISSING")
        for q in qualitative:
            encode(data, q)
            data.drop(q, axis=1, inplace=True)
        data.dropna(how="any", axis=0, inplace=True)
        X = data.drop(["SalePrice", "Id"], axis=1)
        Y = data["SalePrice"]
        XFit, XBlindTest, yFit, yBlindTest = train_test_split(X, Y, test_size=0.2)
        column_head = pd.Index(["SalePrice"]).append(XFit.columns)
        train = pd.DataFrame(np.column_stack([yFit, XFit]), columns=column_head)
        blind = pd.DataFrame(
            np.column_stack([yBlindTest, XBlindTest]), columns=column_head
        )

        train.to_csv("./lib/data/trainSamples.csv", index=False)
        blind.to_csv("./lib/data/testSamples.csv", index=False)
        file = open("./lib/dict/encoding.pickle", "wb")
        pickle.dump(encodingDict, file)
        file.close()

