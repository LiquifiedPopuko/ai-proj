import pandas as pd
import requests
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np


def load_test_data():
    return pd.read_csv("./lib/data/testSamples.csv")


def convert_to_query(x):
    tuple_list = list(zip("&" + x.index + "=", x.astype(str)))
    return "".join(["".join(item) for item in tuple_list])


if __name__ == "__main__":
    data = load_test_data()
    data["query"] = data.drop("SalePrice", axis=1).apply(
        lambda x: convert_to_query(x), axis=1
    )
    dataframe_test = data
    dataframe_test["predict_nn"] = dataframe_test.apply(
        lambda x: requests.get(r"http://127.0.0.1:80/estimate?" + x["query"]).json()[
            "rating"
        ],
        axis=1,
    ).astype("float64")
    dataframe_test["predict_reg"] = dataframe_test.apply(
        lambda x: requests.get(
            r"http://127.0.0.1:80/estimate-regression?" + x["query"]
        ).json()["rating"],
        axis=1,
    ).astype("float64")
    print(dataframe_test[["SalePrice", "predict_nn", "predict_reg"]])
    print(
        f'rmse_nn = {mean_squared_error(dataframe_test["SalePrice"],dataframe_test["predict_nn"], squared=False)}'
    )
    print(
        f'rmse_reg = {mean_squared_error(dataframe_test["SalePrice"],dataframe_test["predict_reg"], squared=False)}'
    )
    print(
        f'mape_nn = {mean_absolute_percentage_error(dataframe_test["SalePrice"],dataframe_test["predict_nn"])}'
    )
    print(
        f'mape_reg = {mean_absolute_percentage_error(dataframe_test["SalePrice"],dataframe_test["predict_reg"])}'
    )
    print("Kaggle score is log1p then rmse")
    print(
        f'kaggle_score_nn = {mean_squared_error(np.log1p(dataframe_test["SalePrice"]),np.log1p(dataframe_test["predict_nn"]), squared=False)}'
    )
    print(
        f'kaggle_score_reg = {mean_squared_error(np.log1p(dataframe_test["SalePrice"]),np.log1p(dataframe_test["predict_reg"]), squared=False)}'
    )
    print(
        f'log1p_mse_nn = {mean_squared_error(np.log1p(dataframe_test["SalePrice"]),np.log1p(dataframe_test["predict_nn"]))}'
    )
    print(
        f'log1p_mse_reg = {mean_squared_error(np.log1p(dataframe_test["SalePrice"]),np.log1p(dataframe_test["predict_reg"]))}'
    )
