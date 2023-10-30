import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from scikeras.wrappers import KerasRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class BuildEstimator:
    @staticmethod
    def generateBestModel():
        fit = pd.read_csv("./lib/data/trainSamples.csv")
        x_fit = fit.drop(["SalePrice"], axis=1)
        y_fit = fit["SalePrice"]

        blind_test = pd.read_csv("./lib/data/testSamples.csv")
        x_blind_test = blind_test.drop(["SalePrice"], axis=1)
        y_blind_test = blind_test["SalePrice"]

        search_params = {
            "model__hidden_layer_dim": [10, 50, 100],
            "model__batch_size": [64, 128, 256, 512, 1024],
            "model__epochs": [10, 25, 50],
            "model__optimizer__learning_rate": [0.001, 0.01, 0.1],
        }
        pipe = Pipeline(
            [
                ("standard_scaler", StandardScaler()),
                ("pca", PCA(n_components=0.5, svd_solver="full")),
                (
                    "model",
                    KerasRegressor(
                        model=BuildEstimator.createModel,
                        verbose=0,
                        loss="mean_absolute_error",
                        optimizer="adam",
                        hidden_layer_dim="100"
                    ),
                ),
            ]
        )
        cv = GridSearchCV(
            estimator=pipe,
            param_grid=search_params,
            cv=10,
            verbose=0,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            error_score=0.0,
        )
        res = cv.fit(x_fit, y_fit)

        optimizedModel = res.best_estimator_
        y_pred_fit = optimizedModel.predict(x_fit)
        y_pred_test = optimizedModel.predict(x_blind_test)

        fit_score = mean_squared_error(y_fit, y_pred_fit)
        test_score = mean_squared_error(y_blind_test, y_pred_test)

        print("fit mse = %.2f and test mse = %.2f" % (fit_score, test_score))
        print("Best: %f using %s" % (res.best_score_, res.best_params_))

        return optimizedModel

    @staticmethod
    def createModel(hidden_layer_dim, meta):
        n_features_in_ = meta["n_features_in_"]
        X_shape_ = meta["X_shape_"]
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(n_features_in_, input_shape=X_shape_[1:], kernel_initializer='normal'))
        model.add(
            tf.keras.layers.Dense(
                hidden_layer_dim, activation="relu", kernel_initializer='normal'
            )
        )
        model.add(
            tf.keras.layers.Dense(
                hidden_layer_dim, activation="relu", kernel_initializer='normal'
            )
        )
        model.add(
            tf.keras.layers.Dense(
                hidden_layer_dim, activation="relu", kernel_initializer='normal'
            )
        )
        model.add(tf.keras.layers.Dense(1, activation="linear", kernel_initializer='normal'))
        return model
