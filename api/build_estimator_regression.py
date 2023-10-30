import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


class BuildRegressorEstimator:
    @staticmethod
    def getBestPipeline(x, y):
        param_grid = {
            'pca__n_components': [0.1, 0.3, 0.5, 0.7, 0.9],
            'model__max_depth': [2, 3, 5, 7, 10],
            'model__n_estimators': [10, 100, 500],
        }

        pipe = Pipeline(steps=[
                ('standard_scaler',StandardScaler()),
                ('pca', PCA(svd_solver='full')),
                ('model', XGBRegressor(colsample_bytree=0.6, subsample=0.6, missing=0))
        ])
        cv = GridSearchCV(pipe,param_grid,cv=10,verbose=0,scoring='neg_root_mean_squared_error',n_jobs=-1,error_score=0.0)
        cv.fit(x, y)

        print("Best: %f using %s" % (cv.best_score_, cv.best_params_))

        return cv

    @staticmethod
    def generateBestModel():
        fit = pd.read_csv("./lib/data/trainSamples.csv")
        x_fit = fit.drop(["SalePrice"], axis=1)
        y_fit = fit["SalePrice"]

        blind_test = pd.read_csv("./lib/data/testSamples.csv")
        x_blind_test = blind_test.drop(["SalePrice"], axis=1)
        y_blind_test = blind_test["SalePrice"]

        optimizedModel = BuildRegressorEstimator.getBestPipeline(x_fit.values,y_fit.values).best_estimator_
        yPredFit  = optimizedModel.predict(x_fit)
        yPredTest = optimizedModel.predict(x_blind_test)

        fit_score = mean_squared_error(y_fit,yPredFit)
        test_score = mean_squared_error(y_blind_test,yPredTest)

        print("fit mse = %.2f and test mse = %.2f" %(fit_score,test_score))

        return optimizedModel
