import pickle
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

data = pd.read_csv("./lib/data/train.csv")
fit = pd.read_csv("./lib/data/trainSamples.csv")
x_fit = fit.drop(["SalePrice"], axis=1)
y_fit = fit["SalePrice"]

class GetResources:

	@staticmethod
	def getModel():
		pipeline = joblib.load('./lib/model/model_nn.joblib')
		pipeline.named_steps['model'].model_ = load_model('./lib/model/model_nn.keras')
		return pipeline

	@staticmethod
	def getModelRegression():
		pipeline = joblib.load('./lib/model/model_regression.joblib')
		return pipeline

	@staticmethod
	def getLookupDict():
		dict_file =  './lib/dict/encoding.pickle'
		file = open(dict_file,'rb')
		lookupDict = pickle.load(file)
		file.close()
		return lookupDict

	@staticmethod
	def getXScaler(name):
		scaler = StandardScaler()
		scaler.fit(data[name].values.reshape(-1, 1))
		return scaler

	@staticmethod
	def getYScaler():
		scaler = StandardScaler()
		scaler.fit(data["SalePrice"].values.reshape(-1, 1))
		return scaler

	@staticmethod
	def getColumnNames():
		return data.drop(["SalePrice"], axis=1).columns

	@staticmethod
	def qualitativeColumns():
		return [f for f in data.columns if data.dtypes[f] == "object"]
