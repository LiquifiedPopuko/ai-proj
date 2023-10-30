from flask import Flask, jsonify, request
from getResources import GetResources
from collections import OrderedDict
import pandas as pd
import tensorflow as tf

print(tf.__version__)

app = Flask(__name__)
app.model = GetResources.getModel()
app.model_regression = GetResources.getModelRegression()
app.lookupDict = GetResources.getLookupDict()
app.priceScaler = GetResources.getYScaler()
app.columnNames = GetResources.getColumnNames()
app.qualitativeColumns = GetResources.qualitativeColumns()

@app.route('/estimate', methods=['GET'])
def estimate():
	argList = request.args.to_dict(flat=False)
	queryDF=pd.DataFrame.from_dict(OrderedDict(argList))
	# print(app.lookupDict)
	try:
		for feat in queryDF.columns:
			# print(f'{feat}={queryDF[feat].values[0]}')
			if feat in app.qualitativeColumns:
				print(f'{feat} is qualitative')
				value = queryDF[feat].values[0]
				queryDF.drop(feat, axis=1, inplace=True)
				feat = feat + "_E"
				if feat in app.lookupDict:
					# print(f'{feat}={app.lookupDict[feat]}')
					try:
						queryDF[feat] = app.lookupDict[feat][value]
					except:
						queryDF[feat] = app.lookupDict[feat]["MISSING"]
			else:
				queryDF[feat] = queryDF[feat].astype("float64")
	except Exception as e:
		print(e)
		return "Error - check params"
	# print(queryDF)
	estimatedRating = app.model.predict(queryDF)[0]

	return jsonify(rating=str(estimatedRating))

@app.route('/estimate-regression', methods=['GET'])
def estimate_regression():
	argList = request.args.to_dict(flat=False)
	queryDF=pd.DataFrame.from_dict(OrderedDict(argList))
	# print(app.lookupDict)
	try:
		for feat in queryDF.columns:
			# print(f'{feat}={queryDF[feat].values[0]}')
			if feat in app.qualitativeColumns:
				print(f'{feat} is qualitative')
				value = queryDF[feat].values[0]
				queryDF.drop(feat, axis=1, inplace=True)
				feat = feat + "_E"
				if feat in app.lookupDict:
					# print(f'{feat}={app.lookupDict[feat]}')
					try:
						queryDF[feat] = app.lookupDict[feat][value]
					except:
						queryDF[feat] = app.lookupDict[feat]["MISSING"]
			else:
				queryDF[feat] = queryDF[feat].astype("float64")
	except Exception as e:
		print(e)
		return "Error - check params"
	# print(queryDF)
	estimatedRating = app.model_regression.predict(queryDF)[0]

	return jsonify(rating=str(estimatedRating))

@app.route('/', methods=['GET'])
def hello():
	return 'hello'

if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0')
