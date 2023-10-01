from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from xgboost import XGBClassifier
import numpy as np
from category_encoders import TargetEncoder
import pickle

MODEL_FEATURES = ['loan_term', 'loan_lux_asset_ratio', 'residential_assets_value', 'res_asset_ratio', 'loan_income_ratio', 'lux_asset_ratio', 'bank_asset_ratio', 'education', 'loan_amount', 'com_asset_ratio']
CATEGORIC_FEATURES = ['education']

app = Flask(__name__)
model = pickle.load(open("model.pkl","rb"))
encoder = pickle.load(open("encoder.pkl","rb"))


@app.route('/')
def home():
    return "The Service is up & Running..."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['data']
        data = pd.DataFrame.from_dict([data])
        data = data.loc[:,MODEL_FEATURES]
        data.loc[:,CATEGORIC_FEATURES] = encoder.transform(data.loc[:,CATEGORIC_FEATURES])

        for col in data.select_dtypes(object).columns:
            data[col] = data[col].astype(np.float64)

        prediction = model.predict(data)[0]
        response = {"prediction": int(prediction)}

        return jsonify(response), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)