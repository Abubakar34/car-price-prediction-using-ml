import pickle
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("predict.html")


@app.route("/predict", methods=["POST"])
def predict():
    model = pickle.load(open("./model/LinearRegressionModel.pkl", "rb"))

    present_price = float(request.form["present_price"])
    kms_driven = float(request.form["kms_driven"])
    fuel_type = request.form["fuel_type"]
    seller_type = request.form["seller_type"]
    transmission = request.form["transmission"]
    owner = int(request.form["owner"])
    year = int(request.form["year"])

    # Assuming fuel_type, seller_type, and transmission are preprocessed inside the model
    columns = ["Present_Price", "Fuel_Type", "Selling_type", "Transmission"]
    input = [present_price, fuel_type, seller_type, transmission]

    prediction = model.predict(
        pd.DataFrame(columns=columns, data=np.array(input).reshape(1, 4))
    )

    # Return prediction as JSON
    return jsonify({"prediction": f"{float(prediction[0]):.2f}"})


if __name__ == "__main__":
    app.run(debug=True)
