from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model_predict = joblib.load("./covid_prediction.joblib")

@app.route("/predict", methods=["POST"])

def predict():
    data = request.json["data"]
    prediction = model_predict.predict(data)
    return jsonify({
        "Prediction" : prediction.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)