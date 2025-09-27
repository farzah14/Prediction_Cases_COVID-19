from flask import Flask, request, jsonify
import json
import joblib
from datetime import datetime
import os

app = Flask(__name__)

# Load model
model_predict = joblib.load("./covid_prediction.joblib")

# File log untuk menyimpan prediksi
name_file = "result_predict.json"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ambil data dari request
        input_data = request.json.get("data")
        if input_data is None:
            return jsonify({"error": "Field 'data' tidak ditemukan"}), 400

        # Lakukan prediksi
        prediction = model_predict.predict(input_data).tolist()

        # Siapkan payload untuk disimpan
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "prediction": prediction
        }

        # Baca log lama (jika ada)
        if os.path.exists(name_file):
            with open(name_file, "r", encoding="utf-8") as f:
                try:
                    logs = json.load(f)
                except json.JSONDecodeError:
                    logs = []  # Jika file corrupt/empty, mulai dari list kosong
        else:
            logs = []

        # Tambahkan entri baru
        logs.append(log_entry)

        # Simpan kembali ke file
        with open(name_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=4)

        # Kirim respons
        return jsonify({"Prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)