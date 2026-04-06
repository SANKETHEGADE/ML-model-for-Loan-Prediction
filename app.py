from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        data = np.array([data])
        data = scaler.transform(data)

        prediction = model.predict(data)[0]

        result = "Loan Approved ✅" if prediction == 1 else "Loan Not Approved ❌"

        return render_template("index.html", prediction_text=result)

    except:
        return render_template("index.html", prediction_text="Error in input")

if __name__ == "__main__":
    app.run(debug=True)