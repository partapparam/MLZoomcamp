import pickle

from flask import Flask
from flask import request
from flask import jsonify

model_file = "model1.bin"
dv_file = "dv.bin"

with open(model_file, "rb") as f_in:
    model = pickle.load(f_in)
with open(dv_file, "rb") as f_dv_in:
    dv = pickle.load(f_dv_in)

app = Flask("bank")


@app.route("/predict", methods=["POST"])
def predict():
    client = request.get_json()

    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]
    score = y_pred >= 0.5

    result = {"credit_prop": float(y_pred), "score": bool(score)}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
