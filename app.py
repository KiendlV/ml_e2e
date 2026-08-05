from flask import Flask, request, jsonify
import torch
import socket
import joblib

from models.pytorch_deep_model import RainPredictor


app = Flask(__name__)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = RainPredictor()

model.load_state_dict(
    torch.load(
        "models/model_weights_v1/base_ff_model_v1.pth",
        map_location=device
    )
)

model.to(device)

model.eval()


scaler = joblib.load(
    "models/model_weights_v1/scaler.pkl"
)

@app.post("/predict")
def predict():

    data = request.get_json()

    features = data["features"]

    features = scaler.transform(
        [features]
    )


    x = torch.tensor(
        features,
        dtype=torch.float32
    )


    x = x.to(device)


    with torch.inference_mode():

        output = model(x)

        prediction = torch.argmax(
            output,
            dim=1
        )


    return jsonify(
        {
            "prediction": prediction.item()
        }
    )


@app.route("/")
def hello():

    return (
        f"Hello from Kubernetes! "
        f"Host: {socket.gethostname()}"
    )

@app.get("/health")
def health():

    return {
        "status": "ok"
    }, 200

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )