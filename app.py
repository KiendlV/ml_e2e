from flask import Flask, request, jsonify

import torch
import socket


from src.paths import WEIGHTS_DIR

from models.pytorch_deep_model import RainPredictor

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RainPredictor()
model.load_state_dict(torch.load(WEIGHTS_DIR / "base_ff_model_v1.pth", map_location=device))
model.to(device)
model.eval()



@app.post("/predict")
def predict():

    data = request.get_json()

    features = data["features"]

    x = torch.tensor(
        features,
        dtype=torch.float32
    )

    x = x.unsqueeze(0).to(device)

    with torch.inference_mode():
        output = model(x)
        prediction = torch.argmax(output, dim=1)

    return jsonify({
        "prediction": prediction.item()
    })

@app.route("/")
def hello():
    return f"Hello from Kubernetes! Host: {socket.gethostname()}\n"

@app.route("/health")
def health():
    return "OK\n"

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )