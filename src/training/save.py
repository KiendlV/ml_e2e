import torch
import joblib

from src.paths import WEIGHTS_DIR


def save_model(
    model,
    scaler,
    name
):

    WEIGHTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    torch.save(
        model.state_dict(),
        WEIGHTS_DIR / f"{name}.pth"
    )


    joblib.dump(
        scaler,
        WEIGHTS_DIR / "scaler.pkl"
    )