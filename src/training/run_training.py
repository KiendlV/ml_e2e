import mlflow

from src.training.config import load_config
from src.training.data import load_training_data
from src.training.preprocess import preprocess
from src.training.train import train_model
from src.training.evaluate import evaluate_model
from src.training.save import save_model

from models.pytorch_deep_model import RainPredictor


def main():

    config = load_config()


    mlflow.set_experiment(
        "rain-predictor"
    )


    df = load_training_data(
        config["data"]["file"]
    )


    X_train, X_test, y_train, y_test, scaler = preprocess(
        df,
        config["features"],
        config["training"]["test_size"],
        config["training"]["random_state"]
    )


    model = RainPredictor()


    model = train_model(
        model,
        X_train,
        y_train,
        config["training"]["epochs"],
        config["training"]["batch_size"],
        config["training"]["learning_rate"]
    )


    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )


    print(metrics)


    save_model(
        model,
        scaler,
        config["model"]["name"]
    )


if __name__ == "__main__":
    main()