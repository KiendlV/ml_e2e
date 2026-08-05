import torch
from torch.utils.data import (
    TensorDataset,
    DataLoader
)

import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature


def train_model(
    model,
    X_train,
    y_train,
    epochs,
    batch_size,
    learning_rate
):

    dataset = TensorDataset(
        X_train,
        y_train
    )


    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )


    criterion = torch.nn.BCELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )


    with mlflow.start_run():

        mlflow.log_params(
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate
            }
        )


        for epoch in range(epochs):

            for X_batch, y_batch in loader:

                prediction = model(X_batch)

                loss = criterion(
                    prediction,
                    y_batch
                )


                optimizer.zero_grad()

                loss.backward()

                optimizer.step()


            mlflow.log_metric(
                "loss",
                loss.item(),
                step=epoch
            )


            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}: {loss.item()}"
                )

        signature = infer_signature(
            X_train[:1].numpy(),
            model(X_train[:1]).detach().numpy()
        )

        mlflow.pytorch.log_model(
            pytorch_model=model,
            name="rain_predictor",
            input_example=X_train[:1],
            signature=signature
        )


    return model