import torch

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)


def evaluate_model(
    model,
    X_test,
    y_test
):

    with torch.no_grad():

        predictions = model(X_test)

        predictions = (
            predictions > 0.5
        ).int()


    metrics = {

        "accuracy":
            accuracy_score(
                y_test,
                predictions
            ),

        "precision":
            precision_score(
                y_test,
                predictions
            ),

        "recall":
            recall_score(
                y_test,
                predictions
            ),

        "f1":
            f1_score(
                y_test,
                predictions
            )
    }


    return metrics