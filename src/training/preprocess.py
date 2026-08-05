import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess(
    df,
    features,
    test_size,
    random_state
):

    X = df[features]

    df["rain"] = (
        df["rain"] > 0
    ).astype(int)

    y = df["rain"]


    scaler = StandardScaler()

    X = scaler.fit_transform(X)


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )


    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    X_test = torch.tensor(
        X_test,
        dtype=torch.float32
    )


    y_train = torch.tensor(
        y_train.values,
        dtype=torch.float32
    ).reshape(-1,1)


    y_test = torch.tensor(
        y_test.values,
        dtype=torch.float32
    ).reshape(-1,1)


    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    )