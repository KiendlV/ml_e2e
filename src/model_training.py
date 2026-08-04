import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path

from src.paths import PROCESSED_DATA_DIR, WEIGHTS_DIR

from models.pytorch_deep_model import RainPredictor

parquet_data_path = PROCESSED_DATA_DIR / "historical_training_data_2025.parquet"

model_path = WEIGHTS_DIR / "base_ff_model_v1.pth"

df = pd.read_parquet(path = parquet_data_path)

features = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure"
]

X = df[features]

df["rain"] = (df["rain"] > 0).astype(int)
y = df["rain"]



scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)

train_dataset = TensorDataset(X_train, y_train)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

model = RainPredictor()

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 100

for epoch in range(epochs):

    for X_batch, y_batch in train_loader:

        pred = model(X_batch)

        loss = criterion(pred, y_batch)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: {loss.item():.4f}")

WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

torch.save(model.state_dict(), model_path)