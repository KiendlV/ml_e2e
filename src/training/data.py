import pandas as pd
from src.paths import PROCESSED_DATA_DIR


def load_training_data(filename):

    path = PROCESSED_DATA_DIR / filename

    return pd.read_parquet(path)