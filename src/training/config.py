from pathlib import Path
import yaml


CONFIG_PATH = Path("configs/training.yaml")


def load_config():

    with open(CONFIG_PATH, "r") as file:
        return yaml.safe_load(file)