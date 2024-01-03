from dataclasses import dataclass


@dataclass
class Path:
    training_path: str
    test_path: str
    model_save_path: str
    predictions_save_path: str


@dataclass
class Params:
    optimizer: str
    tracking_uri_mlflow: str


@dataclass
class MyConfig:
    path: Path
    params: Params
