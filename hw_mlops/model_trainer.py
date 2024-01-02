import pathlib
import pickle

import pandas as pd
from dvc.api import DVCFileSystem
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class ModelTrainer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.random_state = 42

    def get_train_data(self) -> pd.DataFrame and pd.Series:
        path_to_train = "data/train.csv"
        target = "MedHouseVal"
        # assuming this will be run in the git directory
        fs = DVCFileSystem(pathlib.Path().resolve())
        fs.get(path_to_train, path_to_train)
        df_train = pd.read_csv(path_to_train)
        return df_train.drop(columns=[target]), df_train[target]

    def scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        return pd.DataFrame(data=self.scaler.transform(X), columns=X.columns)

    def train_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model = MLPRegressor(self.random_state)
        self.model.fit(X, y)

    def save_model(self) -> None:
        model_dict = {"model": self.model, "scaler": self.scaler}
        with open("model/model.pkl", "wb") as file:
            pickle.dump(model_dict, file)
