import pickle

import pandas as pd
from dvc.api import DVCFileSystem
from hydra.utils import get_original_cwd
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class ModelTrainer:
    def __init__(self, train_data_path, optimizer, model_save_path):
        self.model = None
        self.scaler = None
        self.random_state = 42
        self.optimizer = optimizer
        self.train_data_path = train_data_path
        self.original_dir = get_original_cwd()
        self.model_save_path = self.original_dir + model_save_path
        self.score_ = 0

    def get_train_data(self) -> pd.DataFrame and pd.Series:
        target = "MedHouseVal"
        # assuming this will be run in the git directory
        fs = DVCFileSystem(self.original_dir)
        local_copy = self.original_dir + self.train_data_path
        fs.get(self.train_data_path, local_copy)
        df_train = pd.read_csv(local_copy)
        return df_train.drop(columns=[target]), df_train[target]

    def scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        return pd.DataFrame(data=self.scaler.transform(X), columns=X.columns)

    def train_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model = MLPRegressor(self.random_state, solver=self.optimizer)
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        self.score_ = r2_score(y_true=y, y_pred=y_pred)

    def save_model(self) -> None:
        model_dict = {"model": self.model, "scaler": self.scaler}
        with open(self.model_save_path, "wb") as file:
            pickle.dump(model_dict, file)
            print(f"model saved at {self.model_save_path}")
