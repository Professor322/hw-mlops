import pickle

import numpy as np
import pandas as pd
from dvc.api import DVCFileSystem
from hydra.utils import get_original_cwd


class InferenceRunner:
    def __init__(self, test_data_path, model_path, predictions_save_path):
        self.test_data_path = test_data_path
        self.original_dir = get_original_cwd()
        self.predictions_save_path = self.original_dir + predictions_save_path
        self.model_path = self.original_dir + model_path
        with open(self.model_path, "rb") as file:
            model_dict = pickle.load(file)
            self.model = model_dict["model"]
            self.scaler = model_dict["scaler"]

    def get_test_data(self) -> pd.DataFrame and pd.Series:
        target = "MedHouseVal"
        # assuming this will be run in the git directory
        fs = DVCFileSystem(self.original_dir)
        local_copy = self.original_dir + self.test_data_path
        fs.get(self.test_data_path, local_copy)
        df_test = pd.read_csv(local_copy)
        return df_test.drop(columns=[target]), df_test[target]

    def scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        self.scaler.fit(X)
        return pd.DataFrame(data=self.scaler.transform(X), columns=X.columns)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.y_pred = self.model.predict(X)
        return self.y_pred

    def save_predictions(self):
        pd.Series(self.y_pred).to_csv(self.predictions_save_path)
        print(f"predictions are saved in {self.predictions_save_path}")
