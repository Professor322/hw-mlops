import pathlib
import pickle

import numpy as np
import pandas as pd
from dvc.api import DVCFileSystem


class InferenceRunner:
    def __init__(self, model_path="model/model.pkl"):
        with open(model_path, "rb") as file:
            model_dict = pickle.load(file)
            self.model = model_dict["model"]
            self.scaler = model_dict["scaler"]

    def get_test_data(self) -> pd.DataFrame and pd.Series:
        path_to_test = "data/test.csv"
        target = "MedHouseVal"
        # assuming this will be run in the git directory
        fs = DVCFileSystem(pathlib.Path().resolve())
        fs.get(path_to_test, path_to_test)
        df_train = pd.read_csv(path_to_test)
        return df_train.drop(columns=[target]), df_train[target]

    def scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        self.scaler.fit(X)
        return pd.DataFrame(data=self.scaler.transform(X), columns=X.columns)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.y_pred = self.model.predict(X)
        return self.y_pred

    def save_predictions(self):
        prediction_results_path = "prediction_results.csv"
        pd.Series(self.y_pred).to_csv(prediction_results_path)
        print(f"predictions are saved in {prediction_results_path}")
