import git
import hydra
import mlflow
from sklearn.metrics import r2_score

from hw_mlops import config
from hw_mlops import inference_runner as ir


@hydra.main(config_path="config", config_name="default_config", version_base="1.2")
def main(cfg: config.MyConfig):
    inference_runner = ir.InferenceRunner(
        test_data_path=cfg.path.test_path,
        predictions_save_path=cfg.path.predictions_save_path,
        model_path=cfg.path.model_save_path,
    )
    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri=cfg.params.tracking_uri_mlflow)

    # Create a new MLflow Experiment
    mlflow.set_experiment("MLOps homework")
    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(cfg)
        repo = git.Repo()
        mlflow.log_param("commit-id", repo.head.object.hexsha)

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Inference Info", "Basic MLP model for california housing")

        X, y = inference_runner.get_test_data()
        X_scaled = inference_runner.scale_features(X)
        y_pred = inference_runner.predict(X_scaled)
        mlflow.log_metric("r2_score_test", r2_score(y, y_pred))
        inference_runner.save_predictions()


if __name__ == "__main__":
    main()
