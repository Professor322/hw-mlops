import git
import hydra
import mlflow
from hydra.core.config_store import ConfigStore

from hw_mlops import config
from hw_mlops import model_trainer as mt

cs = ConfigStore.instance()
cs.store(name="my-config", node=config.MyConfig)


@hydra.main(config_path="config", config_name="default_config", version_base="1.2")
def main(cfg: config.MyConfig) -> None:
    model_trainer = mt.ModelTrainer(
        train_data_path=cfg.path.training_path,
        optimizer=cfg.params.optimizer,
        model_save_path=cfg.path.model_save_path,
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
        mlflow.set_tag("Training Info", "Basic MLP model for california housing")

        X, y = model_trainer.get_train_data()
        X_scaled = model_trainer.scale_features(X)
        model_trainer.train_model(X_scaled, y)
        mlflow.log_metric("r2_score_train", model_trainer.score_)
        # Log the loss metric
        for loss in model_trainer.model.loss_curve_:
            mlflow.log_metric("loss", loss)
        model_trainer.save_model()


if __name__ == "__main__":
    main()
