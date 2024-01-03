import hydra
from hydra.core.config_store import ConfigStore

from hw_mlops import config
from hw_mlops import model_trainer as mt

cs = ConfigStore.instance()
cs.store(name="my-config", node=config.MyConfig)


@hydra.main(config_path="config", config_name="default_config")
def main(cfg: config.MyConfig) -> None:
    model_trainer = mt.ModelTrainer(
        train_data_path=cfg.path.training_path,
        optimizer=cfg.params.optimizer,
        model_save_path=cfg.path.model_save_path,
    )
    X, y = model_trainer.get_train_data()
    X_scaled = model_trainer.scale_features(X)
    model_trainer.train_model(X_scaled, y)
    model_trainer.save_model()


if __name__ == "__main__":
    main()
