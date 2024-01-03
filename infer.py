import hydra

from hw_mlops import config
from hw_mlops import inference_runner as ir


@hydra.main(config_path="config", config_name="default_config")
def main(cfg: config.MyConfig):
    inference_runner = ir.InferenceRunner(
        test_data_path=cfg.path.test_path,
        predictions_save_path=cfg.path.predictions_save_path,
        model_path=cfg.path.model_save_path,
    )
    X, _ = inference_runner.get_test_data()
    X_scaled = inference_runner.scale_features(X)
    inference_runner.predict(X_scaled)
    inference_runner.save_predictions()


if __name__ == "__main__":
    main()
