from hw_mlops import model_trainer as mt


def main() -> None:
    model_trainer = mt.ModelTrainer()
    X, y = model_trainer.get_train_data()
    X_scaled = model_trainer.scale_features(X)
    model_trainer.train_model(X_scaled, y)
    model_trainer.save_model()


if __name__ == "__main__":
    main()
