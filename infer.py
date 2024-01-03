from hw_mlops import inference_runner as ir


def main():
    inference_runner = ir.InferenceRunner()
    X, y = inference_runner.get_test_data()
    X_scaled = inference_runner.scale_features(X)
    inference_runner.predict(X_scaled)
    inference_runner.save_predictions()


if __name__ == "__main__":
    main()
