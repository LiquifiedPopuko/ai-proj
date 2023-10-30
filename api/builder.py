import joblib

from build_estimator import BuildEstimator
from build_estimator_regression import BuildRegressorEstimator
from sample_creator import SampleCreator

if __name__ == "__main__":
    SampleCreator.createBlindTestSamples()
    with open("./lib/model/model_nn.joblib", "wb") as f:
        pipeline = BuildEstimator.generateBestModel()
        pipeline.named_steps["model"].model_.save("./lib/model/model_nn.keras")
        pipeline.named_steps["model"].model_ = None
        joblib.dump(pipeline, f)
    with open("./lib/model/model_regression.joblib", "wb") as f:
        joblib.dump(BuildRegressorEstimator.generateBestModel(), f)
