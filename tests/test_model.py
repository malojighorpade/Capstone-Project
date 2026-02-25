# load test + signature test + performance test

import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle


class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # Get DagsHub token
        dagshub_token = os.getenv("CAPSTONE_TEST")
        if not dagshub_token:
            raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

        # Set MLflow auth
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "malojighorpade"
        repo_name = "Capstone-Project"

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(
            f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"
        )

        # Load latest model
        cls.new_model_name = "my_model"
        cls.new_model_version = cls.get_latest_model_version(
            cls.new_model_name
        )

        cls.new_model_uri = (
            f"models:/{cls.new_model_name}/{cls.new_model_version}"
        )

        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

        # Load SAME vectorizer used in training
        cls.vectorizer = pickle.load(
            open("models/vectorizer.pkl", "rb")
        )

        # Load holdout data
        cls.holdout_data = pd.read_csv(
            "data/processed/test_bow.csv"
        )

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):

        client = mlflow.MlflowClient()

        latest_version = client.get_latest_versions(
            model_name,
            stages=[stage]
        )

        return latest_version[0].version if latest_version else None

    # -------------------------
    # Test 1: Model Load
    # -------------------------
    def test_model_loaded_properly(self):

        self.assertIsNotNone(self.new_model)

    # -------------------------
    # Test 2: Model Signature
    # -------------------------
    def test_model_signature(self):

        input_text = "hi how are you"

        # Vectorize input
        input_data = self.vectorizer.transform([input_text])

        # Use REAL feature names (IMPORTANT FIX)
        input_df = pd.DataFrame(
            input_data.toarray(),
            columns=self.vectorizer.get_feature_names_out()
        )

        # Predict
        prediction = self.new_model.predict(input_df)

        # Check feature count
        self.assertEqual(
            input_df.shape[1],
            len(self.vectorizer.get_feature_names_out())
        )

        # Check output shape
        self.assertEqual(
            len(prediction),
            input_df.shape[0]
        )

        self.assertEqual(
            len(prediction.shape),
            1
        )

    # -------------------------
    # Test 3: Performance
    # -------------------------
    def test_model_performance(self):

        # Split features & labels
        X_holdout = self.holdout_data.iloc[:, :-1]
        y_holdout = self.holdout_data.iloc[:, -1]

        # Predict
        y_pred_new = self.new_model.predict(X_holdout)

        # Metrics
        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new)
        recall_new = recall_score(y_holdout, y_pred_new)
        f1_new = f1_score(y_holdout, y_pred_new)

        # Thresholds
        expected_accuracy = 0.40
        expected_precision = 0.40
        expected_recall = 0.40
        expected_f1 = 0.40

        # Assertions
        self.assertGreaterEqual(
            accuracy_new,
            expected_accuracy,
            f"Accuracy should be at least {expected_accuracy}"
        )

        self.assertGreaterEqual(
            precision_new,
            expected_precision,
            f"Precision should be at least {expected_precision}"
        )

        self.assertGreaterEqual(
            recall_new,
            expected_recall,
            f"Recall should be at least {expected_recall}"
        )

        self.assertGreaterEqual(
            f1_new,
            expected_f1,
            f"F1 score should be at least {expected_f1}"
        )


if __name__ == "__main__":
    unittest.main()