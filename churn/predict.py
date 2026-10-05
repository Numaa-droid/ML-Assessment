"""Load the exported pipeline and score customer profiles.

Used by the API, and also runnable as a standalone script:
    python -m churn.predict path/to/customer.json
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from churn import config
from churn.schemas import CustomerProfile, PredictionResponse


class ChurnPredictor:
    def __init__(self, artifact: dict):
        self.pipeline = artifact["pipeline"]
        self.threshold = artifact["threshold"]
        self.features = artifact["features"]
        self.model_name = artifact["model_name"]
        self.model_version = artifact["model_version"]

    @classmethod
    def load(cls, path: Path = config.MODEL_PATH) -> "ChurnPredictor":
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Model artifact not found at {path}. Train it first with: python -m churn.train"
            )
        return cls(joblib.load(path))

    def predict(self, customer: CustomerProfile) -> PredictionResponse:
        row = pd.DataFrame([customer.model_dump()])[self.features]
        probability = float(self.pipeline.predict_proba(row)[0, 1])
        prediction = int(probability >= self.threshold)
        return PredictionResponse(
            customerID=customer.customerID,
            churn_prediction=prediction,
            churn_label="Yes" if prediction else "No",
            churn_probability=round(probability, 4),
            threshold=self.threshold,
            model_version=self.model_version,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict churn for one customer JSON profile.")
    parser.add_argument("payload", help="Path to a JSON file, or '-' to read from stdin")
    parser.add_argument("--model", default=config.MODEL_PATH, type=Path)
    args = parser.parse_args()

    raw = sys.stdin.read() if args.payload == "-" else Path(args.payload).read_text()
    customer = CustomerProfile.model_validate(json.loads(raw))
    result = ChurnPredictor.load(args.model).predict(customer)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
