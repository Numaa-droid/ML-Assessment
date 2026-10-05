# Telco Customer Churn: Training Pipeline & Inference API

An end-to-end churn model for the [Kaggle Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn): EDA, a leakage-safe scikit-learn pipeline, cross-validated model comparison with hyperparameter tuning, and a FastAPI service that scores a single raw customer profile.

**Result:** a tuned XGBoost pipeline with **test ROC-AUC 0.849, PR-AUC 0.664, and F1 0.638**. At its tuned threshold it catches **74% of churners**, against 53% at the default 0.5 threshold.

---

## Contents

1. [Quick start](#1-quick-start)
2. [Project structure](#2-project-structure)
3. [Approach](#3-approach)
4. [Results](#4-results)
5. [Why accuracy is insufficient](#5-why-accuracy-is-insufficient)
6. [API reference](#6-api-reference)
7. [Limitations and trade-offs](#7-limitations-and-trade-offs)
8. [With two more days: top 3 improvements](#8-with-two-more-days-top-3-improvements)

---

## 1. Quick start

**Requirements:** Python 3.11. Run every command from the project root.

### Set up the environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

- `requirements.txt` pins only what training and serving need.
- `requirements-dev.txt` adds the notebook, plotting and test tools.

### Run the inference API

A trained model is already committed at `models/churn_pipeline.joblib`, so the API runs without training first.

```bash
uvicorn churn.api:app --port 8000
```

In a second terminal:

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
  "customerID": "7590-VHVEG",
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "No",
  "MultipleLines": "No phone service",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85,
  "TotalCharges": "29.85"
}'
```

```json
{
  "customerID": "7590-VHVEG",
  "churn_prediction": 1,
  "churn_label": "Yes",
  "churn_probability": 0.6607,
  "threshold": 0.3335,
  "model_version": "xgboost_tuned-20261005T104330Z"
}
```

Interactive Swagger docs, pre-filled with the sample payload, are at **http://localhost:8000/docs** (the root URL `/` redirects there).

### Standalone prediction script (no server)

```bash
# Reads one customer profile from stdin ('-') or from a JSON file path
echo '{"customerID": "7590-VHVEG", "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service", "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": "29.85"}' | python -m churn.predict -
```

### Docker

```bash
docker compose up --build          # API on http://localhost:8000; add -d to run in the background
docker compose down                # stop and remove the container
```

- If port 8000 is busy, run `API_PORT=8001 docker compose up --build`.
- Without Compose, the equivalent is `docker build -t churn-api .` followed by `docker run --rm -p 8000:8000 churn-api`.
- Compose adds a health check on `/health` and restarts the container if it crashes.

The image installs only the runtime dependencies and runs as a non-root user. It uses the CPU-only `xgboost-cpu` build of the same pinned version, which saves about 470 MB of unused NVIDIA libraries.

### Retrain, test, explore

```bash
python -m churn.train      # about 20s; rewrites models/churn_pipeline.joblib and models/metrics.json
pytest                     # 32 tests
jupyter lab notebooks/eda.ipynb
```

Training is deterministic: re-running it reproduces identical metrics, with fixed seeds throughout.

---

## 2. Project structure

```
├── churn/
│   ├── config.py        # paths, column groups, seed, split/CV settings
│   ├── data.py          # load the CSV (with column checks), split X / y
│   ├── features.py      # preprocessing ColumnTransformer (part of the saved model)
│   ├── train.py         # CV, tuning, threshold selection, model selection, export
│   ├── schemas.py       # Pydantic request/response contracts
│   ├── predict.py       # ChurnPredictor + standalone CLI
│   └── api.py           # FastAPI app
├── data/                # raw Kaggle CSV
├── models/
│   ├── churn_pipeline.joblib   # pipeline + threshold + version metadata
│   └── metrics.json            # full CV/test metrics for every model
├── notebooks/eda.ipynb  # exploratory analysis (outputs saved)
├── tests/               # API, validation and schema/data-consistency tests
├── Dockerfile / docker-compose.yml
├── requirements.txt / requirements-dev.txt
└── pytest.ini
```

---

## 3. Approach

### EDA highlights ([`notebooks/eda.ipynb`](notebooks/eda.ipynb))

- **7,043 customers, 26.5% churn.** The imbalance is about 2.8 : 1.
- **`TotalCharges` is stored as text.** 11 values are a blank `" "`, and **all of them have `tenure = 0`**: new customers who have not been billed yet.
- **The strongest signals:**
  - Contract type: month-to-month customers churn at 43%, two-year customers at 3%.
  - Tenure: the median is 10 months for churners and 38 for retained customers.
  - Fiber optic internet: 42% churn.
  - Electronic check payment: 45% churn.
  - No tech support or online security: about 42% churn.
- **Gender carries no signal.**

### Preprocessing ([`churn/features.py`](churn/features.py))

All preprocessing is a `ColumnTransformer` **inside the model `Pipeline`**, so it is fitted only on training folds and saved together with the model.

| Column type | Steps |
|---|---|
| Numeric (`tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`) | Coerce to numeric → fill blank `TotalCharges` with `tenure × MonthlyCharges` (= 0 for these new customers) → median imputer as a safety net → `StandardScaler` |
| Categorical (15 columns, ≤ 4 levels each) | Most-frequent imputer → `OneHotEncoder(handle_unknown="ignore")` |

- `customerID` is dropped.
- Because the `TotalCharges` cleaning lives in the pipeline, the API sends the raw payload straight through, so training and serving can never diverge.
- A median fill would have invented roughly $1,400 of billing history for customers who joined this month.

### Leakage prevention and validation ([`churn/train.py`](churn/train.py))

1. **Hold out the test set first.** A stratified 80/20 split is made first. The test set is used **only for the final reported numbers**. It plays no part in tuning, threshold choice or model selection.
2. **Fit preprocessing within folds.** Every learned step (imputation, scaling, encoding) is part of the pipeline, so it is re-fitted on each CV training fold.
3. **Use stratified folds.** Validation uses 5-fold `StratifiedKFold`.
4. **Choose the threshold out-of-fold.** Each model's decision threshold maximises F1 on its out-of-fold CV predictions, never on the test set.
5. **Select on CV only.** The shipped model is the one with the highest **mean CV PR-AUC**.

### Class imbalance

Imbalance is handled with:

- stratified splits and folds
- **PR-AUC** as the tuning and selection metric
- a **tuned decision threshold** (0.33 rather than 0.5)

I tested class weights and chose **not** to use them. A class-weighted logistic regression is kept in the results table as a reported-only row to show why:

- Weighting gave **no ranking gain**: CV PR-AUC was 0.660 weighted vs 0.662 unweighted.
- It **inflated the predicted probabilities**: the mean prediction rose from 0.27 to about 0.41 against a true rate of 0.265, and the Brier score worsened from 0.138 to 0.169.
- The API returns a churn probability, so keeping it honest matters. Threshold tuning delivers the same recall gain without distorting the probability.

### Models

| Model | Role | Configuration |
|---|---|---|
| Logistic Regression | Baseline | L2, C = 1.0 |
| Random Forest | Tree ensemble | 400 trees, `min_samples_leaf=5` |
| XGBoost | **Primary**, tuned | `RandomizedSearchCV`, 40 candidates × 5 stratified folds, scored on PR-AUC |

Best XGBoost parameters: `n_estimators=392, learning_rate=0.012, max_depth=4, min_child_weight=8, subsample=0.86, colsample_bytree=0.67, reg_lambda=2.88`.

---

## 4. Results

CV = mean ± std over 5 stratified folds of the training set (5,634 rows). Test = held-out set (1,409 rows, 374 churners). Precision, recall and F1 are measured at each model's tuned threshold. The full breakdown, including confusion matrices, is in [`models/metrics.json`](models/metrics.json).

| Model | CV ROC-AUC | CV PR-AUC | Test ROC-AUC | Test PR-AUC | Test F1 | Test Precision | Test Recall | Test Brier ↓ | Threshold |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.846 ± 0.013 | 0.662 ± 0.019 | 0.842 | 0.634 | 0.616 | 0.537 | 0.722 | 0.138 | 0.327 |
| Logistic Regression + class weights *(comparison only)* | 0.846 ± 0.012 | 0.660 ± 0.019 | 0.842 | 0.633 | 0.620 | 0.540 | 0.727 | 0.169 | 0.574 |
| Random Forest | 0.845 ± 0.009 | 0.656 ± 0.021 | 0.840 | 0.647 | 0.628 | 0.517 | **0.799** | 0.137 | 0.271 |
| **XGBoost, tuned (shipped)** | **0.851 ± 0.011** | **0.672 ± 0.020** | **0.849** | **0.664** | **0.638** | **0.562** | 0.738 | **0.135** | 0.334 |

- **XGBoost wins on CV and confirms it on the test set**, with the best ROC-AUC, PR-AUC, F1, precision and calibration (Brier).
- **The margin over the baseline is small**: about +0.01 PR-AUC, which is within one CV standard deviation. On this dataset the signal is largely linear and additive, and a logistic regression is a strong, more interpretable alternative.
- **Random Forest has the highest recall at its own threshold**, at the cost of precision. Moving along that trade-off is a business choice (see §8).

---

## 5. Why accuracy is insufficient

With 73.5% of customers not churning, **a model that always predicts "No" scores 73.5% accuracy while catching zero churners.** Accuracy rewards the majority class and says nothing about the outcome that matters: finding the customers who are about to leave.

The shipped model on the test set shows this directly:

| Threshold | Accuracy | Churners caught (recall) | Precision | F1 |
|---|---|---|---|---|
| Default 0.5 | **80.9%** | 197 / 374 (53%) | 0.68 | 0.594 |
| Tuned 0.334 | 77.8% | **276 / 374 (74%)** | 0.56 | **0.638** |

Optimising for accuracy would pick the 0.5 threshold and **miss 79 more customers who churn**. For that reason:

- the models are compared on **ROC-AUC**, which is threshold-free ranking quality;
- they are tuned and selected on **PR-AUC**, which focuses on the minority class and is sensitive to imbalance;
- and decisions are reported with **precision, recall and F1** at the operating threshold.

---

## 6. API reference

### `POST /predict`

Accepts one raw customer profile, in the same shape as a CSV row without `Churn`.

| Field | Accepted values |
|---|---|
| `customerID` | optional string, echoed back |
| `gender` | `Female`, `Male` |
| `SeniorCitizen` | `0`, `1` |
| `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling` | `Yes`, `No` |
| `tenure` | integer ≥ 0 |
| `MultipleLines` | `Yes`, `No`, `No phone service` |
| `InternetService` | `DSL`, `Fiber optic`, `No` |
| `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | `Yes`, `No`, `No internet service` |
| `Contract` | `Month-to-month`, `One year`, `Two year` |
| `PaymentMethod` | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` |
| `MonthlyCharges` | number ≥ 0 |
| `TotalCharges` | number or numeric string ≥ 0; blank or omitted is filled with `tenure × MonthlyCharges` |

**Response fields:**
- `churn_prediction`: `0` or `1`
- `churn_label`: `"No"` or `"Yes"`
- `churn_probability`
- `threshold`: the decision threshold the prediction used
- `model_version`

**Invalid input** (an unknown category, a negative number, a non-numeric `TotalCharges`, a missing field) returns **HTTP 422** with a message naming the field and the allowed values. A test checks that the allowed categories match the training data exactly.

### `GET /health`

Returns `{"status": "ok", "model_name": ..., "model_version": ...}`.

- The model is loaded **once at startup**, and the service fails fast if the artifact is missing.
- Set `CHURN_MODEL_PATH` to serve a different artifact.

---

## 7. Limitations and trade-offs

- **The CV score for the tuned XGBoost is slightly optimistic.** The same folds were used to choose its hyperparameters. Nested CV would remove this bias, but it seemed out of scope. The held-out test score (0.664 PR-AUC) confirms the ranking.
- **The shipped model is trained on the 80% training split**, not refitted on all data. That keeps the reported test metrics an exact description of the artifact. Refitting on 100% before a real release is a one-line change.
- **joblib artifacts are tied to library versions.** They are pinned in `requirements.txt` and also recorded inside the artifact (`library_versions`).
- **The threshold optimises F1**, which weighs false positives and false negatives equally. A real deployment should set it from the cost of a retention offer versus the value of a saved customer.

---

## 8. With two more days: top 3 improvements

1. **Experiment tracking and a model registry (MLflow).** Today, versions are a timestamp on a joblib file and metrics live in a JSON file. I would:
   - log every run's parameters, metrics, data hash and git commit;
   - register models with staged promotion (candidate → production), gated on beating the current production model on a fixed evaluation set;
   - have the API load by registry alias instead of a file path.

   This gives full lineage and one-command rollback.

2. **CI/CD with data validation.** I would add a GitHub Actions pipeline that:
   - runs linting and `pytest`;
   - runs a fast training smoke test;
   - builds and scans the Docker image;
   - deploys on merge.

   I would also add an explicit data contract (for example Pandera or Great Expectations) on the training data and incoming requests. It would check schema, ranges, null rates and the `TotalCharges` format, so upstream changes fail loudly at ingestion rather than silently degrading the model.

3. **Production observability and drift monitoring.** I would:
   - log every prediction (inputs, probability, model version, latency) in a structured way;
   - expose Prometheus metrics for latency and error rates;
   - track feature and prediction drift (for example PSI on `tenure`, `MonthlyCharges` and `Contract` mix);
   - join predictions with churn labels once they arrive, to track real precision and recall over time.

   Drift or a performance drop would trigger retraining through the pipeline above.

**Next in line:**
- a batch scoring endpoint
- probability calibration (isotonic) for retention budgeting
- a cost-based decision threshold
- per-prediction SHAP explanations, so the retention team knows *why* a customer is at risk
- ONNX export for a lighter, framework-independent serving image
