"""Train, compare and export churn models.

Run from the project root:  python -m churn.train

Workflow (the held-out test set is touched only for final reporting):
  1. Stratified 80/20 train/test split.
  2. For each candidate: optional hyperparameter search, then 5-fold stratified
     out-of-fold (OOF) predictions on the training set.
  3. Per model, choose the F1-maximising decision threshold on its OOF predictions.
  4. Select the model with the best CV PR-AUC, refit on the full training set,
     report test metrics and save the pipeline + threshold with joblib.

Class imbalance (26.5% churn) is handled by stratification, imbalance-aware
metrics (PR-AUC for tuning/selection) and threshold tuning, *not* class weights:
weighting gave no ranking gain here but inflated predicted probabilities, and
the API returns the probability. A class-weighted logistic regression is kept as
a reported-only comparison row to document that.
"""

import json
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import sklearn
import xgboost
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churn import config
from churn.data import load_raw, split_features_target
from churn.features import build_preprocessor

logger = logging.getLogger(__name__)

SEARCH_ITERATIONS = 40


def make_pipeline(model) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


def candidate_models() -> dict[str, dict]:
    """Each entry: pipeline, optional search space, and whether it may be shipped."""
    rs = config.RANDOM_STATE
    return {
        "logistic_regression": {
            "pipeline": make_pipeline(LogisticRegression(max_iter=2000)),
        },
        "logistic_regression_balanced": {
            "pipeline": make_pipeline(LogisticRegression(max_iter=2000, class_weight="balanced")),
            "selectable": False,  # comparison only: shows the cost of class weights
        },
        "random_forest": {
            "pipeline": make_pipeline(
                RandomForestClassifier(
                    n_estimators=400, min_samples_leaf=5, n_jobs=-1, random_state=rs
                )
            ),
        },
        "xgboost_tuned": {
            "pipeline": make_pipeline(
                XGBClassifier(tree_method="hist", eval_metric="logloss", n_jobs=1, random_state=rs)
            ),
            "search_space": {
                "model__n_estimators": randint(100, 600),
                "model__learning_rate": loguniform(0.01, 0.2),
                "model__max_depth": randint(2, 6),
                "model__min_child_weight": randint(1, 10),
                "model__subsample": uniform(0.6, 0.4),
                "model__colsample_bytree": uniform(0.5, 0.5),
                "model__reg_lambda": loguniform(0.1, 10),
            },
        },
    }


def tune(pipeline: Pipeline, search_space: dict, X, y, cv) -> tuple[Pipeline, dict]:
    search = RandomizedSearchCV(
        pipeline,
        search_space,
        n_iter=SEARCH_ITERATIONS,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
    )
    search.fit(X, y)
    best_params = {k.removeprefix("model__"): _to_builtin(v) for k, v in search.best_params_.items()}
    logger.info("Best params: %s (CV PR-AUC %.4f)", best_params, search.best_score_)
    return clone(search.best_estimator_), best_params


def out_of_fold(pipeline: Pipeline, X, y, cv) -> tuple[np.ndarray, list[dict]]:
    """OOF probabilities plus per-fold ROC-AUC / PR-AUC."""
    oof = np.zeros(len(y))
    folds = []
    for train_idx, val_idx in cv.split(X, y):
        model = clone(pipeline).fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
        y_val = y.iloc[val_idx]
        folds.append(
            {
                "roc_auc": roc_auc_score(y_val, oof[val_idx]),
                "pr_auc": average_precision_score(y_val, oof[val_idx]),
            }
        )
    return oof, folds


def best_f1_threshold(y, proba) -> float:
    precision, recall, thresholds = precision_recall_curve(y, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    # The last precision/recall pair has no threshold.
    return float(thresholds[np.argmax(f1[:-1])])


def evaluate(y, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "f1": f1_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "accuracy": accuracy_score(y, pred),
        "brier": brier_score_loss(y, proba),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _to_builtin(value):
    return value.item() if isinstance(value, np.generic) else value


def _round(metrics: dict) -> dict:
    return {
        k: (_round(v) if isinstance(v, dict) else round(float(v), 4) if isinstance(v, float) else v)
        for k, v in metrics.items()
    }


def summary_table(results: dict, selected: str) -> str:
    lines = [
        "| Model | CV ROC-AUC | CV PR-AUC | Test ROC-AUC | Test PR-AUC | Test F1 | Test Precision | Test Recall | Test Brier | Threshold |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        cv, test = r["cv"], r["test"]
        label = f"**{name}** (selected)" if name == selected else name
        lines.append(
            f"| {label} | {cv['roc_auc_mean']:.3f} ± {cv['roc_auc_std']:.3f} "
            f"| {cv['pr_auc_mean']:.3f} ± {cv['pr_auc_std']:.3f} "
            f"| {test['roc_auc']:.3f} | {test['pr_auc']:.3f} | {test['f1']:.3f} "
            f"| {test['precision']:.3f} | {test['recall']:.3f} | {test['brier']:.3f} "
            f"| {r['threshold']:.3f} |"
        )
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    X, y = split_features_target(load_raw())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    logger.info("Train %d rows, test %d rows, churn rate %.3f", len(X_train), len(X_test), y_train.mean())
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    results, fitted = {}, {}
    for name, spec in candidate_models().items():
        logger.info("=== %s ===", name)
        pipeline, best_params = spec["pipeline"], None
        if "search_space" in spec:
            pipeline, best_params = tune(pipeline, spec["search_space"], X_train, y_train, cv)

        oof, folds = out_of_fold(pipeline, X_train, y_train, cv)
        threshold = best_f1_threshold(y_train, oof)
        fitted[name] = pipeline.fit(X_train, y_train)
        test_proba = fitted[name].predict_proba(X_test)[:, 1]

        results[name] = _round(
            {
                "selectable": spec.get("selectable", True),
                "best_params": best_params,
                "threshold": threshold,
                "cv": {
                    "roc_auc_mean": np.mean([f["roc_auc"] for f in folds]),
                    "roc_auc_std": np.std([f["roc_auc"] for f in folds]),
                    "pr_auc_mean": np.mean([f["pr_auc"] for f in folds]),
                    "pr_auc_std": np.std([f["pr_auc"] for f in folds]),
                    **{f"oof_{k}": v for k, v in evaluate(y_train, oof, threshold).items()
                       if k in ("f1", "precision", "recall", "brier")},
                },
                "test": evaluate(y_test, test_proba, threshold),
                "test_at_0.5": evaluate(y_test, test_proba, 0.5),
            }
        )
        logger.info(
            "CV PR-AUC %.4f | test ROC-AUC %.4f, PR-AUC %.4f, F1 %.4f @ threshold %.3f",
            results[name]["cv"]["pr_auc_mean"], results[name]["test"]["roc_auc"],
            results[name]["test"]["pr_auc"], results[name]["test"]["f1"], threshold,
        )

    # Selection uses CV only; the test set never influences which model ships.
    selected = max(
        (n for n, r in results.items() if r["selectable"]),
        key=lambda n: results[n]["cv"]["pr_auc_mean"],
    )
    trained_at = datetime.now(timezone.utc)
    artifact = {
        "pipeline": fitted[selected],
        "threshold": results[selected]["threshold"],
        "model_name": selected,
        "model_version": f"{selected}-{trained_at:%Y%m%dT%H%M%SZ}",
        "trained_at": trained_at.isoformat(),
        "features": config.FEATURES,
        "library_versions": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__},
    }

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, config.MODEL_PATH)
    report = {
        "selected_model": selected,
        "model_version": artifact["model_version"],
        "selection_metric": "cv_pr_auc_mean",
        "split": {"train_rows": len(X_train), "test_rows": len(X_test), "cv_folds": config.CV_FOLDS},
        "models": results,
    }
    config.METRICS_PATH.write_text(json.dumps(report, indent=2))

    logger.info("Selected %s, saved to %s", selected, config.MODEL_PATH)
    print("\n" + summary_table(results, selected))


if __name__ == "__main__":
    main()
