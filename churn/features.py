"""Preprocessing that is fitted as part of the model pipeline.

Everything that learns from data (imputation, scaling, encoding) lives in this
ColumnTransformer, so it is fitted only on training folds (no leakage) and is
serialized together with the model (no train/serve skew).
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from churn import config


def clean_numeric(X: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric columns and fill blank TotalCharges.

    TotalCharges arrives as text ("29.85", or " " for brand-new customers).
    Blanks only occur at tenure 0, so the charge accrued so far is
    tenure * MonthlyCharges, i.e. 0; that is more faithful than a median fill.
    """
    X = X.apply(pd.to_numeric, errors="coerce")
    X["TotalCharges"] = X["TotalCharges"].fillna(X["tenure"] * X["MonthlyCharges"])
    return X


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("clean", FunctionTransformer(clean_numeric, feature_names_out="one-to-one")),
            # Safety net for any value still missing at inference time.
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, config.NUMERIC_FEATURES),
            ("cat", categorical, config.CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
