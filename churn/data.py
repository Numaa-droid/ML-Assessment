"""Loading the raw dataset and separating features from the target."""

from pathlib import Path

import pandas as pd

from churn import config


def load_raw(path: Path = config.DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV and check that every expected column is present."""
    df = pd.read_csv(path)
    expected = set(config.FEATURES) | {config.ID_COLUMN, config.TARGET}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model features X (ID dropped) and a 0/1 target y.

    Feature cleaning (e.g. TotalCharges parsing) deliberately happens inside
    the sklearn pipeline, not here, so training and inference share one code path.
    """
    labels = df[config.TARGET].str.strip()
    unexpected = set(labels.unique()) - {"Yes", "No"}
    if unexpected:
        raise ValueError(f"Unexpected target values: {sorted(unexpected)}")

    y = (labels == config.POSITIVE_LABEL).astype(int).rename(config.TARGET)
    X = df[config.FEATURES].copy()
    return X, y
