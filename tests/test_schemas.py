from typing import get_args

import pytest

from churn import config
from churn.data import load_raw
from churn.schemas import CustomerProfile


@pytest.mark.parametrize("column", config.CATEGORICAL_FEATURES + ["SeniorCitizen"])
def test_schema_categories_match_training_data(column):
    """The API must accept exactly the categories the model was trained on."""
    allowed = set(get_args(CustomerProfile.model_fields[column].annotation))
    seen = set(load_raw()[column].unique())
    assert allowed == seen
