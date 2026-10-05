import pytest
from fastapi.testclient import TestClient

from churn.api import app
from churn.schemas import SAMPLE_CUSTOMER


@pytest.fixture(scope="session")
def client():
    # Context manager runs the lifespan, i.e. loads the model artifact.
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def customer() -> dict:
    return dict(SAMPLE_CUSTOMER)
