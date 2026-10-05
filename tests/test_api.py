import joblib
import pandas as pd
import pytest

from churn import config


def test_root_redirects_to_docs(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_sample_payload(client, customer):
    response = client.post("/predict", json=customer)
    assert response.status_code == 200
    body = response.json()

    assert body["customerID"] == customer["customerID"]
    assert 0 <= body["churn_probability"] <= 1
    assert body["churn_prediction"] == int(body["churn_probability"] >= body["threshold"])
    assert body["churn_label"] == ("Yes" if body["churn_prediction"] else "No")


def test_api_matches_direct_pipeline(client, customer):
    """Train/serve parity: the API returns what the saved pipeline predicts."""
    artifact = joblib.load(config.MODEL_PATH)
    expected = artifact["pipeline"].predict_proba(pd.DataFrame([customer])[config.FEATURES])[0, 1]
    body = client.post("/predict", json=customer).json()
    assert body["churn_probability"] == pytest.approx(expected, abs=1e-4)


def test_total_charges_string_and_number_agree(client, customer):
    as_string = client.post("/predict", json=customer).json()
    as_number = client.post("/predict", json={**customer, "TotalCharges": 29.85}).json()
    assert as_string["churn_probability"] == as_number["churn_probability"]


@pytest.mark.parametrize("total_charges", [" ", "", None])
def test_blank_total_charges_for_new_customer(client, customer, total_charges):
    payload = {**customer, "tenure": 0, "TotalCharges": total_charges}
    assert client.post("/predict", json=payload).status_code == 200


def test_customer_id_is_optional(client, customer):
    customer.pop("customerID")
    response = client.post("/predict", json=customer)
    assert response.status_code == 200
    assert response.json()["customerID"] is None


@pytest.mark.parametrize(
    "field, value",
    [
        ("Contract", "Weekly"),
        ("gender", "female"),
        ("SeniorCitizen", 2),
        ("tenure", -1),
        ("MonthlyCharges", -5),
        ("TotalCharges", "abc"),
    ],
)
def test_invalid_values_rejected(client, customer, field, value):
    response = client.post("/predict", json={**customer, field: value})
    assert response.status_code == 422


def test_missing_required_field_rejected(client, customer):
    customer.pop("Contract")
    assert client.post("/predict", json=customer).status_code == 422
