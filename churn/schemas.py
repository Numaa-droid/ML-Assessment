"""Request/response contracts for the inference API.

Categorical fields are restricted to the values seen in training, so malformed
input is rejected with a clear 422 instead of being silently one-hot encoded
as "unknown".
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

YesNo = Literal["Yes", "No"]
InternetAddon = Literal["Yes", "No", "No internet service"]

SAMPLE_CUSTOMER = {
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
    "TotalCharges": "29.85",
}


class CustomerProfile(BaseModel):
    """A single raw customer record, in the same shape as the source CSV."""

    model_config = ConfigDict(json_schema_extra={"examples": [SAMPLE_CUSTOMER]})

    customerID: str | None = Field(default=None, description="Echoed back; not used by the model")
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetAddon
    OnlineBackup: InternetAddon
    DeviceProtection: InternetAddon
    TechSupport: InternetAddon
    StreamingTV: InternetAddon
    StreamingMovies: InternetAddon
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float | None = Field(
        default=None,
        ge=0,
        description="Number or numeric string; blank/omitted is filled with tenure * MonthlyCharges",
    )

    @field_validator("TotalCharges", mode="before")
    @classmethod
    def parse_total_charges(cls, value):
        # The source data stores this as text, with " " for brand-new customers.
        if isinstance(value, str):
            value = value.strip()
            return None if value == "" else value
        return value


class PredictionResponse(BaseModel):
    customerID: str | None
    churn_prediction: Literal[0, 1]
    churn_label: YesNo
    churn_probability: float = Field(ge=0, le=1)
    threshold: float
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_name: str
    model_version: str
