"""
create Pydantic models
"""
import os
from pathlib import Path
from typing import List

from pydantic import BaseModel, validator


def must_be_non_negative(v: float) -> float:
    """Check if the v is non-negative

    Parameters
    ----------
    v : float
        value

    Returns
    -------
    float
        v

    Raises
    ------
    ValueError
        Raises error when v is negative
    """
    if v < 0:
        raise ValueError(f"{v} must be non-negative")
    return v


class Location(BaseModel):
    """Specify the locations of inputs and outputs"""

    # working_directory = os.getcwd()
    folder_link: str = str(Path(os.getcwd()).absolute()) + "/"
    actual_predicted = folder_link + "data/final/actual_predicted.csv"
    data_raw: str = folder_link + "data/raw/"
    data_process: str = folder_link + "data/processed/training_dataset.csv"
    data_process_path: str = "data/processed/"
    data_final: str = folder_link + "data/final/predictions.pkl"
    data_process_pkl: str = folder_link + "data/final/training_dataset.pkl"
    model: str = folder_link + "models/model.pkl"
    input_notebook: str = folder_link + "notebooks/analyze_results.ipynb"
    output_notebook: str = folder_link + "notebooks/results.ipynb"

    data_sf_export_deal: str = data_raw + "sf_export - deals.csv"
    data_sf_export_oppty: str = data_raw + "sf_export - opportunity.csv"
    column_reference: str = data_raw + "column_reference.xlsx"


class ProcessConfig(BaseModel):
    """Specify the parameters of the `process` flow"""

    drop_columns: List[str] = ["Id"]
    label: str = "target"
    test_size: float = 0.3

    _validated_test_size = validator("test_size", allow_reuse=True)(
        must_be_non_negative
    )


class ModelParams(BaseModel):
    """Specify the parameters of the `train` flow"""

    C: List[float] = [0.1, 1, 10, 100, 1000]
    # gamma: List[float] = [1, 0.1, 0.01, 0.001, 0.0001]
    penalty: List[str] = ["l1", "l2", "elasticnet"]
    solver: List[str] = [
        "lbfgs",
        "liblinear",
        "newton-cg",
        "newton-cholesky",
        "sag",
        "saga",
    ]

    # _validated_fields = validator("*", allow_reuse=True, each_item=True)(
    #     must_be_non_negative
    # )


class DataframeParams(BaseModel):
    """Specify the parameters of the dataframe preprocessing"""

    footer_drop: str = "Opportunity ID"
    categorical_columns: list[str] = [""]
    funnel_map: dict = {
        "Opportunity Created": 0,
        "Case Review Set": 1,
        "Case Review Completed": 2,
        "Enrollment Meeting Set": 3,
        "Deal": 4,
    }
    funnel_map_reverse = {val: key for (key, val) in funnel_map.items()}
    X_columns = [
        "1-2 years",
        "2-3 years",
        "3-5 years",
        "AMRM Hand-off",
        "APPT. SETTER",
        "AUTO RENEWAL",
        "CAMPAIGN",
        "Case Review Completed",
        "Case Review Set",
        "Closer Initiated",
        "Corp Email Lead",
        "Deal",
        "EARLY RENEWAL",
        "Enrollment Meeting Set",
        "FINANCING LEAD",
        "HOTLEAD-Calendly",
        "HOTLEAD-LMWebForm",
        "HOTLEAD-LinkedIn-LGF",
        "HOTLEAD-email",
        "HOTLEAD-other",
        "HOTLEAD-phone",
        "Hot Transfer",
        "MM REFERRAL",
        "Newsletter Lead",
        "Opportunity Created",
        "RENEWAL",
        "Refer a Colleague",
        "Revived Hotlead",
        "SAVE",
        "SELF GEN",
        "Social Media Engagement",
        "UPGRADE",
        "Case Matched Count",
        "Has Login",
        "Responses Avg Attys per Responded Case",
        "Responses Count of Cases with Engaged or Hired Response",
        "Responses Response Count",
        "Responses Responses with Atty Calendaring Usage",
    ]
