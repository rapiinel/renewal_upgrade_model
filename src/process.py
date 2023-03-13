"""Python script to process the data"""

import glob
import re
import warnings
from datetime import date

import joblib
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from pandarallel import pandarallel
from prefect import flow, task
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from config import DataframeParams, Location, ProcessConfig

pandarallel.initialize(progress_bar=False)
warnings.filterwarnings("ignore")


@task
def get_raw_data(data_location: str):
    """Read raw data

    Parameters
    ----------
    data_location : str
        The location of the raw data
    """
    return pd.read_csv(data_location, encoding="latin-1")


@task
def drop_columns(data: pd.DataFrame, columns: list):
    """Drop unimportant columns

    Parameters
    ----------
    data : pd.DataFrame
        Data to process
    columns : list
        Columns to drop
    """
    return data.drop(columns=columns)


@task
def get_X_y(data: pd.DataFrame, label: str):
    """Get features and label

    Parameters
    ----------
    data : pd.DataFrame
        Data to process
    label : str
        Name of the label
    """
    X = data.drop(columns=label)
    y = data[label]
    return X, y


@task
def split_train_test(X: pd.DataFrame, y: pd.DataFrame, test_size: int):
    """_summary_

    Parameters
    ----------
    X : pd.DataFrame
        Features
    y : pd.DataFrame
        Target
    test_size : int
        Size of the test set
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


@task
def save_processed_data(data: dict, save_location: str):
    """Save processed data

    Parameters
    ----------
    data : dict
        Data to process
    save_location : str
        Where to save the data
    """
    joblib.dump(data, save_location)


def get_breakdown(data, column):
    """get percentage breakdown of columns"""
    data = pd.DataFrame(data[column].value_counts())
    data["% Total"] = data / data.sum()
    data["% Total"] = data.apply(
        lambda x: "{:.2%}".format(x["% Total"]), axis=1
    )
    return data


def get_mem_month(value):
    """get membership month"""
    value = str(value).lower()

    if "membership" in value:
        return int(value.split(" ")[0]) * 12
    elif "months" in value:
        return int(value.split(" ")[0])
    else:
        try:
            return int(value)
        except Exception:
            return "No int found"


def dropoff_checker(value):
    """dropoff checker"""
    stage = value["Stage"]
    value = (
        value[
            [
                "Case Review Set",
                "Case Review Completed",
                "Enrollment Meeting Set",
            ]
        ]
        .notna()
        .tolist()
    )
    if stage == "Deal":
        return "Deal"
    elif value == [True, False, False]:
        return "Case Review Set"
    elif value == [True, True, False]:
        return "Case Review Completed"
    elif (value == [True, True, True]) & (stage == "No Deal"):
        return "Enrollment Meeting Set"
    elif (value == [True, True, True]) & (stage == "Deal"):
        return "Deal"
    elif (value == [False, False, False]) & (stage == "No Deal"):
        return "Opportunity Created"
    elif (value == [False, False, False]) & (stage == "Deal"):
        return "Deal"
    elif stage == "Prospect or Lead":
        return "Opportunity Created"
    elif stage == "Enrollment Meeting Set":
        return "Enrollment Meeting Set"
    elif stage == "Case Review Completed":
        return "Case Review Completed"
    elif stage == "Case Review Set":
        return "Case Review Set"
    elif (stage == "No Deal") & (value[-1] is True):
        return "Enrollment Meeting Set"
    elif (stage == "No Deal") & (value[-2] is True):
        return "Case Review Completed"
    elif (stage == "No Deal") & (value[-3] is True):
        return "Case Review Set"
    else:
        return "Please Check"


def generate_dataset(df, column_id):
    """generating the dataset"""
    temp_list = []
    for id in df[column_id].unique():
        temp_df = df[df[column_id] == id].sort_values(
            by=["Account Activated - converted"], ascending=True
        )
        final = temp_df[["Salesforce Account Id"]].copy()
        final["start date"] = temp_df["Account Activated - converted"]
        final["end date"] = temp_df["Account Activated - converted"].shift(-1)
        final["Membership Length - bin"] = temp_df["Membership Length - bin"]
        final["Membership Length - converted"] = temp_df[
            "Membership Length - converted"
        ]
        final["target"] = temp_df["Membership Type"].shift(-1)
        temp_list.append(final)
    return pd.concat(temp_list).reset_index(drop=True)


def target_checker(row, date_threshold):
    """This is to generate the target variable"""
    if (str(row["target"]) == "None") & (
        row["end date"] <= date_threshold - relativedelta(months=3)
    ):
        return "expired, with 3 months grace period"
    elif (str(row["target"]) == "None") & (row["end date"] <= date_threshold):
        return "expired"
    elif (str(row["target"]) == "None") & (row["end date"] > date_threshold):
        return "active"
    else:
        return row["target"]


@task
def preprocess_deals(
    data: pd.DataFrame, DataFrameParams: DataframeParams = DataframeParams()
):
    """Preprocessing the raw data"""

    data.dropna(subset=[DataFrameParams.footer_drop], inplace=True)
    data["Account Activated - converted"] = pd.to_datetime(
        data["Account Activated"], errors="coerce"
    )

    # removing waitlist and deals that are created in the same date
    data["Deal Name"] = data.apply(
        lambda x: str(x["Deal Name"]).lower(), axis=1
    )
    to_remove_deals = data[
        (data["Deal Name"].str.contains("waitlist"))
        & (data["Account Activated"].isna())
    ]["Salesforce Deal Id"].tolist()
    data = (
        data[~data["Salesforce Deal Id"].isin(to_remove_deals)]
        .reset_index(drop=True)
        .drop(columns=["Deal Name", "Created Date"])
    )

    # dropping not needed columns
    data.drop(
        columns=[
            "Auto-Renewal Opt Out Window Start",
            "Auto Renewal Opt Out Received",
            "Auto-Renewal Date",
            "Auto-Renewal Opt Out Window End",
        ],
        inplace=True,
    )

    # filter
    data["missing %"] = data.loc[:, :"Area of Practice 1"].isna().mean(axis=1)
    data = data[data["missing %"] <= 0.15]
    data.reset_index(drop=True, inplace=True)
    data["Membership Length"] = data.apply(
        lambda x: x["Total # of Installments"]
        if x["Membership Length"]
        in ["Other (see Description)", "Other (see Deal Notes)", "no value"]
        else x["Membership Length"],
        axis=1,
    )
    data.loc[:, :"Membership Length"] = data.loc[
        :, :"Membership Length"
    ].fillna("no value")
    data["Membership Length - converted"] = data.apply(
        lambda x: get_mem_month(x["Membership Length"]), axis=1
    )
    data = data.loc[data["Membership Length - converted"] != "No int found"]
    data.dropna(subset=["Membership Length - converted"], inplace=True)
    data.reset_index(drop=True, inplace=True)
    data["Membership Length - converted"] = data[
        "Membership Length - converted"
    ].astype("int")
    data["Membership Length - bin"] = pd.qcut(
        data["Membership Length - converted"], q=4
    )
    data["Practice Area Count"] = data.apply(
        lambda x: x[
            [
                "Area of Practice 1",
                "Area of Practice 2",
                "Area of Practice 3",
                "Area of Practice 4",
                "Area of Practice 5",
            ]
        ].count(),
        axis=1,
    )
    data["Practice Area Type"] = data.apply(
        lambda x: "Multiple" if x["Practice Area Count"] > 1 else "single",
        axis=1,
    )
    data = generate_dataset(data, "Salesforce Account Id")
    data.dropna(subset=["start date"], inplace=True)
    data["end date"] = data.apply(
        lambda x: x["start date"]
        + relativedelta(months=x["Membership Length - converted"])
        if str(x["end date"]) == "NaT"
        else x["end date"],
        axis=1,
    )
    date_today = pd.to_datetime(date.today())
    data["target"] = data.apply(
        lambda x: target_checker(x, date_today), axis=1
    )
    data = data[data["target"] != "no value"]
    control_list = [
        "expired, with 3 months grace period",
        "Renewal",
        "expired",
        "active",
    ]
    data["target"] = data.apply(
        lambda x: x["target"] if x["target"] in control_list else "Renewal",
        axis=1,
    )

    return data


@task
def oppty_preprocess(
    data: pd.DataFrame, DataFrameParams: DataframeParams = DataframeParams()
):
    """processing opportunity stats"""
    data.dropna(subset=["Salesforce Contact Id"], inplace=True)
    data["missing %"] = (
        data.loc[:, :"Number of Opportunities"].isna().mean(axis=1)
    )

    date_col_list = [
        "Created Date",
        "Case Review Set",
        "Case Review Completed",
        "Enrollment Meeting Set",
        "Order Date",
        "Opportunity Disposition Date",
        "Enrollment Meeting Completed",
        "Enrollment Meeting Date and Time",
    ]

    for date2 in date_col_list:
        data[date2] = pd.to_datetime(data[date2], errors="coerce")

    data = data.assign(
        year=data["Created Date"].dt.year,
        quarter=data["Created Date"].dt.quarter,
        month=data["Created Date"].dt.month,
        quarter_period_type=data["Created Date"].dt.to_period("Q"),
    )

    data = data[data["Stage"].isin(["Deal", "No Deal"])]
    data["Closed Oppty"] = data.apply(
        lambda x: "Closed"
        if (x["Stage"] == "Deal") | (x["Stage"] == "No Deal")
        else "Ongoing",
        axis=1,
    )
    data["Dropout Stage"] = data.apply(lambda x: dropoff_checker(x), axis=1)
    data["Lead Source"].fillna("SELF GEN", inplace=True)

    return data


@task
def get_OneHotEncoder(data: pd.DataFrame):
    """one hot encoder"""
    ohe = preprocessing.OneHotEncoder()
    values = ohe.fit_transform(data[["Lead Source"]])
    data[ohe.categories_[0]] = values.toarray()

    ohe_dropout_stage = preprocessing.OneHotEncoder()
    values_dropout_stage = ohe_dropout_stage.fit_transform(
        data[["Dropout Stage"]]
    )
    data[ohe_dropout_stage.categories_[0]] = values_dropout_stage.toarray()

    data["oppty_count"] = 1
    data.reset_index(drop=True, inplace=True)

    columns = (
        ohe_dropout_stage.categories_[0].tolist()
        + ohe.categories_[0].tolist()
        + ["oppty_count"]
    )
    data = data.pivot_table(
        index=["Salesforce Account Id", "Created Date"],
        values=columns,
        aggfunc="sum",
    ).reset_index(drop=False)

    return data


def data_checker(row):
    """checking inclusion dates"""
    if (row["start date"] <= row["Created Date"]) & (
        row["Created Date"] <= row["end date"]
    ):
        return "within range"
    elif row["start date"] > row["Created Date"]:
        return "below start date"
    elif row["end date"] < row["Created Date"]:
        return "higher than end date"
    else:
        return "Please check"


@task
def combine_deal_oppty(
    deal: pd.DataFrame,
    oppty: pd.DataFrame,
):
    """Combining deals and opportunity dataframe"""
    deal = deal.applymap(str)
    oppty = oppty.applymap(str)

    deal["start date"] = pd.to_datetime(deal["start date"])
    deal["end date"] = pd.to_datetime(deal["end date"])
    oppty["Created Date"] = pd.to_datetime(oppty["Created Date"])

    df = deal.merge(oppty, on="Salesforce Account Id", how="left")
    df = df.loc[:, ~df.columns.duplicated()].copy()

    for col in ["start date", "end date", "Created Date"]:
        df[col] = pd.to_datetime(df[col])
    df["col_checker"] = df.apply(lambda x: data_checker(x), axis=1)

    index = [
        "Salesforce Account Id",
        "start date",
        "end date",
        "Created Date",
        "Membership Length - bin",
        "Membership Length - converted",
        "target",
        "col_checker",
    ]
    values = [x for x in df.columns if x not in index]
    for value in values:
        df[value] = df[value].astype("float")

    df = df.pivot_table(index=index, values=values, aggfunc="sum").reset_index(
        drop=False
    )
    memLength_map = {
        "(-0.001, 12.0]": "Less than 1 year",
        "(12.0, 18.0]": "1-2 years",
        "(18.0, 36.0]": "2-3 years",
        "(36.0, 60.0]": "3-5 years",
    }

    df["Membership Length - bin"] = df["Membership Length - bin"].map(
        memLength_map
    )
    ohe_memLength = preprocessing.OneHotEncoder()

    values = ohe_memLength.fit_transform(df[["Membership Length - bin"]])
    df[ohe_memLength.categories_[0]] = values.toarray()

    col_list = df.loc[:, "AMRM Hand-off":].columns.tolist()

    df_clean = df.apply(lambda x: inclusion_checker(x, col_list), axis=1)
    index_col = [
        "Salesforce Account Id",
        "start date",
        "end date",
        "Membership Length - bin",
        "Membership Length - converted",
        "target",
    ]

    df_clean = df_clean.pivot_table(
        index=index_col,
        values=[x for x in df_clean.columns if x not in index_col],
        aggfunc="sum",
    ).reset_index(drop=False)
    df_clean.drop(
        columns=["Membership Length - bin", "Membership Length - converted"],
        inplace=True,
    )

    return df_clean


def inclusion_checker(row, col_list):
    """date inclusion checker"""
    if row["col_checker"] != "within range":
        for col in col_list:
            row[col] = np.nan
        return row
    else:
        return row


def data_checker_2(row):
    """data checker 2"""
    if (row["start date"] <= row["Matched Attorneys Matched Date"]) & (
        row["Matched Attorneys Matched Date"] <= row["end date"]
    ):
        return "within range"
    elif row["start date"] > row["Matched Attorneys Matched Date"]:
        return "below start date"
    elif row["end date"] < row["Matched Attorneys Matched Date"]:
        return "higher than end date"
    else:
        return "Please check"


def get_attorney_data(data_location: str, df: pd.DataFrame):
    """getting attorney data"""
    file_list = glob.glob(data_location)
    col_name_var = "Matched Attorney Settings Salesforce Account Id"
    temp_list = []
    for file in file_list:
        temp_df = pd.read_csv(file)
        col_var_dict = {col_name_var: "Salesforce Account Id"}
        temp_df.rename(
            columns=col_var_dict,
            inplace=True,
        )
        temp_df = df[
            ["Salesforce Account Id", "start date", "end date"]
        ].merge(temp_df, on="Salesforce Account Id", how="inner")
        temp_df["Matched Attorneys Matched Date"] = pd.to_datetime(
            temp_df["Matched Attorneys Matched Date"]
        )
        temp_df["col_checker"] = temp_df.apply(
            lambda x: data_checker_2(x), axis=1
        )
        temp_list.append(temp_df)
    df_atty = pd.concat(temp_list)
    df_atty.reset_index(drop=True, inplace=True)
    df_atty = df_atty[df_atty["col_checker"] == "within range"]
    df_atty.drop(
        columns=[
            "Matched Attorneys Matched Date",
            "Matched Attorneys Login Name",
            "col_checker",
        ],
        inplace=True,
    )
    index = ["Salesforce Account Id", "start date", "end date"]
    function = {}
    index_col = [
        "Salesforce Account Id",
        "start date",
        "end date",
        "Membership Length - bin",
        "Membership Length - converted",
        "target",
    ]
    for col in [x for x in df_atty.columns if x not in index_col]:
        if "avg" not in str(col).lower():
            function[col] = "sum"
        else:
            function[col] = "mean"

    for col in df_atty.loc[:, "Case Matched Count":].columns:
        df_atty[col] = df_atty[col].apply(
            lambda x: re.sub(r"[^\w]", "", str(x))
            if str(x).lower() != "nan"
            else 0
        )
        df_atty[col] = df_atty[col].astype("int")
    df_atty = df_atty.pivot_table(
        index=index,
        values=[x for x in df_atty.columns if x not in index],
        aggfunc=function,
    ).reset_index(drop=False)
    final_df = df.merge(df_atty, how="left").fillna(0)

    return final_df


@flow
def process(
    location: Location = Location(),
    config: ProcessConfig = ProcessConfig(),
):
    """Flow to process the ata

    Parameters
    ----------
    location : Location, optional
        Locations of inputs and outputs, by default Location()
    config : ProcessConfig, optional
        Configurations for processing data, by default ProcessConfig()
    """
    deal_data = get_raw_data(location.data_sf_export_deal)
    deal_processed = preprocess_deals(deal_data)

    oppty_data = get_raw_data(location.data_sf_export_oppty)
    oppty_processed = oppty_preprocess(oppty_data)
    oppty_encoded = get_OneHotEncoder(oppty_processed)

    combined_df = combine_deal_oppty(deal_processed, oppty_encoded)
    combined_df.to_csv(
        location.data_process + "combined_deal_oppty.csv", index=False
    )

    # processing attorney stats
    df_atty = get_attorney_data(
        location.data_raw + "/looker/*.csv", combined_df
    )
    df_atty.to_csv(location.data_process + "dataset.csv", index=False)

    # processed = drop_columns(data, config.drop_columns)
    # X, y = get_X_y(processed, config.label)
    # split_data = split_train_test(X, y, config.test_size)
    # save_processed_data(split_data, location.data_process)


if __name__ == "__main__":
    process(config=ProcessConfig(test_size=0.1))
