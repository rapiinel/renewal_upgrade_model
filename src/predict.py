"""
Create predict flow
"""
import pickle

import joblib
import numpy as np
import pandas as pd
from prefect import flow, task

from config import DataframeParams, Location, ModelParams, ProcessConfig

# from process import process
# from run_notebook import run_notebook
# from train_model import train


@task
def load_model(model_location: str):
    """loading the trained model"""

    return joblib.load(model_location)


@task
def get_data(data_location: str, x_col: DataframeParams = DataframeParams()):
    """Loading actual dataset that needs predicting"""
    df = pd.read_csv(data_location)
    df_orig = df.reset_index(drop=True).copy()
    df = df[x_col.X_columns].copy()

    return df, df_orig


@flow
def main_flow(location: Location = Location()):
    """Flow to run the prediction procress

    Parameters
    ----------
    location : Location, optional
        Locations of inputs and outputs, by default Location()
    process_config : ProcessConfig, optional
        Configurations for processing data, by default ProcessConfig()
    model_params : ModelParams, optional
        Configurations for training models, by default ModelParams()
    """

    model = load_model(location.model)
    # df, df_orig = get_data(location.data_process_path + "actual_dataset.csv")
    df, df_orig = get_data(location.actual_dataset_for_prediction)
    print(df.head())
    # print(location.model)
    # print(type(model))
    # df = df.reshape(-1,1)
    result = model.predict(df)
    result_proba = model.predict_proba(df)
    result_logproba = model.predict_log_proba(df)

    print(result)
    print("===========================================")
    print(result_proba)
    print("===========================================")
    print(result_logproba)
    print("===========================================")

    df_orig["result"] = result
    df_orig["result"] = df_orig["result"].map(
        {0: "possible churn", 1: "Most likely to renew/upgrade"}
    )
    result_proba = pd.DataFrame(
        result_proba, columns=["Churn Likelihood", "Renew/Upgrade Likelihood"]
    )
    df_orig = pd.concat(
        [
            df_orig[
                [
                    "Salesforce Account Id",
                    "start date",
                    "end date",
                    "target",
                    "result",
                ]
            ],
            result_proba,
        ],
        axis=1,
    )

    # result_logproba = pd.DataFrame(
    #     result_logproba,
    #     columns=["result log proba_churn", "result log proba_renew"],
    # )
    # df_orig = pd.concat([df_orig, result_logproba], axis=1)
    # df['result proba_churn'], df['result proba_renew'] =
    # df['result log proba_churn'], df['result log proba_renew'] = result_logproba

    df_orig.rename(
        columns={
            "start date": "deal activated date",
            "end date": "deal expire date",
            "target": "current status",
            "result": "prediction",
        },
        inplace=True,
    )
    df_orig.to_csv(location.actual_predicted, index=False)
    # process(location, process_config)
    # train(location, model_params)
    # run_notebook(location)


if __name__ == "__main__":
    main_flow()
