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
    df = df[x_col.X_columns].copy()

    return df


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
    df = get_data(location.data_process_path + "actual_dataset.csv")
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

    df["result"] = result
    result_proba = pd.DataFrame(
        result_proba, columns=["result proba_churn", "result proba_renew"]
    )
    df = pd.concat([df, result_proba], axis=1)
    result_logproba = pd.DataFrame(
        result_proba,
        columns=["result log proba_churn", "result log proba_renew"],
    )
    df = pd.concat([df, result_proba], axis=1)
    # df['result proba_churn'], df['result proba_renew'] =
    # df['result log proba_churn'], df['result log proba_renew'] = result_logproba

    df.to_csv(location.actual_predicted, index=False)
    # process(location, process_config)
    # train(location, model_params)
    # run_notebook(location)


if __name__ == "__main__":
    main_flow()
