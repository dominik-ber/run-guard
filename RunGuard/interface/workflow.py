import os
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
from prefect import task, flow
import pandas as pd
from RunGuard.ml_logic.train_test_data import train_test_data_prep




@task
def preprocess_new_data():

    df = pd.read_csv("raw_data/day_approach_maskedID_timeseries.csv")
    X_train, y_train, X_test, y_test = train_test_data_prep(df)
    return X_train, y_train, X_test, y_test



@task
def re_train(min_date: str, max_date: str, split_ratio: str):
    return train(min_date=min_date, max_date=max_date, split_ratio=split_ratio)




@flow(name="RunGuard Training Flow")
def train_flow():
    """
    Build the Prefect workflow for the `taxifare` package. It should:
        - preprocess 1 month of new data, starting from EVALUATION_START_DATE
        - compute `old_mae` by evaluating the current production model in this new month period
        - compute `new_mae` by re-training, then evaluating the current production model on this new month period
        - if the new one is better than the old one, replace the current production model with the new one
        - if neither model is good enough, send a notification!
    """

    #min_date = EVALUATION_START_DATE
    #max_date = str(datetime.strptime(min_date, "%Y-%m-%d") + relativedelta(months=1)).split()[0]

    preprocessed = preprocess_new_data()




if __name__ == "__main__":
    train_flow()
