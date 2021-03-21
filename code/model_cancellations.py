import pandas as pd
import numpy as np
import datetime

from features import X1_cxl_cols, X2_cxl_cols
from xgboost import XGBClassifier

from model_tools import (
    get_preds,
    get_fbeta_score,
    optimize_prob_threshold,
    make_confusion_matrix,
)

DATE_FMT = "%Y-%m-%d"


def get_future_res(df_res, as_of_date):
    otb_mask = (df_res.ResMadeDate <= as_of_date) & (  # reservations made before AOD
        df_res.CheckoutDate > as_of_date
    )  # checking out after AOD

    return df_res.loc[otb_mask].copy()


def get_otb_res(df_res, as_of_date):
    """
    Takes a list of reservations and returns all reservations on-the-books (OTB) as of a given date (df_otb).

    _____
    Parameters:
        - df_res (required): cleaned reservations DataFrame
        - as_of_date (required, str): Date of simulation.
            - e.g. "2017-08-15"
    """
    otb_mask = (
        (df_res.ResMadeDate <= as_of_date)  # reservations made before AOD
        & (df_res.CheckoutDate > as_of_date)  # checking out after AOD
    ) & (
        (df_res.IsCanceled == 0)
        | (
            (  # only include cxls that have not been canceled yet
                (df_res.IsCanceled == 1) & (df_res.ReservationStatusDate >= as_of_date)
            )
        )
    )

    return df_res[otb_mask].copy()


def split_reservations(df_res, as_of_date, features, verbose=1):
    """
    Performs train/test split.

    Training set will contain all reservations made prior to as_of_date that have not yet checked out of the hotel.
    _______
    Parameters:
        - df_res (required): cleaned Reservations DataFrame
        - as_of_date (required, str, DATE_FMT): Date that represents 'today' for Rev Management simulation
        - features (required, list): Hotel-specific, imported from features.py
        - verbose(optional, int): if verbose==0, no output verbiage will be printed to the screen.
        - stay_date: the night for which we're predicting cancels
    """
    as_of_dt = pd.to_datetime(as_of_date, format=DATE_FMT)
    df_res["DaysUntilArrival"] = (df_res.ArrivalDate - as_of_dt).dt.days
    # train: all reservations that have already checked out
    # test: all OTB reservations
    train_mask = df_res["CheckoutDate"] <= as_of_date
    df_train = df_res[train_mask].copy()

    df_test = get_otb_res(df_res, as_of_date)

    X_train = df_train[features].copy()
    X_test = df_test[features].copy()

    y_train = df_train["IsCanceled"].copy()
    y_test = df_test["IsCanceled"].copy()

    if verbose > 1:
        print(
            f"Split complete.\nTraining sample size: {len(X_train)}\nTesting sample Size: {len(X_test)}\n\n"
        )
    return X_train, X_test, y_train, y_test


def model_cancellations(df_res, as_of_date, hotel_num, verbose=1):
    """
    Performs train/test split using split_reservations func,
    Prepares & fits the XGBClassifier model on the training data.
    Model hyperparameters are based on grid search results (hotel-specific).

    Returns X_test, y_test, & trained model.

    Parameters (all required):
        - df_res (pandas.DataFrame)
        - as_of_date (str, i.e. "2017-08-24")
        - hotel_num (int)
    """
    assert hotel_num in (1, 2), ValueError(
        "Invalid hotel_num (must be integer, 1 or 2)."
    )

    if hotel_num == 1:
        feature_cols = X1_cxl_cols
        param_md = 5
        param_ne = 475
        param_lr = 0.11

    if hotel_num == 2:
        feature_cols = X2_cxl_cols
        param_md = 5
        param_ne = 475
        param_lr = 0.11

    model = XGBClassifier(
        objective="binary:logistic",
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=36,
        learning_rate=param_lr,
        n_estimators=param_ne,
        max_depth=param_md,
    )

    X_train, X_test, y_train, y_test = split_reservations(
        df_res,
        as_of_date,
        feature_cols,
        verbose=verbose,
    )
    model.fit(X_train, y_train)

    return X_test, y_test, model


def predict_cancellations(df_res, as_of_date, hotel_num, confusion=True, verbose=1):
    """
    Generates cancellation predictions and returns future-looking reservations dataFrame.

    The resulting DataFrame contains all future (and in-house) reservations touching as_of_date and beyond.

    _____
    Parameters:
        - df_res (pd.DataFrame, required): cleaned reservations DataFrame
        - as_of_date (str DATE_FMT, required): date of simulation
        - hotel_num (int, required):
        - confusion (T/F, optional): whether or not a confusion matrix will be printed
    """

    X_test, y_test, model = model_cancellations(
        df_res,
        as_of_date,
        hotel_num,
        verbose=verbose,
    )
    # make predictions using above model, adjusting occ threshold to optimize F-0.5 score
    X_test_cxl_probas = model.predict_proba(X_test)

    thresh = optimize_prob_threshold(
        model, X_test=X_test, y_test=y_test, confusion=confusion, beta=1
    )

    X_test[["will_come_proba", "cxl_proba"]] = X_test_cxl_probas
    df_res["cxl_proba"] = X_test["cxl_proba"]
    df_res["will_cancel"] = df_res.cxl_proba >= thresh

    # df_otb = df_res.loc[list(X_test.index)].copy()  # PROBLEM HERE
    # df_res[["will_come_proba", "cxl_proba"]] = X_test_cxl_probas
    # df_otb.drop(columns="will_come_proba", inplace=True)
    # df_otb["will_cancel"] = df_otb.cxl_proba >= thresh
    # df_otb["IsCanceled"] = y_test.to
    # return df_otb
    return df_res