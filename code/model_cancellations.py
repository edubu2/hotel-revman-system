import pandas as pd
import numpy as np

from features import X1_cxl_cols, X2_cxl_cols
from xgboost import XGBClassifier

from viz import (
    get_preds,
    get_fbeta_score,
    optimize_prob_threshold,
    make_confusion_matrix,
)


def get_otb_res(df_res, as_of_date):
    """
    Takes a list of reservations and returns all reservations on-the-books (OTB) as of a given date (df_otb).
    _____
    Parameters:
        - df_res (required): cleaned reservations DataFrame
        - as_of_date (required, str): Date of simulation.
            - e.g. "2017-08-15"
    """
    # assert type(df_res) == "pandas.core.frame.DataFrame", "df_res must be a DataFrame."
    as_of_dt = pd.to_datetime(as_of_date)
    otb_mask = (
        (df_res.ResMadeDate <= as_of_date)  # reservations made before AOD
        & (df_res.ArrivalDate <= as_of_date)  # arriving before/on AOD
        & (df_res.CheckoutDate > as_of_date)  # checking out after AOD
    ) & (
        (
            (df_res.ReservationStatus != "Canceled")
            | (
                (  # only include cxls that have not been canceled yet
                    (df_res.IsCanceled == 1)
                    & (df_res.ReservationStatusDate >= as_of_date)
                )
            )
        )
    )

    return df_res[otb_mask]


def split_reservations(df_res, as_of_date, features, print_len):
    """
    Performs train/test split.

    Training set will contain all reservations made prior to as_of_date that have not yet checked out of the hotel.
    _______
    Parameters:
        - df_res (required): cleaned Reservations DataFrame
        - as_of_date (required, str, "%Y-%m-%d"): Date that represents 'today' for Rev Management simulation
        - features (required, list): Hotel-specific, imported from features.py
        - print_len(optional, bool, default=False): Whether or not to print the length of the resulting matrices.
    """
    as_of_dt = pd.to_datetime(as_of_date, format="%Y-%m-%d")
    df_res["DaysUntilArrival"] = (df_res.ArrivalDate - as_of_dt).dt.days

    # train: all reservations that have already checked out
    # test: all OTB reservations
    train_mask = df_res["CheckoutDate"] < as_of_date
    df_train = df_res[train_mask]
    df_test = get_otb_res(df_res, as_of_date)

    X_train = df_train[features].copy()
    X_test = df_test[features].copy()

    y_train = df_train["IsCanceled"].copy()
    y_test = df_test["IsCanceled"].copy()

    if print_len:
        print(
            f"Split complete.\nTraining sample size: {len(X_train)}\nTesting sample Size: {len(X_test)}\n\n"
        )
    return X_train, X_test, y_train, y_test


def model_cancellations(df_res, as_of_date, hotel_num, print_len):
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
    assert hotel_num in (1, 2), "Invalid hotel_num (must be integer, 1 or 2)."

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
        n_jobs=12,
        learning_rate=param_lr,
        n_estimators=param_ne,
        max_depth=param_md,
    )

    X_train, X_test, y_train, y_test = split_reservations(
        df_res,
        as_of_date,
        feature_cols,
        print_len,
    )
    model.fit(X_train, y_train)

    return X_test, y_test, model


def predict_cancellations(
    df_res, as_of_date, hotel_num, print_len=False, confusion=True
):
    """
    Generates cancellation predictions and returns future-looking reservations dataFrame.

    The resulting DataFrame contains all future (and in-house) reservations touching as_of_date and beyond.

    _____
    Parameters:
        - df_res (pd.DataFrame, required): cleaned reservations DataFrame
        - as_of_date (str "%Y-%m-%d", required): date of simulation
        - hotel_num (int, required):
        - confusion (T/F, optional): whether or not a confusion matrix will be printed
    """

    X_test, y_test, model = model_cancellations(
        df_res, as_of_date, hotel_num, print_len=print_len
    )
    # make predictions using above model, adjusting occ threshold to optimize F-0.5 score
    X_test_preds = model.predict_proba(X_test)

    thresh = optimize_prob_threshold(
        model, X_test=X_test, y_test=y_test, confusion=confusion
    )
    df_future_res = df_res.loc[list(X_test.index)].copy()  # PROBLEM HERE
    df_future_res[["will_come_proba", "cxl_proba"]] = X_test_preds
    df_future_res.drop(columns="will_come_proba", inplace=True)
    df_future_res["will_cancel"] = df_future_res.cxl_proba >= thresh
    df_future_res["IsCanceled"] = y_test.to_numpy()
    return df_future_res