import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, confusion_matrix

from features import X1_cxl_cols, X2_cxl_cols
from xgboost import XGBClassifier


def split_reservations(df_res, as_of_date, features, y_col="IsCanceled"):
    """
    Performs train/test split.

    Training set will contain all reservations made prior to as_of_date that have not yet checked out of the hotel.
    _______
    Parameters:
        - df_res (required): cleaned Reservations DataFrame
        - as_of_date (required, str, "%Y-%m-%d"): Date that represents 'today' for Rev Management simulation
        - features (required, list): The column names of the X DataFrame
        - y_col (optional, string): column name of the target variable
    """
    as_of_dt = pd.to_datetime(as_of_date, format="%Y-%m-%d")
    test_mask = (df_res["ResMadeDate"] <= as_of_date) & (
        df_res["CheckoutDate"] > as_of_date
    )

    X_train = df_res[~test_mask][features].copy()
    X_test = df_res[test_mask][features].copy()

    y_train = df_res[~test_mask][y_col].copy()
    y_test = df_res[test_mask][y_col].copy()

    print(
        f"Training sample size: {len(X_train)}\nTesting sample Size: {len(X_test)}\n\n"
    )

    return X_train, X_test, y_train, y_test


def make_confusion_matrix(
    y_test,
    y_predict,
    label_color="black",
    save_to=False,
    threshold=0.5,
    title=None,
    facecolor="#5c5c5c",
):
    # Predict class 1 if probability of being in class 1 is greater than threshold
    # (model.predict(X_test) does this automatically with a threshold of 0.5)

    confusion = confusion_matrix(y_test, y_predict)

    fig, ax = plt.subplots(dpi=130, figsize=(6, 4))
    sns.set(font_scale=1.3)
    group_counts = ["{0:0.0f}".format(value) for value in confusion.flatten()]
    group_percentages = [
        "{0:.2%}".format(value) for value in confusion.flatten() / np.sum(confusion)
    ]
    labels = [f"{v2}\n\n{v3}" for v2, v3 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    label_font = {
        "family": "Arial",
        "color": label_color,
        "weight": "bold",
        "size": 17,
    }

    title_font = {
        "family": "Arial",
        "color": label_color,
        "weight": "bold",
        "size": 27,
    }

    sns.heatmap(
        confusion,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["will_come", "will_cancel"],
        yticklabels=["will_come", "will_cancel"],
    )

    if title == None:
        title = "Confusion Matrix"
    plt.title(title, fontdict=label_font)
    plt.xlabel("Prediction", fontdict=label_font)
    plt.ylabel("Actual", fontdict=label_font)
    if save_to:
        plt.tight_layout()
        plt.savefig(
            save_to, dpi=170, facecolor=facecolor, bbox_inches="tight", pad_inches=1.6
        )
    plt.show()


def get_preds(pred_probas, threshold, y_test):
    df_preds = pd.DataFrame(pred_probas, columns=["no_cxl_proba", "cxl_proba"])
    df_preds["prediction"] = df_preds["cxl_proba"] >= threshold
    df_preds["actual"] = y_test.to_numpy()
    return df_preds


def get_fbeta_score(pred_probas, beta, threshold, y_test):
    df_preds = get_preds(pred_probas, threshold, y_test)
    precision = precision_score(y_test, df_preds["prediction"])
    recall = recall_score(y_test, df_preds["prediction"])
    fbeta_score = ((1 + beta ** 2) * precision * recall) / (
        beta ** 2 * precision + recall
    )
    return round(fbeta_score, 3)


def optimize_prob_threshold(
    model, X_test, y_test, beta=0.5, thresh_start=0.4, thresh_stop=0.95, confusion=False
):
    """
    Takes a trained cancellation XGBoost model and returns X_test, with predictions column (will_cancel)
    """
    pred_probas = model.predict_proba(X_test)

    thresholds = np.arange(thresh_start, thresh_stop, 0.002)
    fbetas = {}  # will hold {prob_thresh: resulting_fbeta_score}

    for t_val in thresholds:
        fbetas[t_val] = get_fbeta_score(pred_probas, beta, t_val, y_test)

    best_thresh = 0
    best_fbeta = 0

    for threshold, fb_score in fbetas.items():
        if fb_score > best_fbeta:
            best_thresh = round(threshold, 3)
            best_fbeta = round(fb_score, 3)
        else:
            continue

    df_preds = get_preds(pred_probas, best_thresh, y_test)
    if confusion:
        make_confusion_matrix(
            df_preds.actual, df_preds.prediction, threshold=best_thresh
        )

        print(
            f"Optimal probability threshold (to maximize F-{beta}): {best_thresh}\nF-{beta} Score: {best_fbeta}\n"
        )

    return best_thresh


def model_cancellations(df_res, as_of_date, hotel_num):
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
        n_jobs=-1,
        learning_rate=param_lr,
        n_estimators=param_ne,
        max_depth=param_md,
    )

    X_train, X_test, y_train, y_test = split_reservations(
        df_res, as_of_date=as_of_date, features=feature_cols
    )
    model.fit(X_train, y_train)

    return X_test, y_test, model


def predict_cancellations(df_res, as_of_date, hotel_num, confusion):
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

    # EW - FIX ISSUE: I AM USING SPLIT_RESERVATIONS TWICE, MAYBE?
    X_test, y_test, model = model_cancellations(df_res, as_of_date, hotel_num)
    # make predictions using above model, adjusting occ threshold to optimize F-0.5 score
    X_test_preds = model.predict_proba(X_test)

    thresh = optimize_prob_threshold(
        model, X_test=X_test, y_test=y_test, confusion=confusion
    )
    df_future_res = df_res.loc[list(X_test.index)].copy()
    df_future_res[["will_come_proba", "cxl_proba"]] = X_test_preds
    df_future_res.drop(columns="will_come_proba", inplace=True)
    df_future_res["will_cancel"] = df_future_res.cxl_proba >= thresh
    df_future_res["IsCanceled"] = y_test.to_numpy()
    return df_future_res