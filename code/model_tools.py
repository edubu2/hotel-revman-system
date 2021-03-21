import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix


def get_preds(pred_probas, threshold, y_test):
    """
    Generates a DataFrame containing the probabilities that each reservation will not cancel, and the probabilities that they will cancel.
    """
    df_preds = pd.DataFrame(pred_probas, columns=["no_cxl_proba", "cxl_proba"])
    df_preds["prediction"] = df_preds["cxl_proba"] >= threshold
    df_preds["actual"] = y_test.to_numpy()
    return df_preds


def get_fbeta_score(pred_probas, threshold, y_test, beta=0.5):
    """Returns the F-Beta score at the given threshold & value of beta."""
    df_preds = get_preds(pred_probas, threshold, y_test)
    precision = precision_score(y_test, df_preds["prediction"])
    recall = recall_score(y_test, df_preds["prediction"])
    fbeta_score = ((1 + beta ** 2) * precision * recall) / (
        beta ** 2 * precision + recall
    )

    return round(fbeta_score, 3)


def optimize_prob_threshold(
    model,
    X_test,
    y_test,
    beta,
    thresh_start=0.35,
    thresh_stop=0.75,
    confusion=False,
):
    """
    Takes a trained cancellation XGBoost model and returns the best probability threshold.
    """
    pred_probas = model.predict_proba(X_test)

    thresholds = np.arange(thresh_start, thresh_stop, 0.01)
    fbetas = {}  # will hold {prob_thresh: resulting_fbeta_score}

    for t_val in thresholds:
        fbetas[t_val] = get_fbeta_score(pred_probas, t_val, y_test, beta)

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


def make_confusion_matrix(
    y_test,
    y_predict,
    label_color="black",
    save_to=False,
    threshold=0.5,
    title=None,
    facecolor="#5c5c5c",
):
    """
    Outputs a confusion matrix for the cancellation predictions.
    """
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
        # "family": "Arial",
        "color": label_color,
        "weight": "bold",
        "size": 17,
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
        title = "Confusion Matrix for Predicted Cancellations"
    plt.title(title, fontdict=label_font)
    plt.xlabel("Prediction", fontdict=label_font)
    plt.ylabel("Actual", fontdict=label_font)
    if save_to:
        plt.tight_layout()
        plt.savefig(
            save_to, dpi=170, facecolor=facecolor, bbox_inches="tight", pad_inches=1.6
        )
    plt.show()
    return
