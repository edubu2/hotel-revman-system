"""
This script trains a model to predict remaining transient pickup (demand).

It then finds the optimal selling price based on resulting room revenue.

Returns a DataFrame containing 31 days of future dates, along with predicted demand
at the optimal selling prices.
"""
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from demand_features import rf_cols, rf2_cols

DATE_FMT = "%Y-%m-%d"


def splits(df_sim, as_of_date, features):
    """
    Splits df_sim into X_train, X_test, y_train, y_test.
    """

    train_mask = df_sim["StayDate"] < as_of_date
    test_mask = df_sim["AsOfDate"] == as_of_date
    df_train = df_sim.loc[train_mask].copy()
    df_test = df_sim.loc[test_mask].copy()

    X_train = df_train[features].copy()
    X_test = df_test[features].copy()
    y_train = df_train["ACTUAL_TRN_RoomsPickup"].copy()
    y_test = df_test["ACTUAL_TRN_RoomsPickup"].copy()

    return X_train, y_train, X_test, y_test


def train_model(
    df_sim, as_of_date, hotel_num, features, X_train, y_train, X_test, y_test
):

    if hotel_num == 1:
        rfm = RandomForestRegressor(  # max_depth=None
            n_estimators=550,
            n_jobs=-1,
            random_state=20,
        )
        rfm.fit(X_train, y_train)
        preds = rfm.predict(X_test)
    else:
        # hotel 2
        rfm = RandomForestRegressor(
            n_estimators=350, max_depth=25, n_jobs=-1, random_state=20
        )
        rfm.fit(X_train, y_train)
        preds = rfm.predict(X_test)

    # add preds back to original
    X_test["Proj_TRN_RemDemand"] = preds.round(0).astype(int)

    mask = df_sim["AsOfDate"] == as_of_date
    df_pricing = df_sim[mask].copy()
    df_pricing["Proj_TRN_RemDemand"] = X_test["Proj_TRN_RemDemand"]

    return df_pricing, rfm, preds


def calculate_rev_at_price(price, df_pricing, model, df_index, features):
    """
    Calculates transient room revenue at predicted selling prices."""
    df = df_pricing.copy()
    df.loc[df_index, "SellingPrice"] = price
    X = df.loc[df_index, features].to_numpy()
    resulting_rn = model.predict([X])[0]
    resulting_rev = round(resulting_rn * price, 2)
    return resulting_rn, resulting_rev


def get_optimal_prices(df_pricing, as_of_date, model, features):
    """
    Models demand at current prices & stores resulting TRN RoomsBooked & Rev.

    Then adjusts prices by 5% increments in both directions, up to 25%.
    """
    # optimize_price func
    indices = list(df_pricing.index)
    price_adjustments = np.delete(
        np.arange(-0.25, 0.30, 0.05).round(2), 5
    )  # delete zero (already have it)
    optimal_prices = []
    for i in indices:  # loop thru stay dates & calculate stats @ original rate
        original_rate = round(df_pricing.loc[i, "SellingPrice"], 2)
        date_X = df_pricing.loc[i, features].to_numpy()
        original_rn = model.predict([date_X])[0]
        original_rev = original_rn * original_rate
        optimal_rate = (
            original_rate,  # will be updated
            original_rn,  # will be updated
            original_rev,  # will be updated
            original_rate,  # will remain
            original_rn,  # will remain
            original_rev,  # will remain
        )

        for pct in price_adjustments:  # now take optimal rate (highest rev)
            new_rate = round(original_rate * (1 + pct), 2)
            resulting_rn, resulting_rev = calculate_rev_at_price(
                new_rate, df_pricing, model, i, features
            )

            if resulting_rev > optimal_rate[1]:
                optimal_rate = (
                    new_rate,
                    resulting_rn,
                    resulting_rev,
                    original_rate,
                    original_rn,
                    original_rev,
                )

            else:
                continue
        optimal_prices.append(optimal_rate)
    assert len(optimal_prices) == 31, AssertionError("Something went wrong.")
    return df_pricing, optimal_prices


def add_rates(df_pricing, optimal_prices):
    """
    Implements price recommendations from optimize_price and returns pricing_df
    """
    new_rates = []
    resulting_rns = []
    resulting_revs = []
    original_rates = []
    original_rns = []
    original_revs = []
    for (
        new_rate,
        resulting_rn,
        resulting_rev,
        original_rate,
        original_rn,
        original_rev,
    ) in optimal_prices:
        new_rates.append(new_rate)
        resulting_rns.append(resulting_rn)
        resulting_revs.append(round(resulting_rev, 2))
        original_rates.append(original_rate)
        original_rns.append(original_rn)
        original_revs.append(round(original_rev, 2))

    df_pricing["OptimalRate"] = new_rates
    df_pricing["TRN_rnPU_AtOptimal"] = resulting_rns
    df_pricing["TRN_RevPU_AtOptimal"] = resulting_revs
    df_pricing["TRN_rnPU_AtOriginal"] = original_rns
    df_pricing["TRN_RN_ProjVsActual_OP"] = (
        df_pricing["TRN_rnPU_AtOriginal"] - df_pricing["ACTUAL_TRN_RoomsPickup"]
    )
    df_pricing["TRN_RevPU_AtOriginal"] = original_revs
    df_pricing["TRN_RevProjVsActual_OP"] = (
        df_pricing["TRN_RevPU_AtOriginal"] - df_pricing["ACTUAL_TRN_RevPickup"]
    )

    return df_pricing


def summarize_model_results(model, y_test, preds):
    """Writes model metrics to STDOUT."""
    r2 = round(r2_score(y_test, preds), 3)
    mae = round(mean_absolute_error(y_test, preds), 3)
    mse = round(mean_squared_error(y_test, preds), 3)

    print(
        f"R² score on test set (stay dates Aug 1 - Aug 31, 2017):                        {r2}"
    )
    print(
        f"MAE (Mean Absolute Error) score on test set (stay dates Aug 1 - Aug 31, 2017): {mae}"
    )
    print(
        f"MSE (Mean Squared Error) score on test set (stay dates Aug 1 - Aug 31, 2017):  {mse}\n"
    )
    pass


def add_display_columns(df_pricing, capacity):
    """
    Adds the following informative columns that will be displayed to app users:
        - RecommendedPriceChange (optimal rate variance to original rate)
        - ProjChgAtOptimal (projected RN & Rev change at optimal rates)
        - DOW (day of week)
        - Actual & Projected Occ
    """
    df_pricing["RecommendedPriceChange"] = (
        df_pricing["OptimalRate"] - df_pricing["SellingPrice"]
    )

    df_pricing["ProjRN_ChgAtOptimal"] = (
        df_pricing["TRN_rnPU_AtOptimal"] - df_pricing["TRN_rnPU_AtOriginal"]
    )

    df_pricing["ProjRevChgAtOptimal"] = (
        df_pricing["TRN_RevPU_AtOptimal"] - df_pricing["TRN_RevPU_AtOriginal"]
    )

    df_pricing["DOW"] = (
        df_pricing["StayDate"]
        .map(lambda x: dt.datetime.strftime(x, format="%a"))
        .astype(str)
    )

    avg_price_change = round(df_pricing["RecommendedPriceChange"].mean(), 2)
    total_rn_opp = round(df_pricing["ProjRN_ChgAtOptimal"].sum(), 2)
    total_rev_opp = round(df_pricing["ProjRevChgAtOptimal"].sum(), 2)

    print(
        f"Average recommended price change...                                            {avg_price_change}"
    )
    print(
        f"Estimated RN (Roomnight) growth after implementing price recommendations...    {total_rn_opp}"
    )
    print(
        f"Estimated revenue growth after implementing price recommendations...           {total_rev_opp}"
    )

    # occupancy columns
    df_pricing["ACTUAL_Occ"] = round(df_pricing["ACTUAL_RoomsSold"] / capacity, 1)
    df_pricing["TotalProjRoomsSold"] = (
        capacity - df_pricing["RemSupply"] + df_pricing["Proj_TRN_RemDemand"]
    )
    df_pricing["ProjOcc"] = round(df_pricing["TotalProjRoomsSold"] / capacity, 2)

    return df_pricing.copy()


def recommend_pricing(hotel_num, df_sim, as_of_date):
    assert hotel_num in (1, 2), ValueError("hotel_num must be (int) 1 or 2.")
    if hotel_num == 1:
        capacity = 187
        features = rf_cols
    else:
        capacity = 226
        features = rf2_cols

    print("Training Random Forest model to predict remaining transient demand...")
    X_train, y_train, X_test, y_test = splits(df_sim, as_of_date, features)
    df_pricing, model, preds = train_model(
        df_sim, as_of_date, hotel_num, features, X_train, y_train, X_test, y_test
    )

    print("Model ready.\n")
    summarize_model_results(model, y_test, preds)

    print("Calculating optimal selling prices...\n")
    df_pricing, optimal_prices = get_optimal_prices(
        df_pricing, as_of_date, model, features
    )
    df_pricing = add_rates(df_pricing, optimal_prices)
    df_pricing = add_display_columns(df_pricing, capacity)

    print("Simulation ready.\n")

    return df_pricing
