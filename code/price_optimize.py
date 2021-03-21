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
from sklearn.metrics import r2_score

from demand_features import rf_cols

DATE_FMT = "%Y-%m-%d"


def splits(df_sim, as_of_date):
    """
    Splits df_sim into X_train, X_test, y_train, y_test.
    """
    train_mask = df_sim["StayDate"] < as_of_date
    test_mask = df_sim["AsOfDate"] == as_of_date
    df_train = df_sim.loc[train_mask].copy()
    df_test = df_sim.loc[test_mask].copy()

    X_train = df_train[rf_cols].copy()
    X_test = df_test[rf_cols].copy()
    y_train = df_train["ACTUAL_TRN_RoomsPickup"].copy()
    y_test = df_test["ACTUAL_TRN_RoomsPickup"].copy()

    return X_train, y_train, X_test, y_test


def train_model(df_sim, as_of_date="2017-08-01"):

    X_train, y_train, X_test, y_test = splits(df_sim, as_of_date)

    rfm = RandomForestRegressor(
        n_estimators=150,
        max_depth=48,
        min_samples_split=2,
        bootstrap=True,
        n_jobs=-1,
        random_state=20,
    )
    rfm.fit(X_train, y_train)
    preds = rfm.predict(X_test)
    print("Finished training model.")
    r2 = round(r2_score(y_test, preds), 4)
    print(f"R² Score on test set: {r2}")

    X_test["ACTUAL_TRN_RoomsPickup"] = preds.round(0).astype(int)
    df_sim["ACTUAL_TRN_RoomsPickup"] = X_test["ACTUAL_TRN_RoomsPickup"]
    mask = df_sim["AsOfDate"] == as_of_date
    df_pricing = df_sim[mask].copy()
    df_pricing.index = pd.to_datetime(pd.Series(df_pricing.StayDate), format=DATE_FMT)
    return df_pricing, rfm


def calculate_rev_at_price(price, df_sim, model, stay_date):
    df = df_sim.copy()
    df.loc[stay_date, "SellingPrice"] = price
    X = df.loc[stay_date, rf_cols].to_numpy()
    trn_rooms_to_book = model.predict([X])[0]
    return trn_rooms_to_book * price


def optimize_price(df_sim, as_of_date):
    """
    Models demand at current prices & stores resulting TRN RoomsBooked & Rev.

    Then adjusts prices by 5% increments in both directions, up to 25%.
    """
    df_pricing, model = train_model(df_sim, as_of_date)
    stay_dates = list(pd.to_datetime(pd.Series(df_pricing.index), format=DATE_FMT))
    price_adjustments = np.delete(np.arange(-0.25, 0.30, 0.05).round(2), 5)
    optimal_prices = []
    for stay_date in stay_dates:
        sd = dt.datetime.strftime(stay_date, format=DATE_FMT)
        original_rate = round(df_pricing.loc[sd, "SellingPrice"], 2)
        date_X = df_pricing.loc[sd, rf_cols].to_numpy()
        pred = model.predict([date_X])[0]
        rev = pred * original_rate
        optimal_rate = (original_rate, rev)

        for pct in price_adjustments:
            new_rate = round(original_rate * (1 + pct), 2)
            resulting_rev = calculate_rev_at_price(new_rate, df_sim, model, sd)

            if resulting_rev > optimal_rate[1]:
                optimal_rate = (new_rate, resulting_rev)
            else:
                continue
        optimal_prices.append(optimal_rate)
    assert len(optimal_prices) == 31, AssertionError("Something went wrong.")
    return optimal_prices


def update_rates(df_sim, as_of_date):
    """
    Implements price recommendations from optimize_price and returns df_optimized.
    """
    optimal_prices = optimize_price(df_sim, as_of_date)
    df_optimized = df_sim.copy()
    new_prices = []
    new_revs = []

    for rate, rev in optimal_prices:
        new_prices.append(rate)
        new_revs.append(rev)

    df_optimized["OptimalRate"] = new_prices
    df_optimized["RevAtOptimalRate"] = new_revs

    return df_optimized