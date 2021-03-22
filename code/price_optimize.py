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


def train_model(df_sim, as_of_date, X_train, y_train, X_test, y_test):

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

    X_test["Proj_TRN_RoomsPickup"] = preds.round(0).astype(int)
    df_sim["Proj_TRN_RoomsPickup"] = X_test["Proj_TRN_RoomsPickup"]
    mask = df_sim["AsOfDate"] == as_of_date
    df_pricing = df_sim[mask].copy()
    return df_pricing, rfm, preds


def calculate_rev_at_price(price, df_pricing, model, df_index):
    """
    Calculates transient room revenue at predicted selling prices."""
    df = df_pricing.copy()
    df.loc[df_index, "SellingPrice"] = price
    X = df.loc[df_index, rf_cols].to_numpy()
    trn_rooms_to_book = model.predict([X])[0]
    resulting_rev = round(trn_rooms_to_book * price, 2)
    return resulting_rev


def get_optimal_prices(df_pricing, as_of_date, model):
    """
    Models demand at current prices & stores resulting TRN RoomsBooked & Rev.

    Then adjusts prices by 5% increments in both directions, up to 25%.
    """
    # optimize_price func
    indices = list(df_pricing.index)
    price_adjustments = np.delete(np.arange(-0.25, 0.30, 0.05).round(2), 5)
    optimal_prices = []
    for i in indices:
        #     sd = dt.datetime.strftime(stay_date, format=DATE_FMT)
        original_rate = round(df_pricing.loc[i, "SellingPrice"], 2)
        date_X = df_pricing.loc[i, rf_cols].to_numpy()
        pred = model.predict([date_X])[0]
        proj_rev_old_price = pred * original_rate
        optimal_rate = (
            original_rate,
            proj_rev_old_price,
            original_rate,
            proj_rev_old_price,
        )

        for pct in price_adjustments:
            new_rate = round(original_rate * (1 + pct), 2)
            resulting_rev = calculate_rev_at_price(new_rate, df_pricing, model, i)

            if resulting_rev > optimal_rate[1]:
                optimal_rate = (
                    new_rate,
                    resulting_rev,
                    original_rate,
                    proj_rev_old_price,
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
    resulting_revs = []
    original_rates = []
    proj_rev_old_prices = []
    for new_rate, resulting_rev, original_rate, proj_rev_old_price in optimal_prices:
        new_rates.append(new_rate)
        resulting_revs.append(resulting_rev)
        original_rates.append(original_rate)
        proj_rev_old_prices.append(round(proj_rev_old_price, 2))

    df_pricing["OptimalRate"] = new_rates
    df_pricing["TRN_RevPickupAtOptimal"] = resulting_revs
    df_pricing["TRN_RevPickupAtOriginal"] = proj_rev_old_prices

    return df_pricing


def summarize_model_results(model, y_test, preds):
    """Writes model metrics to STDOUT."""
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)

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


def add_finishing_touches(df_pricing):
    """
    Adds the following columns:
        - RecommendedPriceChange (optimal rate variance to original rate)
        - RevOpportunity (projected revenue change at optimal rates)
        - DOW (day of week)
    """
    df_pricing["RecommendedPriceChange"] = (
        df_pricing["OptimalRate"] - df_pricing["SellingPrice"]
    )
    avg_price_change = df_pricing["RecommendedPriceChange"].mean()
    print(
        f"Average recommended price change...                                  {avg_price_change}"
    )

    df_pricing["ProjRevOpportunity"] = (
        df_pricing["TRN_RevPickupAtOptimal"] - df_pricing["TRN_RevPickupAtOriginal"]
    )
    total_rev_opp = df_pricing["ProjRevOpportunity"].sum()
    print(
        f"Estimated revenue growth from implementing price recommendations...  {total_rev_opp} "
    )

    df_pricing["DOW"] = (
        df_pricing["StayDate"]
        .map(lambda x: dt.datetime.strftime(x, format="%a"))
        .astype(str)
    )

    return df_pricing.copy()


def recommend_pricing(df_sim, as_of_date):

    print("Training Random Forest model to predict remaining transient demand...")
    X_train, y_train, X_test, y_test = splits(df_sim, as_of_date)
    df_pricing, model, preds = train_model(
        df_sim, as_of_date, X_train, y_train, X_test, y_test
    )

    print("Model ready.\n")
    summarize_model_results(model, y_test, preds)

    print("Calculating optimal selling prices...\n")
    df_pricing, optimal_prices = get_optimal_prices(df_pricing, as_of_date, model)
    df_pricing = add_rates(df_pricing, optimal_prices)
    df_pricing = add_finishing_touches(df_pricing)

    print("Simulation ready.\n")

    return df_pricing
