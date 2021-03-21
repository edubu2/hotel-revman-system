"""
This module contains functions that setup a hotel revenue
management simulation. The 'as_of_date' variable indicates
'today' in simulation terms. We essentially rollback the
reservations list back to the way they were on as_of_date.

This code is optimized for efficiency, since it will be run
hundreds of times in sim_utils.py with different as_of_dates 
for each hotel. This will be the training data for my demand model.

The generate_simulation function wraps all of the others. It can 
be used in Jupyter notebooks and in other scripts by adjusting 
the arguments. For certain purposes, not all functions are needed. 

Then, we capture various time-based performance statistics
and return a DataFrame (df_sim) that will be used to model demand.
"""

from collections import defaultdict
import datetime
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from sim_utils import ly_cols, tm_cols
from model_cancellations import get_otb_res, predict_cancellations, get_future_res

H1_CAPACITY = 187
H2_CAPACITY = 226
DATE_FMT = "%Y-%m-%d"
SIM_PICKLE_FP = "./sims2/pickleh{}_sim_{}.pick"


def setup_sim(df_res, hotel_num, as_of_date="2017-08-01"):
    """
    Takes reservations and returns a DataFrame that can be used
    as a revenue management simulation.

    Very similar to setup.df_to_dbd (does the same thing but uses
    predicted cancels instead of actual)
    ____
    Parameters:
        - df_res (pandas.DataFrame, required): future-looking
          reservations DataFrame containing "will_cancel" column
        - as_of_date (str (DATE_FMT), optional): resulting
          day-by-days DataFrame will start on this day
        - cxl_type (str, optional): either "a" (actual) or "p"
          (predicted). Default      value is "p".
    """

    date = pd.to_datetime(as_of_date, format=DATE_FMT)
    if date + pd.DateOffset(31) > datetime.date(2017, 8, 31):
        end_date = datetime.date(2017, 8, 31)
    else:
        end_date = date + pd.DateOffset(31)
    delta = datetime.timedelta(days=1)

    nightly_stats = {}
    future_res = predict_cancellations(
        df_res, as_of_date, hotel_num, confusion=False, verbose=0
    )

    future_res = get_otb_res(future_res, as_of_date).copy()

    while date <= end_date:

        stay_date_str = datetime.datetime.strftime(date, format=DATE_FMT)

        mask = (future_res.ArrivalDate <= stay_date_str) & (
            future_res.CheckoutDate > stay_date_str
        )
        night_df = future_res[mask].copy()
        date_stats = aggregate_reservations(night_df)
        nightly_stats[stay_date_str] = dict(date_stats)
        date += delta

    df_sim = pd.DataFrame(nightly_stats).transpose().fillna(0)
    df_sim.index = pd.to_datetime(df_sim.index, format=DATE_FMT)
    return df_sim


def fixup_sim(df_sim, as_of_date):
    df_sim["Date"] = pd.to_datetime(df_sim.index, format=DATE_FMT)
    df_sim["TM05_Date"] = df_sim.Date - pd.DateOffset(5)
    df_sim["TM15_Date"] = df_sim.Date - pd.DateOffset(15)
    df_sim["TM30_Date"] = df_sim.Date - pd.DateOffset(30)
    # have to do it this way to prevent performance warning (non-vectorized operation)

    dow = pd.to_datetime(df_sim.index, format=DATE_FMT)
    dow = dow.strftime("%a")
    df_sim.insert(0, "DOW", dow)

    # add STLY date
    stly_lambda = lambda x: pd.to_datetime(x) + relativedelta(
        years=-1, weekday=pd.to_datetime(x).weekday()
    )

    df_sim["STLY_Date"] = pd.to_datetime(df_sim.index.map(stly_lambda), format=DATE_FMT)
    aod_dt = pd.to_datetime(as_of_date, format=DATE_FMT)
    df_sim["DaysUntilArrival"] = (df_sim.Date - aod_dt).dt.days

    return df_sim.copy()


def aggregate_reservations(night_df):
    """
    Takes a reservations DataFrame containing all reservations for a certain day and returns a dictionary of it's contents.
    ______
    Parameters:
        - night_df (pd.DataFrame, required): All OTB reservations that are touching as_of_date.
        - date_string (required, DATE_FMT)

    RETURNS
    -------
        - A dictionary, structured as: {<stay_date>: {'stat': 'val', 'stat', 'val'}, ...}
    """
    date_stats = defaultdict(int)
    date_stats["RoomsOTB"] += len(night_df)
    date_stats["RevOTB"] += night_df.ADR.sum()
    date_stats["CxlForecast"] += night_df.will_cancel.sum()
    tmp = (
        night_df[["ResNum", "CustomerType", "ADR", "will_cancel"]]
        .groupby("CustomerType")
        .agg({"ResNum": "count", "ADR": "sum", "will_cancel": "sum"})
        .rename(columns={"ResNum": "RS", "ADR": "Rev", "will_cancel": "CXL"})
    )

    seg_codes = [
        ("Transient", "TRN"),
        ("Transient-Party", "TRNP"),
        ("Group", "GRP"),
        ("Contract", "CNT"),
    ]
    for seg, code in seg_codes:
        if seg in list(tmp.index):
            date_stats[code + "_RoomsOTB"] += tmp.loc[seg, "RS"]
            date_stats[code + "_RevOTB"] += tmp.loc[seg, "Rev"]
            date_stats[code + "_CxlForecast"] += tmp.loc[seg, "CXL"]
        else:
            date_stats[code + "_RoomsOTB"] += 0
            date_stats[code + "_RevOTB"] += 0
            date_stats[code + "_CxlForecast"] += 0

    return date_stats


def add_cxl_cols(df_sim, df_res, as_of_date):
    """Adds total num of realized cancels as of as_of_date for each future date."""

    def cxl_mapper(night):
        night_ds = datetime.datetime.strftime(night, format=DATE_FMT)

        mask = (
            (df_res.IsCanceled == 1)
            & (df_res.ReservationStatusDate <= as_of_date)
            & (df_res.ArrivalDate <= night_ds)
            & (df_res.CheckoutDate > night_ds)
        )
        df_cxl_res = df_res[mask].copy()
        return len(df_cxl_res)

    df_sim["Realized_Cxls"] = df_sim["Date"].map(cxl_mapper)
    return df_sim.copy()


def add_pricing(df_sim, df_res):
    """
    Adds 'SellingPrice' column to df_sim.

    Contains the average TRANSIENT rate for all booked reservations during a given week,
    including those that cancelled at any time.
    """

    # apply the weekly WD/WE prices to the original df_sim
    def rate_mapper(night):
        night_ds = datetime.datetime.strftime(night, format=DATE_FMT)

        mask = (
            (df_res.ArrivalDate <= night_ds)
            & (df_res.CheckoutDate > night_ds)
            & (df_res.CustomerType == "Transient")
        )
        df_pricing_res = df_res[mask].copy()
        price = round(df_pricing_res.ADR.mean(), 2)
        return price

    df_sim["SellingPrice"] = df_sim["Date"].map(rate_mapper)

    return df_sim.copy()


def add_tminus_cols(df_sim, df_res):
    """
    This function adds booking statistics for a given date compared to the same date LY.
    """

    def apply_tminus_stats(row, tm_date_col):
        night_ds = datetime.datetime.strftime(row["Date"], format=DATE_FMT)
        tminus_date = row[tm_date_col]
        tminus_ds = datetime.datetime.strftime(tminus_date, format=DATE_FMT)
        tminus_otb = get_otb_res(df_res, tminus_ds)
        mask = (tminus_otb["ArrivalDate"] <= night_ds) & (
            tminus_otb["CheckoutDate"] > night_ds
        )
        tminus_otb = tminus_otb[mask].copy()
        night_tm_stats = []

        night_tm_stats.append(len(tminus_otb))  # add total OTB
        night_tm_stats.append(round(np.sum(tminus_otb.ADR), 2))  # add total Rev

        mkt_segs = ["Transient", "Transient-Party", "Group", "Contract"]
        for seg in mkt_segs:
            mask = tminus_otb.CustomerType == seg
            night_tm_stats.append(len(tminus_otb[mask]))  # add segment OTB
            night_tm_stats.append(
                round(np.sum(tminus_otb[mask].ADR), 2)
            )  # add segment Rev

        return tuple(night_tm_stats)

    tms = ["TM30", "TM15", "TM05"]
    for tm in tms:
        tm_col_names = [tm + "_" + col for col in tm_cols]
        df_sim[tm_col_names] = df_sim.apply(
            lambda row: apply_tminus_stats(row, tm + "_Date"),
            result_type="expand",
            axis="columns",
        )

    df_sim.drop(
        columns=["TM30_Date", "TM15_Date", "TM05_Date"], inplace=True, errors="ignore"
    )

    return df_sim.copy().fillna(0)


def generate_simulation(
    df_dbd,
    as_of_date,
    hotel_num,
    df_res,
    confusion=True,
    verbose=1,
):
    """
    Takes reservations and returns a DataFrame that can be used as a revenue management simulation.

    Resulting DataFrame contains all future reservations as of a certain point in time.

    ____
    Parameters:
        - df_dbd (DataFrame, required)
        - as_of_date (str DATE_FMT, required): date of simulation
        - hotel_num (int, required): 1 for h1 and 2 for h2
        - df_res (DataFrame, required)
        - confusion (bool, optional): whether or not to print a confusion matrix plot for
          cancellation predictions.
        - pull_stly (bool, optional): whether or not to pull stly stats for sim.
        - verbose (int, optional): if zero, print statements will be suppressed.
    """
    aod_dt = pd.to_datetime(as_of_date, format=DATE_FMT)
    min_dt = datetime.date(2015, 7, 1)
    assert hotel_num in [1, 2], ValueError(
        "Invalid hotel_num. Must be (integer) 1 or 2."
    )
    assert aod_dt >= min_dt, ValueError(
        "as_of_date must be between 7/1/16 and 8/30/17."
    )

    if hotel_num == 1:
        capacity = H1_CAPACITY
    else:
        capacity = H2_CAPACITY
    if verbose > 0:
        print("Setting up simulation...")
    df_sim = setup_sim(df_res, hotel_num, as_of_date)
    df_sim = fixup_sim(df_sim, as_of_date)
    df_sim = add_cxl_cols(df_sim, df_res, as_of_date)

    if verbose > 0:
        print("Estimating prices...")
    df_sim = add_pricing(df_sim, df_res)

    if verbose > 0:
        print("Pulling T-Minus OTB statistics...")
    df_sim = add_tminus_cols(df_sim, df_res)

    if verbose > 0:
        print(f"\nSimulation setup complete. As of date: {as_of_date}.\n")

    return df_sim.copy()
