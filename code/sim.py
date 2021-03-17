"""
This module contains functions that setup a hotel revenue
management simulation. The 'as_of_date' variable indicates
'today' in simulation terms. We essentially rollback the
reservations list back to the way they were on as_of_date.

Then, we capture various time-based performance statistics
and return a DataFrame (df_sim) that will be used to model demand.
"""

from collections import defaultdict
import datetime
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from utils import stly_cols, ly_cols, tm_cols, pace_tuples
from model_cancellations import get_otb_res, predict_cancellations

H1_CAPACITY = 187
H2_CAPACITY = 226
DATE_FMT = "%Y-%m-%d"
SIM_CSV_FP = "./sims/h{}_sim_{}.csv"


def setup_sim(df_otb, df_res, as_of_date="2017-08-01", end_date=False):
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

    df_dates = df_otb.copy()
    date = pd.to_datetime(as_of_date, format=DATE_FMT)
    if date + pd.DateOffset(31) > datetime.date(2017, 8, 31):
        end_date = datetime.date(2017, 8, 31)
    else:
        end_date = date + pd.DateOffset(31)
    delta = datetime.timedelta(days=1)
    max_los = int(df_dates["LOS"].max())

    nightly_stats = {}

    while date <= end_date:

        stay_date_str = datetime.datetime.strftime(date, format=DATE_FMT)
        # initialize date dict, which will go into nightly_stats as:
        # {'date': {'stat': 'val', 'stat', 'val'}}
        night_df = get_otb_res(df_res, as_of_date)

        mask = (night_df.ArrivalDate <= stay_date_str) & (
            night_df.CheckoutDate > stay_date_str
        )

        night_df = night_df[mask].copy()

        date_stats = aggregate_reservations(night_df, stay_date_str)
        nightly_stats[stay_date_str] = dict(date_stats)
        date += delta

    df_sim = pd.DataFrame(nightly_stats).transpose().fillna(0)
    df_sim.index = pd.to_datetime(df_sim.index, format=DATE_FMT)
    df_sim["Date"] = pd.to_datetime(df_sim.index, format=DATE_FMT)
    df_sim["TM05_Date"] = df_sim.Date - pd.DateOffset(5)
    df_sim["TM15_Date"] = df_sim.Date - pd.DateOffset(15)
    df_sim["TM30_Date"] = df_sim.Date - pd.DateOffset(30)
    # have to do it this way to prevent performance warning (non-vectorized operation)
    df_sim["WeekEndDate"] = df_sim.apply(
        lambda x: x["Date"] + pd.DateOffset(weekday=6), axis=1
    )

    dow = pd.to_datetime(df_sim.index, format=DATE_FMT)
    dow = dow.strftime("%a")
    df_sim.insert(0, "DOW", dow)
    df_sim["WE"] = (df_sim.DOW == "Fri") | (df_sim.DOW == "Sat")
    df_sim["WD"] = df_sim.WE is False

    # add STLY date
    stly_lambda = lambda x: pd.to_datetime(x) + relativedelta(
        years=-1, weekday=pd.to_datetime(x).weekday()
    )

    df_sim["STLY_Date"] = pd.to_datetime(df_sim.index.map(stly_lambda), format=DATE_FMT)

    return df_sim.copy()


def aggregate_reservations(night_df, date_string):
    """
    Takes a reservations DataFrame containing all reservations for a certain day and returns a dictionary of it's contents.
    ______
    Parameters:
        - night_df (pd.DataFrame, required): All OTB reservations that are touching as_of_date.
        - date_string (required, DATE_FMT)
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


def add_sim_cols(df_sim, df_dbd, capacity, pull_lya=True):
    """
    Adds several columns to df_sim, including:
        - 'Occ' (occupancy)
        - 'RevPAR' (revenue per available room)
        - 'ADR' (by segment)
        - 'RemSupply' (RoomsOTB - ProjectedCXLs)
        - 'DOW' (day-of-week)
        - 'WD' (weekday - binary)
        - 'WE' (weekend - binary)
        - 'STLY_Date' (datetime DATE_FMT)
        - 'LYA_' cols (last year actual RoomsSold, ADR, Rev, CXL'd)
        - 'GapToLYA_' cols
        - 'Actual_' cols (including pickup)
    _____
    Parameters:
        - df_sim
        - df_dbd
        - capacity (int, required): number of rooms in the hotel
        - pull_lya (set to False when using save_sims.py)

    """
    # add Occ/RevPAR/RemSupply columns'
    df_sim["Occ"] = round(df_sim["RoomsOTB"].astype("int") / int(capacity), 2)
    df_sim["RevPAR"] = round(df_sim["RevOTB"] / capacity, 2)
    df_sim["RemSupply"] = (
        capacity - df_sim.RoomsOTB.astype(int) + df_sim.CxlForecast.astype(int)
    )
    # to avoid errors, only operate on existing columns
    # Add ADR by segment
    df_sim["ADR_OTB"] = round(df_sim.RevOTB / df_sim.RoomsOTB, 2)

    seg_codes = ["TRN", "TRNP", "GRP", "CNT"]
    for code in seg_codes:
        if df_sim[code + "_RoomsOTB"].sum() > 0:
            df_sim[code + "_ADR_OTB"] = round(
                df_sim[code + "_RevOTB"] / df_sim[code + "_RoomsOTB"], 2
            )
        else:
            df_sim[code + "_ADR_OTB"] = 0
    if pull_lya:

        def apply_ly_cols(row):
            stly_date = row["STLY_Date"]
            stly_date_str = datetime.datetime.strftime(stly_date, format=DATE_FMT)

            df_lya = list(df_dbd.loc[stly_date_str, ly_cols])
            return tuple(df_lya)

        ly_new_cols = ["LYA_" + col for col in ly_cols]
        df_sim[ly_new_cols] = df_sim.apply(apply_ly_cols, axis=1, result_type="expand")

        df_sim.fillna(0, inplace=True)

        # add gap to LYA column
        df_sim["RoomsGapToLYA"] = df_sim.LYA_RoomsSold - df_sim.RoomsOTB
        df_sim["ADR_GapToLYA"] = df_sim.LYA_ADR - df_sim.ADR_OTB

    actual_cols = [
        "RoomsSold",
        "ADR",
        "RoomRev",
        "TRN_RoomsSold",
        "TRN_ADR",
        "TRN_RoomRev",
    ]

    for col in actual_cols:
        df_sim["Actual_" + col] = df_dbd[col]

    # add pickup cols
    df_sim["Actual_RoomsPickup"] = df_sim.Actual_RoomsSold - df_sim.RoomsOTB
    df_sim["Actual_ADR_Pickup"] = df_sim.Actual_ADR - df_sim.ADR_OTB
    df_sim["Actual_RoomRevPickup"] = df_sim.Actual_RoomRev - df_sim.RevOTB

    df_sim["Actual_TRN_RoomsPickup"] = df_sim.Actual_TRN_RoomsSold - df_sim.TRN_RoomsOTB
    df_sim["Actual_TRN_ADR_Pickup"] = df_sim.Actual_TRN_ADR - df_sim.TRN_ADR_OTB
    df_sim["Actual_TRN_RoomRevPickup"] = df_sim.Actual_TRN_RoomRev - df_sim.TRN_RevOTB
    return df_sim.copy()


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


def add_tminus_cols(df_sim, df_dbd, df_res, hotel_num, capacity):
    """
    This function adds booking statistics for a given date compared to the same date LY.
    """

    def apply_tminus_stats(row, tm_date_col):
        night = row["Date"]
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
        night_tm_stats.append(round(np.mean(tminus_otb.ADR), 2))  # add total ADR
        night_tm_stats.append(round(np.sum(tminus_otb.ADR), 2))  # add total Rev

        mkt_segs = ["Transient", "Transient-Party", "Group", "Contract"]
        for seg in mkt_segs:
            mask = tminus_otb.CustomerType == seg
            night_tm_stats.append(len(tminus_otb[mask]))  # add segment OTB
            night_tm_stats.append(
                round(np.mean(tminus_otb[mask].ADR), 2)
            )  # add segment ADR
            # MUST ADD TM REV TO night_tm_stats IN UTILS.PY (IN ORDER) BEFORE UN-COMMENTING THESE OUT
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

    tms = ["TM30", "TM15", "TM05"]
    segs = ["TRN", "TRNP", "GRP", "CNT"]
    # print(df_sim.columns)
    for tm in tms:
        # add total hotel stats first
        df_sim[tm + "_RoomsPickup"] = round(
            df_sim["RoomsOTB"] - df_sim[tm + "_RoomsOTB"], 2
        )
        df_sim[tm + "_ADR_Pickup"] = round(
            df_sim["ADR_OTB"] - df_sim[tm + "_ADR_OTB"], 2
        )
        df_sim[tm + "_RevPickup"] = round(df_sim["RevOTB"] - df_sim[tm + "_RevOTB"], 2)
        for seg in segs:
            # and now segmented stats
            df_sim[tm + "_" + seg + "_RoomsPickup"] = round(
                df_sim[seg + "_RoomsOTB"] - df_sim[tm + "_" + seg + "_RoomsOTB"], 2
            )
            df_sim[tm + "_" + seg + "_ADR_Pickup"] = round(
                df_sim[seg + "_ADR_OTB"] - df_sim[tm + "_" + seg + "_ADR_OTB"], 2
            )
            df_sim[tm + "_" + seg + "_RevPickup"] = round(
                df_sim[seg + "_RevOTB"] - df_sim[tm + "_" + seg + "_RevOTB"], 2
            )
    return df_sim.copy().fillna(0)


def add_stly_cols(df_sim, df_dbd, df_res, hotel_num, as_of_date, capacity, verbose=1):
    """
    Adds the following columns to df_sim:
        - STLY: RoomsOTB, ADR_OTB, TotalRoomsBooked_L30 & L90
    ____
    Parameters:
        - df_sim (pandas.DataFrame, required): simulation DataFrame (future looking)
        - df_dbd (pandas.DataFrame, required): actual hotel-level data for entire dataset
    """

    def apply_STLY_stats(row):
        """This function will be used in add_stly_cols to add STLY stats to df_sim."""

        # pull stly
        stay_date = row["STLY_Date"]
        stay_date_str = datetime.datetime.strftime(stay_date, format=DATE_FMT)
        stly_date = pd.to_datetime(as_of_date) + relativedelta(
            years=-1, weekday=pd.to_datetime(as_of_date).weekday()
        )
        stly_date_str = datetime.datetime.strftime(stly_date, format=DATE_FMT)
        # if verbose > 0:
        #     print(
        #         f"Pulling stats from STLY date {stly_date_str}, stay_date {stay_date_str}..."
        #     )

        stly_sim = pd.read_csv(
            SIM_CSV_FP.format(str(hotel_num), stly_date_str),
            parse_dates=[
                "Date",
                "TM05_Date",
                "TM15_Date",
                "TM30_Date",
                "STLY_Date",
                "WeekEndDate",
            ],
        )
        stly_sim.drop(columns={"Unnamed: 0": "Date"}, inplace=True, errors="ignore")
        stly_sim.index = stly_sim.Date
        stly_stats = []
        for col in stly_cols:
            stat = float(stly_sim.loc[stay_date_str, col])
            stly_stats.append(stat)

        return tuple(stly_stats)

    if verbose > 0:
        num_models = len(df_sim)
        print(f"Training {num_models} models to obtain STLY statistics...\n")

    new_col_names = ["STLY_" + col for col in stly_cols]
    df_sim[new_col_names] = df_sim.apply(
        apply_STLY_stats, result_type="expand", axis="columns"
    )
    # quick way to add ADR columns
    stly_segs = [
        ("STLY_RoomsOTB", "STLY_ADR_OTB", ""),
        ("STLY_TRN_RoomsOTB", "STLY_TRN_ADR_OTB", "TRN_"),
        ("STLY_TRNP_RoomsOTB", "STLY_TRNP_ADR_OTB", "TRNP_"),
        ("STLY_GRP_RoomsOTB", "STLY_GRP_ADR_OTB", "GRP_"),
        ("STLY_CNT_RoomsOTB", "STLY_CNT_ADR_OTB", "CNT_"),
    ]

    for Rooms, adr, name in stly_segs:
        df_sim["STLY_" + name + "RoomRev"] = df_sim[Rooms] * df_sim[adr]

    if verbose > 0:
        print("\nSTLY statistics obtained.\n")
    return df_sim.copy()


def add_pace_cols(df_sim):
    """
    Adds pace & pickup pace columns to df_sim.

    Uses lists generated in utils.py.
    """

    for ty, stly in pace_tuples:
        df_sim[ty + "_Pace"] = df_sim[ty] - df_sim[stly]

    return df_sim.copy()


def generate_simulation(
    df_dbd,
    as_of_date,
    hotel_num,
    df_res,
    confusion=True,
    pull_stly=True,
    verbose=1,
    pull_lya=True,
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
    # df_otb = get_otb_res(df_res, as_of_date)
    df_otb = predict_cancellations(
        df_res,
        as_of_date,
        hotel_num,
        confusion=confusion,
        verbose=verbose,
    )

    if hotel_num == 1:
        capacity = H1_CAPACITY
    else:
        capacity = H2_CAPACITY
    if verbose > 0:
        print("Setting up simulation...")
    df_sim = setup_sim(
        df_otb,
        df_res,
        as_of_date,
    )
    df_sim = add_sim_cols(df_sim, df_dbd, capacity, pull_lya=pull_lya)
    df_sim = add_cxl_cols(df_sim, df_res, as_of_date)

    if verbose > 0:
        print("Estimating prices...")
    df_sim = add_pricing(df_sim, df_res)

    if verbose > 0:
        print("Pulling T-Minus OTB statistics...")
    df_sim = add_tminus_cols(df_sim, df_dbd, df_res, hotel_num, capacity)

    if verbose > 0:
        print("Pulling STLY OTB statistics...")
    if pull_stly:
        df_sim = add_stly_cols(
            df_sim, df_dbd, df_res, hotel_num, as_of_date, capacity, verbose=verbose
        )
        df_sim = add_pace_cols(df_sim)

    if verbose > 0:
        print(f"\nSimulation setup complete. As of date: {as_of_date}\n")

    return df_sim.copy()
