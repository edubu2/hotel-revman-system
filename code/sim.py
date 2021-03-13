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
from model_cancellations import get_otb_res, predict_cancellations

H1_CAPACITY = 187
H2_CAPACITY = 226


def setup_sim(df_otb, df_res, as_of_date="2017-08-01"):
    """
    Takes reservations and returns a DataFrame that can be used
    as a revenue management simulation.

    Very similar to setup.df_to_dbd (does the same thing but uses
    predicted cancels instead of actual)
    ____
    Parameters:
        - df_res (pandas.DataFrame, required): future-looking
          reservations DataFrame containing "will_cancel" column
        - as_of_date (str ("%Y-%m-%d"), optional): resulting
          day-by-days DataFrame will start on this day
        - cxl_type (str, optional): either "a" (actual) or "p"
          (predicted). Default      value is "p".
    """

    df_dates = df_otb.copy()
    date = pd.to_datetime(as_of_date, format="%Y-%m-%d")
    end_date = datetime.date(2017, 8, 31)
    delta = datetime.timedelta(days=1)
    max_los = int(df_dates["LOS"].max())

    nightly_stats = {}

    while date <= end_date:

        date_string = datetime.datetime.strftime(date, format="%Y-%m-%d")
        # initialize date dict, which will go into nightly_stats as:
        # {'date': {'stat': 'val', 'stat', 'val'}}
        night_df = get_otb_res(df_res, date_string)

        mask = (night_df.ArrivalDate <= date_string) & (
            night_df.CheckoutDate > date_string
        )

        night_df = night_df[mask].copy()

        date_stats = aggregate_reservations(night_df, date_string)
        nightly_stats[date_string] = dict(date_stats)
        date += delta

    df_sim = pd.DataFrame(nightly_stats).transpose().fillna(0)
    # df_sim["WeekEndDate"] = df_sim.index + pd.DateOffset(weekday=6)
    df_sim.index = pd.to_datetime(df_sim.index, format="%Y-%m-%d")
    df_sim["Date"] = pd.to_datetime(df_sim.index, format="%Y-%m-%d")
    # have to do it this way to prevent performance warning (non-vectorized operation)
    df_sim["WeekEndDate"] = df_sim.apply(
        lambda x: x["Date"] + pd.DateOffset(weekday=6), axis=1
    )

    dow = pd.to_datetime(df_sim.index, format="%Y-%m-%d")
    dow = dow.strftime("%a")
    df_sim.insert(0, "DOW", dow)
    df_sim["WE"] = (df_sim.DOW == "Fri") | (df_sim.DOW == "Sat")
    df_sim["WD"] = df_sim.WE is False

    # add STLY date
    stly_lambda = lambda x: pd.to_datetime(x) + relativedelta(
        years=-1, weekday=pd.to_datetime(x).weekday()
    )

    df_sim["STLY_Date"] = pd.to_datetime(
        df_sim.index.map(stly_lambda), format="%Y-%m-%d"
    )

    return df_sim


def aggregate_reservations(night_df, date_string):
    """
    Takes a reservations DataFrame containing all reservations for a certain day and returns a dictionary of it's contents.
    ______
    Parameters:
        - night_df (pd.DataFrame, required): All OTB reservations that are touching as_of_date.
        - date_string (required, "%Y-%m-%d")
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

    if "Transient" in list(tmp.index):
        date_stats["Trn_RoomsOTB"] += tmp.loc["Transient", "RS"]
        date_stats["Trn_RevOTB"] += tmp.loc["Transient", "Rev"]
        date_stats["Trn_CxlForecast"] += tmp.loc["Transient", "CXL"]

    if "Transient-Party" in list(tmp.index):
        date_stats["TrnP_RoomsOTB"] += tmp.loc["Transient-Party", "RS"]
        date_stats["TrnP_RevOTB"] += tmp.loc["Transient-Party", "Rev"]
        date_stats["TrnP_CxlForecast"] += tmp.loc["Transient-Party", "CXL"]

    if "Group" in list(tmp.index):
        date_stats["Grp_RoomsOTB"] += tmp.loc["Group", "RS"]
        date_stats["Grp_RevOTB"] += tmp.loc["Group", "Rev"]
        date_stats["Grp_CxlForecast"] += tmp.loc["Group", "CXL"]

    if "Contract" in list(tmp.index):
        date_stats["Cnt_RoomsOTB"] += tmp.loc["Contract", "RS"]
        date_stats["Cnt_RevOTB"] += tmp.loc["Contract", "Rev"]
        date_stats["Cnt_CxlForecast"] += tmp.loc["Contract", "CXL"]

    return date_stats


def add_sim_cols(df_sim, df_dbd, capacity):
    """
    Adds several columns to df_sim, including:
        - 'Occ' (occupancy)
        - 'RevPAR' (revenue per available room)
        - 'ADR' (by segment)
        - 'RemSupply' (RoomsOTB - ProjectedCXLs)
        - 'DOW' (day-of-week)
        - 'WD' (weekday - binary)
        - 'WE' (weekend - binary)
        - 'STLY_Date' (datetime "%Y-%m-%d")
        - 'LYA_' cols (last year actual RoomsSold, ADR, Rev, CXL'd)
    _____
    Parameters:
        - df_sim: day-by-day hotel DF
        - capacity (int, required): number of rooms in the hotel
    """
    # add Occ/RevPAR/RemSupply columns'
    df_sim["Occ"] = round(df_sim["RoomsOTB"] / capacity, 2)
    df_sim["RevPAR"] = round(df_sim["RevOTB"] / capacity, 2)
    df_sim["RemSupply"] = (
        capacity - df_sim.RoomsOTB.astype(int) + df_sim.CxlForecast.astype(int)
    )
    # to avoid errors, only operate on existing columns
    # Add ADR by segment
    df_sim["ADR_OTB"] = round(df_sim.RevOTB / df_sim.RoomsOTB, 2)
    try:
        df_sim["Trn_ADR_OTB"] = round(df_sim.Trn_RevOTB / df_sim.Trn_RoomsOTB, 2)
    except:
        pass
    try:
        df_sim["TrnP_ADR_OTB"] = round(df_sim.TrnP_RevOTB / df_sim.TrnP_RoomsOTB, 2)
    except:
        pass
    try:
        df_sim["Grp_ADR_OTB"] = round(df_sim.Grp_RevOTB / df_sim.Grp_RoomsOTB, 2)
    except:
        pass
    try:
        df_sim["Cnt_ADR_OTB"] = round(df_sim.Cnt_RevOTB / df_sim.Cnt_RoomsOTB, 2)
    except:
        pass

    def apply_ly_cols(row):
        stly_date = row["STLY_Date"]
        stly_date_str = datetime.datetime.strftime(stly_date, format="%Y-%m-%d")
        df_lya = list(
            df_dbd.loc[
                stly_date_str, ["RoomsSold", "ADR", "RoomRev", "RevPAR", "NumCancels"]
            ]
        )
        return tuple(df_lya)

    df_sim[
        ["LYA_RoomsSold", "LYA_ADR", "LYA_RoomRev", "LYA_RevPAR", "LYA_NumCancels"]
    ] = df_sim.apply(apply_ly_cols, axis=1, result_type="expand")

    df_sim.fillna(0, inplace=True)
    return df_sim


def add_pricing(df_sim):
    """
    Adds 'SellingPrice' column to df_sim.

    Contains the average rate for all booked reservations during a given week (WD/WE).
    This gives us an indication of what the hotel's online selling prices.
    """
    # get average WD/WE pricing for each week
    df_sim.index = pd.to_datetime(df_sim.index, format="%Y-%m-%d")
    df_pricing = (
        df_sim[["Trn_RoomsOTB", "Trn_RevOTB", "WD"]]
        .groupby([pd.Grouper(freq="1W"), "WD"])
        .agg("sum")
    )
    df_pricing = df_pricing.reset_index().rename(columns={"level_0": "Date"})
    df_pricing["Date"] = pd.to_datetime(df_pricing.Date, format="%Y-%m-%d")
    df_pricing["Trn_ADR_OTB"] = round(
        df_pricing.Trn_RevOTB / df_pricing.Trn_RoomsOTB, 2
    )
    df_pricing.index = df_pricing.Date

    # apply the weekly WD/WE prices to the original df_sim
    def apply_rates(row):
        wd = row["WD"] == 1
        week_end = datetime.datetime.strftime(row.WeekEndDate, format="%Y-%m-%d")
        mask = df_pricing.WD == wd
        price = df_pricing[mask].loc[week_end, "Trn_ADR_OTB"]
        return price

    df_sim["SellingPrice"] = df_sim.apply(apply_rates, axis=1)

    return df_sim


def add_tminus_cols(df_sim, df_dbd, df_res, hotel_num, as_of_date, capacity):
    """
    This function adds booking statistics for a given date compared to the same date LY.
    """

    def apply_tminus_stats(row, n_days):
        tminus_date = row["Date"] - pd.DateOffset(n_days)
        tminus_ds = datetime.datetime.strftime(tminus_date, format="%Y-%m-%d")
        tminus_otb = get_otb_res(df_res, tminus_ds)
        mask = (tminus_otb["ArrivalDate"] <= tminus_ds) & (
            tminus_otb["CheckoutDate"] > tminus_ds
        )

        tminus_otb = tminus_otb[mask].copy()
        tm_sim = setup_sim(tminus_otb, df_res, tminus_ds)
        tm_sim = add_sim_cols(tm_sim, df_dbd, capacity)
        tm_sim = add_pricing(tm_sim)
        TM_OTB = float(tm_sim.loc[tminus_ds, "RoomsOTB"])
        TM_REV = float(tm_sim.loc[tminus_ds, "ADR_OTB"])
        TM_ADR = round(TM_REV / TM_OTB, 2)
        TM_SELLPRICE = float(tm_sim.loc[tminus_ds, "SellingPrice"])
        TM_CxlForecast = float(tm_sim.loc[tminus_ds, "CxlForecast"])

        try:
            TM_TRN_OTB = float(tm_sim.loc[tminus_ds, "Trn_RoomsOTB"])
            TM_TRN_ADR = float(tm_sim.loc[tminus_ds, "Trn_ADR_OTB"])
        except:
            TM_TRN_OTB = 0
            TM_TRN_ADR = 0
        try:
            TM_TRNP_OTB = float(tm_sim.loc[tminus_ds, "TrnP_RoomsOTB"])
            TM_TRNP_ADR = float(tm_sim.loc[tminus_ds, "TrnP_ADR_OTB"])

        except:
            TM_TRNP_OTB = 0
            TM_TRNP_ADR = 0
        try:
            TM_GRP_OTB = float(tm_sim.loc[tminus_ds, "Grp_RoomsOTB"])
            TM_GRP_ADR = float(tm_sim.loc[tminus_ds, "Grp_ADR_OTB"])
        except:
            TM_GRP_OTB = 0
            TM_GRP_ADR = 0
        try:
            TM_CNT_OTB = float(tm_sim.loc[tminus_ds, "Cnt_RoomsOTB"])
            TM_CNT_ADR = float(tm_sim.loc[tminus_ds, "Cnt_ADR_OTB"])
        except:
            TM_CNT_OTB = 0
            TM_CNT_ADR = 0

        return (
            TM_OTB,
            TM_ADR,
            TM_SELLPRICE,
            TM_TRN_OTB,
            TM_TRN_ADR,
            TM_TRNP_OTB,
            TM_TRNP_ADR,
            TM_GRP_OTB,
            TM_GRP_ADR,
            TM_CNT_OTB,
            TM_CNT_ADR,
        )

    tm_cols = [
        "RoomsOTB",
        "ADR",
        "SellingPrice",
        "TRN_OTB",
        "TRN_ADR",
        "TRNP_OTB",
        "TRNP_ADR",
        "GRP_OTB",
        "GRP_ADR",
        "CNT_OTB",
        "CNT_ADR",
    ]
    t30_col_names = ["tm30_" + col for col in tm_cols]
    t15_col_names = ["tm15_" + col for col in tm_cols]
    t05_col_names = ["tm05_" + col for col in tm_cols]
    print("Pulling t-5 booking stats...")
    df_sim[t05_col_names] = df_sim.apply(
        lambda row: apply_tminus_stats(row, 5), result_type="expand", axis="columns"
    )
    print("Pulling t-15 booking stats...")
    df_sim[t15_col_names] = df_sim.apply(
        lambda row: apply_tminus_stats(row, 15), result_type="expand", axis="columns"
    )
    print("Pulling t-30 booking stats...")
    df_sim[t30_col_names] = df_sim.apply(
        lambda row: apply_tminus_stats(row, 30), result_type="expand", axis="columns"
    )
    return df_sim


def add_stly_cols(df_sim, df_dbd, df_res, hotel_num, as_of_date, capacity):
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
        stly_date = row["STLY_Date"]
        stly_date_str = datetime.datetime.strftime(stly_date, format="%Y-%m-%d")
        print(f"Predicting cancellations as of STLY date {stly_date_str}...")
        stly_otb = predict_cancellations(
            df_res, stly_date_str, hotel_num, confusion=False
        )
        mask = (stly_otb["ArrivalDate"] <= stly_date_str) & (
            stly_otb["CheckoutDate"] > stly_date_str
        )

        stly_otb = stly_otb[mask].copy()
        stly_sim = setup_sim(stly_otb, df_res, stly_date_str)
        stly_sim = add_pricing(stly_sim)
        stly_sim = add_tminus_cols(
            stly_sim, df_dbd, df_res, hotel_num, stly_date_str, capacity
        )
        STLY_OTB = float(stly_sim.loc[stly_date_str, "RoomsOTB"])
        STLY_REV = float(stly_sim.loc[stly_date_str, "RevOTB"])
        STLY_ADR = round(STLY_REV / STLY_OTB, 2)
        STLY_SELLPRICE = float(stly_sim.loc[stly_date_str, "SellingPrice"])
        STLY_CxlForecast = float(stly_sim.loc[stly_date_str, "CxlForecast"])
        STLY_TM05_OTB = float(stly_sim.loc[stly_date_str, "tm05_RoomsOTB"])
        STLY_TM05_ADR = float(stly_sim.loc[stly_date_str, "tm15_ADR"])
        STLY_TM15_OTB = float(stly_sim.loc[stly_date_str, "tm15_RoomsOTB"])
        STLY_TM15_ADR = float(stly_sim.loc[stly_date_str, "tm05_ADR"])
        STLY_TM30_OTB = float(stly_sim.loc[stly_date_str, "tm30_RoomsOTB"])
        STLY_TM30_ADR = float(stly_sim.loc[stly_date_str, "tm30_ADR"])

        try:
            STLY_TRN_OTB = float(stly_sim.loc[stly_date_str, "Trn_RoomsOTB"])
            STLY_TRN_REV = float(stly_sim.loc[stly_date_str, "Trn_RevOTB"])
            STLY_TRN_ADR = round(STLY_TRN_REV / STLY_TRN_OTB, 2)
            STLY_TRN_CXL = float(stly_sim.loc[stly_date_str, "Trn_CxlForecast"])
            STLY_TM05_TRN_OTB = float(stly_sim.loc[stly_date_str, "tm05_Trn_OTB"])
            STLY_TM05_TRN_ADR = float(stly_sim.loc[stly_date_str, "tm05_Trn_ADR"])
            STLY_TM15_TRN_OTB = float(stly_sim.loc[stly_date_str, "tm15_Trn_OTB"])
            STLY_TM15_TRN_ADR = float(stly_sim.loc[stly_date_str, "tm15_Trn_ADR"])
            STLY_TM30_TRN_OTB = float(stly_sim.loc[stly_date_str, "tm30_Trn_OTB"])
            STLY_TM30_TRN_ADR = float(stly_sim.loc[stly_date_str, "tm30_Trn_ADR"])
        except:
            STLY_TRN_OTB = 0
            STLY_TRN_REV = 0
            STLY_TRN_ADR = 0
            STLY_TRN_CXL = 0
            STLY_TM05_TRN_OTB = 0
            STLY_TM05_TRN_ADR = 0
            STLY_TM15_TRN_OTB = 0
            STLY_TM15_TRN_ADR = 0
            STLY_TM30_TRN_OTB = 0
            STLY_TM30_TRN_ADR = 0
        try:
            STLY_TRNP_OTB = float(stly_sim.loc[stly_date_str, "TrnP_RoomsOTB"])
            STLY_TRNP_REV = float(stly_sim.loc[stly_date_str, "TrnP_RevOTB"])
            STLY_TRNP_ADR = round(STLY_TRNP_REV / STLY_TRNP_OTB, 2)
            STLY_TRNP_CXL = float(stly_sim.loc[stly_date_str, "TrnP_CxlForecast"])
            STLY_TM05_TRNP_OTB = float(stly_sim.loc[stly_date_str, "tm05_TrnP_OTB"])
            STLY_TM05_TRNP_ADR = float(stly_sim.loc[stly_date_str, "tm05_TrnP_ADR"])
            STLY_TM15_TRNP_OTB = float(stly_sim.loc[stly_date_str, "tm15_TrnP_OTB"])
            STLY_TM15_TRNP_ADR = float(stly_sim.loc[stly_date_str, "tm15_TrnP_ADR"])
            STLY_TM30_TRNP_OTB = float(stly_sim.loc[stly_date_str, "tm30_TrnP_OTB"])
            STLY_TM30_TRNP_ADR = float(stly_sim.loc[stly_date_str, "tm30_TrnP_ADR"])
        except:
            STLY_TRNP_OTB = 0
            STLY_TRNP_REV = 0
            STLY_TRNP_ADR = 0
            STLY_TRNP_CXL = 0
            STLY_TM05_TRNP_OTB = 0
            STLY_TM05_TRNP_ADR = 0
            STLY_TM15_TRNP_OTB = 0
            STLY_TM15_TRNP_ADR = 0
            STLY_TM30_TRNP_OTB = 0
            STLY_TM30_TRNP_ADR = 0
        try:
            STLY_GRP_OTB = float(stly_sim.loc[stly_date_str, "Grp_RoomsOTB"])
            STLY_GRP_REV = float(stly_sim.loc[stly_date_str, "Grp_RevOTB"])
            STLY_GRP_ADR = round(STLY_GRP_REV / STLY_GRP_OTB, 2)
            STLY_GRP_CXL = float(stly_sim.loc[stly_date_str, "Grp_CxlForecast"])
            STLY_TM05_GRP_OTB = float(stly_sim.loc[stly_date_str, "tm05_Grp_OTB"])
            STLY_TM05_GRP_ADR = float(stly_sim.loc[stly_date_str, "tm05_Grp_ADR"])
            STLY_TM15_GRP_OTB = float(stly_sim.loc[stly_date_str, "tm15_Grp_OTB"])
            STLY_TM15_GRP_ADR = float(stly_sim.loc[stly_date_str, "tm15_Grp_ADR"])
            STLY_TM30_GRP_OTB = float(stly_sim.loc[stly_date_str, "tm30_Grp_OTB"])
            STLY_TM30_GRP_ADR = float(stly_sim.loc[stly_date_str, "tm30_Grp_ADR"])
        except:
            STLY_GRP_OTB = 0
            STLY_GRP_REV = 0
            STLY_GRP_ADR = 0
            STLY_GRP_CXL = 0
            STLY_TM05_GRP_OTB = 0
            STLY_TM05_GRP_ADR = 0
            STLY_TM15_GRP_OTB = 0
            STLY_TM15_GRP_ADR = 0
            STLY_TM30_GRP_OTB = 0
            STLY_TM30_GRP_ADR = 0
        try:
            STLY_CNT_OTB = float(stly_sim.loc[stly_date_str, "Cnt_RoomsOTB"])
            STLY_CNT_REV = float(stly_sim.loc[stly_date_str, "Cnt_RevOTB"])
            STLY_CNT_ADR = round(STLY_CNT_REV / STLY_CNT_OTB, 2)
            STLY_CNT_CXL = float(stly_sim.loc[stly_date_str, "Cnt_CxlForecast"])
            STLY_TM05_CNT_OTB = float(stly_sim.loc[stly_date_str, "tm05_Cnt_OTB"])
            STLY_TM05_CNT_ADR = float(stly_sim.loc[stly_date_str, "tm05_Cnt_ADR"])
            STLY_TM15_CNT_OTB = float(stly_sim.loc[stly_date_str, "tm15_Cnt_OTB"])
            STLY_TM15_CNT_ADR = float(stly_sim.loc[stly_date_str, "tm15_Cnt_ADR"])
            STLY_TM30_CNT_OTB = float(stly_sim.loc[stly_date_str, "tm30_Cnt_OTB"])
            STLY_TM30_CNT_ADR = float(stly_sim.loc[stly_date_str, "tm30_Cnt_ADR"])
        except:
            STLY_CNT_OTB = 0
            STLY_CNT_REV = 0
            STLY_CNT_ADR = 0
            STLY_CNT_CXL = 0
            STLY_TM05_CNT_OTB = 0
            STLY_TM05_CNT_ADR = 0
            STLY_TM15_CNT_OTB = 0
            STLY_TM15_CNT_ADR = 0
            STLY_TM30_CNT_OTB = 0
            STLY_TM30_CNT_ADR = 0

        return (
            STLY_OTB,
            STLY_REV,
            STLY_ADR,
            STLY_SELLPRICE,
            STLY_CxlForecast,
            STLY_TRN_OTB,
            STLY_TRN_REV,
            STLY_TRN_ADR,
            STLY_TRN_CXL,
            STLY_TRNP_OTB,
            STLY_TRNP_REV,
            STLY_TRNP_ADR,
            STLY_TRNP_CXL,
            STLY_GRP_OTB,
            STLY_GRP_REV,
            STLY_GRP_ADR,
            STLY_GRP_CXL,
            STLY_CNT_OTB,
            STLY_CNT_REV,
            STLY_CNT_ADR,
            STLY_CNT_CXL,
            STLY_TM05_OTB,
            STLY_TM05_ADR,
            STLY_TM15_OTB,
            STLY_TM15_ADR,
            STLY_TM30_OTB,
            STLY_TM30_ADR,
            STLY_TM05_TRN_OTB,
            STLY_TM05_TRN_ADR,
            STLY_TM15_TRN_OTB,
            STLY_TM15_TRN_ADR,
            STLY_TM30_TRN_OTB,
            STLY_TM30_TRN_ADR,
            STLY_TM05_TRNP_OTB,
            STLY_TM05_TRNP_ADR,
            STLY_TM15_TRNP_OTB,
            STLY_TM15_TRNP_ADR,
            STLY_TM30_TRNP_OTB,
            STLY_TM30_TRNP_ADR,
            STLY_TM05_GRP_OTB,
            STLY_TM05_GRP_ADR,
            STLY_TM15_GRP_OTB,
            STLY_TM15_GRP_ADR,
            STLY_TM30_GRP_OTB,
            STLY_TM30_GRP_ADR,
            STLY_TM05_CNT_OTB,
            STLY_TM05_CNT_ADR,
            STLY_TM15_CNT_OTB,
            STLY_TM15_CNT_ADR,
            STLY_TM30_CNT_OTB,
            STLY_TM30_CNT_ADR,
        )

    num_models = len(df_sim)
    print(f"Training {num_models} models to obtain STLY statistics...\n")

    new_col_names = [
        "STLY_OTB",
        "STLY_Rev",
        "STLY_ADR",
        "STLY_SellingPrice",
        "STLY_CxlForecast",
        "STLY_TRN_OTB",
        "STLY_TRN_REV",
        "STLY_TRN_ADR",
        "STLY_TRN_CXL",
        "STLY_TRNP_OTB",
        "STLY_TRNP_REV",
        "STLY_TRNP_ADR",
        "STLY_TRNP_CXL",
        "STLY_GRP_OTB",
        "STLY_GRP_REV",
        "STLY_GRP_ADR",
        "STLY_GRP_CXL",
        "STLY_CNT_OTB",
        "STLY_CNT_REV",
        "STLY_CNT_ADR",
        "STLY_CNT_CXL",
        "STLY_TM05_OTB",
        "STLY_TM05_ADR",
        "STLY_TM15_OTB",
        "STLY_TM15_ADR",
        "STLY_TM30_OTB",
        "STLY_TM30_ADR",
        "STLY_TM05_TRN_OTB",
        "STLY_TM05_TRN_ADR",
        "STLY_TM15_TRN_OTB",
        "STLY_TM15_TRN_ADR",
        "STLY_TM30_TRN_OTB",
        "STLY_TM30_TRN_ADR",
        "STLY_TM05_TRNP_OTB",
        "STLY_TM05_TRNP_ADR",
        "STLY_TM15_TRNP_OTB",
        "STLY_TM15_TRNP_ADR",
        "STLY_TM30_TRNP_OTB",
        "STLY_TM30_TRNP_ADR",
        "STLY_TM05_GRP_OTB",
        "STLY_TM05_GRP_ADR",
        "STLY_TM15_GRP_OTB",
        "STLY_TM15_GRP_ADR",
        "STLY_TM30_GRP_OTB",
        "STLY_TM30_GRP_ADR",
        "STLY_TM05_CNT_OTB",
        "STLY_TM05_CNT_ADR",
        "STLY_TM15_CNT_OTB",
        "STLY_TM15_CNT_ADR",
        "STLY_TM30_CNT_OTB",
        "STLY_TM30_CNT_ADR",
    ]
    df_sim[new_col_names] = df_sim.apply(
        apply_STLY_stats, result_type="expand", axis="columns"
    )
    print("\nSTLY statistics obtained.\n")
    return df_sim


def generate_simulation(df_dbd, as_of_date, hotel_num, df_res):
    """
    Takes reservations and returns a DataFrame that can be used as a revenue management simulation.

    Resulting DataFrame contains all future reservations as of a certain point in time.

    ____
    Parameters:
        - df_dbd (DataFrame, required)
        - as_of_date (str "%Y-%m-%d", required): date of simulation
        - hotel_num (int, required): 1 for h1 and 2 for h2
        - df_res (DataFrame, required)
    """
    assert hotel_num in [1, 2], ValueError("Invalid hotel_num.")
    aod_dt = pd.to_datetime(as_of_date, format="%Y-%m-%d")
    min_dt = datetime.date(2016, 7, 1)
    assert aod_dt > min_dt, ValueError("as_of_date must be between 7/1/16 and 8/30/17")
    print("Preparing crystal ball...")
    print("Predicting cancellations on all future reservations...")
    # df_otb = get_otb_res(df_res, as_of_date)
    df_otb = predict_cancellations(df_res, as_of_date, hotel_num, print_len=True)

    if hotel_num == 1:
        capacity = H1_CAPACITY
    else:
        capacity = H2_CAPACITY

    print("Setting up simulation...")
    df_sim = setup_sim(
        df_otb,
        df_res,
        as_of_date,
    )
    df_sim = add_sim_cols(df_sim, df_dbd, capacity)
    print("Estimating prices...")
    df_sim = add_pricing(df_sim)
    df_sim = add_tminus_cols(df_sim, df_dbd, df_res, hotel_num, as_of_date, capacity)
    df_sim = add_stly_cols(
        df_sim,
        df_dbd,
        df_res,
        hotel_num,
        as_of_date,
        capacity,
    )

    print("Simulation setup complete!\n")
    return df_sim
