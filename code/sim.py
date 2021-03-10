import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import *
from collections import defaultdict
from model_cancellations import predict_cancellations

h1_capacity = 187
h2_capacity = 226


def setup_sim(df_future_res, as_of_date="2017-08-01"):
    """
    Takes reservations and returns a DataFrame that can be used as a revenue management simulation.

    Very similar to setup.df_to_dbd (does the same thing but uses predicted cancels instead of actual)
    ____
    Parameters:
        - df_res (pandas.DataFrame, required): future-looking reservations DataFrame containing "will_cancel" column
        - as_of_date (str ("%Y-%m-%d"), optional): resulting day-by-days DataFrame will start on this day
        - cxl_type (str, optional): either "a" (actual) or "p" (predicted). Default value is "p".
    """
    df_dates = df_future_res.copy()
    date = pd.to_datetime(as_of_date, format="%Y-%m-%d")
    end_date = datetime.date(2017, 8, 31)
    delta = datetime.timedelta(days=1)
    max_los = int(df_dates["LOS"].max())

    nightly_stats = {}

    while date <= end_date:

        date_string = datetime.datetime.strftime(date, format="%Y-%m-%d")
        tminus = 0

        # initialize date dict, which will go into nightly_stats as {'date': {'stat': 'val', 'stat', 'val'}}
        date_stats = defaultdict(int)

        # start on the arrival date and move back
        # to capture ALL reservations touching 'date' (and not just those that arrive on 'date')
        for _ in range(max_los):

            #
            date_tminus = date - pd.DateOffset(tminus)

            date_tminus_string = datetime.datetime.strftime(
                date_tminus, format="%Y-%m-%d"
            )

            mask = (
                (df_dates.ArrivalDate == date_tminus_string)
                & (df_dates.LOS >= 1 + tminus)
                & (df_dates.IsCanceled == 0)
            )

            date_stats["RoomsOTB"] += len(df_dates[mask])
            date_stats["RevOTB"] += df_dates[mask].ADR.sum()
            date_stats["CxlForecast"] += df_dates[mask].will_cancel.sum()

            tmp = (
                df_dates[mask][["ResNum", "CustomerType", "ADR", "will_cancel"]]
                .groupby("CustomerType")
                .agg({"ResNum": "count", "ADR": "sum", "will_cancel": "sum"})
                .rename(columns={"ResNum": "RS", "ADR": "Rev", "will_cancel": "CXL"})
            )

            if "Transient" in list(tmp.index):
                date_stats["Trn_RoomsOTB"] += tmp.loc["Transient", "RS"]
                date_stats["Trn_CxlProj"] += tmp.loc["Transient", "CXL"]
                date_stats["Trn_RevOTB"] += tmp.loc["Transient", "Rev"]

            if "Transient-Party" in list(tmp.index):
                date_stats["TrnP_RoomsOTB"] += tmp.loc["Transient-Party", "RS"]
                date_stats["TrnP_CxlProj"] += tmp.loc["Transient-Party", "CXL"]
                date_stats["TrnP_RevOTB"] += tmp.loc["Transient-Party", "Rev"]

            if "Group" in list(tmp.index):
                date_stats["Grp_RoomsOTB"] += tmp.loc["Group", "RS"]
                date_stats["Grp_CxlProj"] += tmp.loc["Group", "CXL"]
                date_stats["Grp_RevOTB"] += tmp.loc["Group", "Rev"]

            if "Contract" in list(tmp.index):
                date_stats["Cnt_RoomsOTB"] += tmp.loc["Contract", "RS"]
                date_stats["Cnt_CxlProj"] += tmp.loc["Contract", "CXL"]
                date_stats["Cnt_RevOTB"] += tmp.loc["Contract", "Rev"]

            tminus += 1

        nightly_stats[date_string] = dict(date_stats)
        date += delta

    return pd.DataFrame(nightly_stats).transpose().fillna(0)


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
        - 'LYA_RoomsSold' (last year actual rooms sold)
        - 'LYA_ADR' (last year actual average daily rate)
        - 'LYA_Occ' (last year actual occupancy)


    _____
    Parameters:
        - df_sim: day-by-day hotel DF
        - capacity (int, required): number of rooms in the hotel
    """
    # add Occ/RevPAR/RemSupply columns'
    df_sim["Occ"] = round(df_sim["RoomsOTB"] / capacity, 2)
    df_sim["RevPAR"] = round(df_sim["RevOTB"] / capacity, 2)
    df_sim["RemSupply"] = df_sim.RoomsOTB.astype(int) - df_sim.CxlForecast.astype(int)
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

    dow = pd.to_datetime(df_sim.index, format="%Y-%m-%d")
    dow = dow.strftime("%a")
    df_sim.insert(0, "DOW", dow)
    df_sim["WE"] = (df_sim.DOW == "Fri") | (df_sim.DOW == "Sat")
    df_sim["WD"] = df_sim.WE == False

    # add STLY date & LYA columns
    stly_lambda = lambda x: pd.to_datetime(x) + relativedelta(
        years=-1, weekday=pd.to_datetime(x).weekday()
    )
    df_sim["STLY_Date"] = df_sim.index.map(stly_lambda)

    def apply_lyas(row):
        stly_date = row["STLY_Date"]
        stly_date_string = datetime.datetime.strftime(stly_date, format="%Y-%m-%d")

        LYA_RoomsSold = df_dbd.loc[stly_date_string, "RoomsSold"]
        LYA_ADR = df_dbd.loc[stly_date_string, "ADR"]
        LYA_Occ = df_dbd.loc[stly_date_string, "Occ"]

        return LYA_RoomsSold, LYA_ADR, LYA_Occ

    df_sim[["LYA_RoomsSold", "LYA_ADR", "LYA_Occ"]] = df_sim.apply(
        apply_lyas, axis=1, result_type="expand"
    )

    df_sim.fillna(0, inplace=True)

    return df_sim


def add_stly_cols(df_sim, df_dbd, df_res, hotel_num, as_of_date, capacity):
    """
    Adds the following columns to df_sim:
        - Last year actual: Rooms Sold, ADR, Cancellations
        - STLY: RoomsOTB, ADR_OTB, TotalRoomsBooked_L30 & L90

    ____
    Parameters:
        - df_sim (pandas.DataFrame, required): simulation DataFrame (future looking)
        - df_dbd (pandas.DataFrame, required): actual hotel-level data for entire dataset
    """
    first_date = df_sim.index.min()
    last_date = df_sim.index.max()

    pull_cols = [
        "RoomsOTB",
        "RevOTB",
        "ADR_OTB",
        "CxlForecast",
        "RemSupply",
        "Trn_RoomsOTB",
        "Trn_CxlProj",
        "Trn_RevOTB",
        "Trn_ADR_OTB",
        "TrnP_RoomsOTB",
        "TrnP_CxlProj",
        "TrnP_RevOTB",
        "TrnP_ADR_OTB",
        "Cnt_RoomsOTB",
        "Cnt_CxlProj",
        "Cnt_RevOTB",
        "Cnt_ADR_OTB",
        "Grp_RoomsOTB",
        "Grp_CxlProj",
        "Grp_RevOTB",
        "Grp_ADR_OTB",
    ]
    stly_col_names = ["STLY_" + col for col in pull_cols]
    # lya_pull_cols = ["RoomsSold", "Occ", "ADR", "RevPAR"]
    # [new_col_names.append("LYA_" + col) for col in lya_pull_cols]

    def apply_STLY_stats(row):
        """This function will be used for df_sim.STLY_Date.map() to add STLY stats to df_sim."""

        # pull stly
        stly_date = row["STLY_Date"]
        date_string = datetime.datetime.strftime(stly_date, format="%Y-%m-%d")
        stly_future_res = predict_cancellations(df_res, date_string, hotel_num, False)
        stly_df_sim = setup_sim(stly_future_res, date_string)
        stly_df_sim = add_sim_cols(stly_df_sim, df_dbd, capacity)
        stly_stats = list(stly_df_sim.loc[date_string, pull_cols])
        # pull LYA (last year actual)
        # lya_stats = list(df_dbd.loc[date_string, lya_pull_cols])

        return stly_stats

    df_sim[stly_col_names] = df_sim.apply(
        apply_STLY_stats, result_type="expand", axis="columns"
    )

    # df_sim[["LYA_RoomsSold", "LYA_Occ", ""]]

    return df_sim


def generate_simulation(df_dbd, as_of_date, hotel_num, df_res):
    """
    Takes reservations and returns a DataFrame that can be used as a revenue management simulation.

    Resulting DataFrame contains all future reservations as of a certain point in time.

    ____
    Parameters:
        - df_future_res (pandas.DataFrame, required): reservations dataframe (with proj. cancels) generated
          from the functions in setup.py and model_cancellations.py
        - as_of_date (str "%Y-%m-%d", required): date of simulation
        - hotel_num (int, required): 1 for h1 and 2 for h2
    """
    assert hotel_num in [1, 2], "Invalid hotel_num."

    if hotel_num == 1:
        capacity = h1_capacity
    else:
        capacity = h2_capacity

    df_future_res = predict_cancellations(
        df_res, as_of_date, hotel_num, confusion=False
    )

    df_sim = setup_sim(df_future_res, as_of_date)
    df_sim = add_sim_cols(df_sim, df_dbd, capacity)
    df_sim = add_stly_cols(
        df_sim,
        df_dbd,
        df_res,
        hotel_num,
        as_of_date,
        capacity,
    )

    return df_sim


def get_last_30():
    # start date = as_of_date - 30
    # as_of start date, run generate_simulation
    # record rooms OTB
    #
    pass