import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import *
from collections import defaultdict


def setup_sim(df_futures, as_of_date="2017-08-01"):
    """
    Takes reservations and returns a DataFrame that can be used as a revenue management simulation.

    Very similar to setup.df_to_dbd (does the same thing but uses predicted cancels instead of actual)

    Our data is made up of reservations containing 'Arrival Date' and 'Length of Stay'.
    This function is used to determine how many rooms were sold on a given night, accounting for
    guests that arrived previously and are staying multiple nights.

    ____
    Parameters:
        - df_res (pandas.DataFrame, required): future-looking reservations DataFrame containing "will_cancel" column
        - as_of_date (str ("%Y-%m-%d"), optional): resulting day-by-days DataFrame will start on this day
        - cxl_type (str, optional): either "a" (actual) or "p" (predicted). Default value is "p".
    """
    df_dates = df_futures.copy()
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

    return pd.DataFrame(nightly_stats).transpose()


def add_sim_cols(df_sim, capacity):
    """
    Adds several columns to df_sim, including:
        - 'Occ' (occupancy)
        - 'RevPAR' (revenue per available room)
        - 'ADR' (by segment)
        - 'DOW' (day-of-week)
        - 'WD' (weekday - binary)
        - 'WE' (weekend - binary)
        - 'STLY_Date' (datetime "%Y-%m-%d")

    _____
    Parameters:
        - df_sim: day-by-day hotel DF
        - capacity (int, required): number of rooms in the hotel
    """
    # add Occ & RevPAR columns'
    df_sim["Occ"] = round(df_sim["RoomsOTB"] / capacity, 2)
    df_sim["RevPAR"] = round(df_sim["RevOTB"] / capacity, 2)

    # Add ADR by segment
    df_sim["ADR_OTB"] = round(df_sim.RevOTB / df_sim.RoomsOTB, 2)
    df_sim["Trn_ADR_OTB"] = round(df_sim.Trn_RevOTB / df_sim.Trn_RoomsOTB, 2)
    df_sim["TrnP_ADR_OTB"] = round(df_sim.TrnP_RevOTB / df_sim.TrnP_RoomsOTB, 2)
    df_sim["Grp_ADR_OTB"] = round(df_sim.Grp_RevOTB / df_sim.Grp_RoomsOTB, 2)
    df_sim["Cnt_ADR_OTB"] = round(df_sim.Cnt_RevOTB / df_sim.Cnt_RoomsOTB, 2)

    dow = pd.to_datetime(df_sim.index, format="%Y-%m-%d")
    dow = dow.strftime("%a")
    df_sim.insert(0, "DOW", dow)
    df_sim["WE"] = (df_sim.DOW == "Fri") | (df_sim.DOW == "Sat")
    df_sim["WD"] = df_sim.WE == False

    # add STLY date
    stly_lambda = lambda x: pd.to_datetime(x) + relativedelta(
        years=-1, weekday=pd.to_datetime(x).weekday()
    )
    df_sim["STLY_Date"] = df_sim.index.map(stly_lambda)

    df_sim.fillna(0, inplace=True)

    return df_sim


def add_stly_cols(df_sim, df_dbd):
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

    return