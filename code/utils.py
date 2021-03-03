import pandas as pd
import numpy as np

from collections import defaultdict
import datetime


def parse_dates(df_res):
    """
    Adds datetime & length-of-stay (LOS) columns to the DataFrame.
    _____________
    Takes:
        - df_res (required, DataFrame): raw hotel reservation data.

    Returns: df_res (DataFrame)
        - New column: 'ArrivalDate'
        - New column: LOS
        - Changed column: 'StatusDate'
    """

    df_res["ArrivalDate"] = pd.to_datetime(
        df_res.ArrivalDateYear.astype(str)
        + "-"
        + df_res.ArrivalDateMonth
        + "-"
        + df_res.ArrivalDateDayOfMonth.astype(str)
    )

    df_res["ReservationStatusDate"] = pd.to_datetime(df_res.ReservationStatusDate)

    df_res["LOS"] = (df_res.StaysInWeekendNights + df_res.StaysInWeekNights).astype(int)

    return df_res


def add_res_columns(df_res):
    """
    Adds several columns to df_res, including:
        - ResNum: unique ID for each booking
    """

    res_nums = list(range(len(df_res)))
    df_res.insert(0, "ResNum", res_nums)
    return df_res


def res_to_stats(df_res):
    """
    Takes a dataFrame (with parsed dates and LOS column) containing a hotel's reservations and
     returns a DataFrame containing nightly hotel room sales.
    """
    mask = df_res["IsCanceled"] == 0
    df_dates = df_res[mask]

    date = datetime.date(2015, 7, 1)
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
        for i in range(max_los + 1):

            date_tminus = date - pd.DateOffset(tminus)

            date_tminus_string = datetime.datetime.strftime(
                date_tminus, format="%Y-%m-%d"
            )

            mask = (
                (df_dates.ArrivalDate == date_tminus_string)
                & (df_dates.LOS >= 1 + tminus)
                & (df_dates.IsCanceled == 0)
            )

            date_stats["RoomsSold"] += len(df_dates[mask])
            date_stats["RoomRev"] += df_dates[mask].ADR.sum()

            tmp = (
                df_dates[mask][["ResNum", "CustomerType", "ADR"]]
                .groupby("CustomerType")
                .agg({"ResNum": "count", "ADR": "sum"})
                .rename(columns={"ResNum": "RS", "ADR": "Rev"})
            )

            c_types = ["Transient", "Transient-Party", "Group", "Contract"]

            if "Transient" in list(tmp.index):
                date_stats["Trn_RoomsSold"] += tmp.loc["Transient", "RS"]
                date_stats["Trn_RoomRev"] += tmp.loc["Transient", "Rev"]
            if "Transient-Party" in list(tmp.index):
                date_stats["TrnP_RoomsSold"] += tmp.loc["Transient-Party", "RS"]
                date_stats["TrnP_RoomRev"] += tmp.loc["Transient-Party", "Rev"]
            if "Group" in list(tmp.index):
                date_stats["Grp_RoomsSold"] += tmp.loc["Group", "RS"]
                date_stats["Grp_RoomRev"] += tmp.loc["Group", "Rev"]
            if "Contract" in list(tmp.index):
                date_stats["Cnt_RoomsSold"] += tmp.loc["Contract", "RS"]
                date_stats["Cnt_RoomRev"] += tmp.loc["Contract", "Rev"]

            tminus += 1

        nightly_stats[date_string] = dict(date_stats)
        date += delta

    return pd.DataFrame(nightly_stats).transpose()


def calculate_stats(stats_df):
    """
    Adds several columns to stats_df, including:
        - 'ADR' (by segment)
        - 'DOW' (day-of-week)
        - 'WD' (weekday - binary)
        - 'WE' (weekend - binary)
    """

    stats_df["ADR"] = round(stats_df.RoomRev / stats_df.RoomsSold, 2)
    stats_df["Trn_ADR"] = round(stats_df.Trn_RoomRev / stats_df.Trn_RoomsSold, 2)
    stats_df["TrnP_ADR"] = round(stats_df.TrnP_RoomRev / stats_df.TrnP_RoomsSold, 2)
    stats_df["Grp_ADR"] = round(stats_df.Grp_RoomRev / stats_df.Grp_RoomsSold, 2)
    stats_df["Cnt_ADR"] = round(stats_df.Cnt_RoomRev / stats_df.Cnt_RoomsSold, 2)

    dow = pd.to_datetime(stats_df.index, format="%Y-%m-%d")
    dow = dow.strftime("%a")
    stats_df.insert(0, "DOW", dow)
    stats_df["WE"] = (stats_df.DOW == "Fri") | (stats_df.DOW == "Sat")
    stats_df["WD"] = stats_df.WE == False

    return stats_df