import pandas as pd
import numpy as np

from collections import defaultdict
import datetime


def parse_dates(df_res):
    """
    Adds datetime & length-of-stay (LOS) columns to the DataFrame.
    Also replaces 'NULL' values with np.NaN
    _____________
    Takes:
        - df_res (required, DataFrame): raw hotel reservation data.

    Returns: df_res (DataFrame)
        - New column: 'ArrivalDate'
        - New column: LOS
        - Changed column: 'StatusDate'
    """
    df_res.loc[0, "Agent"]
    df_res = df_res.replace("       NULL", np.NaN)

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
        - Dummy columns:
            - CustomerType (is_grp, is_trn, is_trnP, contract)
            - ReservationStatus (Check-Out, No-Show, Canceled)
            - MarketSegment(Corp/Direct/Group/OfflineTA/OnlineTA)
            - DistributionChannel (Direct, TA/TO)
            - DepositType (Refundable, Non-Refundable)
            - AgencyBooking (True/False)
            - CompanyListed (True/False)

    """

    res_nums = list(range(len(df_res)))
    df_res.insert(0, "ResNum", res_nums)

    # one-hot-encode CustomerType
    df_res[["is_grp", "is_trn", "is_trnP"]] = pd.get_dummies(
        df_res.CustomerType, drop_first=True
    )

    # one-hot-encode ResStatus (IsCanceled already included, so only keeping no-show (checkout can be inferred))
    df_res[["CheckOut", "No-Show"]] = pd.get_dummies(
        df_res.ReservationStatus, drop_first=True
    )
    df_res.drop(columns=["CheckOut"], inplace=True)

    # one-hot-encode MarketSegment
    df_res[
        ["MS_Corporate", "MS_Direct", "MS_Group", "MS_OfflineTA", "MS_OnlineTA"]
    ] = pd.get_dummies(df_res.MarketSegment, drop_first=True)

    # one-hot-encode DistributionChannel
    df_res[["DC_Direct", "TA_TO"]] = pd.get_dummies(
        df_res.DistributionChannel, drop_first=True
    ).drop(columns="Undefined")

    # one-hot-encode DepositType
    df_res[["DT_NonRefundable", "DT_Refundable"]] = pd.get_dummies(
        df_res.DepositType, drop_first=True
    )

    # Boolean columns (AgencyBooking & CompanyListed)
    df_res["AgencyBooking"] = ~df_res["Agent"].isnull()
    df_res["CompanyListed"] = ~df_res["Company"].isnull()

    return df_res


def res_to_dbd(df_res):
    """
    Takes a dataFrame (with parsed dates and LOS column) containing a hotel's reservations and
    returns a DataFrame containing nightly hotel room sales.

    Our data is made up of reservations containing 'Arrival Date' and 'Length of Stay'.
    This function is used to determine how many rooms were sold on a given night, accounting for
    guests that arrived previously and are staying multiple nights.
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


def add_dbd_columns(dbd_df, capacity):
    """
    Adds several columns to dbd_df, including:
        - 'Occ' (occupancy)
        - 'RevPAR' (revenue per available room)
        - 'ADR' (by segment)
        - 'DOW' (day-of-week)
        - 'WD' (weekday - binary)
        - 'WE' (weekend - binary)
    """
    # add Occ & RevPAR columns
    dbd_df["Occ"] = round(dbd_df["RoomsSold"] / capacity, 2)
    dbd_df["RevPAR"] = round(dbd_df["RoomRev"] / capacity, 2)

    # Add ADR by segment
    dbd_df["ADR"] = round(dbd_df.RoomRev / dbd_df.RoomsSold, 2)
    dbd_df["Trn_ADR"] = round(dbd_df.Trn_RoomRev / dbd_df.Trn_RoomsSold, 2)
    dbd_df["TrnP_ADR"] = round(dbd_df.TrnP_RoomRev / dbd_df.TrnP_RoomsSold, 2)
    dbd_df["Grp_ADR"] = round(dbd_df.Grp_RoomRev / dbd_df.Grp_RoomsSold, 2)
    dbd_df["Cnt_ADR"] = round(dbd_df.Cnt_RoomRev / dbd_df.Cnt_RoomsSold, 2)

    dow = pd.to_datetime(dbd_df.index, format="%Y-%m-%d")
    dow = dow.strftime("%a")
    dbd_df.insert(0, "DOW", dow)
    dbd_df["WE"] = (dbd_df.DOW == "Fri") | (dbd_df.DOW == "Sat")
    dbd_df["WD"] = dbd_df.WE == False
    col_order = [
        "DOW",
        "Occ",
        "RoomsSold",
        "ADR",
        "RoomRev",
        "RevPAR",
        "Trn_RoomsSold",
        "Trn_ADR",
        "Trn_RoomRev",
        "Grp_RoomsSold",
        "Grp_ADR",
        "Grp_RoomRev",
        "TrnP_RoomsSold",
        "TrnP_ADR",
        "TrnP_RoomRev",
        "Cnt_RoomsSold",
        "Cnt_ADR",
        "Cnt_RoomRev",
        "WE",
        "WD",
    ]
    dbd_df = dbd_df[col_order]
    dbd_df.fillna(0, inplace=True)

    return dbd_df


def generate_hotel_dfs(res_filepath, capacity=None):
    """
    Takes in a hotel's raw reservations data (CSV) and capacity & returns two
    formatted DataFrames using the above functions:
        - df_res: cleaned reservations DataFrame
        - df_dbd: day-by-day hotel statistics, by Customer Type

    _______
    Parameters:
        - res_filepath (str, required): relative filepath to hotel's reservation data (CSV)
        - capacity (int, optional): the capacity of the hotel for Occ & RevPAR calculations
            - if omitted, capacity will calculated based on the maximum number of rooms sold on any
              day in the timespan provided
    """
    raw_df = pd.read_csv(res_filepath)
    df_res = parse_dates(raw_df)
    df_res = add_res_columns(df_res)

    df_dbd = res_to_dbd(df_res)

    if capacity is None:
        capacity = df_dbd.RoomsSold.max()

    df_dbd = add_dbd_columns(df_dbd, capacity=capacity)

    return df_res, df_dbd