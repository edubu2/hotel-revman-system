import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, confusion_matrix

from collections import defaultdict
import datetime
from dateutil.relativedelta import *
from features import X1_cxl_cols, X2_cxl_cols
from xgboost import XGBClassifier


def parse_dates(df_res):
    """
    Adds datetime & length-of-stay (LOS) columns to the DataFrame.
    Also replaces 'NULL' values with np.NaN
    _____________
    Takes:
        - df_res (required, DataFrame): raw hotel reservation data.

    Returns: df_res (DataFrame)
        - New column: 'ArrivalDate'
        - New column: 'LOS'
        - New column: 'CheckoutDate'
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

    # add checkout date
    checkout_lam = lambda row: row["ArrivalDate"] + pd.DateOffset(row["LOS"])
    df_res["CheckoutDate"] = df_res.apply(checkout_lam, axis=1)

    return df_res


def add_res_columns(df_res):
    """
    Adds several columns to df_res, including:
        - ResNum: unique ID for each booking
        - PreviousBookings: Prev. Bookings not cxl'd + prev. cxls
        - ResMadeDate: Date reservation was made
        - NumPeople: Adults + children + babies
        - Dummy columns (one-hot encoded for colinearity):
            - CustomerType (is_grp, is_trn, is_trnP, contract)
            - ReservationStatus (Check-Out, No-Show, Canceled)
            - MarketSegment(Corp/Direct/Group/OfflineTA/OnlineTA)
            - DistributionChannel (Direct, TA/TO)
            - Meal (Bed&Breakfast, Halfboard (breakfast + 1 meal), FB (full board))
            - DepositType (Refundable, Non-Refundable)
            - Country (3-digit ISO 3166 Country Codes, top 10 + 'other')
            - AgencyBooking (True/False)
            - CompanyListed (True/False)

    """
    # add unique reservation ID num
    res_nums = list(range(len(df_res)))
    df_res.insert(0, "ResNum", res_nums)

    # add 'PreviousBookings'
    df_res["PreviousBookings"] = (
        df_res.PreviousBookingsNotCanceled + df_res.PreviousCancellations
    )

    # add ResMadeDate
    df_res["ResMadeDate"] = df_res.ArrivalDate - df_res["LeadTime"].map(
        datetime.timedelta
    )

    # add NumPeople
    df_res["Children"] = df_res["Children"].fillna(0)
    df_res["NumPeople"] = df_res.Adults + df_res.Children + df_res.Babies

    # one-hot-encode CustomerType
    df_res[["CT_is_grp", "CT_is_trn", "CT_is_trnP"]] = pd.get_dummies(
        df_res.CustomerType, drop_first=True
    )

    # one-hot-encode ResStatus (IsCanceled already included, so only keeping no-show (checkout can be inferred))
    df_res[["RS_CheckOut", "RS_No-Show"]] = pd.get_dummies(
        df_res.ReservationStatus, drop_first=True
    )
    df_res.drop(columns=["RS_CheckOut"], inplace=True)

    # one-hot-encode MarketSegment (done this way since h1 & h2 have different MktSegs
    mkt_seg_cols = list(pd.get_dummies(df_res.MarketSegment, drop_first=True).columns)
    mkt_seg_cols = ["MS_" + ms_name for ms_name in mkt_seg_cols]
    df_res[mkt_seg_cols] = pd.get_dummies(df_res.MarketSegment, drop_first=True)

    # one-hot-encode DistributionChannel (same situation as MktSeg comment above)
    dist_channel_cols = list(
        pd.get_dummies(df_res.DistributionChannel, drop_first=True).columns
    )
    dist_channel_cols = ["DC_" + channel_name for channel_name in dist_channel_cols]
    df_res[dist_channel_cols] = pd.get_dummies(
        df_res.DistributionChannel, drop_first=True
    )

    # one-hot-encode meal (same situation as above)
    meal_cols = list(pd.get_dummies(df_res.Meal, drop_first=True).columns)
    meal_cols = ["MEAL_" + ms_name.strip() for ms_name in meal_cols]
    df_res[meal_cols] = pd.get_dummies(df_res.Meal, drop_first=True)

    # one-hot-encode Country
    top_ten_countries = list(
        df_res.Country.value_counts().sort_values(ascending=False).head(10).index
    )

    for country in top_ten_countries:
        df_res["FROM_" + country] = df_res.Country == country

    df_res["FROM_other"] = ~df_res.Country.isin(top_ten_countries)

    # one-hot-encode DepositType
    df_res[["DT_NonRefundable", "DT_Refundable"]] = pd.get_dummies(
        df_res.DepositType, drop_first=True
    )

    # Boolean columns (AgencyBooking & CompanyListed)
    df_res["AgencyBooking"] = ~df_res["Agent"].isnull()
    df_res["CompanyListed"] = ~df_res["Company"].isnull()

    return df_res


def res_to_dbd(df_res, first_date="2015-07-01"):
    """
    Takes a dataFrame (with parsed dates and LOS column) containing a hotel's reservations and
    returns a DataFrame containing nightly hotel room sales.

    Our data is made up of reservations containing 'Arrival Date' and 'Length of Stay'.
    This function is used to determine how many rooms were sold on a given night, accounting for
    guests that arrived previously and are staying multiple nights.

    ____
    Parameters:
        - df_res (pandas.DataFrame, required): reservations DataFrame
        - first_date (str ("%Y-%m-%d"), optional): resulting day-by-days DataFrame will start on this day
    """
    mask = df_res["IsCanceled"] == 0
    df_dates = df_res[mask].copy()

    date = pd.to_datetime(first_date, format="%Y-%m-%d")
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


def add_dbd_columns(df_dbd, capacity):
    """
    Adds several columns to df_dbd, including:
        - 'Occ' (occupancy)
        - 'RevPAR' (revenue per available room)
        - 'ADR' (by segment)
        - 'DOW' (day-of-week)
        - 'WD' (weekday - binary)
        - 'WE' (weekend - binary)
        - 'STLY_Date' (datetime "%Y-%m-%d")

    _____
    Parameters:
        - df_dbd: day-by-day hotel DF
        - capacity (int, required): number of rooms in the hotel
    """
    # add Occ & RevPAR columns'
    df_dbd["Occ"] = round(df_dbd["RoomsSold"] / capacity, 2)
    df_dbd["RevPAR"] = round(df_dbd["RoomRev"] / capacity, 2)

    # Add ADR by segment
    df_dbd["ADR"] = round(df_dbd.RoomRev / df_dbd.RoomsSold, 2)
    df_dbd["Trn_ADR"] = round(df_dbd.Trn_RoomRev / df_dbd.Trn_RoomsSold, 2)
    df_dbd["TrnP_ADR"] = round(df_dbd.TrnP_RoomRev / df_dbd.TrnP_RoomsSold, 2)
    df_dbd["Grp_ADR"] = round(df_dbd.Grp_RoomRev / df_dbd.Grp_RoomsSold, 2)
    df_dbd["Cnt_ADR"] = round(df_dbd.Cnt_RoomRev / df_dbd.Cnt_RoomsSold, 2)

    dow = pd.to_datetime(df_dbd.index, format="%Y-%m-%d")
    dow = dow.strftime("%a")
    df_dbd.insert(0, "DOW", dow)
    df_dbd["WE"] = (df_dbd.DOW == "Fri") | (df_dbd.DOW == "Sat")
    df_dbd["WD"] = df_dbd.WE == False
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
    df_dbd = df_dbd[col_order].copy()

    # add STLY date
    stly_lambda = lambda x: pd.to_datetime(x) + relativedelta(
        years=-1, weekday=pd.to_datetime(x).weekday()
    )
    df_dbd["STLY_Date"] = df_dbd.index.map(stly_lambda)

    df_dbd.fillna(0, inplace=True)

    return df_dbd


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