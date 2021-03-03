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

    rooms_sold = defaultdict(int)

    while date <= end_date:

        date_string = datetime.datetime.strftime(date, format="%Y-%m-%d")
        tminus = 0

        # start on the arrival date and move back
        # to capture ALL reservations touching 'date' (and not just those that arrive on 'date')
        for i in range(max_los + 1):

            date_tminus = datetime.datetime.strftime(
                date - pd.DateOffset(tminus), format="%Y-%m-%d"
            )
            mask = (
                (df_dates.ArrivalDate == date_tminus)
                & (df_dates.LOS >= 1 + tminus)
                & (df_dates.IsCanceled == 0)
            )

            # add rooms_sold
            rooms_sold[date_string] += len(df_dates[mask])
            tminus += 1

        date += delta

    return pd.DataFrame(rooms_sold, index=["RoomsSold"]).transpose()
