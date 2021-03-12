import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import *
from collections import defaultdict
from model_cancellations import split_reservations, predict_cancellations

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
    print("Parsing reservations...")
    df_dates = df_future_res.copy()
    date = pd.to_datetime(as_of_date, format="%Y-%m-%d")
    end_date = datetime.date(2017, 8, 31)
    delta = datetime.timedelta(days=1)
    max_los = int(df_dates["LOS"].max())

    nightly_stats = {}

    while date <= end_date:

        date_string = datetime.datetime.strftime(date, format="%Y-%m-%d")

        # initialize date dict, which will go into nightly_stats as {'date': {'stat': 'val', 'stat', 'val'}}
        date_stats = defaultdict(int)

        mask = (
            (df_dates.ArrivalDate <= date_string)
            & (df_dates.ResMadeDate <= date_string)
            & (df_dates.CheckoutDate > date_string)
        ) & (
            (df_dates.IsCanceled == 0)
            | (
                (  # only cxls if they have not been canceled yet
                    (df_dates.IsCanceled == 1)
                    & (df_dates.ReservationStatusDate >= date_string)
                )
            )
        )

        night_df = df_dates[mask].copy()
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

    # add STLY date
    stly_lambda = lambda x: pd.to_datetime(x) + relativedelta(
        years=-1, weekday=pd.to_datetime(x).weekday()
    )
    df_sim["STLY_Date"] = df_sim.index.map(stly_lambda)

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
        stly_future_res = predict_cancellations(
            df_res, stly_date_str, hotel_num, confusion=False
        )
        stly_sim = setup_sim(stly_future_res, stly_date_str)
        stly_sim = add_sim_cols(stly_sim, df_dbd, capacity)

        STLY_OTB = stly_sim.loc[stly_date_str, "RoomsOTB"]
        STLY_REV = stly_sim.loc[stly_date_str, "RevOTB"]
        STLY_ADR = round(STLY_REV / STLY_OTB, 2)
        STLY_CxlForecast = stly_sim.loc[stly_date_str, "CxlForecast"]

        return STLY_OTB, STLY_REV, STLY_ADR, STLY_CxlForecast

    num_models = len(df_sim)
    print(f"Training {num_models} models to obtain STLY statistics...\n")

    new_col_names = ["STLY_OTB", "STLY_Rev", "STLY_ADR", "STLY_CxlForecast"]
    df_sim[new_col_names] = df_sim.apply(
        apply_STLY_stats, result_type="expand", axis="columns"
    )
    print("STLY statistics obtained.")
    return df_sim


def add_pricing(df_sim):
    """
    Adds 'SellingPrice' column to df_sim.

    Contains the average rate for all booked reservations during a given week (WD/WE).
    This gives us an indication of what the hotel's online selling prices.
    """
    # get average WD/WE pricing for each week
    df_sim.index = pd.to_datetime(df_sim.index)
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
    n = len(df_sim)
    print(f"Ignore red warning below. Operation only being applied to {n} rows.")
    df_sim["WeekEndDate"] = df_sim.index + pd.DateOffset(weekday=6)

    # apply the weekly WD/WE prices to the original df_sim
    def apply_rates(row):
        wd = row["WD"] == 1
        date = datetime.datetime.strftime(row.name, format="%Y-%m-%d")
        week_end = datetime.datetime.strftime(row.WeekEndDate, format="%Y-%m-%d")
        mask = df_pricing.WD == wd
        price = df_pricing[mask].loc[week_end, "Trn_ADR_OTB"]
        return price

    print("Estimating selling prices...")
    df_sim["SellingPrice"] = df_sim.apply(apply_rates, axis=1)
    print("Estimated selling prices obtained.")

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
    aod_dt = pd.to_datetime(as_of_date)
    min_dt = datetime.date(2016, 7, 1)
    assert aod_dt > min_dt, "as_of_date must be between 7/1/16 and 8/30/17"
    print("Creating crystal ball...")
    print("Predicting future cancellations...")
    df_future_res = predict_cancellations(df_res, as_of_date, hotel_num)

    if hotel_num == 1:
        capacity = h1_capacity
    else:
        capacity = h2_capacity

    "Setting up simulation..."
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
    df_sim = add_pricing(df_sim)
    print("The simulation has begun.")

    return df_sim