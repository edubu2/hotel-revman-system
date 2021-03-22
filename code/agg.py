"""This script aggregates the data in ../data/otb-data/ folder
that was generated from save_sims.py.

It then adds in calculated statistics/features that will be used
for modeling demand.

Note: must be at least 364 days of actuals for training; the rest
future-looking starting with t+0 on AOD (as-of date).
"""

import datetime as dt
import pandas as pd
import numpy as np
import os

from agg_utils import stly_cols_agg, ly_cols_agg, trash_can, pace_tuples, gap_tuples

DATE_FMT = "%Y-%m-%d"
SIM_AOD = dt.date(2017, 8, 1)  # simulation as-of date
FOLDER = "./sims2/"
H1_CAPACITY = 187
H2_CAPACITY = 226
H1_DBD = pd.read_pickle("./pickle/h1_dbd.pick")
H2_DBD = pd.read_pickle("./pickle/h2_dbd.pick")


def combine_files(hotel_num, sim_aod, prelim_csv_out=None):
    """Combines all required files in FOLDER into one DataFrame."""
    sim_start = SIM_AOD - pd.DateOffset(365 * 2)
    lam_include = (
        lambda x: x[:9] == "h" + str(hotel_num) + "_sim_20"
        and pd.to_datetime(x[7:17]) >= sim_start
        and x[7] == "2"
    )
    otb_files = [f for f in os.listdir(FOLDER) if lam_include(f)]
    otb_files.sort()
    df_sim = pd.DataFrame()
    for otb_data in otb_files:
        df_sim = df_sim.append(pd.read_pickle(FOLDER + otb_data))
    if prelim_csv_out is not None:
        df_sim.to_csv(prelim_csv_out)
        print(f"'{prelim_csv_out}' file saved.")

    return df_sim.copy()


def extract_features(df_sim, df_dbd, capacity):
    """A series of functions that add TY (This Year) features to df_sim."""

    def add_aod(df_sim):
        """Adds "AsOfDate" and "STLY_AsOfDate" columns."""

        def apply_aod(row):
            stay_date = row["Date"]
            stly_stay_date = pd.to_datetime(row["STLY_Date"])
            n_days_b4 = int(row["DaysUntilArrival"])
            as_of_date = pd.to_datetime(
                stay_date - pd.DateOffset(n_days_b4), format=DATE_FMT
            )
            stly_as_of_date = pd.to_datetime(
                stly_stay_date - pd.DateOffset(n_days_b4), format=DATE_FMT
            )
            return as_of_date, stly_as_of_date

        df_sim[["AsOfDate", "STLY_AsOfDate"]] = df_sim[
            ["Date", "STLY_Date", "DaysUntilArrival"]
        ].apply(apply_aod, axis=1, result_type="expand")

        df_sim.rename(
            columns={"Date": "StayDate", "STLY_Date": "STLY_StayDate"}, inplace=True
        )

        return df_sim.copy()

    def onehot(df_sim):
        ohe_dow = pd.get_dummies(df_sim["DayOfWeek"], drop_first=True)
        dow_ohe_cols = list(ohe_dow.columns)
        df_sim[dow_ohe_cols] = ohe_dow

        return df_sim.copy()

    def add_date_info(df_sim):
        df_sim["MonthNum"] = df_sim.StayDate.dt.month
        df_sim["DayOfWeek"] = df_sim.StayDate.map(
            lambda x: dt.datetime.strftime(x, format="%a")
        )
        # add one-hot-encoded date columns
        df_sim = onehot(df_sim)
        is_weekend = (
            (df_sim.Mon == 0)
            & (df_sim.Wed == 0)
            & (df_sim.Tue == 0)
            & (df_sim.Mon == 0)
            & (df_sim.Sun == 0)
            & (df_sim.Thu == 0)
        )
        days = ["Mon", "Tue", "Wed", "Thu", "Sat", "Sun"]
        for d in days:
            df_sim[d] = df_sim[d].astype("bool")
        df_sim["WE"] = is_weekend
        df_sim["week_of_year"] = df_sim.StayDate.map(lambda x: x.weekofyear).astype(
            float
        )
        return df_sim

    def add_rem_supply(df_sim):
        df_sim["RemSupply"] = (
            capacity - df_sim.RoomsOTB.astype(int) + df_sim.CxlForecast.astype(int)
        )
        return df_sim.copy()

    def add_lya(df_sim):
        def apply_ly_cols(row):
            stly_date = row["STLY_StayDate"]
            if pd.to_datetime(stly_date) < dt.date(2015, 8, 1):
                return tuple(np.zeros(len(ly_cols_agg)))  # ignore lya if no data
            stly_date_str = dt.datetime.strftime(stly_date, format=DATE_FMT)
            df_lya = list(df_dbd.loc[stly_date_str, ly_cols_agg])
            return tuple(df_lya)

        # first need ADR OTB
        df_sim["ADR_OTB"] = round(df_sim["RevOTB"] / df_sim["RoomsOTB"], 2)
        df_sim["TRN_ADR_OTB"] = round(df_sim["TRN_RevOTB"] / df_sim["TRN_RoomsOTB"], 2)
        df_sim["TRNP_ADR_OTB"] = round(
            df_sim["TRNP_RevOTB"] / df_sim["TRNP_RoomsOTB"], 2
        )

        ly_new_cols = ["LYA_" + col for col in ly_cols_agg]
        df_sim[ly_new_cols] = df_sim[["STLY_StayDate"]].apply(
            apply_ly_cols, axis=1, result_type="expand"
        )
        df_sim.fillna(0, inplace=True)
        return df_sim.copy()

    def add_actuals(df_sim):
        actual_cols = [
            "RoomsSold",
            "ADR",
            "RoomRev",
            "TRN_RoomsSold",
            "TRN_ADR",
            "TRN_RoomRev",
            "TRNP_RoomsSold",
            "TRNP_ADR",
            "TRNP_RoomRev",
            "NumCancels",
        ]

        def apply_ty_actuals(row):
            date = row["StayDate"]
            date_str = dt.datetime.strftime(date, format=DATE_FMT)
            results = list(df_dbd.loc[date_str, actual_cols])
            return tuple(results)

        new_actual_cols = ["ACTUAL_" + col for col in actual_cols]
        df_sim[new_actual_cols] = df_sim[["StayDate"]].apply(
            apply_ty_actuals, axis=1, result_type="expand"
        )

        # add actual pickup
        df_sim["ACTUAL_RoomsPickup"] = df_sim["ACTUAL_RoomsSold"] - df_sim["RoomsOTB"]
        df_sim["ACTUAL_ADR_Pickup"] = df_sim["ACTUAL_ADR"] - df_sim["ADR_OTB"]
        df_sim["ACTUAL_RevPickup"] = df_sim["ACTUAL_RoomRev"] - df_sim["RevOTB"]

        df_sim["ACTUAL_TRN_RoomsPickup"] = (
            df_sim["ACTUAL_TRN_RoomsSold"] - df_sim["TRN_RoomsOTB"]
        )
        df_sim["ACTUAL_TRN_ADR_Pickup"] = (
            df_sim["ACTUAL_TRN_ADR"] - df_sim["TRN_ADR_OTB"]
        )
        df_sim["ACTUAL_TRN_RevPickup"] = round(
            df_sim["ACTUAL_TRN_RoomRev"] - df_sim["TRN_RevOTB"], 2
        )

        df_sim["ACTUAL_TRNP_RoomsPickup"] = (
            df_sim["ACTUAL_TRNP_RoomsSold"] - df_sim["TRNP_RoomsOTB"]
        )
        df_sim["ACTUAL_TRNP_ADR_Pickup"] = (
            df_sim["ACTUAL_TRNP_ADR"] - df_sim["TRNP_ADR_OTB"]
        )
        df_sim["ACTUAL_TRNP_RevPickup"] = round(
            df_sim["ACTUAL_TRNP_RoomRev"] - df_sim["TRNP_RevOTB"], 2
        )

        df_sim.fillna(0, inplace=True)

        return df_sim

    def add_tminus(df_sim):
        """
        Adds tminus 5, 15, 30 day pickup statistics. Will pull STLY later on to compare
        recent booking vs last year.
        """

        # loop thru tminus windows (for total hotel & TRN) & count bookings
        tms = ["TM30_", "TM15_", "TM05_"]
        segs = ["", "TRN_", "TRNP_"]  # "" for total hotel

        for tm in tms:
            for seg in segs:
                # add tm_seg_adr
                df_sim[tm + seg + "ADR_OTB"] = round(
                    df_sim[tm + seg + "RevOTB"] / df_sim[tm + seg + "RoomsOTB"], 2
                )
                # and now segmented stats
                df_sim[tm + seg + "RoomsPickup"] = round(
                    df_sim[seg + "RoomsOTB"] - df_sim[tm + seg + "RoomsOTB"], 2
                )
                df_sim[tm + seg + "RevPickup"] = round(
                    df_sim[seg + "RevOTB"] - df_sim[tm + seg + "RevOTB"], 2
                )
                df_sim[tm + seg + "ADR_Pickup"] = round(
                    df_sim[seg + "ADR_OTB"] - df_sim[tm + seg + "ADR_OTB"], 2
                )
        return df_sim.copy()

    def add_gaps(df_sim):
        # add gap to lya cols
        for lya, ty_otb, new_col in gap_tuples:
            df_sim[new_col] = df_sim[lya] - df_sim[ty_otb]
        return df_sim

    # back to main
    funcs = [
        add_aod,
        add_rem_supply,
        add_lya,
        add_actuals,
        add_date_info,
        add_tminus,
        add_gaps,
    ]
    for func in funcs:
        df_sim = func(df_sim)

    return df_sim.copy()


def merge_stly(df_sim):
    """
    For each as_of_date + stay_date combo in df_sim from 2016-08-02 to 2017-08-31,
    pull the corresponding 2015-2016 (same-time last year, adjusted for DOW).
    """

    def add_stly_otb(df_sim):
        # first, create unique ID col (id) for each as-of-date/stay-date combo
        # then, add a stly_id column that we can use as left key for our merge (self-join)
        df_sim_ids = df_sim.AsOfDate.astype(str) + " - " + df_sim.StayDate.astype(str)
        df_sim_stly_ids = (
            df_sim.STLY_AsOfDate.astype(str) + " - " + df_sim.STLY_StayDate.astype(str)
        )
        df_sim.insert(0, "id", df_sim_ids)
        df_sim.insert(1, "stly_id", df_sim_stly_ids)

        df_sim = df_sim.merge(
            df_sim[stly_cols_agg],
            left_on="stly_id",
            right_on="id",
            suffixes=(None, "_STLY"),
        )
        return df_sim.copy()

    def add_pace(df_sim):
        # pace_tuples example: ('RoomsOTB', 'RoomsOTB_STLY' )
        for ty_stat, stly_stat in pace_tuples:
            new_stat_name = ty_stat
            if ty_stat[:8] == ["ACTUAL_"]:
                new_stat_name = ty_stat[8:]
            df_sim["Pace_" + new_stat_name] = df_sim[ty_stat] - df_sim[stly_stat]

        return df_sim.copy()

    df_sim = add_stly_otb(df_sim)
    df_sim = add_pace(df_sim)
    return df_sim


def cleanup_sim(df_sim):
    df_sim["RemSupply"] = df_sim["RemSupply"].astype(float)
    df_sim["RemSupply_STLY"] = df_sim["RemSupply_STLY"].astype(float)
    df_sim["Realized_Cxls"] = df_sim["Realized_Cxls"].astype(float)
    df_sim["Realized_Cxls_STLY"] = df_sim["Realized_Cxls_STLY"].astype(float)
    df_sim["DaysUntilArrival"] = df_sim["DaysUntilArrival"].astype(float)
    df_sim["Pace_RemSupply"] = df_sim["Pace_RemSupply"].astype(float)
    df_sim["SellingPrice"] = round(df_sim["SellingPrice"], 2)
    df_sim["Pace_SellingPrice"] = round(df_sim["Pace_SellingPrice"], 2)
    return df_sim


def prep_demand_features(hotel_num, prelim_csv_out=None, results_csv_out=None):
    """
    Wraps several functions that read OTB historical csv files into a DataFrame (df_sim)
    and adds relevant features that will be used to model demand & recommend pricing.

    Parameters:
        - hotel_num (int, required): 1 or 2
        - prelim_csv_out (str, optional): output the pre-processed dataFrame (raw) to this filepath.
        - results_csv_out (str, optional): Save resulting data as csv with given filepath.

    Returns
        - df_sim
    """
    assert hotel_num in (1, 2), ValueError("hotel_num must be either 1 or 2 (int).")
    if hotel_num == 1:
        capacity = H1_CAPACITY
        df_dbd = H1_DBD
    else:
        capacity = H2_CAPACITY
        df_dbd = H2_DBD

    df_sim = combine_files(hotel_num, SIM_AOD, prelim_csv_out=prelim_csv_out)
    df_sim = extract_features(df_sim, df_dbd, capacity)
    df_sim = merge_stly(df_sim)
    df_sim = cleanup_sim(df_sim)

    # drop unnecessary columns
    df_sim.drop(columns=trash_can, inplace=True)
    df_sim.fillna(0, inplace=True)
    if results_csv_out is not None:
        df_sim.to_csv(results_csv_out)
        print(f"'{results_csv_out}' file saved.")
    return df_sim.dropna()
