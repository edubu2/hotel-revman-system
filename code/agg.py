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

from agg_utils import stly_cols_agg, ly_cols_agg, drop_cols_agg, pace_tuples, trash_can

DATE_FMT = "%Y-%m-%d"
SIM_AOD = dt.date(2017, 8, 1) # simulation as-of date
FOLDER = "./sims/pickle/"
H1_CAPACITY = 187
H2_CAPACITY = 226
H1_DBD = pd.read_pickle("./pickle/h1_dbd.pick")
H2_DBD = pd.read_pickle("./pickle/h2_dbd.pick")

def combine_files(hotel_num, sim_aod):
    """Combines all required files in FOLDER into one DataFrame."""
    sim_start = SIM_AOD - pd.DateOffset(365*2) 
    lam_include = lambda x: x[:2] == 'h' + str(hotel_num) and pd.to_datetime(x[7:17]) >= sim_start
    otb_files = [f for f in os.listdir(FOLDER) if lam_include(f)] 
    otb_files.sort()
    df_sim = pd.DataFrame()
    for otb_data in otb_files:
        df_sim = df_sim.append(pd.read_pickle(FOLDER + otb_data))
        
    return df_sim.copy()
    
def add_ty_features(df_sim, df_dbd, capacity):
    """A series of functions that add TY (This Year) features to df_sim."""

    def add_aod(df_sim):
        """Adds "AsOfDate" and "STLY_AsOfDate" columns."""
        def apply_aod(row):
            stay_date = pd.to_datetime(row["Date"])
            stly_stay_date = pd.to_datetime(row["STLY_Date"])
            n_days_b4 = int(row["DaysUntilArrival"])
            as_of_date = pd.to_datetime(stay_date - pd.DateOffset(n_days_b4), format=DATE_FMT)
            stly_as_of_date = pd.to_datetime(stly_stay_date - pd.DateOffset(n_days_b4), format=DATE_FMT)
            return as_of_date, stly_as_of_date

        df_sim[["AsOfDate","STLY_AsOfDate"]] = (df_sim[["Date", "STLY_Date", "DaysUntilArrival"]]
            .apply(apply_aod, axis=1, result_type='expand'))
        
        df_sim.rename(columns={"Date": "StayDate", "STLY_Date": "STLY_StayDate"}, inplace=True)
        
        return df_sim

    def add_rem_supply(df_sim):
        df_sim["RemSupply"] = (
            capacity - df_sim.RoomsOTB.astype(int) + df_sim.CxlForecast.astype(int)
        )
        return df_sim.copy()
    
    def onehot(df_sim):
        ohe_dow = pd.get_dummies(df_sim.DOW, drop_first=True)
        dow_ohe_cols = list(ohe_dow.columns)
        df_sim[dow_ohe_cols] = ohe_dow

        return df_sim.copy()
    
    def add_non_trn(df_sim):
        df_sim["NONTRN_RoomsOTB"] = (
            df_sim.RoomsOTB - df_sim.TRN_RoomsOTB
        )
        df_sim["NONTRN_RevOTB"] = round(df_sim['RevOTB'] - df_sim['TRN_RevOTB'], 2)
        df_sim["NONTRN_ADR_OTB"] = round(df_sim["NONTRN_RevOTB"] / df_sim["NONTRN_RoomsOTB"], 2)
        df_sim["NONTRN_CxlForecast"] = df_sim.CxlForecast - df_sim.TRN_CxlForecast

        return df_sim.copy()
    
    def add_lya(df_sim):
        def apply_ly_cols(row):
            stly_date = row["STLY_StayDate"]
            if pd.to_datetime(stly_date) < dt.date(2015, 8, 1):
                return tuple(np.zeros(len(ly_cols_agg))) # ignore lya if no data
            stly_date_str = dt.datetime.strftime(stly_date, format=DATE_FMT)
            df_lya = list(df_dbd.loc[stly_date_str, ly_cols_agg])
            return tuple(df_lya)

        ly_new_cols = ["LYA_" + col for col in ly_cols_agg]
        df_sim[ly_new_cols] = df_sim[["STLY_StayDate"]].apply(apply_ly_cols, axis=1, result_type="expand")
        df_sim.fillna(0, inplace=True)
        return df_sim.copy()
    
    def add_actuals(df_sim):
        actual_cols = ['RoomsSold', "ADR", "RoomRev", "NumCancels"]
        def apply_ty_actuals(row):
            date = row["StayDate"]
            date_str = dt.datetime.strftime(date, format=DATE_FMT)
            results = list(df_dbd.loc[date_str, actual_cols])
            return tuple(results)

        new_actual_cols = ["ACTUAL_" + col for col in actual_cols]
        df_sim[new_actual_cols] = df_sim[["StayDate"]].apply(apply_ty_actuals, axis=1, result_type="expand")
        df_sim.fillna(0, inplace=True)
        return df_sim
    
    def add_date_cols(df_sim):
        df_sim["WeekNum"] = df_sim.StayDate.dt.isocalendar().week
        return df_sim
    
    def add_tminus(df_sim):
        """
        Adds tminus 5, 15, 30 day pickup statistics. Will pull STLY later on to compare
        recent booking vs last year.
        """
        # first need TRN_ADR
        df_sim["TRN_ADR_OTB"] = round(df_sim["TRN_RevOTB"] / df_sim["TRN_RoomsOTB"])

        # loop thru tminus windows (for total hotel & trn) & count bookings
        tms = ["TM30_", "TM15_", "TM05_"]
        segs = ["", "TRN_"] # "" for total hotel

        for tm in tms:
            for seg in segs:  
                # add tm_seg_adr
                df_sim[tm + seg + "ADR_OTB"] = round(df_sim[tm + seg + "RevOTB"] / df_sim[tm + seg + "RoomsOTB"], 2)
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
            # back to outside loop (iterating thru tms)
            # while we're here, add TM_NONTRN stats (non-transient)
            df_sim[tm +  "NONTRN_RoomsOTB"] = (
                df_sim[tm + "RoomsOTB"]
                - df_sim[tm + "TRN_RoomsOTB"]
            )
            df_sim[tm + "NONTRN_RevOTB"] = (
                df_sim[tm + "RevOTB"]
                - df_sim[tm + "TRN_RevOTB"]
            )
            df_sim[tm + "NONTRN_ADR_OTB"] = round(
                df_sim[tm + "NONTRN_RevOTB"] / df_sim[tm + "NONTRN_RoomsOTB"], 2
            )
            # add TM_NONTRN_OTB Pickup
            df_sim[tm + "NONTRN_RoomsPickup"] = (
                df_sim["NONTRN_RoomsOTB"]
                - df_sim[tm + "NONTRN_RoomsOTB"]
            )
            df_sim[tm + "NONTRN_RevPickup"] = round(
                df_sim["NONTRN_RevOTB"]
                - df_sim[tm + "NONTRN_RevOTB"], 2
            )
            df_sim[tm + "NONTRN_ADR_Pickup"] = (
                df_sim["NONTRN_ADR_OTB"]
                - df_sim[tm + "NONTRN_ADR_OTB"]
            )
        return df_sim.copy()

    def add_lya_gap(df_sim):
        # must be done AFTER NONTRN cols added
        df_sim["RoomsGapToLYA"] = df_sim.LYA_RoomsSold - df_sim.RoomsOTB
        df_sim["RevGapToLYA"] = df_sim.LYA_RoomRev - df_sim.RevOTB
        df_sim["ADR_GapToLYA"] = df_sim.LYA_ADR - df_sim.ADR_OTB
        df_sim["TRN_RoomsGapToLYA"] = df_sim.LYA_TRN_RoomsSold - df_sim.TRN_RoomsOTB
        df_sim["TRN_RevGapToLYA"] = df_sim.LYA_TRN_RoomRev - df_sim.TRN_RevOTB
        df_sim["TRN_ADR_GapToLYA"] = df_sim.LYA_TRN_ADR - df_sim.TRN_ADR_OTB
        df_sim["NONTRN_RoomsGapToLYA"] = df_sim["RoomsGapToLYA"] - df_sim["TRN_RoomsGapToLYA"]
        df_sim["NONTRN_RevGapToLYA"] = df_sim["RevGapToLYA"] - df_sim["TRN_RevGapToLYA"]
        df_sim["NONTRN_ADR_GapToLYA"] = df_sim["ADR_GapToLYA"] - df_sim["TRN_ADR_GapToLYA"]
        return df_sim
    
    # back to main
    funcs = [add_aod, add_rem_supply, onehot, add_non_trn, add_lya, add_actuals, add_date_cols, add_tminus, add_lya_gap]
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
        df_sim_ids = df_sim.AsOfDate.astype(str) + ' - ' + df_sim.StayDate.astype(str)
        df_sim_stly_ids = df_sim.STLY_AsOfDate.astype(str) + ' - ' + df_sim.STLY_StayDate.astype(str)
        df_sim.insert(0, "id", df_sim_ids)
        df_sim.insert(1, "stly_id", df_sim_stly_ids)

        df_sim = df_sim.merge(df_sim[stly_cols_agg], left_on='stly_id', right_on='id', suffixes=(None, "_STLY"))
        return df_sim.copy()

    def add_pace(df_sim):
        # pace_tuples example: ('RoomsOTB', 'RoomsOTB_STLY' )
        for ty_stat, stly_stat in pace_tuples:
            df_sim[ty_stat + '_Pace'] = df_sim[ty_stat] - df_sim[stly_stat]
        
        return df_sim.copy()
    
    df_sim = add_stly_otb(df_sim)
    df_sim = add_pace(df_sim)
    return df_sim

def prep_demand_features(hotel_num):
    """
    Wraps several functions that read OTB historical csv files into a DataFrame (df_sim)
    and adds relevant features that will be used to model demand.
    """
    assert hotel_num in (1, 2), ValueError("hotel_num must be either 1 or 2 (int).")
    if hotel_num == 1:
        capacity = H1_CAPACITY
        df_dbd = H1_DBD
    else:
        capacity = H2_CAPACITY
        df_dbd = H2_DBD
    
    df_sim = combine_files(hotel_num, SIM_AOD)
    df_sim = add_ty_features(df_sim, df_dbd, capacity)
    df_sim = merge_stly(df_sim)
    # drop unnecessary columns
    
    df_sim.drop(columns=trash_can, inplace=True)
    df_sim.fillna(0, inplace=True)
    return df_sim






    
