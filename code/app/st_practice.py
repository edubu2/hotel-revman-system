import streamlit as st

import numpy as np
import pandas as pd

st.title("RightRates")
df = pd.read_csv("../../data/h1_all_sims31.csv")

mask = df.AsOfDate == "2017-08-01"

highlight_cols = [
    "AsOfDate",
    "StayDate",
    "DOW",
    "RemSupply",
    "CxlForecast",
    "TRN_CxlForecast",
    "RoomsOTB",
    "TRN_RoomsOTB",
    # "RoomsOTB_STLY",
    # "TRN_RoomsOTB_STLY",
    "Pace_RoomsOTB",
    "Pace_TRN_RoomsOTB",
    "SellingPrice",
    "Pace_SellingPrice",
    "ACTUAL_RoomsPickup_STLY",
    "ACTUAL_TRN_RoomsPickup_STLY",
    "TRN_RoomsOTB_STLY",
    "Pace_TRN_RoomsOTB",
    "Pace_TM30_TRN_RoomsPickup",
    "Pace_TM15_TRN_RoomsPickup",
    "Pace_TM05_TRN_RoomsPickup",
]
st.table(df[mask][highlight_cols])


# plots
# show 'DaysUntilArrival' vs. 'SellingPrice' (TRN only) (line)
# dow vs. occ (bar)