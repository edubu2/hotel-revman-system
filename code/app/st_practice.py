import streamlit as st

import numpy as np
import pandas as pd
from app_utils import user_display_cols, model_eval_cols

aod = st.date_input("StayDate for Rate Change")
rate_update = st.number_input("Enter new rate.")

st.title("RightRates")
df = pd.read_csv("../../data/results/h1_pricing.csv")

mask = df.AsOfDate == "2017-08-01"

st.table(df[mask][user_display_cols])


# plots
# show 'DaysUntilArrival' vs. 'SellingPrice' (TRN only) (line)
# dow vs. occ (bar)