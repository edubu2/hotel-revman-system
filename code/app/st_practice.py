import streamlit as st

import numpy as np
import pandas as pd
from app_utils import user_display_cols, model_eval_cols, renamed_user_cols
from app_funcs import get_pricing

# gather data that will be displayed to user
df_display = get_pricing(1, user_display_cols)
df_eval = get_pricing(1, model_eval_cols)

# add title & update rates section
st.title("RevCast")
# show users resulting table
st.write(df_display)

st.header("Evaluating Model Performance")
st.write(df_eval)


# plots
# show 'DaysUntilArrival' vs. 'SellingPrice' (TRN only) (line)
# dow vs. occ (bar)