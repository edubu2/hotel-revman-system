"""This script aggregates the data in ../data/otb-data/ folder
that was generated from save_sims 1 & 2.

It then adds in calculated statistics/features that will be used
for modeling demand.
"""

import datetime as dt
import pandas as pd
import numpy as np

from sim_utils import ly_cols

SIM_AOD = dt.date(2017, 8, 1)
DATE_FMT = "%Y-%m-%d"
SIM_AOD_STR = dt.datetime.strftime(SIM_AOD, format=DATE_FMT)
FOLDER = "../data/otb-data/"
FILEPATH = FOLDER + "h{}_sim_{}.pick" # {} will be formatted to contain hotel_num, date


def combine_files(hotel_num, sim_aod):
    """
    Combines all files in FOLDER into one DataFrame.
    """
    df = pd.DataFrame()
    for otb_data in h1_files:
        df = df.append(pd.read_pickle(FOLDER + otb_data))
    
def 
