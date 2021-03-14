"""
This script generates over 700 .csv files containing on the books data 
as of each date from 2016-07-01 to 2017-08-31 for both H1 and H2.
"""

from collections import defaultdict
import pandas as pd
import numpy as np
import datetime
from sim import generate_simulation
import time

DATE_FMT = "%Y-%m-%d"
H1_RES = pd.read_pickle("pickle/h1_res.pick")
H2_RES = pd.read_pickle("pickle/h2_res.pick")
H1_DBD = pd.read_pickle("pickle/h1_dbd.pick")
H2_DBD = pd.read_pickle("pickle/h2_dbd.pick")


def save_sim_records(df_dbd, df_res, hotel_num, skip_existing=False):
    start = datetime.date(2016, 7, 1)
    stop = datetime.date(2017, 8, 1)
    all_dates = [
        datetime.datetime.strftime(start + datetime.timedelta(days=x), format=DATE_FMT)
        for x in range((stop - start).days + 1)
    ]

    folder = "./sims/"

    for as_of_date in all_dates:
        # aod_dt = pd.to_datetime(as_of_date, format=DATE_FMT)
        df_sim = generate_simulation(
            df_dbd,
            as_of_date,
            hotel_num,
            df_res,
            confusion=False,
            pull_stly=False,
            verbose=0,
        )
        print(f"Generated df_sim as of {as_of_date}")
        out_file = str(folder + f"h{str(hotel_num)}_sim_{as_of_date}.csv")
        df_sim.to_csv(out_file)

        print(f"Saved file {out_file}.")

    return


def main(h1_dbd, h1_res, h2_dbd, h2_res, skip_existing):
    save_sim_records(h1_dbd, h1_res, 1)
    print(
        f"Finished retrieving historical OTB records for Hotel 1\n",
        "Sleeping ten seconds for CPU health.",
    )
    time.sleep(10)
    save_sim_records(h2_dbd, h2_res, 2)


main(H1_DBD, H1_RES, H2_DBD, H2_RES)
