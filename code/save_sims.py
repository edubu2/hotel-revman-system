from collections import defaultdict
import pandas as pd
import numpy as np
import datetime
from sim import generate_simulation

h1_res = pd.read_pickle("pickle/h1_res.pick")
h2_res = pd.read_pickle("pickle/h2_res.pick")
h1_dbd = pd.read_pickle("pickle/h1_dbd.pick")
h2_dbd = pd.read_pickle("pickle/h2_dbd.pick")


def save_sim_records(df_dbd, df_res, hotel_num):
    start = datetime.date(2016, 7, 1)
    stop = datetime.date(2017, 8, 1)
    all_dates = [
        datetime.datetime.strftime(
            start + datetime.timedelta(days=x), format="%Y-%m-%d"
        )
        for x in range((stop - start).days + 1)
    ]

    folder = "./sims/"

    for as_of_date in all_dates:
        # aod_dt = pd.to_datetime(as_of_date, format="%Y-%m-%d")
        df_sim = generate_simulation(
            df_dbd, as_of_date, hotel_num, df_res, confusion=False, pull_stly=False
        )
        print(f"Generated df_sim as of {as_of_date}")
        out_file = str(folder + f"h{str(hotel_num)}_sim_{as_of_date}.csv")
        df_sim.to_csv(out_file)
        print(f"Saved file {out_file}.")


def main(h1_dbd, h1_res, h2_dbd, h2_res):
    save_sim_records(h1_dbd, h1_res, 1)
    save_sim_records(h2_dbd, h2_res, 2)


main(h1_dbd, h1_res, h2_dbd, h2_res)
