"""
This script generates over 700 .csv files containing on the books data 
as of each date from 2016-07-01 to 2017-08-31 for both H1 and H2.

It takes several hours for each run. But once it's saved, the data
can be accessed and manipulated very quickly.

INSTRUCTIONS
-------

Upon initial setup, execute this script twice with different parameters:
    first run: 
        - ensure FOLDER = "./sims.pickle/"
        - ensure START = datetime.date(2015, 8, 1)
        - ensure STOP = datetime.date(2017, 8, 1)
        - ensure PULL_EXTENDED = FALSE
    
    second run:
        - ensure FOLDER = "../data/otb-data/"
        - ensure START = datetime.date(2016, 8, 1)
        - ensure STOP = datetime.date(2017, 8, 1)
        - ensure PULL_EXTENDED = FALSE
        

Execute from the command line, like so:
    > python3 save_sims.py
"""

from collections import defaultdict
import pandas as pd
import numpy as np
import datetime
from sim import generate_simulation
import time

# --- ADJUST THESE VARIABLES FROM STEP 1 - STEP 2 ---
# --- SEE INSTRUCTIONS IN DOCSTRING ---
FOLDER = "../data/otb-data/"
START = datetime.date(2016, 8, 1)
STOP = datetime.date(2017, 8, 1)
PULL_EXTENDED = False  # set to True only for run 2 (see instructions in docstring)

# ---- STOP ----

DATE_FMT = "%Y-%m-%d"
H1_RES = pd.read_pickle("pickle/h1_res.pick")
H2_RES = pd.read_pickle("pickle/h2_res.pick")
H1_DBD = pd.read_pickle("pickle/h1_dbd.pick")
H2_DBD = pd.read_pickle("pickle/h2_dbd.pick")


def save_sim_records(
    df_dbd, df_res, hotel_num, skip_existing=False, pull_extended=False
):

    all_dates = [
        datetime.datetime.strftime(START + datetime.timedelta(days=x), format=DATE_FMT)
        for x in range((STOP - START).days + 1)
    ]

    counter = 1

    for as_of_date in all_dates:

        # if counter % 100 == 0:
        #     time.sleep(60)  # save cpu
        out_file = str(FOLDER + f"h{str(hotel_num)}_sim_{as_of_date}.pick")

        if PULL_EXTENDED:
            df_sim = generate_simulation(
                df_dbd,
                as_of_date,
                hotel_num,
                df_res,
                confusion=False,
                pull_stly=True,
                verbose=0,
                pull_lya=True,
                add_pace=True,
                add_cxl_realized=True,
            )

        else:
            df_sim = generate_simulation(
                df_dbd,
                as_of_date,
                hotel_num,
                df_res,
                confusion=False,
                pull_stly=False,
                verbose=0,
                pull_lya=False,
                add_pace=False,
                add_cxl_realized=False,
            )

        df_sim.to_pickle(out_file)
        counter += 1

        print(f"Saved file {out_file}.")

    return


def main(h1_dbd, h1_res, h2_dbd, h2_res):
    print("Starting hotel 1...")
    save_sim_records(h1_dbd, h1_res, 1)
    print(
        f"Finished retrieving historical OTB records for Hotel 1\n",
        "Sleeping ten seconds for CPU health...",
    )
    # time.sleep(10)
    print("Starting hotel 2...")
    save_sim_records(h2_dbd, h2_res, 2)
    print("All files saved.")
    return


if __name__ == "__main__":
    main(H1_DBD, H1_RES, H2_DBD, H2_RES)
