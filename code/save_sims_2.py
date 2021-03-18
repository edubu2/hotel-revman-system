"""
This script generates over 700 .csv files containing on the books data 
as of each date from 2017-07-01 to 2017-08-31 for both H1 and H2. generate_simulation
will use the resulting files of save_sims_1.py when used with the parameters set forth
below. 

This script also takes several hours, but upon completion our data will be easily
accessible.

It takes several hours for each run (very CPU intensive). I highly recommend using a 
virtual machine. 
-------

Upon initial setup, after saving df_res and df_dbd to pickle files, run this script
from your terminal.

Ensure the global params are as follows: 
FOLDER = "../data/otb-data/"
START = datetime.date(2016, 8, 1)
STOP = datetime.date(2017, 8, 1)
PULL_EXTENDED = True
        

Execute from the command line, like so:
    > python3 save_sims.py
"""

from collections import defaultdict
import pandas as pd
import numpy as np
import datetime
from sim import generate_simulation
import time

FOLDER = "../data/otb-data/"
START = datetime.date(2016, 8, 1)
STOP = datetime.date(2017, 8, 1)
PULL_EXTENDED = True  

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
