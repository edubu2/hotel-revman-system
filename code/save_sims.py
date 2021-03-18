"""
ON-THE-BOOKS DATA SERIALIZATION SCRIPTS - PART 1 OF 2

(REQUIRED TO REPRODUCE LOCALLY)

INTRODUCTION
-------

The 'save_sims' ('_1' and '_2') scripts capture future-looking on the books (OTB)
data as_of every date in our sample that has one year of history and serializes them for
later ease of use using Pickle. This means the first year of data cannot be used
directly in the output, but data from that year will be contained in the STLY (same-time-
last-year) features of the save_sims_2 output. The scripts need to be run in order. I am
currently experimenting with starting save_sims_2.py once save_sims_1.py is halfway done.
I will update this when I can confirm it works. It will be extremely CPU intensive and
should not be attempted on a personal computer (unless you have state of the art hardware).

Hyperthreading should be enabled.

This script generates nearly 1000 .csv files containing on the books data 
as of each date from 2015-07-01 to 2017-08-31 for both H1 and H2. 

The subsequent save_sims_2.py script will pull STLY data from 2017-08-01 to
2018-08-01 (hence the need for OTB data from 2015-07-01 - 2016-08-31)

INSTRUCTIONS
------
Upon completion, run save_sims_2.py. save_sims_2.py is nearly identical to this
script, but passes different parameters into generate_simulation function with each
iteration.

It takes several hours for each script to run (very CPU intensive). Once the files are saved,
the data can be accessed and manipulated very quickly.

We need to do this in order to quickly pull STLY OTB data for future simulations.
-------

Upon initial setup, after saving df_res and df_dbd to pickle files *, run this script
from your terminal.

* df_res and df_sim are generated from the 'dbds.py' script. Replace 'df' with 'h<n>';
replace '<n>' with the hotel number. 

Ensure the global params are as follows: 
FOLDER = "./sims/pickle/"
START = datetime.date(2015, 8, 1)
STOP = datetime.date(2017, 8, 1)
PULL_EXTENDED = False
        

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
FOLDER = "./sims/pickle/"
START = datetime.date(2015, 8, 1)
STOP = datetime.date(2017, 8, 31)
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
