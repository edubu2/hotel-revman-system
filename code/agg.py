"""This script aggregates the data in ../data/otb-data/ folder
that was generated from save_sims.py.

It then adds in calculated statistics/features that will be used
for modeling demand.
"""

FOLDER = "../data/otb-data/"


def agg_res(hotel_num):
    """
    Combines all files in FOLDER into one DataFrame.
    """
    df = pd.DataFrame()
    for otb_data in h1_files:
        df = df.append(pd.read_pickle(FOLDER + otb_data))
