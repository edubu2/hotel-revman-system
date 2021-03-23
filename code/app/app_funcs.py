import pandas as pd
from app_utils import renamed_user_cols

DATE_FMT = "%Y-%m-%d"


def get_pricing(hotel_num, cols):
    if hotel_num == 1:
        capacity = 187
    else:
        capacity = 226
    df = pd.read_csv(
        f"../../data/results/h{hotel_num}_pricing_v2.csv",
        parse_dates=["AsOfDate", "StayDate"],
    )
    df.index = pd.DatetimeIndex(df.StayDate).date
    df = df.sort_index()
    df["LYA_Occ"] = df["LYA_RoomsSold"] / capacity
    df.drop(columns="Unnamed: 0", inplace=True)
    df = df[cols]

    return df
