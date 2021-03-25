stly_cols = [  # each must exist in all csv file column names (prefixes will be added)
    "RoomsOTB",
    "RevOTB",
    "SellingPrice",
    "TM05_RoomsOTB",
    "TM05_RevOTB",
    "TM15_RoomsOTB",
    "TM15_RevOTB",
    "TM30_RoomsOTB",
    "TM30_RevOTB",
    "TRN_RoomsOTB",
    "TRN_RevOTB",
    "TM05_TRN_RoomsOTB",
    "TM05_TRN_RevOTB",
    "TM15_TRN_RoomsOTB",
    "TM15_TRN_RevOTB",
    "TM30_TRN_RoomsOTB",
    "TM30_TRN_RevOTB",
    "TRNP_RoomsOTB",
    "TRNP_RevOTB",
    "TM05_TRNP_RoomsOTB",
    "TM05_TRNP_RevOTB",
    "TM15_TRNP_RoomsOTB",
    "TM15_TRNP_RevOTB",
    "TM30_TRNP_RoomsOTB",
    "TM30_TRNP_RevOTB",
    "GRP_RoomsOTB",
    "GRP_RevOTB",
    "CNT_RoomsOTB",
    "CNT_RevOTB",
]

ly_cols = [  # must match df_dbd col names
    "RoomsSold",
    "RoomRev",
    "NumCancels",
    "TRN_RoomsSold",
    "TRN_RoomRev",
    "GRP_RoomsSold",
    "GRP_RoomRev",
    "TRNP_RoomsSold",
    "TRNP_RoomRev",
    "CNT_RoomsSold",
    "CNT_RoomRev",
]

tm_cols = [  # new col names (prefixes will be added) (don't include pickup cols here)
    "RoomsOTB",
    "RevOTB",
    "TRN_RoomsOTB",
    "TRN_RevOTB",
    "TRNP_RoomsOTB",
    "TRNP_RevOTB",
    "GRP_RoomsOTB",
    "GRP_RevOTB",
    "CNT_RoomsOTB",
    "CNT_RevOTB",
]

ty_pace_cols = [
    "RoomsOTB",
    "RevOTB",
    "TRN_RoomsOTB",
    "TRN_RevOTB",
    "TRNP_RoomsOTB",
    "TRNP_RevOTB",
]

stly_pace_cols = ["STLY_" + col for col in ty_pace_cols]


def merge(list1, list2):
    """Merges two lists into a list of tuples. Must be same length."""
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
    return merged_list


pace_tuples = merge(ty_pace_cols, stly_pace_cols)
