stly_cols = [  # each must exist in all csv file column names (prefixes will be added)
    "RoomsOTB",
    "ADR_OTB",
    "RevOTB",
    "SellingPrice",
    "TM05_RoomsOTB",
    "TM15_ADR_OTB",
    "TM05_RevOTB",
    "TM15_RoomsOTB",
    "TM05_ADR_OTB",
    "TM15_RevOTB",
    "TM30_RoomsOTB",
    "TM30_ADR_OTB",
    "TM30_RevOTB",
    "TRN_RoomsOTB",
    "TRN_ADR_OTB",
    "TRN_RevOTB",
    "TM05_TRN_RoomsOTB",
    "TM05_TRN_ADR_OTB",
    "TM05_TRN_RevOTB",
    "TM15_TRN_RoomsOTB",
    "TM15_TRN_ADR_OTB",
    "TM15_TRN_RevOTB",
    "TM30_TRN_RoomsOTB",
    "TM30_TRN_ADR_OTB",
    "TM30_TRN_RevOTB",
    "TRNP_RoomsOTB",
    "TRNP_ADR_OTB",
    "TRNP_RevOTB",
    "TM05_TRNP_RoomsOTB",
    "TM05_TRNP_ADR_OTB",
    "TM05_TRNP_RevOTB",
    "TM15_TRNP_RoomsOTB",
    "TM15_TRNP_ADR_OTB",
    "TM15_TRNP_RevOTB",
    "TM30_TRNP_RoomsOTB",
    "TM30_TRNP_ADR_OTB",
    "TM30_TRNP_RevOTB",
    "GRP_RoomsOTB",
    "GRP_ADR_OTB",
    "GRP_RevOTB",
    "CNT_RoomsOTB",
    "CNT_ADR_OTB",
    "CNT_RevOTB",
]

ly_cols = [  # must match df_dbd col names
    "RoomsSold",
    "ADR",
    "RoomRev",
    "NumCancels",
    "TRN_RoomsSold",
    "TRN_ADR",
    "TRN_RoomRev",
    "GRP_RoomsSold",
    "GRP_ADR",
    "GRP_RoomRev",
    "TRNP_RoomsSold",
    "TRNP_ADR",
    "TRNP_RoomRev",
    "CNT_RoomsSold",
    "CNT_ADR",
    "CNT_RoomRev",
]

tm_cols = [  # new col names (prefixes will be added) (don't include pickup cols here)
    "RoomsOTB",
    # "ADR_OTB",
    "RevOTB",
    "TRN_RoomsOTB",
    # "TRN_ADR_OTB",
    "TRN_RevOTB",
    "TRNP_RoomsOTB",
    # "TRNP_ADR_OTB",
    "TRNP_RevOTB",
    "GRP_RoomsOTB",
    # "GRP_ADR_OTB",
    "GRP_RevOTB",
    "CNT_RoomsOTB",
    # "CNT_ADR_OTB",
    "CNT_RevOTB",
]

ty_pace_cols = [
    "RoomsOTB",
    # "ADR_OTB",
    "RevOTB",
    "TRN_RoomsOTB",
    # "TRN_ADR_OTB",
    "TRN_RevOTB",
    "TM30_RoomsPickup",
    # "TM30_ADR_Pickup",
    "TM30_RevPickup",
    "TM15_RoomsPickup",
    # "TM15_ADR_Pickup",
    "TM15_RevPickup",
    "TM05_RoomsPickup",
    # "TM05_ADR_Pickup",
    "TM05_RevPickup",
    "TM30_TRN_RoomsPickup",
    # "TM30_TRN_ADR_Pickup",
    "TM30_TRN_RevPickup",
    "TM15_TRN_RoomsPickup",
    # "TM15_TRN_ADR_Pickup",
    "TM15_TRN_RevPickup",
    "TM05_TRN_RoomsPickup",
    # "TM05_TRN_ADR_Pickup",
    "TM05_TRN_RevPickup",
    "TM30_NONTRN_RoomsPickup",
    # "TM30_NONTRN_ADR_Pickup",
    "TM30_NONTRN_RevPickup",
    "TM15_NONTRN_RoomsPickup",
    # "TM15_NONTRN_ADR_Pickup",
    "TM15_NONTRN_RevPickup",
    "TM05_NONTRN_RoomsPickup",
    # "TM05_NONTRN_ADR_Pickup",
    "TM05_NONTRN_RevPickup",
]

stly_pace_cols = ["STLY_" + col for col in ty_pace_cols]


def merge(list1, list2):
    """Merges two lists into a list of tuples. Must be same length."""
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
    return merged_list


pace_tuples = merge(ty_pace_cols, stly_pace_cols)
