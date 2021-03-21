stly_cols_agg = [
    "id",
    "AsOfDate",
    "StayDate",
    # "DaysUntilArrival",  # to confirm results - remove once merge working
    "RoomsOTB",
    "ADR_OTB",
    "RevOTB",
    "ACTUAL_RoomsPickup",
    "ACTUAL_ADR_Pickup",
    "ACTUAL_RevPickup",
    "CxlForecast",
    "RemSupply",
    "SellingPrice",
    "Realized_Cxls",
    "TRN_RoomsOTB",
    "TRN_ADR_OTB",
    "TRN_RevOTB",
    "TRN_CxlForecast",
    "ACTUAL_TRN_RoomsPickup",
    "ACTUAL_TRN_ADR_Pickup",
    "ACTUAL_TRN_RevPickup",
    "TM30_RoomsPickup",
    "TM30_ADR_Pickup",
    "TM30_RevPickup",
    "TM30_TRN_RoomsPickup",
    "TM30_TRN_ADR_Pickup",
    "TM30_TRN_RevPickup",
    "TM15_RoomsPickup",
    "TM15_ADR_Pickup",
    "TM15_RevPickup",
    "TM15_TRN_RoomsPickup",
    "TM15_TRN_ADR_Pickup",
    "TM15_TRN_RevPickup",
    "TM05_RoomsPickup",
    "TM05_ADR_Pickup",
    "TM05_RevPickup",
    "TM05_TRN_RoomsPickup",
    "TM05_TRN_ADR_Pickup",
    "TM05_TRN_RevPickup",
]
# new cols start here
ly_cols_agg = [  # must match df_dbd col names
    "RoomsSold",
    "ADR",
    "RoomRev",
    "NumCancels",
    "TRN_RoomsSold",
    "TRN_ADR",
    "TRN_RoomRev",
]

ly_gap_cols = [  # must match df_dbd col names
    "LYA_RoomsSold",
    "LYA_ADR",
    "LYA_RoomRev",
    "LYA_NumCancels",
    "LYA_TRN_RoomsSold",
    "LYA_TRN_ADR",
    "LYA_TRN_RoomRev",
]

ty_gap_cols = [
    "RoomsOTB",
    "ADR_OTB",
    "RevOTB",
    "Realized_Cxls",
    "TRN_RoomsOTB",
    "TRN_ADR_OTB",
    "TRN_RevOTB",
]

new_gap_cols = [
    "OTB_GapToLYA_RoomsSold",
    "OTB_GapToLYA_ADR",
    "OTB_GapToLYA_RoomRev",
    "OTB_GapToLYA_NumCancels",
    "OTB_GapToLYA_TRN_RoomsSold",
    "OTB_GapToLYA_TRN_ADR",
    "OTB_GapToLYA_TRN_RoomRev",
    "OTB_GapToLYA_NONTRN_RoomsSold",
    "OTB_GapToLYA_NONTRN_ADR",
    "OTB_GapToLYA_NONTRN_RoomRev",
]

gap_tuples = list(zip(ly_gap_cols, ty_gap_cols, new_gap_cols))

stly_pace_cols = [
    "RoomsOTB_STLY",
    "ADR_OTB_STLY",
    "RevOTB_STLY",
    "CxlForecast_STLY",
    "RemSupply_STLY",
    "SellingPrice_STLY",
    "TRN_RoomsOTB_STLY",
    "TRN_ADR_OTB_STLY",
    "TRN_RevOTB_STLY",
    "TRN_CxlForecast_STLY",
    "TM30_RoomsPickup_STLY",
    "TM30_ADR_Pickup_STLY",
    "TM30_RevPickup_STLY",
    "TM30_TRN_RoomsPickup_STLY",
    "TM30_TRN_ADR_Pickup_STLY",
    "TM30_TRN_RevPickup_STLY",
    "TM15_RoomsPickup_STLY",
    "TM15_ADR_Pickup_STLY",
    "TM15_RevPickup_STLY",
    "TM15_TRN_RoomsPickup_STLY",
    "TM15_TRN_ADR_Pickup_STLY",
    "TM15_TRN_RevPickup_STLY",
    "TM05_RoomsPickup_STLY",
    "TM05_ADR_Pickup_STLY",
    "TM05_RevPickup_STLY",
    "TM05_TRN_RoomsPickup_STLY",
    "TM05_TRN_ADR_Pickup_STLY",
    "TM05_TRN_RevPickup_STLY",
]

ty_pace_cols = [c[:-5] for c in stly_pace_cols]

pace_tuples = list(zip(ty_pace_cols, stly_pace_cols))

trash_can = [
    "stly_id",
    "TM05_TRNP_RoomsOTB",
    "TM05_TRNP_RevOTB",
    "TM05_GRP_RoomsOTB",
    "TM05_GRP_RevOTB",
    "TM05_CNT_RoomsOTB",
    "TM05_CNT_RevOTB",
    "TM15_TRNP_RoomsOTB",
    "TM15_TRNP_RevOTB",
    "TM15_GRP_RoomsOTB",
    "TM15_GRP_RevOTB",
    "TM15_CNT_RoomsOTB",
    "TM15_CNT_RevOTB",
    "TM30_TRNP_RoomsOTB",
    "TM30_TRNP_RevOTB",
    "TM30_GRP_RoomsOTB",
    "TM30_GRP_RevOTB",
    "TM30_CNT_RoomsOTB",
    "TM30_CNT_RevOTB",
    "TRNP_RevOTB",
    "TRNP_CxlForecast",
    "GRP_RevOTB",
    "GRP_CxlForecast",
    "CNT_RevOTB",
    "CNT_CxlForecast",
    # "TM30_RevOTB",
    # "TM30_TRN_RevOTB",
    # "TM15_RevOTB",
    # "TM15_TRN_RevOTB",
    # "TM05_RevOTB",
    # "TM05_TRN_RevOTB",
    # "TM30_ADR_OTB",
    # "TM30_TRN_ADR_OTB",
    # "TM15_ADR_OTB",
    # "TM15_TRN_ADR_OTB",
    # "TM05_ADR_OTB",
    # "TM05_TRN_ADR_OTB",
]
