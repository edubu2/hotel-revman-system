"""This script simply contains the feature columns for the various models that can be easily imported into Jupyter Notebooks."""

# Features for hotel 1 cancellation forecast
X1_cxl_cols = [
    "LeadTime",
    "LOS",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "ADR",
    "NumPeople",
    "Adults",
    "Children",
    "Babies",
    "TotalOfSpecialRequests",
    "PreviousBookings",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "BookingChanges",
    "DaysInWaitingList",
    "RequiredCarParkingSpaces",
    "IsRepeatedGuest",
    "AgencyBooking",
    "CompanyListed",
    "CT_is_grp",
    "CT_is_trn",
    "CT_is_trnP",
    "RS_No-Show",
    "MS_Corporate",
    "MS_Direct",
    "MS_Groups",
    "MS_Offline TA/TO",
    "MS_Online TA",
    "DC_Direct",
    "DC_TA/TO",
    "DC_Undefined",
    "FROM_PRT",
    "FROM_GBR",
    "FROM_ESP",
    "FROM_IRL",
    "FROM_FRA",
    "FROM_DEU",
    "FROM_CN",
    "FROM_NLD",
    "FROM_USA",
    "FROM_ITA",
    "FROM_other",
    "DT_NonRefundable",
    "DT_Refundable",
    "MEAL_Undefined",
    "MEAL_HB",
    "MEAL_FB",
]

# Features for hotel 2 cancellation forecast
X2_cxl_cols = [
    "LeadTime",
    "LOS",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "ADR",
    "NumPeople",
    "Adults",
    "Children",
    "Babies",
    "TotalOfSpecialRequests",
    "PreviousBookings",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "BookingChanges",
    "DaysInWaitingList",
    "RequiredCarParkingSpaces",
    "IsRepeatedGuest",
    "AgencyBooking",
    "CompanyListed",
    "CT_is_grp",
    "CT_is_trn",
    "CT_is_trnP",
    "RS_No-Show",
    "MS_Complementary",
    "MS_Corporate",
    "MS_Direct",
    "MS_Groups",
    "MS_Offline TA/TO",
    "MS_Online TA",
    "MS_Undefined",
    "DC_Direct",
    "DC_GDS",
    "DC_TA/TO",
    "DC_Undefined",
    "FROM_PRT",
    "FROM_FRA",
    "FROM_DEU",
    "FROM_GBR",
    "FROM_ESP",
    "FROM_ITA",
    "FROM_BEL",
    "FROM_BRA",
    "FROM_USA",
    "FROM_NLD",
    "FROM_other",
    "DT_NonRefundable",
    "DT_Refundable",
    "MEAL_FB",
    "MEAL_HB",
    "MEAL_SC",
]
